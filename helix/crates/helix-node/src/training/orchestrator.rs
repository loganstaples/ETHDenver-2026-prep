//! Multi-Node Training Orchestrator.
//!
//! Coordinates distributed training across multiple workers:
//! - Round coordination and lifecycle management
//! - Worker failure detection and recovery
//! - Gradient synchronization
//! - Heartbeat/liveness monitoring
//! - Byzantine gradient filtering for outlier detection

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use sha2::Digest;
use tokio::sync::{broadcast, mpsc};

use crate::network::messages::{
    GradientMessage, HeartbeatMessage, MessagePayload, NetworkMessage, PeerId, TrainingMessage,
    TrainingParams,
};
use crate::network::runner::{NetworkEvent, NetworkRunner};
use crate::round_commit::{
    RoundCommitConfig, RoundCommitManager, WorkerProof,
};
use crate::sc_client::{SCClient, TrainingProofInputs};
use crate::training::DistributedRoundId;
use ethers::types::U256;

use super::aggregation::AggregationStrategy;
use super::consensus::{
    ConsensusConfig, ConsensusProtocol, ConsensusResult,
    compute_aggregated_commitment, verify_bft_threshold,
};
use super::verification::{
    ByzantineGradientFilter, ByzantineStrategy, FilterResult,
    GradientValidator, ProofVerifier, ValidationResult, VerificationConfig,
};

/// Orchestrator configuration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Minimum workers to start a round.
    pub min_workers: usize,
    /// Maximum workers per round.
    pub max_workers: usize,
    /// Round collection timeout.
    pub collection_timeout: Duration,
    /// Round aggregation timeout.
    pub aggregation_timeout: Duration,
    /// Heartbeat interval.
    pub heartbeat_interval: Duration,
    /// Worker timeout (no heartbeat).
    pub worker_timeout: Duration,
    /// Maximum consecutive failures before exclusion.
    pub max_failures: u32,
    /// Enable automatic recovery.
    pub auto_recovery: bool,
    /// Verification configuration.
    pub verification: VerificationConfig,
    /// Default training parameters.
    pub default_params: TrainingParams,
    /// BFT consensus configuration for gradient aggregation.
    pub consensus: ConsensusConfig,
    /// Enable BFT consensus (2-phase commit) for gradient aggregation.
    /// When disabled, falls back to leader-only aggregation.
    pub consensus_enabled: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            min_workers: 3,
            max_workers: 100,
            collection_timeout: Duration::from_secs(300),
            aggregation_timeout: Duration::from_secs(60),
            heartbeat_interval: Duration::from_secs(10),
            worker_timeout: Duration::from_secs(30),
            max_failures: 3,
            auto_recovery: true,
            verification: VerificationConfig::default(),
            default_params: TrainingParams {
                learning_rate: 0.001,
                batch_size: 32,
                local_epochs: 5,
                max_error_bound: 0.1,
                d_in: 4,
                d_hid: 8,
                d_out: 2,
                model_seed: 42,
            },
            consensus: ConsensusConfig::default(),
            consensus_enabled: true,
        }
    }
}

/// Worker state.
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Worker status.
    pub status: WorkerStatus,
    /// Last heartbeat time.
    pub last_heartbeat: Instant,
    /// Current load (0-100).
    pub load: u8,
    /// Consecutive failures.
    pub failures: u32,
    /// Rounds participated in.
    pub rounds_participated: u64,
    /// Rounds completed successfully.
    pub rounds_completed: u64,
    /// Assigned shard ID.
    pub shard_id: Option<u32>,
}

/// Worker status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Worker is available.
    Available,
    /// Worker is assigned to current round.
    Assigned,
    /// Worker is computing.
    Computing,
    /// Worker submitted gradient.
    Submitted,
    /// Worker is unresponsive.
    Unresponsive,
    /// Worker is excluded (too many failures).
    Excluded,
}

/// Round state.
#[derive(Debug, Clone)]
pub struct RoundState {
    /// Round ID.
    pub id: u64,
    /// Round phase.
    pub phase: RoundPhase,
    /// Model hash at start.
    pub model_hash: [u8; 32],
    /// Training parameters.
    pub params: TrainingParams,
    /// Assigned workers.
    pub workers: HashSet<PeerId>,
    /// Received gradients.
    pub gradients: HashMap<PeerId, CollectedGradient>,
    /// Phase started at.
    pub phase_started: Instant,
    /// Round started at.
    pub started_at: Instant,
}

/// Round phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundPhase {
    /// Recruiting workers.
    Recruiting,
    /// Collecting gradients.
    Collecting,
    /// Aggregating gradients.
    Aggregating,
    /// Committing result.
    Committing,
    /// Round completed.
    Completed,
    /// Round failed.
    Failed,
}

/// Collected gradient from a worker.
#[derive(Debug, Clone)]
pub struct CollectedGradient {
    /// Worker peer ID.
    pub peer_id: PeerId,
    /// Gradient commitment.
    pub commitment: [u8; 32],
    /// Error bound.
    pub error_bound: f64,
    /// Proof.
    pub proof: Vec<u8>,
    /// Received at.
    pub received_at: Instant,
    /// Validation result.
    pub validation: Option<ValidationResult>,
    /// Gradient L2 norm (if provided by the worker). Used for Byzantine
    /// outlier detection. When absent, `error_bound` is used as a proxy.
    pub gradient_norm: Option<f64>,
}

/// Orchestrator event.
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// Round started.
    RoundStarted { round_id: u64, workers: Vec<PeerId> },
    /// Round completed.
    RoundCompleted { round_id: u64, result_hash: [u8; 32] },
    /// Round failed.
    RoundFailed { round_id: u64, reason: String },
    /// Worker joined.
    WorkerJoined { peer_id: PeerId },
    /// Worker left.
    WorkerLeft { peer_id: PeerId },
    /// Worker failed.
    WorkerFailed { peer_id: PeerId, reason: String },
    /// Gradient received.
    GradientReceived { round_id: u64, peer_id: PeerId },
    /// Gradient rejected.
    GradientRejected { round_id: u64, peer_id: PeerId, reason: String },
}

/// Training orchestrator.
pub struct TrainingOrchestrator {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: OrchestratorConfig,
    /// Network runner.
    network: Arc<NetworkRunner>,
    /// Workers.
    workers: Arc<RwLock<HashMap<PeerId, WorkerState>>>,
    /// Current round.
    current_round: Arc<RwLock<Option<RoundState>>>,
    /// Round counter.
    round_counter: Arc<RwLock<u64>>,
    /// Gradient validator.
    validator: Arc<GradientValidator>,
    /// Event sender.
    event_tx: broadcast::Sender<OrchestratorEvent>,
    /// Running state.
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Whether this node is the leader.
    is_leader: Arc<RwLock<bool>>,
    /// Round commit manager for proof aggregation + on-chain submission.
    round_commit: Arc<RwLock<Option<RoundCommitManager>>>,
    /// Smart contract client for on-chain slashing.
    sc_client: Arc<RwLock<Option<Arc<SCClient>>>>,
    /// On-chain model ID for slashing calls.
    model_id: Arc<RwLock<u64>>,
    /// Byzantine gradient filter for outlier detection.
    byzantine_filter: Arc<RwLock<ByzantineGradientFilter>>,
    /// Aggregation strategy.
    aggregation_strategy: Arc<RwLock<AggregationStrategy>>,
    /// BFT consensus protocol for 2-phase commit on gradient aggregation.
    consensus: Arc<RwLock<ConsensusProtocol>>,
}

impl TrainingOrchestrator {
    /// Creates a new training orchestrator.
    pub fn new(
        local_id: PeerId,
        config: OrchestratorConfig,
        network: Arc<NetworkRunner>,
    ) -> Self {
        let validator = Arc::new(GradientValidator::new(
            config.verification.clone(),
            100.0, // max gradient norm
            config.min_workers,
        ));

        let (event_tx, _) = broadcast::channel(1000);

        let byzantine_filter = Arc::new(RwLock::new(ByzantineGradientFilter::new(
            ByzantineStrategy::Combined,
            (config.min_workers / 3).max(1), // Tolerate up to n/3 Byzantine workers
        )));

        let consensus = Arc::new(RwLock::new(ConsensusProtocol::new(
            local_id.clone(),
            config.consensus.clone(),
        )));

        Self {
            local_id,
            config,
            network,
            workers: Arc::new(RwLock::new(HashMap::new())),
            current_round: Arc::new(RwLock::new(None)),
            round_counter: Arc::new(RwLock::new(0)),
            validator,
            event_tx,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            is_leader: Arc::new(RwLock::new(false)),
            round_commit: Arc::new(RwLock::new(None)),
            sc_client: Arc::new(RwLock::new(None)),
            model_id: Arc::new(RwLock::new(0)),
            byzantine_filter,
            aggregation_strategy: Arc::new(RwLock::new(AggregationStrategy::FedAvg)),
            consensus,
        }
    }

    /// Sets the aggregation strategy.
    pub fn set_aggregation_strategy(&self, strategy: AggregationStrategy) {
        *self.aggregation_strategy.write() = strategy;
    }

    /// Returns the current aggregation strategy.
    pub fn aggregation_strategy(&self) -> AggregationStrategy {
        self.aggregation_strategy.read().clone()
    }

    /// Sets up the round commit manager with an on-chain client.
    pub fn set_round_commit_manager(&self, manager: RoundCommitManager) {
        *self.round_commit.write() = Some(manager);
    }

    /// Sets the smart contract client for on-chain slashing.
    pub fn set_sc_client(&self, client: Arc<SCClient>, model_id: u64) {
        *self.sc_client.write() = Some(client);
        *self.model_id.write() = model_id;
    }

    /// Starts the orchestrator.
    pub async fn start(&self) -> mpsc::Receiver<OrchestratorEvent> {
        self.running.store(true, std::sync::atomic::Ordering::SeqCst);

        // Create event receiver
        let (tx, rx) = mpsc::channel(1000);

        // Forward broadcast events to mpsc
        let mut event_rx = self.event_tx.subscribe();
        tokio::spawn(async move {
            while let Ok(event) = event_rx.recv().await {
                if tx.send(event).await.is_err() {
                    break;
                }
            }
        });

        // Start heartbeat loop
        self.start_heartbeat_loop();

        // Start worker monitoring loop
        self.start_monitoring_loop();

        // Start network event processing
        self.start_network_processing();

        rx
    }

    /// Stops the orchestrator.
    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Sets whether this node is the leader.
    pub fn set_leader(&self, is_leader: bool) {
        *self.is_leader.write() = is_leader;
    }

    /// Returns whether this node is the leader.
    pub fn is_leader(&self) -> bool {
        *self.is_leader.read()
    }

    /// Starts a new training round (leader only).
    pub async fn start_round(&self, model_hash: [u8; 32]) -> Result<u64, OrchestratorError> {
        if !self.is_leader() {
            return Err(OrchestratorError::NotLeader);
        }

        // Check if training is paused due to network partition
        if self.network.is_training_paused() {
            return Err(OrchestratorError::TrainingPaused);
        }

        // Check we have enough workers (must be available AND meet reputation threshold)
        let available_workers: Vec<PeerId> = {
            let rep_mgr = self.network.reputation_manager().lock();
            self.workers.read()
                .iter()
                .filter(|(id, w)| {
                    if w.status != WorkerStatus::Available {
                        return false;
                    }
                    if !rep_mgr.is_peer_eligible(id) {
                        log::warn!(
                            "Worker {} excluded from round: reputation {:.1} below minimum threshold",
                            id, rep_mgr.score(id),
                        );
                        return false;
                    }
                    true
                })
                .map(|(id, _)| id.clone())
                .collect()
        };

        if available_workers.len() < self.config.min_workers {
            return Err(OrchestratorError::InsufficientWorkers {
                have: available_workers.len(),
                need: self.config.min_workers,
            });
        }

        // Increment round counter
        let round_id = {
            let mut counter = self.round_counter.write();
            *counter += 1;
            *counter
        };

        // Select workers for this round
        let workers_for_round: HashSet<PeerId> = available_workers
            .into_iter()
            .take(self.config.max_workers)
            .collect();

        // Update worker states
        {
            let mut workers = self.workers.write();
            for peer_id in &workers_for_round {
                if let Some(worker) = workers.get_mut(peer_id) {
                    worker.status = WorkerStatus::Assigned;
                    worker.shard_id = Some(workers_for_round.iter()
                        .position(|p| p == peer_id)
                        .unwrap_or(0) as u32);
                }
            }
        }

        // Create round state
        let round = RoundState {
            id: round_id,
            phase: RoundPhase::Recruiting,
            model_hash,
            params: self.config.default_params.clone(),
            workers: workers_for_round.clone(),
            gradients: HashMap::new(),
            phase_started: Instant::now(),
            started_at: Instant::now(),
        };

        // Store round state
        *self.current_round.write() = Some(round);

        // Broadcast round start message
        self.network.broadcast(MessagePayload::Training(TrainingMessage::RoundStart {
            round_id,
            model_hash,
            params: self.config.default_params.clone(),
        })).await;

        // Emit event
        let _ = self.event_tx.send(OrchestratorEvent::RoundStarted {
            round_id,
            workers: workers_for_round.into_iter().collect(),
        });

        // Transition to collecting phase
        self.transition_to_collecting().await;

        Ok(round_id)
    }

    /// Handles a gradient submission.
    pub async fn handle_gradient(
        &self,
        from: PeerId,
        round_id: u64,
        commitment: [u8; 32],
        error_bound: f64,
        proof: Vec<u8>,
    ) -> Result<(), OrchestratorError> {
        // Check round
        let mut round_guard = self.current_round.write();
        let round = round_guard.as_mut()
            .ok_or(OrchestratorError::NoActiveRound)?;

        if round.id != round_id {
            return Err(OrchestratorError::WrongRound {
                expected: round.id,
                got: round_id,
            });
        }

        if round.phase != RoundPhase::Collecting {
            return Err(OrchestratorError::WrongPhase {
                expected: RoundPhase::Collecting,
                got: round.phase,
            });
        }

        // Check worker is assigned
        if !round.workers.contains(&from) {
            return Err(OrchestratorError::WorkerNotAssigned(from.clone()));
        }

        // Validate gradient
        let validation = self.validator.validate_gradient(
            &from,
            round_id,
            commitment,
            error_bound,
            &proof,
            None,
        ).await;

        if !validation.is_valid {
            // Record negative reputation event for failed verification
            self.network.reputation_manager().lock().record_invalid_proof(&from);

            // Emit rejection event
            let _ = self.event_tx.send(OrchestratorEvent::GradientRejected {
                round_id,
                peer_id: from.clone(),
                reason: validation.reason.clone(),
            });

            // Update worker failure count
            {
                let mut workers = self.workers.write();
                if let Some(worker) = workers.get_mut(&from) {
                    worker.failures += 1;
                    if worker.failures >= self.config.max_failures {
                        worker.status = WorkerStatus::Excluded;
                    }
                }
            }

            if validation.should_slash {
                log::warn!("Worker {} should be slashed: {}", from, validation.reason);
                // Trigger on-chain slashing via challenge_proof
                let sc_client = self.sc_client.read().clone();
                let model_id = *self.model_id.read();
                let proof_clone = proof.clone();
                let from_clone = from.clone();
                let reason = validation.reason.clone();
                if let Some(client) = sc_client {
                    tokio::spawn(async move {
                        log::info!(
                            "Submitting challenge_proof for worker {} on model {} round {}",
                            from_clone, model_id, round_id,
                        );
                        match client.challenge_proof(
                            model_id,
                            round_id,
                            proof_clone,
                            vec![], // public inputs not available in rejection path
                        ).await {
                            Ok(receipt) => {
                                log::info!(
                                    "Slashing tx submitted for worker {}: tx={:?}",
                                    from_clone, receipt.transaction_hash,
                                );
                            }
                            Err(e) => {
                                log::error!(
                                    "Failed to submit slashing tx for worker {}: {}",
                                    from_clone, e,
                                );
                            }
                        }
                    });
                }
            }

            return Err(OrchestratorError::ValidationFailed(validation.reason));
        }

        // Record positive reputation event for valid proof
        self.network.reputation_manager().lock().record_valid_proof(&from);

        // Store gradient
        let gradient = CollectedGradient {
            peer_id: from.clone(),
            commitment,
            error_bound,
            proof,
            received_at: Instant::now(),
            validation: Some(validation),
            gradient_norm: None,
        };

        round.gradients.insert(from.clone(), gradient);

        // Update worker state
        {
            let mut workers = self.workers.write();
            if let Some(worker) = workers.get_mut(&from) {
                worker.status = WorkerStatus::Submitted;
            }
        }

        // Emit event
        let _ = self.event_tx.send(OrchestratorEvent::GradientReceived {
            round_id,
            peer_id: from,
        });

        // Check if we have all gradients
        if round.gradients.len() >= round.workers.len() {
            // All gradients received, transition to aggregation
            drop(round_guard);
            self.transition_to_aggregating().await;
        }

        Ok(())
    }

    /// Handles a heartbeat message.
    pub fn handle_heartbeat(&self, from: PeerId, message: HeartbeatMessage) {
        let from_for_pong = from.clone();
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(&from) {
            worker.last_heartbeat = Instant::now();
            worker.load = message.load;

            // Recover unresponsive worker
            if worker.status == WorkerStatus::Unresponsive && self.config.auto_recovery {
                worker.status = WorkerStatus::Available;
                let _ = self.event_tx.send(OrchestratorEvent::WorkerJoined {
                    peer_id: from.clone(),
                });
            }
        } else {
            // New worker
            let worker = WorkerState {
                peer_id: from.clone(),
                status: WorkerStatus::Available,
                last_heartbeat: Instant::now(),
                load: message.load,
                failures: 0,
                rounds_participated: 0,
                rounds_completed: 0,
                shard_id: None,
            };
            workers.insert(from.clone(), worker);
            let _ = self.event_tx.send(OrchestratorEvent::WorkerJoined {
                peer_id: from,
            });
        }

        // Send pong if it was a ping
        if !message.is_pong {
            let pong = HeartbeatMessage {
                seq: message.seq,
                is_pong: true,
                load: 0,
            };

            let network = self.network.clone();
            tokio::spawn(async move {
                let _ = network.send_direct(&from_for_pong, MessagePayload::Heartbeat(pong)).await;
            });
        }
    }

    /// Registers a worker.
    pub fn register_worker(&self, peer_id: PeerId) {
        let mut workers = self.workers.write();
        if !workers.contains_key(&peer_id) {
            workers.insert(peer_id.clone(), WorkerState {
                peer_id: peer_id.clone(),
                status: WorkerStatus::Available,
                last_heartbeat: Instant::now(),
                load: 0,
                failures: 0,
                rounds_participated: 0,
                rounds_completed: 0,
                shard_id: None,
            });

            let _ = self.event_tx.send(OrchestratorEvent::WorkerJoined { peer_id });
        }
    }

    /// Unregisters a worker.
    pub fn unregister_worker(&self, peer_id: &PeerId) {
        let mut workers = self.workers.write();
        workers.remove(peer_id);
        let _ = self.event_tx.send(OrchestratorEvent::WorkerLeft {
            peer_id: peer_id.clone(),
        });
    }

    /// Returns the current round info.
    pub fn current_round(&self) -> Option<(u64, RoundPhase)> {
        self.current_round.read().as_ref()
            .map(|r| (r.id, r.phase))
    }

    /// Returns worker stats.
    pub fn worker_stats(&self) -> WorkerStats {
        let workers = self.workers.read();
        let total = workers.len();
        let available = workers.values().filter(|w| w.status == WorkerStatus::Available).count();
        let computing = workers.values().filter(|w| w.status == WorkerStatus::Computing).count();
        let unresponsive = workers.values().filter(|w| w.status == WorkerStatus::Unresponsive).count();
        let excluded = workers.values().filter(|w| w.status == WorkerStatus::Excluded).count();

        WorkerStats {
            total,
            available,
            computing,
            unresponsive,
            excluded,
        }
    }

    async fn transition_to_collecting(&self) {
        let mut round_guard = self.current_round.write();
        if let Some(ref mut round) = *round_guard {
            round.phase = RoundPhase::Collecting;
            round.phase_started = Instant::now();

            // Update all assigned workers to computing
            let mut workers = self.workers.write();
            for peer_id in &round.workers {
                if let Some(worker) = workers.get_mut(peer_id) {
                    worker.status = WorkerStatus::Computing;
                    worker.rounds_participated += 1;
                }
            }
        }
    }

    async fn transition_to_aggregating(&self) {
        let mut round_guard = self.current_round.write();
        if let Some(ref mut round) = *round_guard {
            round.phase = RoundPhase::Aggregating;
            round.phase_started = Instant::now();
        }
        drop(round_guard);

        // Perform aggregation
        self.aggregate_and_commit().await;
    }

    async fn aggregate_and_commit(&self) {
        let round_id;
        let gradients: Vec<(PeerId, CollectedGradient)>;
        let worker_ids: Vec<PeerId>;
        let model_hash: [u8; 32];

        {
            let mut round_guard = self.current_round.write();
            if let Some(ref mut round) = *round_guard {
                round_id = round.id;
                model_hash = round.model_hash;
                gradients = round.gradients.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                worker_ids = round.workers.iter().cloned().collect();

                round.phase = RoundPhase::Committing;
                round.phase_started = Instant::now();
            } else {
                return;
            }
        }

        // Try using RoundCommitManager for real proof aggregation + on-chain submission
        let has_commit_manager = self.round_commit.read().is_some();

        if has_commit_manager {
            let result = self.aggregate_with_commit_manager(
                round_id,
                &gradients,
                &worker_ids,
                model_hash,
            ).await;

            match result {
                Ok(result_hash) => {
                    log::info!(
                        "Round {} aggregated via RoundCommitManager, tx={}",
                        round_id,
                        hex::encode(result_hash)
                    );
                    self.network.broadcast(MessagePayload::Gradient(
                        GradientMessage::AggregatedGradient {
                            round_id,
                            commitment: result_hash,
                            proof: vec![], // Proof already submitted on-chain
                        },
                    )).await;
                    self.complete_round(result_hash).await;
                    return;
                }
                Err(e) => {
                    log::warn!(
                        "RoundCommitManager submission failed for round {}: {}, falling back to local aggregation",
                        round_id, e
                    );
                }
            }
        }

        // === Byzantine gradient filtering (Task B3.3) ===
        // Convert gradients to filter format: (id, norm, gradient_vector).
        // NOTE: When actual gradient vectors are unavailable (current protocol only
        // sends commitments + proofs), we use error_bound as a 1-element proxy.
        // This enables TrimmedMean and z-score filtering on a meaningful value.
        // Krum on 1-D data degrades to "reject the most distant error_bound" which
        // is still useful for catching outliers.
        let submissions: Vec<(String, f64, Vec<f32>)> = gradients.iter().map(|(peer_id, grad)| {
            let norm = grad.gradient_norm.unwrap_or(grad.error_bound);
            let grad_proxy = vec![norm as f32];
            (peer_id.0.clone(), norm, grad_proxy)
        }).collect();

        let filter_results = self.byzantine_filter.write().filter_gradients(&submissions);

        // Partition into accepted/rejected
        let mut accepted_commitments = Vec::new();
        for (peer_id_str, result) in &filter_results {
            if result.accepted {
                if let Some((_, grad)) = gradients.iter().find(|(pid, _)| pid.0 == *peer_id_str) {
                    accepted_commitments.push(grad.commitment);
                }
            } else {
                let _ = self.event_tx.send(OrchestratorEvent::GradientRejected {
                    round_id,
                    peer_id: PeerId::from_string(peer_id_str.clone()),
                    reason: result.reason.clone().unwrap_or_else(|| "Byzantine filter".to_string()),
                });
                log::warn!(
                    "Byzantine filter rejected gradient from {}: {:?}",
                    peer_id_str,
                    result.reason
                );
            }
        }

        if accepted_commitments.is_empty() {
            // All rejected — fail the round
            self.fail_round(round_id, "All gradients rejected by Byzantine filter").await;
            return;
        }

        log::info!(
            "Byzantine filter: {}/{} gradients accepted for round {}",
            accepted_commitments.len(),
            gradients.len(),
            round_id,
        );

        // Compute aggregated commitment via SHA-256 of sorted accepted commitments
        let combined = compute_aggregated_commitment(&accepted_commitments);

        // === BFT Consensus (2-Phase Commit) ===
        // If consensus is enabled, run 2PC before accepting the aggregation.
        // The leader proposes the aggregated commitment and waits for 2f+1 votes.
        if self.config.consensus_enabled && worker_ids.len() >= 3 {
            let participants: HashSet<PeerId> = worker_ids.iter().cloned().collect();

            // Check BFT threshold: need n >= 3f+1
            let max_faulty = self.config.consensus.max_faulty;
            if !verify_bft_threshold(participants.len(), max_faulty) {
                log::warn!(
                    "Insufficient participants ({}) for BFT consensus (need >= 3f+1 = {}), \
                     falling back to leader-only aggregation",
                    participants.len(),
                    3 * max_faulty + 1,
                );
            } else {
                // Phase 1: Leader creates and broadcasts proposal
                let propose_msg = {
                    let mut consensus = self.consensus.write();
                    consensus.create_proposal(
                        round_id,
                        participants,
                        &accepted_commitments,
                        gradients.iter().map(|(_, g)| g.error_bound).sum::<f64>()
                            / gradients.len() as f64,
                    )
                };

                self.network.broadcast(MessagePayload::Consensus(propose_msg)).await;

                log::info!(
                    "Consensus proposal broadcast for round {} (commitment={})",
                    round_id,
                    hex::encode(&combined[..8]),
                );

                // Note: Vote collection happens asynchronously via handle_consensus_message.
                // The round completes when quorum is reached in handle_consensus_vote.
                // We return here and let the consensus flow drive completion.
                return;
            }
        }

        // Fallback: no consensus, commit directly
        self.network.broadcast(MessagePayload::Gradient(GradientMessage::AggregatedGradient {
            round_id,
            commitment: combined,
            proof: vec![],
        })).await;

        self.complete_round(combined).await;
    }

    /// Aggregates proofs via RoundCommitManager and submits to chain.
    async fn aggregate_with_commit_manager(
        &self,
        round_id: u64,
        gradients: &[(PeerId, CollectedGradient)],
        worker_ids: &[PeerId],
        model_hash: [u8; 32],
    ) -> Result<[u8; 32], String> {
        let dist_round_id = DistributedRoundId {
            session_id: 1,
            round_number: round_id,
        };

        // Start collection
        let commit_id = {
            let mut manager = self.round_commit.write();
            let manager = manager.as_mut().ok_or("No commit manager")?;
            manager
                .start_collection(dist_round_id, worker_ids.to_vec())
                .map_err(|e| format!("start_collection: {}", e))?
        };

        // Submit each worker's proof
        for (idx, (peer_id, gradient)) in gradients.iter().enumerate() {
            let worker_proof = WorkerProof {
                worker_id: peer_id.clone(),
                proof: gradient.proof.clone(),
                public_inputs: TrainingProofInputs {
                    old_hash_lo: U256::from_big_endian(&model_hash[..16]),
                    old_hash_hi: U256::from_big_endian(&model_hash[16..]),
                    new_hash_lo: U256::from_big_endian(&gradient.commitment[..16]),
                    new_hash_hi: U256::from_big_endian(&gradient.commitment[16..]),
                    loss: U256::zero(),
                    error_bound: U256::from((gradient.error_bound * 1e18) as u64),
                    step_number: U256::from(round_id),
                    error_checksum: U256::zero(),
                },
                error_bound: gradient.error_bound,
                share_index: idx,
                gradient_commitment: gradient.commitment,
                submitted_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let mut manager = self.round_commit.write();
            let manager = manager.as_mut().ok_or("No commit manager")?;
            manager
                .submit_proof(commit_id, worker_proof)
                .map_err(|e| format!("submit_proof: {}", e))?;
        }

        // Use first worker's commitment as new_commitment (representative proof)
        let new_commitment = gradients
            .first()
            .map(|(_, g)| g.commitment)
            .unwrap_or([0u8; 32]);

        // Finalize collection and aggregate
        let aggregated = {
            let mut manager = self.round_commit.write();
            let manager = manager.as_mut().ok_or("No commit manager")?;
            manager
                .finalize_collection(commit_id, model_hash, new_commitment)
                .map_err(|e| format!("finalize_collection: {}", e))?
        };

        // Submit to chain
        let result = {
            let mut manager = self.round_commit.write();
            let manager = manager.as_mut().ok_or("No commit manager")?;
            manager
                .submit_to_chain(commit_id)
                .await
                .map_err(|e| format!("submit_to_chain: {}", e))?
        };

        log::info!(
            "On-chain proof submitted: tx={:?}, block={}, gas={}",
            result.tx_hash,
            result.block_number,
            result.gas_used,
        );

        Ok(aggregated.gradient_commitment)
    }

    /// Fails the current round with the given reason.
    async fn fail_round(&self, round_id: u64, reason: &str) {
        {
            let mut round_guard = self.current_round.write();
            if let Some(ref mut round) = *round_guard {
                round.phase = RoundPhase::Failed;

                // Reset worker states
                let mut workers = self.workers.write();
                for peer_id in &round.workers {
                    if let Some(worker) = workers.get_mut(peer_id) {
                        worker.status = WorkerStatus::Available;
                        worker.shard_id = None;
                    }
                }
            }
        }

        // Emit failure event
        let _ = self.event_tx.send(OrchestratorEvent::RoundFailed {
            round_id,
            reason: reason.to_string(),
        });

        log::error!("Round {} failed: {}", round_id, reason);

        // Clear round
        *self.current_round.write() = None;
    }

    async fn complete_round(&self, result_hash: [u8; 32]) {
        let round_id;

        {
            let mut round_guard = self.current_round.write();
            if let Some(ref mut round) = *round_guard {
                round_id = round.id;
                round.phase = RoundPhase::Completed;

                // Update worker stats
                let mut workers = self.workers.write();
                for peer_id in &round.workers {
                    if let Some(worker) = workers.get_mut(peer_id) {
                        if round.gradients.contains_key(peer_id) {
                            worker.rounds_completed += 1;
                        }
                        worker.status = WorkerStatus::Available;
                        worker.shard_id = None;
                    }
                }
            } else {
                return;
            }
        }

        // Broadcast round complete
        self.network.broadcast(MessagePayload::Training(TrainingMessage::RoundComplete {
            round_id,
            result_hash,
        })).await;

        // Emit event
        let _ = self.event_tx.send(OrchestratorEvent::RoundCompleted {
            round_id,
            result_hash,
        });

        // Clear round
        *self.current_round.write() = None;
    }

    /// Handles an incoming BFT consensus message.
    ///
    /// Routes Propose/Vote/Commit/Abort messages to the consensus protocol
    /// and drives the 2-phase commit to completion.
    pub async fn handle_consensus_message(
        &self,
        from: PeerId,
        message: crate::network::messages::ConsensusMessage,
    ) {
        use crate::network::messages::ConsensusMessage;

        match message {
            ConsensusMessage::Propose {
                round_id,
                aggregated_commitment,
                proposer_binding,
                num_gradients,
                error_bound,
                nonce,
            } => {
                // Non-leader receives proposal — verify and vote
                let local_commitments = self.get_local_gradient_commitments(round_id);
                let participants = self.get_round_participants(round_id);

                let (vote_msg, _events) = {
                    let mut consensus = self.consensus.write();
                    consensus.handle_proposal(
                        &from,
                        round_id,
                        aggregated_commitment,
                        proposer_binding,
                        num_gradients,
                        error_bound,
                        nonce,
                        &local_commitments,
                        participants,
                    )
                };

                // Broadcast vote
                self.network.broadcast(MessagePayload::Consensus(vote_msg)).await;
            }

            ConsensusMessage::Vote {
                round_id,
                accept,
                voter_commitment,
                reason,
            } => {
                // Leader collects votes
                let (decision, _events) = {
                    let mut consensus = self.consensus.write();
                    consensus.handle_vote(&from, round_id, accept, voter_commitment, reason)
                };

                if let Some(decision_msg) = decision {
                    match decision_msg {
                        ConsensusMessage::Commit {
                            round_id: commit_round_id,
                            final_commitment,
                            votes_for,
                            total_participants,
                        } => {
                            log::info!(
                                "Consensus COMMITTED for round {}: {}/{} votes (commitment={})",
                                commit_round_id,
                                votes_for,
                                total_participants,
                                hex::encode(&final_commitment[..8]),
                            );

                            // Broadcast the commit decision
                            self.network.broadcast(
                                MessagePayload::Consensus(ConsensusMessage::Commit {
                                    round_id: commit_round_id,
                                    final_commitment,
                                    votes_for,
                                    total_participants,
                                }),
                            ).await;

                            // Also broadcast the aggregated gradient
                            self.network.broadcast(MessagePayload::Gradient(
                                GradientMessage::AggregatedGradient {
                                    round_id: commit_round_id,
                                    commitment: final_commitment,
                                    proof: vec![],
                                },
                            )).await;

                            // Complete the round
                            self.complete_round(final_commitment).await;

                            // Clear consensus state
                            self.consensus.write().clear_active_round();
                        }
                        ConsensusMessage::Abort { round_id: abort_round_id, reason } => {
                            log::warn!(
                                "Consensus ABORTED for round {}: {}",
                                abort_round_id, reason,
                            );

                            // Broadcast the abort
                            self.network.broadcast(
                                MessagePayload::Consensus(ConsensusMessage::Abort {
                                    round_id: abort_round_id,
                                    reason: reason.clone(),
                                }),
                            ).await;

                            // Fail the round
                            self.fail_round(
                                abort_round_id,
                                &format!("Consensus aborted: {}", reason),
                            ).await;

                            self.consensus.write().clear_active_round();
                        }
                        _ => {}
                    }
                }
            }

            ConsensusMessage::Commit {
                round_id,
                final_commitment,
                votes_for,
                total_participants,
            } => {
                // Non-leader receives commit decision
                let _events = {
                    let mut consensus = self.consensus.write();
                    consensus.handle_commit(
                        round_id,
                        final_commitment,
                        votes_for,
                        total_participants,
                    )
                };

                log::info!(
                    "Received consensus commit for round {} (commitment={})",
                    round_id,
                    hex::encode(&final_commitment[..8]),
                );

                self.consensus.write().clear_active_round();
            }

            ConsensusMessage::Abort { round_id, reason } => {
                // Non-leader receives abort
                let _events = {
                    let mut consensus = self.consensus.write();
                    consensus.handle_abort(round_id, reason.clone())
                };

                log::warn!(
                    "Received consensus abort for round {}: {}",
                    round_id, reason,
                );

                self.consensus.write().clear_active_round();
            }
        }
    }

    /// Returns the gradient commitments from the current round for consensus verification.
    fn get_local_gradient_commitments(&self, round_id: u64) -> Vec<[u8; 32]> {
        let round_guard = self.current_round.read();
        if let Some(ref round) = *round_guard {
            if round.id == round_id {
                let mut commitments: Vec<[u8; 32]> = round
                    .gradients
                    .values()
                    .map(|g| g.commitment)
                    .collect();
                commitments.sort();
                return commitments;
            }
        }
        vec![]
    }

    /// Returns the set of participants for a round.
    fn get_round_participants(&self, round_id: u64) -> HashSet<PeerId> {
        let round_guard = self.current_round.read();
        if let Some(ref round) = *round_guard {
            if round.id == round_id {
                return round.workers.clone();
            }
        }
        HashSet::new()
    }

    /// Returns a reference to the consensus protocol.
    pub fn consensus(&self) -> &Arc<RwLock<ConsensusProtocol>> {
        &self.consensus
    }

    fn start_heartbeat_loop(&self) {
        let running = self.running.clone();
        let network = self.network.clone();
        let local_id = self.local_id.clone();
        let interval = self.config.heartbeat_interval;

        tokio::spawn(async move {
            let mut seq = 0u64;
            let mut ticker = tokio::time::interval(interval);

            while running.load(std::sync::atomic::Ordering::SeqCst) {
                ticker.tick().await;

                let heartbeat = HeartbeatMessage {
                    seq,
                    is_pong: false,
                    load: 50, // Would compute actual load
                };

                network.broadcast(MessagePayload::Heartbeat(heartbeat)).await;
                seq += 1;
            }
        });
    }

    fn start_monitoring_loop(&self) {
        let running = self.running.clone();
        let workers = self.workers.clone();
        let current_round = self.current_round.clone();
        let event_tx = self.event_tx.clone();
        let timeout = self.config.worker_timeout;
        let collection_timeout = self.config.collection_timeout;
        let aggregation_timeout = self.config.aggregation_timeout;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(1));

            while running.load(std::sync::atomic::Ordering::SeqCst) {
                ticker.tick().await;

                // Check worker timeouts
                {
                    let mut workers_guard = workers.write();
                    for (peer_id, worker) in workers_guard.iter_mut() {
                        if worker.status != WorkerStatus::Excluded
                            && worker.status != WorkerStatus::Unresponsive
                            && worker.last_heartbeat.elapsed() > timeout
                        {
                            worker.status = WorkerStatus::Unresponsive;
                            let _ = event_tx.send(OrchestratorEvent::WorkerFailed {
                                peer_id: peer_id.clone(),
                                reason: "Heartbeat timeout".to_string(),
                            });
                        }
                    }
                }

                // Check round phase timeouts
                {
                    let mut round_guard = current_round.write();
                    if let Some(ref mut round) = *round_guard {
                        match round.phase {
                            RoundPhase::Collecting => {
                                if round.phase_started.elapsed() > collection_timeout {
                                    // Force transition to aggregation with what we have
                                    if round.gradients.len() >= 1 {
                                        round.phase = RoundPhase::Aggregating;
                                        round.phase_started = Instant::now();
                                    } else {
                                        round.phase = RoundPhase::Failed;
                                        let _ = event_tx.send(OrchestratorEvent::RoundFailed {
                                            round_id: round.id,
                                            reason: "Collection timeout with no gradients".to_string(),
                                        });
                                    }
                                }
                            }
                            RoundPhase::Aggregating => {
                                if round.phase_started.elapsed() > aggregation_timeout {
                                    round.phase = RoundPhase::Failed;
                                    let _ = event_tx.send(OrchestratorEvent::RoundFailed {
                                        round_id: round.id,
                                        reason: "Aggregation timeout".to_string(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        });
    }

    fn start_network_processing(&self) {
        let running = self.running.clone();
        let event_rx = self.network.event_receiver();
        let orchestrator_workers = self.workers.clone();
        let orchestrator_round = self.current_round.clone();
        let event_tx = self.event_tx.clone();
        let validator = self.validator.clone();

        // Clone self references for the spawned task
        // Using a channel-based approach instead of self reference
        let (gradient_tx, mut gradient_rx) = mpsc::channel::<(PeerId, u64, [u8; 32], f64, Vec<u8>)>(100);
        let (heartbeat_tx, mut heartbeat_rx) = mpsc::channel::<(PeerId, HeartbeatMessage)>(100);
        let (consensus_tx, mut consensus_rx) = mpsc::channel::<(PeerId, crate::network::messages::ConsensusMessage)>(100);

        // Spawn event processing task
        tokio::spawn(async move {
            let mut rx = event_rx.lock().await;
            while running.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::select! {
                    Some(event) = rx.recv() => {
                        match event {
                            NetworkEvent::GradientMessage { from, message } => {
                                match message {
                                    GradientMessage::ShareGradient { round_id, gradient_commitment, commitment_nonce, error_bound, proof } => {
                                        let _ = gradient_tx.send((from, round_id, gradient_commitment, error_bound, proof)).await;
                                    }
                                    _ => {}
                                }
                            }
                            NetworkEvent::Heartbeat { from, message } => {
                                let _ = heartbeat_tx.send((from, message)).await;
                            }
                            NetworkEvent::PeerDiscovered(peer) => {
                                let mut workers = orchestrator_workers.write();
                                if !workers.contains_key(&peer.id) {
                                    workers.insert(peer.id.clone(), WorkerState {
                                        peer_id: peer.id.clone(),
                                        status: WorkerStatus::Available,
                                        last_heartbeat: Instant::now(),
                                        load: 0,
                                        failures: 0,
                                        rounds_participated: 0,
                                        rounds_completed: 0,
                                        shard_id: None,
                                    });
                                    let _ = event_tx.send(OrchestratorEvent::WorkerJoined { peer_id: peer.id });
                                }
                            }
                            NetworkEvent::PeerDisconnected(peer_id) => {
                                orchestrator_workers.write().remove(&peer_id);
                                let _ = event_tx.send(OrchestratorEvent::WorkerLeft { peer_id });
                            }
                            NetworkEvent::ConsensusMessage { from, message } => {
                                let _ = consensus_tx.send((from, message)).await;
                            }
                            _ => {}
                        }
                    }
                    else => break,
                }
            }
        });

        // Spawn gradient handling task
        let workers_clone = self.workers.clone();
        let round_clone = self.current_round.clone();
        let event_tx_clone = self.event_tx.clone();
        let validator_clone = self.validator.clone();
        let max_failures = self.config.max_failures;
        let sc_client_clone = self.sc_client.clone();
        let model_id_clone = self.model_id.clone();
        let reputation_clone = self.network.reputation_manager().clone();

        tokio::spawn(async move {
            while let Some((from, round_id, commitment, error_bound, proof)) = gradient_rx.recv().await {
                // Validate and store gradient
                let validation = validator_clone.validate_gradient(
                    &from,
                    round_id,
                    commitment,
                    error_bound,
                    &proof,
                    None,
                ).await;

                let mut round_guard = round_clone.write();
                if let Some(ref mut round) = *round_guard {
                    if round.id == round_id && round.phase == RoundPhase::Collecting {
                        if validation.is_valid {
                            reputation_clone.lock().record_valid_proof(&from);

                            let gradient = CollectedGradient {
                                peer_id: from.clone(),
                                commitment,
                                error_bound,
                                proof,
                                received_at: Instant::now(),
                                validation: Some(validation),
                                gradient_norm: None,
                            };
                            round.gradients.insert(from.clone(), gradient);

                            let mut workers = workers_clone.write();
                            if let Some(worker) = workers.get_mut(&from) {
                                worker.status = WorkerStatus::Submitted;
                            }

                            let _ = event_tx_clone.send(OrchestratorEvent::GradientReceived {
                                round_id,
                                peer_id: from,
                            });
                        } else {
                            reputation_clone.lock().record_invalid_proof(&from);

                            let mut workers = workers_clone.write();
                            if let Some(worker) = workers.get_mut(&from) {
                                worker.failures += 1;
                                if worker.failures >= max_failures {
                                    worker.status = WorkerStatus::Excluded;
                                }
                            }

                            // Trigger on-chain slashing if warranted
                            if validation.should_slash {
                                let sc_client = sc_client_clone.read().clone();
                                let model_id = *model_id_clone.read();
                                let from_slash = from.clone();
                                let proof_slash = proof.clone();
                                if let Some(client) = sc_client {
                                    tokio::spawn(async move {
                                        log::info!(
                                            "Submitting challenge_proof for worker {} (round {})",
                                            from_slash, round_id,
                                        );
                                        if let Err(e) = client.challenge_proof(
                                            model_id, round_id, proof_slash, vec![],
                                        ).await {
                                            log::error!(
                                                "Failed to submit slashing tx for {}: {}",
                                                from_slash, e,
                                            );
                                        }
                                    });
                                }
                            }

                            let _ = event_tx_clone.send(OrchestratorEvent::GradientRejected {
                                round_id,
                                peer_id: from,
                                reason: validation.reason,
                            });
                        }
                    }
                }
            }
        });

        // Spawn heartbeat handling task
        let workers_clone = self.workers.clone();
        let event_tx_clone = self.event_tx.clone();
        let auto_recovery = self.config.auto_recovery;

        tokio::spawn(async move {
            while let Some((from, message)) = heartbeat_rx.recv().await {
                let mut workers = workers_clone.write();
                if let Some(worker) = workers.get_mut(&from) {
                    worker.last_heartbeat = Instant::now();
                    worker.load = message.load;

                    if worker.status == WorkerStatus::Unresponsive && auto_recovery {
                        worker.status = WorkerStatus::Available;
                        let _ = event_tx_clone.send(OrchestratorEvent::WorkerJoined {
                            peer_id: from,
                        });
                    }
                } else {
                    workers.insert(from.clone(), WorkerState {
                        peer_id: from.clone(),
                        status: WorkerStatus::Available,
                        last_heartbeat: Instant::now(),
                        load: message.load,
                        failures: 0,
                        rounds_participated: 0,
                        rounds_completed: 0,
                        shard_id: None,
                    });
                    let _ = event_tx_clone.send(OrchestratorEvent::WorkerJoined { peer_id: from });
                }
            }
        });

        // Spawn consensus message handling task
        let consensus_proto = self.consensus.clone();
        let consensus_network = self.network.clone();
        let consensus_round = self.current_round.clone();
        let consensus_event_tx = self.event_tx.clone();
        let consensus_workers = self.workers.clone();
        let sc_client_consensus = self.sc_client.clone();
        let model_id_consensus = self.model_id.clone();

        tokio::spawn(async move {
            while let Some((from, message)) = consensus_rx.recv().await {
                use crate::network::messages::ConsensusMessage;

                match message {
                    ConsensusMessage::Propose {
                        round_id,
                        aggregated_commitment,
                        proposer_binding,
                        num_gradients,
                        error_bound,
                        nonce,
                    } => {
                        // Non-leader receives proposal — verify and vote
                        let local_commitments = {
                            let round_guard = consensus_round.read();
                            if let Some(ref round) = *round_guard {
                                if round.id == round_id {
                                    round.gradients.values().map(|g| g.commitment).collect::<Vec<_>>()
                                } else {
                                    vec![]
                                }
                            } else {
                                vec![]
                            }
                        };

                        let participants = {
                            let round_guard = consensus_round.read();
                            if let Some(ref round) = *round_guard {
                                if round.id == round_id {
                                    round.workers.clone()
                                } else {
                                    HashSet::new()
                                }
                            } else {
                                HashSet::new()
                            }
                        };

                        let vote_msg = {
                            let mut consensus = consensus_proto.write();
                            let (msg, _events) = consensus.handle_proposal(
                                &from,
                                round_id,
                                aggregated_commitment,
                                proposer_binding,
                                num_gradients,
                                error_bound,
                                nonce,
                                &local_commitments,
                                participants,
                            );
                            msg
                        };

                        consensus_network.broadcast(MessagePayload::Consensus(vote_msg)).await;
                    }

                    ConsensusMessage::Vote {
                        round_id,
                        accept,
                        voter_commitment,
                        reason,
                    } => {
                        // Leader collects votes
                        let decision = {
                            let mut consensus = consensus_proto.write();
                            let (msg, _events) = consensus.handle_vote(
                                &from, round_id, accept, voter_commitment, reason,
                            );
                            msg
                        };

                        if let Some(decision_msg) = decision {
                            match decision_msg {
                                ConsensusMessage::Commit {
                                    round_id: commit_round_id,
                                    final_commitment,
                                    votes_for,
                                    total_participants,
                                } => {
                                    log::info!(
                                        "Consensus COMMITTED for round {}: {}/{} votes (commitment={})",
                                        commit_round_id, votes_for, total_participants,
                                        hex::encode(&final_commitment[..8]),
                                    );

                                    // Broadcast the commit decision
                                    consensus_network.broadcast(
                                        MessagePayload::Consensus(ConsensusMessage::Commit {
                                            round_id: commit_round_id,
                                            final_commitment,
                                            votes_for,
                                            total_participants,
                                        }),
                                    ).await;

                                    // Also broadcast the aggregated gradient
                                    consensus_network.broadcast(MessagePayload::Gradient(
                                        GradientMessage::AggregatedGradient {
                                            round_id: commit_round_id,
                                            commitment: final_commitment,
                                            proof: vec![],
                                        },
                                    )).await;

                                    // Complete the round and reset workers
                                    {
                                        let mut round_guard = consensus_round.write();
                                        if let Some(ref mut round) = *round_guard {
                                            if round.id == commit_round_id {
                                                round.phase = RoundPhase::Completed;

                                                let mut workers = consensus_workers.write();
                                                for peer_id in &round.workers {
                                                    if let Some(worker) = workers.get_mut(peer_id) {
                                                        if round.gradients.contains_key(peer_id) {
                                                            worker.rounds_completed += 1;
                                                        }
                                                        worker.status = WorkerStatus::Available;
                                                        worker.shard_id = None;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Broadcast round complete to training layer
                                    consensus_network.broadcast(MessagePayload::Training(
                                        TrainingMessage::RoundComplete {
                                            round_id: commit_round_id,
                                            result_hash: final_commitment,
                                        },
                                    )).await;

                                    let _ = consensus_event_tx.send(OrchestratorEvent::RoundCompleted {
                                        round_id: commit_round_id,
                                        result_hash: final_commitment,
                                    });

                                    // Clear round state
                                    *consensus_round.write() = None;
                                    consensus_proto.write().clear_active_round();
                                }
                                ConsensusMessage::Abort { round_id: abort_round_id, reason } => {
                                    log::warn!(
                                        "Consensus ABORTED for round {}: {}",
                                        abort_round_id, reason,
                                    );

                                    consensus_network.broadcast(
                                        MessagePayload::Consensus(ConsensusMessage::Abort {
                                            round_id: abort_round_id,
                                            reason: reason.clone(),
                                        }),
                                    ).await;

                                    // Fail the round and reset workers
                                    {
                                        let mut round_guard = consensus_round.write();
                                        if let Some(ref mut round) = *round_guard {
                                            if round.id == abort_round_id {
                                                round.phase = RoundPhase::Failed;

                                                let mut workers = consensus_workers.write();
                                                for peer_id in &round.workers {
                                                    if let Some(worker) = workers.get_mut(peer_id) {
                                                        worker.status = WorkerStatus::Available;
                                                        worker.shard_id = None;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    let _ = consensus_event_tx.send(OrchestratorEvent::RoundFailed {
                                        round_id: abort_round_id,
                                        reason: format!("Consensus aborted: {}", reason),
                                    });

                                    *consensus_round.write() = None;
                                    consensus_proto.write().clear_active_round();
                                }
                                _ => {}
                            }
                        }
                    }

                    ConsensusMessage::Commit {
                        round_id,
                        final_commitment,
                        votes_for,
                        total_participants,
                    } => {
                        // Non-leader receives commit decision
                        {
                            let mut consensus = consensus_proto.write();
                            let _events = consensus.handle_commit(
                                round_id, final_commitment, votes_for, total_participants,
                            );
                        }

                        log::info!(
                            "Received consensus commit for round {} (commitment={})",
                            round_id, hex::encode(&final_commitment[..8]),
                        );

                        consensus_proto.write().clear_active_round();
                    }

                    ConsensusMessage::Abort { round_id, reason } => {
                        // Non-leader receives abort
                        {
                            let mut consensus = consensus_proto.write();
                            let _events = consensus.handle_abort(round_id, reason.clone());
                        }

                        log::warn!(
                            "Received consensus abort for round {}: {}",
                            round_id, reason,
                        );

                        consensus_proto.write().clear_active_round();
                    }
                }
            }
        });
    }
}

/// Worker statistics.
#[derive(Debug, Clone)]
pub struct WorkerStats {
    pub total: usize,
    pub available: usize,
    pub computing: usize,
    pub unresponsive: usize,
    pub excluded: usize,
}

/// Orchestrator errors.
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("Not the leader")]
    NotLeader,
    #[error("Training paused due to network partition")]
    TrainingPaused,
    #[error("Insufficient workers: have {have}, need {need}")]
    InsufficientWorkers { have: usize, need: usize },
    #[error("No active round")]
    NoActiveRound,
    #[error("Wrong round: expected {expected}, got {got}")]
    WrongRound { expected: u64, got: u64 },
    #[error("Wrong phase: expected {expected:?}, got {got:?}")]
    WrongPhase { expected: RoundPhase, got: RoundPhase },
    #[error("Worker not assigned: {0}")]
    WorkerNotAssigned(PeerId),
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::messages::NodeCapabilities;
    use crate::network::runner::{NetworkRunnerConfig, NetworkRunner};

    fn test_network() -> Arc<NetworkRunner> {
        let local_id = PeerId::random();
        let mut config = NetworkRunnerConfig::default();
        config.transport.listen_addr = "127.0.0.1:0".parse().unwrap();
        Arc::new(NetworkRunner::new(local_id, config, NodeCapabilities::default()).unwrap())
    }

    fn test_orchestrator() -> TrainingOrchestrator {
        let network = test_network();
        let config = OrchestratorConfig::default();
        let local_id = PeerId::random();
        TrainingOrchestrator::new(local_id, config, network)
    }

    #[test]
    fn test_config_default() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.min_workers, 3);
        assert_eq!(config.max_workers, 100);
        assert_eq!(config.max_failures, 3);
        assert!(config.auto_recovery);
        assert!(config.consensus_enabled);
    }

    #[test]
    fn test_worker_state() {
        let worker = WorkerState {
            peer_id: PeerId::random(),
            status: WorkerStatus::Available,
            last_heartbeat: Instant::now(),
            load: 50,
            failures: 0,
            rounds_participated: 0,
            rounds_completed: 0,
            shard_id: None,
        };

        assert_eq!(worker.status, WorkerStatus::Available);
    }

    #[test]
    fn test_register_and_unregister_worker() {
        let orch = test_orchestrator();
        let peer = PeerId::from_string("worker-1");

        orch.register_worker(peer.clone());
        let stats = orch.worker_stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.available, 1);

        orch.unregister_worker(&peer);
        let stats = orch.worker_stats();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_duplicate_register_ignored() {
        let orch = test_orchestrator();
        let peer = PeerId::from_string("worker-1");

        orch.register_worker(peer.clone());
        orch.register_worker(peer.clone()); // duplicate
        let stats = orch.worker_stats();
        assert_eq!(stats.total, 1);
    }

    #[test]
    fn test_worker_stats_breakdown() {
        let orch = test_orchestrator();

        for i in 0..5 {
            orch.register_worker(PeerId::from_string(format!("worker-{}", i)));
        }

        let stats = orch.worker_stats();
        assert_eq!(stats.total, 5);
        assert_eq!(stats.available, 5);
        assert_eq!(stats.computing, 0);
        assert_eq!(stats.unresponsive, 0);
        assert_eq!(stats.excluded, 0);
    }

    #[tokio::test]
    async fn test_start_round_not_leader() {
        let orch = test_orchestrator();
        // By default, not a leader
        assert!(!orch.is_leader());

        let result = orch.start_round([0u8; 32]).await;
        assert!(matches!(result, Err(OrchestratorError::NotLeader)));
    }

    #[tokio::test]
    async fn test_start_round_insufficient_workers() {
        let orch = test_orchestrator();
        orch.set_leader(true);

        // Only register 1 worker, but min is 3
        orch.register_worker(PeerId::from_string("worker-1"));

        let result = orch.start_round([0u8; 32]).await;
        assert!(matches!(result, Err(OrchestratorError::InsufficientWorkers { have: 1, need: 3 })));
    }

    #[tokio::test]
    async fn test_start_round_success() {
        let orch = test_orchestrator();
        orch.set_leader(true);

        // Register enough workers
        for i in 0..5 {
            orch.register_worker(PeerId::from_string(format!("worker-{}", i)));
        }

        let result = orch.start_round([0u8; 32]).await;
        assert!(result.is_ok());
        let round_id = result.unwrap();
        assert_eq!(round_id, 1);

        // Current round should be set
        let (id, phase) = orch.current_round().unwrap();
        assert_eq!(id, 1);
        assert_eq!(phase, RoundPhase::Collecting);
    }

    #[tokio::test]
    async fn test_round_increments_counter() {
        let orch = test_orchestrator();
        orch.set_leader(true);

        for i in 0..5 {
            orch.register_worker(PeerId::from_string(format!("worker-{}", i)));
        }

        let r1 = orch.start_round([0u8; 32]).await.unwrap();
        assert_eq!(r1, 1);

        // Reset workers to available (they got moved to Computing)
        {
            let mut workers = orch.workers.write();
            for w in workers.values_mut() {
                w.status = WorkerStatus::Available;
            }
        }
        *orch.current_round.write() = None;

        let r2 = orch.start_round([1u8; 32]).await.unwrap();
        assert_eq!(r2, 2);
    }

    #[test]
    fn test_set_and_get_leader() {
        let orch = test_orchestrator();
        assert!(!orch.is_leader());
        orch.set_leader(true);
        assert!(orch.is_leader());
        orch.set_leader(false);
        assert!(!orch.is_leader());
    }

    #[test]
    fn test_set_aggregation_strategy() {
        let orch = test_orchestrator();
        assert!(matches!(orch.aggregation_strategy(), AggregationStrategy::FedAvg));

        orch.set_aggregation_strategy(AggregationStrategy::Krum { num_byzantine: 2 });
        assert!(matches!(orch.aggregation_strategy(), AggregationStrategy::Krum { num_byzantine: 2 }));
    }

    #[test]
    fn test_no_active_round_initially() {
        let orch = test_orchestrator();
        assert!(orch.current_round().is_none());
    }

    #[tokio::test]
    async fn test_heartbeat_registers_new_worker() {
        let orch = test_orchestrator();
        let peer = PeerId::from_string("new-worker");

        orch.handle_heartbeat(peer.clone(), HeartbeatMessage {
            seq: 1,
            is_pong: false,
            load: 30,
        });

        let stats = orch.worker_stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.available, 1);
    }

    #[tokio::test]
    async fn test_heartbeat_updates_existing_worker() {
        let orch = test_orchestrator();
        let peer = PeerId::from_string("worker-1");

        orch.register_worker(peer.clone());

        orch.handle_heartbeat(peer.clone(), HeartbeatMessage {
            seq: 1,
            is_pong: false,
            load: 75,
        });

        let workers = orch.workers.read();
        let worker = workers.get(&peer).unwrap();
        assert_eq!(worker.load, 75);
    }

    #[tokio::test]
    async fn test_handle_gradient_no_active_round() {
        let orch = test_orchestrator();
        let result = orch.handle_gradient(
            PeerId::from_string("worker-1"),
            1,
            [0u8; 32],
            0.01,
            vec![],
        ).await;
        assert!(matches!(result, Err(OrchestratorError::NoActiveRound)));
    }

    #[tokio::test]
    async fn test_handle_gradient_wrong_round() {
        let orch = test_orchestrator();
        orch.set_leader(true);

        for i in 0..3 {
            orch.register_worker(PeerId::from_string(format!("worker-{}", i)));
        }
        orch.start_round([0u8; 32]).await.unwrap();

        let result = orch.handle_gradient(
            PeerId::from_string("worker-0"),
            999, // wrong round ID
            [0u8; 32],
            0.01,
            vec![],
        ).await;
        assert!(matches!(result, Err(OrchestratorError::WrongRound { .. })));
    }

    #[tokio::test]
    async fn test_handle_gradient_worker_not_assigned() {
        let orch = test_orchestrator();
        orch.set_leader(true);

        for i in 0..3 {
            orch.register_worker(PeerId::from_string(format!("worker-{}", i)));
        }
        let round_id = orch.start_round([0u8; 32]).await.unwrap();

        // Outsider tries to submit
        let result = orch.handle_gradient(
            PeerId::from_string("outsider"),
            round_id,
            [0u8; 32],
            0.01,
            vec![],
        ).await;
        assert!(matches!(result, Err(OrchestratorError::WorkerNotAssigned(_))));
    }
}
