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
    /// Byzantine gradient filter for outlier detection.
    byzantine_filter: Arc<RwLock<ByzantineGradientFilter>>,
    /// Aggregation strategy.
    aggregation_strategy: Arc<RwLock<AggregationStrategy>>,
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
            byzantine_filter,
            aggregation_strategy: Arc::new(RwLock::new(AggregationStrategy::FedAvg)),
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

        // Check we have enough workers
        let available_workers: Vec<PeerId> = {
            self.workers.read()
                .iter()
                .filter(|(_, w)| w.status == WorkerStatus::Available)
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
                // In production, would trigger slashing
                log::warn!("Worker {} should be slashed: {}", from, validation.reason);
            }

            return Err(OrchestratorError::ValidationFailed(validation.reason));
        }

        // Store gradient
        let gradient = CollectedGradient {
            peer_id: from.clone(),
            commitment,
            error_bound,
            proof,
            received_at: Instant::now(),
            validation: Some(validation),
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
        // Convert gradients to filter format: (id, norm, gradient_proxy)
        let submissions: Vec<(String, f64, Vec<f32>)> = gradients.iter().map(|(peer_id, grad)| {
            // Use commitment bytes as gradient proxy for distance computation
            let grad_proxy: Vec<f32> = grad.commitment.iter().map(|&b| b as f32).collect();
            let norm = grad.error_bound;
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
        accepted_commitments.sort();
        let mut hasher = sha2::Sha256::new();
        for commitment in &accepted_commitments {
            sha2::Digest::update(&mut hasher, commitment);
        }
        let combined: [u8; 32] = sha2::Digest::finalize(hasher).into();

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

        // Spawn event processing task
        tokio::spawn(async move {
            let mut rx = event_rx.lock().await;
            while running.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::select! {
                    Some(event) = rx.recv() => {
                        match event {
                            NetworkEvent::GradientMessage { from, message } => {
                                match message {
                                    GradientMessage::ShareGradient { round_id, gradient_commitment, error_bound, proof } => {
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
                            let gradient = CollectedGradient {
                                peer_id: from.clone(),
                                commitment,
                                error_bound,
                                proof,
                                received_at: Instant::now(),
                                validation: Some(validation),
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
                            let mut workers = workers_clone.write();
                            if let Some(worker) = workers.get_mut(&from) {
                                worker.failures += 1;
                                if worker.failures >= max_failures {
                                    worker.status = WorkerStatus::Excluded;
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

    #[test]
    fn test_config_default() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.min_workers, 3);
        assert_eq!(config.max_workers, 100);
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
}
