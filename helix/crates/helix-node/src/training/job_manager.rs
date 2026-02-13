//! Training Job Lifecycle Manager.
//!
//! Orchestrates the complete lifecycle of a training round from announcement
//! through on-chain finalization. The `TrainingJobManager` is a state machine
//! that drives the following phases:
//!
//! 1. **Announcing** — Broadcast round configuration to all registered workers
//! 2. **WaitingForQuorum** — Collect worker readiness acknowledgements
//! 3. **Distributing** — Send model weights to all participating workers
//! 4. **Training** — Workers compute local training steps and submit proofs
//! 5. **Aggregating** — Byzantine-filter gradients and aggregate commitments
//! 6. **Submitting** — Submit aggregated proof on-chain via OnChainPipeline
//! 7. **Finalizing** — Broadcast results, distribute updated weights
//! 8. **Complete** — Round finished successfully
//!
//! Failure at any phase transitions to `Failed` with a descriptive reason.
//! Worker dropout, timeouts, and invalid proofs are handled gracefully.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, RwLock};

use helix_core::ModelCheckpoint;

use crate::config::{ChainConfig, NodeConfig, TrainingModelConfig};
use crate::network::messages::{
    MessagePayload, ModelDims, NetworkMessage, PeerId, RegistrationMessage,
    RoundManagementMessage, TrainingMessage, TrainingParams,
};
use crate::network::runner::NetworkRunner;
use crate::on_chain_pipeline::OnChainPipeline;
use crate::roles::aggregator::{AggregatedResult, AggregatorNode, CollectedGradient};
use crate::trainer::{average_models, MlpModel};
use crate::training::orchestrator::{OrchestratorEvent, TrainingOrchestrator};

// ============================================================================
// Job State Machine
// ============================================================================

/// Phase of the training job lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobPhase {
    /// Idle — no active job.
    Idle,
    /// Announcing the round to workers via gossip.
    Announcing,
    /// Waiting for enough workers to acknowledge readiness.
    WaitingForQuorum,
    /// Distributing model weights to all participants.
    Distributing,
    /// Workers are training; collecting proofs and gradients.
    Training,
    /// Aggregating collected gradients with Byzantine filtering.
    Aggregating,
    /// Submitting aggregated proof on-chain.
    Submitting,
    /// Broadcasting results and distributing updated weights.
    Finalizing,
    /// Round completed successfully.
    Complete,
    /// Round failed.
    Failed,
}

impl std::fmt::Display for JobPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Announcing => write!(f, "Announcing"),
            Self::WaitingForQuorum => write!(f, "WaitingForQuorum"),
            Self::Distributing => write!(f, "Distributing"),
            Self::Training => write!(f, "Training"),
            Self::Aggregating => write!(f, "Aggregating"),
            Self::Submitting => write!(f, "Submitting"),
            Self::Finalizing => write!(f, "Finalizing"),
            Self::Complete => write!(f, "Complete"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Configuration for a training job.
#[derive(Debug, Clone)]
pub struct TrainingJobConfig {
    /// On-chain model ID (if registered).
    pub model_id: Option<u64>,
    /// Model architecture dimensions.
    pub model_dims: ModelDims,
    /// Reference to the training dataset.
    pub dataset_ref: String,
    /// Number of training steps per worker per round.
    pub steps_per_worker: u32,
    /// Learning rate for this round.
    pub learning_rate: f64,
    /// Maximum error budget for this round.
    pub error_budget: f64,
    /// Minimum workers required to proceed.
    pub min_workers: usize,
    /// Maximum workers allowed per round.
    pub max_workers: usize,
    /// Timeout for worker quorum (how long to wait for enough workers).
    pub quorum_timeout: Duration,
    /// Timeout for weight distribution phase.
    pub distribution_timeout: Duration,
    /// Timeout for training/proof collection.
    pub training_timeout: Duration,
    /// Timeout for gradient aggregation.
    pub aggregation_timeout: Duration,
    /// Timeout for on-chain submission.
    pub submission_timeout: Duration,
    /// Whether to submit proofs on-chain.
    pub submit_on_chain: bool,
    /// Number of sequential rounds to run (0 = single round).
    pub total_rounds: u64,
    /// Minimum stake required for worker participation (in wei).
    pub min_stake_amount: u64,
}

impl Default for TrainingJobConfig {
    fn default() -> Self {
        Self {
            model_id: None,
            model_dims: ModelDims {
                d_in: 4,
                d_hid: 8,
                d_out: 2,
                num_layers: 2,
                num_heads: 0,
                activation_type: 0,
            },
            dataset_ref: String::new(),
            steps_per_worker: 10,
            learning_rate: 0.01,
            error_budget: 0.1,
            min_workers: 1,
            max_workers: 100,
            quorum_timeout: Duration::from_secs(120),
            distribution_timeout: Duration::from_secs(60),
            training_timeout: Duration::from_secs(300),
            aggregation_timeout: Duration::from_secs(60),
            submission_timeout: Duration::from_secs(120),
            submit_on_chain: false,
            total_rounds: 1,
            min_stake_amount: 0,
        }
    }
}

impl TrainingJobConfig {
    /// Creates a job config from existing node config.
    pub fn from_node_config(node_config: &NodeConfig) -> Self {
        let t = &node_config.training;
        let submit_on_chain = node_config.chain.is_some();
        let model_id = node_config.chain.as_ref().and_then(|c| c.model_id);

        Self {
            model_id,
            model_dims: ModelDims {
                d_in: t.d_in,
                d_hid: t.d_hid,
                d_out: t.d_out,
                num_layers: 2,
                num_heads: 0,
                activation_type: 0,
            },
            dataset_ref: String::new(),
            steps_per_worker: 10,
            learning_rate: t.learning_rate,
            error_budget: 0.1,
            min_workers: t.min_workers,
            max_workers: 100,
            quorum_timeout: Duration::from_secs(t.collection_timeout_secs),
            distribution_timeout: Duration::from_secs(60),
            training_timeout: Duration::from_secs(t.collection_timeout_secs),
            aggregation_timeout: Duration::from_secs(60),
            submission_timeout: Duration::from_secs(120),
            submit_on_chain,
            total_rounds: 1,
            min_stake_amount: 0,
        }
    }
}

/// Status of an individual worker within a job.
#[derive(Debug, Clone)]
pub struct WorkerJobState {
    /// Peer ID of the worker.
    pub peer_id: PeerId,
    /// When the worker acknowledged readiness.
    pub ready_at: Option<Instant>,
    /// When the worker received model weights.
    pub weights_sent_at: Option<Instant>,
    /// When the worker submitted their proof.
    pub proof_submitted_at: Option<Instant>,
    /// Whether the worker's proof passed validation.
    pub proof_valid: Option<bool>,
    /// Proof bytes submitted by the worker.
    pub proof: Option<Vec<u8>>,
    /// Public inputs from the worker's proof.
    pub public_inputs: Option<Vec<[u8; 32]>>,
    /// Checkpoint data from the worker after training.
    pub checkpoint_data: Option<Vec<u8>>,
    /// Error bound achieved by this worker.
    pub error_bound: Option<f64>,
    /// New model hash after worker training.
    pub new_model_hash: Option<[u8; 32]>,
    /// Steps completed by this worker.
    pub steps_completed: u32,
}

impl WorkerJobState {
    fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            ready_at: None,
            weights_sent_at: None,
            proof_submitted_at: None,
            proof_valid: None,
            proof: None,
            public_inputs: None,
            checkpoint_data: None,
            error_bound: None,
            new_model_hash: None,
            steps_completed: 0,
        }
    }
}

/// Live snapshot of a training job's state.
#[derive(Debug, Clone)]
pub struct JobSnapshot {
    /// Current phase.
    pub phase: JobPhase,
    /// Round ID.
    pub round_id: u64,
    /// When this phase started.
    pub phase_started: Instant,
    /// When the job started.
    pub job_started: Instant,
    /// Number of workers that acknowledged readiness.
    pub workers_ready: usize,
    /// Number of workers that submitted proofs.
    pub proofs_received: usize,
    /// Total workers in this round.
    pub total_workers: usize,
    /// Workers excluded by Byzantine filtering.
    pub excluded_workers: Vec<PeerId>,
    /// On-chain transaction hash (if submitted).
    pub tx_hash: Option<String>,
    /// Failure reason (if Failed).
    pub failure_reason: Option<String>,
}

/// Events emitted by the job manager during lifecycle.
#[derive(Debug, Clone)]
pub enum JobEvent {
    /// Job phase transitioned.
    PhaseChanged {
        round_id: u64,
        from: JobPhase,
        to: JobPhase,
    },
    /// Worker acknowledged readiness.
    WorkerReady {
        round_id: u64,
        peer_id: PeerId,
        workers_ready: usize,
        workers_needed: usize,
    },
    /// Worker submitted proof.
    ProofReceived {
        round_id: u64,
        peer_id: PeerId,
        proofs_received: usize,
        proofs_expected: usize,
    },
    /// Worker's proof was rejected.
    ProofRejected {
        round_id: u64,
        peer_id: PeerId,
        reason: String,
    },
    /// Worker dropped out or timed out.
    WorkerDropped {
        round_id: u64,
        peer_id: PeerId,
        reason: String,
    },
    /// Aggregation completed.
    AggregationComplete {
        round_id: u64,
        num_contributors: usize,
        num_excluded: usize,
        total_error_bound: f64,
    },
    /// On-chain submission succeeded.
    OnChainSubmitted {
        round_id: u64,
        tx_hash: String,
        gas_used: u64,
    },
    /// Round completed successfully.
    RoundComplete {
        round_id: u64,
        new_model_hash: [u8; 32],
        num_contributors: usize,
        duration: Duration,
    },
    /// Round failed.
    RoundFailed {
        round_id: u64,
        phase: JobPhase,
        reason: String,
    },
}

/// Error types for the job manager.
#[derive(Debug, Clone)]
pub enum JobError {
    /// Not enough workers to meet quorum.
    InsufficientQuorum { have: usize, need: usize },
    /// Phase timeout expired.
    PhaseTimeout { phase: JobPhase, elapsed: Duration },
    /// Aggregation produced no result.
    AggregationFailed(String),
    /// On-chain submission failed.
    OnChainSubmissionFailed(String),
    /// Network error.
    NetworkError(String),
    /// Worker-related failure.
    WorkerFailure(String),
    /// Job was cancelled externally.
    Cancelled,
    /// Invalid state transition.
    InvalidTransition { from: JobPhase, to: JobPhase },
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientQuorum { have, need } => {
                write!(f, "insufficient quorum: {}/{} workers ready", have, need)
            }
            Self::PhaseTimeout { phase, elapsed } => {
                write!(f, "{} phase timed out after {:.1}s", phase, elapsed.as_secs_f64())
            }
            Self::AggregationFailed(msg) => write!(f, "aggregation failed: {}", msg),
            Self::OnChainSubmissionFailed(msg) => write!(f, "on-chain submission failed: {}", msg),
            Self::NetworkError(msg) => write!(f, "network error: {}", msg),
            Self::WorkerFailure(msg) => write!(f, "worker failure: {}", msg),
            Self::Cancelled => write!(f, "job cancelled"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition: {} -> {}", from, to)
            }
        }
    }
}

impl std::error::Error for JobError {}

/// Result of a completed training round.
#[derive(Debug, Clone)]
pub struct RoundResult {
    /// Round ID.
    pub round_id: u64,
    /// New aggregated model (after FedAvg).
    pub aggregated_model: MlpModel,
    /// New model hash.
    pub model_hash: [u8; 32],
    /// Number of contributing workers.
    pub num_contributors: usize,
    /// Workers excluded by Byzantine filtering or validation.
    pub excluded_workers: Vec<PeerId>,
    /// Total error bound from aggregation.
    pub total_error_bound: f64,
    /// On-chain transaction hash (if submitted).
    pub tx_hash: Option<String>,
    /// Wall-clock duration of the round.
    pub duration: Duration,
}

// ============================================================================
// TrainingJobManager
// ============================================================================

/// Orchestrates the complete lifecycle of training rounds.
///
/// The `TrainingJobManager` is created by the aggregator node and driven
/// by the runtime's main event loop. It coordinates workers, manages
/// timeouts, handles failures, and submits results on-chain.
pub struct TrainingJobManager {
    /// Our peer ID.
    local_id: PeerId,
    /// Job configuration.
    config: TrainingJobConfig,
    /// Network runner for P2P communication.
    network: Arc<NetworkRunner>,
    /// On-chain pipeline (None if offline mode).
    on_chain_pipeline: Option<Arc<OnChainPipeline>>,
    /// Current job phase.
    phase: JobPhase,
    /// Current round number (incremented each round).
    round_id: u64,
    /// When the current phase started.
    phase_started: Instant,
    /// When the job started.
    job_started: Instant,
    /// Workers participating in this round.
    workers: HashMap<PeerId, WorkerJobState>,
    /// Current model being trained.
    current_model: MlpModel,
    /// Serialized checkpoint of current model.
    current_checkpoint_bytes: Vec<u8>,
    /// Current model hash.
    current_model_hash: [u8; 32],
    /// Event sender for lifecycle events.
    event_tx: broadcast::Sender<JobEvent>,
    /// Cancel signal receiver.
    cancel_rx: broadcast::Receiver<()>,
    /// Cancel signal sender (held to keep channel alive).
    cancel_tx: broadcast::Sender<()>,
    /// Workers that were excluded during this round.
    excluded_workers: Vec<PeerId>,
    /// On-chain tx hash from submission (if any).
    tx_hash: Option<String>,
    /// Failure reason (populated when entering Failed phase).
    failure_reason: Option<String>,
}

impl TrainingJobManager {
    /// Creates a new training job manager.
    pub fn new(
        local_id: PeerId,
        config: TrainingJobConfig,
        network: Arc<NetworkRunner>,
        on_chain_pipeline: Option<Arc<OnChainPipeline>>,
        model: MlpModel,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        let (cancel_tx, cancel_rx) = broadcast::channel(1);

        let checkpoint = model.to_checkpoint(0);
        let checkpoint_bytes = checkpoint.to_bytes().unwrap_or_default();
        let model_hash = model.commitment();

        Self {
            local_id,
            config,
            network,
            on_chain_pipeline,
            phase: JobPhase::Idle,
            round_id: 0,
            phase_started: Instant::now(),
            job_started: Instant::now(),
            workers: HashMap::new(),
            current_model: model,
            current_checkpoint_bytes: checkpoint_bytes,
            current_model_hash: model_hash,
            event_tx,
            cancel_rx,
            cancel_tx,
            excluded_workers: Vec::new(),
            tx_hash: None,
            failure_reason: None,
        }
    }

    /// Returns a clone of the cancel sender so the caller can cancel the job.
    pub fn cancel_handle(&self) -> broadcast::Sender<()> {
        self.cancel_tx.clone()
    }

    /// Subscribes to job lifecycle events.
    pub fn subscribe_events(&self) -> broadcast::Receiver<JobEvent> {
        self.event_tx.subscribe()
    }

    /// Returns a snapshot of the current job state.
    pub fn snapshot(&self) -> JobSnapshot {
        let workers_ready = self.workers.values()
            .filter(|w| w.ready_at.is_some())
            .count();
        let proofs_received = self.workers.values()
            .filter(|w| w.proof_submitted_at.is_some())
            .count();

        JobSnapshot {
            phase: self.phase.clone(),
            round_id: self.round_id,
            phase_started: self.phase_started,
            job_started: self.job_started,
            workers_ready,
            proofs_received,
            total_workers: self.workers.len(),
            excluded_workers: self.excluded_workers.clone(),
            tx_hash: self.tx_hash.clone(),
            failure_reason: self.failure_reason.clone(),
        }
    }

    /// Returns the current phase.
    pub fn phase(&self) -> &JobPhase {
        &self.phase
    }

    /// Returns the current round ID.
    pub fn round_id(&self) -> u64 {
        self.round_id
    }

    /// Returns a reference to the current model.
    pub fn current_model(&self) -> &MlpModel {
        &self.current_model
    }

    /// Returns the current model hash.
    pub fn current_model_hash(&self) -> [u8; 32] {
        self.current_model_hash
    }

    // ========================================================================
    // State Transitions
    // ========================================================================

    fn transition_to(&mut self, new_phase: JobPhase) {
        let old_phase = self.phase.clone();
        info!(
            "Job round={} transitioning: {} -> {}",
            self.round_id, old_phase, new_phase
        );
        self.phase = new_phase.clone();
        self.phase_started = Instant::now();
        let _ = self.event_tx.send(JobEvent::PhaseChanged {
            round_id: self.round_id,
            from: old_phase,
            to: new_phase,
        });
    }

    fn fail(&mut self, error: JobError) {
        let reason = error.to_string();
        error!(
            "Job round={} failed in phase {}: {}",
            self.round_id, self.phase, reason
        );
        self.failure_reason = Some(reason.clone());
        let phase = self.phase.clone();
        self.transition_to(JobPhase::Failed);
        let _ = self.event_tx.send(JobEvent::RoundFailed {
            round_id: self.round_id,
            phase,
            reason,
        });
    }

    // ========================================================================
    // Main Run Loop
    // ========================================================================

    /// Runs the complete training job lifecycle.
    ///
    /// This is the primary entry point. It runs `total_rounds` sequential
    /// training rounds, each going through the full lifecycle. Returns the
    /// result of the final round, or the error that caused failure.
    ///
    /// Uses a channel-based message receiver for incoming worker messages.
    /// The caller is responsible for forwarding relevant `RoundManagementMessage`
    /// and `GradientMessage` events from the network to the returned sender.
    pub async fn run(
        &mut self,
        mut message_rx: mpsc::Receiver<WorkerMessage>,
    ) -> Result<RoundResult, JobError> {
        info!(
            "TrainingJobManager starting: {} rounds, min_workers={}, model_dims={}x{}x{}",
            self.config.total_rounds,
            self.config.min_workers,
            self.config.model_dims.d_in,
            self.config.model_dims.d_hid,
            self.config.model_dims.d_out,
        );

        let total_rounds = self.config.total_rounds.max(1);
        let mut last_result: Option<RoundResult> = None;

        for round_num in 0..total_rounds {
            self.round_id = round_num + 1;
            self.workers.clear();
            self.excluded_workers.clear();
            self.tx_hash = None;
            self.failure_reason = None;
            self.job_started = Instant::now();

            info!("=== Starting round {}/{} ===", self.round_id, total_rounds);

            match self.run_single_round(&mut message_rx).await {
                Ok(result) => {
                    info!(
                        "Round {} completed: {} contributors, error_bound={:.6}, duration={:.1}s",
                        self.round_id,
                        result.num_contributors,
                        result.total_error_bound,
                        result.duration.as_secs_f64(),
                    );
                    // Update model for next round
                    self.current_model = result.aggregated_model.clone();
                    self.current_model_hash = result.model_hash;
                    let ckpt = self.current_model.to_checkpoint(self.round_id);
                    self.current_checkpoint_bytes = ckpt.to_bytes().unwrap_or_default();
                    last_result = Some(result);
                }
                Err(e) => {
                    error!("Round {} failed: {}", self.round_id, e);
                    return Err(e);
                }
            }
        }

        last_result.ok_or(JobError::AggregationFailed(
            "no rounds completed".to_string(),
        ))
    }

    /// Runs a single training round through all phases.
    async fn run_single_round(
        &mut self,
        message_rx: &mut mpsc::Receiver<WorkerMessage>,
    ) -> Result<RoundResult, JobError> {
        // Phase 1: Announce
        self.phase_announcing().await?;

        // Phase 2: Wait for quorum
        self.phase_wait_for_quorum(message_rx).await?;

        // Phase 3: Distribute model weights
        self.phase_distribute().await?;

        // Phase 4: Training — collect proofs from workers
        self.phase_training(message_rx).await?;

        // Phase 5: Aggregate
        let aggregation_result = self.phase_aggregate().await?;

        // Phase 6: Submit on-chain
        self.phase_submit(&aggregation_result).await?;

        // Phase 7: Finalize — broadcast results, distribute updated weights
        let result = self.phase_finalize(aggregation_result).await?;

        Ok(result)
    }

    // ========================================================================
    // Phase 1: Announcing
    // ========================================================================

    async fn phase_announcing(&mut self) -> Result<(), JobError> {
        self.transition_to(JobPhase::Announcing);

        let deadline = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + self.config.quorum_timeout.as_secs();

        let round_configure = RoundManagementMessage::RoundConfigure {
            round_id: self.round_id,
            model_id: self.config.model_id.unwrap_or(0),
            model_dims: self.config.model_dims.clone(),
            dataset_ref: self.config.dataset_ref.clone(),
            steps_per_worker: self.config.steps_per_worker,
            learning_rate: self.config.learning_rate,
            error_budget: self.config.error_budget,
            min_workers: self.config.min_workers as u32,
            deadline,
            current_model_hash: self.current_model_hash,
            dataset_spec: None,
        };

        info!(
            "Round {}: announcing to network (min_workers={}, deadline={}s from now)",
            self.round_id,
            self.config.min_workers,
            self.config.quorum_timeout.as_secs(),
        );

        self.network.broadcast(
            MessagePayload::RoundManagement(round_configure),
        ).await;

        // Also broadcast via TrainingMessage for backward compatibility
        let params = TrainingParams {
            learning_rate: self.config.learning_rate,
            batch_size: 32,
            local_epochs: self.config.steps_per_worker,
            max_error_bound: self.config.error_budget,
            d_in: self.config.model_dims.d_in,
            d_hid: self.config.model_dims.d_hid,
            d_out: self.config.model_dims.d_out,
            model_seed: 42,
            num_layers: self.config.model_dims.num_layers,
            activation_type: self.config.model_dims.activation_type,
        };

        self.network.broadcast(
            MessagePayload::Training(TrainingMessage::RoundStart {
                round_id: self.round_id,
                model_hash: self.current_model_hash,
                params,
            }),
        ).await;

        Ok(())
    }

    // ========================================================================
    // Phase 2: Waiting for Quorum
    // ========================================================================

    async fn phase_wait_for_quorum(
        &mut self,
        message_rx: &mut mpsc::Receiver<WorkerMessage>,
    ) -> Result<(), JobError> {
        self.transition_to(JobPhase::WaitingForQuorum);

        let deadline = Instant::now() + self.config.quorum_timeout;
        let min_workers = self.config.min_workers;

        info!(
            "Round {}: waiting for quorum ({} workers, timeout={:.0}s)",
            self.round_id,
            min_workers,
            self.config.quorum_timeout.as_secs_f64(),
        );

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let ready_count = self.ready_worker_count();
                if ready_count >= min_workers {
                    // Deadline passed but we have enough workers
                    break;
                }
                self.fail(JobError::InsufficientQuorum {
                    have: ready_count,
                    need: min_workers,
                });
                return Err(JobError::InsufficientQuorum {
                    have: ready_count,
                    need: min_workers,
                });
            }

            // Check for cancellation
            if self.cancel_rx.try_recv().is_ok() {
                self.fail(JobError::Cancelled);
                return Err(JobError::Cancelled);
            }

            tokio::select! {
                msg = message_rx.recv() => {
                    match msg {
                        Some(worker_msg) => self.handle_quorum_message(worker_msg),
                        None => {
                            self.fail(JobError::NetworkError("message channel closed".to_string()));
                            return Err(JobError::NetworkError("message channel closed".to_string()));
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    // Periodic check
                }
            }

            // Check if quorum met
            let ready_count = self.ready_worker_count();
            if ready_count >= min_workers {
                info!(
                    "Round {}: quorum met ({}/{} workers ready)",
                    self.round_id, ready_count, min_workers
                );
                // Give a short grace period for additional workers
                let grace_end = Instant::now() + Duration::from_secs(5);
                while Instant::now() < grace_end && Instant::now() < deadline {
                    tokio::select! {
                        msg = message_rx.recv() => {
                            if let Some(worker_msg) = msg {
                                self.handle_quorum_message(worker_msg);
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                    }

                    if self.ready_worker_count() >= self.config.max_workers {
                        break;
                    }
                }
                break;
            }
        }

        let final_count = self.ready_worker_count();
        info!(
            "Round {}: quorum phase complete ({} workers ready)",
            self.round_id, final_count
        );

        Ok(())
    }

    fn handle_quorum_message(&mut self, msg: WorkerMessage) {
        match msg {
            WorkerMessage::WorkerReady { peer_id, round_id, ready, reason } => {
                if round_id != self.round_id {
                    debug!(
                        "Ignoring WorkerReady for wrong round (got={}, expected={})",
                        round_id, self.round_id
                    );
                    return;
                }

                if !ready {
                    info!(
                        "Worker {} declined round {}: {}",
                        peer_id,
                        round_id,
                        reason.unwrap_or_else(|| "no reason".to_string())
                    );
                    return;
                }

                if self.workers.len() >= self.config.max_workers {
                    debug!(
                        "Worker {} ready but max_workers={} reached",
                        peer_id, self.config.max_workers
                    );
                    return;
                }

                if self.workers.contains_key(&peer_id) {
                    debug!("Worker {} already registered, ignoring duplicate", peer_id);
                    return;
                }

                let mut state = WorkerJobState::new(peer_id.clone());
                state.ready_at = Some(Instant::now());
                self.workers.insert(peer_id.clone(), state);

                let workers_ready = self.ready_worker_count();
                info!(
                    "Round {}: worker {} ready ({}/{} workers)",
                    self.round_id, peer_id, workers_ready, self.config.min_workers
                );

                let _ = self.event_tx.send(JobEvent::WorkerReady {
                    round_id: self.round_id,
                    peer_id,
                    workers_ready,
                    workers_needed: self.config.min_workers,
                });
            }
            WorkerMessage::ParticipateRequest { peer_id, round_id } => {
                if round_id != self.round_id {
                    return;
                }

                if self.workers.len() >= self.config.max_workers {
                    return;
                }

                if !self.workers.contains_key(&peer_id) {
                    let mut state = WorkerJobState::new(peer_id.clone());
                    state.ready_at = Some(Instant::now());
                    self.workers.insert(peer_id.clone(), state);

                    let workers_ready = self.ready_worker_count();
                    info!(
                        "Round {}: worker {} joined via ParticipateRequest ({}/{})",
                        self.round_id, peer_id, workers_ready, self.config.min_workers
                    );

                    let _ = self.event_tx.send(JobEvent::WorkerReady {
                        round_id: self.round_id,
                        peer_id,
                        workers_ready,
                        workers_needed: self.config.min_workers,
                    });
                }
            }
            // Other message types are queued for later phases
            _ => {}
        }
    }

    fn ready_worker_count(&self) -> usize {
        self.workers.values().filter(|w| w.ready_at.is_some()).count()
    }

    // ========================================================================
    // Phase 3: Distributing Model Weights
    // ========================================================================

    async fn phase_distribute(&mut self) -> Result<(), JobError> {
        self.transition_to(JobPhase::Distributing);

        let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
        let num_workers = worker_ids.len();

        info!(
            "Round {}: distributing model weights to {} workers (checkpoint={} bytes)",
            self.round_id,
            num_workers,
            self.current_checkpoint_bytes.len(),
        );

        // Send model weights to each worker
        for peer_id in &worker_ids {
            let payload = MessagePayload::Training(TrainingMessage::ModelWeights {
                round_id: self.round_id,
                checkpoint_data: self.current_checkpoint_bytes.clone(),
                weight_hash: self.current_model_hash,
            });

            match self.network.send_direct(peer_id, payload).await {
                Ok(()) => {
                    if let Some(worker) = self.workers.get_mut(peer_id) {
                        worker.weights_sent_at = Some(Instant::now());
                    }
                    debug!("Sent model weights to worker {}", peer_id);
                }
                Err(e) => {
                    warn!(
                        "Failed to send model weights to worker {}: {}. Removing from round.",
                        peer_id, e
                    );
                    let _ = self.event_tx.send(JobEvent::WorkerDropped {
                        round_id: self.round_id,
                        peer_id: peer_id.clone(),
                        reason: format!("weight distribution failed: {}", e),
                    });
                }
            }
        }

        // Remove workers that didn't get weights
        let failed_workers: Vec<PeerId> = self.workers.iter()
            .filter(|(_, w)| w.weights_sent_at.is_none())
            .map(|(id, _)| id.clone())
            .collect();

        for peer_id in &failed_workers {
            self.workers.remove(peer_id);
        }

        // Check we still have enough workers
        if self.workers.len() < self.config.min_workers {
            let err = JobError::InsufficientQuorum {
                have: self.workers.len(),
                need: self.config.min_workers,
            };
            self.fail(err.clone());
            return Err(err);
        }

        // Signal workers to begin training
        let participants: Vec<String> = self.workers.keys().map(|p| p.to_string()).collect();

        self.network.broadcast(
            MessagePayload::RoundManagement(RoundManagementMessage::BeginTraining {
                round_id: self.round_id,
                participants,
            }),
        ).await;

        info!(
            "Round {}: model distributed to {} workers, training signaled",
            self.round_id,
            self.workers.len(),
        );

        Ok(())
    }

    // ========================================================================
    // Phase 4: Training (Collect Proofs)
    // ========================================================================

    async fn phase_training(
        &mut self,
        message_rx: &mut mpsc::Receiver<WorkerMessage>,
    ) -> Result<(), JobError> {
        self.transition_to(JobPhase::Training);

        let deadline = Instant::now() + self.config.training_timeout;
        let expected_proofs = self.workers.len();

        info!(
            "Round {}: waiting for {} proofs (timeout={:.0}s)",
            self.round_id,
            expected_proofs,
            self.config.training_timeout.as_secs_f64(),
        );

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Timeout — check if we have enough proofs to proceed
                let received = self.proofs_received_count();
                if received >= self.config.min_workers {
                    warn!(
                        "Round {}: training timeout with {}/{} proofs, proceeding with partial results",
                        self.round_id, received, expected_proofs
                    );
                    // Remove workers that didn't submit
                    self.remove_non_submitting_workers();
                    break;
                }

                let err = JobError::PhaseTimeout {
                    phase: JobPhase::Training,
                    elapsed: self.config.training_timeout,
                };
                self.fail(err.clone());
                return Err(err);
            }

            // Check for cancellation
            if self.cancel_rx.try_recv().is_ok() {
                self.fail(JobError::Cancelled);
                return Err(JobError::Cancelled);
            }

            tokio::select! {
                msg = message_rx.recv() => {
                    match msg {
                        Some(worker_msg) => self.handle_training_message(worker_msg).await,
                        None => {
                            let err = JobError::NetworkError("message channel closed".to_string());
                            self.fail(err.clone());
                            return Err(err);
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    // Periodic check for timeouts
                }
            }

            // Check if all proofs received
            if self.proofs_received_count() >= expected_proofs {
                info!(
                    "Round {}: all {} proofs received",
                    self.round_id, expected_proofs
                );
                break;
            }
        }

        let received = self.proofs_received_count();
        info!(
            "Round {}: training phase complete ({} proofs collected)",
            self.round_id, received
        );

        Ok(())
    }

    async fn handle_training_message(&mut self, msg: WorkerMessage) {
        match msg {
            WorkerMessage::ProofSubmission {
                peer_id,
                round_id,
                proof,
                public_inputs,
                error_bound,
                steps_completed,
                new_model_hash,
                checkpoint_data,
            } => {
                if round_id != self.round_id {
                    debug!(
                        "Ignoring proof from {} for wrong round (got={}, expected={})",
                        peer_id, round_id, self.round_id
                    );
                    return;
                }

                if !self.workers.contains_key(&peer_id) {
                    warn!(
                        "Proof from unregistered worker {} for round {}, ignoring",
                        peer_id, round_id
                    );
                    return;
                }

                // Validate error bound
                if !error_bound.is_finite() || error_bound > self.config.error_budget {
                    warn!(
                        "Rejecting proof from {}: error_bound={:.6} exceeds budget={:.6}",
                        peer_id, error_bound, self.config.error_budget
                    );
                    let _ = self.event_tx.send(JobEvent::ProofRejected {
                        round_id: self.round_id,
                        peer_id: peer_id.clone(),
                        reason: format!(
                            "error_bound {:.6} exceeds budget {:.6}",
                            error_bound, self.config.error_budget
                        ),
                    });
                    return;
                }

                // Validate proof is non-empty
                if proof.is_empty() {
                    warn!("Rejecting empty proof from {} for round {}", peer_id, round_id);
                    let _ = self.event_tx.send(JobEvent::ProofRejected {
                        round_id: self.round_id,
                        peer_id: peer_id.clone(),
                        reason: "empty proof".to_string(),
                    });
                    return;
                }

                // Store proof data
                if let Some(worker) = self.workers.get_mut(&peer_id) {
                    worker.proof_submitted_at = Some(Instant::now());
                    worker.proof_valid = Some(true);
                    worker.proof = Some(proof);
                    worker.public_inputs = Some(public_inputs);
                    worker.error_bound = Some(error_bound);
                    worker.new_model_hash = Some(new_model_hash);
                    worker.checkpoint_data = Some(checkpoint_data);
                    worker.steps_completed = steps_completed;
                }

                let proofs_received = self.proofs_received_count();
                let proofs_expected = self.workers.len();

                info!(
                    "Round {}: proof from {} accepted ({}/{} proofs, error_bound={:.6})",
                    self.round_id, peer_id, proofs_received, proofs_expected, error_bound
                );

                let _ = self.event_tx.send(JobEvent::ProofReceived {
                    round_id: self.round_id,
                    peer_id,
                    proofs_received,
                    proofs_expected,
                });
            }
            WorkerMessage::WeightUpdate {
                peer_id,
                round_id,
                checkpoint_data,
                weight_hash,
            } => {
                if round_id != self.round_id {
                    return;
                }

                if let Some(worker) = self.workers.get_mut(&peer_id) {
                    if worker.checkpoint_data.is_none() {
                        worker.checkpoint_data = Some(checkpoint_data);
                        worker.new_model_hash = Some(weight_hash);
                        worker.proof_submitted_at = Some(Instant::now());
                        worker.proof_valid = Some(true);
                        // Set a placeholder proof to mark submission
                        if worker.proof.is_none() {
                            worker.proof = Some(vec![0u8; 1]);
                        }
                        info!(
                            "Round {}: weight update from {} ({}/{})",
                            self.round_id, peer_id,
                            self.proofs_received_count(), self.workers.len(),
                        );
                    }
                }
            }
            // Handle late-arriving quorum messages
            WorkerMessage::WorkerReady { peer_id, round_id, ready, .. } => {
                if round_id == self.round_id && ready && !self.workers.contains_key(&peer_id) {
                    if self.workers.len() < self.config.max_workers {
                        debug!(
                            "Late worker {} joining round {} during training phase",
                            peer_id, round_id
                        );
                        // Don't add them — they missed weight distribution
                    }
                }
            }
            _ => {}
        }
    }

    fn proofs_received_count(&self) -> usize {
        self.workers.values()
            .filter(|w| w.proof_submitted_at.is_some())
            .count()
    }

    fn remove_non_submitting_workers(&mut self) {
        let non_submitters: Vec<PeerId> = self.workers.iter()
            .filter(|(_, w)| w.proof_submitted_at.is_none())
            .map(|(id, _)| id.clone())
            .collect();

        for peer_id in &non_submitters {
            warn!(
                "Round {}: removing non-submitting worker {}",
                self.round_id, peer_id
            );
            self.workers.remove(peer_id);
            let _ = self.event_tx.send(JobEvent::WorkerDropped {
                round_id: self.round_id,
                peer_id: peer_id.clone(),
                reason: "did not submit proof within timeout".to_string(),
            });
        }
    }

    // ========================================================================
    // Phase 5: Aggregation
    // ========================================================================

    async fn phase_aggregate(&mut self) -> Result<AggregationOutput, JobError> {
        self.transition_to(JobPhase::Aggregating);

        let submitters: Vec<(PeerId, WorkerJobState)> = self.workers.iter()
            .filter(|(_, w)| w.checkpoint_data.is_some())
            .map(|(id, w)| (id.clone(), w.clone()))
            .collect();

        if submitters.is_empty() {
            let err = JobError::AggregationFailed("no worker checkpoints to aggregate".to_string());
            self.fail(err.clone());
            return Err(err);
        }

        info!(
            "Round {}: aggregating {} worker submissions",
            self.round_id, submitters.len()
        );

        // FedAvg: deserialize worker models and average
        let mut worker_models = Vec::new();
        let mut failed_deserialize = Vec::new();

        for (peer_id, worker) in &submitters {
            if let Some(ref ckpt_bytes) = worker.checkpoint_data {
                match ModelCheckpoint::from_bytes(ckpt_bytes) {
                    Ok(ckpt) => match MlpModel::from_checkpoint(&ckpt) {
                        Ok(model) => {
                            worker_models.push((peer_id.clone(), model));
                        }
                        Err(e) => {
                            warn!(
                                "Round {}: failed to reconstruct model from {}: {}",
                                self.round_id, peer_id, e
                            );
                            failed_deserialize.push(peer_id.clone());
                        }
                    },
                    Err(e) => {
                        warn!(
                            "Round {}: failed to deserialize checkpoint from {}: {}",
                            self.round_id, peer_id, e
                        );
                        failed_deserialize.push(peer_id.clone());
                    }
                }
            }
        }

        // Track excluded workers
        self.excluded_workers.extend(failed_deserialize);

        if worker_models.is_empty() {
            let err = JobError::AggregationFailed(
                "all worker checkpoints failed deserialization".to_string(),
            );
            self.fail(err.clone());
            return Err(err);
        }

        // Compute FedAvg
        let models: Vec<MlpModel> = worker_models.iter().map(|(_, m)| m.clone()).collect();
        let aggregated = match average_models(&models) {
            Ok(model) => model,
            Err(e) => {
                let err = JobError::AggregationFailed(format!("FedAvg failed: {}", e));
                self.fail(err.clone());
                return Err(err);
            }
        };

        let new_model_hash = aggregated.commitment();
        let total_error_bound: f64 = submitters.iter()
            .filter(|(id, _)| !self.excluded_workers.contains(id))
            .filter_map(|(_, w)| w.error_bound)
            .sum::<f64>()
            / worker_models.len().max(1) as f64;

        let contributors: Vec<PeerId> = worker_models.iter().map(|(id, _)| id.clone()).collect();

        info!(
            "Round {}: FedAvg aggregation complete ({} contributors, error_bound={:.6})",
            self.round_id,
            contributors.len(),
            total_error_bound,
        );

        let _ = self.event_tx.send(JobEvent::AggregationComplete {
            round_id: self.round_id,
            num_contributors: contributors.len(),
            num_excluded: self.excluded_workers.len(),
            total_error_bound,
        });

        Ok(AggregationOutput {
            aggregated_model: aggregated,
            model_hash: new_model_hash,
            contributors,
            total_error_bound,
        })
    }

    // ========================================================================
    // Phase 6: On-Chain Submission
    // ========================================================================

    async fn phase_submit(
        &mut self,
        aggregation: &AggregationOutput,
    ) -> Result<(), JobError> {
        if !self.config.submit_on_chain {
            debug!("Round {}: on-chain submission disabled, skipping", self.round_id);
            return Ok(());
        }

        let pipeline = match &self.on_chain_pipeline {
            Some(p) => Arc::clone(p),
            None => {
                debug!("Round {}: no on-chain pipeline configured, skipping", self.round_id);
                return Ok(());
            }
        };

        self.transition_to(JobPhase::Submitting);

        info!(
            "Round {}: submitting aggregated proof on-chain ({} contributors)",
            self.round_id,
            aggregation.contributors.len(),
        );

        // Start the round on-chain if not already started
        match pipeline.start_round_on_chain().await {
            Ok(on_chain_round) => {
                info!(
                    "Round {}: on-chain round {} started",
                    self.round_id, on_chain_round
                );
            }
            Err(e) => {
                warn!(
                    "Round {}: failed to start on-chain round (may already exist): {}",
                    self.round_id, e
                );
                // Continue — round may already be active
            }
        }

        // Build TrainingProofResultV2 entries from worker submissions
        // For now, use the aggregated commitment as the submission
        // The OnChainPipeline handles RLC aggregation internally
        let worker_proofs = self.build_worker_proofs();

        if worker_proofs.is_empty() {
            warn!("Round {}: no valid worker proofs for on-chain submission", self.round_id);
            return Ok(());
        }

        match pipeline.submit_aggregated_round(self.round_id, worker_proofs).await {
            Ok(Some(submission)) => {
                self.tx_hash = Some(submission.tx_hash.clone());
                info!(
                    "Round {}: on-chain submission succeeded (tx={}, gas={}, proofs={})",
                    self.round_id,
                    submission.tx_hash,
                    submission.gas_used,
                    submission.proofs_aggregated,
                );
                let _ = self.event_tx.send(JobEvent::OnChainSubmitted {
                    round_id: self.round_id,
                    tx_hash: submission.tx_hash,
                    gas_used: submission.gas_used,
                });
            }
            Ok(None) => {
                info!(
                    "Round {}: no on-chain submission produced (proofs may be empty)",
                    self.round_id
                );
            }
            Err(e) => {
                let err = JobError::OnChainSubmissionFailed(e.to_string());
                // Don't fail the round for on-chain issues — the training result is still valid
                warn!("Round {}: on-chain submission failed: {}", self.round_id, e);
                let _ = self.event_tx.send(JobEvent::RoundFailed {
                    round_id: self.round_id,
                    phase: JobPhase::Submitting,
                    reason: err.to_string(),
                });
            }
        }

        Ok(())
    }

    fn build_worker_proofs(&self) -> Vec<helix_prover::TrainingProofResultV2> {
        use helix_prover::TrainingProofResultV2;
        use halo2curves::bn256::Fr;
        use halo2curves::ff::Field;

        let mut proofs = Vec::new();

        for (_, worker) in &self.workers {
            if let (Some(ref proof), Some(ref public_inputs)) = (&worker.proof, &worker.public_inputs) {
                if proof.len() <= 1 {
                    // Placeholder proof, skip
                    continue;
                }

                // Convert [u8; 32] public inputs to Fr via from_raw
                let pi_fr: Vec<Fr> = public_inputs.iter()
                    .map(|bytes| {
                        let mut repr = [0u64; 4];
                        for (i, chunk) in bytes.chunks(8).enumerate().take(4) {
                            repr[i] = u64::from_le_bytes(chunk.try_into().unwrap_or([0u8; 8]));
                        }
                        Fr::from_raw(repr)
                    })
                    .collect();

                if pi_fr.len() >= 8 {
                    proofs.push(TrainingProofResultV2 {
                        proof: proof.clone(),
                        public_inputs: pi_fr.clone(),
                        loss: pi_fr.get(4).copied().unwrap_or(Fr::ZERO),
                        total_error: pi_fr.get(5).copied().unwrap_or(Fr::ZERO),
                        step_number: worker.steps_completed as u64,
                        old_state_hash: (
                            pi_fr.get(0).copied().unwrap_or(Fr::ZERO),
                            pi_fr.get(1).copied().unwrap_or(Fr::ZERO),
                        ),
                        new_state_hash: (
                            pi_fr.get(2).copied().unwrap_or(Fr::ZERO),
                            pi_fr.get(3).copied().unwrap_or(Fr::ZERO),
                        ),
                        verified: false,
                        generation_time: Duration::from_secs(0),
                        verification_time: None,
                        attempts: 1,
                        from_cache: false,
                        witness_hash: None,
                    });
                }
            }
        }

        proofs
    }

    // ========================================================================
    // Phase 7: Finalization
    // ========================================================================

    async fn phase_finalize(
        &mut self,
        aggregation: AggregationOutput,
    ) -> Result<RoundResult, JobError> {
        self.transition_to(JobPhase::Finalizing);

        let duration = self.job_started.elapsed();

        // Update current model
        self.current_model = aggregation.aggregated_model.clone();
        self.current_model_hash = aggregation.model_hash;
        let ckpt = self.current_model.to_checkpoint(self.round_id);
        self.current_checkpoint_bytes = ckpt.to_bytes().unwrap_or_default();

        // Broadcast updated weights to all workers
        self.network.broadcast(
            MessagePayload::Training(TrainingMessage::UpdatedWeights {
                round_id: self.round_id,
                checkpoint_data: self.current_checkpoint_bytes.clone(),
                weight_hash: self.current_model_hash,
            }),
        ).await;

        // Broadcast round completion via RoundManagementMessage
        let tx_hash_bytes: Option<[u8; 32]> = self.tx_hash.as_ref().and_then(|h| {
            let hex_str = h.strip_prefix("0x").unwrap_or(h);
            let bytes = hex::decode(hex_str).ok()?;
            if bytes.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                Some(arr)
            } else {
                None
            }
        });

        self.network.broadcast(
            MessagePayload::RoundManagement(RoundManagementMessage::RoundCompleted {
                round_id: self.round_id,
                final_commitment: aggregation.model_hash,
                tx_hash: tx_hash_bytes,
                new_model_hash: aggregation.model_hash,
                total_error_bound: aggregation.total_error_bound,
                num_contributors: aggregation.contributors.len() as u32,
            }),
        ).await;

        // Also broadcast via TrainingMessage for backward compatibility
        self.network.broadcast(
            MessagePayload::Training(TrainingMessage::RoundComplete {
                round_id: self.round_id,
                result_hash: aggregation.model_hash,
            }),
        ).await;

        info!(
            "Round {}: finalized (model_hash={}, contributors={}, duration={:.1}s)",
            self.round_id,
            hex::encode(&aggregation.model_hash[..8]),
            aggregation.contributors.len(),
            duration.as_secs_f64(),
        );

        let result = RoundResult {
            round_id: self.round_id,
            aggregated_model: aggregation.aggregated_model,
            model_hash: aggregation.model_hash,
            num_contributors: aggregation.contributors.len(),
            excluded_workers: self.excluded_workers.clone(),
            total_error_bound: aggregation.total_error_bound,
            tx_hash: self.tx_hash.clone(),
            duration,
        };

        let _ = self.event_tx.send(JobEvent::RoundComplete {
            round_id: self.round_id,
            new_model_hash: result.model_hash,
            num_contributors: result.num_contributors,
            duration,
        });

        self.transition_to(JobPhase::Complete);

        Ok(result)
    }
}

// ============================================================================
// Internal Types
// ============================================================================

/// Output of the aggregation phase.
#[derive(Debug, Clone)]
struct AggregationOutput {
    /// The FedAvg-aggregated model.
    aggregated_model: MlpModel,
    /// Hash of the aggregated model.
    model_hash: [u8; 32],
    /// Workers who contributed.
    contributors: Vec<PeerId>,
    /// Average error bound.
    total_error_bound: f64,
}

// ============================================================================
// Message Types
// ============================================================================

/// Messages forwarded from the network to the job manager.
///
/// The runtime's event loop receives `NetworkEvent` variants and converts
/// relevant ones into `WorkerMessage` before sending to the job manager.
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    /// Worker acknowledges round configuration.
    WorkerReady {
        peer_id: PeerId,
        round_id: u64,
        ready: bool,
        reason: Option<String>,
    },
    /// Worker requests to participate (legacy path).
    ParticipateRequest {
        peer_id: PeerId,
        round_id: u64,
    },
    /// Worker submits proof of training.
    ProofSubmission {
        peer_id: PeerId,
        round_id: u64,
        proof: Vec<u8>,
        public_inputs: Vec<[u8; 32]>,
        error_bound: f64,
        steps_completed: u32,
        new_model_hash: [u8; 32],
        checkpoint_data: Vec<u8>,
    },
    /// Worker sends weight update (FedAvg path without ZK proofs).
    WeightUpdate {
        peer_id: PeerId,
        round_id: u64,
        checkpoint_data: Vec<u8>,
        weight_hash: [u8; 32],
    },
    /// Worker registration message.
    Registration {
        peer_id: PeerId,
        message: RegistrationMessage,
    },
}

/// Converts network events into worker messages for the job manager.
///
/// This function is called by the runtime to bridge the network layer
/// with the job manager's message channel. Returns `None` for events
/// that aren't relevant to the job manager.
pub fn network_event_to_worker_message(
    from: PeerId,
    payload: &MessagePayload,
) -> Option<WorkerMessage> {
    match payload {
        MessagePayload::RoundManagement(rm) => match rm {
            RoundManagementMessage::WorkerReady { round_id, ready, reason } => {
                Some(WorkerMessage::WorkerReady {
                    peer_id: from,
                    round_id: *round_id,
                    ready: *ready,
                    reason: reason.clone(),
                })
            }
            RoundManagementMessage::ProofSubmission {
                round_id,
                proof,
                public_inputs,
                error_bound,
                steps_completed,
                new_model_hash,
                checkpoint_data,
            } => {
                Some(WorkerMessage::ProofSubmission {
                    peer_id: from,
                    round_id: *round_id,
                    proof: proof.clone(),
                    public_inputs: public_inputs.clone(),
                    error_bound: *error_bound,
                    steps_completed: *steps_completed,
                    new_model_hash: *new_model_hash,
                    checkpoint_data: checkpoint_data.clone(),
                })
            }
            _ => None,
        },
        MessagePayload::Training(tm) => match tm {
            TrainingMessage::ParticipateRequest { round_id } => {
                Some(WorkerMessage::ParticipateRequest {
                    peer_id: from,
                    round_id: *round_id,
                })
            }
            _ => None,
        },
        MessagePayload::Gradient(gm) => match gm {
            crate::network::messages::GradientMessage::WeightUpdate {
                round_id,
                checkpoint_data,
                weight_hash,
            } => {
                Some(WorkerMessage::WeightUpdate {
                    peer_id: from,
                    round_id: *round_id,
                    checkpoint_data: checkpoint_data.clone(),
                    weight_hash: *weight_hash,
                })
            }
            _ => None,
        },
        MessagePayload::Registration(reg) => {
            Some(WorkerMessage::Registration {
                peer_id: from,
                message: reg.clone(),
            })
        }
        _ => None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_phase_display() {
        assert_eq!(format!("{}", JobPhase::Idle), "Idle");
        assert_eq!(format!("{}", JobPhase::Announcing), "Announcing");
        assert_eq!(format!("{}", JobPhase::WaitingForQuorum), "WaitingForQuorum");
        assert_eq!(format!("{}", JobPhase::Distributing), "Distributing");
        assert_eq!(format!("{}", JobPhase::Training), "Training");
        assert_eq!(format!("{}", JobPhase::Aggregating), "Aggregating");
        assert_eq!(format!("{}", JobPhase::Submitting), "Submitting");
        assert_eq!(format!("{}", JobPhase::Finalizing), "Finalizing");
        assert_eq!(format!("{}", JobPhase::Complete), "Complete");
        assert_eq!(format!("{}", JobPhase::Failed), "Failed");
    }

    #[test]
    fn test_job_error_display() {
        let err = JobError::InsufficientQuorum { have: 1, need: 3 };
        assert!(err.to_string().contains("1/3"));

        let err = JobError::PhaseTimeout {
            phase: JobPhase::Training,
            elapsed: Duration::from_secs(300),
        };
        assert!(err.to_string().contains("Training"));
        assert!(err.to_string().contains("300"));

        let err = JobError::Cancelled;
        assert_eq!(err.to_string(), "job cancelled");
    }

    #[test]
    fn test_job_config_default() {
        let config = TrainingJobConfig::default();
        assert_eq!(config.min_workers, 1);
        assert_eq!(config.max_workers, 100);
        assert_eq!(config.steps_per_worker, 10);
        assert!(!config.submit_on_chain);
        assert_eq!(config.total_rounds, 1);
    }

    #[test]
    fn test_job_config_from_node_config() {
        let mut node_config = NodeConfig::default();
        node_config.training.d_in = 10;
        node_config.training.d_hid = 20;
        node_config.training.d_out = 5;
        node_config.training.learning_rate = 0.001;
        node_config.training.min_workers = 3;

        let job_config = TrainingJobConfig::from_node_config(&node_config);
        assert_eq!(job_config.model_dims.d_in, 10);
        assert_eq!(job_config.model_dims.d_hid, 20);
        assert_eq!(job_config.model_dims.d_out, 5);
        assert_eq!(job_config.learning_rate, 0.001);
        assert_eq!(job_config.min_workers, 3);
        assert!(!job_config.submit_on_chain);
    }

    #[test]
    fn test_job_config_from_node_config_with_chain() {
        let mut node_config = NodeConfig::default();
        node_config.chain = Some(ChainConfig {
            coordinator_address: "0x1234".to_string(),
            model_ipfs_hash: "QmTest".to_string(),
            model_id: Some(42),
            min_stake_wei: "1000000000000000000".to_string(),
            stake_amount_wei: "1000000000000000000".to_string(),
            round_duration_secs: 600,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            proof_queue_interval_secs: 5,
        });

        let job_config = TrainingJobConfig::from_node_config(&node_config);
        assert!(job_config.submit_on_chain);
        assert_eq!(job_config.model_id, Some(42));
    }

    #[test]
    fn test_worker_job_state_new() {
        let peer_id = PeerId::random();
        let state = WorkerJobState::new(peer_id.clone());
        assert_eq!(state.peer_id, peer_id);
        assert!(state.ready_at.is_none());
        assert!(state.weights_sent_at.is_none());
        assert!(state.proof_submitted_at.is_none());
        assert!(state.checkpoint_data.is_none());
        assert_eq!(state.steps_completed, 0);
    }

    #[test]
    fn test_network_event_to_worker_message_worker_ready() {
        let peer_id = PeerId::random();
        let payload = MessagePayload::RoundManagement(
            RoundManagementMessage::WorkerReady {
                round_id: 5,
                ready: true,
                reason: None,
            },
        );

        let msg = network_event_to_worker_message(peer_id.clone(), &payload);
        assert!(msg.is_some());
        match msg.unwrap() {
            WorkerMessage::WorkerReady { peer_id: id, round_id, ready, .. } => {
                assert_eq!(id, peer_id);
                assert_eq!(round_id, 5);
                assert!(ready);
            }
            _ => panic!("expected WorkerReady"),
        }
    }

    #[test]
    fn test_network_event_to_worker_message_proof_submission() {
        let peer_id = PeerId::random();
        let payload = MessagePayload::RoundManagement(
            RoundManagementMessage::ProofSubmission {
                round_id: 3,
                proof: vec![1, 2, 3],
                public_inputs: vec![[0u8; 32]; 8],
                error_bound: 0.05,
                steps_completed: 10,
                new_model_hash: [42u8; 32],
                checkpoint_data: vec![4, 5, 6],
            },
        );

        let msg = network_event_to_worker_message(peer_id.clone(), &payload);
        assert!(msg.is_some());
        match msg.unwrap() {
            WorkerMessage::ProofSubmission { round_id, error_bound, steps_completed, .. } => {
                assert_eq!(round_id, 3);
                assert_eq!(error_bound, 0.05);
                assert_eq!(steps_completed, 10);
            }
            _ => panic!("expected ProofSubmission"),
        }
    }

    #[test]
    fn test_network_event_to_worker_message_participate_request() {
        let peer_id = PeerId::random();
        let payload = MessagePayload::Training(
            TrainingMessage::ParticipateRequest { round_id: 7 },
        );

        let msg = network_event_to_worker_message(peer_id.clone(), &payload);
        assert!(msg.is_some());
        match msg.unwrap() {
            WorkerMessage::ParticipateRequest { round_id, .. } => {
                assert_eq!(round_id, 7);
            }
            _ => panic!("expected ParticipateRequest"),
        }
    }

    #[test]
    fn test_network_event_to_worker_message_irrelevant() {
        let peer_id = PeerId::random();

        // Heartbeat should return None
        let payload = MessagePayload::Heartbeat(crate::network::messages::HeartbeatMessage {
            seq: 1,
            is_pong: false,
            load: 50,
        });
        assert!(network_event_to_worker_message(peer_id.clone(), &payload).is_none());

        // RoundConfigure should return None (aggregator -> worker, not worker -> aggregator)
        let payload = MessagePayload::RoundManagement(
            RoundManagementMessage::RoundConfigure {
                round_id: 1,
                model_id: 0,
                model_dims: ModelDims {
                    d_in: 4, d_hid: 8, d_out: 2,
                    num_layers: 2, num_heads: 0, activation_type: 0,
                },
                dataset_ref: String::new(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                min_workers: 3,
                deadline: 0,
                current_model_hash: [0u8; 32],
                dataset_spec: None,
            },
        );
        assert!(network_event_to_worker_message(peer_id, &payload).is_none());
    }

    #[test]
    fn test_job_snapshot() {
        let peer_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );

        let snapshot = manager.snapshot();
        assert_eq!(snapshot.phase, JobPhase::Idle);
        assert_eq!(snapshot.round_id, 0);
        assert_eq!(snapshot.workers_ready, 0);
        assert_eq!(snapshot.proofs_received, 0);
        assert_eq!(snapshot.total_workers, 0);
        assert!(snapshot.tx_hash.is_none());
        assert!(snapshot.failure_reason.is_none());
    }

    #[test]
    fn test_transition_to() {
        let peer_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );

        let mut event_rx = manager.subscribe_events();

        manager.transition_to(JobPhase::Announcing);
        assert_eq!(*manager.phase(), JobPhase::Announcing);

        // Check event was emitted
        let event = event_rx.try_recv();
        assert!(event.is_ok());
        match event.unwrap() {
            JobEvent::PhaseChanged { from, to, .. } => {
                assert_eq!(from, JobPhase::Idle);
                assert_eq!(to, JobPhase::Announcing);
            }
            _ => panic!("expected PhaseChanged"),
        }
    }

    #[test]
    fn test_fail_transition() {
        let peer_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );

        manager.transition_to(JobPhase::Training);
        manager.fail(JobError::Cancelled);

        assert_eq!(*manager.phase(), JobPhase::Failed);
        assert_eq!(manager.failure_reason.as_deref(), Some("job cancelled"));
    }

    #[test]
    fn test_handle_quorum_message_worker_ready() {
        let peer_id = PeerId::random();
        let worker_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        // Add a ready worker
        manager.handle_quorum_message(WorkerMessage::WorkerReady {
            peer_id: worker_id.clone(),
            round_id: 1,
            ready: true,
            reason: None,
        });

        assert_eq!(manager.ready_worker_count(), 1);
        assert!(manager.workers.contains_key(&worker_id));

        // Wrong round should be ignored
        let worker_id2 = PeerId::random();
        manager.handle_quorum_message(WorkerMessage::WorkerReady {
            peer_id: worker_id2.clone(),
            round_id: 99,
            ready: true,
            reason: None,
        });
        assert_eq!(manager.ready_worker_count(), 1);
        assert!(!manager.workers.contains_key(&worker_id2));
    }

    #[test]
    fn test_handle_quorum_message_not_ready() {
        let peer_id = PeerId::random();
        let worker_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        manager.handle_quorum_message(WorkerMessage::WorkerReady {
            peer_id: worker_id,
            round_id: 1,
            ready: false,
            reason: Some("busy".to_string()),
        });

        assert_eq!(manager.ready_worker_count(), 0);
    }

    #[test]
    fn test_handle_quorum_message_max_workers() {
        let peer_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let mut config = TrainingJobConfig::default();
        config.max_workers = 2;
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        // Add 2 workers (at max)
        for _ in 0..2 {
            manager.handle_quorum_message(WorkerMessage::WorkerReady {
                peer_id: PeerId::random(),
                round_id: 1,
                ready: true,
                reason: None,
            });
        }
        assert_eq!(manager.ready_worker_count(), 2);

        // Third should be rejected
        manager.handle_quorum_message(WorkerMessage::WorkerReady {
            peer_id: PeerId::random(),
            round_id: 1,
            ready: true,
            reason: None,
        });
        assert_eq!(manager.ready_worker_count(), 2);
    }

    #[test]
    fn test_handle_quorum_message_duplicate_worker() {
        let peer_id = PeerId::random();
        let worker_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        // Add worker once
        manager.handle_quorum_message(WorkerMessage::WorkerReady {
            peer_id: worker_id.clone(),
            round_id: 1,
            ready: true,
            reason: None,
        });

        // Add same worker again — should be ignored
        manager.handle_quorum_message(WorkerMessage::WorkerReady {
            peer_id: worker_id,
            round_id: 1,
            ready: true,
            reason: None,
        });

        assert_eq!(manager.ready_worker_count(), 1);
    }

    #[tokio::test]
    async fn test_handle_training_message_proof_submission() {
        let peer_id = PeerId::random();
        let worker_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let mut config = TrainingJobConfig::default();
        config.error_budget = 0.1;
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        // Register worker first
        let mut state = WorkerJobState::new(worker_id.clone());
        state.ready_at = Some(Instant::now());
        state.weights_sent_at = Some(Instant::now());
        manager.workers.insert(worker_id.clone(), state);

        // Submit proof
        manager.handle_training_message(WorkerMessage::ProofSubmission {
            peer_id: worker_id.clone(),
            round_id: 1,
            proof: vec![1, 2, 3, 4],
            public_inputs: vec![[0u8; 32]; 8],
            error_bound: 0.05,
            steps_completed: 10,
            new_model_hash: [42u8; 32],
            checkpoint_data: vec![5, 6, 7, 8],
        }).await;

        assert_eq!(manager.proofs_received_count(), 1);
        let worker = manager.workers.get(&worker_id).unwrap();
        assert!(worker.proof_submitted_at.is_some());
        assert_eq!(worker.error_bound, Some(0.05));
        assert_eq!(worker.steps_completed, 10);
    }

    #[tokio::test]
    async fn test_handle_training_message_reject_invalid_error_bound() {
        let peer_id = PeerId::random();
        let worker_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let mut config = TrainingJobConfig::default();
        config.error_budget = 0.1;
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        let mut state = WorkerJobState::new(worker_id.clone());
        state.ready_at = Some(Instant::now());
        manager.workers.insert(worker_id.clone(), state);

        // Submit proof with error bound exceeding budget
        manager.handle_training_message(WorkerMessage::ProofSubmission {
            peer_id: worker_id.clone(),
            round_id: 1,
            proof: vec![1, 2, 3],
            public_inputs: vec![[0u8; 32]; 8],
            error_bound: 0.5, // Exceeds 0.1 budget
            steps_completed: 10,
            new_model_hash: [42u8; 32],
            checkpoint_data: vec![5, 6, 7],
        }).await;

        assert_eq!(manager.proofs_received_count(), 0);
    }

    #[tokio::test]
    async fn test_handle_training_message_reject_empty_proof() {
        let peer_id = PeerId::random();
        let worker_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        let mut state = WorkerJobState::new(worker_id.clone());
        state.ready_at = Some(Instant::now());
        manager.workers.insert(worker_id.clone(), state);

        // Submit empty proof
        manager.handle_training_message(WorkerMessage::ProofSubmission {
            peer_id: worker_id,
            round_id: 1,
            proof: vec![],
            public_inputs: vec![[0u8; 32]; 8],
            error_bound: 0.05,
            steps_completed: 10,
            new_model_hash: [42u8; 32],
            checkpoint_data: vec![5, 6, 7],
        }).await;

        assert_eq!(manager.proofs_received_count(), 0);
    }

    #[tokio::test]
    async fn test_handle_training_message_wrong_round() {
        let peer_id = PeerId::random();
        let worker_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        let mut state = WorkerJobState::new(worker_id.clone());
        state.ready_at = Some(Instant::now());
        manager.workers.insert(worker_id.clone(), state);

        // Wrong round
        manager.handle_training_message(WorkerMessage::ProofSubmission {
            peer_id: worker_id,
            round_id: 99,
            proof: vec![1, 2, 3],
            public_inputs: vec![[0u8; 32]; 8],
            error_bound: 0.05,
            steps_completed: 10,
            new_model_hash: [42u8; 32],
            checkpoint_data: vec![5, 6, 7],
        }).await;

        assert_eq!(manager.proofs_received_count(), 0);
    }

    #[tokio::test]
    async fn test_remove_non_submitting_workers() {
        let peer_id = PeerId::random();
        let model = MlpModel::new_random(4, 8, 2, 42);
        let config = TrainingJobConfig::default();
        let network = create_test_network();

        let mut manager = TrainingJobManager::new(
            peer_id,
            config,
            network,
            None,
            model,
        );
        manager.round_id = 1;

        // Add 3 workers, 2 submitted
        let w1 = PeerId::random();
        let w2 = PeerId::random();
        let w3 = PeerId::random();

        for wid in [&w1, &w2, &w3] {
            let mut state = WorkerJobState::new(wid.clone());
            state.ready_at = Some(Instant::now());
            manager.workers.insert(wid.clone(), state);
        }

        // Mark w1 and w2 as submitted
        manager.workers.get_mut(&w1).unwrap().proof_submitted_at = Some(Instant::now());
        manager.workers.get_mut(&w2).unwrap().proof_submitted_at = Some(Instant::now());

        assert_eq!(manager.workers.len(), 3);
        manager.remove_non_submitting_workers();
        assert_eq!(manager.workers.len(), 2);
        assert!(!manager.workers.contains_key(&w3));
    }

    // Helper to create a test network runner (minimal, for unit tests)
    fn create_test_network() -> Arc<NetworkRunner> {
        use crate::network::runner::NetworkRunnerBuilder;
        Arc::new(
            NetworkRunnerBuilder::new()
                .local_id(PeerId::random())
                .build()
                .expect("test network runner should build"),
        )
    }
}
