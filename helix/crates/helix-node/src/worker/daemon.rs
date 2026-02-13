//! Worker Daemon — autonomous training round lifecycle management.
//!
//! The `WorkerDaemon` subscribes to network events and autonomously:
//! 1. Evaluates incoming `RoundConfigure` announcements
//! 2. Registers with the aggregator if the round is acceptable
//! 3. Receives model weights and training parameters
//! 4. Executes training steps with ZK proof generation
//! 5. Submits proofs and updated weights to the aggregator
//! 6. Tracks participation history, earnings, and reputation
//!
//! Supports both **automatic** mode (auto-joins profitable rounds) and
//! **manual** mode (waits for explicit RPC-triggered joins).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use log::{error, info, warn};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::network::messages::{
    MessagePayload, ModelDims, NodeCapabilities, PeerId, RegistrationMessage,
    RoundManagementMessage, TrainingMessage,
};
use crate::network::runner::NetworkRunner;
use crate::trainer::{MlpModel, Trainer};
use crate::training::consensus::compute_hiding_gradient_commitment;
use crate::training::distribution::{
    self, ChunkReceiver, DataAssignment, ModelPackage, ModelPackageChunk,
    verify_model_package,
};
use crate::worker::evaluator::{EvaluatorConfig, RejectReason, RoundEvaluator};
use crate::worker::tracker::{EarningsSummary, ParticipationTracker};

use helix_core::ModelCheckpoint;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the worker daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDaemonConfig {
    /// Whether auto-join is enabled at startup.
    #[serde(default = "default_auto_join")]
    pub auto_join: bool,

    /// Maximum model parameters the worker can handle.
    #[serde(default = "default_max_model_params")]
    pub max_model_params: usize,

    /// Maximum error budget the worker is willing to accept.
    #[serde(default = "default_max_error_budget")]
    pub max_error_budget: f64,

    /// Minimum seconds of slack before deadline to consider joining.
    #[serde(default = "default_min_deadline_slack_secs")]
    pub min_deadline_slack_secs: u64,

    /// Maximum number of steps per round the worker is willing to do.
    #[serde(default = "default_max_steps_per_round")]
    pub max_steps_per_round: u32,

    /// Maximum participation records to keep.
    #[serde(default = "default_max_history_records")]
    pub max_history_records: usize,

    /// Base reward per proof for earnings estimation (in wei).
    #[serde(default = "default_base_reward_per_proof_wei")]
    pub base_reward_per_proof_wei: u64,
}

fn default_auto_join() -> bool { true }
fn default_max_model_params() -> usize { 1_000_000 }
fn default_max_error_budget() -> f64 { 1.0 }
fn default_min_deadline_slack_secs() -> u64 { 30 }
fn default_max_steps_per_round() -> u32 { 1000 }
fn default_max_history_records() -> usize { 500 }
fn default_base_reward_per_proof_wei() -> u64 { 1_000_000_000_000_000 } // 0.001 ETH

impl Default for WorkerDaemonConfig {
    fn default() -> Self {
        Self {
            auto_join: default_auto_join(),
            max_model_params: default_max_model_params(),
            max_error_budget: default_max_error_budget(),
            min_deadline_slack_secs: default_min_deadline_slack_secs(),
            max_steps_per_round: default_max_steps_per_round(),
            max_history_records: default_max_history_records(),
            base_reward_per_proof_wei: default_base_reward_per_proof_wei(),
        }
    }
}

// ============================================================================
// Active Round State
// ============================================================================

/// Phase of an active training round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundPhase {
    /// Registered with aggregator, waiting for model weights.
    Registered,
    /// Model weights received, waiting for BeginTraining signal.
    WeightsReceived,
    /// Actively training (executing steps, generating proofs).
    Training,
    /// Proof submitted, waiting for round completion.
    ProofSubmitted,
}

impl std::fmt::Display for RoundPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Registered => write!(f, "registered"),
            Self::WeightsReceived => write!(f, "weights_received"),
            Self::Training => write!(f, "training"),
            Self::ProofSubmitted => write!(f, "proof_submitted"),
        }
    }
}

/// State of an active training round.
#[derive(Debug, Clone)]
pub struct ActiveRoundState {
    /// Round ID.
    pub round_id: u64,
    /// Model ID.
    pub model_id: u64,
    /// Model architecture dimensions.
    pub model_dims: ModelDims,
    /// Number of training steps to perform.
    pub steps_per_worker: u32,
    /// Learning rate.
    pub learning_rate: f64,
    /// Error budget.
    pub error_budget: f64,
    /// Round deadline (Unix timestamp).
    pub deadline: u64,
    /// Current phase.
    pub phase: RoundPhase,
    /// Aggregator peer ID (who sent the RoundConfigure).
    pub aggregator_id: PeerId,
    /// Dataset reference (IPFS hash, URL, etc.).
    pub dataset_ref: String,
    /// Hash of current model weights (from aggregator).
    pub current_model_hash: [u8; 32],
    /// Steps completed so far in this round.
    pub steps_completed: u32,
    /// Proofs submitted so far in this round.
    pub proofs_submitted: u32,
    /// Timestamp when we joined this round.
    pub joined_at: u64,
}

// ============================================================================
// Worker Status (for RPC)
// ============================================================================

/// Worker daemon status, exposed via RPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    /// Whether auto-join is enabled.
    pub auto_join: bool,
    /// Whether the worker is currently in a round.
    pub in_round: bool,
    /// Active round details (if in a round).
    pub active_round: Option<ActiveRoundSummary>,
    /// Total rounds participated in.
    pub total_rounds: u64,
    /// Total rounds succeeded.
    pub rounds_succeeded: u64,
    /// Success rate (0.0 to 1.0).
    pub success_rate: f64,
    /// Reputation score (0.0 to 1.0).
    pub reputation_score: f64,
}

/// Summary of the active round for RPC reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveRoundSummary {
    /// Round ID.
    pub round_id: u64,
    /// Model ID.
    pub model_id: u64,
    /// Current phase.
    pub phase: String,
    /// Steps completed.
    pub steps_completed: u32,
    /// Steps required.
    pub steps_required: u32,
    /// Time elapsed in seconds.
    pub elapsed_secs: u64,
}

// ============================================================================
// Daemon Events
// ============================================================================

/// Events emitted by the daemon for logging and external integration.
#[derive(Debug, Clone)]
pub enum WorkerDaemonEvent {
    /// A round was announced and evaluated.
    RoundEvaluated {
        round_id: u64,
        accepted: bool,
        reason: Option<String>,
    },
    /// Worker registered with aggregator for a round.
    Registered { round_id: u64 },
    /// Model weights received from aggregator.
    WeightsReceived { round_id: u64, bytes: usize },
    /// Training started for a round.
    TrainingStarted { round_id: u64, steps: u32 },
    /// A training step completed.
    StepCompleted { round_id: u64, step: u32, loss: f64 },
    /// Proof submitted for a round.
    ProofSubmitted { round_id: u64, proof_bytes: usize },
    /// Round completed.
    RoundCompleted { round_id: u64, success: bool },
    /// Round failed.
    RoundFailed { round_id: u64, reason: String },
}

// ============================================================================
// WorkerDaemon
// ============================================================================

/// Autonomous worker daemon that manages training round participation.
///
/// The daemon processes network events and manages the full lifecycle of
/// training round participation, from discovery through proof submission.
pub struct WorkerDaemon {
    /// Local peer ID.
    local_id: PeerId,
    /// Daemon configuration.
    config: WorkerDaemonConfig,
    /// Round evaluator for opportunity assessment.
    evaluator: RoundEvaluator,
    /// Participation history tracker.
    tracker: Arc<RwLock<ParticipationTracker>>,
    /// Whether auto-join is enabled (mutable at runtime via RPC).
    auto_join: Arc<AtomicBool>,
    /// Currently active round (None if idle).
    ///
    /// Public for integration test setup. Production code should use
    /// `active_round()` and `current_round_id()` accessors.
    pub active_round: Arc<RwLock<Option<ActiveRoundState>>>,
    /// Trainer for the current round (separate from runtime's trainer to avoid conflicts).
    trainer: Arc<RwLock<Option<Trainer>>>,
    /// Event log (recent events for debugging/RPC).
    events: Arc<RwLock<Vec<WorkerDaemonEvent>>>,
    /// Maximum events to keep.
    max_events: usize,
    /// Chunk receiver for in-progress chunked model transfers.
    chunk_receiver: Arc<RwLock<Option<ChunkReceiver>>>,
    /// Current data assignment for the active round.
    data_assignment: Arc<RwLock<Option<DataAssignment>>>,
}

impl WorkerDaemon {
    /// Creates a new worker daemon.
    pub fn new(
        local_id: PeerId,
        config: WorkerDaemonConfig,
        capabilities: NodeCapabilities,
    ) -> Self {
        let evaluator_config = EvaluatorConfig {
            max_model_params: config.max_model_params,
            max_error_budget: config.max_error_budget,
            min_deadline_slack_secs: config.min_deadline_slack_secs,
            max_steps_per_round: config.max_steps_per_round,
        };

        let auto_join = Arc::new(AtomicBool::new(config.auto_join));

        let mut tracker = ParticipationTracker::new(config.max_history_records);
        tracker.set_base_reward(config.base_reward_per_proof_wei);

        Self {
            local_id,
            config,
            evaluator: RoundEvaluator::new(evaluator_config, capabilities),
            tracker: Arc::new(RwLock::new(tracker)),
            auto_join,
            active_round: Arc::new(RwLock::new(None)),
            trainer: Arc::new(RwLock::new(None)),
            events: Arc::new(RwLock::new(Vec::new())),
            max_events: 200,
            chunk_receiver: Arc::new(RwLock::new(None)),
            data_assignment: Arc::new(RwLock::new(None)),
        }
    }

    // ========================================================================
    // Public Accessors (for RPC integration)
    // ========================================================================

    /// Returns the current worker status.
    pub fn status(&self) -> WorkerStatus {
        let tracker = self.tracker.read();
        let summary = tracker.earnings_summary();
        let active = self.active_round.read();

        let active_round = active.as_ref().map(|r| {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            ActiveRoundSummary {
                round_id: r.round_id,
                model_id: r.model_id,
                phase: r.phase.to_string(),
                steps_completed: r.steps_completed,
                steps_required: r.steps_per_worker,
                elapsed_secs: now.saturating_sub(r.joined_at),
            }
        });

        WorkerStatus {
            auto_join: self.auto_join.load(Ordering::Relaxed),
            in_round: active.is_some(),
            active_round,
            total_rounds: summary.rounds_participated,
            rounds_succeeded: summary.rounds_succeeded,
            success_rate: summary.success_rate,
            reputation_score: summary.reputation_score,
        }
    }

    /// Returns the earnings summary.
    pub fn earnings(&self) -> EarningsSummary {
        self.tracker.read().earnings_summary()
    }

    /// Sets auto-join mode. Returns the previous value.
    pub fn set_auto_join(&self, enabled: bool) -> bool {
        let prev = self.auto_join.swap(enabled, Ordering::Relaxed);
        info!(
            "Worker daemon auto-join: {} -> {}",
            prev, enabled,
        );
        prev
    }

    /// Returns whether auto-join is enabled.
    pub fn is_auto_join(&self) -> bool {
        self.auto_join.load(Ordering::Relaxed)
    }

    /// Returns the active round state (if any).
    pub fn active_round(&self) -> Option<ActiveRoundState> {
        self.active_round.read().clone()
    }

    /// Returns the current round ID (if any).
    pub fn current_round_id(&self) -> Option<u64> {
        self.active_round.read().as_ref().map(|r| r.round_id)
    }

    /// Returns recent daemon events.
    pub fn recent_events(&self) -> Vec<WorkerDaemonEvent> {
        self.events.read().clone()
    }

    /// Returns a shared reference to the tracker.
    pub fn tracker(&self) -> Arc<RwLock<ParticipationTracker>> {
        self.tracker.clone()
    }

    // ========================================================================
    // Event Processing — called from runtime's event loop
    // ========================================================================

    /// Processes a RoundManagement network event.
    ///
    /// This is the main entry point called from the runtime's worker event loop
    /// whenever a `RoundManagementMessage` arrives.
    pub async fn handle_round_management(
        &self,
        from: &PeerId,
        message: &RoundManagementMessage,
        network: &Arc<NetworkRunner>,
    ) {
        match message {
            RoundManagementMessage::RoundConfigure {
                round_id,
                model_id,
                model_dims,
                dataset_ref,
                steps_per_worker,
                learning_rate,
                error_budget,
                min_workers: _,
                deadline,
                current_model_hash,
                dataset_spec: _,
            } => {
                self.handle_round_configure(
                    from,
                    *round_id,
                    *model_id,
                    model_dims.clone(),
                    dataset_ref.clone(),
                    *steps_per_worker,
                    *learning_rate,
                    *error_budget,
                    *deadline,
                    *current_model_hash,
                    network,
                ).await;
            }
            RoundManagementMessage::BeginTraining { round_id, participants } => {
                self.handle_begin_training(*round_id, participants, network).await;
            }
            RoundManagementMessage::ProofAcknowledged { round_id, valid, reason } => {
                self.handle_proof_acknowledged(*round_id, *valid, reason.as_deref());
            }
            RoundManagementMessage::RoundCompleted {
                round_id,
                final_commitment: _,
                tx_hash: _,
                new_model_hash: _,
                total_error_bound,
                num_contributors: _,
            } => {
                self.handle_round_completed(*round_id, *total_error_bound);
            }
            RoundManagementMessage::RoundFailed { round_id, reason } => {
                self.handle_round_failed(*round_id, reason);
            }
            // WorkerReady is sent by workers, not handled by workers
            RoundManagementMessage::WorkerReady { .. } => {}
            // ProofSubmission is sent by workers, not handled by workers
            RoundManagementMessage::ProofSubmission { .. } => {}
            // Model package delivery — handled in handle_model_package_delivery
            RoundManagementMessage::ModelPackageDelivery { package_data } => {
                self.handle_model_package_delivery(package_data, network).await;
            }
            // Chunked model package delivery
            RoundManagementMessage::ModelPackageChunkDelivery { chunk_data } => {
                self.handle_model_package_chunk(chunk_data, network).await;
            }
            // Data assignment delivery
            RoundManagementMessage::DataAssignmentDelivery { assignment_data } => {
                self.handle_data_assignment(assignment_data);
            }
            // DistributionAcknowledged is sent by workers, not handled by workers
            RoundManagementMessage::DistributionAcknowledged { .. } => {}
        }
    }

    /// Processes a Training network event (ModelWeights/UpdatedWeights).
    pub async fn handle_training_message(
        &self,
        _from: &PeerId,
        message: &TrainingMessage,
        _network: &Arc<NetworkRunner>,
    ) {
        match message {
            TrainingMessage::ModelWeights { round_id, checkpoint_data, weight_hash: _ }
            | TrainingMessage::UpdatedWeights { round_id, checkpoint_data, weight_hash: _ } => {
                self.handle_model_weights(*round_id, checkpoint_data);
            }
            _ => {}
        }
    }

    /// Processes a Registration network event (WorkerRegistered).
    pub fn handle_registration_message(
        &self,
        _from: &PeerId,
        message: &RegistrationMessage,
    ) {
        match message {
            RegistrationMessage::WorkerRegistered {
                accepted,
                worker_slot,
                reject_reason,
                current_round_id: _,
            } => {
                self.handle_worker_registered(*accepted, *worker_slot, reject_reason.as_deref());
            }
            _ => {}
        }
    }

    /// Manually joins a round (called from RPC when auto-join is disabled).
    ///
    /// Returns `Ok(())` if the join was initiated, `Err(reason)` if the round
    /// was rejected by the evaluator (ignoring the auto-join check).
    pub async fn manual_join(
        &self,
        round_id: u64,
        model_id: u64,
        model_dims: ModelDims,
        dataset_ref: String,
        steps_per_worker: u32,
        learning_rate: f64,
        error_budget: f64,
        deadline: u64,
        current_model_hash: [u8; 32],
        aggregator_id: PeerId,
        network: &Arc<NetworkRunner>,
    ) -> Result<(), String> {
        // Evaluate but skip the auto-join check
        let current = self.current_round_id();
        let eval_result = self.evaluator.evaluate(
            &model_dims,
            steps_per_worker,
            error_budget,
            deadline,
            current,
            true, // force auto_join=true for manual joins
        );

        if let Err(reason) = eval_result {
            return Err(reason.to_string());
        }

        self.initiate_join(
            &aggregator_id, round_id, model_id, model_dims,
            dataset_ref, steps_per_worker, learning_rate, error_budget,
            deadline, current_model_hash, network,
        ).await;

        Ok(())
    }

    // ========================================================================
    // Internal Handlers
    // ========================================================================

    async fn handle_round_configure(
        &self,
        from: &PeerId,
        round_id: u64,
        model_id: u64,
        model_dims: ModelDims,
        dataset_ref: String,
        steps_per_worker: u32,
        learning_rate: f64,
        error_budget: f64,
        deadline: u64,
        current_model_hash: [u8; 32],
        network: &Arc<NetworkRunner>,
    ) {
        info!(
            "RoundConfigure #{} from {} (model={}, {}x{}x{}, steps={}, lr={}, budget={:.6})",
            round_id, from, model_id,
            model_dims.d_in, model_dims.d_hid, model_dims.d_out,
            steps_per_worker, learning_rate, error_budget,
        );

        let auto_join = self.auto_join.load(Ordering::Relaxed);
        let current = self.current_round_id();

        let eval_result = self.evaluator.evaluate(
            &model_dims,
            steps_per_worker,
            error_budget,
            deadline,
            current,
            auto_join,
        );

        match eval_result {
            Ok(()) => {
                self.emit(WorkerDaemonEvent::RoundEvaluated {
                    round_id,
                    accepted: true,
                    reason: None,
                });

                self.initiate_join(
                    from, round_id, model_id, model_dims,
                    dataset_ref, steps_per_worker, learning_rate, error_budget,
                    deadline, current_model_hash, network,
                ).await;
            }
            Err(reason) => {
                info!("Round #{} rejected: {}", round_id, reason);
                self.emit(WorkerDaemonEvent::RoundEvaluated {
                    round_id,
                    accepted: false,
                    reason: Some(reason.to_string()),
                });

                // Send WorkerReady(false) so aggregator knows we're declining
                if reason != RejectReason::AutoJoinDisabled {
                    network.broadcast(MessagePayload::RoundManagement(
                        RoundManagementMessage::WorkerReady {
                            round_id,
                            ready: false,
                            reason: Some(reason.to_string()),
                        },
                    )).await;
                }
            }
        }
    }

    async fn initiate_join(
        &self,
        aggregator_id: &PeerId,
        round_id: u64,
        model_id: u64,
        model_dims: ModelDims,
        dataset_ref: String,
        steps_per_worker: u32,
        learning_rate: f64,
        error_budget: f64,
        deadline: u64,
        current_model_hash: [u8; 32],
        network: &Arc<NetworkRunner>,
    ) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Set active round state
        {
            let mut active = self.active_round.write();
            *active = Some(ActiveRoundState {
                round_id,
                model_id,
                model_dims: model_dims.clone(),
                steps_per_worker,
                learning_rate,
                error_budget,
                deadline,
                phase: RoundPhase::Registered,
                aggregator_id: aggregator_id.clone(),
                dataset_ref,
                current_model_hash,
                steps_completed: 0,
                proofs_submitted: 0,
                joined_at: now,
            });
        }

        // Record in tracker
        self.tracker.write().record_join(round_id, model_id);

        // Initialize trainer for this round's dimensions
        {
            let mut trainer = self.trainer.write();
            *trainer = Some(Trainer::from_params(
                model_dims.d_in,
                model_dims.d_hid,
                model_dims.d_out,
                learning_rate,
                now, // use current time as seed for random init
                model_dims.num_layers,
                model_dims.activation_type,
            ));
        }

        info!("Joining round #{} (model={})", round_id, model_id);

        // Send registration to aggregator
        let capabilities = self.evaluator.config();
        network.broadcast(MessagePayload::Registration(
            RegistrationMessage::WorkerRegister {
                capabilities: NodeCapabilities {
                    can_train: true,
                    can_aggregate: false,
                    can_prove: true,
                    gpu_memory_mb: 0,
                    cpu_cores: std::thread::available_parallelism()
                        .map(|n| n.get() as u32)
                        .unwrap_or(4),
                    storage_gb: 10,
                },
                listen_addr: String::new(), // filled by NetworkRunner
                public_key: None,
                max_concurrent_steps: capabilities.max_steps_per_round,
                available_memory_mb: 4096,
            },
        )).await;

        // Send WorkerReady
        network.broadcast(MessagePayload::RoundManagement(
            RoundManagementMessage::WorkerReady {
                round_id,
                ready: true,
                reason: None,
            },
        )).await;

        self.emit(WorkerDaemonEvent::Registered { round_id });
    }

    fn handle_model_weights(&self, round_id: u64, checkpoint_data: &[u8]) {
        let mut active = self.active_round.write();
        let round = match active.as_mut() {
            Some(r) if r.round_id == round_id => r,
            _ => {
                warn!(
                    "Received model weights for unknown/mismatched round {} (active={:?})",
                    round_id,
                    active.as_ref().map(|r| r.round_id),
                );
                return;
            }
        };

        match ModelCheckpoint::from_bytes(checkpoint_data) {
            Ok(ckpt) => {
                match MlpModel::from_checkpoint(&ckpt) {
                    Ok(model) => {
                        let mut trainer = self.trainer.write();
                        *trainer = Some(Trainer::with_model(model, round.learning_rate));
                        round.phase = RoundPhase::WeightsReceived;
                        info!(
                            "Model weights loaded for round {} ({} bytes, step={})",
                            round_id, checkpoint_data.len(), ckpt.step_number,
                        );
                        self.emit(WorkerDaemonEvent::WeightsReceived {
                            round_id,
                            bytes: checkpoint_data.len(),
                        });
                    }
                    Err(e) => {
                        error!("Failed to deserialize model from checkpoint for round {}: {}", round_id, e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to parse checkpoint for round {}: {}", round_id, e);
            }
        }
    }

    async fn handle_begin_training(
        &self,
        round_id: u64,
        participants: &[String],
        network: &Arc<NetworkRunner>,
    ) {
        // Check if we're a participant
        let local_str = self.local_id.0.clone();
        if !participants.contains(&local_str) {
            info!(
                "BeginTraining for round {} but we're not in participant list ({} workers)",
                round_id, participants.len(),
            );
            return;
        }

        // Verify we're in the right round and phase
        {
            let active = self.active_round.read();
            match active.as_ref() {
                Some(r) if r.round_id == round_id => {
                    if r.phase != RoundPhase::Registered && r.phase != RoundPhase::WeightsReceived {
                        warn!(
                            "BeginTraining for round {} but phase is {} (expected registered/weights_received)",
                            round_id, r.phase,
                        );
                        return;
                    }
                }
                _ => {
                    warn!("BeginTraining for round {} but not our active round", round_id);
                    return;
                }
            }
        }

        info!(
            "BeginTraining for round {} ({} participants)",
            round_id, participants.len(),
        );

        self.execute_training(round_id, network).await;
    }

    /// Executes the training steps for the active round.
    async fn execute_training(
        &self,
        round_id: u64,
        network: &Arc<NetworkRunner>,
    ) {
        let (steps_per_worker, d_in, d_out, model_seed) = {
            let mut active = self.active_round.write();
            let round = match active.as_mut() {
                Some(r) if r.round_id == round_id => r,
                _ => return,
            };
            round.phase = RoundPhase::Training;
            (
                round.steps_per_worker,
                round.model_dims.d_in,
                round.model_dims.d_out,
                round_id, // use round_id as part of data seed
            )
        };

        self.emit(WorkerDaemonEvent::TrainingStarted {
            round_id,
            steps: steps_per_worker,
        });

        // Load training data from the data assignment if available,
        // otherwise fall back to synthetic data generation.
        let assigned_data = self.load_training_data(d_in, d_out, steps_per_worker);

        let mut total_loss = 0.0;
        let mut total_error_bound = 0.0;
        let mut last_proof_bytes: Vec<u8> = Vec::new();
        let mut last_public_inputs: Vec<[u8; 32]> = Vec::new();
        let mut last_model_hash = [0u8; 32];

        for step in 0..steps_per_worker {
            // Use assigned data if available, otherwise generate synthetic data
            let (x, target) = if let Some(ref batches) = assigned_data {
                let batch_idx = step as usize % batches.len();
                batches[batch_idx].clone()
            } else {
                generate_training_data(d_in, d_out, model_seed + step as u64)
            };

            let step_result = {
                let mut trainer = self.trainer.write();
                let tr = match trainer.as_mut() {
                    Some(t) => t,
                    None => {
                        error!("No trainer available for round {} step {}", round_id, step);
                        self.fail_round(round_id, "no trainer available");
                        return;
                    }
                };

                tr.train_step(&x, &target)
            };

            match step_result {
                Ok(proved) => {
                    let loss = proved.loss;
                    total_loss = loss;
                    total_error_bound += loss * 0.01; // error bound estimation

                    last_proof_bytes = proved.evm_bundle
                        .as_ref()
                        .map(|b| b.evm_proof.clone())
                        .unwrap_or_else(|| proved.proof_result.proof.clone());

                    last_public_inputs = if let Some(ref bundle) = proved.evm_bundle {
                        bundle.evm_public_inputs.clone()
                    } else {
                        // No EVM bundle: create placeholder PIs (8 x 32 bytes)
                        vec![[0u8; 32]; proved.proof_result.public_inputs.len()]
                    };

                    last_model_hash = proved.commitment;

                    // Update step count
                    {
                        let mut active = self.active_round.write();
                        if let Some(r) = active.as_mut() {
                            r.steps_completed = step + 1;
                        }
                    }

                    self.emit(WorkerDaemonEvent::StepCompleted {
                        round_id,
                        step: step + 1,
                        loss,
                    });

                    info!(
                        "Round {} step {}/{} complete (loss={:.6})",
                        round_id, step + 1, steps_per_worker, loss,
                    );
                }
                Err(e) => {
                    error!("Training step failed for round {} step {}: {}", round_id, step, e);
                    self.fail_round(round_id, &format!("training step {} failed: {}", step, e));
                    return;
                }
            }
        }

        // Submit proof to aggregator
        let checkpoint_data = {
            let trainer = self.trainer.read();
            trainer.as_ref().and_then(|tr| {
                let ckpt = tr.model().to_checkpoint(tr.step_count());
                ckpt.to_bytes().ok()
            }).unwrap_or_default()
        };

        network.broadcast(MessagePayload::RoundManagement(
            RoundManagementMessage::ProofSubmission {
                round_id,
                proof: last_proof_bytes.clone(),
                public_inputs: last_public_inputs,
                error_bound: total_error_bound,
                steps_completed: steps_per_worker,
                new_model_hash: last_model_hash,
                checkpoint_data,
            },
        )).await;

        // Also send gradient commitment for the consensus layer
        let (hiding_commitment, nonce) = compute_hiding_gradient_commitment(&last_model_hash);
        network.broadcast(MessagePayload::Gradient(
            crate::network::messages::GradientMessage::ShareGradient {
                round_id,
                gradient_commitment: hiding_commitment,
                commitment_nonce: nonce,
                error_bound: total_error_bound,
                proof: last_proof_bytes.clone(),
            },
        )).await;

        // Send weight update
        {
            let trainer = self.trainer.read();
            if let Some(tr) = trainer.as_ref() {
                let ckpt = tr.model().to_checkpoint(tr.step_count());
                if let Ok(ckpt_bytes) = ckpt.to_bytes() {
                    network.broadcast(MessagePayload::Gradient(
                        crate::network::messages::GradientMessage::WeightUpdate {
                            round_id,
                            checkpoint_data: ckpt_bytes,
                            weight_hash: tr.model().commitment(),
                        },
                    )).await;
                }
            }
        }

        // Update phase
        {
            let mut active = self.active_round.write();
            if let Some(r) = active.as_mut() {
                r.proofs_submitted += 1;
                r.phase = RoundPhase::ProofSubmitted;
            }
        }

        self.emit(WorkerDaemonEvent::ProofSubmitted {
            round_id,
            proof_bytes: last_proof_bytes.len(),
        });

        info!(
            "Round {} training complete: {} steps, loss={:.6}, proof={} bytes",
            round_id, steps_per_worker, total_loss, last_proof_bytes.len(),
        );
    }

    // ========================================================================
    // Data Loading from Assignments
    // ========================================================================

    /// Loads training data from the current data assignment.
    ///
    /// Returns a vector of (input, target) batches if a data assignment is present
    /// and the data can be loaded. Returns `None` to fall back to synthetic data.
    fn load_training_data(
        &self,
        d_in: usize,
        d_out: usize,
        steps: u32,
    ) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
        let assignment = self.data_assignment.read();
        let assignment = assignment.as_ref()?;

        match &assignment.source {
            distribution::DataSourceType::Inline { data, format } => {
                match format {
                    distribution::DataFormat::Csv => {
                        self.load_csv_data(data, d_in, d_out, steps, assignment)
                    }
                    distribution::DataFormat::RawF32 => {
                        self.load_raw_f32_data(data, d_in, d_out, steps)
                    }
                    distribution::DataFormat::BinaryTensor => {
                        // Binary tensor uses helix-core's format
                        warn!("BinaryTensor format not yet supported for inline data, using synthetic data");
                        None
                    }
                }
            }
            distribution::DataSourceType::S3 { uri, .. } => {
                info!("S3 data source ({}) — loading at runtime via helix-core", uri);
                // S3 loading is async and requires the s3-fetch feature in helix-core.
                // For the training loop, the worker would pre-fetch this before training starts.
                // For now, log and fall back to synthetic data.
                warn!("S3 data loading not yet wired into synchronous training loop, using synthetic data");
                None
            }
            distribution::DataSourceType::Ipfs { cid, .. } => {
                info!("IPFS data source ({}) — loading at runtime via helix-core", cid);
                warn!("IPFS data loading not yet wired into synchronous training loop, using synthetic data");
                None
            }
            distribution::DataSourceType::Http { url } => {
                info!("HTTP data source ({}) — loading at runtime", url);
                warn!("HTTP data loading not yet wired into synchronous training loop, using synthetic data");
                None
            }
        }
    }

    /// Parses inline CSV data into training batches.
    fn load_csv_data(
        &self,
        data: &[u8],
        d_in: usize,
        d_out: usize,
        steps: u32,
        assignment: &DataAssignment,
    ) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
        let text = match std::str::from_utf8(data) {
            Ok(s) => s,
            Err(e) => {
                warn!("CSV data is not valid UTF-8: {}", e);
                return None;
            }
        };

        let mut batches = Vec::new();
        let mut batch_x = Vec::new();
        let mut batch_target = Vec::new();
        let batch_size = assignment.batch_size.max(1) as usize;

        for line in text.lines().skip(1) {
            // skip header row
            let values: Vec<f64> = line
                .split(',')
                .filter_map(|v| v.trim().parse::<f64>().ok())
                .collect();

            if values.is_empty() {
                continue;
            }

            // Extract features and labels using column indices
            let features: Vec<f64> = if assignment.feature_columns.is_empty() {
                // Default: all columns except the last `d_out` are features
                let end = values.len().saturating_sub(d_out);
                values[..end].to_vec()
            } else {
                assignment.feature_columns.iter()
                    .filter_map(|&i| values.get(i).copied())
                    .collect()
            };

            let labels: Vec<f64> = if assignment.label_columns.is_empty() {
                // Default: last `d_out` columns are labels
                let start = values.len().saturating_sub(d_out);
                values[start..].to_vec()
            } else {
                assignment.label_columns.iter()
                    .filter_map(|&i| values.get(i).copied())
                    .collect()
            };

            // Pad/truncate to expected dimensions
            let mut x = features;
            x.resize(d_in, 0.0);
            let mut t = labels;
            t.resize(d_out, 0.0);

            batch_x.extend_from_slice(&x);
            batch_target.extend_from_slice(&t);

            if batch_x.len() >= d_in * batch_size {
                // Average the batch into a single training sample
                // (the Trainer expects single-sample (x, target) pairs)
                let avg_x: Vec<f64> = (0..d_in)
                    .map(|i| {
                        (0..batch_size)
                            .map(|b| batch_x[b * d_in + i])
                            .sum::<f64>() / batch_size as f64
                    })
                    .collect();
                let avg_t: Vec<f64> = (0..d_out)
                    .map(|i| {
                        (0..batch_size)
                            .map(|b| batch_target[b * d_out + i])
                            .sum::<f64>() / batch_size as f64
                    })
                    .collect();

                batches.push((avg_x, avg_t));
                batch_x.clear();
                batch_target.clear();
            }
        }

        // Flush remaining partial batch
        if !batch_x.is_empty() {
            let num_samples = batch_x.len() / d_in;
            if num_samples > 0 {
                let avg_x: Vec<f64> = (0..d_in)
                    .map(|i| {
                        (0..num_samples)
                            .map(|b| batch_x[b * d_in + i])
                            .sum::<f64>() / num_samples as f64
                    })
                    .collect();
                let avg_t: Vec<f64> = (0..d_out)
                    .map(|i| {
                        (0..num_samples)
                            .map(|b| batch_target[b * d_out + i])
                            .sum::<f64>() / num_samples as f64
                    })
                    .collect();
                batches.push((avg_x, avg_t));
            }
        }

        if batches.is_empty() {
            warn!("CSV data produced no training batches, falling back to synthetic data");
            return None;
        }

        info!(
            "Loaded {} training batches from CSV data ({} bytes, batch_size={})",
            batches.len(), data.len(), batch_size,
        );

        Some(batches)
    }

    /// Parses raw float32 data into training batches.
    fn load_raw_f32_data(
        &self,
        data: &[u8],
        d_in: usize,
        d_out: usize,
        _steps: u32,
    ) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
        let sample_size = (d_in + d_out) * 4; // 4 bytes per f32
        if data.len() < sample_size {
            warn!("RawF32 data too small ({} bytes, need at least {})", data.len(), sample_size);
            return None;
        }

        let num_samples = data.len() / sample_size;
        let mut batches = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let offset = i * sample_size;
            let mut x = Vec::with_capacity(d_in);
            let mut t = Vec::with_capacity(d_out);

            for j in 0..d_in {
                let pos = offset + j * 4;
                if pos + 4 <= data.len() {
                    let bytes: [u8; 4] = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                    x.push(f32::from_le_bytes(bytes) as f64);
                }
            }

            for j in 0..d_out {
                let pos = offset + (d_in + j) * 4;
                if pos + 4 <= data.len() {
                    let bytes: [u8; 4] = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                    t.push(f32::from_le_bytes(bytes) as f64);
                }
            }

            if x.len() == d_in && t.len() == d_out {
                batches.push((x, t));
            }
        }

        if batches.is_empty() {
            warn!("RawF32 data produced no training samples");
            return None;
        }

        info!(
            "Loaded {} training samples from RawF32 data ({} bytes)",
            batches.len(), data.len(),
        );

        Some(batches)
    }

    // ========================================================================
    // Model Package & Data Assignment Handlers
    // ========================================================================

    /// Handles receipt of a complete model package from the aggregator.
    async fn handle_model_package_delivery(
        &self,
        package_data: &[u8],
        network: &Arc<NetworkRunner>,
    ) {
        let package: ModelPackage = match bincode::deserialize(package_data) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to deserialize model package: {}", e);
                return;
            }
        };

        let round_id = package.round_id;

        // Verify the package matches our active round
        {
            let active = self.active_round.read();
            match active.as_ref() {
                Some(r) if r.round_id == round_id => {}
                _ => {
                    warn!(
                        "Received model package for unknown round {} (active={:?})",
                        round_id,
                        active.as_ref().map(|r| r.round_id),
                    );
                    return;
                }
            }
        }

        // Verify model integrity and load it
        match verify_model_package(&package) {
            Ok(model) => {
                // Also verify against the on-chain commitment hash
                let active = self.active_round.read();
                if let Some(ref round) = *active {
                    let model_commitment = model.commitment();
                    if model_commitment != round.current_model_hash {
                        // For resume rounds, the hash may differ — only warn
                        if !package.is_resume {
                            warn!(
                                "Model commitment mismatch for round {}: \
                                 on-chain={}, received={}",
                                round_id,
                                hex::encode(&round.current_model_hash[..8]),
                                hex::encode(&model_commitment[..8]),
                            );
                        }
                    }
                }
                drop(active);

                // Set up the trainer with the received model
                {
                    let mut trainer = self.trainer.write();
                    *trainer = Some(Trainer::with_model(
                        model,
                        package.hyperparameters.learning_rate,
                    ));
                }

                // Update round phase
                {
                    let mut active = self.active_round.write();
                    if let Some(ref mut r) = *active {
                        r.phase = RoundPhase::WeightsReceived;
                    }
                }

                info!(
                    "Model package loaded for round {} ({} bytes, compressed={}, resume={}, step={})",
                    round_id,
                    package_data.len(),
                    package.compressed,
                    package.is_resume,
                    package.starting_step,
                );

                self.emit(WorkerDaemonEvent::WeightsReceived {
                    round_id,
                    bytes: package_data.len(),
                });

                // Send acknowledgement
                let loaded_hash = {
                    let trainer = self.trainer.read();
                    trainer.as_ref().map(|t| t.model().commitment())
                };
                network.broadcast(MessagePayload::RoundManagement(
                    RoundManagementMessage::DistributionAcknowledged {
                        round_id,
                        success: true,
                        error: None,
                        loaded_weight_hash: loaded_hash,
                    },
                )).await;
            }
            Err(e) => {
                error!(
                    "Model package verification failed for round {}: {}",
                    round_id, e,
                );

                // Send failure acknowledgement
                network.broadcast(MessagePayload::RoundManagement(
                    RoundManagementMessage::DistributionAcknowledged {
                        round_id,
                        success: false,
                        error: Some(e.clone()),
                        loaded_weight_hash: None,
                    },
                )).await;

                self.fail_round(round_id, &format!("model verification failed: {}", e));
            }
        }
    }

    /// Handles receipt of a model package chunk for chunked transfer.
    async fn handle_model_package_chunk(
        &self,
        chunk_data: &[u8],
        network: &Arc<NetworkRunner>,
    ) {
        let chunk: ModelPackageChunk = match bincode::deserialize(chunk_data) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to deserialize model package chunk: {}", e);
                return;
            }
        };

        let round_id = chunk.round_id;
        let model_id = chunk.model_id;
        let chunk_index = chunk.chunk_index;
        let total_chunks = chunk.total_chunks;

        // Initialize receiver if this is the first chunk
        {
            let mut receiver = self.chunk_receiver.write();
            if receiver.is_none() || chunk_index == 0 {
                *receiver = Some(ChunkReceiver::new(round_id, model_id, total_chunks));
            }
        }

        // Add chunk to receiver
        let is_complete = {
            let mut receiver = self.chunk_receiver.write();
            match receiver.as_mut() {
                Some(recv) => match recv.add_chunk(chunk) {
                    Ok(complete) => {
                        info!(
                            "Chunk {}/{} received for round {} ({:.0}%)",
                            chunk_index + 1,
                            total_chunks,
                            round_id,
                            recv.progress() * 100.0,
                        );
                        complete
                    }
                    Err(e) => {
                        error!("Chunk {} failed integrity check: {}", chunk_index, e);
                        return;
                    }
                },
                None => {
                    error!("No chunk receiver available");
                    return;
                }
            }
        };

        // If all chunks received, assemble and process
        if is_complete {
            let receiver = self.chunk_receiver.write().take();
            if let Some(recv) = receiver {
                let elapsed = recv.elapsed();
                let bytes = recv.bytes_received();
                info!(
                    "All {} chunks received for round {} ({} bytes in {:.1}s)",
                    total_chunks, round_id, bytes, elapsed.as_secs_f64(),
                );

                match recv.assemble() {
                    Ok(package) => {
                        let package_bytes = match bincode::serialize(&package) {
                            Ok(b) => b,
                            Err(e) => {
                                error!("Failed to re-serialize assembled package: {}", e);
                                return;
                            }
                        };
                        self.handle_model_package_delivery(&package_bytes, network).await;
                    }
                    Err(e) => {
                        error!("Failed to assemble model package chunks: {}", e);
                        self.fail_round(round_id, &format!("chunk assembly failed: {}", e));
                    }
                }
            }
        }
    }

    /// Handles receipt of a data assignment from the aggregator.
    fn handle_data_assignment(&self, assignment_data: &[u8]) {
        let assignment: DataAssignment = match bincode::deserialize(assignment_data) {
            Ok(a) => a,
            Err(e) => {
                error!("Failed to deserialize data assignment: {}", e);
                return;
            }
        };

        let round_id = assignment.round_id;

        // Verify the assignment matches our active round
        {
            let active = self.active_round.read();
            match active.as_ref() {
                Some(r) if r.round_id == round_id => {}
                _ => {
                    warn!(
                        "Received data assignment for unknown round {} (active={:?})",
                        round_id,
                        active.as_ref().map(|r| r.round_id),
                    );
                    return;
                }
            }
        }

        info!(
            "Data assignment received for round {}: worker {}/{}, {} samples, source={:?}",
            round_id,
            assignment.worker_index + 1,
            assignment.total_workers,
            assignment.num_samples,
            match &assignment.source {
                distribution::DataSourceType::S3 { uri, .. } => format!("S3({})", uri),
                distribution::DataSourceType::Ipfs { cid, .. } => format!("IPFS({})", cid),
                distribution::DataSourceType::Inline { data, .. } => format!("Inline({} bytes)", data.len()),
                distribution::DataSourceType::Http { url } => format!("HTTP({})", url),
            },
        );

        *self.data_assignment.write() = Some(assignment);
    }

    fn handle_worker_registered(
        &self,
        accepted: bool,
        worker_slot: Option<u32>,
        reject_reason: Option<&str>,
    ) {
        if accepted {
            info!(
                "Registration accepted (slot={:?})",
                worker_slot,
            );
        } else {
            let reason = reject_reason.unwrap_or("unknown");
            warn!("Registration rejected: {}", reason);

            // Clear active round if registration failed
            let round_id = self.current_round_id();
            if let Some(rid) = round_id {
                self.fail_round(rid, &format!("registration rejected: {}", reason));
            }
        }
    }

    fn handle_proof_acknowledged(&self, round_id: u64, valid: bool, reason: Option<&str>) {
        let active = self.active_round.read();
        if let Some(r) = active.as_ref() {
            if r.round_id == round_id {
                if valid {
                    info!("Proof acknowledged for round {}", round_id);
                } else {
                    warn!(
                        "Proof rejected for round {}: {}",
                        round_id,
                        reason.unwrap_or("unknown"),
                    );
                }
            }
        }
    }

    fn handle_round_completed(&self, round_id: u64, total_error_bound: f64) {
        let mut active = self.active_round.write();
        let was_our_round = active.as_ref().map_or(false, |r| r.round_id == round_id);

        if was_our_round {
            let round = active.take().unwrap();
            let proof_hash = format!("round_{}_proof", round_id);

            self.tracker.write().record_completion(
                round_id,
                round.steps_completed,
                round.proofs_submitted,
                total_error_bound,
                0.0, // final loss not in the completion message
                proof_hash,
            );

            // Clear trainer and distribution state
            *self.trainer.write() = None;
            *self.chunk_receiver.write() = None;
            *self.data_assignment.write() = None;

            info!(
                "Round {} completed: {} steps, {} proofs",
                round_id, round.steps_completed, round.proofs_submitted,
            );

            // Must drop active before emit to avoid double-borrow
            drop(active);

            self.emit(WorkerDaemonEvent::RoundCompleted {
                round_id,
                success: true,
            });
        } else {
            info!("Round {} completed (not our round)", round_id);
        }
    }

    fn handle_round_failed(&self, round_id: u64, reason: &str) {
        let mut active = self.active_round.write();
        let was_our_round = active.as_ref().map_or(false, |r| r.round_id == round_id);

        if was_our_round {
            active.take();
            self.tracker.write().record_failure(round_id);
            *self.trainer.write() = None;

            warn!("Round {} failed: {}", round_id, reason);

            drop(active);

            self.emit(WorkerDaemonEvent::RoundFailed {
                round_id,
                reason: reason.to_string(),
            });
        }
    }

    fn fail_round(&self, round_id: u64, reason: &str) {
        let mut active = self.active_round.write();
        if active.as_ref().map_or(false, |r| r.round_id == round_id) {
            active.take();
            self.tracker.write().record_failure(round_id);
            *self.trainer.write() = None;
            *self.chunk_receiver.write() = None;
            *self.data_assignment.write() = None;

            drop(active);

            self.emit(WorkerDaemonEvent::RoundFailed {
                round_id,
                reason: reason.to_string(),
            });
        }
    }

    fn emit(&self, event: WorkerDaemonEvent) {
        let mut events = self.events.write();
        if events.len() >= self.max_events {
            events.remove(0);
        }
        events.push(event);
    }
}

// ============================================================================
// Training Data Generation
// ============================================================================

/// Generates synthetic training data for a given seed.
///
/// Identical to `crate::runtime::generate_training_data` but accessible
/// without depending on the runtime module's internal structure.
fn generate_training_data(d_in: usize, d_out: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use rand::SeedableRng;
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x: Vec<f64> = (0..d_in).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f64> = (0..d_out).map(|_| rng.gen_range(0.0..1.0)).collect();
    (x, target)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_capabilities() -> NodeCapabilities {
        NodeCapabilities {
            can_train: true,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 4096,
            cpu_cores: 8,
            storage_gb: 100,
        }
    }

    fn test_config() -> WorkerDaemonConfig {
        WorkerDaemonConfig::default()
    }

    fn test_dims() -> ModelDims {
        ModelDims {
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            num_layers: 2,
            num_heads: 0,
            activation_type: 0,
        }
    }

    #[test]
    fn test_daemon_creation() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );
        assert!(daemon.is_auto_join());
        assert!(daemon.active_round().is_none());
        assert_eq!(daemon.current_round_id(), None);
    }

    #[test]
    fn test_daemon_status_idle() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );
        let status = daemon.status();
        assert!(status.auto_join);
        assert!(!status.in_round);
        assert!(status.active_round.is_none());
        assert_eq!(status.total_rounds, 0);
    }

    #[test]
    fn test_set_auto_join() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );
        assert!(daemon.is_auto_join());

        let prev = daemon.set_auto_join(false);
        assert!(prev); // was true
        assert!(!daemon.is_auto_join());

        let prev = daemon.set_auto_join(true);
        assert!(!prev); // was false
        assert!(daemon.is_auto_join());
    }

    #[test]
    fn test_daemon_earnings_empty() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );
        let earnings = daemon.earnings();
        assert_eq!(earnings.total_earnings_wei, 0);
        assert_eq!(earnings.rounds_participated, 0);
    }

    #[test]
    fn test_round_phase_display() {
        assert_eq!(RoundPhase::Registered.to_string(), "registered");
        assert_eq!(RoundPhase::WeightsReceived.to_string(), "weights_received");
        assert_eq!(RoundPhase::Training.to_string(), "training");
        assert_eq!(RoundPhase::ProofSubmitted.to_string(), "proof_submitted");
    }

    #[test]
    fn test_worker_daemon_config_default() {
        let config = WorkerDaemonConfig::default();
        assert!(config.auto_join);
        assert_eq!(config.max_model_params, 1_000_000);
        assert_eq!(config.max_error_budget, 1.0);
        assert_eq!(config.min_deadline_slack_secs, 30);
        assert_eq!(config.max_steps_per_round, 1000);
        assert_eq!(config.max_history_records, 500);
    }

    #[test]
    fn test_worker_daemon_config_serialization() {
        let config = WorkerDaemonConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WorkerDaemonConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.auto_join, config.auto_join);
        assert_eq!(deserialized.max_model_params, config.max_model_params);
    }

    #[test]
    fn test_event_log_capping() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );
        // max_events is 200
        for i in 0..250 {
            daemon.emit(WorkerDaemonEvent::RoundEvaluated {
                round_id: i,
                accepted: true,
                reason: None,
            });
        }
        assert_eq!(daemon.recent_events().len(), 200);
    }

    #[test]
    fn test_active_round_state() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );

        // Simulate setting an active round
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 42,
                model_id: 1,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: 0,
                phase: RoundPhase::Registered,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: "test".to_string(),
                current_model_hash: [0u8; 32],
                steps_completed: 0,
                proofs_submitted: 0,
                joined_at: 1000,
            });
        }

        assert_eq!(daemon.current_round_id(), Some(42));
        let status = daemon.status();
        assert!(status.in_round);
        assert_eq!(status.active_round.unwrap().round_id, 42);
    }

    #[test]
    fn test_fail_round() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );

        // Setup active round and tracker
        daemon.tracker.write().record_join(42, 1);
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 42,
                model_id: 1,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: 0,
                phase: RoundPhase::Training,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: "test".to_string(),
                current_model_hash: [0u8; 32],
                steps_completed: 3,
                proofs_submitted: 0,
                joined_at: 1000,
            });
        }

        daemon.fail_round(42, "test failure");

        assert!(daemon.active_round().is_none());
        assert_eq!(daemon.current_round_id(), None);

        let record = daemon.tracker.read().get_round_record(42).cloned();
        assert!(record.is_some());
        assert!(!record.unwrap().success);
    }

    #[test]
    fn test_handle_round_completed_cleans_up() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );

        // Setup active round
        daemon.tracker.write().record_join(42, 1);
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 42,
                model_id: 1,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: 0,
                phase: RoundPhase::ProofSubmitted,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: "test".to_string(),
                current_model_hash: [0u8; 32],
                steps_completed: 10,
                proofs_submitted: 1,
                joined_at: 1000,
            });
        }

        daemon.handle_round_completed(42, 0.05);

        assert!(daemon.active_round().is_none());
        let record = daemon.tracker.read().get_round_record(42).cloned();
        assert!(record.is_some());
        assert!(record.unwrap().success);
    }

    #[test]
    fn test_handle_round_completed_ignores_other_rounds() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );

        // Setup active round 42
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 42,
                model_id: 1,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: 0,
                phase: RoundPhase::Training,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: "test".to_string(),
                current_model_hash: [0u8; 32],
                steps_completed: 5,
                proofs_submitted: 0,
                joined_at: 1000,
            });
        }

        // Complete a different round — should not affect our active round
        daemon.handle_round_completed(99, 0.05);
        assert_eq!(daemon.current_round_id(), Some(42));
    }

    #[test]
    fn test_handle_proof_acknowledged_valid() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );

        // Setup active round
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 42,
                model_id: 1,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: 0,
                phase: RoundPhase::ProofSubmitted,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: "test".to_string(),
                current_model_hash: [0u8; 32],
                steps_completed: 10,
                proofs_submitted: 1,
                joined_at: 1000,
            });
        }

        // Should not panic or modify state
        daemon.handle_proof_acknowledged(42, true, None);
        assert_eq!(daemon.current_round_id(), Some(42));
    }

    #[test]
    fn test_handle_worker_registered_rejected() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );

        // Setup active round
        daemon.tracker.write().record_join(42, 1);
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 42,
                model_id: 1,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: 0,
                phase: RoundPhase::Registered,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: "test".to_string(),
                current_model_hash: [0u8; 32],
                steps_completed: 0,
                proofs_submitted: 0,
                joined_at: 1000,
            });
        }

        daemon.handle_worker_registered(false, None, Some("round full"));

        // Should have failed the round
        assert!(daemon.active_round().is_none());
    }

    #[test]
    fn test_daemon_with_disabled_auto_join() {
        let config = WorkerDaemonConfig {
            auto_join: false,
            ..Default::default()
        };
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            config,
            test_capabilities(),
        );
        assert!(!daemon.is_auto_join());

        let status = daemon.status();
        assert!(!status.auto_join);
    }

    #[test]
    fn test_multiple_rounds_tracking() {
        let daemon = WorkerDaemon::new(
            PeerId::random(),
            test_config(),
            test_capabilities(),
        );

        // Simulate multiple rounds
        for round_id in 1..=5 {
            daemon.tracker.write().record_join(round_id, 1);

            if round_id % 2 == 0 {
                daemon.tracker.write().record_completion(
                    round_id, 10, 1, 0.01, 0.5, format!("proof_{}", round_id),
                );
            } else {
                daemon.tracker.write().record_failure(round_id);
            }
        }

        let earnings = daemon.earnings();
        assert_eq!(earnings.rounds_participated, 5);
        assert_eq!(earnings.rounds_succeeded, 2);
        assert_eq!(earnings.rounds_failed, 3);
    }

    // ========================================================================
    // Data Loading Tests
    // ========================================================================

    #[test]
    fn test_load_training_data_no_assignment() {
        let daemon = WorkerDaemon::new(
            PeerId::from_string("worker-1"),
            test_config(),
            test_capabilities(),
        );

        // No data assignment set → returns None (falls back to synthetic)
        let result = daemon.load_training_data(4, 2, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_load_training_data_csv_inline() {
        let daemon = WorkerDaemon::new(
            PeerId::from_string("worker-1"),
            test_config(),
            test_capabilities(),
        );

        // Set up a CSV data assignment
        let csv = "\
feat1,feat2,feat3,feat4,label1,label2
0.1,0.2,0.3,0.4,0.5,0.6
0.7,0.8,0.9,1.0,1.1,1.2
0.2,0.3,0.4,0.5,0.6,0.7
0.8,0.9,1.0,1.1,1.2,1.3";

        let assignment = distribution::DataAssignment {
            round_id: 1,
            worker_index: 0,
            total_workers: 1,
            source: distribution::DataSourceType::Inline {
                data: csv.as_bytes().to_vec(),
                format: distribution::DataFormat::Csv,
            },
            byte_offset: 0,
            byte_length: csv.len() as u64,
            num_samples: 4,
            shard_hash: None,
            shuffle_seed: 42,
            batch_size: 2,
            format: distribution::DataFormat::Csv,
            feature_columns: vec![0, 1, 2, 3],
            label_columns: vec![4, 5],
        };

        *daemon.data_assignment.write() = Some(assignment);

        let result = daemon.load_training_data(4, 2, 10);
        assert!(result.is_some());
        let batches = result.unwrap();
        // 4 rows with batch_size=2 → 2 batches
        assert_eq!(batches.len(), 2);

        // Each batch should have d_in=4 inputs and d_out=2 outputs
        for (x, t) in &batches {
            assert_eq!(x.len(), 4);
            assert_eq!(t.len(), 2);
            // Values should be non-zero (came from CSV)
            assert!(x.iter().any(|v| *v != 0.0));
            assert!(t.iter().any(|v| *v != 0.0));
        }
    }

    #[test]
    fn test_load_training_data_csv_default_columns() {
        let daemon = WorkerDaemon::new(
            PeerId::from_string("worker-1"),
            test_config(),
            test_capabilities(),
        );

        // CSV with no explicit column mapping — defaults to last d_out as labels
        let csv = "\
a,b,c,d
0.1,0.2,0.3,0.4
0.5,0.6,0.7,0.8";

        let assignment = distribution::DataAssignment {
            round_id: 1,
            worker_index: 0,
            total_workers: 1,
            source: distribution::DataSourceType::Inline {
                data: csv.as_bytes().to_vec(),
                format: distribution::DataFormat::Csv,
            },
            byte_offset: 0,
            byte_length: csv.len() as u64,
            num_samples: 2,
            shard_hash: None,
            shuffle_seed: 42,
            batch_size: 1,
            format: distribution::DataFormat::Csv,
            feature_columns: vec![],  // empty → auto-detect
            label_columns: vec![],    // empty → auto-detect
        };

        *daemon.data_assignment.write() = Some(assignment);

        // d_in=2, d_out=2 → first 2 columns are features, last 2 are labels
        let result = daemon.load_training_data(2, 2, 10);
        assert!(result.is_some());
        let batches = result.unwrap();
        assert_eq!(batches.len(), 2); // batch_size=1, 2 rows

        // First batch: features=[0.1, 0.2], labels=[0.3, 0.4]
        assert!((batches[0].0[0] - 0.1).abs() < 0.001);
        assert!((batches[0].0[1] - 0.2).abs() < 0.001);
        assert!((batches[0].1[0] - 0.3).abs() < 0.001);
        assert!((batches[0].1[1] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_load_training_data_raw_f32() {
        let daemon = WorkerDaemon::new(
            PeerId::from_string("worker-1"),
            test_config(),
            test_capabilities(),
        );

        // Create raw f32 data: 2 samples of (d_in=3, d_out=1)
        // Each sample is 4 * (3 + 1) = 16 bytes
        let mut data = Vec::new();
        // Sample 1: x=[0.1, 0.2, 0.3], t=[0.5]
        for v in &[0.1f32, 0.2, 0.3, 0.5] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        // Sample 2: x=[0.4, 0.5, 0.6], t=[0.8]
        for v in &[0.4f32, 0.5, 0.6, 0.8] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let assignment = distribution::DataAssignment {
            round_id: 1,
            worker_index: 0,
            total_workers: 1,
            source: distribution::DataSourceType::Inline {
                data: data.clone(),
                format: distribution::DataFormat::RawF32,
            },
            byte_offset: 0,
            byte_length: data.len() as u64,
            num_samples: 2,
            shard_hash: None,
            shuffle_seed: 42,
            batch_size: 1,
            format: distribution::DataFormat::RawF32,
            feature_columns: vec![],
            label_columns: vec![],
        };

        *daemon.data_assignment.write() = Some(assignment);

        let result = daemon.load_training_data(3, 1, 10);
        assert!(result.is_some());
        let batches = result.unwrap();
        assert_eq!(batches.len(), 2);

        // Verify values (f32→f64 conversion)
        assert!((batches[0].0[0] - 0.1).abs() < 0.001);
        assert!((batches[0].0[1] - 0.2).abs() < 0.001);
        assert!((batches[0].0[2] - 0.3).abs() < 0.001);
        assert!((batches[0].1[0] - 0.5).abs() < 0.001);

        assert!((batches[1].0[0] - 0.4).abs() < 0.001);
        assert!((batches[1].1[0] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_load_training_data_empty_csv() {
        let daemon = WorkerDaemon::new(
            PeerId::from_string("worker-1"),
            test_config(),
            test_capabilities(),
        );

        // CSV with only header, no data rows
        let csv = "feat1,feat2,label\n";

        let assignment = distribution::DataAssignment {
            round_id: 1,
            worker_index: 0,
            total_workers: 1,
            source: distribution::DataSourceType::Inline {
                data: csv.as_bytes().to_vec(),
                format: distribution::DataFormat::Csv,
            },
            byte_offset: 0,
            byte_length: csv.len() as u64,
            num_samples: 0,
            shard_hash: None,
            shuffle_seed: 42,
            batch_size: 1,
            format: distribution::DataFormat::Csv,
            feature_columns: vec![],
            label_columns: vec![],
        };

        *daemon.data_assignment.write() = Some(assignment);

        // Empty CSV should return None (falls back to synthetic)
        let result = daemon.load_training_data(2, 1, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_data_assignment_cleared_on_round_complete() {
        let daemon = WorkerDaemon::new(
            PeerId::from_string("worker-1"),
            test_config(),
            test_capabilities(),
        );

        // Simulate having an active round with data assignment
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 1,
                model_id: 100,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: u64::MAX,
                phase: RoundPhase::ProofSubmitted,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: String::new(),
                current_model_hash: [0u8; 32],
                steps_completed: 10,
                proofs_submitted: 1,
                joined_at: 0,
            });
        }

        // Set a data assignment
        *daemon.data_assignment.write() = Some(distribution::DataAssignment {
            round_id: 1,
            worker_index: 0,
            total_workers: 1,
            source: distribution::DataSourceType::Inline {
                data: vec![1, 2, 3],
                format: distribution::DataFormat::Csv,
            },
            byte_offset: 0,
            byte_length: 3,
            num_samples: 1,
            shard_hash: None,
            shuffle_seed: 42,
            batch_size: 1,
            format: distribution::DataFormat::Csv,
            feature_columns: vec![],
            label_columns: vec![],
        });

        assert!(daemon.data_assignment.read().is_some());

        // Complete the round
        daemon.handle_round_completed(1, 0.01);

        // Data assignment should be cleared
        assert!(daemon.data_assignment.read().is_none());
        assert!(daemon.chunk_receiver.read().is_none());
        assert!(daemon.active_round.read().is_none());
    }

    #[test]
    fn test_data_assignment_cleared_on_fail() {
        let daemon = WorkerDaemon::new(
            PeerId::from_string("worker-1"),
            test_config(),
            test_capabilities(),
        );

        // Simulate active round
        {
            let mut active = daemon.active_round.write();
            *active = Some(ActiveRoundState {
                round_id: 2,
                model_id: 100,
                model_dims: test_dims(),
                steps_per_worker: 10,
                learning_rate: 0.01,
                error_budget: 0.1,
                deadline: u64::MAX,
                phase: RoundPhase::Training,
                aggregator_id: PeerId::from_string("agg-1"),
                dataset_ref: String::new(),
                current_model_hash: [0u8; 32],
                steps_completed: 5,
                proofs_submitted: 0,
                joined_at: 0,
            });
        }

        *daemon.data_assignment.write() = Some(distribution::DataAssignment {
            round_id: 2,
            worker_index: 0,
            total_workers: 1,
            source: distribution::DataSourceType::Inline {
                data: vec![1],
                format: distribution::DataFormat::Csv,
            },
            byte_offset: 0,
            byte_length: 1,
            num_samples: 1,
            shard_hash: None,
            shuffle_seed: 42,
            batch_size: 1,
            format: distribution::DataFormat::Csv,
            feature_columns: vec![],
            label_columns: vec![],
        });

        assert!(daemon.data_assignment.read().is_some());

        // Fail the round
        daemon.fail_round(2, "test failure");

        // Everything should be cleared
        assert!(daemon.data_assignment.read().is_none());
        assert!(daemon.chunk_receiver.read().is_none());
        assert!(daemon.active_round.read().is_none());
    }
}
