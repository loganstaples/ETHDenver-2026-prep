//! Distributed Training Coordinator.
//!
//! This module provides a high-level distributed training coordinator that integrates
//! all distributed training components including:
//! - Training round state machine
//! - Worker synchronization barriers
//! - Fault tolerance and recovery
//! - Distributed checkpointing
//! - Smart contract integration
//!
//! The DistributedTrainingCoordinator orchestrates multi-worker training sessions
//! with Byzantine fault tolerance, automatic failure recovery, and on-chain commits.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use tokio::sync::{broadcast, mpsc, oneshot};

use crate::network::messages::PeerId;

use super::coordinator::{
    CoordinatorConfig, CoordinatorError, CoordinatorEvent, CoordinatorState,
    NodeId, NodeRole, PeerInfo, TrainingCoordinator,
};
use super::distributed_checkpoint::{
    CheckpointEvent, CheckpointProgress, CheckpointSyncHelper, DistributedCheckpoint,
    DistributedCheckpointConfig, DistributedCheckpointCoordinator, DistributedCheckpointId,
    ResumableTrainingState, ResumeCoordinator, ShareCheckpoint, WorkerCheckpointContribution,
};
use super::fault_tolerance::{
    FaultEvent, FaultToleranceConfig, FaultToleranceManager, ReplacementAction,
    WorkerHealth, WorkerReplacement,
};
use super::model::{ModelGradient, ModelWeights};
use super::round::RoundId;
use super::state_machine::{
    DistributedRound, StateMachineConfig, DistributedRoundId, DistributedRoundState,
    DistributedTrainingStateMachine, RoundFailureReason, RoundWorker, StateMachineEvent,
    WorkerRoundState,
};
use super::synchronization::{
    BarrierConfig, BarrierEvent, BarrierManager, BarrierResult, SyncCoordinator,
    SyncPhase,
};

// ============================================================================
// Distributed Training Configuration
// ============================================================================

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedTrainingConfig {
    /// Base coordinator configuration.
    pub coordinator_config: CoordinatorConfig,
    /// State machine configuration.
    pub round_config: StateMachineConfig,
    /// Synchronization configuration.
    pub sync_config: BarrierConfig,
    /// Fault tolerance configuration.
    pub fault_tolerance_config: FaultToleranceConfig,
    /// Checkpoint configuration.
    pub checkpoint_config: DistributedCheckpointConfig,
    /// This worker's ID.
    pub worker_id: PeerId,
    /// Minimum workers required for training.
    pub min_workers: usize,
    /// Maximum workers allowed.
    pub max_workers: usize,
    /// Enable automatic failure recovery.
    pub auto_recovery: bool,
    /// Enable automatic worker replacement.
    pub auto_replacement: bool,
    /// Smart contract address for round commits.
    pub contract_address: Option<String>,
    /// RPC endpoint for smart contract interaction.
    pub rpc_endpoint: Option<String>,
    /// Whether to submit proofs on-chain.
    pub submit_proofs_onchain: bool,
    /// Timeout for round completion.
    pub round_timeout: Duration,
    /// Timeout for gradient collection.
    pub collection_timeout: Duration,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            coordinator_config: CoordinatorConfig::default(),
            round_config: StateMachineConfig::default(),
            sync_config: BarrierConfig::default(),
            fault_tolerance_config: FaultToleranceConfig::default(),
            checkpoint_config: DistributedCheckpointConfig::default(),
            worker_id: PeerId::from_string("default-worker"),
            min_workers: 2,
            max_workers: 100,
            auto_recovery: true,
            auto_replacement: true,
            contract_address: None,
            rpc_endpoint: None,
            submit_proofs_onchain: false,
            round_timeout: Duration::from_secs(300),
            collection_timeout: Duration::from_secs(60),
        }
    }
}

// ============================================================================
// Distributed Training State
// ============================================================================

/// High-level state of distributed training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedTrainingState {
    /// Not initialized.
    Uninitialized,
    /// Initializing components.
    Initializing,
    /// Waiting for workers to join.
    WaitingForWorkers,
    /// Training in progress.
    Training,
    /// Paused (e.g., for recovery).
    Paused,
    /// Recovering from failure.
    Recovering,
    /// Training completed.
    Completed,
    /// Failed permanently.
    Failed,
    /// Shutting down.
    ShuttingDown,
}

/// Information about a worker in distributed training.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    /// Worker's peer ID.
    pub peer_id: PeerId,
    /// Worker's node ID (string representation).
    pub node_id: NodeId,
    /// Worker's stake.
    pub stake: u64,
    /// Worker's health status.
    pub health: WorkerHealth,
    /// Current round state.
    pub round_state: Option<WorkerRoundState>,
    /// Number of rounds completed.
    pub rounds_completed: u64,
    /// Number of gradients submitted.
    pub gradients_submitted: u64,
    /// Last activity timestamp.
    pub last_active: Instant,
    /// Whether worker is the current leader.
    pub is_leader: bool,
    /// MPC share index.
    pub share_index: Option<usize>,
}

impl WorkerInfo {
    /// Creates a new worker info.
    pub fn new(peer_id: PeerId, stake: u64) -> Self {
        let node_id = NodeId::new(format!("{}", peer_id));
        Self {
            peer_id,
            node_id,
            stake,
            health: WorkerHealth::Healthy,
            round_state: None,
            rounds_completed: 0,
            gradients_submitted: 0,
            last_active: Instant::now(),
            is_leader: false,
            share_index: None,
        }
    }

    /// Updates activity timestamp.
    pub fn touch(&mut self) {
        self.last_active = Instant::now();
    }
}

// ============================================================================
// Distributed Training Events
// ============================================================================

/// Events from distributed training.
#[derive(Debug, Clone)]
pub enum DistributedTrainingEvent {
    /// Training started.
    TrainingStarted {
        session_id: u64,
        num_workers: usize,
    },
    /// Worker joined training.
    WorkerJoined {
        worker_id: PeerId,
        total_workers: usize,
    },
    /// Worker left training.
    WorkerLeft {
        worker_id: PeerId,
        reason: String,
        total_workers: usize,
    },
    /// Worker replaced after failure.
    WorkerReplaced {
        old_worker: PeerId,
        new_worker: PeerId,
    },
    /// Round started.
    RoundStarted {
        round_id: DistributedRoundId,
        leader: PeerId,
    },
    /// Round phase changed.
    RoundPhaseChanged {
        round_id: DistributedRoundId,
        phase: DistributedRoundState,
    },
    /// Gradient share collected.
    GradientShareCollected {
        round_id: DistributedRoundId,
        worker_id: PeerId,
        shares_collected: usize,
        shares_required: usize,
    },
    /// Aggregation completed.
    AggregationCompleted {
        round_id: DistributedRoundId,
        num_contributors: usize,
        error_bound: f64,
    },
    /// Round committed on-chain.
    RoundCommitted {
        round_id: DistributedRoundId,
        tx_hash: Option<String>,
    },
    /// Round completed.
    RoundCompleted {
        round_id: DistributedRoundId,
        duration: Duration,
    },
    /// Round failed.
    RoundFailed {
        round_id: DistributedRoundId,
        reason: String,
    },
    /// Barrier synchronized.
    BarrierSynchronized {
        phase: SyncPhase,
        participants: usize,
    },
    /// Checkpoint created.
    CheckpointCreated {
        checkpoint_id: DistributedCheckpointId,
    },
    /// Recovery started.
    RecoveryStarted {
        from_checkpoint: Option<DistributedCheckpointId>,
    },
    /// Recovery completed.
    RecoveryCompleted {
        recovered_round: u64,
    },
    /// Training converged.
    TrainingConverged {
        total_rounds: u64,
        final_loss: f64,
    },
    /// Training failed.
    TrainingFailed {
        reason: String,
    },
    /// Fault detected.
    FaultDetected {
        worker_id: PeerId,
        fault_type: String,
    },
    /// Leader changed.
    LeaderChanged {
        old_leader: Option<PeerId>,
        new_leader: PeerId,
    },
}

// ============================================================================
// Gradient Share Collection
// ============================================================================

/// Collected gradient share from a worker.
#[derive(Debug, Clone)]
pub struct GradientShare {
    /// Worker that submitted the share.
    pub worker_id: PeerId,
    /// Share index in MPC scheme.
    pub share_index: usize,
    /// The actual share data.
    pub share_data: Vec<f32>,
    /// Error bound for this share.
    pub error_bound: f64,
    /// Commitment to the share.
    pub commitment: [u8; 32],
    /// Proof data.
    pub proof: Vec<u8>,
    /// Timestamp.
    pub timestamp: u64,
}

/// Gradient share collector for a round.
#[derive(Debug)]
pub struct GradientShareCollector {
    /// Round ID.
    round_id: DistributedRoundId,
    /// Required workers.
    required_workers: HashSet<PeerId>,
    /// Collected shares.
    shares: HashMap<PeerId, GradientShare>,
    /// MPC threshold for reconstruction.
    threshold: usize,
    /// Collection started at.
    started_at: Instant,
    /// Timeout.
    timeout: Duration,
}

impl GradientShareCollector {
    /// Creates a new share collector.
    pub fn new(
        round_id: DistributedRoundId,
        workers: Vec<PeerId>,
        threshold: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            round_id,
            required_workers: workers.into_iter().collect(),
            shares: HashMap::new(),
            threshold,
            started_at: Instant::now(),
            timeout,
        }
    }

    /// Adds a share from a worker.
    pub fn add_share(&mut self, share: GradientShare) -> Result<(), String> {
        if !self.required_workers.contains(&share.worker_id) {
            return Err(format!("Worker {} not in required workers", share.worker_id));
        }

        if self.shares.contains_key(&share.worker_id) {
            return Err(format!("Already have share from {}", share.worker_id));
        }

        let worker_id = share.worker_id.clone();
        self.shares.insert(worker_id, share);
        Ok(())
    }

    /// Checks if we have enough shares for aggregation.
    pub fn has_enough_shares(&self) -> bool {
        self.shares.len() >= self.threshold
    }

    /// Checks if collection is complete (all workers submitted).
    pub fn is_complete(&self) -> bool {
        self.required_workers
            .iter()
            .all(|w| self.shares.contains_key(w))
    }

    /// Checks if collection has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.started_at.elapsed() > self.timeout
    }

    /// Returns missing workers.
    pub fn missing_workers(&self) -> Vec<PeerId> {
        self.required_workers
            .iter()
            .filter(|w| !self.shares.contains_key(w))
            .cloned()
            .collect()
    }

    /// Returns collected shares.
    pub fn shares(&self) -> &HashMap<PeerId, GradientShare> {
        &self.shares
    }

    /// Takes all collected shares.
    pub fn take_shares(self) -> HashMap<PeerId, GradientShare> {
        self.shares
    }
}

// ============================================================================
// Distributed Training Coordinator
// ============================================================================

/// Main distributed training coordinator.
pub struct DistributedTrainingCoordinator {
    /// Configuration.
    config: DistributedTrainingConfig,
    /// Current state.
    state: DistributedTrainingState,
    /// Training session ID.
    session_id: u64,
    /// Base coordinator.
    coordinator: TrainingCoordinator,
    /// Distributed round state machine.
    state_machine: DistributedTrainingStateMachine,
    /// Synchronization coordinator.
    sync_coordinator: SyncCoordinator,
    /// Fault tolerance manager.
    fault_manager: FaultToleranceManager,
    /// Distributed checkpoint coordinator.
    checkpoint_coordinator: Arc<DistributedCheckpointCoordinator>,
    /// Resume coordinator.
    resume_coordinator: ResumeCoordinator,
    /// Active workers.
    workers: HashMap<PeerId, WorkerInfo>,
    /// Current leader.
    current_leader: Option<PeerId>,
    /// Current round share collector.
    share_collector: Option<GradientShareCollector>,
    /// Current checkpoint in progress.
    active_checkpoint: Option<DistributedCheckpointId>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<DistributedTrainingEvent>,
    /// Pending events.
    pending_events: Vec<DistributedTrainingEvent>,
    /// Round counter.
    round_counter: u64,
    /// Total rounds completed.
    rounds_completed: u64,
    /// Training started at.
    started_at: Option<Instant>,
    /// Smart contract client for on-chain operations.
    sc_client: Option<Arc<crate::sc_client::SCClient>>,
    /// Model ID on the smart contract.
    model_id: Option<u64>,
    /// Current model commitment hash.
    model_commitment: [u8; 32],
    /// Accumulated error bound.
    total_error_bound: f64,
}

impl DistributedTrainingCoordinator {
    /// Creates a new distributed training coordinator.
    pub fn new(
        config: DistributedTrainingConfig,
    ) -> Result<Self, DistributedCoordinatorError> {
        // Create base coordinator
        let coordinator = TrainingCoordinator::new(config.coordinator_config.clone());

        // Create state machine
        let state_machine = DistributedTrainingStateMachine::new(config.round_config.clone());

        // Create sync coordinator
        let sync_coordinator = SyncCoordinator::new(config.sync_config.clone());

        // Create fault tolerance manager
        let fault_manager = FaultToleranceManager::new(config.fault_tolerance_config.clone());

        // Create checkpoint coordinator
        let checkpoint_coordinator = Arc::new(
            DistributedCheckpointCoordinator::new(config.checkpoint_config.clone())
                .map_err(|e| DistributedCoordinatorError::InitializationFailed(e.to_string()))?
        );

        // Create resume coordinator
        let resume_coordinator = ResumeCoordinator::new(checkpoint_coordinator.clone());

        let (event_tx, _) = broadcast::channel(256);

        let session_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(Self {
            config,
            state: DistributedTrainingState::Uninitialized,
            session_id,
            coordinator,
            state_machine,
            sync_coordinator,
            fault_manager,
            checkpoint_coordinator,
            resume_coordinator,
            workers: HashMap::new(),
            current_leader: None,
            share_collector: None,
            active_checkpoint: None,
            event_tx,
            pending_events: Vec::new(),
            round_counter: 0,
            rounds_completed: 0,
            started_at: None,
            sc_client: None,
            model_id: None,
            model_commitment: [0u8; 32],
            total_error_bound: 0.0,
        })
    }

    /// Sets the smart contract client for on-chain operations.
    pub fn set_sc_client(&mut self, client: Arc<crate::sc_client::SCClient>) {
        self.sc_client = Some(client);
    }

    /// Sets the model ID for on-chain tracking.
    pub fn set_model_id(&mut self, model_id: u64) {
        self.model_id = Some(model_id);
    }

    /// Sets the initial model commitment.
    pub fn set_model_commitment(&mut self, commitment: [u8; 32]) {
        self.model_commitment = commitment;
    }

    /// Returns the current model commitment.
    pub fn model_commitment(&self) -> [u8; 32] {
        self.model_commitment
    }

    /// Returns the model ID.
    pub fn model_id(&self) -> Option<u64> {
        self.model_id
    }

    /// Returns the total accumulated error bound.
    pub fn total_error_bound(&self) -> f64 {
        self.total_error_bound
    }

    /// Subscribes to training events.
    pub fn subscribe(&self) -> broadcast::Receiver<DistributedTrainingEvent> {
        self.event_tx.subscribe()
    }

    /// Returns the current state.
    pub fn state(&self) -> DistributedTrainingState {
        self.state
    }

    /// Returns the session ID.
    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Returns active workers.
    pub fn workers(&self) -> &HashMap<PeerId, WorkerInfo> {
        &self.workers
    }

    /// Returns the current leader.
    pub fn current_leader(&self) -> Option<PeerId> {
        self.current_leader.clone()
    }

    /// Returns whether this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.current_leader == Some(self.config.worker_id.clone())
    }

    /// Returns rounds completed.
    pub fn rounds_completed(&self) -> u64 {
        self.rounds_completed
    }

    // ========================================================================
    // Initialization and Worker Management
    // ========================================================================

    /// Initializes the coordinator with a model.
    pub fn initialize(&mut self, model: ModelWeights) -> Result<(), DistributedCoordinatorError> {
        if self.state != DistributedTrainingState::Uninitialized {
            return Err(DistributedCoordinatorError::AlreadyInitialized);
        }

        self.state = DistributedTrainingState::Initializing;

        // Initialize base coordinator
        self.coordinator.initialize(model)
            .map_err(|e| DistributedCoordinatorError::CoordinatorError(e))?;

        // Register self as a worker
        let self_info = WorkerInfo::new(self.config.worker_id.clone(), self.config.coordinator_config.stake);
        self.workers.insert(self.config.worker_id.clone(), self_info);

        // Initialize fault manager with self
        self.fault_manager.register_worker(self.config.worker_id.clone());

        self.state = DistributedTrainingState::WaitingForWorkers;

        Ok(())
    }

    /// Adds a worker to the training session.
    pub fn add_worker(
        &mut self,
        worker_id: PeerId,
        stake: u64,
        address: String,
    ) -> Result<(), DistributedCoordinatorError> {
        if self.workers.len() >= self.config.max_workers {
            return Err(DistributedCoordinatorError::TooManyWorkers);
        }

        if self.workers.contains_key(&worker_id) {
            return Err(DistributedCoordinatorError::WorkerAlreadyExists(worker_id));
        }

        // Add to base coordinator
        self.coordinator.register_peer(
            NodeId::new(format!("{}", worker_id)),
            stake,
            address,
        );

        // Add to our workers
        let info = WorkerInfo::new(worker_id.clone(), stake);
        self.workers.insert(worker_id.clone(), info);

        // Register with fault manager
        self.fault_manager.register_worker(worker_id.clone());

        self.pending_events.push(DistributedTrainingEvent::WorkerJoined {
            worker_id,
            total_workers: self.workers.len(),
        });

        // Check if we have enough workers to start
        if self.state == DistributedTrainingState::WaitingForWorkers
            && self.workers.len() >= self.config.min_workers
        {
            // Ready to start training
        }

        Ok(())
    }

    /// Removes a worker from the training session.
    pub fn remove_worker(
        &mut self,
        worker_id: PeerId,
        reason: String,
    ) -> Result<(), DistributedCoordinatorError> {
        if !self.workers.contains_key(&worker_id) {
            return Err(DistributedCoordinatorError::WorkerNotFound(worker_id));
        }

        // Remove from base coordinator
        self.coordinator.unregister_peer(&NodeId::new(format!("{}", worker_id)));

        // Remove from our workers
        self.workers.remove(&worker_id);

        // Unregister from fault manager
        self.fault_manager.unregister_worker(&worker_id);

        self.pending_events.push(DistributedTrainingEvent::WorkerLeft {
            worker_id,
            reason,
            total_workers: self.workers.len(),
        });

        // Check if we need to trigger recovery
        if self.state == DistributedTrainingState::Training
            && self.workers.len() < self.config.min_workers
        {
            self.state = DistributedTrainingState::Paused;
        }

        Ok(())
    }

    // ========================================================================
    // Training Lifecycle
    // ========================================================================

    /// Starts distributed training.
    pub async fn start_training(&mut self) -> Result<(), DistributedCoordinatorError> {
        if self.state != DistributedTrainingState::WaitingForWorkers {
            return Err(DistributedCoordinatorError::WrongState(self.state));
        }

        if self.workers.len() < self.config.min_workers {
            return Err(DistributedCoordinatorError::InsufficientWorkers {
                required: self.config.min_workers,
                available: self.workers.len(),
            });
        }

        self.state = DistributedTrainingState::Training;
        self.started_at = Some(Instant::now());

        // Elect initial leader
        self.elect_leader();

        self.pending_events.push(DistributedTrainingEvent::TrainingStarted {
            session_id: self.session_id,
            num_workers: self.workers.len(),
        });

        // Start the first round
        self.start_round().await?;

        Ok(())
    }

    /// Starts a new training round.
    pub async fn start_round(&mut self) -> Result<DistributedRoundId, DistributedCoordinatorError> {
        if self.state != DistributedTrainingState::Training {
            return Err(DistributedCoordinatorError::WrongState(self.state));
        }

        self.round_counter += 1;

        // Create distributed round
        let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
        let round_id = self.state_machine.create_round(worker_ids.clone())
            .map_err(|e| DistributedCoordinatorError::StateMachineError(e))?;

        // Start round in base coordinator
        self.coordinator.start_round()
            .map_err(|e| DistributedCoordinatorError::CoordinatorError(e))?;

        // Elect leader for this round
        self.elect_leader_for_round(round_id);

        // Initialize barrier for worker synchronization
        self.sync_coordinator.create_barrier(
            SyncPhase::Ready,
            worker_ids.clone(),
            self.config.collection_timeout,
        );

        // Initialize share collector
        let threshold = (worker_ids.len() * 2 + 2) / 3; // 2/3 threshold
        self.share_collector = Some(GradientShareCollector::new(
            round_id,
            worker_ids,
            threshold,
            self.config.collection_timeout,
        ));

        self.pending_events.push(DistributedTrainingEvent::RoundStarted {
            round_id,
            leader: self.current_leader.clone().unwrap_or_else(|| self.config.worker_id.clone()),
        });

        Ok(round_id)
    }

    /// Submits a gradient share for the current round.
    pub async fn submit_gradient_share(
        &mut self,
        share: GradientShare,
    ) -> Result<(), DistributedCoordinatorError> {
        let collector = self.share_collector.as_mut()
            .ok_or(DistributedCoordinatorError::NoActiveRound)?;

        let round_id = collector.round_id;
        let worker_id = share.worker_id.clone();

        // Add share to collector
        collector.add_share(share)
            .map_err(|e| DistributedCoordinatorError::InvalidShare(e))?;

        // Update worker activity
        if let Some(worker) = self.workers.get_mut(&worker_id) {
            worker.touch();
            worker.gradients_submitted += 1;
        }

        // Record heartbeat
        self.fault_manager.record_heartbeat(&worker_id, 10.0);

        // Emit event
        let shares_collected = collector.shares().len();
        let shares_required = collector.threshold;

        self.pending_events.push(DistributedTrainingEvent::GradientShareCollected {
            round_id,
            worker_id: worker_id.clone(),
            shares_collected,
            shares_required,
        });

        // Transition state machine
        self.state_machine.worker_submitted(round_id, worker_id.clone())
            .map_err(|e| DistributedCoordinatorError::StateMachineError(e))?;

        // Check if we can proceed to aggregation
        if collector.has_enough_shares() {
            self.transition_to_aggregation(round_id).await?;
        }

        Ok(())
    }

    /// Transitions to aggregation phase.
    async fn transition_to_aggregation(
        &mut self,
        round_id: DistributedRoundId,
    ) -> Result<(), DistributedCoordinatorError> {
        // Transition state machine
        self.state_machine.start_aggregation(round_id)
            .map_err(|e| DistributedCoordinatorError::StateMachineError(e))?;

        self.pending_events.push(DistributedTrainingEvent::RoundPhaseChanged {
            round_id,
            phase: DistributedRoundState::Aggregating,
        });

        // Synchronize workers at aggregation barrier
        let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
        self.sync_coordinator.create_barrier(
            SyncPhase::AggregationReady,
            worker_ids,
            Duration::from_secs(30),
        );

        Ok(())
    }

    /// Performs aggregation (leader only).
    pub async fn aggregate(&mut self) -> Result<(), DistributedCoordinatorError> {
        if !self.is_leader() {
            return Err(DistributedCoordinatorError::NotLeader);
        }

        let collector = self.share_collector.take()
            .ok_or(DistributedCoordinatorError::NoActiveRound)?;

        let round_id = collector.round_id;
        let shares = collector.take_shares();

        // Perform base coordinator aggregation
        let result = self.coordinator.aggregate()
            .map_err(|e| DistributedCoordinatorError::CoordinatorError(e))?;

        // Transition state machine
        self.state_machine.complete_aggregation(round_id)
            .map_err(|e| DistributedCoordinatorError::StateMachineError(e))?;

        self.pending_events.push(DistributedTrainingEvent::AggregationCompleted {
            round_id,
            num_contributors: shares.len(),
            error_bound: result.combined_error_bound,
        });

        // Proceed to commit phase
        self.transition_to_commit(round_id).await?;

        Ok(())
    }

    /// Transitions to commit phase.
    async fn transition_to_commit(
        &mut self,
        round_id: DistributedRoundId,
    ) -> Result<(), DistributedCoordinatorError> {
        self.state_machine.start_commit(round_id)
            .map_err(|e| DistributedCoordinatorError::StateMachineError(e))?;

        self.pending_events.push(DistributedTrainingEvent::RoundPhaseChanged {
            round_id,
            phase: DistributedRoundState::Committing,
        });

        // Submit to smart contract if configured
        let tx_hash = if self.config.submit_proofs_onchain {
            self.submit_to_contract(round_id).await?
        } else {
            None
        };

        self.pending_events.push(DistributedTrainingEvent::RoundCommitted {
            round_id,
            tx_hash,
        });

        // Complete the round
        self.complete_round(round_id).await?;

        Ok(())
    }

    /// Submits round proof to smart contract.
    async fn submit_to_contract(
        &self,
        round_id: DistributedRoundId,
    ) -> Result<Option<String>, DistributedCoordinatorError> {
        // Check if we have the necessary components for on-chain submission
        let client = match &self.sc_client {
            Some(c) => c,
            None => return Ok(None), // No client configured, skip on-chain submission
        };

        let model_id = match self.model_id {
            Some(id) => id,
            None => return Ok(None), // No model ID, skip on-chain submission
        };

        // Get the current round from state machine to extract proof data
        let round = match self.state_machine.current_round() {
            Some(r) => r,
            None => return Err(DistributedCoordinatorError::NoActiveRound),
        };

        // Collect proofs from submitted workers
        let submitted_workers: Vec<_> = round.workers.iter()
            .filter(|(_, w)| w.state == crate::training::state_machine::WorkerRoundState::Submitted)
            .collect();

        if submitted_workers.is_empty() {
            return Err(DistributedCoordinatorError::InvalidShare("No submitted proofs".to_string()));
        }

        // Compute public inputs for the combined proof
        // Split the commitment hashes into lo/hi halves
        let old_commitment = round.initial_commitment;
        let new_commitment = round.new_commitment.unwrap_or(old_commitment);

        let old_hash_lo = ethers::types::U256::from_big_endian(&old_commitment[..16]);
        let old_hash_hi = ethers::types::U256::from_big_endian(&old_commitment[16..]);
        let new_hash_lo = ethers::types::U256::from_big_endian(&new_commitment[..16]);
        let new_hash_hi = ethers::types::U256::from_big_endian(&new_commitment[16..]);

        // Compute aggregate error bound
        let total_error: f64 = submitted_workers.iter()
            .filter_map(|(_, w)| w.error_bound)
            .sum();

        let inputs = crate::sc_client::TrainingProofInputs {
            old_hash_lo,
            old_hash_hi,
            new_hash_lo,
            new_hash_hi,
            loss: ethers::types::U256::zero(), // Would come from training metrics
            error_bound: ethers::types::U256::from((total_error * 1e18) as u64),
            step_number: ethers::types::U256::from(round_id.round_number),
            error_checksum: ethers::types::U256::zero(),
        };

        // Create a combined proof (in production, this would be a proper aggregated proof)
        // For now, we just use an empty proof placeholder that would be replaced with actual ZK proof
        let combined_proof = vec![0u8; 32]; // Placeholder

        // Submit to the smart contract
        match client.submit_proof(model_id, round_id.round_number, combined_proof, &inputs).await {
            Ok(receipt) => {
                let tx_hash = format!("{:?}", receipt.transaction_hash);
                Ok(Some(tx_hash))
            }
            Err(e) => {
                // Log the error but don't fail the round for on-chain issues
                // The training can continue locally even if on-chain submission fails
                tracing::warn!("Failed to submit proof on-chain: {}", e);
                Err(DistributedCoordinatorError::CheckpointError(format!("On-chain submission failed: {}", e)))
            }
        }
    }

    /// Registers a model on-chain and returns the model ID.
    pub async fn register_model_onchain(
        &mut self,
        ipfs_hash: &str,
        initial_commitment: [u8; 32],
        min_stake: u64,
    ) -> Result<u64, DistributedCoordinatorError> {
        let client = self.sc_client.as_ref()
            .ok_or(DistributedCoordinatorError::InitializationFailed("No SC client configured".to_string()))?;

        let commitment = ethers::types::U256::from_big_endian(&initial_commitment);
        let stake = ethers::types::U256::from(min_stake);

        // Default model architecture: 2-layer MLP with ReLU
        let model_arch = (2u32, 2u32, 1u32, 2u32, 0u8);
        let (_, model_id) = client.register_model(ipfs_hash, commitment, stake, model_arch).await
            .map_err(|e| DistributedCoordinatorError::CheckpointError(format!("Failed to register model: {}", e)))?;

        self.model_id = Some(model_id);
        self.model_commitment = initial_commitment;

        Ok(model_id)
    }

    /// Starts a round on-chain.
    pub async fn start_round_onchain(
        &self,
        duration_secs: u64,
    ) -> Result<(), DistributedCoordinatorError> {
        let client = self.sc_client.as_ref()
            .ok_or(DistributedCoordinatorError::InitializationFailed("No SC client configured".to_string()))?;

        let model_id = self.model_id
            .ok_or(DistributedCoordinatorError::InitializationFailed("No model ID set".to_string()))?;

        client.start_round(model_id, duration_secs).await
            .map_err(|e| DistributedCoordinatorError::CheckpointError(format!("Failed to start round: {}", e)))?;

        Ok(())
    }

    /// Stakes on the model (for worker registration).
    pub async fn stake_onchain(
        &self,
        amount: u64,
    ) -> Result<(), DistributedCoordinatorError> {
        let client = self.sc_client.as_ref()
            .ok_or(DistributedCoordinatorError::InitializationFailed("No SC client configured".to_string()))?;

        let model_id = self.model_id
            .ok_or(DistributedCoordinatorError::InitializationFailed("No model ID set".to_string()))?;

        let stake_amount = ethers::types::U256::from(amount);

        client.stake(model_id, stake_amount).await
            .map_err(|e| DistributedCoordinatorError::CheckpointError(format!("Failed to stake: {}", e)))?;

        Ok(())
    }

    /// Gets the current model state from the smart contract.
    pub async fn get_model_state_onchain(&self) -> Result<crate::sc_client::ModelState, DistributedCoordinatorError> {
        let client = self.sc_client.as_ref()
            .ok_or(DistributedCoordinatorError::InitializationFailed("No SC client configured".to_string()))?;

        let model_id = self.model_id
            .ok_or(DistributedCoordinatorError::InitializationFailed("No model ID set".to_string()))?;

        client.get_model_state(model_id).await
            .map_err(|e| DistributedCoordinatorError::CheckpointError(format!("Failed to get model state: {}", e)))
    }

    /// Completes a training round.
    async fn complete_round(
        &mut self,
        round_id: DistributedRoundId,
    ) -> Result<(), DistributedCoordinatorError> {
        let start_time = self.started_at.unwrap_or_else(Instant::now);

        // Complete in base coordinator
        self.coordinator.complete_round()
            .map_err(|e| DistributedCoordinatorError::CoordinatorError(e))?;

        // Complete in state machine
        self.state_machine.complete_round(round_id)
            .map_err(|e| DistributedCoordinatorError::StateMachineError(e))?;

        self.rounds_completed += 1;

        // Update worker stats
        for worker in self.workers.values_mut() {
            worker.rounds_completed += 1;
        }

        self.pending_events.push(DistributedTrainingEvent::RoundCompleted {
            round_id,
            duration: start_time.elapsed(),
        });

        // Check if we should checkpoint
        if self.should_checkpoint() {
            self.create_checkpoint().await?;
        }

        // Check for convergence
        if self.coordinator.metrics().has_converged() {
            self.state = DistributedTrainingState::Completed;
            self.pending_events.push(DistributedTrainingEvent::TrainingConverged {
                total_rounds: self.rounds_completed,
                final_loss: self.coordinator.metrics().current_loss().unwrap_or(0.0),
            });
        }

        Ok(())
    }

    // ========================================================================
    // Leader Election
    // ========================================================================

    /// Elects a leader from available workers.
    fn elect_leader(&mut self) {
        let leader = self.compute_leader(0);
        let old_leader = self.current_leader.clone();
        self.current_leader = Some(leader.clone());

        // Update worker info
        for (id, worker) in self.workers.iter_mut() {
            worker.is_leader = *id == leader;
        }

        if old_leader != self.current_leader {
            self.pending_events.push(DistributedTrainingEvent::LeaderChanged {
                old_leader,
                new_leader: leader,
            });
        }
    }

    /// Elects a leader for a specific round.
    fn elect_leader_for_round(&mut self, round_id: DistributedRoundId) {
        let leader = self.compute_leader(round_id.round_number);
        let old_leader = self.current_leader.clone();
        self.current_leader = Some(leader.clone());

        // Update worker info
        for (id, worker) in self.workers.iter_mut() {
            worker.is_leader = *id == leader;
        }

        if old_leader != self.current_leader {
            self.pending_events.push(DistributedTrainingEvent::LeaderChanged {
                old_leader,
                new_leader: leader,
            });
        }
    }

    /// Computes leader for a given round using deterministic rotation.
    fn compute_leader(&self, round: u64) -> PeerId {
        let mut worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
        worker_ids.sort_by_key(|id| id.0.clone());

        if worker_ids.is_empty() {
            return self.config.worker_id.clone();
        }

        let leader_idx = (round as usize) % worker_ids.len();
        worker_ids[leader_idx].clone()
    }

    // ========================================================================
    // Checkpointing
    // ========================================================================

    /// Checks if a checkpoint should be created.
    fn should_checkpoint(&self) -> bool {
        let interval = self.config.checkpoint_config.base_config.checkpoint_interval;
        self.rounds_completed > 0 && self.rounds_completed % interval == 0
    }

    /// Creates a distributed checkpoint.
    pub async fn create_checkpoint(&mut self) -> Result<DistributedCheckpointId, DistributedCoordinatorError> {
        let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();

        let checkpoint_id = self.checkpoint_coordinator
            .initiate_checkpoint(self.rounds_completed, worker_ids)
            .await
            .map_err(|e| DistributedCoordinatorError::CheckpointError(e.to_string()))?;

        self.active_checkpoint = Some(checkpoint_id);

        self.pending_events.push(DistributedTrainingEvent::CheckpointCreated {
            checkpoint_id,
        });

        Ok(checkpoint_id)
    }

    /// Submits this worker's checkpoint contribution.
    pub async fn submit_checkpoint_contribution(
        &mut self,
        contribution: WorkerCheckpointContribution,
    ) -> Result<(), DistributedCoordinatorError> {
        let checkpoint_id = self.active_checkpoint
            .ok_or(DistributedCoordinatorError::NoActiveCheckpoint)?;

        self.checkpoint_coordinator
            .contribute(checkpoint_id, contribution)
            .await
            .map_err(|e| DistributedCoordinatorError::CheckpointError(e.to_string()))?;

        Ok(())
    }

    // ========================================================================
    // Fault Tolerance
    // ========================================================================

    /// Processes heartbeat from a worker.
    pub fn process_heartbeat(&mut self, worker_id: PeerId) {
        self.fault_manager.record_heartbeat(&worker_id, 10.0);

        if let Some(worker) = self.workers.get_mut(&worker_id) {
            worker.touch();
            worker.health = WorkerHealth::Healthy;
        }
    }

    /// Checks for worker failures.
    pub async fn check_failures(&mut self) -> Vec<PeerId> {
        let failed = self.fault_manager.detect_failures();

        for worker_id in &failed {
            if let Some(worker) = self.workers.get_mut(worker_id) {
                worker.health = WorkerHealth::Failed;
            }

            self.pending_events.push(DistributedTrainingEvent::FaultDetected {
                worker_id: worker_id.clone(),
                fault_type: "Heartbeat timeout".to_string(),
            });
        }

        // Handle automatic recovery if enabled
        if self.config.auto_recovery && !failed.is_empty() {
            self.handle_worker_failures(&failed).await;
        }

        failed
    }

    /// Handles worker failures.
    async fn handle_worker_failures(&mut self, failed_workers: &[PeerId]) {
        for worker_id in failed_workers {
            // Remove failed worker
            let _ = self.remove_worker(worker_id.clone(), "Heartbeat timeout".to_string());

            // Check if we need to pause training
            if self.workers.len() < self.config.min_workers {
                self.state = DistributedTrainingState::Paused;

                // Trigger emergency checkpoint
                if self.config.auto_recovery {
                    let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
                    let _ = self.checkpoint_coordinator
                        .emergency_checkpoint(
                            self.rounds_completed,
                            worker_ids,
                            format!("Worker {} failed", worker_id),
                        )
                        .await;
                }
            }

            // Check if leader failed
            if self.current_leader.as_ref() == Some(worker_id) {
                self.elect_leader();
            }
        }
    }

    /// Attempts recovery from a checkpoint.
    pub async fn recover_from_checkpoint(
        &mut self,
        checkpoint_id: Option<DistributedCheckpointId>,
    ) -> Result<(), DistributedCoordinatorError> {
        self.state = DistributedTrainingState::Recovering;

        self.pending_events.push(DistributedTrainingEvent::RecoveryStarted {
            from_checkpoint: checkpoint_id,
        });

        // Find checkpoint to recover from
        let checkpoint = if let Some(id) = checkpoint_id {
            self.resume_coordinator.find_checkpoint(id)
        } else {
            self.resume_coordinator.find_resume_checkpoint()
        };

        if let Some(ckpt) = checkpoint {
            // Verify workers can resume
            let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
            self.resume_coordinator
                .verify_resume_compatibility(&ckpt, &worker_ids)
                .map_err(|e| DistributedCoordinatorError::CheckpointError(e.to_string()))?;

            // Resume from checkpoint
            let recovered_round = ckpt.progress.current_round;

            self.pending_events.push(DistributedTrainingEvent::RecoveryCompleted {
                recovered_round,
            });
        }

        self.state = DistributedTrainingState::Training;

        Ok(())
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Waits for all workers at a barrier.
    pub async fn wait_at_barrier(
        &mut self,
        phase: SyncPhase,
    ) -> Result<BarrierResult, DistributedCoordinatorError> {
        let result = self.sync_coordinator
            .arrive_and_wait(phase, self.config.worker_id.clone(), self.config.collection_timeout)
            .await;

        if matches!(result, BarrierResult::AllArrived { .. }) {
            self.pending_events.push(DistributedTrainingEvent::BarrierSynchronized {
                phase,
                participants: self.workers.len(),
            });
        }

        Ok(result)
    }

    // ========================================================================
    // Event Handling
    // ========================================================================

    /// Drains pending events.
    pub fn drain_events(&mut self) -> Vec<DistributedTrainingEvent> {
        let events = std::mem::take(&mut self.pending_events);

        // Broadcast events
        for event in &events {
            let _ = self.event_tx.send(event.clone());
        }

        events
    }

    /// Processes state machine events.
    pub fn process_state_machine_events(&mut self) {
        let events = self.state_machine.drain_events();

        for event in events {
            match event {
                StateMachineEvent::RoundCreated { round_id: _ } => {
                    // Already handled in start_round
                }
                StateMachineEvent::StateTransition {
                    round_id,
                    from: _,
                    to,
                } => {
                    self.pending_events.push(DistributedTrainingEvent::RoundPhaseChanged {
                        round_id,
                        phase: to,
                    });
                }
                StateMachineEvent::WorkerJoined { round_id: _, peer_id: _, worker_count: _ } => {
                    // Update worker state
                }
                StateMachineEvent::RoundCompleted { round_id: _, duration: _, workers_participated: _ } => {
                    // Already handled in complete_round
                }
                StateMachineEvent::RoundFailed { round_id, reason, message: _ } => {
                    self.pending_events.push(DistributedTrainingEvent::RoundFailed {
                        round_id,
                        reason: format!("{:?}", reason),
                    });
                }
                _ => {}
            }
        }
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    /// Shuts down the coordinator gracefully.
    pub async fn shutdown(&mut self) {
        self.state = DistributedTrainingState::ShuttingDown;

        // Create final checkpoint
        let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
        let _ = self.checkpoint_coordinator
            .emergency_checkpoint(
                self.rounds_completed,
                worker_ids,
                "Shutdown".to_string(),
            )
            .await;

        // Shutdown base coordinator
        self.coordinator.shutdown();
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Distributed coordinator errors.
#[derive(Debug)]
pub enum DistributedCoordinatorError {
    /// Already initialized.
    AlreadyInitialized,
    /// Initialization failed.
    InitializationFailed(String),
    /// Wrong state for operation.
    WrongState(DistributedTrainingState),
    /// Insufficient workers.
    InsufficientWorkers { required: usize, available: usize },
    /// Too many workers.
    TooManyWorkers,
    /// Worker already exists.
    WorkerAlreadyExists(PeerId),
    /// Worker not found.
    WorkerNotFound(PeerId),
    /// Not the leader.
    NotLeader,
    /// No active round.
    NoActiveRound,
    /// No active checkpoint.
    NoActiveCheckpoint,
    /// Invalid share.
    InvalidShare(String),
    /// Coordinator error.
    CoordinatorError(CoordinatorError),
    /// State machine error.
    StateMachineError(String),
    /// Checkpoint error.
    CheckpointError(String),
    /// Synchronization error.
    SyncError(String),
    /// Recovery failed.
    RecoveryFailed(String),
}

impl std::fmt::Display for DistributedCoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInitialized => write!(f, "Already initialized"),
            Self::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            Self::WrongState(state) => write!(f, "Wrong state: {:?}", state),
            Self::InsufficientWorkers { required, available } => {
                write!(f, "Insufficient workers: need {}, have {}", required, available)
            }
            Self::TooManyWorkers => write!(f, "Too many workers"),
            Self::WorkerAlreadyExists(id) => write!(f, "Worker already exists: {}", id),
            Self::WorkerNotFound(id) => write!(f, "Worker not found: {}", id),
            Self::NotLeader => write!(f, "Not the leader"),
            Self::NoActiveRound => write!(f, "No active round"),
            Self::NoActiveCheckpoint => write!(f, "No active checkpoint"),
            Self::InvalidShare(msg) => write!(f, "Invalid share: {}", msg),
            Self::CoordinatorError(e) => write!(f, "Coordinator error: {}", e),
            Self::StateMachineError(e) => write!(f, "State machine error: {}", e),
            Self::CheckpointError(e) => write!(f, "Checkpoint error: {}", e),
            Self::SyncError(e) => write!(f, "Synchronization error: {}", e),
            Self::RecoveryFailed(e) => write!(f, "Recovery failed: {}", e),
        }
    }
}

impl std::error::Error for DistributedCoordinatorError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_peer_id(id: u8) -> PeerId {
        PeerId::from_string(&format!("test-peer-{}", id))
    }

    fn create_test_config() -> DistributedTrainingConfig {
        DistributedTrainingConfig {
            worker_id: create_test_peer_id(1),
            min_workers: 2,
            max_workers: 10,
            ..Default::default()
        }
    }

    #[test]
    fn test_gradient_share_collector() {
        let round_id = DistributedRoundId {
            session_id: 1,
            round_number: 1,
        };

        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
            create_test_peer_id(3),
        ];

        let mut collector = GradientShareCollector::new(
            round_id,
            workers.clone(),
            2,
            Duration::from_secs(60),
        );

        assert!(!collector.has_enough_shares());
        assert!(!collector.is_complete());

        // Add first share
        let share1 = GradientShare {
            worker_id: create_test_peer_id(1),
            share_index: 0,
            share_data: vec![1.0, 2.0],
            error_bound: 0.01,
            commitment: [0u8; 32],
            proof: vec![],
            timestamp: 12345,
        };
        collector.add_share(share1).unwrap();

        assert!(!collector.has_enough_shares());

        // Add second share
        let share2 = GradientShare {
            worker_id: create_test_peer_id(2),
            share_index: 1,
            share_data: vec![3.0, 4.0],
            error_bound: 0.02,
            commitment: [1u8; 32],
            proof: vec![],
            timestamp: 12346,
        };
        collector.add_share(share2).unwrap();

        assert!(collector.has_enough_shares());
        assert!(!collector.is_complete());

        // Add third share
        let share3 = GradientShare {
            worker_id: create_test_peer_id(3),
            share_index: 2,
            share_data: vec![5.0, 6.0],
            error_bound: 0.03,
            commitment: [2u8; 32],
            proof: vec![],
            timestamp: 12347,
        };
        collector.add_share(share3).unwrap();

        assert!(collector.is_complete());
    }

    #[test]
    fn test_gradient_share_collector_duplicate() {
        let round_id = DistributedRoundId {
            session_id: 1,
            round_number: 1,
        };

        let workers = vec![create_test_peer_id(1), create_test_peer_id(2)];
        let mut collector = GradientShareCollector::new(
            round_id,
            workers,
            2,
            Duration::from_secs(60),
        );

        let share = GradientShare {
            worker_id: create_test_peer_id(1),
            share_index: 0,
            share_data: vec![1.0],
            error_bound: 0.01,
            commitment: [0u8; 32],
            proof: vec![],
            timestamp: 12345,
        };

        collector.add_share(share.clone()).unwrap();

        // Duplicate should fail
        let result = collector.add_share(share);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_share_collector_unknown_worker() {
        let round_id = DistributedRoundId {
            session_id: 1,
            round_number: 1,
        };

        let workers = vec![create_test_peer_id(1), create_test_peer_id(2)];
        let mut collector = GradientShareCollector::new(
            round_id,
            workers,
            2,
            Duration::from_secs(60),
        );

        let share = GradientShare {
            worker_id: create_test_peer_id(99), // Unknown worker
            share_index: 0,
            share_data: vec![1.0],
            error_bound: 0.01,
            commitment: [0u8; 32],
            proof: vec![],
            timestamp: 12345,
        };

        let result = collector.add_share(share);
        assert!(result.is_err());
    }

    #[test]
    fn test_worker_info() {
        let worker_id = create_test_peer_id(1);
        let mut info = WorkerInfo::new(worker_id, 1000);

        assert_eq!(info.stake, 1000);
        assert_eq!(info.health, WorkerHealth::Healthy);
        assert_eq!(info.rounds_completed, 0);
        assert!(!info.is_leader);

        let old_time = info.last_active;
        std::thread::sleep(std::time::Duration::from_millis(10));
        info.touch();
        assert!(info.last_active > old_time);
    }

    #[test]
    fn test_distributed_training_state() {
        let state = DistributedTrainingState::Uninitialized;
        assert_eq!(state, DistributedTrainingState::Uninitialized);

        let state = DistributedTrainingState::Training;
        assert_eq!(state, DistributedTrainingState::Training);
    }

    #[test]
    fn test_leader_computation() {
        // This would require a full coordinator setup
        // Testing the basic logic separately
        let mut worker_ids = vec![
            create_test_peer_id(3),
            create_test_peer_id(1),
            create_test_peer_id(2),
        ];
        worker_ids.sort_by_key(|id| id.0.clone());

        // After sorting: test-peer-1, test-peer-2, test-peer-3
        assert_eq!(worker_ids[0].0, "test-peer-1");
        assert_eq!(worker_ids[1].0, "test-peer-2");
        assert_eq!(worker_ids[2].0, "test-peer-3");

        // Round 0 -> worker 0
        let leader_idx = 0 % worker_ids.len();
        assert_eq!(worker_ids[leader_idx].0, "test-peer-1");

        // Round 1 -> worker 1
        let leader_idx = 1 % worker_ids.len();
        assert_eq!(worker_ids[leader_idx].0, "test-peer-2");

        // Round 2 -> worker 2
        let leader_idx = 2 % worker_ids.len();
        assert_eq!(worker_ids[leader_idx].0, "test-peer-3");

        // Round 3 -> back to worker 0
        let leader_idx = 3 % worker_ids.len();
        assert_eq!(worker_ids[leader_idx].0, "test-peer-1");
    }

    #[test]
    fn test_distributed_coordinator_error_display() {
        let err = DistributedCoordinatorError::InsufficientWorkers {
            required: 3,
            available: 1,
        };
        assert_eq!(
            format!("{}", err),
            "Insufficient workers: need 3, have 1"
        );

        let err = DistributedCoordinatorError::NotLeader;
        assert_eq!(format!("{}", err), "Not the leader");
    }
}
