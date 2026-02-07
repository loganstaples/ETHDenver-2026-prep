//! Training Session Manager.
//!
//! Provides a unified high-level API for multi-worker distributed training sessions.
//! This module integrates all distributed training components including:
//!
//! - Session creation and lifecycle management
//! - Worker registration with staking
//! - Round orchestration through the state machine
//! - Gradient share collection from workers
//! - Proof aggregation and verification
//! - Smart contract round commits
//! - Timeout handling and failure recovery
//! - Heartbeat-based liveness detection
//! - Session state persistence for resume
//!
//! # Architecture
//!
//! The `TrainingSessionManager` acts as the central coordinator for distributed training:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  TrainingSessionManager                          │
//! │  ┌─────────────┐  ┌───────────────┐  ┌────────────────────┐    │
//! │  │   Session   │  │   Workers     │  │   State Machine    │    │
//! │  │   Config    │  │   Registry    │  │   (Round FSM)      │    │
//! │  └─────────────┘  └───────────────┘  └────────────────────┘    │
//! │  ┌─────────────┐  ┌───────────────┐  ┌────────────────────┐    │
//! │  │  Gradient   │  │    Proof      │  │     SC Client      │    │
//! │  │  Collector  │  │   Aggregator  │  │   (On-chain)       │    │
//! │  └─────────────┘  └───────────────┘  └────────────────────┘    │
//! │  ┌─────────────┐  ┌───────────────┐  ┌────────────────────┐    │
//! │  │   Fault     │  │  Checkpoint   │  │   Event            │    │
//! │  │  Tolerance  │  │  Coordinator  │  │   Broadcasting     │    │
//! │  └─────────────┘  └───────────────┘  └────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ethers::types::{H256, U256};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use tokio::sync::{broadcast, mpsc, oneshot};

use crate::network::messages::PeerId;
use crate::sc_client::{SCClient, TrainingProofInputs, ModelState};
use crate::round_commit::{
    RoundCommitManager, RoundCommitConfig, RoundCommitId, RoundCommitResult,
    WorkerProof, ProofAggregator, AggregatedRoundProof, RoundCommitError,
};

use super::coordinator::NodeId;
use super::distributed_checkpoint::{
    DistributedCheckpointConfig, DistributedCheckpointCoordinator, DistributedCheckpointId,
    WorkerCheckpointContribution, ResumableTrainingState,
};
use super::distributed_coordinator::{
    DistributedTrainingConfig, DistributedTrainingCoordinator, DistributedTrainingEvent,
    DistributedTrainingState, GradientShare, WorkerInfo,
};
use super::fault_tolerance::{
    FaultToleranceConfig, FaultToleranceManager, WorkerHealth,
};
use super::model::ModelWeights;
use super::state_machine::{
    DistributedRoundId, DistributedRoundState, DistributedTrainingStateMachine,
    StateMachineConfig, WorkerRoundState,
};
use super::synchronization::{BarrierConfig, SyncCoordinator, SyncPhase, BarrierResult};

// ============================================================================
// Session Configuration
// ============================================================================

/// Configuration for a training session.
#[derive(Debug, Clone)]
pub struct TrainingSessionConfig {
    /// Unique session identifier.
    pub session_id: u64,
    /// Model ID on the smart contract.
    pub model_id: u64,
    /// Model owner's address.
    pub model_owner: String,
    /// Minimum workers required to start training.
    pub min_workers: usize,
    /// Maximum workers allowed.
    pub max_workers: usize,
    /// Minimum stake required per worker (in wei).
    pub min_stake: u64,
    /// Total training rounds to complete.
    pub total_rounds: u64,
    /// Timeout for worker registration.
    pub registration_timeout: Duration,
    /// Timeout for each training round.
    pub round_timeout: Duration,
    /// Timeout for gradient collection.
    pub collection_timeout: Duration,
    /// Checkpoint interval (rounds between checkpoints).
    pub checkpoint_interval: u64,
    /// Whether to submit proofs on-chain.
    pub submit_proofs_onchain: bool,
    /// RPC endpoint for smart contract interaction.
    pub rpc_endpoint: Option<String>,
    /// Coordinator contract address.
    pub contract_address: Option<String>,
    /// Enable automatic failure recovery.
    pub auto_recovery: bool,
    /// Heartbeat interval for liveness detection.
    pub heartbeat_interval: Duration,
    /// Heartbeat timeout before marking worker as failed.
    pub heartbeat_timeout: Duration,
}

impl Default for TrainingSessionConfig {
    fn default() -> Self {
        Self {
            session_id: Self::generate_session_id(),
            model_id: 0,
            model_owner: String::new(),
            min_workers: 3,
            max_workers: 100,
            min_stake: 1_000_000_000_000_000_000, // 1 ETH
            total_rounds: 100,
            registration_timeout: Duration::from_secs(300),
            round_timeout: Duration::from_secs(600),
            collection_timeout: Duration::from_secs(120),
            checkpoint_interval: 10,
            submit_proofs_onchain: false,
            rpc_endpoint: None,
            contract_address: None,
            auto_recovery: true,
            heartbeat_interval: Duration::from_secs(30),
            heartbeat_timeout: Duration::from_secs(90),
        }
    }
}

impl TrainingSessionConfig {
    /// Generates a unique session ID based on current timestamp.
    fn generate_session_id() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Creates a config for local testing (no on-chain interaction).
    pub fn for_local_testing(min_workers: usize) -> Self {
        Self {
            min_workers,
            max_workers: min_workers * 2,
            min_stake: 0,
            total_rounds: 10,
            registration_timeout: Duration::from_secs(30),
            round_timeout: Duration::from_secs(60),
            collection_timeout: Duration::from_secs(30),
            checkpoint_interval: 5,
            submit_proofs_onchain: false,
            auto_recovery: false,
            heartbeat_interval: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(15),
            ..Default::default()
        }
    }
}

// ============================================================================
// Session State
// ============================================================================

/// High-level state of a training session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session created but not started.
    Created,
    /// Waiting for workers to register.
    WaitingForWorkers,
    /// Training is in progress.
    Training,
    /// Session is paused (e.g., for recovery).
    Paused,
    /// Recovering from a failure.
    Recovering,
    /// Training completed successfully.
    Completed,
    /// Session failed.
    Failed,
    /// Session was cancelled.
    Cancelled,
}

/// Summary of a completed training session.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Session ID.
    pub session_id: u64,
    /// Total rounds completed.
    pub rounds_completed: u64,
    /// Final state.
    pub final_state: SessionState,
    /// Workers that participated.
    pub workers: Vec<PeerId>,
    /// Total duration.
    pub duration: Duration,
    /// Initial model commitment.
    pub initial_commitment: [u8; 32],
    /// Final model commitment.
    pub final_commitment: [u8; 32],
    /// Total error bound accumulated.
    pub total_error_bound: f64,
    /// Transaction hashes for on-chain commits.
    pub tx_hashes: Vec<H256>,
}

// ============================================================================
// Session Events
// ============================================================================

/// Events emitted by the training session manager.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Session created.
    SessionCreated { session_id: u64 },
    /// Worker registered.
    WorkerRegistered {
        session_id: u64,
        worker_id: PeerId,
        stake: u64,
        total_workers: usize,
    },
    /// Worker left or was removed.
    WorkerLeft {
        session_id: u64,
        worker_id: PeerId,
        reason: String,
    },
    /// Training started.
    TrainingStarted {
        session_id: u64,
        num_workers: usize,
    },
    /// Round started.
    RoundStarted {
        session_id: u64,
        round_number: u64,
        leader: PeerId,
    },
    /// Gradient share received.
    GradientReceived {
        session_id: u64,
        round_number: u64,
        worker_id: PeerId,
        received: usize,
        expected: usize,
    },
    /// Aggregation completed.
    AggregationCompleted {
        session_id: u64,
        round_number: u64,
        num_contributors: usize,
        error_bound: f64,
    },
    /// Round committed on-chain.
    RoundCommitted {
        session_id: u64,
        round_number: u64,
        tx_hash: Option<H256>,
    },
    /// Round completed.
    RoundCompleted {
        session_id: u64,
        round_number: u64,
        duration: Duration,
    },
    /// Round failed.
    RoundFailed {
        session_id: u64,
        round_number: u64,
        reason: String,
    },
    /// Checkpoint created.
    CheckpointCreated {
        session_id: u64,
        checkpoint_id: DistributedCheckpointId,
    },
    /// Worker heartbeat received.
    HeartbeatReceived {
        session_id: u64,
        worker_id: PeerId,
    },
    /// Worker timed out.
    WorkerTimeout {
        session_id: u64,
        worker_id: PeerId,
    },
    /// Session paused.
    SessionPaused {
        session_id: u64,
        reason: String,
    },
    /// Session resumed.
    SessionResumed {
        session_id: u64,
        from_round: u64,
    },
    /// Session completed.
    SessionCompleted {
        session_id: u64,
        summary: SessionSummary,
    },
    /// Session failed.
    SessionFailed {
        session_id: u64,
        reason: String,
    },
}

// ============================================================================
// Registered Worker
// ============================================================================

/// Information about a registered worker in a session.
#[derive(Debug, Clone)]
pub struct RegisteredWorker {
    /// Worker's peer ID.
    pub peer_id: PeerId,
    /// Worker's stake amount.
    pub stake: u64,
    /// Worker's Ethereum address (for payment/slashing).
    pub eth_address: String,
    /// MPC share index assigned to this worker.
    pub share_index: usize,
    /// Worker's current health status.
    pub health: WorkerHealth,
    /// Last heartbeat timestamp.
    pub last_heartbeat: Instant,
    /// Current round state.
    pub round_state: Option<WorkerRoundState>,
    /// Rounds completed by this worker.
    pub rounds_completed: u64,
    /// Gradients submitted by this worker.
    pub gradients_submitted: u64,
    /// Whether this worker has been slashed.
    pub is_slashed: bool,
    /// Registration timestamp.
    pub registered_at: Instant,
}

impl RegisteredWorker {
    /// Creates a new registered worker.
    pub fn new(peer_id: PeerId, stake: u64, eth_address: String, share_index: usize) -> Self {
        let now = Instant::now();
        Self {
            peer_id,
            stake,
            eth_address,
            share_index,
            health: WorkerHealth::Healthy,
            last_heartbeat: now,
            round_state: None,
            rounds_completed: 0,
            gradients_submitted: 0,
            is_slashed: false,
            registered_at: now,
        }
    }

    /// Updates the heartbeat timestamp.
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    /// Checks if the worker is healthy based on heartbeat timeout.
    pub fn is_healthy(&self, timeout: Duration) -> bool {
        self.last_heartbeat.elapsed() < timeout && matches!(self.health, WorkerHealth::Healthy)
    }
}

// ============================================================================
// Training Session Manager
// ============================================================================

/// Main coordinator for multi-worker distributed training sessions.
///
/// The `TrainingSessionManager` provides a high-level API for:
/// - Creating and managing training sessions
/// - Registering workers with stake verification
/// - Orchestrating training rounds through the state machine
/// - Collecting gradient shares from workers
/// - Aggregating proofs and submitting to smart contracts
/// - Handling worker failures and timeouts
/// - Checkpointing and resuming sessions
pub struct TrainingSessionManager {
    /// Session configuration.
    config: TrainingSessionConfig,
    /// Current session state.
    state: SessionState,
    /// Registered workers.
    workers: HashMap<PeerId, RegisteredWorker>,
    /// Next share index to assign.
    next_share_index: usize,
    /// Distributed training state machine.
    state_machine: DistributedTrainingStateMachine,
    /// Synchronization coordinator.
    sync_coordinator: Arc<SyncCoordinator>,
    /// Fault tolerance manager.
    fault_manager: FaultToleranceManager,
    /// Checkpoint coordinator.
    checkpoint_coordinator: Option<Arc<DistributedCheckpointCoordinator>>,
    /// Round commit manager.
    commit_manager: RoundCommitManager,
    /// Smart contract client.
    sc_client: Option<Arc<SCClient>>,
    /// Current round number.
    current_round: u64,
    /// Current model commitment.
    model_commitment: [u8; 32],
    /// Initial model commitment.
    initial_commitment: [u8; 32],
    /// Model weights (held by session manager during training).
    model_weights: Option<ModelWeights>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<SessionEvent>,
    /// Pending events to emit.
    pending_events: Vec<SessionEvent>,
    /// Transaction hashes for completed rounds.
    tx_hashes: Vec<H256>,
    /// Accumulated error bound.
    total_error_bound: f64,
    /// Session started timestamp.
    started_at: Option<Instant>,
}

impl TrainingSessionManager {
    /// Creates a new training session manager.
    pub fn new(config: TrainingSessionConfig) -> Self {
        let (event_tx, _) = broadcast::channel(256);

        // Create state machine config from session config
        let state_machine_config = StateMachineConfig {
            min_workers: config.min_workers,
            max_workers: config.max_workers,
            min_submission_fraction: 0.67,
            worker_inactivity_timeout: config.heartbeat_timeout,
            allow_degradation: config.auto_recovery,
            min_degraded_workers: (config.min_workers + 1) / 2,
            ..Default::default()
        };

        let state_machine = DistributedTrainingStateMachine::new(state_machine_config);

        // Create fault tolerance manager
        let fault_config = FaultToleranceConfig {
            heartbeat_timeout: config.heartbeat_timeout,
            max_missed_heartbeats: 3,
            ..Default::default()
        };
        let fault_manager = FaultToleranceManager::new(fault_config);

        // Create sync coordinator
        let sync_config = BarrierConfig::default();
        let sync_coordinator = Arc::new(SyncCoordinator::new(sync_config));

        // Create round commit manager
        let commit_config = RoundCommitConfig {
            model_id: config.model_id,
            min_proofs: config.min_workers,
            collection_timeout: config.collection_timeout,
            ..Default::default()
        };
        let commit_manager = RoundCommitManager::new(commit_config);

        Self {
            config,
            state: SessionState::Created,
            workers: HashMap::new(),
            next_share_index: 0,
            state_machine,
            sync_coordinator,
            fault_manager,
            checkpoint_coordinator: None,
            commit_manager,
            sc_client: None,
            current_round: 0,
            model_commitment: [0u8; 32],
            initial_commitment: [0u8; 32],
            model_weights: None,
            event_tx,
            pending_events: Vec::new(),
            tx_hashes: Vec::new(),
            total_error_bound: 0.0,
            started_at: None,
        }
    }

    /// Creates a session manager with smart contract client.
    pub async fn with_sc_client(
        config: TrainingSessionConfig,
        sc_client: Arc<SCClient>,
    ) -> Result<Self, SessionError> {
        let mut manager = Self::new(config);
        manager.sc_client = Some(sc_client.clone());
        manager.commit_manager.set_client(sc_client);
        Ok(manager)
    }

    /// Subscribes to session events.
    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.event_tx.subscribe()
    }

    /// Returns the current session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Returns the session ID.
    pub fn session_id(&self) -> u64 {
        self.config.session_id
    }

    /// Returns the current round number.
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Returns the current model commitment.
    pub fn model_commitment(&self) -> [u8; 32] {
        self.model_commitment
    }

    /// Returns registered workers.
    pub fn workers(&self) -> &HashMap<PeerId, RegisteredWorker> {
        &self.workers
    }

    /// Returns the number of registered workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Returns healthy workers.
    pub fn healthy_workers(&self) -> Vec<PeerId> {
        self.workers
            .iter()
            .filter(|(_, w)| w.is_healthy(self.config.heartbeat_timeout))
            .map(|(id, _)| id.clone())
            .collect()
    }

    // ========================================================================
    // Session Lifecycle
    // ========================================================================

    /// Initializes the session with model weights.
    pub fn initialize(&mut self, model: ModelWeights) -> Result<(), SessionError> {
        if self.state != SessionState::Created {
            return Err(SessionError::InvalidState {
                expected: SessionState::Created,
                actual: self.state,
            });
        }

        // Compute initial model commitment
        self.model_commitment = compute_model_commitment(&model);
        self.initial_commitment = self.model_commitment;
        self.model_weights = Some(model);

        // Setup checkpoint coordinator if directory configured
        // (Would be configured via TrainingSessionConfig in a full implementation)

        self.state = SessionState::WaitingForWorkers;

        self.pending_events.push(SessionEvent::SessionCreated {
            session_id: self.config.session_id,
        });

        Ok(())
    }

    /// Registers a worker for the training session.
    pub fn register_worker(
        &mut self,
        peer_id: PeerId,
        stake: u64,
        eth_address: String,
    ) -> Result<usize, SessionError> {
        if self.state != SessionState::WaitingForWorkers && self.state != SessionState::Created {
            return Err(SessionError::NotAcceptingWorkers);
        }

        if self.workers.len() >= self.config.max_workers {
            return Err(SessionError::MaxWorkersReached);
        }

        if self.workers.contains_key(&peer_id) {
            return Err(SessionError::WorkerAlreadyRegistered(peer_id));
        }

        if stake < self.config.min_stake {
            return Err(SessionError::InsufficientStake {
                required: self.config.min_stake,
                provided: stake,
            });
        }

        // Assign share index
        let share_index = self.next_share_index;
        self.next_share_index += 1;

        // Register worker
        let worker = RegisteredWorker::new(peer_id.clone(), stake, eth_address, share_index);
        self.workers.insert(peer_id.clone(), worker);

        // Register with fault manager
        self.fault_manager.register_worker(peer_id.clone());

        self.pending_events.push(SessionEvent::WorkerRegistered {
            session_id: self.config.session_id,
            worker_id: peer_id,
            stake,
            total_workers: self.workers.len(),
        });

        Ok(share_index)
    }

    /// Removes a worker from the session.
    pub fn remove_worker(&mut self, peer_id: &PeerId, reason: &str) -> Result<(), SessionError> {
        if !self.workers.contains_key(peer_id) {
            return Err(SessionError::WorkerNotFound(peer_id.clone()));
        }

        self.workers.remove(peer_id);
        self.fault_manager.unregister_worker(peer_id);

        self.pending_events.push(SessionEvent::WorkerLeft {
            session_id: self.config.session_id,
            worker_id: peer_id.clone(),
            reason: reason.to_string(),
        });

        // Check if we need to pause due to insufficient workers
        if self.state == SessionState::Training && self.workers.len() < self.config.min_workers {
            self.state = SessionState::Paused;
            self.pending_events.push(SessionEvent::SessionPaused {
                session_id: self.config.session_id,
                reason: "Insufficient workers".to_string(),
            });
        }

        Ok(())
    }

    /// Starts the training session.
    pub fn start_training(&mut self) -> Result<(), SessionError> {
        if self.state != SessionState::WaitingForWorkers {
            return Err(SessionError::InvalidState {
                expected: SessionState::WaitingForWorkers,
                actual: self.state,
            });
        }

        if self.workers.len() < self.config.min_workers {
            return Err(SessionError::InsufficientWorkers {
                required: self.config.min_workers,
                available: self.workers.len(),
            });
        }

        self.state = SessionState::Training;
        self.started_at = Some(Instant::now());

        self.pending_events.push(SessionEvent::TrainingStarted {
            session_id: self.config.session_id,
            num_workers: self.workers.len(),
        });

        Ok(())
    }

    // ========================================================================
    // Round Management
    // ========================================================================

    /// Starts a new training round.
    pub fn start_round(&mut self) -> Result<DistributedRoundId, SessionError> {
        if self.state != SessionState::Training {
            return Err(SessionError::InvalidState {
                expected: SessionState::Training,
                actual: self.state,
            });
        }

        if self.current_round >= self.config.total_rounds {
            return Err(SessionError::TrainingComplete);
        }

        self.current_round += 1;

        // Start round in state machine
        let round_id = self.state_machine
            .start_round(self.model_commitment, None)
            .map_err(|e| SessionError::StateMachineError(e))?;

        // Add workers to the round
        let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
        for (i, worker_id) in worker_ids.iter().enumerate() {
            let stake = self.workers.get(worker_id).map(|w| w.stake).unwrap_or(1000);
            self.state_machine.add_worker(worker_id.clone(), stake)
                .map_err(|e| SessionError::StateMachineError(e))?;
        }

        // Initialize gradient collection
        let round_commit_id = RoundCommitId::from_distributed_round(round_id);
        self.commit_manager
            .start_collection(round_id, worker_ids.clone())
            .map_err(|e| SessionError::CommitError(e))?;

        // Determine leader (round-robin)
        let leader_idx = (self.current_round as usize - 1) % worker_ids.len();
        let leader = worker_ids[leader_idx].clone();

        // Try to start distribution
        self.state_machine.try_start_distribution()
            .map_err(|e| SessionError::StateMachineError(e))?;

        // Mark distribution complete to transition to Computing state
        // (In production, this would happen after shares are actually distributed)
        self.state_machine.mark_distribution_complete()
            .map_err(|e| SessionError::StateMachineError(e))?;

        self.pending_events.push(SessionEvent::RoundStarted {
            session_id: self.config.session_id,
            round_number: self.current_round,
            leader,
        });

        Ok(round_id)
    }

    /// Submits a gradient share from a worker.
    pub fn submit_gradient_share(&mut self, share: GradientShare) -> Result<(), SessionError> {
        let worker_id = share.worker_id.clone();

        // Verify worker is registered
        if !self.workers.contains_key(&worker_id) {
            return Err(SessionError::WorkerNotFound(worker_id));
        }

        // Update worker state
        if let Some(worker) = self.workers.get_mut(&worker_id) {
            worker.gradients_submitted += 1;
            worker.heartbeat();
        }

        // Record in fault manager
        self.fault_manager.record_heartbeat(&worker_id, 10.0);

        // Record submission in state machine
        self.state_machine.record_submission(&worker_id, share.commitment, share.error_bound)
            .map_err(|e| SessionError::StateMachineError(e))?;

        // Get counts for event
        let round = self.state_machine.current_round()
            .ok_or(SessionError::NoActiveRound)?;
        let received = round.submitted_count();
        let expected = round.active_worker_count();

        self.pending_events.push(SessionEvent::GradientReceived {
            session_id: self.config.session_id,
            round_number: self.current_round,
            worker_id,
            received,
            expected,
        });

        Ok(())
    }

    /// Checks if enough gradients have been collected.
    pub fn has_enough_gradients(&self) -> bool {
        self.state_machine.current_round()
            .map(|r| r.has_min_submissions())
            .unwrap_or(false)
    }

    /// Completes the current round with aggregation and optional on-chain commit.
    pub async fn complete_round(
        &mut self,
        old_commitment: [u8; 32],
        new_commitment: [u8; 32],
        aggregated_error: f64,
    ) -> Result<RoundCommitResult, SessionError> {
        // Transition state machine through collection and aggregation
        self.state_machine.try_start_collection()
            .map_err(|e| SessionError::StateMachineError(e))?;

        self.state_machine.try_start_aggregation()
            .map_err(|e| SessionError::StateMachineError(e))?;

        let round_id = self.state_machine.current_round()
            .map(|r| r.id)
            .ok_or(SessionError::NoActiveRound)?;

        // Record aggregation
        self.state_machine.record_aggregation(
            new_commitment,
            new_commitment,
            self.workers.len(),
            0,
        ).map_err(|e| SessionError::StateMachineError(e))?;

        // Update model commitment
        self.model_commitment = new_commitment;
        self.total_error_bound += aggregated_error;

        // Create commit result (mock if no SC client)
        let commit_result = if self.config.submit_proofs_onchain && self.sc_client.is_some() {
            // Real on-chain submission
            let commit_id = RoundCommitId::from_distributed_round(round_id);
            self.commit_manager.finalize_collection(commit_id, old_commitment, new_commitment)
                .map_err(|e| SessionError::CommitError(e))?;

            self.commit_manager.submit_to_chain(commit_id).await
                .map_err(|e| SessionError::CommitError(e))?
        } else {
            // Mock commit for local testing
            RoundCommitResult {
                commit_id: RoundCommitId::from_distributed_round(round_id),
                tx_hash: H256::random(),
                block_number: self.current_round,
                gas_used: U256::from(100_000),
                new_commitment: U256::from_big_endian(&new_commitment),
            }
        };

        self.tx_hashes.push(commit_result.tx_hash);

        // Record commit in state machine
        self.state_machine.record_commit(commit_result.tx_hash.into())
            .map_err(|e| SessionError::StateMachineError(e))?;

        // Update worker stats
        for worker in self.workers.values_mut() {
            worker.rounds_completed += 1;
        }

        let duration = self.started_at.map(|s| s.elapsed()).unwrap_or_default();

        self.pending_events.push(SessionEvent::AggregationCompleted {
            session_id: self.config.session_id,
            round_number: self.current_round,
            num_contributors: self.workers.len(),
            error_bound: aggregated_error,
        });

        self.pending_events.push(SessionEvent::RoundCommitted {
            session_id: self.config.session_id,
            round_number: self.current_round,
            tx_hash: Some(commit_result.tx_hash),
        });

        self.pending_events.push(SessionEvent::RoundCompleted {
            session_id: self.config.session_id,
            round_number: self.current_round,
            duration,
        });

        // Check if we should checkpoint
        if self.current_round % self.config.checkpoint_interval == 0 {
            // Would create checkpoint here
        }

        // Check if training is complete
        if self.current_round >= self.config.total_rounds {
            self.complete_session()?;
        }

        Ok(commit_result)
    }

    /// Completes the training session.
    fn complete_session(&mut self) -> Result<SessionSummary, SessionError> {
        self.state = SessionState::Completed;

        let summary = SessionSummary {
            session_id: self.config.session_id,
            rounds_completed: self.current_round,
            final_state: self.state,
            workers: self.workers.keys().cloned().collect(),
            duration: self.started_at.map(|s| s.elapsed()).unwrap_or_default(),
            initial_commitment: self.initial_commitment,
            final_commitment: self.model_commitment,
            total_error_bound: self.total_error_bound,
            tx_hashes: self.tx_hashes.clone(),
        };

        self.pending_events.push(SessionEvent::SessionCompleted {
            session_id: self.config.session_id,
            summary: summary.clone(),
        });

        Ok(summary)
    }

    // ========================================================================
    // Heartbeat and Liveness
    // ========================================================================

    /// Processes a heartbeat from a worker.
    pub fn process_heartbeat(&mut self, worker_id: &PeerId) {
        if let Some(worker) = self.workers.get_mut(worker_id) {
            worker.heartbeat();
        }
        self.fault_manager.record_heartbeat(worker_id, 10.0);

        self.pending_events.push(SessionEvent::HeartbeatReceived {
            session_id: self.config.session_id,
            worker_id: worker_id.clone(),
        });
    }

    /// Checks for worker timeouts and handles them.
    pub fn check_worker_timeouts(&mut self) -> Vec<PeerId> {
        let timeout = self.config.heartbeat_timeout;
        let mut timed_out = Vec::new();

        for (peer_id, worker) in &mut self.workers {
            if !worker.is_healthy(timeout) {
                worker.health = WorkerHealth::Failed;
                timed_out.push(peer_id.clone());
            }
        }

        for peer_id in &timed_out {
            self.pending_events.push(SessionEvent::WorkerTimeout {
                session_id: self.config.session_id,
                worker_id: peer_id.clone(),
            });
        }

        // Handle auto-recovery if enabled
        if self.config.auto_recovery && !timed_out.is_empty() {
            for peer_id in &timed_out {
                let _ = self.remove_worker(peer_id, "Heartbeat timeout");
            }
        }

        timed_out
    }

    // ========================================================================
    // Event Handling
    // ========================================================================

    /// Drains and broadcasts pending events.
    pub fn drain_events(&mut self) -> Vec<SessionEvent> {
        let events = std::mem::take(&mut self.pending_events);
        for event in &events {
            let _ = self.event_tx.send(event.clone());
        }
        events
    }

    // ========================================================================
    // Checkpointing
    // ========================================================================

    /// Creates a checkpoint of the current session state.
    pub async fn create_checkpoint(&self) -> Result<DistributedCheckpointId, SessionError> {
        let coordinator = self.checkpoint_coordinator.as_ref()
            .ok_or(SessionError::NoCheckpointCoordinator)?;

        let worker_ids: Vec<PeerId> = self.workers.keys().cloned().collect();
        let checkpoint_id = coordinator
            .initiate_checkpoint(self.current_round, worker_ids)
            .await
            .map_err(|e| SessionError::CheckpointError(e.to_string()))?;

        Ok(checkpoint_id)
    }

    /// Resumes a session from a checkpoint.
    pub async fn resume_from_checkpoint(
        &mut self,
        checkpoint_id: DistributedCheckpointId,
    ) -> Result<(), SessionError> {
        // Would restore state from checkpoint here
        self.state = SessionState::Training;

        self.pending_events.push(SessionEvent::SessionResumed {
            session_id: self.config.session_id,
            from_round: self.current_round,
        });

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Computes a commitment hash for model weights.
fn compute_model_commitment(model: &ModelWeights) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Hash model metadata
    hasher.update(model.metadata.name.as_bytes());
    hasher.update(&model.metadata.hidden_dim.to_le_bytes());
    hasher.update(&model.metadata.num_layers.to_le_bytes());

    // Hash layer weights
    for layer in &model.layers {
        for (_, weight) in &layer.weights {
            // Hash a subset of weights for efficiency
            let sample_size = weight.data.len().min(1000);
            for val in weight.data.iter().take(sample_size) {
                hasher.update(&val.to_le_bytes());
            }
        }
    }

    let result = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

// ============================================================================
// Errors
// ============================================================================

/// Errors from the training session manager.
#[derive(Debug)]
pub enum SessionError {
    /// Invalid session state for operation.
    InvalidState {
        expected: SessionState,
        actual: SessionState,
    },
    /// Not accepting worker registrations.
    NotAcceptingWorkers,
    /// Maximum workers reached.
    MaxWorkersReached,
    /// Worker already registered.
    WorkerAlreadyRegistered(PeerId),
    /// Worker not found.
    WorkerNotFound(PeerId),
    /// Insufficient stake.
    InsufficientStake {
        required: u64,
        provided: u64,
    },
    /// Insufficient workers.
    InsufficientWorkers {
        required: usize,
        available: usize,
    },
    /// No active round.
    NoActiveRound,
    /// Training already complete.
    TrainingComplete,
    /// State machine error.
    StateMachineError(String),
    /// Commit error.
    CommitError(RoundCommitError),
    /// Checkpoint error.
    CheckpointError(String),
    /// No checkpoint coordinator configured.
    NoCheckpointCoordinator,
    /// Smart contract error.
    ContractError(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidState { expected, actual } => {
                write!(f, "Invalid state: expected {:?}, actual {:?}", expected, actual)
            }
            Self::NotAcceptingWorkers => write!(f, "Not accepting worker registrations"),
            Self::MaxWorkersReached => write!(f, "Maximum workers reached"),
            Self::WorkerAlreadyRegistered(id) => write!(f, "Worker already registered: {}", id),
            Self::WorkerNotFound(id) => write!(f, "Worker not found: {}", id),
            Self::InsufficientStake { required, provided } => {
                write!(f, "Insufficient stake: required {}, provided {}", required, provided)
            }
            Self::InsufficientWorkers { required, available } => {
                write!(f, "Insufficient workers: required {}, available {}", required, available)
            }
            Self::NoActiveRound => write!(f, "No active round"),
            Self::TrainingComplete => write!(f, "Training already complete"),
            Self::StateMachineError(e) => write!(f, "State machine error: {}", e),
            Self::CommitError(e) => write!(f, "Commit error: {}", e),
            Self::CheckpointError(e) => write!(f, "Checkpoint error: {}", e),
            Self::NoCheckpointCoordinator => write!(f, "No checkpoint coordinator configured"),
            Self::ContractError(e) => write!(f, "Smart contract error: {}", e),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<RoundCommitError> for SessionError {
    fn from(e: RoundCommitError) -> Self {
        Self::CommitError(e)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_peer_id(id: u8) -> PeerId {
        PeerId::from_string(&format!("test-peer-{}", id))
    }

    fn create_test_model() -> ModelWeights {
        use super::super::model::{ModelMetadata, LayerWeights, WeightData};

        ModelWeights {
            metadata: ModelMetadata {
                name: "test-model".to_string(),
                hidden_dim: 64,
                num_layers: 2,
                num_heads: 2,
                vocab_size: 100,
                ..Default::default()
            },
            layers: vec![
                LayerWeights {
                    layer_idx: 0,
                    weights: vec![
                        ("w1".to_string(), WeightData {
                            shape: vec![64, 64],
                            data: vec![0.1; 64 * 64],
                            error_bound: 0.0,
                        }),
                    ].into_iter().collect(),
                },
            ],
            embeddings: None,
            lm_head: None,
            extra_weights: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_session_creation() {
        let config = TrainingSessionConfig::for_local_testing(3);
        let manager = TrainingSessionManager::new(config);

        assert_eq!(manager.state(), SessionState::Created);
        assert_eq!(manager.worker_count(), 0);
    }

    #[test]
    fn test_session_initialization() {
        let config = TrainingSessionConfig::for_local_testing(3);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        manager.initialize(model).unwrap();

        assert_eq!(manager.state(), SessionState::WaitingForWorkers);
        assert_ne!(manager.model_commitment, [0u8; 32]);
    }

    #[test]
    fn test_worker_registration() {
        let config = TrainingSessionConfig::for_local_testing(2);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        manager.initialize(model).unwrap();

        // Register workers
        let share0 = manager.register_worker(
            create_test_peer_id(1),
            1000,
            "0x1234".to_string(),
        ).unwrap();
        assert_eq!(share0, 0);

        let share1 = manager.register_worker(
            create_test_peer_id(2),
            1000,
            "0x5678".to_string(),
        ).unwrap();
        assert_eq!(share1, 1);

        assert_eq!(manager.worker_count(), 2);
    }

    #[test]
    fn test_duplicate_worker_registration() {
        let config = TrainingSessionConfig::for_local_testing(2);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        manager.initialize(model).unwrap();

        manager.register_worker(
            create_test_peer_id(1),
            1000,
            "0x1234".to_string(),
        ).unwrap();

        // Duplicate registration should fail
        let result = manager.register_worker(
            create_test_peer_id(1),
            1000,
            "0x1234".to_string(),
        );
        assert!(matches!(result, Err(SessionError::WorkerAlreadyRegistered(_))));
    }

    #[test]
    fn test_start_training() {
        let config = TrainingSessionConfig::for_local_testing(2);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        manager.initialize(model).unwrap();

        // Register enough workers
        manager.register_worker(create_test_peer_id(1), 1000, "0x1".to_string()).unwrap();
        manager.register_worker(create_test_peer_id(2), 1000, "0x2".to_string()).unwrap();

        // Start training
        manager.start_training().unwrap();

        assert_eq!(manager.state(), SessionState::Training);
    }

    #[test]
    fn test_start_training_insufficient_workers() {
        let config = TrainingSessionConfig::for_local_testing(3);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        manager.initialize(model).unwrap();

        // Only register 2 workers when 3 are required
        manager.register_worker(create_test_peer_id(1), 1000, "0x1".to_string()).unwrap();
        manager.register_worker(create_test_peer_id(2), 1000, "0x2".to_string()).unwrap();

        // Should fail
        let result = manager.start_training();
        assert!(matches!(result, Err(SessionError::InsufficientWorkers { .. })));
    }

    #[test]
    fn test_start_round() {
        let config = TrainingSessionConfig::for_local_testing(2);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        manager.initialize(model).unwrap();
        manager.register_worker(create_test_peer_id(1), 1000, "0x1".to_string()).unwrap();
        manager.register_worker(create_test_peer_id(2), 1000, "0x2".to_string()).unwrap();
        manager.start_training().unwrap();

        // Start round
        let round_id = manager.start_round().unwrap();

        assert_eq!(manager.current_round(), 1);
        assert_eq!(round_id.round_number, 1);
    }

    #[test]
    fn test_heartbeat_processing() {
        let config = TrainingSessionConfig::for_local_testing(2);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        manager.initialize(model).unwrap();
        let peer_id = create_test_peer_id(1);
        manager.register_worker(peer_id.clone(), 1000, "0x1".to_string()).unwrap();

        // Process heartbeat
        manager.process_heartbeat(&peer_id);

        let worker = manager.workers().get(&peer_id).unwrap();
        assert!(worker.is_healthy(Duration::from_secs(10)));
    }

    #[test]
    fn test_event_emission() {
        let config = TrainingSessionConfig::for_local_testing(2);
        let mut manager = TrainingSessionManager::new(config);
        let model = create_test_model();

        let mut receiver = manager.subscribe();

        manager.initialize(model).unwrap();

        // Drain events
        let events = manager.drain_events();

        assert!(!events.is_empty());
        assert!(matches!(events[0], SessionEvent::SessionCreated { .. }));
    }
}
