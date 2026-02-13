//! Round Manager — Full State Machine for Aggregator Training Rounds.
//!
//! Implements the complete round lifecycle:
//!
//! ```text
//! Idle → Configuring → WaitingForWorkers → Training → CollectingProofs
//!      → Aggregating → SubmittingOnChain → Complete
//! ```
//!
//! The aggregator drives all state transitions. Workers register explicitly,
//! receive round configuration, signal readiness, perform training, and submit
//! proofs. The aggregator validates proofs, aggregates results, submits on-chain,
//! and broadcasts completion with the final commitment hash.

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::network::messages::{
    ModelDims, NodeCapabilities, PeerId, RegistrationMessage, RoundManagementMessage,
};
use crate::sc_client::TrainingProofInputs;

// ============================================================================
// Worker Registry
// ============================================================================

/// Information about a registered worker.
#[derive(Debug, Clone)]
pub struct RegisteredWorker {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Worker capabilities.
    pub capabilities: NodeCapabilities,
    /// Listen address.
    pub listen_addr: String,
    /// Public key bytes (if provided).
    pub public_key: Option<Vec<u8>>,
    /// Maximum concurrent steps.
    pub max_concurrent_steps: u32,
    /// Available memory in MB.
    pub available_memory_mb: u64,
    /// Assigned slot in the registry.
    pub slot: u32,
    /// When the worker registered.
    pub registered_at: Instant,
    /// Last heartbeat time.
    pub last_heartbeat: Instant,
    /// Whether the worker is currently participating in a round.
    pub in_round: bool,
    /// Total rounds completed.
    pub rounds_completed: u64,
    /// Total rounds failed.
    pub rounds_failed: u64,
}

/// Worker registry tracks all registered workers.
#[derive(Debug)]
pub struct WorkerRegistry {
    /// Registered workers by peer ID.
    workers: HashMap<PeerId, RegisteredWorker>,
    /// Next slot number to assign.
    next_slot: u32,
    /// Maximum allowed workers.
    max_workers: u32,
    /// Worker timeout (no heartbeat for this long → considered dead).
    worker_timeout: Duration,
}

impl WorkerRegistry {
    /// Creates a new worker registry.
    pub fn new(max_workers: u32, worker_timeout: Duration) -> Self {
        Self {
            workers: HashMap::new(),
            next_slot: 0,
            max_workers,
            worker_timeout,
        }
    }

    /// Registers a new worker. Returns the registration response.
    pub fn register(
        &mut self,
        peer_id: PeerId,
        capabilities: NodeCapabilities,
        listen_addr: String,
        public_key: Option<Vec<u8>>,
        max_concurrent_steps: u32,
        available_memory_mb: u64,
    ) -> RegistrationMessage {
        // Check if already registered
        if self.workers.contains_key(&peer_id) {
            return RegistrationMessage::WorkerRegistered {
                accepted: true,
                worker_slot: Some(self.workers[&peer_id].slot),
                reject_reason: None,
                current_round_id: None,
            };
        }

        // Check capacity
        if self.workers.len() as u32 >= self.max_workers {
            return RegistrationMessage::WorkerRegistered {
                accepted: false,
                worker_slot: None,
                reject_reason: Some("Registry full".to_string()),
                current_round_id: None,
            };
        }

        // Check minimum capabilities
        if !capabilities.can_train {
            return RegistrationMessage::WorkerRegistered {
                accepted: false,
                worker_slot: None,
                reject_reason: Some("Worker must have training capability".to_string()),
                current_round_id: None,
            };
        }

        let slot = self.next_slot;
        self.next_slot += 1;

        let now = Instant::now();
        self.workers.insert(
            peer_id.clone(),
            RegisteredWorker {
                peer_id,
                capabilities,
                listen_addr,
                public_key,
                max_concurrent_steps,
                available_memory_mb,
                slot,
                registered_at: now,
                last_heartbeat: now,
                in_round: false,
                rounds_completed: 0,
                rounds_failed: 0,
            },
        );

        RegistrationMessage::WorkerRegistered {
            accepted: true,
            worker_slot: Some(slot),
            reject_reason: None,
            current_round_id: None,
        }
    }

    /// Deregisters a worker.
    pub fn deregister(&mut self, peer_id: &PeerId) -> RegistrationMessage {
        self.workers.remove(peer_id);
        RegistrationMessage::WorkerDeregistered {
            acknowledged: true,
        }
    }

    /// Updates a worker's heartbeat timestamp.
    pub fn heartbeat(&mut self, peer_id: &PeerId) {
        if let Some(worker) = self.workers.get_mut(peer_id) {
            worker.last_heartbeat = Instant::now();
        }
    }

    /// Removes stale workers that haven't sent a heartbeat within the timeout.
    pub fn prune_stale(&mut self) -> Vec<PeerId> {
        let timeout = self.worker_timeout;
        let stale: Vec<PeerId> = self
            .workers
            .iter()
            .filter(|(_, w)| w.last_heartbeat.elapsed() > timeout)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &stale {
            self.workers.remove(id);
        }

        stale
    }

    /// Returns the number of registered workers.
    pub fn count(&self) -> usize {
        self.workers.len()
    }

    /// Returns all registered worker peer IDs.
    pub fn worker_ids(&self) -> Vec<PeerId> {
        self.workers.keys().cloned().collect()
    }

    /// Returns all registered workers.
    pub fn workers(&self) -> &HashMap<PeerId, RegisteredWorker> {
        &self.workers
    }

    /// Returns a mutable reference to a worker by peer ID.
    pub fn get_mut(&mut self, peer_id: &PeerId) -> Option<&mut RegisteredWorker> {
        self.workers.get_mut(peer_id)
    }

    /// Returns a reference to a worker by peer ID.
    pub fn get(&self, peer_id: &PeerId) -> Option<&RegisteredWorker> {
        self.workers.get(peer_id)
    }

    /// Checks if a worker is registered.
    pub fn is_registered(&self, peer_id: &PeerId) -> bool {
        self.workers.contains_key(peer_id)
    }

    /// Returns available (not in-round) workers that can train.
    pub fn available_workers(&self) -> Vec<&RegisteredWorker> {
        self.workers
            .values()
            .filter(|w| !w.in_round && w.capabilities.can_train)
            .collect()
    }

    /// Marks workers as participating in a round.
    pub fn assign_to_round(&mut self, peer_ids: &[PeerId]) {
        for id in peer_ids {
            if let Some(w) = self.workers.get_mut(id) {
                w.in_round = true;
            }
        }
    }

    /// Releases workers from round participation.
    pub fn release_from_round(&mut self, peer_ids: &[PeerId]) {
        for id in peer_ids {
            if let Some(w) = self.workers.get_mut(id) {
                w.in_round = false;
            }
        }
    }

    /// Records a round completion for a worker.
    pub fn record_round_complete(&mut self, peer_id: &PeerId) {
        if let Some(w) = self.workers.get_mut(peer_id) {
            w.rounds_completed += 1;
            w.in_round = false;
        }
    }

    /// Records a round failure for a worker.
    pub fn record_round_failed(&mut self, peer_id: &PeerId) {
        if let Some(w) = self.workers.get_mut(peer_id) {
            w.rounds_failed += 1;
            w.in_round = false;
        }
    }
}

// ============================================================================
// Round Configuration
// ============================================================================

/// Configuration for a single training round.
#[derive(Debug, Clone)]
pub struct RoundConfiguration {
    /// Unique round identifier.
    pub round_id: u64,
    /// On-chain model ID.
    pub model_id: u64,
    /// Model architecture dimensions.
    pub model_dims: ModelDims,
    /// Reference to the training dataset.
    pub dataset_ref: String,
    /// Number of training steps each worker should perform.
    pub steps_per_worker: u32,
    /// Learning rate.
    pub learning_rate: f64,
    /// Maximum error budget.
    pub error_budget: f64,
    /// Minimum workers required to proceed.
    pub min_workers: u32,
    /// Round deadline (Unix timestamp).
    pub deadline: u64,
    /// Current model weight hash.
    pub current_model_hash: [u8; 32],
}

impl RoundConfiguration {
    /// Converts to the wire format message.
    pub fn to_message(&self) -> RoundManagementMessage {
        RoundManagementMessage::RoundConfigure {
            round_id: self.round_id,
            model_id: self.model_id,
            model_dims: self.model_dims.clone(),
            dataset_ref: self.dataset_ref.clone(),
            steps_per_worker: self.steps_per_worker,
            learning_rate: self.learning_rate,
            error_budget: self.error_budget,
            min_workers: self.min_workers,
            deadline: self.deadline,
            current_model_hash: self.current_model_hash,
        }
    }
}

// ============================================================================
// Proof Tracking
// ============================================================================

/// A worker's submitted proof.
#[derive(Debug, Clone)]
pub struct WorkerProofSubmission {
    /// Worker peer ID.
    pub peer_id: PeerId,
    /// ZK proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs (8 × 32-byte field elements).
    pub public_inputs: Vec<[u8; 32]>,
    /// Error bound.
    pub error_bound: f64,
    /// Steps completed.
    pub steps_completed: u32,
    /// New model hash after training.
    pub new_model_hash: [u8; 32],
    /// Serialized checkpoint data.
    pub checkpoint_data: Vec<u8>,
    /// When the proof was received.
    pub received_at: Instant,
    /// Whether the proof passed local validation.
    pub locally_valid: Option<bool>,
}

/// Tracks proof collection for a round.
#[derive(Debug)]
pub struct ProofTracker {
    /// Round ID.
    round_id: u64,
    /// Expected workers.
    expected_workers: Vec<PeerId>,
    /// Collected proofs by peer ID.
    proofs: HashMap<PeerId, WorkerProofSubmission>,
    /// Workers that failed to submit.
    missing_workers: Vec<PeerId>,
    /// Collection started at.
    started_at: Instant,
    /// Collection timeout.
    timeout: Duration,
}

impl ProofTracker {
    /// Creates a new proof tracker.
    pub fn new(round_id: u64, expected_workers: Vec<PeerId>, timeout: Duration) -> Self {
        Self {
            round_id,
            expected_workers: expected_workers.clone(),
            proofs: HashMap::new(),
            missing_workers: expected_workers,
            started_at: Instant::now(),
            timeout,
        }
    }

    /// Submits a proof from a worker. Returns true if the proof was accepted (new).
    pub fn submit_proof(&mut self, submission: WorkerProofSubmission) -> bool {
        let peer_id = submission.peer_id.clone();

        // Only accept from expected workers
        if !self.expected_workers.contains(&peer_id) {
            return false;
        }

        // Don't accept duplicates
        if self.proofs.contains_key(&peer_id) {
            return false;
        }

        self.missing_workers.retain(|id| id != &peer_id);
        self.proofs.insert(peer_id, submission);
        true
    }

    /// Returns whether all expected proofs have been collected.
    pub fn is_complete(&self) -> bool {
        self.missing_workers.is_empty()
    }

    /// Returns whether the collection has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.started_at.elapsed() > self.timeout
    }

    /// Returns collected proof count.
    pub fn collected_count(&self) -> usize {
        self.proofs.len()
    }

    /// Returns expected proof count.
    pub fn expected_count(&self) -> usize {
        self.expected_workers.len()
    }

    /// Returns all collected proofs.
    pub fn proofs(&self) -> &HashMap<PeerId, WorkerProofSubmission> {
        &self.proofs
    }

    /// Consumes the tracker and returns collected proofs.
    pub fn into_proofs(self) -> HashMap<PeerId, WorkerProofSubmission> {
        self.proofs
    }

    /// Returns the list of workers that haven't submitted yet.
    pub fn missing_workers(&self) -> &[PeerId] {
        &self.missing_workers
    }
}

// ============================================================================
// Round Manager State Machine
// ============================================================================

/// Round manager state — the full lifecycle of a training round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundManagerState {
    /// No round in progress. Waiting for round initiation.
    Idle,
    /// Round is being configured. Aggregator has created a RoundConfiguration
    /// and is broadcasting it to registered workers.
    Configuring { round_id: u64 },
    /// Waiting for enough workers to signal readiness.
    WaitingForWorkers {
        round_id: u64,
        ready_count: u32,
        required_count: u32,
    },
    /// Workers are performing training steps.
    Training { round_id: u64 },
    /// Collecting ZK proofs from workers.
    CollectingProofs {
        round_id: u64,
        collected: u32,
        expected: u32,
    },
    /// Aggregating proofs and computing final result.
    Aggregating { round_id: u64 },
    /// Submitting the aggregated result on-chain.
    SubmittingOnChain { round_id: u64 },
    /// Round complete.
    Complete {
        round_id: u64,
        final_commitment: [u8; 32],
    },
    /// Round failed.
    Failed { round_id: u64, reason: String },
}

impl std::fmt::Display for RoundManagerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Configuring { round_id } => write!(f, "Configuring(round={})", round_id),
            Self::WaitingForWorkers { round_id, ready_count, required_count } => {
                write!(f, "WaitingForWorkers(round={}, {}/{})", round_id, ready_count, required_count)
            }
            Self::Training { round_id } => write!(f, "Training(round={})", round_id),
            Self::CollectingProofs { round_id, collected, expected } => {
                write!(f, "CollectingProofs(round={}, {}/{})", round_id, collected, expected)
            }
            Self::Aggregating { round_id } => write!(f, "Aggregating(round={})", round_id),
            Self::SubmittingOnChain { round_id } => write!(f, "SubmittingOnChain(round={})", round_id),
            Self::Complete { round_id, .. } => write!(f, "Complete(round={})", round_id),
            Self::Failed { round_id, reason } => write!(f, "Failed(round={}, {})", round_id, reason),
        }
    }
}

/// Events emitted by the round manager for external consumers.
#[derive(Debug, Clone)]
pub enum RoundManagerEvent {
    /// A worker registered.
    WorkerRegistered { peer_id: PeerId, slot: u32 },
    /// A worker deregistered.
    WorkerDeregistered { peer_id: PeerId },
    /// Round state changed.
    StateChanged { old: RoundManagerState, new: RoundManagerState },
    /// Round configured and sent to workers.
    RoundConfigured { round_id: u64, config: RoundConfiguration },
    /// Worker signaled readiness.
    WorkerReady { round_id: u64, peer_id: PeerId },
    /// Training started.
    TrainingStarted { round_id: u64, participants: Vec<PeerId> },
    /// Proof received from worker.
    ProofReceived { round_id: u64, peer_id: PeerId, valid: bool },
    /// Aggregation complete.
    AggregationComplete { round_id: u64, commitment: [u8; 32] },
    /// On-chain submission started.
    OnChainSubmissionStarted { round_id: u64 },
    /// On-chain submission succeeded.
    OnChainSubmissionComplete { round_id: u64, tx_hash: [u8; 32] },
    /// Round completed.
    RoundCompleted { round_id: u64, final_commitment: [u8; 32] },
    /// Round failed.
    RoundFailed { round_id: u64, reason: String },
}

/// Aggregated result after combining worker proofs.
#[derive(Debug, Clone)]
pub struct AggregatedResult {
    /// Selected/aggregated proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for the aggregated proof.
    pub public_inputs: TrainingProofInputs,
    /// Final model hash after aggregation.
    pub new_model_hash: [u8; 32],
    /// Total error bound.
    pub total_error_bound: f64,
    /// Contributing workers.
    pub contributors: Vec<PeerId>,
}

/// Configuration for the RoundManager.
#[derive(Debug, Clone)]
pub struct RoundManagerConfig {
    /// Maximum workers allowed.
    pub max_workers: u32,
    /// Worker heartbeat timeout.
    pub worker_timeout: Duration,
    /// Proof collection timeout.
    pub proof_collection_timeout: Duration,
    /// Worker readiness timeout.
    pub readiness_timeout: Duration,
    /// Training timeout.
    pub training_timeout: Duration,
    /// Aggregation timeout.
    pub aggregation_timeout: Duration,
    /// On-chain submission timeout.
    pub submission_timeout: Duration,
}

impl Default for RoundManagerConfig {
    fn default() -> Self {
        Self {
            max_workers: 100,
            worker_timeout: Duration::from_secs(60),
            proof_collection_timeout: Duration::from_secs(300),
            readiness_timeout: Duration::from_secs(60),
            training_timeout: Duration::from_secs(600),
            aggregation_timeout: Duration::from_secs(120),
            submission_timeout: Duration::from_secs(120),
        }
    }
}

/// The Round Manager orchestrates the complete lifecycle of training rounds.
///
/// It maintains:
/// - A worker registry for explicit registration/deregistration
/// - The current round state machine
/// - Proof tracking for the active round
/// - Round configuration
/// - Event history
pub struct RoundManager {
    /// Worker registry.
    registry: WorkerRegistry,
    /// Current state.
    state: RoundManagerState,
    /// Configuration.
    config: RoundManagerConfig,
    /// Round counter.
    round_counter: u64,
    /// Current round configuration (set when configuring).
    current_round_config: Option<RoundConfiguration>,
    /// Workers that signaled readiness for current round.
    ready_workers: Vec<PeerId>,
    /// Proof tracker for current round.
    proof_tracker: Option<ProofTracker>,
    /// Aggregated result for current round.
    aggregated_result: Option<AggregatedResult>,
    /// When the current phase started.
    phase_started: Option<Instant>,
    /// Event log.
    events: Vec<RoundManagerEvent>,
    /// Completed round history: round_id → final_commitment.
    completed_rounds: HashMap<u64, [u8; 32]>,
}

impl RoundManager {
    /// Creates a new RoundManager.
    pub fn new(config: RoundManagerConfig) -> Self {
        Self {
            registry: WorkerRegistry::new(config.max_workers, config.worker_timeout),
            state: RoundManagerState::Idle,
            config,
            round_counter: 0,
            current_round_config: None,
            ready_workers: Vec::new(),
            proof_tracker: None,
            aggregated_result: None,
            phase_started: None,
            events: Vec::new(),
            completed_rounds: HashMap::new(),
        }
    }

    // ========================================================================
    // Public accessors
    // ========================================================================

    /// Returns the current state.
    pub fn state(&self) -> &RoundManagerState {
        &self.state
    }

    /// Returns the worker registry.
    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    /// Returns a mutable reference to the worker registry.
    pub fn registry_mut(&mut self) -> &mut WorkerRegistry {
        &mut self.registry
    }

    /// Returns the current round configuration.
    pub fn current_round_config(&self) -> Option<&RoundConfiguration> {
        self.current_round_config.as_ref()
    }

    /// Returns the proof tracker.
    pub fn proof_tracker(&self) -> Option<&ProofTracker> {
        self.proof_tracker.as_ref()
    }

    /// Returns the aggregated result.
    pub fn aggregated_result(&self) -> Option<&AggregatedResult> {
        self.aggregated_result.as_ref()
    }

    /// Drains and returns all pending events.
    pub fn drain_events(&mut self) -> Vec<RoundManagerEvent> {
        std::mem::take(&mut self.events)
    }

    /// Returns completed round history.
    pub fn completed_rounds(&self) -> &HashMap<u64, [u8; 32]> {
        &self.completed_rounds
    }

    // ========================================================================
    // Worker Registration
    // ========================================================================

    /// Handles a worker registration request.
    pub fn handle_register(
        &mut self,
        peer_id: PeerId,
        capabilities: NodeCapabilities,
        listen_addr: String,
        public_key: Option<Vec<u8>>,
        max_concurrent_steps: u32,
        available_memory_mb: u64,
    ) -> RegistrationMessage {
        let response = self.registry.register(
            peer_id.clone(),
            capabilities,
            listen_addr,
            public_key,
            max_concurrent_steps,
            available_memory_mb,
        );

        if let RegistrationMessage::WorkerRegistered { accepted: true, worker_slot: Some(slot), .. } = &response {
            self.events.push(RoundManagerEvent::WorkerRegistered {
                peer_id,
                slot: *slot,
            });
        }

        response
    }

    /// Handles a worker deregistration request.
    pub fn handle_deregister(&mut self, peer_id: &PeerId) -> RegistrationMessage {
        let response = self.registry.deregister(peer_id);
        self.events.push(RoundManagerEvent::WorkerDeregistered {
            peer_id: peer_id.clone(),
        });
        response
    }

    /// Updates a worker's heartbeat.
    pub fn handle_heartbeat(&mut self, peer_id: &PeerId) {
        self.registry.heartbeat(peer_id);
    }

    /// Prunes stale workers and returns their IDs.
    pub fn prune_stale_workers(&mut self) -> Vec<PeerId> {
        self.registry.prune_stale()
    }

    // ========================================================================
    // State Transitions
    // ========================================================================

    fn transition_to(&mut self, new_state: RoundManagerState) {
        let old_state = self.state.clone();
        self.state = new_state.clone();
        self.phase_started = Some(Instant::now());
        self.events.push(RoundManagerEvent::StateChanged {
            old: old_state,
            new: new_state,
        });
    }

    /// Starts a new round by moving to Configuring state.
    /// Returns the RoundConfiguration to broadcast to workers.
    pub fn start_round(&mut self, config: RoundConfiguration) -> Result<RoundConfiguration, RoundManagerError> {
        // Must be Idle to start a new round
        if self.state != RoundManagerState::Idle {
            return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "Configuring".to_string(),
                reason: "Can only start a round from Idle state".to_string(),
            });
        }

        // Check minimum workers are available
        let available = self.registry.available_workers().len();
        if (available as u32) < config.min_workers {
            return Err(RoundManagerError::InsufficientWorkers {
                available: available as u32,
                required: config.min_workers,
            });
        }

        let round_id = config.round_id;
        self.current_round_config = Some(config.clone());
        self.ready_workers.clear();
        self.round_counter = round_id;

        self.transition_to(RoundManagerState::Configuring { round_id });
        self.events.push(RoundManagerEvent::RoundConfigured {
            round_id,
            config: config.clone(),
        });

        Ok(config)
    }

    /// After broadcasting the configuration, move to WaitingForWorkers.
    pub fn configuration_sent(&mut self) -> Result<(), RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::Configuring { round_id } => *round_id,
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "WaitingForWorkers".to_string(),
                reason: "Must be in Configuring state".to_string(),
            }),
        };

        let required = self
            .current_round_config
            .as_ref()
            .map(|c| c.min_workers)
            .unwrap_or(1);

        self.transition_to(RoundManagerState::WaitingForWorkers {
            round_id,
            ready_count: 0,
            required_count: required,
        });

        Ok(())
    }

    /// Handles a worker signaling readiness for the current round.
    /// Returns Ok(true) if the minimum worker count was reached and we should transition.
    pub fn handle_worker_ready(
        &mut self,
        peer_id: &PeerId,
        round_id: u64,
        ready: bool,
    ) -> Result<bool, RoundManagerError> {
        let (state_round_id, required) = match &self.state {
            RoundManagerState::WaitingForWorkers { round_id, required_count, .. } => {
                (*round_id, *required_count)
            }
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "N/A".to_string(),
                reason: "Not in WaitingForWorkers state".to_string(),
            }),
        };

        if round_id != state_round_id {
            return Err(RoundManagerError::WrongRound {
                expected: state_round_id,
                got: round_id,
            });
        }

        if !self.registry.is_registered(peer_id) {
            return Err(RoundManagerError::UnregisteredWorker {
                peer_id: peer_id.clone(),
            });
        }

        if ready && !self.ready_workers.contains(peer_id) {
            self.ready_workers.push(peer_id.clone());
            self.events.push(RoundManagerEvent::WorkerReady {
                round_id,
                peer_id: peer_id.clone(),
            });
        }

        let ready_count = self.ready_workers.len() as u32;

        // Update state
        self.state = RoundManagerState::WaitingForWorkers {
            round_id,
            ready_count,
            required_count: required,
        };

        Ok(ready_count >= required)
    }

    /// Begins training by transitioning from WaitingForWorkers → Training.
    /// Returns the list of participating workers.
    pub fn begin_training(&mut self) -> Result<Vec<PeerId>, RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::WaitingForWorkers { round_id, .. } => *round_id,
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "Training".to_string(),
                reason: "Must be in WaitingForWorkers state".to_string(),
            }),
        };

        let participants = self.ready_workers.clone();
        self.registry.assign_to_round(&participants);

        self.transition_to(RoundManagerState::Training { round_id });
        self.events.push(RoundManagerEvent::TrainingStarted {
            round_id,
            participants: participants.clone(),
        });

        Ok(participants)
    }

    /// Transitions from Training → CollectingProofs when workers have been
    /// notified to submit their proofs.
    pub fn start_collecting_proofs(&mut self) -> Result<(), RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::Training { round_id } => *round_id,
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "CollectingProofs".to_string(),
                reason: "Must be in Training state".to_string(),
            }),
        };

        let expected_workers = self.ready_workers.clone();
        let expected = expected_workers.len() as u32;

        self.proof_tracker = Some(ProofTracker::new(
            round_id,
            expected_workers,
            self.config.proof_collection_timeout,
        ));

        self.transition_to(RoundManagerState::CollectingProofs {
            round_id,
            collected: 0,
            expected,
        });

        Ok(())
    }

    /// Handles a proof submission from a worker.
    /// Returns Ok(true) if all expected proofs have been collected.
    pub fn handle_proof_submission(
        &mut self,
        peer_id: PeerId,
        round_id: u64,
        proof: Vec<u8>,
        public_inputs: Vec<[u8; 32]>,
        error_bound: f64,
        steps_completed: u32,
        new_model_hash: [u8; 32],
        checkpoint_data: Vec<u8>,
    ) -> Result<bool, RoundManagerError> {
        let (state_round_id, expected) = match &self.state {
            RoundManagerState::CollectingProofs { round_id, expected, .. } => (*round_id, *expected),
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "N/A".to_string(),
                reason: "Not in CollectingProofs state".to_string(),
            }),
        };

        if round_id != state_round_id {
            return Err(RoundManagerError::WrongRound {
                expected: state_round_id,
                got: round_id,
            });
        }

        let submission = WorkerProofSubmission {
            peer_id: peer_id.clone(),
            proof,
            public_inputs,
            error_bound,
            steps_completed,
            new_model_hash,
            checkpoint_data,
            received_at: Instant::now(),
            locally_valid: None,
        };

        let tracker = self.proof_tracker.as_mut().ok_or_else(|| {
            RoundManagerError::InternalError("Proof tracker not initialized".to_string())
        })?;

        let accepted = tracker.submit_proof(submission);

        self.events.push(RoundManagerEvent::ProofReceived {
            round_id,
            peer_id,
            valid: accepted,
        });

        let collected = tracker.collected_count() as u32;
        self.state = RoundManagerState::CollectingProofs {
            round_id,
            collected,
            expected,
        };

        Ok(tracker.is_complete())
    }

    /// Transitions from CollectingProofs → Aggregating.
    pub fn start_aggregation(&mut self) -> Result<HashMap<PeerId, WorkerProofSubmission>, RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::CollectingProofs { round_id, .. } => *round_id,
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "Aggregating".to_string(),
                reason: "Must be in CollectingProofs state".to_string(),
            }),
        };

        let tracker = self.proof_tracker.take().ok_or_else(|| {
            RoundManagerError::InternalError("Proof tracker not initialized".to_string())
        })?;

        let proofs = tracker.into_proofs();

        self.transition_to(RoundManagerState::Aggregating { round_id });

        Ok(proofs)
    }

    /// Sets the aggregated result and transitions to SubmittingOnChain.
    pub fn set_aggregated_result(
        &mut self,
        result: AggregatedResult,
    ) -> Result<(), RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::Aggregating { round_id } => *round_id,
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "SubmittingOnChain".to_string(),
                reason: "Must be in Aggregating state".to_string(),
            }),
        };

        let commitment = result.new_model_hash;
        self.aggregated_result = Some(result);

        self.events.push(RoundManagerEvent::AggregationComplete {
            round_id,
            commitment,
        });

        self.transition_to(RoundManagerState::SubmittingOnChain { round_id });
        self.events.push(RoundManagerEvent::OnChainSubmissionStarted { round_id });

        Ok(())
    }

    /// Records successful on-chain submission and transitions to Complete.
    /// Returns the RoundCompleted message to broadcast.
    pub fn on_chain_submission_complete(
        &mut self,
        tx_hash: [u8; 32],
    ) -> Result<RoundManagementMessage, RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::SubmittingOnChain { round_id } => *round_id,
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "Complete".to_string(),
                reason: "Must be in SubmittingOnChain state".to_string(),
            }),
        };

        let result = self.aggregated_result.as_ref().ok_or_else(|| {
            RoundManagerError::InternalError("No aggregated result".to_string())
        })?;

        let final_commitment = result.new_model_hash;
        let total_error_bound = result.total_error_bound;
        let num_contributors = result.contributors.len() as u32;

        // Release workers from round
        let participants = self.ready_workers.clone();
        for pid in &participants {
            self.registry.record_round_complete(pid);
        }

        self.completed_rounds.insert(round_id, final_commitment);

        self.transition_to(RoundManagerState::Complete {
            round_id,
            final_commitment,
        });

        self.events.push(RoundManagerEvent::OnChainSubmissionComplete {
            round_id,
            tx_hash,
        });
        self.events.push(RoundManagerEvent::RoundCompleted {
            round_id,
            final_commitment,
        });

        Ok(RoundManagementMessage::RoundCompleted {
            round_id,
            final_commitment,
            tx_hash: Some(tx_hash),
            new_model_hash: final_commitment,
            total_error_bound,
            num_contributors,
        })
    }

    /// Completes a round without on-chain submission (offline mode).
    pub fn complete_offline(&mut self) -> Result<RoundManagementMessage, RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::Aggregating { round_id }
            | RoundManagerState::SubmittingOnChain { round_id } => *round_id,
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "Complete".to_string(),
                reason: "Must be in Aggregating or SubmittingOnChain state".to_string(),
            }),
        };

        let result = self.aggregated_result.as_ref().ok_or_else(|| {
            RoundManagerError::InternalError("No aggregated result".to_string())
        })?;

        let final_commitment = result.new_model_hash;
        let total_error_bound = result.total_error_bound;
        let num_contributors = result.contributors.len() as u32;

        let participants = self.ready_workers.clone();
        for pid in &participants {
            self.registry.record_round_complete(pid);
        }

        self.completed_rounds.insert(round_id, final_commitment);

        self.transition_to(RoundManagerState::Complete {
            round_id,
            final_commitment,
        });

        self.events.push(RoundManagerEvent::RoundCompleted {
            round_id,
            final_commitment,
        });

        Ok(RoundManagementMessage::RoundCompleted {
            round_id,
            final_commitment,
            tx_hash: None,
            new_model_hash: final_commitment,
            total_error_bound,
            num_contributors,
        })
    }

    /// Fails the current round with a reason.
    pub fn fail_round(&mut self, reason: String) -> Result<RoundManagementMessage, RoundManagerError> {
        let round_id = match &self.state {
            RoundManagerState::Idle => return Err(RoundManagerError::InvalidTransition {
                from: "Idle".to_string(),
                to: "Failed".to_string(),
                reason: "No round in progress to fail".to_string(),
            }),
            RoundManagerState::Complete { round_id, .. } => *round_id,
            RoundManagerState::Failed { round_id, .. } => *round_id,
            RoundManagerState::Configuring { round_id }
            | RoundManagerState::WaitingForWorkers { round_id, .. }
            | RoundManagerState::Training { round_id }
            | RoundManagerState::CollectingProofs { round_id, .. }
            | RoundManagerState::Aggregating { round_id }
            | RoundManagerState::SubmittingOnChain { round_id } => *round_id,
        };

        // Release workers from round
        let participants = self.ready_workers.clone();
        for pid in &participants {
            self.registry.record_round_failed(pid);
        }

        self.transition_to(RoundManagerState::Failed {
            round_id,
            reason: reason.clone(),
        });

        self.events.push(RoundManagerEvent::RoundFailed {
            round_id,
            reason: reason.clone(),
        });

        Ok(RoundManagementMessage::RoundFailed { round_id, reason })
    }

    /// Resets to Idle after a round completes or fails.
    pub fn reset_to_idle(&mut self) -> Result<(), RoundManagerError> {
        match &self.state {
            RoundManagerState::Complete { .. } | RoundManagerState::Failed { .. } => {}
            _ => return Err(RoundManagerError::InvalidTransition {
                from: self.state.to_string(),
                to: "Idle".to_string(),
                reason: "Can only reset from Complete or Failed state".to_string(),
            }),
        }

        self.current_round_config = None;
        self.ready_workers.clear();
        self.proof_tracker = None;
        self.aggregated_result = None;
        self.phase_started = None;

        self.transition_to(RoundManagerState::Idle);

        Ok(())
    }

    /// Returns the next round ID (auto-incrementing).
    pub fn next_round_id(&self) -> u64 {
        self.round_counter + 1
    }

    /// Checks for phase timeouts and returns the timed-out phase if any.
    pub fn check_timeout(&self) -> Option<&RoundManagerState> {
        let started = self.phase_started?;
        let elapsed = started.elapsed();

        match &self.state {
            RoundManagerState::WaitingForWorkers { .. } if elapsed > self.config.readiness_timeout => {
                Some(&self.state)
            }
            RoundManagerState::Training { .. } if elapsed > self.config.training_timeout => {
                Some(&self.state)
            }
            RoundManagerState::CollectingProofs { .. } if elapsed > self.config.proof_collection_timeout => {
                Some(&self.state)
            }
            RoundManagerState::Aggregating { .. } if elapsed > self.config.aggregation_timeout => {
                Some(&self.state)
            }
            RoundManagerState::SubmittingOnChain { .. } if elapsed > self.config.submission_timeout => {
                Some(&self.state)
            }
            _ => None,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Round manager errors.
#[derive(Debug, Clone)]
pub enum RoundManagerError {
    /// Invalid state transition.
    InvalidTransition {
        from: String,
        to: String,
        reason: String,
    },
    /// Not enough workers available.
    InsufficientWorkers {
        available: u32,
        required: u32,
    },
    /// Wrong round ID.
    WrongRound {
        expected: u64,
        got: u64,
    },
    /// Worker not registered.
    UnregisteredWorker {
        peer_id: PeerId,
    },
    /// Internal error.
    InternalError(String),
}

impl std::fmt::Display for RoundManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition { from, to, reason } => {
                write!(f, "Invalid transition {} → {}: {}", from, to, reason)
            }
            Self::InsufficientWorkers { available, required } => {
                write!(f, "Insufficient workers: {}/{} required", available, required)
            }
            Self::WrongRound { expected, got } => {
                write!(f, "Wrong round: expected {}, got {}", expected, got)
            }
            Self::UnregisteredWorker { peer_id } => {
                write!(f, "Worker {} not registered", peer_id)
            }
            Self::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for RoundManagerError {}

// ============================================================================
// Helper: Compute commitment from model hash
// ============================================================================

/// Computes a SHA-256 commitment from multiple model hashes (FedAvg result).
pub fn compute_model_commitment(model_hashes: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for hash in model_hashes {
        hasher.update(hash);
    }
    hasher.finalize().into()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_capabilities() -> NodeCapabilities {
        NodeCapabilities {
            can_train: true,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 4096,
            cpu_cores: 8,
            storage_gb: 100,
        }
    }

    fn make_config(round_id: u64, min_workers: u32) -> RoundConfiguration {
        RoundConfiguration {
            round_id,
            model_id: 1,
            model_dims: ModelDims {
                d_in: 4,
                d_hid: 8,
                d_out: 2,
                num_layers: 1,
                num_heads: 0,
                activation_type: 0,
            },
            dataset_ref: "ipfs://QmTest123".to_string(),
            steps_per_worker: 5,
            learning_rate: 0.001,
            error_budget: 0.1,
            min_workers,
            deadline: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            current_model_hash: [0xAA; 32],
        }
    }

    #[test]
    fn test_worker_registration() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        let peer = PeerId::from_string("worker-1");
        let resp = mgr.handle_register(
            peer.clone(),
            make_capabilities(),
            "127.0.0.1:8000".to_string(),
            None,
            4,
            8192,
        );

        match resp {
            RegistrationMessage::WorkerRegistered { accepted, worker_slot, .. } => {
                assert!(accepted);
                assert_eq!(worker_slot, Some(0));
            }
            _ => panic!("Expected WorkerRegistered"),
        }

        assert_eq!(mgr.registry().count(), 1);
        assert!(mgr.registry().is_registered(&peer));
    }

    #[test]
    fn test_worker_registration_duplicate() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        let peer = PeerId::from_string("worker-1");
        mgr.handle_register(peer.clone(), make_capabilities(), "addr".to_string(), None, 4, 8192);
        let resp = mgr.handle_register(peer.clone(), make_capabilities(), "addr".to_string(), None, 4, 8192);

        match resp {
            RegistrationMessage::WorkerRegistered { accepted, worker_slot, .. } => {
                assert!(accepted);
                assert_eq!(worker_slot, Some(0)); // Same slot
            }
            _ => panic!("Expected WorkerRegistered"),
        }

        assert_eq!(mgr.registry().count(), 1);
    }

    #[test]
    fn test_worker_registration_no_training_capability() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        let peer = PeerId::from_string("worker-bad");
        let caps = NodeCapabilities {
            can_train: false,
            ..make_capabilities()
        };
        let resp = mgr.handle_register(peer, caps, "addr".to_string(), None, 4, 8192);

        match resp {
            RegistrationMessage::WorkerRegistered { accepted, reject_reason, .. } => {
                assert!(!accepted);
                assert!(reject_reason.is_some());
            }
            _ => panic!("Expected WorkerRegistered"),
        }

        assert_eq!(mgr.registry().count(), 0);
    }

    #[test]
    fn test_worker_deregistration() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        let peer = PeerId::from_string("worker-1");
        mgr.handle_register(peer.clone(), make_capabilities(), "addr".to_string(), None, 4, 8192);
        assert_eq!(mgr.registry().count(), 1);

        mgr.handle_deregister(&peer);
        assert_eq!(mgr.registry().count(), 0);
        assert!(!mgr.registry().is_registered(&peer));
    }

    #[test]
    fn test_full_round_lifecycle() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Register 3 workers
        let workers: Vec<PeerId> = (1..=3)
            .map(|i| PeerId::from_string(format!("worker-{}", i)))
            .collect();

        for w in &workers {
            mgr.handle_register(
                w.clone(),
                make_capabilities(),
                format!("127.0.0.1:{}", 8000 + w.0.chars().last().unwrap().to_digit(10).unwrap()),
                None,
                4,
                8192,
            );
        }
        assert_eq!(mgr.registry().count(), 3);

        // 1. Start round (Idle → Configuring)
        let config = make_config(1, 3);
        let config = mgr.start_round(config).unwrap();
        assert!(matches!(mgr.state(), RoundManagerState::Configuring { round_id: 1 }));

        // 2. Configuration sent (Configuring → WaitingForWorkers)
        mgr.configuration_sent().unwrap();
        assert!(matches!(
            mgr.state(),
            RoundManagerState::WaitingForWorkers { round_id: 1, ready_count: 0, required_count: 3 }
        ));

        // 3. Workers signal readiness
        for (i, w) in workers.iter().enumerate() {
            let reached = mgr.handle_worker_ready(w, 1, true).unwrap();
            if i < 2 {
                assert!(!reached);
            } else {
                assert!(reached); // 3rd worker triggers threshold
            }
        }

        // 4. Begin training (WaitingForWorkers → Training)
        let participants = mgr.begin_training().unwrap();
        assert_eq!(participants.len(), 3);
        assert!(matches!(mgr.state(), RoundManagerState::Training { round_id: 1 }));

        // 5. Start collecting proofs (Training → CollectingProofs)
        mgr.start_collecting_proofs().unwrap();
        assert!(matches!(
            mgr.state(),
            RoundManagerState::CollectingProofs { round_id: 1, collected: 0, expected: 3 }
        ));

        // 6. Workers submit proofs
        for (i, w) in workers.iter().enumerate() {
            let all_collected = mgr.handle_proof_submission(
                w.clone(),
                1,
                vec![0x42; 100], // mock proof
                vec![[0u8; 32]; 8], // mock public inputs
                0.05,
                5,
                [0xBB; 32],
                vec![0x01; 50], // mock checkpoint
            ).unwrap();

            if i < 2 {
                assert!(!all_collected);
            } else {
                assert!(all_collected);
            }
        }

        // 7. Start aggregation (CollectingProofs → Aggregating)
        let proofs = mgr.start_aggregation().unwrap();
        assert_eq!(proofs.len(), 3);
        assert!(matches!(mgr.state(), RoundManagerState::Aggregating { round_id: 1 }));

        // 8. Set aggregated result (Aggregating → SubmittingOnChain)
        let aggregated = AggregatedResult {
            proof: vec![0xDE; 200],
            public_inputs: TrainingProofInputs {
                old_hash_lo: ethers::types::U256::from(1),
                old_hash_hi: ethers::types::U256::from(2),
                new_hash_lo: ethers::types::U256::from(3),
                new_hash_hi: ethers::types::U256::from(4),
                loss: ethers::types::U256::from(100),
                error_bound: ethers::types::U256::from(10),
                step_number: ethers::types::U256::from(1),
                error_checksum: ethers::types::U256::from(0),
            },
            new_model_hash: [0xCC; 32],
            total_error_bound: 0.15,
            contributors: workers.clone(),
        };
        mgr.set_aggregated_result(aggregated).unwrap();
        assert!(matches!(mgr.state(), RoundManagerState::SubmittingOnChain { round_id: 1 }));

        // 9. On-chain submission complete (SubmittingOnChain → Complete)
        let completion_msg = mgr.on_chain_submission_complete([0xDD; 32]).unwrap();
        match &completion_msg {
            RoundManagementMessage::RoundCompleted {
                round_id,
                final_commitment,
                tx_hash,
                num_contributors,
                ..
            } => {
                assert_eq!(*round_id, 1);
                assert_eq!(*final_commitment, [0xCC; 32]);
                assert!(tx_hash.is_some());
                assert_eq!(*num_contributors, 3);
            }
            _ => panic!("Expected RoundCompleted message"),
        }

        assert!(matches!(
            mgr.state(),
            RoundManagerState::Complete { round_id: 1, .. }
        ));

        // Check completed rounds
        assert_eq!(mgr.completed_rounds().len(), 1);
        assert_eq!(mgr.completed_rounds()[&1], [0xCC; 32]);

        // 10. Reset to idle for next round
        mgr.reset_to_idle().unwrap();
        assert!(matches!(mgr.state(), RoundManagerState::Idle));

        // Workers should be released
        for w in &workers {
            let worker = mgr.registry().get(w).unwrap();
            assert!(!worker.in_round);
            assert_eq!(worker.rounds_completed, 1);
        }
    }

    #[test]
    fn test_round_failure() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Register 3 workers
        for i in 1..=3 {
            mgr.handle_register(
                PeerId::from_string(format!("worker-{}", i)),
                make_capabilities(),
                "addr".to_string(),
                None,
                4,
                8192,
            );
        }

        let config = make_config(1, 3);
        mgr.start_round(config).unwrap();
        mgr.configuration_sent().unwrap();

        // Fail the round
        let fail_msg = mgr.fail_round("Test failure".to_string()).unwrap();
        match fail_msg {
            RoundManagementMessage::RoundFailed { round_id, reason } => {
                assert_eq!(round_id, 1);
                assert_eq!(reason, "Test failure");
            }
            _ => panic!("Expected RoundFailed message"),
        }

        assert!(matches!(
            mgr.state(),
            RoundManagerState::Failed { round_id: 1, .. }
        ));

        // Can reset after failure
        mgr.reset_to_idle().unwrap();
        assert!(matches!(mgr.state(), RoundManagerState::Idle));
    }

    #[test]
    fn test_insufficient_workers() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Only register 1 worker
        mgr.handle_register(
            PeerId::from_string("worker-1"),
            make_capabilities(),
            "addr".to_string(),
            None,
            4,
            8192,
        );

        let config = make_config(1, 3); // requires 3
        let result = mgr.start_round(config);
        assert!(result.is_err());
        match result.unwrap_err() {
            RoundManagerError::InsufficientWorkers { available, required } => {
                assert_eq!(available, 1);
                assert_eq!(required, 3);
            }
            _ => panic!("Expected InsufficientWorkers"),
        }
    }

    #[test]
    fn test_invalid_state_transitions() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Can't go from Idle to WaitingForWorkers directly
        assert!(mgr.configuration_sent().is_err());

        // Can't begin training from Idle
        assert!(mgr.begin_training().is_err());

        // Can't start collecting from Idle
        assert!(mgr.start_collecting_proofs().is_err());

        // Can't start aggregation from Idle
        assert!(mgr.start_aggregation().is_err());

        // Can't reset from Idle
        assert!(mgr.reset_to_idle().is_err());
    }

    #[test]
    fn test_unregistered_worker_readiness() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Register workers and start round
        for i in 1..=3 {
            mgr.handle_register(
                PeerId::from_string(format!("worker-{}", i)),
                make_capabilities(),
                "addr".to_string(),
                None,
                4,
                8192,
            );
        }

        let config = make_config(1, 3);
        mgr.start_round(config).unwrap();
        mgr.configuration_sent().unwrap();

        // Unregistered worker tries to signal readiness
        let unknown = PeerId::from_string("unknown-worker");
        let result = mgr.handle_worker_ready(&unknown, 1, true);
        assert!(result.is_err());
        match result.unwrap_err() {
            RoundManagerError::UnregisteredWorker { peer_id } => {
                assert_eq!(peer_id, unknown);
            }
            _ => panic!("Expected UnregisteredWorker"),
        }
    }

    #[test]
    fn test_wrong_round_id_readiness() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        for i in 1..=3 {
            mgr.handle_register(
                PeerId::from_string(format!("worker-{}", i)),
                make_capabilities(),
                "addr".to_string(),
                None,
                4,
                8192,
            );
        }

        let config = make_config(1, 3);
        mgr.start_round(config).unwrap();
        mgr.configuration_sent().unwrap();

        let worker = PeerId::from_string("worker-1");
        let result = mgr.handle_worker_ready(&worker, 99, true); // wrong round
        assert!(result.is_err());
        match result.unwrap_err() {
            RoundManagerError::WrongRound { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 99);
            }
            _ => panic!("Expected WrongRound"),
        }
    }

    #[test]
    fn test_offline_completion() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Register workers
        let workers: Vec<PeerId> = (1..=2)
            .map(|i| PeerId::from_string(format!("worker-{}", i)))
            .collect();
        for w in &workers {
            mgr.handle_register(w.clone(), make_capabilities(), "addr".to_string(), None, 4, 8192);
        }

        // Go through lifecycle to Aggregating
        let config = make_config(1, 2);
        mgr.start_round(config).unwrap();
        mgr.configuration_sent().unwrap();
        for w in &workers {
            mgr.handle_worker_ready(w, 1, true).unwrap();
        }
        mgr.begin_training().unwrap();
        mgr.start_collecting_proofs().unwrap();
        for w in &workers {
            mgr.handle_proof_submission(
                w.clone(), 1, vec![0x42; 10], vec![[0u8; 32]; 8],
                0.05, 5, [0xBB; 32], vec![0x01; 10],
            ).unwrap();
        }
        mgr.start_aggregation().unwrap();

        // Set result and complete offline
        let result = AggregatedResult {
            proof: vec![],
            public_inputs: TrainingProofInputs {
                old_hash_lo: ethers::types::U256::from(0),
                old_hash_hi: ethers::types::U256::from(0),
                new_hash_lo: ethers::types::U256::from(0),
                new_hash_hi: ethers::types::U256::from(0),
                loss: ethers::types::U256::from(0),
                error_bound: ethers::types::U256::from(0),
                step_number: ethers::types::U256::from(0),
                error_checksum: ethers::types::U256::from(0),
            },
            new_model_hash: [0xEE; 32],
            total_error_bound: 0.1,
            contributors: workers.clone(),
        };
        mgr.set_aggregated_result(result).unwrap();

        // Complete offline (no on-chain submission)
        let msg = mgr.complete_offline().unwrap();
        match msg {
            RoundManagementMessage::RoundCompleted { tx_hash, .. } => {
                assert!(tx_hash.is_none()); // No tx hash in offline mode
            }
            _ => panic!("Expected RoundCompleted"),
        }

        assert!(matches!(mgr.state(), RoundManagerState::Complete { .. }));
    }

    #[test]
    fn test_proof_tracker_rejects_unknown_worker() {
        let workers = vec![PeerId::from_string("w1"), PeerId::from_string("w2")];
        let mut tracker = ProofTracker::new(1, workers, Duration::from_secs(300));

        let submission = WorkerProofSubmission {
            peer_id: PeerId::from_string("unknown"),
            proof: vec![],
            public_inputs: vec![],
            error_bound: 0.0,
            steps_completed: 0,
            new_model_hash: [0; 32],
            checkpoint_data: vec![],
            received_at: Instant::now(),
            locally_valid: None,
        };

        assert!(!tracker.submit_proof(submission));
        assert_eq!(tracker.collected_count(), 0);
    }

    #[test]
    fn test_proof_tracker_rejects_duplicate() {
        let w1 = PeerId::from_string("w1");
        let workers = vec![w1.clone()];
        let mut tracker = ProofTracker::new(1, workers, Duration::from_secs(300));

        let submission = WorkerProofSubmission {
            peer_id: w1.clone(),
            proof: vec![0x42],
            public_inputs: vec![],
            error_bound: 0.05,
            steps_completed: 5,
            new_model_hash: [0xBB; 32],
            checkpoint_data: vec![],
            received_at: Instant::now(),
            locally_valid: None,
        };

        assert!(tracker.submit_proof(submission.clone()));
        // Duplicate should be rejected
        let dup = WorkerProofSubmission {
            peer_id: w1,
            proof: vec![0x43],
            ..submission
        };
        assert!(!tracker.submit_proof(dup));
        assert_eq!(tracker.collected_count(), 1);
    }

    #[test]
    fn test_worker_registry_max_capacity() {
        let mut registry = WorkerRegistry::new(2, Duration::from_secs(60));

        let resp1 = registry.register(
            PeerId::from_string("w1"), make_capabilities(),
            "addr1".to_string(), None, 4, 8192,
        );
        assert!(matches!(resp1, RegistrationMessage::WorkerRegistered { accepted: true, .. }));

        let resp2 = registry.register(
            PeerId::from_string("w2"), make_capabilities(),
            "addr2".to_string(), None, 4, 8192,
        );
        assert!(matches!(resp2, RegistrationMessage::WorkerRegistered { accepted: true, .. }));

        // Third should be rejected
        let resp3 = registry.register(
            PeerId::from_string("w3"), make_capabilities(),
            "addr3".to_string(), None, 4, 8192,
        );
        match resp3 {
            RegistrationMessage::WorkerRegistered { accepted, reject_reason, .. } => {
                assert!(!accepted);
                assert!(reject_reason.unwrap().contains("full"));
            }
            _ => panic!("Expected WorkerRegistered"),
        }
    }

    #[test]
    fn test_events_are_emitted() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Register a worker
        mgr.handle_register(
            PeerId::from_string("w1"),
            make_capabilities(),
            "addr".to_string(),
            None,
            4,
            8192,
        );

        let events = mgr.drain_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], RoundManagerEvent::WorkerRegistered { .. }));

        // Events should be empty after drain
        assert!(mgr.drain_events().is_empty());
    }

    #[test]
    fn test_compute_model_commitment() {
        let h1 = [0xAA; 32];
        let h2 = [0xBB; 32];
        let c1 = compute_model_commitment(&[h1, h2]);
        let c2 = compute_model_commitment(&[h1, h2]);
        assert_eq!(c1, c2); // deterministic

        // Different order → different commitment
        let c3 = compute_model_commitment(&[h2, h1]);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_three_worker_integration_round() {
        // Full 3-worker integration test simulating:
        // register → configure → ready → train → proof → aggregate → submit → complete

        let mut mgr = RoundManager::new(RoundManagerConfig {
            max_workers: 10,
            worker_timeout: Duration::from_secs(60),
            proof_collection_timeout: Duration::from_secs(300),
            readiness_timeout: Duration::from_secs(60),
            training_timeout: Duration::from_secs(600),
            aggregation_timeout: Duration::from_secs(120),
            submission_timeout: Duration::from_secs(120),
        });

        // === Phase 1: Registration ===
        let w1 = PeerId::from_string("worker-alpha");
        let w2 = PeerId::from_string("worker-beta");
        let w3 = PeerId::from_string("worker-gamma");
        let all_workers = vec![w1.clone(), w2.clone(), w3.clone()];

        for w in &all_workers {
            let resp = mgr.handle_register(
                w.clone(),
                NodeCapabilities {
                    can_train: true,
                    can_aggregate: false,
                    can_prove: true,
                    gpu_memory_mb: 8192,
                    cpu_cores: 16,
                    storage_gb: 500,
                },
                format!("10.0.0.{}:9000", all_workers.iter().position(|x| x == w).unwrap() + 1),
                None,
                8,
                16384,
            );
            assert!(matches!(resp, RegistrationMessage::WorkerRegistered { accepted: true, .. }));
        }
        assert_eq!(mgr.registry().count(), 3);

        // === Phase 2: Configure round ===
        let config = RoundConfiguration {
            round_id: 1,
            model_id: 42,
            model_dims: ModelDims {
                d_in: 10,
                d_hid: 20,
                d_out: 5,
                num_layers: 2,
                num_heads: 0,
                activation_type: 0,
            },
            dataset_ref: "ipfs://QmAbcDef12345".to_string(),
            steps_per_worker: 10,
            learning_rate: 0.0001,
            error_budget: 0.05,
            min_workers: 3,
            deadline: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 7200,
            current_model_hash: [0x11; 32],
        };

        let round_config = mgr.start_round(config).unwrap();
        assert_eq!(round_config.model_id, 42);
        assert_eq!(round_config.steps_per_worker, 10);

        mgr.configuration_sent().unwrap();

        // === Phase 3: Workers signal readiness ===
        for w in &all_workers {
            let reached = mgr.handle_worker_ready(w, 1, true).unwrap();
            if w == &w3 {
                assert!(reached, "Should reach quorum on 3rd worker");
            }
        }

        // === Phase 4: Begin training ===
        let participants = mgr.begin_training().unwrap();
        assert_eq!(participants.len(), 3);
        assert!(matches!(mgr.state(), RoundManagerState::Training { round_id: 1 }));

        // Verify workers are marked as in-round
        for w in &all_workers {
            assert!(mgr.registry().get(w).unwrap().in_round);
        }

        // === Phase 5: Transition to proof collection ===
        mgr.start_collecting_proofs().unwrap();

        // === Phase 6: Workers submit proofs ===
        // Each worker submits a proof with a different model hash
        let worker_hashes: Vec<[u8; 32]> = vec![[0x22; 32], [0x33; 32], [0x44; 32]];

        for (i, w) in all_workers.iter().enumerate() {
            let mut fake_inputs = vec![[0u8; 32]; 8];
            // Set step number in public input
            fake_inputs[6][0] = 10; // 10 steps

            let complete = mgr.handle_proof_submission(
                w.clone(),
                1,
                vec![0xAB; 1856], // Realistic proof size
                fake_inputs,
                0.03 + (i as f64 * 0.01),
                10,
                worker_hashes[i],
                vec![0xCD; 256], // Checkpoint data
            ).unwrap();

            if i == 2 {
                assert!(complete, "All 3 proofs should be collected");
            }
        }

        // === Phase 7: Aggregate ===
        let collected_proofs = mgr.start_aggregation().unwrap();
        assert_eq!(collected_proofs.len(), 3);

        // Compute aggregated commitment
        let final_hash = compute_model_commitment(&worker_hashes);

        let aggregated = AggregatedResult {
            proof: collected_proofs.values().next().unwrap().proof.clone(), // Use best proof
            public_inputs: TrainingProofInputs {
                old_hash_lo: ethers::types::U256::from(1),
                old_hash_hi: ethers::types::U256::from(0),
                new_hash_lo: ethers::types::U256::from(3),
                new_hash_hi: ethers::types::U256::from(0),
                loss: ethers::types::U256::from(50),
                error_bound: ethers::types::U256::from(5),
                step_number: ethers::types::U256::from(10),
                error_checksum: ethers::types::U256::from(0),
            },
            new_model_hash: final_hash,
            total_error_bound: 0.09,
            contributors: all_workers.clone(),
        };

        mgr.set_aggregated_result(aggregated).unwrap();
        assert!(matches!(mgr.state(), RoundManagerState::SubmittingOnChain { round_id: 1 }));

        // === Phase 8: On-chain submission ===
        let fake_tx_hash = [0xFF; 32];
        let completion_msg = mgr.on_chain_submission_complete(fake_tx_hash).unwrap();

        match &completion_msg {
            RoundManagementMessage::RoundCompleted {
                round_id,
                final_commitment,
                tx_hash,
                new_model_hash,
                total_error_bound,
                num_contributors,
            } => {
                assert_eq!(*round_id, 1);
                assert_eq!(*final_commitment, final_hash);
                assert_eq!(tx_hash.unwrap(), fake_tx_hash);
                assert_eq!(*new_model_hash, final_hash);
                assert!(*total_error_bound < 0.1);
                assert_eq!(*num_contributors, 3);
            }
            _ => panic!("Expected RoundCompleted"),
        }

        // === Phase 9: Verify completion ===
        assert!(matches!(
            mgr.state(),
            RoundManagerState::Complete { round_id: 1, .. }
        ));

        // Workers should be released and have 1 completed round
        mgr.reset_to_idle().unwrap();
        for w in &all_workers {
            let info = mgr.registry().get(w).unwrap();
            assert!(!info.in_round);
            assert_eq!(info.rounds_completed, 1);
            assert_eq!(info.rounds_failed, 0);
        }

        // Verify events were emitted (just check count, we tested specifics above)
        // Events were drained implicitly by the assertions, but we can check completed_rounds
        assert_eq!(mgr.completed_rounds().len(), 1);

        // === Phase 10: Start a second round ===
        let config2 = make_config(2, 2); // Only need 2 workers this time
        mgr.start_round(config2).unwrap();
        assert!(matches!(mgr.state(), RoundManagerState::Configuring { round_id: 2 }));
    }

    #[test]
    fn test_worker_join_between_rounds() {
        let mut mgr = RoundManager::new(RoundManagerConfig::default());

        // Start with 3 workers
        for i in 1..=3 {
            mgr.handle_register(
                PeerId::from_string(format!("w{}", i)),
                make_capabilities(),
                "addr".to_string(),
                None,
                4,
                8192,
            );
        }

        // Complete a round
        let config = make_config(1, 2);
        mgr.start_round(config).unwrap();
        mgr.configuration_sent().unwrap();
        for i in 1..=2 {
            mgr.handle_worker_ready(&PeerId::from_string(format!("w{}", i)), 1, true).unwrap();
        }
        mgr.begin_training().unwrap();
        mgr.start_collecting_proofs().unwrap();
        for i in 1..=2 {
            mgr.handle_proof_submission(
                PeerId::from_string(format!("w{}", i)), 1,
                vec![0x42; 10], vec![[0u8; 32]; 8],
                0.05, 5, [0xBB; 32], vec![],
            ).unwrap();
        }
        mgr.start_aggregation().unwrap();
        mgr.set_aggregated_result(AggregatedResult {
            proof: vec![],
            public_inputs: TrainingProofInputs {
                old_hash_lo: ethers::types::U256::from(0),
                old_hash_hi: ethers::types::U256::from(0),
                new_hash_lo: ethers::types::U256::from(0),
                new_hash_hi: ethers::types::U256::from(0),
                loss: ethers::types::U256::from(0),
                error_bound: ethers::types::U256::from(0),
                step_number: ethers::types::U256::from(0),
                error_checksum: ethers::types::U256::from(0),
            },
            new_model_hash: [0xBB; 32],
            total_error_bound: 0.1,
            contributors: vec![PeerId::from_string("w1"), PeerId::from_string("w2")],
        }).unwrap();
        mgr.complete_offline().unwrap();
        mgr.reset_to_idle().unwrap();

        // New worker joins between rounds
        let new_worker = PeerId::from_string("w4-new");
        let resp = mgr.handle_register(
            new_worker.clone(),
            make_capabilities(),
            "new-addr".to_string(),
            None,
            4,
            8192,
        );
        assert!(matches!(resp, RegistrationMessage::WorkerRegistered { accepted: true, .. }));
        assert_eq!(mgr.registry().count(), 4);

        // Worker leaves between rounds
        mgr.handle_deregister(&PeerId::from_string("w3"));
        assert_eq!(mgr.registry().count(), 3);

        // Can start next round with new composition
        let config2 = make_config(2, 3);
        assert!(mgr.start_round(config2).is_ok());
    }
}
