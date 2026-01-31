//! Distributed Training Round State Machine.
//!
//! Implements a comprehensive state machine for distributed training rounds with:
//! - Well-defined state transitions with guards
//! - Timeout handling per state
//! - Event emission for observability
//! - Support for graceful degradation and recovery

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;

use crate::network::messages::PeerId;

/// Unique identifier for a distributed training round.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct DistributedRoundId(pub u64);

impl DistributedRoundId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for DistributedRoundId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DRound({})", self.0)
    }
}

/// State of a distributed training round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedRoundState {
    /// Initial state - round created but not started.
    Init,
    /// Waiting for minimum workers to join.
    WaitingForWorkers,
    /// Distributing model shares to workers.
    Distributing,
    /// Workers are computing gradients.
    Computing,
    /// Collecting gradient shares from workers.
    Collecting,
    /// Aggregating collected gradients.
    Aggregating,
    /// Committing result on-chain.
    Committing,
    /// Round completed successfully.
    Completed,
    /// Round failed.
    Failed(RoundFailureReason),
}

impl DistributedRoundState {
    /// Returns whether this is a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_))
    }

    /// Returns whether the round is in an active state.
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Init | Self::Completed | Self::Failed(_))
    }

    /// Returns the default timeout for this state.
    pub fn default_timeout(&self) -> Duration {
        match self {
            Self::Init => Duration::from_secs(60),
            Self::WaitingForWorkers => Duration::from_secs(120),
            Self::Distributing => Duration::from_secs(60),
            Self::Computing => Duration::from_secs(300),
            Self::Collecting => Duration::from_secs(120),
            Self::Aggregating => Duration::from_secs(60),
            Self::Committing => Duration::from_secs(120),
            Self::Completed | Self::Failed(_) => Duration::from_secs(0),
        }
    }

    /// Returns the next expected state in the happy path.
    pub fn next_state(&self) -> Option<DistributedRoundState> {
        match self {
            Self::Init => Some(Self::WaitingForWorkers),
            Self::WaitingForWorkers => Some(Self::Distributing),
            Self::Distributing => Some(Self::Computing),
            Self::Computing => Some(Self::Collecting),
            Self::Collecting => Some(Self::Aggregating),
            Self::Aggregating => Some(Self::Committing),
            Self::Committing => Some(Self::Completed),
            Self::Completed | Self::Failed(_) => None,
        }
    }
}

/// Reasons for round failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundFailureReason {
    /// Timeout waiting for workers.
    WorkerTimeout,
    /// Not enough workers joined.
    InsufficientWorkers,
    /// Share distribution failed.
    DistributionFailed,
    /// Computation timeout.
    ComputationTimeout,
    /// Collection timeout.
    CollectionTimeout,
    /// Too many workers failed during computation.
    TooManyFailures,
    /// Aggregation error.
    AggregationError,
    /// On-chain commit failed.
    CommitFailed,
    /// Leader failure.
    LeaderFailed,
    /// Cancelled by coordinator.
    Cancelled,
}

/// Worker participation state within a round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerRoundState {
    /// Worker has joined but not received shares.
    Joined,
    /// Worker has received model shares.
    SharesReceived,
    /// Worker is computing.
    Computing,
    /// Worker has submitted gradient.
    Submitted,
    /// Worker failed or timed out.
    Failed,
    /// Worker left the round.
    Left,
}

/// Worker information within a round.
#[derive(Debug, Clone)]
pub struct RoundWorker {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Worker's state in this round.
    pub state: WorkerRoundState,
    /// Worker's stake.
    pub stake: u64,
    /// Shard index assigned to this worker.
    pub shard_index: u32,
    /// Time worker joined.
    pub joined_at: Instant,
    /// Last activity time.
    pub last_activity: Instant,
    /// Gradient commitment (if submitted).
    pub gradient_commitment: Option<[u8; 32]>,
    /// Error bound (if submitted).
    pub error_bound: Option<f64>,
}

impl RoundWorker {
    pub fn new(peer_id: PeerId, stake: u64, shard_index: u32) -> Self {
        let now = Instant::now();
        Self {
            peer_id,
            state: WorkerRoundState::Joined,
            stake,
            shard_index,
            joined_at: now,
            last_activity: now,
            gradient_commitment: None,
            error_bound: None,
        }
    }

    pub fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            WorkerRoundState::Joined
                | WorkerRoundState::SharesReceived
                | WorkerRoundState::Computing
                | WorkerRoundState::Submitted
        )
    }
}

/// Event emitted by the state machine.
#[derive(Debug, Clone)]
pub enum StateMachineEvent {
    /// State changed.
    StateChanged {
        round_id: DistributedRoundId,
        from: DistributedRoundState,
        to: DistributedRoundState,
    },
    /// Worker joined.
    WorkerJoined {
        round_id: DistributedRoundId,
        peer_id: PeerId,
        worker_count: usize,
    },
    /// Worker left or failed.
    WorkerLeft {
        round_id: DistributedRoundId,
        peer_id: PeerId,
        reason: String,
    },
    /// Shares distributed to all workers.
    SharesDistributed {
        round_id: DistributedRoundId,
        worker_count: usize,
    },
    /// Gradient received from worker.
    GradientReceived {
        round_id: DistributedRoundId,
        peer_id: PeerId,
        received_count: usize,
        expected_count: usize,
    },
    /// Aggregation completed.
    AggregationCompleted {
        round_id: DistributedRoundId,
        included_count: usize,
        excluded_count: usize,
        aggregated_commitment: [u8; 32],
    },
    /// Round committed on-chain.
    Committed {
        round_id: DistributedRoundId,
        tx_hash: [u8; 32],
        new_model_commitment: [u8; 32],
    },
    /// Timeout occurred.
    Timeout {
        round_id: DistributedRoundId,
        state: DistributedRoundState,
    },
    /// Round completed.
    RoundCompleted {
        round_id: DistributedRoundId,
        duration: Duration,
        workers_participated: usize,
    },
    /// Round failed.
    RoundFailed {
        round_id: DistributedRoundId,
        reason: RoundFailureReason,
        message: String,
    },
}

/// Configuration for the state machine.
#[derive(Debug, Clone)]
pub struct StateMachineConfig {
    /// Minimum workers required to start.
    pub min_workers: usize,
    /// Maximum workers allowed.
    pub max_workers: usize,
    /// Minimum submission fraction to proceed.
    pub min_submission_fraction: f64,
    /// Timeout overrides per state.
    pub state_timeouts: HashMap<String, Duration>,
    /// Worker inactivity timeout.
    pub worker_inactivity_timeout: Duration,
    /// Allow graceful degradation (continue with fewer workers).
    pub allow_degradation: bool,
    /// Minimum workers for degraded mode.
    pub min_degraded_workers: usize,
}

impl Default for StateMachineConfig {
    fn default() -> Self {
        Self {
            min_workers: 3,
            max_workers: 100,
            min_submission_fraction: 0.67,
            state_timeouts: HashMap::new(),
            worker_inactivity_timeout: Duration::from_secs(60),
            allow_degradation: true,
            min_degraded_workers: 2,
        }
    }
}

/// A distributed training round managed by the state machine.
#[derive(Debug)]
pub struct DistributedRound {
    /// Round identifier.
    pub id: DistributedRoundId,
    /// Current state.
    pub state: DistributedRoundState,
    /// Configuration.
    pub config: StateMachineConfig,
    /// Workers in this round.
    pub workers: HashMap<PeerId, RoundWorker>,
    /// Model commitment at round start.
    pub initial_commitment: [u8; 32],
    /// Aggregated gradient commitment (after aggregation).
    pub aggregated_commitment: Option<[u8; 32]>,
    /// New model commitment (after update).
    pub new_commitment: Option<[u8; 32]>,
    /// On-chain transaction hash (after commit).
    pub tx_hash: Option<[u8; 32]>,
    /// Round creation time.
    pub created_at: Instant,
    /// Current state entry time.
    pub state_entered_at: Instant,
    /// State transition history.
    pub state_history: Vec<(DistributedRoundState, Instant)>,
    /// Next shard index to assign.
    next_shard_index: u32,
}

impl DistributedRound {
    /// Creates a new distributed round.
    pub fn new(
        id: DistributedRoundId,
        config: StateMachineConfig,
        initial_commitment: [u8; 32],
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            state: DistributedRoundState::Init,
            config,
            workers: HashMap::new(),
            initial_commitment,
            aggregated_commitment: None,
            new_commitment: None,
            tx_hash: None,
            created_at: now,
            state_entered_at: now,
            state_history: vec![(DistributedRoundState::Init, now)],
            next_shard_index: 0,
        }
    }

    /// Returns the timeout for the current state.
    pub fn current_timeout(&self) -> Duration {
        let state_name = format!("{:?}", self.state);
        self.config
            .state_timeouts
            .get(&state_name)
            .copied()
            .unwrap_or_else(|| self.state.default_timeout())
    }

    /// Checks if the current state has timed out.
    pub fn is_timed_out(&self) -> bool {
        let timeout = self.current_timeout();
        timeout > Duration::ZERO && self.state_entered_at.elapsed() > timeout
    }

    /// Returns the duration spent in the current state.
    pub fn time_in_state(&self) -> Duration {
        self.state_entered_at.elapsed()
    }

    /// Returns the total round duration.
    pub fn duration(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Returns the number of active workers.
    pub fn active_worker_count(&self) -> usize {
        self.workers.values().filter(|w| w.is_active()).count()
    }

    /// Returns the number of workers who have submitted.
    pub fn submitted_count(&self) -> usize {
        self.workers
            .values()
            .filter(|w| w.state == WorkerRoundState::Submitted)
            .count()
    }

    /// Returns whether we have enough workers to proceed.
    pub fn has_min_workers(&self) -> bool {
        self.active_worker_count() >= self.config.min_workers
    }

    /// Returns whether we have enough submissions to proceed.
    pub fn has_min_submissions(&self) -> bool {
        let active = self.active_worker_count();
        if active == 0 {
            return false;
        }
        let submitted = self.submitted_count();
        let fraction = submitted as f64 / active as f64;
        fraction >= self.config.min_submission_fraction
    }

    /// Returns whether we can proceed in degraded mode.
    pub fn can_proceed_degraded(&self) -> bool {
        self.config.allow_degradation
            && self.active_worker_count() >= self.config.min_degraded_workers
    }

    /// Adds a worker to the round.
    pub fn add_worker(&mut self, peer_id: PeerId, stake: u64) -> Result<u32, String> {
        if self.workers.len() >= self.config.max_workers {
            return Err("Maximum workers reached".to_string());
        }

        if self.workers.contains_key(&peer_id) {
            return Err("Worker already in round".to_string());
        }

        if !matches!(
            self.state,
            DistributedRoundState::Init | DistributedRoundState::WaitingForWorkers
        ) {
            return Err(format!(
                "Cannot add worker in state {:?}",
                self.state
            ));
        }

        let shard_index = self.next_shard_index;
        self.next_shard_index += 1;

        let worker = RoundWorker::new(peer_id.clone(), stake, shard_index);
        self.workers.insert(peer_id, worker);

        Ok(shard_index)
    }

    /// Removes a worker from the round.
    pub fn remove_worker(&mut self, peer_id: &PeerId) -> Option<RoundWorker> {
        if let Some(mut worker) = self.workers.remove(peer_id) {
            worker.state = WorkerRoundState::Left;
            Some(worker)
        } else {
            None
        }
    }

    /// Marks a worker as failed.
    pub fn mark_worker_failed(&mut self, peer_id: &PeerId, reason: &str) {
        if let Some(worker) = self.workers.get_mut(peer_id) {
            worker.state = WorkerRoundState::Failed;
            worker.touch();
        }
    }

    /// Marks that a worker received their shares.
    pub fn mark_shares_received(&mut self, peer_id: &PeerId) -> Result<(), String> {
        if self.state != DistributedRoundState::Distributing {
            return Err(format!(
                "Cannot mark shares received in state {:?}",
                self.state
            ));
        }

        if let Some(worker) = self.workers.get_mut(peer_id) {
            if worker.state == WorkerRoundState::Joined {
                worker.state = WorkerRoundState::SharesReceived;
                worker.touch();
                Ok(())
            } else {
                Err(format!(
                    "Worker not in Joined state: {:?}",
                    worker.state
                ))
            }
        } else {
            Err("Worker not found".to_string())
        }
    }

    /// Marks that a worker started computing.
    pub fn mark_computing(&mut self, peer_id: &PeerId) -> Result<(), String> {
        if !matches!(
            self.state,
            DistributedRoundState::Computing | DistributedRoundState::Distributing
        ) {
            return Err(format!(
                "Cannot mark computing in state {:?}",
                self.state
            ));
        }

        if let Some(worker) = self.workers.get_mut(peer_id) {
            if matches!(
                worker.state,
                WorkerRoundState::SharesReceived | WorkerRoundState::Joined
            ) {
                worker.state = WorkerRoundState::Computing;
                worker.touch();
                Ok(())
            } else {
                Err(format!(
                    "Worker not in correct state: {:?}",
                    worker.state
                ))
            }
        } else {
            Err("Worker not found".to_string())
        }
    }

    /// Records a gradient submission from a worker.
    pub fn record_submission(
        &mut self,
        peer_id: &PeerId,
        commitment: [u8; 32],
        error_bound: f64,
    ) -> Result<(), String> {
        if !matches!(
            self.state,
            DistributedRoundState::Computing | DistributedRoundState::Collecting
        ) {
            return Err(format!(
                "Cannot record submission in state {:?}",
                self.state
            ));
        }

        if let Some(worker) = self.workers.get_mut(peer_id) {
            if !matches!(
                worker.state,
                WorkerRoundState::Computing | WorkerRoundState::SharesReceived
            ) {
                return Err(format!(
                    "Worker not in correct state: {:?}",
                    worker.state
                ));
            }

            worker.state = WorkerRoundState::Submitted;
            worker.gradient_commitment = Some(commitment);
            worker.error_bound = Some(error_bound);
            worker.touch();
            Ok(())
        } else {
            Err("Worker not found".to_string())
        }
    }

    /// Sets the aggregation result.
    pub fn set_aggregation_result(
        &mut self,
        commitment: [u8; 32],
        new_model_commitment: [u8; 32],
    ) -> Result<(), String> {
        if self.state != DistributedRoundState::Aggregating {
            return Err(format!(
                "Cannot set aggregation in state {:?}",
                self.state
            ));
        }

        self.aggregated_commitment = Some(commitment);
        self.new_commitment = Some(new_model_commitment);
        Ok(())
    }

    /// Sets the commit transaction hash.
    pub fn set_tx_hash(&mut self, hash: [u8; 32]) -> Result<(), String> {
        if self.state != DistributedRoundState::Committing {
            return Err(format!(
                "Cannot set tx hash in state {:?}",
                self.state
            ));
        }

        self.tx_hash = Some(hash);
        Ok(())
    }

    /// Transitions to a new state.
    fn transition_to(&mut self, new_state: DistributedRoundState) {
        let now = Instant::now();
        self.state = new_state;
        self.state_entered_at = now;
        self.state_history.push((new_state, now));
    }

    /// Returns workers who should be marked as inactive.
    pub fn check_inactive_workers(&self) -> Vec<PeerId> {
        let timeout = self.config.worker_inactivity_timeout;
        self.workers
            .iter()
            .filter(|(_, w)| {
                w.is_active()
                    && w.state != WorkerRoundState::Submitted
                    && w.last_activity.elapsed() > timeout
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
}

/// State machine for managing distributed training rounds.
pub struct DistributedTrainingStateMachine {
    /// Current round.
    current_round: Option<DistributedRound>,
    /// Completed rounds history.
    completed_rounds: Vec<RoundSummary>,
    /// Round counter.
    round_counter: u64,
    /// Event sender.
    event_tx: broadcast::Sender<StateMachineEvent>,
    /// Event receiver (for cloning).
    _event_rx: broadcast::Receiver<StateMachineEvent>,
    /// Default configuration.
    default_config: StateMachineConfig,
}

/// Summary of a completed round.
#[derive(Debug, Clone)]
pub struct RoundSummary {
    pub id: DistributedRoundId,
    pub final_state: DistributedRoundState,
    pub duration: Duration,
    pub workers_joined: usize,
    pub workers_submitted: usize,
    pub initial_commitment: [u8; 32],
    pub final_commitment: Option<[u8; 32]>,
    pub tx_hash: Option<[u8; 32]>,
}

impl DistributedTrainingStateMachine {
    /// Creates a new state machine.
    pub fn new(config: StateMachineConfig) -> Self {
        let (event_tx, event_rx) = broadcast::channel(1000);
        Self {
            current_round: None,
            completed_rounds: Vec::new(),
            round_counter: 0,
            event_tx,
            _event_rx: event_rx,
            default_config: config,
        }
    }

    /// Subscribes to state machine events.
    pub fn subscribe(&self) -> broadcast::Receiver<StateMachineEvent> {
        self.event_tx.subscribe()
    }

    /// Starts a new round.
    pub fn start_round(
        &mut self,
        initial_commitment: [u8; 32],
        config: Option<StateMachineConfig>,
    ) -> Result<DistributedRoundId, String> {
        if let Some(ref round) = self.current_round {
            if !round.state.is_terminal() {
                return Err(format!(
                    "Cannot start new round while round {} is in state {:?}",
                    round.id, round.state
                ));
            }
        }

        // Archive previous round
        if let Some(old_round) = self.current_round.take() {
            self.archive_round(old_round);
        }

        self.round_counter += 1;
        let round_id = DistributedRoundId::new(self.round_counter);
        let config = config.unwrap_or_else(|| self.default_config.clone());

        let mut round = DistributedRound::new(round_id, config, initial_commitment);

        // Transition to WaitingForWorkers
        let old_state = round.state;
        round.transition_to(DistributedRoundState::WaitingForWorkers);

        self.emit(StateMachineEvent::StateChanged {
            round_id,
            from: old_state,
            to: round.state,
        });

        self.current_round = Some(round);
        Ok(round_id)
    }

    /// Adds a worker to the current round.
    pub fn add_worker(&mut self, peer_id: PeerId, stake: u64) -> Result<u32, String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        let shard_index = round.add_worker(peer_id.clone(), stake)?;
        let worker_count = round.active_worker_count();

        self.emit(StateMachineEvent::WorkerJoined {
            round_id: round.id,
            peer_id,
            worker_count,
        });

        Ok(shard_index)
    }

    /// Removes a worker from the current round.
    pub fn remove_worker(&mut self, peer_id: &PeerId, reason: &str) -> Result<(), String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;
        let round_id = round.id;

        if round.remove_worker(peer_id).is_some() {
            self.emit(StateMachineEvent::WorkerLeft {
                round_id,
                peer_id: peer_id.clone(),
                reason: reason.to_string(),
            });
        }

        Ok(())
    }

    /// Attempts to advance to Distributing state.
    pub fn try_start_distribution(&mut self) -> Result<bool, String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        if round.state != DistributedRoundState::WaitingForWorkers {
            return Ok(false);
        }

        if !round.has_min_workers() {
            return Ok(false);
        }

        let old_state = round.state;
        round.transition_to(DistributedRoundState::Distributing);

        self.emit(StateMachineEvent::StateChanged {
            round_id: round.id,
            from: old_state,
            to: round.state,
        });

        Ok(true)
    }

    /// Marks that shares have been distributed to all workers.
    pub fn mark_distribution_complete(&mut self) -> Result<(), String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        if round.state != DistributedRoundState::Distributing {
            return Err(format!(
                "Not in Distributing state: {:?}",
                round.state
            ));
        }

        let worker_count = round.active_worker_count();
        let old_state = round.state;
        round.transition_to(DistributedRoundState::Computing);

        self.emit(StateMachineEvent::SharesDistributed {
            round_id: round.id,
            worker_count,
        });

        self.emit(StateMachineEvent::StateChanged {
            round_id: round.id,
            from: old_state,
            to: round.state,
        });

        Ok(())
    }

    /// Records a gradient submission.
    pub fn record_submission(
        &mut self,
        peer_id: &PeerId,
        commitment: [u8; 32],
        error_bound: f64,
    ) -> Result<(), String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;
        let round_id = round.id;

        round.record_submission(peer_id, commitment, error_bound)?;

        let received = round.submitted_count();
        let expected = round.active_worker_count();

        self.emit(StateMachineEvent::GradientReceived {
            round_id,
            peer_id: peer_id.clone(),
            received_count: received,
            expected_count: expected,
        });

        Ok(())
    }

    /// Attempts to advance to Collecting state.
    pub fn try_start_collection(&mut self) -> Result<bool, String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        if round.state != DistributedRoundState::Computing {
            return Ok(false);
        }

        // Check if we have submissions or timeout
        if round.submitted_count() > 0
            && (round.has_min_submissions() || round.is_timed_out())
        {
            let old_state = round.state;
            round.transition_to(DistributedRoundState::Collecting);

            self.emit(StateMachineEvent::StateChanged {
                round_id: round.id,
                from: old_state,
                to: round.state,
            });

            return Ok(true);
        }

        Ok(false)
    }

    /// Attempts to advance to Aggregating state.
    pub fn try_start_aggregation(&mut self) -> Result<bool, String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        if round.state != DistributedRoundState::Collecting {
            return Ok(false);
        }

        if round.has_min_submissions() || round.is_timed_out() {
            let old_state = round.state;
            round.transition_to(DistributedRoundState::Aggregating);

            self.emit(StateMachineEvent::StateChanged {
                round_id: round.id,
                from: old_state,
                to: round.state,
            });

            return Ok(true);
        }

        Ok(false)
    }

    /// Records aggregation completion.
    pub fn record_aggregation(
        &mut self,
        aggregated_commitment: [u8; 32],
        new_model_commitment: [u8; 32],
        included_count: usize,
        excluded_count: usize,
    ) -> Result<(), String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        round.set_aggregation_result(aggregated_commitment, new_model_commitment)?;

        let old_state = round.state;
        round.transition_to(DistributedRoundState::Committing);

        self.emit(StateMachineEvent::AggregationCompleted {
            round_id: round.id,
            included_count,
            excluded_count,
            aggregated_commitment,
        });

        self.emit(StateMachineEvent::StateChanged {
            round_id: round.id,
            from: old_state,
            to: round.state,
        });

        Ok(())
    }

    /// Records successful on-chain commit.
    pub fn record_commit(&mut self, tx_hash: [u8; 32]) -> Result<(), String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        round.set_tx_hash(tx_hash)?;

        let new_commitment = round.new_commitment.unwrap_or([0u8; 32]);
        let old_state = round.state;
        let duration = round.duration();
        let workers_participated = round.submitted_count();

        round.transition_to(DistributedRoundState::Completed);

        self.emit(StateMachineEvent::Committed {
            round_id: round.id,
            tx_hash,
            new_model_commitment: new_commitment,
        });

        self.emit(StateMachineEvent::StateChanged {
            round_id: round.id,
            from: old_state,
            to: round.state,
        });

        self.emit(StateMachineEvent::RoundCompleted {
            round_id: round.id,
            duration,
            workers_participated,
        });

        Ok(())
    }

    /// Marks the round as failed.
    pub fn fail_round(&mut self, reason: RoundFailureReason, message: &str) -> Result<(), String> {
        let round = self.current_round.as_mut().ok_or("No active round")?;

        if round.state.is_terminal() {
            return Err("Round already in terminal state".to_string());
        }

        let old_state = round.state;
        round.transition_to(DistributedRoundState::Failed(reason));

        self.emit(StateMachineEvent::RoundFailed {
            round_id: round.id,
            reason,
            message: message.to_string(),
        });

        self.emit(StateMachineEvent::StateChanged {
            round_id: round.id,
            from: old_state,
            to: round.state,
        });

        Ok(())
    }

    /// Checks for timeouts and handles them.
    pub fn check_timeouts(&mut self) -> Option<RoundFailureReason> {
        let round = match self.current_round.as_mut() {
            Some(r) if !r.state.is_terminal() => r,
            _ => return None,
        };

        if !round.is_timed_out() {
            return None;
        }

        let round_id = round.id;
        let state = round.state;

        self.emit(StateMachineEvent::Timeout { round_id, state });

        // Determine failure reason based on state
        let reason = match state {
            DistributedRoundState::WaitingForWorkers => {
                if round.can_proceed_degraded() {
                    // Try to proceed in degraded mode
                    if let Ok(true) = self.try_start_distribution() {
                        return None;
                    }
                }
                RoundFailureReason::InsufficientWorkers
            }
            DistributedRoundState::Distributing => RoundFailureReason::DistributionFailed,
            DistributedRoundState::Computing => {
                if round.submitted_count() > 0 && round.can_proceed_degraded() {
                    // Try to collect what we have
                    if let Ok(true) = self.try_start_collection() {
                        return None;
                    }
                }
                RoundFailureReason::ComputationTimeout
            }
            DistributedRoundState::Collecting => {
                if round.submitted_count() > 0 {
                    // Try to aggregate what we have
                    if let Ok(true) = self.try_start_aggregation() {
                        return None;
                    }
                }
                RoundFailureReason::CollectionTimeout
            }
            DistributedRoundState::Aggregating => RoundFailureReason::AggregationError,
            DistributedRoundState::Committing => RoundFailureReason::CommitFailed,
            _ => return None,
        };

        let _ = self.fail_round(reason, &format!("Timeout in state {:?}", state));
        Some(reason)
    }

    /// Checks for inactive workers and marks them as failed.
    pub fn check_inactive_workers(&mut self) -> Vec<PeerId> {
        let round = match self.current_round.as_mut() {
            Some(r) if !r.state.is_terminal() => r,
            _ => return Vec::new(),
        };

        let inactive = round.check_inactive_workers();
        let round_id = round.id;

        for peer_id in &inactive {
            round.mark_worker_failed(peer_id, "Inactivity timeout");
            self.emit(StateMachineEvent::WorkerLeft {
                round_id,
                peer_id: peer_id.clone(),
                reason: "Inactivity timeout".to_string(),
            });
        }

        inactive
    }

    /// Returns the current round.
    pub fn current_round(&self) -> Option<&DistributedRound> {
        self.current_round.as_ref()
    }

    /// Returns mutable access to the current round.
    pub fn current_round_mut(&mut self) -> Option<&mut DistributedRound> {
        self.current_round.as_mut()
    }

    /// Returns the completed rounds.
    pub fn completed_rounds(&self) -> &[RoundSummary] {
        &self.completed_rounds
    }

    /// Archives a completed round.
    fn archive_round(&mut self, round: DistributedRound) {
        let summary = RoundSummary {
            id: round.id,
            final_state: round.state,
            duration: round.duration(),
            workers_joined: round.workers.len(),
            workers_submitted: round.submitted_count(),
            initial_commitment: round.initial_commitment,
            final_commitment: round.new_commitment,
            tx_hash: round.tx_hash,
        };
        self.completed_rounds.push(summary);

        // Keep only last 100 summaries
        if self.completed_rounds.len() > 100 {
            self.completed_rounds.remove(0);
        }
    }

    /// Emits an event.
    fn emit(&self, event: StateMachineEvent) {
        let _ = self.event_tx.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_peer_id(idx: usize) -> PeerId {
        PeerId::from_string(&format!("test-peer-{}", idx))
    }

    #[test]
    fn test_round_lifecycle() {
        let config = StateMachineConfig {
            min_workers: 2,
            ..Default::default()
        };
        let mut sm = DistributedTrainingStateMachine::new(config);

        // Start round
        let round_id = sm.start_round([1u8; 32], None).unwrap();
        assert_eq!(round_id.0, 1);

        // Add workers
        sm.add_worker(create_peer_id(0), 1000).unwrap();
        sm.add_worker(create_peer_id(1), 1000).unwrap();

        // Should be able to start distribution
        assert!(sm.try_start_distribution().unwrap());
        assert_eq!(
            sm.current_round().unwrap().state,
            DistributedRoundState::Distributing
        );

        // Mark distribution complete
        sm.mark_distribution_complete().unwrap();
        assert_eq!(
            sm.current_round().unwrap().state,
            DistributedRoundState::Computing
        );

        // Record submissions
        sm.record_submission(&create_peer_id(0), [2u8; 32], 0.01).unwrap();
        sm.record_submission(&create_peer_id(1), [3u8; 32], 0.02).unwrap();

        // Should be able to start collection
        assert!(sm.try_start_collection().unwrap());
        assert_eq!(
            sm.current_round().unwrap().state,
            DistributedRoundState::Collecting
        );

        // Should be able to start aggregation
        assert!(sm.try_start_aggregation().unwrap());
        assert_eq!(
            sm.current_round().unwrap().state,
            DistributedRoundState::Aggregating
        );

        // Record aggregation
        sm.record_aggregation([4u8; 32], [5u8; 32], 2, 0).unwrap();
        assert_eq!(
            sm.current_round().unwrap().state,
            DistributedRoundState::Committing
        );

        // Record commit
        sm.record_commit([6u8; 32]).unwrap();
        assert_eq!(
            sm.current_round().unwrap().state,
            DistributedRoundState::Completed
        );
    }

    #[test]
    fn test_worker_management() {
        let config = StateMachineConfig {
            min_workers: 2,
            max_workers: 5,
            ..Default::default()
        };
        let mut sm = DistributedTrainingStateMachine::new(config);

        sm.start_round([1u8; 32], None).unwrap();

        // Add workers
        let shard0 = sm.add_worker(create_peer_id(0), 1000).unwrap();
        let shard1 = sm.add_worker(create_peer_id(1), 2000).unwrap();

        assert_eq!(shard0, 0);
        assert_eq!(shard1, 1);
        assert_eq!(sm.current_round().unwrap().active_worker_count(), 2);

        // Remove worker
        sm.remove_worker(&create_peer_id(0), "Left voluntarily").unwrap();
        assert_eq!(sm.current_round().unwrap().active_worker_count(), 1);
    }

    #[test]
    fn test_failure_handling() {
        let config = StateMachineConfig {
            min_workers: 3,
            ..Default::default()
        };
        let mut sm = DistributedTrainingStateMachine::new(config);

        sm.start_round([1u8; 32], None).unwrap();

        // Only add 2 workers (below minimum)
        sm.add_worker(create_peer_id(0), 1000).unwrap();
        sm.add_worker(create_peer_id(1), 1000).unwrap();

        // Cannot start distribution
        assert!(!sm.try_start_distribution().unwrap());

        // Fail the round
        sm.fail_round(RoundFailureReason::InsufficientWorkers, "Not enough workers")
            .unwrap();

        assert!(sm.current_round().unwrap().state.is_terminal());
    }

    #[test]
    fn test_state_transitions() {
        let state = DistributedRoundState::Init;
        assert_eq!(state.next_state(), Some(DistributedRoundState::WaitingForWorkers));

        let state = DistributedRoundState::Computing;
        assert_eq!(state.next_state(), Some(DistributedRoundState::Collecting));

        let state = DistributedRoundState::Completed;
        assert_eq!(state.next_state(), None);
    }
}
