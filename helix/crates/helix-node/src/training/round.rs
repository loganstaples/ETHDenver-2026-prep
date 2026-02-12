//! Training Round State Machine.
//!
//! Implements a complete federated learning round with state transitions,
//! timeout handling, and participant tracking.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Unique identifier for a training round.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RoundId(pub u64);

impl RoundId {
    /// Creates a new round ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the next round ID.
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for RoundId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Round({})", self.0)
    }
}

/// State of a training round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    /// Round is initializing, waiting for participants to join.
    Initializing,
    /// Collecting gradient submissions from participants.
    Collecting,
    /// Aggregating collected gradients.
    Aggregating,
    /// Committing aggregated gradient to chain.
    Committing,
    /// Round completed successfully.
    Completed,
    /// Round failed due to timeout or insufficient participation.
    Failed(RoundFailure),
}

/// Reasons for round failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundFailure {
    /// Timed out waiting for gradients.
    CollectionTimeout,
    /// Timed out during aggregation.
    AggregationTimeout,
    /// Timed out committing to chain.
    CommitTimeout,
    /// Insufficient participants.
    InsufficientParticipants,
    /// Invalid aggregation (e.g., too many Byzantine nodes).
    InvalidAggregation,
    /// On-chain transaction failed.
    TransactionFailed,
}

/// Configuration for a training round.
#[derive(Debug, Clone)]
pub struct RoundConfig {
    /// Minimum number of participants required.
    pub min_participants: usize,
    /// Maximum number of participants allowed.
    pub max_participants: usize,
    /// Timeout for the collection phase.
    pub collection_timeout: Duration,
    /// Timeout for the aggregation phase.
    pub aggregation_timeout: Duration,
    /// Timeout for the commit phase.
    pub commit_timeout: Duration,
    /// Minimum stake required to participate.
    pub min_stake: u64,
    /// Whether to use Byzantine-fault-tolerant aggregation.
    pub byzantine_tolerant: bool,
    /// Maximum fraction of Byzantine nodes to tolerate (e.g., 0.33).
    pub max_byzantine_fraction: f64,
}

impl Default for RoundConfig {
    fn default() -> Self {
        Self {
            min_participants: 3,
            max_participants: 100,
            collection_timeout: Duration::from_secs(300), // 5 minutes
            aggregation_timeout: Duration::from_secs(60),  // 1 minute
            commit_timeout: Duration::from_secs(120),      // 2 minutes
            min_stake: 1000,
            byzantine_tolerant: true,
            max_byzantine_fraction: 0.33,
        }
    }
}

/// A participant in a training round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Unique identifier (typically Ethereum address).
    pub id: String,
    /// Stake amount (determines voting weight).
    pub stake: u64,
    /// Whether gradient has been submitted.
    pub submitted: bool,
    /// Submission timestamp.
    pub submission_time: Option<u64>,
    /// Hash of submitted gradient.
    pub gradient_hash: Option<[u8; 32]>,
    /// Proof of valid gradient computation.
    pub proof: Option<Vec<u8>>,
}

impl Participant {
    /// Creates a new participant.
    pub fn new(id: String, stake: u64) -> Self {
        Self {
            id,
            stake,
            submitted: false,
            submission_time: None,
            gradient_hash: None,
            proof: None,
        }
    }

    /// Marks the participant as having submitted.
    pub fn mark_submitted(&mut self, gradient_hash: [u8; 32], proof: Vec<u8>, timestamp: u64) {
        self.submitted = true;
        self.submission_time = Some(timestamp);
        self.gradient_hash = Some(gradient_hash);
        self.proof = Some(proof);
    }
}

/// A gradient submission from a participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientSubmission {
    /// Participant ID.
    pub participant_id: String,
    /// Hash of the gradient.
    pub gradient_hash: [u8; 32],
    /// Serialized gradient data.
    pub gradient_data: Vec<u8>,
    /// ZK proof of valid computation.
    pub proof: Vec<u8>,
    /// Public inputs accompanying the proof (8 × 32-byte big-endian field elements).
    /// Layout: [old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, loss, error_bound, step_number, error_checksum].
    /// Required for proof aggregation via RLCAggregationProver.
    #[serde(default)]
    pub public_inputs: Vec<[u8; 32]>,
    /// Claimed error bound.
    pub error_bound: f64,
    /// Timestamp of submission.
    pub timestamp: u64,
}

/// Result of aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// Hash of the aggregated gradient.
    pub aggregated_hash: [u8; 32],
    /// Serialized aggregated gradient.
    pub aggregated_gradient: Vec<u8>,
    /// Combined error bound.
    pub combined_error_bound: f64,
    /// Total stake of contributors.
    pub total_stake: u64,
    /// Number of contributors included.
    pub num_contributors: usize,
    /// IDs of excluded participants (Byzantine detection).
    pub excluded_participants: Vec<String>,
    /// Aggregated ZK proof bytes (single proof covering all participants' proofs).
    /// When present, this is a real KZG proof from the SHPLONKAggregationCircuit
    /// with 8 public inputs matching the on-chain contract interface.
    #[serde(default)]
    pub aggregated_proof: Vec<u8>,
    /// Public inputs for the aggregated proof (8 × 32 bytes when present).
    #[serde(default)]
    pub aggregated_proof_public_inputs: Vec<[u8; 32]>,
}

/// A training round.
#[derive(Debug)]
pub struct TrainingRound {
    /// Round identifier.
    pub id: RoundId,
    /// Current state.
    pub state: RoundState,
    /// Round configuration.
    pub config: RoundConfig,
    /// Registered participants.
    pub participants: HashMap<String, Participant>,
    /// Received gradient submissions.
    pub submissions: HashMap<String, GradientSubmission>,
    /// Aggregation result (set after aggregation).
    pub aggregation_result: Option<AggregationResult>,
    /// Model commitment at round start.
    pub initial_commitment: [u8; 32],
    /// Model commitment after update.
    pub final_commitment: Option<[u8; 32]>,
    /// Time when this round started.
    pub started_at: Instant,
    /// Time when current phase started.
    pub phase_started_at: Instant,
    /// History of state transitions.
    pub state_history: Vec<(RoundState, Instant)>,
}

impl TrainingRound {
    /// Creates a new training round.
    pub fn new(id: RoundId, config: RoundConfig, initial_commitment: [u8; 32]) -> Self {
        let now = Instant::now();
        Self {
            id,
            state: RoundState::Initializing,
            config,
            participants: HashMap::new(),
            submissions: HashMap::new(),
            aggregation_result: None,
            initial_commitment,
            final_commitment: None,
            started_at: now,
            phase_started_at: now,
            state_history: vec![(RoundState::Initializing, now)],
        }
    }

    /// Registers a participant for this round.
    pub fn register_participant(&mut self, id: String, stake: u64) -> Result<(), RoundError> {
        if self.state != RoundState::Initializing {
            return Err(RoundError::WrongState {
                expected: RoundState::Initializing,
                actual: self.state,
            });
        }

        if stake < self.config.min_stake {
            return Err(RoundError::InsufficientStake {
                required: self.config.min_stake,
                provided: stake,
            });
        }

        if self.participants.len() >= self.config.max_participants {
            return Err(RoundError::MaxParticipantsReached);
        }

        if self.participants.contains_key(&id) {
            return Err(RoundError::AlreadyRegistered(id));
        }

        self.participants.insert(id.clone(), Participant::new(id, stake));
        Ok(())
    }

    /// Transitions to the collection phase.
    pub fn start_collection(&mut self) -> Result<(), RoundError> {
        if self.state != RoundState::Initializing {
            return Err(RoundError::WrongState {
                expected: RoundState::Initializing,
                actual: self.state,
            });
        }

        if self.participants.len() < self.config.min_participants {
            return Err(RoundError::InsufficientParticipants {
                required: self.config.min_participants,
                registered: self.participants.len(),
            });
        }

        self.transition_to(RoundState::Collecting);
        Ok(())
    }

    /// Submits a gradient from a participant.
    pub fn submit_gradient(&mut self, submission: GradientSubmission) -> Result<(), RoundError> {
        if self.state != RoundState::Collecting {
            return Err(RoundError::WrongState {
                expected: RoundState::Collecting,
                actual: self.state,
            });
        }

        let participant = self.participants.get_mut(&submission.participant_id)
            .ok_or_else(|| RoundError::UnknownParticipant(submission.participant_id.clone()))?;

        if participant.submitted {
            return Err(RoundError::AlreadySubmitted(submission.participant_id.clone()));
        }

        participant.mark_submitted(
            submission.gradient_hash,
            submission.proof.clone(),
            submission.timestamp,
        );

        self.submissions.insert(submission.participant_id.clone(), submission);
        Ok(())
    }

    /// Transitions to the aggregation phase.
    pub fn start_aggregation(&mut self) -> Result<(), RoundError> {
        if self.state != RoundState::Collecting {
            return Err(RoundError::WrongState {
                expected: RoundState::Collecting,
                actual: self.state,
            });
        }

        if self.submissions.len() < self.config.min_participants {
            return Err(RoundError::InsufficientSubmissions {
                required: self.config.min_participants,
                received: self.submissions.len(),
            });
        }

        self.transition_to(RoundState::Aggregating);
        Ok(())
    }

    /// Sets the aggregation result.
    pub fn set_aggregation_result(&mut self, result: AggregationResult) -> Result<(), RoundError> {
        if self.state != RoundState::Aggregating {
            return Err(RoundError::WrongState {
                expected: RoundState::Aggregating,
                actual: self.state,
            });
        }

        self.aggregation_result = Some(result);
        self.transition_to(RoundState::Committing);
        Ok(())
    }

    /// Marks the round as completed with the new commitment.
    pub fn complete(&mut self, final_commitment: [u8; 32]) -> Result<(), RoundError> {
        if self.state != RoundState::Committing {
            return Err(RoundError::WrongState {
                expected: RoundState::Committing,
                actual: self.state,
            });
        }

        self.final_commitment = Some(final_commitment);
        self.transition_to(RoundState::Completed);
        Ok(())
    }

    /// Marks the round as failed.
    pub fn fail(&mut self, reason: RoundFailure) {
        self.transition_to(RoundState::Failed(reason));
    }

    /// Checks for timeouts and updates state accordingly.
    pub fn check_timeouts(&mut self) -> Option<RoundFailure> {
        let elapsed = self.phase_started_at.elapsed();

        match self.state {
            RoundState::Collecting if elapsed > self.config.collection_timeout => {
                let failure = RoundFailure::CollectionTimeout;
                self.fail(failure);
                Some(failure)
            }
            RoundState::Aggregating if elapsed > self.config.aggregation_timeout => {
                let failure = RoundFailure::AggregationTimeout;
                self.fail(failure);
                Some(failure)
            }
            RoundState::Committing if elapsed > self.config.commit_timeout => {
                let failure = RoundFailure::CommitTimeout;
                self.fail(failure);
                Some(failure)
            }
            _ => None,
        }
    }

    /// Returns the number of submitted gradients.
    pub fn submission_count(&self) -> usize {
        self.submissions.len()
    }

    /// Returns the total stake of participants who have submitted.
    pub fn submitted_stake(&self) -> u64 {
        self.participants
            .values()
            .filter(|p| p.submitted)
            .map(|p| p.stake)
            .sum()
    }

    /// Returns the total stake of all participants.
    pub fn total_stake(&self) -> u64 {
        self.participants.values().map(|p| p.stake).sum()
    }

    /// Returns the fraction of stake that has submitted.
    pub fn submission_progress(&self) -> f64 {
        let total = self.total_stake();
        if total == 0 {
            return 0.0;
        }
        self.submitted_stake() as f64 / total as f64
    }

    /// Returns whether the round is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, RoundState::Completed | RoundState::Failed(_))
    }

    /// Returns the duration of the round so far.
    pub fn duration(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns the duration of the current phase.
    pub fn phase_duration(&self) -> Duration {
        self.phase_started_at.elapsed()
    }

    /// Internal: transitions to a new state.
    fn transition_to(&mut self, new_state: RoundState) {
        let now = Instant::now();
        self.state = new_state;
        self.phase_started_at = now;
        self.state_history.push((new_state, now));
    }
}

/// Errors that can occur during round operations.
#[derive(Debug, Clone)]
pub enum RoundError {
    /// Wrong state for the operation.
    WrongState {
        expected: RoundState,
        actual: RoundState,
    },
    /// Insufficient stake to participate.
    InsufficientStake {
        required: u64,
        provided: u64,
    },
    /// Maximum participants reached.
    MaxParticipantsReached,
    /// Participant already registered.
    AlreadyRegistered(String),
    /// Unknown participant.
    UnknownParticipant(String),
    /// Gradient already submitted.
    AlreadySubmitted(String),
    /// Insufficient participants to start.
    InsufficientParticipants {
        required: usize,
        registered: usize,
    },
    /// Insufficient submissions to aggregate.
    InsufficientSubmissions {
        required: usize,
        received: usize,
    },
}

impl std::fmt::Display for RoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongState { expected, actual } => {
                write!(f, "Wrong state: expected {:?}, got {:?}", expected, actual)
            }
            Self::InsufficientStake { required, provided } => {
                write!(f, "Insufficient stake: required {}, provided {}", required, provided)
            }
            Self::MaxParticipantsReached => write!(f, "Maximum participants reached"),
            Self::AlreadyRegistered(id) => write!(f, "Participant already registered: {}", id),
            Self::UnknownParticipant(id) => write!(f, "Unknown participant: {}", id),
            Self::AlreadySubmitted(id) => write!(f, "Gradient already submitted by: {}", id),
            Self::InsufficientParticipants { required, registered } => {
                write!(f, "Insufficient participants: required {}, registered {}", required, registered)
            }
            Self::InsufficientSubmissions { required, received } => {
                write!(f, "Insufficient submissions: required {}, received {}", required, received)
            }
        }
    }
}

impl std::error::Error for RoundError {}

/// Manages multiple training rounds.
#[derive(Debug)]
pub struct RoundManager {
    /// Configuration for new rounds.
    pub config: RoundConfig,
    /// Current active round.
    pub current_round: Option<TrainingRound>,
    /// History of completed rounds.
    pub completed_rounds: Vec<RoundSummary>,
    /// Next round ID.
    next_round_id: u64,
}

/// Summary of a completed round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundSummary {
    /// Round ID.
    pub id: RoundId,
    /// Final state.
    pub final_state: RoundState,
    /// Duration of the round.
    pub duration_secs: u64,
    /// Number of participants.
    pub num_participants: usize,
    /// Number of submissions.
    pub num_submissions: usize,
    /// Total stake that participated.
    pub total_stake: u64,
    /// Initial model commitment.
    pub initial_commitment: [u8; 32],
    /// Final model commitment (if successful).
    pub final_commitment: Option<[u8; 32]>,
}

impl RoundManager {
    /// Creates a new round manager.
    pub fn new(config: RoundConfig) -> Self {
        Self {
            config,
            current_round: None,
            completed_rounds: Vec::new(),
            next_round_id: 0,
        }
    }

    /// Starts a new training round.
    pub fn start_round(&mut self, initial_commitment: [u8; 32]) -> Result<RoundId, RoundError> {
        if self.current_round.as_ref().map_or(false, |r| !r.is_terminal()) {
            // There's an active round - can't start a new one
            return Err(RoundError::WrongState {
                expected: RoundState::Completed,
                actual: self.current_round.as_ref().unwrap().state,
            });
        }

        // Archive the old round if present
        if let Some(old_round) = self.current_round.take() {
            self.archive_round(old_round);
        }

        let round_id = RoundId::new(self.next_round_id);
        self.next_round_id += 1;

        self.current_round = Some(TrainingRound::new(
            round_id,
            self.config.clone(),
            initial_commitment,
        ));

        Ok(round_id)
    }

    /// Returns a reference to the current round.
    pub fn current(&self) -> Option<&TrainingRound> {
        self.current_round.as_ref()
    }

    /// Returns a mutable reference to the current round.
    pub fn current_mut(&mut self) -> Option<&mut TrainingRound> {
        self.current_round.as_mut()
    }

    /// Archives a completed round.
    fn archive_round(&mut self, round: TrainingRound) {
        let summary = RoundSummary {
            id: round.id,
            final_state: round.state,
            duration_secs: round.duration().as_secs(),
            num_participants: round.participants.len(),
            num_submissions: round.submissions.len(),
            total_stake: round.total_stake(),
            initial_commitment: round.initial_commitment,
            final_commitment: round.final_commitment,
        };
        self.completed_rounds.push(summary);
    }

    /// Returns the number of completed rounds.
    pub fn completed_count(&self) -> usize {
        self.completed_rounds.len()
    }

    /// Returns the success rate of rounds.
    pub fn success_rate(&self) -> f64 {
        if self.completed_rounds.is_empty() {
            return 0.0;
        }
        let successes = self.completed_rounds
            .iter()
            .filter(|r| r.final_state == RoundState::Completed)
            .count();
        successes as f64 / self.completed_rounds.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_lifecycle() {
        let config = RoundConfig {
            min_participants: 2,
            ..Default::default()
        };
        let mut round = TrainingRound::new(
            RoundId::new(0),
            config,
            [0u8; 32],
        );

        // Register participants
        round.register_participant("node1".to_string(), 1000).unwrap();
        round.register_participant("node2".to_string(), 2000).unwrap();

        // Start collection
        round.start_collection().unwrap();
        assert_eq!(round.state, RoundState::Collecting);

        // Submit gradients
        round.submit_gradient(GradientSubmission {
            participant_id: "node1".to_string(),
            gradient_hash: [1u8; 32],
            gradient_data: vec![1, 2, 3],
            proof: vec![4, 5, 6],
            public_inputs: vec![],
            error_bound: 0.01,
            timestamp: 1000,
        }).unwrap();

        round.submit_gradient(GradientSubmission {
            participant_id: "node2".to_string(),
            gradient_hash: [2u8; 32],
            gradient_data: vec![7, 8, 9],
            proof: vec![10, 11, 12],
            public_inputs: vec![],
            error_bound: 0.02,
            timestamp: 1001,
        }).unwrap();

        // Start aggregation
        round.start_aggregation().unwrap();
        assert_eq!(round.state, RoundState::Aggregating);

        // Set aggregation result
        round.set_aggregation_result(AggregationResult {
            aggregated_hash: [3u8; 32],
            aggregated_gradient: vec![13, 14, 15],
            combined_error_bound: 0.015,
            total_stake: 3000,
            num_contributors: 2,
            excluded_participants: vec![],
            aggregated_proof: vec![],
            aggregated_proof_public_inputs: vec![],
        }).unwrap();
        assert_eq!(round.state, RoundState::Committing);

        // Complete
        round.complete([4u8; 32]).unwrap();
        assert_eq!(round.state, RoundState::Completed);
        assert!(round.is_terminal());
    }

    #[test]
    fn test_round_manager() {
        let config = RoundConfig {
            min_participants: 1,
            ..Default::default()
        };
        let mut manager = RoundManager::new(config);

        let round_id = manager.start_round([0u8; 32]).unwrap();
        assert_eq!(round_id.0, 0);

        // Complete the round
        let round = manager.current_mut().unwrap();
        round.register_participant("node1".to_string(), 1000).unwrap();
        round.start_collection().unwrap();
        round.submit_gradient(GradientSubmission {
            participant_id: "node1".to_string(),
            gradient_hash: [1u8; 32],
            gradient_data: vec![],
            proof: vec![],
            public_inputs: vec![],
            error_bound: 0.01,
            timestamp: 0,
        }).unwrap();
        round.start_aggregation().unwrap();
        round.set_aggregation_result(AggregationResult {
            aggregated_hash: [2u8; 32],
            aggregated_gradient: vec![],
            combined_error_bound: 0.01,
            total_stake: 1000,
            num_contributors: 1,
            excluded_participants: vec![],
            aggregated_proof: vec![],
            aggregated_proof_public_inputs: vec![],
        }).unwrap();
        round.complete([3u8; 32]).unwrap();

        // Start next round
        let round_id_2 = manager.start_round([3u8; 32]).unwrap();
        assert_eq!(round_id_2.0, 1);
        assert_eq!(manager.completed_count(), 1);
    }
}
