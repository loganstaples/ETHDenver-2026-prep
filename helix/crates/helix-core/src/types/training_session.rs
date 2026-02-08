//! Training session orchestration type.
//!
//! Tracks the full lifecycle of a distributed training run:
//! Register -> Share -> Train -> Prove -> Aggregate -> Commit
//!
//! A `TrainingSession` coordinates multiple participants across rounds,
//! ensuring correct state transitions and tracking error commitments
//! throughout the training process.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

use super::coordination::{NodeId, RoundDescriptor, SessionId, TrainingParams, TrainingStepReceipt};
use super::error_commitment::ErrorCommitmentTracker;

/// The phase of a training session lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SessionPhase {
    /// Model registered, waiting for participants to join.
    Registered,
    /// Data shards distributed, participants preparing.
    DataShared,
    /// Active training round in progress.
    Training,
    /// Proofs being generated for completed training steps.
    Proving,
    /// Proofs submitted, aggregating results.
    Aggregating,
    /// Round committed on-chain, ready for next round or finalization.
    Committed,
    /// Session completed successfully.
    Finalized,
    /// Session failed or was cancelled.
    Failed,
}

impl fmt::Display for SessionPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionPhase::Registered => write!(f, "Registered"),
            SessionPhase::DataShared => write!(f, "DataShared"),
            SessionPhase::Training => write!(f, "Training"),
            SessionPhase::Proving => write!(f, "Proving"),
            SessionPhase::Aggregating => write!(f, "Aggregating"),
            SessionPhase::Committed => write!(f, "Committed"),
            SessionPhase::Finalized => write!(f, "Finalized"),
            SessionPhase::Failed => write!(f, "Failed"),
        }
    }
}

impl SessionPhase {
    /// Returns the valid next phases from this phase.
    pub fn valid_transitions(&self) -> &[SessionPhase] {
        match self {
            SessionPhase::Registered => &[SessionPhase::DataShared, SessionPhase::Failed],
            SessionPhase::DataShared => &[SessionPhase::Training, SessionPhase::Failed],
            SessionPhase::Training => &[SessionPhase::Proving, SessionPhase::Failed],
            SessionPhase::Proving => &[SessionPhase::Aggregating, SessionPhase::Failed],
            SessionPhase::Aggregating => &[SessionPhase::Committed, SessionPhase::Failed],
            SessionPhase::Committed => &[
                SessionPhase::Training,    // next round
                SessionPhase::Finalized,   // done
                SessionPhase::Failed,
            ],
            SessionPhase::Finalized => &[],
            SessionPhase::Failed => &[],
        }
    }

    /// Returns true if this phase is terminal.
    pub fn is_terminal(&self) -> bool {
        matches!(self, SessionPhase::Finalized | SessionPhase::Failed)
    }
}

/// Tracks a participant's state within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantState {
    /// Node identity.
    pub node_id: NodeId,
    /// When the participant joined (unix timestamp).
    pub joined_at: u64,
    /// Number of rounds completed by this participant.
    pub rounds_completed: u64,
    /// Number of valid proofs submitted.
    pub proofs_submitted: u64,
    /// Whether the participant is currently active.
    pub active: bool,
    /// Accumulated stake for this session.
    pub stake_amount: u64,
}

/// Summary of a completed round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundSummary {
    /// Round number.
    pub round_number: u64,
    /// Round descriptor.
    pub descriptor: RoundDescriptor,
    /// Receipts from participants who completed this round.
    pub receipts: Vec<TrainingStepReceipt>,
    /// Weight commitment after this round.
    pub post_weight_commitment: [u8; 32],
    /// Error checksum after this round.
    pub error_checksum: u64,
    /// When this round was committed (unix timestamp).
    pub committed_at: u64,
    /// On-chain transaction hash, if submitted.
    pub tx_hash: Option<[u8; 32]>,
}

/// Configuration for a training session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Model identifier.
    pub model_id: [u8; 32],
    /// Training parameters.
    pub params: TrainingParams,
    /// Maximum number of rounds.
    pub max_rounds: u64,
    /// Minimum participants required per round.
    pub min_participants: usize,
    /// Maximum participants allowed.
    pub max_participants: usize,
    /// Round deadline duration in seconds.
    pub round_deadline_secs: u64,
    /// Whether to require on-chain proof submission.
    pub require_on_chain: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model_id: [0u8; 32],
            params: TrainingParams::default(),
            max_rounds: 100,
            min_participants: 1,
            max_participants: 32,
            round_deadline_secs: 300,
            require_on_chain: false,
        }
    }
}

/// Orchestrates the full lifecycle of a distributed training run.
///
/// State machine: Register -> Share -> Train -> Prove -> Aggregate -> Commit
/// After Commit, loops back to Train for the next round, or finalizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSession {
    /// Unique session identifier.
    pub session_id: SessionId,
    /// Session configuration.
    pub config: SessionConfig,
    /// Current phase.
    phase: SessionPhase,
    /// Phase transition history (phase, timestamp).
    phase_history: Vec<(SessionPhase, u64)>,
    /// Initiator node.
    pub initiator: NodeId,
    /// Participants and their state.
    participants: HashMap<[u8; 32], ParticipantState>,
    /// Current round number.
    current_round: u64,
    /// Completed round summaries.
    rounds: Vec<RoundSummary>,
    /// Current weight commitment.
    weight_commitment: [u8; 32],
    /// Error commitment tracker.
    error_tracker: ErrorCommitmentTracker,
    /// When the session was created (unix timestamp).
    pub created_at: u64,
    /// Pending receipts for the current round.
    pending_receipts: Vec<TrainingStepReceipt>,
    /// Failure reason if session failed.
    failure_reason: Option<String>,
}

impl TrainingSession {
    /// Creates a new training session.
    pub fn new(
        initiator: NodeId,
        config: SessionConfig,
        start_time: u64,
    ) -> Self {
        let session_id = SessionId::derive(&config.model_id, start_time, &initiator);
        let error_tracker = ErrorCommitmentTracker::with_history(
            config.model_id,
            config.params.error_budget,
        );

        let mut session = Self {
            session_id,
            config,
            phase: SessionPhase::Registered,
            phase_history: Vec::new(),
            initiator,
            participants: HashMap::new(),
            current_round: 0,
            rounds: Vec::new(),
            weight_commitment: [0u8; 32],
            error_tracker,
            created_at: start_time,
            pending_receipts: Vec::new(),
            failure_reason: None,
        };
        session.phase_history.push((SessionPhase::Registered, start_time));
        session
    }

    /// Returns the current phase.
    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    /// Returns the current round number.
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Returns the current weight commitment.
    pub fn weight_commitment(&self) -> [u8; 32] {
        self.weight_commitment
    }

    /// Returns the error commitment tracker.
    pub fn error_tracker(&self) -> &ErrorCommitmentTracker {
        &self.error_tracker
    }

    /// Returns mutable reference to error tracker.
    pub fn error_tracker_mut(&mut self) -> &mut ErrorCommitmentTracker {
        &mut self.error_tracker
    }

    /// Returns the number of participants.
    pub fn participant_count(&self) -> usize {
        self.participants.values().filter(|p| p.active).count()
    }

    /// Returns the completed rounds.
    pub fn rounds(&self) -> &[RoundSummary] {
        &self.rounds
    }

    /// Returns the failure reason, if any.
    pub fn failure_reason(&self) -> Option<&str> {
        self.failure_reason.as_deref()
    }

    /// Returns the phase transition history.
    pub fn phase_history(&self) -> &[(SessionPhase, u64)] {
        &self.phase_history
    }

    // === State transitions ===

    /// Transitions to a new phase with validation.
    fn transition_to(&mut self, new_phase: SessionPhase, timestamp: u64) -> Result<(), String> {
        let valid = self.phase.valid_transitions();
        if !valid.contains(&new_phase) {
            return Err(format!(
                "invalid transition: {} -> {} (valid: {:?})",
                self.phase, new_phase, valid
            ));
        }
        self.phase = new_phase;
        self.phase_history.push((new_phase, timestamp));
        Ok(())
    }

    /// Adds a participant to the session.
    /// Only valid in Registered or DataShared phases.
    pub fn add_participant(
        &mut self,
        node_id: NodeId,
        stake_amount: u64,
        timestamp: u64,
    ) -> Result<(), String> {
        if self.phase != SessionPhase::Registered && self.phase != SessionPhase::DataShared {
            return Err(format!(
                "cannot add participants in {} phase",
                self.phase
            ));
        }
        let active_count = self.participants.values().filter(|p| p.active).count();
        if active_count >= self.config.max_participants {
            return Err(format!(
                "max participants ({}) reached",
                self.config.max_participants
            ));
        }

        let key = *node_id.as_bytes();
        self.participants.insert(key, ParticipantState {
            node_id,
            joined_at: timestamp,
            rounds_completed: 0,
            proofs_submitted: 0,
            active: true,
            stake_amount,
        });
        Ok(())
    }

    /// Removes a participant from the session.
    pub fn remove_participant(&mut self, node_id: &NodeId) -> Result<(), String> {
        let key = node_id.as_bytes();
        match self.participants.get_mut(key as &[u8; 32]) {
            Some(p) => {
                p.active = false;
                Ok(())
            }
            None => Err("participant not found".to_string()),
        }
    }

    /// Transitions to DataShared phase.
    /// Requires at least `min_participants` active participants.
    pub fn share_data(&mut self, timestamp: u64) -> Result<(), String> {
        let active_count = self.participant_count();
        if active_count < self.config.min_participants {
            return Err(format!(
                "need at least {} participants, have {}",
                self.config.min_participants, active_count
            ));
        }
        self.transition_to(SessionPhase::DataShared, timestamp)
    }

    /// Starts a new training round.
    /// Transitions from DataShared or Committed to Training.
    pub fn start_training(
        &mut self,
        initial_weight_commitment: [u8; 32],
        timestamp: u64,
    ) -> Result<RoundDescriptor, String> {
        if self.phase != SessionPhase::DataShared && self.phase != SessionPhase::Committed {
            return Err(format!(
                "cannot start training in {} phase",
                self.phase
            ));
        }

        if self.current_round >= self.config.max_rounds {
            return Err(format!(
                "max rounds ({}) reached",
                self.config.max_rounds
            ));
        }

        // Set weight commitment for this round
        if self.phase == SessionPhase::DataShared {
            self.weight_commitment = initial_weight_commitment;
        }

        let active_participants: Vec<NodeId> = self.participants
            .values()
            .filter(|p| p.active)
            .map(|p| p.node_id)
            .collect();

        let descriptor = RoundDescriptor {
            model_id: self.config.model_id,
            round_number: self.current_round,
            participants: active_participants,
            deadline: timestamp + self.config.round_deadline_secs,
            params: self.config.params.clone(),
            session_id: self.session_id,
            weight_commitment: self.weight_commitment,
        };

        self.pending_receipts.clear();
        self.transition_to(SessionPhase::Training, timestamp)?;
        Ok(descriptor)
    }

    /// Completes the training phase and transitions to proving.
    pub fn finish_training(&mut self, timestamp: u64) -> Result<(), String> {
        self.transition_to(SessionPhase::Proving, timestamp)
    }

    /// Submits a proof receipt for the current round.
    pub fn submit_receipt(
        &mut self,
        receipt: TrainingStepReceipt,
    ) -> Result<(), String> {
        if self.phase != SessionPhase::Proving {
            return Err(format!(
                "cannot submit receipts in {} phase",
                self.phase
            ));
        }

        // Validate receipt matches session
        if receipt.session_id != self.session_id {
            return Err("receipt session_id mismatch".to_string());
        }
        if receipt.round_number != self.current_round {
            return Err(format!(
                "receipt round {} does not match current round {}",
                receipt.round_number, self.current_round
            ));
        }

        // Validate participant is active
        let key = receipt.node_id.as_bytes();
        match self.participants.get(key) {
            Some(p) if p.active => {}
            _ => return Err("receipt from unknown or inactive participant".to_string()),
        }

        self.pending_receipts.push(receipt);
        Ok(())
    }

    /// Transitions to aggregation phase.
    pub fn start_aggregation(&mut self, timestamp: u64) -> Result<(), String> {
        if self.pending_receipts.is_empty() {
            return Err("no receipts to aggregate".to_string());
        }
        self.transition_to(SessionPhase::Aggregating, timestamp)
    }

    /// Commits the round results and transitions to Committed.
    ///
    /// `new_weight_commitment` is the post-aggregation weight hash.
    /// `step_error` is the error accumulated in this round.
    pub fn commit_round(
        &mut self,
        new_weight_commitment: [u8; 32],
        step_error: f64,
        timestamp: u64,
        tx_hash: Option<[u8; 32]>,
    ) -> Result<(), String> {
        if self.phase != SessionPhase::Aggregating {
            return Err(format!(
                "cannot commit in {} phase",
                self.phase
            ));
        }

        // Update error tracker
        self.error_tracker.record_step(step_error);
        let error_checksum = self.error_tracker.checksum_compact();

        // Build round summary
        let descriptor = RoundDescriptor {
            model_id: self.config.model_id,
            round_number: self.current_round,
            participants: self.pending_receipts.iter().map(|r| r.node_id).collect(),
            deadline: 0,
            params: self.config.params.clone(),
            session_id: self.session_id,
            weight_commitment: self.weight_commitment,
        };

        let summary = RoundSummary {
            round_number: self.current_round,
            descriptor,
            receipts: std::mem::take(&mut self.pending_receipts),
            post_weight_commitment: new_weight_commitment,
            error_checksum,
            committed_at: timestamp,
            tx_hash,
        };

        // Update participant stats
        for receipt in &summary.receipts {
            let key = receipt.node_id.as_bytes();
            if let Some(p) = self.participants.get_mut(key) {
                p.rounds_completed += 1;
                p.proofs_submitted += 1;
            }
        }

        self.rounds.push(summary);
        self.weight_commitment = new_weight_commitment;
        self.current_round += 1;
        self.transition_to(SessionPhase::Committed, timestamp)
    }

    /// Finalizes the session. No more rounds will be run.
    pub fn finalize(&mut self, timestamp: u64) -> Result<SessionSummary, String> {
        if self.phase != SessionPhase::Committed {
            return Err(format!(
                "cannot finalize in {} phase",
                self.phase
            ));
        }
        self.transition_to(SessionPhase::Finalized, timestamp)?;
        Ok(self.summary())
    }

    /// Marks the session as failed with a reason.
    pub fn fail(&mut self, reason: String, timestamp: u64) -> Result<(), String> {
        if self.phase.is_terminal() {
            return Err(format!(
                "cannot fail from terminal phase {}",
                self.phase
            ));
        }
        self.failure_reason = Some(reason);
        self.transition_to(SessionPhase::Failed, timestamp)
    }

    /// Produces a summary of the session.
    pub fn summary(&self) -> SessionSummary {
        let total_proofs: u64 = self.participants.values().map(|p| p.proofs_submitted).sum();
        SessionSummary {
            session_id: self.session_id,
            model_id: self.config.model_id,
            phase: self.phase,
            rounds_completed: self.rounds.len() as u64,
            total_proofs,
            participant_count: self.participant_count(),
            accumulated_error: self.error_tracker.accumulated_error(),
            within_budget: self.error_tracker.within_budget(),
            final_weight_commitment: self.weight_commitment,
            created_at: self.created_at,
        }
    }

    /// Computes a deterministic session hash covering all committed state.
    pub fn session_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.session_id.as_bytes());
        hasher.update(self.current_round.to_le_bytes());
        hasher.update(self.weight_commitment);
        for round in &self.rounds {
            hasher.update(round.post_weight_commitment);
            hasher.update(round.error_checksum.to_le_bytes());
        }
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// Summary of a training session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session identifier.
    pub session_id: SessionId,
    /// Model identifier.
    pub model_id: [u8; 32],
    /// Current phase.
    pub phase: SessionPhase,
    /// Number of completed rounds.
    pub rounds_completed: u64,
    /// Total proofs submitted across all participants.
    pub total_proofs: u64,
    /// Number of active participants.
    pub participant_count: usize,
    /// Total accumulated error.
    pub accumulated_error: f64,
    /// Whether training is within error budget.
    pub within_budget: bool,
    /// Final (or current) weight commitment.
    pub final_weight_commitment: [u8; 32],
    /// When the session was created.
    pub created_at: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::coordination::GradientCommitment;

    fn test_node(seed: u8) -> NodeId {
        NodeId::from_bytes([seed; 32])
    }

    fn test_config() -> SessionConfig {
        SessionConfig {
            model_id: [1u8; 32],
            params: TrainingParams::default(),
            max_rounds: 10,
            min_participants: 1,
            max_participants: 4,
            round_deadline_secs: 60,
            require_on_chain: false,
        }
    }

    fn make_receipt(session_id: SessionId, node_id: NodeId, round: u64) -> TrainingStepReceipt {
        TrainingStepReceipt {
            round_number: round,
            node_id,
            session_id,
            proof_hash: [0xaa; 32],
            gradient_commitment: GradientCommitment {
                gradient_hash: [1u8; 32],
                node_id,
                round_number: round,
                old_weight_hash: [2u8; 32],
                new_weight_hash: [3u8; 32],
                error_bound: 0.001,
            },
            error_checksum: 12345,
            verified_at: 1000,
            tx_hash: None,
        }
    }

    #[test]
    fn test_session_phase_transitions() {
        let phase = SessionPhase::Registered;
        let valid = phase.valid_transitions();
        assert!(valid.contains(&SessionPhase::DataShared));
        assert!(valid.contains(&SessionPhase::Failed));
        assert!(!valid.contains(&SessionPhase::Training));
    }

    #[test]
    fn test_session_full_lifecycle() {
        let initiator = test_node(1);
        let config = test_config();
        let mut session = TrainingSession::new(initiator, config, 1000);

        assert_eq!(session.phase(), SessionPhase::Registered);

        // Add participants
        session.add_participant(test_node(1), 100, 1001).unwrap();
        session.add_participant(test_node(2), 100, 1002).unwrap();
        assert_eq!(session.participant_count(), 2);

        // Share data
        session.share_data(1010).unwrap();
        assert_eq!(session.phase(), SessionPhase::DataShared);

        // Start training
        let round = session.start_training([0xAA; 32], 1020).unwrap();
        assert_eq!(session.phase(), SessionPhase::Training);
        assert_eq!(round.round_number, 0);
        assert_eq!(round.participants.len(), 2);

        // Finish training
        session.finish_training(1030).unwrap();
        assert_eq!(session.phase(), SessionPhase::Proving);

        // Submit receipts
        let receipt = make_receipt(session.session_id, test_node(1), 0);
        session.submit_receipt(receipt).unwrap();

        // Aggregate
        session.start_aggregation(1040).unwrap();
        assert_eq!(session.phase(), SessionPhase::Aggregating);

        // Commit
        session.commit_round([0xBB; 32], 0.001, 1050, None).unwrap();
        assert_eq!(session.phase(), SessionPhase::Committed);
        assert_eq!(session.current_round(), 1);
        assert_eq!(session.rounds().len(), 1);

        // Finalize
        let summary = session.finalize(1060).unwrap();
        assert_eq!(summary.rounds_completed, 1);
        assert!(summary.within_budget);
        assert_eq!(session.phase(), SessionPhase::Finalized);
    }

    #[test]
    fn test_session_invalid_transition() {
        let initiator = test_node(1);
        let config = test_config();
        let mut session = TrainingSession::new(initiator, config, 1000);

        // Cannot go directly to Training from Registered
        let result = session.start_training([0u8; 32], 1001);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_min_participants() {
        let initiator = test_node(1);
        let mut config = test_config();
        config.min_participants = 2;
        let mut session = TrainingSession::new(initiator, config, 1000);

        session.add_participant(test_node(1), 100, 1001).unwrap();

        // Not enough participants
        let result = session.share_data(1010);
        assert!(result.is_err());

        session.add_participant(test_node(2), 100, 1002).unwrap();
        session.share_data(1010).unwrap();
    }

    #[test]
    fn test_session_max_participants() {
        let initiator = test_node(1);
        let mut config = test_config();
        config.max_participants = 2;
        let mut session = TrainingSession::new(initiator, config, 1000);

        session.add_participant(test_node(1), 100, 1001).unwrap();
        session.add_participant(test_node(2), 100, 1002).unwrap();
        let result = session.add_participant(test_node(3), 100, 1003);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_receipt_validation() {
        let initiator = test_node(1);
        let config = test_config();
        let mut session = TrainingSession::new(initiator, config, 1000);

        session.add_participant(test_node(1), 100, 1001).unwrap();
        session.share_data(1010).unwrap();
        session.start_training([0xAA; 32], 1020).unwrap();
        session.finish_training(1030).unwrap();

        // Wrong session
        let bad_receipt = make_receipt(
            SessionId::from_bytes([99u8; 32]),
            test_node(1),
            0,
        );
        assert!(session.submit_receipt(bad_receipt).is_err());

        // Wrong round
        let bad_receipt = make_receipt(session.session_id, test_node(1), 5);
        assert!(session.submit_receipt(bad_receipt).is_err());

        // Unknown participant
        let bad_receipt = make_receipt(session.session_id, test_node(99), 0);
        assert!(session.submit_receipt(bad_receipt).is_err());
    }

    #[test]
    fn test_session_fail() {
        let initiator = test_node(1);
        let config = test_config();
        let mut session = TrainingSession::new(initiator, config, 1000);

        session.fail("test failure".to_string(), 1001).unwrap();
        assert_eq!(session.phase(), SessionPhase::Failed);
        assert_eq!(session.failure_reason(), Some("test failure"));
    }

    #[test]
    fn test_session_multi_round() {
        let initiator = test_node(1);
        let config = test_config();
        let mut session = TrainingSession::new(initiator, config, 1000);

        session.add_participant(test_node(1), 100, 1001).unwrap();
        session.share_data(1010).unwrap();

        // Round 0
        session.start_training([0xAA; 32], 1020).unwrap();
        session.finish_training(1030).unwrap();
        let receipt = make_receipt(session.session_id, test_node(1), 0);
        session.submit_receipt(receipt).unwrap();
        session.start_aggregation(1040).unwrap();
        session.commit_round([0xBB; 32], 0.001, 1050, None).unwrap();

        // Round 1 (loops back to Training from Committed)
        session.start_training([0xBB; 32], 1060).unwrap();
        session.finish_training(1070).unwrap();
        let receipt = make_receipt(session.session_id, test_node(1), 1);
        session.submit_receipt(receipt).unwrap();
        session.start_aggregation(1080).unwrap();
        session.commit_round([0xCC; 32], 0.002, 1090, None).unwrap();

        assert_eq!(session.current_round(), 2);
        assert_eq!(session.rounds().len(), 2);
        assert!((session.error_tracker().accumulated_error() - 0.003).abs() < 1e-15);
    }

    #[test]
    fn test_session_hash_deterministic() {
        let initiator = test_node(1);
        let config = test_config();
        let session = TrainingSession::new(initiator, config, 1000);

        let h1 = session.session_hash();
        let h2 = session.session_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_session_summary() {
        let initiator = test_node(1);
        let config = test_config();
        let session = TrainingSession::new(initiator, config, 1000);

        let summary = session.summary();
        assert_eq!(summary.rounds_completed, 0);
        assert_eq!(summary.participant_count, 0);
        assert!(summary.within_budget);
    }

    #[test]
    fn test_terminal_phases_cannot_fail() {
        let initiator = test_node(1);
        let config = test_config();
        let mut session = TrainingSession::new(initiator, config, 1000);

        session.add_participant(test_node(1), 100, 1001).unwrap();
        session.share_data(1010).unwrap();
        session.start_training([0xAA; 32], 1020).unwrap();
        session.finish_training(1030).unwrap();
        let receipt = make_receipt(session.session_id, test_node(1), 0);
        session.submit_receipt(receipt).unwrap();
        session.start_aggregation(1040).unwrap();
        session.commit_round([0xBB; 32], 0.001, 1050, None).unwrap();
        session.finalize(1060).unwrap();

        // Cannot fail from Finalized
        let result = session.fail("too late".to_string(), 1070);
        assert!(result.is_err());
    }
}
