//! BFT 2-Phase Commit Consensus for Gradient Aggregation.
//!
//! Implements a Byzantine-fault-tolerant consensus protocol for agreeing on
//! aggregated gradient commitments. The protocol uses a simple 2-phase commit:
//!
//! **Phase 1 (Propose):** The leader computes the aggregated gradient commitment
//! from collected worker gradients and broadcasts a `Propose` message with a
//! cryptographic binding commitment.
//!
//! **Phase 2 (Vote):** Each participant independently verifies the proposal by
//! recomputing the commitment from their local view of gradients, then broadcasts
//! a `Vote` (accept/reject) with their own computed commitment.
//!
//! **Decision:** If at least `2f + 1` votes are received (where `f` is the max
//! number of Byzantine nodes), the proposal is committed. Otherwise, it is aborted.
//!
//! The cryptographic binding commitment prevents the leader from equivocating
//! (sending different proposals to different participants).

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime};

use sha2::{Digest, Sha256};

use crate::network::messages::{ConsensusMessage, PeerId};

/// Configuration for the BFT consensus protocol.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Maximum number of Byzantine (faulty) nodes the protocol tolerates.
    /// The quorum requirement is `2f + 1` out of `n` total participants.
    pub max_faulty: usize,
    /// Timeout for the propose phase (waiting for proposal from leader).
    pub propose_timeout: Duration,
    /// Timeout for the vote phase (waiting for 2f+1 votes).
    pub vote_timeout: Duration,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            max_faulty: 1,
            propose_timeout: Duration::from_secs(30),
            vote_timeout: Duration::from_secs(30),
        }
    }
}

/// Current phase of a consensus round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusPhase {
    /// No active consensus.
    Idle,
    /// Phase 1: Leader has proposed, waiting for votes.
    Proposing,
    /// Phase 2: Collecting votes.
    Voting,
    /// Consensus reached, committed.
    Committed,
    /// Consensus failed, aborted.
    Aborted,
}

/// A proposal for aggregated gradient commitment.
#[derive(Debug, Clone)]
pub struct ConsensusProposal {
    /// Round ID.
    pub round_id: u64,
    /// SHA-256 of sorted accepted gradient commitments.
    pub aggregated_commitment: [u8; 32],
    /// Cryptographic binding: H(aggregated_commitment || proposer_id || nonce).
    pub proposer_binding: [u8; 32],
    /// Who proposed this.
    pub proposer: PeerId,
    /// Number of gradients included.
    pub num_gradients: usize,
    /// Combined error bound.
    pub error_bound: f64,
    /// Nonce used in binding commitment.
    pub nonce: [u8; 16],
    /// When the proposal was created.
    pub created_at: Instant,
}

/// A vote on a consensus proposal.
#[derive(Debug, Clone)]
pub struct ConsensusVote {
    /// Round ID.
    pub round_id: u64,
    /// Who cast this vote.
    pub voter: PeerId,
    /// Accept or reject.
    pub accept: bool,
    /// Voter's independently computed commitment.
    pub voter_commitment: [u8; 32],
    /// Reason for rejection (if any).
    pub reason: Option<String>,
}

/// Result of a consensus round.
#[derive(Debug, Clone)]
pub enum ConsensusResult {
    /// Quorum reached, gradient commitment agreed upon.
    Committed {
        round_id: u64,
        commitment: [u8; 32],
        votes_for: usize,
        votes_against: usize,
        total_participants: usize,
    },
    /// Insufficient votes or too many rejections.
    Aborted {
        round_id: u64,
        reason: String,
        votes_for: usize,
        votes_against: usize,
    },
    /// Timed out waiting for votes.
    Timeout {
        round_id: u64,
        votes_received: usize,
        votes_required: usize,
    },
}

/// Event emitted by the consensus protocol.
#[derive(Debug, Clone)]
pub enum ConsensusEvent {
    /// A proposal was received and is being voted on.
    ProposalReceived {
        round_id: u64,
        proposer: PeerId,
        aggregated_commitment: [u8; 32],
    },
    /// A vote was received.
    VoteReceived {
        round_id: u64,
        voter: PeerId,
        accept: bool,
    },
    /// Consensus was reached.
    ConsensusReached(ConsensusResult),
    /// Consensus failed.
    ConsensusFailed(ConsensusResult),
}

/// State of a single consensus round.
#[derive(Debug)]
pub struct ConsensusRound {
    /// Round ID.
    pub round_id: u64,
    /// Current phase.
    pub phase: ConsensusPhase,
    /// The proposal (set when proposing or when receiving a proposal).
    pub proposal: Option<ConsensusProposal>,
    /// Collected votes indexed by voter.
    pub votes: HashMap<PeerId, ConsensusVote>,
    /// Set of expected participants.
    pub participants: HashSet<PeerId>,
    /// When this consensus round's current phase started.
    pub phase_started: Instant,
    /// Configuration.
    pub config: ConsensusConfig,
}

impl ConsensusRound {
    /// Creates a new consensus round.
    pub fn new(
        round_id: u64,
        participants: HashSet<PeerId>,
        config: ConsensusConfig,
    ) -> Self {
        Self {
            round_id,
            phase: ConsensusPhase::Idle,
            proposal: None,
            votes: HashMap::new(),
            participants,
            phase_started: Instant::now(),
            config,
        }
    }

    /// Returns the quorum size: 2f + 1.
    pub fn quorum_size(&self) -> usize {
        2 * self.config.max_faulty + 1
    }

    /// Returns the total number of participants.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Returns how many accept votes we have.
    pub fn accept_count(&self) -> usize {
        self.votes.values().filter(|v| v.accept).count()
    }

    /// Returns how many reject votes we have.
    pub fn reject_count(&self) -> usize {
        self.votes.values().filter(|v| !v.accept).count()
    }

    /// Checks if quorum has been reached (2f+1 accept votes).
    pub fn has_quorum(&self) -> bool {
        self.accept_count() >= self.quorum_size()
    }

    /// Checks if the round cannot possibly reach quorum (too many rejects).
    pub fn is_irrecoverable(&self) -> bool {
        let max_possible_accepts = self.participant_count() - self.reject_count();
        max_possible_accepts < self.quorum_size()
    }

    /// Checks if the current phase has timed out.
    pub fn is_timed_out(&self) -> bool {
        let timeout = match self.phase {
            ConsensusPhase::Proposing => self.config.propose_timeout,
            ConsensusPhase::Voting => self.config.vote_timeout,
            _ => return false,
        };
        self.phase_started.elapsed() > timeout
    }
}

/// The BFT consensus protocol manager.
///
/// Manages consensus rounds for gradient aggregation using 2-phase commit.
/// Each training round triggers a consensus round to agree on the aggregated
/// gradient commitment before it's accepted.
pub struct ConsensusProtocol {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: ConsensusConfig,
    /// Active consensus round (at most one at a time).
    active_round: Option<ConsensusRound>,
    /// History of completed consensus results (bounded).
    history: Vec<ConsensusResult>,
    /// Maximum history size.
    max_history: usize,
}

impl ConsensusProtocol {
    /// Creates a new consensus protocol instance.
    pub fn new(local_id: PeerId, config: ConsensusConfig) -> Self {
        Self {
            local_id,
            config,
            active_round: None,
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Returns a reference to the active round, if any.
    pub fn active_round(&self) -> Option<&ConsensusRound> {
        self.active_round.as_ref()
    }

    /// Returns the consensus config.
    pub fn config(&self) -> &ConsensusConfig {
        &self.config
    }

    /// Creates a proposal as the leader.
    ///
    /// Computes the cryptographic binding commitment and transitions to Proposing phase.
    /// Returns the ConsensusMessage::Propose to broadcast.
    pub fn create_proposal(
        &mut self,
        round_id: u64,
        participants: HashSet<PeerId>,
        gradient_commitments: &[[u8; 32]],
        error_bound: f64,
    ) -> ConsensusMessage {
        // Compute aggregated commitment: SHA-256 of sorted gradient commitments
        let aggregated_commitment = compute_aggregated_commitment(gradient_commitments);

        // Generate nonce for binding commitment
        let nonce = generate_nonce();

        // Compute binding: H(aggregated_commitment || proposer_id || nonce)
        let proposer_binding = compute_binding(
            &aggregated_commitment,
            &self.local_id,
            &nonce,
        );

        let proposal = ConsensusProposal {
            round_id,
            aggregated_commitment,
            proposer_binding,
            proposer: self.local_id.clone(),
            num_gradients: gradient_commitments.len(),
            error_bound,
            nonce,
            created_at: Instant::now(),
        };

        let msg = ConsensusMessage::Propose {
            round_id,
            aggregated_commitment,
            proposer_binding,
            num_gradients: gradient_commitments.len(),
            error_bound,
            nonce,
        };

        let mut round = ConsensusRound::new(round_id, participants, self.config.clone());
        round.phase = ConsensusPhase::Proposing;
        round.proposal = Some(proposal);
        round.phase_started = Instant::now();

        self.active_round = Some(round);
        msg
    }

    /// Handles a received Propose message (non-leader participants).
    ///
    /// Verifies the binding commitment, computes own commitment from local gradient
    /// view, and returns a Vote message.
    pub fn handle_proposal(
        &mut self,
        from: &PeerId,
        round_id: u64,
        aggregated_commitment: [u8; 32],
        proposer_binding: [u8; 32],
        num_gradients: usize,
        error_bound: f64,
        nonce: [u8; 16],
        local_gradient_commitments: &[[u8; 32]],
        participants: HashSet<PeerId>,
    ) -> (ConsensusMessage, Vec<ConsensusEvent>) {
        let mut events = Vec::new();

        events.push(ConsensusEvent::ProposalReceived {
            round_id,
            proposer: from.clone(),
            aggregated_commitment,
        });

        // Verify the binding commitment
        let expected_binding = compute_binding(&aggregated_commitment, from, &nonce);
        if expected_binding != proposer_binding {
            return (
                ConsensusMessage::Vote {
                    round_id,
                    accept: false,
                    voter_commitment: [0u8; 32],
                    reason: Some("Invalid proposer binding commitment".to_string()),
                },
                events,
            );
        }

        // Compute our own aggregated commitment from local view
        let our_commitment = compute_aggregated_commitment(local_gradient_commitments);

        // Compare our commitment with the proposal
        let accept = our_commitment == aggregated_commitment;
        let reason = if !accept {
            Some("Commitment mismatch: local view differs from proposal".to_string())
        } else {
            None
        };

        // Set up the consensus round if not already active
        let proposal = ConsensusProposal {
            round_id,
            aggregated_commitment,
            proposer_binding,
            proposer: from.clone(),
            num_gradients,
            error_bound,
            nonce,
            created_at: Instant::now(),
        };

        let mut round = ConsensusRound::new(round_id, participants, self.config.clone());
        round.phase = ConsensusPhase::Voting;
        round.proposal = Some(proposal);
        round.phase_started = Instant::now();
        self.active_round = Some(round);

        (
            ConsensusMessage::Vote {
                round_id,
                accept,
                voter_commitment: our_commitment,
                reason,
            },
            events,
        )
    }

    /// Handles a received Vote message.
    ///
    /// Collects the vote and checks if quorum has been reached.
    /// Returns an optional decision message (Commit or Abort) and events.
    pub fn handle_vote(
        &mut self,
        from: &PeerId,
        round_id: u64,
        accept: bool,
        voter_commitment: [u8; 32],
        reason: Option<String>,
    ) -> (Option<ConsensusMessage>, Vec<ConsensusEvent>) {
        let mut events = Vec::new();

        events.push(ConsensusEvent::VoteReceived {
            round_id,
            voter: from.clone(),
            accept,
        });

        let round = match self.active_round.as_mut() {
            Some(r) if r.round_id == round_id => r,
            _ => return (None, events),
        };

        // Only accept votes from known participants
        if !round.participants.contains(from) {
            return (None, events);
        }

        // Don't accept duplicate votes
        if round.votes.contains_key(from) {
            return (None, events);
        }

        // Record the vote
        round.votes.insert(from.clone(), ConsensusVote {
            round_id,
            voter: from.clone(),
            accept,
            voter_commitment,
            reason,
        });

        // Transition to voting phase if still proposing
        if round.phase == ConsensusPhase::Proposing {
            round.phase = ConsensusPhase::Voting;
            round.phase_started = Instant::now();
        }

        // Check for quorum
        if round.has_quorum() {
            let commitment = round.proposal.as_ref()
                .map(|p| p.aggregated_commitment)
                .unwrap_or([0u8; 32]);

            let result = ConsensusResult::Committed {
                round_id,
                commitment,
                votes_for: round.accept_count(),
                votes_against: round.reject_count(),
                total_participants: round.participant_count(),
            };

            round.phase = ConsensusPhase::Committed;
            events.push(ConsensusEvent::ConsensusReached(result.clone()));

            self.history.push(result);
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }

            return (
                Some(ConsensusMessage::Commit {
                    round_id,
                    final_commitment: commitment,
                    votes_for: round.accept_count(),
                    total_participants: round.participant_count(),
                }),
                events,
            );
        }

        // Check if irrecoverable (too many rejects)
        if round.is_irrecoverable() {
            let result = ConsensusResult::Aborted {
                round_id,
                reason: format!(
                    "Too many rejections: {} reject, {} accept, need {}",
                    round.reject_count(),
                    round.accept_count(),
                    round.quorum_size(),
                ),
                votes_for: round.accept_count(),
                votes_against: round.reject_count(),
            };

            round.phase = ConsensusPhase::Aborted;
            events.push(ConsensusEvent::ConsensusFailed(result.clone()));

            self.history.push(result);
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }

            return (
                Some(ConsensusMessage::Abort {
                    round_id,
                    reason: "Insufficient accept votes for quorum".to_string(),
                }),
                events,
            );
        }

        (None, events)
    }

    /// Checks for timeout and returns a decision if the round has timed out.
    pub fn check_timeout(&mut self) -> Option<(ConsensusMessage, Vec<ConsensusEvent>)> {
        let round = self.active_round.as_mut()?;

        if !round.is_timed_out() {
            return None;
        }

        let mut events = Vec::new();

        // Check if we have quorum despite timeout
        if round.has_quorum() {
            let commitment = round.proposal.as_ref()
                .map(|p| p.aggregated_commitment)
                .unwrap_or([0u8; 32]);

            let result = ConsensusResult::Committed {
                round_id: round.round_id,
                commitment,
                votes_for: round.accept_count(),
                votes_against: round.reject_count(),
                total_participants: round.participant_count(),
            };

            round.phase = ConsensusPhase::Committed;
            events.push(ConsensusEvent::ConsensusReached(result.clone()));
            self.history.push(result);

            return Some((
                ConsensusMessage::Commit {
                    round_id: round.round_id,
                    final_commitment: commitment,
                    votes_for: round.accept_count(),
                    total_participants: round.participant_count(),
                },
                events,
            ));
        }

        // Timeout without quorum
        let result = ConsensusResult::Timeout {
            round_id: round.round_id,
            votes_received: round.votes.len(),
            votes_required: round.quorum_size(),
        };

        round.phase = ConsensusPhase::Aborted;
        events.push(ConsensusEvent::ConsensusFailed(result.clone()));
        self.history.push(result);

        Some((
            ConsensusMessage::Abort {
                round_id: round.round_id,
                reason: format!(
                    "Timeout: received {}/{} votes (need {})",
                    round.votes.len(),
                    round.participant_count(),
                    round.quorum_size(),
                ),
            },
            events,
        ))
    }

    /// Handles a Commit message (from leader after quorum).
    ///
    /// Non-leader participants use this to finalize their local state.
    pub fn handle_commit(
        &mut self,
        round_id: u64,
        final_commitment: [u8; 32],
        votes_for: usize,
        total_participants: usize,
    ) -> Vec<ConsensusEvent> {
        let mut events = Vec::new();

        if let Some(round) = self.active_round.as_mut() {
            if round.round_id == round_id {
                round.phase = ConsensusPhase::Committed;

                let result = ConsensusResult::Committed {
                    round_id,
                    commitment: final_commitment,
                    votes_for,
                    votes_against: 0,
                    total_participants,
                };

                events.push(ConsensusEvent::ConsensusReached(result.clone()));
                self.history.push(result);
            }
        }

        events
    }

    /// Handles an Abort message.
    pub fn handle_abort(
        &mut self,
        round_id: u64,
        reason: String,
    ) -> Vec<ConsensusEvent> {
        let mut events = Vec::new();

        if let Some(round) = self.active_round.as_mut() {
            if round.round_id == round_id {
                round.phase = ConsensusPhase::Aborted;

                let result = ConsensusResult::Aborted {
                    round_id,
                    reason,
                    votes_for: round.accept_count(),
                    votes_against: round.reject_count(),
                };

                events.push(ConsensusEvent::ConsensusFailed(result.clone()));
                self.history.push(result);
            }
        }

        events
    }

    /// Clears the active round (call after processing a Commit or Abort).
    pub fn clear_active_round(&mut self) {
        self.active_round = None;
    }

    /// Returns whether there's an active consensus round.
    pub fn is_active(&self) -> bool {
        self.active_round.is_some()
    }

    /// Returns consensus history.
    pub fn history(&self) -> &[ConsensusResult] {
        &self.history
    }
}

/// Computes the aggregated commitment from a set of gradient commitments.
///
/// Sorts the commitments lexicographically, then SHA-256 hashes them together.
/// This ensures deterministic computation regardless of order of receipt.
pub fn compute_aggregated_commitment(commitments: &[[u8; 32]]) -> [u8; 32] {
    let mut sorted = commitments.to_vec();
    sorted.sort();

    let mut hasher = Sha256::new();
    for commitment in &sorted {
        hasher.update(commitment);
    }
    hasher.finalize().into()
}

/// Computes a cryptographic binding commitment.
///
/// Binds the proposer to a specific aggregated commitment value, preventing
/// equivocation (sending different proposals to different participants).
pub fn compute_binding(
    aggregated_commitment: &[u8; 32],
    proposer_id: &PeerId,
    nonce: &[u8; 16],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"HELIX-CONSENSUS-BINDING-V1");
    hasher.update(aggregated_commitment);
    hasher.update(proposer_id.0.as_bytes());
    hasher.update(nonce);
    hasher.finalize().into()
}

/// Generates a random 16-byte nonce for binding commitments.
fn generate_nonce() -> [u8; 16] {
    use rand::Rng;
    let mut nonce = [0u8; 16];
    rand::thread_rng().fill(&mut nonce);
    nonce
}

/// Verifies that the number of participants is sufficient for BFT consensus.
///
/// For `f` Byzantine nodes, we need `n >= 3f + 1` total participants.
pub fn verify_bft_threshold(num_participants: usize, max_faulty: usize) -> bool {
    num_participants >= 3 * max_faulty + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_participants(n: usize) -> (Vec<PeerId>, HashSet<PeerId>) {
        let peers: Vec<PeerId> = (0..n)
            .map(|i| PeerId::from_string(format!("peer-{}", i)))
            .collect();
        let set: HashSet<PeerId> = peers.iter().cloned().collect();
        (peers, set)
    }

    #[test]
    fn test_consensus_config_defaults() {
        let config = ConsensusConfig::default();
        assert_eq!(config.max_faulty, 1);
        assert_eq!(config.propose_timeout, Duration::from_secs(30));
        assert_eq!(config.vote_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_quorum_size() {
        let (_, participants) = make_participants(7);
        let round = ConsensusRound::new(1, participants, ConsensusConfig {
            max_faulty: 2,
            ..Default::default()
        });
        // 2f+1 = 5
        assert_eq!(round.quorum_size(), 5);
    }

    #[test]
    fn test_bft_threshold() {
        // f=1 needs n >= 4
        assert!(!verify_bft_threshold(3, 1));
        assert!(verify_bft_threshold(4, 1));

        // f=2 needs n >= 7
        assert!(!verify_bft_threshold(6, 2));
        assert!(verify_bft_threshold(7, 2));
    }

    #[test]
    fn test_aggregated_commitment_deterministic() {
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        let c3 = [3u8; 32];

        // Same commitments in different order should produce same result
        let result1 = compute_aggregated_commitment(&[c1, c2, c3]);
        let result2 = compute_aggregated_commitment(&[c3, c1, c2]);
        let result3 = compute_aggregated_commitment(&[c2, c3, c1]);

        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
    }

    #[test]
    fn test_binding_commitment() {
        let commitment = [42u8; 32];
        let proposer = PeerId::from_string("leader-1");
        let nonce = [7u8; 16];

        let binding = compute_binding(&commitment, &proposer, &nonce);

        // Same inputs should produce same binding
        let binding2 = compute_binding(&commitment, &proposer, &nonce);
        assert_eq!(binding, binding2);

        // Different proposer should produce different binding
        let other = PeerId::from_string("leader-2");
        let binding3 = compute_binding(&commitment, &other, &nonce);
        assert_ne!(binding, binding3);

        // Different nonce should produce different binding
        let nonce2 = [8u8; 16];
        let binding4 = compute_binding(&commitment, &proposer, &nonce2);
        assert_ne!(binding, binding4);
    }

    #[test]
    fn test_full_consensus_happy_path() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig {
            max_faulty: 1,
            ..Default::default()
        };

        // Leader creates proposal
        let mut leader = ConsensusProtocol::new(peers[0].clone(), config.clone());
        let commitments = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let propose_msg = leader.create_proposal(
            1,
            participants.clone(),
            &commitments,
            0.05,
        );

        // Extract proposal fields
        let (round_id, agg_commitment, binding, num_grads, error_bound, nonce) = match propose_msg {
            ConsensusMessage::Propose {
                round_id,
                aggregated_commitment,
                proposer_binding,
                num_gradients,
                error_bound,
                nonce,
            } => (round_id, aggregated_commitment, proposer_binding, num_gradients, error_bound, nonce),
            _ => panic!("Expected Propose message"),
        };

        assert_eq!(round_id, 1);
        assert_eq!(num_grads, 3);

        // Workers receive proposal and vote
        for i in 1..4 {
            let mut worker = ConsensusProtocol::new(peers[i].clone(), config.clone());
            let (vote_msg, events) = worker.handle_proposal(
                &peers[0],
                round_id,
                agg_commitment,
                binding,
                num_grads,
                error_bound,
                nonce,
                &commitments, // Same commitments -> should accept
                participants.clone(),
            );

            assert!(!events.is_empty());

            let (accept, voter_commitment) = match vote_msg {
                ConsensusMessage::Vote { accept, voter_commitment, .. } => (accept, voter_commitment),
                _ => panic!("Expected Vote message"),
            };
            assert!(accept, "Worker {} should accept", i);
            assert_eq!(voter_commitment, agg_commitment);

            // Leader processes vote
            let (decision, _events) = leader.handle_vote(
                &peers[i],
                round_id,
                accept,
                voter_commitment,
                None,
            );

            // After 3 votes with f=1, quorum (2f+1=3) should be reached
            if i == 3 {
                assert!(decision.is_some());
                match decision.unwrap() {
                    ConsensusMessage::Commit { votes_for, .. } => {
                        assert!(votes_for >= 3);
                    }
                    _ => panic!("Expected Commit message"),
                }
            }
        }

        // Leader should be committed
        assert_eq!(
            leader.active_round().unwrap().phase,
            ConsensusPhase::Committed,
        );
    }

    #[test]
    fn test_consensus_reject_on_mismatch() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig {
            max_faulty: 1,
            ..Default::default()
        };

        let mut leader = ConsensusProtocol::new(peers[0].clone(), config.clone());
        let commitments = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let propose_msg = leader.create_proposal(1, participants.clone(), &commitments, 0.05);

        let (round_id, agg_commitment, binding, num_grads, error_bound, nonce) = match propose_msg {
            ConsensusMessage::Propose {
                round_id,
                aggregated_commitment,
                proposer_binding,
                num_gradients,
                error_bound,
                nonce,
            } => (round_id, aggregated_commitment, proposer_binding, num_gradients, error_bound, nonce),
            _ => panic!("Expected Propose"),
        };

        // Worker with different local view rejects
        let mut worker = ConsensusProtocol::new(peers[1].clone(), config.clone());
        let different_commitments = [[4u8; 32], [5u8; 32], [6u8; 32]]; // Different!
        let (vote_msg, _) = worker.handle_proposal(
            &peers[0],
            round_id,
            agg_commitment,
            binding,
            num_grads,
            error_bound,
            nonce,
            &different_commitments,
            participants.clone(),
        );

        match vote_msg {
            ConsensusMessage::Vote { accept, reason, .. } => {
                assert!(!accept, "Worker should reject mismatched commitments");
                assert!(reason.is_some());
            }
            _ => panic!("Expected Vote"),
        }
    }

    #[test]
    fn test_consensus_invalid_binding_rejected() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig::default();

        let mut worker = ConsensusProtocol::new(peers[1].clone(), config.clone());

        // Fake binding that doesn't match
        let fake_binding = [0xFFu8; 32];
        let commitments = [[1u8; 32], [2u8; 32]];

        let (vote_msg, _) = worker.handle_proposal(
            &peers[0],
            1,
            compute_aggregated_commitment(&commitments),
            fake_binding,
            2,
            0.05,
            [0u8; 16],
            &commitments,
            participants,
        );

        match vote_msg {
            ConsensusMessage::Vote { accept, reason, .. } => {
                assert!(!accept);
                assert!(reason.unwrap().contains("binding"));
            }
            _ => panic!("Expected Vote"),
        }
    }

    #[test]
    fn test_consensus_abort_on_too_many_rejects() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig {
            max_faulty: 1,
            ..Default::default()
        };

        let mut leader = ConsensusProtocol::new(peers[0].clone(), config.clone());
        let commitments = [[1u8; 32], [2u8; 32]];
        let _propose_msg = leader.create_proposal(1, participants.clone(), &commitments, 0.05);

        // Send 2 reject votes out of 4 participants (quorum = 3)
        // After 2 rejects, max possible accepts = 4-2 = 2 < 3 = quorum -> irrecoverable
        for i in 1..=2 {
            let (decision, _) = leader.handle_vote(
                &peers[i],
                1,
                false,
                [0u8; 32],
                Some("disagree".to_string()),
            );

            if i == 2 {
                assert!(decision.is_some());
                match decision.unwrap() {
                    ConsensusMessage::Abort { reason, .. } => {
                        assert!(reason.contains("quorum"));
                    }
                    _ => panic!("Expected Abort"),
                }
            }
        }

        assert_eq!(
            leader.active_round().unwrap().phase,
            ConsensusPhase::Aborted,
        );
    }

    #[test]
    fn test_consensus_duplicate_vote_ignored() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig {
            max_faulty: 1,
            ..Default::default()
        };

        let mut leader = ConsensusProtocol::new(peers[0].clone(), config);
        let commitments = [[1u8; 32]];
        let _propose = leader.create_proposal(1, participants, &commitments, 0.05);

        // First vote from peer-1
        let (d1, _) = leader.handle_vote(&peers[1], 1, true, [0u8; 32], None);
        assert!(d1.is_none()); // Not yet quorum

        // Duplicate vote from peer-1 should be ignored
        let (d2, _) = leader.handle_vote(&peers[1], 1, true, [0u8; 32], None);
        assert!(d2.is_none());

        // Only 1 vote should be recorded
        assert_eq!(leader.active_round().unwrap().votes.len(), 1);
    }

    #[test]
    fn test_consensus_unknown_voter_ignored() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig::default();

        let mut leader = ConsensusProtocol::new(peers[0].clone(), config);
        let commitments = [[1u8; 32]];
        let _propose = leader.create_proposal(1, participants, &commitments, 0.05);

        // Vote from unknown peer
        let unknown = PeerId::from_string("unknown-peer");
        let (decision, _) = leader.handle_vote(&unknown, 1, true, [0u8; 32], None);
        assert!(decision.is_none());
        assert_eq!(leader.active_round().unwrap().votes.len(), 0);
    }

    #[test]
    fn test_consensus_history() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig {
            max_faulty: 1,
            ..Default::default()
        };

        let mut leader = ConsensusProtocol::new(peers[0].clone(), config.clone());
        let commitments = [[1u8; 32]];

        // Run a round to completion
        let _propose = leader.create_proposal(1, participants.clone(), &commitments, 0.05);
        for i in 1..=3 {
            leader.handle_vote(&peers[i], 1, true, compute_aggregated_commitment(&commitments), None);
        }

        assert_eq!(leader.history().len(), 1);
        match &leader.history()[0] {
            ConsensusResult::Committed { round_id, .. } => assert_eq!(*round_id, 1),
            _ => panic!("Expected committed result"),
        }
    }

    #[test]
    fn test_consensus_handle_commit() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig::default();

        let commitments = [[1u8; 32], [2u8; 32]];
        let mut worker = ConsensusProtocol::new(peers[1].clone(), config.clone());

        // Worker receives proposal
        let agg = compute_aggregated_commitment(&commitments);
        let nonce = [0u8; 16];
        let binding = compute_binding(&agg, &peers[0], &nonce);
        worker.handle_proposal(
            &peers[0], 1, agg, binding, 2, 0.05, nonce, &commitments, participants,
        );

        // Worker receives Commit from leader
        let events = worker.handle_commit(1, agg, 3, 4);
        assert!(!events.is_empty());
        assert_eq!(worker.active_round().unwrap().phase, ConsensusPhase::Committed);
    }

    #[test]
    fn test_consensus_handle_abort() {
        let (peers, participants) = make_participants(4);
        let config = ConsensusConfig::default();

        let commitments = [[1u8; 32]];
        let mut worker = ConsensusProtocol::new(peers[1].clone(), config.clone());

        // Set up a round
        let agg = compute_aggregated_commitment(&commitments);
        let nonce = [0u8; 16];
        let binding = compute_binding(&agg, &peers[0], &nonce);
        worker.handle_proposal(
            &peers[0], 1, agg, binding, 1, 0.05, nonce, &commitments, participants,
        );

        // Worker receives Abort
        let events = worker.handle_abort(1, "test abort".to_string());
        assert!(!events.is_empty());
        assert_eq!(worker.active_round().unwrap().phase, ConsensusPhase::Aborted);
    }
}
