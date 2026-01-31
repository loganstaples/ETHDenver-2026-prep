//! Byzantine fault detection and handling for MPC.
//!
//! This module detects and handles malicious behavior from parties including:
//! - Message tampering
//! - Message dropping/delay
//! - Inconsistent values across operations
//! - Replay attacks
//! - Invalid protocol messages
//!
//! # Detection Strategies
//!
//! 1. **MAC verification**: Every message includes a MAC that is checked
//! 2. **Commitment verification**: All values are committed before revealing
//! 3. **Consistency checks**: Values are checked across multiple operations
//! 4. **Timeout detection**: Unresponsive parties are detected via timeouts
//! 5. **Statistical sampling**: Random spot checks catch cheating with high probability

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Types of Byzantine faults that can be detected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FaultType {
    /// Party sent an invalid MAC
    InvalidMAC,
    /// Party's revealed value doesn't match commitment
    CommitmentMismatch,
    /// Party's values are inconsistent across operations
    InconsistentValues,
    /// Party is unresponsive
    Timeout,
    /// Party sent a message with an old sequence number
    ReplayAttack,
    /// Party sent malformed protocol message
    MalformedMessage,
    /// Party's computation result is incorrect
    IncorrectComputation,
    /// Party violated protocol (e.g., sent message out of order)
    ProtocolViolation,
    /// Party's gradient appears to be poisoned
    GradientPoisoning,
    /// Party submitted invalid Beaver triple
    InvalidBeaver,
}

impl std::fmt::Display for FaultType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMAC => write!(f, "Invalid MAC"),
            Self::CommitmentMismatch => write!(f, "Commitment mismatch"),
            Self::InconsistentValues => write!(f, "Inconsistent values"),
            Self::Timeout => write!(f, "Timeout"),
            Self::ReplayAttack => write!(f, "Replay attack"),
            Self::MalformedMessage => write!(f, "Malformed message"),
            Self::IncorrectComputation => write!(f, "Incorrect computation"),
            Self::ProtocolViolation => write!(f, "Protocol violation"),
            Self::GradientPoisoning => write!(f, "Gradient poisoning"),
            Self::InvalidBeaver => write!(f, "Invalid Beaver triple"),
        }
    }
}

/// A detected fault with metadata.
#[derive(Debug, Clone)]
pub struct DetectedFault {
    /// The party that committed the fault
    pub party: PartyId,
    /// Type of fault
    pub fault_type: FaultType,
    /// Round in which fault was detected
    pub round: u64,
    /// Human-readable description
    pub description: String,
    /// Evidence (e.g., hash of bad message)
    pub evidence: Option<Vec<u8>>,
    /// Timestamp of detection
    pub detected_at: Instant,
}

impl DetectedFault {
    pub fn new(
        party: PartyId,
        fault_type: FaultType,
        round: u64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            party,
            fault_type,
            round,
            description: description.into(),
            evidence: None,
            detected_at: Instant::now(),
        }
    }

    pub fn with_evidence(mut self, evidence: Vec<u8>) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// Generates a cryptographic proof of the fault for slashing.
    pub fn generate_proof(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.party.0.as_bytes());
        hasher.update(self.fault_type.to_string().as_bytes());
        hasher.update(&self.round.to_le_bytes());
        hasher.update(self.description.as_bytes());
        if let Some(evidence) = &self.evidence {
            hasher.update(evidence);
        }
        hasher.finalize().into()
    }
}

/// Status of a party in the protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartyStatus {
    /// Party is active and behaving correctly
    Active,
    /// Party is suspected of misbehavior
    Suspected,
    /// Party has been confirmed as faulty
    Faulty,
    /// Party has been excluded from the protocol
    Excluded,
    /// Party is unresponsive
    Unresponsive,
}

/// State tracking for a single party.
#[derive(Debug)]
pub struct PartyState {
    /// Party ID
    pub party_id: PartyId,
    /// Current status
    pub status: PartyStatus,
    /// Number of faults detected
    pub fault_count: usize,
    /// Last message timestamp
    pub last_seen: Instant,
    /// Expected next sequence number
    pub expected_sequence: u64,
    /// Suspected faults (not yet confirmed)
    pub suspected_faults: Vec<FaultType>,
    /// Confirmed faults
    pub confirmed_faults: Vec<DetectedFault>,
}

impl PartyState {
    pub fn new(party_id: PartyId) -> Self {
        Self {
            party_id,
            status: PartyStatus::Active,
            fault_count: 0,
            last_seen: Instant::now(),
            expected_sequence: 0,
            suspected_faults: Vec::new(),
            confirmed_faults: Vec::new(),
        }
    }

    /// Records a message from this party.
    pub fn record_message(&mut self, sequence: u64) {
        self.last_seen = Instant::now();
        if sequence >= self.expected_sequence {
            self.expected_sequence = sequence + 1;
        }
    }

    /// Records a suspected fault.
    pub fn suspect(&mut self, fault_type: FaultType) {
        self.suspected_faults.push(fault_type);
        if self.suspected_faults.len() >= 2 {
            self.status = PartyStatus::Suspected;
        }
    }

    /// Confirms a fault.
    pub fn confirm_fault(&mut self, fault: DetectedFault) {
        self.fault_count += 1;
        self.confirmed_faults.push(fault);

        // Auto-exclude after too many faults
        if self.fault_count >= 3 {
            self.status = PartyStatus::Faulty;
        } else {
            self.status = PartyStatus::Suspected;
        }
    }

    /// Excludes the party.
    pub fn exclude(&mut self) {
        self.status = PartyStatus::Excluded;
    }

    /// Checks if party has timed out.
    pub fn check_timeout(&mut self, timeout: Duration) -> bool {
        if self.last_seen.elapsed() > timeout {
            self.status = PartyStatus::Unresponsive;
            true
        } else {
            false
        }
    }
}

/// Byzantine fault detector for MPC sessions.
pub struct ByzantineDetector {
    /// State for each party
    party_states: HashMap<String, PartyState>,
    /// Round number
    current_round: u64,
    /// Timeout for party responses
    timeout: Duration,
    /// Maximum faults before exclusion
    max_faults: usize,
    /// All detected faults
    all_faults: Vec<DetectedFault>,
    /// Parties that have been excluded
    excluded_parties: HashSet<String>,
}

impl ByzantineDetector {
    pub fn new(timeout: Duration, max_faults: usize) -> Self {
        Self {
            party_states: HashMap::new(),
            current_round: 0,
            timeout,
            max_faults,
            all_faults: Vec::new(),
            excluded_parties: HashSet::new(),
        }
    }

    /// Registers a party for monitoring.
    pub fn register_party(&mut self, party_id: PartyId) {
        self.party_states
            .insert(party_id.0.clone(), PartyState::new(party_id));
    }

    /// Advances to a new round.
    pub fn advance_round(&mut self) {
        self.current_round += 1;
    }

    /// Records a message from a party.
    pub fn record_message(&mut self, party_id: &str, sequence: u64) -> MPCResult<()> {
        if self.excluded_parties.contains(party_id) {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new(party_id),
                description: "Message from excluded party".into(),
            });
        }

        // Check for replay - need to check first before mutating
        let replay_detected = if let Some(state) = self.party_states.get(party_id) {
            if sequence < state.expected_sequence {
                Some((state.expected_sequence, sequence))
            } else {
                None
            }
        } else {
            None
        };

        if let Some((expected, received)) = replay_detected {
            self.report_fault(
                PartyId::new(party_id),
                FaultType::ReplayAttack,
                format!("Sequence {} < expected {}", received, expected),
            );
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new(party_id),
                description: "Replay attack detected".into(),
            });
        }

        // Now update the state
        if let Some(state) = self.party_states.get_mut(party_id) {
            state.record_message(sequence);
        }

        Ok(())
    }

    /// Reports a detected fault.
    pub fn report_fault(
        &mut self,
        party_id: PartyId,
        fault_type: FaultType,
        description: impl Into<String>,
    ) {
        let fault = DetectedFault::new(
            party_id.clone(),
            fault_type.clone(),
            self.current_round,
            description,
        );

        self.all_faults.push(fault.clone());

        if let Some(state) = self.party_states.get_mut(&party_id.0) {
            state.confirm_fault(fault);

            if state.fault_count >= self.max_faults {
                self.exclude_party(&party_id.0);
            }
        }
    }

    /// Reports a fault with evidence.
    pub fn report_fault_with_evidence(
        &mut self,
        party_id: PartyId,
        fault_type: FaultType,
        description: impl Into<String>,
        evidence: Vec<u8>,
    ) {
        let fault = DetectedFault::new(
            party_id.clone(),
            fault_type.clone(),
            self.current_round,
            description,
        )
        .with_evidence(evidence);

        self.all_faults.push(fault.clone());

        if let Some(state) = self.party_states.get_mut(&party_id.0) {
            state.confirm_fault(fault);

            if state.fault_count >= self.max_faults {
                self.exclude_party(&party_id.0);
            }
        }
    }

    /// Excludes a party from the protocol.
    pub fn exclude_party(&mut self, party_id: &str) {
        self.excluded_parties.insert(party_id.to_string());
        if let Some(state) = self.party_states.get_mut(party_id) {
            state.exclude();
        }
    }

    /// Checks for timed out parties.
    pub fn check_timeouts(&mut self) -> Vec<PartyId> {
        let mut timed_out = Vec::new();

        for state in self.party_states.values_mut() {
            if state.check_timeout(self.timeout) {
                timed_out.push(state.party_id.clone());
                self.all_faults.push(DetectedFault::new(
                    state.party_id.clone(),
                    FaultType::Timeout,
                    self.current_round,
                    "Party timeout",
                ));
            }
        }

        timed_out
    }

    /// Returns the status of a party.
    pub fn party_status(&self, party_id: &str) -> Option<PartyStatus> {
        self.party_states.get(party_id).map(|s| s.status)
    }

    /// Returns all active parties.
    pub fn active_parties(&self) -> Vec<PartyId> {
        self.party_states
            .values()
            .filter(|s| s.status == PartyStatus::Active)
            .map(|s| s.party_id.clone())
            .collect()
    }

    /// Returns all faults for a party.
    pub fn party_faults(&self, party_id: &str) -> Vec<&DetectedFault> {
        self.all_faults
            .iter()
            .filter(|f| f.party.0 == party_id)
            .collect()
    }

    /// Returns all excluded parties.
    pub fn excluded_parties(&self) -> &HashSet<String> {
        &self.excluded_parties
    }

    /// Checks if there are enough active parties to continue.
    pub fn has_quorum(&self, min_parties: usize) -> bool {
        self.active_parties().len() >= min_parties
    }

    /// Generates a summary of all detected faults.
    pub fn fault_summary(&self) -> HashMap<FaultType, usize> {
        let mut summary = HashMap::new();
        for fault in &self.all_faults {
            *summary.entry(fault.fault_type.clone()).or_insert(0) += 1;
        }
        summary
    }
}

/// Helper to detect specific types of Byzantine behavior.
pub struct ByzantineChecker;

impl ByzantineChecker {
    /// Checks if a party's values are consistent with previous values.
    pub fn check_value_consistency(
        previous: &[f64],
        current: &[f64],
        max_change: f64,
    ) -> Option<String> {
        if previous.len() != current.len() {
            return Some(format!(
                "Length changed: {} -> {}",
                previous.len(),
                current.len()
            ));
        }

        for (i, (p, c)) in previous.iter().zip(current.iter()).enumerate() {
            let change = (c - p).abs();
            if change > max_change {
                return Some(format!(
                    "Element {} changed too much: {} -> {} (delta {})",
                    i, p, c, change
                ));
            }
        }

        None
    }

    /// Checks if a gradient appears to be poisoned.
    pub fn check_gradient_poisoning(
        gradient: &[f64],
        mean_threshold: f64,
        max_element: f64,
        max_norm: f64,
    ) -> Option<String> {
        // Check mean
        let mean: f64 = gradient.iter().sum::<f64>() / gradient.len() as f64;
        if mean.abs() > mean_threshold {
            return Some(format!("Gradient mean {} exceeds threshold", mean.abs()));
        }

        // Check individual elements
        for (i, g) in gradient.iter().enumerate() {
            if g.abs() > max_element {
                return Some(format!("Element {} = {} exceeds max", i, g));
            }
        }

        // Check L2 norm
        let norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm > max_norm {
            return Some(format!("Gradient norm {} exceeds max", norm));
        }

        None
    }

    /// Checks if Beaver triple shares are valid.
    pub fn check_beaver_shares(
        a_shares: &[f64],
        b_shares: &[f64],
        c_shares: &[f64],
    ) -> Option<String> {
        let a: f64 = a_shares.iter().sum();
        let b: f64 = b_shares.iter().sum();
        let c: f64 = c_shares.iter().sum();

        let expected_c = a * b;
        if (c - expected_c).abs() > 1e-6 {
            return Some(format!(
                "Invalid Beaver: c={} but a*b={} (a={}, b={})",
                c, expected_c, a, b
            ));
        }

        None
    }

    /// Checks protocol message ordering.
    pub fn check_protocol_order(
        expected_phase: &str,
        received_phase: &str,
    ) -> Option<String> {
        if expected_phase != received_phase {
            return Some(format!(
                "Protocol violation: expected {} but got {}",
                expected_phase, received_phase
            ));
        }
        None
    }
}

/// Accusation for on-chain slashing.
#[derive(Debug, Clone)]
pub struct SlashingAccusation {
    /// Accused party
    pub accused: PartyId,
    /// Accuser party
    pub accuser: PartyId,
    /// Fault type
    pub fault_type: FaultType,
    /// Round number
    pub round: u64,
    /// Cryptographic proof
    pub proof: [u8; 32],
    /// Evidence data
    pub evidence: Vec<u8>,
}

impl SlashingAccusation {
    /// Creates a new slashing accusation.
    pub fn from_fault(fault: &DetectedFault, accuser: PartyId) -> Self {
        Self {
            accused: fault.party.clone(),
            accuser,
            fault_type: fault.fault_type.clone(),
            round: fault.round,
            proof: fault.generate_proof(),
            evidence: fault.evidence.clone().unwrap_or_default(),
        }
    }

    /// Serializes for on-chain submission.
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(self.accused.0.as_bytes());
        data.extend_from_slice(self.accuser.0.as_bytes());
        data.extend_from_slice(self.fault_type.to_string().as_bytes());
        data.extend_from_slice(&self.round.to_le_bytes());
        data.extend_from_slice(&self.proof);
        data.extend_from_slice(&self.evidence);
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_detection() {
        let mut detector = ByzantineDetector::new(Duration::from_secs(30), 3);

        let party0 = PartyId::from_index(0);
        let party1 = PartyId::from_index(1);

        detector.register_party(party0.clone());
        detector.register_party(party1.clone());

        // Record normal message
        assert!(detector.record_message("party-0", 0).is_ok());
        assert!(detector.record_message("party-0", 1).is_ok());

        // Replay should fail
        assert!(detector.record_message("party-0", 0).is_err());

        // Check status
        assert_eq!(detector.party_status("party-0"), Some(PartyStatus::Suspected));
    }

    #[test]
    fn test_fault_accumulation() {
        let mut detector = ByzantineDetector::new(Duration::from_secs(30), 2);

        let party = PartyId::from_index(0);
        detector.register_party(party.clone());

        // First fault
        detector.report_fault(party.clone(), FaultType::InvalidMAC, "Bad MAC");
        assert_eq!(detector.party_status("party-0"), Some(PartyStatus::Suspected));

        // Second fault - should be excluded
        detector.report_fault(party.clone(), FaultType::IncorrectComputation, "Bad result");
        assert!(detector.excluded_parties().contains("party-0"));
    }

    #[test]
    fn test_quorum() {
        let mut detector = ByzantineDetector::new(Duration::from_secs(30), 1);

        for i in 0..5 {
            detector.register_party(PartyId::from_index(i));
        }

        assert!(detector.has_quorum(3));
        assert!(detector.has_quorum(5));

        // Exclude some parties
        detector.exclude_party("party-0");
        detector.exclude_party("party-1");

        assert!(detector.has_quorum(3));
        assert!(!detector.has_quorum(5));
    }

    #[test]
    fn test_byzantine_checker() {
        // Value consistency
        let prev = vec![1.0, 2.0, 3.0];
        let curr = vec![1.1, 2.0, 3.0];
        assert!(ByzantineChecker::check_value_consistency(&prev, &curr, 0.5).is_none());

        let bad = vec![1.0, 5.0, 3.0];
        assert!(ByzantineChecker::check_value_consistency(&prev, &bad, 0.5).is_some());

        // Gradient poisoning
        let good_grad = vec![0.1, -0.1, 0.05];
        assert!(
            ByzantineChecker::check_gradient_poisoning(&good_grad, 0.5, 1.0, 2.0).is_none()
        );

        let poisoned = vec![0.1, 100.0, 0.05];
        assert!(
            ByzantineChecker::check_gradient_poisoning(&poisoned, 0.5, 1.0, 2.0).is_some()
        );

        // Beaver verification
        let a = vec![1.0, 1.0, 1.0]; // sum = 3
        let b = vec![2.0, 1.0, 1.0]; // sum = 4
        let c = vec![4.0, 4.0, 4.0]; // sum = 12 = 3 * 4
        assert!(ByzantineChecker::check_beaver_shares(&a, &b, &c).is_none());

        let bad_c = vec![4.0, 4.0, 5.0]; // sum = 13 != 12
        assert!(ByzantineChecker::check_beaver_shares(&a, &b, &bad_c).is_some());
    }

    #[test]
    fn test_slashing_accusation() {
        let fault = DetectedFault::new(
            PartyId::from_index(0),
            FaultType::InvalidMAC,
            5,
            "Party sent invalid MAC",
        )
        .with_evidence(vec![1, 2, 3, 4]);

        let accusation = SlashingAccusation::from_fault(&fault, PartyId::from_index(1));

        assert_eq!(accusation.accused.0, "party-0");
        assert_eq!(accusation.accuser.0, "party-1");
        assert_eq!(accusation.round, 5);
        assert!(!accusation.serialize().is_empty());
    }
}
