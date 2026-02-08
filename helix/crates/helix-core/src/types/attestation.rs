//! Training attestation types for verifiable training participation.
//!
//! An attestation is a signed statement by a node that it correctly executed
//! a training step. Attestations chain together to form a verifiable record
//! of a node's training history within a session.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

use super::coordination::{NodeId, SessionId};

/// A signed statement attesting to the correct execution of a training step.
///
/// Each attestation is cryptographically linked to:
/// - The node that performed the step
/// - The session and round context
/// - The error commitment checksum (matching on-chain)
/// - The previous attestation (forming a chain)
///
/// The signature covers the attestation digest, which binds all fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingAttestation {
    /// The node making this attestation.
    pub node_id: NodeId,
    /// Session this attestation belongs to.
    pub session_id: SessionId,
    /// Training step number.
    pub step_number: u64,
    /// Round number within the session.
    pub round_number: u64,
    /// Error commitment checksum (compact u64, matches on-chain).
    pub error_checksum: u64,
    /// Hash of the old weights (before this step).
    pub old_weight_hash: [u8; 32],
    /// Hash of the new weights (after this step).
    pub new_weight_hash: [u8; 32],
    /// Hash of the previous attestation in the chain (zeroed for first).
    pub previous_attestation_hash: [u8; 32],
    /// Unix timestamp when attestation was created.
    pub timestamp: u64,
    /// Ed25519 signature over the attestation digest (64 bytes).
    /// None if attestation is unsigned (e.g., during construction).
    pub signature: Option<Vec<u8>>,
}

impl TrainingAttestation {
    /// Computes the attestation digest that is signed.
    ///
    /// The digest binds together all attestation fields (excluding signature)
    /// into a single 32-byte hash.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.node_id.as_bytes());
        hasher.update(self.session_id.as_bytes());
        hasher.update(self.step_number.to_le_bytes());
        hasher.update(self.round_number.to_le_bytes());
        hasher.update(self.error_checksum.to_le_bytes());
        hasher.update(self.old_weight_hash);
        hasher.update(self.new_weight_hash);
        hasher.update(self.previous_attestation_hash);
        hasher.update(self.timestamp.to_le_bytes());
        let result = hasher.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&result);
        digest
    }

    /// Returns the hash of this attestation (used as `previous_attestation_hash` for the next).
    pub fn attestation_hash(&self) -> [u8; 32] {
        self.digest()
    }

    /// Returns true if this is the first attestation in a chain.
    pub fn is_genesis(&self) -> bool {
        self.previous_attestation_hash == [0u8; 32]
    }

    /// Checks structural validity (not signature verification).
    pub fn is_structurally_valid(&self) -> bool {
        // Step 0 must be genesis
        if self.step_number == 0 && !self.is_genesis() {
            return false;
        }
        // Non-genesis must reference a previous attestation
        if self.step_number > 0 && self.is_genesis() {
            return false;
        }
        true
    }
}

impl fmt::Display for TrainingAttestation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Attestation(node={:?}, session={:?}, step={}, round={})",
            self.node_id, self.session_id, self.step_number, self.round_number
        )
    }
}

/// An ordered chain of training attestations from a single node.
///
/// The chain enforces:
/// - Sequential step numbers
/// - Consistent node identity
/// - Hash-linking between consecutive attestations
/// - Session consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationChain {
    /// The node that produced all attestations in this chain.
    pub node_id: NodeId,
    /// The session these attestations belong to.
    pub session_id: SessionId,
    /// Ordered attestations (by step number).
    attestations: Vec<TrainingAttestation>,
}

impl AttestationChain {
    /// Creates a new empty attestation chain.
    pub fn new(node_id: NodeId, session_id: SessionId) -> Self {
        Self {
            node_id,
            session_id,
            attestations: Vec::new(),
        }
    }

    /// Appends an attestation to the chain, verifying chain integrity.
    ///
    /// Returns an error string if the attestation doesn't fit the chain.
    pub fn append(&mut self, attestation: TrainingAttestation) -> Result<(), String> {
        // Check node identity
        if attestation.node_id != self.node_id {
            return Err(format!(
                "attestation node {:?} does not match chain node {:?}",
                attestation.node_id, self.node_id
            ));
        }

        // Check session
        if attestation.session_id != self.session_id {
            return Err("attestation session does not match chain session".to_string());
        }

        // Check step number is sequential
        let expected_step = self.attestations.len() as u64;
        if attestation.step_number != expected_step {
            return Err(format!(
                "expected step {}, got {}",
                expected_step, attestation.step_number
            ));
        }

        // Check hash linking
        if self.attestations.is_empty() {
            if !attestation.is_genesis() {
                return Err("first attestation must be genesis (zero previous hash)".to_string());
            }
        } else {
            let last = self.attestations.last().unwrap();
            let expected_hash = last.attestation_hash();
            if attestation.previous_attestation_hash != expected_hash {
                return Err("previous_attestation_hash does not match last attestation".to_string());
            }
        }

        self.attestations.push(attestation);
        Ok(())
    }

    /// Returns the number of attestations in the chain.
    pub fn len(&self) -> usize {
        self.attestations.len()
    }

    /// Returns true if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.attestations.is_empty()
    }

    /// Returns the attestations in order.
    pub fn attestations(&self) -> &[TrainingAttestation] {
        &self.attestations
    }

    /// Returns the last attestation, if any.
    pub fn latest(&self) -> Option<&TrainingAttestation> {
        self.attestations.last()
    }

    /// Returns the hash that the next attestation should reference.
    pub fn next_previous_hash(&self) -> [u8; 32] {
        match self.attestations.last() {
            Some(last) => last.attestation_hash(),
            None => [0u8; 32], // Genesis
        }
    }

    /// Verifies the entire chain integrity (hash linking and step sequencing).
    pub fn verify_integrity(&self) -> Result<(), String> {
        for (i, attestation) in self.attestations.iter().enumerate() {
            if attestation.node_id != self.node_id {
                return Err(format!("attestation {} has wrong node_id", i));
            }
            if attestation.session_id != self.session_id {
                return Err(format!("attestation {} has wrong session_id", i));
            }
            if attestation.step_number != i as u64 {
                return Err(format!(
                    "attestation {} has step_number {}, expected {}",
                    i, attestation.step_number, i
                ));
            }
            if i == 0 {
                if !attestation.is_genesis() {
                    return Err("first attestation is not genesis".to_string());
                }
            } else {
                let prev = &self.attestations[i - 1];
                if attestation.previous_attestation_hash != prev.attestation_hash() {
                    return Err(format!(
                        "attestation {} has broken hash link",
                        i
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node() -> NodeId {
        NodeId::from_bytes([1u8; 32])
    }

    fn test_session() -> SessionId {
        SessionId::from_bytes([10u8; 32])
    }

    fn make_attestation(step: u64, prev_hash: [u8; 32]) -> TrainingAttestation {
        TrainingAttestation {
            node_id: test_node(),
            session_id: test_session(),
            step_number: step,
            round_number: step,
            error_checksum: 12345 + step,
            old_weight_hash: [2u8; 32],
            new_weight_hash: [3u8; 32],
            previous_attestation_hash: prev_hash,
            timestamp: 1700000000 + step,
            signature: None,
        }
    }

    #[test]
    fn test_attestation_digest_deterministic() {
        let a = make_attestation(0, [0u8; 32]);
        assert_eq!(a.digest(), a.digest());
    }

    #[test]
    fn test_attestation_digest_changes_with_step() {
        let a = make_attestation(0, [0u8; 32]);
        let b = make_attestation(1, [0u8; 32]);
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn test_genesis_attestation() {
        let a = make_attestation(0, [0u8; 32]);
        assert!(a.is_genesis());
        assert!(a.is_structurally_valid());
    }

    #[test]
    fn test_non_genesis_with_zero_hash_invalid() {
        let a = make_attestation(1, [0u8; 32]);
        assert!(!a.is_structurally_valid());
    }

    #[test]
    fn test_attestation_chain_happy_path() {
        let mut chain = AttestationChain::new(test_node(), test_session());
        assert!(chain.is_empty());

        // Add genesis
        let a0 = make_attestation(0, [0u8; 32]);
        chain.append(a0.clone()).unwrap();
        assert_eq!(chain.len(), 1);

        // Add second
        let a1 = make_attestation(1, a0.attestation_hash());
        chain.append(a1.clone()).unwrap();
        assert_eq!(chain.len(), 2);

        // Add third
        let a2 = make_attestation(2, a1.attestation_hash());
        chain.append(a2).unwrap();
        assert_eq!(chain.len(), 3);

        // Verify integrity
        chain.verify_integrity().unwrap();
    }

    #[test]
    fn test_attestation_chain_rejects_wrong_node() {
        let mut chain = AttestationChain::new(test_node(), test_session());

        let mut bad = make_attestation(0, [0u8; 32]);
        bad.node_id = NodeId::from_bytes([99u8; 32]);

        assert!(chain.append(bad).is_err());
    }

    #[test]
    fn test_attestation_chain_rejects_wrong_step() {
        let mut chain = AttestationChain::new(test_node(), test_session());

        // Try to add step 1 without step 0
        let a1 = make_attestation(1, [0u8; 32]);
        assert!(chain.append(a1).is_err());
    }

    #[test]
    fn test_attestation_chain_rejects_broken_hash_link() {
        let mut chain = AttestationChain::new(test_node(), test_session());

        let a0 = make_attestation(0, [0u8; 32]);
        chain.append(a0).unwrap();

        // Wrong previous hash
        let a1 = make_attestation(1, [0xff; 32]);
        assert!(chain.append(a1).is_err());
    }

    #[test]
    fn test_attestation_chain_next_previous_hash() {
        let mut chain = AttestationChain::new(test_node(), test_session());
        assert_eq!(chain.next_previous_hash(), [0u8; 32]); // Genesis

        let a0 = make_attestation(0, [0u8; 32]);
        let expected = a0.attestation_hash();
        chain.append(a0).unwrap();
        assert_eq!(chain.next_previous_hash(), expected);
    }

    #[test]
    fn test_attestation_display() {
        let a = make_attestation(0, [0u8; 32]);
        let s = format!("{}", a);
        assert!(s.contains("step=0"));
        assert!(s.contains("round=0"));
    }
}
