//! Proof chain type for state hash continuity validation.
//!
//! A `ProofChain` maintains an ordered sequence of proof entries where each
//! entry's `new_weight_hash` must equal the next entry's `old_weight_hash`,
//! forming an unbroken chain of state transitions.
//!
//! This ensures that no training steps are skipped, reordered, or tampered
//! with between consecutive ZK proofs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::coordination::SessionId;

/// A single entry in a proof chain.
///
/// Each entry represents one verified training step with its state hashes
/// and associated metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofEntry {
    /// Training step number.
    pub step_number: u64,
    /// Weight hash before this step.
    pub old_weight_hash: [u8; 32],
    /// Weight hash after this step.
    pub new_weight_hash: [u8; 32],
    /// Hash of the ZK proof for this step.
    pub proof_hash: [u8; 32],
    /// Error bound for this step.
    pub error_bound: f64,
    /// Error checksum (compact u64, contract-compatible).
    pub error_checksum: u64,
    /// When this proof was verified (unix timestamp).
    pub verified_at: u64,
    /// On-chain transaction hash, if submitted.
    pub tx_hash: Option<[u8; 32]>,
}

impl ProofEntry {
    /// Computes a deterministic hash of this entry for chain integrity.
    pub fn entry_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.step_number.to_le_bytes());
        hasher.update(self.old_weight_hash);
        hasher.update(self.new_weight_hash);
        hasher.update(self.proof_hash);
        hasher.update(self.error_bound.to_le_bytes());
        hasher.update(self.error_checksum.to_le_bytes());
        hasher.update(self.verified_at.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// Result of a proof chain integrity verification.
#[derive(Debug, Clone)]
pub struct ChainVerification {
    /// Whether the chain is fully valid.
    pub valid: bool,
    /// Number of entries verified.
    pub entries_checked: usize,
    /// Any continuity gaps found (step numbers where new_hash[i] != old_hash[i+1]).
    pub gaps: Vec<ContinuityGap>,
    /// Step number sequence errors (out-of-order or duplicated steps).
    pub sequence_errors: Vec<SequenceError>,
}

/// A continuity gap between two consecutive proof entries.
#[derive(Debug, Clone)]
pub struct ContinuityGap {
    /// Step number of the earlier entry.
    pub step_before: u64,
    /// Step number of the later entry.
    pub step_after: u64,
    /// The new_weight_hash from the earlier entry.
    pub expected_hash: [u8; 32],
    /// The old_weight_hash from the later entry.
    pub actual_hash: [u8; 32],
}

/// A sequence error in the proof chain.
#[derive(Debug, Clone)]
pub struct SequenceError {
    /// Index in the chain where the error was found.
    pub index: usize,
    /// Expected step number.
    pub expected_step: u64,
    /// Actual step number found.
    pub actual_step: u64,
}

/// An ordered chain of ZK proof entries with state hash continuity validation.
///
/// Ensures that `new_weight_hash[i] == old_weight_hash[i+1]` for all consecutive
/// pairs, forming a verifiable chain of training state transitions.
///
/// The chain also validates:
/// - Sequential step numbering (no gaps or reordering)
/// - No duplicate proof entries
/// - Consistent session context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofChain {
    /// Session this chain belongs to.
    pub session_id: SessionId,
    /// Model identifier.
    pub model_id: [u8; 32],
    /// Initial weight hash (genesis state).
    genesis_hash: [u8; 32],
    /// Ordered proof entries.
    entries: Vec<ProofEntry>,
    /// Set of seen proof hashes for replay detection.
    seen_proof_hashes: Vec<[u8; 32]>,
}

impl ProofChain {
    /// Creates a new proof chain with the given genesis (initial) weight hash.
    pub fn new(
        session_id: SessionId,
        model_id: [u8; 32],
        genesis_hash: [u8; 32],
    ) -> Self {
        Self {
            session_id,
            model_id,
            genesis_hash,
            entries: Vec::new(),
            seen_proof_hashes: Vec::new(),
        }
    }

    /// Returns the genesis (initial) weight hash.
    pub fn genesis_hash(&self) -> [u8; 32] {
        self.genesis_hash
    }

    /// Returns the current head weight hash (new_weight_hash of the last entry,
    /// or genesis_hash if empty).
    pub fn head_hash(&self) -> [u8; 32] {
        self.entries.last()
            .map(|e| e.new_weight_hash)
            .unwrap_or(self.genesis_hash)
    }

    /// Returns the number of entries in the chain.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the entries in order.
    pub fn entries(&self) -> &[ProofEntry] {
        &self.entries
    }

    /// Returns the expected step number for the next entry.
    pub fn next_step(&self) -> u64 {
        self.entries.last()
            .map(|e| e.step_number + 1)
            .unwrap_or(0)
    }

    /// Appends a proof entry to the chain, validating continuity.
    ///
    /// Checks:
    /// 1. Step number is sequential
    /// 2. old_weight_hash matches previous new_weight_hash (or genesis)
    /// 3. Proof hash hasn't been seen before (replay detection)
    pub fn append(&mut self, entry: ProofEntry) -> Result<(), String> {
        // Check step number
        let expected_step = self.next_step();
        if entry.step_number != expected_step {
            return Err(format!(
                "expected step {}, got {}",
                expected_step, entry.step_number
            ));
        }

        // Check state hash continuity
        let expected_old_hash = self.head_hash();
        if entry.old_weight_hash != expected_old_hash {
            return Err(format!(
                "state hash discontinuity at step {}: expected old_hash {:02x}{:02x}..., got {:02x}{:02x}...",
                entry.step_number,
                expected_old_hash[0], expected_old_hash[1],
                entry.old_weight_hash[0], entry.old_weight_hash[1],
            ));
        }

        // Check for proof replay
        if self.seen_proof_hashes.contains(&entry.proof_hash) {
            return Err(format!(
                "duplicate proof hash at step {}",
                entry.step_number
            ));
        }

        self.seen_proof_hashes.push(entry.proof_hash);
        self.entries.push(entry);
        Ok(())
    }

    /// Verifies the full chain integrity.
    ///
    /// Returns a detailed verification report including any gaps or errors.
    pub fn verify(&self) -> ChainVerification {
        let mut gaps = Vec::new();
        let mut sequence_errors = Vec::new();

        for (i, entry) in self.entries.iter().enumerate() {
            // Check step sequence
            let expected_step = i as u64;
            if entry.step_number != expected_step {
                sequence_errors.push(SequenceError {
                    index: i,
                    expected_step,
                    actual_step: entry.step_number,
                });
            }

            // Check state hash continuity
            let expected_old = if i == 0 {
                self.genesis_hash
            } else {
                self.entries[i - 1].new_weight_hash
            };

            if entry.old_weight_hash != expected_old {
                gaps.push(ContinuityGap {
                    step_before: if i == 0 { 0 } else { self.entries[i - 1].step_number },
                    step_after: entry.step_number,
                    expected_hash: expected_old,
                    actual_hash: entry.old_weight_hash,
                });
            }
        }

        let valid = gaps.is_empty() && sequence_errors.is_empty();
        ChainVerification {
            valid,
            entries_checked: self.entries.len(),
            gaps,
            sequence_errors,
        }
    }

    /// Computes a Merkle-like hash over the entire chain for compact verification.
    pub fn chain_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.session_id.as_bytes());
        hasher.update(self.model_id);
        hasher.update(self.genesis_hash);
        hasher.update((self.entries.len() as u64).to_le_bytes());
        for entry in &self.entries {
            hasher.update(entry.entry_hash());
        }
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Returns the total accumulated error across all entries.
    pub fn total_error(&self) -> f64 {
        self.entries.iter().map(|e| e.error_bound).sum()
    }

    /// Returns the latest error checksum, or 0 if empty.
    pub fn latest_error_checksum(&self) -> u64 {
        self.entries.last().map(|e| e.error_checksum).unwrap_or(0)
    }

    /// Extracts the sequence of weight hashes (genesis -> all new_weight_hashes).
    pub fn weight_hash_sequence(&self) -> Vec<[u8; 32]> {
        let mut seq = Vec::with_capacity(self.entries.len() + 1);
        seq.push(self.genesis_hash);
        for entry in &self.entries {
            seq.push(entry.new_weight_hash);
        }
        seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_session() -> SessionId {
        SessionId::from_bytes([10u8; 32])
    }

    fn make_entry(step: u64, old_hash: [u8; 32], new_hash: [u8; 32]) -> ProofEntry {
        ProofEntry {
            step_number: step,
            old_weight_hash: old_hash,
            new_weight_hash: new_hash,
            proof_hash: {
                // Deterministic but unique proof hash per step
                let mut h = [0u8; 32];
                h[0..8].copy_from_slice(&step.to_le_bytes());
                h[8] = 0xAA;
                h
            },
            error_bound: 0.001,
            error_checksum: 12345 + step,
            verified_at: 1700000000 + step,
            tx_hash: None,
        }
    }

    #[test]
    fn test_proof_chain_basic() {
        let genesis = [0xAA; 32];
        let chain = ProofChain::new(test_session(), [1u8; 32], genesis);

        assert!(chain.is_empty());
        assert_eq!(chain.head_hash(), genesis);
        assert_eq!(chain.next_step(), 0);
    }

    #[test]
    fn test_proof_chain_append() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];
        let hash_2 = [0xCC; 32];

        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);

        chain.append(make_entry(0, genesis, hash_1)).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.head_hash(), hash_1);

        chain.append(make_entry(1, hash_1, hash_2)).unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.head_hash(), hash_2);
    }

    #[test]
    fn test_proof_chain_continuity_check() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];

        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);
        chain.append(make_entry(0, genesis, hash_1)).unwrap();

        // Discontinuous: old_hash doesn't match previous new_hash
        let bad_entry = make_entry(1, [0xFF; 32], [0xDD; 32]);
        let result = chain.append(bad_entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("discontinuity"));
    }

    #[test]
    fn test_proof_chain_step_sequence() {
        let genesis = [0xAA; 32];
        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);

        // Skip step 0, try to add step 1
        let entry = make_entry(1, genesis, [0xBB; 32]);
        let result = chain.append(entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expected step 0"));
    }

    #[test]
    fn test_proof_chain_replay_detection() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];
        let hash_2 = [0xCC; 32];

        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);

        let entry0 = make_entry(0, genesis, hash_1);
        chain.append(entry0.clone()).unwrap();

        // Try to replay with same proof hash
        let mut replay = make_entry(1, hash_1, hash_2);
        replay.proof_hash = entry0.proof_hash; // Same proof hash
        let result = chain.append(replay);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate proof"));
    }

    #[test]
    fn test_proof_chain_verify_valid() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];
        let hash_2 = [0xCC; 32];

        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);
        chain.append(make_entry(0, genesis, hash_1)).unwrap();
        chain.append(make_entry(1, hash_1, hash_2)).unwrap();

        let verification = chain.verify();
        assert!(verification.valid);
        assert_eq!(verification.entries_checked, 2);
        assert!(verification.gaps.is_empty());
        assert!(verification.sequence_errors.is_empty());
    }

    #[test]
    fn test_proof_chain_chain_hash() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];

        let mut chain1 = ProofChain::new(test_session(), [1u8; 32], genesis);
        chain1.append(make_entry(0, genesis, hash_1)).unwrap();

        let mut chain2 = ProofChain::new(test_session(), [1u8; 32], genesis);
        chain2.append(make_entry(0, genesis, hash_1)).unwrap();

        // Same chain should produce same hash
        assert_eq!(chain1.chain_hash(), chain2.chain_hash());

        // Different chain should produce different hash
        let mut chain3 = ProofChain::new(test_session(), [2u8; 32], genesis);
        chain3.append(make_entry(0, genesis, hash_1)).unwrap();
        assert_ne!(chain1.chain_hash(), chain3.chain_hash());
    }

    #[test]
    fn test_proof_chain_total_error() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];
        let hash_2 = [0xCC; 32];

        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);
        chain.append(make_entry(0, genesis, hash_1)).unwrap();
        chain.append(make_entry(1, hash_1, hash_2)).unwrap();

        // Each entry has error_bound = 0.001
        assert!((chain.total_error() - 0.002).abs() < 1e-15);
    }

    #[test]
    fn test_proof_chain_weight_hash_sequence() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];
        let hash_2 = [0xCC; 32];

        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);
        chain.append(make_entry(0, genesis, hash_1)).unwrap();
        chain.append(make_entry(1, hash_1, hash_2)).unwrap();

        let seq = chain.weight_hash_sequence();
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[0], genesis);
        assert_eq!(seq[1], hash_1);
        assert_eq!(seq[2], hash_2);
    }

    #[test]
    fn test_proof_chain_empty_verify() {
        let chain = ProofChain::new(test_session(), [1u8; 32], [0xAA; 32]);
        let verification = chain.verify();
        assert!(verification.valid);
        assert_eq!(verification.entries_checked, 0);
    }

    #[test]
    fn test_proof_chain_long_chain() {
        let mut chain = ProofChain::new(test_session(), [1u8; 32], [0u8; 32]);
        let mut current_hash = [0u8; 32];

        for step in 0..100 {
            let mut next_hash = [0u8; 32];
            next_hash[0..8].copy_from_slice(&(step + 1u64).to_le_bytes());

            chain.append(make_entry(step, current_hash, next_hash)).unwrap();
            current_hash = next_hash;
        }

        assert_eq!(chain.len(), 100);
        let verification = chain.verify();
        assert!(verification.valid);
        assert_eq!(verification.entries_checked, 100);
    }

    #[test]
    fn test_proof_entry_hash_deterministic() {
        let entry = make_entry(0, [0xAA; 32], [0xBB; 32]);
        assert_eq!(entry.entry_hash(), entry.entry_hash());
    }

    #[test]
    fn test_proof_entry_hash_uniqueness() {
        let e1 = make_entry(0, [0xAA; 32], [0xBB; 32]);
        let e2 = make_entry(1, [0xBB; 32], [0xCC; 32]);
        assert_ne!(e1.entry_hash(), e2.entry_hash());
    }

    #[test]
    fn test_latest_error_checksum() {
        let genesis = [0xAA; 32];
        let hash_1 = [0xBB; 32];

        let chain_empty = ProofChain::new(test_session(), [1u8; 32], genesis);
        assert_eq!(chain_empty.latest_error_checksum(), 0);

        let mut chain = ProofChain::new(test_session(), [1u8; 32], genesis);
        chain.append(make_entry(0, genesis, hash_1)).unwrap();
        assert_eq!(chain.latest_error_checksum(), 12345); // 12345 + 0
    }
}
