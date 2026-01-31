//! MPC-specific Zero-Knowledge Proofs.
//!
//! This module provides ZK proofs specialized for MPC operations:
//!
//! - **Share Validity**: Prove that a share is valid (correctly formed,
//!   matches commitment) without revealing the share value
//! - **Aggregation**: Prove that gradient aggregation was performed correctly
//! - **MAC Verification**: Prove that MACs are valid in ZK
//! - **Batched Proofs**: Efficient batch verification of multiple proofs
//!
//! # Security Properties
//!
//! These proofs provide:
//! - **Soundness**: Invalid proofs are rejected with overwhelming probability
//! - **Zero-Knowledge**: Proofs reveal nothing about secret values
//! - **Composability**: Proofs can be combined for complex statements

pub mod aggregation;
pub mod mac_verification;
pub mod share_validity;

pub use aggregation::{
    AggregationProof, AggregationProver, AggregationVerifier, GradientAggregationWitness,
    GradientShareInput,
};
pub use mac_verification::{MACProof, MACProver, MACVerifier as ZKMACVerifier, MACWitness};
pub use share_validity::{
    ShareValidityProof, ShareValidityProver, ShareValidityVerifier, ShareValidityWitness,
};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use sha2::{Digest, Sha256};

/// Common trait for all MPC proofs.
pub trait MPCProof: Sized {
    /// The witness type for this proof.
    type Witness;

    /// Returns the size of the proof in bytes.
    fn size(&self) -> usize;

    /// Serializes the proof to bytes.
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserializes a proof from bytes.
    fn from_bytes(bytes: &[u8]) -> MPCResult<Self>;

    /// Returns the proof type identifier.
    fn proof_type(&self) -> ProofType;
}

/// Types of MPC proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofType {
    /// Share validity proof.
    ShareValidity,
    /// Gradient aggregation proof.
    Aggregation,
    /// MAC verification proof.
    MACVerification,
    /// Batched proof.
    Batched,
}

/// A batched collection of proofs for efficient verification.
#[derive(Debug, Clone)]
pub struct BatchedProof {
    /// Individual proofs.
    pub proofs: Vec<BatchProofEntry>,
    /// Random challenges for batching.
    pub challenges: Vec<Fr>,
    /// Aggregated commitment.
    pub aggregate_commitment: [u8; 32],
}

/// An entry in a batched proof.
#[derive(Debug, Clone)]
pub struct BatchProofEntry {
    /// Proof type.
    pub proof_type: ProofType,
    /// Serialized proof.
    pub proof_data: Vec<u8>,
    /// Public inputs for this proof.
    pub public_inputs: Vec<Fr>,
}

impl BatchedProof {
    /// Creates a new batched proof.
    pub fn new() -> Self {
        Self {
            proofs: Vec::new(),
            challenges: Vec::new(),
            aggregate_commitment: [0u8; 32],
        }
    }

    /// Adds a share validity proof to the batch.
    pub fn add_share_validity(&mut self, proof: ShareValidityProof) {
        self.proofs.push(BatchProofEntry {
            proof_type: ProofType::ShareValidity,
            proof_data: proof.to_bytes(),
            public_inputs: proof.public_inputs(),
        });
    }

    /// Adds an aggregation proof to the batch.
    pub fn add_aggregation(&mut self, proof: AggregationProof) {
        self.proofs.push(BatchProofEntry {
            proof_type: ProofType::Aggregation,
            proof_data: proof.to_bytes(),
            public_inputs: proof.public_inputs(),
        });
    }

    /// Adds a MAC verification proof to the batch.
    pub fn add_mac_verification(&mut self, proof: MACProof) {
        self.proofs.push(BatchProofEntry {
            proof_type: ProofType::MACVerification,
            proof_data: proof.to_bytes(),
            public_inputs: proof.public_inputs(),
        });
    }

    /// Generates random challenges for batched verification.
    pub fn generate_challenges(&mut self, seed: u64) {
        use rand::{RngCore, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        self.challenges = (0..self.proofs.len())
            .map(|_| Fr::random(&mut rng))
            .collect();
    }

    /// Computes the aggregate commitment.
    pub fn compute_aggregate(&mut self) {
        let mut hasher = Sha256::new();

        for (entry, challenge) in self.proofs.iter().zip(self.challenges.iter()) {
            hasher.update(&[entry.proof_type as u8]);
            hasher.update(&entry.proof_data);
            hasher.update(&challenge.to_bytes_le());
        }

        self.aggregate_commitment = hasher.finalize().into();
    }

    /// Returns the number of proofs in the batch.
    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    /// Checks if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }

    /// Total size of all proofs in bytes.
    pub fn total_size(&self) -> usize {
        self.proofs.iter().map(|e| e.proof_data.len()).sum()
    }
}

impl Default for BatchedProof {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch verifier for multiple proofs.
pub struct BatchVerifier {
    /// Share validity verifier.
    share_verifier: ShareValidityVerifier,
    /// Aggregation verifier.
    agg_verifier: AggregationVerifier,
    /// MAC verifier.
    mac_verifier: ZKMACVerifier,
}

impl BatchVerifier {
    /// Creates a new batch verifier.
    pub fn new() -> Self {
        Self {
            share_verifier: ShareValidityVerifier::new(),
            agg_verifier: AggregationVerifier::new(),
            mac_verifier: ZKMACVerifier::new(),
        }
    }

    /// Verifies a batched proof.
    pub fn verify(&self, batch: &BatchedProof) -> MPCResult<bool> {
        if batch.challenges.len() != batch.proofs.len() {
            return Err(MPCError::ProtocolError(
                "Challenge count mismatch".into(),
            ));
        }

        // Verify each proof individually.
        for entry in &batch.proofs {
            let valid = match entry.proof_type {
                ProofType::ShareValidity => {
                    let proof = ShareValidityProof::from_bytes(&entry.proof_data)?;
                    self.share_verifier.verify(&proof)?
                }
                ProofType::Aggregation => {
                    let proof = AggregationProof::from_bytes(&entry.proof_data)?;
                    self.agg_verifier.verify(&proof)?
                }
                ProofType::MACVerification => {
                    let proof = MACProof::from_bytes(&entry.proof_data)?;
                    self.mac_verifier.verify(&proof)?
                }
                ProofType::Batched => {
                    // Recursive batched proofs not supported.
                    return Err(MPCError::ProtocolError(
                        "Nested batched proofs not supported".into(),
                    ));
                }
            };

            if !valid {
                return Ok(false);
            }
        }

        // Verify aggregate commitment.
        let mut hasher = Sha256::new();
        for (entry, challenge) in batch.proofs.iter().zip(batch.challenges.iter()) {
            hasher.update(&[entry.proof_type as u8]);
            hasher.update(&entry.proof_data);
            hasher.update(&challenge.to_bytes_le());
        }
        let expected: [u8; 32] = hasher.finalize().into();

        Ok(expected == batch.aggregate_commitment)
    }
}

impl Default for BatchVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about proof generation.
#[derive(Debug, Clone, Default)]
pub struct ProofStats {
    /// Number of proofs generated.
    pub num_proofs: usize,
    /// Total generation time in milliseconds.
    pub total_time_ms: u64,
    /// Total proof size in bytes.
    pub total_size_bytes: usize,
    /// Average proof size.
    pub avg_size_bytes: usize,
    /// Average generation time.
    pub avg_time_ms: u64,
}

impl ProofStats {
    /// Creates new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a proof generation.
    pub fn record(&mut self, size: usize, time_ms: u64) {
        self.num_proofs += 1;
        self.total_size_bytes += size;
        self.total_time_ms += time_ms;

        if self.num_proofs > 0 {
            self.avg_size_bytes = self.total_size_bytes / self.num_proofs;
            self.avg_time_ms = self.total_time_ms / self.num_proofs as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_proof() {
        let mut batch = BatchedProof::new();
        assert!(batch.is_empty());

        // Add some mock proofs.
        batch.proofs.push(BatchProofEntry {
            proof_type: ProofType::ShareValidity,
            proof_data: vec![1, 2, 3],
            public_inputs: vec![Fr::ZERO],
        });

        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());

        batch.generate_challenges(42);
        assert_eq!(batch.challenges.len(), 1);

        batch.compute_aggregate();
        assert_ne!(batch.aggregate_commitment, [0u8; 32]);
    }

    #[test]
    fn test_proof_stats() {
        let mut stats = ProofStats::new();

        stats.record(1000, 100);
        stats.record(2000, 200);

        assert_eq!(stats.num_proofs, 2);
        assert_eq!(stats.total_size_bytes, 3000);
        assert_eq!(stats.total_time_ms, 300);
        assert_eq!(stats.avg_size_bytes, 1500);
        assert_eq!(stats.avg_time_ms, 150);
    }
}
