//! Hash-based share commitments.
//!
//! When shares are distributed, the dealer also publishes a commitment
//! to each share. This allows later verification that a party is using
//! their actual share (not a modified one) without revealing the share.
//!
//! Commitment scheme: H(share_data || blinding_factor)
//! where H is SHA-256.

use sha2::{Sha256, Digest};
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::types::PartyId;
use crate::sharing::tensor::TensorShare;

/// A commitment to a share value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareCommitment {
    /// The party this commitment is for.
    pub party: PartyId,
    /// The commitment hash.
    pub hash: [u8; 32],
    /// Description of what was committed.
    pub description: String,
}

impl ShareCommitment {
    /// Creates a commitment to a scalar share.
    pub fn commit_scalar(
        party: &PartyId,
        value: f64,
        blinding: &[u8; 32],
        description: impl Into<String>,
    ) -> Self {
        let hash = Self::hash_with_blinding(&value.to_le_bytes(), blinding);
        Self {
            party: party.clone(),
            hash,
            description: description.into(),
        }
    }

    /// Creates a commitment to a vector share.
    pub fn commit_vector(
        party: &PartyId,
        values: &[f64],
        blinding: &[u8; 32],
        description: impl Into<String>,
    ) -> Self {
        let mut data = Vec::with_capacity(values.len() * 8);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let hash = Self::hash_with_blinding(&data, blinding);
        Self {
            party: party.clone(),
            hash,
            description: description.into(),
        }
    }

    /// Creates a commitment to a tensor share.
    pub fn commit_tensor(
        party: &PartyId,
        tensor: &TensorShare,
        blinding: &[u8; 32],
        description: impl Into<String>,
    ) -> Self {
        Self::commit_vector(party, &tensor.data, blinding, description)
    }

    /// Verifies a commitment against a value.
    pub fn verify_scalar(&self, value: f64, blinding: &[u8; 32]) -> bool {
        let expected = Self::hash_with_blinding(&value.to_le_bytes(), blinding);
        self.hash == expected
    }

    /// Verifies a commitment against a vector.
    pub fn verify_vector(&self, values: &[f64], blinding: &[u8; 32]) -> bool {
        let mut data = Vec::with_capacity(values.len() * 8);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let expected = Self::hash_with_blinding(&data, blinding);
        self.hash == expected
    }

    /// Verifies a commitment against a tensor share.
    pub fn verify_tensor(&self, tensor: &TensorShare, blinding: &[u8; 32]) -> bool {
        self.verify_vector(&tensor.data, blinding)
    }

    /// Computes H(data || blinding).
    fn hash_with_blinding(data: &[u8], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.update(blinding);
        hasher.finalize().into()
    }
}

/// Generates random blinding factors for commitments.
pub struct BlindingGenerator {
    rng: ChaCha20Rng,
}

impl BlindingGenerator {
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Generates a random 32-byte blinding factor.
    pub fn generate(&mut self) -> [u8; 32] {
        let mut blinding = [0u8; 32];
        self.rng.fill(&mut blinding);
        blinding
    }

    /// Generates n blinding factors.
    pub fn generate_batch(&mut self, n: usize) -> Vec<[u8; 32]> {
        (0..n).map(|_| self.generate()).collect()
    }
}

impl Default for BlindingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// A set of commitments for all shares in a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCommitments {
    /// Commitments for each party's shares.
    pub per_party: Vec<Vec<ShareCommitment>>,
    /// Hash of all commitments together (for on-chain anchoring).
    pub root_hash: [u8; 32],
}

impl ModelCommitments {
    /// Creates commitments for all parties' model shares.
    pub fn create(
        party_commitments: Vec<Vec<ShareCommitment>>,
    ) -> Self {
        // Compute root hash over all commitments.
        let mut hasher = Sha256::new();
        for party_comms in &party_commitments {
            for comm in party_comms {
                hasher.update(&comm.hash);
            }
        }
        let root_hash: [u8; 32] = hasher.finalize().into();

        Self {
            per_party: party_commitments,
            root_hash,
        }
    }

    /// Verifies a specific party's commitment for a named share.
    pub fn verify_party_share(
        &self,
        party_index: usize,
        description: &str,
        values: &[f64],
        blinding: &[u8; 32],
    ) -> bool {
        if party_index >= self.per_party.len() {
            return false;
        }

        self.per_party[party_index]
            .iter()
            .find(|c| c.description == description)
            .map_or(false, |c| c.verify_vector(values, blinding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_commitment() {
        let party = PartyId::from_index(0);
        let mut gen = BlindingGenerator::with_seed(42);
        let blinding = gen.generate();

        let value = 42.0;
        let commitment = ShareCommitment::commit_scalar(&party, value, &blinding, "test");

        assert!(commitment.verify_scalar(value, &blinding));
        assert!(!commitment.verify_scalar(43.0, &blinding));

        // Wrong blinding fails.
        let wrong_blinding = gen.generate();
        assert!(!commitment.verify_scalar(value, &wrong_blinding));
    }

    #[test]
    fn test_vector_commitment() {
        let party = PartyId::from_index(1);
        let mut gen = BlindingGenerator::with_seed(42);
        let blinding = gen.generate();

        let values = vec![1.0, 2.0, 3.0];
        let commitment = ShareCommitment::commit_vector(&party, &values, &blinding, "weights");

        assert!(commitment.verify_vector(&values, &blinding));
        assert!(!commitment.verify_vector(&[1.0, 2.0, 4.0], &blinding));
    }

    #[test]
    fn test_model_commitments() {
        let mut gen = BlindingGenerator::with_seed(42);

        let party0_comms = vec![
            ShareCommitment::commit_vector(
                &PartyId::from_index(0),
                &[1.0, 2.0],
                &gen.generate(),
                "layer0",
            ),
        ];

        let party1_comms = vec![
            ShareCommitment::commit_vector(
                &PartyId::from_index(1),
                &[3.0, 4.0],
                &gen.generate(),
                "layer0",
            ),
        ];

        let mc = ModelCommitments::create(vec![party0_comms, party1_comms]);
        assert_ne!(mc.root_hash, [0u8; 32]);
        assert_eq!(mc.per_party.len(), 2);
    }
}
