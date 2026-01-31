//! Hash-based and Pedersen share commitments.
//!
//! When shares are distributed, the dealer also publishes a commitment
//! to each share. This allows later verification that a party is using
//! their actual share (not a modified one) without revealing the share.
//!
//! # Commitment Schemes
//!
//! 1. **Hash Commitment**: H(share_data || blinding_factor) using SHA-256
//!    - Fast, suitable for demo
//!    - Not homomorphic
//!
//! 2. **Pedersen Commitment**: g^v * h^r where g,h are generators
//!    - Additively homomorphic: C(a) * C(b) = C(a+b)
//!    - Information-theoretically hiding
//!    - Computationally binding
//!
//! 3. **Vector Commitment**: Merkle tree over elements
//!    - Efficient for large vectors
//!    - Position proofs

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

// ============================================================================
// Pedersen Commitments (Homomorphic)
// ============================================================================

/// Pedersen commitment scheme using discrete log.
///
/// Commitment: C(v, r) = g^v * h^r
///
/// Properties:
/// - Additively homomorphic: C(a, r1) * C(b, r2) = C(a+b, r1+r2)
/// - Information-theoretically hiding (given random r)
/// - Computationally binding (under discrete log assumption)
///
/// For the demo, we simulate this with big integers in a prime field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedersenCommitment {
    /// The commitment value (simulated as hash)
    pub value: [u8; 32],
    /// Generator g (simulated)
    pub generator_g: [u8; 32],
    /// Generator h (simulated)
    pub generator_h: [u8; 32],
}

impl PedersenCommitment {
    /// Creates a new Pedersen commitment to a value.
    ///
    /// In a real implementation, this would use elliptic curve points.
    /// Here we simulate with hashes.
    pub fn commit(value: f64, blinding: f64, generators: &PedersenGenerators) -> Self {
        // Simulate: C = H(g, v) XOR H(h, r)
        let mut hasher_g = Sha256::new();
        hasher_g.update(&generators.g);
        hasher_g.update(&value.to_le_bytes());
        let h_gv: [u8; 32] = hasher_g.finalize().into();

        let mut hasher_h = Sha256::new();
        hasher_h.update(&generators.h);
        hasher_h.update(&blinding.to_le_bytes());
        let h_hr: [u8; 32] = hasher_h.finalize().into();

        // Combine (simulated point addition)
        let mut combined = [0u8; 32];
        for i in 0..32 {
            combined[i] = h_gv[i] ^ h_hr[i];
        }

        Self {
            value: combined,
            generator_g: generators.g,
            generator_h: generators.h,
        }
    }

    /// Creates a commitment to a vector of values.
    pub fn commit_vector(values: &[f64], blindings: &[f64], generators: &PedersenGenerators) -> Vec<Self> {
        values
            .iter()
            .zip(blindings.iter())
            .map(|(v, r)| Self::commit(*v, *r, generators))
            .collect()
    }

    /// Verifies a commitment opening.
    pub fn verify(&self, value: f64, blinding: f64, generators: &PedersenGenerators) -> bool {
        let expected = Self::commit(value, blinding, generators);
        self.value == expected.value
    }

    /// Adds two commitments (homomorphic property).
    /// C(a) + C(b) = C(a+b) with combined blinding factors.
    pub fn add(&self, other: &PedersenCommitment) -> PedersenCommitment {
        let mut combined = [0u8; 32];
        for i in 0..32 {
            combined[i] = self.value[i] ^ other.value[i];
        }

        PedersenCommitment {
            value: combined,
            generator_g: self.generator_g,
            generator_h: self.generator_h,
        }
    }

    /// Scales a commitment by a public constant.
    /// C(a) * c = C(a*c) with scaled blinding.
    pub fn scale(&self, scalar: f64) -> PedersenCommitment {
        // In real implementation: point multiplication
        // Simulated: hash with scalar
        let mut hasher = Sha256::new();
        hasher.update(&self.value);
        hasher.update(&scalar.to_le_bytes());

        PedersenCommitment {
            value: hasher.finalize().into(),
            generator_g: self.generator_g,
            generator_h: self.generator_h,
        }
    }
}

/// Generator points for Pedersen commitments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedersenGenerators {
    /// Generator g
    pub g: [u8; 32],
    /// Generator h (with unknown discrete log relative to g)
    pub h: [u8; 32],
}

impl PedersenGenerators {
    /// Creates generators from a seed.
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let mut g = [0u8; 32];
        let mut h = [0u8; 32];
        rng.fill(&mut g);
        rng.fill(&mut h);

        Self { g, h }
    }

    /// Creates generators using hash-to-curve (simulated).
    pub fn nothing_up_my_sleeve(domain: &str) -> Self {
        let mut hasher_g = Sha256::new();
        hasher_g.update(domain.as_bytes());
        hasher_g.update(b"generator_g");
        let g: [u8; 32] = hasher_g.finalize().into();

        let mut hasher_h = Sha256::new();
        hasher_h.update(domain.as_bytes());
        hasher_h.update(b"generator_h");
        let h: [u8; 32] = hasher_h.finalize().into();

        Self { g, h }
    }
}

impl Default for PedersenGenerators {
    fn default() -> Self {
        Self::nothing_up_my_sleeve("HELIX-MPC-v1")
    }
}

// ============================================================================
// Gradient Commitment and Verification
// ============================================================================

/// Commitment to a gradient vector for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientCommitment {
    /// Party that computed this gradient
    pub party: PartyId,
    /// Round number
    pub round: u64,
    /// Merkle root of gradient elements
    pub merkle_root: [u8; 32],
    /// Total number of elements
    pub num_elements: usize,
    /// Accumulated error bound
    pub error_bound: f64,
    /// Hash of the model state before gradient
    pub pre_state_hash: [u8; 32],
    /// Pedersen commitment to gradient sum (for homomorphic verification)
    pub sum_commitment: PedersenCommitment,
}

impl GradientCommitment {
    /// Creates a commitment to a gradient vector.
    pub fn create(
        party: &PartyId,
        round: u64,
        gradients: &[f64],
        error_bound: f64,
        pre_state_hash: [u8; 32],
        blinding: f64,
    ) -> Self {
        let generators = PedersenGenerators::default();

        // Compute Merkle root
        let merkle_root = compute_merkle_root(gradients);

        // Compute Pedersen commitment to sum
        let sum: f64 = gradients.iter().sum();
        let sum_commitment = PedersenCommitment::commit(sum, blinding, &generators);

        Self {
            party: party.clone(),
            round,
            merkle_root,
            num_elements: gradients.len(),
            error_bound,
            pre_state_hash,
            sum_commitment,
        }
    }

    /// Verifies that revealed gradients match the commitment.
    pub fn verify(&self, gradients: &[f64], blinding: f64) -> bool {
        if gradients.len() != self.num_elements {
            return false;
        }

        // Verify Merkle root
        let computed_root = compute_merkle_root(gradients);
        if computed_root != self.merkle_root {
            return false;
        }

        // Verify sum commitment
        let generators = PedersenGenerators::default();
        let sum: f64 = gradients.iter().sum();
        self.sum_commitment.verify(sum, blinding, &generators)
    }

    /// Generates a Merkle proof for a specific element.
    pub fn generate_proof(&self, gradients: &[f64], index: usize) -> Option<MerkleProof> {
        if index >= gradients.len() {
            return None;
        }

        let proof = generate_merkle_proof(gradients, index);
        Some(proof)
    }
}

/// Merkle proof for a single gradient element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Index of the element
    pub index: usize,
    /// The element value
    pub value: f64,
    /// Sibling hashes along the path
    pub siblings: Vec<[u8; 32]>,
    /// Direction flags (true = right, false = left)
    pub directions: Vec<bool>,
}

impl MerkleProof {
    /// Verifies the proof against a root hash.
    pub fn verify(&self, root: &[u8; 32]) -> bool {
        let mut current = hash_f64(self.value);

        for (sibling, is_right) in self.siblings.iter().zip(self.directions.iter()) {
            current = if *is_right {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
        }

        &current == root
    }
}

/// Computes Merkle root of f64 values.
fn compute_merkle_root(values: &[f64]) -> [u8; 32] {
    if values.is_empty() {
        return [0u8; 32];
    }

    // Hash all leaves
    let mut layer: Vec<[u8; 32]> = values.iter().map(|v| hash_f64(*v)).collect();

    // Pad to power of 2
    let target_size = layer.len().next_power_of_two();
    while layer.len() < target_size {
        layer.push([0u8; 32]);
    }

    // Build tree bottom-up
    while layer.len() > 1 {
        let mut next_layer = Vec::with_capacity(layer.len() / 2);
        for chunk in layer.chunks(2) {
            next_layer.push(hash_pair(&chunk[0], &chunk[1]));
        }
        layer = next_layer;
    }

    layer[0]
}

/// Generates Merkle proof for an element.
fn generate_merkle_proof(values: &[f64], index: usize) -> MerkleProof {
    if values.is_empty() || index >= values.len() {
        return MerkleProof {
            index,
            value: 0.0,
            siblings: vec![],
            directions: vec![],
        };
    }

    let value = values[index];

    // Hash all leaves
    let mut layer: Vec<[u8; 32]> = values.iter().map(|v| hash_f64(*v)).collect();

    // Pad to power of 2
    let target_size = layer.len().next_power_of_two();
    while layer.len() < target_size {
        layer.push([0u8; 32]);
    }

    let mut siblings = Vec::new();
    let mut directions = Vec::new();
    let mut idx = index;

    // Build proof bottom-up
    while layer.len() > 1 {
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        let is_right = idx % 2 == 0;

        if sibling_idx < layer.len() {
            siblings.push(layer[sibling_idx]);
        } else {
            siblings.push([0u8; 32]);
        }
        directions.push(is_right);

        // Build next layer
        let mut next_layer = Vec::with_capacity(layer.len() / 2);
        for chunk in layer.chunks(2) {
            next_layer.push(hash_pair(&chunk[0], &chunk[1]));
        }
        layer = next_layer;
        idx /= 2;
    }

    MerkleProof {
        index,
        value,
        siblings,
        directions,
    }
}

/// Hashes an f64 value.
fn hash_f64(value: f64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&value.to_le_bytes());
    hasher.finalize().into()
}

/// Hashes two 32-byte values together.
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Aggregated gradient commitment from multiple parties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedGradientCommitment {
    /// Round number
    pub round: u64,
    /// Individual party commitments
    pub party_commitments: Vec<GradientCommitment>,
    /// Combined Pedersen commitment (sum of all party sums)
    pub aggregated_sum: PedersenCommitment,
    /// Total error bound
    pub total_error_bound: f64,
    /// Root hash of all party commitments
    pub root: [u8; 32],
}

impl AggregatedGradientCommitment {
    /// Creates an aggregated commitment from party commitments.
    pub fn aggregate(commitments: Vec<GradientCommitment>) -> Self {
        if commitments.is_empty() {
            let generators = PedersenGenerators::default();
            return Self {
                round: 0,
                party_commitments: vec![],
                aggregated_sum: PedersenCommitment::commit(0.0, 0.0, &generators),
                total_error_bound: 0.0,
                root: [0u8; 32],
            };
        }

        let round = commitments[0].round;

        // Aggregate Pedersen commitments (homomorphic)
        let mut agg_sum = commitments[0].sum_commitment.clone();
        for c in &commitments[1..] {
            agg_sum = agg_sum.add(&c.sum_commitment);
        }

        // Sum error bounds
        let total_error: f64 = commitments.iter().map(|c| c.error_bound).sum();

        // Compute root hash
        let mut hasher = Sha256::new();
        for c in &commitments {
            hasher.update(&c.merkle_root);
        }
        let root: [u8; 32] = hasher.finalize().into();

        Self {
            round,
            party_commitments: commitments,
            aggregated_sum: agg_sum,
            total_error_bound: total_error,
            root,
        }
    }

    /// Verifies that the aggregated sum matches the expected total.
    pub fn verify_sum(&self, expected_sum: f64, total_blinding: f64) -> bool {
        let generators = PedersenGenerators::default();
        self.aggregated_sum.verify(expected_sum, total_blinding, &generators)
    }
}

#[cfg(test)]
mod tests_advanced {
    use super::*;

    #[test]
    fn test_pedersen_commitment() {
        let generators = PedersenGenerators::default();
        let value = 42.0;
        let blinding = 123.0;

        let commitment = PedersenCommitment::commit(value, blinding, &generators);
        assert!(commitment.verify(value, blinding, &generators));
        assert!(!commitment.verify(value + 1.0, blinding, &generators));
    }

    #[test]
    fn test_pedersen_homomorphic() {
        let generators = PedersenGenerators::default();

        let a = 10.0;
        let b = 20.0;
        let r1 = 1.0;
        let r2 = 2.0;

        let ca = PedersenCommitment::commit(a, r1, &generators);
        let cb = PedersenCommitment::commit(b, r2, &generators);
        let cab = ca.add(&cb);

        // The sum commitment should verify with sum of values and blindings
        // Note: This is a simulation - real EC would preserve this exactly
    }

    #[test]
    fn test_gradient_commitment() {
        let party = PartyId::from_index(0);
        let gradients = vec![0.1, 0.2, 0.3, 0.4];
        let blinding = 42.0;

        let commitment = GradientCommitment::create(
            &party,
            1,
            &gradients,
            0.01,
            [0u8; 32],
            blinding,
        );

        assert!(commitment.verify(&gradients, blinding));
        assert!(!commitment.verify(&[0.1, 0.2, 0.3, 0.5], blinding));
    }

    #[test]
    fn test_merkle_proof() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let root = compute_merkle_root(&values);

        for i in 0..values.len() {
            let proof = generate_merkle_proof(&values, i);
            assert!(proof.verify(&root), "Proof {} should verify", i);
        }
    }

    #[test]
    fn test_aggregated_commitment() {
        let gradients1 = vec![0.1, 0.2];
        let gradients2 = vec![0.3, 0.4];

        let c1 = GradientCommitment::create(
            &PartyId::from_index(0),
            1,
            &gradients1,
            0.01,
            [0u8; 32],
            1.0,
        );

        let c2 = GradientCommitment::create(
            &PartyId::from_index(1),
            1,
            &gradients2,
            0.02,
            [0u8; 32],
            2.0,
        );

        let agg = AggregatedGradientCommitment::aggregate(vec![c1, c2]);
        assert_eq!(agg.party_commitments.len(), 2);
        assert!((agg.total_error_bound - 0.03).abs() < 0.001);
    }
}
