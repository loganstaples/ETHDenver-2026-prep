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
use crate::field::{Fr, ct_eq_hash};

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
        Self::commit_fr_vector(party, &tensor.data, blinding, description)
    }

    /// Creates a commitment to a vector of Fr field elements.
    pub fn commit_fr_vector(
        party: &PartyId,
        values: &[Fr],
        blinding: &[u8; 32],
        description: impl Into<String>,
    ) -> Self {
        let mut data = Vec::with_capacity(values.len() * 32);
        for v in values {
            data.extend_from_slice(&v.to_bytes_le());
        }
        let hash = Self::hash_with_blinding(&data, blinding);
        Self {
            party: party.clone(),
            hash,
            description: description.into(),
        }
    }

    /// Verifies a commitment against a value (constant-time comparison).
    pub fn verify_scalar(&self, value: f64, blinding: &[u8; 32]) -> bool {
        let expected = Self::hash_with_blinding(&value.to_le_bytes(), blinding);
        ct_eq_hash(&self.hash, &expected).to_bool()
    }

    /// Verifies a commitment against a vector (constant-time comparison).
    pub fn verify_vector(&self, values: &[f64], blinding: &[u8; 32]) -> bool {
        let mut data = Vec::with_capacity(values.len() * 8);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let expected = Self::hash_with_blinding(&data, blinding);
        ct_eq_hash(&self.hash, &expected).to_bool()
    }

    /// Verifies a commitment against a tensor share.
    pub fn verify_tensor(&self, tensor: &TensorShare, blinding: &[u8; 32]) -> bool {
        self.verify_fr_vector(&tensor.data, blinding)
    }

    /// Verifies a commitment against a vector of Fr field elements (constant-time comparison).
    pub fn verify_fr_vector(&self, values: &[Fr], blinding: &[u8; 32]) -> bool {
        let mut data = Vec::with_capacity(values.len() * 32);
        for v in values {
            data.extend_from_slice(&v.to_bytes_le());
        }
        let expected = Self::hash_with_blinding(&data, blinding);
        ct_eq_hash(&self.hash, &expected).to_bool()
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
// Pedersen Commitments (Homomorphic, real EC point operations)
// ============================================================================

use halo2curves::bn256::{G1Affine, G1};
use halo2curves::group::{Group, Curve};
use halo2curves::bn256::Fr as Halo2Fr;
use halo2curves::ff::{Field, PrimeField};

/// Pedersen commitment scheme using BN254 elliptic curve.
///
/// Commitment: C(v, r) = v*G + r*H where G, H are generator points.
///
/// Properties:
/// - Additively homomorphic: C(a, r1) + C(b, r2) = C(a+b, r1+r2)
/// - Information-theoretically hiding (given random r)
/// - Computationally binding (under discrete log assumption)
#[derive(Debug, Clone)]
pub struct PedersenCommitment {
    /// The commitment point on BN254 G1
    pub point: G1Affine,
}

impl Serialize for PedersenCommitment {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use halo2curves::group::GroupEncoding;
        let bytes = self.point.to_bytes();
        serializer.serialize_bytes(bytes.as_ref())
    }
}

impl<'de> Deserialize<'de> for PedersenCommitment {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use halo2curves::group::GroupEncoding;
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let mut repr = <G1Affine as GroupEncoding>::Repr::default();
        if bytes.len() != repr.as_ref().len() {
            return Err(serde::de::Error::custom("invalid point length"));
        }
        repr.as_mut().copy_from_slice(&bytes);
        let point = G1Affine::from_bytes(&repr);
        if point.is_some().into() {
            Ok(Self { point: point.unwrap() })
        } else {
            Err(serde::de::Error::custom("invalid curve point"))
        }
    }
}

impl PartialEq for PedersenCommitment {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}

impl PedersenCommitment {
    /// Creates a new Pedersen commitment: C = value*G + blinding*H
    pub fn commit(value: &Fr, blinding: &Fr, generators: &PedersenGenerators) -> Self {
        let g_proj = G1::from(generators.g);
        let h_proj = G1::from(generators.h);
        let point = (g_proj * value.inner() + h_proj * blinding.inner()).to_affine();
        Self { point }
    }

    /// Creates a commitment from an f64 value (convenience wrapper).
    pub fn commit_f64(value: f64, blinding: f64, generators: &PedersenGenerators) -> Self {
        Self::commit(&Fr::from_f64(value), &Fr::from_f64(blinding), generators)
    }

    /// Creates a commitment to a vector of values.
    pub fn commit_vector(values: &[Fr], blindings: &[Fr], generators: &PedersenGenerators) -> Vec<Self> {
        values
            .iter()
            .zip(blindings.iter())
            .map(|(v, r)| Self::commit(v, r, generators))
            .collect()
    }

    /// Verifies a commitment opening: checks that point == value*G + blinding*H
    pub fn verify(&self, value: &Fr, blinding: &Fr, generators: &PedersenGenerators) -> bool {
        let expected = Self::commit(value, blinding, generators);
        self.point == expected.point
    }

    /// Verifies with f64 values (convenience wrapper).
    pub fn verify_f64(&self, value: f64, blinding: f64, generators: &PedersenGenerators) -> bool {
        self.verify(&Fr::from_f64(value), &Fr::from_f64(blinding), generators)
    }

    /// Adds two commitments (homomorphic property).
    /// C(a, r1) + C(b, r2) = C(a+b, r1+r2)
    pub fn add(&self, other: &PedersenCommitment) -> PedersenCommitment {
        let sum = (G1::from(self.point) + G1::from(other.point)).to_affine();
        PedersenCommitment { point: sum }
    }

    /// Scales a commitment by a scalar.
    /// scalar * C(a, r) = C(scalar*a, scalar*r)
    pub fn scale(&self, scalar: &Fr) -> PedersenCommitment {
        let scaled = (G1::from(self.point) * scalar.inner()).to_affine();
        PedersenCommitment { point: scaled }
    }
}

/// Generator points for Pedersen commitments on BN254 G1.
#[derive(Debug, Clone)]
pub struct PedersenGenerators {
    /// Generator g (standard BN254 generator)
    pub g: G1Affine,
    /// Generator h (nothing-up-my-sleeve derived, unknown discrete log w.r.t. g)
    pub h: G1Affine,
}

impl Serialize for PedersenGenerators {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use halo2curves::group::GroupEncoding;
        use serde::ser::SerializeTuple;
        let g_bytes = self.g.to_bytes();
        let h_bytes = self.h.to_bytes();
        let mut t = serializer.serialize_tuple(2)?;
        t.serialize_element(g_bytes.as_ref())?;
        t.serialize_element(h_bytes.as_ref())?;
        t.end()
    }
}

impl<'de> Deserialize<'de> for PedersenGenerators {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use halo2curves::group::GroupEncoding;
        let (g_bytes, h_bytes): (Vec<u8>, Vec<u8>) = Deserialize::deserialize(deserializer)?;
        let mut g_repr = <G1Affine as GroupEncoding>::Repr::default();
        let mut h_repr = <G1Affine as GroupEncoding>::Repr::default();
        if g_bytes.len() != g_repr.as_ref().len() || h_bytes.len() != h_repr.as_ref().len() {
            return Err(serde::de::Error::custom("invalid generator bytes length"));
        }
        g_repr.as_mut().copy_from_slice(&g_bytes);
        h_repr.as_mut().copy_from_slice(&h_bytes);
        let g = G1Affine::from_bytes(&g_repr);
        let h = G1Affine::from_bytes(&h_repr);
        if g.is_some().into() && h.is_some().into() {
            Ok(Self { g: g.unwrap(), h: h.unwrap() })
        } else {
            Err(serde::de::Error::custom("invalid generator curve points"))
        }
    }
}

impl PedersenGenerators {
    /// Creates generators from a seed (for testing).
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let scalar = Halo2Fr::random(&mut rng);
        let g = G1Affine::generator();
        let h = (G1::generator() * scalar).to_affine();
        Self { g, h }
    }

    /// Creates generators using nothing-up-my-sleeve derivation.
    ///
    /// g is the standard BN254 generator. h is derived by hashing the domain
    /// string to a scalar and multiplying the generator by it. This ensures
    /// nobody knows the discrete log of h relative to g.
    pub fn nothing_up_my_sleeve(domain: &str) -> Self {
        let g = G1Affine::generator();

        // Derive h by hashing domain to get a scalar, then h = scalar * G
        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        hasher.update(b"pedersen_generator_h");
        let hash: [u8; 32] = hasher.finalize().into();

        // Convert hash to Fr by interpreting as little-endian bytes mod r.
        // Use from_repr which reduces mod the field modulus.
        let mut repr = <Halo2Fr as PrimeField>::Repr::default();
        repr.as_mut().copy_from_slice(&hash);
        // If the bytes happen to be >= modulus, from_repr returns None.
        // In that case, just zero the high byte and retry (deterministic).
        let scalar = Halo2Fr::from_repr(repr).unwrap_or_else(|| {
            let mut adjusted = hash;
            adjusted[31] &= 0x0F; // Clear high nibble to ensure < modulus
            let mut repr2 = <Halo2Fr as PrimeField>::Repr::default();
            repr2.as_mut().copy_from_slice(&adjusted);
            Halo2Fr::from_repr(repr2).unwrap_or(Halo2Fr::ONE)
        });
        let h = (G1::generator() * scalar).to_affine();

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
        blinding: &Fr,
    ) -> Self {
        let generators = PedersenGenerators::default();

        // Compute Merkle root
        let merkle_root = compute_merkle_root(gradients);

        // Compute Pedersen commitment to sum
        let sum: f64 = gradients.iter().sum();
        let sum_commitment = PedersenCommitment::commit(&Fr::from_f64(sum), blinding, &generators);

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
    pub fn verify(&self, gradients: &[f64], blinding: &Fr) -> bool {
        if gradients.len() != self.num_elements {
            return false;
        }

        // Verify Merkle root (constant-time)
        let computed_root = compute_merkle_root(gradients);
        if !ct_eq_hash(&computed_root, &self.merkle_root).to_bool() {
            return false;
        }

        // Verify sum commitment
        let generators = PedersenGenerators::default();
        let sum: f64 = gradients.iter().sum();
        self.sum_commitment.verify(&Fr::from_f64(sum), blinding, &generators)
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
    /// Verifies the proof against a root hash (constant-time comparison).
    pub fn verify(&self, root: &[u8; 32]) -> bool {
        let mut current = hash_f64(self.value);

        for (sibling, is_right) in self.siblings.iter().zip(self.directions.iter()) {
            current = if *is_right {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
        }

        ct_eq_hash(&current, root).to_bool()
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
                aggregated_sum: PedersenCommitment::commit(&Fr::ZERO, &Fr::ZERO, &generators),
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
    pub fn verify_sum(&self, expected_sum: &Fr, total_blinding: &Fr) -> bool {
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
        let value = Fr::from_f64(42.0);
        let blinding = Fr::from_f64(123.0);

        let commitment = PedersenCommitment::commit(&value, &blinding, &generators);
        assert!(commitment.verify(&value, &blinding, &generators));
        assert!(!commitment.verify(&Fr::from_f64(43.0), &blinding, &generators));
    }

    #[test]
    fn test_pedersen_homomorphic() {
        let generators = PedersenGenerators::default();

        let a = Fr::from_f64(10.0);
        let b = Fr::from_f64(20.0);
        let r1 = Fr::from_f64(1.0);
        let r2 = Fr::from_f64(2.0);

        let ca = PedersenCommitment::commit(&a, &r1, &generators);
        let cb = PedersenCommitment::commit(&b, &r2, &generators);
        let cab = ca.add(&cb);

        // Real homomorphic property: C(a,r1) + C(b,r2) = C(a+b, r1+r2)
        let sum_val = Fr::add(&a, &b);
        let sum_blind = Fr::add(&r1, &r2);
        assert!(
            cab.verify(&sum_val, &sum_blind, &generators),
            "Homomorphic addition must hold: C(a)+C(b) should verify with (a+b, r1+r2)"
        );
    }

    #[test]
    fn test_pedersen_scale() {
        let generators = PedersenGenerators::default();

        let v = Fr::from_f64(5.0);
        let r = Fr::from_f64(7.0);
        let scalar = Fr::from_f64(3.0);

        let c = PedersenCommitment::commit(&v, &r, &generators);
        let scaled = c.scale(&scalar);

        // scalar * C(v, r) = C(scalar*v, scalar*r)
        // Use raw field multiplication to match EC scalar multiplication in scale()
        let sv = Fr::mul(&v, &scalar);
        let sr = Fr::mul(&r, &scalar);
        assert!(
            scaled.verify(&sv, &sr, &generators),
            "Scalar multiplication must hold"
        );
    }

    #[test]
    fn test_gradient_commitment() {
        let party = PartyId::from_index(0);
        let gradients = vec![0.1, 0.2, 0.3, 0.4];
        let blinding = Fr::from_f64(42.0);

        let commitment = GradientCommitment::create(
            &party,
            1,
            &gradients,
            0.01,
            [0u8; 32],
            &blinding,
        );

        assert!(commitment.verify(&gradients, &blinding));
        assert!(!commitment.verify(&[0.1, 0.2, 0.3, 0.5], &blinding));
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

        let r1 = Fr::from_f64(1.0);
        let r2 = Fr::from_f64(2.0);

        let c1 = GradientCommitment::create(
            &PartyId::from_index(0),
            1,
            &gradients1,
            0.01,
            [0u8; 32],
            &r1,
        );

        let c2 = GradientCommitment::create(
            &PartyId::from_index(1),
            1,
            &gradients2,
            0.02,
            [0u8; 32],
            &r2,
        );

        let agg = AggregatedGradientCommitment::aggregate(vec![c1, c2]);
        assert_eq!(agg.party_commitments.len(), 2);
        assert!((agg.total_error_bound - 0.03).abs() < 0.001);

        // Verify homomorphic sum: (0.1+0.2) + (0.3+0.4) = 1.0
        let total_sum = Fr::from_f64(1.0);
        let total_blind = Fr::add(&r1, &r2);
        assert!(agg.verify_sum(&total_sum, &total_blind));
    }
}
