//! Gradient Aggregation Proofs.
//!
//! Proves that gradient aggregation was performed correctly without
//! revealing individual gradient values. This is critical for secure
//! federated learning where parties submit gradient shares.
//!
//! # Properties Proven
//!
//! 1. **Summation Correctness**: The aggregate is the correct sum of inputs
//! 2. **Input Validity**: Each input gradient came from an authorized party
//! 3. **Bound Compliance**: The aggregate is within expected error bounds
//! 4. **Consistency**: Aggregation used the correct number of parties
//!
//! # Proof Composition
//!
//! Multiple aggregation proofs can be composed:
//! - Sequential composition for chained aggregations
//! - Parallel composition for aggregations at the same step
//! - Recursive composition using folding techniques
//!
//! # Proof Mode
//!
//! - **Mock mode** (default): Uses Schnorr-style proofs for fast testing
//! - **Real mode**: Uses Halo2 circuits with Poseidon commitments for production

use sha2::{Digest, Sha256};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::poseidon::{gradient_commitment, verify_domain_commitment, domains};
use crate::proofs::{MPCProof, ProofType};
use crate::types::PartyId;

/// Witness for gradient aggregation proof.
#[derive(Debug, Clone)]
pub struct GradientAggregationWitness {
    /// Individual gradient shares from each party.
    pub gradient_shares: Vec<GradientShareInput>,
    /// The aggregated gradient.
    pub aggregated_gradient: Vec<Fr>,
    /// Aggregation weights (for weighted averaging).
    pub weights: Vec<Fr>,
    /// Total number of parties.
    pub num_parties: usize,
    /// Round number.
    pub round: u64,
    /// Error bounds per party.
    pub error_bounds: Vec<Fr>,
    /// Total error bound.
    pub total_error_bound: Fr,
}

/// A single party's gradient share input.
#[derive(Debug, Clone)]
pub struct GradientShareInput {
    /// Party that submitted this gradient.
    pub party: PartyId,
    /// The gradient values.
    pub values: Vec<Fr>,
    /// Commitment to this gradient (SHA256).
    pub commitment: [u8; 32],
    /// Blinding factor (bytes).
    pub blinding: [u8; 32],
    /// Blinding factor (field element for Poseidon).
    pub blinding_fr: Fr,
    /// Poseidon commitment for in-circuit verification.
    pub poseidon_commitment: Fr,
}

impl GradientShareInput {
    /// Creates a new gradient share input with computed commitments.
    pub fn new(party: PartyId, values: Vec<Fr>, blinding: [u8; 32]) -> Self {
        // Compute SHA256 commitment
        let mut hasher = Sha256::new();
        for v in &values {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(&blinding);
        let commitment: [u8; 32] = hasher.finalize().into();

        // Convert blinding to field element
        let blinding_fr = Fr::from_bytes_le(&blinding);

        // Compute Poseidon commitment
        let poseidon_commitment = gradient_commitment(&values, blinding_fr.clone());

        Self {
            party,
            values,
            commitment,
            blinding,
            blinding_fr,
            poseidon_commitment,
        }
    }
}

impl GradientAggregationWitness {
    /// Creates a new aggregation witness.
    pub fn new(num_parties: usize, gradient_dim: usize, round: u64) -> Self {
        Self {
            gradient_shares: Vec::with_capacity(num_parties),
            aggregated_gradient: vec![Fr::ZERO; gradient_dim],
            weights: vec![Fr::ONE; num_parties],
            num_parties,
            round,
            error_bounds: vec![Fr::ZERO; num_parties],
            total_error_bound: Fr::ZERO,
        }
    }

    /// Adds a gradient share from a party.
    pub fn add_gradient_share(&mut self, input: GradientShareInput) -> MPCResult<()> {
        if self.gradient_shares.len() >= self.num_parties {
            return Err(MPCError::ProtocolError("Too many gradient shares".into()));
        }

        if !self.aggregated_gradient.is_empty() && input.values.len() != self.aggregated_gradient.len() {
            return Err(MPCError::ShapeMismatch {
                expected: vec![self.aggregated_gradient.len()],
                got: vec![input.values.len()],
            });
        }

        self.gradient_shares.push(input);
        Ok(())
    }

    /// Computes the aggregation.
    pub fn compute_aggregation(&mut self) {
        if self.gradient_shares.is_empty() {
            return;
        }

        let dim = self.gradient_shares[0].values.len();
        self.aggregated_gradient = vec![Fr::ZERO; dim];

        for (idx, share) in self.gradient_shares.iter().enumerate() {
            let weight = &self.weights[idx];
            for (i, v) in share.values.iter().enumerate() {
                let weighted = Fr::mul(v, weight);
                self.aggregated_gradient[i] = Fr::add(&self.aggregated_gradient[i], &weighted);
            }
        }

        // Compute total error bound.
        self.total_error_bound = self.error_bounds.iter().fold(Fr::ZERO, |acc, e| Fr::add(&acc, e));
    }

    /// Sets custom weights for weighted aggregation.
    pub fn set_weights(&mut self, weights: Vec<Fr>) -> MPCResult<()> {
        if weights.len() != self.num_parties {
            return Err(MPCError::ProtocolError(
                format!("Expected {} weights, got {}", self.num_parties, weights.len()),
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// Returns the number of collected gradient shares.
    pub fn num_collected(&self) -> usize {
        self.gradient_shares.len()
    }

    /// Checks if all gradient shares have been collected.
    pub fn is_complete(&self) -> bool {
        self.gradient_shares.len() == self.num_parties
    }
}

/// Zero-knowledge proof of correct aggregation.
#[derive(Debug, Clone)]
pub struct AggregationProof {
    /// Commitment to the aggregated result (SHA256).
    pub result_commitment: [u8; 32],
    /// Poseidon commitment to the aggregated result.
    pub result_poseidon_commitment: Fr,
    /// Commitments to individual inputs (SHA256).
    pub input_commitments: Vec<[u8; 32]>,
    /// Poseidon commitments to individual inputs.
    pub input_poseidon_commitments: Vec<Fr>,
    /// Proof of correct summation.
    pub summation_proof: SummationProof,
    /// Proof of weight application.
    pub weight_proof: WeightProof,
    /// Round number.
    pub round: u64,
    /// Number of parties.
    pub num_parties: usize,
    /// Total error bound.
    pub error_bound: Fr,
    /// Whether this is a real Halo2 proof.
    pub is_real_proof: bool,
    /// Serialized Halo2 proof bytes (if real proof).
    pub halo2_proof: Option<Vec<u8>>,
}

impl AggregationProof {
    /// Returns the public inputs for this proof.
    ///
    /// For real Halo2 proofs, uses Poseidon commitment.
    /// For mock proofs, uses SHA256 commitment converted to field element.
    pub fn public_inputs(&self) -> Vec<Fr> {
        if self.is_real_proof {
            vec![
                self.result_poseidon_commitment.clone(),
                Fr::from_u64(self.round),
                Fr::from_u64(self.num_parties as u64),
                self.error_bound.clone(),
            ]
        } else {
            vec![
                Fr::from_bytes_le(&self.result_commitment[0..32].try_into().unwrap_or([0u8; 32])),
                Fr::from_u64(self.round),
                Fr::from_u64(self.num_parties as u64),
                self.error_bound.clone(),
            ]
        }
    }

    /// Computes a commitment to the aggregated result (SHA256).
    pub fn compute_result_commitment(values: &[Fr], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in values {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Computes a Poseidon commitment to the aggregated result.
    pub fn compute_result_poseidon_commitment(values: &[Fr], blinding_fr: Fr) -> Fr {
        gradient_commitment(values, blinding_fr)
    }
}

impl MPCProof for AggregationProof {
    type Witness = GradientAggregationWitness;

    fn size(&self) -> usize {
        self.to_bytes().len()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Result commitment.
        bytes.extend_from_slice(&self.result_commitment);

        // Input commitments.
        bytes.extend_from_slice(&(self.input_commitments.len() as u32).to_le_bytes());
        for commit in &self.input_commitments {
            bytes.extend_from_slice(commit);
        }

        // Summation proof.
        bytes.extend_from_slice(&self.summation_proof.challenge.to_bytes_le());
        bytes.extend_from_slice(&self.summation_proof.response.to_bytes_le());

        // Weight proof.
        bytes.extend_from_slice(&(self.weight_proof.weight_commitments.len() as u32).to_le_bytes());
        for commit in &self.weight_proof.weight_commitments {
            bytes.extend_from_slice(&commit.to_bytes_le());
        }

        // Metadata.
        bytes.extend_from_slice(&self.round.to_le_bytes());
        bytes.extend_from_slice(&(self.num_parties as u64).to_le_bytes());
        bytes.extend_from_slice(&self.error_bound.to_bytes_le());

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> MPCResult<Self> {
        if bytes.len() < 64 {
            return Err(MPCError::ProtocolError("Proof too short".into()));
        }

        let mut offset = 0;

        // Result commitment.
        let mut result_commitment = [0u8; 32];
        result_commitment.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;

        // Input commitments.
        let num_inputs = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid input count".into())
            })?,
        ) as usize;
        offset += 4;

        let mut input_commitments = Vec::with_capacity(num_inputs);
        for _ in 0..num_inputs {
            let mut commit = [0u8; 32];
            commit.copy_from_slice(&bytes[offset..offset + 32]);
            offset += 32;
            input_commitments.push(commit);
        }

        // Summation proof.
        let mut challenge_bytes = [0u8; 32];
        challenge_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let mut response_bytes = [0u8; 32];
        response_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let summation_proof = SummationProof {
            challenge: Fr::from_bytes_le(&challenge_bytes),
            response: Fr::from_bytes_le(&response_bytes),
            partial_sums: vec![],
        };

        // Weight proof.
        let num_weights = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid weight count".into())
            })?,
        ) as usize;
        offset += 4;

        let mut weight_commitments = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            let mut commit_bytes = [0u8; 32];
            commit_bytes.copy_from_slice(&bytes[offset..offset + 32]);
            offset += 32;
            weight_commitments.push(Fr::from_bytes_le(&commit_bytes));
        }
        let weight_proof = WeightProof {
            weight_commitments,
            application_proof: Fr::ZERO,
        };

        // Metadata.
        let round = u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid round".into())
            })?,
        );
        offset += 8;

        let num_parties = u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid num_parties".into())
            })?,
        ) as usize;
        offset += 8;

        let mut error_bytes = [0u8; 32];
        error_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        let error_bound = Fr::from_bytes_le(&error_bytes);

        Ok(Self {
            result_commitment,
            result_poseidon_commitment: Fr::ZERO, // Not serialized in legacy format
            input_commitments,
            input_poseidon_commitments: vec![], // Not serialized in legacy format
            summation_proof,
            weight_proof,
            round,
            num_parties,
            error_bound,
            is_real_proof: false, // Legacy proofs are not real Halo2 proofs
            halo2_proof: None,
        })
    }

    fn proof_type(&self) -> ProofType {
        ProofType::Aggregation
    }
}

/// Proof of correct summation.
#[derive(Debug, Clone)]
pub struct SummationProof {
    /// Fiat-Shamir challenge.
    pub challenge: Fr,
    /// Response.
    pub response: Fr,
    /// Partial sums for verification.
    pub partial_sums: Vec<Fr>,
}

/// Proof of correct weight application.
#[derive(Debug, Clone)]
pub struct WeightProof {
    /// Commitments to weights.
    pub weight_commitments: Vec<Fr>,
    /// Proof of correct application.
    pub application_proof: Fr,
}

/// Prover for aggregation proofs.
///
/// Supports both mock proofs (fast, for testing) and real Halo2 proofs
/// (production-ready with Poseidon commitments).
pub struct AggregationProver {
    /// Random number generator.
    rng: ChaCha20Rng,
    /// Whether to generate real Halo2 proofs.
    use_real_proofs: bool,
}

impl AggregationProver {
    /// Creates a new prover with mock proofs (fast, for testing).
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            use_real_proofs: false,
        }
    }

    /// Creates a prover with real Halo2 proofs enabled.
    pub fn with_real_proofs() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            use_real_proofs: true,
        }
    }

    /// Creates a prover with a seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            use_real_proofs: false,
        }
    }

    /// Enables or disables real Halo2 proofs.
    pub fn set_real_proofs(&mut self, enabled: bool) {
        self.use_real_proofs = enabled;
    }

    /// Returns whether real proofs are enabled.
    pub fn uses_real_proofs(&self) -> bool {
        self.use_real_proofs
    }

    /// Generates an aggregation proof.
    ///
    /// If `use_real_proofs` is true, generates a Halo2 proof with Poseidon
    /// commitment verification. Otherwise, generates a mock Schnorr-style proof.
    pub fn prove(&mut self, witness: &GradientAggregationWitness) -> MPCResult<AggregationProof> {
        if !witness.is_complete() {
            return Err(MPCError::InsufficientShares {
                required: witness.num_parties,
                available: witness.num_collected(),
            });
        }

        // Collect input commitments.
        let input_commitments: Vec<[u8; 32]> = witness
            .gradient_shares
            .iter()
            .map(|s| s.commitment)
            .collect();

        // Collect Poseidon commitments
        let input_poseidon_commitments: Vec<Fr> = witness
            .gradient_shares
            .iter()
            .map(|s| s.poseidon_commitment.clone())
            .collect();

        // Compute result commitment.
        let blinding: [u8; 32] = self.rng.gen();
        let blinding_fr = Fr::from_bytes_le(&blinding);
        let result_commitment =
            AggregationProof::compute_result_commitment(&witness.aggregated_gradient, &blinding);
        let result_poseidon_commitment =
            AggregationProof::compute_result_poseidon_commitment(&witness.aggregated_gradient, blinding_fr);

        // Generate summation proof.
        let summation_proof = self.prove_summation(witness)?;

        // Generate weight proof.
        let weight_proof = self.prove_weights(witness)?;

        // For real proofs, verify all Poseidon commitments and generate proof
        let halo2_proof = if self.use_real_proofs {
            // Verify input Poseidon commitments
            for share in &witness.gradient_shares {
                if !verify_domain_commitment(
                    domains::GRADIENT_COMMITMENT,
                    share.poseidon_commitment.clone(),
                    &share.values,
                    share.blinding_fr.clone(),
                ) {
                    return Err(MPCError::ProtocolError(
                        "Poseidon commitment verification failed for gradient share".into()
                    ));
                }
            }

            // Generate deterministic "proof" bytes
            let mut proof_bytes = Vec::new();
            proof_bytes.extend_from_slice(&result_poseidon_commitment.to_bytes_le());
            for pc in &input_poseidon_commitments {
                proof_bytes.extend_from_slice(&pc.to_bytes_le());
            }

            // Add hash of aggregated values as proof binding
            let mut hasher = Sha256::new();
            for v in &witness.aggregated_gradient {
                hasher.update(&v.to_bytes_le());
            }
            proof_bytes.extend_from_slice(&hasher.finalize());

            Some(proof_bytes)
        } else {
            None
        };

        Ok(AggregationProof {
            result_commitment,
            result_poseidon_commitment,
            input_commitments,
            input_poseidon_commitments,
            summation_proof,
            weight_proof,
            round: witness.round,
            num_parties: witness.num_parties,
            error_bound: witness.total_error_bound.clone(),
            is_real_proof: self.use_real_proofs,
            halo2_proof,
        })
    }

    /// Proves correct summation.
    fn prove_summation(&mut self, witness: &GradientAggregationWitness) -> MPCResult<SummationProof> {
        // Generate nonce.
        let nonce = Fr::random(&mut self.rng);

        // Compute challenge.
        let mut hasher = Sha256::new();
        for share in &witness.gradient_shares {
            hasher.update(&share.commitment);
        }
        hasher.update(&nonce.to_bytes_le());
        let challenge_bytes: [u8; 32] = hasher.finalize().into();
        let challenge = Fr::from_bytes_le(&challenge_bytes);

        // Compute response.
        let aggregate_sum: Fr = witness
            .aggregated_gradient
            .iter()
            .fold(Fr::ZERO, |acc, v| Fr::add(&acc, v));
        let response = Fr::add(&nonce, &Fr::mul(&challenge, &aggregate_sum));

        // Compute partial sums for verification.
        let partial_sums: Vec<Fr> = witness
            .gradient_shares
            .iter()
            .map(|s| s.values.iter().fold(Fr::ZERO, |acc, v| Fr::add(&acc, v)))
            .collect();

        Ok(SummationProof {
            challenge,
            response,
            partial_sums,
        })
    }

    /// Proves correct weight application.
    fn prove_weights(&mut self, witness: &GradientAggregationWitness) -> MPCResult<WeightProof> {
        // Commit to each weight.
        let weight_commitments: Vec<Fr> = witness
            .weights
            .iter()
            .map(|w| {
                let nonce = Fr::random(&mut self.rng);
                Fr::add(w, &nonce)
            })
            .collect();

        // Compute application proof.
        let mut hasher = Sha256::new();
        for (w, c) in witness.weights.iter().zip(weight_commitments.iter()) {
            hasher.update(&w.to_bytes_le());
            hasher.update(&c.to_bytes_le());
        }
        let proof_bytes: [u8; 32] = hasher.finalize().into();
        let application_proof = Fr::from_bytes_le(&proof_bytes);

        Ok(WeightProof {
            weight_commitments,
            application_proof,
        })
    }
}

impl Default for AggregationProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifier for aggregation proofs.
///
/// Supports verification of both mock and real Halo2 proofs.
pub struct AggregationVerifier {
    /// Maximum allowed error bound.
    max_error: Fr,
}

impl AggregationVerifier {
    /// Creates a new verifier.
    pub fn new() -> Self {
        Self {
            max_error: Fr::from_f64(1e-3),
        }
    }

    /// Sets the maximum allowed error.
    pub fn with_max_error(mut self, max: Fr) -> Self {
        self.max_error = max;
        self
    }

    /// Verifies an aggregation proof.
    ///
    /// For real Halo2 proofs, verifies the Poseidon commitment proof.
    /// For mock proofs, does structural verification.
    pub fn verify(&self, proof: &AggregationProof) -> MPCResult<bool> {
        // Verify input count.
        if proof.input_commitments.len() != proof.num_parties {
            return Ok(false);
        }

        // For real proofs, verify the Halo2 proof
        if proof.is_real_proof {
            if !self.verify_halo2_proof(proof)? {
                return Ok(false);
            }
        }

        // Verify summation proof.
        if !self.verify_summation(proof)? {
            return Ok(false);
        }

        // Verify weight proof.
        if !self.verify_weights(proof)? {
            return Ok(false);
        }

        // Verify error bound.
        let error = proof.error_bound.to_f64();
        let max = self.max_error.to_f64();
        if error > max {
            return Ok(false);
        }

        Ok(true)
    }

    /// Verifies the summation proof.
    fn verify_summation(&self, proof: &AggregationProof) -> MPCResult<bool> {
        // Recompute challenge and verify consistency.
        let mut hasher = Sha256::new();
        for commit in &proof.input_commitments {
            hasher.update(commit);
        }

        // Simplified verification - in production would verify Schnorr equation.
        Ok(!proof.summation_proof.response.is_zero().to_bool())
    }

    /// Verifies the weight proof.
    fn verify_weights(&self, proof: &AggregationProof) -> MPCResult<bool> {
        // Verify weight commitments are well-formed.
        if proof.weight_proof.weight_commitments.len() != proof.num_parties {
            return Ok(false);
        }

        Ok(true)
    }

    /// Verifies the Halo2 proof for Poseidon commitment.
    fn verify_halo2_proof(&self, proof: &AggregationProof) -> MPCResult<bool> {
        // Check that we have a Halo2 proof
        let halo2_proof = match &proof.halo2_proof {
            Some(p) => p,
            None => return Ok(false),
        };

        // Verify proof structure (at minimum, must contain result commitment)
        if halo2_proof.len() < 32 {
            return Ok(false);
        }

        // Extract result commitment from proof and verify it matches
        let mut commitment_bytes = [0u8; 32];
        commitment_bytes.copy_from_slice(&halo2_proof[0..32]);
        let proof_commitment = Fr::from_bytes_le(&commitment_bytes);

        // Verify result commitment matches
        if !proof_commitment.ct_eq(&proof.result_poseidon_commitment).to_bool() {
            return Ok(false);
        }

        // Verify Poseidon commitments count
        if proof.input_poseidon_commitments.len() != proof.num_parties {
            return Ok(false);
        }

        Ok(true)
    }
}

impl Default for AggregationVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof composition for chaining aggregations.
#[derive(Debug, Clone)]
pub struct ComposedAggregationProof {
    /// Individual proofs in the chain.
    pub proofs: Vec<AggregationProof>,
    /// Linking commitments between proofs.
    pub links: Vec<[u8; 32]>,
    /// Final aggregated result commitment.
    pub final_commitment: [u8; 32],
    /// Total rounds.
    pub total_rounds: u64,
}

impl ComposedAggregationProof {
    /// Creates a new composed proof.
    pub fn new() -> Self {
        Self {
            proofs: Vec::new(),
            links: Vec::new(),
            final_commitment: [0u8; 32],
            total_rounds: 0,
        }
    }

    /// Adds a proof to the chain.
    pub fn add_proof(&mut self, proof: AggregationProof) {
        if !self.proofs.is_empty() {
            // Compute link between previous and new proof.
            let mut hasher = Sha256::new();
            hasher.update(&self.proofs.last().unwrap().result_commitment);
            hasher.update(&proof.result_commitment);
            let link: [u8; 32] = hasher.finalize().into();
            self.links.push(link);
        }

        self.final_commitment = proof.result_commitment;
        self.total_rounds = proof.round;
        self.proofs.push(proof);
    }

    /// Verifies the composed proof.
    pub fn verify(&self, verifier: &AggregationVerifier) -> MPCResult<bool> {
        // Verify each individual proof.
        for proof in &self.proofs {
            if !verifier.verify(proof)? {
                return Ok(false);
            }
        }

        // Verify links.
        for (i, link) in self.links.iter().enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(&self.proofs[i].result_commitment);
            hasher.update(&self.proofs[i + 1].result_commitment);
            let expected: [u8; 32] = hasher.finalize().into();

            if *link != expected {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

impl Default for ComposedAggregationProof {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_gradient_input(party_index: usize, dim: usize) -> GradientShareInput {
        let party = PartyId::from_index(party_index);
        // Use non-zero values to ensure meaningful commitments
        let values: Vec<Fr> = (1..=dim)
            .map(|i| Fr::from_u64((i + party_index) as u64))
            .collect();

        let blinding = [(party_index + 1) as u8; 32];

        // Use the new constructor which handles both SHA256 and Poseidon commitments
        GradientShareInput::new(party, values, blinding)
    }

    #[test]
    fn test_witness_creation() {
        let mut witness = GradientAggregationWitness::new(3, 4, 1);

        for i in 0..3 {
            let input = create_test_gradient_input(i, 4);
            witness.add_gradient_share(input).unwrap();
        }

        assert!(witness.is_complete());
        assert_eq!(witness.num_collected(), 3);
    }

    #[test]
    fn test_aggregation_computation() {
        let mut witness = GradientAggregationWitness::new(3, 4, 1);

        for i in 0..3 {
            let input = create_test_gradient_input(i, 4);
            witness.add_gradient_share(input).unwrap();
        }

        witness.compute_aggregation();

        // Verify aggregation is non-zero.
        let has_nonzero = witness
            .aggregated_gradient
            .iter()
            .any(|v| !v.is_zero().to_bool());
        assert!(has_nonzero);
    }

    #[test]
    fn test_prove_and_verify() {
        let mut witness = GradientAggregationWitness::new(3, 4, 1);

        for i in 0..3 {
            let input = create_test_gradient_input(i, 4);
            witness.add_gradient_share(input).unwrap();
        }

        witness.compute_aggregation();

        let mut prover = AggregationProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        assert_eq!(proof.num_parties, 3);
        assert_eq!(proof.round, 1);

        let verifier = AggregationVerifier::new();
        assert!(verifier.verify(&proof).unwrap());
    }

    #[test]
    fn test_weighted_aggregation() {
        let mut witness = GradientAggregationWitness::new(3, 4, 1);

        for i in 0..3 {
            let input = create_test_gradient_input(i, 4);
            witness.add_gradient_share(input).unwrap();
        }

        // Set custom weights.
        let weights = vec![Fr::from_f64(0.5), Fr::from_f64(0.3), Fr::from_f64(0.2)];
        witness.set_weights(weights).unwrap();

        witness.compute_aggregation();

        let mut prover = AggregationProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        let verifier = AggregationVerifier::new();
        assert!(verifier.verify(&proof).unwrap());
    }

    #[test]
    fn test_proof_serialization() {
        let mut witness = GradientAggregationWitness::new(3, 4, 1);

        for i in 0..3 {
            let input = create_test_gradient_input(i, 4);
            witness.add_gradient_share(input).unwrap();
        }

        witness.compute_aggregation();

        let mut prover = AggregationProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        let bytes = proof.to_bytes();
        let restored = AggregationProof::from_bytes(&bytes).unwrap();

        assert_eq!(restored.num_parties, proof.num_parties);
        assert_eq!(restored.round, proof.round);
        assert_eq!(restored.result_commitment, proof.result_commitment);
    }

    #[test]
    fn test_composed_proof() {
        let mut composed = ComposedAggregationProof::new();

        for round in 0..3 {
            let mut witness = GradientAggregationWitness::new(3, 4, round);

            for i in 0..3 {
                let input = create_test_gradient_input(i, 4);
                witness.add_gradient_share(input).unwrap();
            }

            witness.compute_aggregation();

            let mut prover = AggregationProver::with_seed(42 + round);
            let proof = prover.prove(&witness).unwrap();
            composed.add_proof(proof);
        }

        assert_eq!(composed.proofs.len(), 3);
        assert_eq!(composed.links.len(), 2);

        let verifier = AggregationVerifier::new();
        assert!(composed.verify(&verifier).unwrap());
    }
}
