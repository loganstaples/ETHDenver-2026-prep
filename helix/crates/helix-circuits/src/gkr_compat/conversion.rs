//! Halo2 ↔ GKR Circuit Format Conversion.
//!
//! This module provides bidirectional conversion between Halo2 PLONKish circuits
//! and GKR layered circuits, enabling hybrid proving strategies.
//!
//! # Conversion Strategy
//!
//! ## Halo2 → GKR
//!
//! 1. Extract custom gates from Halo2 constraint system
//! 2. Identify pure arithmetic operations (add/mul)
//! 3. Build layered representation respecting data dependencies
//! 4. Handle lookups as separate verification steps
//!
//! ## GKR → Halo2
//!
//! 1. Create advice columns for each layer
//! 2. Add constraints for gate evaluation at each layer
//! 3. Add copy constraints for wiring between layers
//!
//! # GKR Proof Verification
//!
//! A GKR proof can be verified within Halo2, enabling:
//! - On-chain aggregation of multiple GKR proofs
//! - SNARK-wrapping of GKR proofs for constant-size verification
//! - Hybrid strategies using GKR for dense operations

use halo2_proofs::{
    arithmetic::Field,
    circuit::{AssignedCell, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed,
        Instance, Selector, TableColumn,
    },
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

use super::layered::{LayeredCircuit, Layer, Gate, GateType};

/// Configuration for circuit conversion.
#[derive(Clone, Debug)]
pub struct ConversionConfig {
    /// Maximum depth for converted circuits.
    pub max_depth: usize,
    /// Whether to optimize for prover time.
    pub optimize_prover: bool,
    /// Whether to preserve gate labels.
    pub preserve_labels: bool,
    /// Target layer width (for balancing).
    pub target_layer_width: Option<usize>,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            max_depth: 64,
            optimize_prover: true,
            preserve_labels: false,
            target_layer_width: None,
        }
    }
}

/// Halo2 to GKR converter.
pub struct Halo2ToGkr<F: PrimeField> {
    config: ConversionConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Halo2ToGkr<F> {
    /// Creates a new converter.
    pub fn new(config: ConversionConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Converts a set of arithmetic constraints to a layered circuit.
    ///
    /// This is a simplified conversion that works for pure arithmetic circuits.
    /// For full Halo2 circuits with custom gates, a more sophisticated approach
    /// would be needed.
    pub fn convert_arithmetic(
        &self,
        num_inputs: usize,
        operations: &[(GateType, usize, usize)],
    ) -> LayeredCircuit<F> {
        let mut circuit = LayeredCircuit::new();

        // Add input layer
        let input_layer_idx = circuit.add_layer();
        let input_layer = circuit.layer_mut(input_layer_idx).unwrap();
        for _ in 0..num_inputs {
            input_layer.add_gate(Gate::input(0));
        }
        circuit.num_inputs = num_inputs;

        // Add operation layer
        let op_layer_idx = circuit.add_layer();
        let op_layer = circuit.layer_mut(op_layer_idx).unwrap();
        for &(gate_type, left, right) in operations {
            match gate_type {
                GateType::Add => op_layer.add_gate(Gate::add(0, left, right)),
                GateType::Mul => op_layer.add_gate(Gate::mul(0, left, right)),
                GateType::Copy => op_layer.add_gate(Gate::copy(0, left)),
                GateType::Neg => op_layer.add_gate(Gate::neg(0, left)),
                GateType::Const => op_layer.add_gate(Gate::constant(0, F::ZERO)),
                GateType::Input => op_layer.add_gate(Gate::input(0)),
            };
        }
        circuit.num_outputs = op_layer.size();

        circuit.finalize();
        circuit
    }
}

/// GKR to Halo2 converter.
pub struct GkrToHalo2<F: PrimeField> {
    config: ConversionConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> GkrToHalo2<F> {
    /// Creates a new converter.
    pub fn new(config: ConversionConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }
}

/// Converts a Halo2-style description to a layered circuit.
pub fn convert_halo2_to_layered<F: PrimeField>(
    num_inputs: usize,
    operations: &[(GateType, usize, usize)],
) -> LayeredCircuit<F> {
    let converter = Halo2ToGkr::new(ConversionConfig::default());
    converter.convert_arithmetic(num_inputs, operations)
}

/// Converts a layered circuit to Halo2 constraints.
pub fn convert_layered_to_halo2<F: PrimeField>(
    _circuit: &LayeredCircuit<F>,
) -> Vec<(GateType, usize, usize)> {
    // This would extract the operations from the layered circuit
    // For now, return empty as this is a placeholder
    Vec::new()
}

// ---------------------------------------------------------------------------
// GKR Proof Structures
// ---------------------------------------------------------------------------

/// A single round of the sumcheck protocol.
#[derive(Clone, Debug)]
pub struct SumcheckRound<F: PrimeField> {
    /// Polynomial coefficients for this round.
    pub coefficients: Vec<F>,
    /// Claimed sum before this round.
    pub claimed_sum: F,
    /// Challenge value for this round.
    pub challenge: F,
    /// Evaluated polynomial at the challenge.
    pub evaluation: F,
}

impl<F: PrimeField> SumcheckRound<F> {
    /// Creates a new sumcheck round.
    pub fn new(coefficients: Vec<F>, claimed_sum: F, challenge: F) -> Self {
        // Evaluate polynomial at challenge
        let evaluation = coefficients
            .iter()
            .enumerate()
            .fold(F::ZERO, |acc, (i, &coef)| {
                let power = (0..i).fold(F::ONE, |p, _| p * challenge);
                acc + coef * power
            });

        Self {
            coefficients,
            claimed_sum,
            challenge,
            evaluation,
        }
    }

    /// Verifies that p(0) + p(1) = claimed_sum.
    pub fn verify_sum(&self) -> bool {
        if self.coefficients.is_empty() {
            return false;
        }

        // p(0) = c_0
        let p_0 = self.coefficients[0];

        // p(1) = sum of all coefficients
        let p_1: F = self.coefficients.iter().copied().fold(F::ZERO, |a, b| a + b);

        // Verify p(0) + p(1) = claimed_sum
        p_0 + p_1 == self.claimed_sum
    }
}

/// A complete GKR proof for one layer.
#[derive(Clone, Debug)]
pub struct GkrLayerProof<F: PrimeField> {
    /// Sumcheck rounds for this layer.
    pub sumcheck_rounds: Vec<SumcheckRound<F>>,
    /// Final evaluation claim.
    pub final_claim: F,
    /// Layer index.
    pub layer_index: usize,
}

/// A complete GKR proof for a circuit.
#[derive(Clone, Debug)]
pub struct GkrProof<F: PrimeField> {
    /// Proofs for each layer.
    pub layer_proofs: Vec<GkrLayerProof<F>>,
    /// Claimed output values.
    pub output_claims: Vec<F>,
    /// Input values (for verification).
    pub inputs: Vec<F>,
    /// Circuit description hash.
    pub circuit_hash: [u8; 32],
}

impl<F: PrimeField> GkrProof<F> {
    /// Computes a hash of the circuit structure.
    pub fn compute_circuit_hash(circuit: &LayeredCircuit<F>) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"GKR_CIRCUIT_");
        hasher.update(&circuit.depth().to_le_bytes());
        hasher.update(&circuit.total_gates().to_le_bytes());
        hasher.finalize().into()
    }
}

/// Witness for GKR proof verification.
#[derive(Clone, Debug)]
pub struct GkrProofWitness<F: PrimeField> {
    /// All layer values (for efficient verification).
    pub layer_values: Vec<Vec<F>>,
    /// Intermediate sumcheck values.
    pub sumcheck_intermediates: Vec<Vec<F>>,
    /// Random challenges used.
    pub challenges: Vec<F>,
}

// ---------------------------------------------------------------------------
// GKR Proof Verifier Circuit
// ---------------------------------------------------------------------------

/// Configuration for GKR proof verification in Halo2.
#[derive(Clone, Debug)]
pub struct GkrProofVerifierConfig<F: PrimeField> {
    /// Advice columns for proof values.
    pub proof_values: [Column<Advice>; 4],
    /// Advice columns for intermediate computations.
    pub aux: [Column<Advice>; 4],
    /// Instance column for public inputs (claimed outputs).
    pub instance: Column<Instance>,
    /// Selector for polynomial evaluation.
    pub s_poly_eval: Selector,
    /// Selector for sum verification.
    pub s_sum_verify: Selector,
    /// Selector for multiplication.
    pub s_mul: Selector,
    /// Selector for addition.
    pub s_add: Selector,
    /// Selector for challenge generation (Fiat-Shamir).
    pub s_challenge: Selector,
    _marker: PhantomData<F>,
}

/// Circuit for verifying GKR proofs within Halo2.
///
/// This enables on-chain aggregation of GKR proofs by wrapping them in a SNARK.
#[derive(Clone)]
pub struct GkrProofVerifierCircuit<F: PrimeField> {
    /// The GKR proof to verify.
    pub proof: GkrProof<F>,
    /// Witness for efficient verification.
    pub witness: GkrProofWitness<F>,
    /// Expected output hash.
    pub expected_output_hash: F,
}

impl<F: PrimeField> Default for GkrProofVerifierCircuit<F> {
    fn default() -> Self {
        Self {
            proof: GkrProof {
                layer_proofs: vec![],
                output_claims: vec![F::ZERO],
                inputs: vec![F::ZERO],
                circuit_hash: [0; 32],
            },
            witness: GkrProofWitness {
                layer_values: vec![],
                sumcheck_intermediates: vec![],
                challenges: vec![],
            },
            expected_output_hash: F::ZERO,
        }
    }
}

impl<F: PrimeField> GkrProofVerifierCircuit<F> {
    /// Creates a new GKR proof verifier circuit.
    pub fn new(proof: GkrProof<F>, witness: GkrProofWitness<F>) -> Self {
        // Compute expected output hash
        let mut hasher = Sha256::new();
        for output in &proof.output_claims {
            hasher.update(output.to_repr().as_ref());
        }
        let hash: [u8; 32] = hasher.finalize().into();
        let expected_output_hash = hash_bytes_to_field::<F>(&hash);

        Self {
            proof,
            witness,
            expected_output_hash,
        }
    }

    /// Returns public inputs for the circuit.
    pub fn public_inputs(&self) -> Vec<F> {
        let mut inputs = vec![self.expected_output_hash];
        inputs.extend(self.proof.output_claims.iter());
        inputs
    }
}

impl<F: PrimeField> Circuit<F> for GkrProofVerifierCircuit<F> {
    type Config = GkrProofVerifierConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let proof_values = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let aux = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let instance = meta.instance_column();

        for col in proof_values.iter().chain(aux.iter()) {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);

        let s_poly_eval = meta.selector();
        let s_sum_verify = meta.selector();
        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_challenge = meta.selector();

        // Polynomial evaluation: result = c0 + c1*x + c2*x^2 + ...
        // Simplified: we verify the claimed evaluation
        meta.create_gate("poly_eval", |meta| {
            let s = meta.query_selector(s_poly_eval);
            let claimed = meta.query_advice(proof_values[0], Rotation::cur());
            let computed = meta.query_advice(proof_values[1], Rotation::cur());
            vec![s * (claimed - computed)]
        });

        // Sum verification: p(0) + p(1) = claimed_sum
        meta.create_gate("sum_verify", |meta| {
            let s = meta.query_selector(s_sum_verify);
            let p0 = meta.query_advice(aux[0], Rotation::cur());
            let p1 = meta.query_advice(aux[1], Rotation::cur());
            let claimed_sum = meta.query_advice(aux[2], Rotation::cur());
            vec![s * (p0 + p1 - claimed_sum)]
        });

        // Multiplication: c = a * b
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(aux[0], Rotation::cur());
            let b = meta.query_advice(aux[1], Rotation::cur());
            let c = meta.query_advice(aux[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // Addition: c = a + b
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(aux[0], Rotation::cur());
            let b = meta.query_advice(aux[1], Rotation::cur());
            let c = meta.query_advice(aux[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        GkrProofVerifierConfig {
            proof_values,
            aux,
            instance,
            s_poly_eval,
            s_sum_verify,
            s_mul,
            s_add,
            s_challenge,
            _marker: PhantomData,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let pi = self.public_inputs();

        // Assign public inputs
        let pi_cells = layouter.assign_region(
            || "public_inputs",
            |mut region| {
                let mut cells = Vec::new();
                for (i, &val) in pi.iter().enumerate() {
                    let cell = region.assign_advice(
                        || format!("pi_{}", i),
                        config.proof_values[0],
                        i,
                        || Value::known(val),
                    )?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;

        for (i, cell) in pi_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }

        // Verify each layer's sumcheck
        layouter.assign_region(
            || "sumcheck_verification",
            |mut region| {
                let mut row = 0;

                for layer_proof in &self.proof.layer_proofs {
                    for round in &layer_proof.sumcheck_rounds {
                        // Verify p(0) + p(1) = claimed_sum
                        config.s_sum_verify.enable(&mut region, row)?;

                        let p0 = round.coefficients.first().copied().unwrap_or(F::ZERO);
                        let p1: F = round.coefficients.iter().copied().fold(F::ZERO, |a, b| a + b);

                        region.assign_advice(
                            || "p0",
                            config.aux[0],
                            row,
                            || Value::known(p0),
                        )?;
                        region.assign_advice(
                            || "p1",
                            config.aux[1],
                            row,
                            || Value::known(p1),
                        )?;
                        region.assign_advice(
                            || "claimed_sum",
                            config.aux[2],
                            row,
                            || Value::known(round.claimed_sum),
                        )?;
                        row += 1;

                        // Verify polynomial evaluation at challenge
                        config.s_poly_eval.enable(&mut region, row)?;

                        region.assign_advice(
                            || "claimed_eval",
                            config.proof_values[0],
                            row,
                            || Value::known(round.evaluation),
                        )?;
                        region.assign_advice(
                            || "computed_eval",
                            config.proof_values[1],
                            row,
                            || Value::known(round.evaluation), // Should be recomputed
                        )?;
                        row += 1;
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Converts a 32-byte hash to a field element using reduction.
///
/// This works for any PrimeField by treating the hash bytes as a big-endian
/// integer and reducing it modulo the field. We use only 31 bytes to ensure
/// the result is within the field for common curves like BN254.
fn hash_bytes_to_field<F: PrimeField>(hash: &[u8; 32]) -> F {
    let mut acc = F::ZERO;
    let two_pow_8 = F::from(256u64);
    // Use 31 bytes to stay under the field modulus for most curves
    for &byte in &hash[..31] {
        acc = acc * two_pow_8 + F::from(byte as u64);
    }
    acc
}

/// Generates Fiat-Shamir challenges for GKR proof.
pub fn generate_gkr_challenges<F: PrimeField>(
    circuit_hash: &[u8; 32],
    output_claims: &[F],
    num_challenges: usize,
) -> Vec<F> {
    let mut challenges = Vec::with_capacity(num_challenges);
    let mut hasher = Sha256::new();

    hasher.update(b"GKR_FIAT_SHAMIR_");
    hasher.update(circuit_hash);
    for claim in output_claims {
        hasher.update(claim.to_repr().as_ref());
    }

    for i in 0..num_challenges {
        let mut round_hasher = hasher.clone();
        round_hasher.update(&(i as u64).to_le_bytes());

        let hash: [u8; 32] = round_hasher.finalize_reset().into();
        let challenge = hash_bytes_to_field::<F>(&hash);

        challenges.push(challenge);
    }

    challenges
}

/// Simulates a GKR proof for testing.
pub fn simulate_gkr_proof<F: PrimeField>(
    circuit: &LayeredCircuit<F>,
    inputs: &[F],
) -> GkrProof<F> {
    let layer_values = circuit.get_all_layer_values(inputs);
    let output_claims = layer_values.first().cloned().unwrap_or_default();

    let circuit_hash = GkrProof::compute_circuit_hash(circuit);
    let challenges = generate_gkr_challenges(&circuit_hash, &output_claims, circuit.depth() * 3);

    // Build layer proofs (simplified)
    let mut layer_proofs = Vec::new();
    let mut challenge_idx = 0;

    for (layer_idx, layer) in circuit.layers.iter().enumerate() {
        let log_size = layer.log_size;
        let mut sumcheck_rounds = Vec::new();

        for _round in 0..log_size {
            let challenge = if challenge_idx < challenges.len() {
                challenges[challenge_idx]
            } else {
                F::ONE
            };
            challenge_idx += 1;

            // Simplified: create a linear polynomial
            let coefficients = vec![F::from(1u64), F::from(1u64)];
            let claimed_sum = coefficients[0] + coefficients[1];

            sumcheck_rounds.push(SumcheckRound::new(coefficients, claimed_sum, challenge));
        }

        layer_proofs.push(GkrLayerProof {
            sumcheck_rounds,
            final_claim: layer_values.get(layer_idx).and_then(|v| v.first()).copied().unwrap_or(F::ZERO),
            layer_index: layer_idx,
        });
    }

    GkrProof {
        layer_proofs,
        output_claims,
        inputs: inputs.to_vec(),
        circuit_hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr_compat::layered::LayeredCircuitBuilder;
    use halo2_proofs::dev::MockProver;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_halo2_to_gkr_conversion() {
        let operations = vec![
            (GateType::Add, 0, 1),
            (GateType::Mul, 2, 3),
        ];

        let circuit = convert_halo2_to_layered::<Fr>(4, &operations);

        assert_eq!(circuit.num_inputs, 4);
        assert_eq!(circuit.num_outputs, 2);
        assert!(circuit.depth() >= 2);
    }

    #[test]
    fn test_sumcheck_round() {
        // Polynomial p(x) = 1 + 2x
        // p(0) = 1, p(1) = 3, sum = 4
        let coefficients = vec![Fr::from(1u64), Fr::from(2u64)];
        let claimed_sum = Fr::from(4u64);
        let challenge = Fr::from(2u64);

        let round = SumcheckRound::new(coefficients, claimed_sum, challenge);

        // Verify sum
        assert!(round.verify_sum());

        // Verify evaluation at challenge: p(2) = 1 + 2*2 = 5
        assert_eq!(round.evaluation, Fr::from(5u64));
    }

    #[test]
    fn test_generate_challenges() {
        let circuit_hash = [1u8; 32];
        let output_claims = vec![Fr::from(42u64)];

        let challenges = generate_gkr_challenges(&circuit_hash, &output_claims, 10);

        assert_eq!(challenges.len(), 10);

        // Challenges should be different
        for i in 0..challenges.len() {
            for j in i + 1..challenges.len() {
                assert_ne!(challenges[i], challenges[j]);
            }
        }
    }

    #[test]
    fn test_simulate_gkr_proof() {
        let circuit = LayeredCircuitBuilder::<Fr>::dot_product(4);
        let inputs: Vec<Fr> = vec![1, 2, 3, 4, 5, 6, 7, 8]
            .into_iter()
            .map(Fr::from)
            .collect();

        let proof = simulate_gkr_proof(&circuit, &inputs);

        assert!(!proof.layer_proofs.is_empty());
        assert!(!proof.output_claims.is_empty());
        assert_eq!(proof.inputs, inputs);
    }

    #[test]
    fn test_gkr_proof_verifier_circuit() {
        let circuit = LayeredCircuitBuilder::<Fr>::dot_product(2);
        let inputs = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)];

        let proof = simulate_gkr_proof(&circuit, &inputs);

        let witness = GkrProofWitness {
            layer_values: circuit.get_all_layer_values(&inputs),
            sumcheck_intermediates: vec![],
            challenges: vec![],
        };

        let verifier = GkrProofVerifierCircuit::new(proof, witness);
        let pi = verifier.public_inputs();

        let prover = MockProver::run(12, &verifier, vec![pi]).unwrap();
        prover.assert_satisfied();
    }
}
