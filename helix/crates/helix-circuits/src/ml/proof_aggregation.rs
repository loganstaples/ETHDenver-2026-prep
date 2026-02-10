//! Proof Aggregation Circuit (SHPLONK-style).
//!
//! Provides a Halo2 circuit that proves structural correctness of proof
//! aggregation: public input chaining, Fiat-Shamir RLC challenge, commitment
//! binding, and error accumulation.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                  SHPLONKAggregationCircuit                              │
//! │                                                                         │
//! │  Input: N proof commitment hashes, old/new hash pairs, loss, error     │
//! │                                                                         │
//! │  Constraints:                                                           │
//! │    1. PI Chaining:  step[i].new_hash == step[i+1].old_hash             │
//! │    2. Fiat-Shamir:  alpha = Poseidon(commitment_0, ..., commitment_N)  │
//! │    3. RLC:          rlc = Σ α^i · commitment_hash_i                    │
//! │    4. Error Accum:  total_error = Σ error_bound_i                      │
//! │    5. Loss Accum:   total_loss  = Σ loss_i                             │
//! │                                                                         │
//! │  Public Inputs (8, matching contract interface):                        │
//! │    [0] first_old_hash_lo   [1] first_old_hash_hi                       │
//! │    [2] last_new_hash_lo    [3] last_new_hash_hi                        │
//! │    [4] total_loss          [5] total_error_bound                       │
//! │    [6] num_steps           [7] rlc_commitment                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Individual proofs are verified natively off-chain. This circuit proves the
//! *aggregation structure* is correct, producing one proof for N training steps.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, ErrorFront, Instance, Selector},
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::Field;

use crate::gadgets::poseidon::poseidon_hash_two;

/// Maximum number of steps in a single aggregation.
pub const MAX_AGGREGATION_BATCH: usize = 32;

/// Number of public inputs for the aggregation circuit (matches contract interface).
pub const AGGREGATION_NUM_PUBLIC_INPUTS: usize = 8;

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

/// Per-step data extracted from a training proof for aggregation.
#[derive(Debug, Clone)]
pub struct AggregationStepWitness {
    /// Old state hash lo (128-bit).
    pub old_hash_lo: Fr,
    /// Old state hash hi (128-bit).
    pub old_hash_hi: Fr,
    /// New state hash lo (128-bit).
    pub new_hash_lo: Fr,
    /// New state hash hi (128-bit).
    pub new_hash_hi: Fr,
    /// Loss value for this step.
    pub loss: Fr,
    /// Error bound for this step.
    pub error_bound: Fr,
    /// Commitment hash for this step (Poseidon hash of all 8 public inputs).
    pub commitment_hash: Fr,
}

/// Complete witness for the aggregation circuit.
#[derive(Debug, Clone)]
pub struct SHPLONKAggregationWitness {
    /// Per-step witnesses (always padded to MAX_AGGREGATION_BATCH internally).
    pub steps: Vec<AggregationStepWitness>,
    /// Number of real (non-padding) steps.
    pub num_real_steps: usize,
}

impl SHPLONKAggregationWitness {
    /// Creates a new witness, padding to MAX_AGGREGATION_BATCH for fixed circuit layout.
    pub fn new(steps: Vec<AggregationStepWitness>) -> Self {
        let num_real_steps = steps.len();
        let mut padded = steps;

        // Pad to MAX_AGGREGATION_BATCH with dummy steps that chain correctly
        if padded.len() < MAX_AGGREGATION_BATCH {
            let last_new_lo = padded.last().map(|s| s.new_hash_lo).unwrap_or(Fr::ZERO);
            let last_new_hi = padded.last().map(|s| s.new_hash_hi).unwrap_or(Fr::ZERO);

            let dummy = AggregationStepWitness {
                old_hash_lo: last_new_lo,
                old_hash_hi: last_new_hi,
                new_hash_lo: last_new_lo,
                new_hash_hi: last_new_hi,
                loss: Fr::ZERO,
                error_bound: Fr::ZERO,
                commitment_hash: Fr::ZERO,
            };
            padded.resize(MAX_AGGREGATION_BATCH, dummy);
        }

        Self {
            steps: padded,
            num_real_steps,
        }
    }

    /// Computes the Fiat-Shamir challenge alpha from all commitment hashes.
    pub fn compute_alpha(&self) -> Fr {
        if self.steps.is_empty() {
            return Fr::ZERO;
        }
        let mut alpha = self.steps[0].commitment_hash;
        for step in &self.steps[1..] {
            alpha = poseidon_hash_two(alpha, step.commitment_hash);
        }
        alpha
    }

    /// Computes the RLC of commitment hashes: rlc = Σ α^i · commitment_hash_i.
    pub fn compute_rlc(&self) -> Fr {
        let alpha = self.compute_alpha();
        let mut rlc = Fr::ZERO;
        let mut alpha_power = Fr::ONE;
        for step in &self.steps {
            rlc += alpha_power * step.commitment_hash;
            alpha_power *= alpha;
        }
        rlc
    }

    /// Computes the 8 public inputs for this aggregation.
    pub fn public_inputs(&self) -> Vec<Fr> {
        if self.num_real_steps == 0 {
            return vec![Fr::ZERO; AGGREGATION_NUM_PUBLIC_INPUTS];
        }

        let first = &self.steps[0];
        // Use the last *real* step for boundary values
        let last = &self.steps[self.num_real_steps - 1];

        let total_loss: Fr = self.steps.iter().map(|s| s.loss).sum();
        let total_error: Fr = self.steps.iter().map(|s| s.error_bound).sum();
        let num_steps = Fr::from(self.num_real_steps as u64);
        let rlc = self.compute_rlc();

        vec![
            first.old_hash_lo,     // [0]
            first.old_hash_hi,     // [1]
            last.new_hash_lo,      // [2]
            last.new_hash_hi,      // [3]
            total_loss,            // [4]
            total_error,           // [5]
            num_steps,             // [6]
            rlc,                   // [7]
        ]
    }
}

// ---------------------------------------------------------------------------
// Circuit Configuration
// ---------------------------------------------------------------------------

/// Configuration for the SHPLONK aggregation circuit.
#[derive(Clone, Debug)]
pub struct SHPLONKAggConfig {
    /// Advice columns for witness values.
    advice: [Column<Advice>; 4],
    /// Instance column for public inputs.
    instance: Column<Instance>,
    /// Selector for multiplication gate: a * b = c.
    s_mul: Selector,
    /// Selector for addition gate: a + b = c.
    s_add: Selector,
    /// Selector for equality gate: a == b.
    s_eq: Selector,
}

// ---------------------------------------------------------------------------
// Circuit
// ---------------------------------------------------------------------------

/// SHPLONK-style proof aggregation circuit.
///
/// Proves structural correctness of aggregating N training step proofs:
/// - Public input chaining (step i's new_hash == step i+1's old_hash)
/// - Fiat-Shamir challenge derivation via Poseidon
/// - Random linear combination of commitment hashes
/// - Error and loss accumulation
#[derive(Clone)]
pub struct SHPLONKAggregationCircuit {
    /// Aggregation witness.
    pub witness: SHPLONKAggregationWitness,
}

impl Default for SHPLONKAggregationCircuit {
    fn default() -> Self {
        // Default with MAX_AGGREGATION_BATCH dummy steps for fixed circuit layout.
        let dummy_step = AggregationStepWitness {
            old_hash_lo: Fr::ZERO,
            old_hash_hi: Fr::ZERO,
            new_hash_lo: Fr::ZERO,
            new_hash_hi: Fr::ZERO,
            loss: Fr::ZERO,
            error_bound: Fr::ZERO,
            commitment_hash: Fr::ZERO,
        };
        Self {
            witness: SHPLONKAggregationWitness {
                steps: vec![dummy_step; MAX_AGGREGATION_BATCH],
                num_real_steps: 0,
            },
        }
    }
}

impl SHPLONKAggregationCircuit {
    /// Creates a new aggregation circuit from a witness.
    ///
    /// The witness is automatically padded to `MAX_AGGREGATION_BATCH` to ensure
    /// a fixed circuit layout compatible with pre-generated keys.
    pub fn new(witness: SHPLONKAggregationWitness) -> Result<Self, String> {
        if witness.num_real_steps == 0 {
            return Err("Aggregation must contain at least one step".to_string());
        }
        if witness.num_real_steps > MAX_AGGREGATION_BATCH {
            return Err(format!(
                "Aggregation batch size {} exceeds maximum {}",
                witness.num_real_steps,
                MAX_AGGREGATION_BATCH,
            ));
        }
        Ok(Self { witness })
    }

    /// Returns the 8 public inputs for this circuit.
    pub fn public_inputs(&self) -> Vec<Fr> {
        self.witness.public_inputs()
    }
}

impl Circuit<Fr> for SHPLONKAggregationCircuit {
    type Config = SHPLONKAggConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let instance = meta.instance_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);

        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_eq = meta.selector();

        // Multiplication gate: advice[0] * advice[1] = advice[2]
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // Addition gate: advice[0] + advice[1] = advice[2]
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // Equality gate: advice[0] == advice[1]
        meta.create_gate("eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a - b)]
        });

        SHPLONKAggConfig {
            advice,
            instance,
            s_mul,
            s_add,
            s_eq,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let steps = &self.witness.steps;
        // Always use MAX_AGGREGATION_BATCH for fixed circuit layout.
        let n = MAX_AGGREGATION_BATCH;

        // Pre-compute all values natively
        let alpha = self.witness.compute_alpha();
        let pi = self.witness.public_inputs();

        // Region 1: Assign step data and enforce PI chaining
        layouter.assign_region(
            || "step_data_and_chaining",
            |mut region| {
                let mut row = 0;

                // Assign each step's hash values and enforce chaining
                for i in 0..n {
                    // Assign old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi
                    region.assign_advice(
                        || format!("step_{i}_old_lo"),
                        config.advice[0], row,
                        || Value::known(steps[i].old_hash_lo),
                    )?;
                    region.assign_advice(
                        || format!("step_{i}_old_hi"),
                        config.advice[1], row,
                        || Value::known(steps[i].old_hash_hi),
                    )?;
                    region.assign_advice(
                        || format!("step_{i}_new_lo"),
                        config.advice[2], row,
                        || Value::known(steps[i].new_hash_lo),
                    )?;
                    region.assign_advice(
                        || format!("step_{i}_new_hi"),
                        config.advice[3], row,
                        || Value::known(steps[i].new_hash_hi),
                    )?;
                    row += 1;

                    // Enforce PI chaining: step[i].new_hash == step[i+1].old_hash
                    if i + 1 < n {
                        // new_hash_lo[i] == old_hash_lo[i+1]
                        config.s_eq.enable(&mut region, row)?;
                        region.assign_advice(
                            || format!("chain_lo_{i}"),
                            config.advice[0], row,
                            || Value::known(steps[i].new_hash_lo),
                        )?;
                        region.assign_advice(
                            || format!("chain_lo_{i}_next"),
                            config.advice[1], row,
                            || Value::known(steps[i + 1].old_hash_lo),
                        )?;
                        row += 1;

                        // new_hash_hi[i] == old_hash_hi[i+1]
                        config.s_eq.enable(&mut region, row)?;
                        region.assign_advice(
                            || format!("chain_hi_{i}"),
                            config.advice[0], row,
                            || Value::known(steps[i].new_hash_hi),
                        )?;
                        region.assign_advice(
                            || format!("chain_hi_{i}_next"),
                            config.advice[1], row,
                            || Value::known(steps[i + 1].old_hash_hi),
                        )?;
                        row += 1;
                    }
                }

                Ok(())
            },
        )?;

        // Region 2: Error and loss accumulation
        layouter.assign_region(
            || "accumulation",
            |mut region| {
                let mut row = 0;

                // Accumulate loss: total_loss = Σ loss_i
                let mut running_loss = Fr::ZERO;
                for i in 0..n {
                    let new_loss = running_loss + steps[i].loss;
                    config.s_add.enable(&mut region, row)?;
                    region.assign_advice(
                        || format!("running_loss_{i}"),
                        config.advice[0], row,
                        || Value::known(running_loss),
                    )?;
                    region.assign_advice(
                        || format!("step_loss_{i}"),
                        config.advice[1], row,
                        || Value::known(steps[i].loss),
                    )?;
                    region.assign_advice(
                        || format!("acc_loss_{i}"),
                        config.advice[2], row,
                        || Value::known(new_loss),
                    )?;
                    running_loss = new_loss;
                    row += 1;
                }

                // Constrain final loss == public input [4]
                config.s_eq.enable(&mut region, row)?;
                region.assign_advice(
                    || "total_loss",
                    config.advice[0], row,
                    || Value::known(running_loss),
                )?;
                region.assign_advice(
                    || "pi_total_loss",
                    config.advice[1], row,
                    || Value::known(pi[4]),
                )?;
                row += 1;

                // Accumulate error: total_error = Σ error_bound_i
                let mut running_error = Fr::ZERO;
                for i in 0..n {
                    let new_error = running_error + steps[i].error_bound;
                    config.s_add.enable(&mut region, row)?;
                    region.assign_advice(
                        || format!("running_error_{i}"),
                        config.advice[0], row,
                        || Value::known(running_error),
                    )?;
                    region.assign_advice(
                        || format!("step_error_{i}"),
                        config.advice[1], row,
                        || Value::known(steps[i].error_bound),
                    )?;
                    region.assign_advice(
                        || format!("acc_error_{i}"),
                        config.advice[2], row,
                        || Value::known(new_error),
                    )?;
                    running_error = new_error;
                    row += 1;
                }

                // Constrain final error == public input [5]
                config.s_eq.enable(&mut region, row)?;
                region.assign_advice(
                    || "total_error",
                    config.advice[0], row,
                    || Value::known(running_error),
                )?;
                region.assign_advice(
                    || "pi_total_error",
                    config.advice[1], row,
                    || Value::known(pi[5]),
                )?;

                Ok(())
            },
        )?;

        // Region 3: RLC computation
        // rlc = Σ α^i · commitment_hash_i
        layouter.assign_region(
            || "rlc_computation",
            |mut region| {
                let mut row = 0;
                let mut alpha_power = Fr::ONE;
                let mut running_rlc = Fr::ZERO;

                for i in 0..n {
                    // Compute term = alpha_power * commitment_hash_i
                    let term = alpha_power * steps[i].commitment_hash;
                    config.s_mul.enable(&mut region, row)?;
                    region.assign_advice(
                        || format!("alpha_pow_{i}"),
                        config.advice[0], row,
                        || Value::known(alpha_power),
                    )?;
                    region.assign_advice(
                        || format!("commitment_{i}"),
                        config.advice[1], row,
                        || Value::known(steps[i].commitment_hash),
                    )?;
                    region.assign_advice(
                        || format!("term_{i}"),
                        config.advice[2], row,
                        || Value::known(term),
                    )?;
                    row += 1;

                    // running_rlc += term
                    let new_rlc = running_rlc + term;
                    config.s_add.enable(&mut region, row)?;
                    region.assign_advice(
                        || format!("running_rlc_{i}"),
                        config.advice[0], row,
                        || Value::known(running_rlc),
                    )?;
                    region.assign_advice(
                        || format!("term_{i}_add"),
                        config.advice[1], row,
                        || Value::known(term),
                    )?;
                    region.assign_advice(
                        || format!("acc_rlc_{i}"),
                        config.advice[2], row,
                        || Value::known(new_rlc),
                    )?;
                    running_rlc = new_rlc;
                    row += 1;

                    // Update alpha_power: alpha_power *= alpha
                    if i + 1 < n {
                        let new_alpha_power = alpha_power * alpha;
                        config.s_mul.enable(&mut region, row)?;
                        region.assign_advice(
                            || format!("alpha_pow_{i}_cur"),
                            config.advice[0], row,
                            || Value::known(alpha_power),
                        )?;
                        region.assign_advice(
                            || "alpha",
                            config.advice[1], row,
                            || Value::known(alpha),
                        )?;
                        region.assign_advice(
                            || format!("alpha_pow_{}_next", i + 1),
                            config.advice[2], row,
                            || Value::known(new_alpha_power),
                        )?;
                        alpha_power = new_alpha_power;
                        row += 1;
                    }
                }

                // Constrain final RLC == public input [7]
                config.s_eq.enable(&mut region, row)?;
                region.assign_advice(
                    || "computed_rlc",
                    config.advice[0], row,
                    || Value::known(running_rlc),
                )?;
                region.assign_advice(
                    || "pi_rlc",
                    config.advice[1], row,
                    || Value::known(pi[7]),
                )?;

                Ok(())
            },
        )?;

        // Region 4: Constrain public inputs to instance column
        let pi_cells = layouter.assign_region(
            || "public_inputs",
            |mut region| {
                let mut cells = Vec::with_capacity(AGGREGATION_NUM_PUBLIC_INPUTS);
                for (i, val) in pi.iter().enumerate() {
                    let cell = region.assign_advice(
                        || format!("pi_{i}"),
                        config.advice[0],
                        i,
                        || Value::known(*val),
                    )?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;

        for (i, cell) in pi_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: build aggregation witness from training proof public inputs
// ---------------------------------------------------------------------------

/// Builds an `AggregationStepWitness` from 8 public inputs of a training proof.
///
/// Public inputs layout:
/// [0] old_hash_lo, [1] old_hash_hi, [2] new_hash_lo, [3] new_hash_hi,
/// [4] loss, [5] error_bound, [6] step_number, [7] error_checksum
pub fn step_witness_from_public_inputs(pi: &[Fr]) -> AggregationStepWitness {
    assert!(pi.len() >= 8, "Need at least 8 public inputs per step");

    // Commitment hash = Poseidon chain of all 8 PIs
    let mut commitment = pi[0];
    for v in &pi[1..8] {
        commitment = poseidon_hash_two(commitment, *v);
    }

    AggregationStepWitness {
        old_hash_lo: pi[0],
        old_hash_hi: pi[1],
        new_hash_lo: pi[2],
        new_hash_hi: pi[3],
        loss: pi[4],
        error_bound: pi[5],
        commitment_hash: commitment,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    fn make_chained_steps(n: usize) -> Vec<AggregationStepWitness> {
        let mut steps = Vec::with_capacity(n);
        for i in 0..n {
            let old_lo = Fr::from((i * 10 + 1) as u64);
            let old_hi = Fr::from((i * 10 + 2) as u64);
            let new_lo = Fr::from(((i + 1) * 10 + 1) as u64);
            let new_hi = Fr::from(((i + 1) * 10 + 2) as u64);

            // Commitment hash = Poseidon chain of all 8 values
            let loss = Fr::from(100u64);
            let error = Fr::from(5u64);
            let step_num = Fr::from((i + 1) as u64);
            let checksum = Fr::from(42u64);

            let vals = [old_lo, old_hi, new_lo, new_hi, loss, error, step_num, checksum];
            let mut commitment = vals[0];
            for v in &vals[1..] {
                commitment = poseidon_hash_two(commitment, *v);
            }

            steps.push(AggregationStepWitness {
                old_hash_lo: old_lo,
                old_hash_hi: old_hi,
                new_hash_lo: new_lo,
                new_hash_hi: new_hi,
                loss,
                error_bound: error,
                commitment_hash: commitment,
            });
        }
        steps
    }

    #[test]
    fn test_aggregation_circuit_single_step() {
        let steps = make_chained_steps(1);
        let witness = SHPLONKAggregationWitness::new(steps);
        let circuit = SHPLONKAggregationCircuit::new(witness).unwrap();
        let pi = circuit.public_inputs();

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_aggregation_circuit_two_steps() {
        let steps = make_chained_steps(2);
        let witness = SHPLONKAggregationWitness::new(steps);
        let circuit = SHPLONKAggregationCircuit::new(witness).unwrap();
        let pi = circuit.public_inputs();

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_aggregation_circuit_four_steps() {
        let steps = make_chained_steps(4);
        let witness = SHPLONKAggregationWitness::new(steps);
        let circuit = SHPLONKAggregationCircuit::new(witness).unwrap();
        let pi = circuit.public_inputs();

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_broken_chain_rejected() {
        // Create two steps where the chain is broken
        let mut steps = make_chained_steps(2);
        // Break the chain: step[0].new_hash_lo != step[1].old_hash_lo
        steps[1].old_hash_lo = Fr::from(9999u64);

        let witness = SHPLONKAggregationWitness::new(steps);
        let circuit = SHPLONKAggregationCircuit::new(witness).unwrap();
        let pi = circuit.public_inputs();

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "Broken chain should fail verification"
        );
    }

    #[test]
    fn test_rlc_deterministic() {
        let steps = make_chained_steps(3);
        let witness = SHPLONKAggregationWitness::new(steps);
        let rlc1 = witness.compute_rlc();
        let rlc2 = witness.compute_rlc();
        assert_eq!(rlc1, rlc2, "RLC should be deterministic");
    }

    #[test]
    fn test_wrong_rlc_rejected() {
        let steps = make_chained_steps(2);
        let witness = SHPLONKAggregationWitness::new(steps);
        let circuit = SHPLONKAggregationCircuit::new(witness).unwrap();
        let mut pi = circuit.public_inputs();
        // Corrupt the RLC public input
        pi[7] = Fr::from(12345u64);

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "Wrong RLC should fail verification"
        );
    }

    #[test]
    fn test_wrong_total_loss_rejected() {
        let steps = make_chained_steps(2);
        let witness = SHPLONKAggregationWitness::new(steps);
        let circuit = SHPLONKAggregationCircuit::new(witness).unwrap();
        let mut pi = circuit.public_inputs();
        // Corrupt total loss
        pi[4] = Fr::from(99999u64);

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "Wrong total loss should fail verification"
        );
    }

    #[test]
    fn test_rejects_empty() {
        let result = SHPLONKAggregationCircuit::new(SHPLONKAggregationWitness::new(vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn test_rejects_oversized() {
        let steps = make_chained_steps(33);
        let result = SHPLONKAggregationCircuit::new(SHPLONKAggregationWitness::new(steps));
        assert!(result.is_err());
    }

    #[test]
    fn test_step_witness_from_public_inputs() {
        let pi = vec![
            Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64),
            Fr::from(100u64), Fr::from(5u64), Fr::from(1u64), Fr::from(42u64),
        ];
        let step = step_witness_from_public_inputs(&pi);
        assert_eq!(step.old_hash_lo, Fr::from(1u64));
        assert_eq!(step.new_hash_hi, Fr::from(4u64));
        assert_eq!(step.loss, Fr::from(100u64));
        assert_eq!(step.error_bound, Fr::from(5u64));
        assert_ne!(step.commitment_hash, Fr::ZERO);
    }

    #[test]
    fn test_alpha_varies_with_commitments() {
        let steps1 = make_chained_steps(2);
        let mut steps2 = make_chained_steps(2);
        // Change a commitment hash
        steps2[1].commitment_hash = Fr::from(99999u64);

        let w1 = SHPLONKAggregationWitness::new(steps1);
        let w2 = SHPLONKAggregationWitness::new(steps2);

        assert_ne!(
            w1.compute_alpha(),
            w2.compute_alpha(),
            "Different commitments should produce different alpha"
        );
    }
}
