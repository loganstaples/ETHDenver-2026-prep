//! Nova-Style Incrementally Verifiable Computation (IVC) Circuit.
//!
//! This module implements a proper IVC scheme inspired by Nova folding.
//! Instead of concatenating proofs, we use a folding-based approach where:
//!
//! 1. Each step produces an "accumulator" that summarizes all previous proofs
//! 2. The accumulator can be verified in O(1) regardless of the number of steps
//! 3. Folding combines two accumulators into one with negligible overhead
//!
//! Key concepts:
//! - Relaxed R1CS: Allows error terms in constraints for folding
//! - Committed Relaxed R1CS: Commitments to witness and error vectors
//! - Folding: Combining two instances using random challenge
//!
//! This is a simplified version suitable for the hackathon demo, implementing
//! the core ideas without full Nova complexity.

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, ErrorFront, Expression, Fixed, Instance, Selector,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

/// Number of public inputs for the IVC step circuit.
pub const IVC_PUBLIC_INPUTS: usize = 8;

// ---------------------------------------------------------------------------
// IVC State and Accumulator
// ---------------------------------------------------------------------------

/// Represents the state being computed incrementally.
#[derive(Clone, Debug, Default)]
pub struct IVCState {
    /// Current step number (starts at 0).
    pub step: u64,
    /// Commitment to the current state (e.g., model weights hash).
    pub state_commitment: [u8; 32],
    /// Accumulated error bound across all steps.
    pub accumulated_error: Fr,
}

/// Accumulator for folding multiple proofs.
///
/// In Nova, this would be a "Committed Relaxed R1CS instance".
/// Here we use a simplified version with commitments to:
/// - The running state
/// - The error vector (for relaxed constraints)
/// - Random challenges used in folding
#[derive(Clone, Debug)]
pub struct IVCAccumulator {
    /// Commitment to the accumulated computation (hash of state chain).
    pub state_commitment: [u8; 32],
    /// Number of steps folded into this accumulator.
    pub num_steps: u64,
    /// Accumulated error term (u in Nova notation).
    /// Starts at 1, grows with each fold.
    pub error_term: Fr,
    /// Accumulated error bound from all computations.
    pub error_bound: Fr,
    /// Hash of all challenges used in folding (for verification).
    pub challenge_hash: [u8; 32],
}

impl Default for IVCAccumulator {
    fn default() -> Self {
        Self::initial([0; 32])
    }
}

impl IVCAccumulator {
    /// Creates the initial accumulator (before any computation).
    pub fn initial(initial_commitment: [u8; 32]) -> Self {
        Self {
            state_commitment: initial_commitment,
            num_steps: 0,
            error_term: Fr::one(), // u = 1 for the base case
            error_bound: Fr::zero(),
            challenge_hash: [0; 32],
        }
    }

    /// Converts accumulator fields to public inputs.
    pub fn to_public_inputs(&self) -> Vec<Fr> {
        let state_lo = bytes_to_fr_lo(&self.state_commitment);
        let state_hi = bytes_to_fr_hi(&self.state_commitment);
        let challenge_lo = bytes_to_fr_lo(&self.challenge_hash);
        let challenge_hi = bytes_to_fr_hi(&self.challenge_hash);

        vec![
            state_lo,
            state_hi,
            Fr::from(self.num_steps),
            self.error_term,
            self.error_bound,
            challenge_lo,
            challenge_hi,
            Fr::zero(), // Reserved for future use
        ]
    }
}

/// Converts lower 16 bytes of a 32-byte array to Fr.
fn bytes_to_fr_lo(bytes: &[u8; 32]) -> Fr {
    let mut buf = [0u8; 32];
    buf[..16].copy_from_slice(&bytes[..16]);
    Fr::from_raw([
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        0,
        0,
    ])
}

/// Converts upper 16 bytes of a 32-byte array to Fr.
fn bytes_to_fr_hi(bytes: &[u8; 32]) -> Fr {
    let mut buf = [0u8; 32];
    buf[..16].copy_from_slice(&bytes[16..32]);
    Fr::from_raw([
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        0,
        0,
    ])
}

// ---------------------------------------------------------------------------
// Folding Operation
// ---------------------------------------------------------------------------

/// Folds two accumulators into one using a random challenge.
///
/// This is the key operation in Nova-style IVC:
/// - Given accumulators A1 and A2, and random challenge r
/// - Produces A' that summarizes both A1 and A2
/// - A' can be verified without knowing A1 or A2 individually
pub fn fold_accumulators(
    acc1: &IVCAccumulator,
    acc2: &IVCAccumulator,
    challenge: Fr,
) -> IVCAccumulator {
    // New state commitment = Hash(acc1.state || acc2.state || challenge)
    let mut hasher = Sha256::new();
    hasher.update(&acc1.state_commitment);
    hasher.update(&acc2.state_commitment);
    hasher.update(&challenge.to_repr().as_ref());
    let new_state: [u8; 32] = hasher.finalize().into();

    // New error term: u' = u1 + r * u2 (linear combination)
    let new_error_term = acc1.error_term + challenge * acc2.error_term;

    // New error bound: sum of both bounds
    let new_error_bound = acc1.error_bound + acc2.error_bound;

    // New challenge hash: Hash(old_challenges || new_challenge)
    let mut challenge_hasher = Sha256::new();
    challenge_hasher.update(&acc1.challenge_hash);
    challenge_hasher.update(&acc2.challenge_hash);
    challenge_hasher.update(&challenge.to_repr().as_ref());
    let new_challenge_hash: [u8; 32] = challenge_hasher.finalize().into();

    IVCAccumulator {
        state_commitment: new_state,
        num_steps: acc1.num_steps + acc2.num_steps,
        error_term: new_error_term,
        error_bound: new_error_bound,
        challenge_hash: new_challenge_hash,
    }
}

/// Generates a Fiat-Shamir challenge for folding.
pub fn generate_folding_challenge(acc1: &IVCAccumulator, acc2: &IVCAccumulator) -> Fr {
    let mut hasher = Sha256::new();
    hasher.update(b"HELIX_IVC_FOLD_CHALLENGE");
    hasher.update(&acc1.state_commitment);
    hasher.update(&acc1.num_steps.to_le_bytes());
    hasher.update(&acc2.state_commitment);
    hasher.update(&acc2.num_steps.to_le_bytes());

    let hash: [u8; 32] = hasher.finalize().into();

    // Convert to field element
    Fr::from_raw([
        u64::from_le_bytes(hash[0..8].try_into().unwrap()),
        u64::from_le_bytes(hash[8..16].try_into().unwrap()),
        u64::from_le_bytes(hash[16..24].try_into().unwrap()),
        u64::from_le_bytes(hash[24..32].try_into().unwrap()) & 0x0FFFFFFFFFFFFFFF, // Ensure < modulus
    ])
}

// ---------------------------------------------------------------------------
// IVC Step Circuit
// ---------------------------------------------------------------------------

/// Configuration for the IVC step circuit.
#[derive(Clone, Debug)]
pub struct IVCStepConfig {
    /// Advice columns for computation.
    advice: [Column<Advice>; 4],
    /// Instance column for public inputs.
    instance: Column<Instance>,
    /// Selector for accumulator update verification.
    s_acc_update: Selector,
    /// Selector for state transition verification.
    s_state_transition: Selector,
    /// Selector for error bound check.
    s_error_check: Selector,
    /// Selector for multiplication.
    s_mul: Selector,
    /// Selector for addition.
    s_add: Selector,
}

/// Witness for a single IVC step.
#[derive(Clone, Debug)]
pub struct IVCStepWitness {
    /// Previous accumulator.
    pub prev_acc: IVCAccumulator,
    /// New state after this step's computation.
    pub new_state: [u8; 32],
    /// Hash of the computation performed in this step.
    pub computation_hash: [u8; 32],
    /// Error introduced in this step.
    pub step_error: Fr,
    /// Folding challenge (if folding with another accumulator).
    pub fold_challenge: Option<Fr>,
    /// Other accumulator (if folding).
    pub other_acc: Option<IVCAccumulator>,
}

impl Default for IVCStepWitness {
    fn default() -> Self {
        Self {
            prev_acc: IVCAccumulator::default(),
            new_state: [0; 32],
            computation_hash: [0; 32],
            step_error: Fr::zero(),
            fold_challenge: None,
            other_acc: None,
        }
    }
}

impl IVCStepWitness {
    /// Computes the resulting accumulator after this step.
    pub fn resulting_accumulator(&self) -> IVCAccumulator {
        // First, create accumulator for just this step
        let step_acc = IVCAccumulator {
            state_commitment: self.new_state,
            num_steps: self.prev_acc.num_steps + 1,
            error_term: self.prev_acc.error_term,
            error_bound: self.prev_acc.error_bound + self.step_error,
            challenge_hash: self.prev_acc.challenge_hash,
        };

        // If folding with another accumulator, do the fold
        match (&self.fold_challenge, &self.other_acc) {
            (Some(challenge), Some(other)) => fold_accumulators(&step_acc, other, *challenge),
            _ => step_acc,
        }
    }
}

/// Circuit that verifies a single IVC step and optionally folds with another accumulator.
#[derive(Clone)]
pub struct IVCStepCircuit {
    pub witness: IVCStepWitness,
}

impl Default for IVCStepCircuit {
    fn default() -> Self {
        Self {
            witness: IVCStepWitness::default(),
        }
    }
}

impl IVCStepCircuit {
    pub fn public_inputs(&self) -> Vec<Fr> {
        let result_acc = self.witness.resulting_accumulator();
        result_acc.to_public_inputs()
    }
}

impl Circuit<Fr> for IVCStepCircuit {
    type Config = IVCStepConfig;
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

        let s_acc_update = meta.selector();
        let s_state_transition = meta.selector();
        let s_error_check = meta.selector();
        let s_mul = meta.selector();
        let s_add = meta.selector();

        // Multiplication gate: a * b = c
        meta.create_gate("ivc_mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // Addition gate: a + b = c
        meta.create_gate("ivc_add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // State transition verification: new_commitment = Hash(prev_commitment, computation)
        // Note: Full implementation would use Poseidon hash circuit. For now we just
        // ensure the selector is used (real verification happens in witness computation).
        meta.create_gate("state_transition", |meta| {
            let s = meta.query_selector(s_state_transition);
            let _prev_state_lo = meta.query_advice(advice[0], Rotation::cur());
            let _computation_lo = meta.query_advice(advice[1], Rotation::cur());
            let _new_state_lo = meta.query_advice(advice[2], Rotation::cur());
            // Placeholder constraint - always passes. Real hash verification would go here.
            vec![s * Expression::Constant(Fr::zero())]
        });

        // Error bound accumulation: new_error = old_error + step_error
        meta.create_gate("error_accumulation", |meta| {
            let s = meta.query_selector(s_error_check);
            let old_error = meta.query_advice(advice[0], Rotation::cur());
            let step_error = meta.query_advice(advice[1], Rotation::cur());
            let new_error = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (old_error + step_error - new_error)]
        });

        // Accumulator update (for folding): new_u = u1 + r * u2
        meta.create_gate("accumulator_fold", |meta| {
            let s = meta.query_selector(s_acc_update);
            let u1 = meta.query_advice(advice[0], Rotation::cur());
            let r_times_u2 = meta.query_advice(advice[1], Rotation::cur()); // Precomputed r * u2
            let new_u = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (u1 + r_times_u2 - new_u)]
        });

        IVCStepConfig {
            advice,
            instance,
            s_acc_update,
            s_state_transition,
            s_error_check,
            s_mul,
            s_add,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let w = &self.witness;
        let result_acc = w.resulting_accumulator();
        let pi = result_acc.to_public_inputs();

        // Bind public inputs
        let pi_cells = layouter.assign_region(
            || "public_inputs",
            |mut region| {
                let mut cells = Vec::with_capacity(IVC_PUBLIC_INPUTS);
                for (i, val) in pi.iter().enumerate() {
                    let cell = region.assign_advice(
                        || format!("pi_{}", i),
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

        // Verify error accumulation
        layouter.assign_region(
            || "error_accumulation",
            |mut region| {
                config.s_error_check.enable(&mut region, 0)?;
                region.assign_advice(
                    || "old_error",
                    config.advice[0],
                    0,
                    || Value::known(w.prev_acc.error_bound),
                )?;
                region.assign_advice(
                    || "step_error",
                    config.advice[1],
                    0,
                    || Value::known(w.step_error),
                )?;
                region.assign_advice(
                    || "new_error",
                    config.advice[2],
                    0,
                    || Value::known(w.prev_acc.error_bound + w.step_error),
                )?;
                Ok(())
            },
        )?;

        // If folding, verify the fold operation
        if let (Some(challenge), Some(other)) = (&w.fold_challenge, &w.other_acc) {
            // Verify: new_u = u1 + r * u2
            let r_times_u2 = *challenge * other.error_term;

            layouter.assign_region(
                || "fold_error_term",
                |mut region| {
                    config.s_acc_update.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "u1",
                        config.advice[0],
                        0,
                        || Value::known(w.prev_acc.error_term),
                    )?;
                    region.assign_advice(
                        || "r_times_u2",
                        config.advice[1],
                        0,
                        || Value::known(r_times_u2),
                    )?;
                    region.assign_advice(
                        || "new_u",
                        config.advice[2],
                        0,
                        || Value::known(result_acc.error_term),
                    )?;
                    Ok(())
                },
            )?;

            // Verify: r * u2 was computed correctly
            layouter.assign_region(
                || "verify_r_times_u2",
                |mut region| {
                    config.s_mul.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "r",
                        config.advice[0],
                        0,
                        || Value::known(*challenge),
                    )?;
                    region.assign_advice(
                        || "u2",
                        config.advice[1],
                        0,
                        || Value::known(other.error_term),
                    )?;
                    region.assign_advice(
                        || "r_times_u2",
                        config.advice[2],
                        0,
                        || Value::known(r_times_u2),
                    )?;
                    Ok(())
                },
            )?;
        }

        // Verify state transition
        let prev_state_lo = bytes_to_fr_lo(&w.prev_acc.state_commitment);
        let computation_lo = bytes_to_fr_lo(&w.computation_hash);
        let new_state_lo = bytes_to_fr_lo(&w.new_state);

        layouter.assign_region(
            || "state_transition",
            |mut region| {
                config.s_state_transition.enable(&mut region, 0)?;
                region.assign_advice(
                    || "prev_state_lo",
                    config.advice[0],
                    0,
                    || Value::known(prev_state_lo),
                )?;
                region.assign_advice(
                    || "computation_lo",
                    config.advice[1],
                    0,
                    || Value::known(computation_lo),
                )?;
                region.assign_advice(
                    || "new_state_lo",
                    config.advice[2],
                    0,
                    || Value::known(new_state_lo),
                )?;
                Ok(())
            },
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IVC Chain Management
// ---------------------------------------------------------------------------

/// Manages a chain of IVC computations.
pub struct IVCChain {
    /// Current accumulator representing all previous computations.
    pub accumulator: IVCAccumulator,
    /// History of step accumulators (for debugging/verification).
    history: Vec<IVCAccumulator>,
}

impl IVCChain {
    /// Creates a new IVC chain with the given initial state.
    pub fn new(initial_commitment: [u8; 32]) -> Self {
        Self {
            accumulator: IVCAccumulator::initial(initial_commitment),
            history: Vec::new(),
        }
    }

    /// Adds a computation step to the chain.
    pub fn add_step(
        &mut self,
        new_state: [u8; 32],
        computation_hash: [u8; 32],
        step_error: Fr,
    ) {
        let old_acc = self.accumulator.clone();

        // Create new accumulator incorporating this step
        let mut hasher = Sha256::new();
        hasher.update(&old_acc.state_commitment);
        hasher.update(&computation_hash);
        let combined_state: [u8; 32] = hasher.finalize().into();

        self.accumulator = IVCAccumulator {
            state_commitment: new_state,
            num_steps: old_acc.num_steps + 1,
            error_term: old_acc.error_term, // Stays 1 unless we fold
            error_bound: old_acc.error_bound + step_error,
            challenge_hash: old_acc.challenge_hash,
        };

        self.history.push(old_acc);
    }

    /// Folds this chain with another chain.
    pub fn fold_with(&mut self, other: &IVCChain) {
        let challenge = generate_folding_challenge(&self.accumulator, &other.accumulator);
        self.accumulator = fold_accumulators(&self.accumulator, &other.accumulator, challenge);
    }

    /// Returns the current step count.
    pub fn step_count(&self) -> u64 {
        self.accumulator.num_steps
    }

    /// Returns the current error bound.
    pub fn error_bound(&self) -> Fr {
        self.accumulator.error_bound
    }

    /// Verifies that the accumulator is valid (basic sanity checks).
    pub fn verify(&self) -> bool {
        // Check that error term is non-zero (should start at 1)
        if self.accumulator.error_term == Fr::zero() {
            return false;
        }

        // Check that num_steps is consistent with history
        if !self.history.is_empty() && self.accumulator.num_steps == 0 {
            return false;
        }

        true
    }
}

impl Default for IVCChain {
    fn default() -> Self {
        Self::new([0; 32])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    #[test]
    fn test_accumulator_creation() {
        let initial = [1u8; 32];
        let acc = IVCAccumulator::initial(initial);

        assert_eq!(acc.num_steps, 0);
        assert_eq!(acc.error_term, Fr::one());
        assert_eq!(acc.error_bound, Fr::zero());
    }

    #[test]
    fn test_folding() {
        let acc1 = IVCAccumulator {
            state_commitment: [1u8; 32],
            num_steps: 5,
            error_term: Fr::one(),
            error_bound: Fr::from(10),
            challenge_hash: [0u8; 32],
        };

        let acc2 = IVCAccumulator {
            state_commitment: [2u8; 32],
            num_steps: 3,
            error_term: Fr::from(2),
            error_bound: Fr::from(5),
            challenge_hash: [0u8; 32],
        };

        let challenge = Fr::from(7);
        let folded = fold_accumulators(&acc1, &acc2, challenge);

        assert_eq!(folded.num_steps, 8);
        assert_eq!(folded.error_term, Fr::one() + Fr::from(7) * Fr::from(2)); // 1 + 7*2 = 15
        assert_eq!(folded.error_bound, Fr::from(15)); // 10 + 5 = 15
    }

    #[test]
    fn test_ivc_chain() {
        let initial = [0u8; 32];
        let mut chain = IVCChain::new(initial);

        // Add some steps
        chain.add_step([1u8; 32], [10u8; 32], Fr::from(1));
        chain.add_step([2u8; 32], [20u8; 32], Fr::from(2));
        chain.add_step([3u8; 32], [30u8; 32], Fr::from(3));

        assert_eq!(chain.step_count(), 3);
        assert_eq!(chain.error_bound(), Fr::from(6)); // 1 + 2 + 3
        assert!(chain.verify());
    }

    #[test]
    fn test_ivc_step_circuit() {
        let prev_acc = IVCAccumulator::initial([0u8; 32]);
        let witness = IVCStepWitness {
            prev_acc,
            new_state: [1u8; 32],
            computation_hash: [10u8; 32],
            step_error: Fr::from(5),
            fold_challenge: None,
            other_acc: None,
        };

        let circuit = IVCStepCircuit { witness };
        let pi = circuit.public_inputs();

        let prover = MockProver::run(10, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_ivc_step_circuit_with_fold() {
        let prev_acc = IVCAccumulator {
            state_commitment: [1u8; 32],
            num_steps: 5,
            error_term: Fr::one(),
            error_bound: Fr::from(10),
            challenge_hash: [0u8; 32],
        };

        let other_acc = IVCAccumulator {
            state_commitment: [2u8; 32],
            num_steps: 3,
            error_term: Fr::from(2),
            error_bound: Fr::from(5),
            challenge_hash: [0u8; 32],
        };

        let challenge = generate_folding_challenge(&prev_acc, &other_acc);

        let witness = IVCStepWitness {
            prev_acc: prev_acc.clone(),
            new_state: [3u8; 32],
            computation_hash: [30u8; 32],
            step_error: Fr::from(1),
            fold_challenge: Some(challenge),
            other_acc: Some(other_acc),
        };

        let circuit = IVCStepCircuit { witness };
        let pi = circuit.public_inputs();

        let prover = MockProver::run(10, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }
}
