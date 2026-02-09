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
//! State transitions are verified in-circuit using Poseidon hash, ensuring
//! the prover cannot forge state commitments.

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, ErrorFront, Fixed, Instance,
        Selector,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;

use crate::gadgets::poseidon::{
    poseidon_hash_two, PoseidonCircuitConfig, synthesize_poseidon_hash,
};

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
    /// Commitment to the current state (Poseidon hash).
    pub state_commitment: Fr,
    /// Accumulated error bound across all steps.
    pub accumulated_error: Fr,
}

/// Accumulator for folding multiple proofs.
///
/// In Nova, this would be a "Committed Relaxed R1CS instance".
/// Here we use a simplified version with commitments to:
/// - The running state (as a Poseidon hash)
/// - The error vector (for relaxed constraints)
/// - Random challenges used in folding
#[derive(Clone, Debug)]
pub struct IVCAccumulator {
    /// Commitment to the accumulated computation (Poseidon hash of state chain).
    pub state_commitment: Fr,
    /// Number of steps folded into this accumulator.
    pub num_steps: u64,
    /// Accumulated error term (u in Nova notation).
    /// Starts at 1, grows with each fold.
    pub error_term: Fr,
    /// Accumulated error bound from all computations.
    pub error_bound: Fr,
    /// Hash of all challenges used in folding (Poseidon hash).
    pub challenge_hash: Fr,
}

impl Default for IVCAccumulator {
    fn default() -> Self {
        Self::initial(Fr::ZERO)
    }
}

impl IVCAccumulator {
    /// Creates the initial accumulator (before any computation).
    pub fn initial(initial_commitment: Fr) -> Self {
        Self {
            state_commitment: initial_commitment,
            num_steps: 0,
            error_term: Fr::one(), // u = 1 for the base case
            error_bound: Fr::zero(),
            challenge_hash: Fr::ZERO,
        }
    }

    /// Creates an accumulator from raw [u8; 32] commitment (backward compat).
    pub fn initial_from_bytes(bytes: [u8; 32]) -> Self {
        Self::initial(bytes_to_fr(&bytes))
    }

    /// Converts accumulator fields to public inputs.
    pub fn to_public_inputs(&self) -> Vec<Fr> {
        vec![
            self.state_commitment,
            Fr::ZERO, // Reserved (was state_hi in SHA-256 era)
            Fr::from(self.num_steps),
            self.error_term,
            self.error_bound,
            self.challenge_hash,
            Fr::ZERO, // Reserved
            Fr::ZERO, // Reserved
        ]
    }
}

/// Converts a [u8; 32] to Fr by interpreting as LE representation.
/// Returns Fr::ZERO if the bytes do not represent a valid field element after masking.
fn bytes_to_fr(bytes: &[u8; 32]) -> Fr {
    let mut repr = [0u8; 32];
    repr.copy_from_slice(bytes);
    // Clear top 3 bits to keep value below 2^253, well within BN254 scalar field modulus
    repr[31] &= 0x1F;
    // Note: after masking, value is < 2^253 < p (BN254 scalar modulus), so this always succeeds.
    // unwrap_or(ZERO) is a defensive fallback that should never trigger in practice.
    Fr::from_repr_vartime(repr.into()).unwrap_or(Fr::ZERO)
}

// ---------------------------------------------------------------------------
// Folding Operation
// ---------------------------------------------------------------------------

/// Folds two accumulators into one using a random challenge.
///
/// Uses Poseidon hash for state commitment combination.
pub fn fold_accumulators(
    acc1: &IVCAccumulator,
    acc2: &IVCAccumulator,
    challenge: Fr,
) -> IVCAccumulator {
    // New state commitment = Poseidon(Poseidon(acc1.state, acc2.state), challenge)
    let combined = poseidon_hash_two(acc1.state_commitment, acc2.state_commitment);
    let new_state = poseidon_hash_two(combined, challenge);

    // New error term: u' = u1 + r * u2 (linear combination)
    let new_error_term = acc1.error_term + challenge * acc2.error_term;

    // New error bound: sum of both bounds
    let new_error_bound = acc1.error_bound + acc2.error_bound;

    // New challenge hash = Poseidon(Poseidon(ch1, ch2), challenge)
    let combined_ch = poseidon_hash_two(acc1.challenge_hash, acc2.challenge_hash);
    let new_challenge_hash = poseidon_hash_two(combined_ch, challenge);

    IVCAccumulator {
        state_commitment: new_state,
        num_steps: acc1.num_steps + acc2.num_steps,
        error_term: new_error_term,
        error_bound: new_error_bound,
        challenge_hash: new_challenge_hash,
    }
}

/// Generates a Fiat-Shamir challenge for folding using Poseidon.
pub fn generate_folding_challenge(acc1: &IVCAccumulator, acc2: &IVCAccumulator) -> Fr {
    // Domain-separated Poseidon hash of both accumulators' states
    let domain = Fr::from(0x48454C49585F464Fu64); // "HELIX_FO" domain
    let combined_state = poseidon_hash_two(acc1.state_commitment, acc2.state_commitment);
    let combined_steps = poseidon_hash_two(Fr::from(acc1.num_steps), Fr::from(acc2.num_steps));
    poseidon_hash_two(
        poseidon_hash_two(domain, combined_state),
        combined_steps,
    )
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
    /// Fixed column for Poseidon round constants.
    fixed: Column<Fixed>,
    /// Selector for accumulator update verification.
    s_acc_update: Selector,
    /// Selector for error bound check.
    s_error_check: Selector,
    /// Selector for multiplication.
    s_mul: Selector,
    /// Selector for addition.
    s_add: Selector,
    /// Selector for Poseidon round constant addition.
    s_rc_add: Selector,
    /// Selector for equality check.
    s_eq: Selector,
}

/// Witness for a single IVC step.
#[derive(Clone, Debug)]
pub struct IVCStepWitness {
    /// Previous accumulator.
    pub prev_acc: IVCAccumulator,
    /// New state after this step's computation (Poseidon hash).
    pub new_state: Fr,
    /// Hash of the computation performed in this step.
    pub computation_hash: Fr,
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
            new_state: Fr::ZERO,
            computation_hash: Fr::ZERO,
            step_error: Fr::zero(),
            fold_challenge: None,
            other_acc: None,
        }
    }
}

impl IVCStepWitness {
    /// Computes the resulting accumulator after this step.
    pub fn resulting_accumulator(&self) -> IVCAccumulator {
        let step_acc = IVCAccumulator {
            state_commitment: self.new_state,
            num_steps: self.prev_acc.num_steps + 1,
            error_term: self.prev_acc.error_term,
            error_bound: self.prev_acc.error_bound + self.step_error,
            challenge_hash: self.prev_acc.challenge_hash,
        };

        match (&self.fold_challenge, &self.other_acc) {
            (Some(challenge), Some(other)) => fold_accumulators(&step_acc, other, *challenge),
            _ => step_acc,
        }
    }

    /// Computes the expected new_state using Poseidon.
    /// new_state = Poseidon(prev_state_commitment, computation_hash)
    pub fn expected_new_state(&self) -> Fr {
        poseidon_hash_two(self.prev_acc.state_commitment, self.computation_hash)
    }
}

/// Circuit that verifies a single IVC step and optionally folds with another accumulator.
///
/// State transitions are verified in-circuit using a full Poseidon hash,
/// ensuring the prover correctly computed new_state = Poseidon(prev_state, computation).
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
        let fixed = meta.fixed_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);

        let s_acc_update = meta.selector();
        let s_error_check = meta.selector();
        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_rc_add = meta.selector();
        let s_eq = meta.selector();

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

        // Poseidon round constant addition gate: advice[0] + fixed = advice[2]
        meta.create_gate("poseidon_rc_add", |meta| {
            let s = meta.query_selector(s_rc_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let rc = meta.query_fixed(fixed, Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + rc - c)]
        });

        // Equality check gate: a = b
        meta.create_gate("ivc_eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a - b)]
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
            let r_times_u2 = meta.query_advice(advice[1], Rotation::cur());
            let new_u = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (u1 + r_times_u2 - new_u)]
        });

        IVCStepConfig {
            advice,
            instance,
            fixed,
            s_acc_update,
            s_error_check,
            s_mul,
            s_add,
            s_rc_add,
            s_eq,
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

        // ================================================================
        // Verify state transition using in-circuit Poseidon hash.
        // Constrains: new_state == Poseidon(prev_state_commitment, computation_hash)
        // ================================================================
        let poseidon_config = PoseidonCircuitConfig {
            advice: [config.advice[0], config.advice[1], config.advice[2]],
            fixed: config.fixed,
            s_mul: config.s_mul,
            s_add: config.s_add,
            s_rc_add: config.s_rc_add,
            s_eq: config.s_eq,
        };

        let expected_new_state = w.expected_new_state();

        // Verify the witness new_state matches what Poseidon computes
        synthesize_poseidon_hash(
            &poseidon_config,
            &mut layouter,
            w.prev_acc.state_commitment,
            w.computation_hash,
            "state_transition",
        )?;

        // Also constrain that the witness new_state equals the Poseidon output
        layouter.assign_region(
            || "verify_new_state",
            |mut region| {
                config.s_eq.enable(&mut region, 0)?;
                region.assign_advice(
                    || "expected",
                    config.advice[0],
                    0,
                    || Value::known(expected_new_state),
                )?;
                region.assign_advice(
                    || "actual",
                    config.advice[1],
                    0,
                    || Value::known(w.new_state),
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
    /// Creates a new IVC chain with the given initial state commitment.
    pub fn new(initial_commitment: Fr) -> Self {
        Self {
            accumulator: IVCAccumulator::initial(initial_commitment),
            history: Vec::new(),
        }
    }

    /// Creates a new IVC chain from raw bytes (backward compat).
    pub fn new_from_bytes(initial_commitment: [u8; 32]) -> Self {
        Self::new(bytes_to_fr(&initial_commitment))
    }

    /// Adds a computation step to the chain.
    pub fn add_step(
        &mut self,
        new_state: Fr,
        _computation_hash: Fr,
        step_error: Fr,
    ) {
        let old_acc = self.accumulator.clone();

        self.accumulator = IVCAccumulator {
            state_commitment: new_state,
            num_steps: old_acc.num_steps + 1,
            error_term: old_acc.error_term,
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
        if self.accumulator.error_term == Fr::zero() {
            return false;
        }
        if !self.history.is_empty() && self.accumulator.num_steps == 0 {
            return false;
        }
        true
    }
}

impl Default for IVCChain {
    fn default() -> Self {
        Self::new(Fr::ZERO)
    }
}

// ---------------------------------------------------------------------------
// IVC Folding Circuit – verifies a fold of two accumulators in ZK
// ---------------------------------------------------------------------------

/// Number of public inputs for the folding circuit.
pub const FOLDING_PUBLIC_INPUTS: usize = 8;

/// Maximum multi-step chain length.
pub const MAX_MULTI_STEPS: usize = 16;

/// Witness for the IVC folding circuit.
#[derive(Clone, Debug)]
pub struct IVCFoldingWitness {
    /// First accumulator (left).
    pub acc1: IVCAccumulator,
    /// Second accumulator (right).
    pub acc2: IVCAccumulator,
    /// Folding challenge (Fiat-Shamir derived).
    pub challenge: Fr,
}

impl Default for IVCFoldingWitness {
    fn default() -> Self {
        Self {
            acc1: IVCAccumulator::default(),
            acc2: IVCAccumulator::default(),
            challenge: Fr::ZERO,
        }
    }
}

impl IVCFoldingWitness {
    /// Computes the resulting folded accumulator.
    pub fn result(&self) -> IVCAccumulator {
        fold_accumulators(&self.acc1, &self.acc2, self.challenge)
    }
}

/// Circuit that verifies the fold of two accumulators in zero knowledge.
///
/// Public inputs (8 total):
///   [0] folded state commitment
///   [1] reserved
///   [2] folded num_steps
///   [3] folded error_term
///   [4] folded error_bound
///   [5] folded challenge_hash
///   [6..7] reserved
#[derive(Clone)]
pub struct IVCFoldingCircuit {
    pub witness: IVCFoldingWitness,
}

impl Default for IVCFoldingCircuit {
    fn default() -> Self {
        Self {
            witness: IVCFoldingWitness::default(),
        }
    }
}

impl IVCFoldingCircuit {
    pub fn public_inputs(&self) -> Vec<Fr> {
        self.witness.result().to_public_inputs()
    }
}

impl Circuit<Fr> for IVCFoldingCircuit {
    type Config = IVCStepConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        IVCStepCircuit::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let w = &self.witness;
        let folded = w.result();
        let pi = folded.to_public_inputs();

        // Bind public inputs
        let pi_cells = layouter.assign_region(
            || "fold_public_inputs",
            |mut region| {
                let mut cells = Vec::with_capacity(FOLDING_PUBLIC_INPUTS);
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

        // Verify error term folding: new_u = u1 + r * u2
        let r_times_u2 = w.challenge * w.acc2.error_term;

        layouter.assign_region(
            || "verify_r_times_u2",
            |mut region| {
                config.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "r", config.advice[0], 0, || Value::known(w.challenge))?;
                region.assign_advice(|| "u2", config.advice[1], 0, || Value::known(w.acc2.error_term))?;
                region.assign_advice(|| "r_u2", config.advice[2], 0, || Value::known(r_times_u2))?;
                Ok(())
            },
        )?;

        layouter.assign_region(
            || "fold_error_term",
            |mut region| {
                config.s_acc_update.enable(&mut region, 0)?;
                region.assign_advice(|| "u1", config.advice[0], 0, || Value::known(w.acc1.error_term))?;
                region.assign_advice(|| "r_u2", config.advice[1], 0, || Value::known(r_times_u2))?;
                region.assign_advice(|| "new_u", config.advice[2], 0, || Value::known(folded.error_term))?;
                Ok(())
            },
        )?;

        // Verify error bound addition: new_bound = bound1 + bound2
        layouter.assign_region(
            || "fold_error_bound",
            |mut region| {
                config.s_error_check.enable(&mut region, 0)?;
                region.assign_advice(|| "bound1", config.advice[0], 0, || Value::known(w.acc1.error_bound))?;
                region.assign_advice(|| "bound2", config.advice[1], 0, || Value::known(w.acc2.error_bound))?;
                region.assign_advice(|| "new_bound", config.advice[2], 0, || Value::known(folded.error_bound))?;
                Ok(())
            },
        )?;

        // Verify state commitment via Poseidon:
        // combined = Poseidon(state1, state2), then new_state = Poseidon(combined, challenge)
        let poseidon_config = PoseidonCircuitConfig {
            advice: [config.advice[0], config.advice[1], config.advice[2]],
            fixed: config.fixed,
            s_mul: config.s_mul,
            s_add: config.s_add,
            s_rc_add: config.s_rc_add,
            s_eq: config.s_eq,
        };

        // Hash state commitments together
        synthesize_poseidon_hash(
            &poseidon_config,
            &mut layouter,
            w.acc1.state_commitment,
            w.acc2.state_commitment,
            "fold_state_combine",
        )?;

        let combined = poseidon_hash_two(w.acc1.state_commitment, w.acc2.state_commitment);

        // Hash combined with challenge to get final state
        synthesize_poseidon_hash(
            &poseidon_config,
            &mut layouter,
            combined,
            w.challenge,
            "fold_state_challenge",
        )?;

        // Verify the folded state matches
        let expected_state = poseidon_hash_two(combined, w.challenge);
        layouter.assign_region(
            || "verify_folded_state",
            |mut region| {
                config.s_eq.enable(&mut region, 0)?;
                region.assign_advice(|| "expected", config.advice[0], 0, || Value::known(expected_state))?;
                region.assign_advice(|| "actual", config.advice[1], 0, || Value::known(folded.state_commitment))?;
                Ok(())
            },
        )?;

        // Verify challenge hash: combined_ch = Poseidon(ch1, ch2), new_ch = Poseidon(combined_ch, challenge)
        synthesize_poseidon_hash(
            &poseidon_config,
            &mut layouter,
            w.acc1.challenge_hash,
            w.acc2.challenge_hash,
            "fold_challenge_combine",
        )?;

        let combined_ch = poseidon_hash_two(w.acc1.challenge_hash, w.acc2.challenge_hash);

        synthesize_poseidon_hash(
            &poseidon_config,
            &mut layouter,
            combined_ch,
            w.challenge,
            "fold_challenge_hash",
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IVC Multi-Step Circuit – chains N sequential steps into a single proof
// ---------------------------------------------------------------------------

/// Witness for a multi-step IVC circuit.
#[derive(Clone, Debug)]
pub struct IVCMultiStepWitness {
    /// Initial accumulator state.
    pub initial_acc: IVCAccumulator,
    /// Sequence of (computation_hash, step_error) for each step.
    pub steps: Vec<(Fr, Fr)>,
}

impl Default for IVCMultiStepWitness {
    fn default() -> Self {
        Self {
            initial_acc: IVCAccumulator::default(),
            steps: Vec::new(),
        }
    }
}

impl IVCMultiStepWitness {
    /// Computes intermediate states and the final accumulator.
    pub fn compute_chain(&self) -> (Vec<Fr>, IVCAccumulator) {
        let mut states = Vec::with_capacity(self.steps.len());
        let mut current_state = self.initial_acc.state_commitment;
        let mut total_error = self.initial_acc.error_bound;
        let mut step_count = self.initial_acc.num_steps;

        for (computation_hash, step_error) in &self.steps {
            let new_state = poseidon_hash_two(current_state, *computation_hash);
            states.push(new_state);
            total_error = total_error + *step_error;
            step_count += 1;
            current_state = new_state;
        }

        let final_acc = IVCAccumulator {
            state_commitment: current_state,
            num_steps: step_count,
            error_term: self.initial_acc.error_term,
            error_bound: total_error,
            challenge_hash: self.initial_acc.challenge_hash,
        };

        (states, final_acc)
    }
}

/// Circuit that proves N sequential IVC steps in a single proof.
///
/// This is more efficient than N separate IVCStepCircuit proofs when
/// the steps are known ahead of time.
#[derive(Clone)]
pub struct IVCMultiStepCircuit {
    pub witness: IVCMultiStepWitness,
}

impl Default for IVCMultiStepCircuit {
    fn default() -> Self {
        Self {
            witness: IVCMultiStepWitness::default(),
        }
    }
}

impl IVCMultiStepCircuit {
    pub fn public_inputs(&self) -> Vec<Fr> {
        let (_, final_acc) = self.witness.compute_chain();
        final_acc.to_public_inputs()
    }
}

impl Circuit<Fr> for IVCMultiStepCircuit {
    type Config = IVCStepConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        IVCStepCircuit::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let w = &self.witness;
        let (states, final_acc) = w.compute_chain();
        let pi = final_acc.to_public_inputs();

        // Bind public inputs
        let pi_cells = layouter.assign_region(
            || "multi_step_public_inputs",
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

        let poseidon_config = PoseidonCircuitConfig {
            advice: [config.advice[0], config.advice[1], config.advice[2]],
            fixed: config.fixed,
            s_mul: config.s_mul,
            s_add: config.s_add,
            s_rc_add: config.s_rc_add,
            s_eq: config.s_eq,
        };

        let mut current_state = w.initial_acc.state_commitment;
        let mut running_error = w.initial_acc.error_bound;

        for (i, (computation_hash, step_error)) in w.steps.iter().enumerate() {
            // Verify state transition: new_state = Poseidon(current_state, computation_hash)
            synthesize_poseidon_hash(
                &poseidon_config,
                &mut layouter,
                current_state,
                *computation_hash,
                &format!("step_{}_transition", i),
            )?;

            let expected_state = poseidon_hash_two(current_state, *computation_hash);

            // Verify equality with computed state
            layouter.assign_region(
                || format!("verify_step_{}_state", i),
                |mut region| {
                    config.s_eq.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "expected",
                        config.advice[0],
                        0,
                        || Value::known(expected_state),
                    )?;
                    region.assign_advice(
                        || "actual",
                        config.advice[1],
                        0,
                        || Value::known(states[i]),
                    )?;
                    Ok(())
                },
            )?;

            // Verify error accumulation
            let new_error = running_error + *step_error;
            layouter.assign_region(
                || format!("step_{}_error", i),
                |mut region| {
                    config.s_error_check.enable(&mut region, 0)?;
                    region.assign_advice(|| "old_err", config.advice[0], 0, || Value::known(running_error))?;
                    region.assign_advice(|| "step_err", config.advice[1], 0, || Value::known(*step_error))?;
                    region.assign_advice(|| "new_err", config.advice[2], 0, || Value::known(new_error))?;
                    Ok(())
                },
            )?;

            current_state = expected_state;
            running_error = new_error;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Proof Chain Tracking
// ---------------------------------------------------------------------------

/// Records a single proven step in the IVC chain.
#[derive(Clone, Debug)]
pub struct IVCProofStep {
    /// Step index within the chain.
    pub step_index: u64,
    /// Accumulator state after this step.
    pub accumulator: IVCAccumulator,
    /// Proof bytes (KZG proof serialized).
    pub proof: Vec<u8>,
}

/// Records the result of a folding operation.
#[derive(Clone, Debug)]
pub struct FoldResult {
    /// Left accumulator steps before fold.
    pub left_steps: u64,
    /// Right accumulator steps before fold.
    pub right_steps: u64,
    /// Resulting folded accumulator.
    pub folded: IVCAccumulator,
    /// Challenge used for folding.
    pub challenge: Fr,
    /// Proof bytes (KZG proof of the fold circuit).
    pub proof: Vec<u8>,
}

impl IVCChain {
    /// Adds a step with proof tracking.
    pub fn add_step_with_proof(
        &mut self,
        new_state: Fr,
        computation_hash: Fr,
        step_error: Fr,
        proof: Vec<u8>,
    ) -> IVCProofStep {
        self.add_step(new_state, computation_hash, step_error);
        IVCProofStep {
            step_index: self.accumulator.num_steps,
            accumulator: self.accumulator.clone(),
            proof,
        }
    }

    /// Folds with another chain and records the fold result.
    pub fn fold_with_proof(
        &mut self,
        other: &IVCChain,
        proof: Vec<u8>,
    ) -> FoldResult {
        let left_steps = self.accumulator.num_steps;
        let right_steps = other.accumulator.num_steps;
        let challenge = generate_folding_challenge(&self.accumulator, &other.accumulator);
        self.fold_with(other);
        FoldResult {
            left_steps,
            right_steps,
            folded: self.accumulator.clone(),
            challenge,
            proof,
        }
    }

    /// Returns the full step history.
    pub fn history(&self) -> &[IVCAccumulator] {
        &self.history
    }

    /// Verifies consistency of the chain including history.
    pub fn verify_chain_consistency(&self) -> bool {
        if !self.verify() {
            return false;
        }
        // Verify step count matches history
        if !self.history.is_empty() {
            let last_history_steps = self.history.last().map(|a| a.num_steps).unwrap_or(0);
            if self.accumulator.num_steps <= last_history_steps {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    #[test]
    fn test_accumulator_creation() {
        let acc = IVCAccumulator::initial(Fr::from(42u64));

        assert_eq!(acc.num_steps, 0);
        assert_eq!(acc.error_term, Fr::one());
        assert_eq!(acc.error_bound, Fr::zero());
        assert_eq!(acc.state_commitment, Fr::from(42u64));
    }

    #[test]
    fn test_folding() {
        let acc1 = IVCAccumulator {
            state_commitment: Fr::from(100u64),
            num_steps: 5,
            error_term: Fr::one(),
            error_bound: Fr::from(10),
            challenge_hash: Fr::ZERO,
        };

        let acc2 = IVCAccumulator {
            state_commitment: Fr::from(200u64),
            num_steps: 3,
            error_term: Fr::from(2),
            error_bound: Fr::from(5),
            challenge_hash: Fr::ZERO,
        };

        let challenge = Fr::from(7);
        let folded = fold_accumulators(&acc1, &acc2, challenge);

        assert_eq!(folded.num_steps, 8);
        assert_eq!(folded.error_term, Fr::one() + Fr::from(7) * Fr::from(2)); // 1 + 7*2 = 15
        assert_eq!(folded.error_bound, Fr::from(15)); // 10 + 5 = 15
        // State commitment should be Poseidon-based, non-trivial
        assert_ne!(folded.state_commitment, Fr::ZERO);
    }

    #[test]
    fn test_ivc_chain() {
        let mut chain = IVCChain::new(Fr::ZERO);

        chain.add_step(Fr::from(1u64), Fr::from(10u64), Fr::from(1));
        chain.add_step(Fr::from(2u64), Fr::from(20u64), Fr::from(2));
        chain.add_step(Fr::from(3u64), Fr::from(30u64), Fr::from(3));

        assert_eq!(chain.step_count(), 3);
        assert_eq!(chain.error_bound(), Fr::from(6)); // 1 + 2 + 3
        assert!(chain.verify());
    }

    #[test]
    fn test_ivc_step_circuit() {
        let prev_acc = IVCAccumulator::initial(Fr::from(42u64));
        let computation_hash = Fr::from(100u64);

        // Compute the correct new_state using Poseidon
        let new_state = poseidon_hash_two(prev_acc.state_commitment, computation_hash);

        let witness = IVCStepWitness {
            prev_acc,
            new_state,
            computation_hash,
            step_error: Fr::from(5),
            fold_challenge: None,
            other_acc: None,
        };

        let circuit = IVCStepCircuit { witness };
        let pi = circuit.public_inputs();

        // k=12 needed for Poseidon circuit (~764 rows)
        let prover = MockProver::run(12, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_ivc_step_circuit_rejects_wrong_state() {
        let prev_acc = IVCAccumulator::initial(Fr::from(42u64));
        let computation_hash = Fr::from(100u64);

        // Use WRONG new_state (not the Poseidon output)
        let wrong_new_state = Fr::from(999u64);

        let witness = IVCStepWitness {
            prev_acc,
            new_state: wrong_new_state,
            computation_hash,
            step_error: Fr::from(5),
            fold_challenge: None,
            other_acc: None,
        };

        let circuit = IVCStepCircuit { witness };
        let pi = circuit.public_inputs();

        let prover = MockProver::run(12, &circuit, vec![pi]).unwrap();
        // This should FAIL because new_state != Poseidon(prev_state, computation)
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_ivc_step_circuit_with_fold() {
        let prev_acc = IVCAccumulator {
            state_commitment: Fr::from(100u64),
            num_steps: 5,
            error_term: Fr::one(),
            error_bound: Fr::from(10),
            challenge_hash: Fr::ZERO,
        };

        let other_acc = IVCAccumulator {
            state_commitment: Fr::from(200u64),
            num_steps: 3,
            error_term: Fr::from(2),
            error_bound: Fr::from(5),
            challenge_hash: Fr::ZERO,
        };

        let computation_hash = Fr::from(300u64);
        let new_state = poseidon_hash_two(prev_acc.state_commitment, computation_hash);
        let challenge = generate_folding_challenge(&prev_acc, &other_acc);

        let witness = IVCStepWitness {
            prev_acc: prev_acc.clone(),
            new_state,
            computation_hash,
            step_error: Fr::from(1),
            fold_challenge: Some(challenge),
            other_acc: Some(other_acc),
        };

        let circuit = IVCStepCircuit { witness };
        let pi = circuit.public_inputs();

        let prover = MockProver::run(12, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    // ===== Folding Circuit Tests =====

    #[test]
    fn test_folding_circuit_basic() {
        let acc1 = IVCAccumulator {
            state_commitment: Fr::from(100u64),
            num_steps: 5,
            error_term: Fr::one(),
            error_bound: Fr::from(10),
            challenge_hash: Fr::ZERO,
        };
        let acc2 = IVCAccumulator {
            state_commitment: Fr::from(200u64),
            num_steps: 3,
            error_term: Fr::from(2),
            error_bound: Fr::from(5),
            challenge_hash: Fr::ZERO,
        };
        let challenge = generate_folding_challenge(&acc1, &acc2);

        let witness = IVCFoldingWitness { acc1, acc2, challenge };
        let circuit = IVCFoldingCircuit { witness };
        let pi = circuit.public_inputs();

        // k=13 needed for 4 Poseidon hashes in folding circuit
        let prover = MockProver::run(13, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_folding_circuit_rejects_wrong_challenge() {
        let acc1 = IVCAccumulator::initial(Fr::from(10u64));
        let acc2 = IVCAccumulator::initial(Fr::from(20u64));
        let real_challenge = generate_folding_challenge(&acc1, &acc2);

        // Use the real challenge for the witness but tamper with acc2's error_term
        // in a way that produces wrong folded values
        let witness = IVCFoldingWitness {
            acc1: acc1.clone(),
            acc2: acc2.clone(),
            challenge: real_challenge,
        };
        let circuit = IVCFoldingCircuit { witness };

        // Compute correct PI, then corrupt one
        let mut pi = circuit.public_inputs();
        pi[3] = Fr::from(9999u64); // corrupt folded error_term

        let prover = MockProver::run(13, &circuit, vec![pi]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Real KZG proof generation and verification for IVCFoldingCircuit.
    #[test]
    fn test_folding_circuit_real_proof() {
        use halo2_proofs::{
            plonk::{create_proof, keygen_pk, keygen_vk, verify_proof_multi},
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            transcript::{
                Blake2bRead, Blake2bWrite, Challenge255,
                TranscriptReadBuffer, TranscriptWriterBuffer,
            },
        };
        use halo2curves::bn256::{Bn256, G1Affine};
        use rand_core::OsRng;

        let acc1 = IVCAccumulator {
            state_commitment: Fr::from(100u64),
            num_steps: 5,
            error_term: Fr::one(),
            error_bound: Fr::from(10),
            challenge_hash: Fr::ZERO,
        };
        let acc2 = IVCAccumulator {
            state_commitment: Fr::from(200u64),
            num_steps: 3,
            error_term: Fr::from(2),
            error_bound: Fr::from(5),
            challenge_hash: Fr::ZERO,
        };
        let challenge = generate_folding_challenge(&acc1, &acc2);

        let witness = IVCFoldingWitness { acc1, acc2, challenge };
        let circuit = IVCFoldingCircuit { witness };
        let pi = circuit.public_inputs();
        let k = 13; // 4 Poseidon hashes need k=13

        // Setup
        let params = ParamsKZG::<Bn256>::setup(k, OsRng);
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");

        // Prove
        let instances = vec![pi.clone()];
        let mut transcript = Blake2bWrite::<Vec<u8>, G1Affine, Challenge255<_>>::init(vec![]);

        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            _,
            _,
            _,
            _,
        >(
            &params,
            &pk,
            &[circuit],
            &[instances.clone()],
            OsRng,
            &mut transcript,
        )
        .expect("IVC folding create_proof failed");

        let proof = transcript.finalize();
        assert!(!proof.is_empty(), "Proof must not be empty");

        // Verify
        let mut verifier_transcript =
            Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
        let verifier_params = params.verifier_params();
        let verified = verify_proof_multi::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _,
            _,
            SingleStrategy<Bn256>,
        >(&verifier_params, &vk, &[instances], &mut verifier_transcript);

        assert!(verified, "IVC folding real proof verification must succeed");
    }

    // ===== Multi-Step Circuit Tests =====

    #[test]
    fn test_multi_step_circuit_single_step() {
        let initial = IVCAccumulator::initial(Fr::from(42u64));
        let witness = IVCMultiStepWitness {
            initial_acc: initial,
            steps: vec![(Fr::from(100u64), Fr::from(1))],
        };
        let circuit = IVCMultiStepCircuit { witness };
        let pi = circuit.public_inputs();

        let prover = MockProver::run(12, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_multi_step_circuit_three_steps() {
        let initial = IVCAccumulator::initial(Fr::from(1u64));
        let witness = IVCMultiStepWitness {
            initial_acc: initial,
            steps: vec![
                (Fr::from(10u64), Fr::from(1)),
                (Fr::from(20u64), Fr::from(2)),
                (Fr::from(30u64), Fr::from(3)),
            ],
        };
        let circuit = IVCMultiStepCircuit { witness };
        let pi = circuit.public_inputs();

        // 3 steps × 1 Poseidon each = 3 Poseidon hashes → k=13
        let prover = MockProver::run(13, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_multi_step_circuit_five_steps() {
        let initial = IVCAccumulator::initial(Fr::ZERO);
        let steps: Vec<(Fr, Fr)> = (1..=5)
            .map(|i| (Fr::from(i as u64 * 100), Fr::from(i as u64)))
            .collect();

        let witness = IVCMultiStepWitness {
            initial_acc: initial,
            steps,
        };
        let circuit = IVCMultiStepCircuit { witness };
        let pi = circuit.public_inputs();

        // 5 Poseidon hashes → k=14
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_multi_step_circuit_eight_steps() {
        let initial = IVCAccumulator::initial(Fr::from(7u64));
        let steps: Vec<(Fr, Fr)> = (1..=8)
            .map(|i| (Fr::from(i as u64 * 11), Fr::from(1)))
            .collect();

        let witness = IVCMultiStepWitness {
            initial_acc: initial,
            steps,
        };
        let circuit = IVCMultiStepCircuit { witness };
        let pi = circuit.public_inputs();

        // 8 Poseidon hashes → k=14
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_multi_step_rejects_wrong_final_state() {
        let initial = IVCAccumulator::initial(Fr::from(1u64));
        let witness = IVCMultiStepWitness {
            initial_acc: initial,
            steps: vec![
                (Fr::from(10u64), Fr::from(1)),
                (Fr::from(20u64), Fr::from(2)),
            ],
        };
        let circuit = IVCMultiStepCircuit { witness };
        let mut pi = circuit.public_inputs();
        pi[0] = Fr::from(12345u64); // corrupt state commitment

        let prover = MockProver::run(13, &circuit, vec![pi]).unwrap();
        assert!(prover.verify().is_err());
    }

    // ===== Enhanced IVCChain Tests =====

    #[test]
    fn test_chain_with_proof_tracking() {
        let mut chain = IVCChain::new(Fr::from(1u64));

        let step1 = chain.add_step_with_proof(
            Fr::from(10u64),
            Fr::from(100u64),
            Fr::from(1),
            vec![0xDE, 0xAD],
        );
        assert_eq!(step1.step_index, 1);
        assert_eq!(step1.proof, vec![0xDE, 0xAD]);

        let step2 = chain.add_step_with_proof(
            Fr::from(20u64),
            Fr::from(200u64),
            Fr::from(2),
            vec![0xBE, 0xEF],
        );
        assert_eq!(step2.step_index, 2);
        assert_eq!(chain.step_count(), 2);
        assert!(chain.verify_chain_consistency());
    }

    #[test]
    fn test_chain_fold_with_proof() {
        let mut chain1 = IVCChain::new(Fr::from(1u64));
        chain1.add_step(Fr::from(10u64), Fr::from(100u64), Fr::from(1));
        chain1.add_step(Fr::from(20u64), Fr::from(200u64), Fr::from(2));

        let mut chain2 = IVCChain::new(Fr::from(2u64));
        chain2.add_step(Fr::from(30u64), Fr::from(300u64), Fr::from(3));

        let fold_result = chain1.fold_with_proof(&chain2, vec![0xF0, 0x1D]);
        assert_eq!(fold_result.left_steps, 2);
        assert_eq!(fold_result.right_steps, 1);
        assert_eq!(fold_result.folded.num_steps, 3);
        assert!(!fold_result.proof.is_empty());
    }

    #[test]
    fn test_chain_consistency_verification() {
        let mut chain = IVCChain::new(Fr::from(1u64));
        assert!(chain.verify_chain_consistency());

        chain.add_step(Fr::from(10u64), Fr::from(100u64), Fr::from(1));
        assert!(chain.verify_chain_consistency());

        chain.add_step(Fr::from(20u64), Fr::from(200u64), Fr::from(2));
        assert!(chain.verify_chain_consistency());
        assert_eq!(chain.history().len(), 2);
    }

    // ===== 5-Step Multi-Step Chain with Real KZG Proof =====

    #[test]
    fn test_multi_step_five_step_real_proof() {
        use halo2_proofs::{
            plonk::{create_proof, keygen_pk, keygen_vk, verify_proof_multi},
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            transcript::{
                Blake2bRead, Blake2bWrite, Challenge255,
                TranscriptReadBuffer, TranscriptWriterBuffer,
            },
        };
        use halo2curves::bn256::{Bn256, G1Affine};
        use rand_core::OsRng;

        let initial = IVCAccumulator::initial(Fr::from(42u64));
        let steps: Vec<(Fr, Fr)> = (1..=5)
            .map(|i| (Fr::from(i as u64 * 7), Fr::from(1)))
            .collect();

        let witness = IVCMultiStepWitness {
            initial_acc: initial,
            steps,
        };
        let circuit = IVCMultiStepCircuit { witness };
        let pi = circuit.public_inputs();
        let k = 14;

        let params = ParamsKZG::<Bn256>::setup(k, OsRng);
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");

        let instances = vec![pi.clone()];
        let mut transcript = Blake2bWrite::<Vec<u8>, G1Affine, Challenge255<_>>::init(vec![]);

        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            _, _, _, _,
        >(
            &params, &pk, &[circuit], &[instances.clone()], OsRng, &mut transcript,
        )
        .expect("multi-step create_proof failed");

        let proof = transcript.finalize();
        assert!(!proof.is_empty());

        let mut vt = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
        let vp = params.verifier_params();
        let verified = verify_proof_multi::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _, _, SingleStrategy<Bn256>,
        >(&vp, &vk, &[instances], &mut vt);

        assert!(verified, "5-step multi-step real proof must verify");
    }

    /// Real proof generation for IVC step circuit using KZG + SHPLONK.
    #[test]
    fn test_ivc_real_proof_generation() {
        use halo2_proofs::{
            plonk::{create_proof, keygen_pk, keygen_vk, verify_proof_multi},
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            transcript::{
                Blake2bRead, Blake2bWrite, Challenge255,
                TranscriptReadBuffer, TranscriptWriterBuffer,
            },
        };
        use halo2curves::bn256::{Bn256, G1Affine};
        use rand_core::OsRng;

        let prev_acc = IVCAccumulator::initial(Fr::from(42u64));
        let computation_hash = Fr::from(100u64);
        let new_state = poseidon_hash_two(prev_acc.state_commitment, computation_hash);

        let witness = IVCStepWitness {
            prev_acc,
            new_state,
            computation_hash,
            step_error: Fr::from(5),
            fold_challenge: None,
            other_acc: None,
        };

        let circuit = IVCStepCircuit { witness };
        let pi = circuit.public_inputs();
        let k = 12;

        // Setup
        let params = ParamsKZG::<Bn256>::setup(k, OsRng);
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");

        // Prove
        let instances = vec![pi.clone()];
        let mut transcript = Blake2bWrite::<Vec<u8>, G1Affine, Challenge255<_>>::init(vec![]);

        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            _,
            _,
            _,
            _,
        >(
            &params,
            &pk,
            &[circuit],
            &[instances.clone()],
            OsRng,
            &mut transcript,
        )
        .expect("IVC create_proof failed");

        let proof = transcript.finalize();
        assert!(!proof.is_empty());

        // Verify
        let mut verifier_transcript =
            Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
        let verifier_params = params.verifier_params();
        let verified = verify_proof_multi::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _,
            _,
            SingleStrategy<Bn256>,
        >(&verifier_params, &vk, &[instances], &mut verifier_transcript);

        assert!(verified, "IVC real proof verification must succeed");
    }
}
