//! End-to-end integration test infrastructure.
//!
//! Provides shared utilities for full-system integration tests that exercise
//! the complete HELIX pipeline: training → proof generation → aggregation →
//! on-chain verification.
//!
//! # Key Components
//!
//! - [`shared_prover`]: Lazy-initialized prover (expensive keygen, reused across tests)
//! - [`simulate_worker`]: Worker simulation (trains + generates proofs)
//! - [`verify_commitment_chain`]: Validates commitment chaining across proof steps
//! - [`corrupt_proof`]: Creates corrupted proofs for Byzantine testing
//! - [`compute_initial_commitment`]: Computes on-chain commitment from weights

use std::sync::OnceLock;

use halo2curves::bn256::Fr;

use helix_circuits::ml::training_step_v2::compute_state_hash_v2;
use helix_prover::{
    MLTrainingProverV2, RetryConfig, TrainingProofResultV2, TrainingWeights, V2ProverConfig,
};

#[cfg(feature = "on-chain")]
use super::anvil::{compute_hash_pair, fr_to_u256};

#[cfg(feature = "on-chain")]
use ethers::types::U256;

// ============================================================================
// Constants
// ============================================================================

/// Model dimensions used across all E2E tests: 2 inputs, 2 hidden, 1 output.
pub const D_IN: usize = 2;
pub const D_HID: usize = 2;
pub const D_OUT: usize = 1;

/// Number of public inputs per proof.
pub const NUM_PUBLIC_INPUTS: usize = 8;

// ============================================================================
// Shared Prover (Lazy Singleton)
// ============================================================================

static PROVER: OnceLock<MLTrainingProverV2> = OnceLock::new();

/// Returns a shared prover instance. Keygen is expensive (~5s for k=14),
/// so we initialize once and reuse across all tests in the same binary.
pub fn shared_prover() -> &'static MLTrainingProverV2 {
    PROVER.get_or_init(|| {
        let config = V2ProverConfig {
            k: 14,
            relu_range: 256,
            exp_range: 256,
            exp_scale: 1000,
            self_verify: true,
            use_witness_cache: false,
            use_freivalds: false,
            enable_tracing: false,
            retry: RetryConfig::none(),
            ..V2ProverConfig::default()
        };
        MLTrainingProverV2::with_config(D_IN, D_HID, D_OUT, config)
    })
}

// ============================================================================
// Model Setup
// ============================================================================

/// Creates circuit-safe initial weights for a 2×2×1 model.
///
/// Values are small (Fr(1)-Fr(3)) so that quantized activations stay
/// within the ±256 ReLU lookup range used by the circuit.
pub fn initial_weights() -> TrainingWeights {
    TrainingWeights::new(
        D_IN,
        D_HID,
        D_OUT,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)], // w1: [2×2]
        vec![Fr::zero(), Fr::zero()],                                           // b1: [2]
        vec![Fr::from(1u64), Fr::from(1u64)],                                   // w2: [1×2]
        vec![Fr::zero()],                                                        // b2: [1]
    )
}

/// Learning rate for test training (Fr field element).
pub fn lr() -> Fr {
    Fr::from(1u64)
}

/// Base error per operation.
pub fn base_error() -> Fr {
    Fr::from(1u64)
}

/// Default model ID (first registered model = 0) as 32-byte LE array.
pub fn default_model_id() -> [u8; 32] {
    [0u8; 32]
}

/// Converts a U256 model ID to a 32-byte LE array for the circuit witness.
#[cfg(feature = "on-chain")]
pub fn model_id_to_bytes(model_id: U256) -> [u8; 32] {
    let mut be = [0u8; 32];
    model_id.to_big_endian(&mut be);
    let mut le = [0u8; 32];
    for (i, b) in be.iter().enumerate() {
        le[31 - i] = *b;
    }
    le
}

/// Default error budget matching contract's DEFAULT_MAX_ERROR_BOUND = 1e18.
pub fn default_error_budget() -> Fr {
    Fr::from(1_000_000_000_000_000_000u64)
}

/// Circuit-safe training dataset: 4 samples of (input, target).
///
/// All values are small Fr elements to stay within ReLU lookup range.
pub fn training_data() -> Vec<(Vec<Fr>, Vec<Fr>)> {
    vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(3u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(4u64)]),
        (vec![Fr::from(2u64), Fr::from(2u64)], vec![Fr::from(6u64)]),
    ]
}

// ============================================================================
// Proof Generation Helpers
// ============================================================================

/// Generates a proof for one training step with default model_id and error_budget.
pub fn prove_step(
    weights: &TrainingWeights,
    x: &[Fr],
    target: &[Fr],
    step_number: u64,
) -> (TrainingProofResultV2, TrainingWeights) {
    prove_step_with_params(
        weights,
        x,
        target,
        step_number,
        default_model_id(),
        default_error_budget(),
    )
}

/// Generates a proof with explicit model_id and error_budget for PI[7] matching.
pub fn prove_step_with_params(
    weights: &TrainingWeights,
    x: &[Fr],
    target: &[Fr],
    step_number: u64,
    model_id: [u8; 32],
    error_budget: Fr,
) -> (TrainingProofResultV2, TrainingWeights) {
    let prover = shared_prover();

    let witness = MLTrainingProverV2::build_witness_with_params(
        weights.d_in,
        weights.d_hid,
        weights.d_out,
        x,
        target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        lr(),
        step_number,
        base_error(),
        model_id,
        error_budget,
    );

    let result = prover.prove(&witness).expect("Proof generation failed");

    let new_weights = TrainingWeights::new(
        weights.d_in,
        weights.d_hid,
        weights.d_out,
        witness.w1_new.clone(),
        witness.b1_new.clone(),
        witness.w2_new.clone(),
        witness.b2_new.clone(),
    );

    (result, new_weights)
}

// ============================================================================
// Commitment Helpers
// ============================================================================

/// Computes the state hash (lo, hi) for a set of weights.
pub fn compute_state_hash(weights: &TrainingWeights) -> (Fr, Fr) {
    compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2)
}

/// Computes the initial on-chain commitment (keccak256 hash pair) for model registration.
#[cfg(feature = "on-chain")]
pub fn compute_initial_commitment(weights: &TrainingWeights) -> U256 {
    let hash = compute_state_hash(weights);
    let lo = fr_to_u256(&hash.0);
    let hi = fr_to_u256(&hash.1);
    compute_hash_pair(lo, hi)
}

// ============================================================================
// Worker Simulation
// ============================================================================

/// Result from a simulated worker's training run.
#[derive(Debug)]
pub struct WorkerResult {
    /// Worker index.
    pub worker_id: usize,
    /// Per-step proof results and updated weights.
    pub steps: Vec<(TrainingProofResultV2, TrainingWeights)>,
    /// Final weights after all steps.
    pub final_weights: TrainingWeights,
}

/// Simulates a worker training for `num_steps` on the given dataset.
///
/// Each worker uses the same initial weights but may process different
/// data orderings. Proofs are generated for every step.
pub fn simulate_worker(
    worker_id: usize,
    init_weights: &TrainingWeights,
    dataset: &[(Vec<Fr>, Vec<Fr>)],
    num_steps: usize,
) -> WorkerResult {
    let mut weights = init_weights.clone();
    let mut steps = Vec::with_capacity(num_steps);

    for step in 0..num_steps {
        let (x, target) = &dataset[step % dataset.len()];
        let step_number = (step + 1) as u64;
        let (result, new_weights) = prove_step(&weights, x, target, step_number);
        assert!(result.verified, "Worker {} step {} proof not self-verified", worker_id, step_number);
        steps.push((result, new_weights.clone()));
        weights = new_weights;
    }

    WorkerResult {
        worker_id,
        steps,
        final_weights: weights,
    }
}

/// Simulates a worker with explicit model_id and error_budget
/// (needed when model_id is known from on-chain registration).
#[cfg(feature = "on-chain")]
pub fn simulate_worker_with_params(
    worker_id: usize,
    init_weights: &TrainingWeights,
    dataset: &[(Vec<Fr>, Vec<Fr>)],
    num_steps: usize,
    model_id: [u8; 32],
    error_budget: Fr,
) -> WorkerResult {
    let mut weights = init_weights.clone();
    let mut steps = Vec::with_capacity(num_steps);

    for step in 0..num_steps {
        let (x, target) = &dataset[step % dataset.len()];
        let step_number = (step + 1) as u64;
        let (result, new_weights) =
            prove_step_with_params(&weights, x, target, step_number, model_id, error_budget);
        assert!(
            result.verified,
            "Worker {} step {} proof not self-verified",
            worker_id, step_number
        );
        steps.push((result, new_weights.clone()));
        weights = new_weights;
    }

    WorkerResult {
        worker_id,
        steps,
        final_weights: weights,
    }
}

// ============================================================================
// Aggregation & Verification Helpers
// ============================================================================

/// Verifies that a sequence of proofs has valid commitment chaining.
///
/// Each proof's `new_state_hash` must match the next proof's `old_state_hash`.
pub fn verify_commitment_chain(results: &[&TrainingProofResultV2]) -> bool {
    for window in results.windows(2) {
        let prev = window[0];
        let next = window[1];
        if prev.new_state_hash != next.old_state_hash {
            return false;
        }
    }
    true
}

/// Creates a corrupted copy of a proof by flipping bytes in the proof data.
///
/// This simulates a Byzantine worker submitting invalid proofs.
pub fn corrupt_proof(result: &TrainingProofResultV2) -> TrainingProofResultV2 {
    let mut corrupted = result.clone();
    if corrupted.proof.len() > 20 {
        // Flip bytes at multiple positions to ensure corruption
        corrupted.proof[10] ^= 0xFF;
        corrupted.proof[11] ^= 0xFF;
        corrupted.proof[20] ^= 0xFF;
    }
    corrupted
}

/// Verifies a proof using the shared prover's verification key.
pub fn verify_proof(result: &TrainingProofResultV2) -> bool {
    let prover = shared_prover();
    prover.verify_result(result)
}

// ============================================================================
// Async Worker Helpers (for tokio-based tests)
// ============================================================================

/// Spawns a worker simulation as a tokio blocking task.
///
/// Returns a JoinHandle that resolves to the WorkerResult.
pub fn spawn_worker(
    worker_id: usize,
    init_weights: TrainingWeights,
    dataset: Vec<(Vec<Fr>, Vec<Fr>)>,
    num_steps: usize,
) -> tokio::task::JoinHandle<WorkerResult> {
    tokio::task::spawn_blocking(move || {
        simulate_worker(worker_id, &init_weights, &dataset, num_steps)
    })
}

/// Spawns a worker with explicit params as a tokio blocking task.
#[cfg(feature = "on-chain")]
pub fn spawn_worker_with_params(
    worker_id: usize,
    init_weights: TrainingWeights,
    dataset: Vec<(Vec<Fr>, Vec<Fr>)>,
    num_steps: usize,
    model_id: [u8; 32],
    error_budget: Fr,
) -> tokio::task::JoinHandle<WorkerResult> {
    tokio::task::spawn_blocking(move || {
        simulate_worker_with_params(
            worker_id,
            &init_weights,
            &dataset,
            num_steps,
            model_id,
            error_budget,
        )
    })
}
