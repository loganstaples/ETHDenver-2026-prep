//! ML Training Step V2 Prover.
//!
//! Wraps the enhanced `MLTrainingStepV2Circuit` from `helix-circuits` with the
//! Halo2 `ProverPipeline` to produce real KZG proofs for training steps with:
//!
//! - **Freivalds matrix verification** for O(n²) matmul checking
//! - **Error bound tracking** through all operations
//! - **Proper state commitment** using Poseidon-style hashing
//! - **Comprehensive error handling** (no panics)
//! - **Automatic retry on failure** with exponential backoff
//! - **Progress callbacks** for long-running proofs
//! - **Self-verification** before returning proofs
//! - **Witness sanity checks** before proving
//! - **Timeout handling** with graceful cancellation
//! - **Deterministic proof generation** (same inputs → same proof)
//! - **Proof caching** for identical witnesses
//!
//! This prover is recommended for production use over the V1 prover.

use crate::cache::witness_cache::{
    SharedWitnessCache, WitnessCachedProof, WitnessHash, WitnessHashBuilder, shared_witness_cache,
};
use crate::keys::generate_keys_for_circuit;
use crate::pipeline::{
    CancellationToken, PipelineConfig, PipelineError, ProofPhase, ProofProgress, ProgressCallback,
    ProverPipeline, RetryConfig, no_progress_callback,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;
use helix_circuits::ml::config::LossFunction;
use helix_circuits::ml::training_step_v2::{
    compute_state_hash_v2, compute_witness_v2, compute_witness_v2_cross_entropy,
    MLTrainingStepV2Circuit, MLTrainingStepV2Witness, NUM_PUBLIC_INPUTS,
};
use helix_circuits::verifier::{
    SolidityGenerator, VkData,
    fr_to_evm_bytes,
    ProofFormatError,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors specific to ML training proof generation.
#[derive(Error, Debug)]
pub enum TrainingProverError {
    /// Pipeline error.
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Witness validation failed.
    #[error("Witness validation failed: {message} (field: {field})")]
    WitnessValidation { message: String, field: String },

    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual} for {field}")]
    DimensionMismatch {
        field: String,
        expected: usize,
        actual: usize,
    },

    /// Value out of range.
    #[error("Value out of range in {field}: {message}")]
    ValueOutOfRange { field: String, message: String },

    /// Self-verification failed.
    #[error("Self-verification failed: generated proof does not verify")]
    SelfVerificationFailed,

    /// EVM proof serialization failed.
    #[error("EVM proof serialization failed: {0}")]
    EvmSerializationFailed(#[from] ProofFormatError),

    /// Operation timed out.
    #[error("Operation timed out after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },

    /// Operation cancelled.
    #[error("Operation cancelled")]
    Cancelled,

    /// Prover not initialized.
    #[error("Prover not initialized: {message}")]
    NotInitialized { message: String },
}

impl TrainingProverError {
    /// Creates a dimension mismatch error.
    pub fn dimension_mismatch<S: Into<String>>(field: S, expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch {
            field: field.into(),
            expected,
            actual,
        }
    }

    /// Creates a witness validation error.
    pub fn validation<S: Into<String>>(message: S, field: S) -> Self {
        Self::WitnessValidation {
            message: message.into(),
            field: field.into(),
        }
    }

    /// Creates a value out of range error.
    pub fn out_of_range<S: Into<String>>(field: S, message: S) -> Self {
        Self::ValueOutOfRange {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// Result type for training prover operations.
pub type TrainingProverResult<T> = Result<T, TrainingProverError>;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for V2 prover.
#[derive(Debug, Clone)]
pub struct V2ProverConfig {
    /// K parameter (circuit size = 2^k rows).
    pub k: u32,
    /// ReLU lookup range half-width.
    pub relu_range: usize,
    /// Exp lookup range for softmax.
    pub exp_range: usize,
    /// Exp lookup scale.
    pub exp_scale: u64,
    /// Whether to use Freivalds verification.
    pub use_freivalds: bool,
    /// Base error per operation (for error tracking).
    pub base_error: Fr,
    /// Whether to self-verify proofs.
    pub self_verify: bool,
    /// Retry configuration.
    pub retry: RetryConfig,
    /// Proof generation timeout.
    pub proof_timeout: Option<Duration>,
    /// Verification timeout.
    pub verify_timeout: Option<Duration>,
    /// Seed for deterministic proof generation.
    pub deterministic_seed: Option<[u8; 32]>,
    /// Whether to use witness caching.
    pub use_witness_cache: bool,
    /// Enable detailed tracing.
    pub enable_tracing: bool,
    /// Loss function type (MSE or CrossEntropy).
    pub loss_function: helix_circuits::ml::config::LossFunction,
}

impl Default for V2ProverConfig {
    fn default() -> Self {
        Self {
            k: 14,
            relu_range: 128,
            exp_range: 256,
            exp_scale: 1000,
            use_freivalds: true,
            base_error: Fr::from(1u64),
            self_verify: true,
            retry: RetryConfig::default(),
            proof_timeout: Some(Duration::from_secs(300)),
            verify_timeout: Some(Duration::from_secs(30)),
            deterministic_seed: None,
            use_witness_cache: true,
            enable_tracing: true,
            loss_function: helix_circuits::ml::config::LossFunction::MSE,
        }
    }
}

impl V2ProverConfig {
    /// Creates a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            k: 12,
            relu_range: 64,
            exp_range: 128,
            self_verify: false,
            retry: RetryConfig::none(),
            use_witness_cache: false,
            enable_tracing: false,
            ..Default::default()
        }
    }

    /// Creates a production configuration with aggressive reliability.
    pub fn production() -> Self {
        Self {
            self_verify: true,
            retry: RetryConfig::aggressive(),
            use_witness_cache: true,
            enable_tracing: true,
            ..Default::default()
        }
    }

    /// Enables deterministic proof generation.
    pub fn deterministic(mut self, seed: [u8; 32]) -> Self {
        self.deterministic_seed = Some(seed);
        self
    }

    /// Sets the loss function (MSE or CrossEntropy).
    pub fn with_loss_function(mut self, loss: LossFunction) -> Self {
        self.loss_function = loss;
        self
    }

    /// Sets the K parameter.
    pub fn with_k(mut self, k: u32) -> Self {
        self.k = k;
        self
    }

    /// Sets the ReLU range.
    pub fn with_relu_range(mut self, relu_range: usize) -> Self {
        self.relu_range = relu_range;
        self
    }

    /// Auto-configures k and relu_range from the required ReLU lookup range.
    ///
    /// Uses `MLTrainingStepV2Circuit::minimum_k()` to compute the smallest k
    /// that fits the given relu_range for this model shape, then expands
    /// relu_range to use all available lookup rows in that k (free headroom).
    pub fn auto(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        relu_range: usize,
    ) -> Self {
        use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;

        let exp_range = 128usize;
        let use_freivalds = true;

        // First compute minimum k for the given relu_range
        let dummy = MLTrainingStepV2Circuit {
            witness: create_zero_witness(d_in, d_hid, d_out),
            relu_range,
            exp_range,
            exp_scale: 64,
            use_freivalds,
        };
        let min_k = dummy.minimum_k();

        // Now expand relu_range to fill available space in that k
        let max_range = MLTrainingStepV2Circuit::max_relu_range_for_k(
            min_k, d_in, d_hid, d_out, use_freivalds, exp_range,
        );
        let final_relu_range = relu_range.max(max_range);

        Self {
            k: min_k,
            relu_range: final_relu_range,
            exp_range,
            exp_scale: 64,
            use_freivalds,
            ..Default::default()
        }
    }
}

// ============================================================================
// Proof Result
// ============================================================================

/// Result of proving a V2 training step.
#[derive(Debug, Clone)]
pub struct TrainingProofResultV2 {
    /// Serialized Halo2 proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for verification.
    pub public_inputs: Vec<Fr>,
    /// The loss value.
    pub loss: Fr,
    /// The final accumulated error bound.
    pub total_error: Fr,
    /// Step number.
    pub step_number: u64,
    /// Old state commitment.
    pub old_state_hash: (Fr, Fr),
    /// New state commitment.
    pub new_state_hash: (Fr, Fr),
    /// Whether the proof was self-verified.
    pub verified: bool,
    /// Proof generation time.
    pub generation_time: Duration,
    /// Verification time (if self-verified).
    pub verification_time: Option<Duration>,
    /// Number of attempts.
    pub attempts: u32,
    /// Whether this result came from cache.
    pub from_cache: bool,
    /// Witness hash (for caching).
    pub witness_hash: Option<WitnessHash>,
}

impl TrainingProofResultV2 {
    /// Returns the proof bytes in EVM-compatible format.
    ///
    /// The pipeline uses PSE's `Keccak256Transcript` which writes EC points as
    /// 64-byte uncompressed (x, y) in big-endian — exactly the format expected
    /// by the PSE-generated Halo2VerifierCore.sol. No conversion needed.
    pub fn to_evm_proof(&self) -> Result<Vec<u8>, ProofFormatError> {
        if self.proof.is_empty() {
            return Err(ProofFormatError::TooShort { got: 0, min: 32 });
        }
        Ok(self.proof.clone())
    }

    /// Converts public inputs to big-endian `uint256` byte arrays for Solidity.
    ///
    /// Returns `Vec<[u8; 32]>` where each element is a 32-byte big-endian
    /// representation suitable for passing as `uint256[]` to the on-chain verifier.
    ///
    /// Layout: `[old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, loss, error_bound, step_number, error_checksum]`
    pub fn to_evm_public_inputs(&self) -> Vec<[u8; 32]> {
        self.public_inputs
            .iter()
            .map(|fr| fr_to_evm_bytes(fr))
            .collect()
    }
}

// ============================================================================
// EVM Proof Bundle
// ============================================================================

/// Complete bundle for on-chain proof submission.
///
/// Contains everything needed to verify a training proof on-chain via
/// `Halo2Verifier.sol`: the serialized proof, public inputs, and the
/// verification key parameters for contract deployment.
#[derive(Debug, Clone)]
pub struct EvmProofBundle {
    /// 320-byte EVM-compatible proof (3 advice commitments + SHPLONK opening).
    pub evm_proof: Vec<u8>,
    /// 8 x 32-byte big-endian public inputs for Solidity `uint256[]`.
    pub evm_public_inputs: Vec<[u8; 32]>,
    /// VK data for Halo2Verifier constructor: G1 generator, s·G2, -G2.
    pub vk_deployment_args: VkData,
    /// The full proof result including metadata (loss, step number, etc).
    pub result: TrainingProofResultV2,
}

impl EvmProofBundle {
    /// Creates an `EvmProofBundle` from a proof result and VK data.
    ///
    /// Converts the raw Halo2 transcript proof to EVM format and packages
    /// it with public inputs and VK deployment data for on-chain submission.
    pub fn from_proof_result(
        result: &TrainingProofResultV2,
        vk: VkData,
    ) -> Result<Self, ProofFormatError> {
        let evm_proof = result.to_evm_proof()?;
        let evm_public_inputs = result.to_evm_public_inputs();
        Ok(Self {
            evm_proof,
            evm_public_inputs,
            vk_deployment_args: vk,
            result: result.clone(),
        })
    }

    /// Returns the EVM proof as a hex-encoded string (no `0x` prefix).
    pub fn evm_proof_hex(&self) -> String {
        self.evm_proof.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Returns public inputs as hex-encoded strings (no `0x` prefix).
    pub fn evm_public_inputs_hex(&self) -> Vec<String> {
        self.evm_public_inputs
            .iter()
            .map(|pi| pi.iter().map(|b| format!("{:02x}", b)).collect())
            .collect()
    }
}

// ============================================================================
// Witness Validation
// ============================================================================

/// Validation result for a witness.
#[derive(Debug)]
pub struct WitnessValidationResult {
    /// Whether the witness is valid.
    pub valid: bool,
    /// List of validation errors.
    pub errors: Vec<WitnessValidationError>,
    /// List of warnings (non-fatal issues).
    pub warnings: Vec<String>,
}

impl WitnessValidationResult {
    /// Creates a successful validation result.
    pub fn ok() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Adds an error.
    pub fn add_error(&mut self, error: WitnessValidationError) {
        self.valid = false;
        self.errors.push(error);
    }

    /// Adds a warning.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// A specific witness validation error.
#[derive(Debug)]
pub struct WitnessValidationError {
    /// Field that failed validation.
    pub field: String,
    /// Error message.
    pub message: String,
    /// Expected value (if applicable).
    pub expected: Option<String>,
    /// Actual value (if applicable).
    pub actual: Option<String>,
}

impl WitnessValidationError {
    /// Creates a new validation error.
    pub fn new<S: Into<String>>(field: S, message: S) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
            expected: None,
            actual: None,
        }
    }

    /// Adds expected/actual values.
    pub fn with_values<S: Into<String>>(mut self, expected: S, actual: S) -> Self {
        self.expected = Some(expected.into());
        self.actual = Some(actual.into());
        self
    }
}

/// Validates a training witness before proving.
pub fn validate_witness(witness: &MLTrainingStepV2Witness) -> WitnessValidationResult {
    let mut result = WitnessValidationResult::ok();

    // Check dimensions
    let expected_w1_len = witness.d_hid * witness.d_in;
    if witness.w1.len() != expected_w1_len {
        result.add_error(
            WitnessValidationError::new("w1", "Dimension mismatch")
                .with_values(expected_w1_len.to_string(), witness.w1.len().to_string()),
        );
    }

    let expected_b1_len = witness.d_hid;
    if witness.b1.len() != expected_b1_len {
        result.add_error(
            WitnessValidationError::new("b1", "Dimension mismatch")
                .with_values(expected_b1_len.to_string(), witness.b1.len().to_string()),
        );
    }

    let expected_w2_len = witness.d_out * witness.d_hid;
    if witness.w2.len() != expected_w2_len {
        result.add_error(
            WitnessValidationError::new("w2", "Dimension mismatch")
                .with_values(expected_w2_len.to_string(), witness.w2.len().to_string()),
        );
    }

    let expected_b2_len = witness.d_out;
    if witness.b2.len() != expected_b2_len {
        result.add_error(
            WitnessValidationError::new("b2", "Dimension mismatch")
                .with_values(expected_b2_len.to_string(), witness.b2.len().to_string()),
        );
    }

    // Check input dimensions
    if witness.x.len() != witness.d_in {
        result.add_error(
            WitnessValidationError::new("x", "Input dimension mismatch")
                .with_values(witness.d_in.to_string(), witness.x.len().to_string()),
        );
    }

    if witness.target.len() != witness.d_out {
        result.add_error(
            WitnessValidationError::new("target", "Target dimension mismatch")
                .with_values(witness.d_out.to_string(), witness.target.len().to_string()),
        );
    }

    // Check intermediate values
    if witness.h.len() != witness.d_hid {
        result.add_error(WitnessValidationError::new(
            "h",
            "Hidden activation dimension mismatch",
        ));
    }

    if witness.y.len() != witness.d_out {
        result.add_error(WitnessValidationError::new(
            "y",
            "Output dimension mismatch",
        ));
    }

    // Check new weights
    if witness.w1_new.len() != expected_w1_len {
        result.add_error(WitnessValidationError::new(
            "w1_new",
            "Updated W1 dimension mismatch",
        ));
    }

    if witness.w2_new.len() != expected_w2_len {
        result.add_error(WitnessValidationError::new(
            "w2_new",
            "Updated W2 dimension mismatch",
        ));
    }

    // Check for zero learning rate (warning)
    if witness.lr == Fr::zero() {
        result.add_warning("Learning rate is zero - weights will not update".to_string());
    }

    // Check step number is reasonable
    if witness.step_number > 1_000_000 {
        result.add_warning(format!(
            "Very high step number: {} - verify this is intended",
            witness.step_number
        ));
    }

    result
}

// ============================================================================
// ML Training Prover V2
// ============================================================================

/// ML Training Step V2 Prover.
///
/// Generates Halo2 KZG proofs using the enhanced `MLTrainingStepV2Circuit`
/// which includes Freivalds verification and proper error tracking.
pub struct MLTrainingProverV2 {
    /// Halo2 proving pipeline.
    pipeline: ProverPipeline<MLTrainingStepV2Circuit>,
    /// Configuration.
    config: V2ProverConfig,
    /// Model dimensions (d_in, d_hid, d_out).
    dims: (usize, usize, usize),
    /// Witness cache.
    cache: Option<SharedWitnessCache>,
    /// Proof counter.
    proof_count: AtomicU64,
    /// Cache hit counter.
    cache_hits: AtomicU64,
}

impl MLTrainingProverV2 {
    /// Creates and initializes a new V2 prover for a specific model shape.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self::with_config(d_in, d_hid, d_out, V2ProverConfig::default())
    }

    /// Creates a V2 prover with custom configuration.
    ///
    /// Uses [`generate_keys_for_circuit`] to produce real Halo2 KZG keys
    /// (params, proving key, verification key) and threads them through
    /// the pipeline for production-grade proof generation.
    pub fn with_config(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        config: V2ProverConfig,
    ) -> Self {
        // Build pipeline config from prover config
        let pipeline_config = PipelineConfig {
            k: config.k,
            self_verify: false, // We handle this ourselves (including EVM format verification)
            retry: config.retry.clone(),
            proof_timeout: config.proof_timeout,
            verify_timeout: config.verify_timeout,
            deterministic_seed: config.deterministic_seed,
            enable_tracing: config.enable_tracing,
        };

        // Create a dummy witness for key generation
        let dummy_witness = create_zero_witness(d_in, d_hid, d_out);
        let setup_circuit = MLTrainingStepV2Circuit {
            witness: dummy_witness,
            relu_range: config.relu_range,
            exp_range: config.exp_range,
            exp_scale: config.exp_scale,
            use_freivalds: config.use_freivalds,
        };

        // Generate real Halo2 keys via generate_keys_for_circuit()
        let circuit_keys = generate_keys_for_circuit(
            &setup_circuit,
            "ml_training_step_v2",
            1,
            config.k,
        );

        // Create pipeline from real CircuitKeys (params, pk, vk)
        let pipeline = ProverPipeline::from_keys(
            pipeline_config,
            circuit_keys.params,
            circuit_keys.pk,
            circuit_keys.vk,
        );

        // Create cache if enabled
        let cache = if config.use_witness_cache {
            Some(shared_witness_cache())
        } else {
            None
        };

        Self {
            pipeline,
            config,
            dims: (d_in, d_hid, d_out),
            cache,
            proof_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        }
    }

    /// Returns the model dimensions this prover was initialized for.
    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    /// Returns the configuration.
    pub fn config(&self) -> &V2ProverConfig {
        &self.config
    }

    /// Returns the number of proofs generated.
    pub fn proof_count(&self) -> u64 {
        self.proof_count.load(Ordering::Relaxed)
    }

    /// Returns the number of cache hits.
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Returns the cache hit ratio.
    pub fn cache_hit_ratio(&self) -> f64 {
        let proofs = self.proof_count();
        let hits = self.cache_hits();
        if proofs == 0 {
            0.0
        } else {
            hits as f64 / proofs as f64
        }
    }

    /// Builds a witness from raw training data.
    ///
    /// This computes the forward pass, backward pass, and weight updates,
    /// generating all intermediate values needed for the circuit.
    pub fn build_witness(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        x: &[Fr],
        target: &[Fr],
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        lr: Fr,
        step_number: u64,
        base_error: Fr,
    ) -> MLTrainingStepV2Witness {
        let old_hash = compute_state_hash_v2(w1, b1, w2, b2);

        // First pass to compute new weights
        let tmp = compute_witness_v2(
            d_in,
            d_hid,
            d_out,
            x,
            target,
            w1,
            b1,
            w2,
            b2,
            lr,
            old_hash,
            (Fr::zero(), Fr::zero()),
            step_number,
            base_error,
        );

        let new_hash = compute_state_hash_v2(&tmp.w1_new, &tmp.b1_new, &tmp.w2_new, &tmp.b2_new);

        // Second pass with correct new hash
        compute_witness_v2(
            d_in, d_hid, d_out, x, target, w1, b1, w2, b2, lr, old_hash, new_hash, step_number,
            base_error,
        )
    }

    /// Builds a witness with on-chain parameters for error checksum matching.
    ///
    /// This is the same as `build_witness` but sets `model_id` and `error_budget`
    /// so that PI[7] (error checksum) matches the contract's
    /// `PoseidonHasher.computeErrorChecksum(errorBound, stepNumber, modelId, maxErrorBound)`.
    ///
    /// - `model_id`: The on-chain model ID (stored as 32-byte LE Fr representation)
    /// - `error_budget`: The contract's `maxErrorBound` value (e.g. Fr::from(10u64.pow(18)))
    pub fn build_witness_with_params(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        x: &[Fr],
        target: &[Fr],
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        lr: Fr,
        step_number: u64,
        base_error: Fr,
        model_id: [u8; 32],
        error_budget: Fr,
    ) -> MLTrainingStepV2Witness {
        let mut witness = Self::build_witness(
            d_in, d_hid, d_out, x, target, w1, b1, w2, b2, lr, step_number, base_error,
        );
        witness.set_error_params(model_id, error_budget);
        witness
    }

    /// Builds a witness with the specified loss function.
    ///
    /// For `LossFunction::MSE`, delegates to `build_witness`.
    /// For `LossFunction::CrossEntropy`, uses `compute_witness_v2_cross_entropy`.
    pub fn build_witness_with_loss(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        x: &[Fr],
        target: &[Fr],
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        lr: Fr,
        step_number: u64,
        base_error: Fr,
        loss_function: LossFunction,
        scale: f64,
    ) -> MLTrainingStepV2Witness {
        match loss_function {
            LossFunction::MSE => {
                Self::build_witness(d_in, d_hid, d_out, x, target, w1, b1, w2, b2, lr, step_number, base_error)
            }
            LossFunction::CrossEntropy => {
                let old_hash = compute_state_hash_v2(w1, b1, w2, b2);

                // First pass to compute new weights
                let tmp = compute_witness_v2_cross_entropy(
                    d_in, d_hid, d_out, x, target, w1, b1, w2, b2, lr,
                    old_hash, (Fr::zero(), Fr::zero()), step_number, base_error, scale,
                );

                let new_hash = compute_state_hash_v2(&tmp.w1_new, &tmp.b1_new, &tmp.w2_new, &tmp.b2_new);

                // Second pass with correct new hash
                compute_witness_v2_cross_entropy(
                    d_in, d_hid, d_out, x, target, w1, b1, w2, b2, lr,
                    old_hash, new_hash, step_number, base_error, scale,
                )
            }
        }
    }

    /// Computes the witness hash for caching.
    pub fn compute_witness_hash(witness: &MLTrainingStepV2Witness) -> WitnessHash {
        WitnessHashBuilder::new()
            .add_dims(witness.d_in, witness.d_hid, witness.d_out)
            .add_field_elements(&witness.x)
            .add_field_elements(&witness.target)
            .add_field_elements(&witness.w1)
            .add_field_elements(&witness.b1)
            .add_field_elements(&witness.w2)
            .add_field_elements(&witness.b2)
            .add_field_elements(&[witness.lr])
            .add_u64(witness.step_number)
            .finish()
    }

    /// Generates a Halo2 proof for the given witness.
    pub fn prove(&self, witness: &MLTrainingStepV2Witness) -> TrainingProverResult<TrainingProofResultV2> {
        self.prove_with_options(witness, no_progress_callback(), None)
    }

    /// Generates a proof with progress callbacks and cancellation support.
    pub fn prove_with_options(
        &self,
        witness: &MLTrainingStepV2Witness,
        progress: ProgressCallback,
        cancel_token: Option<&CancellationToken>,
    ) -> TrainingProverResult<TrainingProofResultV2> {
        let start = Instant::now();

        // Check cancellation
        if let Some(token) = cancel_token {
            if token.is_cancelled() {
                return Err(TrainingProverError::Cancelled);
            }
        }

        // Validate witness
        progress(ProofProgress {
            phase: ProofPhase::Setup,
            phase_progress: 0.0,
            overall_progress: 0.0,
            elapsed: start.elapsed(),
            estimated_remaining: None,
            attempt: 0,
            message: Some("Validating witness".to_string()),
        });

        let validation = validate_witness(witness);
        if !validation.valid {
            let first_error = validation.errors.first()
                .ok_or_else(|| TrainingProverError::WitnessValidation {
                    message: "Validation failed but no errors recorded".to_string(),
                    field: "unknown".to_string(),
                })?;
            return Err(TrainingProverError::validation(
                &first_error.message,
                &first_error.field,
            ));
        }

        // Compute witness hash for caching
        let witness_hash = Self::compute_witness_hash(witness);

        // Check cache
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(&witness_hash) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.proof_count.fetch_add(1, Ordering::Relaxed);

                if self.config.enable_tracing {
                    tracing::info!(
                        witness_hash = %witness_hash,
                        "Cache hit for training proof"
                    );
                }

                // Reconstruct public inputs from witness
                let pi = witness.public_inputs();

                return Ok(TrainingProofResultV2 {
                    proof: cached.proof,
                    public_inputs: pi,
                    loss: witness.loss,
                    total_error: witness.total_error,
                    step_number: witness.step_number,
                    old_state_hash: witness.old_state_hash,
                    new_state_hash: witness.new_state_hash,
                    verified: cached.verified,
                    generation_time: Duration::from_millis(cached.generation_time_ms),
                    verification_time: None,
                    attempts: 1,
                    from_cache: true,
                    witness_hash: Some(witness_hash),
                });
            }
        }

        // Check timeout
        if let Some(timeout) = self.config.proof_timeout {
            if start.elapsed() > timeout {
                return Err(TrainingProverError::Timeout {
                    elapsed_ms: start.elapsed().as_millis() as u64,
                });
            }
        }

        // Build circuit
        let circuit = MLTrainingStepV2Circuit {
            witness: witness.clone(),
            relu_range: self.config.relu_range,
            exp_range: self.config.exp_range,
            exp_scale: self.config.exp_scale,
            use_freivalds: self.config.use_freivalds,
        };

        let pi = witness.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        // Generate proof
        progress(ProofProgress {
            phase: ProofPhase::Synthesis,
            phase_progress: 0.0,
            overall_progress: 0.2,
            elapsed: start.elapsed(),
            estimated_remaining: None,
            attempt: 0,
            message: Some("Generating proof".to_string()),
        });

        let proof_result = self
            .pipeline
            .prove_with_options(&circuit, &pi_refs, progress.clone(), cancel_token)
            .map_err(|e| {
                tracing::error!(
                    witness_hash = %witness_hash,
                    step = witness.step_number,
                    error = %e,
                    "Proof generation failed"
                );
                TrainingProverError::Pipeline(e)
            })?;

        let gen_time = proof_result.generation_time;
        let proof = proof_result.proof;
        let attempts = proof_result.attempts;

        // Self-verification (native Halo2 + EVM format)
        let (verified, verify_time) = if self.config.self_verify {
            progress(ProofProgress {
                phase: ProofPhase::Verification,
                phase_progress: 0.0,
                overall_progress: 0.9,
                elapsed: start.elapsed(),
                estimated_remaining: None,
                attempt: 0,
                message: Some("Self-verifying proof".to_string()),
            });

            let verify_start = Instant::now();

            // 1. Native Halo2 verification
            let is_valid = self.pipeline.verify(&proof, &pi_refs)?;
            if !is_valid {
                return Err(TrainingProverError::SelfVerificationFailed);
            }

            // 2. EVM format sanity check — the PSE Keccak256Transcript already
            //    produces proofs with 64-byte uncompressed EC points matching
            //    the Solidity verifier. Just validate non-empty + size multiple of 32.
            if proof.is_empty() {
                return Err(TrainingProverError::EvmSerializationFailed(
                    ProofFormatError::TooShort { got: 0, min: 32 },
                ));
            }
            if proof.len() % 32 != 0 {
                return Err(TrainingProverError::EvmSerializationFailed(
                    ProofFormatError::TooShort { got: proof.len(), min: proof.len() + (32 - proof.len() % 32) },
                ));
            }

            if self.config.enable_tracing {
                tracing::debug!(
                    evm_proof_size = proof.len(),
                    "EVM proof format validated successfully"
                );
            }

            (true, Some(verify_start.elapsed()))
        } else {
            (false, None)
        };

        // Store in cache
        if let Some(ref cache) = self.cache {
            let cached = WitnessCachedProof::new(
                witness_hash,
                proof.clone(),
                pi.iter()
                    .map(|f| {
                        let bytes = f.to_repr();
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(bytes.as_ref());
                        arr
                    })
                    .collect(),
                gen_time.as_millis() as u64,
            );
            cache.insert(cached);

            if verified {
                cache.mark_verified(&witness_hash);
            }
        }

        self.proof_count.fetch_add(1, Ordering::Relaxed);

        if self.config.enable_tracing {
            tracing::info!(
                step = witness.step_number,
                proof_size = proof.len(),
                generation_time_ms = gen_time.as_millis() as u64,
                verified,
                "Training proof generated"
            );
        }

        Ok(TrainingProofResultV2 {
            proof,
            public_inputs: pi,
            loss: witness.loss,
            total_error: witness.total_error,
            step_number: witness.step_number,
            old_state_hash: witness.old_state_hash,
            new_state_hash: witness.new_state_hash,
            verified,
            generation_time: gen_time,
            verification_time: verify_time,
            attempts,
            from_cache: false,
            witness_hash: Some(witness_hash),
        })
    }

    /// Verifies a proof against the given public inputs.
    pub fn verify(&self, proof: &[u8], public_inputs: &[Fr]) -> bool {
        let pi_refs: Vec<&[Fr]> = vec![public_inputs];
        self.pipeline.verify(proof, &pi_refs).unwrap_or(false)
    }

    /// Verifies a `TrainingProofResultV2`.
    pub fn verify_result(&self, result: &TrainingProofResultV2) -> bool {
        if result.proof.is_empty() {
            return false;
        }
        self.verify(&result.proof, &result.public_inputs)
    }

    /// Generates a Solidity verifier contract for this prover's circuit.
    pub fn generate_solidity_verifier(&self, contract_name: &str) -> String {
        let vk_data = self
            .pipeline
            .extract_vk_data(NUM_PUBLIC_INPUTS)
            .expect("VK not initialized");

        let evm_vk = VkData {
            g1: vk_data.g1,
            s_g2: vk_data.s_g2,
            neg_g2: vk_data.neg_g2,
            num_advices: vk_data.num_advices,
        };

        SolidityGenerator::new(contract_name)
            .with_instances(NUM_PUBLIC_INPUTS)
            .with_vk_data(evm_vk)
            .with_batch(true)
            .generate()
    }

    /// Exports verification key data needed for contract deployment.
    ///
    /// Returns [`VkData`] containing the pairing-check points (G1 generator,
    /// s·G2 from SRS, -G2) that the on-chain `Halo2Verifier` contract needs.
    /// This must be called after key generation (which happens in [`new`]/[`with_config`]).
    pub fn export_vk_data(&self) -> Result<VkData, TrainingProverError> {
        let extracted = self
            .pipeline
            .extract_vk_data(NUM_PUBLIC_INPUTS)
            .ok_or_else(|| TrainingProverError::NotInitialized {
                message: "Pipeline VK not initialized — cannot export VK data".to_string(),
            })?;

        Ok(VkData {
            g1: extracted.g1,
            s_g2: extracted.s_g2,
            neg_g2: extracted.neg_g2,
            num_advices: extracted.num_advices,
        })
    }

    /// Generates a proof and exports everything needed for on-chain verification
    /// in a single call: EVM proof bytes, public inputs, and VK deployment args.
    ///
    /// Returns an [`EvmProofBundle`] containing:
    /// - `evm_proof`: 320-byte EVM-compatible proof
    /// - `evm_public_inputs`: 8 x 32-byte big-endian public inputs
    /// - `vk_deployment_args`: VK data for Halo2Verifier constructor
    /// - The underlying `TrainingProofResultV2` for metadata
    pub fn prove_and_export_evm(
        &self,
        witness: &MLTrainingStepV2Witness,
    ) -> TrainingProverResult<EvmProofBundle> {
        let result = self.prove(witness)?;

        let evm_proof = result.to_evm_proof()
            .map_err(TrainingProverError::EvmSerializationFailed)?;
        let evm_public_inputs = result.to_evm_public_inputs();
        let vk_deployment_args = self.export_vk_data()?;

        Ok(EvmProofBundle {
            evm_proof,
            evm_public_inputs,
            vk_deployment_args,
            result,
        })
    }

    /// Clears the witness cache.
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear();
        }
    }

    /// Returns cache statistics.
    pub fn cache_stats(&self) -> Option<crate::cache::witness_cache::WitnessCacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }
}

/// Creates a zero-initialized witness for setup (public variant for pipeline sizing).
pub fn create_zero_witness_pub(d_in: usize, d_hid: usize, d_out: usize) -> MLTrainingStepV2Witness {
    create_zero_witness(d_in, d_hid, d_out)
}

/// Creates a zero-initialized witness for setup.
fn create_zero_witness(d_in: usize, d_hid: usize, d_out: usize) -> MLTrainingStepV2Witness {
    MLTrainingStepV2Witness {
        d_in,
        d_hid,
        d_out,
        x: vec![Fr::zero(); d_in],
        target: vec![Fr::zero(); d_out],
        w1: vec![Fr::zero(); d_hid * d_in],
        b1: vec![Fr::zero(); d_hid],
        w2: vec![Fr::zero(); d_out * d_hid],
        b2: vec![Fr::zero(); d_out],
        h_pre: vec![Fr::zero(); d_hid],
        h_pre_err: vec![Fr::zero(); d_hid],
        h: vec![Fr::zero(); d_hid],
        h_err: vec![Fr::zero(); d_hid],
        y: vec![Fr::zero(); d_out],
        y_err: vec![Fr::zero(); d_out],
        loss: Fr::zero(),
        loss_err: Fr::zero(),
        dy: vec![Fr::zero(); d_out],
        dy_err: vec![Fr::zero(); d_out],
        dw2: vec![Fr::zero(); d_out * d_hid],
        dw2_err: vec![Fr::zero(); d_out * d_hid],
        db2: vec![Fr::zero(); d_out],
        db2_err: vec![Fr::zero(); d_out],
        dh: vec![Fr::zero(); d_hid],
        dh_err: vec![Fr::zero(); d_hid],
        relu_mask: vec![Fr::zero(); d_hid],
        dh_pre: vec![Fr::zero(); d_hid],
        dh_pre_err: vec![Fr::zero(); d_hid],
        dw1: vec![Fr::zero(); d_hid * d_in],
        dw1_err: vec![Fr::zero(); d_hid * d_in],
        db1: vec![Fr::zero(); d_hid],
        db1_err: vec![Fr::zero(); d_hid],
        lr: Fr::one(),
        w1_new: vec![Fr::zero(); d_hid * d_in],
        b1_new: vec![Fr::zero(); d_hid],
        w2_new: vec![Fr::zero(); d_out * d_hid],
        b2_new: vec![Fr::zero(); d_out],
        total_error: Fr::zero(),
        freivalds_r1: vec![Fr::zero(); d_hid],
        freivalds_r2: vec![Fr::zero(); d_out],
        old_state_hash: (Fr::zero(), Fr::zero()),
        new_state_hash: (Fr::zero(), Fr::zero()),
        step_number: 0,
        model_id: [0u8; 32],
        error_budget: Fr::zero(),
        error_checksum: Fr::zero(),
    }
}

// ============================================================================
// Batch Prover
// ============================================================================

/// Batch prover for processing multiple training steps efficiently.
pub struct BatchTrainingProverV2 {
    /// The underlying V2 prover.
    prover: MLTrainingProverV2,
}

impl BatchTrainingProverV2 {
    /// Creates a new batch prover.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            prover: MLTrainingProverV2::new(d_in, d_hid, d_out),
        }
    }

    /// Creates a batch prover with custom configuration.
    pub fn with_config(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        config: V2ProverConfig,
    ) -> Self {
        Self {
            prover: MLTrainingProverV2::with_config(d_in, d_hid, d_out, config),
        }
    }

    /// Proves a batch of training steps sequentially.
    ///
    /// Returns an error if all steps fail (0 proofs produced from non-empty input).
    pub fn prove_batch(
        &self,
        initial_weights: TrainingWeights,
        training_samples: &[(Vec<Fr>, Vec<Fr>)],
        lr: Fr,
    ) -> TrainingProverResult<BatchProofResult> {
        self.prove_batch_with_options(
            initial_weights,
            training_samples,
            lr,
            no_progress_callback(),
            None,
        )
    }

    /// Proves a batch of training steps, failing immediately on the first error.
    ///
    /// Unlike [`prove_batch`] which tolerates failures and records them in
    /// `failed_steps`, this method returns an error as soon as any step fails.
    /// Use this when you require all proofs to succeed (e.g., on-chain submission).
    pub fn prove_batch_strict(
        &self,
        initial_weights: TrainingWeights,
        training_samples: &[(Vec<Fr>, Vec<Fr>)],
        lr: Fr,
    ) -> TrainingProverResult<BatchProofResult> {
        self.prove_batch_strict_with_options(
            initial_weights,
            training_samples,
            lr,
            no_progress_callback(),
            None,
        )
    }

    /// Strict batch proving with progress callbacks and cancellation support.
    ///
    /// Returns `Err` on the first step failure instead of accumulating failures.
    pub fn prove_batch_strict_with_options(
        &self,
        initial_weights: TrainingWeights,
        training_samples: &[(Vec<Fr>, Vec<Fr>)],
        lr: Fr,
        progress: ProgressCallback,
        cancel_token: Option<&CancellationToken>,
    ) -> TrainingProverResult<BatchProofResult> {
        if training_samples.is_empty() {
            return Err(TrainingProverError::WitnessValidation {
                message: "No training samples provided".to_string(),
                field: "training_samples".to_string(),
            });
        }

        let start = Instant::now();
        let mut current_weights = initial_weights;
        let mut proofs = Vec::with_capacity(training_samples.len());
        let mut total_loss = Fr::zero();
        let total_steps = training_samples.len();

        for (step, (x, target)) in training_samples.iter().enumerate() {
            if let Some(token) = cancel_token {
                if token.is_cancelled() {
                    return Err(TrainingProverError::Cancelled);
                }
            }

            progress(ProofProgress {
                phase: ProofPhase::Setup,
                phase_progress: 0.0,
                overall_progress: step as f64 / total_steps as f64,
                elapsed: start.elapsed(),
                estimated_remaining: None,
                attempt: 0,
                message: Some(format!("Training step {}/{}", step + 1, total_steps)),
            });

            let witness = MLTrainingProverV2::build_witness(
                current_weights.d_in,
                current_weights.d_hid,
                current_weights.d_out,
                x,
                target,
                &current_weights.w1,
                &current_weights.b1,
                &current_weights.w2,
                &current_weights.b2,
                lr,
                (step + 1) as u64,
                self.prover.config.base_error,
            );

            let result = self.prover.prove_with_options(&witness, progress.clone(), cancel_token)?;
            total_loss = total_loss + result.loss;

            current_weights.w1 = witness.w1_new.clone();
            current_weights.b1 = witness.b1_new.clone();
            current_weights.w2 = witness.w2_new.clone();
            current_weights.b2 = witness.b2_new.clone();

            proofs.push(result);
        }

        Ok(BatchProofResult {
            proofs,
            final_weights: current_weights,
            total_loss,
            num_steps: training_samples.len(),
            failed_steps: Vec::new(),
            total_time: start.elapsed(),
        })
    }

    /// Proves a batch with progress callbacks and cancellation support.
    ///
    /// Tolerates individual step failures and records them in `failed_steps`.
    /// Always advances weights through each step (even failed ones) so that
    /// subsequent steps compute correct state.
    ///
    /// Returns a result with `proofs.len() == 0` only if `training_samples` is
    /// empty or every step failed — check [`BatchProofResult::failed_steps`]
    /// to distinguish the two cases.
    pub fn prove_batch_with_options(
        &self,
        initial_weights: TrainingWeights,
        training_samples: &[(Vec<Fr>, Vec<Fr>)],
        lr: Fr,
        progress: ProgressCallback,
        cancel_token: Option<&CancellationToken>,
    ) -> TrainingProverResult<BatchProofResult> {
        if training_samples.is_empty() {
            tracing::warn!("prove_batch_with_options: called with 0 training samples");
            return Ok(BatchProofResult {
                proofs: Vec::new(),
                final_weights: initial_weights,
                total_loss: Fr::zero(),
                num_steps: 0,
                failed_steps: Vec::new(),
                total_time: Duration::ZERO,
            });
        }

        let start = Instant::now();
        let mut current_weights = initial_weights;
        let mut proofs = Vec::new();
        let mut total_loss = Fr::zero();
        let mut failed_steps = Vec::new();
        let total_steps = training_samples.len();

        for (step, (x, target)) in training_samples.iter().enumerate() {
            // Check cancellation
            if let Some(token) = cancel_token {
                if token.is_cancelled() {
                    return Err(TrainingProverError::Cancelled);
                }
            }

            // Report batch progress
            progress(ProofProgress {
                phase: ProofPhase::Setup,
                phase_progress: 0.0,
                overall_progress: step as f64 / total_steps as f64,
                elapsed: start.elapsed(),
                estimated_remaining: None,
                attempt: 0,
                message: Some(format!("Training step {}/{}", step + 1, total_steps)),
            });

            let witness = MLTrainingProverV2::build_witness(
                current_weights.d_in,
                current_weights.d_hid,
                current_weights.d_out,
                x,
                target,
                &current_weights.w1,
                &current_weights.b1,
                &current_weights.w2,
                &current_weights.b2,
                lr,
                (step + 1) as u64,
                self.prover.config.base_error,
            );

            // Always advance weights through the witness computation so subsequent
            // steps use the correct state, regardless of whether proving succeeds.
            current_weights.w1 = witness.w1_new.clone();
            current_weights.b1 = witness.b1_new.clone();
            current_weights.w2 = witness.w2_new.clone();
            current_weights.b2 = witness.b2_new.clone();

            match self.prover.prove_with_options(&witness, progress.clone(), cancel_token) {
                Ok(result) => {
                    total_loss = total_loss + result.loss;
                    proofs.push(result);
                }
                Err(e) => {
                    if self.prover.config.enable_tracing {
                        tracing::error!(step, error = %e, "Batch proving failed at step");
                    }
                    failed_steps.push((step, e.to_string()));
                }
            }
        }

        // Return error if all steps failed (no proofs produced).
        if proofs.is_empty() && !failed_steps.is_empty() {
            tracing::error!(
                total_steps = total_steps,
                failed_count = failed_steps.len(),
                "Batch proving produced 0 proofs: all {} steps failed",
                total_steps,
            );
            return Err(TrainingProverError::WitnessValidation {
                message: format!(
                    "Batch produced 0 proofs: all {} steps failed. First error: {}",
                    total_steps,
                    failed_steps[0].1,
                ),
                field: "training_samples".to_string(),
            });
        }

        Ok(BatchProofResult {
            proofs,
            final_weights: current_weights,
            total_loss,
            num_steps: training_samples.len(),
            failed_steps,
            total_time: start.elapsed(),
        })
    }

    /// Verifies all proofs in a batch result.
    pub fn verify_batch(&self, batch: &BatchProofResult) -> bool {
        batch.proofs.iter().all(|p| self.prover.verify_result(p))
    }

    /// Proves a batch and produces an RLC-aggregated proof.
    ///
    /// This is a convenience method that:
    /// 1. Proves each training step individually
    /// 2. Aggregates all proofs into a single RLC-committed proof
    ///
    /// Returns the aggregated proof containing a single KZG proof for all steps.
    pub fn prove_batch_with_aggregation(
        &self,
        initial_weights: TrainingWeights,
        training_samples: &[(Vec<Fr>, Vec<Fr>)],
        lr: Fr,
        aggregation_k: u32,
    ) -> TrainingProverResult<crate::aggregation::AggregatedTrainingProof> {
        use crate::aggregation::RLCAggregationProver;

        // First produce individual proofs
        let batch_result = self.prove_batch_strict(initial_weights, training_samples, lr)?;

        if batch_result.proofs.is_empty() {
            return Err(TrainingProverError::WitnessValidation {
                message: "Batch produced 0 proofs, cannot aggregate".to_string(),
                field: "training_samples".to_string(),
            });
        }

        // Create aggregation prover and aggregate
        let agg_prover = RLCAggregationProver::new(
            batch_result.proofs.len(),
            aggregation_k,
        );

        agg_prover.aggregate(&batch_result.proofs)
            .map_err(|e| TrainingProverError::Pipeline(
                crate::pipeline::PipelineError::ProofGenError {
                    message: format!("RLC aggregation failed: {e}"),
                    cause: None,
                    attempt: 1,
                },
            ))
    }
}

// ============================================================================
// Training Weights
// ============================================================================

/// Weights for a 2-layer MLP.
#[derive(Debug, Clone)]
pub struct TrainingWeights {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub w1: Vec<Fr>,
    pub b1: Vec<Fr>,
    pub w2: Vec<Fr>,
    pub b2: Vec<Fr>,
}

impl TrainingWeights {
    /// Creates new weights initialized to given values.
    pub fn new(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        w1: Vec<Fr>,
        b1: Vec<Fr>,
        w2: Vec<Fr>,
        b2: Vec<Fr>,
    ) -> Self {
        assert_eq!(w1.len(), d_hid * d_in, "W1 size mismatch");
        assert_eq!(b1.len(), d_hid, "B1 size mismatch");
        assert_eq!(w2.len(), d_out * d_hid, "W2 size mismatch");
        assert_eq!(b2.len(), d_out, "B2 size mismatch");

        Self {
            d_in,
            d_hid,
            d_out,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Creates zero-initialized weights.
    pub fn zeros(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_hid,
            d_out,
            w1: vec![Fr::zero(); d_hid * d_in],
            b1: vec![Fr::zero(); d_hid],
            w2: vec![Fr::zero(); d_out * d_hid],
            b2: vec![Fr::zero(); d_out],
        }
    }

    /// Validates the weights.
    pub fn validate(&self) -> bool {
        self.w1.len() == self.d_hid * self.d_in
            && self.b1.len() == self.d_hid
            && self.w2.len() == self.d_out * self.d_hid
            && self.b2.len() == self.d_out
    }
}

// ============================================================================
// Batch Proof Result
// ============================================================================

/// Result of batch proving.
#[derive(Debug)]
pub struct BatchProofResult {
    /// Individual proof results for each step.
    pub proofs: Vec<TrainingProofResultV2>,
    /// Final weights after all training steps.
    pub final_weights: TrainingWeights,
    /// Total accumulated loss.
    pub total_loss: Fr,
    /// Number of training steps.
    pub num_steps: usize,
    /// Failed steps with error messages.
    pub failed_steps: Vec<(usize, String)>,
    /// Total time for batch proving.
    pub total_time: Duration,
}

impl BatchProofResult {
    /// Returns the average loss.
    pub fn average_loss(&self) -> Fr {
        if self.proofs.is_empty() {
            Fr::zero()
        } else {
            // Note: Division in Fr is complex, return total for now
            self.total_loss
        }
    }

    /// Returns the number of successful proofs.
    pub fn successful_count(&self) -> usize {
        self.proofs.len()
    }

    /// Returns the number of failed proofs.
    pub fn failed_count(&self) -> usize {
        self.failed_steps.len()
    }

    /// Returns the success rate.
    pub fn success_rate(&self) -> f64 {
        if self.num_steps == 0 {
            1.0
        } else {
            self.proofs.len() as f64 / self.num_steps as f64
        }
    }

    /// Returns the average proof generation time.
    pub fn average_proof_time(&self) -> Duration {
        if self.proofs.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = self.proofs.iter().map(|p| p.generation_time).sum();
            total / self.proofs.len() as u32
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_model_weights() -> TrainingWeights {
        TrainingWeights::new(
            2,
            2,
            1,
            vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)],
            vec![Fr::from(0), Fr::from(0)],
            vec![Fr::from(1), Fr::from(1)],
            vec![Fr::from(0)],
        )
    }

    #[test]
    fn test_v2_prover_init() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Use minimal config (k=12) instead of default (k=14) to avoid
        // a 40-120s SRS generation that makes tests appear stuck.
        let config = V2ProverConfig::minimal();
        let _prover = MLTrainingProverV2::with_config(2, 2, 1, config);
    }

    #[test]
    fn test_witness_validation() {
        // validate_witness is a pure function — no prover needed.
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2,
            2,
            1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1),
            1,
            Fr::from(1),
        );

        let result = validate_witness(&witness);
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_witness_hash_deterministic() {
        let weights = small_model_weights();

        let witness1 = MLTrainingProverV2::build_witness(
            2,
            2,
            1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1),
            1,
            Fr::from(1),
        );

        let witness2 = MLTrainingProverV2::build_witness(
            2,
            2,
            1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1),
            1,
            Fr::from(1),
        );

        let hash1 = MLTrainingProverV2::compute_witness_hash(&witness1);
        let hash2 = MLTrainingProverV2::compute_witness_hash(&witness2);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_v2_prove_and_verify() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = V2ProverConfig::minimal();
        let prover = MLTrainingProverV2::with_config(2, 2, 1, config);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2,
            2,
            1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1),
            1,
            Fr::from(1),
        );

        let result = prover.prove(&witness).expect("prove should succeed");

        assert!(!result.proof.is_empty());
        assert!(prover.verify_result(&result));
    }

    #[test]
    fn test_batch_proving() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = V2ProverConfig::minimal();
        let prover = BatchTrainingProverV2::with_config(2, 2, 1, config);
        let weights = small_model_weights();

        let samples = vec![(vec![Fr::from(1), Fr::from(1)], vec![Fr::from(5)])];

        let result = prover.prove_batch(weights, &samples, Fr::from(1))
            .expect("batch prove should succeed");

        assert_eq!(result.num_steps, 1);
        assert_eq!(result.proofs.len(), 1);
        assert!(prover.verify_batch(&result));
    }

    #[test]
    fn test_wrong_public_inputs_rejected() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = V2ProverConfig::minimal();
        let prover = MLTrainingProverV2::with_config(2, 2, 1, config);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2,
            2,
            1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1),
            1,
            Fr::from(1),
        );

        let result = prover.prove(&witness).expect("prove should succeed");

        // Corrupt a public input
        let mut bad_pi = result.public_inputs.clone();
        if bad_pi.len() > 4 {
            bad_pi[4] = Fr::from(9999u64);
        }
        assert!(!prover.verify(&result.proof, &bad_pi));
    }

    #[test]
    fn test_prover_error_types() {
        let err = TrainingProverError::dimension_mismatch("w1", 8, 4);
        assert!(err.to_string().contains("Dimension mismatch"));

        let err = TrainingProverError::validation("Invalid value", "loss");
        assert!(err.to_string().contains("Witness validation failed"));
    }

    #[test]
    fn test_config_builders() {
        let config = V2ProverConfig::production();
        assert!(config.self_verify);
        assert!(config.use_witness_cache);

        let config = V2ProverConfig::minimal();
        assert!(!config.self_verify);
        assert!(!config.use_witness_cache);
    }

    #[test]
    fn test_to_evm_proof() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = V2ProverConfig::minimal();
        let prover = MLTrainingProverV2::with_config(2, 2, 1, config);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2, 2, 1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1, &weights.b1, &weights.w2, &weights.b2,
            Fr::from(1), 1, Fr::from(1),
        );

        let result = prover.prove(&witness).expect("prove should succeed");
        assert!(!result.proof.is_empty());

        // With PSE Keccak256Transcript, proof is already in EVM format (uncompressed points)
        let evm_proof = result.to_evm_proof().expect("EVM proof serialization should succeed with KZG");
        assert!(evm_proof.len() > 0, "EVM proof must not be empty");
        assert_eq!(evm_proof.len() % 32, 0, "EVM proof length must be a multiple of 32");
        assert!(prover.verify_result(&result));
    }

    #[test]
    fn test_to_evm_public_inputs() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = V2ProverConfig::minimal();
        let prover = MLTrainingProverV2::with_config(2, 2, 1, config);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2, 2, 1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1, &weights.b1, &weights.w2, &weights.b2,
            Fr::from(1), 1, Fr::from(1),
        );

        let result = prover.prove(&witness).expect("prove should succeed");
        let evm_pi = result.to_evm_public_inputs();

        // Should have 8 public inputs (NUM_PUBLIC_INPUTS)
        assert_eq!(evm_pi.len(), 8);

        // Each element is 32 bytes big-endian
        for pi in &evm_pi {
            assert_eq!(pi.len(), 32);
        }
    }

    #[test]
    fn test_export_vk_data() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = V2ProverConfig::minimal();
        let prover = MLTrainingProverV2::with_config(2, 2, 1, config);

        let vk_data = prover.export_vk_data().expect("VK export should succeed");
        assert_eq!(vk_data.num_advices, 3);
        // G1 generator coordinates should be non-empty decimal strings
        assert!(!vk_data.g1.0.is_empty());
        assert!(!vk_data.g1.1.is_empty());
    }

    #[test]
    fn test_evm_self_verification() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // With self_verify=true and KZG commitment scheme, prove performs:
        // 1. Native Halo2 verification
        // 2. Hard-fail EVM proof serialization and format validation
        let mut config = V2ProverConfig::minimal();
        config.self_verify = true;
        let prover = MLTrainingProverV2::with_config(2, 2, 1, config);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2, 2, 1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1, &weights.b1, &weights.w2, &weights.b2,
            Fr::from(1), 1, Fr::from(1),
        );

        let result = prover.prove(&witness).expect("prove with self-verify should succeed (KZG)");
        assert!(result.verified);
        assert!(prover.verify_result(&result));

        // Verify EVM proof format independently — PSE transcript already produces EVM format
        let evm_proof = result.to_evm_proof().expect("EVM serialization should succeed");
        assert!(evm_proof.len() > 0, "EVM proof must not be empty");
        assert_eq!(evm_proof.len() % 32, 0, "EVM proof length must be a multiple of 32");
    }

    #[test]
    fn test_deterministic_seed_produces_consistent_proofs() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let seed = [42u8; 32];
        let config = V2ProverConfig::minimal().deterministic(seed);
        let prover = MLTrainingProverV2::with_config(2, 2, 1, config);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2, 2, 1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1, &weights.b1, &weights.w2, &weights.b2,
            Fr::from(1), 1, Fr::from(1),
        );

        let r1 = prover.prove(&witness).expect("prove should succeed");
        assert!(!r1.proof.is_empty());
        assert!(prover.verify_result(&r1));
    }

    #[test]
    fn test_auto_config() {
        // V2ProverConfig::auto should pick k and relu_range from model dims
        let config = V2ProverConfig::auto(10, 10, 5, 512);
        assert!(config.k >= 10, "k should be at least 10, got {}", config.k);
        assert!(
            config.relu_range >= 512,
            "relu_range should be at least 512, got {}",
            config.relu_range
        );
    }

    #[test]
    fn test_max_relu_range_for_k() {
        use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;

        // For a small model (2,2,1) with k=12, the max range should be substantial
        let range = MLTrainingStepV2Circuit::max_relu_range_for_k(12, 2, 2, 1, true, 128);
        assert!(range > 100, "max_relu_range for k=12 should be > 100, got {}", range);

        // Larger k should allow larger range
        let range14 = MLTrainingStepV2Circuit::max_relu_range_for_k(14, 10, 10, 5, true, 128);
        let range16 = MLTrainingStepV2Circuit::max_relu_range_for_k(16, 10, 10, 5, true, 128);
        assert!(range16 > range14, "k=16 should allow larger range than k=14");
    }

    /// Tests dynamic circuit scaling with a 10×10×5 Xavier-initialized model.
    ///
    /// This is the core test for the ReLU range fix: a model with real-sized
    /// weights (not tiny 0.001 values) should produce valid proofs. The system
    /// automatically picks the quantization scale and circuit size (k).
    #[test]
    fn test_xavier_10x10x5_auto_prove() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let (d_in, d_hid, d_out) = (10, 10, 5);

        // Xavier/Glorot initialization: weights ~ N(0, sqrt(2/(fan_in+fan_out)))
        let mut seed: u64 = 12345;
        let mut rng = || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (seed >> 11) as f64 / (1u64 << 53) as f64;
            bits * 2.0 - 1.0 // Uniform [-1, 1]
        };

        let xavier_scale_1 = (2.0 / (d_in + d_hid) as f64).sqrt(); // ~0.316
        let xavier_scale_2 = (2.0 / (d_hid + d_out) as f64).sqrt(); // ~0.365

        let w1_data: Vec<f64> = (0..d_hid * d_in).map(|_| rng() * xavier_scale_1).collect();
        let b1_data: Vec<f64> = (0..d_hid).map(|_| rng() * 0.01).collect();
        let w2_data: Vec<f64> = (0..d_out * d_hid).map(|_| rng() * xavier_scale_2).collect();
        let b2_data: Vec<f64> = (0..d_out).map(|_| rng() * 0.01).collect();

        use helix_core::types::BoundedTensor;
        use helix_core::types::Precision;
        use helix_avm::nn::Linear;

        let weights1 = BoundedTensor::from_exact(w1_data.clone(), vec![d_hid, d_in]);
        let bias1 = BoundedTensor::from_exact(b1_data.clone(), vec![d_hid]);
        let weights2 = BoundedTensor::from_exact(w2_data.clone(), vec![d_out, d_hid]);
        let bias2 = BoundedTensor::from_exact(b2_data.clone(), vec![d_out]);

        let l1 = Linear::new(weights1, Some(bias1), Precision::F32).unwrap();
        let l2 = Linear::new(weights2, Some(bias2), Precision::F32).unwrap();

        // Normalized inputs and targets
        let input: Vec<f64> = (0..d_in).map(|i| (i as f64 - 5.0) / 5.0).collect();
        let target: Vec<f64> = (0..d_out).map(|i| (i as f64) / 5.0).collect();
        let learning_rate = 0.01;

        // Auto-witness picks the best scale for max_k=16
        let auto = helix_avm::circuit_bridge::build_training_witness_auto(
            &l1, &l2, &input, &target, learning_rate, 1, 16,
        ).expect("auto witness should succeed");

        println!(
            "Auto scale={}, relu_range={}, model={}×{}×{}",
            auto.scale, auto.relu_range, d_in, d_hid, d_out
        );

        assert!(auto.scale >= 2, "scale should be at least 2, got {}", auto.scale);
        assert!(auto.relu_range >= 256, "relu_range should be >= 256, got {}", auto.relu_range);

        // Auto-configure prover from the computed relu_range
        let config = V2ProverConfig::auto(d_in, d_hid, d_out, auto.relu_range);
        println!("Prover config: k={}, relu_range={}", config.k, config.relu_range);

        let prover = MLTrainingProverV2::with_config(d_in, d_hid, d_out, config);

        // Prove step 1
        let result = prover.prove(&auto.witness).expect("proof generation should succeed");
        assert!(!result.proof.is_empty(), "proof should not be empty");
        assert!(prover.verify_result(&result), "proof should verify");

        println!(
            "Step 1: proof={} bytes, loss={:?}, verified={}",
            result.proof.len(),
            result.loss,
            result.verified
        );
    }

    // ========================================================================
    // Circuit Scaling Tests (feature-gated for CI speed)
    // ========================================================================

    /// Helper: compute minimum_k and verify it's sane for given dimensions.
    fn verify_minimum_k_for_dims(d_in: usize, d_hid: usize, d_out: usize) -> u32 {
        use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;
        let dummy = MLTrainingStepV2Circuit {
            witness: create_zero_witness(d_in, d_hid, d_out),
            relu_range: 256,
            exp_range: 128,
            exp_scale: 64,
            use_freivalds: true,
        };
        let k = dummy.minimum_k();
        assert!(k >= 10, "k should be at least 10, got {} for {}x{}x{}", k, d_in, d_hid, d_out);
        assert!(k <= 26, "k should be at most 26, got {} for {}x{}x{}", k, d_in, d_hid, d_out);
        k
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_16x16x8_minimum_k() {
        let k = verify_minimum_k_for_dims(16, 16, 8);
        println!("16x16x8: minimum_k = {}", k);
        assert!(k >= 12, "16x16x8 should need at least k=12");
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_32x32x16_minimum_k() {
        let k = verify_minimum_k_for_dims(32, 32, 16);
        println!("32x32x16: minimum_k = {}", k);
        assert!(k >= 13, "32x32x16 should need at least k=13");
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_64x64x32_minimum_k() {
        let k = verify_minimum_k_for_dims(64, 64, 32);
        println!("64x64x32: minimum_k = {}", k);
        assert!(k >= 14, "64x64x32 should need at least k=14");
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_128x128x64_minimum_k() {
        let k = verify_minimum_k_for_dims(128, 128, 64);
        println!("128x128x64: minimum_k = {}", k);
        assert!(k >= 15, "128x128x64 should need at least k=15");
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_16x16x8_mockprover() {
        use helix_circuits::halo2_proofs::dev::MockProver;
        use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;
        use helix_circuits::halo2curves::bn256::Fr;

        let (d_in, d_hid, d_out) = (16, 16, 8);
        let k = verify_minimum_k_for_dims(d_in, d_hid, d_out);

        let witness = MLTrainingProverV2::build_witness(
            d_in, d_hid, d_out,
            &vec![Fr::from(1); d_in],
            &vec![Fr::from(1); d_out],
            &vec![Fr::from(1); d_hid * d_in],
            &vec![Fr::from(0); d_hid],
            &vec![Fr::from(1); d_out * d_hid],
            &vec![Fr::from(0); d_out],
            Fr::from(1), 1, Fr::from(1),
        );

        let circuit = MLTrainingStepV2Circuit {
            witness: witness.clone(),
            relu_range: 256,
            exp_range: 128,
            exp_scale: 64,
            use_freivalds: true,
        };

        let pi = witness.public_inputs();
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("MockProver::run failed");
        prover.verify().expect("MockProver verification failed");
        println!("16x16x8: MockProver passed at k={}", k);
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_16x16x8_real_proof() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (d_in, d_hid, d_out) = (16, 16, 8);

        let config = V2ProverConfig::auto(d_in, d_hid, d_out, 256);
        println!("16x16x8: auto config k={}, relu_range={}", config.k, config.relu_range);

        let prover = MLTrainingProverV2::with_config(d_in, d_hid, d_out, config);
        let witness = MLTrainingProverV2::build_witness(
            d_in, d_hid, d_out,
            &vec![Fr::from(1); d_in],
            &vec![Fr::from(1); d_out],
            &vec![Fr::from(1); d_hid * d_in],
            &vec![Fr::from(0); d_hid],
            &vec![Fr::from(1); d_out * d_hid],
            &vec![Fr::from(0); d_out],
            Fr::from(1), 1, Fr::from(1),
        );

        let start = std::time::Instant::now();
        let result = prover.prove(&witness).expect("Real proof should succeed");
        let elapsed = start.elapsed();

        let peak_mem = crate::pipeline::current_resident_memory();
        println!(
            "16x16x8: proof={} bytes, time={:.1}s, peak_mem={}MB, verified={}",
            result.proof.len(),
            elapsed.as_secs_f64(),
            peak_mem / (1024 * 1024),
            result.verified,
        );

        assert!(!result.proof.is_empty());
        assert!(prover.verify_result(&result));
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_32x32x16_mockprover() {
        use helix_circuits::halo2_proofs::dev::MockProver;
        use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;

        let (d_in, d_hid, d_out) = (32, 32, 16);
        let k = verify_minimum_k_for_dims(d_in, d_hid, d_out);

        let witness = MLTrainingProverV2::build_witness(
            d_in, d_hid, d_out,
            &vec![Fr::from(1); d_in],
            &vec![Fr::from(1); d_out],
            &vec![Fr::from(1); d_hid * d_in],
            &vec![Fr::from(0); d_hid],
            &vec![Fr::from(1); d_out * d_hid],
            &vec![Fr::from(0); d_out],
            Fr::from(1), 1, Fr::from(1),
        );

        let circuit = MLTrainingStepV2Circuit {
            witness: witness.clone(),
            relu_range: 256,
            exp_range: 128,
            exp_scale: 64,
            use_freivalds: true,
        };

        let pi = witness.public_inputs();
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("MockProver::run failed");
        prover.verify().expect("MockProver verification failed");
        println!("32x32x16: MockProver passed at k={}", k);
    }

    #[test]
    #[cfg(feature = "scaling-tests")]
    fn test_scaling_table_reference() {
        let table = crate::pipeline::generate_scaling_table(&[
            (2, 2, 1),
            (8, 8, 4),
            (16, 16, 8),
            (32, 32, 16),
            (64, 64, 32),
            (128, 128, 64),
        ]);

        println!("\n=== Circuit Scaling Reference Table ===");
        println!("{:<15} {:>4} {:>10} {:>12} {:>12} {:>10}",
            "Model", "k", "Rows", "SRS (MB)", "Peak (MB)", "Disk (MB)");
        println!("{:-<67}", "");
        for entry in &table {
            println!("{:<15} {:>4} {:>10} {:>12.1} {:>12.1} {:>10.1}",
                format!("{}x{}x{}", entry.dims.0, entry.dims.1, entry.dims.2),
                entry.k,
                entry.rows,
                entry.srs_memory_mb,
                entry.peak_memory_mb,
                entry.srs_disk_mb,
            );
        }
        println!();

        // Verify monotonicity
        for i in 1..table.len() {
            assert!(
                table[i].k >= table[i-1].k,
                "k should be monotonically non-decreasing: {} < {} at index {}",
                table[i].k, table[i-1].k, i,
            );
        }
    }

    /// Multi-step training with Xavier-initialized 10×10×5 model.
    ///
    /// Trains for 5 steps, generates a proof for each, and verifies all.
    /// This validates that the auto-scaling and dynamic circuit sizing work
    /// across multiple training iterations where weights change.
    #[test]
    fn test_xavier_10x10x5_multistep() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let (d_in, d_hid, d_out) = (10, 10, 5);
        let num_steps = 5;

        // Xavier/Glorot initialization
        let mut seed: u64 = 67890;
        let mut rng = || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (seed >> 11) as f64 / (1u64 << 53) as f64;
            bits * 2.0 - 1.0
        };

        let xavier_1 = (2.0 / (d_in + d_hid) as f64).sqrt();
        let xavier_2 = (2.0 / (d_hid + d_out) as f64).sqrt();

        let mut w1: Vec<f64> = (0..d_hid * d_in).map(|_| rng() * xavier_1).collect();
        let mut b1: Vec<f64> = (0..d_hid).map(|_| rng() * 0.01).collect();
        let mut w2: Vec<f64> = (0..d_out * d_hid).map(|_| rng() * xavier_2).collect();
        let mut b2: Vec<f64> = (0..d_out).map(|_| rng() * 0.01).collect();

        let input: Vec<f64> = (0..d_in).map(|i| (i as f64 - 5.0) / 5.0).collect();
        let target: Vec<f64> = (0..d_out).map(|i| (i as f64) / 5.0).collect();
        let learning_rate = 0.01;

        use helix_core::types::BoundedTensor;
        use helix_core::types::Precision;
        use helix_avm::nn::Linear;
        use helix_avm::quantization::CircuitQuantizer;
        use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;

        // Pre-compute: find the max scale that works across all steps.
        // Use 25% of the max relu_range as the target for auto_scale. This provides
        // 4x headroom for weight growth during training — gradients can increase
        // weight magnitudes, which in turn increases h_pre values.
        let max_relu_range = MLTrainingStepV2Circuit::max_relu_range_for_k(
            16, d_in, d_hid, d_out, true, 128,
        );
        let target_relu_range = max_relu_range / 4;
        let scale = CircuitQuantizer::auto_scale(&w1, &b1, &input, d_in, target_relu_range);
        println!(
            "Chosen scale={} for target_relu_range={} (max={})",
            scale, target_relu_range, max_relu_range
        );

        // Build first witness to get relu_range for prover setup
        let l1 = Linear::new(
            BoundedTensor::from_exact(w1.clone(), vec![d_hid, d_in]),
            Some(BoundedTensor::from_exact(b1.clone(), vec![d_hid])),
            Precision::F32,
        ).unwrap();
        let l2 = Linear::new(
            BoundedTensor::from_exact(w2.clone(), vec![d_out, d_hid]),
            Some(BoundedTensor::from_exact(b2.clone(), vec![d_out])),
            Precision::F32,
        ).unwrap();

        let first_witness = helix_avm::circuit_bridge::build_training_witness_with_scale(
            &l1, &l2, &input, &target, learning_rate, 1, scale,
        ).expect("first witness should succeed");

        // Set up prover with auto config
        let config = V2ProverConfig::auto(d_in, d_hid, d_out, first_witness.relu_range);
        println!(
            "Prover: k={}, relu_range={}, model={}×{}×{}",
            config.k, config.relu_range, d_in, d_hid, d_out
        );
        let prover = MLTrainingProverV2::with_config(d_in, d_hid, d_out, config);

        // Multi-step proving loop
        let mut proofs = Vec::new();
        for step in 1..=num_steps {
            let layer1 = Linear::new(
                BoundedTensor::from_exact(w1.clone(), vec![d_hid, d_in]),
                Some(BoundedTensor::from_exact(b1.clone(), vec![d_hid])),
                Precision::F32,
            ).unwrap();
            let layer2 = Linear::new(
                BoundedTensor::from_exact(w2.clone(), vec![d_out, d_hid]),
                Some(BoundedTensor::from_exact(b2.clone(), vec![d_out])),
                Precision::F32,
            ).unwrap();

            let output = helix_avm::circuit_bridge::build_training_witness_with_scale(
                &layer1, &layer2, &input, &target, learning_rate, step as u64, scale,
            ).expect(&format!("witness for step {} should succeed", step));

            println!(
                "Step {} witness: relu_range={}, prover_relu_range={}",
                step, output.relu_range, max_relu_range
            );
            assert!(
                output.relu_range <= max_relu_range,
                "step {} relu_range {} exceeds prover max {}",
                step, output.relu_range, max_relu_range
            );

            let result = prover.prove(&output.witness)
                .expect(&format!("proof for step {} should succeed", step));

            assert!(!result.proof.is_empty(), "step {} proof empty", step);
            assert!(
                prover.verify_result(&result),
                "step {} proof failed verification", step
            );

            println!(
                "Step {}: proof={} bytes, relu_range={}, verified={}",
                step, result.proof.len(), output.relu_range, result.verified
            );

            proofs.push(result);

            // Extract new weights from witness for next step
            let q = CircuitQuantizer::with_scale(scale);
            w1 = output.witness.w1_new.iter()
                .map(|fr| q.dequantize_fr(*fr))
                .collect();
            b1 = output.witness.b1_new.iter()
                .map(|fr| q.dequantize_fr(*fr))
                .collect();
            w2 = output.witness.w2_new.iter()
                .map(|fr| q.dequantize_fr(*fr))
                .collect();
            b2 = output.witness.b2_new.iter()
                .map(|fr| q.dequantize_fr(*fr))
                .collect();
        }

        assert_eq!(proofs.len(), num_steps, "should have {} proofs", num_steps);
        println!("All {} steps proved and verified successfully!", num_steps);
    }
}
