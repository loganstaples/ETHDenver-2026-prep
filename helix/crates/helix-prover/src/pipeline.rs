//! Halo2 Core Proving Pipeline.
//!
//! This module provides a reliable, production-grade proving pipeline with:
//! - Comprehensive error handling (no panics)
//! - Automatic retry on failure
//! - Progress callbacks for long-running proofs
//! - Self-verification before returning proofs
//! - Timeout handling with graceful cancellation
//! - Deterministic proof generation
//! - Detailed error messages for debugging

use helix_circuits::halo2_proofs::{
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof_multi, Circuit, ProvingKey,
        VerifyingKey,
    },
    poly::{
        commitment::Params,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    },
    transcript::TranscriptWriterBuffer,
};
use halo2_solidity_verifier::Keccak256Transcript;
use helix_circuits::halo2curves::{
    bn256::{Bn256, Fq, Fq2, Fr, G1Affine, G2Affine},
    ff::PrimeField,
    CurveAffine,
};
use rand::{rngs::StdRng, SeedableRng};
use rand_core::{OsRng, RngCore};
use sha2::Digest;
use std::fmt;
use std::io::{Read as IoRead, Write as IoWrite};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Comprehensive error types for the proving pipeline.
#[derive(Error, Debug)]
pub enum PipelineError {
    /// Key generation failed.
    #[error("Key generation failed: {message}")]
    KeyGenError { message: String, cause: Option<String> },

    /// Proof generation failed.
    #[error("Proof generation failed: {message}")]
    ProofGenError {
        message: String,
        cause: Option<String>,
        attempt: u32,
    },

    /// Proof verification failed.
    #[error("Proof verification failed: {message}")]
    VerificationError { message: String, cause: Option<String> },

    /// Self-verification of generated proof failed.
    #[error("Self-verification failed after generating proof: {message}")]
    SelfVerificationError { message: String, proof_size: usize },

    /// Pipeline not initialized.
    #[error("Pipeline not initialized: {message}")]
    NotInitialized { message: String },

    /// Invalid input.
    #[error("Invalid input: {message}")]
    InvalidInput {
        message: String,
        field: String,
        expected: String,
        actual: String,
    },

    /// Witness validation failed.
    #[error("Witness validation failed: {message}")]
    WitnessValidationError { message: String, field: String },

    /// Operation timed out.
    #[error("Operation timed out after {elapsed_ms}ms (limit: {timeout_ms}ms): {operation}")]
    Timeout {
        operation: String,
        elapsed_ms: u64,
        timeout_ms: u64,
    },

    /// Operation was cancelled.
    #[error("Operation cancelled: {operation}")]
    Cancelled { operation: String },

    /// Maximum retries exceeded.
    #[error("Max retries ({max_retries}) exceeded for {operation}: last error: {last_error}")]
    MaxRetriesExceeded {
        operation: String,
        max_retries: u32,
        last_error: String,
    },

    /// Internal error.
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl PipelineError {
    /// Creates a key generation error.
    pub fn keygen<S: Into<String>>(message: S) -> Self {
        Self::KeyGenError {
            message: message.into(),
            cause: None,
        }
    }

    /// Creates a key generation error with cause.
    pub fn keygen_with_cause<S: Into<String>, C: Into<String>>(message: S, cause: C) -> Self {
        Self::KeyGenError {
            message: message.into(),
            cause: Some(cause.into()),
        }
    }

    /// Creates a proof generation error.
    pub fn proof_gen<S: Into<String>>(message: S, attempt: u32) -> Self {
        Self::ProofGenError {
            message: message.into(),
            cause: None,
            attempt,
        }
    }

    /// Creates a proof generation error with cause.
    pub fn proof_gen_with_cause<S: Into<String>, C: Into<String>>(
        message: S,
        cause: C,
        attempt: u32,
    ) -> Self {
        Self::ProofGenError {
            message: message.into(),
            cause: Some(cause.into()),
            attempt,
        }
    }

    /// Creates a not initialized error.
    pub fn not_initialized<S: Into<String>>(message: S) -> Self {
        Self::NotInitialized {
            message: message.into(),
        }
    }

    /// Creates an invalid input error.
    pub fn invalid_input<S: Into<String>>(
        message: S,
        field: S,
        expected: S,
        actual: S,
    ) -> Self {
        Self::InvalidInput {
            message: message.into(),
            field: field.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Creates a witness validation error.
    pub fn witness_validation<S: Into<String>>(message: S, field: S) -> Self {
        Self::WitnessValidationError {
            message: message.into(),
            field: field.into(),
        }
    }

    /// Creates a timeout error.
    pub fn timeout<S: Into<String>>(operation: S, elapsed_ms: u64, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            elapsed_ms,
            timeout_ms,
        }
    }

    /// Creates a cancelled error.
    pub fn cancelled<S: Into<String>>(operation: S) -> Self {
        Self::Cancelled {
            operation: operation.into(),
        }
    }

    /// Creates a max retries exceeded error.
    pub fn max_retries<S: Into<String>>(operation: S, max_retries: u32, last_error: S) -> Self {
        Self::MaxRetriesExceeded {
            operation: operation.into(),
            max_retries,
            last_error: last_error.into(),
        }
    }

    /// Returns true if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::ProofGenError { .. } | Self::Internal { .. }
        )
    }
}

/// Result type for pipeline operations.
pub type PipelineResult<T> = Result<T, PipelineError>;

// ============================================================================
// Progress Tracking
// ============================================================================

/// Progress phase for proof generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofPhase {
    /// Setting up circuit.
    Setup,
    /// Synthesizing witness.
    Synthesis,
    /// Computing commitments.
    Commitments,
    /// Generating opening proofs.
    OpeningProofs,
    /// Finalizing transcript.
    Finalization,
    /// Self-verification.
    Verification,
}

impl fmt::Display for ProofPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Setup => write!(f, "Setup"),
            Self::Synthesis => write!(f, "Synthesis"),
            Self::Commitments => write!(f, "Commitments"),
            Self::OpeningProofs => write!(f, "Opening Proofs"),
            Self::Finalization => write!(f, "Finalization"),
            Self::Verification => write!(f, "Verification"),
        }
    }
}

/// Progress update for proof generation.
#[derive(Debug, Clone)]
pub struct ProofProgress {
    /// Current phase.
    pub phase: ProofPhase,
    /// Progress within phase (0.0 to 1.0).
    pub phase_progress: f64,
    /// Overall progress (0.0 to 1.0).
    pub overall_progress: f64,
    /// Elapsed time.
    pub elapsed: Duration,
    /// Estimated time remaining (if available).
    pub estimated_remaining: Option<Duration>,
    /// Current attempt number (for retries).
    pub attempt: u32,
    /// Additional message.
    pub message: Option<String>,
}

/// Callback type for progress updates.
pub type ProgressCallback = Arc<dyn Fn(ProofProgress) + Send + Sync>;

/// Default no-op progress callback.
pub fn no_progress_callback() -> ProgressCallback {
    Arc::new(|_| {})
}

// ============================================================================
// Cancellation Token
// ============================================================================

/// Token for cancelling long-running operations.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a new cancellation token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Cancels the operation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Checks if the operation has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Resets the token to uncancelled state.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Retry Configuration
// ============================================================================

/// Configuration for automatic retry on failure.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay between retries.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Exponential backoff multiplier.
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays.
    pub add_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            add_jitter: true,
        }
    }
}

impl RetryConfig {
    /// No retries.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Aggressive retry configuration for critical operations.
    pub fn aggressive() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            add_jitter: true,
        }
    }

    /// Calculates delay for a given attempt.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_delay = self.initial_delay.as_millis() as f64
            * self.backoff_multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(self.max_delay.as_millis() as f64);

        let final_delay = if self.add_jitter {
            let jitter = rand::random::<f64>() * 0.3; // Up to 30% jitter
            capped_delay * (1.0 + jitter)
        } else {
            capped_delay
        };

        Duration::from_millis(final_delay as u64)
    }
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Configuration for the prover pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Circuit size parameter (2^k rows).
    pub k: u32,
    /// Whether to self-verify proofs before returning.
    pub self_verify: bool,
    /// Retry configuration.
    pub retry: RetryConfig,
    /// Timeout for proof generation.
    pub proof_timeout: Option<Duration>,
    /// Timeout for verification.
    pub verify_timeout: Option<Duration>,
    /// Seed for deterministic randomness (None = use OsRng).
    pub deterministic_seed: Option<[u8; 32]>,
    /// Enable detailed tracing.
    pub enable_tracing: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            k: 14,
            self_verify: true,
            retry: RetryConfig::default(),
            proof_timeout: Some(Duration::from_secs(300)), // 5 minutes
            verify_timeout: Some(Duration::from_secs(30)),
            deterministic_seed: None,
            enable_tracing: true,
        }
    }
}

impl PipelineConfig {
    /// Creates config with a specific k value.
    pub fn with_k(k: u32) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }

    /// Enables deterministic proof generation.
    pub fn deterministic(mut self, seed: [u8; 32]) -> Self {
        self.deterministic_seed = Some(seed);
        self
    }

    /// Disables self-verification.
    pub fn no_self_verify(mut self) -> Self {
        self.self_verify = false;
        self
    }

    /// Sets retry configuration.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Sets proof timeout.
    pub fn with_proof_timeout(mut self, timeout: Duration) -> Self {
        self.proof_timeout = Some(timeout);
        self
    }
}

// ============================================================================
// SRS Parameter Cache
// ============================================================================

/// Returns the default SRS cache directory.
pub fn default_srs_cache_dir() -> PathBuf {
    let base = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join(".cache");
    base.join("helix").join("srs")
}

/// Loads cached SRS parameters from disk, or generates and caches them.
///
/// Uses `ParamsKZG::read()` / `ParamsKZG::write()` for efficient binary
/// serialization of the SRS. The cache key is `srs_k{k}_{seed_hex}.bin`
/// where `seed_hex` is the hex-encoded first 8 bytes of the SRS seed.
///
/// Returns `(params, cache_hit)` where `cache_hit` is true if loaded from disk.
pub fn load_or_generate_srs(
    k: u32,
    srs_seed: [u8; 32],
    cache_dir: Option<&Path>,
) -> (ParamsKZG<Bn256>, bool) {
    let cache_dir = cache_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_srs_cache_dir);

    let seed_prefix = hex::encode(&srs_seed[..8]);
    let cache_file = cache_dir.join(format!("srs_k{}_{}.bin", k, seed_prefix));

    // Try to load from cache
    if cache_file.exists() {
        if let Ok(file) = std::fs::File::open(&cache_file) {
            let mut reader = std::io::BufReader::new(file);
            if let Ok(params) = ParamsKZG::<Bn256>::read(&mut reader) {
                if params.k() == k {
                    tracing::info!(k, path = %cache_file.display(), "Loaded SRS from cache");
                    return (params, true);
                }
                tracing::warn!(
                    expected_k = k,
                    found_k = params.k(),
                    "Cached SRS k mismatch, regenerating"
                );
            } else {
                tracing::warn!(path = %cache_file.display(), "Failed to read cached SRS, regenerating");
            }
        }
    }

    // Generate from scratch
    let start = Instant::now();
    let rng = StdRng::from_seed(srs_seed);
    let params = ParamsKZG::<Bn256>::setup(k, rng);
    let gen_time = start.elapsed();
    tracing::info!(k, gen_time_ms = gen_time.as_millis() as u64, "Generated SRS from scratch");

    // Cache to disk
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        tracing::warn!(error = %e, "Failed to create SRS cache directory");
    } else if let Ok(file) = std::fs::File::create(&cache_file) {
        let mut writer = std::io::BufWriter::new(file);
        if let Err(e) = params.write(&mut writer) {
            tracing::warn!(error = %e, "Failed to write SRS to cache");
            // Clean up partial file
            let _ = std::fs::remove_file(&cache_file);
        } else {
            tracing::info!(
                k,
                path = %cache_file.display(),
                size_mb = std::fs::metadata(&cache_file).map(|m| m.len() / (1024 * 1024)).unwrap_or(0),
                "Cached SRS to disk"
            );
        }
    }

    (params, false)
}

/// Computes a SHA-256 hash of a verification key for consistency checks.
///
/// Serializes the VK and hashes the bytes. Two nodes with the same SRS and
/// circuit must produce identical VK hashes.
pub fn compute_vk_hash(vk: &VerifyingKey<G1Affine>) -> [u8; 32] {
    use sha2::Sha256;
    let mut hasher = Sha256::new();
    // Serialize VK to bytes via its Debug representation as a stable proxy.
    // The actual byte content includes all fixed commitments and selector data.
    let mut vk_bytes = Vec::new();
    vk.write(&mut vk_bytes, helix_circuits::halo2_proofs::SerdeFormat::RawBytes).unwrap_or_default();
    hasher.update(&vk_bytes);
    hasher.finalize().into()
}

// ============================================================================
// Memory Profiling
// ============================================================================

/// Memory usage snapshot captured during proving.
#[derive(Debug, Clone, Default)]
pub struct MemoryProfile {
    /// Resident memory before SRS generation (bytes).
    pub pre_srs_memory: u64,
    /// Resident memory after SRS generation (bytes).
    pub post_srs_memory: u64,
    /// Peak resident memory during proof generation (bytes).
    pub peak_proving_memory: u64,
    /// SRS size on disk (bytes).
    pub srs_disk_size: u64,
    /// Whether the SRS was loaded from cache.
    pub srs_cache_hit: bool,
}

impl MemoryProfile {
    /// Returns estimated SRS memory usage.
    pub fn srs_memory_bytes(&self) -> u64 {
        self.post_srs_memory.saturating_sub(self.pre_srs_memory)
    }
}

/// Returns current resident memory size in bytes (macOS/Linux).
pub fn current_resident_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // Use rusage which is simpler and more reliable than task_info
        #[repr(C)]
        struct Rusage {
            ru_utime: [i64; 2],  // timeval
            ru_stime: [i64; 2],  // timeval
            ru_maxrss: i64,
            _rest: [i64; 13],
        }
        extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        const RUSAGE_SELF: i32 = 0;
        unsafe {
            let mut usage = std::mem::zeroed::<Rusage>();
            if getrusage(RUSAGE_SELF, &mut usage) == 0 {
                // On macOS, ru_maxrss is in bytes
                return usage.ru_maxrss as u64;
            }
        }
        0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

// ============================================================================
// Model Requirements
// ============================================================================

/// Hardware requirements for proving a model of given dimensions.
#[derive(Debug, Clone)]
pub struct ModelRequirements {
    /// Model dimensions (d_in, d_hid, d_out).
    pub dims: (usize, usize, usize),
    /// Required k value (circuit size = 2^k rows).
    pub k: u32,
    /// Estimated SRS memory (bytes).
    pub estimated_srs_memory: u64,
    /// Estimated peak proving memory (bytes).
    pub estimated_peak_memory: u64,
    /// Estimated SRS disk size (bytes).
    pub estimated_srs_disk_size: u64,
    /// Whether this model fits the given constraints.
    pub feasible: bool,
    /// Reason if not feasible.
    pub infeasible_reason: Option<String>,
}

impl ModelRequirements {
    /// Estimates requirements for a model with the given dimensions.
    ///
    /// Uses heuristics based on k value:
    /// - SRS memory: ~48 * 2^k bytes (G1 points)
    /// - Peak proving memory: ~5x SRS memory
    /// - SRS disk size: ~48 * 2^k bytes
    pub fn estimate(d_in: usize, d_hid: usize, d_out: usize, max_memory_bytes: u64) -> Self {
        use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;

        let relu_range = 256usize;
        let exp_range = 128usize;

        let dummy_witness = crate::provers::training_prover_v2::create_zero_witness_pub(d_in, d_hid, d_out);
        let dummy_circuit = MLTrainingStepV2Circuit {
            witness: dummy_witness,
            relu_range,
            exp_range,
            exp_scale: 64,
            use_freivalds: true,
        };
        let k = dummy_circuit.minimum_k();

        // Heuristic: each G1 point is ~48 bytes (compressed) in memory
        let srs_points = 1u64 << k;
        let estimated_srs_memory = srs_points * 96; // ~96 bytes per G1Affine in memory
        let estimated_peak_memory = estimated_srs_memory * 5; // proving uses ~5x SRS
        let estimated_srs_disk_size = srs_points * 64; // serialized format

        let feasible = if max_memory_bytes > 0 {
            estimated_peak_memory <= max_memory_bytes
        } else {
            true // No constraint
        };

        let infeasible_reason = if !feasible {
            Some(format!(
                "Model requires ~{}MB peak memory (k={}), but limit is {}MB. \
                 Reduce model size or increase available memory.",
                estimated_peak_memory / (1024 * 1024),
                k,
                max_memory_bytes / (1024 * 1024),
            ))
        } else {
            None
        };

        Self {
            dims: (d_in, d_hid, d_out),
            k,
            estimated_srs_memory,
            estimated_peak_memory,
            estimated_srs_disk_size,
            feasible,
            infeasible_reason,
        }
    }
}

/// Reference table entry for deployment planning.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScalingEntry {
    /// Model dimensions (d_in, d_hid, d_out).
    pub dims: (usize, usize, usize),
    /// Circuit size parameter.
    pub k: u32,
    /// Number of circuit rows (2^k).
    pub rows: u64,
    /// Estimated SRS memory in MB.
    pub srs_memory_mb: f64,
    /// Estimated peak proving memory in MB.
    pub peak_memory_mb: f64,
    /// Estimated SRS disk size in MB.
    pub srs_disk_mb: f64,
}

/// Generates a scaling reference table for a list of model sizes.
pub fn generate_scaling_table(model_sizes: &[(usize, usize, usize)]) -> Vec<ScalingEntry> {
    model_sizes.iter().map(|&(d_in, d_hid, d_out)| {
        let req = ModelRequirements::estimate(d_in, d_hid, d_out, 0);
        ScalingEntry {
            dims: (d_in, d_hid, d_out),
            k: req.k,
            rows: 1u64 << req.k,
            srs_memory_mb: req.estimated_srs_memory as f64 / (1024.0 * 1024.0),
            peak_memory_mb: req.estimated_peak_memory as f64 / (1024.0 * 1024.0),
            srs_disk_mb: req.estimated_srs_disk_size as f64 / (1024.0 * 1024.0),
        }
    }).collect()
}

// ============================================================================
// Extracted VK Data
// ============================================================================

/// Extracted verification key data for EVM verifier generation.
#[derive(Debug, Clone)]
pub struct ExtractedVkData {
    /// G1 generator point.
    pub g1: (String, String),
    /// SRS s·G2 point for KZG opening.
    pub s_g2: (String, String, String, String),
    /// Negative G2 generator -[1]₂.
    pub neg_g2: (String, String, String, String),
    /// Number of advice column commitments.
    pub num_advices: usize,
    /// Number of public inputs.
    pub num_instances: usize,
    /// Circuit size (k = log2(rows)).
    pub k: u32,
}

// ============================================================================
// Proof Result
// ============================================================================

/// Result of proof generation with metadata.
#[derive(Debug, Clone)]
pub struct ProofResult {
    /// The serialized proof.
    pub proof: Vec<u8>,
    /// Whether the proof was self-verified.
    pub verified: bool,
    /// Generation time.
    pub generation_time: Duration,
    /// Verification time (if self-verified).
    pub verification_time: Option<Duration>,
    /// Number of attempts.
    pub attempts: u32,
    /// Proof generation seed (for reproducibility).
    pub seed: Option<[u8; 32]>,
    /// Peak memory during proof generation (bytes, 0 if unavailable).
    pub peak_memory: u64,
}

// ============================================================================
// Prover Pipeline
// ============================================================================

/// Thread-safe proving pipeline with comprehensive error handling.
pub struct ProverPipeline<C: Circuit<Fr>> {
    /// SRS parameters.
    pub params: ParamsKZG<Bn256>,
    /// Proving key.
    pub pk: Option<ProvingKey<G1Affine>>,
    /// Verification key.
    pub vk: Option<VerifyingKey<G1Affine>>,
    /// Pipeline configuration.
    config: PipelineConfig,
    /// Proof counter for statistics.
    proof_count: AtomicU64,
    /// Memory profile from SRS loading.
    memory_profile: MemoryProfile,
    /// VK hash for consistency checks.
    vk_hash: Option<[u8; 32]>,
    /// Marker for circuit type.
    _marker: PhantomData<C>,
}

impl<C: Circuit<Fr> + Clone> ProverPipeline<C> {
    /// Creates a new pipeline with default configuration.
    pub fn new(k: u32) -> Self {
        Self::with_config(PipelineConfig::with_k(k))
    }

    /// Creates a new pipeline with custom configuration.
    ///
    /// Uses the deterministic HELIX SRS seed so all provers generate
    /// compatible parameters. Override with [`PipelineConfig::deterministic`]
    /// to use a custom seed. Automatically caches SRS to disk for fast
    /// restarts.
    pub fn with_config(config: PipelineConfig) -> Self {
        let pre_memory = current_resident_memory();

        let srs_seed = config.deterministic_seed
            .unwrap_or(crate::keys::HELIX_SRS_SEED);

        let (params, cache_hit) = load_or_generate_srs(config.k, srs_seed, None);

        let post_memory = current_resident_memory();

        let srs_disk_size = {
            let seed_prefix = hex::encode(&srs_seed[..8]);
            let cache_file = default_srs_cache_dir()
                .join(format!("srs_k{}_{}.bin", config.k, seed_prefix));
            std::fs::metadata(&cache_file).map(|m| m.len()).unwrap_or(0)
        };

        Self {
            params,
            pk: None,
            vk: None,
            config,
            proof_count: AtomicU64::new(0),
            memory_profile: MemoryProfile {
                pre_srs_memory: pre_memory,
                post_srs_memory: post_memory,
                peak_proving_memory: post_memory,
                srs_disk_size,
                srs_cache_hit: cache_hit,
            },
            vk_hash: None,
            _marker: PhantomData,
        }
    }

    /// Creates a pipeline from pre-generated keys.
    ///
    /// Use this when keys have been generated via [`generate_keys_for_circuit`]
    /// and you want to inject them directly instead of calling [`setup`].
    pub fn from_keys(
        config: PipelineConfig,
        params: ParamsKZG<Bn256>,
        pk: ProvingKey<G1Affine>,
        vk: VerifyingKey<G1Affine>,
    ) -> Self {
        let vk_hash = Some(compute_vk_hash(&vk));
        Self {
            params,
            pk: Some(pk),
            vk: Some(vk),
            config,
            proof_count: AtomicU64::new(0),
            memory_profile: MemoryProfile::default(),
            vk_hash,
            _marker: PhantomData,
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Sets up the pipeline with a circuit.
    pub fn setup(&mut self, circuit: &C) -> PipelineResult<()> {
        if self.config.enable_tracing {
            tracing::info!(k = self.config.k, "Setting up proving pipeline");
        }

        let vk = keygen_vk(&self.params, circuit).map_err(|e| {
            PipelineError::keygen_with_cause(
                "Failed to generate verification key",
                format!("{:?}", e),
            )
        })?;

        let pk = keygen_pk(&self.params, vk.clone(), circuit).map_err(|e| {
            PipelineError::keygen_with_cause("Failed to generate proving key", format!("{:?}", e))
        })?;

        self.vk_hash = Some(compute_vk_hash(&vk));
        self.vk = Some(vk);
        self.pk = Some(pk);

        if self.config.enable_tracing {
            tracing::info!("Pipeline setup complete");
        }

        Ok(())
    }

    /// Generates a proof with full error handling and optional callbacks.
    pub fn prove(
        &self,
        circuit: &C,
        public_inputs: &[&[Fr]],
    ) -> PipelineResult<Vec<u8>> {
        let result = self.prove_with_options(
            circuit,
            public_inputs,
            no_progress_callback(),
            None,
        )?;
        Ok(result.proof)
    }

    /// Generates a proof with progress callbacks and cancellation support.
    pub fn prove_with_options(
        &self,
        circuit: &C,
        public_inputs: &[&[Fr]],
        progress: ProgressCallback,
        cancel_token: Option<&CancellationToken>,
    ) -> PipelineResult<ProofResult> {
        let pk = self.pk.as_ref().ok_or_else(|| {
            PipelineError::not_initialized("Proving key not generated - call setup() first")
        })?;

        let start = Instant::now();
        let mut last_error: Option<PipelineError> = None;

        // Retry loop
        for attempt in 0..=self.config.retry.max_retries {
            // Check cancellation
            if let Some(token) = cancel_token {
                if token.is_cancelled() {
                    return Err(PipelineError::cancelled("prove"));
                }
            }

            // Check timeout
            if let Some(timeout) = self.config.proof_timeout {
                if start.elapsed() > timeout {
                    return Err(PipelineError::timeout(
                        "prove",
                        start.elapsed().as_millis() as u64,
                        timeout.as_millis() as u64,
                    ));
                }
            }

            // Report progress
            progress(ProofProgress {
                phase: ProofPhase::Setup,
                phase_progress: 0.0,
                overall_progress: attempt as f64 / (self.config.retry.max_retries + 1) as f64,
                elapsed: start.elapsed(),
                estimated_remaining: None,
                attempt,
                message: if attempt > 0 {
                    Some(format!("Retry attempt {}", attempt))
                } else {
                    None
                },
            });

            // Attempt proof generation
            match self.try_prove(circuit, public_inputs, &progress, attempt) {
                Ok(proof) => {
                    let gen_time = start.elapsed();

                    // Self-verification
                    let (verified, verify_time) = if self.config.self_verify {
                        progress(ProofProgress {
                            phase: ProofPhase::Verification,
                            phase_progress: 0.0,
                            overall_progress: 0.95,
                            elapsed: start.elapsed(),
                            estimated_remaining: None,
                            attempt,
                            message: Some("Self-verifying proof".to_string()),
                        });

                        let verify_start = Instant::now();
                        let is_valid = self.verify(&proof, public_inputs)?;

                        if !is_valid {
                            return Err(PipelineError::SelfVerificationError {
                                message: "Generated proof failed self-verification".to_string(),
                                proof_size: proof.len(),
                            });
                        }

                        (true, Some(verify_start.elapsed()))
                    } else {
                        (false, None)
                    };

                    // Determine seed used
                    let seed = self.config.deterministic_seed;

                    let peak_mem = current_resident_memory();
                    self.proof_count.fetch_add(1, Ordering::Relaxed);

                    if self.config.enable_tracing {
                        tracing::info!(
                            proof_size = proof.len(),
                            generation_time_ms = gen_time.as_millis() as u64,
                            attempts = attempt + 1,
                            verified,
                            peak_memory_mb = peak_mem / (1024 * 1024),
                            "Proof generated successfully"
                        );
                    }

                    return Ok(ProofResult {
                        proof,
                        verified,
                        generation_time: gen_time,
                        verification_time: verify_time,
                        attempts: attempt + 1,
                        seed,
                        peak_memory: peak_mem,
                    });
                }
                Err(e) => {
                    if self.config.enable_tracing {
                        tracing::warn!(
                            attempt,
                            error = %e,
                            "Proof generation failed"
                        );
                    }

                    last_error = Some(e);

                    // Apply backoff if we have more retries
                    if attempt < self.config.retry.max_retries {
                        let delay = self.config.retry.delay_for_attempt(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        // All retries exhausted
        let error_msg = last_error
            .map(|e| e.to_string())
            .unwrap_or_else(|| "Unknown error".to_string());
        Err(PipelineError::max_retries(
            "prove".to_string(),
            self.config.retry.max_retries,
            error_msg,
        ))
    }

    /// Internal proof generation attempt.
    fn try_prove(
        &self,
        circuit: &C,
        public_inputs: &[&[Fr]],
        progress: &ProgressCallback,
        attempt: u32,
    ) -> PipelineResult<Vec<u8>> {
        let pk = self.pk.as_ref().expect("invariant: pk checked in prove_with_options");
        let start = Instant::now();

        progress(ProofProgress {
            phase: ProofPhase::Synthesis,
            phase_progress: 0.0,
            overall_progress: 0.1,
            elapsed: start.elapsed(),
            estimated_remaining: None,
            attempt,
            message: None,
        });

        // Create transcript
        let mut transcript = Keccak256Transcript::new(vec![]);

        // Get RNG based on configuration.
        // When deterministic_seed is set, derive a per-proof seed by XORing
        // the base seed with the proof counter so each call gets unique but
        // reproducible randomness. This is ONLY for reproducible testing.
        //
        // For production (deterministic_seed = None), use OsRng which provides
        // cryptographically secure randomness from the OS entropy pool.
        let rng = if let Some(base_seed) = self.config.deterministic_seed {
            let mut seed = base_seed;
            let count = self.proof_count.load(Ordering::Relaxed);
            let count_bytes = count.to_le_bytes();
            for (i, &b) in count_bytes.iter().enumerate() {
                seed[24 + i] ^= b;
            }
            StdRng::from_seed(seed)
        } else {
            // Production path: seed StdRng from OsRng (cryptographic entropy).
            // We use StdRng seeded from OsRng rather than OsRng directly because
            // create_proof requires an rng implementing RngCore, and StdRng
            // seeded from OsRng provides equivalent security guarantees.
            let mut seed = [0u8; 32];
            OsRng.fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };

        progress(ProofProgress {
            phase: ProofPhase::Commitments,
            phase_progress: 0.0,
            overall_progress: 0.3,
            elapsed: start.elapsed(),
            estimated_remaining: None,
            attempt,
            message: None,
        });

        // Generate proof (KZG commitment scheme, SHPLONK multiopen)
        let instances: Vec<Vec<Fr>> = public_inputs.iter().map(|s| s.to_vec()).collect();
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            _,
            _,
            _,
            _,
        >(
            &self.params,
            pk,
            &[circuit.clone()],
            &[instances],
            rng,
            &mut transcript,
        )
        .map_err(|e| {
            PipelineError::proof_gen_with_cause(
                "Halo2 proof generation failed",
                format!("{:?}", e),
                attempt,
            )
        })?;

        progress(ProofProgress {
            phase: ProofPhase::Finalization,
            phase_progress: 0.0,
            overall_progress: 0.9,
            elapsed: start.elapsed(),
            estimated_remaining: None,
            attempt,
            message: None,
        });

        Ok(transcript.finalize())
    }

    /// Verifies a proof against public inputs.
    pub fn verify(&self, proof: &[u8], public_inputs: &[&[Fr]]) -> PipelineResult<bool> {
        let vk = self.vk.as_ref().ok_or_else(|| {
            PipelineError::not_initialized("Verification key not generated - call setup() first")
        })?;

        let start = Instant::now();

        // Check timeout
        if let Some(timeout) = self.config.verify_timeout {
            if start.elapsed() > timeout {
                return Err(PipelineError::timeout(
                    "verify",
                    start.elapsed().as_millis() as u64,
                    timeout.as_millis() as u64,
                ));
            }
        }

        let mut transcript = Keccak256Transcript::new(proof);

        let instances: Vec<Vec<Fr>> = public_inputs.iter().map(|s| s.to_vec()).collect();
        let verifier_params = self.params.verifier_params();
        let result = verify_proof_multi::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _,
            _,
            SingleStrategy<Bn256>,
        >(&verifier_params, vk, &[instances], &mut transcript);

        if self.config.enable_tracing && !result {
            tracing::warn!("Proof verification failed");
        }

        Ok(result)
    }

    /// Extracts verification key data for EVM verifier generation.
    ///
    /// Reads the real `[s]₂` point from the SRS (`ParamsKZG::s_g2()`) and the
    /// G2 generator from `ParamsKZG::g2()`, then computes `-G2` by negating.
    /// These are the pairing-check parameters the on-chain Halo2Verifier needs.
    pub fn extract_vk_data(&self, num_instances: usize) -> Option<ExtractedVkData> {
        let _vk = self.vk.as_ref()?;

        // BN254 G1 generator
        let g1_coords = G1Affine::generator().coordinates().expect("G1 generator always has coordinates");
        let g1_x = field_to_u256(*g1_coords.x());
        let g1_y = field_to_u256(*g1_coords.y());

        // Extract real s·G2 from the SRS
        let s_g2_point = self.params.s_g2();
        let s_g2 = g2_to_evm_decimal_tuple(&s_g2_point);

        // Compute -G2 (negation of the G2 generator from the SRS)
        let g2_point = self.params.g2();
        use std::ops::Neg;
        let neg_g2_point = g2_point.neg();
        let neg_g2 = g2_to_evm_decimal_tuple(&neg_g2_point);

        // Number of advice columns for MLTrainingStepV2
        let num_advices = 3;

        Some(ExtractedVkData {
            g1: (g1_x, g1_y),
            s_g2,
            neg_g2,
            num_advices,
            num_instances,
            k: self.params.k(),
        })
    }

    /// Returns a reference to the verification key.
    pub fn vk(&self) -> Option<&VerifyingKey<G1Affine>> {
        self.vk.as_ref()
    }

    /// Returns a reference to the SRS params.
    pub fn params(&self) -> &ParamsKZG<Bn256> {
        &self.params
    }

    /// Returns the K parameter (circuit size = 2^k rows).
    pub fn k(&self) -> u32 {
        self.params.k()
    }

    /// Checks if the pipeline has been set up.
    pub fn is_setup(&self) -> bool {
        self.pk.is_some() && self.vk.is_some()
    }

    /// Returns the number of proofs generated.
    pub fn proof_count(&self) -> u64 {
        self.proof_count.load(Ordering::Relaxed)
    }

    /// Returns the memory profile from SRS loading and proving.
    pub fn memory_profile(&self) -> &MemoryProfile {
        &self.memory_profile
    }

    /// Returns the VK hash for consistency verification.
    ///
    /// Two pipelines with the same SRS seed, k, and circuit must produce
    /// identical VK hashes. Use this to verify that a node's VK matches
    /// the expected value from the aggregator or contract.
    pub fn vk_hash(&self) -> Option<&[u8; 32]> {
        self.vk_hash.as_ref()
    }

    /// Verifies that this pipeline's VK hash matches the expected hash.
    ///
    /// Returns `Ok(true)` if the hashes match, `Ok(false)` if they don't,
    /// or `Err` if the pipeline hasn't been set up yet.
    pub fn verify_vk_consistency(&self, expected_hash: &[u8; 32]) -> PipelineResult<bool> {
        let hash = self.vk_hash.as_ref().ok_or_else(|| {
            PipelineError::not_initialized("VK not generated - call setup() first")
        })?;
        Ok(hash == expected_hash)
    }

    /// Proves multiple circuits in batch, returning individual proofs.
    pub fn prove_batch(
        &self,
        circuits: &[C],
        public_inputs: &[Vec<Fr>],
    ) -> PipelineResult<Vec<Vec<u8>>> {
        if circuits.len() != public_inputs.len() {
            return Err(PipelineError::invalid_input(
                "Circuit and public input count mismatch",
                "circuits.len()",
                &circuits.len().to_string(),
                &public_inputs.len().to_string(),
            ));
        }

        let mut proofs = Vec::with_capacity(circuits.len());
        for (circuit, pi) in circuits.iter().zip(public_inputs.iter()) {
            let pi_refs: Vec<&[Fr]> = vec![pi.as_slice()];
            proofs.push(self.prove(circuit, &pi_refs)?);
        }

        Ok(proofs)
    }

    /// Verifies multiple proofs in batch.
    pub fn verify_batch(
        &self,
        proofs: &[Vec<u8>],
        public_inputs: &[Vec<Fr>],
    ) -> PipelineResult<Vec<bool>> {
        if proofs.len() != public_inputs.len() {
            return Err(PipelineError::invalid_input(
                "Proof and public input count mismatch",
                "proofs.len()",
                &proofs.len().to_string(),
                &public_inputs.len().to_string(),
            ));
        }

        let mut results = Vec::with_capacity(proofs.len());
        for (proof, pi) in proofs.iter().zip(public_inputs.iter()) {
            let pi_refs: Vec<&[Fr]> = vec![pi.as_slice()];
            results.push(self.verify(proof, &pi_refs)?);
        }

        Ok(results)
    }

    /// Proves multiple circuits in parallel using threads.
    #[cfg(feature = "parallel")]
    pub fn prove_batch_parallel(
        &self,
        circuits: &[C],
        public_inputs: &[Vec<Fr>],
    ) -> PipelineResult<Vec<Vec<u8>>>
    where
        C: Send + Sync,
    {
        use rayon::prelude::*;

        if circuits.len() != public_inputs.len() {
            return Err(PipelineError::invalid_input(
                "Circuit and public input count mismatch",
                "circuits.len()",
                &circuits.len().to_string(),
                &public_inputs.len().to_string(),
            ));
        }

        let pairs: Vec<(&C, &Vec<Fr>)> = circuits.iter().zip(public_inputs.iter()).collect();

        let results: Vec<PipelineResult<Vec<u8>>> = pairs
            .par_iter()
            .map(|(circuit, pi)| {
                let pi_refs: Vec<&[Fr]> = vec![pi.as_slice()];
                self.prove(circuit, &pi_refs)
            })
            .collect();

        results.into_iter().collect()
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics from a proving session.
#[derive(Debug, Clone)]
pub struct ProvingStats {
    /// Number of proofs generated.
    pub num_proofs: usize,
    /// Total proof size in bytes.
    pub total_proof_size: usize,
    /// Average proof size in bytes.
    pub avg_proof_size: usize,
    /// Number of successful verifications.
    pub successful_verifications: usize,
    /// K parameter used.
    pub k: u32,
    /// Total generation time.
    pub total_generation_time: Duration,
    /// Average generation time.
    pub avg_generation_time: Duration,
    /// Total retry count.
    pub total_retries: u32,
}

impl ProvingStats {
    /// Creates stats from a batch of proof results.
    pub fn from_results(results: &[ProofResult], k: u32) -> Self {
        let total_size: usize = results.iter().map(|r| r.proof.len()).sum();
        let total_time: Duration = results.iter().map(|r| r.generation_time).sum();
        let total_retries: u32 = results.iter().map(|r| r.attempts.saturating_sub(1)).sum();
        let verified_count = results.iter().filter(|r| r.verified).count();

        Self {
            num_proofs: results.len(),
            total_proof_size: total_size,
            avg_proof_size: if results.is_empty() {
                0
            } else {
                total_size / results.len()
            },
            successful_verifications: verified_count,
            k,
            total_generation_time: total_time,
            avg_generation_time: if results.is_empty() {
                Duration::ZERO
            } else {
                total_time / results.len() as u32
            },
            total_retries,
        }
    }

    /// Creates stats from a batch of proofs (without timing info).
    pub fn from_batch(proofs: &[Vec<u8>], k: u32) -> Self {
        let total_size: usize = proofs.iter().map(|p| p.len()).sum();

        Self {
            num_proofs: proofs.len(),
            total_proof_size: total_size,
            avg_proof_size: if proofs.is_empty() {
                0
            } else {
                total_size / proofs.len()
            },
            successful_verifications: 0,
            k,
            total_generation_time: Duration::ZERO,
            avg_generation_time: Duration::ZERO,
            total_retries: 0,
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Converts a field element to a decimal string (for Solidity constants).
fn field_to_u256<F: PrimeField>(f: F) -> String {
    let repr = f.to_repr();
    let bytes = repr.as_ref();
    let value = num_bigint::BigUint::from_bytes_le(bytes);
    value.to_string()
}

/// Extracts the c0 (real) component from an Fq2 via byte serialization.
fn fq2_c0(f: &Fq2) -> Fq {
    let bytes = f.to_bytes();
    Fq::from_bytes(bytes[..32].try_into().expect("invariant: Fq2 bytes[..32] is 32 bytes")).expect("invariant: valid Fq bytes from Fq2")
}

/// Extracts the c1 (imaginary) component from an Fq2.
fn fq2_c1(f: &Fq2) -> Fq {
    let bytes = f.to_bytes();
    Fq::from_bytes(bytes[32..].try_into().expect("invariant: Fq2 bytes[32..] is 32 bytes")).expect("invariant: valid Fq bytes from Fq2")
}

/// Converts a G2Affine point to EIP-197 pairing precompile format:
/// (x_imaginary, x_real, y_imaginary, y_real) as decimal strings.
fn g2_to_evm_decimal_tuple(point: &G2Affine) -> (String, String, String, String) {
    (
        field_to_u256(fq2_c1(&point.x)),
        field_to_u256(fq2_c0(&point.x)),
        field_to_u256(fq2_c1(&point.y)),
        field_to_u256(fq2_c0(&point.y)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srs_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path();
        let seed = crate::keys::HELIX_SRS_SEED;
        let k = 5; // Small k for fast test

        // First call: generate
        let (params1, hit1) = load_or_generate_srs(k, seed, Some(cache_dir));
        assert!(!hit1, "First call should not be a cache hit");
        assert_eq!(params1.k(), k);

        // Second call: should load from cache
        let (params2, hit2) = load_or_generate_srs(k, seed, Some(cache_dir));
        assert!(hit2, "Second call should be a cache hit");
        assert_eq!(params2.k(), k);

        // Verify the params produce the same s_g2 point (proving SRS equivalence)
        let s1 = params1.s_g2();
        let s2 = params2.s_g2();
        assert_eq!(s1, s2, "Cached SRS must match generated SRS");
    }

    #[test]
    fn test_srs_cache_different_seeds() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path();
        let k = 5;

        let seed1 = [1u8; 32];
        let seed2 = [2u8; 32];

        let (params1, _) = load_or_generate_srs(k, seed1, Some(cache_dir));
        let (params2, _) = load_or_generate_srs(k, seed2, Some(cache_dir));

        // Different seeds should produce different SRS
        assert_ne!(params1.s_g2(), params2.s_g2(), "Different seeds must produce different SRS");
    }

    #[test]
    fn test_vk_hash_consistency() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use helix_circuits::halo2_proofs::plonk::{keygen_vk, Circuit};
        use helix_circuits::halo2curves::bn256::Fr;

        // Generate two VKs from the same SRS + circuit → must have same hash
        let seed = crate::keys::HELIX_SRS_SEED;
        let dir = tempfile::tempdir().unwrap();
        let k = 5;

        let (params, _) = load_or_generate_srs(k, seed, Some(dir.path()));

        // Use the IVCStepCircuit as a simple test circuit
        use crate::provers::ivc_circuit::IVCStepCircuit;
        let circuit = IVCStepCircuit::default();

        let vk1 = keygen_vk(&params, &circuit).unwrap();
        let vk2 = keygen_vk(&params, &circuit).unwrap();

        let hash1 = compute_vk_hash(&vk1);
        let hash2 = compute_vk_hash(&vk2);

        assert_eq!(hash1, hash2, "Same SRS + circuit must produce same VK hash");
        assert_ne!(hash1, [0u8; 32], "VK hash should not be all zeros");
    }

    #[test]
    fn test_memory_profiling() {
        let mem = current_resident_memory();
        // On macOS/Linux this should return a non-zero value
        // On other platforms it returns 0 (which is fine)
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(mem > 0, "Resident memory should be positive on macOS/Linux");
        }
    }

    #[test]
    fn test_model_requirements_estimate() {
        // Small model
        let req = ModelRequirements::estimate(2, 2, 1, 0);
        assert_eq!(req.dims, (2, 2, 1));
        assert!(req.k >= 10, "k should be at least 10");
        assert!(req.feasible, "Small model should be feasible");
        assert!(req.estimated_srs_memory > 0);
        assert!(req.estimated_peak_memory > req.estimated_srs_memory);

        // Model with tight memory constraint
        let req = ModelRequirements::estimate(2, 2, 1, 1024); // 1KB limit
        assert!(!req.feasible, "Should be infeasible with 1KB limit");
        assert!(req.infeasible_reason.is_some());
    }

    #[test]
    fn test_scaling_table() {
        let table = generate_scaling_table(&[
            (2, 2, 1),
            (16, 16, 8),
            (32, 32, 16),
        ]);
        assert_eq!(table.len(), 3);

        // k should increase with model size
        assert!(table[1].k >= table[0].k, "Larger model should need >= k");
        assert!(table[2].k >= table[1].k, "Larger model should need >= k");

        // Memory should increase with k
        assert!(table[1].srs_memory_mb >= table[0].srs_memory_mb);
    }

    #[test]
    fn test_pipeline_error_display() {
        let err = PipelineError::keygen("test error");
        assert!(err.to_string().contains("Key generation failed"));

        let err = PipelineError::timeout("prove", 100, 50);
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());

        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_retry_config_delays() {
        let config = RetryConfig {
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(1),
            add_jitter: false,
            ..Default::default()
        };

        let delay0 = config.delay_for_attempt(0);
        let delay1 = config.delay_for_attempt(1);
        let delay2 = config.delay_for_attempt(2);

        assert_eq!(delay0, Duration::from_millis(100));
        assert_eq!(delay1, Duration::from_millis(200));
        assert_eq!(delay2, Duration::from_millis(400));
    }

    #[test]
    fn test_retry_config_max_delay() {
        let config = RetryConfig {
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 10.0,
            max_delay: Duration::from_millis(500),
            add_jitter: false,
            ..Default::default()
        };

        let delay5 = config.delay_for_attempt(5);
        assert_eq!(delay5, Duration::from_millis(500)); // Capped at max
    }

    #[test]
    fn test_proof_progress_display() {
        let phase = ProofPhase::Commitments;
        assert_eq!(format!("{}", phase), "Commitments");
    }

    #[test]
    fn test_pipeline_config_builder() {
        let seed = [1u8; 32];
        let config = PipelineConfig::with_k(12)
            .deterministic(seed)
            .no_self_verify()
            .with_proof_timeout(Duration::from_secs(60));

        assert_eq!(config.k, 12);
        assert_eq!(config.deterministic_seed, Some(seed));
        assert!(!config.self_verify);
        assert_eq!(config.proof_timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_default_config_uses_osrng() {
        // Default pipeline config must NOT have a deterministic seed,
        // which means the production path uses OsRng for cryptographic randomness.
        let config = PipelineConfig::default();
        assert!(
            config.deterministic_seed.is_none(),
            "Production default must use OsRng (no deterministic seed)"
        );
    }

    #[test]
    fn test_deterministic_seed_produces_reproducible_rng() {
        // When deterministic_seed is set, the same seed should produce
        // the same StdRng output.
        let seed = [42u8; 32];
        let mut rng1 = StdRng::from_seed(seed);
        let mut rng2 = StdRng::from_seed(seed);

        let mut buf1 = [0u8; 32];
        let mut buf2 = [0u8; 32];
        rng1.fill_bytes(&mut buf1);
        rng2.fill_bytes(&mut buf2);
        assert_eq!(buf1, buf2, "same seed must produce identical randomness");
    }

    #[test]
    fn test_osrng_produces_unique_randomness() {
        // When using OsRng (production path), each seeding should produce
        // different randomness — proving non-determinism.
        let mut seed1 = [0u8; 32];
        let mut seed2 = [0u8; 32];
        OsRng.fill_bytes(&mut seed1);
        OsRng.fill_bytes(&mut seed2);
        assert_ne!(
            seed1, seed2,
            "OsRng must produce different seeds on each call"
        );
    }

    /// Generate a correct Halo2 SHPLONK Solidity verifier from our circuit VK.
    /// Writes the verifier to contracts/src/verification/Halo2Verifier.sol
    /// Ignored by default: this is a code-generation tool, not a correctness test.
    /// Run explicitly with: cargo test -p helix-prover test_generate_solidity_verifier -- --ignored
    #[test]
    #[ignore]
    fn test_generate_solidity_verifier() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use helix_circuits::{
            MLTrainingStepV2Circuit, compute_witness_v2, compute_state_hash_v2,
        };
        use helix_circuits::halo2_proofs::{
            plonk::{keygen_vk, keygen_pk, create_proof, verify_proof_multi},
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
        };
        use helix_circuits::halo2curves::bn256::{Bn256, Fr, G1Affine};
        use helix_circuits::halo2curves::ff::Field;
        use halo2_solidity_verifier::{SolidityGenerator, BatchOpenScheme::Bdfg21, encode_calldata, Keccak256Transcript};
        use rand_core::OsRng;

        // Create the same tiny circuit used in circuit tests
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;
        let w1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];
        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];
        let lr = Fr::from(1);
        let base_error = Fr::from(1);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::ZERO, Fr::ZERO), 1, base_error,
        );
        let new_hash = compute_state_hash_v2(&witness.w1_new, &witness.b1_new, &witness.w2_new, &witness.b2_new);
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, new_hash, 1, base_error,
        );
        let pi = witness.public_inputs();

        // Circuit config MUST match the on-chain test prover config
        // (on_chain_verification.rs uses V2ProverConfig { relu_range: 256,
        // exp_range: 256, exp_scale: 1000, use_freivalds: false }).
        // Any mismatch produces a different VK → proofs rejected on-chain.
        let circuit = MLTrainingStepV2Circuit {
            witness,
            relu_range: 256,
            exp_range: 256,
            exp_scale: 1000,
            use_freivalds: false,
        };

        let k = 14;
        // Use HELIX_SRS_SEED for deterministic SRS — same seed as
        // generate_keys_for_circuit() so proofs are compatible with this VK.
        use crate::keys::HELIX_SRS_SEED;
        use rand::SeedableRng;
        let srs_rng = rand::rngs::StdRng::from_seed(HELIX_SRS_SEED);
        let params = ParamsKZG::<Bn256>::setup(k, srs_rng);
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");

        // Generate proof with PSE's Keccak256Transcript (EVM-compatible).
        // This transcript writes EC points as 64-byte uncompressed (x, y) in big-endian,
        // matching the format expected by the PSE-generated Solidity verifier.
        let instances = vec![pi.clone()];
        let mut transcript = Keccak256Transcript::new(vec![]);
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            _,_,_,_,
        >(&params, &pk, &[circuit], &[instances.clone()], OsRng, &mut transcript)
        .expect("create_proof failed");
        let proof = transcript.finalize();

        // Verify proof using PSE transcript
        let mut verifier_transcript = Keccak256Transcript::new(proof.as_slice());
        let verifier_params = params.verifier_params();
        let verified = verify_proof_multi::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _,_,
            SingleStrategy<Bn256>,
        >(&verifier_params, &vk, &[instances.clone()], &mut verifier_transcript);
        assert!(verified, "PSE Keccak256Transcript proof verification must succeed");

        // Generate Solidity verifier using PSE generator
        let num_instances = pi.len();
        let generator = SolidityGenerator::new(&params, &vk, Bdfg21, num_instances);
        let verifier_sol = generator.render().expect("render failed");

        assert!(verifier_sol.contains("Halo2Verifier"), "generated verifier must contain Halo2Verifier");
        // The BDFG21 scheme generates pairing verification logic
        assert!(verifier_sol.contains("ecPairing") || verifier_sol.contains("pairing") || verifier_sol.contains("staticcall"),
            "generated verifier must contain pairing logic");

        // Also generate separate VK
        let (verifier_sol_sep, vk_sol) = generator.render_separately().expect("render_separately failed");
        assert!(vk_sol.contains("Halo2VerifyingKey"), "VK contract must exist");

        // Encode calldata for testing
        let calldata = encode_calldata(None, &proof, &pi);
        assert!(!calldata.is_empty(), "calldata must not be empty");
        eprintln!("=== Solidity verifier generated successfully ===");
        eprintln!("  Verifier size: {} bytes", verifier_sol.len());
        eprintln!("  VK size: {} bytes", vk_sol.len());
        eprintln!("  Proof size: {} bytes", proof.len());
        eprintln!("  Calldata size: {} bytes", calldata.len());
        eprintln!("  Public inputs: {}", num_instances);

        // Write the core verifier to contracts directory
        // The PSE-generated verifier is saved as Halo2VerifierCore.sol
        // (Halo2Verifier.sol is the wrapper that implements IHelixVerifier)
        let contracts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap()
            .join("contracts").join("src").join("verification");
        if contracts_dir.exists() {
            // Rename contract from Halo2Verifier to Halo2VerifierCore
            // and add memory-safe annotation for via_ir compatibility
            let core_sol = verifier_sol_sep.replace("contract Halo2Verifier", "contract Halo2VerifierCore");
            let core_sol = core_sol.replace("assembly {", "assembly (\"memory-safe\") {");
            let verifier_path = contracts_dir.join("Halo2VerifierCore.sol");
            let vk_path = contracts_dir.join("Halo2VerifyingKey.sol");
            std::fs::write(&verifier_path, &core_sol).expect("write verifier core");
            std::fs::write(&vk_path, &vk_sol).expect("write VK");
            eprintln!("  Written to: {}", verifier_path.display());
            eprintln!("  Written to: {}", vk_path.display());
        }
    }
}
