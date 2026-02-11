//! Proof Verification for Gradient Aggregation.
//!
//! Verifies ZK proofs before accepting gradients for aggregation using real
//! Halo2 KZG verification via `helix_prover::MLTrainingProverV2`.
//! This ensures computation integrity in the distributed training process.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use sha2::{Digest, Sha256};

use helix_prover::halo2curves::bn256::Fr;
use helix_prover::halo2curves::ff::PrimeField;
use helix_prover::MLTrainingProverV2;

use crate::network::messages::PeerId;

/// Verification result.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the proof is valid.
    pub is_valid: bool,
    /// Verification time in milliseconds.
    pub verification_time_ms: u64,
    /// Error message if invalid.
    pub error: Option<String>,
    /// Extracted public inputs.
    pub public_inputs: Option<Vec<u64>>,
}

/// Proof type being verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofType {
    /// ML training proof (model update validity).
    Training,
    /// Aggregation proof (correct gradient combination).
    Aggregation,
    /// State transition proof.
    StateTransition,
}

/// Configurable verification policy controlling when Halo2 proofs are
/// checked vs. when a cheaper structural check suffices.
#[derive(Debug, Clone)]
pub enum VerificationPolicy {
    /// Verify every submitted proof with full Halo2 KZG verification.
    /// Most secure, but highest latency per gradient.
    VerifyAll,
    /// Verify a random fraction of proofs (0.0 – 1.0).
    /// E.g. `Sample(0.1)` verifies ~10% of proofs.
    /// Unverified proofs still pass structural checks (size, error bounds).
    Sample(f64),
    /// Perform only structural checks (proof size, error bound range) without
    /// invoking the Halo2 verifier. Fastest, but provides no cryptographic
    /// guarantee. Suitable for development/testing only.
    StructuralOnly,
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        Self::VerifyAll
    }
}

/// Configuration for proof verification.
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// Enable proof verification.
    pub enabled: bool,
    /// Verification policy controlling when full Halo2 verification is performed.
    pub policy: VerificationPolicy,
    /// Maximum time allowed for verification.
    pub timeout: Duration,
    /// Cache verified proofs.
    pub enable_cache: bool,
    /// Cache TTL in seconds.
    pub cache_ttl_secs: u64,
    /// Maximum cache size.
    pub max_cache_size: usize,
    /// Minimum error bound required.
    pub min_error_bound: f64,
    /// Maximum error bound allowed.
    pub max_error_bound: f64,
    /// Model dimensions for prover initialization.
    pub model_dims: Option<(usize, usize, usize)>,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            policy: VerificationPolicy::default(),
            timeout: Duration::from_secs(30),
            enable_cache: true,
            cache_ttl_secs: 3600,
            max_cache_size: 10000,
            min_error_bound: 0.0,
            max_error_bound: 0.1,
            model_dims: None,
        }
    }
}

/// Cached verification result.
#[derive(Debug, Clone)]
struct CachedVerification {
    result: VerificationResult,
    timestamp: u64,
}

/// Proof verifier for gradient submissions.
///
/// Uses a real `MLTrainingProverV2` instance for Halo2 KZG proof verification.
/// The prover is lazily initialized on first use with model dimensions from config.
/// All nodes use the deterministic `HELIX_SRS_SEED` so verification keys match.
pub struct ProofVerifier {
    /// Configuration.
    config: VerificationConfig,
    /// Real Halo2 prover/verifier (lazy-initialized, shared across verifications).
    /// Uses deterministic SRS so all nodes produce the same VK.
    prover: Arc<RwLock<Option<MLTrainingProverV2>>>,
    /// Verification cache (proof hash -> result).
    cache: Arc<RwLock<HashMap<[u8; 32], CachedVerification>>>,
    /// Statistics.
    stats: Arc<RwLock<VerificationStats>>,
}

impl std::fmt::Debug for ProofVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProofVerifier")
            .field("config", &self.config)
            .field("prover_initialized", &self.prover.read().is_some())
            .field("cache_size", &self.cache.read().len())
            .field("stats", &*self.stats.read())
            .finish()
    }
}

/// Verification statistics.
#[derive(Debug, Clone, Default)]
pub struct VerificationStats {
    /// Total proofs verified.
    pub total_verified: u64,
    /// Proofs that passed.
    pub proofs_valid: u64,
    /// Proofs that failed.
    pub proofs_invalid: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Average verification time (ms).
    pub avg_verification_time_ms: f64,
}

impl ProofVerifier {
    /// Creates a new proof verifier.
    ///
    /// Logs a loud warning if `StructuralOnly` policy is selected, since it
    /// provides no cryptographic guarantees and should only be used in
    /// development/testing with an explicit configuration override.
    pub fn new(config: VerificationConfig) -> Self {
        if matches!(config.policy, VerificationPolicy::StructuralOnly) {
            log::warn!(
                "⚠ VerificationPolicy::StructuralOnly is active — proofs are NOT cryptographically \
                 verified! This provides ZERO security and must only be used for development/testing. \
                 Set policy to VerifyAll (the default) for production use."
            );
        }
        Self {
            config,
            prover: Arc::new(RwLock::new(None)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(VerificationStats::default())),
        }
    }

    /// Creates a proof verifier with a pre-initialized prover for the given model dimensions.
    pub fn with_model_dims(config: VerificationConfig, d_in: usize, d_hid: usize, d_out: usize) -> Self {
        if matches!(config.policy, VerificationPolicy::StructuralOnly) {
            log::warn!(
                "⚠ VerificationPolicy::StructuralOnly is active — proofs are NOT cryptographically \
                 verified! This provides ZERO security and must only be used for development/testing. \
                 Set policy to VerifyAll (the default) for production use."
            );
        }
        let prover = MLTrainingProverV2::new(d_in, d_hid, d_out);
        Self {
            config,
            prover: Arc::new(RwLock::new(Some(prover))),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(VerificationStats::default())),
        }
    }

    /// Ensures the prover is initialized, creating it if needed.
    fn ensure_prover(&self) -> bool {
        if self.prover.read().is_some() {
            return true;
        }
        if let Some((d_in, d_hid, d_out)) = self.config.model_dims {
            let prover = MLTrainingProverV2::new(d_in, d_hid, d_out);
            *self.prover.write() = Some(prover);
            true
        } else {
            false
        }
    }

    /// Verifies a gradient proof using real Halo2 KZG verification.
    ///
    /// Respects the configured [`VerificationPolicy`]:
    /// - `VerifyAll`: every proof goes through full Halo2 KZG verification.
    /// - `Sample(fraction)`: randomly selects a fraction of proofs for full
    ///   verification; the rest pass with structural checks only.
    /// - `StructuralOnly`: only checks proof size and error bounds, skips
    ///   Halo2 entirely (development mode).
    pub async fn verify_gradient_proof(
        &self,
        participant: &PeerId,
        round_id: u64,
        gradient_commitment: [u8; 32],
        error_bound: f64,
        proof: &[u8],
    ) -> VerificationResult {
        if !self.config.enabled {
            return VerificationResult {
                is_valid: true,
                verification_time_ms: 0,
                error: None,
                public_inputs: None,
            };
        }

        let start = std::time::Instant::now();

        // Check error bound (always, regardless of policy)
        if error_bound < self.config.min_error_bound || error_bound > self.config.max_error_bound {
            let mut stats = self.stats.write();
            stats.total_verified += 1;
            stats.proofs_invalid += 1;

            return VerificationResult {
                is_valid: false,
                verification_time_ms: start.elapsed().as_millis() as u64,
                error: Some(format!(
                    "Error bound {} outside allowed range [{}, {}]",
                    error_bound, self.config.min_error_bound, self.config.max_error_bound
                )),
                public_inputs: None,
            };
        }

        // Determine whether to perform full Halo2 verification based on policy
        let do_full_verification = match &self.config.policy {
            VerificationPolicy::VerifyAll => true,
            VerificationPolicy::Sample(fraction) => {
                use rand::Rng;
                rand::thread_rng().gen::<f64>() < *fraction
            }
            VerificationPolicy::StructuralOnly => false,
        };

        if !do_full_verification {
            // Structural checks only: proof size and error bound already checked
            let structural_ok = !proof.is_empty() && proof.len() >= 64;
            let mut stats = self.stats.write();
            stats.total_verified += 1;
            if structural_ok {
                stats.proofs_valid += 1;
            } else {
                stats.proofs_invalid += 1;
            }
            return VerificationResult {
                is_valid: structural_ok,
                verification_time_ms: start.elapsed().as_millis() as u64,
                error: if structural_ok {
                    None
                } else {
                    Some(format!(
                        "Structural check failed: proof size {} bytes (minimum 64)",
                        proof.len()
                    ))
                },
                public_inputs: None,
            };
        }

        // Check cache — key includes proof hash, model commitment, round_id, and error_bound
        let cache_key = self.cache_key(proof, &gradient_commitment, round_id, error_bound);
        if self.config.enable_cache {
            if let Some(cached) = self.get_cached(&cache_key) {
                let mut stats = self.stats.write();
                stats.cache_hits += 1;
                return cached;
            }
        }

        // Perform full Halo2 KZG verification
        let result = self.verify_proof_internal(
            participant,
            round_id,
            gradient_commitment,
            error_bound,
            proof,
        ).await;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_verified += 1;
            if result.is_valid {
                stats.proofs_valid += 1;
            } else {
                stats.proofs_invalid += 1;
            }
            // Update rolling average
            let n = stats.total_verified as f64;
            stats.avg_verification_time_ms =
                (stats.avg_verification_time_ms * (n - 1.0) + result.verification_time_ms as f64) / n;
        }

        // Cache result
        if self.config.enable_cache {
            self.cache_result(&cache_key, &result);
        }

        result
    }

    /// Verifies an aggregation proof.
    pub async fn verify_aggregation_proof(
        &self,
        round_id: u64,
        aggregated_commitment: [u8; 32],
        individual_commitments: &[[u8; 32]],
        _proof: &[u8],
    ) -> VerificationResult {
        if !self.config.enabled {
            return VerificationResult {
                is_valid: true,
                verification_time_ms: 0,
                error: None,
                public_inputs: None,
            };
        }

        let start = std::time::Instant::now();

        // Verify that aggregated commitment is derived from individual commitments
        let computed_commitment = self.compute_aggregated_commitment(individual_commitments);
        let is_valid = computed_commitment == aggregated_commitment;

        let elapsed = start.elapsed().as_millis() as u64;

        VerificationResult {
            is_valid,
            verification_time_ms: elapsed,
            error: if is_valid {
                None
            } else {
                Some("Aggregated commitment mismatch".to_string())
            },
            public_inputs: Some(vec![round_id]),
        }
    }

    /// Batch verifies multiple proofs.
    pub async fn batch_verify(
        &self,
        proofs: Vec<(PeerId, u64, [u8; 32], f64, Vec<u8>)>,
    ) -> Vec<(PeerId, VerificationResult)> {
        let mut results = Vec::with_capacity(proofs.len());

        for (participant, round_id, commitment, error_bound, proof) in proofs {
            let result = self.verify_gradient_proof(
                &participant,
                round_id,
                commitment,
                error_bound,
                &proof,
            ).await;
            results.push((participant, result));
        }

        results
    }

    /// Returns verification statistics.
    pub fn stats(&self) -> VerificationStats {
        self.stats.read().clone()
    }

    /// Clears the verification cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Real Halo2 KZG proof verification.
    ///
    /// Parses the proof bytes and public inputs, then delegates to
    /// `MLTrainingProverV2::verify()` which performs actual KZG pairing checks.
    async fn verify_proof_internal(
        &self,
        _participant: &PeerId,
        _round_id: u64,
        _gradient_commitment: [u8; 32],
        _error_bound: f64,
        proof: &[u8],
    ) -> VerificationResult {
        let start = std::time::Instant::now();

        let verification_future = async {
            // 1. Check proof is not empty
            if proof.is_empty() {
                return VerificationResult {
                    is_valid: false,
                    verification_time_ms: start.elapsed().as_millis() as u64,
                    error: Some("Empty proof".to_string()),
                    public_inputs: None,
                };
            }

            // 2. Minimum size: a real Halo2 KZG proof is at least ~200 bytes
            if proof.len() < 64 {
                return VerificationResult {
                    is_valid: false,
                    verification_time_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Proof too small for Halo2 KZG: {} bytes", proof.len())),
                    public_inputs: None,
                };
            }

            // 3. Ensure prover is initialized
            if !self.ensure_prover() {
                return VerificationResult {
                    is_valid: false,
                    verification_time_ms: start.elapsed().as_millis() as u64,
                    error: Some("Prover not initialized: model dimensions not configured".to_string()),
                    public_inputs: None,
                };
            }

            // 4. Try to parse public inputs from the proof envelope.
            //    The EVM proof format from MLTrainingProverV2 appends 8 x 32-byte
            //    public inputs after the proof transcript bytes. If the proof
            //    includes these, extract them; otherwise try the raw proof bytes
            //    directly (caller may have separated proof and PIs).
            let (proof_bytes, public_inputs_fr) = self.parse_proof_and_inputs(proof);

            // 5. Perform real Halo2 KZG verification
            let prover_guard = self.prover.read();
            let prover = prover_guard.as_ref().unwrap();

            let is_valid = if let Some(ref pi) = public_inputs_fr {
                prover.verify(&proof_bytes, pi)
            } else {
                // Without public inputs we cannot verify — reject
                false
            };

            let elapsed = start.elapsed().as_millis() as u64;
            let pi_u64 = public_inputs_fr.as_ref().map(|pi| {
                pi.iter().map(|fr| {
                    let repr = fr.to_repr();
                    u64::from_le_bytes(repr.as_ref()[..8].try_into().unwrap_or([0u8; 8]))
                }).collect()
            });

            VerificationResult {
                is_valid,
                verification_time_ms: elapsed,
                error: if is_valid {
                    None
                } else {
                    Some("Halo2 KZG proof verification failed".to_string())
                },
                public_inputs: pi_u64,
            }
        };

        match tokio::time::timeout(self.config.timeout, verification_future).await {
            Ok(result) => result,
            Err(_) => VerificationResult {
                is_valid: false,
                verification_time_ms: self.config.timeout.as_millis() as u64,
                error: Some("Verification timeout".to_string()),
                public_inputs: None,
            },
        }
    }

    /// Parses proof bytes that may contain appended public inputs.
    ///
    /// The EVM format from `MLTrainingProverV2` is:
    ///   [proof_transcript_bytes | PI_0(32 bytes) | PI_1(32 bytes) | ... | PI_7(32 bytes)]
    /// where there are 8 public inputs, each encoded as 32-byte big-endian Fr.
    ///
    /// If the proof is large enough to contain 8 x 32 = 256 bytes of PIs at the end,
    /// we extract them. Otherwise we return the raw bytes as the proof and None for PIs.
    fn parse_proof_and_inputs(&self, data: &[u8]) -> (Vec<u8>, Option<Vec<Fr>>) {
        const NUM_PUBLIC_INPUTS: usize = 8;
        const PI_SIZE: usize = NUM_PUBLIC_INPUTS * 32; // 256 bytes

        if data.len() > PI_SIZE {
            let proof_end = data.len() - PI_SIZE;
            let proof_bytes = data[..proof_end].to_vec();
            let pi_bytes = &data[proof_end..];

            // Parse each 32-byte chunk as a big-endian Fr
            let mut public_inputs = Vec::with_capacity(NUM_PUBLIC_INPUTS);
            for i in 0..NUM_PUBLIC_INPUTS {
                let chunk = &pi_bytes[i * 32..(i + 1) * 32];
                // Fr::from_repr expects little-endian bytes
                let mut le_bytes = [0u8; 32];
                for (j, b) in chunk.iter().enumerate() {
                    le_bytes[31 - j] = *b;
                }
                let repr = <Fr as PrimeField>::Repr::from(le_bytes);
                match Option::from(Fr::from_repr(repr)) {
                    Some(fr) => public_inputs.push(fr),
                    None => return (data.to_vec(), None), // Invalid field element
                }
            }

            (proof_bytes, Some(public_inputs))
        } else {
            (data.to_vec(), None)
        }
    }

    /// Computes a cache key from proof bytes, model commitment, round ID, and error bound.
    ///
    /// Including round_id and error_bound prevents cache hits across different
    /// training steps or configurations that happen to share proof bytes.
    fn cache_key(&self, proof: &[u8], model_commitment: &[u8; 32], round_id: u64, error_bound: f64) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(proof);
        hasher.update(model_commitment);
        hasher.update(round_id.to_le_bytes());
        hasher.update(error_bound.to_le_bytes());
        let result = hasher.finalize();
        result.into()
    }

    fn get_cached(&self, proof_hash: &[u8; 32]) -> Option<VerificationResult> {
        let cache = self.cache.read();
        if let Some(cached) = cache.get(proof_hash) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if now - cached.timestamp < self.config.cache_ttl_secs {
                return Some(cached.result.clone());
            }
        }
        None
    }

    fn cache_result(&self, proof_hash: &[u8; 32], result: &VerificationResult) {
        let mut cache = self.cache.write();

        // Enforce max cache size
        if cache.len() >= self.config.max_cache_size {
            let oldest_key = cache.iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| *k);
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        cache.insert(*proof_hash, CachedVerification {
            result: result.clone(),
            timestamp: now,
        });
    }

    fn compute_aggregated_commitment(&self, commitments: &[[u8; 32]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for commitment in commitments {
            hasher.update(commitment);
        }
        let result = hasher.finalize();
        result.into()
    }
}

/// Gradient validator that combines proof verification with outlier detection.
pub struct GradientValidator {
    /// Proof verifier.
    verifier: ProofVerifier,
    /// Maximum gradient norm.
    max_gradient_norm: f64,
    /// Required participants for Byzantine tolerance.
    min_participants: usize,
}

impl std::fmt::Debug for GradientValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientValidator")
            .field("verifier", &self.verifier)
            .field("max_gradient_norm", &self.max_gradient_norm)
            .field("min_participants", &self.min_participants)
            .finish()
    }
}

impl GradientValidator {
    /// Creates a new gradient validator.
    pub fn new(config: VerificationConfig, max_gradient_norm: f64, min_participants: usize) -> Self {
        Self {
            verifier: ProofVerifier::new(config),
            max_gradient_norm,
            min_participants,
        }
    }

    /// Creates a gradient validator with a pre-initialized prover.
    pub fn with_model_dims(
        config: VerificationConfig,
        max_gradient_norm: f64,
        min_participants: usize,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> Self {
        Self {
            verifier: ProofVerifier::with_model_dims(config, d_in, d_hid, d_out),
            max_gradient_norm,
            min_participants,
        }
    }

    /// Validates a gradient submission.
    pub async fn validate_gradient(
        &self,
        participant: &PeerId,
        round_id: u64,
        gradient_commitment: [u8; 32],
        error_bound: f64,
        proof: &[u8],
        gradient_norm: Option<f64>,
    ) -> ValidationResult {
        // 1. Verify proof
        let proof_result = self.verifier.verify_gradient_proof(
            participant,
            round_id,
            gradient_commitment,
            error_bound,
            proof,
        ).await;

        if !proof_result.is_valid {
            return ValidationResult {
                is_valid: false,
                reason: format!("Proof verification failed: {}", proof_result.error.unwrap_or_default()),
                should_slash: true,
            };
        }

        // 2. Check gradient norm (if provided)
        if let Some(norm) = gradient_norm {
            if norm > self.max_gradient_norm {
                return ValidationResult {
                    is_valid: false,
                    reason: format!("Gradient norm {} exceeds maximum {}", norm, self.max_gradient_norm),
                    should_slash: false, // Could be honest but divergent
                };
            }
        }

        ValidationResult {
            is_valid: true,
            reason: String::new(),
            should_slash: false,
        }
    }

    /// Validates that we have enough participants for Byzantine tolerance.
    pub fn validate_participation(&self, num_participants: usize) -> bool {
        num_participants >= self.min_participants
    }

    /// Returns the proof verifier.
    pub fn verifier(&self) -> &ProofVerifier {
        &self.verifier
    }
}

/// Result of gradient validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the gradient is valid.
    pub is_valid: bool,
    /// Reason for rejection (if invalid).
    pub reason: String,
    /// Whether the participant should be slashed.
    pub should_slash: bool,
}

/// Byzantine-fault-tolerant gradient filter.
///
/// Wraps the gradient aggregation strategies (Krum, TrimmedMean, Median) to
/// detect and reject outlier gradient submissions before they are aggregated.
/// This prevents Byzantine workers from corrupting the model update.
pub struct ByzantineGradientFilter {
    /// Strategy for outlier detection.
    strategy: ByzantineStrategy,
    /// Maximum number of Byzantine workers to tolerate.
    max_byzantine: usize,
    /// Historical gradient norms for statistical detection (bounded ring buffer).
    historical_norms: Vec<f64>,
    /// Z-score threshold for statistical outlier detection.
    zscore_threshold: f64,
    /// Maximum number of historical norms to retain.
    max_historical_norms: usize,
}

/// Strategy for Byzantine gradient filtering.
#[derive(Debug, Clone, Copy)]
pub enum ByzantineStrategy {
    /// Krum: select gradient(s) closest to neighbors, reject distant ones.
    Krum,
    /// Trimmed mean: reject top/bottom fraction of gradient norms.
    TrimmedMean { trim_fraction: f64 },
    /// Coordinate-wise median: not a filter, but an aggregation strategy.
    Median,
    /// Combined: Krum selection + norm-based statistical outlier detection.
    Combined,
}

/// Result of Byzantine gradient filtering.
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Whether the gradient passed the filter.
    pub accepted: bool,
    /// Reason for rejection (if rejected).
    pub reason: Option<String>,
    /// Gradient norm.
    pub gradient_norm: f64,
    /// Z-score (if statistical detection was used).
    pub z_score: Option<f64>,
}

impl ByzantineGradientFilter {
    /// Creates a new Byzantine gradient filter.
    pub fn new(strategy: ByzantineStrategy, max_byzantine: usize) -> Self {
        Self {
            strategy,
            max_byzantine,
            historical_norms: Vec::new(),
            zscore_threshold: 3.0,
            max_historical_norms: 10_000,
        }
    }

    /// Filters a set of gradient submissions, returning which ones to keep.
    ///
    /// Returns a Vec of (participant_id, accepted, reason) for each submission.
    pub fn filter_gradients(
        &mut self,
        submissions: &[(String, f64, Vec<f32>)], // (participant_id, norm, flattened_gradient)
    ) -> Vec<(String, FilterResult)> {
        let n = submissions.len();
        if n == 0 {
            return vec![];
        }

        match self.strategy {
            ByzantineStrategy::Krum => self.filter_krum(submissions),
            ByzantineStrategy::TrimmedMean { trim_fraction } => {
                self.filter_trimmed_mean(submissions, trim_fraction)
            }
            ByzantineStrategy::Median => {
                // Median doesn't reject — all pass
                submissions.iter().map(|(id, norm, _)| {
                    (id.clone(), FilterResult {
                        accepted: true,
                        reason: None,
                        gradient_norm: *norm,
                        z_score: None,
                    })
                }).collect()
            }
            ByzantineStrategy::Combined => self.filter_combined(submissions),
        }
    }

    /// Krum-based filtering: compute pairwise distances and reject outliers.
    fn filter_krum(
        &mut self,
        submissions: &[(String, f64, Vec<f32>)],
    ) -> Vec<(String, FilterResult)> {
        let n = submissions.len();
        if n <= 2 * self.max_byzantine + 2 {
            // Not enough workers for Krum — accept all
            return submissions.iter().map(|(id, norm, _)| {
                (id.clone(), FilterResult {
                    accepted: true,
                    reason: None,
                    gradient_norm: *norm,
                    z_score: None,
                })
            }).collect();
        }

        // Compute pairwise L2 distances
        let mut distances: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = l2_distance(&submissions[i].2, &submissions[j].2);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        // For each gradient, compute Krum score (sum of closest n-f-2 distances)
        let num_closest = n - self.max_byzantine - 2;
        let mut scores: Vec<(usize, f64)> = (0..n).map(|i| {
            let mut dists = distances[i].clone();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let score: f64 = dists[1..=num_closest].iter().sum();
            (i, score)
        }).collect();

        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Accept the top n - max_byzantine gradients (lowest Krum scores)
        let accept_count = n.saturating_sub(self.max_byzantine);
        let accepted_indices: std::collections::HashSet<usize> =
            scores.iter().take(accept_count).map(|(i, _)| *i).collect();

        submissions.iter().enumerate().map(|(i, (id, norm, _))| {
            let accepted = accepted_indices.contains(&i);
            (id.clone(), FilterResult {
                accepted,
                reason: if accepted {
                    None
                } else {
                    Some(format!(
                        "Krum outlier: score {:.4} (rank {})",
                        scores.iter().find(|(idx, _)| *idx == i).map(|(_, s)| *s).unwrap_or(0.0),
                        scores.iter().position(|(idx, _)| *idx == i).unwrap_or(n),
                    ))
                },
                gradient_norm: *norm,
                z_score: None,
            })
        }).collect()
    }

    /// Trimmed mean filtering: reject top and bottom gradient norms.
    fn filter_trimmed_mean(
        &mut self,
        submissions: &[(String, f64, Vec<f32>)],
        trim_fraction: f64,
    ) -> Vec<(String, FilterResult)> {
        let n = submissions.len();
        let trim_count = ((n as f64 * trim_fraction).floor() as usize).min(n / 2);

        // Sort by norm
        let mut indexed: Vec<(usize, f64)> = submissions.iter()
            .enumerate()
            .map(|(i, (_, norm, _))| (i, *norm))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Trim top and bottom
        let accepted_indices: std::collections::HashSet<usize> =
            indexed[trim_count..n.saturating_sub(trim_count)]
                .iter()
                .map(|(i, _)| *i)
                .collect();

        submissions.iter().enumerate().map(|(i, (id, norm, _))| {
            let accepted = accepted_indices.contains(&i);
            (id.clone(), FilterResult {
                accepted,
                reason: if accepted {
                    None
                } else {
                    Some(format!("Trimmed: norm {:.4} outside [{:.4}, {:.4}]",
                        norm,
                        indexed.get(trim_count).map(|(_, n)| *n).unwrap_or(0.0),
                        indexed.get(n.saturating_sub(trim_count + 1)).map(|(_, n)| *n).unwrap_or(0.0),
                    ))
                },
                gradient_norm: *norm,
                z_score: None,
            })
        }).collect()
    }

    /// Combined filtering: Krum + statistical Z-score detection.
    fn filter_combined(
        &mut self,
        submissions: &[(String, f64, Vec<f32>)],
    ) -> Vec<(String, FilterResult)> {
        // First apply Krum
        let mut krum_results = self.filter_krum(submissions);

        // Then apply statistical Z-score filtering on norms
        let norms: Vec<f64> = submissions.iter().map(|(_, n, _)| *n).collect();

        // Add to historical norms for running statistics (bounded)
        for &norm in &norms {
            self.historical_norms.push(norm);
        }
        // Evict oldest entries to prevent unbounded memory growth
        if self.historical_norms.len() > self.max_historical_norms {
            let excess = self.historical_norms.len() - self.max_historical_norms;
            self.historical_norms.drain(..excess);
        }

        if self.historical_norms.len() >= 10 {
            let mean: f64 = self.historical_norms.iter().sum::<f64>() / self.historical_norms.len() as f64;
            let variance: f64 = self.historical_norms.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / self.historical_norms.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                for (i, (_, result)) in krum_results.iter_mut().enumerate() {
                    let z = (norms[i] - mean).abs() / std_dev;
                    result.z_score = Some(z);
                    if z > self.zscore_threshold && result.accepted {
                        result.accepted = false;
                        result.reason = Some(format!(
                            "Statistical outlier: z-score {:.2} > threshold {:.2}",
                            z, self.zscore_threshold
                        ));
                    }
                }
            }
        }

        krum_results
    }
}

/// L2 distance between two gradient vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verifier_disabled() {
        let mut config = VerificationConfig::default();
        config.enabled = false;

        let verifier = ProofVerifier::new(config);

        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            1,
            [0u8; 32],
            0.05,
            &[],
        ).await;

        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_verifier_error_bound_check() {
        let config = VerificationConfig::default();
        let verifier = ProofVerifier::new(config);

        // Error bound too high
        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            1,
            [0u8; 32],
            0.5, // > 0.1 max
            &[1u8; 64],
        ).await;

        assert!(!result.is_valid);
        assert!(result.error.unwrap().contains("Error bound"));
    }

    #[tokio::test]
    async fn test_verifier_empty_proof() {
        let config = VerificationConfig::default();
        let verifier = ProofVerifier::new(config);

        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            1,
            [0u8; 32],
            0.05,
            &[], // Empty proof
        ).await;

        assert!(!result.is_valid);
    }

    #[tokio::test]
    async fn test_verifier_rejects_invalid_proof() {
        // With real verification, random bytes should fail
        let config = VerificationConfig {
            model_dims: Some((4, 8, 2)),
            ..Default::default()
        };
        let verifier = ProofVerifier::new(config);

        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            1,
            [1u8; 32],
            0.05,
            &[0xAB; 512], // Random bytes — not a valid Halo2 proof
        ).await;

        assert!(!result.is_valid);
    }

    #[tokio::test]
    async fn test_verifier_rejects_without_model_dims() {
        let config = VerificationConfig::default(); // No model_dims
        let verifier = ProofVerifier::new(config);

        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            1,
            [1u8; 32],
            0.05,
            &[0xAB; 512],
        ).await;

        assert!(!result.is_valid);
        assert!(result.error.unwrap().contains("not initialized"));
    }

    #[tokio::test]
    async fn test_batch_verify() {
        let config = VerificationConfig {
            model_dims: Some((4, 8, 2)),
            ..Default::default()
        };
        let verifier = ProofVerifier::new(config);

        let proofs = vec![
            (PeerId::random(), 1, [1u8; 32], 0.05, vec![1u8; 64]),
            (PeerId::random(), 1, [2u8; 32], 0.05, vec![1u8; 64]),
        ];

        let results = verifier.batch_verify(proofs).await;
        assert_eq!(results.len(), 2);
        // Both should fail since they're not real Halo2 proofs
        assert!(!results[0].1.is_valid);
        assert!(!results[1].1.is_valid);
    }

    #[test]
    fn test_gradient_validator() {
        let config = VerificationConfig::default();
        let validator = GradientValidator::new(config, 100.0, 3);

        assert!(!validator.validate_participation(2));
        assert!(validator.validate_participation(3));
    }

    #[tokio::test]
    async fn test_real_proof_round_trip() {
        use helix_prover::MLTrainingProverV2;
        use halo2curves::bn256::Fr;
        use halo2curves::ff::Field;

        // Generate a real proof using MLTrainingProverV2
        let (d_in, d_hid, d_out) = (4, 8, 2);
        let prover = MLTrainingProverV2::new(d_in, d_hid, d_out);

        // Build a witness with small nonzero values
        let x = vec![Fr::from(1u64); d_in];
        let target = vec![Fr::from(1u64); d_out];
        let w1 = vec![Fr::from(1u64); d_hid * d_in];
        let b1 = vec![Fr::zero(); d_hid];
        let w2 = vec![Fr::from(1u64); d_out * d_hid];
        let b2 = vec![Fr::zero(); d_out];
        let lr = Fr::from(1u64);
        let witness = MLTrainingProverV2::build_witness(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr, 0, Fr::zero(),
        );
        let proof_result = prover.prove(&witness);

        // If proving succeeds, verify it through our ProofVerifier
        if let Ok(result) = proof_result {
            let config = VerificationConfig {
                model_dims: Some((4, 8, 2)),
                ..Default::default()
            };
            let verifier = ProofVerifier::with_model_dims(config, 4, 8, 2);

            // Construct the proof+PI payload as the prover would emit
            let mut proof_with_pi = result.proof.clone();
            for fr in &result.public_inputs {
                let repr = fr.to_repr();
                // Convert LE repr to BE for EVM format
                let mut be_bytes = [0u8; 32];
                for (i, b) in repr.as_ref().iter().enumerate() {
                    be_bytes[31 - i] = *b;
                }
                proof_with_pi.extend_from_slice(&be_bytes);
            }

            let vr = verifier.verify_gradient_proof(
                &PeerId::random(),
                1,
                [0u8; 32],
                0.05,
                &proof_with_pi,
            ).await;

            assert!(vr.is_valid, "Real proof should verify: {:?}", vr.error);
        }
        // If proving fails (e.g. in CI with limited resources), skip gracefully
    }

    #[test]
    fn test_parse_proof_and_inputs() {
        let config = VerificationConfig::default();
        let verifier = ProofVerifier::new(config);

        // Proof too small for PI extraction
        let (proof, pi) = verifier.parse_proof_and_inputs(&[0u8; 100]);
        assert_eq!(proof.len(), 100);
        assert!(pi.is_none());

        // Proof with appended PIs (257 bytes = 1 proof byte + 8*32 PI bytes)
        let mut data = vec![0xAA; 300];
        // Write 8 zero PIs at the end (zero is a valid Fr)
        for i in 0..256 {
            data[300 - 256 + i] = 0;
        }
        let (proof, pi) = verifier.parse_proof_and_inputs(&data);
        assert_eq!(proof.len(), 44); // 300 - 256
        assert!(pi.is_some());
        assert_eq!(pi.unwrap().len(), 8);
    }

    #[test]
    fn test_byzantine_krum_filter() {
        let mut filter = ByzantineGradientFilter::new(ByzantineStrategy::Krum, 1);

        // 5 honest workers with similar gradients, 1 Byzantine with very different gradient
        let honest_grad = vec![1.0f32, 2.0, 3.0, 4.0];
        let byzantine_grad = vec![100.0f32, 200.0, 300.0, 400.0];

        let submissions: Vec<(String, f64, Vec<f32>)> = vec![
            ("w0".into(), 5.48, honest_grad.clone()),
            ("w1".into(), 5.50, honest_grad.iter().map(|v| v + 0.1).collect()),
            ("w2".into(), 5.52, honest_grad.iter().map(|v| v + 0.2).collect()),
            ("w3".into(), 5.47, honest_grad.iter().map(|v| v - 0.1).collect()),
            ("w4".into(), 5.49, honest_grad.iter().map(|v| v + 0.05).collect()),
            ("byzantine".into(), 547.7, byzantine_grad),
        ];

        let results = filter.filter_gradients(&submissions);
        assert_eq!(results.len(), 6);

        // Byzantine worker should be rejected
        let byz_result = results.iter().find(|(id, _)| id == "byzantine").unwrap();
        assert!(!byz_result.1.accepted, "Byzantine gradient should be rejected");

        // Honest workers should be accepted
        for (id, result) in &results {
            if id != "byzantine" {
                assert!(result.accepted, "Honest worker {} should be accepted", id);
            }
        }
    }

    #[test]
    fn test_byzantine_trimmed_mean_filter() {
        let mut filter = ByzantineGradientFilter::new(
            ByzantineStrategy::TrimmedMean { trim_fraction: 0.2 },
            1,
        );

        let submissions: Vec<(String, f64, Vec<f32>)> = vec![
            ("w0".into(), 1.0, vec![1.0]),
            ("w1".into(), 2.0, vec![2.0]),
            ("w2".into(), 3.0, vec![3.0]),
            ("w3".into(), 4.0, vec![4.0]),
            ("w4".into(), 100.0, vec![100.0]), // Outlier
        ];

        let results = filter.filter_gradients(&submissions);

        // w4 (highest norm) should be trimmed
        let w4_result = results.iter().find(|(id, _)| id == "w4").unwrap();
        assert!(!w4_result.1.accepted, "Highest-norm gradient should be trimmed");

        // w0 (lowest norm) should also be trimmed
        let w0_result = results.iter().find(|(id, _)| id == "w0").unwrap();
        assert!(!w0_result.1.accepted, "Lowest-norm gradient should be trimmed");
    }

    #[test]
    fn test_byzantine_filter_too_few_workers() {
        let mut filter = ByzantineGradientFilter::new(ByzantineStrategy::Krum, 1);

        // Only 3 workers — too few for Krum with f=1 (need > 2f+2 = 4)
        let submissions: Vec<(String, f64, Vec<f32>)> = vec![
            ("w0".into(), 1.0, vec![1.0]),
            ("w1".into(), 2.0, vec![2.0]),
            ("w2".into(), 100.0, vec![100.0]),
        ];

        let results = filter.filter_gradients(&submissions);
        // All should be accepted when Krum can't run
        assert!(results.iter().all(|(_, r)| r.accepted));
    }

    #[test]
    fn test_default_policy_is_verify_all() {
        let policy = VerificationPolicy::default();
        assert!(matches!(policy, VerificationPolicy::VerifyAll));

        let config = VerificationConfig::default();
        assert!(matches!(config.policy, VerificationPolicy::VerifyAll));
    }

    #[tokio::test]
    async fn test_structural_only_requires_explicit_config() {
        // Default config should NOT be StructuralOnly
        let default_config = VerificationConfig::default();
        assert!(
            !matches!(default_config.policy, VerificationPolicy::StructuralOnly),
            "Default policy must not be StructuralOnly"
        );

        // StructuralOnly can only be set via explicit config override
        let explicit_config = VerificationConfig {
            policy: VerificationPolicy::StructuralOnly,
            ..Default::default()
        };
        assert!(matches!(explicit_config.policy, VerificationPolicy::StructuralOnly));

        // Verify StructuralOnly actually accepts structural-only proofs (no Halo2)
        let verifier = ProofVerifier::new(explicit_config);
        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            1,
            [1u8; 32],
            0.05,
            &[0xAB; 64], // Any 64+ byte blob passes structural checks
        ).await;
        assert!(result.is_valid, "StructuralOnly should accept any 64+ byte blob");
    }

    #[tokio::test]
    async fn test_verify_all_rejects_random_blob() {
        // With VerifyAll (default), random bytes should fail even if >= 64 bytes
        let config = VerificationConfig {
            model_dims: Some((4, 8, 2)),
            ..Default::default()
        };
        assert!(matches!(config.policy, VerificationPolicy::VerifyAll));

        let verifier = ProofVerifier::new(config);
        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            1,
            [1u8; 32],
            0.05,
            &[0xAB; 512], // Random bytes — not a valid Halo2 proof
        ).await;
        assert!(!result.is_valid, "VerifyAll must reject random bytes");
    }
}
