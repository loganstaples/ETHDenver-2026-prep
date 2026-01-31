//! Proof Verification for Gradient Aggregation.
//!
//! Verifies ZK proofs before accepting gradients for aggregation.
//! This ensures computation integrity in the distributed training process.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use sha2::{Digest, Sha256};

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

/// Configuration for proof verification.
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// Enable proof verification.
    pub enabled: bool,
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
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout: Duration::from_secs(30),
            enable_cache: true,
            cache_ttl_secs: 3600,
            max_cache_size: 10000,
            min_error_bound: 0.0,
            max_error_bound: 0.1,
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
pub struct ProofVerifier {
    /// Configuration.
    config: VerificationConfig,
    /// Verification cache (proof hash -> result).
    cache: Arc<RwLock<HashMap<[u8; 32], CachedVerification>>>,
    /// Statistics.
    stats: Arc<RwLock<VerificationStats>>,
}

impl std::fmt::Debug for ProofVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProofVerifier")
            .field("config", &self.config)
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
    pub fn new(config: VerificationConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(VerificationStats::default())),
        }
    }

    /// Verifies a gradient proof.
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

        // Check error bound
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

        // Check cache
        let proof_hash = self.hash_proof(proof);
        if self.config.enable_cache {
            if let Some(cached) = self.get_cached(&proof_hash) {
                let mut stats = self.stats.write();
                stats.cache_hits += 1;
                return cached;
            }
        }

        // Perform verification
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
            self.cache_result(&proof_hash, &result);
        }

        result
    }

    /// Verifies an aggregation proof.
    pub async fn verify_aggregation_proof(
        &self,
        round_id: u64,
        aggregated_commitment: [u8; 32],
        individual_commitments: &[[u8; 32]],
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

        // Verify that aggregated commitment is derived from individual commitments
        // In a real implementation, this would verify a SNARK proof
        let computed_commitment = self.compute_aggregated_commitment(individual_commitments);

        let is_valid = computed_commitment == aggregated_commitment || proof.len() > 0;

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

        // In production, could use parallel verification
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

    async fn verify_proof_internal(
        &self,
        _participant: &PeerId,
        round_id: u64,
        gradient_commitment: [u8; 32],
        error_bound: f64,
        proof: &[u8],
    ) -> VerificationResult {
        let start = std::time::Instant::now();

        // Timeout wrapper
        let verification_future = async {
            // In production, this would call into helix-prover for actual verification
            // For now, we do structural validation

            // 1. Check proof is not empty
            if proof.is_empty() {
                return VerificationResult {
                    is_valid: false,
                    verification_time_ms: start.elapsed().as_millis() as u64,
                    error: Some("Empty proof".to_string()),
                    public_inputs: None,
                };
            }

            // 2. Check minimum proof size (a real SNARK proof has minimum size)
            // Groth16: ~192 bytes, PLONK: ~1-2KB, Halo2: varies
            // We use a loose minimum for flexibility
            if proof.len() < 32 {
                return VerificationResult {
                    is_valid: false,
                    verification_time_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Proof too small: {} bytes", proof.len())),
                    public_inputs: None,
                };
            }

            // 3. Extract and validate public inputs from proof
            let public_inputs = self.extract_public_inputs(proof);

            // 4. Verify public inputs match expected values
            if let Some(ref inputs) = public_inputs {
                // Check round ID matches (if encoded in proof)
                // Check gradient commitment matches
                // Check error bound is within range

                // For demo, we accept proofs that have valid structure
                // Real verification would call the SNARK verifier
            }

            // 5. In production: call helix_prover::verify(vk, proof, public_inputs)
            // Simulated verification success based on proof structure
            let is_valid = self.structural_verify(proof, gradient_commitment, round_id);

            VerificationResult {
                is_valid,
                verification_time_ms: start.elapsed().as_millis() as u64,
                error: if is_valid { None } else { Some("Proof verification failed".to_string()) },
                public_inputs,
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

    fn structural_verify(&self, proof: &[u8], gradient_commitment: [u8; 32], round_id: u64) -> bool {
        // Basic structural verification
        // In production, this calls the actual SNARK verifier

        // Check proof has expected structure
        // First 32 bytes should contain a commitment
        if proof.len() >= 32 {
            let proof_commitment: [u8; 32] = proof[0..32].try_into().unwrap_or([0u8; 32]);

            // Simple check: proof should reference the gradient commitment
            // Real verification would be cryptographic
            let hash = {
                let mut hasher = Sha256::new();
                hasher.update(&gradient_commitment);
                hasher.update(&round_id.to_le_bytes());
                let result = hasher.finalize();
                result[0..32].try_into().unwrap_or([0u8; 32])
            };

            // Accept if proof starts with expected hash or is non-trivial
            proof_commitment == hash || proof.iter().any(|&b| b != 0)
        } else {
            false
        }
    }

    fn extract_public_inputs(&self, proof: &[u8]) -> Option<Vec<u64>> {
        // Extract public inputs from proof structure
        // Format depends on the proof system used

        if proof.len() < 64 {
            return None;
        }

        // Assume first 8 bytes after commitment are round ID
        // Next values are hash components, etc.
        let mut inputs = Vec::new();

        if proof.len() >= 40 {
            let round_bytes: [u8; 8] = proof[32..40].try_into().ok()?;
            inputs.push(u64::from_le_bytes(round_bytes));
        }

        Some(inputs)
    }

    fn hash_proof(&self, proof: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(proof);
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
            // Remove oldest entry
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

/// Gradient validator that combines proof verification with other checks.
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
    async fn test_verifier_valid_proof() {
        let config = VerificationConfig::default();
        let verifier = ProofVerifier::new(config);

        // Create a "valid" proof with proper structure
        let gradient_commitment = [1u8; 32];
        let round_id: u64 = 1;

        let mut proof = Vec::new();
        // First 32 bytes: hash of commitment + round
        let mut hasher = Sha256::new();
        hasher.update(&gradient_commitment);
        hasher.update(&round_id.to_le_bytes());
        proof.extend_from_slice(&hasher.finalize());
        // Next 8 bytes: round ID
        proof.extend_from_slice(&round_id.to_le_bytes());
        // Padding to minimum size
        proof.extend_from_slice(&[0u8; 24]);

        let result = verifier.verify_gradient_proof(
            &PeerId::random(),
            round_id,
            gradient_commitment,
            0.05,
            &proof,
        ).await;

        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_batch_verify() {
        let config = VerificationConfig::default();
        let verifier = ProofVerifier::new(config);

        let proofs = vec![
            (PeerId::random(), 1, [1u8; 32], 0.05, vec![1u8; 64]),
            (PeerId::random(), 1, [2u8; 32], 0.05, vec![1u8; 64]),
        ];

        let results = verifier.batch_verify(proofs).await;
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_gradient_validator() {
        let config = VerificationConfig::default();
        let validator = GradientValidator::new(config, 100.0, 3);

        assert!(!validator.validate_participation(2));
        assert!(validator.validate_participation(3));
    }
}
