//! Verifier Node Role.
//!
//! Implements the verifier node role which validates proofs
//! and maintains network consensus. Supports full Halo2 KZG verification
//! (default), structural validation, replay detection, and
//! concurrency-limited verification.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};

use parking_lot::RwLock as SyncRwLock;

use helix_prover::halo2curves::bn256::Fr;
use helix_prover::halo2curves::ff::PrimeField;
use helix_prover::MLTrainingProverV2;

use crate::network::messages::{NodeCapabilities, PeerId};
use crate::sc_client::TrainingProofInputs;

/// Minimum proof size in bytes for a valid KZG proof.
const MIN_PROOF_SIZE: usize = 384;

/// Maximum number of recent proof IDs tracked for replay detection.
const REPLAY_CACHE_CAPACITY: usize = 10_000;

/// Maximum reasonable step number.
const MAX_STEP_NUMBER: u64 = 1_000_000;

/// Verifier state.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifierState {
    /// Ready to verify.
    Ready,
    /// Currently verifying a proof.
    Verifying { proof_id: String },
    /// Error state.
    Error { message: String },
}

/// Verification policy — controls how strictly proofs are validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationPolicy {
    /// Full Halo2 KZG proof verification (structural checks + cryptographic verification).
    /// Most secure — this is the production default.
    VerifyAll,
    /// Structural validation only: size check, public input count, range checks.
    /// Suitable for testing environments where full proving is too slow.
    Structural,
    /// Permissive mode: only checks that proof is non-empty (for demos).
    Permissive,
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        Self::VerifyAll
    }
}

/// Configuration for verifier node.
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Maximum concurrent verifications.
    pub max_concurrent: usize,
    /// Verification timeout (seconds).
    pub timeout_secs: u64,
    /// Verification policy.
    pub policy: VerificationPolicy,
    /// Model dimensions (d_in, d_hid, d_out) for Halo2 prover initialization.
    /// Required when policy is `VerifyAll`.
    pub model_dims: Option<(usize, usize, usize)>,
}

impl Default for VerifierConfig {
    /// Secure-by-default: uses full Halo2 KZG verification with demo model dimensions.
    ///
    /// In production, always use `VerifierConfig::for_model(d_in, d_hid, d_out)` with
    /// the actual model dimensions. Use `VerifierConfig::structural()` only for testing.
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            timeout_secs: 60,
            policy: VerificationPolicy::VerifyAll,
            model_dims: Some((2, 2, 1)),
        }
    }
}

impl VerifierConfig {
    /// Creates a config for testing that uses structural validation only.
    ///
    /// **WARNING**: This should NEVER be used in production. Structural validation
    /// only checks proof size and public input ranges — it does NOT verify the
    /// cryptographic proof. Any data that meets the size requirements will be accepted.
    pub fn structural() -> Self {
        Self {
            max_concurrent: 4,
            timeout_secs: 60,
            policy: VerificationPolicy::Structural,
            model_dims: None,
        }
    }

    /// Creates a config for production use with full Halo2 KZG verification.
    ///
    /// Requires model dimensions to initialize the prover circuit.
    pub fn for_model(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            max_concurrent: 4,
            timeout_secs: 60,
            policy: VerificationPolicy::VerifyAll,
            model_dims: Some((d_in, d_hid, d_out)),
        }
    }

    /// Creates a config for demo use with the standard demo MLP (2, 2, 1).
    pub fn demo() -> Self {
        Self::for_model(2, 2, 1)
    }
}

/// Verifier node statistics.
#[derive(Debug, Clone, Default)]
pub struct VerifierStats {
    /// Total proofs verified.
    pub proofs_verified: u64,
    /// Proofs accepted.
    pub proofs_accepted: u64,
    /// Proofs rejected.
    pub proofs_rejected: u64,
    /// Average verification time (ms).
    pub avg_verify_time_ms: f64,
    /// Replays detected.
    pub replays_detected: u64,
}

/// Result of structural proof validation with a reason for rejection.
#[derive(Debug, Clone)]
pub enum VerifyResult {
    /// Proof is valid.
    Valid,
    /// Proof is invalid with a reason.
    Invalid(String),
}

impl VerifyResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }
}

/// Verifier node role.
pub struct VerifierNode {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: VerifierConfig,
    /// Current state.
    state: Arc<RwLock<VerifierState>>,
    /// Statistics.
    stats: Arc<RwLock<VerifierStats>>,
    /// Replay detection cache.
    replay_cache: Arc<RwLock<HashSet<String>>>,
    /// Concurrency semaphore.
    semaphore: Arc<Semaphore>,
    /// Halo2 prover for real KZG verification (lazy-initialized).
    halo2_prover: Arc<SyncRwLock<Option<MLTrainingProverV2>>>,
}

impl VerifierNode {
    /// Creates a new verifier node.
    pub fn new(local_id: PeerId, config: VerifierConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        // Pre-initialize prover if model_dims are given and policy requires it.
        let prover = if config.policy == VerificationPolicy::VerifyAll {
            config.model_dims.map(|(d_in, d_hid, d_out)| {
                MLTrainingProverV2::new(d_in, d_hid, d_out)
            })
        } else {
            None
        };
        Self {
            local_id,
            config,
            state: Arc::new(RwLock::new(VerifierState::Ready)),
            stats: Arc::new(RwLock::new(VerifierStats::default())),
            replay_cache: Arc::new(RwLock::new(HashSet::new())),
            semaphore,
            halo2_prover: Arc::new(SyncRwLock::new(prover)),
        }
    }

    /// Ensures the Halo2 prover is initialized, creating it lazily if model_dims are available.
    fn ensure_halo2_prover(&self) -> bool {
        if self.halo2_prover.read().is_some() {
            return true;
        }
        if let Some((d_in, d_hid, d_out)) = self.config.model_dims {
            let prover = MLTrainingProverV2::new(d_in, d_hid, d_out);
            *self.halo2_prover.write() = Some(prover);
            true
        } else {
            false
        }
    }

    /// Converts `TrainingProofInputs` (U256 values) to Halo2 `Fr` field elements.
    /// Returns all 8 public inputs needed for native Halo2 KZG verification.
    fn inputs_to_fr(inputs: &TrainingProofInputs) -> Vec<Fr> {
        let mut result = Vec::with_capacity(8);
        for u256 in &[
            inputs.old_hash_lo,
            inputs.old_hash_hi,
            inputs.new_hash_lo,
            inputs.new_hash_hi,
            inputs.loss,
            inputs.error_bound,
            inputs.step_number,
            inputs.error_checksum,
        ] {
            let mut le_bytes = [0u8; 32];
            u256.to_little_endian(&mut le_bytes);
            let repr = <Fr as PrimeField>::Repr::from(le_bytes);
            match Option::from(Fr::from_repr(repr)) {
                Some(fr) => result.push(fr),
                None => {
                    // If any U256 value doesn't fit in Fr, use zero
                    result.push(Fr::from(0u64));
                }
            }
        }
        result
    }

    /// Performs full Halo2 KZG verification of a proof against public inputs.
    fn halo2_verify(&self, proof: &[u8], inputs: &TrainingProofInputs) -> VerifyResult {
        if !self.ensure_halo2_prover() {
            return VerifyResult::Invalid(
                "Halo2 prover not initialized: model_dims not configured".to_string()
            );
        }

        let public_inputs_fr = Self::inputs_to_fr(inputs);
        let prover_guard = self.halo2_prover.read();
        let prover = prover_guard.as_ref().unwrap();

        if prover.verify(proof, &public_inputs_fr) {
            VerifyResult::Valid
        } else {
            VerifyResult::Invalid("Halo2 KZG proof verification failed".to_string())
        }
    }

    /// Gets the node's capabilities.
    pub fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            can_train: false,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 100,
        }
    }

    /// Gets current state.
    pub async fn get_state(&self) -> VerifierState {
        self.state.read().await.clone()
    }

    /// Performs structural validation of proof bytes and public inputs.
    fn structural_check(proof: &[u8], inputs: &TrainingProofInputs) -> VerifyResult {
        // Size check: KZG proofs must be at least 384 bytes
        if proof.len() < MIN_PROOF_SIZE {
            return VerifyResult::Invalid(format!(
                "Proof too small: {} bytes (minimum {})",
                proof.len(),
                MIN_PROOF_SIZE
            ));
        }

        // Public inputs: old/new hashes must be non-zero
        if inputs.old_hash_lo.is_zero() && inputs.old_hash_hi.is_zero() {
            return VerifyResult::Invalid("Old weight hash is zero".into());
        }
        if inputs.new_hash_lo.is_zero() && inputs.new_hash_hi.is_zero() {
            return VerifyResult::Invalid("New weight hash is zero".into());
        }

        // Step number must be reasonable
        if inputs.step_number.as_u64() > MAX_STEP_NUMBER {
            return VerifyResult::Invalid(format!(
                "Step number {} exceeds maximum {}",
                inputs.step_number,
                MAX_STEP_NUMBER
            ));
        }

        VerifyResult::Valid
    }

    /// Verifies a proof with structural validation, replay detection, and concurrency control.
    pub async fn verify_proof(
        &self,
        proof_id: String,
        proof: &[u8],
        inputs: &TrainingProofInputs,
    ) -> VerifyResult {
        // Acquire concurrency permit
        let _permit = self.semaphore.acquire().await.unwrap();

        let start = Instant::now();

        // Replay detection
        {
            let mut cache = self.replay_cache.write().await;
            if cache.contains(&proof_id) {
                let mut stats = self.stats.write().await;
                stats.replays_detected += 1;
                return VerifyResult::Invalid(format!("Replay detected: {}", proof_id));
            }
            // Evict oldest entries if at capacity (simple clear strategy)
            if cache.len() >= REPLAY_CACHE_CAPACITY {
                cache.clear();
            }
            cache.insert(proof_id.clone());
        }

        // Update state
        {
            let mut state = self.state.write().await;
            *state = VerifierState::Verifying { proof_id: proof_id.clone() };
        }

        // Validate based on policy
        let result = match self.config.policy {
            VerificationPolicy::VerifyAll => {
                // Structural checks first, then full Halo2 KZG verification
                let structural = Self::structural_check(proof, inputs);
                if !structural.is_valid() {
                    structural
                } else {
                    self.halo2_verify(proof, inputs)
                }
            }
            VerificationPolicy::Structural => {
                Self::structural_check(proof, inputs)
            }
            VerificationPolicy::Permissive => {
                if proof.is_empty() {
                    VerifyResult::Invalid("Empty proof".into())
                } else {
                    VerifyResult::Valid
                }
            }
        };

        // Update stats
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        {
            let mut stats = self.stats.write().await;
            stats.proofs_verified += 1;
            if result.is_valid() {
                stats.proofs_accepted += 1;
            } else {
                stats.proofs_rejected += 1;
            }
            // Running average
            let n = stats.proofs_verified as f64;
            stats.avg_verify_time_ms =
                stats.avg_verify_time_ms * ((n - 1.0) / n) + elapsed_ms / n;
        }

        // Return to ready state
        {
            let mut state = self.state.write().await;
            *state = VerifierState::Ready;
        }

        result
    }

    /// Legacy verify_proof for backward compatibility (permissive, no inputs).
    pub async fn verify_proof_legacy(&self, proof_id: String, proof: &[u8]) -> bool {
        use ethers::types::U256;
        let dummy_inputs = TrainingProofInputs {
            old_hash_lo: U256::from(1),
            old_hash_hi: U256::from(1),
            new_hash_lo: U256::from(1),
            new_hash_hi: U256::from(1),
            loss: U256::zero(),
            error_bound: U256::from(10),
            step_number: U256::from(1),
            error_checksum: U256::zero(),
        };
        // Use permissive check for legacy callers
        let result = if proof.is_empty() {
            VerifyResult::Invalid("Empty proof".into())
        } else {
            self.verify_proof(proof_id, proof, &dummy_inputs).await
        };
        result.is_valid()
    }

    /// Gets statistics.
    pub async fn get_stats(&self) -> VerifierStats {
        self.stats.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethers::types::U256;

    /// Returns a `VerifierConfig` using `Structural` policy so unit tests that
    /// fabricate proof bytes don't need a real Halo2 prover.
    fn structural_config() -> VerifierConfig {
        VerifierConfig::structural()
    }

    fn make_valid_inputs() -> TrainingProofInputs {
        TrainingProofInputs {
            old_hash_lo: U256::from(1),
            old_hash_hi: U256::from(2),
            new_hash_lo: U256::from(3),
            new_hash_hi: U256::from(4),
            loss: U256::from(100),
            error_bound: U256::from(10),
            step_number: U256::from(1),
            error_checksum: U256::zero(),
        }
    }

    fn make_valid_proof() -> Vec<u8> {
        vec![0xAB; 512] // > MIN_PROOF_SIZE
    }

    #[tokio::test]
    async fn test_verifier_init() {
        let local_id = PeerId::random();
        let node = VerifierNode::new(local_id, structural_config());

        assert!(matches!(node.get_state().await, VerifierState::Ready));
    }

    #[test]
    fn test_default_policy_is_verify_all() {
        let config = VerifierConfig::default();
        assert_eq!(config.policy, VerificationPolicy::VerifyAll);
        assert!(config.model_dims.is_some(), "default must include model_dims for VerifyAll");
    }

    #[test]
    fn test_structural_constructor_for_testing() {
        let config = VerifierConfig::structural();
        assert_eq!(config.policy, VerificationPolicy::Structural);
        assert!(config.model_dims.is_none());
    }

    #[test]
    fn test_for_model_constructor() {
        let config = VerifierConfig::for_model(4, 8, 2);
        assert_eq!(config.policy, VerificationPolicy::VerifyAll);
        assert_eq!(config.model_dims, Some((4, 8, 2)));
    }

    #[test]
    fn test_demo_constructor() {
        let config = VerifierConfig::demo();
        assert_eq!(config.policy, VerificationPolicy::VerifyAll);
        assert_eq!(config.model_dims, Some((2, 2, 1)));
    }

    #[tokio::test]
    async fn test_verify_all_rejects_without_model_dims() {
        // VerifyAll without model_dims should reject proofs (prover can't init)
        let config = VerifierConfig {
            policy: VerificationPolicy::VerifyAll,
            model_dims: None,
            ..Default::default()
        };
        let node = VerifierNode::new(PeerId::random(), config);
        let result = node.verify_proof("p1".into(), &make_valid_proof(), &make_valid_inputs()).await;
        assert!(!result.is_valid());
        if let VerifyResult::Invalid(reason) = result {
            assert!(reason.contains("not initialized"));
        }
    }

    #[tokio::test]
    async fn test_verify_all_rejects_fake_proof() {
        // VerifyAll with model_dims should reject random bytes
        let config = VerifierConfig {
            policy: VerificationPolicy::VerifyAll,
            model_dims: Some((2, 2, 1)),
            ..Default::default()
        };
        let node = VerifierNode::new(PeerId::random(), config);
        let result = node.verify_proof("p1".into(), &make_valid_proof(), &make_valid_inputs()).await;
        assert!(!result.is_valid());
        if let VerifyResult::Invalid(reason) = result {
            assert!(reason.contains("Halo2 KZG"));
        }
    }

    #[tokio::test]
    async fn test_empty_proof_rejected() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let result = node.verify_proof("p1".into(), &[], &make_valid_inputs()).await;
        assert!(!result.is_valid());
    }

    #[tokio::test]
    async fn test_undersized_proof_rejected() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let small_proof = vec![1u8; 100]; // < 384
        let result = node.verify_proof("p1".into(), &small_proof, &make_valid_inputs()).await;
        assert!(!result.is_valid());
        if let VerifyResult::Invalid(reason) = result {
            assert!(reason.contains("too small"));
        }
    }

    #[tokio::test]
    async fn test_valid_structural_proof_accepted() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let result = node.verify_proof("p1".into(), &make_valid_proof(), &make_valid_inputs()).await;
        assert!(result.is_valid());
    }

    #[tokio::test]
    async fn test_zero_old_hash_rejected() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let mut inputs = make_valid_inputs();
        inputs.old_hash_lo = U256::zero();
        inputs.old_hash_hi = U256::zero();
        let result = node.verify_proof("p1".into(), &make_valid_proof(), &inputs).await;
        assert!(!result.is_valid());
    }

    #[tokio::test]
    async fn test_zero_new_hash_rejected() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let mut inputs = make_valid_inputs();
        inputs.new_hash_lo = U256::zero();
        inputs.new_hash_hi = U256::zero();
        let result = node.verify_proof("p1".into(), &make_valid_proof(), &inputs).await;
        assert!(!result.is_valid());
    }

    #[tokio::test]
    async fn test_excessive_step_number_rejected() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let mut inputs = make_valid_inputs();
        inputs.step_number = U256::from(MAX_STEP_NUMBER + 1);
        let result = node.verify_proof("p1".into(), &make_valid_proof(), &inputs).await;
        assert!(!result.is_valid());
    }

    #[tokio::test]
    async fn test_replay_detection() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let proof = make_valid_proof();
        let inputs = make_valid_inputs();

        // First submission succeeds
        let result = node.verify_proof("same-id".into(), &proof, &inputs).await;
        assert!(result.is_valid());

        // Second submission with same ID is replay
        let result = node.verify_proof("same-id".into(), &proof, &inputs).await;
        assert!(!result.is_valid());
        if let VerifyResult::Invalid(reason) = result {
            assert!(reason.contains("Replay"));
        }

        let stats = node.get_stats().await;
        assert_eq!(stats.replays_detected, 1);
    }

    #[tokio::test]
    async fn test_concurrency_limit_respected() {
        let config = VerifierConfig {
            max_concurrent: 2,
            policy: VerificationPolicy::Structural,
            ..Default::default()
        };
        let node = Arc::new(VerifierNode::new(PeerId::random(), config));
        let proof = make_valid_proof();
        let inputs = make_valid_inputs();

        // Launch 3 concurrent verifications
        let mut handles = Vec::new();
        for i in 0..3 {
            let node_clone = Arc::clone(&node);
            let proof_clone = proof.clone();
            let inputs_clone = inputs.clone();
            handles.push(tokio::spawn(async move {
                node_clone
                    .verify_proof(format!("concurrent-{}", i), &proof_clone, &inputs_clone)
                    .await
            }));
        }

        // All should complete (semaphore queues rather than rejects)
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_valid());
        }

        let stats = node.get_stats().await;
        assert_eq!(stats.proofs_verified, 3);
    }

    #[tokio::test]
    async fn test_permissive_policy_accepts_small_proof() {
        let config = VerifierConfig {
            policy: VerificationPolicy::Permissive,
            ..Default::default()
        };
        let node = VerifierNode::new(PeerId::random(), config);
        let small_proof = vec![1u8; 10]; // Would fail structural check
        let result = node.verify_proof("p1".into(), &small_proof, &make_valid_inputs()).await;
        assert!(result.is_valid());
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let node = VerifierNode::new(PeerId::random(), structural_config());
        let inputs = make_valid_inputs();

        // One valid, one invalid
        node.verify_proof("p1".into(), &make_valid_proof(), &inputs).await;
        node.verify_proof("p2".into(), &[], &inputs).await;

        let stats = node.get_stats().await;
        assert_eq!(stats.proofs_verified, 2);
        assert_eq!(stats.proofs_accepted, 1);
        assert_eq!(stats.proofs_rejected, 1);
    }
}
