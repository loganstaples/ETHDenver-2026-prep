//! Optional ZK Proof Generation for MPC Checkpoints.
//!
//! This module provides optional zero-knowledge proof generation tied to
//! MPC training checkpoints. When enabled, it generates lightweight state
//! transition proofs at configurable intervals, providing an additional
//! layer of cryptographic verification beyond the SPDZ MACs.
//!
//! # Design Principles
//!
//! - **Zero overhead when disabled**: When `ZKConfig::enabled` is false,
//!   no prover infrastructure is initialized and `on_checkpoint()` returns
//!   immediately.
//!
//! - **Lazy initialization**: The `CheckpointProver` is only created on
//!   the first checkpoint that requires a proof, avoiding startup cost
//!   if proofs are rarely needed.
//!
//! - **Configurable frequency**: Proofs can be generated every N checkpoints
//!   (e.g., every 5th checkpoint) to balance verification cost vs. coverage.
//!
//! # Usage
//!
//! ```ignore
//! use helix_mpc::zk_integration::{ZKConfig, ZKCheckpointManager, ZKCheckpointData};
//!
//! // Disabled (no overhead)
//! let mut mgr = ZKCheckpointManager::new(ZKConfig::disabled());
//! assert!(mgr.on_checkpoint(&data).unwrap().is_none());
//!
//! // Enabled with proof every 5 checkpoints
//! let config = ZKConfig {
//!     enabled: true,
//!     checkpoint_freq: 5,
//!     num_weights: 1000,
//! };
//! let mut mgr = ZKCheckpointManager::new(config);
//! // Checkpoints 1-4: no proof
//! // Checkpoint 5: generates proof
//! ```

use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;
use helix_circuits::ml::state_transition::{
    StateTransitionWitness, compute_field_hash, split_hash,
};
use tracing::{debug, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for optional ZK proof generation at checkpoints.
#[derive(Debug, Clone)]
pub struct ZKConfig {
    /// Whether ZK proof generation is enabled.
    /// When false, no prover is initialized and on_checkpoint() is a no-op.
    pub enabled: bool,
    /// Generate a proof every N checkpoints.
    /// E.g., checkpoint_freq=5 means proofs at checkpoints 5, 10, 15, ...
    pub checkpoint_freq: u64,
    /// Number of weights in the model (determines circuit size).
    pub num_weights: usize,
}

impl ZKConfig {
    /// Creates a disabled configuration (zero overhead).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            checkpoint_freq: 1,
            num_weights: 0,
        }
    }

    /// Creates an enabled configuration with specified parameters.
    pub fn enabled(num_weights: usize, checkpoint_freq: u64) -> Self {
        Self {
            enabled: true,
            checkpoint_freq: checkpoint_freq.max(1), // at least every checkpoint
            num_weights,
        }
    }
}

impl Default for ZKConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

// ============================================================================
// Checkpoint Data
// ============================================================================

/// Data needed to generate a ZK proof at a checkpoint.
#[derive(Debug, Clone)]
pub struct ZKCheckpointData {
    /// Old weights (at the start of the checkpoint interval).
    pub old_weights: Vec<Fr>,
    /// New weights (at the end of the checkpoint interval).
    pub new_weights: Vec<Fr>,
    /// Error bound accumulated during this interval.
    pub error_bound: Fr,
}

impl ZKCheckpointData {
    /// Creates checkpoint data from weight vectors.
    pub fn new(old_weights: Vec<Fr>, new_weights: Vec<Fr>, error_bound: Fr) -> Self {
        Self {
            old_weights,
            new_weights,
            error_bound,
        }
    }
}

// ============================================================================
// Proof Result
// ============================================================================

/// Result of a ZK checkpoint proof generation.
#[derive(Debug, Clone)]
pub struct ZKCheckpointProof {
    /// Serialized proof bytes.
    pub proof_bytes: Vec<u8>,
    /// Public inputs used.
    pub public_inputs: Vec<Fr>,
    /// Whether the proof was self-verified.
    pub verified: bool,
    /// Checkpoint number at which this proof was generated.
    pub checkpoint_number: u64,
    /// Proof generation time in milliseconds.
    pub generation_time_ms: u64,
}

// ============================================================================
// ZK Checkpoint Manager
// ============================================================================

/// Manages optional ZK proof generation at MPC checkpoints.
///
/// When ZK is disabled, this is essentially a no-op wrapper with zero overhead.
/// When enabled, it lazily initializes the `CheckpointProver` on the first
/// checkpoint that needs a proof, and generates proofs at the configured frequency.
pub struct ZKCheckpointManager {
    /// Configuration.
    config: ZKConfig,
    /// Lazily initialized prover (only when enabled and first proof is needed).
    prover: Option<helix_prover::provers::checkpoint_prover::CheckpointProver>,
    /// Number of checkpoints since the last proof was generated.
    checkpoints_since_last_proof: u64,
    /// Total number of checkpoints processed.
    total_checkpoints: u64,
    /// Total number of proofs generated.
    proofs_generated: u64,
}

impl ZKCheckpointManager {
    /// Creates a new ZK checkpoint manager.
    ///
    /// If `config.enabled` is false, no prover infrastructure is allocated.
    pub fn new(config: ZKConfig) -> Self {
        if config.enabled {
            info!(
                num_weights = config.num_weights,
                checkpoint_freq = config.checkpoint_freq,
                "ZK checkpoint manager created (enabled)"
            );
        } else {
            debug!("ZK checkpoint manager created (disabled, zero overhead)");
        }

        Self {
            config,
            prover: None,
            checkpoints_since_last_proof: 0,
            total_checkpoints: 0,
            proofs_generated: 0,
        }
    }

    /// Returns whether ZK proof generation is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Returns whether the prover has been initialized.
    pub fn is_prover_initialized(&self) -> bool {
        self.prover.is_some()
    }

    /// Returns the number of proofs generated so far.
    pub fn proofs_generated(&self) -> u64 {
        self.proofs_generated
    }

    /// Returns the total number of checkpoints processed.
    pub fn total_checkpoints(&self) -> u64 {
        self.total_checkpoints
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ZKConfig {
        &self.config
    }

    /// Lazily initializes the prover.
    ///
    /// This is called on the first checkpoint that needs a proof.
    /// Key generation happens here, which may take a few seconds.
    fn ensure_prover(&mut self) {
        if self.prover.is_none() {
            info!(
                num_weights = self.config.num_weights,
                "Lazily initializing CheckpointProver (first proof requested)"
            );
            let prover_config = helix_prover::provers::checkpoint_prover::CheckpointProverConfig::new(
                self.config.num_weights,
            );
            self.prover = Some(
                helix_prover::provers::checkpoint_prover::CheckpointProver::new(prover_config),
            );
            info!("CheckpointProver initialized successfully");
        }
    }

    /// Processes a checkpoint and optionally generates a ZK proof.
    ///
    /// Returns `Some(proof)` if a proof was generated at this checkpoint,
    /// or `None` if no proof was needed (either disabled or not at the
    /// configured frequency).
    ///
    /// # Errors
    ///
    /// Returns an error if proof generation fails (only when a proof is
    /// actually attempted).
    pub fn on_checkpoint(
        &mut self,
        data: &ZKCheckpointData,
    ) -> Result<Option<ZKCheckpointProof>, String> {
        self.total_checkpoints += 1;
        self.checkpoints_since_last_proof += 1;

        // Early return if disabled
        if !self.config.enabled {
            return Ok(None);
        }

        // Check if we need to generate a proof at this checkpoint
        if self.checkpoints_since_last_proof < self.config.checkpoint_freq {
            debug!(
                checkpoints_since_last = self.checkpoints_since_last_proof,
                freq = self.config.checkpoint_freq,
                "Checkpoint processed, no proof needed yet"
            );
            return Ok(None);
        }

        // Time to generate a proof
        info!(
            checkpoint = self.total_checkpoints,
            proofs_so_far = self.proofs_generated,
            "Generating ZK checkpoint proof"
        );

        // Validate data dimensions
        if data.old_weights.len() != self.config.num_weights {
            return Err(format!(
                "old_weights has {} elements but expected {}",
                data.old_weights.len(),
                self.config.num_weights,
            ));
        }
        if data.new_weights.len() != self.config.num_weights {
            return Err(format!(
                "new_weights has {} elements but expected {}",
                data.new_weights.len(),
                self.config.num_weights,
            ));
        }

        // Lazily initialize prover
        self.ensure_prover();

        // Build witness
        let witness = StateTransitionWitness::new(
            data.old_weights.clone(),
            data.new_weights.clone(),
            data.error_bound,
        );

        // Generate proof
        let start = std::time::Instant::now();
        let prover = self.prover.as_ref().unwrap();
        let result = prover.prove(&witness).map_err(|e| format!("Proof generation failed: {}", e))?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Update counters
        self.proofs_generated += 1;
        self.checkpoints_since_last_proof = 0;

        info!(
            checkpoint = self.total_checkpoints,
            proof_size = result.proof_bytes.len(),
            generation_time_ms = elapsed_ms,
            verified = result.verified,
            total_proofs = self.proofs_generated,
            "ZK checkpoint proof generated successfully"
        );

        Ok(Some(ZKCheckpointProof {
            proof_bytes: result.proof_bytes,
            public_inputs: result.public_inputs,
            verified: result.verified,
            checkpoint_number: self.total_checkpoints,
            generation_time_ms: elapsed_ms,
        }))
    }

    /// Verifies a previously generated checkpoint proof.
    ///
    /// Returns an error if the prover is not initialized (no proofs have
    /// been generated yet).
    pub fn verify_proof(&self, proof: &ZKCheckpointProof) -> Result<bool, String> {
        let prover = self.prover.as_ref().ok_or_else(|| {
            "Prover not initialized - no proofs have been generated yet".to_string()
        })?;

        let pi_refs: Vec<&[Fr]> = vec![&proof.public_inputs];
        prover
            .verify_raw(&proof.proof_bytes, &proof.public_inputs)
            .map_err(|e| format!("Verification failed: {}", e))
    }

    /// Returns a summary of the manager's state.
    pub fn summary(&self) -> String {
        format!(
            "ZKCheckpointManager: enabled={}, freq={}, weights={}, \
             checkpoints={}, proofs={}, prover_initialized={}",
            self.config.enabled,
            self.config.checkpoint_freq,
            self.config.num_weights,
            self.total_checkpoints,
            self.proofs_generated,
            self.prover.is_some(),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::ff::Field;
    use std::sync::LazyLock;

    /// Lock to serialize proof-heavy tests (same pattern as helix-prover).
    static ZK_TEST_LOCK: LazyLock<std::sync::Mutex<()>> =
        LazyLock::new(|| std::sync::Mutex::new(()));

    /// Helper to create test checkpoint data.
    fn make_test_data(n: usize) -> ZKCheckpointData {
        let old: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();
        let new: Vec<Fr> = (0..n).map(|i| Fr::from((i + 2) as u64)).collect();
        ZKCheckpointData::new(old, new, Fr::from(100u64))
    }

    #[test]
    fn test_zk_disabled_no_overhead() {
        let mut mgr = ZKCheckpointManager::new(ZKConfig::disabled());

        assert!(!mgr.is_enabled());
        assert!(!mgr.is_prover_initialized());

        // Process 100 checkpoints - no prover should ever be created
        let data = make_test_data(8);
        for _ in 0..100 {
            let result = mgr.on_checkpoint(&data).unwrap();
            assert!(result.is_none(), "Disabled manager should never produce proofs");
        }

        assert!(!mgr.is_prover_initialized(), "Prover should never be initialized when disabled");
        assert_eq!(mgr.proofs_generated(), 0);
        assert_eq!(mgr.total_checkpoints(), 100);
    }

    #[test]
    fn test_zk_config_defaults() {
        let disabled = ZKConfig::disabled();
        assert!(!disabled.enabled);

        let enabled = ZKConfig::enabled(100, 5);
        assert!(enabled.enabled);
        assert_eq!(enabled.num_weights, 100);
        assert_eq!(enabled.checkpoint_freq, 5);

        // checkpoint_freq should be at least 1
        let min_freq = ZKConfig::enabled(100, 0);
        assert_eq!(min_freq.checkpoint_freq, 1);
    }

    #[test]
    fn test_zk_checkpoint_frequency() {
        // This test exercises the full proof pipeline, which is heavy.
        // Only run when the prover test lock is available.
        let _lock = ZK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let num_weights = 8;
        let config = ZKConfig::enabled(num_weights, 3);
        let mut mgr = ZKCheckpointManager::new(config);

        let data = make_test_data(num_weights);

        // Track which checkpoints produce proofs
        let mut proof_checkpoints = Vec::new();

        for i in 1..=10 {
            let result = mgr.on_checkpoint(&data).unwrap();
            if result.is_some() {
                proof_checkpoints.push(i);
            }
        }

        // With freq=3, proofs should be generated at checkpoints 3, 6, 9
        assert_eq!(
            proof_checkpoints,
            vec![3, 6, 9],
            "Proofs should be generated at checkpoints 3, 6, 9"
        );
        assert_eq!(mgr.proofs_generated(), 3);
        assert_eq!(mgr.total_checkpoints(), 10);
        assert!(mgr.is_prover_initialized());
    }

    #[test]
    fn test_zk_lazy_initialization() {
        let num_weights = 8;
        let config = ZKConfig::enabled(num_weights, 5);
        let mut mgr = ZKCheckpointManager::new(config);

        // Prover should not be initialized yet
        assert!(!mgr.is_prover_initialized());

        let data = make_test_data(num_weights);

        // Process 4 checkpoints (below freq=5 threshold)
        for _ in 0..4 {
            let _ = mgr.on_checkpoint(&data).unwrap();
        }

        // Prover should still not be initialized (no proof needed yet)
        assert!(!mgr.is_prover_initialized(), "Prover should not initialize until first proof is needed");
    }

    #[test]
    fn test_zk_dimension_mismatch_error() {
        let num_weights = 8;
        let config = ZKConfig::enabled(num_weights, 1); // every checkpoint
        let mut mgr = ZKCheckpointManager::new(config);

        // Create data with wrong number of weights
        let wrong_data = make_test_data(4); // 4 instead of 8

        let result = mgr.on_checkpoint(&wrong_data);
        assert!(result.is_err(), "Mismatched dimensions should be rejected");
    }

    #[test]
    fn test_zk_proof_verification() {
        let _lock = ZK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let num_weights = 8;
        let config = ZKConfig::enabled(num_weights, 1);
        let mut mgr = ZKCheckpointManager::new(config);

        let data = make_test_data(num_weights);

        // Generate a proof
        let proof = mgr.on_checkpoint(&data).unwrap().expect("Should generate proof");

        // Verify it
        let is_valid = mgr.verify_proof(&proof).expect("Verification should succeed");
        assert!(is_valid, "Generated proof should verify");
    }

    #[test]
    fn test_zk_verify_without_prover_errors() {
        let mgr = ZKCheckpointManager::new(ZKConfig::disabled());

        let fake_proof = ZKCheckpointProof {
            proof_bytes: vec![0u8; 100],
            public_inputs: vec![Fr::ZERO; 6],
            verified: false,
            checkpoint_number: 1,
            generation_time_ms: 0,
        };

        let result = mgr.verify_proof(&fake_proof);
        assert!(result.is_err(), "Verification without prover should error");
    }

    #[test]
    fn test_zk_summary() {
        let config = ZKConfig::enabled(100, 5);
        let mgr = ZKCheckpointManager::new(config);
        let summary = mgr.summary();
        assert!(summary.contains("enabled=true"));
        assert!(summary.contains("freq=5"));
        assert!(summary.contains("weights=100"));
    }
}
