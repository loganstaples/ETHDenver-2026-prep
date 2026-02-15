//! Checkpoint Prover for State Transition Proofs.
//!
//! Wraps the `StateTransitionCircuit` from `helix-circuits` with the
//! Halo2 `ProverPipeline` to produce real KZG proofs for MPC checkpoint
//! verification.
//!
//! This prover is lightweight compared to `MLTrainingProverV2`:
//! - Smaller circuit (only verifies weight delta consistency)
//! - Faster proof generation (~10x faster than full training step proofs)
//! - Designed for optional use at MPC checkpoint boundaries
//!
//! # Usage
//!
//! ```ignore
//! use helix_prover::provers::checkpoint_prover::*;
//!
//! // Configure for 1000-weight model
//! let config = CheckpointProverConfig::new(1000);
//! let prover = CheckpointProver::new(config);
//!
//! // Generate proof for a weight transition
//! let witness = StateTransitionWitness::new(old_weights, new_weights, error_bound);
//! let result = prover.prove(&witness)?;
//!
//! // Verify the proof
//! assert!(prover.verify(&result)?);
//! ```

use crate::keys::generate_keys_for_circuit;
use crate::pipeline::{
    CancellationToken, PipelineConfig, PipelineError, ProofPhase, ProofProgress, ProgressCallback,
    ProverPipeline, RetryConfig, no_progress_callback,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use helix_circuits::ml::state_transition::{
    StateTransitionCircuit, StateTransitionWitness,
    NUM_PUBLIC_INPUTS, compute_field_hash, split_hash,
};
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

// ============================================================================
// Error Types
// ============================================================================

/// Errors specific to checkpoint proof generation.
#[derive(Error, Debug)]
pub enum CheckpointProverError {
    /// Pipeline error.
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Witness validation failed.
    #[error("Witness validation failed: {0}")]
    WitnessValidation(String),

    /// Self-verification failed.
    #[error("Self-verification failed: generated proof does not verify")]
    SelfVerificationFailed,

    /// Operation timed out.
    #[error("Operation timed out after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },

    /// Operation cancelled.
    #[error("Operation cancelled")]
    Cancelled,
}

/// Result type for checkpoint prover operations.
pub type CheckpointProverResult<T> = Result<T, CheckpointProverError>;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the checkpoint prover.
#[derive(Debug, Clone)]
pub struct CheckpointProverConfig {
    /// K parameter (circuit size = 2^k rows).
    pub k: u32,
    /// Number of weights in the model.
    pub num_weights: usize,
    /// Whether to self-verify proofs after generation.
    pub self_verify: bool,
    /// Retry configuration.
    pub retry: RetryConfig,
    /// Proof generation timeout.
    pub proof_timeout: Option<Duration>,
    /// Verification timeout.
    pub verify_timeout: Option<Duration>,
}

impl CheckpointProverConfig {
    /// Creates a configuration for the given number of weights.
    ///
    /// Automatically computes the minimum k needed.
    pub fn new(num_weights: usize) -> Self {
        let dummy = StateTransitionCircuit {
            witness: StateTransitionWitness::default(),
            num_weights,
        };
        let k = dummy.minimum_k().max(10); // Ensure minimum k for halo2

        Self {
            k,
            num_weights,
            self_verify: true,
            retry: RetryConfig::default(),
            proof_timeout: Some(Duration::from_secs(120)),
            verify_timeout: Some(Duration::from_secs(30)),
        }
    }

    /// Creates a minimal configuration for testing.
    pub fn minimal(num_weights: usize) -> Self {
        let dummy = StateTransitionCircuit {
            witness: StateTransitionWitness::default(),
            num_weights,
        };
        let k = dummy.minimum_k().max(10);

        Self {
            k,
            num_weights,
            self_verify: false,
            retry: RetryConfig::none(),
            proof_timeout: Some(Duration::from_secs(60)),
            verify_timeout: Some(Duration::from_secs(10)),
        }
    }

    /// Sets the k parameter explicitly (overrides auto-computation).
    pub fn with_k(mut self, k: u32) -> Self {
        self.k = k;
        self
    }

    /// Enables or disables self-verification.
    pub fn with_self_verify(mut self, verify: bool) -> Self {
        self.self_verify = verify;
        self
    }
}

// ============================================================================
// Proof Result
// ============================================================================

/// Result of generating a checkpoint proof.
#[derive(Debug, Clone)]
pub struct CheckpointProofResult {
    /// Serialized Halo2 proof bytes.
    pub proof_bytes: Vec<u8>,
    /// Public inputs for verification.
    pub public_inputs: Vec<Fr>,
    /// Whether the proof was self-verified.
    pub verified: bool,
    /// Proof generation time.
    pub generation_time: Duration,
    /// Verification time (if self-verified).
    pub verification_time: Option<Duration>,
    /// Number of attempts.
    pub attempts: u32,
}

impl CheckpointProofResult {
    /// Returns the proof bytes in EVM-compatible format.
    ///
    /// The pipeline uses PSE's Keccak256Transcript which produces proofs
    /// directly compatible with the Halo2 Solidity verifier.
    pub fn to_evm_proof(&self) -> Vec<u8> {
        self.proof_bytes.clone()
    }

    /// Returns public inputs as big-endian uint256 byte arrays for Solidity.
    pub fn to_evm_public_inputs(&self) -> Vec<[u8; 32]> {
        self.public_inputs
            .iter()
            .map(|fr| {
                let repr = fr.to_repr();
                let bytes: &[u8] = repr.as_ref();
                // Convert LE repr to BE for EVM
                let mut be = [0u8; 32];
                for (i, b) in bytes.iter().enumerate() {
                    be[31 - i] = *b;
                }
                be
            })
            .collect()
    }

    /// Returns the proof size in bytes.
    pub fn proof_size(&self) -> usize {
        self.proof_bytes.len()
    }
}

// ============================================================================
// Checkpoint Prover
// ============================================================================

/// Checkpoint prover that generates KZG proofs for state transitions.
///
/// This wraps the `StateTransitionCircuit` with the Halo2 proving pipeline,
/// providing real KZG proof generation and verification.
pub struct CheckpointProver {
    /// Halo2 proving pipeline.
    pipeline: ProverPipeline<StateTransitionCircuit>,
    /// Configuration.
    config: CheckpointProverConfig,
}

impl CheckpointProver {
    /// Creates and initializes a new checkpoint prover.
    ///
    /// This performs key generation (SRS setup, keygen_vk, keygen_pk), which
    /// may take a few seconds for the first call. Subsequent calls with the
    /// same k will benefit from SRS caching.
    #[instrument(skip_all, fields(k = config.k, num_weights = config.num_weights))]
    pub fn new(config: CheckpointProverConfig) -> Self {
        let pipeline_config = PipelineConfig {
            k: config.k,
            self_verify: false, // We handle self-verification ourselves
            retry: config.retry.clone(),
            proof_timeout: config.proof_timeout,
            verify_timeout: config.verify_timeout,
            deterministic_seed: None,
            enable_tracing: true,
        };

        // Create a dummy circuit for key generation
        let dummy_witness = StateTransitionWitness {
            old_weights: vec![Fr::ZERO; config.num_weights],
            new_weights: vec![Fr::ZERO; config.num_weights],
            delta: vec![Fr::ZERO; config.num_weights],
            error_bound: Fr::ZERO,
        };
        let setup_circuit = StateTransitionCircuit {
            witness: dummy_witness,
            num_weights: config.num_weights,
        };

        // Generate real Halo2 keys
        let circuit_keys = generate_keys_for_circuit(
            &setup_circuit,
            "state_transition_checkpoint",
            1,
            config.k,
        );

        // Create pipeline from real CircuitKeys
        let pipeline = ProverPipeline::from_keys(
            pipeline_config,
            circuit_keys.params,
            circuit_keys.pk,
            circuit_keys.vk,
        );

        info!(
            k = config.k,
            num_weights = config.num_weights,
            "Checkpoint prover initialized"
        );

        Self { pipeline, config }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CheckpointProverConfig {
        &self.config
    }

    /// Generates a KZG proof for a state transition.
    #[instrument(skip_all)]
    pub fn prove(
        &self,
        witness: &StateTransitionWitness,
    ) -> CheckpointProverResult<CheckpointProofResult> {
        self.prove_with_options(witness, no_progress_callback(), None)
    }

    /// Generates a proof with progress callbacks and cancellation support.
    #[instrument(skip_all)]
    pub fn prove_with_options(
        &self,
        witness: &StateTransitionWitness,
        progress: ProgressCallback,
        cancel_token: Option<&CancellationToken>,
    ) -> CheckpointProverResult<CheckpointProofResult> {
        info!("Checkpoint proof generation started");
        let start = Instant::now();

        // Check cancellation
        if let Some(token) = cancel_token {
            if token.is_cancelled() {
                return Err(CheckpointProverError::Cancelled);
            }
        }

        // Validate witness
        witness.validate().map_err(CheckpointProverError::WitnessValidation)?;

        if witness.num_weights() != self.config.num_weights {
            return Err(CheckpointProverError::WitnessValidation(format!(
                "witness has {} weights but prover configured for {}",
                witness.num_weights(),
                self.config.num_weights,
            )));
        }

        // Check timeout
        if let Some(timeout) = self.config.proof_timeout {
            if start.elapsed() > timeout {
                return Err(CheckpointProverError::Timeout {
                    elapsed_ms: start.elapsed().as_millis() as u64,
                });
            }
        }

        // Build circuit
        let circuit = StateTransitionCircuit {
            witness: witness.clone(),
            num_weights: self.config.num_weights,
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
            message: Some("Generating checkpoint proof".to_string()),
        });

        let proof_result = self
            .pipeline
            .prove_with_options(&circuit, &pi_refs, progress.clone(), cancel_token)
            .map_err(|e| {
                error!(error = %e, "Checkpoint proof generation failed");
                CheckpointProverError::Pipeline(e)
            })?;

        let gen_time = proof_result.generation_time;
        let proof = proof_result.proof;
        let attempts = proof_result.attempts;

        // Self-verification
        let (verified, verify_time) = if self.config.self_verify {
            progress(ProofProgress {
                phase: ProofPhase::Verification,
                phase_progress: 0.0,
                overall_progress: 0.9,
                elapsed: start.elapsed(),
                estimated_remaining: None,
                attempt: 0,
                message: Some("Self-verifying checkpoint proof".to_string()),
            });

            let verify_start = Instant::now();
            let is_valid = self.pipeline.verify(&proof, &pi_refs)?;

            if !is_valid {
                return Err(CheckpointProverError::SelfVerificationFailed);
            }

            (true, Some(verify_start.elapsed()))
        } else {
            (false, None)
        };

        info!(
            proof_size = proof.len(),
            generation_time_ms = gen_time.as_millis() as u64,
            verified,
            "Checkpoint proof generation complete"
        );

        Ok(CheckpointProofResult {
            proof_bytes: proof,
            public_inputs: pi,
            verified,
            generation_time: gen_time,
            verification_time: verify_time,
            attempts,
        })
    }

    /// Verifies a previously generated checkpoint proof.
    pub fn verify(&self, result: &CheckpointProofResult) -> CheckpointProverResult<bool> {
        let pi_refs: Vec<&[Fr]> = vec![&result.public_inputs];
        let is_valid = self.pipeline.verify(&result.proof_bytes, &pi_refs)?;
        Ok(is_valid)
    }

    /// Verifies a proof given raw bytes and public inputs.
    pub fn verify_raw(
        &self,
        proof_bytes: &[u8],
        public_inputs: &[Fr],
    ) -> CheckpointProverResult<bool> {
        let pi_refs: Vec<&[Fr]> = vec![public_inputs];
        let is_valid = self.pipeline.verify(proof_bytes, &pi_refs)?;
        Ok(is_valid)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create test weights.
    fn make_test_weights(n: usize) -> (Vec<Fr>, Vec<Fr>) {
        let old: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();
        let new: Vec<Fr> = (0..n).map(|i| Fr::from((i + 2) as u64)).collect();
        (old, new)
    }

    #[test]
    fn test_checkpoint_proof_generation() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let num_weights = 8;
        let (old_w, new_w) = make_test_weights(num_weights);
        let witness = StateTransitionWitness::new(old_w, new_w, Fr::from(100u64));

        // Create prover with minimal config
        let config = CheckpointProverConfig::minimal(num_weights);
        let prover = CheckpointProver::new(config);

        // Generate proof
        let result = prover.prove(&witness).expect("Proof generation should succeed");

        // Verify proof is non-empty and properly sized
        assert!(!result.proof_bytes.is_empty(), "Proof must be non-empty");
        assert!(result.proof_bytes.len() > 32, "Proof should be a real KZG proof");
        assert_eq!(
            result.public_inputs.len(),
            NUM_PUBLIC_INPUTS,
            "Must have {} public inputs",
            NUM_PUBLIC_INPUTS
        );

        // Verify the proof
        let is_valid = prover.verify(&result).expect("Verification should not error");
        assert!(is_valid, "Generated proof must verify");
    }

    #[test]
    fn test_proof_size() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let num_weights = 8;
        let (old_w, new_w) = make_test_weights(num_weights);
        let witness = StateTransitionWitness::new(old_w, new_w, Fr::from(50u64));

        let config = CheckpointProverConfig::minimal(num_weights);
        let prover = CheckpointProver::new(config);
        let result = prover.prove(&witness).expect("Proof generation should succeed");

        // KZG proofs from SHPLONK should be reasonably sized
        // Typically 1-3 KB for small circuits
        assert!(result.proof_size() > 0, "Proof must have positive size");
        assert!(
            result.proof_size() < 100_000,
            "Proof should be less than 100KB, got {} bytes",
            result.proof_size()
        );

        // Proof should be a multiple of 32 bytes (field element size)
        assert_eq!(
            result.proof_size() % 32,
            0,
            "Proof size should be a multiple of 32 bytes"
        );

        // EVM public inputs should be properly formatted
        let evm_pi = result.to_evm_public_inputs();
        assert_eq!(evm_pi.len(), NUM_PUBLIC_INPUTS);
        for pi in &evm_pi {
            assert_eq!(pi.len(), 32, "Each EVM PI should be 32 bytes");
        }
    }

    #[test]
    fn test_checkpoint_prover_config_auto_k() {
        // Small model
        let config_small = CheckpointProverConfig::new(10);
        assert!(config_small.k >= 8, "k should be at least 8");

        // Larger model
        let config_large = CheckpointProverConfig::new(1000);
        assert!(
            config_large.k >= config_small.k,
            "Larger model needs larger or equal k"
        );
    }

    #[test]
    fn test_witness_dimension_mismatch_rejected() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let num_weights = 8;
        let config = CheckpointProverConfig::minimal(num_weights);
        let prover = CheckpointProver::new(config);

        // Create witness with wrong number of weights
        let (old_w, new_w) = make_test_weights(4); // 4 instead of 8
        let witness = StateTransitionWitness::new(old_w, new_w, Fr::from(100u64));

        let result = prover.prove(&witness);
        assert!(result.is_err(), "Mismatched witness dimensions should be rejected");
    }
}
