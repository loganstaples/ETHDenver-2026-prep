//! Optional ZK proof layer for the full orchestrator.
//!
//! When `--zk-proofs` is enabled, this module provides lazy-initialized ZK proof
//! generation at checkpoint boundaries. The ZK prover is only created when the
//! first proof is requested, avoiding any overhead when ZK is disabled.
//!
//! The prover generates `StateTransitionCircuit` proofs showing that model
//! weights transitioned validly between two consecutive checkpoints. These proofs
//! can be submitted on-chain via `submitCheckpointWithProof()` for external
//! verifiability by parties that didn't participate in MPC.
//!
//! # Design
//!
//! - **Zero overhead when disabled**: No SRS loading, no circuit setup, no imports
//! - **Lazy initialization**: First proof request triggers SRS + keygen (~2-5s)
//! - **Incremental proving**: Each proof covers the transition from the previous
//!   checkpoint's weights to the current ones
//! - **Non-fatal failures**: ZK proof failures don't halt the pipeline; MPC+MAC
//!   is the primary correctness mechanism

use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use helix_circuits::ml::state_transition::StateTransitionWitness;
use helix_mpc::e2e_integration::FinalWeights;
use helix_prover::halo2curves::bn256::Fr as Halo2Fr;
use helix_prover::{CheckpointProofResult, CheckpointProver, CheckpointProverConfig};
use tracing::{debug, info, warn};

/// Configuration for the optional ZK proof layer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZkProofConfig {
    /// Whether ZK proof generation is enabled.
    pub enabled: bool,
    /// Generate a ZK proof every N checkpoints (1 = every checkpoint).
    pub checkpoint_frequency: usize,
    /// Whether to self-verify proofs after generation.
    pub self_verify: bool,
}

impl Default for ZkProofConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint_frequency: 1,
            self_verify: true,
        }
    }
}

/// Result of a ZK proof generation attempt for a checkpoint.
#[derive(Debug, Clone)]
pub struct ZkCheckpointProofResult {
    /// The generated proof.
    pub proof: CheckpointProofResult,
    /// Training step this proof covers (transition endpoint).
    pub step: usize,
    /// Total time including any lazy initialization.
    pub total_time: Duration,
}

/// Tracks ZK proof generation statistics.
#[derive(Debug, Default)]
pub struct ZkProofStats {
    /// Number of proofs generated.
    pub proofs_generated: usize,
    /// Number of proofs that self-verified.
    pub proofs_verified: usize,
    /// Total proving time.
    pub total_proving_time: Duration,
    /// Prover initialization time (one-time cost).
    pub init_time: Option<Duration>,
}

/// Lazy-initialized ZK proof layer for the orchestrator.
///
/// Only performs expensive operations (SRS loading, key generation) when the
/// first proof is actually requested. Tracks weight state across checkpoints
/// to generate incremental state transition proofs.
pub struct ZkProofLayer {
    config: ZkProofConfig,
    /// The actual Halo2 prover, lazily initialized on first use.
    prover: Option<CheckpointProver>,
    /// Weights at the previous ZK checkpoint (flattened `Vec<Halo2Fr>`).
    prev_weights: Option<Vec<Halo2Fr>>,
    /// Total number of weights in the model.
    num_weights: usize,
    /// Running statistics.
    stats: ZkProofStats,
    /// Number of checkpoints seen so far (for frequency tracking).
    checkpoints_seen: usize,
}

impl ZkProofLayer {
    /// Creates a new ZK proof layer.
    ///
    /// No expensive operations happen here. SRS loading and key generation are
    /// deferred until the first proof is actually requested.
    pub fn new(config: ZkProofConfig, d_in: usize, d_hid: usize, d_out: usize) -> Self {
        let num_weights = d_in * d_hid + d_hid + d_hid * d_out + d_out;
        Self {
            config,
            prover: None,
            prev_weights: None,
            num_weights,
            stats: ZkProofStats::default(),
            checkpoints_seen: 0,
        }
    }

    /// Returns whether this layer is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Returns the proof statistics.
    pub fn stats(&self) -> &ZkProofStats {
        &self.stats
    }

    /// Returns the number of proofs generated.
    pub fn proofs_generated(&self) -> usize {
        self.stats.proofs_generated
    }

    /// Sets the baseline (initial) weights for ZK proving.
    ///
    /// Must be called before `process_checkpoint`. The baseline weights represent
    /// the model state before MPC training begins, and serve as the "old state"
    /// for the first ZK state transition proof.
    pub fn set_baseline_weights(&mut self, w1: &[f64], b1: &[f64], w2: &[f64], b2: &[f64]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let weights = FinalWeights {
            w1: w1.to_vec(),
            b1: b1.to_vec(),
            w2: w2.to_vec(),
            b2: b2.to_vec(),
        };
        let elements = weights_to_field_elements(&weights, self.num_weights)?;
        debug!(
            num_weights = elements.len(),
            "ZK layer: baseline weights set from initial model state"
        );
        self.prev_weights = Some(elements);
        Ok(())
    }

    /// Returns whether the given checkpoint index (0-based) should have a ZK proof.
    pub fn should_prove_checkpoint(&self, checkpoint_index: usize) -> bool {
        if !self.config.enabled {
            return false;
        }
        if self.prev_weights.is_none() {
            return false;
        }
        (checkpoint_index + 1) % self.config.checkpoint_frequency == 0
    }

    /// Processes a checkpoint and optionally generates a ZK proof.
    ///
    /// Call `set_baseline_weights` first with the initial model weights.
    /// Then call this at each checkpoint with the actual reconstructed weights
    /// at that checkpoint. Generates a proof at frequency intervals showing
    /// the state transition from the previous proven checkpoint to this one.
    ///
    /// Returns `Ok(Some(result))` if a proof was generated, `Ok(None)` if this
    /// checkpoint is not a ZK frequency match or baseline is not yet set.
    pub fn process_checkpoint(
        &mut self,
        checkpoint_index: usize,
        step: usize,
        weights: &FinalWeights,
        loss: f64,
    ) -> Result<Option<ZkCheckpointProofResult>> {
        self.process_checkpoint_inner(checkpoint_index, step, weights, loss, false)
    }

    /// Like `process_checkpoint` but forces proof generation regardless of frequency.
    /// Used for the final checkpoint to ensure at least one ZK proof is generated.
    pub fn process_final_checkpoint(
        &mut self,
        checkpoint_index: usize,
        step: usize,
        weights: &FinalWeights,
        loss: f64,
    ) -> Result<Option<ZkCheckpointProofResult>> {
        self.process_checkpoint_inner(checkpoint_index, step, weights, loss, true)
    }

    fn process_checkpoint_inner(
        &mut self,
        checkpoint_index: usize,
        step: usize,
        weights: &FinalWeights,
        loss: f64,
        force: bool,
    ) -> Result<Option<ZkCheckpointProofResult>> {
        if !self.config.enabled {
            return Ok(None);
        }

        self.checkpoints_seen += 1;

        // Convert f64 weights to Halo2Fr field elements
        let current_weights = weights_to_field_elements(weights, self.num_weights)?;

        // If no baseline set, store as baseline (backward compat with tests)
        if self.prev_weights.is_none() {
            debug!(
                step = step,
                num_weights = current_weights.len(),
                "ZK layer: recording baseline weights (first checkpoint)"
            );
            self.prev_weights = Some(current_weights);
            return Ok(None);
        }

        // Check if this checkpoint should produce a ZK proof (skip check if forced)
        if !force && !self.should_prove_checkpoint(checkpoint_index) {
            return Ok(None);
        }

        let total_start = Instant::now();

        // Lazy-init the prover
        self.ensure_initialized()?;

        let prev = self.prev_weights.as_ref().unwrap();

        // Convert loss to field element (scale by 10^9 to preserve precision)
        let error_bound = {
            let scaled = (loss.abs() * 1e9) as u64;
            Halo2Fr::from(scaled)
        };

        let witness = StateTransitionWitness::new(prev.clone(), current_weights.clone(), error_bound);

        info!(
            step = step,
            checkpoint_index = checkpoint_index,
            num_weights = self.num_weights,
            "ZK layer: generating state transition proof"
        );

        let prover = self.prover.as_ref().unwrap();
        match prover.prove(&witness) {
            Ok(proof_result) => {
                let total_time = total_start.elapsed();

                info!(
                    step = step,
                    proof_size = proof_result.proof_size(),
                    generation_ms = proof_result.generation_time.as_millis() as u64,
                    verified = proof_result.verified,
                    total_ms = total_time.as_millis() as u64,
                    "ZK layer: proof generated successfully"
                );

                self.stats.proofs_generated += 1;
                if proof_result.verified {
                    self.stats.proofs_verified += 1;
                }
                self.stats.total_proving_time += proof_result.generation_time;

                // Update baseline for next proof
                self.prev_weights = Some(current_weights);

                Ok(Some(ZkCheckpointProofResult {
                    proof: proof_result,
                    step,
                    total_time,
                }))
            }
            Err(e) => {
                // Update baseline even on failure so next proof attempt starts
                // from the correct state
                self.prev_weights = Some(current_weights);
                Err(anyhow!("ZK proof generation failed at step {}: {}", step, e))
            }
        }
    }

    /// Ensures the prover is initialized (lazy init).
    fn ensure_initialized(&mut self) -> Result<()> {
        if self.prover.is_some() {
            return Ok(());
        }

        info!(
            num_weights = self.num_weights,
            "ZK layer: initializing prover (SRS + keygen)..."
        );
        let init_start = Instant::now();

        let mut config = CheckpointProverConfig::new(self.num_weights);
        config.self_verify = self.config.self_verify;

        let prover = CheckpointProver::new(config);

        let init_time = init_start.elapsed();
        self.stats.init_time = Some(init_time);

        info!(
            init_ms = init_time.as_millis() as u64,
            k = prover.config().k,
            "ZK layer: prover initialized"
        );

        self.prover = Some(prover);
        Ok(())
    }
}

/// Converts `FinalWeights` (f64) to `Vec<Halo2Fr>` field elements.
///
/// Uses the same fixed-point scaling as the rest of HELIX: multiply by 10^9
/// and convert to u64. This matches the MPC field element representation.
///
/// Returns an error if any weight is NaN, infinite, or exceeds the
/// representable range (~18.4 billion before scaling overflow).
fn weights_to_field_elements(weights: &FinalWeights, expected_len: usize) -> Result<Vec<Halo2Fr>> {
    let scale = 1_000_000_000u64; // 10^9 for fixed-point representation
    let max_abs = (u64::MAX as f64) / (scale as f64); // ~18.4e9
    let mut flat: Vec<Halo2Fr> = Vec::with_capacity(expected_len);

    for (i, &v) in weights
        .w1
        .iter()
        .chain(weights.b1.iter())
        .chain(weights.w2.iter())
        .chain(weights.b2.iter())
        .enumerate()
    {
        if !v.is_finite() {
            return Err(anyhow!(
                "Weight at index {} is not finite ({}); cannot convert to field element",
                i, v
            ));
        }
        if v.abs() > max_abs {
            return Err(anyhow!(
                "Weight at index {} has magnitude {} which exceeds representable range ({:.1e})",
                i, v, max_abs
            ));
        }
        // Handle negative values: use the field's additive inverse
        if v >= 0.0 {
            flat.push(Halo2Fr::from((v * scale as f64) as u64));
        } else {
            // For negative values, compute the additive inverse in the field
            let pos = Halo2Fr::from((v.abs() * scale as f64) as u64);
            flat.push(-pos);
        }
    }

    debug_assert_eq!(
        flat.len(),
        expected_len,
        "Weight conversion produced {} elements but expected {}",
        flat.len(),
        expected_len
    );

    Ok(flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_weights(d_in: usize, d_hid: usize, d_out: usize) -> FinalWeights {
        let w1 = vec![0.01; d_hid * d_in];
        let b1 = vec![0.0; d_hid];
        let w2 = vec![0.02; d_out * d_hid];
        let b2 = vec![0.0; d_out];
        FinalWeights { w1, b1, w2, b2 }
    }

    #[test]
    fn test_zk_layer_disabled_by_default() {
        let config = ZkProofConfig::default();
        assert!(!config.enabled);

        let layer = ZkProofLayer::new(config, 4, 2, 3);
        assert!(!layer.is_enabled());
    }

    #[test]
    fn test_zk_layer_no_overhead_when_disabled() {
        let config = ZkProofConfig::default(); // enabled = false
        let mut layer = ZkProofLayer::new(config, 4, 2, 3);
        let weights = make_test_weights(4, 2, 3);

        // Should return None immediately, no prover initialization
        let result = layer.process_checkpoint(0, 10, &weights, 1.5).unwrap();
        assert!(result.is_none());
        assert!(layer.prover.is_none()); // Prover never created
        assert_eq!(layer.proofs_generated(), 0);
    }

    #[test]
    fn test_zk_layer_first_checkpoint_is_baseline() {
        let config = ZkProofConfig {
            enabled: true,
            checkpoint_frequency: 1,
            self_verify: false,
        };
        let mut layer = ZkProofLayer::new(config, 4, 2, 3);
        let weights = make_test_weights(4, 2, 3);

        // First checkpoint stores baseline, no proof
        let result = layer.process_checkpoint(0, 10, &weights, 1.5).unwrap();
        assert!(result.is_none());
        assert!(layer.prev_weights.is_some());
        // Prover not yet initialized (lazy)
        assert!(layer.prover.is_none());
    }

    #[test]
    fn test_zk_layer_frequency_filtering() {
        let config = ZkProofConfig {
            enabled: true,
            checkpoint_frequency: 2,
            self_verify: false,
        };
        let mut layer = ZkProofLayer::new(config, 4, 2, 3);
        // Must set baseline for should_prove_checkpoint to return true
        layer.set_baseline_weights(&[0.01; 8], &[0.0; 2], &[0.02; 6], &[0.0; 3]).unwrap();

        // checkpoint_frequency=2 means prove every 2nd checkpoint
        assert!(!layer.should_prove_checkpoint(0)); // (0+1)%2 = 1 != 0
        assert!(layer.should_prove_checkpoint(1));  // (1+1)%2 = 0
        assert!(!layer.should_prove_checkpoint(2)); // (2+1)%2 = 1 != 0
        assert!(layer.should_prove_checkpoint(3));  // (3+1)%2 = 0
    }

    #[test]
    fn test_weights_to_field_elements_length() {
        let weights = make_test_weights(4, 2, 3);
        let expected = 4 * 2 + 2 + 2 * 3 + 3; // 17
        let elements = weights_to_field_elements(&weights, expected).unwrap();
        assert_eq!(elements.len(), expected);
    }

    #[test]
    fn test_weights_to_field_elements_negative() {
        let weights = FinalWeights {
            w1: vec![-0.5, 0.5],
            b1: vec![0.0],
            w2: vec![1.0],
            b2: vec![-1.0],
        };
        let elements = weights_to_field_elements(&weights, 5).unwrap();
        assert_eq!(elements.len(), 5);
        // Negative value should be the field's additive inverse
        // -0.5 * 1e9 = 500_000_000 → field negation
        let pos = Halo2Fr::from(500_000_000u64);
        assert_eq!(elements[0], -pos);
        // Positive value
        assert_eq!(elements[1], Halo2Fr::from(500_000_000u64));
    }

    #[test]
    fn test_weights_to_field_elements_rejects_nan() {
        let weights = FinalWeights {
            w1: vec![f64::NAN],
            b1: vec![],
            w2: vec![],
            b2: vec![],
        };
        assert!(weights_to_field_elements(&weights, 1).is_err());
    }

    #[test]
    fn test_weights_to_field_elements_rejects_infinity() {
        let weights = FinalWeights {
            w1: vec![f64::INFINITY],
            b1: vec![],
            w2: vec![],
            b2: vec![],
        };
        assert!(weights_to_field_elements(&weights, 1).is_err());

        let weights_neg = FinalWeights {
            w1: vec![f64::NEG_INFINITY],
            b1: vec![],
            w2: vec![],
            b2: vec![],
        };
        assert!(weights_to_field_elements(&weights_neg, 1).is_err());
    }

    #[test]
    fn test_zk_proof_generation_real() {
        // This test actually generates a real KZG proof
        let config = ZkProofConfig {
            enabled: true,
            checkpoint_frequency: 1,
            self_verify: true,
        };
        let mut layer = ZkProofLayer::new(config, 4, 2, 3);

        let weights_v1 = FinalWeights {
            w1: vec![0.01; 8],
            b1: vec![0.0; 2],
            w2: vec![0.02; 6],
            b2: vec![0.0; 3],
        };
        let weights_v2 = FinalWeights {
            w1: vec![0.015; 8],
            b1: vec![0.001; 2],
            w2: vec![0.025; 6],
            b2: vec![0.001; 3],
        };

        // Baseline
        let result = layer.process_checkpoint(0, 10, &weights_v1, 2.0).unwrap();
        assert!(result.is_none());

        // Real proof
        let result = layer.process_checkpoint(1, 20, &weights_v2, 1.5).unwrap();
        assert!(result.is_some());

        let zk_result = result.unwrap();
        assert_eq!(zk_result.step, 20);
        assert!(!zk_result.proof.proof_bytes.is_empty());
        assert!(zk_result.proof.proof_size() > 32); // Real KZG proof
        assert!(zk_result.proof.verified);
        assert_eq!(zk_result.proof.public_inputs.len(), 6);
        assert_eq!(layer.proofs_generated(), 1);
        assert_eq!(layer.stats().proofs_verified, 1);
    }
}
