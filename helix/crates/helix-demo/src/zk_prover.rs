//! Lazy-loaded ZK proof generation for optional checkpoint verification.
//!
//! When `--zk-proofs` is enabled, this module provides a `LazyZkProver` that
//! defers all expensive operations (SRS loading, circuit key generation) until
//! the first proof is actually requested. When ZK proofs are disabled, none of
//! this code runs—no SRS loading, no circuit setup, no prover invocation.
//!
//! The prover generates `StateTransitionCircuit` proofs showing that model
//! weights transitioned validly between two checkpoints.

use std::time::Instant;

use anyhow::Result;
use helix_circuits::ml::state_transition::StateTransitionWitness;
use helix_prover::{
    CheckpointProofResult, CheckpointProver, CheckpointProverConfig,
};
use helix_prover::halo2curves::bn256::Fr as Halo2Fr;

use crate::display;

/// Result of a ZK checkpoint proof generation attempt.
#[derive(Clone)]
pub struct ZkCheckpointResult {
    /// The proof result from the prover.
    pub proof: CheckpointProofResult,
    /// The step number this proof covers.
    pub step: u64,
    /// Total generation time including any lazy init (milliseconds).
    pub total_time_ms: u64,
}

/// Tracks weight state across checkpoints for ZK proof generation.
///
/// At each ZK-enabled checkpoint, we reconstruct the current weights from
/// party shares. The proof shows the transition from the previous checkpoint's
/// weights to the current ones.
pub struct LazyZkProver {
    /// The actual Halo2 prover, lazily initialized on first use.
    prover: Option<CheckpointProver>,
    /// Weights at the previous checkpoint (flattened Vec<Halo2Fr>).
    prev_weights: Option<Vec<Halo2Fr>>,
    /// Total number of weights in the model.
    num_weights: usize,
    /// All proof results generated during this session.
    results: Vec<ZkCheckpointResult>,
}

impl LazyZkProver {
    /// Creates a new lazy ZK prover for a model with the given dimensions.
    ///
    /// No expensive operations happen here — SRS loading and key generation
    /// are deferred until `generate_proof` is called.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        let num_weights = d_in * d_hid + d_hid + d_hid * d_out + d_out;
        Self {
            prover: None,
            prev_weights: None,
            num_weights,
            results: Vec::new(),
        }
    }

    /// Ensures the prover is initialized. Called lazily on first proof request.
    fn ensure_initialized(&mut self) -> Result<()> {
        if self.prover.is_some() {
            return Ok(());
        }

        let config = CheckpointProverConfig::new(self.num_weights);

        display::zk_prover_init(self.num_weights, config.k);
        let init_start = Instant::now();

        let prover = CheckpointProver::new(config);

        display::zk_prover_init_done(init_start.elapsed().as_millis() as u64);

        self.prover = Some(prover);
        Ok(())
    }

    /// Generate a ZK proof for a checkpoint transition.
    ///
    /// `current_weights` must be pre-reconstructed Halo2Fr field elements
    /// (sum of all parties' additive shares). Only the trusted operator
    /// should call this — weight privacy is maintained because the proof's
    /// public inputs contain only hashes, never raw weights.
    pub fn generate_proof(
        &mut self,
        step: u64,
        current_weights: Vec<Halo2Fr>,
        error_bound_f64: f64,
    ) -> Result<Option<ZkCheckpointResult>> {
        let total_start = Instant::now();

        // First checkpoint: store initial weights, no proof to generate
        if self.prev_weights.is_none() {
            self.prev_weights = Some(current_weights);
            return Ok(None);
        }

        // Lazy-init the prover (SRS + keygen)
        self.ensure_initialized()?;

        let prev = self.prev_weights.as_ref().unwrap();

        // Build witness: old weights → new weights
        let error_bound = {
            // Convert f64 error bound to field element.
            // Use a simple scaling: multiply by 10^9 and take integer part.
            let scaled = (error_bound_f64.abs() * 1e9) as u64;
            Halo2Fr::from(scaled)
        };

        let witness = StateTransitionWitness::new(
            prev.clone(),
            current_weights.clone(),
            error_bound,
        );

        display::zk_proof_start(step);

        let prover = self.prover.as_ref().unwrap();
        match prover.prove(&witness) {
            Ok(proof_result) => {
                let total_time_ms = total_start.elapsed().as_millis() as u64;
                display::zk_proof_success(
                    step,
                    proof_result.proof_size(),
                    proof_result.generation_time.as_millis() as u64,
                    proof_result.verified,
                );
                let result = ZkCheckpointResult {
                    proof: proof_result,
                    step,
                    total_time_ms,
                };
                self.prev_weights = Some(current_weights);
                let cloned = result.clone();
                self.results.push(result);
                Ok(Some(cloned))
            }
            Err(e) => {
                display::zk_proof_failed(step, &e.to_string());
                // Update previous weights even on failure so next proof
                // can still be attempted from the correct base.
                self.prev_weights = Some(current_weights);
                Err(anyhow::anyhow!("ZK proof generation failed: {}", e))
            }
        }
    }

    /// Returns the number of proofs generated.
    pub fn proofs_generated(&self) -> usize {
        self.results.len()
    }

    /// Returns the number of proofs that were self-verified.
    pub fn proofs_verified(&self) -> usize {
        self.results.iter().filter(|r| r.proof.verified).count()
    }

    /// Returns the total proving time in milliseconds.
    pub fn total_proving_time_ms(&self) -> u64 {
        self.results.iter().map(|r| r.proof.generation_time.as_millis() as u64).sum()
    }

    /// Returns all proof results.
    #[allow(dead_code)]
    pub fn results(&self) -> &[ZkCheckpointResult] {
        &self.results
    }
}
