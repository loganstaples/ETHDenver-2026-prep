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
use helix_mpc::field::Fr as MpcFr;
use helix_mpc::mpc_trainer::MPCTrainer;
use helix_mpc::session::transport::LocalTransport;
use helix_prover::{
    CheckpointProofResult, CheckpointProver, CheckpointProverConfig,
};
use helix_prover::halo2curves::bn256::Fr as Halo2Fr;

use crate::display;

/// Result of a ZK checkpoint proof generation attempt.
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

    /// Generates a ZK proof for the weight transition at this checkpoint.
    ///
    /// On the first call, this also initializes the prover (SRS + keygen).
    /// On the first checkpoint, we store the initial weights and skip proof
    /// generation (no previous state to prove transition from).
    ///
    /// Returns `Ok(Some(result))` if a proof was generated, `Ok(None)` if
    /// this was the first checkpoint (no previous weights to prove against).
    pub fn generate_proof(
        &mut self,
        step: u64,
        trainers: &[MPCTrainer<LocalTransport>],
        error_bound_f64: f64,
    ) -> Result<Option<ZkCheckpointResult>> {
        let total_start = Instant::now();

        // Reconstruct current weights from party shares as Halo2Fr
        let current_weights = reconstruct_weights_halo2(trainers, self.num_weights);

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

                // Update previous weights for next checkpoint
                self.prev_weights = Some(current_weights);

                self.results.push(result);

                // Return reference to the last result
                Ok(Some(ZkCheckpointResult {
                    proof: self.results.last().unwrap().proof.clone(),
                    step: self.results.last().unwrap().step,
                    total_time_ms: self.results.last().unwrap().total_time_ms,
                }))
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

    /// Records the initial weights without generating a proof.
    /// Call this at the first MAC checkpoint to establish the baseline.
    #[allow(dead_code)]
    pub fn record_initial_weights(
        &mut self,
        trainers: &[MPCTrainer<LocalTransport>],
    ) {
        let weights = reconstruct_weights_halo2(trainers, self.num_weights);
        self.prev_weights = Some(weights);
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

/// Reconstructs model weights from party shares as Halo2Fr field elements.
///
/// Sums the additive shares across all parties to recover the plaintext
/// weight values. Returns a flattened vector: [W1 | b1 | W2 | b2].
fn reconstruct_weights_halo2(
    trainers: &[MPCTrainer<LocalTransport>],
    expected_len: usize,
) -> Vec<Halo2Fr> {
    let (first_w1, first_b1, first_w2, first_b2) = trainers[0].weight_shares();

    let mut w1_sum: Vec<MpcFr> = first_w1.to_vec();
    let mut b1_sum: Vec<MpcFr> = first_b1.to_vec();
    let mut w2_sum: Vec<MpcFr> = first_w2.to_vec();
    let mut b2_sum: Vec<MpcFr> = first_b2.to_vec();

    for trainer in trainers.iter().skip(1) {
        let (w1, b1, w2, b2) = trainer.weight_shares();
        for (i, s) in w1.iter().enumerate() {
            w1_sum[i] = MpcFr::add(&w1_sum[i], s);
        }
        for (i, s) in b1.iter().enumerate() {
            b1_sum[i] = MpcFr::add(&b1_sum[i], s);
        }
        for (i, s) in w2.iter().enumerate() {
            w2_sum[i] = MpcFr::add(&w2_sum[i], s);
        }
        for (i, s) in b2.iter().enumerate() {
            b2_sum[i] = MpcFr::add(&b2_sum[i], s);
        }
    }

    // Flatten into a single Vec<Halo2Fr>: [W1 | b1 | W2 | b2]
    let mut flat: Vec<Halo2Fr> = Vec::with_capacity(expected_len);
    flat.extend(w1_sum.iter().map(|f| *f.inner()));
    flat.extend(b1_sum.iter().map(|f| *f.inner()));
    flat.extend(w2_sum.iter().map(|f| *f.inner()));
    flat.extend(b2_sum.iter().map(|f| *f.inner()));

    debug_assert_eq!(flat.len(), expected_len,
        "Reconstructed {} weights but expected {}", flat.len(), expected_len);

    flat
}
