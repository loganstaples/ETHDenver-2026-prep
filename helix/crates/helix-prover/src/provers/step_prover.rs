//! Training Step Prover.
//!
//! Orchestrates the full proving pipeline for a single training step:
//! chunk → parallel prove → aggregate → IVC chain.
//!
//! Two proving modes are supported:
//!
//! 1. **Chunked mode** (`prove_training_step`): Splits the computation into
//!    chunks, proves each with `IVCStepCircuit`, aggregates, then chains via IVC.
//!    Suitable for very large models that don't fit in a single circuit.
//!
//! 2. **ML circuit mode** (`prove_ml_training_step`): Uses the real
//!    `MLTrainingStepV2Circuit` to generate a single Halo2 KZG proof covering
//!    forward + backward + weight update for a 2-layer MLP, then chains
//!    the result into the IVC sequence.

use crate::aggregation::{AggregatedProof, ProofAggregator};
use crate::chunking::{ChunkId, ChunkingConfig, ComputationChunker};
use crate::ivc::{IVCConfig, IVCProver, IVCStep};
use crate::parallel::{ChunkProof, ParallelConfig, ParallelProver};
use crate::provers::training_prover::{MLTrainingProver, TrainingProofResult};
use helix_circuits::halo2curves::bn256::Fr;

/// Result of proving a complete training step.
#[derive(Debug)]
pub struct TrainingStepProof {
    /// Aggregated proof for the forward/backward/update chunks.
    pub aggregated_proof: AggregatedProof,
    /// IVC state proof linking this step to the chain.
    pub ivc_proof: Vec<u8>,
    /// Total number of chunks proved.
    pub num_chunks: usize,
    /// Total error bound for this step.
    pub total_error_bound: f64,
    /// Step number in the training sequence.
    pub step_number: u64,
}

/// Result of proving a training step with the real ML circuit.
#[derive(Debug)]
pub struct MLTrainingStepProof {
    /// The Halo2 KZG proof from MLTrainingStepV2Circuit.
    pub ml_proof: TrainingProofResult,
    /// IVC state proof linking this step to the chain.
    pub ivc_proof: Vec<u8>,
    /// Step number in the training sequence.
    pub step_number: u64,
}

/// Training data for a single step (weights, inputs, targets).
#[derive(Debug, Clone)]
pub struct TrainingStepData {
    /// Input dimension.
    pub d_in: usize,
    /// Hidden dimension.
    pub d_hid: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Input features (d_in elements).
    pub x: Vec<Fr>,
    /// Target values (d_out elements).
    pub target: Vec<Fr>,
    /// Layer 1 weights (d_hid × d_in, row-major).
    pub w1: Vec<Fr>,
    /// Layer 1 biases (d_hid elements).
    pub b1: Vec<Fr>,
    /// Layer 2 weights (d_out × d_hid, row-major).
    pub w2: Vec<Fr>,
    /// Layer 2 biases (d_out elements).
    pub b2: Vec<Fr>,
    /// Learning rate (quantised).
    pub lr: Fr,
}

/// Configuration for the training step prover.
#[derive(Debug, Clone)]
pub struct TrainingStepConfig {
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Number of model parameters (for weight update).
    pub num_parameters: usize,
    /// Chunking configuration.
    pub chunking: ChunkingConfig,
    /// Parallel proving configuration.
    pub parallel: ParallelConfig,
    /// IVC configuration.
    pub ivc: IVCConfig,
}

impl Default for TrainingStepConfig {
    fn default() -> Self {
        Self {
            num_layers: 8,
            num_parameters: 1000,
            chunking: ChunkingConfig::default(),
            parallel: ParallelConfig::default(),
            ivc: IVCConfig::default(),
        }
    }
}

/// Training step prover — orchestrates chunk → prove → aggregate → IVC.
///
/// Also holds an optional `MLTrainingProver` for direct ML circuit proofs.
pub struct TrainingStepProver {
    config: TrainingStepConfig,
    ivc_prover: IVCProver,
    step_count: u64,
    /// Real ML training prover (lazily initialised on first ML proof request).
    ml_prover: Option<MLTrainingProver>,
}

impl TrainingStepProver {
    /// Creates a new training step prover.
    pub fn new(initial_commitment: [u8; 32]) -> Self {
        Self::with_config(initial_commitment, TrainingStepConfig::default())
    }

    /// Creates a prover with custom configuration.
    pub fn with_config(initial_commitment: [u8; 32], config: TrainingStepConfig) -> Self {
        let ivc_prover = IVCProver::with_config(initial_commitment, config.ivc.clone());
        Self {
            config,
            ivc_prover,
            step_count: 0,
            ml_prover: None,
        }
    }

    /// Ensures the ML prover is initialised and returns a reference.
    fn ensure_ml_prover(
        &mut self,
        k: u32,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> Result<&MLTrainingProver, String> {
        if self.ml_prover.is_none() {
            self.ml_prover = Some(MLTrainingProver::new(k, d_in, d_hid, d_out));
        }
        self.ml_prover.as_ref().ok_or_else(|| "ML prover initialization failed".to_string())
    }

    /// Proves a single training step (forward + backward + weight update).
    ///
    /// `input_commitment` is the state before this step.
    /// `output_commitment` is the state after this step.
    pub fn prove_training_step(
        &mut self,
        input_commitment: [u8; 32],
        output_commitment: [u8; 32],
    ) -> Result<TrainingStepProof, String> {
        self.step_count += 1;

        // 1. Chunk the computation
        let mut chunker = ComputationChunker::with_config(self.config.chunking.clone());
        let forward_ids = chunker.chunk_forward_pass(self.config.num_layers, input_commitment);
        let backward_ids = chunker.chunk_backward_pass(
            self.config.num_layers,
            output_commitment,
            &forward_ids,
        );
        let _weight_id = chunker.chunk_weight_update(self.config.num_parameters, &backward_ids);

        let all_chunks: Vec<_> = chunker.chunks().to_vec();
        let num_chunks = all_chunks.len();

        // 2. Prove chunks in parallel
        let parallel_prover = ParallelProver::with_config(self.config.parallel.clone());
        parallel_prover.submit_batch(all_chunks);
        parallel_prover.start();
        let chunk_proofs = parallel_prover.wait_all()
            .map_err(|e| format!("Batch proving failed: {e}"))?;
        parallel_prover.stop();

        // 3. Aggregate proofs
        let mut aggregator = ProofAggregator::new();
        aggregator.add_proofs(chunk_proofs);
        let aggregated_proof = aggregator
            .aggregate_single()
            .ok_or("Aggregation produced no proof")?;

        let total_error_bound = aggregated_proof.total_error_bound;

        // 4. Derive computation hash for IVC step
        let computation_hash = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(aggregated_proof.root_commitment);
            h.update(self.step_count.to_le_bytes());
            let digest: [u8; 32] = h.finalize().into();
            digest
        };

        // 5. Add IVC step
        let ivc_step = IVCStep {
            step: self.step_count,
            input_state: input_commitment,
            output_state: output_commitment,
            computation_hash,
            step_error: total_error_bound,
            proof: aggregated_proof.proof.clone(),
        };
        self.ivc_prover.add_step(ivc_step)?;

        // 6. Finalize IVC proof for this step
        let ivc_proof = self
            .ivc_prover
            .finalize()
            .unwrap_or_else(|e| {
                tracing::error!("IVC proof finalization failed for chunked training step: {e}");
                Vec::new()
            });

        Ok(TrainingStepProof {
            aggregated_proof,
            ivc_proof,
            num_chunks,
            total_error_bound,
            step_number: self.step_count,
        })
    }

    /// Proves a training step using the real `MLTrainingStepV2Circuit`.
    ///
    /// This generates a Halo2 KZG proof that the forward pass, backward pass,
    /// and weight update were computed correctly for a 2-layer MLP, then chains
    /// the result into the IVC sequence.
    ///
    /// `k` controls the circuit size (2^k rows). 14 comfortably fits a 4×8×2
    /// model.
    pub fn prove_ml_training_step(
        &mut self,
        data: &TrainingStepData,
        k: u32,
    ) -> Result<MLTrainingStepProof, String> {
        self.step_count += 1;
        let current_step = self.step_count;

        // 1. Build witness and generate ML proof.
        let ml_prover = self.ensure_ml_prover(k, data.d_in, data.d_hid, data.d_out)?;

        let witness = MLTrainingProver::build_witness(
            data.d_in,
            data.d_hid,
            data.d_out,
            &data.x,
            &data.target,
            &data.w1,
            &data.b1,
            &data.w2,
            &data.b2,
            data.lr,
            current_step,
        );
        let ml_result = ml_prover.prove(&witness);

        // Verify our own proof before chaining (sanity check).
        if !ml_prover.verify_result(&ml_result) {
            return Err("ML proof self-verification failed".to_string());
        }

        // 2. Derive commitments from the ML proof's state hashes for IVC.
        let input_commitment = fr_pair_to_bytes(ml_result.old_state_hash);
        let output_commitment = fr_pair_to_bytes(ml_result.new_state_hash);

        // On the first ML step, re-anchor the IVC chain to the ML state hash
        // (the initial commitment passed to `new()` is a placeholder).
        if self.step_count == 1 {
            self.ivc_prover.reset(input_commitment);
        }


        // 3. Compute a computation hash from the ML proof.
        let computation_hash = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(&ml_result.proof);
            h.update(self.step_count.to_le_bytes());
            let digest: [u8; 32] = h.finalize().into();
            digest
        };

        // 4. Add IVC step linking this ML proof to the chain.
        let ivc_step = IVCStep {
            step: self.step_count,
            input_state: input_commitment,
            output_state: output_commitment,
            computation_hash,
            step_error: 0.0, // ML circuit tracks error bound in public inputs
            proof: ml_result.proof.clone(),
        };
        self.ivc_prover.add_step(ivc_step)?;

        // 5. Finalize IVC proof.
        let ivc_proof = self
            .ivc_prover
            .finalize()
            .unwrap_or_else(|e| {
                tracing::error!("IVC proof finalization failed for ML training step: {e}");
                Vec::new()
            });

        Ok(MLTrainingStepProof {
            ml_proof: ml_result,
            ivc_proof,
            step_number: self.step_count,
        })
    }

    /// Returns the current IVC state.
    pub fn ivc_state(&self) -> &crate::ivc::IVCState {
        self.ivc_prover.state()
    }

    /// Returns the number of steps proved so far.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Verifies an ML training proof using this prover's own pipeline (same SRS/VK).
    ///
    /// KZG verification requires the same SRS and verification key that was
    /// used for proof generation. This method uses the internal ML prover's
    /// pipeline, which guarantees matching keys.
    pub fn verify_ml_proof(&self, result: &TrainingProofResult) -> bool {
        match &self.ml_prover {
            Some(prover) => prover.verify_result(result),
            None => false,
        }
    }
}

/// Converts a pair of `Fr` elements (lo, hi) into a 32-byte commitment.
///
/// Takes the first 16 bytes of each field element's little-endian representation.
fn fr_pair_to_bytes(pair: (Fr, Fr)) -> [u8; 32] {
    use helix_circuits::halo2curves::ff::PrimeField;
    let lo_repr = pair.0.to_repr();
    let hi_repr = pair.1.to_repr();
    let lo_bytes = lo_repr.as_ref();
    let hi_bytes = hi_repr.as_ref();

    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&lo_bytes[..16]);
    out[16..].copy_from_slice(&hi_bytes[..16]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_step_prover_init() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prover = TrainingStepProver::new([0u8; 32]);
        assert_eq!(prover.step_count(), 0);
    }

    #[test]
    fn test_prove_single_training_step() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = TrainingStepConfig {
            num_layers: 4,
            num_parameters: 100,
            chunking: ChunkingConfig {
                layers_per_chunk: 2,
                ..Default::default()
            },
            parallel: ParallelConfig {
                num_threads: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut prover = TrainingStepProver::with_config([0u8; 32], config);

        let result = prover.prove_training_step([0u8; 32], [1u8; 32]);
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.step_number, 1);
        assert!(proof.num_chunks > 0);
        assert!(!proof.aggregated_proof.proof.is_empty());
        assert!(!proof.ivc_proof.is_empty());
    }

    #[test]
    fn test_prove_ml_training_step() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let data = TrainingStepData {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            x: vec![Fr::from(1), Fr::from(1)],
            target: vec![Fr::from(5)],
            w1: vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)],
            b1: vec![Fr::from(0), Fr::from(0)],
            w2: vec![Fr::from(1), Fr::from(1)],
            b2: vec![Fr::from(0)],
            lr: Fr::from(1),
        };

        let mut prover = TrainingStepProver::new([0u8; 32]);
        let result = prover.prove_ml_training_step(&data, 12);
        assert!(result.is_ok(), "ML training step failed: {:?}", result.err());

        let proof = result.unwrap();
        assert_eq!(proof.step_number, 1);
        assert!(!proof.ml_proof.proof.is_empty());
        assert!(!proof.ivc_proof.is_empty());
    }

    #[test]
    fn test_ml_proof_chains_into_ivc() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Verify that a single ML step correctly anchors into IVC state.
        let data = TrainingStepData {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            x: vec![Fr::from(1), Fr::from(1)],
            target: vec![Fr::from(5)],
            w1: vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)],
            b1: vec![Fr::from(0), Fr::from(0)],
            w2: vec![Fr::from(1), Fr::from(1)],
            b2: vec![Fr::from(0)],
            lr: Fr::from(1),
        };

        let mut prover = TrainingStepProver::new([0u8; 32]);
        let result = prover
            .prove_ml_training_step(&data, 12)
            .expect("prove failed");

        // IVC state reflects the step.
        assert_eq!(prover.ivc_state().step, 1);
        assert!(prover.ivc_state().proof.is_some());

        // The IVC state commitment should be the ML proof's new state hash.
        let expected_commitment =
            fr_pair_to_bytes(result.ml_proof.new_state_hash);
        assert_eq!(prover.ivc_state().state_commitment, expected_commitment);
    }

    #[test]
    fn test_ml_proof_independent_verification() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Generate an ML proof and verify it using the prover's own pipeline.
        // KZG verification requires the same SRS/VK, so we verify through the
        // step prover rather than creating a separate prover instance.
        let data = TrainingStepData {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            x: vec![Fr::from(1), Fr::from(1)],
            target: vec![Fr::from(5)],
            w1: vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)],
            b1: vec![Fr::from(0), Fr::from(0)],
            w2: vec![Fr::from(1), Fr::from(1)],
            b2: vec![Fr::from(0)],
            lr: Fr::from(1),
        };

        let mut prover = TrainingStepProver::new([0u8; 32]);
        let result = prover
            .prove_ml_training_step(&data, 12)
            .expect("prove failed");

        // Verify using the step prover's own ML prover (same SRS/VK).
        assert!(prover.verify_ml_proof(&result.ml_proof));

        // Tamper with proof and confirm rejection.
        let mut bad = result.ml_proof.clone();
        bad.public_inputs[4] = Fr::from(9999u64); // corrupt loss
        assert!(!prover.verify_ml_proof(&bad));
    }
}
