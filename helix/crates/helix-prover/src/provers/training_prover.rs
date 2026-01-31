//! ML Training Step Prover.
//!
//! Wraps the `MLTrainingStepCircuit` from `helix-circuits` with the Halo2
//! `ProverPipeline` to produce and verify real KZG proofs for training steps.
//!
//! Usage:
//! ```ignore
//! let prover = MLTrainingProver::new(K);
//!
//! // Build witness from training data
//! let witness = MLTrainingProver::build_witness(
//!     d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr, step,
//! );
//!
//! // Generate proof
//! let result = prover.prove(&witness);
//!
//! // Verify proof
//! assert!(prover.verify(&result.proof, &result.public_inputs));
//! ```

use crate::pipeline::ProverPipeline;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::ml::training_step::{
    compute_state_hash, compute_witness, MLTrainingStepCircuit, MLTrainingStepWitness,
    NUM_PUBLIC_INPUTS,
};

/// Result of proving a training step.
#[derive(Debug, Clone)]
pub struct TrainingProofResult {
    /// Serialised Halo2 proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for verification (NUM_PUBLIC_INPUTS elements).
    pub public_inputs: Vec<Fr>,
    /// The loss value (also in public_inputs[4]).
    pub loss: Fr,
    /// The accumulated error bound.
    pub total_error: Fr,
    /// Step number.
    pub step_number: u64,
    /// Old state commitment.
    pub old_state_hash: (Fr, Fr),
    /// New state commitment.
    pub new_state_hash: (Fr, Fr),
}

/// ML Training Step Prover.
///
/// Generates real Halo2 KZG proofs for ML training steps using the
/// `MLTrainingStepCircuit`.
pub struct MLTrainingProver {
    /// Halo2 proving pipeline (params, pk, vk).
    pipeline: ProverPipeline<MLTrainingStepCircuit>,
    /// ReLU lookup table half-range used in the circuit.
    relu_range: usize,
}

impl MLTrainingProver {
    /// Creates and initialises a new prover.
    ///
    /// `k` controls the circuit size: 2^k rows.  A value of 14 (16 384 rows)
    /// comfortably fits a 4×8×2 MLP training step.
    pub fn new(k: u32) -> Self {
        Self::with_relu_range(k, 128)
    }

    /// Creates a prover with a custom ReLU lookup range.
    pub fn with_relu_range(k: u32, relu_range: usize) -> Self {
        let mut pipeline = ProverPipeline::new(k);
        let empty = MLTrainingStepCircuit {
            relu_range,
            ..Default::default()
        };
        pipeline.setup(&empty);
        Self {
            pipeline,
            relu_range,
        }
    }

    /// Builds a witness from raw training data (all in `Fr`).
    ///
    /// This is a convenience wrapper around `compute_witness` + `compute_state_hash`
    /// that correctly computes both old and new state hashes.
    pub fn build_witness(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        x: &[Fr],
        target: &[Fr],
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        lr: Fr,
        step_number: u64,
    ) -> MLTrainingStepWitness {
        let old_hash = compute_state_hash(w1, b1, w2, b2);

        // First pass to compute new weights.
        let tmp = compute_witness(
            d_in,
            d_hid,
            d_out,
            x,
            target,
            w1,
            b1,
            w2,
            b2,
            lr,
            old_hash,
            (Fr::ZERO, Fr::ZERO),
            step_number,
        );

        let new_hash = compute_state_hash(&tmp.w1_new, &tmp.b1_new, &tmp.w2_new, &tmp.b2_new);

        // Second pass with correct new hash.
        compute_witness(
            d_in,
            d_hid,
            d_out,
            x,
            target,
            w1,
            b1,
            w2,
            b2,
            lr,
            old_hash,
            new_hash,
            step_number,
        )
    }

    /// Generates a Halo2 proof for the given witness.
    pub fn prove(&self, witness: &MLTrainingStepWitness) -> TrainingProofResult {
        let circuit = MLTrainingStepCircuit {
            witness: witness.clone(),
            relu_range: self.relu_range,
        };

        let pi = witness.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];
        let proof = self.pipeline.prove(&circuit, &pi_refs);

        TrainingProofResult {
            proof,
            public_inputs: pi.clone(),
            loss: witness.loss,
            total_error: witness.total_error,
            step_number: witness.step_number,
            old_state_hash: witness.old_state_hash,
            new_state_hash: witness.new_state_hash,
        }
    }

    /// Verifies a proof against the given public inputs.
    pub fn verify(&self, proof: &[u8], public_inputs: &[Fr]) -> bool {
        let pi_refs: Vec<&[Fr]> = vec![public_inputs];
        self.pipeline.verify(proof, &pi_refs)
    }

    /// Verifies a `TrainingProofResult`.
    pub fn verify_result(&self, result: &TrainingProofResult) -> bool {
        self.verify(&result.proof, &result.public_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_model_witness() -> MLTrainingStepWitness {
        MLTrainingProver::build_witness(
            2,
            2,
            1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &[Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)],
            &[Fr::from(0), Fr::from(0)],
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(0)],
            Fr::from(1),
            1,
        )
    }

    #[test]
    fn test_prover_init() {
        let _prover = MLTrainingProver::new(14);
    }

    #[test]
    fn test_prove_and_verify() {
        let prover = MLTrainingProver::new(14);
        let witness = small_model_witness();
        let result = prover.prove(&witness);

        assert!(!result.proof.is_empty());
        assert_eq!(result.public_inputs.len(), NUM_PUBLIC_INPUTS);
        assert!(prover.verify_result(&result));
    }

    #[test]
    fn test_wrong_public_inputs_rejected() {
        let prover = MLTrainingProver::new(14);
        let witness = small_model_witness();
        let result = prover.prove(&witness);

        // Corrupt a public input and verify it's rejected.
        let mut bad_pi = result.public_inputs.clone();
        bad_pi[4] = Fr::from(9999u64); // corrupt loss
        assert!(!prover.verify(&result.proof, &bad_pi));
    }

    #[test]
    fn test_4x4x2_model() {
        let d_in = 4;
        let d_hid = 4;
        let d_out = 2;

        let w1: Vec<Fr> = (0..d_hid * d_in)
            .map(|i| Fr::from((i % 3 + 1) as u64))
            .collect();
        let b1 = vec![Fr::from(0); d_hid];
        let w2: Vec<Fr> = (0..d_out * d_hid)
            .map(|i| Fr::from((i % 2 + 1) as u64))
            .collect();
        let b2 = vec![Fr::from(0); d_out];
        let x: Vec<Fr> = (0..d_in).map(|i| Fr::from((i + 1) as u64)).collect();
        let target: Vec<Fr> = (0..d_out).map(|_| Fr::from(10u64)).collect();

        let prover = MLTrainingProver::new(14);
        let witness =
            MLTrainingProver::build_witness(d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, Fr::from(1), 1);
        let result = prover.prove(&witness);

        assert!(prover.verify_result(&result));
    }
}
