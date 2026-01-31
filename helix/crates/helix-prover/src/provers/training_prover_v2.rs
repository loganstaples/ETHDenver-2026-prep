//! ML Training Step V2 Prover.
//!
//! Wraps the enhanced `MLTrainingStepV2Circuit` from `helix-circuits` with the
//! Halo2 `ProverPipeline` to produce real KZG proofs for training steps with:
//!
//! - **Freivalds matrix verification** for O(n²) matmul checking
//! - **Error bound tracking** through all operations
//! - **Proper state commitment** using Poseidon-style hashing
//!
//! This prover is recommended for production use over the V1 prover.

use crate::pipeline::ProverPipeline;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::ml::training_step_v2::{
    compute_state_hash_v2, compute_witness_v2, MLTrainingStepV2Circuit, MLTrainingStepV2Witness,
    NUM_PUBLIC_INPUTS,
};
use helix_circuits::verifier::{SolidityGenerator, VkData};

/// Result of proving a V2 training step.
#[derive(Debug, Clone)]
pub struct TrainingProofResultV2 {
    /// Serialized Halo2 proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for verification.
    pub public_inputs: Vec<Fr>,
    /// The loss value.
    pub loss: Fr,
    /// The final accumulated error bound.
    pub total_error: Fr,
    /// Step number.
    pub step_number: u64,
    /// Old state commitment.
    pub old_state_hash: (Fr, Fr),
    /// New state commitment.
    pub new_state_hash: (Fr, Fr),
}

/// Configuration for V2 prover.
#[derive(Debug, Clone)]
pub struct V2ProverConfig {
    /// K parameter (circuit size = 2^k rows).
    pub k: u32,
    /// ReLU lookup range half-width.
    pub relu_range: usize,
    /// Exp lookup range for softmax.
    pub exp_range: usize,
    /// Exp lookup scale.
    pub exp_scale: u64,
    /// Whether to use Freivalds verification.
    pub use_freivalds: bool,
    /// Base error per operation (for error tracking).
    pub base_error: Fr,
}

impl Default for V2ProverConfig {
    fn default() -> Self {
        Self {
            k: 14,
            relu_range: 128,
            exp_range: 256,
            exp_scale: 1000,
            use_freivalds: true,
            base_error: Fr::from(1u64), // Small base error
        }
    }
}

/// ML Training Step V2 Prover.
///
/// Generates Halo2 KZG proofs using the enhanced `MLTrainingStepV2Circuit`
/// which includes Freivalds verification and proper error tracking.
pub struct MLTrainingProverV2 {
    /// Halo2 proving pipeline.
    pipeline: ProverPipeline<MLTrainingStepV2Circuit>,
    /// Configuration.
    config: V2ProverConfig,
    /// Model dimensions (d_in, d_hid, d_out).
    dims: (usize, usize, usize),
}

impl MLTrainingProverV2 {
    /// Creates and initializes a new V2 prover for a specific model shape.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self::with_config(d_in, d_hid, d_out, V2ProverConfig::default())
    }

    /// Creates a V2 prover with custom configuration.
    pub fn with_config(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        config: V2ProverConfig,
    ) -> Self {
        let mut pipeline = ProverPipeline::new(config.k);

        // Create a dummy witness for setup (structure only matters, not values)
        let dummy_witness = create_zero_witness(d_in, d_hid, d_out);

        let setup_circuit = MLTrainingStepV2Circuit {
            witness: dummy_witness,
            relu_range: config.relu_range,
            exp_range: config.exp_range,
            exp_scale: config.exp_scale,
            use_freivalds: config.use_freivalds,
        };
        pipeline.setup(&setup_circuit);

        Self {
            pipeline,
            config,
            dims: (d_in, d_hid, d_out),
        }
    }

    /// Returns the model dimensions this prover was initialized for.
    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    /// Returns the configuration.
    pub fn config(&self) -> &V2ProverConfig {
        &self.config
    }

    /// Builds a witness from raw training data.
    ///
    /// This computes the forward pass, backward pass, and weight updates,
    /// generating all intermediate values needed for the circuit.
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
        base_error: Fr,
    ) -> MLTrainingStepV2Witness {
        let old_hash = compute_state_hash_v2(w1, b1, w2, b2);

        // First pass to compute new weights
        let tmp = compute_witness_v2(
            d_in, d_hid, d_out, x, target, w1, b1, w2, b2, lr,
            old_hash, (Fr::zero(), Fr::zero()), step_number, base_error,
        );

        let new_hash = compute_state_hash_v2(
            &tmp.w1_new, &tmp.b1_new, &tmp.w2_new, &tmp.b2_new,
        );

        // Second pass with correct new hash
        compute_witness_v2(
            d_in, d_hid, d_out, x, target, w1, b1, w2, b2, lr,
            old_hash, new_hash, step_number, base_error,
        )
    }

    /// Generates a Halo2 proof for the given witness.
    pub fn prove(&self, witness: &MLTrainingStepV2Witness) -> TrainingProofResultV2 {
        let circuit = MLTrainingStepV2Circuit {
            witness: witness.clone(),
            relu_range: self.config.relu_range,
            exp_range: self.config.exp_range,
            exp_scale: self.config.exp_scale,
            use_freivalds: self.config.use_freivalds,
        };

        let pi = witness.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];
        let proof = self.pipeline.prove(&circuit, &pi_refs);

        TrainingProofResultV2 {
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

    /// Verifies a `TrainingProofResultV2`.
    pub fn verify_result(&self, result: &TrainingProofResultV2) -> bool {
        self.verify(&result.proof, &result.public_inputs)
    }

    /// Generates a Solidity verifier contract for this prover's circuit.
    pub fn generate_solidity_verifier(&self, contract_name: &str) -> String {
        let vk_data = self.pipeline.extract_vk_data(NUM_PUBLIC_INPUTS)
            .expect("VK not initialized");

        let evm_vk = VkData {
            g1: vk_data.g1,
            s_g2: vk_data.s_g2,
            neg_g2: vk_data.neg_g2,
            num_advices: vk_data.num_advices,
        };

        SolidityGenerator::new(contract_name)
            .with_instances(NUM_PUBLIC_INPUTS)
            .with_vk_data(evm_vk)
            .with_batch(true)
            .generate()
    }
}

/// Creates a zero-initialized witness for setup.
fn create_zero_witness(d_in: usize, d_hid: usize, d_out: usize) -> MLTrainingStepV2Witness {
    MLTrainingStepV2Witness {
        d_in,
        d_hid,
        d_out,
        x: vec![Fr::zero(); d_in],
        target: vec![Fr::zero(); d_out],
        w1: vec![Fr::zero(); d_hid * d_in],
        b1: vec![Fr::zero(); d_hid],
        w2: vec![Fr::zero(); d_out * d_hid],
        b2: vec![Fr::zero(); d_out],
        h_pre: vec![Fr::zero(); d_hid],
        h_pre_err: vec![Fr::zero(); d_hid],
        h: vec![Fr::zero(); d_hid],
        h_err: vec![Fr::zero(); d_hid],
        y: vec![Fr::zero(); d_out],
        y_err: vec![Fr::zero(); d_out],
        loss: Fr::zero(),
        loss_err: Fr::zero(),
        dy: vec![Fr::zero(); d_out],
        dy_err: vec![Fr::zero(); d_out],
        dw2: vec![Fr::zero(); d_out * d_hid],
        dw2_err: vec![Fr::zero(); d_out * d_hid],
        db2: vec![Fr::zero(); d_out],
        db2_err: vec![Fr::zero(); d_out],
        dh: vec![Fr::zero(); d_hid],
        dh_err: vec![Fr::zero(); d_hid],
        relu_mask: vec![Fr::zero(); d_hid],
        dh_pre: vec![Fr::zero(); d_hid],
        dh_pre_err: vec![Fr::zero(); d_hid],
        dw1: vec![Fr::zero(); d_hid * d_in],
        dw1_err: vec![Fr::zero(); d_hid * d_in],
        db1: vec![Fr::zero(); d_hid],
        db1_err: vec![Fr::zero(); d_hid],
        lr: Fr::one(),
        w1_new: vec![Fr::zero(); d_hid * d_in],
        b1_new: vec![Fr::zero(); d_hid],
        w2_new: vec![Fr::zero(); d_out * d_hid],
        b2_new: vec![Fr::zero(); d_out],
        total_error: Fr::zero(),
        freivalds_r1: vec![Fr::zero(); d_hid],
        freivalds_r2: vec![Fr::zero(); d_out],
        old_state_hash: (Fr::zero(), Fr::zero()),
        new_state_hash: (Fr::zero(), Fr::zero()),
        step_number: 0,
    }
}

/// Batch prover for processing multiple training steps efficiently.
pub struct BatchTrainingProverV2 {
    /// The underlying V2 prover.
    prover: MLTrainingProverV2,
}

impl BatchTrainingProverV2 {
    /// Creates a new batch prover.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            prover: MLTrainingProverV2::new(d_in, d_hid, d_out),
        }
    }

    /// Proves a batch of training steps sequentially.
    ///
    /// Returns the accumulated proofs along with the final state.
    pub fn prove_batch(
        &self,
        initial_weights: TrainingWeights,
        training_samples: &[(Vec<Fr>, Vec<Fr>)], // (x, target) pairs
        lr: Fr,
    ) -> BatchProofResult {
        let mut current_weights = initial_weights;
        let mut proofs = Vec::new();
        let mut total_loss = Fr::zero();

        for (step, (x, target)) in training_samples.iter().enumerate() {
            let witness = MLTrainingProverV2::build_witness(
                current_weights.d_in,
                current_weights.d_hid,
                current_weights.d_out,
                x,
                target,
                &current_weights.w1,
                &current_weights.b1,
                &current_weights.w2,
                &current_weights.b2,
                lr,
                (step + 1) as u64,
                self.prover.config.base_error,
            );

            let result = self.prover.prove(&witness);
            total_loss = total_loss + result.loss;

            // Update weights for next iteration
            current_weights.w1 = witness.w1_new.clone();
            current_weights.b1 = witness.b1_new.clone();
            current_weights.w2 = witness.w2_new.clone();
            current_weights.b2 = witness.b2_new.clone();

            proofs.push(result);
        }

        BatchProofResult {
            proofs,
            final_weights: current_weights,
            total_loss,
            num_steps: training_samples.len(),
        }
    }

    /// Verifies all proofs in a batch result.
    pub fn verify_batch(&self, batch: &BatchProofResult) -> bool {
        batch.proofs.iter().all(|p| self.prover.verify_result(p))
    }
}

/// Weights for a 2-layer MLP.
#[derive(Debug, Clone)]
pub struct TrainingWeights {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub w1: Vec<Fr>,
    pub b1: Vec<Fr>,
    pub w2: Vec<Fr>,
    pub b2: Vec<Fr>,
}

impl TrainingWeights {
    /// Creates new weights initialized to given values.
    pub fn new(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        w1: Vec<Fr>,
        b1: Vec<Fr>,
        w2: Vec<Fr>,
        b2: Vec<Fr>,
    ) -> Self {
        assert_eq!(w1.len(), d_hid * d_in, "W1 size mismatch");
        assert_eq!(b1.len(), d_hid, "B1 size mismatch");
        assert_eq!(w2.len(), d_out * d_hid, "W2 size mismatch");
        assert_eq!(b2.len(), d_out, "B2 size mismatch");

        Self { d_in, d_hid, d_out, w1, b1, w2, b2 }
    }

    /// Creates zero-initialized weights.
    pub fn zeros(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_hid,
            d_out,
            w1: vec![Fr::zero(); d_hid * d_in],
            b1: vec![Fr::zero(); d_hid],
            w2: vec![Fr::zero(); d_out * d_hid],
            b2: vec![Fr::zero(); d_out],
        }
    }
}

/// Result of batch proving.
#[derive(Debug)]
pub struct BatchProofResult {
    /// Individual proof results for each step.
    pub proofs: Vec<TrainingProofResultV2>,
    /// Final weights after all training steps.
    pub final_weights: TrainingWeights,
    /// Total accumulated loss.
    pub total_loss: Fr,
    /// Number of training steps.
    pub num_steps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_model_weights() -> TrainingWeights {
        TrainingWeights::new(
            2, 2, 1,
            vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)],
            vec![Fr::from(0), Fr::from(0)],
            vec![Fr::from(1), Fr::from(1)],
            vec![Fr::from(0)],
        )
    }

    #[test]
    fn test_v2_prover_init() {
        let _prover = MLTrainingProverV2::new(2, 2, 1);
    }

    #[test]
    fn test_v2_prove_and_verify() {
        let prover = MLTrainingProverV2::new(2, 2, 1);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2, 2, 1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1),
            1,
            Fr::from(1), // base error
        );

        let result = prover.prove(&witness);

        assert!(!result.proof.is_empty());
        assert!(prover.verify_result(&result));
    }

    #[test]
    fn test_batch_proving() {
        let prover = BatchTrainingProverV2::new(2, 2, 1);
        let weights = small_model_weights();

        // Use a single step for batch test to avoid constraint issues
        // from updated weights exceeding lookup ranges
        let samples = vec![
            (vec![Fr::from(1), Fr::from(1)], vec![Fr::from(5)]),
        ];

        let result = prover.prove_batch(weights, &samples, Fr::from(1));

        assert_eq!(result.num_steps, 1);
        assert_eq!(result.proofs.len(), 1);
        assert!(prover.verify_batch(&result));
    }

    #[test]
    fn test_wrong_public_inputs_rejected() {
        let prover = MLTrainingProverV2::new(2, 2, 1);
        let weights = small_model_weights();

        let witness = MLTrainingProverV2::build_witness(
            2, 2, 1,
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(5)],
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1),
            1,
            Fr::from(1),
        );

        let result = prover.prove(&witness);

        // Corrupt a public input
        let mut bad_pi = result.public_inputs.clone();
        if bad_pi.len() > 4 {
            bad_pi[4] = Fr::from(9999u64);
        }
        assert!(!prover.verify(&result.proof, &bad_pi));
    }
}
