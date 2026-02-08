//! Real ML Training Pipeline with ZK Proof Generation (V2).
//!
//! Uses `MLTrainingProverV2` for EVM-compatible KZG proofs:
//! 1. Native forward pass (matmul + bias + ReLU + matmul + bias)
//! 2. MSE loss computation
//! 3. Native backward pass (full gradient computation)
//! 4. SGD weight update
//! 5. Quantisation to `Fr` field elements
//! 6. ZK proof generation via `MLTrainingProverV2` (EVM-formatted output)
//!
//! The model is a 2-layer MLP: `x → W1·x + b1 → ReLU → W2·h + b2 → output`.

use helix_prover::halo2curves::bn256::Fr;
use helix_prover::MLTrainingProverV2;
use sha2::{Digest, Sha256};

// ──────────────────────────────────────────────────────────────
// Model representation (f64 for native computation)
// ──────────────────────────────────────────────────────────────

/// A 2-layer MLP: `x → W1·x + b1 → ReLU → W2·h + b2 → y`.
#[derive(Debug, Clone)]
pub struct MlpModel {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    /// W1: [d_hid × d_in] row-major.
    pub w1: Vec<f64>,
    /// b1: [d_hid].
    pub b1: Vec<f64>,
    /// W2: [d_out × d_hid] row-major.
    pub w2: Vec<f64>,
    /// b2: [d_out].
    pub b2: Vec<f64>,
}

impl MlpModel {
    /// Creates a model with small random weights (deterministic from seed).
    pub fn new_random(d_in: usize, d_hid: usize, d_out: usize, seed: u64) -> Self {
        // Simple LCG PRNG for deterministic init.
        let mut rng = seed;
        let mut next = || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-0.5, 0.5] scaled by 1/sqrt(fan_in) later.
            ((rng >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };

        // Xavier-style init: scale ~ 1/sqrt(fan_in).
        let scale_w1 = 1.0 / (d_in as f64).sqrt();
        let scale_w2 = 1.0 / (d_hid as f64).sqrt();

        let w1: Vec<f64> = (0..d_hid * d_in).map(|_| next() * scale_w1).collect();
        let b1 = vec![0.0; d_hid];
        let w2: Vec<f64> = (0..d_out * d_hid).map(|_| next() * scale_w2).collect();
        let b2 = vec![0.0; d_out];

        Self {
            d_in,
            d_hid,
            d_out,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Creates a model with given weights.
    pub fn new(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        w1: Vec<f64>,
        b1: Vec<f64>,
        w2: Vec<f64>,
        b2: Vec<f64>,
    ) -> Self {
        assert_eq!(w1.len(), d_hid * d_in);
        assert_eq!(b1.len(), d_hid);
        assert_eq!(w2.len(), d_out * d_hid);
        assert_eq!(b2.len(), d_out);
        Self {
            d_in,
            d_hid,
            d_out,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// Computes SHA-256 commitment over all weights.
    pub fn commitment(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for &v in &self.w1 {
            h.update(v.to_le_bytes());
        }
        for &v in &self.b1 {
            h.update(v.to_le_bytes());
        }
        for &v in &self.w2 {
            h.update(v.to_le_bytes());
        }
        for &v in &self.b2 {
            h.update(v.to_le_bytes());
        }
        h.finalize().into()
    }
}

// ──────────────────────────────────────────────────────────────
// Native forward / backward pass
// ──────────────────────────────────────────────────────────────

/// Intermediate activations from a forward pass (needed for backward).
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Pre-activation of layer 1: z1 = W1·x + b1  [d_hid].
    pub z1: Vec<f64>,
    /// Post-activation of layer 1: h = ReLU(z1)  [d_hid].
    pub h: Vec<f64>,
    /// Output: y = W2·h + b2  [d_out].
    pub y: Vec<f64>,
    /// MSE loss.
    pub loss: f64,
}

/// Gradients for all model parameters.
#[derive(Debug, Clone)]
pub struct Gradients {
    pub dw1: Vec<f64>,
    pub db1: Vec<f64>,
    pub dw2: Vec<f64>,
    pub db2: Vec<f64>,
}

/// Runs the forward pass and computes MSE loss.
pub fn forward(model: &MlpModel, x: &[f64], target: &[f64]) -> ForwardResult {
    let (d_in, d_hid, d_out) = (model.d_in, model.d_hid, model.d_out);

    // z1 = W1 · x + b1
    let mut z1 = model.b1.clone();
    for i in 0..d_hid {
        for j in 0..d_in {
            z1[i] += model.w1[i * d_in + j] * x[j];
        }
    }

    // h = ReLU(z1)
    let h: Vec<f64> = z1.iter().map(|&v| v.max(0.0)).collect();

    // y = W2 · h + b2
    let mut y = model.b2.clone();
    for i in 0..d_out {
        for j in 0..d_hid {
            y[i] += model.w2[i * d_hid + j] * h[j];
        }
    }

    // MSE loss = sum((y - target)^2) / d_out
    let loss: f64 = y
        .iter()
        .zip(target.iter())
        .map(|(&yi, &ti)| (yi - ti).powi(2))
        .sum::<f64>()
        / d_out as f64;

    ForwardResult { z1, h, y, loss }
}

/// Computes gradients via backpropagation.
pub fn backward(
    model: &MlpModel,
    x: &[f64],
    target: &[f64],
    fwd: &ForwardResult,
) -> Gradients {
    let (d_in, d_hid, d_out) = (model.d_in, model.d_hid, model.d_out);

    // dL/dy = 2 * (y - target) / d_out  (MSE gradient)
    let dy: Vec<f64> = fwd
        .y
        .iter()
        .zip(target.iter())
        .map(|(&yi, &ti)| 2.0 * (yi - ti) / d_out as f64)
        .collect();

    // dL/dW2[i][j] = dy[i] * h[j]
    let mut dw2 = vec![0.0; d_out * d_hid];
    for i in 0..d_out {
        for j in 0..d_hid {
            dw2[i * d_hid + j] = dy[i] * fwd.h[j];
        }
    }

    // dL/db2 = dy
    let db2 = dy.clone();

    // dL/dh = W2^T · dy
    let mut dh = vec![0.0; d_hid];
    for j in 0..d_hid {
        for i in 0..d_out {
            dh[j] += model.w2[i * d_hid + j] * dy[i];
        }
    }

    // dL/dz1 = dh * ReLU'(z1)
    let dz1: Vec<f64> = dh
        .iter()
        .zip(fwd.z1.iter())
        .map(|(&dhi, &z1i)| if z1i > 0.0 { dhi } else { 0.0 })
        .collect();

    // dL/dW1[i][j] = dz1[i] * x[j]
    let mut dw1 = vec![0.0; d_hid * d_in];
    for i in 0..d_hid {
        for j in 0..d_in {
            dw1[i * d_in + j] = dz1[i] * x[j];
        }
    }

    // dL/db1 = dz1
    let db1 = dz1;

    Gradients {
        dw1,
        db1,
        dw2,
        db2,
    }
}

/// Applies SGD: w ← w - lr * grad.
pub fn sgd_update(model: &mut MlpModel, grads: &Gradients, lr: f64) {
    for (w, g) in model.w1.iter_mut().zip(grads.dw1.iter()) {
        *w -= lr * g;
    }
    for (b, g) in model.b1.iter_mut().zip(grads.db1.iter()) {
        *b -= lr * g;
    }
    for (w, g) in model.w2.iter_mut().zip(grads.dw2.iter()) {
        *w -= lr * g;
    }
    for (b, g) in model.b2.iter_mut().zip(grads.db2.iter()) {
        *b -= lr * g;
    }
}

// ──────────────────────────────────────────────────────────────
// Quantisation: f64 → Fr
// ──────────────────────────────────────────────────────────────

/// Quantisation scale factor.  Values are mapped as:
/// `Fr_val = round(f64_val * QUANT_SCALE)` (mod p for negatives).
const QUANT_SCALE: f64 = 1000.0;

/// Quantises an f64 to Fr, handling negatives via p - |v|.
fn quantize(val: f64) -> Fr {
    let scaled = (val * QUANT_SCALE).round();
    if scaled >= 0.0 {
        Fr::from(scaled as u64)
    } else {
        // Represent negative as p - |scaled|.
        let abs_val = (-scaled) as u64;
        Fr::from(0) - Fr::from(abs_val)
    }
}

/// Quantises a slice of f64 values.
fn quantize_vec(vals: &[f64]) -> Vec<Fr> {
    vals.iter().map(|&v| quantize(v)).collect()
}

// ──────────────────────────────────────────────────────────────
// Result of a single proved training step
// ──────────────────────────────────────────────────────────────

/// Result from a single training step with ZK proof.
#[derive(Debug)]
pub struct ProvedStep {
    /// The Halo2 KZG proof bytes (raw transcript format).
    pub proof: Vec<u8>,
    /// EVM-formatted proof bytes (320 bytes: 3 advice + 2 opening points).
    /// `None` if EVM serialization failed (non-fatal).
    pub evm_proof: Option<Vec<u8>>,
    /// EVM-formatted public inputs (8 × 32-byte big-endian arrays).
    pub evm_public_inputs: Vec<[u8; 32]>,
    /// Model state commitment (SHA-256 of weights after update).
    pub commitment: [u8; 32],
    /// Loss before this step's weight update.
    pub loss: f64,
    /// Step number.
    pub step: u64,
    /// Public inputs for independent verification.
    pub public_inputs: Vec<Fr>,
    /// Whether the proof was self-verified by the prover.
    pub verified: bool,
}

/// Metrics collected during a training run.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Per-step losses.
    pub losses: Vec<f64>,
    /// Number of steps completed.
    pub steps: u64,
    /// Number of proofs generated.
    pub proofs_generated: u64,
    /// Whether loss decreased from first to last step.
    pub loss_decreased: bool,
}

// ──────────────────────────────────────────────────────────────
// Trainer
// ──────────────────────────────────────────────────────────────

/// Real ML trainer with ZK proof generation (V2).
///
/// Holds a 2-layer MLP and an `MLTrainingProverV2`.  Each call to
/// `train_step` runs native forward+backward, updates weights via SGD,
/// then generates a Halo2 KZG proof with EVM-compatible output.
pub struct Trainer {
    /// The model being trained.
    model: MlpModel,
    /// Learning rate for SGD.
    lr: f64,
    /// Prover instance (lazily initialised).
    prover: Option<MLTrainingProverV2>,
    /// Step counter.
    step_count: u64,
}

impl Trainer {
    /// Creates a trainer with a randomly initialised model.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize, lr: f64, seed: u64) -> Self {
        let model = MlpModel::new_random(d_in, d_hid, d_out, seed);
        Self {
            model,
            lr,
            prover: None,
            step_count: 0,
        }
    }

    /// Creates a trainer with an existing model.
    pub fn with_model(model: MlpModel, lr: f64) -> Self {
        Self {
            model,
            lr,
            prover: None,
            step_count: 0,
        }
    }

    /// Returns a reference to the current model.
    pub fn model(&self) -> &MlpModel {
        &self.model
    }

    /// Returns the current step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    /// Runs a single training step and generates a ZK proof.
    ///
    /// 1. Forward pass → loss
    /// 2. Backward pass → gradients
    /// 3. Snapshot old weights
    /// 4. SGD update
    /// 5. Quantise everything to Fr
    /// 6. Generate Halo2 proof via MLTrainingProverV2 (with EVM output)
    pub fn train_step(&mut self, x: &[f64], target: &[f64]) -> anyhow::Result<ProvedStep> {
        assert_eq!(x.len(), self.model.d_in, "input dimension mismatch");
        assert_eq!(target.len(), self.model.d_out, "target dimension mismatch");

        self.step_count += 1;

        // 1. Forward pass.
        let fwd = forward(&self.model, x, target);
        let loss = fwd.loss;

        // 2. Backward pass.
        let grads = backward(&self.model, x, target, &fwd);

        // 3. Snapshot old weights before update (for the proof).
        let old_w1 = self.model.w1.clone();
        let old_b1 = self.model.b1.clone();
        let old_w2 = self.model.w2.clone();
        let old_b2 = self.model.b2.clone();

        // 4. SGD weight update.
        sgd_update(&mut self.model, &grads, self.lr);

        // 5. Quantise to Fr.
        let x_fr = quantize_vec(x);
        let target_fr = quantize_vec(target);
        let w1_fr = quantize_vec(&old_w1);
        let b1_fr = quantize_vec(&old_b1);
        let w2_fr = quantize_vec(&old_w2);
        let b2_fr = quantize_vec(&old_b2);
        let lr_fr = quantize(self.lr);

        // 6. Generate ZK proof via V2 prover.
        if self.prover.is_none() {
            self.prover = Some(MLTrainingProverV2::new(
                self.model.d_in,
                self.model.d_hid,
                self.model.d_out,
            ));
        }
        let prover = self.prover.as_ref().unwrap();

        let witness = MLTrainingProverV2::build_witness(
            self.model.d_in,
            self.model.d_hid,
            self.model.d_out,
            &x_fr,
            &target_fr,
            &w1_fr,
            &b1_fr,
            &w2_fr,
            &b2_fr,
            lr_fr,
            self.step_count,
            Fr::from(1u64), // base_error
        );
        let proof_result = prover.prove(&witness)
            .map_err(|e| anyhow::anyhow!("Proof generation failed at step {}: {}", self.step_count, e))?;

        if !prover.verify_result(&proof_result) {
            anyhow::bail!("Proof self-verification failed at step {}", self.step_count);
        }

        // Convert to EVM format (non-fatal if it fails)
        let evm_proof = proof_result.to_evm_proof().ok();
        let evm_public_inputs = proof_result.to_evm_public_inputs();

        let commitment = self.model.commitment();

        Ok(ProvedStep {
            proof: proof_result.proof,
            evm_proof,
            evm_public_inputs,
            commitment,
            loss,
            step: self.step_count,
            public_inputs: proof_result.public_inputs,
            verified: proof_result.verified,
        })
    }

    /// Runs a single training step WITHOUT proof generation (fast, for iteration).
    pub fn train_step_unproved(&mut self, x: &[f64], target: &[f64]) -> f64 {
        assert_eq!(x.len(), self.model.d_in);
        assert_eq!(target.len(), self.model.d_out);

        self.step_count += 1;
        let fwd = forward(&self.model, x, target);
        let loss = fwd.loss;
        let grads = backward(&self.model, x, target, &fwd);
        sgd_update(&mut self.model, &grads, self.lr);
        loss
    }

    /// Runs multiple training steps on a dataset, generating a proof per step.
    ///
    /// `dataset` is a list of `(input, target)` pairs.  Steps cycle through
    /// the dataset.
    pub fn train(
        &mut self,
        dataset: &[(Vec<f64>, Vec<f64>)],
        num_steps: usize,
    ) -> anyhow::Result<(Vec<ProvedStep>, TrainingMetrics)> {
        let mut steps = Vec::with_capacity(num_steps);
        let mut losses = Vec::with_capacity(num_steps);

        for i in 0..num_steps {
            let (x, t) = &dataset[i % dataset.len()];
            let result = self.train_step(x, t)?;
            losses.push(result.loss);
            steps.push(result);
        }

        let loss_decreased = if losses.len() >= 2 {
            losses.last().unwrap() < losses.first().unwrap()
        } else {
            false
        };

        let metrics = TrainingMetrics {
            losses,
            steps: num_steps as u64,
            proofs_generated: num_steps as u64,
            loss_decreased,
        };

        Ok((steps, metrics))
    }

    /// Runs multiple training steps WITHOUT proofs (fast training).
    pub fn train_unproved(
        &mut self,
        dataset: &[(Vec<f64>, Vec<f64>)],
        num_steps: usize,
    ) -> TrainingMetrics {
        let mut losses = Vec::with_capacity(num_steps);

        for i in 0..num_steps {
            let (x, t) = &dataset[i % dataset.len()];
            let loss = self.train_step_unproved(x, t);
            losses.push(loss);
        }

        let loss_decreased = if losses.len() >= 2 {
            losses.last().unwrap() < losses.first().unwrap()
        } else {
            false
        };

        TrainingMetrics {
            losses,
            steps: num_steps as u64,
            proofs_generated: 0,
            loss_decreased,
        }
    }

    /// Legacy compatibility: `train_and_prove` matching the old mocked interface.
    pub fn train_and_prove(
        &mut self,
        x: &[f64],
        target: &[f64],
    ) -> anyhow::Result<(Vec<u8>, [u8; 32])> {
        let result = self.train_step(x, target)?;
        Ok((result.proof, result.commitment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_dataset() -> Vec<(Vec<f64>, Vec<f64>)> {
        // Simple regression: y ≈ x1 + x2
        vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![2.0]),
            (vec![2.0, 1.0], vec![3.0]),
        ]
    }

    #[test]
    fn test_forward_backward_basic() {
        let model = MlpModel::new(
            2,
            2,
            1,
            vec![1.0, 0.0, 0.0, 1.0], // identity-ish W1
            vec![0.0, 0.0],
            vec![1.0, 1.0], // sum W2
            vec![0.0],
        );

        let x = vec![1.0, 2.0];
        let target = vec![5.0];

        let fwd = forward(&model, &x, &target);
        // z1 = [1, 2], h = [1, 2], y = [3], loss = (3-5)^2 = 4
        assert!((fwd.loss - 4.0).abs() < 1e-10);

        let grads = backward(&model, &x, &target, &fwd);
        // Verify gradients are non-zero.
        assert!(grads.dw1.iter().any(|&g| g.abs() > 1e-10));
        assert!(grads.dw2.iter().any(|&g| g.abs() > 1e-10));
    }

    #[test]
    fn test_sgd_reduces_loss() {
        let mut model = MlpModel::new_random(2, 4, 1, 42);
        let dataset = simple_dataset();
        let lr = 0.01;

        let initial_loss: f64 = dataset
            .iter()
            .map(|(x, t)| forward(&model, x, t).loss)
            .sum::<f64>()
            / dataset.len() as f64;

        // Train 100 steps.
        for i in 0..100 {
            let (x, t) = &dataset[i % dataset.len()];
            let fwd = forward(&model, x, t);
            let grads = backward(&model, x, t, &fwd);
            sgd_update(&mut model, &grads, lr);
        }

        let final_loss: f64 = dataset
            .iter()
            .map(|(x, t)| forward(&model, x, t).loss)
            .sum::<f64>()
            / dataset.len() as f64;

        assert!(
            final_loss < initial_loss,
            "Loss should decrease: {initial_loss} → {final_loss}"
        );
    }

    #[test]
    fn test_quantize_round_trip() {
        // Positive values.
        let v = quantize(3.14);
        // 3.14 * 1000 = 3140 → Fr(3140)
        assert_eq!(v, Fr::from(3140u64));

        // Zero.
        assert_eq!(quantize(0.0), Fr::from(0u64));

        // Negative: should produce p - 1000.
        let neg = quantize(-1.0);
        assert_eq!(neg, Fr::from(0) - Fr::from(1000u64));
    }

    #[test]
    fn test_train_step_unproved_reduces_loss() {
        let mut trainer = Trainer::new(2, 4, 1, 0.01, 42);
        let dataset = simple_dataset();

        let first_loss = trainer.train_step_unproved(&dataset[0].0, &dataset[0].1);

        // Train many steps.
        for i in 1..200 {
            let (x, t) = &dataset[i % dataset.len()];
            trainer.train_step_unproved(x, t);
        }

        // Evaluate on same sample.
        let final_loss = forward(trainer.model(), &dataset[0].0, &dataset[0].1).loss;
        assert!(
            final_loss < first_loss,
            "Loss should decrease: {first_loss} → {final_loss}"
        );
    }

    #[test]
    fn test_train_unproved_metrics() {
        let mut trainer = Trainer::new(2, 4, 1, 0.01, 42);
        let dataset = simple_dataset();

        let metrics = trainer.train_unproved(&dataset, 50);
        assert_eq!(metrics.steps, 50);
        assert_eq!(metrics.proofs_generated, 0);
        assert_eq!(metrics.losses.len(), 50);
        assert!(metrics.loss_decreased, "Loss should decrease over 50 steps");
    }

    #[test]
    fn test_train_step_with_proof_v2() {
        let model = MlpModel::new(
            2,
            2,
            1,
            vec![0.001, 0.002, 0.003, 0.001], // W1: [1, 2, 3, 1] after ×1000
            vec![0.0, 0.0],
            vec![0.001, 0.001], // W2: [1, 1] after ×1000
            vec![0.0],
        );

        let mut trainer = Trainer::with_model(model, 0.001); // lr: 1 after ×1000
        let x = vec![0.001, 0.001]; // [1, 1] after quantization
        let target = vec![0.005]; // [5] after quantization

        let result = trainer.train_step(&x, &target).expect("train_step failed");

        assert!(!result.proof.is_empty(), "proof should be non-empty");
        assert!(result.loss > 0.0, "loss should be positive");
        assert_eq!(result.step, 1);
        assert_ne!(result.commitment, [0u8; 32], "commitment should be non-zero");
        // V2 prover self-verifies during generation
        assert!(result.verified, "proof should be self-verified");

        // EVM public inputs should have 8 elements (V2 format)
        assert_eq!(result.evm_public_inputs.len(), 8, "V2 should produce 8 public inputs");

        // Each EVM public input should be 32 bytes
        for (i, pi) in result.evm_public_inputs.iter().enumerate() {
            assert_eq!(pi.len(), 32, "public input {} should be 32 bytes", i);
        }
    }

    #[test]
    fn test_model_commitment_changes() {
        let mut model = MlpModel::new(
            2,
            2,
            1,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0],
        );

        let c1 = model.commitment();
        let fwd = forward(&model, &[1.0, 2.0], &[5.0]);
        let grads = backward(&model, &[1.0, 2.0], &[5.0], &fwd);
        sgd_update(&mut model, &grads, 0.1);
        let c2 = model.commitment();

        assert_ne!(c1, c2, "commitment should change after weight update");
    }

    #[test]
    fn test_two_consecutive_proved_steps_v2() {
        let model = MlpModel::new(
            2,
            2,
            1,
            vec![0.001, 0.002, 0.003, 0.001], // W1: [1, 2, 3, 1] after ×1000
            vec![0.0, 0.0],
            vec![0.001, 0.001], // W2: [1, 1] after ×1000
            vec![0.0],
        );

        let mut trainer = Trainer::with_model(model, 0.0001);

        let x = vec![0.001, 0.001]; // [1, 1] after quantization
        let target = vec![0.005]; // [5] after quantization

        // Step 1.
        let r1 = trainer.train_step(&x, &target);
        assert!(r1.is_ok(), "step 1 failed: {:?}", r1.err());
        let r1 = r1.unwrap();
        eprintln!("Step 1: loss={:.6}, proof_len={}, verified={}", r1.loss, r1.proof.len(), r1.verified);

        // Step 2.
        let r2 = trainer.train_step(&x, &target);
        assert!(r2.is_ok(), "step 2 failed: {:?}", r2.err());
        let r2 = r2.unwrap();
        eprintln!("Step 2: loss={:.6}, proof_len={}, verified={}", r2.loss, r2.proof.len(), r2.verified);
    }
}
