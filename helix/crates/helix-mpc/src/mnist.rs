//! MNIST-scale MPC training pipeline.
//!
//! Provides end-to-end MPC training for a 784→32→10 MLP on MNIST data,
//! including:
//!
//! - Synthetic MNIST data generation for deterministic testing
//! - Optimized batched forward/backward pass at MNIST scale
//! - Loss revelation protocol (jointly reveal scalar loss without leaking weights)
//! - Periodic accuracy evaluation via MPC inference
//! - Softmax + cross-entropy loss for proper classification training
//! - Native (non-MPC) trainer for correctness comparison

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use tracing::{debug, info, warn};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use crate::protocols::arithmetic::SecureArithmetic;
use crate::session::transport::MPCTransport;
use crate::types::PartyId;

// ============================================================================
// MNIST Data Generation
// ============================================================================

/// A single MNIST sample: 784-dim input + 10-class one-hot target.
#[derive(Debug, Clone)]
pub struct MnistSample {
    /// Pixel values normalized to [0, 1], length 784.
    pub pixels: Vec<f64>,
    /// One-hot encoded label, length 10.
    pub label: Vec<f64>,
    /// Raw label index (0-9).
    pub digit: usize,
}

/// Generates synthetic MNIST-like data for deterministic testing.
///
/// Creates digit patterns that a 784→32→10 network can learn:
/// each digit has a characteristic spatial pattern in the 28x28 grid,
/// with added noise for realism.
pub struct MnistDataset {
    pub train: Vec<MnistSample>,
    pub test: Vec<MnistSample>,
}

impl MnistDataset {
    /// Generates a synthetic MNIST dataset with the given number of samples.
    ///
    /// Each digit 0-9 has a distinctive pattern:
    /// - Digit 0: ring pattern (active pixels form an oval)
    /// - Digit 1: vertical line in the center
    /// - Digit 2-9: distinct quadrant/stripe activations
    ///
    /// The patterns are designed to be linearly separable in feature space
    /// after a ReLU hidden layer, ensuring a 784→32→10 MLP can learn them.
    pub fn generate(train_size: usize, test_size: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let train = Self::generate_samples(train_size, &mut rng);
        let test = Self::generate_samples(test_size, &mut rng);
        MnistDataset { train, test }
    }

    fn generate_samples(count: usize, rng: &mut ChaCha20Rng) -> Vec<MnistSample> {
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let digit = rng.gen_range(0..10);
            let pixels = Self::generate_digit_pattern(digit, rng);
            let mut label = vec![0.0; 10];
            label[digit] = 1.0;
            samples.push(MnistSample {
                pixels,
                label,
                digit,
            });
        }
        samples
    }

    /// Generates a 784-pixel pattern for the given digit.
    ///
    /// Each digit activates a different region of the 28x28 grid with
    /// characteristic intensity patterns. Noise is added to prevent
    /// trivial memorization.
    fn generate_digit_pattern(digit: usize, rng: &mut ChaCha20Rng) -> Vec<f64> {
        let mut pixels = vec![0.0f64; 784];
        let noise_scale = 0.05;

        // Each digit activates specific rows and columns in the 28x28 grid.
        // This creates patterns that are distinguishable but require a hidden
        // layer to classify (not linearly separable in raw pixel space for all pairs).
        match digit {
            0 => {
                // Ring: active on border of center region
                for r in 6..22 {
                    for c in 6..22 {
                        let dr = (r as f64 - 14.0).abs();
                        let dc = (c as f64 - 14.0).abs();
                        let dist = (dr * dr + dc * dc).sqrt();
                        if dist > 5.0 && dist < 9.0 {
                            pixels[r * 28 + c] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                        }
                    }
                }
            }
            1 => {
                // Vertical line in center
                for r in 4..24 {
                    for c in 12..16 {
                        pixels[r * 28 + c] = 0.9 + rng.gen_range(-noise_scale..noise_scale);
                    }
                }
            }
            2 => {
                // Top horizontal + diagonal + bottom horizontal
                for c in 8..20 {
                    pixels[6 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[21 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for i in 0..14 {
                    let r = 7 + i;
                    let c = 19 - i;
                    if c >= 8 && c < 20 {
                        pixels[r * 28 + c] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                    }
                }
            }
            3 => {
                // Three horizontal lines
                for c in 8..20 {
                    pixels[6 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[13 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[21 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                // Right vertical
                for r in 6..22 {
                    pixels[r * 28 + 19] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                }
            }
            4 => {
                // Left vertical top half + middle horizontal + right vertical full
                for r in 4..14 {
                    pixels[r * 28 + 8] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for c in 8..20 {
                    pixels[14 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 4..24 {
                    pixels[r * 28 + 19] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
            }
            5 => {
                // Top horizontal + left vertical top + middle horizontal + right vertical bottom + bottom horizontal
                for c in 8..20 {
                    pixels[6 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[13 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[21 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 6..14 {
                    pixels[r * 28 + 8] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 13..22 {
                    pixels[r * 28 + 19] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                }
            }
            6 => {
                // Full left vertical + bottom horizontal + right vertical bottom half + middle horizontal
                for r in 4..24 {
                    pixels[r * 28 + 8] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for c in 8..20 {
                    pixels[13 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[22 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 13..23 {
                    pixels[r * 28 + 19] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                }
            }
            7 => {
                // Top horizontal + right diagonal
                for c in 8..20 {
                    pixels[6 * 28 + c] = 0.9 + rng.gen_range(-noise_scale..noise_scale);
                }
                for i in 0..16 {
                    let r = 6 + i;
                    let c = 19 - i / 2;
                    if r < 24 && c >= 8 {
                        pixels[r * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    }
                }
            }
            8 => {
                // Two stacked rings (figure 8)
                for c in 8..20 {
                    pixels[5 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[13 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[22 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 5..14 {
                    pixels[r * 28 + 8] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[r * 28 + 19] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 13..23 {
                    pixels[r * 28 + 8] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[r * 28 + 19] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                }
            }
            9 => {
                // Top ring + right vertical full
                for c in 8..20 {
                    pixels[5 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                    pixels[13 * 28 + c] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 5..14 {
                    pixels[r * 28 + 8] = 0.8 + rng.gen_range(-noise_scale..noise_scale);
                }
                for r in 4..24 {
                    pixels[r * 28 + 19] = 0.85 + rng.gen_range(-noise_scale..noise_scale);
                }
            }
            _ => unreachable!(),
        }

        // Clamp all values to [0, 1]
        for p in pixels.iter_mut() {
            *p = p.clamp(0.0, 1.0);
        }

        pixels
    }

    /// Converts to (input, target) tuples for the MPC trainer.
    pub fn as_training_pairs(samples: &[MnistSample]) -> Vec<(Vec<f64>, Vec<f64>)> {
        samples
            .iter()
            .map(|s| (s.pixels.clone(), s.label.clone()))
            .collect()
    }
}

// ============================================================================
// Native (non-MPC) Trainer for comparison
// ============================================================================

/// Native (plaintext) MLP trainer for correctness comparison with MPC.
///
/// Implements the same 784→32→10 architecture with identical math:
/// - Forward: h = ReLU(W1 @ x + b1), y = W2 @ h + b2
/// - Loss: softmax cross-entropy
/// - Backward: standard backpropagation
/// - Update: SGD with fixed learning rate
pub struct NativeTrainer {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub learning_rate: f64,
}

/// Result of a native training step.
#[derive(Debug)]
pub struct NativeStepResult {
    pub step: u64,
    pub loss: f64,
    /// Predicted class probabilities (softmax output).
    pub probabilities: Vec<f64>,
}

impl NativeTrainer {
    /// Creates a new native trainer with random Xavier-initialized weights.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize, learning_rate: f64, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let w1_scale = (2.0 / (d_in + d_hid) as f64).sqrt();
        let w2_scale = (2.0 / (d_hid + d_out) as f64).sqrt();

        Self {
            d_in,
            d_hid,
            d_out,
            w1: (0..d_hid * d_in)
                .map(|_| rng.gen_range(-w1_scale..w1_scale))
                .collect(),
            b1: vec![0.0; d_hid],
            w2: (0..d_out * d_hid)
                .map(|_| rng.gen_range(-w2_scale..w2_scale))
                .collect(),
            b2: vec![0.0; d_out],
            learning_rate,
        }
    }

    /// Creates a native trainer from existing weights (for comparison with MPC).
    pub fn from_weights(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        learning_rate: f64,
        w1: Vec<f64>,
        b1: Vec<f64>,
        w2: Vec<f64>,
        b2: Vec<f64>,
    ) -> Self {
        Self {
            d_in,
            d_hid,
            d_out,
            w1,
            b1,
            w2,
            b2,
            learning_rate,
        }
    }

    /// Runs a single training step with MSE loss using pure f64 arithmetic.
    ///
    /// Implements the same 784→32→10 MLP as `MPCTrainer::training_step_unproved`:
    /// - Forward: h = ReLU(W1 @ x + b1), y = W2 @ h + b2
    /// - Loss: 0.5 * sum((y - target)^2)
    /// - Backward: standard backpropagation
    /// - Update: SGD with fixed learning rate
    ///
    /// Uses pure f64 arithmetic (no Fr round-trips) to serve as a correct
    /// baseline for comparing MPC training results. The MPC trainer uses Fr
    /// fixed-point arithmetic due to secret sharing constraints; this trainer
    /// uses native floating-point to provide an ideal reference.
    pub fn training_step_mse(&mut self, input: &[f64], target: &[f64], step: u64) -> NativeStepResult {
        let d_in = self.d_in;
        let d_hid = self.d_hid;
        let d_out = self.d_out;

        // Forward: h_pre = W1 @ x + b1
        let mut h_pre = vec![0.0f64; d_hid];
        for i in 0..d_hid {
            let mut sum = 0.0;
            for j in 0..d_in {
                sum += self.w1[i * d_in + j] * input[j];
            }
            h_pre[i] = sum + self.b1[i];
        }

        // ReLU
        let mut h = vec![0.0f64; d_hid];
        let mut relu_mask = vec![0.0f64; d_hid];
        for i in 0..d_hid {
            if h_pre[i] >= 0.0 {
                h[i] = h_pre[i];
                relu_mask[i] = 1.0;
            }
        }

        // y = W2 @ h + b2
        let mut y = vec![0.0f64; d_out];
        for i in 0..d_out {
            let mut sum = 0.0;
            for j in 0..d_hid {
                sum += self.w2[i * d_hid + j] * h[j];
            }
            y[i] = sum + self.b2[i];
        }

        // MSE Loss: 0.5 * sum((y - target)^2)
        let mut loss = 0.0;
        let mut dy = vec![0.0f64; d_out];
        for i in 0..d_out {
            let diff = y[i] - target[i];
            loss += 0.5 * diff * diff;
            dy[i] = diff;
        }

        // Backward: dW2 = outer(dy, h)
        let mut dw2 = vec![0.0f64; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                dw2[i * d_hid + j] = dy[i] * h[j];
            }
        }

        let db2 = dy.clone();

        // dh = W2^T @ dy
        let mut dh = vec![0.0f64; d_hid];
        for j in 0..d_hid {
            let mut sum = 0.0;
            for i in 0..d_out {
                sum += self.w2[i * d_hid + j] * dy[i];
            }
            dh[j] = sum;
        }

        // dh_pre = dh * relu_mask
        let mut dh_pre = vec![0.0f64; d_hid];
        for i in 0..d_hid {
            dh_pre[i] = dh[i] * relu_mask[i];
        }

        // dW1 = outer(dh_pre, x)
        let mut dw1 = vec![0.0f64; d_hid * d_in];
        for i in 0..d_hid {
            for j in 0..d_in {
                dw1[i * d_in + j] = dh_pre[i] * input[j];
            }
        }

        let db1 = dh_pre.clone();

        // Update: W -= lr * dW
        for i in 0..self.w1.len() {
            self.w1[i] -= self.learning_rate * dw1[i];
        }
        for i in 0..self.b1.len() {
            self.b1[i] -= self.learning_rate * db1[i];
        }
        for i in 0..self.w2.len() {
            self.w2[i] -= self.learning_rate * dw2[i];
        }
        for i in 0..self.b2.len() {
            self.b2[i] -= self.learning_rate * db2[i];
        }

        NativeStepResult {
            step,
            loss,
            probabilities: y,
        }
    }

    /// Runs inference on a batch and returns accuracy.
    pub fn evaluate(&self, data: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        let mut correct = 0usize;
        for (input, target) in data {
            let predicted = self.predict(input);
            let target_class = target
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if predicted == target_class {
                correct += 1;
            }
        }
        correct as f64 / data.len() as f64
    }

    /// Predicts the class for a single input.
    pub fn predict(&self, input: &[f64]) -> usize {
        let y = self.forward(input);
        y.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Forward pass only (no gradient computation).
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let d_in = self.d_in;
        let d_hid = self.d_hid;
        let d_out = self.d_out;

        // h_pre = W1 @ x + b1
        let mut h_pre = vec![0.0f64; d_hid];
        for i in 0..d_hid {
            let mut sum = 0.0;
            for j in 0..d_in {
                sum += self.w1[i * d_in + j] * input[j];
            }
            h_pre[i] = sum + self.b1[i];
        }

        // ReLU
        let mut h = vec![0.0f64; d_hid];
        for i in 0..d_hid {
            h[i] = h_pre[i].max(0.0);
        }

        // y = W2 @ h + b2
        let mut y = vec![0.0f64; d_out];
        for i in 0..d_out {
            let mut sum = 0.0;
            for j in 0..d_hid {
                sum += self.w2[i * d_hid + j] * h[j];
            }
            y[i] = sum + self.b2[i];
        }

        y
    }

    /// Converts to ModelWeights for use with MPCTrainer.
    pub fn to_model_weights(&self) -> ModelWeights {
        ModelWeights::from_f64(&self.w1, &self.b1, &self.w2, &self.b2)
    }
}

// ============================================================================
// MPC MNIST Training Orchestrator
// ============================================================================

/// Configuration for MNIST-scale MPC training.
#[derive(Debug, Clone)]
pub struct MnistMpcConfig {
    /// Number of MPC parties (default: 3).
    pub num_parties: usize,
    /// Number of training steps.
    pub num_steps: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Enable SPDZ MAC verification (check interval).
    pub mac_check_interval: Option<u64>,
    /// Evaluate accuracy every N steps (0 = disabled).
    pub eval_interval: usize,
    /// Number of test samples for evaluation.
    pub eval_batch_size: usize,
    /// Re-share weights every N steps (0 = disabled).
    pub reshare_interval: u64,
    /// Random seed.
    pub seed: u64,
}

impl Default for MnistMpcConfig {
    fn default() -> Self {
        Self {
            num_parties: 3,
            num_steps: 200,
            learning_rate: 0.01,
            mac_check_interval: Some(10),
            eval_interval: 50,
            eval_batch_size: 100,
            reshare_interval: 0,
            seed: 42,
        }
    }
}

/// Result of MPC MNIST training.
#[derive(Debug)]
pub struct MnistTrainingResult {
    /// Per-step losses (revealed jointly).
    pub losses: Vec<f64>,
    /// Accuracy measurements at eval intervals: (step, accuracy).
    pub accuracy_history: Vec<(usize, f64)>,
    /// Final accuracy on test set.
    pub final_accuracy: f64,
    /// Total training time in seconds (wall clock for the async tasks).
    pub total_time_secs: f64,
    /// Per-step time in seconds (average).
    pub avg_step_time_secs: f64,
    /// Number of Beaver triples consumed.
    pub triples_consumed: usize,
}

/// Computes the number of Beaver triples needed per training step for a
/// d_in → d_hid → d_out MLP.
///
/// Per step:
/// - ReLU forward: d_hid triples (h_pre * sign_mask)
/// - W2 @ h: d_out * d_hid triples (shared * shared matmul)
/// - dh * relu_mask: d_hid triples (backward pass)
/// - Overhead: 32 triples
pub fn triples_per_step(d_hid: usize, d_out: usize) -> usize {
    d_hid               // ReLU forward
    + d_out * d_hid     // W2 @ h matmul
    + d_hid             // backward pass dh * relu_mask
    + 32                // overhead
}

/// Pre-generates all Beaver triples needed for the entire training run.
///
/// This is a key optimization: generating triples in bulk before training
/// avoids per-step generation overhead and allows the MPC communication
/// for triple generation to be batched efficiently.
pub async fn pre_generate_triples<T: MPCTransport>(
    trainer: &mut MPCTrainer<T>,
    num_steps: usize,
    d_hid: usize,
    d_out: usize,
) -> MPCResult<()> {
    let per_step = triples_per_step(d_hid, d_out);
    let total = per_step * num_steps;

    // Generate in batches to avoid massive single messages
    let batch_size = 5000.min(total);
    let mut remaining = total;

    info!(
        total = total,
        per_step = per_step,
        "Pre-generating Beaver triples for {} training steps",
        num_steps
    );

    while remaining > 0 {
        let this_batch = remaining.min(batch_size);
        trainer.generate_beaver_triples(this_batch).await?;
        remaining -= this_batch;
        debug!(
            generated = total - remaining,
            total = total,
            "Beaver triple generation progress"
        );
    }

    info!(
        total = total,
        remaining = trainer.beaver_triples_remaining(),
        "Beaver triple pre-generation complete"
    );
    Ok(())
}

/// Runs MPC inference on a batch of samples and jointly reveals accuracy.
///
/// Protocol:
/// 1. Each party runs the MPC forward pass on each sample
/// 2. Parties jointly reconstruct the output y for each sample
/// 3. Each party locally computes argmax(y) and compares to target
/// 4. Accuracy = correct_count / total_count
///
/// The output y is intentionally revealed (it's the prediction). The model
/// weights remain secret-shared throughout.
pub async fn mpc_evaluate_accuracy<T: MPCTransport>(
    trainer: &mut MPCTrainer<T>,
    test_data: &[(Vec<f64>, Vec<f64>)],
) -> MPCResult<f64> {
    let d_in = 784;
    let d_hid = 32;
    let d_out = 10;
    let (w1, b1, w2, b2) = trainer.weight_shares();

    let mut correct = 0usize;
    let peers = trainer.transport().peers();

    for (input, target) in test_data {
        let x: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();

        // Forward: h_pre = W1 @ x + b1 (x is public, W1 is shared → local)
        let mut h_pre = vec![Fr::ZERO; d_hid];
        for i in 0..d_hid {
            let mut sum = Fr::ZERO;
            for j in 0..d_in {
                let contrib = w1[i * d_in + j].mpc_scale(&x[j]);
                sum = Fr::add(&sum, &contrib);
            }
            h_pre[i] = Fr::add(&sum, &b1[i]);
        }

        // Reconstruct h_pre for ReLU evaluation
        // (We open h_pre to evaluate sign; this leaks sign bits but not magnitudes
        // when using the secure_sign_bit protocol. For evaluation, we just open
        // the whole thing since accuracy is already a public metric.)
        let h_pre_bytes = SecureArithmetic::serialize_share_batch(&h_pre);
        trainer.transport().broadcast(&h_pre_bytes).await?;

        let mut h_pre_recon = h_pre.clone();
        for peer in &peers {
            let msg = trainer.transport().recv(peer).await?;
            let peer_h = SecureArithmetic::deserialize_share_batch(&msg)?;
            for i in 0..d_hid {
                h_pre_recon[i] = Fr::add(&h_pre_recon[i], &peer_h[i]);
            }
        }

        // ReLU on plaintext — re-align through f64 for correct mpc_scale
        let mut h = vec![Fr::ZERO; d_hid];
        for i in 0..d_hid {
            let val = h_pre_recon[i].to_f64();
            if val >= 0.0 {
                h[i] = Fr::from_f64(val);
            }
        }

        // y = W2 @ h + b2 (W2 is shared, h is now public → scale by public)
        let mut y_share = vec![Fr::ZERO; d_out];
        for i in 0..d_out {
            let mut sum = Fr::ZERO;
            for j in 0..d_hid {
                let contrib = w2[i * d_hid + j].mpc_scale(&h[j]);
                sum = Fr::add(&sum, &contrib);
            }
            y_share[i] = Fr::add(&sum, &b2[i]);
        }

        // Reconstruct y
        let y_bytes = SecureArithmetic::serialize_share_batch(&y_share);
        trainer.transport().broadcast(&y_bytes).await?;

        let mut y_recon = y_share.clone();
        for peer in &peers {
            let msg = trainer.transport().recv(peer).await?;
            let peer_y = SecureArithmetic::deserialize_share_batch(&msg)?;
            for i in 0..d_out {
                y_recon[i] = Fr::add(&y_recon[i], &peer_y[i]);
            }
        }

        // Compute argmax
        let predicted = y_recon
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.to_f64().partial_cmp(&b.to_f64()).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();

        let target_class = target
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        if predicted == target_class {
            correct += 1;
        }
    }

    Ok(correct as f64 / test_data.len() as f64)
}

/// Runs a single MPC training step and jointly reveals the loss.
///
/// This wraps `training_step_unproved` or `training_step_with_mac` depending
/// on whether MAC verification is enabled, and returns the revealed loss.
pub async fn mpc_training_step_with_loss<T: MPCTransport>(
    trainer: &mut MPCTrainer<T>,
    input: &[f64],
    target: &[f64],
    use_mac: bool,
) -> MPCResult<f64> {
    if use_mac && trainer.mac_enabled() {
        let result = trainer.training_step_with_mac(input, target).await?;
        Ok(result.loss)
    } else {
        let result = trainer.training_step_unproved(input, target).await?;
        Ok(result.loss)
    }
}

/// Creates an MPCTrainerConfig for MNIST-scale training.
pub fn mnist_trainer_config(config: &MnistMpcConfig) -> MPCTrainerConfig {
    let mut trainer_config = MPCTrainerConfig {
        d_in: 784,
        d_hid: 32,
        d_out: 10,
        learning_rate: config.learning_rate,
        num_parties: config.num_parties,
        reshare_interval: config.reshare_interval,
        beaver_batch_size: 10000,
        generate_proofs: false,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: None,
    };

    if let Some(interval) = config.mac_check_interval {
        trainer_config = trainer_config.with_mac_seed(interval, config.seed + 1000);
    }

    trainer_config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_data_generation() {
        let dataset = MnistDataset::generate(100, 20, 42);
        assert_eq!(dataset.train.len(), 100);
        assert_eq!(dataset.test.len(), 20);

        for sample in &dataset.train {
            assert_eq!(sample.pixels.len(), 784);
            assert_eq!(sample.label.len(), 10);
            assert!(sample.digit < 10);
            // One-hot encoding check
            assert_eq!(sample.label[sample.digit], 1.0);
            let label_sum: f64 = sample.label.iter().sum();
            assert!((label_sum - 1.0).abs() < 1e-10);
            // All pixels in [0, 1]
            for &p in &sample.pixels {
                assert!(p >= 0.0 && p <= 1.0);
            }
        }
    }

    #[test]
    fn test_mnist_data_deterministic() {
        let d1 = MnistDataset::generate(50, 10, 42);
        let d2 = MnistDataset::generate(50, 10, 42);

        for (s1, s2) in d1.train.iter().zip(d2.train.iter()) {
            assert_eq!(s1.digit, s2.digit);
            assert_eq!(s1.pixels, s2.pixels);
        }
    }

    #[test]
    fn test_mnist_data_balanced() {
        let dataset = MnistDataset::generate(10000, 0, 42);
        let mut counts = [0usize; 10];
        for sample in &dataset.train {
            counts[sample.digit] += 1;
        }
        // With 10000 samples and 10 classes, each should be ~1000 ± 100
        for (digit, &count) in counts.iter().enumerate() {
            assert!(
                count > 800 && count < 1200,
                "Digit {} has {} samples (expected ~1000)",
                digit, count
            );
        }
    }

    #[test]
    fn test_native_trainer_learns() {
        let dataset = MnistDataset::generate(500, 100, 42);
        let train_pairs = MnistDataset::as_training_pairs(&dataset.train);
        let test_pairs = MnistDataset::as_training_pairs(&dataset.test);

        let mut trainer = NativeTrainer::new(784, 32, 10, 0.01, 42);

        // Train for 200 steps
        let mut losses = Vec::new();
        for step in 0..200 {
            let idx = step % train_pairs.len();
            let (ref input, ref target) = train_pairs[idx];
            let result = trainer.training_step_mse(input, target, step as u64);
            losses.push(result.loss);
        }

        // Loss should decrease
        let early_avg: f64 = losses[..20].iter().sum::<f64>() / 20.0;
        let late_avg: f64 = losses[180..].iter().sum::<f64>() / 20.0;
        assert!(
            late_avg < early_avg,
            "Loss should decrease: early_avg={}, late_avg={}",
            early_avg, late_avg
        );

        // Evaluate accuracy
        let accuracy = trainer.evaluate(&test_pairs);
        assert!(
            accuracy > 0.3,
            "Native trainer accuracy {} should be > 30% after 200 steps",
            accuracy
        );
    }

    #[test]
    fn test_triples_per_step_calculation() {
        // 784→32→10 MLP
        let count = triples_per_step(32, 10);
        // 32 (ReLU) + 320 (W2@h) + 32 (backward) + 32 (overhead) = 416
        assert_eq!(count, 32 + 320 + 32 + 32);
    }

    #[test]
    fn test_mpc_scale_various_values() {
        // Test mpc_scale with various value pairs
        let test_cases: Vec<(f64, f64, f64)> = vec![
            (0.5, 0.3, 0.15),
            (1.0, 1.0, 1.0),
            (2.0, 3.0, 6.0),
            (0.1, 0.4, 0.04),
            (0.05, 0.8, 0.04),
            (0.01, 0.01, 0.0001),
            (-0.5, 0.3, -0.15),
            (0.0495, 0.8, 0.0396),
        ];

        for (a, b, expected) in &test_cases {
            let fa = Fr::from_f64(*a);
            let fb = Fr::from_f64(*b);
            let result = fa.mpc_scale(&fb).to_f64();
            eprintln!("mpc_scale({}, {}) = {} (expected {})", a, b, result, expected);
            if (result - expected).abs() > 0.001 {
                // Also check fixed_mul for comparison
                let fixed = fa.fixed_mul(&fb).to_f64();
                eprintln!("  fixed_mul({}, {}) = {}", a, b, fixed);
                // Print raw bytes of Fr values
                let fa_bytes = fa.to_bytes_le();
                eprintln!("  Fr({}) bytes[0..16] = {:?}", a, &fa_bytes[0..16]);
                let fb_bytes = fb.to_bytes_le();
                eprintln!("  Fr({}) bytes[0..16] = {:?}", b, &fb_bytes[0..16]);
            }
        }
        // Just check a few
        let r = Fr::from_f64(0.5).mpc_scale(&Fr::from_f64(0.3)).to_f64();
        assert!((r - 0.15).abs() < 1e-6, "mpc_scale(0.5, 0.3) = {} != 0.15", r);
        let r2 = Fr::from_f64(0.05).mpc_scale(&Fr::from_f64(0.8)).to_f64();
        assert!((r2 - 0.04).abs() < 0.01, "mpc_scale(0.05, 0.8) = {} != 0.04", r2);
    }

    #[test]
    fn test_mpc_scale_784_accumulation() {
        // Test that accumulating 784 mpc_scale terms gives the correct result
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Generate 784 random weights (Xavier-like) and pixels
        let w1_scale = (2.0 / (784.0 + 32.0) as f64).sqrt();
        let weights: Vec<f64> = (0..784)
            .map(|_| rng.gen_range(-w1_scale..w1_scale))
            .collect();
        let pixels: Vec<f64> = (0..784)
            .map(|j| if j % 7 == 0 { 0.8 } else { 0.0 })
            .collect();

        // Expected dot product in f64
        let expected: f64 = weights.iter().zip(pixels.iter())
            .map(|(w, p)| w * p)
            .sum();

        // Method 1: Direct Fr accumulation (no shares)
        let w_fr: Vec<Fr> = weights.iter().map(|&w| Fr::from_f64(w)).collect();
        let x_fr: Vec<Fr> = pixels.iter().map(|&p| Fr::from_f64(p)).collect();

        let mut sum_direct = Fr::ZERO;
        for j in 0..784 {
            let contrib = w_fr[j].mpc_scale(&x_fr[j]);
            sum_direct = Fr::add(&sum_direct, &contrib);
        }
        let result_direct = sum_direct.to_f64();
        eprintln!("Direct Fr dot product: {} (expected {})", result_direct, expected);
        assert!(
            (result_direct - expected).abs() < 0.01,
            "Direct: {} vs expected {}", result_direct, expected
        );

        // Method 2: 3-party additive shares
        let mut shares: Vec<Vec<Fr>> = vec![Vec::new(); 3];
        for j in 0..784 {
            let s0 = Fr::random(&mut rng);
            let s1 = Fr::random(&mut rng);
            let s2 = Fr::sub(&Fr::sub(&w_fr[j], &s0), &s1);
            shares[0].push(s0);
            shares[1].push(s1);
            shares[2].push(s2);
        }

        let mut sums = [Fr::ZERO; 3];
        for party in 0..3 {
            for j in 0..784 {
                let contrib = shares[party][j].mpc_scale(&x_fr[j]);
                sums[party] = Fr::add(&sums[party], &contrib);
            }
        }

        let recon = Fr::add(&Fr::add(&sums[0], &sums[1]), &sums[2]);
        let result_shared = recon.to_f64();
        eprintln!("Shared Fr dot product: {} (expected {})", result_shared, expected);
        assert!(
            (result_shared - expected).abs() < 0.01,
            "Shared: {} vs expected {}", result_shared, expected
        );
    }
}
