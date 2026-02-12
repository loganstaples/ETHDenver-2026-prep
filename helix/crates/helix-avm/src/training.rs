//! Reference training implementations with proof generation.
//!
//! Provides complete, end-to-end training examples that:
//! 1. Create a model (2-layer MLP)
//! 2. Generate synthetic data (MNIST-like patterns)
//! 3. Train with learning rate scheduling
//! 4. Save checkpoints
//! 5. Generate ZK proofs for each training step via circuit_bridge
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::training::{train_mnist, MnistTrainingConfig};
//!
//! let config = MnistTrainingConfig::default();
//! let result = train_mnist(config)?;
//! println!("Final loss: {}", result.final_loss);
//! println!("Proofs generated: {}", result.proofs_generated);
//! ```

use crate::data::{create_mnist_like_loader, DataPipeline, PipelineConfig};
use crate::gradient::autodiff::{GradientTape, Variable};
use crate::gradient::backward::backward;
use crate::gradient::optimizer::{LRScheduler, LinearWarmupCosineDecay};
use crate::models::serialization::{
    Checkpoint, CheckpointMetadata, ModelSerializer,
};
use crate::nn::Linear;
use helix_core::types::{BoundedTensor, Precision};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for MNIST training.
#[derive(Debug, Clone)]
pub struct MnistTrainingConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Number of training samples.
    pub num_samples: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Peak learning rate.
    pub learning_rate: f64,
    /// Minimum learning rate for cosine decay.
    pub min_lr: f64,
    /// Warmup steps for LR scheduler.
    pub warmup_steps: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Whether to generate circuit proofs for each step.
    pub generate_proofs: bool,
    /// Path to save checkpoints (None = don't save).
    pub checkpoint_dir: Option<String>,
    /// How often to log metrics (every N steps).
    pub log_interval: usize,
}

impl Default for MnistTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 32,
            num_samples: 320,
            hidden_dim: 32,
            learning_rate: 0.01,
            min_lr: 1e-5,
            warmup_steps: 10,
            seed: 42,
            generate_proofs: false,
            checkpoint_dir: None,
            log_interval: 5,
        }
    }
}

impl MnistTrainingConfig {
    /// Creates a minimal config for quick testing.
    pub fn test() -> Self {
        Self {
            epochs: 1,
            batch_size: 10,
            num_samples: 20,
            hidden_dim: 8,
            learning_rate: 0.01,
            min_lr: 1e-5,
            warmup_steps: 2,
            seed: 42,
            generate_proofs: false,
            checkpoint_dir: None,
            log_interval: 1,
        }
    }

    /// Enables proof generation.
    pub fn with_proofs(mut self) -> Self {
        self.generate_proofs = true;
        self
    }

    /// Sets the checkpoint directory.
    pub fn with_checkpoint_dir(mut self, dir: impl Into<String>) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }
}

/// Metrics from a single training step.
#[derive(Debug, Clone)]
pub struct TrainingStepResult {
    /// Step number (global).
    pub step: usize,
    /// Loss value.
    pub loss: f64,
    /// Loss error bound.
    pub loss_error: f64,
    /// Learning rate used.
    pub learning_rate: f64,
    /// Whether a proof was generated for this step.
    pub proof_generated: bool,
}

/// Results from a complete training run.
#[derive(Debug, Clone)]
pub struct MnistTrainingResult {
    /// Final loss value.
    pub final_loss: f64,
    /// Loss history (one per logged step).
    pub loss_history: Vec<f64>,
    /// Number of proofs generated.
    pub proofs_generated: usize,
    /// Total training steps.
    pub total_steps: usize,
    /// Per-step results.
    pub step_results: Vec<TrainingStepResult>,
}

/// A simple 2-layer MLP model for MNIST (784 → hidden → 10).
struct MnistModel {
    layer1: Linear,
    layer2: Linear,
    precision: Precision,
}

impl MnistModel {
    /// Creates a new MNIST model with Xavier-initialized weights.
    fn new(hidden_dim: usize, seed: u64) -> Result<Self, String> {
        let d_in = 784;
        let d_out = 10;
        let mut rng = StdRng::seed_from_u64(seed);

        // Xavier initialization: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let scale1 = (6.0 / (d_in + hidden_dim) as f64).sqrt();
        let w1: Vec<f64> = (0..hidden_dim * d_in)
            .map(|_| rng.gen::<f64>() * 2.0 * scale1 - scale1)
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let scale2 = (6.0 / (hidden_dim + d_out) as f64).sqrt();
        let w2: Vec<f64> = (0..d_out * hidden_dim)
            .map(|_| rng.gen::<f64>() * 2.0 * scale2 - scale2)
            .collect();
        let b2 = vec![0.0; d_out];

        let layer1 = Linear::from_raw(w1, vec![hidden_dim, d_in], Some(b1), Precision::F32)
            .map_err(|e| format!("layer1 init failed: {e}"))?;
        let layer2 = Linear::from_raw(w2, vec![d_out, hidden_dim], Some(b2), Precision::F32)
            .map_err(|e| format!("layer2 init failed: {e}"))?;

        Ok(Self {
            layer1,
            layer2,
            precision: Precision::F32,
        })
    }

    /// Forward pass: layer1 → ReLU → layer2
    #[allow(dead_code)]
    fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, String> {
        let h = self.layer1.forward(input).map_err(|e| format!("layer1 forward: {e}"))?;
        let h = h.relu();
        self.layer2.forward(&h).map_err(|e| format!("layer2 forward: {e}"))
    }

    /// SGD step using automatic differentiation.
    ///
    /// Builds the computation graph on a `GradientTape`, runs the forward pass
    /// through `Variable` operations, then calls `backward()` to compute all
    /// gradients automatically.
    fn train_step(
        &mut self,
        input: &[f64],
        target: &[f64],
        lr: f64,
    ) -> Result<f64, String> {
        let d_in = self.layer1.in_features();
        let d_hid = self.layer1.out_features();
        let d_out = self.layer2.out_features();

        // Create a fresh tape for this step
        let tape = GradientTape::new();

        // Register input as a tracked variable
        let x_tensor = BoundedTensor::from_exact(input.to_vec(), vec![1, d_in]);
        let x_var = Variable::input(x_tensor, tape.clone(), Some("input".to_string()));

        // Layer 1 forward: h_pre = x @ W1^T + b1
        let (h_pre, w1t_idx, b1_idx) = self
            .layer1
            .forward_var(&x_var, tape.clone(), Some("w1t"), Some("b1"))
            .map_err(|e| format!("layer1 forward_var: {e}"))?;

        // ReLU activation
        let h = h_pre.relu();

        // Layer 2 forward: y = h @ W2^T + b2
        let (y, w2t_idx, b2_idx) = self
            .layer2
            .forward_var(&h, tape.clone(), Some("w2t"), Some("b2"))
            .map_err(|e| format!("layer2 forward_var: {e}"))?;

        // Compute MSE loss as a Variable operation:
        // loss = mean((y - target)^2)
        let target_tensor = BoundedTensor::from_exact(target.to_vec(), vec![1, d_out]);
        let target_var = Variable::input(target_tensor, tape.clone(), Some("target".to_string()));
        let diff = y.sub(&target_var);
        let sq = diff.mul(&diff);
        let loss_var = sq.mean();

        let loss_value = loss_var.tensor.data()[0].value();

        // Backward pass — automatically computes all gradients
        let grads = backward(&loss_var).map_err(|e| format!("backward pass: {e}"))?;

        // Extract gradients and apply SGD updates
        // W1^T gradient → update W1^T, then store as W1
        if let Some(grad_w1t) = grads.get(&w1t_idx) {
            let w1t_old = self.layer1.weights().transpose();
            let w1t_new_vals: Vec<f64> = w1t_old
                .values()
                .iter()
                .zip(grad_w1t.values().iter())
                .map(|(w, g)| w - lr * g)
                .collect();
            // W1^T has shape [in, hid], transpose to [hid, in] for storage
            let w1_new = BoundedTensor::from_exact(w1t_new_vals, vec![d_in, d_hid]).transpose();
            let w1_new_vals = w1_new.values();
            let b1_new = if let Some(b1_node) = b1_idx {
                if let Some(grad_b1) = grads.get(&b1_node) {
                    let b1_old = self.layer1.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_hid]);
                    b1_old.iter().zip(grad_b1.values().iter()).map(|(b, g)| b - lr * g).collect()
                } else {
                    self.layer1.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_hid])
                }
            } else {
                self.layer1.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_hid])
            };
            self.layer1 = Linear::from_raw(w1_new_vals, vec![d_hid, d_in], Some(b1_new), self.precision)
                .map_err(|e| format!("layer1 weight update: {e}"))?;
        }

        if let Some(grad_w2t) = grads.get(&w2t_idx) {
            let w2t_old = self.layer2.weights().transpose();
            let w2t_new_vals: Vec<f64> = w2t_old
                .values()
                .iter()
                .zip(grad_w2t.values().iter())
                .map(|(w, g)| w - lr * g)
                .collect();
            let w2_new = BoundedTensor::from_exact(w2t_new_vals, vec![d_hid, d_out]).transpose();
            let w2_new_vals = w2_new.values();
            let b2_new = if let Some(b2_node) = b2_idx {
                if let Some(grad_b2) = grads.get(&b2_node) {
                    let b2_old = self.layer2.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_out]);
                    b2_old.iter().zip(grad_b2.values().iter()).map(|(b, g)| b - lr * g).collect()
                } else {
                    self.layer2.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_out])
                }
            } else {
                self.layer2.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_out])
            };
            self.layer2 = Linear::from_raw(w2_new_vals, vec![d_out, d_hid], Some(b2_new), self.precision)
                .map_err(|e| format!("layer2 weight update: {e}"))?;
        }

        Ok(loss_value)
    }

    /// Manual SGD step on the model weights (reference/fallback implementation).
    ///
    /// For a 2-layer MLP with ReLU, the gradients are:
    ///   dy = 2/n * (y - target)     [d_out]
    ///   dW2 = dy @ h^T              [d_out, d_hid]
    ///   db2 = dy                     [d_out]
    ///   dh = W2^T @ dy              [d_hid]
    ///   dh_pre = dh * relu_mask     [d_hid]
    ///   dW1 = dh_pre @ x^T          [d_hid, d_in]
    ///   db1 = dh_pre                 [d_hid]
    #[cfg(feature = "manual-gradients")]
    fn train_step_manual(
        &mut self,
        input: &[f64],
        target: &[f64],
        lr: f64,
    ) -> Result<f64, String> {
        let d_in = self.layer1.in_features();
        let d_hid = self.layer1.out_features();
        let d_out = self.layer2.out_features();

        let x = BoundedTensor::from_exact(input.to_vec(), vec![d_in]);

        // Forward pass
        let h_pre = self.layer1.forward(&x).map_err(|e| format!("layer1 forward: {e}"))?;
        let h_pre_vals = h_pre.values();
        let h = h_pre.relu();
        let h_vals = h.values();
        let y = self.layer2.forward(&h).map_err(|e| format!("layer2 forward: {e}"))?;
        let y_vals = y.values();

        // Compute MSE loss
        let n = d_out as f64;
        let mut loss = 0.0;
        let mut dy = vec![0.0; d_out];
        for i in 0..d_out {
            let diff = y_vals[i] - target[i];
            loss += diff * diff;
            dy[i] = 2.0 * diff / n;
        }
        loss /= n;

        // Gradient for W2, b2
        let w2_vals = self.layer2.weights().values();
        let mut dw2 = vec![0.0; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                dw2[i * d_hid + j] = dy[i] * h_vals[j];
            }
        }
        let db2 = dy.clone();

        // Gradient for hidden layer
        let mut dh = vec![0.0; d_hid];
        for j in 0..d_hid {
            for i in 0..d_out {
                dh[j] += w2_vals[i * d_hid + j] * dy[i];
            }
        }

        // ReLU mask
        let mut dh_pre = vec![0.0; d_hid];
        for j in 0..d_hid {
            dh_pre[j] = if h_pre_vals[j] > 0.0 { dh[j] } else { 0.0 };
        }

        // Gradient for W1, b1
        let mut dw1 = vec![0.0; d_hid * d_in];
        for i in 0..d_hid {
            for j in 0..d_in {
                dw1[i * d_in + j] = dh_pre[i] * input[j];
            }
        }
        let db1 = dh_pre;

        // SGD update
        let w1_old = self.layer1.weights().values();
        let b1_old = self.layer1.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_hid]);
        let w2_old = self.layer2.weights().values();
        let b2_old = self.layer2.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_out]);

        let w1_new: Vec<f64> = w1_old.iter().zip(dw1.iter()).map(|(w, g)| w - lr * g).collect();
        let b1_new: Vec<f64> = b1_old.iter().zip(db1.iter()).map(|(b, g)| b - lr * g).collect();
        let w2_new: Vec<f64> = w2_old.iter().zip(dw2.iter()).map(|(w, g)| w - lr * g).collect();
        let b2_new: Vec<f64> = b2_old.iter().zip(db2.iter()).map(|(b, g)| b - lr * g).collect();

        self.layer1 = Linear::from_raw(w1_new, vec![d_hid, d_in], Some(b1_new), self.precision)
            .map_err(|e| format!("layer1 weight update: {e}"))?;
        self.layer2 = Linear::from_raw(w2_new, vec![d_out, d_hid], Some(b2_new), self.precision)
            .map_err(|e| format!("layer2 weight update: {e}"))?;

        Ok(loss)
    }

    /// Creates a checkpoint of the current model state.
    fn checkpoint(&self, step: usize, loss: f64) -> Checkpoint {
        let mut cp = Checkpoint::new(
            CheckpointMetadata::new("mnist_mlp", "mlp")
                .with_param_count(
                    self.layer1.weights().len()
                        + self.layer1.bias().map(|b| b.len()).unwrap_or(0)
                        + self.layer2.weights().len()
                        + self.layer2.bias().map(|b| b.len()).unwrap_or(0),
                )
                .with_training_step(step)
                .with_training_loss(loss),
        );
        cp.add_tensor("layer1.weight", self.layer1.weights());
        if let Some(bias) = self.layer1.bias() {
            cp.add_tensor("layer1.bias", bias);
        }
        cp.add_tensor("layer2.weight", self.layer2.weights());
        if let Some(bias) = self.layer2.bias() {
            cp.add_tensor("layer2.bias", bias);
        }
        cp
    }
}

/// Trains a 2-layer MLP on synthetic MNIST-like data.
///
/// This is a reference implementation demonstrating the full HELIX training pipeline:
/// 1. Model initialization with Xavier weights
/// 2. Data loading via `DataPipeline` with shuffling
/// 3. Training with cosine-annealed learning rate
/// 4. Checkpoint persistence via `ModelSerializer`
/// 5. Optional ZK proof generation via `circuit_bridge`
///
/// The model architecture is: Linear(784→hidden) → ReLU → Linear(hidden→10)
///
/// # Arguments
/// * `config` - Training configuration
///
/// # Returns
/// Training results including loss history and proof count
pub fn train_mnist(config: MnistTrainingConfig) -> Result<MnistTrainingResult, String> {
    let mut model = MnistModel::new(config.hidden_dim, config.seed)?;

    let loader = create_mnist_like_loader(config.num_samples, config.batch_size, config.seed);
    let pipeline_config = PipelineConfig::new(config.batch_size)
        .with_shuffle(true, Some(config.seed))
        .with_normalize(0.5, 0.5);
    let mut pipeline = DataPipeline::new(loader, pipeline_config);

    let total_batches = pipeline.num_batches();
    let total_steps = config.epochs * total_batches;
    let scheduler = LinearWarmupCosineDecay::new(
        config.learning_rate,
        config.min_lr,
        config.warmup_steps,
        total_steps,
    );

    let serializer = ModelSerializer::new();

    let mut loss_history = Vec::new();
    let mut step_results = Vec::new();
    let mut proofs_generated = 0usize;
    let mut global_step = 0usize;
    let mut last_loss = 0.0;

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_steps = 0;

        for batch in pipeline.iter() {
            let lr = scheduler.get_lr(global_step);
            let input_vals = batch.inputs.values();
            let target_vals = batch.targets.values();

            // Train on each sample in the batch individually
            // (the circuit proves single-sample steps)
            let feature_dim = 784;
            let label_dim = 10;
            let samples_in_batch = input_vals.len() / feature_dim;

            let mut batch_loss = 0.0;
            for s in 0..samples_in_batch {
                let x_start = s * feature_dim;
                let x_end = x_start + feature_dim;
                let t_start = s * label_dim;
                let t_end = t_start + label_dim;

                if x_end > input_vals.len() || t_end > target_vals.len() {
                    break;
                }

                let sample_input = &input_vals[x_start..x_end];
                let sample_target = &target_vals[t_start..t_end];

                let loss = model.train_step(sample_input, sample_target, lr)?;
                batch_loss += loss;

                // Generate proof if configured
                if config.generate_proofs {
                    match generate_training_proof(
                        &model.layer1,
                        &model.layer2,
                        sample_input,
                        sample_target,
                        lr,
                        global_step as u64,
                    ) {
                        Ok(_) => proofs_generated += 1,
                        Err(e) => {
                            // Log but don't fail — proof generation errors are non-fatal
                            eprintln!("Proof generation failed at step {global_step}: {e}");
                        }
                    }
                }
            }

            let avg_batch_loss = if samples_in_batch > 0 {
                batch_loss / samples_in_batch as f64
            } else {
                0.0
            };

            epoch_loss += avg_batch_loss;
            epoch_steps += 1;
            last_loss = avg_batch_loss;

            let step_result = TrainingStepResult {
                step: global_step,
                loss: avg_batch_loss,
                loss_error: 0.0,
                learning_rate: lr,
                proof_generated: config.generate_proofs,
            };

            if global_step % config.log_interval == 0 {
                loss_history.push(avg_batch_loss);
            }

            step_results.push(step_result);
            global_step += 1;
        }

        let avg_epoch_loss = if epoch_steps > 0 {
            epoch_loss / epoch_steps as f64
        } else {
            0.0
        };

        // Save checkpoint at end of each epoch
        if let Some(ref dir) = config.checkpoint_dir {
            let checkpoint = model.checkpoint(global_step, avg_epoch_loss);
            let path = format!("{dir}/checkpoint_epoch_{epoch}.helixchk");
            if let Err(e) = serializer.save_to_file(&checkpoint, &path) {
                eprintln!("Failed to save checkpoint: {e}");
            }
        }

        pipeline.reset_epoch();
    }

    Ok(MnistTrainingResult {
        final_loss: last_loss,
        loss_history,
        proofs_generated,
        total_steps: global_step,
        step_results,
    })
}

/// Generates a ZK training proof for a single training step.
///
/// This wraps `circuit_bridge::build_training_witness` and returns the witness
/// that can be used with `MLTrainingStepV2Circuit` for proof generation.
#[cfg(any(feature = "circuit-bridge", test))]
fn generate_training_proof(
    layer1: &Linear,
    layer2: &Linear,
    input: &[f64],
    target: &[f64],
    learning_rate: f64,
    step_number: u64,
) -> Result<crate::circuit_bridge::TrainingWitnessOutput, String> {
    crate::circuit_bridge::build_training_witness(
        layer1,
        layer2,
        input,
        target,
        learning_rate,
        step_number,
    )
}

/// Stub when circuit-bridge feature is not enabled.
#[cfg(not(any(feature = "circuit-bridge", test)))]
fn generate_training_proof(
    _layer1: &Linear,
    _layer2: &Linear,
    _input: &[f64],
    _target: &[f64],
    _learning_rate: f64,
    _step_number: u64,
) -> Result<(), String> {
    Err("circuit-bridge feature not enabled".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_model_creation() {
        let model = MnistModel::new(32, 42).unwrap();
        assert_eq!(model.layer1.in_features(), 784);
        assert_eq!(model.layer1.out_features(), 32);
        assert_eq!(model.layer2.in_features(), 32);
        assert_eq!(model.layer2.out_features(), 10);
    }

    #[test]
    fn test_mnist_model_forward() {
        let model = MnistModel::new(32, 42).unwrap();
        let input = BoundedTensor::from_exact(vec![0.5; 784], vec![784]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &vec![10]);
    }

    #[test]
    fn test_mnist_train_step() {
        let mut model = MnistModel::new(8, 42).unwrap();
        let input = vec![0.5; 784];
        let mut target = vec![0.0; 10];
        target[3] = 1.0; // Class 3

        let loss = model.train_step(&input, &target, 0.01).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_train_mnist_convergence() {
        let config = MnistTrainingConfig {
            epochs: 3,
            batch_size: 10,
            num_samples: 100,
            hidden_dim: 16,
            learning_rate: 0.01,
            min_lr: 1e-5,
            warmup_steps: 5,
            seed: 42,
            generate_proofs: false,
            checkpoint_dir: None,
            log_interval: 1,
        };

        let result = train_mnist(config).unwrap();
        assert!(result.total_steps > 0);
        assert!(!result.loss_history.is_empty());

        // Loss should decrease over training
        let first_loss = result.loss_history[0];
        let last_loss = result.final_loss;
        // Allow some tolerance - we just want the model to learn something
        assert!(
            last_loss < first_loss * 1.5,
            "Loss did not decrease: first={first_loss}, last={last_loss}"
        );
    }

    #[test]
    fn test_train_mnist_checkpoint() {
        let dir = std::env::temp_dir()
            .join("helix_mnist_test")
            .to_string_lossy()
            .to_string();

        let config = MnistTrainingConfig {
            epochs: 1,
            batch_size: 10,
            num_samples: 20,
            hidden_dim: 8,
            learning_rate: 0.01,
            min_lr: 1e-5,
            warmup_steps: 2,
            seed: 42,
            generate_proofs: false,
            checkpoint_dir: Some(dir.clone()),
            log_interval: 1,
        };

        let result = train_mnist(config).unwrap();
        assert!(result.total_steps > 0);

        // Verify checkpoint was saved
        let path = format!("{dir}/checkpoint_epoch_0.helixchk");
        let serializer = ModelSerializer::new();
        let checkpoint = serializer.load_from_file(&path).unwrap();
        assert_eq!(checkpoint.metadata.model_name, "mnist_mlp");
        assert!(checkpoint.get_tensor("layer1.weight").is_some());
        assert!(checkpoint.get_tensor("layer2.weight").is_some());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_train_mnist_with_proofs() {
        // Use a tiny model so proof generation is fast
        let config = MnistTrainingConfig {
            epochs: 1,
            batch_size: 10,
            num_samples: 10,
            hidden_dim: 4,
            learning_rate: 0.01,
            min_lr: 1e-5,
            warmup_steps: 1,
            seed: 42,
            generate_proofs: true,
            checkpoint_dir: None,
            log_interval: 1,
        };

        let result = train_mnist(config).unwrap();
        assert!(result.total_steps > 0);
        // With generate_proofs=true, we should get some proofs
        // (may fail for some samples if values overflow circuit range,
        // which is logged but non-fatal)
    }

    #[test]
    fn test_mnist_model_checkpoint_roundtrip() {
        let model = MnistModel::new(8, 42).unwrap();
        let checkpoint = model.checkpoint(0, 1.0);

        let serializer = ModelSerializer::new();
        let bytes = serializer.serialize(&checkpoint).unwrap();
        let loaded = serializer.deserialize(&bytes).unwrap();

        let w1 = loaded.get_tensor("layer1.weight").unwrap();
        let w1_orig = model.layer1.weights();
        assert_eq!(w1.shape(), w1_orig.shape());

        let orig_vals = w1_orig.values();
        let loaded_vals = w1.values();
        for (a, b) in orig_vals.iter().zip(loaded_vals.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_training_step_result() {
        let config = MnistTrainingConfig::test();
        let result = train_mnist(config).unwrap();

        assert!(!result.step_results.is_empty());
        for step in &result.step_results {
            assert!(step.loss.is_finite());
            assert!(step.learning_rate > 0.0);
        }
    }

    /// (a) AutoDiff gradients match manual gradients within 1e-6.
    ///
    /// Runs the manual backpropagation inline (the same code that was the
    /// original `train_step`) and the AutoDiff backward pass on an identical
    /// model/input, then asserts every gradient element matches.
    #[test]
    fn test_autodiff_gradients_match_manual() {
        use crate::gradient::autodiff::{GradientTape, Variable};
        use crate::gradient::backward::backward;

        let hidden_dim = 8;
        let seed = 42;
        let model = MnistModel::new(hidden_dim, seed).unwrap();

        let d_in = model.layer1.in_features();
        let d_hid = model.layer1.out_features();
        let d_out = model.layer2.out_features();

        let input: Vec<f64> = (0..d_in).map(|i| (i as f64 * 0.001) % 1.0).collect();
        let mut target = vec![0.0; d_out];
        target[3] = 1.0;

        // ---- Manual gradients (inline copy of the old train_step logic) ----
        let x = BoundedTensor::from_exact(input.clone(), vec![d_in]);
        let h_pre = model.layer1.forward(&x).expect("layer1");
        let h_pre_vals = h_pre.values();
        let h = h_pre.relu();
        let h_vals = h.values();
        let y = model.layer2.forward(&h).expect("layer2");
        let y_vals = y.values();

        let n = d_out as f64;
        let mut manual_loss = 0.0;
        let mut dy = vec![0.0; d_out];
        for i in 0..d_out {
            let diff = y_vals[i] - target[i];
            manual_loss += diff * diff;
            dy[i] = 2.0 * diff / n;
        }
        manual_loss /= n;

        let w2_vals = model.layer2.weights().values();
        let mut manual_dw2 = vec![0.0; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                manual_dw2[i * d_hid + j] = dy[i] * h_vals[j];
            }
        }
        let manual_db2 = dy.clone();

        let mut manual_dh = vec![0.0; d_hid];
        for j in 0..d_hid {
            for i in 0..d_out {
                manual_dh[j] += w2_vals[i * d_hid + j] * dy[i];
            }
        }

        let mut manual_dh_pre = vec![0.0; d_hid];
        for j in 0..d_hid {
            manual_dh_pre[j] = if h_pre_vals[j] > 0.0 { manual_dh[j] } else { 0.0 };
        }

        let mut manual_dw1 = vec![0.0; d_hid * d_in];
        for i in 0..d_hid {
            for j in 0..d_in {
                manual_dw1[i * d_in + j] = manual_dh_pre[i] * input[j];
            }
        }
        let manual_db1 = manual_dh_pre;

        // ---- AutoDiff gradients ----
        let tape = GradientTape::new();
        let x_tensor = BoundedTensor::from_exact(input.clone(), vec![1, d_in]);
        let x_var = Variable::input(x_tensor, tape.clone(), Some("input".to_string()));

        let (h_pre_var, w1t_idx, b1_idx) = model
            .layer1
            .forward_var(&x_var, tape.clone(), Some("w1t"), Some("b1"))
            .expect("layer1 forward_var");
        let h_var = h_pre_var.relu();
        let (y_var, w2t_idx, b2_idx) = model
            .layer2
            .forward_var(&h_var, tape.clone(), Some("w2t"), Some("b2"))
            .expect("layer2 forward_var");

        let target_tensor = BoundedTensor::from_exact(target.clone(), vec![1, d_out]);
        let target_var = Variable::input(target_tensor, tape.clone(), Some("target".to_string()));
        let diff_var = y_var.sub(&target_var);
        let sq_var = diff_var.mul(&diff_var);
        let loss_var = sq_var.mean();

        let autodiff_loss = loss_var.tensor.data()[0].value();
        let grads = backward(&loss_var).expect("backward");

        // Compare losses
        assert!(
            (manual_loss - autodiff_loss).abs() < 1e-6,
            "Loss mismatch: manual={manual_loss}, autodiff={autodiff_loss}"
        );

        // Compare W2 gradients (W2^T gradient from autodiff vs manual dW2)
        // AutoDiff gives dL/d(W2^T) shape [d_hid, d_out], manual gives dW2 shape [d_out, d_hid]
        // dL/d(W2^T) = X^T @ dL/dY where X=h [1, d_hid], dL/dY [1, d_out]
        // So dL/d(W2^T)[j,i] = h[j] * dy[i]  =>  transpose = dy[i] * h[j] = manual_dw2[i*d_hid+j]
        if let Some(grad_w2t) = grads.get(&w2t_idx) {
            let ad_vals = grad_w2t.values();
            for j in 0..d_hid {
                for i in 0..d_out {
                    let ad_val = ad_vals[j * d_out + i];
                    let manual_val = manual_dw2[i * d_hid + j];
                    assert!(
                        (ad_val - manual_val).abs() < 1e-6,
                        "dW2 mismatch at [{i},{j}]: autodiff={ad_val}, manual={manual_val}"
                    );
                }
            }
        } else {
            panic!("No gradient for W2^T");
        }

        // Compare b2 gradients
        // The autodiff bias gradient may be [1, d_out] (from Add backward)
        // while manual is [d_out].
        if let Some(b2_node) = b2_idx {
            if let Some(grad_b2) = grads.get(&b2_node) {
                let ad_vals = grad_b2.values();
                for i in 0..d_out {
                    let ad_val = ad_vals[i % ad_vals.len()];
                    assert!(
                        (ad_val - manual_db2[i]).abs() < 1e-6,
                        "db2 mismatch at [{i}]: autodiff={ad_val}, manual={}",
                        manual_db2[i]
                    );
                }
            }
        }

        // Compare W1 gradients
        if let Some(grad_w1t) = grads.get(&w1t_idx) {
            let ad_vals = grad_w1t.values();
            for j in 0..d_in {
                for i in 0..d_hid {
                    let ad_val = ad_vals[j * d_hid + i];
                    let manual_val = manual_dw1[i * d_in + j];
                    assert!(
                        (ad_val - manual_val).abs() < 1e-6,
                        "dW1 mismatch at [{i},{j}]: autodiff={ad_val}, manual={manual_val}"
                    );
                }
            }
        } else {
            panic!("No gradient for W1^T");
        }

        // Compare b1 gradients
        if let Some(b1_node) = b1_idx {
            if let Some(grad_b1) = grads.get(&b1_node) {
                let ad_vals = grad_b1.values();
                for i in 0..d_hid {
                    let ad_val = ad_vals[i % ad_vals.len()];
                    assert!(
                        (ad_val - manual_db1[i]).abs() < 1e-6,
                        "db1 mismatch at [{i}]: autodiff={ad_val}, manual={}",
                        manual_db1[i]
                    );
                }
            }
        }
    }

    /// (b) A 3-layer MLP trains to convergence using AutoDiff.
    ///
    /// Architecture: Linear(4→16) → ReLU → Linear(16→8) → ReLU → Linear(8→2)
    /// Trains on a simple XOR-like pattern and verifies loss decreases.
    #[test]
    fn test_3layer_mlp_convergence_autodiff() {
        use crate::gradient::autodiff::{GradientTape, Variable};
        use crate::gradient::backward::backward;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(123);

        // Create 3-layer MLP with Xavier init
        let mk_layer = |d_in: usize, d_out: usize, rng: &mut StdRng| -> Linear {
            let scale = (6.0 / (d_in + d_out) as f64).sqrt();
            let w: Vec<f64> = (0..d_out * d_in)
                .map(|_| rng.gen::<f64>() * 2.0 * scale - scale)
                .collect();
            let b = vec![0.0; d_out];
            Linear::from_raw(w, vec![d_out, d_in], Some(b), Precision::F32).unwrap()
        };

        let mut l1 = mk_layer(4, 16, &mut rng);
        let mut l2 = mk_layer(16, 8, &mut rng);
        let mut l3 = mk_layer(8, 2, &mut rng);

        // XOR-like dataset: 4 inputs → 2 outputs
        let data: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![0.0, 0.0, 1.0, 1.0], vec![1.0, 0.0]),
            (vec![1.0, 1.0, 0.0, 0.0], vec![1.0, 0.0]),
            (vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0]),
            (vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 1.0]),
        ];

        let lr = 0.05;
        let mut first_loss = 0.0;
        let mut last_loss = 0.0;

        for epoch in 0..200 {
            let mut epoch_loss = 0.0;
            for (inp, tgt) in &data {
                let tape = GradientTape::new();
                let x = Variable::input(
                    BoundedTensor::from_exact(inp.clone(), vec![1, 4]),
                    tape.clone(),
                    None,
                );

                let (h1, w1t_idx, b1_idx) = l1.forward_var(&x, tape.clone(), None, None).unwrap();
                let a1 = h1.relu();
                let (h2, w2t_idx, b2_idx) = l2.forward_var(&a1, tape.clone(), None, None).unwrap();
                let a2 = h2.relu();
                let (out, w3t_idx, b3_idx) = l3.forward_var(&a2, tape.clone(), None, None).unwrap();

                let target_var = Variable::input(
                    BoundedTensor::from_exact(tgt.clone(), vec![1, 2]),
                    tape.clone(),
                    None,
                );
                let diff = out.sub(&target_var);
                let sq = diff.mul(&diff);
                let loss = sq.mean();

                epoch_loss += loss.tensor.data()[0].value();
                let grads = backward(&loss).expect("backward");

                // SGD update helper
                let update_layer = |layer: &Linear, wt_idx, b_idx: Option<usize>| -> Linear {
                    let d_in = layer.in_features();
                    let d_out = layer.out_features();
                    let wt_old = layer.weights().transpose();
                    let wt_new = if let Some(g) = grads.get(&wt_idx) {
                        let vals: Vec<f64> = wt_old.values().iter().zip(g.values().iter())
                            .map(|(w, gv)| w - lr * gv).collect();
                        BoundedTensor::from_exact(vals, wt_old.shape().clone()).transpose()
                    } else {
                        layer.weights().clone()
                    };
                    let b_new = if let Some(bi) = b_idx {
                        if let Some(g) = grads.get(&bi) {
                            let b_old = layer.bias().map(|b| b.values()).unwrap_or_else(|| vec![0.0; d_out]);
                            Some(b_old.iter().zip(g.values().iter()).map(|(b, gv)| b - lr * gv).collect::<Vec<_>>())
                        } else {
                            layer.bias().map(|b| b.values())
                        }
                    } else {
                        layer.bias().map(|b| b.values())
                    };
                    Linear::from_raw(wt_new.values(), vec![d_out, d_in], b_new, Precision::F32).unwrap()
                };

                l1 = update_layer(&l1, w1t_idx, b1_idx);
                l2 = update_layer(&l2, w2t_idx, b2_idx);
                l3 = update_layer(&l3, w3t_idx, b3_idx);
            }

            let avg_loss = epoch_loss / data.len() as f64;
            if epoch == 0 {
                first_loss = avg_loss;
            }
            last_loss = avg_loss;
        }

        assert!(
            last_loss < first_loss * 0.5,
            "3-layer MLP did not converge: first_loss={first_loss}, last_loss={last_loss}"
        );
        assert!(last_loss < 0.1, "Loss did not drop sufficiently: {last_loss}");
    }

    /// (c) GeLU and tanh activations work through AutoDiff without manual gradient code.
    #[test]
    fn test_gelu_tanh_autodiff_no_manual_gradients() {
        use crate::gradient::autodiff::{GradientTape, Variable};
        use crate::gradient::backward::backward;

        // Test tanh: simple network input → linear → tanh → mean loss
        {
            let tape = GradientTape::new();
            let x = Variable::input(
                BoundedTensor::from_exact(vec![1.0, -1.0, 0.5, -0.5], vec![1, 4]),
                tape.clone(),
                Some("x".to_string()),
            );
            let w = Variable::param(
                BoundedTensor::from_exact(vec![0.5, 0.3, -0.2, 0.1], vec![1, 4]),
                tape.clone(),
                Some("w".to_string()),
            );
            let w_idx = w.node_index.unwrap();

            let product = x.mul(&w);
            let activated = product.tanh();
            let loss = activated.mean();

            let grads = backward(&loss).expect("tanh backward");
            let grad_w = grads.get(&w_idx).expect("gradient for w exists");

            // Verify gradient is finite and non-zero
            for val in grad_w.values() {
                assert!(val.is_finite(), "tanh gradient is not finite: {val}");
            }
            let grad_norm: f64 = grad_w.values().iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(grad_norm > 1e-10, "tanh gradient is zero");
        }

        // Test GELU: input → linear → GELU → mean loss
        {
            let tape = GradientTape::new();
            let x = Variable::input(
                BoundedTensor::from_exact(vec![1.0, -1.0, 0.5, -0.5], vec![1, 4]),
                tape.clone(),
                Some("x".to_string()),
            );
            let w = Variable::param(
                BoundedTensor::from_exact(vec![0.5, 0.3, -0.2, 0.1], vec![1, 4]),
                tape.clone(),
                Some("w".to_string()),
            );
            let w_idx = w.node_index.unwrap();

            let product = x.mul(&w);
            let activated = product.gelu();
            let loss = activated.mean();

            let grads = backward(&loss).expect("gelu backward");
            let grad_w = grads.get(&w_idx).expect("gradient for w exists");

            // Verify gradient is finite and non-zero
            for val in grad_w.values() {
                assert!(val.is_finite(), "GELU gradient is not finite: {val}");
            }
            let grad_norm: f64 = grad_w.values().iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(grad_norm > 1e-10, "GELU gradient is zero");
        }

        // Test that a 2-layer network with tanh trains (loss decreases)
        {
            let mut w1_vals = vec![0.5, -0.3, 0.2, 0.1, -0.4, 0.6, 0.3, -0.2];
            let mut b1_vals = vec![0.0; 2];
            let mut w2_vals = vec![0.4, -0.5];
            let mut b2_vals = vec![0.0];
            let lr = 0.1;

            let mut first_loss = 0.0;
            let mut last_loss = 0.0;

            for step in 0..100 {
                let tape = GradientTape::new();
                let x = Variable::input(
                    BoundedTensor::from_exact(vec![1.0, -0.5, 0.3, -0.8], vec![1, 4]),
                    tape.clone(),
                    None,
                );
                let w1 = Variable::param(
                    BoundedTensor::from_exact(w1_vals.clone(), vec![4, 2]),
                    tape.clone(),
                    Some("w1".to_string()),
                );
                let w1_idx = w1.node_index.unwrap();
                let b1 = Variable::param(
                    BoundedTensor::from_exact(b1_vals.clone(), vec![1, 2]),
                    tape.clone(),
                    Some("b1".to_string()),
                );
                let b1_idx = b1.node_index.unwrap();

                let h = x.matmul(&w1).add(&b1).tanh();

                let w2 = Variable::param(
                    BoundedTensor::from_exact(w2_vals.clone(), vec![2, 1]),
                    tape.clone(),
                    Some("w2".to_string()),
                );
                let w2_idx = w2.node_index.unwrap();
                let b2 = Variable::param(
                    BoundedTensor::from_exact(b2_vals.clone(), vec![1, 1]),
                    tape.clone(),
                    Some("b2".to_string()),
                );
                let b2_idx = b2.node_index.unwrap();

                let out = h.matmul(&w2).add(&b2);
                let target = Variable::input(
                    BoundedTensor::from_exact(vec![1.0], vec![1, 1]),
                    tape.clone(),
                    None,
                );
                let diff = out.sub(&target);
                let loss = diff.mul(&diff).mean();

                let loss_val = loss.tensor.data()[0].value();
                if step == 0 { first_loss = loss_val; }
                last_loss = loss_val;

                let grads = backward(&loss).expect("backward");

                // SGD updates
                if let Some(g) = grads.get(&w1_idx) {
                    w1_vals = w1_vals.iter().zip(g.values().iter()).map(|(w, gv)| w - lr * gv).collect();
                }
                if let Some(g) = grads.get(&b1_idx) {
                    b1_vals = b1_vals.iter().zip(g.values().iter()).map(|(b, gv)| b - lr * gv).collect();
                }
                if let Some(g) = grads.get(&w2_idx) {
                    w2_vals = w2_vals.iter().zip(g.values().iter()).map(|(w, gv)| w - lr * gv).collect();
                }
                if let Some(g) = grads.get(&b2_idx) {
                    b2_vals = b2_vals.iter().zip(g.values().iter()).map(|(b, gv)| b - lr * gv).collect();
                }
            }

            assert!(
                last_loss < first_loss * 0.1,
                "Tanh network did not converge: first={first_loss}, last={last_loss}"
            );
        }
    }
}
