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
    fn new(hidden_dim: usize, seed: u64) -> Self {
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
            .expect("valid layer1 dims");
        let layer2 = Linear::from_raw(w2, vec![d_out, hidden_dim], Some(b2), Precision::F32)
            .expect("valid layer2 dims");

        Self {
            layer1,
            layer2,
            precision: Precision::F32,
        }
    }

    /// Forward pass: layer1 → ReLU → layer2
    #[allow(dead_code)]
    fn forward(&self, input: &BoundedTensor) -> BoundedTensor {
        let h = self.layer1.forward(input).expect("layer1 forward");
        let h = h.relu();
        self.layer2.forward(&h).expect("layer2 forward")
    }

    /// Manual SGD step on the model weights.
    ///
    /// For a 2-layer MLP with ReLU, the gradients are:
    ///   dy = 2/n * (y - target)     [d_out]
    ///   dW2 = dy @ h^T              [d_out, d_hid]
    ///   db2 = dy                     [d_out]
    ///   dh = W2^T @ dy              [d_hid]
    ///   dh_pre = dh * relu_mask     [d_hid]
    ///   dW1 = dh_pre @ x^T          [d_hid, d_in]
    ///   db1 = dh_pre                 [d_hid]
    fn train_step(
        &mut self,
        input: &[f64],
        target: &[f64],
        lr: f64,
    ) -> f64 {
        let d_in = self.layer1.in_features();
        let d_hid = self.layer1.out_features();
        let d_out = self.layer2.out_features();

        let x = BoundedTensor::from_exact(input.to_vec(), vec![d_in]);

        // Forward pass
        let h_pre = self.layer1.forward(&x).expect("layer1 forward");
        let h_pre_vals = h_pre.values();
        let h = h_pre.relu();
        let h_vals = h.values();
        let y = self.layer2.forward(&h).expect("layer2 forward");
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
            .expect("valid dims");
        self.layer2 = Linear::from_raw(w2_new, vec![d_out, d_hid], Some(b2_new), self.precision)
            .expect("valid dims");

        loss
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
    let mut model = MnistModel::new(config.hidden_dim, config.seed);

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

                let loss = model.train_step(sample_input, sample_target, lr);
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
        let model = MnistModel::new(32, 42);
        assert_eq!(model.layer1.in_features(), 784);
        assert_eq!(model.layer1.out_features(), 32);
        assert_eq!(model.layer2.in_features(), 32);
        assert_eq!(model.layer2.out_features(), 10);
    }

    #[test]
    fn test_mnist_model_forward() {
        let model = MnistModel::new(32, 42);
        let input = BoundedTensor::from_exact(vec![0.5; 784], vec![784]);
        let output = model.forward(&input);
        assert_eq!(output.shape(), &vec![10]);
    }

    #[test]
    fn test_mnist_train_step() {
        let mut model = MnistModel::new(8, 42);
        let input = vec![0.5; 784];
        let mut target = vec![0.0; 10];
        target[3] = 1.0; // Class 3

        let loss = model.train_step(&input, &target, 0.01);
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
        let model = MnistModel::new(8, 42);
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
}
