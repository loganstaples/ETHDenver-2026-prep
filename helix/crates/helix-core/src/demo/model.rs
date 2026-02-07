//! Demo Model Training for HELIX.
//!
//! Provides a simple demo model using BoundedTensors and the Provable trait
//! to showcase HELIX's error-bounded computation and witness generation
//! for the distributed verifiable training pipeline.

use crate::error::HelixResult;
use crate::traits::{Provable, SimpleWitness, Witness};
use crate::types::{BoundedTensor, BoundedValue, ErrorMargin};
use serde::{Deserialize, Serialize};

/// Fixed-point quantization scale (12 decimal places, matches TensorWitness::SCALE).
const QUANTIZE_SCALE: f64 = 1e12;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Demo model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoModelConfig {
    /// Model name.
    pub name: String,
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Activation function.
    pub activation: String,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for DemoModelConfig {
    fn default() -> Self {
        Self {
            name: "helix-demo-mlp".to_string(),
            input_dim: 784,  // MNIST-like
            hidden_dim: 256,
            output_dim: 10,
            num_layers: 3,
            activation: "relu".to_string(),
            dropout: 0.1,
        }
    }
}

// =============================================================================
// DEMO MODEL
// =============================================================================

/// Simple demo model for showcasing HELIX with error-bounded tensors.
///
/// All weights and biases are stored as `BoundedTensor` values, which track
/// numerical error margins through every computation. This enables HELIX to
/// generate verifiable proofs that the training was performed correctly.
#[derive(Debug, Clone)]
pub struct DemoModel {
    config: DemoModelConfig,
    /// Weight tensors per layer, each shaped [in_dim, out_dim].
    weights: Vec<BoundedTensor>,
    /// Bias tensors per layer, each shaped [out_dim].
    biases: Vec<BoundedTensor>,
}

impl DemoModel {
    /// Creates a new demo model with Xavier-initialized BoundedTensor weights.
    ///
    /// All initial weights carry an error margin of 1e-7 to account for
    /// floating-point representation imprecision.
    pub fn new(config: DemoModelConfig) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut prev_dim = config.input_dim;
        for i in 0..config.num_layers {
            let next_dim = if i == config.num_layers - 1 {
                config.output_dim
            } else {
                config.hidden_dim
            };

            // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
            let scale = (2.0 / (prev_dim + next_dim) as f64).sqrt();
            let layer_weight_data: Vec<f64> = (0..prev_dim * next_dim)
                .map(|j| ((j * 7 + 13) % 100) as f64 / 100.0 * scale - scale / 2.0)
                .collect();

            let w = BoundedTensor::from_approximate(
                layer_weight_data,
                vec![prev_dim, next_dim],
                1e-7,
            );
            weights.push(w);

            let b = BoundedTensor::try_zeros(vec![next_dim])
                .expect("bias initialization should not fail");
            biases.push(b);

            prev_dim = next_dim;
        }

        Self { config, weights, biases }
    }

    /// Forward pass through the network using bounded arithmetic.
    ///
    /// Accepts either a 1D vector `[input_dim]` (single sample) or a 2D matrix
    /// `[batch, input_dim]`. Internally reshapes 1D to `[1, input_dim]` for
    /// matmul, then squeezes back to 1D at the end if the input was 1D.
    ///
    /// For each layer: z = input @ W + b, then activation.
    /// The final layer applies softmax for classification.
    pub fn forward(&self, input: &BoundedTensor) -> HelixResult<BoundedTensor> {
        let was_1d = input.is_vector();

        // Reshape 1D [n] -> 2D [1, n] for matmul compatibility
        let mut x = if was_1d {
            let n = input.shape()[0];
            BoundedTensor::new(input.data().to_vec(), vec![1, n])
        } else {
            input.clone()
        };

        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            // z = x @ W + b
            let z = x.matmul(w)?;
            let z = add_bias(&z, b)?;

            // Apply activation (except last layer which uses softmax)
            if layer_idx < self.config.num_layers - 1 {
                x = relu(&z);
            } else {
                x = softmax(&z)?;
            }
        }

        // Squeeze back to 1D if input was 1D
        if was_1d {
            let n = x.shape()[1];
            Ok(BoundedTensor::new(x.data().to_vec(), vec![n]))
        } else {
            Ok(x)
        }
    }

    /// Performs a training step: forward pass, loss computation, and simplified
    /// gradient update on the last layer.
    ///
    /// Returns the average loss as a `BoundedValue<f64>` with tracked error.
    pub fn train_step(
        &mut self,
        inputs: &[BoundedTensor],
        targets: &[usize],
        lr: f64,
    ) -> HelixResult<BoundedValue<f64>> {
        let mut total_loss = BoundedValue::exact(0.0);

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let output = self.forward(input)?;

            // Cross-entropy loss: -log(output[target])
            // Input is a 1D vector [output_dim] for a single sample
            let target_prob = output.data()[target].value().max(1e-10);
            let target_err = output.data()[target].absolute_error();
            let loss = BoundedValue::new(
                -target_prob.ln(),
                // Error propagation for log: |1/x| * epsilon_x
                ErrorMargin::absolute((1.0 / target_prob) * target_err),
            );
            total_loss = total_loss.saturating_add(loss);

            // Simplified gradient: softmax output - one-hot target
            let grad_data: Vec<BoundedValue<f64>> = output.data().iter().enumerate().map(|(i, v)| {
                if i == target {
                    BoundedValue::new(v.value() - 1.0, v.error())
                } else {
                    *v
                }
            }).collect();

            // Update last layer weights using simplified gradient descent
            if let Some(last_weights) = self.weights.last_mut() {
                let out_dim = self.config.output_dim;
                for i in 0..last_weights.len() {
                    let grad_val = grad_data[i % out_dim].value();
                    let current = last_weights.data()[i];
                    let update = BoundedValue::<f64>::with_absolute_error(
                        lr * grad_val * 0.01,
                        lr * grad_data[i % out_dim].absolute_error() * 0.01,
                    );
                    last_weights.data_mut()[i] = current.saturating_add(
                        BoundedValue::new(-update.value(), update.error()),
                    );
                }
            }
        }

        let n = BoundedValue::exact(1.0 / inputs.len() as f64);
        Ok(total_loss.saturating_mul(n))
    }

    /// Gets model parameters as a flat vector of f64 values.
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for w in &self.weights {
            params.extend(w.data().iter().map(|v| v.value()));
        }
        for b in &self.biases {
            params.extend(b.data().iter().map(|v| v.value()));
        }
        params
    }

    /// Gets total parameter count.
    pub fn num_parameters(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }

    /// Serializes model parameters to bytes (little-endian f64).
    pub fn serialize(&self) -> Vec<u8> {
        let params = self.get_parameters();
        params.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
}

// =============================================================================
// PROVABLE TRAINING STEP
// =============================================================================

/// A single demo training step that implements `Provable` for witness generation.
///
/// This captures the input, output, loss, and step number so that a ZK proof
/// can verify the computation was performed correctly.
pub struct DemoTrainingStep {
    /// The input tensor for this step.
    pub input: BoundedTensor,
    /// The output tensor (model predictions) for this step.
    pub output: BoundedTensor,
    /// The computed loss with tracked error bound.
    pub loss: BoundedValue<f64>,
    /// The training step number.
    pub step: usize,
}

impl Provable for DemoTrainingStep {
    type Witness = SimpleWitness;

    fn generate_witness(&self) -> Self::Witness {
        let mut witness = SimpleWitness::empty();

        // Add quantized input values to witness
        for v in self.input.data() {
            witness.push(quantize_to_u64(v.value()));
        }

        // Add quantized output values to witness
        for v in self.output.data() {
            witness.push(quantize_to_u64(v.value()));
        }

        witness
    }

    fn public_inputs(&self) -> Vec<u64> {
        // Public inputs format: [loss, error_bound, step_number]
        vec![
            quantize_to_u64(self.loss.value()),
            quantize_to_u64(self.loss.absolute_error()),
            self.step as u64,
        ]
    }

    fn circuit_id(&self) -> &'static str {
        "demo_training_step_v1"
    }
}

// =============================================================================
// TRAINER
// =============================================================================

/// Training loop for demo model with error-bounded computation.
pub struct DemoTrainer {
    model: DemoModel,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
}

impl DemoTrainer {
    /// Creates a new trainer.
    pub fn new(config: DemoModelConfig, lr: f64, epochs: usize, batch_size: usize) -> Self {
        Self {
            model: DemoModel::new(config),
            learning_rate: lr,
            epochs,
            batch_size,
        }
    }

    /// Trains on synthetic data using BoundedTensors throughout.
    ///
    /// Returns a `TrainingResult` that includes the final loss error bound
    /// and optionally generates a `DemoTrainingStep` witness at the end.
    pub fn train_synthetic(&mut self) -> TrainingResult {
        let mut history = Vec::new();
        let mut last_loss_error = 0.0;

        // Generate synthetic data as BoundedTensors
        let (inputs, targets) = self.generate_synthetic_data(1000);

        for epoch in 0..self.epochs {
            let mut epoch_loss = BoundedValue::exact(0.0);
            let num_batches = (inputs.len() + self.batch_size - 1) / self.batch_size;

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = (start + self.batch_size).min(inputs.len());

                let batch_inputs: Vec<BoundedTensor> = inputs[start..end].to_vec();
                let batch_targets: Vec<usize> = targets[start..end].to_vec();

                let loss = self.model.train_step(
                    &batch_inputs,
                    &batch_targets,
                    self.learning_rate,
                ).unwrap_or_else(|_| BoundedValue::exact(f64::NAN));

                epoch_loss = epoch_loss.saturating_add(loss);
            }

            let scale = BoundedValue::exact(1.0 / num_batches as f64);
            let avg_loss = epoch_loss.saturating_mul(scale);
            history.push(avg_loss.value());
            last_loss_error = avg_loss.absolute_error();

            // Compute accuracy
            let accuracy = self.compute_accuracy(&inputs, &targets);

            if epoch % 10 == 0 || epoch == self.epochs - 1 {
                println!(
                    "Epoch {}: loss={:.4} +/- {:.2e}, accuracy={:.2}%",
                    epoch,
                    avg_loss.value(),
                    avg_loss.absolute_error(),
                    accuracy * 100.0,
                );
            }
        }

        let final_accuracy = self.compute_accuracy(&inputs, &targets);

        // Generate a witness from the last training step for demonstration
        if let Some(first_input) = inputs.first() {
            if let Ok(output) = self.model.forward(first_input) {
                let demo_step = DemoTrainingStep {
                    input: first_input.clone(),
                    output: output.clone(),
                    loss: BoundedValue::<f64>::with_absolute_error(
                        *history.last().unwrap_or(&1.0),
                        last_loss_error,
                    ),
                    step: self.epochs,
                };
                let witness = demo_step.generate_witness();
                let elements = witness.to_field_elements();
                let public = demo_step.public_inputs();
                println!(
                    "Witness generated: {} elements, public_inputs={:?}, circuit={}",
                    elements.len(),
                    public,
                    demo_step.circuit_id(),
                );
            }
        }

        TrainingResult {
            final_loss: *history.last().unwrap_or(&1.0),
            final_loss_error: last_loss_error,
            final_accuracy,
            epochs_completed: self.epochs,
            loss_history: history,
        }
    }

    fn generate_synthetic_data(&self, n: usize) -> (Vec<BoundedTensor>, Vec<usize>) {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for i in 0..n {
            // Generate deterministic input data
            let input_data: Vec<f64> = (0..self.model.config.input_dim)
                .map(|j| ((i * 31 + j * 17) % 100) as f64 / 100.0)
                .collect();

            let input = BoundedTensor::from_approximate(
                input_data.clone(),
                vec![self.model.config.input_dim],
                1e-7,
            );

            // Target based on sum of features
            let target = (input_data.iter().sum::<f64>() * 10.0) as usize
                % self.model.config.output_dim;

            inputs.push(input);
            targets.push(target);
        }

        (inputs, targets)
    }

    fn compute_accuracy(&self, inputs: &[BoundedTensor], targets: &[usize]) -> f64 {
        let mut correct = 0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            if let Ok(output) = self.model.forward(input) {
                let predicted = output
                    .data()
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.value().partial_cmp(&b.value()).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if predicted == target {
                    correct += 1;
                }
            }
        }

        correct as f64 / inputs.len() as f64
    }

    /// Gets a reference to the trained model.
    pub fn get_model(&self) -> &DemoModel {
        &self.model
    }
}

// =============================================================================
// TRAINING RESULT
// =============================================================================

/// Training result including error bound information.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final loss value.
    pub final_loss: f64,
    /// Error bound on the final loss.
    pub final_loss_error: f64,
    /// Final accuracy on training data.
    pub final_accuracy: f64,
    /// Number of epochs completed.
    pub epochs_completed: usize,
    /// Loss history (one value per epoch).
    pub loss_history: Vec<f64>,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Add bias vector to each row of a 2D tensor.
///
/// Given x of shape [rows, cols] and bias of shape [cols], produces
/// output[i][j] = x[i][j] + bias[j].
fn add_bias(x: &BoundedTensor, bias: &BoundedTensor) -> HelixResult<BoundedTensor> {
    if x.is_matrix() {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        assert_eq!(bias.shape()[0], cols, "bias dimension must match columns");

        let mut data = Vec::with_capacity(x.len());
        for i in 0..rows {
            for j in 0..cols {
                let x_val = *x.get(&[i, j]).unwrap();
                let b_val = *bias.get(&[j]).unwrap();
                data.push(x_val.saturating_add(b_val));
            }
        }

        BoundedTensor::try_new(data, vec![rows, cols])
    } else if x.is_vector() {
        // Single sample (1D): element-wise add
        let len = x.shape()[0];
        assert_eq!(bias.shape()[0], len, "bias dimension must match vector length");

        let mut data = Vec::with_capacity(len);
        for j in 0..len {
            let x_val = *x.get(&[j]).unwrap();
            let b_val = *bias.get(&[j]).unwrap();
            data.push(x_val.saturating_add(b_val));
        }

        BoundedTensor::try_new(data, vec![len])
    } else {
        panic!("add_bias: x must be 1D or 2D tensor");
    }
}

/// ReLU activation: max(0, x) applied element-wise with error preservation.
fn relu(x: &BoundedTensor) -> BoundedTensor {
    x.map(|v| {
        let val = v.value().max(0.0);
        BoundedValue::new(val, v.error())
    })
}

/// Softmax over last axis of a 2D or 1D tensor with error tracking.
///
/// Uses the max-subtraction trick for numerical stability, and propagates
/// error through the exp and division operations.
fn softmax(x: &BoundedTensor) -> HelixResult<BoundedTensor> {
    if x.is_matrix() {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let mut data = Vec::with_capacity(x.len());

        for i in 0..rows {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..cols {
                let val = x.get(&[i, j]).unwrap().value();
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp(x - max)
            let mut exp_values: Vec<(f64, f64)> = Vec::with_capacity(cols);
            let mut sum_exp = 0.0;
            for j in 0..cols {
                let bounded = x.get(&[i, j]).unwrap();
                let shifted = bounded.value() - max_val;
                let exp_val = shifted.exp();
                let exp_err = exp_val * bounded.absolute_error();
                exp_values.push((exp_val, exp_err));
                sum_exp += exp_val;
            }

            // Normalize
            for (exp_val, exp_err) in exp_values {
                let prob = exp_val / sum_exp;
                let prob_err = (exp_err / sum_exp).min(1.0);
                data.push(BoundedValue::new(prob, ErrorMargin::absolute(prob_err)));
            }
        }

        BoundedTensor::try_new(data, vec![rows, cols])
    } else if x.is_vector() {
        let len = x.shape()[0];
        let mut data = Vec::with_capacity(len);

        // Find max for numerical stability
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..len {
            let val = x.get(&[j]).unwrap().value();
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max)
        let mut exp_values: Vec<(f64, f64)> = Vec::with_capacity(len);
        let mut sum_exp = 0.0;
        for j in 0..len {
            let bounded = x.get(&[j]).unwrap();
            let shifted = bounded.value() - max_val;
            let exp_val = shifted.exp();
            let exp_err = exp_val * bounded.absolute_error();
            exp_values.push((exp_val, exp_err));
            sum_exp += exp_val;
        }

        // Normalize
        for (exp_val, exp_err) in exp_values {
            let prob = exp_val / sum_exp;
            let prob_err = (exp_err / sum_exp).min(1.0);
            data.push(BoundedValue::new(prob, ErrorMargin::absolute(prob_err)));
        }

        BoundedTensor::try_new(data, vec![len])
    } else {
        panic!("softmax: tensor must be 1D or 2D");
    }
}

/// Quantize a f64 value to u64 using fixed-point representation.
///
/// Uses SCALE=1e12 for 12 decimal places of precision, matching
/// TensorWitness::SCALE used in the ZK circuit interface.
fn quantize_to_u64(value: f64) -> u64 {
    let scaled = (value * QUANTIZE_SCALE).clamp(0.0, u64::MAX as f64);
    scaled as u64
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_model_creation() {
        let config = DemoModelConfig::default();
        let model = DemoModel::new(config);

        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let config = DemoModelConfig {
            input_dim: 16,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let model = DemoModel::new(config);

        let input = BoundedTensor::from_approximate(vec![0.5; 16], vec![16], 1e-7);
        let output = model.forward(&input).unwrap();

        assert_eq!(output.len(), 4);
        // Softmax output should sum to ~1.0
        let sum: f64 = output.data().iter().map(|v| v.value()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {}", sum);
        // All values should have tracked error
        assert!(output.is_finite());
    }

    #[test]
    fn test_forward_pass_error_tracking() {
        let config = DemoModelConfig {
            input_dim: 16,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let model = DemoModel::new(config);

        let input = BoundedTensor::from_approximate(vec![0.5; 16], vec![16], 1e-7);
        let output = model.forward(&input).unwrap();

        // Error should be non-zero (propagated through matmul, add, relu, softmax)
        let max_err = output.max_error();
        assert!(max_err > 0.0, "error should propagate through forward pass");
        assert!(max_err < 1.0, "error should remain reasonable, got {}", max_err);
    }

    #[test]
    fn test_training() {
        let config = DemoModelConfig {
            input_dim: 16,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };

        let mut trainer = DemoTrainer::new(config, 0.01, 10, 32);
        let result = trainer.train_synthetic();

        assert_eq!(result.epochs_completed, 10);
        assert!(result.final_accuracy > 0.0);
        assert!(result.final_loss.is_finite());
        assert!(result.final_loss_error >= 0.0, "loss error should be non-negative");
    }

    #[test]
    fn test_provable_witness_generation() {
        let config = DemoModelConfig {
            input_dim: 16,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let model = DemoModel::new(config);

        let input = BoundedTensor::from_approximate(vec![0.5; 16], vec![16], 1e-7);
        let output = model.forward(&input).unwrap();
        let loss = BoundedValue::<f64>::with_absolute_error(0.5, 1e-4);

        let step = DemoTrainingStep {
            input: input.clone(),
            output: output.clone(),
            loss,
            step: 42,
        };

        // Test witness generation
        let witness = step.generate_witness();
        let elements = witness.to_field_elements();
        // Should contain input (16) + output (4) = 20 quantized values
        assert_eq!(elements.len(), 20);

        // Test public inputs
        let public = step.public_inputs();
        assert_eq!(public.len(), 3);
        assert_eq!(public[2], 42); // step number

        // Test circuit ID
        assert_eq!(step.circuit_id(), "demo_training_step_v1");
    }

    #[test]
    fn test_quantize_to_u64() {
        assert_eq!(quantize_to_u64(1.0), 1_000_000_000_000);
        assert_eq!(quantize_to_u64(0.0), 0);
        assert_eq!(quantize_to_u64(0.5), 500_000_000_000);
        // Negative values clamp to 0
        assert_eq!(quantize_to_u64(-1.0), 0);
    }

    #[test]
    fn test_get_parameters_and_serialize() {
        let config = DemoModelConfig {
            input_dim: 16,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let model = DemoModel::new(config);

        let params = model.get_parameters();
        assert_eq!(params.len(), model.num_parameters());

        let bytes = model.serialize();
        assert_eq!(bytes.len(), model.num_parameters() * 8); // f64 = 8 bytes
    }
}
