//! Quantized Neural Network Layers.
//!
//! This module provides quantized versions of neural network layers that operate
//! directly on INT8 and INT4 representations. These layers support both forward
//! and backward passes for quantization-aware training (QAT).
//!
//! # Architecture
//!
//! The quantized layers follow HELIX's approximate computing model:
//! - All operations track error bounds through computation
//! - Quantized forward passes use INT8/INT4 arithmetic
//! - Backward passes support both full-precision and fake-quantized gradients
//! - Mixed-precision configurations for optimal accuracy/efficiency tradeoffs
//!
//! # Usage
//!
//! ```ignore
//! use helix_avm::nn::quantized::{QuantizedLinear, QuantizedLinearConfig};
//!
//! // Create a quantized linear layer
//! let config = QuantizedLinearConfig::int8_symmetric(768, 3072);
//! let layer = QuantizedLinear::from_config(config);
//!
//! // Forward pass with quantized activations
//! let output = layer.forward(&input)?;
//!
//! // Backward pass for training
//! let grad_input = layer.backward(&grad_output, &input)?;
//! ```

pub mod activation;
pub mod linear;

// Re-export main types
pub use activation::{
    QuantizedActivation, QuantizedActivationConfig, QuantizedActivationType,
    QuantizedGELU, QuantizedReLU, QuantizedSiLU,
};
pub use linear::{
    QuantizedLinear, QuantizedLinearConfig, QuantizedLinearGradients,
};

use helix_core::types::{BoundedTensor, BoundedValue, Shape};
use crate::quantization::{
    int8_tensor::Int8Tensor,
    int4_tensor::Int4Tensor,
    mixed_precision::{MixedPrecisionTensor, PrecisionLevel},
    schemes::{QuantScheme, TensorQuantParams},
};

/// Error type for quantized layer operations.
#[derive(Debug, thiserror::Error)]
pub enum QuantizedLayerError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Shape, actual: Shape },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Precision mismatch: expected {expected:?}, got {actual:?}")]
    PrecisionMismatch {
        expected: PrecisionLevel,
        actual: PrecisionLevel,
    },
}

/// Trait for quantized neural network layers.
pub trait QuantizedLayer: std::fmt::Debug + Clone + Send + Sync {
    /// Error type for this layer.
    type Error: std::error::Error;

    /// Performs the forward pass with quantized computation.
    fn forward(&self, input: &MixedPrecisionTensor) -> Result<MixedPrecisionTensor, Self::Error>;

    /// Performs the backward pass for training.
    /// Returns gradients with respect to input.
    fn backward(
        &self,
        grad_output: &BoundedTensor,
        input: &MixedPrecisionTensor,
    ) -> Result<BoundedTensor, Self::Error>;

    /// Returns the input dimension if applicable.
    fn input_dim(&self) -> Option<usize> {
        None
    }

    /// Returns the output dimension if applicable.
    fn output_dim(&self) -> Option<usize> {
        None
    }

    /// Returns the precision level for weights.
    fn weight_precision(&self) -> PrecisionLevel;

    /// Returns the precision level for activations.
    fn activation_precision(&self) -> PrecisionLevel;

    /// Returns the quantization error bound for this layer.
    fn quantization_error(&self) -> f64;

    /// Returns the number of parameters in this layer.
    fn num_parameters(&self) -> usize;
}

/// Trait for layers with trainable parameters.
pub trait Trainable: QuantizedLayer {
    /// Updates parameters with gradients.
    fn apply_gradients(&mut self, learning_rate: f64);

    /// Returns the weight gradients.
    fn weight_gradients(&self) -> Option<&BoundedTensor>;

    /// Returns the bias gradients.
    fn bias_gradients(&self) -> Option<&BoundedTensor>;

    /// Clears accumulated gradients.
    fn zero_gradients(&mut self);
}

/// Configuration for quantization-aware training.
#[derive(Debug, Clone)]
pub struct QATConfig {
    /// Whether to use straight-through estimator for gradients.
    pub straight_through: bool,
    /// Learning rate scale for quantized parameters.
    pub lr_scale: f64,
    /// Whether to freeze quantization parameters after warmup.
    pub freeze_quant_params: bool,
    /// Number of warmup steps before freezing.
    pub warmup_steps: usize,
    /// Whether to use fake quantization during training.
    pub fake_quantize: bool,
    /// Gradient clipping threshold.
    pub gradient_clip: Option<f64>,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            straight_through: true,
            lr_scale: 1.0,
            freeze_quant_params: false,
            warmup_steps: 1000,
            fake_quantize: true,
            gradient_clip: Some(1.0),
        }
    }
}

/// State for tracking training progress.
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current training step.
    pub step: usize,
    /// Accumulated loss.
    pub accumulated_loss: f64,
    /// Gradient norm history.
    pub gradient_norms: Vec<f64>,
    /// Quantization error history.
    pub quant_errors: Vec<f64>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            step: 0,
            accumulated_loss: 0.0,
            gradient_norms: Vec::new(),
            quant_errors: Vec::new(),
        }
    }
}

impl TrainingState {
    /// Advances to the next step.
    pub fn step(&mut self) {
        self.step += 1;
    }

    /// Records gradient norm.
    pub fn record_gradient_norm(&mut self, norm: f64) {
        self.gradient_norms.push(norm);
        if self.gradient_norms.len() > 1000 {
            self.gradient_norms.remove(0);
        }
    }

    /// Records quantization error.
    pub fn record_quant_error(&mut self, error: f64) {
        self.quant_errors.push(error);
        if self.quant_errors.len() > 1000 {
            self.quant_errors.remove(0);
        }
    }

    /// Returns the average gradient norm.
    pub fn avg_gradient_norm(&self) -> f64 {
        if self.gradient_norms.is_empty() {
            0.0
        } else {
            self.gradient_norms.iter().sum::<f64>() / self.gradient_norms.len() as f64
        }
    }

    /// Returns the average quantization error.
    pub fn avg_quant_error(&self) -> f64 {
        if self.quant_errors.is_empty() {
            0.0
        } else {
            self.quant_errors.iter().sum::<f64>() / self.quant_errors.len() as f64
        }
    }
}

/// A sequential container for quantized layers.
#[derive(Debug, Clone)]
pub struct QuantizedSequential<L: QuantizedLayer> {
    layers: Vec<L>,
}

impl<L: QuantizedLayer> QuantizedSequential<L> {
    /// Creates an empty sequential container.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Adds a layer to the sequence.
    pub fn add(mut self, layer: L) -> Self {
        self.layers.push(layer);
        self
    }

    /// Returns the number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns a reference to a layer by index.
    pub fn get(&self, index: usize) -> Option<&L> {
        self.layers.get(index)
    }

    /// Returns a mutable reference to a layer by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut L> {
        self.layers.get_mut(index)
    }

    /// Performs forward pass through all layers.
    pub fn forward(&self, input: &MixedPrecisionTensor) -> Result<MixedPrecisionTensor, L::Error> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// Returns the total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }

    /// Returns the total quantization error.
    pub fn total_quantization_error(&self) -> f64 {
        self.layers.iter().map(|l| l.quantization_error()).sum()
    }
}

impl<L: QuantizedLayer> Default for QuantizedSequential<L> {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility for computing gradient statistics.
#[derive(Debug, Clone, Default)]
pub struct GradientStats {
    /// Mean gradient value.
    pub mean: f64,
    /// Standard deviation of gradients.
    pub std: f64,
    /// Maximum absolute gradient.
    pub max_abs: f64,
    /// L2 norm of gradients.
    pub l2_norm: f64,
    /// Fraction of zero gradients.
    pub sparsity: f64,
}

impl GradientStats {
    /// Computes statistics from a gradient tensor.
    pub fn from_tensor(tensor: &BoundedTensor) -> Self {
        let data = tensor.data();
        if data.is_empty() {
            return Self::default();
        }

        let n = data.len() as f64;
        let sum: f64 = data.iter().map(|v| v.value()).sum();
        let mean = sum / n;

        let var_sum: f64 = data.iter().map(|v| (v.value() - mean).powi(2)).sum();
        let std = (var_sum / n).sqrt();

        let max_abs = data.iter().map(|v| v.value().abs()).fold(0.0_f64, f64::max);
        let l2_norm = data.iter().map(|v| v.value().powi(2)).sum::<f64>().sqrt();
        let zeros = data.iter().filter(|v| v.value().abs() < 1e-10).count();
        let sparsity = zeros as f64 / n;

        Self {
            mean,
            std,
            max_abs,
            l2_norm,
            sparsity,
        }
    }
}

/// Computes the accuracy loss from quantization.
pub fn compute_accuracy_loss(
    fp32_output: &BoundedTensor,
    quantized_output: &MixedPrecisionTensor,
) -> AccuracyMetrics {
    let quant_bounded = quantized_output.to_bounded();

    let fp32_data = fp32_output.data();
    let quant_data = quant_bounded.data();

    if fp32_data.len() != quant_data.len() {
        return AccuracyMetrics::default();
    }

    let n = fp32_data.len() as f64;
    let mut sum_sq_error = 0.0;
    let mut sum_abs_error = 0.0;
    let mut max_error = 0.0_f64;
    let mut sum_relative_error = 0.0;
    let mut relative_count = 0;

    for (fp, q) in fp32_data.iter().zip(quant_data.iter()) {
        let error = (fp.value() - q.value()).abs();
        sum_sq_error += error * error;
        sum_abs_error += error;
        max_error = max_error.max(error);

        if fp.value().abs() > 1e-10 {
            sum_relative_error += error / fp.value().abs();
            relative_count += 1;
        }
    }

    let signal_power: f64 = fp32_data.iter().map(|v| v.value().powi(2)).sum();
    let noise_power = sum_sq_error;
    let snr = if noise_power > 1e-10 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    };

    AccuracyMetrics {
        mse: sum_sq_error / n,
        rmse: (sum_sq_error / n).sqrt(),
        mae: sum_abs_error / n,
        max_error,
        mean_relative_error: if relative_count > 0 {
            sum_relative_error / relative_count as f64
        } else {
            0.0
        },
        snr_db: snr,
    }
}

/// Accuracy metrics for quantized vs FP32 comparison.
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetrics {
    /// Mean squared error.
    pub mse: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Maximum absolute error.
    pub max_error: f64,
    /// Mean relative error.
    pub mean_relative_error: f64,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
}

impl AccuracyMetrics {
    /// Returns true if accuracy loss is within acceptable bounds (<1%).
    pub fn is_acceptable(&self) -> bool {
        self.mean_relative_error < 0.01 && self.snr_db > 30.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_stats() {
        let data: Vec<BoundedValue<f64>> = vec![1.0, -1.0, 2.0, -2.0, 0.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let tensor = BoundedTensor::new(data, vec![5]);

        let stats = GradientStats::from_tensor(&tensor);
        assert_eq!(stats.mean, 0.0);
        assert!(stats.max_abs > 1.9);
        assert!(stats.sparsity < 0.5);
    }

    #[test]
    fn test_accuracy_metrics() {
        let fp32_data: Vec<BoundedValue<f64>> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let fp32 = BoundedTensor::new(fp32_data, vec![4]);

        // Quantized with small error
        let quant_data: Vec<BoundedValue<f64>> = vec![1.01, 2.01, 3.01, 4.01]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let quant = BoundedTensor::new(quant_data, vec![4]);
        let quant_mp = MixedPrecisionTensor::Float(quant);

        let metrics = compute_accuracy_loss(&fp32, &quant_mp);
        assert!(metrics.rmse < 0.02);
        assert!(metrics.is_acceptable());
    }
}
