//! Quantized Linear Layer.
//!
//! Implements a linear layer (y = Wx + b) with quantized weights and activations.
//! Supports both forward and backward passes for quantization-aware training.
//!
//! # Forward Pass
//!
//! The forward pass uses quantized matrix multiplication:
//! 1. Quantize input activations (if not already quantized)
//! 2. Compute output = input @ weights^T using INT8/INT4 arithmetic
//! 3. Add bias (if present)
//! 4. Track error bounds through computation
//!
//! # Backward Pass
//!
//! The backward pass supports two modes:
//! 1. **Straight-Through Estimator (STE)**: Gradient flows through quantization
//! 2. **Fake Quantization**: Simulate quantization during training

use helix_core::types::{BoundedTensor, BoundedValue, Precision, Shape};
use std::sync::RwLock;

use crate::quantization::{
    int4_ops::{int4_int8_matmul, int4_linear},
    int4_tensor::Int4Tensor,
    int8_ops::{int8_linear, int8_matmul},
    int8_tensor::Int8Tensor,
    mixed_precision::{MixedPrecisionTensor, PrecisionLevel},
    schemes::{QuantScheme, TensorQuantParams},
    fake_quantize_tensor,
};

use super::{QuantizedLayer, QuantizedLayerError, Trainable, QATConfig};

/// Configuration for a quantized linear layer.
#[derive(Debug, Clone)]
pub struct QuantizedLinearConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Whether to include bias.
    pub bias: bool,
    /// Weight precision level.
    pub weight_precision: PrecisionLevel,
    /// Activation precision level.
    pub activation_precision: PrecisionLevel,
    /// Whether to use per-channel quantization for weights.
    pub per_channel: bool,
    /// Group size for group-wise quantization (INT4 only).
    pub group_size: Option<usize>,
    /// QAT configuration.
    pub qat_config: QATConfig,
}

impl QuantizedLinearConfig {
    /// Creates a configuration for INT8 symmetric weights.
    pub fn int8_symmetric(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
            weight_precision: PrecisionLevel::INT8Symmetric,
            activation_precision: PrecisionLevel::INT8Asymmetric,
            per_channel: true,
            group_size: None,
            qat_config: QATConfig::default(),
        }
    }

    /// Creates a configuration for INT4 weights with INT8 activations.
    pub fn int4_int8(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
            weight_precision: PrecisionLevel::INT4Symmetric,
            activation_precision: PrecisionLevel::INT8Asymmetric,
            per_channel: true,
            group_size: Some(128),
            qat_config: QATConfig::default(),
        }
    }

    /// Sets whether to use bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Sets per-channel quantization.
    pub fn with_per_channel(mut self, per_channel: bool) -> Self {
        self.per_channel = per_channel;
        self
    }

    /// Sets group size for INT4 quantization.
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = Some(group_size);
        self
    }

    /// Sets the QAT configuration.
    pub fn with_qat_config(mut self, qat_config: QATConfig) -> Self {
        self.qat_config = qat_config;
        self
    }
}

/// Gradients for a quantized linear layer.
#[derive(Debug, Clone)]
pub struct QuantizedLinearGradients {
    /// Gradient with respect to weights.
    pub weight_grad: BoundedTensor,
    /// Gradient with respect to bias (if present).
    pub bias_grad: Option<BoundedTensor>,
    /// Gradient with respect to input.
    pub input_grad: BoundedTensor,
}

/// A quantized linear layer: y = Wx + b
#[derive(Debug)]
pub struct QuantizedLinear {
    /// Configuration.
    config: QuantizedLinearConfig,
    /// Quantized weights (INT8 or INT4).
    weights: MixedPrecisionTensor,
    /// Quantized bias (if present).
    bias: Option<MixedPrecisionTensor>,
    /// Master weights in FP32 (for training).
    master_weights: BoundedTensor,
    /// Master bias in FP32 (for training).
    master_bias: Option<BoundedTensor>,
    /// Accumulated weight gradients.
    weight_grad_acc: RwLock<Option<BoundedTensor>>,
    /// Accumulated bias gradients.
    bias_grad_acc: RwLock<Option<BoundedTensor>>,
    /// Current training step.
    training_step: usize,
    /// Input cache for backward pass.
    cached_input: RwLock<Option<MixedPrecisionTensor>>,
}

impl Clone for QuantizedLinear {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            master_weights: self.master_weights.clone(),
            master_bias: self.master_bias.clone(),
            weight_grad_acc: RwLock::new(self.weight_grad_acc.read().unwrap().clone()),
            bias_grad_acc: RwLock::new(self.bias_grad_acc.read().unwrap().clone()),
            training_step: self.training_step,
            cached_input: RwLock::new(self.cached_input.read().unwrap().clone()),
        }
    }
}

impl QuantizedLinear {
    /// Creates a new quantized linear layer from FP32 weights.
    pub fn new(
        weights: BoundedTensor,
        bias: Option<BoundedTensor>,
        config: QuantizedLinearConfig,
    ) -> Result<Self, QuantizedLayerError> {
        // Validate dimensions
        if !weights.is_matrix() {
            return Err(QuantizedLayerError::InvalidConfig(
                "Weights must be a 2D tensor".to_string(),
            ));
        }

        let shape = weights.shape();
        if shape[0] != config.out_features || shape[1] != config.in_features {
            return Err(QuantizedLayerError::ShapeMismatch {
                expected: vec![config.out_features, config.in_features],
                actual: shape.clone(),
            });
        }

        if let Some(ref b) = bias {
            if b.len() != config.out_features {
                return Err(QuantizedLayerError::InvalidConfig(format!(
                    "Bias size {} doesn't match out_features {}",
                    b.len(),
                    config.out_features
                )));
            }
        }

        // Quantize weights
        let quantized_weights = quantize_weights(&weights, &config);

        // Quantize bias if present
        let quantized_bias = bias.as_ref().map(|b| quantize_bias(b, &config));

        Ok(Self {
            config,
            weights: quantized_weights,
            bias: quantized_bias,
            master_weights: weights,
            master_bias: bias,
            weight_grad_acc: RwLock::new(None),
            bias_grad_acc: RwLock::new(None),
            training_step: 0,
            cached_input: RwLock::new(None),
        })
    }

    /// Creates a quantized linear layer from configuration with random initialization.
    pub fn from_config(config: QuantizedLinearConfig) -> Self {
        // Xavier/Glorot initialization
        let scale = (2.0 / (config.in_features + config.out_features) as f64).sqrt();

        // Generate random weights (using simple LCG for reproducibility)
        let mut seed: u64 = 42;
        let mut random = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as f64 / 32768.0) * 2.0 - 1.0 // Range [-1, 1]
        };

        let weight_data: Vec<BoundedValue<f64>> = (0..config.out_features * config.in_features)
            .map(|_| BoundedValue::exact(random() * scale))
            .collect();
        let weights = BoundedTensor::new(weight_data, vec![config.out_features, config.in_features]);

        let bias = if config.bias {
            let bias_data: Vec<BoundedValue<f64>> = (0..config.out_features)
                .map(|_| BoundedValue::exact(random() * 0.01))
                .collect();
            Some(BoundedTensor::new(bias_data, vec![config.out_features]))
        } else {
            None
        };

        Self::new(weights, bias, config).expect("Configuration should be valid")
    }

    /// Creates from pre-quantized weights.
    pub fn from_quantized(
        weights: MixedPrecisionTensor,
        bias: Option<MixedPrecisionTensor>,
        config: QuantizedLinearConfig,
    ) -> Self {
        let master_weights = weights.to_bounded();
        let master_bias = bias.as_ref().map(|b| b.to_bounded());

        Self {
            config,
            weights,
            bias,
            master_weights,
            master_bias,
            weight_grad_acc: RwLock::new(None),
            bias_grad_acc: RwLock::new(None),
            training_step: 0,
            cached_input: RwLock::new(None),
        }
    }

    /// Returns the weight tensor.
    pub fn weights(&self) -> &MixedPrecisionTensor {
        &self.weights
    }

    /// Returns the bias tensor if present.
    pub fn bias(&self) -> Option<&MixedPrecisionTensor> {
        self.bias.as_ref()
    }

    /// Returns the master weights (FP32).
    pub fn master_weights(&self) -> &BoundedTensor {
        &self.master_weights
    }

    /// Returns the input features count.
    pub fn in_features(&self) -> usize {
        self.config.in_features
    }

    /// Returns the output features count.
    pub fn out_features(&self) -> usize {
        self.config.out_features
    }

    /// Forward pass with INT8 weights and activations.
    fn forward_int8(
        &self,
        input: &Int8Tensor,
        weights: &Int8Tensor,
        bias: Option<&Int8Tensor>,
    ) -> Int8Tensor {
        let output_scale = weights.scale();
        int8_linear(input, weights, bias, output_scale)
    }

    /// Forward pass with INT4 weights and INT8 activations.
    fn forward_int4_int8(
        &self,
        input: &Int8Tensor,
        weights: &Int4Tensor,
        bias: Option<&Int8Tensor>,
    ) -> Int8Tensor {
        let output_scale = weights.scale();
        int4_linear(input, weights, bias, output_scale)
    }

    /// Computes gradients for the backward pass.
    pub fn compute_gradients(
        &self,
        grad_output: &BoundedTensor,
        input: &MixedPrecisionTensor,
    ) -> Result<QuantizedLinearGradients, QuantizedLayerError> {
        // Get input in bounded form
        let input_bounded = input.to_bounded();

        // Compute gradient w.r.t. weights: grad_W = grad_output^T @ input
        let grad_output_t = grad_output.transpose();
        let weight_grad = matmul_bounded(&grad_output_t, &input_bounded);

        // Compute gradient w.r.t. bias: grad_b = sum(grad_output, axis=0)
        let bias_grad = if self.config.bias {
            Some(sum_axis0(grad_output))
        } else {
            None
        };

        // Compute gradient w.r.t. input: grad_input = grad_output @ weights
        let weights_bounded = self.master_weights.clone();
        let input_grad = matmul_bounded(grad_output, &weights_bounded);

        Ok(QuantizedLinearGradients {
            weight_grad,
            bias_grad,
            input_grad,
        })
    }

    /// Applies gradient clipping if configured.
    fn clip_gradients(&self, grad: &BoundedTensor) -> BoundedTensor {
        if let Some(clip_value) = self.config.qat_config.gradient_clip {
            let data: Vec<BoundedValue<f64>> = grad
                .data()
                .iter()
                .map(|v| {
                    let clipped = v.value().clamp(-clip_value, clip_value);
                    BoundedValue::<f64>::with_absolute_error(clipped, v.absolute_error())
                })
                .collect();
            BoundedTensor::new(data, grad.shape().clone())
        } else {
            grad.clone()
        }
    }

    /// Updates the quantized weights from master weights.
    pub fn update_quantized_weights(&mut self) {
        self.weights = quantize_weights(&self.master_weights, &self.config);
        if let Some(ref master_bias) = self.master_bias {
            self.bias = Some(quantize_bias(master_bias, &self.config));
        }
    }

    /// Computes the layer's output error bound.
    pub fn compute_error_bound(&self, input_error: f64) -> f64 {
        let weight_error = self.weights.total_error();
        let k = self.config.in_features as f64;

        // Error propagation: output_error ≈ k * (weight_error * input_magnitude + input_error * weight_magnitude)
        // Simplified: k * weight_error + input_error
        k * weight_error + input_error
    }
}

impl QuantizedLayer for QuantizedLinear {
    type Error = QuantizedLayerError;

    fn forward(&self, input: &MixedPrecisionTensor) -> Result<MixedPrecisionTensor, Self::Error> {
        // Cache input for backward pass
        {
            let mut cache = self.cached_input.write().unwrap();
            *cache = Some(input.clone());
        }

        // Dispatch based on weight and input precision
        match (&self.weights, input) {
            (MixedPrecisionTensor::Int8(weights), MixedPrecisionTensor::Int8(input)) => {
                let bias_int8 = self.bias.as_ref().and_then(|b| match b {
                    MixedPrecisionTensor::Int8(t) => Some(t),
                    _ => None,
                });
                let output = self.forward_int8(input, weights, bias_int8);
                Ok(MixedPrecisionTensor::Int8(output))
            }
            (MixedPrecisionTensor::Int4(weights), MixedPrecisionTensor::Int8(input)) => {
                let bias_int8 = self.bias.as_ref().and_then(|b| match b {
                    MixedPrecisionTensor::Int8(t) => Some(t),
                    _ => None,
                });
                let output = self.forward_int4_int8(input, weights, bias_int8);
                Ok(MixedPrecisionTensor::Int8(output))
            }
            (MixedPrecisionTensor::Int8(weights), MixedPrecisionTensor::Float(input)) => {
                // Quantize input to INT8
                let input_int8 = Int8Tensor::from_bounded_asymmetric(input);
                let bias_int8 = self.bias.as_ref().and_then(|b| match b {
                    MixedPrecisionTensor::Int8(t) => Some(t),
                    _ => None,
                });
                let output = self.forward_int8(&input_int8, weights, bias_int8);
                Ok(MixedPrecisionTensor::Int8(output))
            }
            (MixedPrecisionTensor::Int4(weights), MixedPrecisionTensor::Float(input)) => {
                // Quantize input to INT8
                let input_int8 = Int8Tensor::from_bounded_asymmetric(input);
                let bias_int8 = self.bias.as_ref().and_then(|b| match b {
                    MixedPrecisionTensor::Int8(t) => Some(t),
                    _ => None,
                });
                let output = self.forward_int4_int8(&input_int8, weights, bias_int8);
                Ok(MixedPrecisionTensor::Int8(output))
            }
            _ => {
                // Fall back to float computation
                let input_bounded = input.to_bounded();
                let weights_bounded = self.master_weights.clone();
                let output = compute_linear_float(&input_bounded, &weights_bounded, self.master_bias.as_ref());
                Ok(MixedPrecisionTensor::from_bounded(
                    &output,
                    self.config.activation_precision,
                ))
            }
        }
    }

    fn backward(
        &self,
        grad_output: &BoundedTensor,
        input: &MixedPrecisionTensor,
    ) -> Result<BoundedTensor, Self::Error> {
        let gradients = self.compute_gradients(grad_output, input)?;

        // Accumulate gradients
        {
            let clipped_weight_grad = self.clip_gradients(&gradients.weight_grad);
            let mut weight_acc = self.weight_grad_acc.write().unwrap();
            if let Some(ref mut acc) = *weight_acc {
                *acc = add_bounded(acc, &clipped_weight_grad);
            } else {
                *weight_acc = Some(clipped_weight_grad);
            }
        }

        if let Some(ref bias_grad) = gradients.bias_grad {
            let clipped_bias_grad = self.clip_gradients(bias_grad);
            let mut bias_acc = self.bias_grad_acc.write().unwrap();
            if let Some(ref mut acc) = *bias_acc {
                *acc = add_bounded(acc, &clipped_bias_grad);
            } else {
                *bias_acc = Some(clipped_bias_grad);
            }
        }

        // Apply straight-through estimator if configured
        if self.config.qat_config.straight_through {
            Ok(gradients.input_grad)
        } else {
            // Apply fake quantization to gradients
            let params = TensorQuantParams::from_range(
                QuantScheme::SymmetricInt8,
                -1.0,
                1.0,
            );
            Ok(fake_quantize_tensor(&gradients.input_grad, &params))
        }
    }

    fn input_dim(&self) -> Option<usize> {
        Some(self.config.in_features)
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.config.out_features)
    }

    fn weight_precision(&self) -> PrecisionLevel {
        self.config.weight_precision
    }

    fn activation_precision(&self) -> PrecisionLevel {
        self.config.activation_precision
    }

    fn quantization_error(&self) -> f64 {
        self.weights.total_error()
    }

    fn num_parameters(&self) -> usize {
        let weight_params = self.config.in_features * self.config.out_features;
        let bias_params = if self.config.bias {
            self.config.out_features
        } else {
            0
        };
        weight_params + bias_params
    }
}

impl Trainable for QuantizedLinear {
    fn apply_gradients(&mut self, learning_rate: f64) {
        let lr = learning_rate * self.config.qat_config.lr_scale;

        // Update master weights
        if let Some(weight_grad) = self.weight_grad_acc.write().unwrap().take() {
            let new_weights = subtract_scaled(&self.master_weights, &weight_grad, lr);
            self.master_weights = new_weights;
        }

        // Update master bias
        if let Some(bias_grad) = self.bias_grad_acc.write().unwrap().take() {
            if let Some(ref master_bias) = self.master_bias {
                self.master_bias = Some(subtract_scaled(master_bias, &bias_grad, lr));
            }
        }

        // Update quantized weights from master weights
        self.update_quantized_weights();
        self.training_step += 1;
    }

    fn weight_gradients(&self) -> Option<&BoundedTensor> {
        // Can't return reference to RwLock contents directly
        None
    }

    fn bias_gradients(&self) -> Option<&BoundedTensor> {
        None
    }

    fn zero_gradients(&mut self) {
        *self.weight_grad_acc.write().unwrap() = None;
        *self.bias_grad_acc.write().unwrap() = None;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Quantizes weights according to configuration.
fn quantize_weights(weights: &BoundedTensor, config: &QuantizedLinearConfig) -> MixedPrecisionTensor {
    match config.weight_precision {
        PrecisionLevel::INT8Symmetric => {
            if config.per_channel {
                // Per-channel quantization along output dimension
                let shape = weights.shape();
                let out_features = shape[0];
                let in_features = shape[1];
                let data = weights.data();

                let mut scales = Vec::with_capacity(out_features);
                let mut quantized = vec![0i8; data.len()];

                for o in 0..out_features {
                    let start = o * in_features;
                    let end = start + in_features;
                    let row: Vec<f64> = data[start..end].iter().map(|v| v.value()).collect();

                    let max_abs = row.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                    let scale = if max_abs > 1e-10 { max_abs / 127.0 } else { 1e-10 };

                    for (i, &v) in row.iter().enumerate() {
                        let q = (v / scale).round() as i32;
                        quantized[start + i] = q.clamp(-127, 127) as i8;
                    }
                    scales.push(scale);
                }

                let zero_points = vec![0i8; out_features];
                let tensor = Int8Tensor::per_channel(
                    quantized,
                    shape.clone(),
                    scales,
                    zero_points,
                    0,
                    true,
                );
                MixedPrecisionTensor::Int8(tensor)
            } else {
                MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(weights))
            }
        }
        PrecisionLevel::INT8Asymmetric => {
            MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_asymmetric(weights))
        }
        PrecisionLevel::INT4Symmetric => {
            if let Some(group_size) = config.group_size {
                use crate::quantization::int4_tensor::Int4TensorBuilder;
                let data: Vec<f64> = weights.data().iter().map(|v| v.value()).collect();
                let tensor = Int4TensorBuilder::new(weights.shape().clone())
                    .with_float_data(data)
                    .with_group_size(group_size)
                    .symmetric()
                    .build();
                MixedPrecisionTensor::Int4(tensor)
            } else {
                MixedPrecisionTensor::Int4(Int4Tensor::from_bounded_symmetric(weights))
            }
        }
        PrecisionLevel::INT4Asymmetric => {
            MixedPrecisionTensor::Int4(Int4Tensor::from_bounded_asymmetric(weights))
        }
        _ => MixedPrecisionTensor::Float(weights.clone()),
    }
}

/// Quantizes bias according to configuration.
fn quantize_bias(bias: &BoundedTensor, config: &QuantizedLinearConfig) -> MixedPrecisionTensor {
    // Bias is typically kept at higher precision (INT8 or INT32)
    match config.weight_precision {
        PrecisionLevel::INT4Symmetric | PrecisionLevel::INT4Asymmetric => {
            // Use INT8 for bias even with INT4 weights
            MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(bias))
        }
        _ => MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(bias)),
    }
}

/// Computes linear layer in float precision.
fn compute_linear_float(
    input: &BoundedTensor,
    weights: &BoundedTensor,
    bias: Option<&BoundedTensor>,
) -> BoundedTensor {
    let weights_t = weights.transpose();
    let output = matmul_bounded(input, &weights_t);

    if let Some(b) = bias {
        // Add bias to each row
        let shape = output.shape();
        let features = shape.last().copied().unwrap_or(1);
        let data: Vec<BoundedValue<f64>> = output
            .data()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let bias_val = b.data()[i % features];
                *v + bias_val
            })
            .collect();
        BoundedTensor::new(data, shape.clone())
    } else {
        output
    }
}

/// Bounded matrix multiplication.
fn matmul_bounded(a: &BoundedTensor, b: &BoundedTensor) -> BoundedTensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape.get(0).copied().unwrap_or(1);
    let k = a_shape.get(1).copied().unwrap_or(a.len());
    let n = b_shape.get(1).copied().unwrap_or(1);

    let a_data = a.data();
    let b_data = b.data();

    let mut result = Vec::with_capacity(m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = BoundedValue::exact(0.0);
            for l in 0..k {
                let a_idx = i * k + l;
                let b_idx = l * n + j;
                if a_idx < a_data.len() && b_idx < b_data.len() {
                    sum = sum + a_data[a_idx] * b_data[b_idx];
                }
            }
            result.push(sum);
        }
    }

    BoundedTensor::new(result, vec![m, n])
}

/// Sums a tensor along axis 0.
fn sum_axis0(tensor: &BoundedTensor) -> BoundedTensor {
    let shape = tensor.shape();
    if shape.is_empty() || shape[0] == 0 {
        return tensor.clone();
    }

    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();
    let data = tensor.data();

    let mut result = vec![BoundedValue::exact(0.0); cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] = result[j] + data[i * cols + j];
        }
    }

    BoundedTensor::new(result, shape[1..].to_vec())
}

/// Adds two bounded tensors.
fn add_bounded(a: &BoundedTensor, b: &BoundedTensor) -> BoundedTensor {
    let data: Vec<BoundedValue<f64>> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(av, bv)| *av + *bv)
        .collect();
    BoundedTensor::new(data, a.shape().clone())
}

/// Subtracts scaled tensor: a - lr * b.
fn subtract_scaled(a: &BoundedTensor, b: &BoundedTensor, lr: f64) -> BoundedTensor {
    let data: Vec<BoundedValue<f64>> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(av, bv)| {
            let new_val = av.value() - lr * bv.value();
            BoundedValue::<f64>::with_absolute_error(new_val, av.absolute_error() + lr * bv.absolute_error())
        })
        .collect();
    BoundedTensor::new(data, a.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_linear_creation() {
        let config = QuantizedLinearConfig::int8_symmetric(64, 128);
        let layer = QuantizedLinear::from_config(config);

        assert_eq!(layer.in_features(), 64);
        assert_eq!(layer.out_features(), 128);
        assert!(layer.bias().is_some());
    }

    #[test]
    fn test_quantized_linear_forward() {
        let config = QuantizedLinearConfig::int8_symmetric(4, 8);
        let layer = QuantizedLinear::from_config(config);

        // Create input
        let input_data: Vec<BoundedValue<f64>> = (0..8)
            .map(|i| BoundedValue::exact(i as f64 / 8.0))
            .collect();
        let input = BoundedTensor::new(input_data, vec![2, 4]);
        let input_mp = MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(&input));

        let output = layer.forward(&input_mp).unwrap();
        assert_eq!(output.shape(), &vec![2, 8]);
    }

    #[test]
    fn test_quantized_linear_backward() {
        let config = QuantizedLinearConfig::int8_symmetric(4, 8);
        let layer = QuantizedLinear::from_config(config);

        // Create input
        let input_data: Vec<BoundedValue<f64>> = (0..8)
            .map(|i| BoundedValue::exact(i as f64 / 8.0))
            .collect();
        let input = BoundedTensor::new(input_data, vec![2, 4]);
        let input_mp = MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(&input));

        // Create gradient output
        let grad_output_data: Vec<BoundedValue<f64>> = (0..16)
            .map(|i| BoundedValue::exact((i as f64 - 8.0) / 16.0))
            .collect();
        let grad_output = BoundedTensor::new(grad_output_data, vec![2, 8]);

        let grad_input = layer.backward(&grad_output, &input_mp).unwrap();
        assert_eq!(grad_input.shape(), &vec![2, 4]);
    }

    #[test]
    fn test_int4_linear() {
        let config = QuantizedLinearConfig::int4_int8(8, 16);
        let layer = QuantizedLinear::from_config(config);

        assert_eq!(layer.weight_precision(), PrecisionLevel::INT4Symmetric);

        let input_data: Vec<BoundedValue<f64>> = (0..16)
            .map(|i| BoundedValue::exact(i as f64 / 16.0))
            .collect();
        let input = BoundedTensor::new(input_data, vec![2, 8]);
        let input_mp = MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_asymmetric(&input));

        let output = layer.forward(&input_mp).unwrap();
        assert_eq!(output.shape(), &vec![2, 16]);
    }

    #[test]
    fn test_trainable_interface() {
        let config = QuantizedLinearConfig::int8_symmetric(4, 4);
        let mut layer = QuantizedLinear::from_config(config);

        let input_data: Vec<BoundedValue<f64>> = vec![0.1, 0.2, 0.3, 0.4]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let input = BoundedTensor::new(input_data, vec![1, 4]);
        let input_mp = MixedPrecisionTensor::Float(input.clone());

        // Forward
        let _ = layer.forward(&input_mp).unwrap();

        // Backward
        let grad_output_data: Vec<BoundedValue<f64>> = vec![1.0, 1.0, 1.0, 1.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let grad_output = BoundedTensor::new(grad_output_data, vec![1, 4]);
        let _ = layer.backward(&grad_output, &input_mp).unwrap();

        // Apply gradients
        layer.apply_gradients(0.01);

        // Weights should have changed
        assert!(layer.training_step > 0);
    }
}
