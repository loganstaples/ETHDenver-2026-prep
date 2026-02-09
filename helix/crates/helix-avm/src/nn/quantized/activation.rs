//! Quantized Activation Functions.
//!
//! Provides quantized implementations of common activation functions with
//! proper error bound tracking and backward pass support for training.
//!
//! # Supported Activations
//!
//! - **ReLU**: max(0, x) - Simple and efficient in quantized form
//! - **GELU**: x * Φ(x) - Uses sigmoid approximation for efficiency
//! - **SiLU/Swish**: x * σ(x) - Popular in modern architectures
//! - **Sigmoid**: 1 / (1 + exp(-x)) - Piecewise linear approximation
//! - **Tanh**: Uses polynomial approximation
//!
//! # Implementation Notes
//!
//! Quantized activations typically:
//! 1. Operate directly on INT8 values where possible
//! 2. Use lookup tables for complex functions
//! 3. Fall back to dequantize-compute-requantize for accuracy

use helix_core::types::{BoundedTensor, BoundedValue};

use crate::quantization::{
    int8_ops::{int8_gelu, int8_leaky_relu, int8_relu, int8_sigmoid, int8_silu, int8_tanh},
    int8_tensor::Int8Tensor,
    int4_ops::{int4_gelu, int4_relu},
    mixed_precision::{MixedPrecisionTensor, PrecisionLevel},
};

use super::{QuantizedLayer, QuantizedLayerError};

/// Type of quantized activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedActivationType {
    /// ReLU: max(0, x)
    ReLU,
    /// Leaky ReLU: max(αx, x) where α is small
    LeakyReLU(OrderedF64),
    /// GELU: x * Φ(x)
    GELU,
    /// SiLU/Swish: x * σ(x)
    SiLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// Softmax (along specified axis)
    Softmax(usize),
    /// No activation (identity)
    Identity,
}

/// Wrapper for f64 that implements Eq for use in enum.
#[derive(Debug, Clone, Copy)]
pub struct OrderedF64(pub f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedF64 {}

impl QuantizedActivationType {
    /// Creates a LeakyReLU with the given negative slope.
    pub fn leaky_relu(negative_slope: f64) -> Self {
        Self::LeakyReLU(OrderedF64(negative_slope))
    }

    /// Creates a Softmax along the given axis.
    pub fn softmax(axis: usize) -> Self {
        Self::Softmax(axis)
    }

    /// Returns whether this activation has a simple gradient.
    pub fn has_simple_gradient(&self) -> bool {
        matches!(self, Self::ReLU | Self::LeakyReLU(_) | Self::Identity)
    }

    /// Returns the approximation error for this activation.
    pub fn approximation_error(&self) -> f64 {
        match self {
            Self::ReLU | Self::Identity => 0.0,
            Self::LeakyReLU(_) => 0.0,
            Self::GELU => 0.01, // Sigmoid approximation error
            Self::SiLU => 0.005,
            Self::Sigmoid => 0.02, // Piecewise linear error
            Self::Tanh => 0.01,
            Self::Softmax(_) => 0.01,
        }
    }
}

/// Configuration for quantized activation layers.
#[derive(Debug, Clone)]
pub struct QuantizedActivationConfig {
    /// Type of activation.
    pub activation_type: QuantizedActivationType,
    /// Input/output precision.
    pub precision: PrecisionLevel,
    /// Whether to use lookup tables for complex activations.
    pub use_lookup_table: bool,
    /// Size of lookup table (if used).
    pub lookup_table_size: usize,
}

impl Default for QuantizedActivationConfig {
    fn default() -> Self {
        Self {
            activation_type: QuantizedActivationType::ReLU,
            precision: PrecisionLevel::INT8Asymmetric,
            use_lookup_table: false,
            lookup_table_size: 256,
        }
    }
}

impl QuantizedActivationConfig {
    /// Creates a ReLU configuration.
    pub fn relu() -> Self {
        Self {
            activation_type: QuantizedActivationType::ReLU,
            ..Default::default()
        }
    }

    /// Creates a GELU configuration.
    pub fn gelu() -> Self {
        Self {
            activation_type: QuantizedActivationType::GELU,
            ..Default::default()
        }
    }

    /// Creates a SiLU configuration.
    pub fn silu() -> Self {
        Self {
            activation_type: QuantizedActivationType::SiLU,
            ..Default::default()
        }
    }

    /// Creates a Sigmoid configuration.
    pub fn sigmoid() -> Self {
        Self {
            activation_type: QuantizedActivationType::Sigmoid,
            ..Default::default()
        }
    }

    /// Enables lookup table optimization.
    pub fn with_lookup_table(mut self, size: usize) -> Self {
        self.use_lookup_table = true;
        self.lookup_table_size = size;
        self
    }
}

/// Generic quantized activation layer.
#[derive(Debug, Clone)]
pub struct QuantizedActivation {
    /// Configuration.
    config: QuantizedActivationConfig,
    /// Lookup table for the activation (if enabled).
    lookup_table: Option<Vec<i8>>,
    /// Output scale for activations that change range.
    output_scale: f64,
}

impl QuantizedActivation {
    /// Creates a new quantized activation layer.
    pub fn new(config: QuantizedActivationConfig) -> Self {
        let lookup_table = if config.use_lookup_table {
            Some(build_lookup_table(&config))
        } else {
            None
        };

        Self {
            config,
            lookup_table,
            output_scale: 1.0 / 127.0, // Default scale
        }
    }

    /// Creates a ReLU activation.
    pub fn relu() -> Self {
        Self::new(QuantizedActivationConfig::relu())
    }

    /// Creates a GELU activation.
    pub fn gelu() -> Self {
        Self::new(QuantizedActivationConfig::gelu())
    }

    /// Creates a SiLU activation.
    pub fn silu() -> Self {
        Self::new(QuantizedActivationConfig::silu())
    }

    /// Creates a Sigmoid activation.
    pub fn sigmoid() -> Self {
        Self::new(QuantizedActivationConfig::sigmoid())
    }

    /// Sets the output scale.
    pub fn with_output_scale(mut self, scale: f64) -> Self {
        self.output_scale = scale;
        self
    }

    /// Applies activation using lookup table.
    fn apply_lookup_table(&self, input: &Int8Tensor) -> Int8Tensor {
        let table = self.lookup_table.as_ref().unwrap();
        let data = input.data();

        let output: Vec<i8> = data
            .iter()
            .map(|&v| {
                // Map signed byte to table index
                let idx = (v as i32 + 128).clamp(0, 255) as usize;
                table[idx]
            })
            .collect();

        Int8Tensor::new(
            output,
            input.shape().clone(),
            self.output_scale,
            0,
            true,
        )
    }

    /// Computes the backward gradient for the activation.
    fn compute_gradient(
        &self,
        grad_output: &BoundedTensor,
        input: &BoundedTensor,
    ) -> BoundedTensor {
        let grad_data = grad_output.data();
        let input_data = input.data();

        let gradient: Vec<BoundedValue<f64>> = grad_data
            .iter()
            .zip(input_data.iter())
            .map(|(g, x)| {
                let grad_value = match self.config.activation_type {
                    QuantizedActivationType::ReLU => {
                        if x.value() > 0.0 {
                            g.value()
                        } else {
                            0.0
                        }
                    }
                    QuantizedActivationType::LeakyReLU(OrderedF64(alpha)) => {
                        if x.value() > 0.0 {
                            g.value()
                        } else {
                            g.value() * alpha
                        }
                    }
                    QuantizedActivationType::GELU => {
                        // GELU gradient: Φ(x) + x * φ(x)
                        let x_val = x.value();
                        let sigmoid = 1.0 / (1.0 + (-1.702 * x_val).exp());
                        let dsigmoid = 1.702 * sigmoid * (1.0 - sigmoid);
                        let grad = sigmoid + x_val * dsigmoid;
                        g.value() * grad
                    }
                    QuantizedActivationType::SiLU => {
                        // SiLU gradient: σ(x) + x * σ(x) * (1 - σ(x))
                        let x_val = x.value();
                        let sigmoid = 1.0 / (1.0 + (-x_val).exp());
                        let grad = sigmoid * (1.0 + x_val * (1.0 - sigmoid));
                        g.value() * grad
                    }
                    QuantizedActivationType::Sigmoid => {
                        // Sigmoid gradient: σ(x) * (1 - σ(x))
                        let sigmoid = 1.0 / (1.0 + (-x.value()).exp());
                        g.value() * sigmoid * (1.0 - sigmoid)
                    }
                    QuantizedActivationType::Tanh => {
                        // Tanh gradient: 1 - tanh²(x)
                        let tanh = x.value().tanh();
                        g.value() * (1.0 - tanh * tanh)
                    }
                    QuantizedActivationType::Identity => g.value(),
                    QuantizedActivationType::Softmax(_) => {
                        // Softmax gradient is more complex - use Jacobian
                        // For now, return simplified gradient
                        g.value()
                    }
                };

                BoundedValue::<f64>::with_absolute_error(grad_value, g.absolute_error() + x.absolute_error())
            })
            .collect();

        BoundedTensor::new(gradient, grad_output.shape().clone())
    }
}

impl QuantizedLayer for QuantizedActivation {
    type Error = QuantizedLayerError;

    fn forward(&self, input: &MixedPrecisionTensor) -> Result<MixedPrecisionTensor, Self::Error> {
        match input {
            MixedPrecisionTensor::Int8(tensor) => {
                // Use lookup table if available
                if self.lookup_table.is_some() {
                    let output = self.apply_lookup_table(tensor);
                    return Ok(MixedPrecisionTensor::Int8(output));
                }

                // Otherwise use direct computation
                let output_scale = tensor.scale();
                let output = match self.config.activation_type {
                    QuantizedActivationType::ReLU => int8_relu(tensor),
                    QuantizedActivationType::LeakyReLU(OrderedF64(alpha)) => {
                        int8_leaky_relu(tensor, alpha, output_scale)
                    }
                    QuantizedActivationType::GELU => int8_gelu(tensor, output_scale),
                    QuantizedActivationType::SiLU => int8_silu(tensor, output_scale),
                    QuantizedActivationType::Sigmoid => int8_sigmoid(tensor, output_scale),
                    QuantizedActivationType::Tanh => int8_tanh(tensor, output_scale),
                    QuantizedActivationType::Softmax(axis) => {
                        use crate::quantization::int8_ops::int8_softmax;
                        int8_softmax(tensor, axis, 1.0 / 127.0) // Softmax output scale
                    }
                    QuantizedActivationType::Identity => tensor.clone(),
                };
                Ok(MixedPrecisionTensor::Int8(output))
            }
            MixedPrecisionTensor::Int4(tensor) => {
                let output_scale = tensor.scale();
                let output = match self.config.activation_type {
                    QuantizedActivationType::ReLU => int4_relu(tensor),
                    QuantizedActivationType::GELU => int4_gelu(tensor, output_scale),
                    _ => {
                        // Fall back to dequantize-compute-requantize
                        let bounded = tensor.dequantize();
                        let activated = apply_float_activation(&bounded, self.config.activation_type);
                        return Ok(MixedPrecisionTensor::from_bounded(
                            &activated,
                            PrecisionLevel::INT4Symmetric,
                        ));
                    }
                };
                Ok(MixedPrecisionTensor::Int4(output))
            }
            MixedPrecisionTensor::Float(tensor) => {
                let activated = apply_float_activation(tensor, self.config.activation_type);
                Ok(MixedPrecisionTensor::Float(activated))
            }
        }
    }

    fn backward(
        &self,
        grad_output: &BoundedTensor,
        input: &MixedPrecisionTensor,
    ) -> Result<BoundedTensor, Self::Error> {
        let input_bounded = input.to_bounded();
        Ok(self.compute_gradient(grad_output, &input_bounded))
    }

    fn weight_precision(&self) -> PrecisionLevel {
        // Activations don't have weights
        PrecisionLevel::FP32
    }

    fn activation_precision(&self) -> PrecisionLevel {
        self.config.precision
    }

    fn quantization_error(&self) -> f64 {
        self.config.activation_type.approximation_error()
    }

    fn num_parameters(&self) -> usize {
        // Activations typically don't have learnable parameters
        0
    }
}

/// Specialized ReLU layer for convenience.
#[derive(Debug, Clone)]
pub struct QuantizedReLU {
    inner: QuantizedActivation,
}

impl QuantizedReLU {
    /// Creates a new quantized ReLU.
    pub fn new() -> Self {
        Self {
            inner: QuantizedActivation::relu(),
        }
    }
}

impl Default for QuantizedReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedLayer for QuantizedReLU {
    type Error = QuantizedLayerError;

    fn forward(&self, input: &MixedPrecisionTensor) -> Result<MixedPrecisionTensor, Self::Error> {
        self.inner.forward(input)
    }

    fn backward(
        &self,
        grad_output: &BoundedTensor,
        input: &MixedPrecisionTensor,
    ) -> Result<BoundedTensor, Self::Error> {
        self.inner.backward(grad_output, input)
    }

    fn weight_precision(&self) -> PrecisionLevel {
        self.inner.weight_precision()
    }

    fn activation_precision(&self) -> PrecisionLevel {
        self.inner.activation_precision()
    }

    fn quantization_error(&self) -> f64 {
        0.0 // ReLU has no approximation error
    }

    fn num_parameters(&self) -> usize {
        0
    }
}

/// Specialized GELU layer for convenience.
#[derive(Debug, Clone)]
pub struct QuantizedGELU {
    inner: QuantizedActivation,
}

impl QuantizedGELU {
    /// Creates a new quantized GELU.
    pub fn new() -> Self {
        Self {
            inner: QuantizedActivation::gelu(),
        }
    }

    /// Creates with lookup table optimization.
    pub fn with_lookup_table() -> Self {
        Self {
            inner: QuantizedActivation::new(
                QuantizedActivationConfig::gelu().with_lookup_table(256),
            ),
        }
    }
}

impl Default for QuantizedGELU {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedLayer for QuantizedGELU {
    type Error = QuantizedLayerError;

    fn forward(&self, input: &MixedPrecisionTensor) -> Result<MixedPrecisionTensor, Self::Error> {
        self.inner.forward(input)
    }

    fn backward(
        &self,
        grad_output: &BoundedTensor,
        input: &MixedPrecisionTensor,
    ) -> Result<BoundedTensor, Self::Error> {
        self.inner.backward(grad_output, input)
    }

    fn weight_precision(&self) -> PrecisionLevel {
        self.inner.weight_precision()
    }

    fn activation_precision(&self) -> PrecisionLevel {
        self.inner.activation_precision()
    }

    fn quantization_error(&self) -> f64 {
        0.01 // GELU approximation error
    }

    fn num_parameters(&self) -> usize {
        0
    }
}

/// Specialized SiLU layer for convenience.
#[derive(Debug, Clone)]
pub struct QuantizedSiLU {
    inner: QuantizedActivation,
}

impl QuantizedSiLU {
    /// Creates a new quantized SiLU.
    pub fn new() -> Self {
        Self {
            inner: QuantizedActivation::silu(),
        }
    }
}

impl Default for QuantizedSiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedLayer for QuantizedSiLU {
    type Error = QuantizedLayerError;

    fn forward(&self, input: &MixedPrecisionTensor) -> Result<MixedPrecisionTensor, Self::Error> {
        self.inner.forward(input)
    }

    fn backward(
        &self,
        grad_output: &BoundedTensor,
        input: &MixedPrecisionTensor,
    ) -> Result<BoundedTensor, Self::Error> {
        self.inner.backward(grad_output, input)
    }

    fn weight_precision(&self) -> PrecisionLevel {
        self.inner.weight_precision()
    }

    fn activation_precision(&self) -> PrecisionLevel {
        self.inner.activation_precision()
    }

    fn quantization_error(&self) -> f64 {
        0.005
    }

    fn num_parameters(&self) -> usize {
        0
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Builds a lookup table for the activation function.
fn build_lookup_table(config: &QuantizedActivationConfig) -> Vec<i8> {
    let size = config.lookup_table_size;
    let mut table = Vec::with_capacity(size);

    // Input scale: maps [-128, 127] to some range
    let input_scale = 4.0 / 127.0; // Assume input range [-4, 4]

    for i in 0..size {
        let quantized_input = i as i32 - 128; // Convert to signed
        let x = quantized_input as f64 * input_scale;

        let y = match config.activation_type {
            QuantizedActivationType::ReLU => x.max(0.0),
            QuantizedActivationType::LeakyReLU(OrderedF64(alpha)) => {
                if x > 0.0 { x } else { alpha * x }
            }
            QuantizedActivationType::GELU => {
                let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
                x * sigmoid
            }
            QuantizedActivationType::SiLU => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }
            QuantizedActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            QuantizedActivationType::Tanh => x.tanh(),
            QuantizedActivationType::Identity => x,
            QuantizedActivationType::Softmax(_) => x, // Softmax can't use lookup table directly
        };

        // Quantize output
        let output_scale = match config.activation_type {
            QuantizedActivationType::Sigmoid | QuantizedActivationType::Tanh => 1.0 / 127.0,
            _ => input_scale, // Keep same scale for most activations
        };

        let quantized_output = (y / output_scale).round() as i32;
        table.push(quantized_output.clamp(-127, 127) as i8);
    }

    table
}

/// Applies activation in float precision.
fn apply_float_activation(
    tensor: &BoundedTensor,
    activation_type: QuantizedActivationType,
) -> BoundedTensor {
    let data: Vec<BoundedValue<f64>> = tensor
        .data()
        .iter()
        .map(|v| {
            let x = v.value();
            let y = match activation_type {
                QuantizedActivationType::ReLU => x.max(0.0),
                QuantizedActivationType::LeakyReLU(OrderedF64(alpha)) => {
                    if x > 0.0 { x } else { alpha * x }
                }
                QuantizedActivationType::GELU => {
                    let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
                    x * sigmoid
                }
                QuantizedActivationType::SiLU => {
                    let sigmoid = 1.0 / (1.0 + (-x).exp());
                    x * sigmoid
                }
                QuantizedActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                QuantizedActivationType::Tanh => x.tanh(),
                QuantizedActivationType::Identity => x,
                QuantizedActivationType::Softmax(_) => x, // Handle separately
            };
            BoundedValue::<f64>::with_absolute_error(y, v.absolute_error())
        })
        .collect();

    BoundedTensor::new(data, tensor.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_relu() {
        let relu = QuantizedReLU::new();

        let input_data: Vec<BoundedValue<f64>> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let input = BoundedTensor::new(input_data, vec![5]);
        let input_mp = MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(&input));

        let output = relu.forward(&input_mp).unwrap();
        let output_bounded = output.to_bounded();
        let output_values = output_bounded.data();

        // Negative values should be close to zero
        assert!(output_values[0].value() >= 0.0 || output_values[0].value().abs() < 0.1);
        assert!(output_values[1].value() >= 0.0 || output_values[1].value().abs() < 0.1);
        // Positive values should be preserved
        assert!(output_values[3].value() > 0.5);
        assert!(output_values[4].value() > 1.0);
    }

    #[test]
    fn test_quantized_gelu() {
        let gelu = QuantizedGELU::new();

        let input_data: Vec<BoundedValue<f64>> = vec![-1.0, 0.0, 1.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let input = BoundedTensor::new(input_data, vec![3]);
        let input_mp = MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(&input));

        let output = gelu.forward(&input_mp).unwrap();
        let output_bounded = output.to_bounded();
        let values = output_bounded.data();

        // GELU(-1) ≈ -0.16
        assert!(values[0].value() < 0.0);
        // GELU(0) = 0
        assert!(values[1].value().abs() < 0.1);
        // GELU(1) ≈ 0.84
        assert!(values[2].value() > 0.0);
    }

    #[test]
    fn test_gelu_backward() {
        let gelu = QuantizedGELU::new();

        let input_data: Vec<BoundedValue<f64>> = vec![-1.0, 0.0, 1.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let input = BoundedTensor::new(input_data, vec![3]);
        let input_mp = MixedPrecisionTensor::Float(input.clone());

        let grad_output_data: Vec<BoundedValue<f64>> = vec![1.0, 1.0, 1.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let grad_output = BoundedTensor::new(grad_output_data, vec![3]);

        let grad_input = gelu.backward(&grad_output, &input_mp).unwrap();
        assert_eq!(grad_input.len(), 3);
    }

    #[test]
    fn test_lookup_table() {
        let gelu_with_lut = QuantizedGELU::with_lookup_table();

        let input_data: Vec<BoundedValue<f64>> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let input = BoundedTensor::new(input_data, vec![5]);
        let input_mp = MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(&input));

        let output = gelu_with_lut.forward(&input_mp).unwrap();
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_activation_config() {
        let config = QuantizedActivationConfig::gelu().with_lookup_table(256);
        assert_eq!(config.activation_type, QuantizedActivationType::GELU);
        assert!(config.use_lookup_table);
        assert_eq!(config.lookup_table_size, 256);
    }

    #[test]
    fn test_silu() {
        let silu = QuantizedSiLU::new();

        let input_data: Vec<BoundedValue<f64>> = vec![-1.0, 0.0, 1.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let input = BoundedTensor::new(input_data, vec![3]);
        let input_mp = MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(&input));

        let output = silu.forward(&input_mp).unwrap();
        let values = output.to_bounded().data().iter().map(|v| v.value()).collect::<Vec<_>>();

        // SiLU(-1) ≈ -0.27
        assert!(values[0] < 0.0);
        // SiLU(0) = 0
        assert!(values[1].abs() < 0.1);
        // SiLU(1) ≈ 0.73
        assert!(values[2] > 0.0);
    }
}
