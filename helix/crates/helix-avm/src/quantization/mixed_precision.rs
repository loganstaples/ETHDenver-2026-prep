//! Mixed-Precision Execution Support.
//!
//! Provides infrastructure for running neural networks with different precision
//! levels for different operations. This is crucial for:
//!
//! 1. **Optimal Performance**: Use lower precision where possible without sacrificing accuracy
//! 2. **Memory Efficiency**: Store weights in INT4/INT8, compute in INT8/INT16
//! 3. **Training Support**: FP32 master weights with quantized forward/backward passes
//! 4. **Error Control**: Track error bounds across precision transitions
//!
//! Common mixed-precision configurations:
//! - INT4 weights + INT8 activations (W4A8): Best for LLM inference
//! - INT8 weights + INT8 activations (W8A8): Good balance of accuracy and efficiency
//! - FP32 accumulator with INT8 operands: Standard for training

use helix_core::types::{BoundedTensor, BoundedValue, Shape};
use std::collections::HashMap;

use super::int4_tensor::Int4Tensor;
use super::int8_tensor::Int8Tensor;
use super::schemes::QuantScheme;

/// Precision level for a computation or storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionLevel {
    /// Full 32-bit floating point.
    FP32,
    /// 16-bit floating point (half precision).
    FP16,
    /// Brain floating point (8-bit exponent, 7-bit mantissa).
    BF16,
    /// 8-bit floating point (E4M3 format).
    FP8E4M3,
    /// 8-bit floating point (E5M2 format).
    FP8E5M2,
    /// 8-bit integer (symmetric).
    INT8Symmetric,
    /// 8-bit integer (asymmetric).
    INT8Asymmetric,
    /// 4-bit integer (symmetric).
    INT4Symmetric,
    /// 4-bit integer (asymmetric).
    INT4Asymmetric,
}

impl PrecisionLevel {
    /// Returns the number of bits.
    pub fn bits(&self) -> usize {
        match self {
            PrecisionLevel::FP32 => 32,
            PrecisionLevel::FP16 | PrecisionLevel::BF16 => 16,
            PrecisionLevel::FP8E4M3 | PrecisionLevel::FP8E5M2 => 8,
            PrecisionLevel::INT8Symmetric | PrecisionLevel::INT8Asymmetric => 8,
            PrecisionLevel::INT4Symmetric | PrecisionLevel::INT4Asymmetric => 4,
        }
    }

    /// Returns the memory size per element in bytes.
    pub fn bytes_per_element(&self) -> f64 {
        self.bits() as f64 / 8.0
    }

    /// Returns the typical quantization error for this precision.
    pub fn typical_error(&self, dynamic_range: f64) -> f64 {
        match self {
            PrecisionLevel::FP32 => dynamic_range * 1e-7,
            PrecisionLevel::FP16 | PrecisionLevel::BF16 => dynamic_range * 1e-3,
            PrecisionLevel::FP8E4M3 | PrecisionLevel::FP8E5M2 => dynamic_range * 0.01,
            PrecisionLevel::INT8Symmetric | PrecisionLevel::INT8Asymmetric => dynamic_range / 254.0,
            PrecisionLevel::INT4Symmetric | PrecisionLevel::INT4Asymmetric => dynamic_range / 14.0,
        }
    }

    /// Returns whether this is an integer format.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            PrecisionLevel::INT8Symmetric
                | PrecisionLevel::INT8Asymmetric
                | PrecisionLevel::INT4Symmetric
                | PrecisionLevel::INT4Asymmetric
        )
    }

    /// Returns whether this is symmetric quantization.
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            PrecisionLevel::INT8Symmetric | PrecisionLevel::INT4Symmetric
        )
    }

    /// Converts to QuantScheme if applicable.
    pub fn to_quant_scheme(&self) -> Option<QuantScheme> {
        match self {
            PrecisionLevel::INT8Symmetric => Some(QuantScheme::SymmetricInt8),
            PrecisionLevel::INT8Asymmetric => Some(QuantScheme::AsymmetricInt8),
            PrecisionLevel::INT4Symmetric => Some(QuantScheme::SymmetricInt4),
            PrecisionLevel::INT4Asymmetric => Some(QuantScheme::AsymmetricInt4),
            PrecisionLevel::FP8E4M3 => Some(QuantScheme::FP8E4M3),
            PrecisionLevel::FP8E5M2 => Some(QuantScheme::FP8E5M2),
            _ => None,
        }
    }
}

/// Configuration for mixed-precision execution.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision for weight storage.
    pub weight_precision: PrecisionLevel,
    /// Precision for activation storage.
    pub activation_precision: PrecisionLevel,
    /// Precision for accumulator during computation.
    pub accumulator_precision: PrecisionLevel,
    /// Precision for gradient computation.
    pub gradient_precision: PrecisionLevel,
    /// Precision for master weights (training only).
    pub master_weight_precision: PrecisionLevel,
    /// Per-layer precision overrides.
    pub layer_overrides: HashMap<String, LayerPrecisionConfig>,
    /// Whether to use dynamic quantization for activations.
    pub dynamic_activation_quant: bool,
    /// Group size for group-wise quantization (INT4 only).
    pub group_size: Option<usize>,
    /// Whether to enable gradient checkpointing.
    pub gradient_checkpointing: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            weight_precision: PrecisionLevel::INT8Symmetric,
            activation_precision: PrecisionLevel::INT8Asymmetric,
            accumulator_precision: PrecisionLevel::FP32,
            gradient_precision: PrecisionLevel::FP32,
            master_weight_precision: PrecisionLevel::FP32,
            layer_overrides: HashMap::new(),
            dynamic_activation_quant: true,
            group_size: None,
            gradient_checkpointing: false,
        }
    }
}

impl MixedPrecisionConfig {
    /// Creates a W8A8 (INT8 weights and activations) configuration.
    pub fn w8a8() -> Self {
        Self::default()
    }

    /// Creates a W4A8 (INT4 weights, INT8 activations) configuration.
    pub fn w4a8() -> Self {
        Self {
            weight_precision: PrecisionLevel::INT4Symmetric,
            activation_precision: PrecisionLevel::INT8Asymmetric,
            accumulator_precision: PrecisionLevel::FP32,
            group_size: Some(128), // Common for LLMs
            ..Default::default()
        }
    }

    /// Creates a W4A16 (INT4 weights, FP16 activations) configuration.
    pub fn w4a16() -> Self {
        Self {
            weight_precision: PrecisionLevel::INT4Symmetric,
            activation_precision: PrecisionLevel::FP16,
            accumulator_precision: PrecisionLevel::FP32,
            group_size: Some(128),
            ..Default::default()
        }
    }

    /// Creates a full FP32 configuration (for reference).
    pub fn fp32() -> Self {
        Self {
            weight_precision: PrecisionLevel::FP32,
            activation_precision: PrecisionLevel::FP32,
            accumulator_precision: PrecisionLevel::FP32,
            gradient_precision: PrecisionLevel::FP32,
            master_weight_precision: PrecisionLevel::FP32,
            dynamic_activation_quant: false,
            ..Default::default()
        }
    }

    /// Creates a FP16 mixed-precision training configuration.
    pub fn fp16_training() -> Self {
        Self {
            weight_precision: PrecisionLevel::FP16,
            activation_precision: PrecisionLevel::FP16,
            accumulator_precision: PrecisionLevel::FP32,
            gradient_precision: PrecisionLevel::FP16,
            master_weight_precision: PrecisionLevel::FP32,
            dynamic_activation_quant: false,
            ..Default::default()
        }
    }

    /// Creates a quantization-aware training configuration.
    pub fn qat_int8() -> Self {
        Self {
            weight_precision: PrecisionLevel::INT8Symmetric,
            activation_precision: PrecisionLevel::INT8Asymmetric,
            accumulator_precision: PrecisionLevel::FP32,
            gradient_precision: PrecisionLevel::FP32,
            master_weight_precision: PrecisionLevel::FP32,
            dynamic_activation_quant: true,
            ..Default::default()
        }
    }

    /// Adds a layer-specific precision override.
    pub fn with_layer_override(mut self, layer_name: &str, config: LayerPrecisionConfig) -> Self {
        self.layer_overrides.insert(layer_name.to_string(), config);
        self
    }

    /// Sets the group size for INT4 quantization.
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = Some(group_size);
        self
    }

    /// Enables gradient checkpointing.
    pub fn with_gradient_checkpointing(mut self) -> Self {
        self.gradient_checkpointing = true;
        self
    }

    /// Gets the effective precision for a layer.
    pub fn get_layer_precision(&self, layer_name: &str) -> LayerPrecisionConfig {
        self.layer_overrides
            .get(layer_name)
            .cloned()
            .unwrap_or_else(|| LayerPrecisionConfig {
                weight_precision: self.weight_precision,
                activation_precision: self.activation_precision,
                accumulator_precision: self.accumulator_precision,
            })
    }

    /// Computes memory footprint for a model.
    pub fn compute_memory_footprint(&self, weight_elements: usize, activation_elements: usize) -> MemoryFootprint {
        let weight_bytes = weight_elements as f64 * self.weight_precision.bytes_per_element();
        let activation_bytes = activation_elements as f64 * self.activation_precision.bytes_per_element();
        let master_bytes = weight_elements as f64 * self.master_weight_precision.bytes_per_element();

        let fp32_weight_bytes = weight_elements as f64 * 4.0;
        let fp32_activation_bytes = activation_elements as f64 * 4.0;

        MemoryFootprint {
            weight_bytes: weight_bytes as usize,
            activation_bytes: activation_bytes as usize,
            master_weight_bytes: master_bytes as usize,
            total_bytes: (weight_bytes + activation_bytes) as usize,
            compression_ratio_weights: fp32_weight_bytes / weight_bytes,
            compression_ratio_activations: fp32_activation_bytes / activation_bytes,
        }
    }
}

/// Per-layer precision configuration.
#[derive(Debug, Clone)]
pub struct LayerPrecisionConfig {
    /// Precision for weights in this layer.
    pub weight_precision: PrecisionLevel,
    /// Precision for activations in this layer.
    pub activation_precision: PrecisionLevel,
    /// Precision for accumulator in this layer.
    pub accumulator_precision: PrecisionLevel,
}

impl LayerPrecisionConfig {
    /// Creates a new layer precision config.
    pub fn new(
        weight_precision: PrecisionLevel,
        activation_precision: PrecisionLevel,
        accumulator_precision: PrecisionLevel,
    ) -> Self {
        Self {
            weight_precision,
            activation_precision,
            accumulator_precision,
        }
    }

    /// Creates a high-precision config for sensitive layers.
    pub fn high_precision() -> Self {
        Self {
            weight_precision: PrecisionLevel::FP32,
            activation_precision: PrecisionLevel::FP32,
            accumulator_precision: PrecisionLevel::FP32,
        }
    }

    /// Creates a low-precision config for robust layers.
    pub fn low_precision() -> Self {
        Self {
            weight_precision: PrecisionLevel::INT4Symmetric,
            activation_precision: PrecisionLevel::INT8Asymmetric,
            accumulator_precision: PrecisionLevel::INT8Symmetric,
        }
    }
}

/// Memory footprint analysis.
#[derive(Debug, Clone)]
pub struct MemoryFootprint {
    /// Weight storage in bytes.
    pub weight_bytes: usize,
    /// Activation storage in bytes.
    pub activation_bytes: usize,
    /// Master weight storage in bytes (training only).
    pub master_weight_bytes: usize,
    /// Total memory in bytes.
    pub total_bytes: usize,
    /// Compression ratio for weights vs FP32.
    pub compression_ratio_weights: f64,
    /// Compression ratio for activations vs FP32.
    pub compression_ratio_activations: f64,
}

impl MemoryFootprint {
    /// Returns total compression ratio vs FP32.
    pub fn total_compression_ratio(&self) -> f64 {
        (self.compression_ratio_weights + self.compression_ratio_activations) / 2.0
    }
}

/// Mixed-precision tensor that can hold data in various formats.
#[derive(Debug, Clone)]
pub enum MixedPrecisionTensor {
    /// Full precision float tensor.
    Float(BoundedTensor),
    /// INT8 quantized tensor.
    Int8(Int8Tensor),
    /// INT4 quantized tensor.
    Int4(Int4Tensor),
}

impl MixedPrecisionTensor {
    /// Creates from a BoundedTensor with the specified precision.
    pub fn from_bounded(tensor: &BoundedTensor, precision: PrecisionLevel) -> Self {
        match precision {
            PrecisionLevel::FP32 | PrecisionLevel::FP16 | PrecisionLevel::BF16 => {
                MixedPrecisionTensor::Float(tensor.clone())
            }
            PrecisionLevel::INT8Symmetric => {
                MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_symmetric(tensor))
            }
            PrecisionLevel::INT8Asymmetric => {
                MixedPrecisionTensor::Int8(Int8Tensor::from_bounded_asymmetric(tensor))
            }
            PrecisionLevel::INT4Symmetric => {
                MixedPrecisionTensor::Int4(Int4Tensor::from_bounded_symmetric(tensor))
            }
            PrecisionLevel::INT4Asymmetric => {
                MixedPrecisionTensor::Int4(Int4Tensor::from_bounded_asymmetric(tensor))
            }
            _ => MixedPrecisionTensor::Float(tensor.clone()),
        }
    }

    /// Converts to BoundedTensor (dequantizes if needed).
    pub fn to_bounded(&self) -> BoundedTensor {
        match self {
            MixedPrecisionTensor::Float(t) => t.clone(),
            MixedPrecisionTensor::Int8(t) => t.dequantize(),
            MixedPrecisionTensor::Int4(t) => t.dequantize(),
        }
    }

    /// Returns the shape.
    pub fn shape(&self) -> &Shape {
        match self {
            MixedPrecisionTensor::Float(t) => t.shape(),
            MixedPrecisionTensor::Int8(t) => t.shape(),
            MixedPrecisionTensor::Int4(t) => t.shape(),
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        match self {
            MixedPrecisionTensor::Float(t) => t.len(),
            MixedPrecisionTensor::Int8(t) => t.len(),
            MixedPrecisionTensor::Int4(t) => t.len(),
        }
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the precision level.
    pub fn precision(&self) -> PrecisionLevel {
        match self {
            MixedPrecisionTensor::Float(_) => PrecisionLevel::FP32,
            MixedPrecisionTensor::Int8(t) => {
                if t.is_symmetric() {
                    PrecisionLevel::INT8Symmetric
                } else {
                    PrecisionLevel::INT8Asymmetric
                }
            }
            MixedPrecisionTensor::Int4(t) => {
                if t.is_symmetric() {
                    PrecisionLevel::INT4Symmetric
                } else {
                    PrecisionLevel::INT4Asymmetric
                }
            }
        }
    }

    /// Returns memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            MixedPrecisionTensor::Float(t) => t.len() * 8, // f64
            MixedPrecisionTensor::Int8(t) => t.len(),
            MixedPrecisionTensor::Int4(t) => t.memory_bytes(),
        }
    }

    /// Converts to a different precision.
    pub fn to_precision(&self, precision: PrecisionLevel) -> Self {
        let bounded = self.to_bounded();
        MixedPrecisionTensor::from_bounded(&bounded, precision)
    }

    /// Returns the total error bound.
    pub fn total_error(&self) -> f64 {
        match self {
            MixedPrecisionTensor::Float(t) => t.max_error(),
            MixedPrecisionTensor::Int8(t) => t.total_error(),
            MixedPrecisionTensor::Int4(t) => t.total_error(),
        }
    }
}

/// Mixed-precision executor for neural network operations.
#[derive(Debug)]
pub struct MixedPrecisionExecutor {
    /// Configuration.
    config: MixedPrecisionConfig,
    /// Statistics about precision conversions.
    stats: ExecutionStats,
}

impl MixedPrecisionExecutor {
    /// Creates a new executor with the given configuration.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            config,
            stats: ExecutionStats::default(),
        }
    }

    /// Creates an executor with W8A8 configuration.
    pub fn w8a8() -> Self {
        Self::new(MixedPrecisionConfig::w8a8())
    }

    /// Creates an executor with W4A8 configuration.
    pub fn w4a8() -> Self {
        Self::new(MixedPrecisionConfig::w4a8())
    }

    /// Returns the configuration.
    pub fn config(&self) -> &MixedPrecisionConfig {
        &self.config
    }

    /// Returns execution statistics.
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Resets execution statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ExecutionStats::default();
    }

    /// Quantizes weights for a layer.
    pub fn quantize_weights(&mut self, weights: &BoundedTensor, layer_name: &str) -> MixedPrecisionTensor {
        let precision = self.config.get_layer_precision(layer_name).weight_precision;
        self.stats.weight_quantizations += 1;
        MixedPrecisionTensor::from_bounded(weights, precision)
    }

    /// Quantizes activations for a layer.
    pub fn quantize_activations(
        &mut self,
        activations: &BoundedTensor,
        layer_name: &str,
    ) -> MixedPrecisionTensor {
        let precision = self.config.get_layer_precision(layer_name).activation_precision;
        self.stats.activation_quantizations += 1;
        MixedPrecisionTensor::from_bounded(activations, precision)
    }

    /// Performs a linear layer operation with mixed precision.
    pub fn linear(
        &mut self,
        input: &MixedPrecisionTensor,
        weights: &MixedPrecisionTensor,
        bias: Option<&MixedPrecisionTensor>,
        layer_name: &str,
    ) -> MixedPrecisionTensor {
        self.stats.operations += 1;

        // Get layer config
        let layer_config = self.config.get_layer_precision(layer_name);

        // Dispatch based on precision combination
        match (input, weights) {
            (MixedPrecisionTensor::Int8(input), MixedPrecisionTensor::Int8(weights)) => {
                use super::int8_ops::int8_linear;
                let output_scale = weights.scale(); // Use weight scale for output
                let bias_int8 = bias.and_then(|b| match b {
                    MixedPrecisionTensor::Int8(t) => Some(t),
                    _ => None,
                });
                let result = int8_linear(input, weights, bias_int8, output_scale);
                MixedPrecisionTensor::Int8(result)
            }
            (MixedPrecisionTensor::Int8(input), MixedPrecisionTensor::Int4(weights)) => {
                use super::int4_ops::int4_linear;
                let output_scale = weights.scale();
                let bias_int8 = bias.and_then(|b| match b {
                    MixedPrecisionTensor::Int8(t) => Some(t),
                    _ => None,
                });
                let result = int4_linear(input, weights, bias_int8, output_scale);
                MixedPrecisionTensor::Int8(result)
            }
            _ => {
                // Fall back to float computation
                let input_bounded = input.to_bounded();
                let weights_bounded = weights.to_bounded();
                let bias_bounded = bias.map(|b| b.to_bounded());

                // Compute in float and convert back
                let output = compute_float_linear(&input_bounded, &weights_bounded, bias_bounded.as_ref());
                MixedPrecisionTensor::from_bounded(&output, layer_config.activation_precision)
            }
        }
    }

    /// Performs a matmul operation with mixed precision.
    pub fn matmul(
        &mut self,
        a: &MixedPrecisionTensor,
        b: &MixedPrecisionTensor,
        layer_name: &str,
    ) -> MixedPrecisionTensor {
        self.stats.operations += 1;

        match (a, b) {
            (MixedPrecisionTensor::Int8(a), MixedPrecisionTensor::Int8(b)) => {
                use super::int8_ops::int8_matmul;
                let output_scale = a.scale() * b.scale();
                let result = int8_matmul(a, b, output_scale);
                MixedPrecisionTensor::Int8(result)
            }
            (MixedPrecisionTensor::Int4(a), MixedPrecisionTensor::Int4(b)) => {
                use super::int4_ops::int4_matmul;
                let output_scale = a.scale() * b.scale();
                let result = int4_matmul(a, b, output_scale);
                MixedPrecisionTensor::Int4(result)
            }
            _ => {
                // Fall back to float
                let layer_config = self.config.get_layer_precision(layer_name);
                let a_bounded = a.to_bounded();
                let b_bounded = b.to_bounded();
                let output = compute_float_matmul(&a_bounded, &b_bounded);
                MixedPrecisionTensor::from_bounded(&output, layer_config.activation_precision)
            }
        }
    }

    /// Applies an activation function with mixed precision.
    pub fn activation(
        &mut self,
        input: &MixedPrecisionTensor,
        activation_type: ActivationType,
        _layer_name: &str,
    ) -> MixedPrecisionTensor {
        self.stats.operations += 1;

        match input {
            MixedPrecisionTensor::Int8(t) => {
                use super::int8_ops::{int8_gelu, int8_relu, int8_silu};
                let output_scale = t.scale();
                let result = match activation_type {
                    ActivationType::ReLU => int8_relu(t),
                    ActivationType::GELU => int8_gelu(t, output_scale),
                    ActivationType::SiLU => int8_silu(t, output_scale),
                    ActivationType::Sigmoid => {
                        use super::int8_ops::int8_sigmoid;
                        int8_sigmoid(t, output_scale)
                    }
                    ActivationType::Tanh => {
                        use super::int8_ops::int8_tanh;
                        int8_tanh(t, output_scale)
                    }
                };
                MixedPrecisionTensor::Int8(result)
            }
            MixedPrecisionTensor::Int4(t) => {
                use super::int4_ops::{int4_gelu, int4_relu};
                let output_scale = t.scale();
                let result = match activation_type {
                    ActivationType::ReLU => int4_relu(t),
                    ActivationType::GELU => int4_gelu(t, output_scale),
                    _ => {
                        // For other activations, convert through float
                        let bounded = t.dequantize();
                        let activated = apply_float_activation(&bounded, activation_type);
                        return MixedPrecisionTensor::from_bounded(
                            &activated,
                            PrecisionLevel::INT4Symmetric,
                        );
                    }
                };
                MixedPrecisionTensor::Int4(result)
            }
            MixedPrecisionTensor::Float(t) => {
                let activated = apply_float_activation(t, activation_type);
                MixedPrecisionTensor::Float(activated)
            }
        }
    }

    /// Applies layer normalization with mixed precision.
    pub fn layer_norm(
        &mut self,
        input: &MixedPrecisionTensor,
        gamma: &MixedPrecisionTensor,
        beta: &MixedPrecisionTensor,
        eps: f64,
        layer_name: &str,
    ) -> MixedPrecisionTensor {
        self.stats.operations += 1;

        match (input, gamma, beta) {
            (
                MixedPrecisionTensor::Int8(input),
                MixedPrecisionTensor::Int8(gamma),
                MixedPrecisionTensor::Int8(beta),
            ) => {
                use super::int8_ops::int8_layer_norm;
                let output_scale = input.scale();
                let result = int8_layer_norm(input, gamma, beta, eps, output_scale);
                MixedPrecisionTensor::Int8(result)
            }
            _ => {
                // Fall back to float
                let input_bounded = input.to_bounded();
                let gamma_bounded = gamma.to_bounded();
                let beta_bounded = beta.to_bounded();
                let output = compute_float_layer_norm(&input_bounded, &gamma_bounded, &beta_bounded, eps);
                let layer_config = self.config.get_layer_precision(layer_name);
                MixedPrecisionTensor::from_bounded(&output, layer_config.activation_precision)
            }
        }
    }

    /// Applies softmax with mixed precision.
    pub fn softmax(
        &mut self,
        input: &MixedPrecisionTensor,
        axis: usize,
        layer_name: &str,
    ) -> MixedPrecisionTensor {
        self.stats.operations += 1;

        match input {
            MixedPrecisionTensor::Int8(t) => {
                use super::int8_ops::int8_softmax;
                let output_scale = 1.0 / 255.0; // Softmax output is [0, 1]
                let result = int8_softmax(t, axis, output_scale);
                MixedPrecisionTensor::Int8(result)
            }
            _ => {
                // Softmax in float for better accuracy
                let bounded = input.to_bounded();
                let output = compute_float_softmax(&bounded, axis);
                let layer_config = self.config.get_layer_precision(layer_name);
                MixedPrecisionTensor::from_bounded(&output, layer_config.activation_precision)
            }
        }
    }
}

/// Activation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// ReLU: max(0, x)
    ReLU,
    /// GELU: x * Φ(x)
    GELU,
    /// SiLU/Swish: x * sigmoid(x)
    SiLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
}

/// Execution statistics.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Number of weight quantizations.
    pub weight_quantizations: usize,
    /// Number of activation quantizations.
    pub activation_quantizations: usize,
    /// Number of operations performed.
    pub operations: usize,
    /// Total error accumulated.
    pub total_error: f64,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Computes linear layer in float precision.
fn compute_float_linear(
    input: &BoundedTensor,
    weights: &BoundedTensor,
    bias: Option<&BoundedTensor>,
) -> BoundedTensor {
    // Simplified implementation - in production would use optimized ops
    let weights_t = weights.transpose();
    let output = compute_float_matmul(input, &weights_t);

    if let Some(b) = bias {
        // Add bias to each row
        let shape = output.shape();
        let features = shape.last().copied().unwrap_or(1);
        let mut data = output.data().to_vec();

        for (i, val) in data.iter_mut().enumerate() {
            let bias_val = b.data()[i % features];
            *val = *val + bias_val;
        }

        BoundedTensor::new(data, shape.clone())
    } else {
        output
    }
}

/// Computes matrix multiplication in float precision.
fn compute_float_matmul(a: &BoundedTensor, b: &BoundedTensor) -> BoundedTensor {
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
                let a_val = a_data[i * k + l];
                let b_val = b_data[l * n + j];
                sum = sum + a_val * b_val;
            }
            result.push(sum);
        }
    }

    BoundedTensor::new(result, vec![m, n])
}

/// Applies activation function in float precision.
fn apply_float_activation(tensor: &BoundedTensor, activation_type: ActivationType) -> BoundedTensor {
    let data: Vec<BoundedValue<f64>> = tensor
        .data()
        .iter()
        .map(|v| {
            let x = v.value();
            let y = match activation_type {
                ActivationType::ReLU => x.max(0.0),
                ActivationType::GELU => {
                    let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
                    x * sigmoid
                }
                ActivationType::SiLU => {
                    let sigmoid = 1.0 / (1.0 + (-x).exp());
                    x * sigmoid
                }
                ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                ActivationType::Tanh => x.tanh(),
            };
            BoundedValue::<f64>::with_absolute_error(y, v.absolute_error())
        })
        .collect();

    BoundedTensor::new(data, tensor.shape().clone())
}

/// Computes layer normalization in float precision.
fn compute_float_layer_norm(
    input: &BoundedTensor,
    gamma: &BoundedTensor,
    beta: &BoundedTensor,
    eps: f64,
) -> BoundedTensor {
    let shape = input.shape();
    let norm_size = shape.last().copied().unwrap_or(1);
    let num_groups = input.len() / norm_size;

    let data = input.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();

    let mut result = Vec::with_capacity(input.len());

    for g in 0..num_groups {
        let start = g * norm_size;
        let group_data = &data[start..start + norm_size];

        // Compute mean
        let sum: f64 = group_data.iter().map(|v| v.value()).sum();
        let mean = sum / norm_size as f64;

        // Compute variance
        let var_sum: f64 = group_data.iter().map(|v| (v.value() - mean).powi(2)).sum();
        let variance = var_sum / norm_size as f64;
        let std_inv = 1.0 / (variance + eps).sqrt();

        // Normalize
        for i in 0..norm_size {
            let x = group_data[i].value();
            let normalized = (x - mean) * std_inv;
            let g_val = gamma_data[i % norm_size].value();
            let b_val = beta_data[i % norm_size].value();
            let y = normalized * g_val + b_val;
            result.push(BoundedValue::<f64>::with_absolute_error(y, group_data[i].absolute_error()));
        }
    }

    BoundedTensor::new(result, shape.clone())
}

/// Computes softmax in float precision.
fn compute_float_softmax(input: &BoundedTensor, axis: usize) -> BoundedTensor {
    let shape = input.shape();
    let softmax_size = shape.get(axis).copied().unwrap_or(1);
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let data = input.data();
    let mut result = vec![BoundedValue::exact(0.0); input.len()];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Find max for stability
            let mut max_val = f64::NEG_INFINITY;
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                max_val = max_val.max(data[idx].value());
            }

            // Compute exp and sum
            let mut exp_sum = 0.0;
            let mut exps = Vec::with_capacity(softmax_size);
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let e = (data[idx].value() - max_val).exp();
                exps.push(e);
                exp_sum += e;
            }

            // Normalize
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let prob = exps[s] / exp_sum;
                result[idx] = BoundedValue::<f64>::with_absolute_error(prob, data[idx].absolute_error());
            }
        }
    }

    BoundedTensor::new(result, shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_levels() {
        assert_eq!(PrecisionLevel::INT8Symmetric.bits(), 8);
        assert_eq!(PrecisionLevel::INT4Symmetric.bits(), 4);
        assert_eq!(PrecisionLevel::FP32.bits(), 32);

        assert!(PrecisionLevel::INT8Symmetric.is_integer());
        assert!(!PrecisionLevel::FP32.is_integer());
    }

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig::w4a8();
        assert_eq!(config.weight_precision, PrecisionLevel::INT4Symmetric);
        assert_eq!(config.activation_precision, PrecisionLevel::INT8Asymmetric);
    }

    #[test]
    fn test_mixed_precision_tensor() {
        let tensor = BoundedTensor::from_exact(vec![0.5, -0.5, 1.0, -1.0], vec![4]);

        let int8 = MixedPrecisionTensor::from_bounded(&tensor, PrecisionLevel::INT8Symmetric);
        let int4 = MixedPrecisionTensor::from_bounded(&tensor, PrecisionLevel::INT4Symmetric);

        assert_eq!(int8.precision(), PrecisionLevel::INT8Symmetric);
        assert_eq!(int4.precision(), PrecisionLevel::INT4Symmetric);

        // INT4 uses less memory
        assert!(int4.memory_bytes() < int8.memory_bytes());
    }

    #[test]
    fn test_memory_footprint() {
        let config = MixedPrecisionConfig::w4a8();
        let footprint = config.compute_memory_footprint(1_000_000, 100_000);

        // INT4 weights should have 8x compression
        assert!(footprint.compression_ratio_weights > 7.0);
        // INT8 activations should have 4x compression
        assert!(footprint.compression_ratio_activations > 3.0);
    }

    #[test]
    fn test_executor() {
        let mut executor = MixedPrecisionExecutor::w8a8();

        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let weights = BoundedTensor::from_exact(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]);

        let input_mp = executor.quantize_activations(&input, "test");
        let weights_mp = executor.quantize_weights(&weights, "test");

        let output = executor.linear(&input_mp, &weights_mp, None, "test");

        assert_eq!(output.shape(), &vec![2, 2]);
        assert!(executor.stats().operations > 0);
    }
}
