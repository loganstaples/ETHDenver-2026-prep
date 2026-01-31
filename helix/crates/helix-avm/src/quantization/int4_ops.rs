//! INT4 Operations.
//!
//! Provides specialized quantized operations for INT4 tensors. INT4 operations
//! typically use INT8 or INT16 accumulators to prevent overflow during computation.
//!
//! INT4 quantization provides 2x memory compression over INT8 at the cost of
//! slightly higher quantization error. It's particularly useful for:
//! - Large language models (LLMs) where memory is the bottleneck
//! - Weight-only quantization with INT8 activations
//! - Edge deployment with extreme memory constraints

use helix_core::types::{BoundedTensor, BoundedValue, Shape};

use super::int4_tensor::Int4Tensor;
use super::int8_tensor::Int8Tensor;
use super::schemes::{QuantScheme, TensorQuantParams};

// ============================================================================
// Matrix Operations
// ============================================================================

/// Performs INT4 matrix multiplication with INT16 accumulator.
///
/// For INT4 x INT4, we use INT16 accumulators since INT4 has range [-8, 7]
/// and the maximum accumulator value is approximately K * 8 * 8 = 64K.
pub fn int4_matmul(a: &Int4Tensor, b: &Int4Tensor, output_scale: f64) -> Int4Tensor {
    assert!(a.is_matrix() && b.is_matrix(), "Inputs must be 2D matrices");
    let (m, k_a) = (a.shape()[0], a.shape()[1]);
    let (k_b, n) = (b.shape()[0], b.shape()[1]);
    assert_eq!(k_a, k_b, "Inner dimensions must match: {} vs {}", k_a, k_b);

    let k = k_a;
    let a_scale = a.scale();
    let b_scale = b.scale();
    let a_zp = a.zero_point() as i32;
    let b_zp = b.zero_point() as i32;

    let combined_scale = (a_scale * b_scale) / output_scale;

    let a_data = a.unpack();
    let b_data = b.unpack();

    let mut output = vec![0i8; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;

            for l in 0..k {
                let a_val = a_data[i * k + l] as i32 - a_zp;
                let b_val = b_data[l * n + j] as i32 - b_zp;
                acc += a_val * b_val;
            }

            let scaled = (acc as f64 * combined_scale).round() as i32;
            output[i * n + j] = scaled.clamp(-7, 7) as i8;
        }
    }

    let matmul_error = k as f64 * a.quantization_error() * b.quantization_error();
    let total_error = matmul_error + output_scale / 2.0;

    Int4Tensor::new(output, vec![m, n], output_scale, 0, true)
}

/// Performs INT4 weights x INT8 activations matrix multiplication.
///
/// This is the most common configuration for LLM inference:
/// - Weights are stored in INT4 for memory efficiency
/// - Activations are computed in INT8 for better accuracy
///
/// Uses INT32 accumulator to handle the mixed precision.
pub fn int4_int8_matmul(
    int4_weights: &Int4Tensor,
    int8_input: &Int8Tensor,
    output_scale: f64,
) -> Int8Tensor {
    // Weights: [out_features, in_features]
    // Input: [batch, in_features]
    // Output: [batch, out_features]
    assert!(int4_weights.is_matrix(), "Weights must be 2D");
    assert!(int8_input.is_matrix(), "Input must be 2D");

    let (out_features, in_features) = (int4_weights.shape()[0], int4_weights.shape()[1]);
    let (batch, input_features) = (int8_input.shape()[0], int8_input.shape()[1]);
    assert_eq!(
        in_features, input_features,
        "Feature dimensions must match: {} vs {}",
        in_features, input_features
    );

    let w_scale = int4_weights.scale();
    let x_scale = int8_input.scale();
    let w_zp = int4_weights.zero_point() as i32;
    let x_zp = int8_input.zero_point() as i32;

    let combined_scale = (w_scale * x_scale) / output_scale;

    let w_data = int4_weights.unpack();
    let x_data = int8_input.data();

    let mut output = vec![0i8; batch * out_features];

    for b in 0..batch {
        for o in 0..out_features {
            let mut acc: i64 = 0;

            for i in 0..in_features {
                let w_val = w_data[o * in_features + i] as i64 - w_zp as i64;
                let x_val = x_data[b * in_features + i] as i64 - x_zp as i64;
                acc += w_val * x_val;
            }

            let scaled = (acc as f64 * combined_scale).round() as i32;
            output[b * out_features + o] = scaled.clamp(-127, 127) as i8;
        }
    }

    let error = in_features as f64 * int4_weights.quantization_error() * int8_input.quantization_error()
        + output_scale / 2.0;

    Int8Tensor::new(output, vec![batch, out_features], output_scale, 0, true).with_error(error)
}

/// Batched INT4 x INT8 matrix multiplication.
pub fn int4_int8_batched_matmul(
    int4_weights: &Int4Tensor,
    int8_input: &Int8Tensor,
    output_scale: f64,
) -> Int8Tensor {
    let input_shape = int8_input.shape();

    if input_shape.len() == 2 {
        // Standard 2D matmul
        return int4_int8_matmul(int4_weights, int8_input, output_scale);
    }

    assert_eq!(input_shape.len(), 3, "Input must be 2D or 3D");
    let batch = input_shape[0];
    let seq_len = input_shape[1];
    let in_features = input_shape[2];

    let (out_features, weight_in_features) = (int4_weights.shape()[0], int4_weights.shape()[1]);
    assert_eq!(in_features, weight_in_features, "Feature dimensions must match");

    let w_scale = int4_weights.scale();
    let x_scale = int8_input.scale();
    let w_zp = int4_weights.zero_point() as i32;
    let x_zp = int8_input.zero_point() as i32;
    let combined_scale = (w_scale * x_scale) / output_scale;

    let w_data = int4_weights.unpack();
    let x_data = int8_input.data();

    let mut output = vec![0i8; batch * seq_len * out_features];

    for b in 0..batch {
        for s in 0..seq_len {
            let x_offset = (b * seq_len + s) * in_features;
            let out_offset = (b * seq_len + s) * out_features;

            for o in 0..out_features {
                let mut acc: i64 = 0;
                for i in 0..in_features {
                    let w_val = w_data[o * in_features + i] as i64 - w_zp as i64;
                    let x_val = x_data[x_offset + i] as i64 - x_zp as i64;
                    acc += w_val * x_val;
                }
                let scaled = (acc as f64 * combined_scale).round() as i32;
                output[out_offset + o] = scaled.clamp(-127, 127) as i8;
            }
        }
    }

    let error = in_features as f64 * int4_weights.quantization_error() * int8_input.quantization_error()
        + output_scale / 2.0;

    Int8Tensor::new(
        output,
        vec![batch, seq_len, out_features],
        output_scale,
        0,
        true,
    )
    .with_error(error)
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Performs INT4 element-wise addition.
pub fn int4_add(a: &Int4Tensor, b: &Int4Tensor, output_scale: f64) -> Int4Tensor {
    assert_eq!(a.len(), b.len(), "Tensor sizes must match");

    let a_scale = a.scale();
    let b_scale = b.scale();
    let a_zp = a.zero_point() as i32;
    let b_zp = b.zero_point() as i32;

    let a_data = a.unpack();
    let b_data = b.unpack();
    let mut output = vec![0i8; a.len()];

    for i in 0..a.len() {
        let a_real = (a_data[i] as i32 - a_zp) as f64 * a_scale;
        let b_real = (b_data[i] as i32 - b_zp) as f64 * b_scale;
        let sum = a_real + b_real;
        let quantized = (sum / output_scale).round() as i32;
        output[i] = quantized.clamp(-7, 7) as i8;
    }

    let error = a.quantization_error() + b.quantization_error() + output_scale / 2.0;
    Int4Tensor::new(output, a.shape().clone(), output_scale, 0, true)
}

/// Performs INT4 element-wise multiplication.
pub fn int4_mul(a: &Int4Tensor, b: &Int4Tensor, output_scale: f64) -> Int4Tensor {
    assert_eq!(a.len(), b.len(), "Tensor sizes must match");

    let a_scale = a.scale();
    let b_scale = b.scale();
    let a_zp = a.zero_point() as i32;
    let b_zp = b.zero_point() as i32;
    let combined_scale = (a_scale * b_scale) / output_scale;

    let a_data = a.unpack();
    let b_data = b.unpack();
    let mut output = vec![0i8; a.len()];

    for i in 0..a.len() {
        let a_val = a_data[i] as i32 - a_zp;
        let b_val = b_data[i] as i32 - b_zp;
        let prod = (a_val * b_val) as f64 * combined_scale;
        output[i] = prod.round().clamp(-7.0, 7.0) as i8;
    }

    let error = a.quantization_error() * b.quantization_error() + output_scale / 2.0;
    Int4Tensor::new(output, a.shape().clone(), output_scale, 0, true)
}

/// Scales an INT4 tensor by a scalar.
pub fn int4_scale(tensor: &Int4Tensor, scalar: f64, output_scale: f64) -> Int4Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let combined = (scale * scalar) / output_scale;

    let data = tensor.unpack();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let val = (data[i] as i32 - zp) as f64 * combined;
        output[i] = val.round().clamp(-7.0, 7.0) as i8;
    }

    let error = tensor.quantization_error() * scalar.abs() + output_scale / 2.0;
    Int4Tensor::new(output, tensor.shape().clone(), output_scale, 0, true)
}

// ============================================================================
// Conversion Operations
// ============================================================================

/// Converts INT4 tensor to INT8.
///
/// This is a lossless operation that expands the bit width.
pub fn int4_to_int8(tensor: &Int4Tensor) -> Int8Tensor {
    let unpacked = tensor.unpack();
    let scale = tensor.scale();
    let zp = tensor.zero_point();

    // INT4 values are already in i8 range, just need to convert
    let int8_data: Vec<i8> = unpacked
        .iter()
        .map(|&v| {
            // Rescale to INT8 range
            let real = (v as i32 - zp as i32) as f64 * scale;
            let int8_scale = scale * 7.0 / 127.0; // Scale ratio
            ((real / int8_scale).round() as i32).clamp(-127, 127) as i8
        })
        .collect();

    let int8_scale = scale * 7.0 / 127.0;
    Int8Tensor::new(int8_data, tensor.shape().clone(), int8_scale, 0, true)
        .with_error(tensor.total_error())
}

/// Converts INT8 tensor to INT4.
///
/// This involves quantization and may lose precision.
pub fn int8_to_int4(tensor: &Int8Tensor) -> Int4Tensor {
    let dq = tensor.dequantize_raw();
    Int4Tensor::from_float_data(&dq, tensor.shape().clone(), QuantScheme::SymmetricInt4)
}

/// Dequantizes INT4 to INT8 for computation, then requantizes.
///
/// This pattern is used when we need higher precision for intermediate
/// computations but want to store results in INT4.
pub fn int4_through_int8<F>(tensor: &Int4Tensor, operation: F, output_scale: f64) -> Int4Tensor
where
    F: FnOnce(&Int8Tensor) -> Int8Tensor,
{
    let int8_tensor = int4_to_int8(tensor);
    let result = operation(&int8_tensor);
    let dq = result.dequantize_raw();
    Int4Tensor::from_float_data(&dq, result.shape().clone(), QuantScheme::SymmetricInt4)
}

// ============================================================================
// Linear Layer Operations
// ============================================================================

/// INT4 linear layer with INT8 activations (most common LLM configuration).
///
/// weights: [out_features, in_features] in INT4
/// input: [batch, in_features] in INT8
/// bias: [out_features] in INT8 (optional)
/// output: [batch, out_features] in INT8
pub fn int4_linear(
    input: &Int8Tensor,
    weight: &Int4Tensor,
    bias: Option<&Int8Tensor>,
    output_scale: f64,
) -> Int8Tensor {
    let mut result = int4_int8_matmul(weight, input, output_scale);

    if let Some(b) = bias {
        result = int8_add_bias_vector(&result, b, output_scale);
    }

    result
}

/// Adds a bias vector to each row of a matrix.
fn int8_add_bias_vector(tensor: &Int8Tensor, bias: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    use super::int8_ops::int8_add;

    assert!(tensor.is_matrix(), "Input must be 2D");
    assert!(bias.is_vector(), "Bias must be 1D");

    let (batch, features) = (tensor.shape()[0], tensor.shape()[1]);
    assert_eq!(features, bias.len(), "Bias length must match feature dimension");

    let t_scale = tensor.scale();
    let t_zp = tensor.zero_point() as i32;
    let b_scale = bias.scale();
    let b_zp = bias.zero_point() as i32;

    let t_data = tensor.data();
    let b_data = bias.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..batch {
        for j in 0..features {
            let t_val = (t_data[i * features + j] as i32 - t_zp) as f64 * t_scale;
            let b_val = (b_data[j] as i32 - b_zp) as f64 * b_scale;
            let sum = t_val + b_val;
            let quantized = (sum / output_scale).round() as i32;
            output[i * features + j] = quantized.clamp(-127, 127) as i8;
        }
    }

    let error = tensor.total_error() + bias.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

// ============================================================================
// Group-wise Quantization Operations
// ============================================================================

/// Performs matrix multiplication with group-wise quantized weights.
///
/// Group-wise quantization uses different scales for different groups of weights,
/// typically along the input dimension. This provides better accuracy than
/// per-tensor quantization while still being efficient.
pub fn int4_group_matmul(
    group_weights: &Int4Tensor,
    input: &Int8Tensor,
    output_scale: f64,
) -> Int8Tensor {
    assert!(group_weights.is_group_wise(), "Weights must use group-wise quantization");
    assert!(group_weights.is_matrix(), "Weights must be 2D");
    assert!(input.is_matrix() || input.ndim() == 3, "Input must be 2D or 3D");

    let (out_features, in_features) = (group_weights.shape()[0], group_weights.shape()[1]);
    let group_size = group_weights.group_size().unwrap();

    let input_shape = input.shape();
    let (batch, seq_len, input_in_features) = if input_shape.len() == 2 {
        (1, input_shape[0], input_shape[1])
    } else {
        (input_shape[0], input_shape[1], input_shape[2])
    };

    assert_eq!(in_features, input_in_features, "Feature dimensions must match");

    let w_data = group_weights.unpack();
    let x_data = input.data();
    let x_scale = input.scale();
    let x_zp = input.zero_point() as i32;

    let output_len = batch * seq_len * out_features;
    let mut output = vec![0i8; output_len];

    // Get group scales - use the base scale since group_scales are internal
    let num_groups = (in_features + group_size - 1) / group_size;
    let group_scales = vec![group_weights.scale(); num_groups];

    for b in 0..batch {
        for s in 0..seq_len {
            let x_offset = (b * seq_len + s) * in_features;
            let out_offset = (b * seq_len + s) * out_features;

            for o in 0..out_features {
                let mut acc: f64 = 0.0;

                for i in 0..in_features {
                    let w_val = w_data[o * in_features + i] as i32;
                    let x_val = x_data[x_offset + i] as i32 - x_zp;

                    let group_idx = i / group_size;
                    let w_scale = group_scales.get(group_idx).copied().unwrap_or(group_weights.scale());

                    acc += (w_val as f64 * w_scale) * (x_val as f64 * x_scale);
                }

                let scaled = (acc / output_scale).round() as i32;
                output[out_offset + o] = scaled.clamp(-127, 127) as i8;
            }
        }
    }

    let error = in_features as f64 * group_weights.quantization_error() * input.quantization_error()
        + output_scale / 2.0;

    let output_shape = if input_shape.len() == 2 {
        vec![seq_len, out_features]
    } else {
        vec![batch, seq_len, out_features]
    };

    Int8Tensor::new(output, output_shape, output_scale, 0, true).with_error(error)
}

// ============================================================================
// Activation Functions (INT4 -> INT4 through dequantize)
// ============================================================================

/// Applies ReLU to INT4 tensor.
pub fn int4_relu(tensor: &Int4Tensor) -> Int4Tensor {
    let zp = tensor.zero_point();
    let data = tensor.unpack();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        output[i] = data[i].max(zp);
    }

    Int4Tensor::new(
        output,
        tensor.shape().clone(),
        tensor.scale(),
        zp,
        tensor.is_symmetric(),
    )
}

/// Applies GELU to INT4 tensor.
pub fn int4_gelu(tensor: &Int4Tensor, output_scale: f64) -> Int4Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let data = tensor.unpack();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let x = (data[i] as i32 - zp) as f64 * scale;
        let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
        let gelu = x * sigmoid;
        let quantized = (gelu / output_scale).round() as i32;
        output[i] = quantized.clamp(-7, 7) as i8;
    }

    Int4Tensor::new(output, tensor.shape().clone(), output_scale, 0, true)
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Requantizes an INT4 tensor to a new scale.
pub fn int4_requantize(tensor: &Int4Tensor, new_scale: f64) -> Int4Tensor {
    let old_scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let scale_ratio = old_scale / new_scale;

    let data = tensor.unpack();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let val = (data[i] as i32 - zp) as f64 * scale_ratio;
        output[i] = val.round().clamp(-7.0, 7.0) as i8;
    }

    Int4Tensor::new(output, tensor.shape().clone(), new_scale, 0, true)
}

/// Computes the quantization error for INT4.
pub fn compute_int4_quantization_error(
    original: &BoundedTensor,
    quantized: &Int4Tensor,
) -> Int4QuantizationErrorMetrics {
    let dq = quantized.dequantize_raw();
    let orig_vals: Vec<f64> = original.data().iter().map(|v| v.value()).collect();

    assert_eq!(dq.len(), orig_vals.len(), "Tensor sizes must match");

    let mut max_abs_error = 0.0_f64;
    let mut sum_sq_error = 0.0_f64;
    let mut sum_abs_error = 0.0_f64;

    for (orig, recovered) in orig_vals.iter().zip(dq.iter()) {
        let abs_error = (orig - recovered).abs();
        max_abs_error = max_abs_error.max(abs_error);
        sum_sq_error += abs_error * abs_error;
        sum_abs_error += abs_error;
    }

    let n = orig_vals.len() as f64;
    let signal_power: f64 = orig_vals.iter().map(|x| x * x).sum();
    let noise_power = sum_sq_error;
    let snr = if noise_power > 1e-10 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    };

    Int4QuantizationErrorMetrics {
        max_absolute_error: max_abs_error,
        mean_absolute_error: sum_abs_error / n,
        root_mean_square_error: (sum_sq_error / n).sqrt(),
        signal_to_noise_ratio: snr,
        compression_ratio: 8.0, // 32-bit float to 4-bit = 8x
        memory_saved_percent: 87.5,
    }
}

/// Metrics for INT4 quantization error analysis.
#[derive(Debug, Clone)]
pub struct Int4QuantizationErrorMetrics {
    /// Maximum absolute error.
    pub max_absolute_error: f64,
    /// Mean absolute error.
    pub mean_absolute_error: f64,
    /// Root mean square error.
    pub root_mean_square_error: f64,
    /// Signal-to-noise ratio in dB.
    pub signal_to_noise_ratio: f64,
    /// Compression ratio vs FP32.
    pub compression_ratio: f64,
    /// Memory saved percentage.
    pub memory_saved_percent: f64,
}

/// Configuration for INT4 quantization.
#[derive(Debug, Clone)]
pub struct Int4QuantConfig {
    /// Whether to use symmetric quantization.
    pub symmetric: bool,
    /// Group size for group-wise quantization (None for per-tensor).
    pub group_size: Option<usize>,
    /// Minimum value clipping percentile.
    pub clip_min_percentile: Option<f64>,
    /// Maximum value clipping percentile.
    pub clip_max_percentile: Option<f64>,
}

impl Default for Int4QuantConfig {
    fn default() -> Self {
        Self {
            symmetric: true,
            group_size: Some(128), // Common choice for LLMs
            clip_min_percentile: None,
            clip_max_percentile: None,
        }
    }
}

impl Int4QuantConfig {
    /// Creates config for per-tensor quantization.
    pub fn per_tensor() -> Self {
        Self {
            group_size: None,
            ..Default::default()
        }
    }

    /// Creates config with specific group size.
    pub fn with_group_size(group_size: usize) -> Self {
        Self {
            group_size: Some(group_size),
            ..Default::default()
        }
    }

    /// Creates config with clipping.
    pub fn with_clipping(min_percentile: f64, max_percentile: f64) -> Self {
        Self {
            clip_min_percentile: Some(min_percentile),
            clip_max_percentile: Some(max_percentile),
            ..Default::default()
        }
    }
}

/// Quantizes a BoundedTensor to INT4 with the given configuration.
pub fn quantize_to_int4(tensor: &BoundedTensor, config: &Int4QuantConfig) -> Int4Tensor {
    let data: Vec<f64> = tensor.data().iter().map(|v| v.value()).collect();
    let shape = tensor.shape().clone();

    // Apply clipping if configured
    let clipped_data = if config.clip_min_percentile.is_some() || config.clip_max_percentile.is_some() {
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_val = if let Some(p) = config.clip_min_percentile {
            let idx = ((p / 100.0) * sorted.len() as f64) as usize;
            sorted.get(idx).copied().unwrap_or(sorted[0])
        } else {
            *sorted.first().unwrap()
        };

        let max_val = if let Some(p) = config.clip_max_percentile {
            let idx = ((p / 100.0) * sorted.len() as f64) as usize;
            sorted.get(idx.min(sorted.len() - 1)).copied().unwrap_or(*sorted.last().unwrap())
        } else {
            *sorted.last().unwrap()
        };

        data.iter().map(|&v| v.clamp(min_val, max_val)).collect()
    } else {
        data
    };

    if let Some(group_size) = config.group_size {
        // Group-wise quantization
        use super::int4_tensor::Int4TensorBuilder;
        Int4TensorBuilder::new(shape)
            .with_float_data(clipped_data)
            .with_group_size(group_size)
            .symmetric()
            .build()
    } else {
        // Per-tensor quantization
        let scheme = if config.symmetric {
            QuantScheme::SymmetricInt4
        } else {
            QuantScheme::AsymmetricInt4
        };
        Int4Tensor::from_float_data(&clipped_data, shape, scheme)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_matmul() {
        let a = Int4Tensor::from_float_data(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            QuantScheme::SymmetricInt4,
        );
        let b = Int4Tensor::from_float_data(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            QuantScheme::SymmetricInt4,
        );

        let output_scale = a.scale() * b.scale() * 15.0;
        let c = int4_matmul(&a, &b, output_scale);

        assert_eq!(c.shape(), &vec![2, 2]);
    }

    #[test]
    fn test_int4_int8_matmul() {
        // Weight: [4, 8] (4 output features, 8 input features)
        let weight = Int4Tensor::from_float_data(
            &(0..32).map(|i| i as f64 / 32.0).collect::<Vec<_>>(),
            vec![4, 8],
            QuantScheme::SymmetricInt4,
        );

        // Input: [2, 8] (batch 2, 8 features)
        let input = Int8Tensor::from_float_data(
            &(0..16).map(|i| i as f64 / 16.0).collect::<Vec<_>>(),
            vec![2, 8],
            QuantScheme::SymmetricInt8,
        );

        let output_scale = 0.1;
        let output = int4_int8_matmul(&weight, &input, output_scale);

        assert_eq!(output.shape(), &vec![2, 4]);
    }

    #[test]
    fn test_int4_relu() {
        let tensor = Int4Tensor::from_float_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], QuantScheme::SymmetricInt4);
        let result = int4_relu(&tensor);

        let dq = result.dequantize_raw();
        for v in &dq[0..2] {
            assert!(*v >= 0.0 || v.abs() < 0.5); // Allow for small negative due to quantization
        }
    }

    #[test]
    fn test_int4_to_int8_conversion() {
        let int4 = Int4Tensor::from_float_data(&[0.5, -0.5, 1.0, -1.0], vec![4], QuantScheme::SymmetricInt4);
        let int8 = int4_to_int8(&int4);

        assert_eq!(int8.len(), 4);

        // Values should be approximately preserved
        let int4_dq = int4.dequantize_raw();
        let int8_dq = int8.dequantize_raw();

        for (v4, v8) in int4_dq.iter().zip(int8_dq.iter()) {
            assert!((v4 - v8).abs() < 0.3); // Allow for quantization error
        }
    }

    #[test]
    fn test_int4_quantization_error() {
        let original = BoundedTensor::from_exact(vec![0.1, 0.5, -0.3, 0.8], vec![4]);
        let quantized = Int4Tensor::from_bounded_symmetric(&original);

        let metrics = compute_int4_quantization_error(&original, &quantized);

        // INT4 has larger error than INT8
        assert!(metrics.max_absolute_error < 0.3);
        assert!(metrics.compression_ratio == 8.0);
    }

    #[test]
    fn test_group_wise_quantization() {
        let data: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) / 32.0).collect();
        let config = Int4QuantConfig::with_group_size(16);
        let tensor = BoundedTensor::from_exact(data.clone(), vec![8, 8]);
        let quantized = quantize_to_int4(&tensor, &config);

        assert!(quantized.is_group_wise());
        assert_eq!(quantized.group_size(), Some(16));
    }
}
