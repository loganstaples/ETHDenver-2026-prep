//! INT8 Operations.
//!
//! Provides specialized quantized operations for INT8 tensors with proper
//! error bound tracking. These operations are designed for efficient inference
//! and training with minimal accuracy loss.
//!
//! All operations follow the HELIX approximate computing model:
//! 1. Compute in higher precision (INT32) accumulator
//! 2. Track error bounds through computation
//! 3. Requantize to INT8 for storage

use helix_core::types::BoundedTensor;
use thiserror::Error;

use super::int8_tensor::Int8Tensor;

/// Errors from quantized operations.
#[derive(Error, Debug)]
pub enum QuantizedOpError {
    #[error("Unsupported tensor shape combination for batched matmul: {a_shape:?} @ {b_shape:?}")]
    UnsupportedShape {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Performs INT8 matrix multiplication with INT32 accumulator.
///
/// Computes C = A @ B where:
/// - A has shape [M, K] with scale_a and zero_point_a
/// - B has shape [K, N] with scale_b and zero_point_b
/// - Output C has shape [M, N] with output_scale
///
/// The computation uses INT32 accumulators to prevent overflow:
/// acc = Σ(a_i - zp_a) * (b_i - zp_b)
/// output = round(acc * scale_a * scale_b / output_scale) + output_zp
pub fn int8_matmul(a: &Int8Tensor, b: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    assert!(a.is_matrix() && b.is_matrix(), "Inputs must be 2D matrices");
    let (m, k_a) = (a.shape()[0], a.shape()[1]);
    let (k_b, n) = (b.shape()[0], b.shape()[1]);
    assert_eq!(k_a, k_b, "Inner dimensions must match: {} vs {}", k_a, k_b);

    let k = k_a;
    let a_scale = a.scale();
    let b_scale = b.scale();
    let a_zp = a.zero_point() as i32;
    let b_zp = b.zero_point() as i32;

    // Combined scale for requantization
    let combined_scale = (a_scale * b_scale) / output_scale;

    let a_data = a.data();
    let b_data = b.data();

    let mut output = vec![0i8; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;

            for l in 0..k {
                let a_val = a_data[i * k + l] as i32 - a_zp;
                let b_val = b_data[l * n + j] as i32 - b_zp;
                acc += a_val * b_val;
            }

            // Requantize
            let scaled = (acc as f64 * combined_scale).round() as i32;
            output[i * n + j] = scaled.clamp(-127, 127) as i8;
        }
    }

    // Error bound: accumulation error + quantization error
    let matmul_error = k as f64 * a.quantization_error() * b.quantization_error();
    let total_error = matmul_error + output_scale / 2.0;

    Int8Tensor::new(output, vec![m, n], output_scale, 0, true).with_error(total_error)
}

/// Performs batched INT8 matrix multiplication.
///
/// For tensors with shape [batch, M, K] @ [batch, K, N] -> [batch, M, N]
/// or broadcasting [batch, M, K] @ [K, N] -> [batch, M, N]
pub fn int8_batched_matmul(
    a: &Int8Tensor,
    b: &Int8Tensor,
    output_scale: f64,
) -> Result<Int8Tensor, QuantizedOpError> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    match (a_shape.len(), b_shape.len()) {
        (3, 3) => {
            // Full batched: [batch, M, K] @ [batch, K, N]
            let batch = a_shape[0];
            assert_eq!(batch, b_shape[0], "Batch dimensions must match");
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];
            assert_eq!(k, b_shape[1], "Inner dimensions must match");

            let a_scale = a.scale();
            let b_scale = b.scale();
            let a_zp = a.zero_point() as i32;
            let b_zp = b.zero_point() as i32;
            let combined_scale = (a_scale * b_scale) / output_scale;

            let a_data = a.data();
            let b_data = b.data();
            let mut output = vec![0i8; batch * m * n];

            for batch_idx in 0..batch {
                let a_offset = batch_idx * m * k;
                let b_offset = batch_idx * k * n;
                let out_offset = batch_idx * m * n;

                for i in 0..m {
                    for j in 0..n {
                        let mut acc: i32 = 0;
                        for l in 0..k {
                            let a_val = a_data[a_offset + i * k + l] as i32 - a_zp;
                            let b_val = b_data[b_offset + l * n + j] as i32 - b_zp;
                            acc += a_val * b_val;
                        }
                        let scaled = (acc as f64 * combined_scale).round() as i32;
                        output[out_offset + i * n + j] = scaled.clamp(-127, 127) as i8;
                    }
                }
            }

            let error = k as f64 * a.quantization_error() * b.quantization_error() + output_scale / 2.0;
            Ok(Int8Tensor::new(output, vec![batch, m, n], output_scale, 0, true).with_error(error))
        }
        (3, 2) => {
            // Broadcast: [batch, M, K] @ [K, N]
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[1];
            assert_eq!(k, b_shape[0], "Inner dimensions must match");

            let a_scale = a.scale();
            let b_scale = b.scale();
            let a_zp = a.zero_point() as i32;
            let b_zp = b.zero_point() as i32;
            let combined_scale = (a_scale * b_scale) / output_scale;

            let a_data = a.data();
            let b_data = b.data();
            let mut output = vec![0i8; batch * m * n];

            for batch_idx in 0..batch {
                let a_offset = batch_idx * m * k;
                let out_offset = batch_idx * m * n;

                for i in 0..m {
                    for j in 0..n {
                        let mut acc: i32 = 0;
                        for l in 0..k {
                            let a_val = a_data[a_offset + i * k + l] as i32 - a_zp;
                            let b_val = b_data[l * n + j] as i32 - b_zp;
                            acc += a_val * b_val;
                        }
                        let scaled = (acc as f64 * combined_scale).round() as i32;
                        output[out_offset + i * n + j] = scaled.clamp(-127, 127) as i8;
                    }
                }
            }

            let error = k as f64 * a.quantization_error() * b.quantization_error() + output_scale / 2.0;
            Ok(Int8Tensor::new(output, vec![batch, m, n], output_scale, 0, true).with_error(error))
        }
        (2, 2) => {
            // Standard 2D matmul
            Ok(int8_matmul(a, b, output_scale))
        }
        _ => Err(QuantizedOpError::UnsupportedShape {
            a_shape: a_shape.to_vec(),
            b_shape: b_shape.to_vec(),
        }),
    }
}

/// Performs INT8 matrix-vector multiplication.
pub fn int8_matvec(matrix: &Int8Tensor, vector: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    assert!(matrix.is_matrix(), "First input must be 2D matrix");
    assert!(vector.is_vector(), "Second input must be 1D vector");

    let (m, k) = (matrix.shape()[0], matrix.shape()[1]);
    assert_eq!(k, vector.len(), "Matrix columns must match vector length");

    let m_scale = matrix.scale();
    let v_scale = vector.scale();
    let m_zp = matrix.zero_point() as i32;
    let v_zp = vector.zero_point() as i32;
    let combined_scale = (m_scale * v_scale) / output_scale;

    let m_data = matrix.data();
    let v_data = vector.data();
    let mut output = vec![0i8; m];

    for i in 0..m {
        let mut acc: i32 = 0;
        for j in 0..k {
            let m_val = m_data[i * k + j] as i32 - m_zp;
            let v_val = v_data[j] as i32 - v_zp;
            acc += m_val * v_val;
        }
        let scaled = (acc as f64 * combined_scale).round() as i32;
        output[i] = scaled.clamp(-127, 127) as i8;
    }

    let error = k as f64 * matrix.quantization_error() * vector.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, vec![m], output_scale, 0, true).with_error(error)
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Performs INT8 element-wise addition.
pub fn int8_add(a: &Int8Tensor, b: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    assert_eq!(a.len(), b.len(), "Tensor sizes must match");

    let a_scale = a.scale();
    let b_scale = b.scale();
    let a_zp = a.zero_point() as i32;
    let b_zp = b.zero_point() as i32;

    let a_data = a.data();
    let b_data = b.data();
    let mut output = vec![0i8; a.len()];

    for i in 0..a.len() {
        // Dequantize, add, requantize
        let a_real = (a_data[i] as i32 - a_zp) as f64 * a_scale;
        let b_real = (b_data[i] as i32 - b_zp) as f64 * b_scale;
        let sum = a_real + b_real;
        let quantized = (sum / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    let error = a.quantization_error() + b.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, a.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Performs INT8 element-wise subtraction.
pub fn int8_sub(a: &Int8Tensor, b: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    assert_eq!(a.len(), b.len(), "Tensor sizes must match");

    let a_scale = a.scale();
    let b_scale = b.scale();
    let a_zp = a.zero_point() as i32;
    let b_zp = b.zero_point() as i32;

    let a_data = a.data();
    let b_data = b.data();
    let mut output = vec![0i8; a.len()];

    for i in 0..a.len() {
        let a_real = (a_data[i] as i32 - a_zp) as f64 * a_scale;
        let b_real = (b_data[i] as i32 - b_zp) as f64 * b_scale;
        let diff = a_real - b_real;
        let quantized = (diff / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    let error = a.quantization_error() + b.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, a.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Performs INT8 element-wise multiplication (Hadamard product).
pub fn int8_mul(a: &Int8Tensor, b: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    assert_eq!(a.len(), b.len(), "Tensor sizes must match");

    let a_scale = a.scale();
    let b_scale = b.scale();
    let a_zp = a.zero_point() as i32;
    let b_zp = b.zero_point() as i32;
    let combined_scale = (a_scale * b_scale) / output_scale;

    let a_data = a.data();
    let b_data = b.data();
    let mut output = vec![0i8; a.len()];

    for i in 0..a.len() {
        let a_val = a_data[i] as i32 - a_zp;
        let b_val = b_data[i] as i32 - b_zp;
        let prod = (a_val * b_val) as f64 * combined_scale;
        output[i] = prod.round().clamp(-127.0, 127.0) as i8;
    }

    let error = a.quantization_error() * b.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, a.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Scales an INT8 tensor by a scalar.
pub fn int8_scale(tensor: &Int8Tensor, scalar: f64, output_scale: f64) -> Int8Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let combined = (scale * scalar) / output_scale;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let val = (data[i] as i32 - zp) as f64 * combined;
        output[i] = val.round().clamp(-127.0, 127.0) as i8;
    }

    let error = tensor.quantization_error() * scalar.abs() + output_scale / 2.0;
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Adds a scalar bias to each element.
pub fn int8_add_bias(tensor: &Int8Tensor, bias: f64, output_scale: f64) -> Int8Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let val = (data[i] as i32 - zp) as f64 * scale + bias;
        let quantized = (val / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    let error = tensor.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

// ============================================================================
// Activation Functions
// ============================================================================

/// Applies ReLU activation in INT8.
pub fn int8_relu(tensor: &Int8Tensor) -> Int8Tensor {
    let zp = tensor.zero_point();
    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        // ReLU: max(0, x) -> in quantized space, max(zero_point, x)
        output[i] = data[i].max(zp);
    }

    // ReLU doesn't add quantization error
    Int8Tensor::new(
        output,
        tensor.shape().clone(),
        tensor.scale(),
        tensor.zero_point(),
        tensor.is_symmetric(),
    )
    .with_error(tensor.total_error())
}

/// Applies Leaky ReLU activation in INT8.
pub fn int8_leaky_relu(tensor: &Int8Tensor, negative_slope: f64, output_scale: f64) -> Int8Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let val = (data[i] as i32 - zp) as f64 * scale;
        let activated = if val >= 0.0 { val } else { val * negative_slope };
        let quantized = (activated / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    let error = tensor.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Applies GELU approximation in INT8.
///
/// Uses the fast sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
pub fn int8_gelu(tensor: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let x = (data[i] as i32 - zp) as f64 * scale;
        let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
        let gelu = x * sigmoid;
        let quantized = (gelu / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    // GELU approximation adds some error
    let error = tensor.quantization_error() + output_scale / 2.0 + 0.01; // 1% approximation error
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Applies SiLU (Swish) activation in INT8.
///
/// SiLU(x) = x * sigmoid(x)
pub fn int8_silu(tensor: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let x = (data[i] as i32 - zp) as f64 * scale;
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        let silu = x * sigmoid;
        let quantized = (silu / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    let error = tensor.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Applies sigmoid activation in INT8.
///
/// Uses piecewise linear approximation for efficiency.
pub fn int8_sigmoid(tensor: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let x = (data[i] as i32 - zp) as f64 * scale;
        // Piecewise linear approximation
        let sigmoid = if x <= -4.0 {
            0.0
        } else if x >= 4.0 {
            1.0
        } else {
            0.5 + 0.125 * x
        };
        let quantized = (sigmoid / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    let error = output_scale / 2.0 + 0.02; // Piecewise approximation error
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

/// Applies tanh activation in INT8.
pub fn int8_tanh(tensor: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let x = (data[i] as i32 - zp) as f64 * scale;
        // Piecewise linear approximation for tanh
        let tanh = if x <= -3.0 {
            -1.0
        } else if x >= 3.0 {
            1.0
        } else {
            x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
        };
        let quantized = (tanh / output_scale).round() as i32;
        output[i] = quantized.clamp(-127, 127) as i8;
    }

    let error = output_scale / 2.0 + 0.01;
    Int8Tensor::new(output, tensor.shape().clone(), output_scale, 0, true).with_error(error)
}

// ============================================================================
// Normalization Operations
// ============================================================================

/// Applies layer normalization in INT8.
pub fn int8_layer_norm(
    tensor: &Int8Tensor,
    gamma: &Int8Tensor,
    beta: &Int8Tensor,
    eps: f64,
    output_scale: f64,
) -> Int8Tensor {
    let shape = tensor.shape();
    assert!(shape.len() >= 1, "Input must have at least 1 dimension");

    let norm_size = *shape.last().unwrap();
    let num_groups = tensor.len() / norm_size;

    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let gamma_scale = gamma.scale();
    let gamma_zp = gamma.zero_point() as i32;
    let beta_scale = beta.scale();
    let beta_zp = beta.zero_point() as i32;

    let data = tensor.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();
    let mut output = vec![0i8; tensor.len()];

    for g in 0..num_groups {
        let start = g * norm_size;
        let end = start + norm_size;
        let group_data = &data[start..end];

        // Compute mean in float
        let mut sum = 0.0;
        for &v in group_data {
            sum += (v as i32 - zp) as f64 * scale;
        }
        let mean = sum / norm_size as f64;

        // Compute variance
        let mut var_sum = 0.0;
        for &v in group_data {
            let x = (v as i32 - zp) as f64 * scale;
            var_sum += (x - mean) * (x - mean);
        }
        let variance = var_sum / norm_size as f64;
        let std_inv = 1.0 / (variance + eps).sqrt();

        // Normalize and apply gamma/beta
        for i in 0..norm_size {
            let x = (group_data[i] as i32 - zp) as f64 * scale;
            let normalized = (x - mean) * std_inv;

            let g_val = (gamma_data[i % norm_size] as i32 - gamma_zp) as f64 * gamma_scale;
            let b_val = (beta_data[i % norm_size] as i32 - beta_zp) as f64 * beta_scale;

            let y = normalized * g_val + b_val;
            let quantized = (y / output_scale).round() as i32;
            output[start + i] = quantized.clamp(-127, 127) as i8;
        }
    }

    let error = tensor.quantization_error() + gamma.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, shape.clone(), output_scale, 0, true).with_error(error)
}

/// Applies RMS normalization in INT8.
pub fn int8_rms_norm(
    tensor: &Int8Tensor,
    gamma: &Int8Tensor,
    eps: f64,
    output_scale: f64,
) -> Int8Tensor {
    let shape = tensor.shape();
    assert!(shape.len() >= 1, "Input must have at least 1 dimension");

    let norm_size = *shape.last().unwrap();
    let num_groups = tensor.len() / norm_size;

    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let gamma_scale = gamma.scale();
    let gamma_zp = gamma.zero_point() as i32;

    let data = tensor.data();
    let gamma_data = gamma.data();
    let mut output = vec![0i8; tensor.len()];

    for g in 0..num_groups {
        let start = g * norm_size;
        let end = start + norm_size;
        let group_data = &data[start..end];

        // Compute RMS
        let mut sq_sum = 0.0;
        for &v in group_data {
            let x = (v as i32 - zp) as f64 * scale;
            sq_sum += x * x;
        }
        let rms = (sq_sum / norm_size as f64 + eps).sqrt();
        let rms_inv = 1.0 / rms;

        // Normalize and apply gamma
        for i in 0..norm_size {
            let x = (group_data[i] as i32 - zp) as f64 * scale;
            let normalized = x * rms_inv;

            let g_val = (gamma_data[i % norm_size] as i32 - gamma_zp) as f64 * gamma_scale;
            let y = normalized * g_val;

            let quantized = (y / output_scale).round() as i32;
            output[start + i] = quantized.clamp(-127, 127) as i8;
        }
    }

    let error = tensor.quantization_error() + gamma.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, shape.clone(), output_scale, 0, true).with_error(error)
}

// ============================================================================
// Softmax and Attention
// ============================================================================

/// Applies softmax in INT8.
pub fn int8_softmax(tensor: &Int8Tensor, axis: usize, output_scale: f64) -> Int8Tensor {
    let shape = tensor.shape();
    assert!(axis < shape.len(), "Axis out of bounds");

    let softmax_size = shape[axis];
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Find max for stability
            let mut max_val = f64::NEG_INFINITY;
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let x = (data[idx] as i32 - zp) as f64 * scale;
                max_val = max_val.max(x);
            }

            // Compute exp(x - max) and sum
            let mut exp_sum = 0.0;
            let mut exps = Vec::with_capacity(softmax_size);
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let x = (data[idx] as i32 - zp) as f64 * scale;
                let e = (x - max_val).exp();
                exps.push(e);
                exp_sum += e;
            }

            // Normalize and quantize
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let prob = exps[s] / exp_sum;
                let quantized = (prob / output_scale).round() as i32;
                output[idx] = quantized.clamp(0, 127) as i8; // Softmax outputs are positive
            }
        }
    }

    // Softmax introduces significant approximation error
    let error = output_scale / 2.0 + 0.01;
    Int8Tensor::new(output, shape.clone(), output_scale, 0, false).with_error(error)
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Computes the sum along an axis.
pub fn int8_sum(tensor: &Int8Tensor, axis: usize, output_scale: f64) -> Int8Tensor {
    let shape = tensor.shape();
    assert!(axis < shape.len(), "Axis out of bounds");

    let reduce_size = shape[axis];
    let mut new_shape: Vec<usize> = shape.iter().cloned().collect();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let data = tensor.data();

    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let mut output = vec![0i8; outer_size * inner_size];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut sum: f64 = 0.0;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                sum += (data[idx] as i32 - zp) as f64 * scale;
            }
            let quantized = (sum / output_scale).round() as i32;
            output[outer * inner_size + inner] = quantized.clamp(-127, 127) as i8;
        }
    }

    let error = reduce_size as f64 * tensor.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, new_shape, output_scale, 0, true).with_error(error)
}

/// Computes the mean along an axis.
pub fn int8_mean(tensor: &Int8Tensor, axis: usize, output_scale: f64) -> Int8Tensor {
    let shape = tensor.shape();
    assert!(axis < shape.len(), "Axis out of bounds");

    let reduce_size = shape[axis];
    let mut new_shape: Vec<usize> = shape.iter().cloned().collect();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    let scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let data = tensor.data();

    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let mut output = vec![0i8; outer_size * inner_size];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut sum: f64 = 0.0;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                sum += (data[idx] as i32 - zp) as f64 * scale;
            }
            let mean = sum / reduce_size as f64;
            let quantized = (mean / output_scale).round() as i32;
            output[outer * inner_size + inner] = quantized.clamp(-127, 127) as i8;
        }
    }

    let error = tensor.quantization_error() + output_scale / 2.0;
    Int8Tensor::new(output, new_shape, output_scale, 0, true).with_error(error)
}

// ============================================================================
// Linear Layer
// ============================================================================

/// Performs a quantized linear layer: y = Wx + b
pub fn int8_linear(
    input: &Int8Tensor,
    weight: &Int8Tensor,
    bias: Option<&Int8Tensor>,
    output_scale: f64,
) -> Int8Tensor {
    // Weight is [out_features, in_features], input is [batch, in_features]
    // Output is [batch, out_features]
    let weight_t = weight.transpose();
    let mut result = int8_matmul(input, &weight_t, output_scale);

    if let Some(b) = bias {
        result = int8_add_bias_vector(&result, b, output_scale);
    }

    result
}

/// Adds a bias vector to each row of a matrix.
fn int8_add_bias_vector(tensor: &Int8Tensor, bias: &Int8Tensor, output_scale: f64) -> Int8Tensor {
    assert!(tensor.is_matrix(), "Input must be 2D");
    assert!(bias.is_vector(), "Bias must be 1D");

    let (batch, features) = (tensor.shape()[0], tensor.shape()[1]);
    assert_eq!(
        features,
        bias.len(),
        "Bias length must match feature dimension"
    );

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
// Utility Functions
// ============================================================================

/// Requantizes an INT8 tensor to a new scale.
pub fn int8_requantize(tensor: &Int8Tensor, new_scale: f64) -> Int8Tensor {
    let old_scale = tensor.scale();
    let zp = tensor.zero_point() as i32;
    let scale_ratio = old_scale / new_scale;

    let data = tensor.data();
    let mut output = vec![0i8; tensor.len()];

    for i in 0..tensor.len() {
        let val = (data[i] as i32 - zp) as f64 * scale_ratio;
        output[i] = val.round().clamp(-127.0, 127.0) as i8;
    }

    let error = tensor.total_error() + new_scale / 2.0;
    Int8Tensor::new(output, tensor.shape().clone(), new_scale, 0, true).with_error(error)
}

/// Concatenates INT8 tensors along an axis.
pub fn int8_concat(tensors: &[&Int8Tensor], axis: usize, output_scale: f64) -> Int8Tensor {
    assert!(!tensors.is_empty(), "Cannot concatenate empty list");

    let first_shape = tensors[0].shape();
    for t in tensors.iter().skip(1) {
        for (i, &dim) in t.shape().iter().enumerate() {
            if i != axis {
                assert_eq!(
                    dim, first_shape[i],
                    "Shapes must match except along concat axis"
                );
            }
        }
    }

    // Calculate output shape
    let mut out_shape = first_shape.clone();
    out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();

    // Compute strides for output
    let mut strides = vec![1; out_shape.len()];
    for i in (0..out_shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }

    let outer_size: usize = first_shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = first_shape[axis + 1..].iter().product::<usize>().max(1);

    let mut output = vec![0i8; out_shape.iter().product()];

    let mut axis_offset = 0;
    for t in tensors {
        let t_scale = t.scale();
        let t_zp = t.zero_point() as i32;
        let t_axis_size = t.shape()[axis];
        let t_data = t.data();

        for outer in 0..outer_size {
            for a in 0..t_axis_size {
                for inner in 0..inner_size {
                    let src_idx = outer * t_axis_size * inner_size + a * inner_size + inner;
                    let dst_idx =
                        outer * out_shape[axis] * inner_size + (axis_offset + a) * inner_size + inner;

                    // Requantize if needed
                    let val = (t_data[src_idx] as i32 - t_zp) as f64 * t_scale;
                    let quantized = (val / output_scale).round() as i32;
                    output[dst_idx] = quantized.clamp(-127, 127) as i8;
                }
            }
        }
        axis_offset += t_axis_size;
    }

    let max_error = tensors.iter().map(|t| t.total_error()).fold(0.0_f64, f64::max);
    Int8Tensor::new(output, out_shape, output_scale, 0, true).with_error(max_error + output_scale / 2.0)
}

/// Computes the quantization error between original and quantized tensors.
pub fn compute_quantization_error(original: &BoundedTensor, quantized: &Int8Tensor) -> QuantizationErrorMetrics {
    let dq = quantized.dequantize_raw();
    let orig_vals: Vec<f64> = original.data().iter().map(|v| v.value()).collect();

    assert_eq!(dq.len(), orig_vals.len(), "Tensor sizes must match");

    let mut max_abs_error = 0.0_f64;
    let mut sum_sq_error = 0.0_f64;
    let mut sum_abs_error = 0.0_f64;
    let mut sum_relative_error = 0.0_f64;
    let mut relative_count = 0;

    for (orig, recovered) in orig_vals.iter().zip(dq.iter()) {
        let abs_error = (orig - recovered).abs();
        max_abs_error = max_abs_error.max(abs_error);
        sum_sq_error += abs_error * abs_error;
        sum_abs_error += abs_error;

        if orig.abs() > 1e-10 {
            sum_relative_error += abs_error / orig.abs();
            relative_count += 1;
        }
    }

    let n = orig_vals.len() as f64;
    QuantizationErrorMetrics {
        max_absolute_error: max_abs_error,
        mean_absolute_error: sum_abs_error / n,
        root_mean_square_error: (sum_sq_error / n).sqrt(),
        mean_relative_error: if relative_count > 0 {
            sum_relative_error / relative_count as f64
        } else {
            0.0
        },
        signal_to_noise_ratio: compute_snr(&orig_vals, &dq),
    }
}

/// Computes signal-to-noise ratio.
fn compute_snr(original: &[f64], quantized: &[f64]) -> f64 {
    let signal_power: f64 = original.iter().map(|x| x * x).sum();
    let noise_power: f64 = original
        .iter()
        .zip(quantized.iter())
        .map(|(o, q)| (o - q) * (o - q))
        .sum();

    if noise_power > 1e-10 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    }
}

/// Metrics for quantization error analysis.
#[derive(Debug, Clone)]
pub struct QuantizationErrorMetrics {
    /// Maximum absolute error.
    pub max_absolute_error: f64,
    /// Mean absolute error.
    pub mean_absolute_error: f64,
    /// Root mean square error.
    pub root_mean_square_error: f64,
    /// Mean relative error (for non-zero values).
    pub mean_relative_error: f64,
    /// Signal-to-noise ratio in dB.
    pub signal_to_noise_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schemes::QuantScheme;

    #[test]
    fn test_int8_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a = Int8Tensor::from_float_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], QuantScheme::SymmetricInt8);
        let b = Int8Tensor::from_float_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], QuantScheme::SymmetricInt8);

        // Expected output: [[22, 28], [49, 64]], max value is 64
        // Output scale must accommodate the actual output range
        let output_scale = 64.0 / 127.0; // ~0.504 to fit max value of 64
        let c = int8_matmul(&a, &b, output_scale);

        assert_eq!(c.shape(), &vec![2, 2]);
        let dq = c.dequantize_raw();
        // Expected: [[22, 28], [49, 64]]
        // Allow tolerance for quantization error (input quantization + output quantization)
        assert!((dq[0] - 22.0).abs() < 3.0, "dq[0]={}, expected ~22", dq[0]);
        assert!((dq[1] - 28.0).abs() < 3.0, "dq[1]={}, expected ~28", dq[1]);
        assert!((dq[2] - 49.0).abs() < 3.0, "dq[2]={}, expected ~49", dq[2]);
        assert!((dq[3] - 64.0).abs() < 3.0, "dq[3]={}, expected ~64", dq[3]);
    }

    #[test]
    fn test_int8_add() {
        let a = Int8Tensor::from_float_data(&[1.0, 2.0, 3.0, 4.0], vec![4], QuantScheme::SymmetricInt8);
        let b = Int8Tensor::from_float_data(&[0.5, 1.0, 1.5, 2.0], vec![4], QuantScheme::SymmetricInt8);

        let output_scale = a.scale().max(b.scale());
        let c = int8_add(&a, &b, output_scale);

        let dq = c.dequantize_raw();
        assert!((dq[0] - 1.5).abs() < 0.1);
        assert!((dq[1] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_int8_relu() {
        let a = Int8Tensor::from_float_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], QuantScheme::SymmetricInt8);
        let b = int8_relu(&a);

        let dq = b.dequantize_raw();
        assert!(dq[0] >= 0.0);
        assert!(dq[1] >= 0.0);
        assert!(dq[2] >= 0.0);
        assert!(dq[3] > 0.0);
        assert!(dq[4] > 0.0);
    }

    #[test]
    fn test_int8_gelu() {
        let a = Int8Tensor::from_float_data(&[-1.0, 0.0, 1.0], vec![3], QuantScheme::SymmetricInt8);
        let output_scale = a.scale() * 2.0;
        let b = int8_gelu(&a, output_scale);

        let dq = b.dequantize_raw();
        // GELU(-1) ≈ -0.16, GELU(0) = 0, GELU(1) ≈ 0.84
        assert!(dq[0] < 0.0);
        assert!(dq[1].abs() < 0.1);
        assert!(dq[2] > 0.0);
    }

    #[test]
    fn test_int8_layer_norm() {
        let x = Int8Tensor::from_float_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], QuantScheme::SymmetricInt8);
        let gamma = Int8Tensor::from_float_data(&[1.0, 1.0], vec![2], QuantScheme::SymmetricInt8);
        let beta = Int8Tensor::from_float_data(&[0.0, 0.0], vec![2], QuantScheme::SymmetricInt8);

        let output_scale = 0.1;
        let result = int8_layer_norm(&x, &gamma, &beta, 1e-5, output_scale);

        // Normalized values should have mean ≈ 0, variance ≈ 1
        let dq = result.dequantize_raw();
        let mean = dq.iter().sum::<f64>() / dq.len() as f64;
        assert!(mean.abs() < 0.5); // Layer norm applied per row
    }

    #[test]
    fn test_quantization_error_metrics() {
        let original = BoundedTensor::from_exact(vec![0.1, 0.5, -0.3, 0.8], vec![4]);
        let quantized = Int8Tensor::from_bounded_symmetric(&original);

        let metrics = compute_quantization_error(&original, &quantized);
        assert!(metrics.max_absolute_error < 0.02);
        assert!(metrics.mean_absolute_error < 0.01);
        assert!(metrics.signal_to_noise_ratio > 30.0); // Good SNR for INT8
    }
}
