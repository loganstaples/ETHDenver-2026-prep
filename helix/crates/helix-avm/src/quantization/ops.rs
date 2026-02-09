//! Quantized Operations.
//!
//! Provides quantized versions of common neural network operations
//! that operate directly on integer representations.

use super::quantize::QuantizedTensor;
use super::schemes::TensorQuantParams;

/// Performs quantized matrix multiplication.
///
/// Computes C = A @ B where all inputs are quantized.
/// Output is stored in INT32 accumulator and then requantized.
pub fn quantized_matmul(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
    output_params: &TensorQuantParams,
) -> QuantizedTensor {
    let a_shape = &a.shape;
    let b_shape = &b.shape;
    
    assert!(!a_shape.is_empty() && !b_shape.is_empty());
    
    let m = if a_shape.len() >= 2 { a_shape[0] } else { 1 };
    let k = if a_shape.len() >= 2 { a_shape[1] } else { a_shape[0] };
    let n = if b_shape.len() >= 2 { b_shape[1] } else { 1 };
    
    // Validate dimensions
    let b_k = if b_shape.len() >= 2 { b_shape[0] } else { b_shape[0] };
    assert_eq!(k, b_k, "Matrix dimensions must align");
    
    // Get quantization parameters
    let a_scale = a.params.scales[0];
    let a_zp = a.params.zero_points[0];
    let b_scale = b.params.scales[0];
    let b_zp = b.params.zero_points[0];
    let out_scale = output_params.scales[0];
    let out_zp = output_params.zero_points[0];
    
    // Combined scale for requantization
    let combined_scale = (a_scale * b_scale) / out_scale;
    
    let mut output = vec![0i64; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut accumulator: i64 = 0;
            
            for l in 0..k {
                let a_val = a.data[i * k + l] - a_zp;
                let b_val = b.data[l * n + j] - b_zp;
                accumulator += a_val * b_val;
            }
            
            // Requantize: scale accumulator and add output zero point
            let scaled = (accumulator as f64 * combined_scale).round() as i64;
            let (qmin, qmax) = output_params.scheme.range();
            output[i * n + j] = (scaled + out_zp).clamp(qmin, qmax);
        }
    }
    
    QuantizedTensor::new(
        output,
        vec![m, n],
        output_params.clone(),
    )
}

/// Performs quantized element-wise addition.
///
/// For tensors with different scales, handles rescaling.
pub fn quantized_add(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
    output_params: &TensorQuantParams,
) -> QuantizedTensor {
    assert_eq!(a.data.len(), b.data.len(), "Tensor sizes must match");
    
    let a_scale = a.params.scales[0];
    let a_zp = a.params.zero_points[0];
    let b_scale = b.params.scales[0];
    let b_zp = b.params.zero_points[0];
    let out_scale = output_params.scales[0];
    let out_zp = output_params.zero_points[0];
    
    let (qmin, qmax) = output_params.scheme.range();
    
    let output: Vec<i64> = a.data.iter().zip(&b.data)
        .map(|(&a_val, &b_val)| {
            // Dequantize both inputs
            let a_real = (a_val - a_zp) as f64 * a_scale;
            let b_real = (b_val - b_zp) as f64 * b_scale;
            
            // Add in float domain
            let sum = a_real + b_real;
            
            // Requantize
            let quantized = (sum / out_scale).round() as i64 + out_zp;
            quantized.clamp(qmin, qmax)
        })
        .collect();
    
    QuantizedTensor::new(output, a.shape.clone(), output_params.clone())
}

/// Performs quantized ReLU.
///
/// Sets negative values to zero (actually to zero_point).
pub fn quantized_relu(tensor: &QuantizedTensor) -> QuantizedTensor {
    let zp = tensor.params.zero_points[0];
    let (qmin, qmax) = tensor.params.scheme.range();
    
    // ReLU: max(x, 0) which in quantized space is max(x, zero_point)
    let output: Vec<i64> = tensor.data.iter()
        .map(|&v| {
            if v < zp {
                zp.clamp(qmin, qmax)
            } else {
                v
            }
        })
        .collect();
    
    QuantizedTensor::new(output, tensor.shape.clone(), tensor.params.clone())
}

/// Performs quantized sigmoid approximation.
///
/// Uses piecewise linear approximation for efficiency.
pub fn quantized_sigmoid(
    tensor: &QuantizedTensor,
    output_params: &TensorQuantParams,
) -> QuantizedTensor {
    let scale = tensor.params.scales[0];
    let zp = tensor.params.zero_points[0];
    let out_scale = output_params.scales[0];
    let out_zp = output_params.zero_points[0];
    let (qmin, qmax) = output_params.scheme.range();
    
    let output: Vec<i64> = tensor.data.iter()
        .map(|&v| {
            // Dequantize
            let x = (v - zp) as f64 * scale;
            
            // Piecewise linear approximation of sigmoid
            let y = if x <= -4.0 {
                0.0
            } else if x >= 4.0 {
                1.0
            } else {
                // Linear approximation: 0.5 + 0.125 * x for |x| < 4
                0.5 + 0.125 * x
            };
            
            // Requantize
            let quantized = (y / out_scale).round() as i64 + out_zp;
            quantized.clamp(qmin, qmax)
        })
        .collect();
    
    QuantizedTensor::new(output, tensor.shape.clone(), output_params.clone())
}

/// Performs quantized GELU approximation.
pub fn quantized_gelu(
    tensor: &QuantizedTensor,
    output_params: &TensorQuantParams,
) -> QuantizedTensor {
    let scale = tensor.params.scales[0];
    let zp = tensor.params.zero_points[0];
    let out_scale = output_params.scales[0];
    let out_zp = output_params.zero_points[0];
    let (qmin, qmax) = output_params.scheme.range();
    
    let output: Vec<i64> = tensor.data.iter()
        .map(|&v| {
            // Dequantize
            let x = (v - zp) as f64 * scale;
            
            // Fast GELU approximation: x * sigmoid(1.702 * x)
            let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
            let y = x * sigmoid;
            
            // Requantize
            let quantized = (y / out_scale).round() as i64 + out_zp;
            quantized.clamp(qmin, qmax)
        })
        .collect();
    
    QuantizedTensor::new(output, tensor.shape.clone(), output_params.clone())
}

/// Performs quantized layer normalization.
///
/// Computes mean and variance in floating point, then requantizes.
pub fn quantized_layer_norm(
    tensor: &QuantizedTensor,
    gamma: &QuantizedTensor,
    beta: &QuantizedTensor,
    output_params: &TensorQuantParams,
    eps: f64,
) -> QuantizedTensor {
    let shape = &tensor.shape;
    let data = &tensor.data;
    
    // Assume last dimension is the normalization dimension
    let norm_size = shape.last().copied().unwrap_or(1);
    let num_groups = data.len() / norm_size;
    
    let scale = tensor.params.scales[0];
    let zp = tensor.params.zero_points[0];
    let out_scale = output_params.scales[0];
    let out_zp = output_params.zero_points[0];
    let (qmin, qmax) = output_params.scheme.range();
    
    let gamma_scale = gamma.params.scales[0];
    let gamma_zp = gamma.params.zero_points[0];
    let beta_scale = beta.params.scales[0];
    let beta_zp = beta.params.zero_points[0];
    
    let mut output = Vec::with_capacity(data.len());
    
    for g in 0..num_groups {
        let start = g * norm_size;
        let end = start + norm_size;
        let group_data = &data[start..end];
        
        // Compute mean and variance in float
        let mut sum = 0.0;
        for &v in group_data {
            sum += (v - zp) as f64 * scale;
        }
        let mean = sum / norm_size as f64;
        
        let mut var_sum = 0.0;
        for &v in group_data {
            let x = (v - zp) as f64 * scale;
            var_sum += (x - mean) * (x - mean);
        }
        let variance = var_sum / norm_size as f64;
        let std_inv = 1.0 / (variance + eps).sqrt();
        
        // Normalize and apply gamma/beta
        for (i, &v) in group_data.iter().enumerate() {
            let x = (v - zp) as f64 * scale;
            let normalized = (x - mean) * std_inv;
            
            let g_val = (gamma.data[i % norm_size] - gamma_zp) as f64 * gamma_scale;
            let b_val = (beta.data[i % norm_size] - beta_zp) as f64 * beta_scale;
            
            let y = normalized * g_val + b_val;
            
            // Requantize
            let quantized = (y / out_scale).round() as i64 + out_zp;
            output.push(quantized.clamp(qmin, qmax));
        }
    }
    
    QuantizedTensor::new(output, shape.clone(), output_params.clone())
}

/// Performs quantized softmax.
///
/// Uses fixed-point arithmetic where possible.
pub fn quantized_softmax(
    tensor: &QuantizedTensor,
    output_params: &TensorQuantParams,
    axis: usize,
) -> QuantizedTensor {
    let shape = &tensor.shape;
    let data = &tensor.data;
    
    let scale = tensor.params.scales[0];
    let zp = tensor.params.zero_points[0];
    let out_scale = output_params.scales[0];
    let out_zp = output_params.zero_points[0];
    let (qmin, qmax) = output_params.scheme.range();
    
    // Compute softmax dimension sizes
    let softmax_size = shape.get(axis).copied().unwrap_or(1);
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = shape.get(axis + 1..).map(|s| s.iter().product()).unwrap_or(1);
    
    let mut output = vec![0i64; data.len()];
    
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let x = (data[idx] - zp) as f64 * scale;
                max_val = max_val.max(x);
            }
            
            // Compute exp(x - max) and sum
            let mut exp_sum = 0.0;
            let mut exps = Vec::with_capacity(softmax_size);
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let x = (data[idx] - zp) as f64 * scale;
                let e = (x - max_val).exp();
                exps.push(e);
                exp_sum += e;
            }
            
            // Normalize and quantize
            for s in 0..softmax_size {
                let idx = outer * softmax_size * inner_size + s * inner_size + inner;
                let prob = exps[s] / exp_sum;
                let quantized = (prob / out_scale).round() as i64 + out_zp;
                output[idx] = quantized.clamp(qmin, qmax);
            }
        }
    }
    
    QuantizedTensor::new(output, shape.clone(), output_params.clone())
}

/// Quantized linear layer (matmul + bias).
pub fn quantized_linear(
    input: &QuantizedTensor,
    weight: &QuantizedTensor,
    bias: Option<&QuantizedTensor>,
    output_params: &TensorQuantParams,
) -> QuantizedTensor {
    // Compute input @ weight^T
    let mut result = quantized_matmul(input, weight, output_params);
    
    // Add bias if present
    if let Some(b) = bias {
        result = quantized_add(&result, b, output_params);
    }
    
    result
}

/// Computes quantization error for an operation.
pub fn compute_op_error(
    input_params: &TensorQuantParams,
    weight_params: &TensorQuantParams,
    output_params: &TensorQuantParams,
    k: usize, // accumulation dimension
) -> f64 {
    let input_error = input_params.quantization_error();
    let weight_error = weight_params.quantization_error();
    let output_error = output_params.quantization_error();
    
    // Error propagation through matmul
    let matmul_error = k as f64 * (input_error * weight_error);
    
    matmul_error + output_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schemes::QuantScheme;

    fn create_quantized_tensor(data: Vec<i64>, shape: Vec<usize>) -> QuantizedTensor {
        let params = TensorQuantParams::new(QuantScheme::SymmetricInt8, 0.01, 0);
        QuantizedTensor::new(data, shape, params)
    }

    #[test]
    fn test_quantized_relu() {
        let tensor = create_quantized_tensor(vec![-10, -5, 0, 5, 10], vec![5]);
        let result = quantized_relu(&tensor);
        
        assert_eq!(result.data, vec![0, 0, 0, 5, 10]);
    }

    #[test]
    fn test_quantized_add() {
        let a = create_quantized_tensor(vec![10, 20, 30], vec![3]);
        let b = create_quantized_tensor(vec![5, 10, 15], vec![3]);
        
        let output_params = TensorQuantParams::new(QuantScheme::SymmetricInt8, 0.01, 0);
        let result = quantized_add(&a, &b, &output_params);
        
        assert_eq!(result.data, vec![15, 30, 45]);
    }

    #[test]
    fn test_quantized_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a = create_quantized_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3],
        );
        let b = create_quantized_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![3, 2],
        );
        
        let output_params = TensorQuantParams::new(QuantScheme::SymmetricInt8, 0.01, 0);
        let result = quantized_matmul(&a, &b, &output_params);
        
        assert_eq!(result.dims(), &[2, 2]);
        // Result should be [[22, 28], [49, 64]] (before scaling)
    }
}
