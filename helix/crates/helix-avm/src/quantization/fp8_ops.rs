//! FP8 Operations.
//!
//! Provides operations on FP8 tensors using the emulation pattern:
//! dequantize to f64, compute in full precision, then requantize to FP8.
//!
//! This approach is appropriate for software emulation of FP8 arithmetic.
//! On hardware with native FP8 support (e.g., NVIDIA H100), these operations
//! would be replaced with hardware-accelerated versions.

use super::fp8_tensor::{Fp8Format, Fp8Tensor};

// ============================================================================
// Matrix Operations
// ============================================================================

/// FP8 matrix multiplication: dequantize -> f64 matmul -> requantize.
///
/// Computes C = A @ B where:
/// - A has shape [M, K]
/// - B has shape [K, N]
/// - Output C has shape [M, N] in the specified output format.
pub fn fp8_matmul(a: &Fp8Tensor, b: &Fp8Tensor, output_format: Fp8Format) -> Fp8Tensor {
    assert!(a.is_matrix() && b.is_matrix(), "Inputs must be 2D matrices");
    let (m, k_a) = (a.shape()[0], a.shape()[1]);
    let (k_b, n) = (b.shape()[0], b.shape()[1]);
    assert_eq!(k_a, k_b, "Inner dimensions must match: {} vs {}", k_a, k_b);

    let k = k_a;

    // Dequantize inputs
    let a_vals = a.dequantize_raw();
    let b_vals = b.dequantize_raw();

    // Perform matmul in f64
    let mut result = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a_vals[i * k + l] * b_vals[l * n + j];
            }
            result[i * n + j] = acc;
        }
    }

    // Requantize to FP8
    let result_tensor = helix_core::types::BoundedTensor::from_exact(result, vec![m, n]);
    Fp8Tensor::from_bounded_tensor(&result_tensor, output_format)
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// FP8 element-wise addition.
///
/// Dequantizes both inputs, adds element-wise, and requantizes.
pub fn fp8_add(a: &Fp8Tensor, b: &Fp8Tensor) -> Fp8Tensor {
    assert_eq!(a.len(), b.len(), "Tensor sizes must match");
    assert_eq!(a.shape(), b.shape(), "Tensor shapes must match");

    let a_vals = a.dequantize_raw();
    let b_vals = b.dequantize_raw();

    let result: Vec<f64> = a_vals
        .iter()
        .zip(b_vals.iter())
        .map(|(av, bv)| av + bv)
        .collect();

    // Use the format of the first operand for the output
    let output_format = a.format();
    let result_tensor =
        helix_core::types::BoundedTensor::from_exact(result, a.shape().to_vec());
    Fp8Tensor::from_bounded_tensor(&result_tensor, output_format)
}

/// Scales an FP8 tensor by a scalar.
///
/// This is efficient: we can simply multiply the per-tensor scale
/// without re-encoding, but for correctness with clamping we dequantize/requantize.
pub fn fp8_scale(tensor: &Fp8Tensor, scalar: f64) -> Fp8Tensor {
    let vals = tensor.dequantize_raw();
    let result: Vec<f64> = vals.iter().map(|v| v * scalar).collect();

    let result_tensor =
        helix_core::types::BoundedTensor::from_exact(result, tensor.shape().to_vec());
    Fp8Tensor::from_bounded_tensor(&result_tensor, tensor.format())
}

// ============================================================================
// Activation Functions
// ============================================================================

/// FP8 ReLU activation.
///
/// ReLU is simple: clamp negative values to zero. Because zero is exactly
/// representable in FP8, we can operate directly on the bit patterns.
pub fn fp8_relu(tensor: &Fp8Tensor) -> Fp8Tensor {
    let vals = tensor.dequantize_raw();
    let result: Vec<f64> = vals.iter().map(|&v| v.max(0.0)).collect();

    let result_tensor =
        helix_core::types::BoundedTensor::from_exact(result, tensor.shape().to_vec());
    Fp8Tensor::from_bounded_tensor(&result_tensor, tensor.format())
}

/// FP8 GELU activation (dequantize, compute, requantize).
///
/// Uses the sigmoid approximation: GELU(x) ~ x * sigmoid(1.702 * x)
pub fn fp8_gelu(tensor: &Fp8Tensor) -> Fp8Tensor {
    let vals = tensor.dequantize_raw();
    let result: Vec<f64> = vals
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
            x * sigmoid
        })
        .collect();

    let result_tensor =
        helix_core::types::BoundedTensor::from_exact(result, tensor.shape().to_vec());
    Fp8Tensor::from_bounded_tensor(&result_tensor, tensor.format())
}

// ============================================================================
// Linear Layer
// ============================================================================

/// FP8 linear layer: y = x @ W^T + b.
///
/// - `input`: shape [batch, in_features]
/// - `weights`: shape [out_features, in_features]
/// - `bias`: optional, shape [out_features]
/// - Returns: shape [batch, out_features]
pub fn fp8_linear(
    input: &Fp8Tensor,
    weights: &Fp8Tensor,
    bias: Option<&Fp8Tensor>,
    output_format: Fp8Format,
) -> Fp8Tensor {
    assert!(input.is_matrix(), "Input must be 2D [batch, in_features]");
    assert!(weights.is_matrix(), "Weights must be 2D [out_features, in_features]");

    let in_features = input.shape()[1];
    let weight_in = weights.shape()[1];
    assert_eq!(
        in_features, weight_in,
        "Input features {} must match weight features {}",
        in_features, weight_in
    );

    // Compute x @ W^T
    let weight_t = weights.transpose();
    let mut result = fp8_matmul(input, &weight_t, output_format);

    // Add bias if provided
    if let Some(b) = bias {
        assert!(b.is_vector(), "Bias must be 1D");
        assert_eq!(
            b.len(),
            weights.shape()[0],
            "Bias length must match out_features"
        );

        let result_vals = result.dequantize_raw();
        let bias_vals = b.dequantize_raw();
        let (batch, out_features) = (result.shape()[0], result.shape()[1]);

        let mut biased = Vec::with_capacity(result_vals.len());
        for i in 0..batch {
            for j in 0..out_features {
                biased.push(result_vals[i * out_features + j] + bias_vals[j]);
            }
        }

        let biased_tensor =
            helix_core::types::BoundedTensor::from_exact(biased, result.shape().to_vec());
        result = Fp8Tensor::from_bounded_tensor(&biased_tensor, output_format);
    }

    result
}

// ============================================================================
// Format Conversion
// ============================================================================

/// Converts an FP8 tensor from one format to another (E4M3 <-> E5M2).
///
/// Dequantizes and requantizes in the target format. Useful for the common
/// pattern of using E4M3 in the forward pass and E5M2 in the backward pass.
pub fn fp8_cast(tensor: &Fp8Tensor, target_format: Fp8Format) -> Fp8Tensor {
    if tensor.format() == target_format {
        return tensor.clone();
    }

    let bounded = tensor.to_bounded_tensor();
    Fp8Tensor::from_bounded_tensor(&bounded, target_format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_core::types::BoundedTensor;

    #[test]
    fn test_fp8_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a_data = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let b_data = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        );

        let a = Fp8Tensor::from_bounded_tensor(&a_data, Fp8Format::E4M3);
        let b = Fp8Tensor::from_bounded_tensor(&b_data, Fp8Format::E4M3);

        let c = fp8_matmul(&a, &b, Fp8Format::E4M3);
        assert_eq!(c.shape(), &[2, 2]);

        let dq = c.dequantize_raw();
        // Expected: [[22, 28], [49, 64]]
        assert!(
            (dq[0] - 22.0).abs() < 5.0,
            "Expected ~22, got {}",
            dq[0]
        );
        assert!(
            (dq[1] - 28.0).abs() < 5.0,
            "Expected ~28, got {}",
            dq[1]
        );
        assert!(
            (dq[2] - 49.0).abs() < 8.0,
            "Expected ~49, got {}",
            dq[2]
        );
        assert!(
            (dq[3] - 64.0).abs() < 10.0,
            "Expected ~64, got {}",
            dq[3]
        );
    }

    #[test]
    fn test_fp8_add() {
        let a_data = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b_data = BoundedTensor::from_exact(vec![0.5, 1.0, 1.5, 2.0], vec![4]);

        let a = Fp8Tensor::from_bounded_tensor(&a_data, Fp8Format::E4M3);
        let b = Fp8Tensor::from_bounded_tensor(&b_data, Fp8Format::E4M3);

        let c = fp8_add(&a, &b);
        let dq = c.dequantize_raw();

        assert!(
            (dq[0] - 1.5).abs() < 0.5,
            "Expected ~1.5, got {}",
            dq[0]
        );
        assert!(
            (dq[1] - 3.0).abs() < 0.5,
            "Expected ~3.0, got {}",
            dq[1]
        );
        assert!(
            (dq[2] - 4.5).abs() < 0.5,
            "Expected ~4.5, got {}",
            dq[2]
        );
        assert!(
            (dq[3] - 6.0).abs() < 1.0,
            "Expected ~6.0, got {}",
            dq[3]
        );
    }

    #[test]
    fn test_fp8_relu() {
        let data = BoundedTensor::from_exact(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let tensor = Fp8Tensor::from_bounded_tensor(&data, Fp8Format::E4M3);
        let result = fp8_relu(&tensor);

        let dq = result.dequantize_raw();
        assert!(dq[0] >= -0.01, "ReLU(-2) should be ~0, got {}", dq[0]);
        assert!(dq[1] >= -0.01, "ReLU(-1) should be ~0, got {}", dq[1]);
        assert!(dq[2] >= -0.01, "ReLU(0) should be ~0, got {}", dq[2]);
        assert!(dq[3] > 0.5, "ReLU(1) should be ~1, got {}", dq[3]);
        assert!(dq[4] > 1.0, "ReLU(2) should be ~2, got {}", dq[4]);
    }

    #[test]
    fn test_fp8_gelu() {
        let data = BoundedTensor::from_exact(vec![-1.0, 0.0, 1.0], vec![3]);
        let tensor = Fp8Tensor::from_bounded_tensor(&data, Fp8Format::E4M3);
        let result = fp8_gelu(&tensor);

        let dq = result.dequantize_raw();
        // GELU(-1) ~ -0.16, GELU(0) = 0, GELU(1) ~ 0.84
        assert!(dq[0] < 0.0, "GELU(-1) should be negative, got {}", dq[0]);
        assert!(
            dq[1].abs() < 0.1,
            "GELU(0) should be ~0, got {}",
            dq[1]
        );
        assert!(dq[2] > 0.0, "GELU(1) should be positive, got {}", dq[2]);
    }

    #[test]
    fn test_fp8_linear() {
        // Input: [2, 3], Weights: [2, 3], Bias: [2]
        // Output: [2, 2]
        let input_data = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let weight_data = BoundedTensor::from_exact(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![2, 3],
        );
        let bias_data = BoundedTensor::from_exact(vec![0.1, 0.2], vec![2]);

        let input = Fp8Tensor::from_bounded_tensor(&input_data, Fp8Format::E4M3);
        let weights = Fp8Tensor::from_bounded_tensor(&weight_data, Fp8Format::E4M3);
        let bias = Fp8Tensor::from_bounded_tensor(&bias_data, Fp8Format::E4M3);

        let result = fp8_linear(&input, &weights, Some(&bias), Fp8Format::E4M3);
        assert_eq!(result.shape(), &[2, 2]);

        // Expected without bias:
        // [1*0.1+2*0.2+3*0.3, 1*0.4+2*0.5+3*0.6] = [1.4, 3.2]
        // [4*0.1+5*0.2+6*0.3, 4*0.4+5*0.5+6*0.6] = [3.2, 7.1]
        // With bias: [1.5, 3.4], [3.3, 7.3]
        let dq = result.dequantize_raw();
        assert!(
            (dq[0] - 1.5).abs() < 1.0,
            "Expected ~1.5, got {}",
            dq[0]
        );
        assert!(
            (dq[1] - 3.4).abs() < 1.5,
            "Expected ~3.4, got {}",
            dq[1]
        );
    }

    #[test]
    fn test_fp8_cast() {
        let data = BoundedTensor::from_exact(vec![1.0, -2.0, 3.0, -4.0], vec![4]);
        let e4m3 = Fp8Tensor::from_bounded_tensor(&data, Fp8Format::E4M3);

        // Cast to E5M2
        let e5m2 = fp8_cast(&e4m3, Fp8Format::E5M2);
        assert_eq!(e5m2.format(), Fp8Format::E5M2);
        assert_eq!(e5m2.shape(), e4m3.shape());

        let dq_e4m3 = e4m3.dequantize_raw();
        let dq_e5m2 = e5m2.dequantize_raw();

        // Values should be approximately the same
        for (a, b) in dq_e4m3.iter().zip(dq_e5m2.iter()) {
            assert!(
                (a - b).abs() < 1.5,
                "Cast error too large: E4M3={}, E5M2={}",
                a,
                b
            );
        }

        // Cast same format should be a no-op clone
        let same = fp8_cast(&e4m3, Fp8Format::E4M3);
        assert_eq!(same.data(), e4m3.data());
    }

    #[test]
    fn test_fp8_scale() {
        let data = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let tensor = Fp8Tensor::from_bounded_tensor(&data, Fp8Format::E4M3);

        let scaled = fp8_scale(&tensor, 2.0);
        let dq = scaled.dequantize_raw();

        assert!(
            (dq[0] - 2.0).abs() < 0.5,
            "Expected ~2.0, got {}",
            dq[0]
        );
        assert!(
            (dq[1] - 4.0).abs() < 1.0,
            "Expected ~4.0, got {}",
            dq[1]
        );
        assert!(
            (dq[2] - 6.0).abs() < 1.0,
            "Expected ~6.0, got {}",
            dq[2]
        );
        assert!(
            (dq[3] - 8.0).abs() < 1.5,
            "Expected ~8.0, got {}",
            dq[3]
        );
    }

    #[test]
    fn test_fp8_matmul_e5m2() {
        // Test with E5M2 format (more appropriate for gradients)
        let a_data = BoundedTensor::from_exact(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
        );
        let b_data = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );

        let a = Fp8Tensor::from_bounded_tensor(&a_data, Fp8Format::E5M2);
        let b = Fp8Tensor::from_bounded_tensor(&b_data, Fp8Format::E5M2);

        let c = fp8_matmul(&a, &b, Fp8Format::E5M2);
        assert_eq!(c.shape(), &[2, 2]);

        let dq = c.dequantize_raw();
        // Expected: [[10*1+20*3, 10*2+20*4], [30*1+40*3, 30*2+40*4]]
        //         = [[70, 100], [150, 220]]
        assert!(
            (dq[0] - 70.0).abs() < 20.0,
            "Expected ~70, got {}",
            dq[0]
        );
        assert!(
            (dq[1] - 100.0).abs() < 30.0,
            "Expected ~100, got {}",
            dq[1]
        );
    }
}
