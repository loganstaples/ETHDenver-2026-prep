//! Approximate matrix multiplication with error tracking.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};
use thiserror::Error;

/// Errors from matrix operations.
#[derive(Error, Debug)]
pub enum MatMulError {
    #[error("Invalid dimensions for matmul: ({0}, {1}) x ({2}, {3})")]
    DimensionMismatch(usize, usize, usize, usize),

    #[error("Matmul requires 2D tensors, got {0}D and {1}D")]
    NotMatrix(usize, usize),
}

/// Matrix multiplication with error propagation.
///
/// For matrices A (m x k) and B (k x n), computes C = A @ B (m x n).
/// Error accumulates through the k multiplications and additions.
///
/// When the `blas-matmul` feature is enabled (default), uses ndarray for
/// hardware-accelerated matmul (Apple Accelerate on macOS, OpenBLAS on Linux).
/// Falls back to a naive O(n^3) loop otherwise.
pub fn matmul(
    a: &BoundedTensor,
    b: &BoundedTensor,
    precision: Precision,
) -> Result<BoundedTensor, MatMulError> {
    // Check dimensions
    if !a.is_matrix() || !b.is_matrix() {
        return Err(MatMulError::NotMatrix(a.ndim(), b.ndim()));
    }

    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);

    if k1 != k2 {
        return Err(MatMulError::DimensionMismatch(m, k1, k2, n));
    }

    #[cfg(feature = "blas-matmul")]
    {
        matmul_blas(a, b, precision, m, k1, n)
    }

    #[cfg(not(feature = "blas-matmul"))]
    {
        matmul_naive(a, b, precision, m, k1, n)
    }
}

/// BLAS-accelerated matmul using ndarray.
#[cfg(feature = "blas-matmul")]
fn matmul_blas(
    a: &BoundedTensor,
    b: &BoundedTensor,
    precision: Precision,
    m: usize,
    k: usize,
    n: usize,
) -> Result<BoundedTensor, MatMulError> {
    use ndarray::Array2;
    use crate::arithmetic::error_propagation::propagate_matmul;

    let precision_error = precision.max_relative_error();

    // Extract raw f64 values into flat Vec
    let a_values = a.values();
    let b_values = b.values();

    // Wrap as ndarray views and compute matmul
    let a_mat = Array2::from_shape_vec((m, k), a_values).expect("shape mismatch for a");
    let b_mat = Array2::from_shape_vec((k, n), b_values).expect("shape mismatch for b");
    let c_mat = a_mat.dot(&b_mat);

    // Compute aggregate error bound using existing propagation formula
    let a_data = a.data();
    let b_data = b.data();

    let max_a_value = a_data.iter().map(|v| v.value().abs()).fold(0.0_f64, f64::max);
    let max_a_error = a_data
        .iter()
        .map(|v| v.absolute_error())
        .fold(0.0_f64, f64::max);
    let max_b_value = b_data.iter().map(|v| v.value().abs()).fold(0.0_f64, f64::max);
    let max_b_error = b_data
        .iter()
        .map(|v| v.absolute_error())
        .fold(0.0_f64, f64::max);

    let uniform_error = propagate_matmul(k, max_a_value, max_a_error, max_b_value, max_b_error, precision_error);

    // Build result tensor with uniform error margin
    let result_data: Vec<BoundedValue<f64>> = c_mat
        .iter()
        .map(|&val| {
            let final_error = uniform_error + val.abs() * precision_error;
            BoundedValue::new(val, ErrorMargin::absolute(final_error))
        })
        .collect();

    Ok(BoundedTensor::new(result_data, vec![m, n]))
}

/// Naive O(n^3) matmul with per-element error tracking.
#[cfg(not(feature = "blas-matmul"))]
fn matmul_naive(
    a: &BoundedTensor,
    b: &BoundedTensor,
    precision: Precision,
    m: usize,
    k: usize,
    n: usize,
) -> Result<BoundedTensor, MatMulError> {
    let precision_error = precision.max_relative_error();
    let mut result_data = Vec::with_capacity(m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = BoundedValue::<f64>::exact(0.0);
            let mut accumulated_error = 0.0;

            for l in 0..k {
                let a_val = a.get(&[i, l]).unwrap();
                let b_val = b.get(&[l, j]).unwrap();

                // Multiply with error propagation
                let prod_val = a_val.value() * b_val.value();
                let prod_error = a_val.value().abs() * b_val.absolute_error()
                    + b_val.value().abs() * a_val.absolute_error()
                    + a_val.absolute_error() * b_val.absolute_error()
                    + prod_val.abs() * precision_error;

                // Add to sum
                let new_sum = sum.value() + prod_val;
                let new_error = sum.absolute_error() + prod_error;

                sum = BoundedValue::new(new_sum, ErrorMargin::absolute(new_error));
                accumulated_error = new_error;
            }

            // Add final rounding error for the sum
            let final_error = accumulated_error + sum.value().abs() * precision_error;
            result_data.push(BoundedValue::new(sum.value(), ErrorMargin::absolute(final_error)));
        }
    }

    Ok(BoundedTensor::new(result_data, vec![m, n]))
}

/// Dot product of two vectors.
pub fn dot(
    a: &BoundedTensor,
    b: &BoundedTensor,
    precision: Precision,
) -> Result<BoundedValue<f64>, MatMulError> {
    if !a.is_vector() || !b.is_vector() {
        return Err(MatMulError::NotMatrix(a.ndim(), b.ndim()));
    }

    if a.len() != b.len() {
        return Err(MatMulError::DimensionMismatch(a.len(), 1, b.len(), 1));
    }

    let precision_error = precision.max_relative_error();
    let mut sum = 0.0;
    let mut error = 0.0;

    for i in 0..a.len() {
        let a_val = a.get(&[i]).unwrap();
        let b_val = b.get(&[i]).unwrap();

        let prod = a_val.value() * b_val.value();
        let prod_err = a_val.value().abs() * b_val.absolute_error()
            + b_val.value().abs() * a_val.absolute_error()
            + prod.abs() * precision_error;

        sum += prod;
        error += prod_err;
    }

    Ok(BoundedValue::new(sum, ErrorMargin::absolute(error)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        // A = [[1, 0], [0, 1]]
        let identity = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let x = BoundedTensor::from_exact(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let result = matmul(&identity, &x, Precision::F32).unwrap();
        let values = result.values();

        assert!((values[0] - 3.0).abs() < 1e-10);
        assert!((values[1] - 4.0).abs() < 1e-10);
        assert!((values[2] - 5.0).abs() < 1e-10);
        assert!((values[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_simple() {
        // [[1, 2], [3, 4]] @ [[1], [1]] = [[3], [7]]
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = BoundedTensor::from_exact(vec![1.0, 1.0], vec![2, 1]);

        let result = matmul(&a, &b, Precision::F32).unwrap();
        let values = result.values();

        assert!((values[0] - 3.0).abs() < 1e-10);
        assert!((values[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = BoundedTensor::zeros(vec![2, 3]);
        let b = BoundedTensor::zeros(vec![4, 2]);
        assert!(matmul(&a, &b, Precision::F32).is_err());
    }

    #[test]
    fn test_matmul_larger() {
        // 4x3 @ 3x2 = 4x2
        let a_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let b_data: Vec<f64> = (1..=6).map(|x| x as f64).collect();

        let a = BoundedTensor::from_exact(a_data, vec![4, 3]);
        let b = BoundedTensor::from_exact(b_data, vec![3, 2]);

        let result = matmul(&a, &b, Precision::F32).unwrap();
        assert_eq!(result.shape(), &[4, 2]);

        // C[0,0] = 1*1 + 2*3 + 3*5 = 22
        assert!((result.values()[0] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_error_tracking() {
        // Create tensors with known error
        let a = BoundedTensor::new(
            vec![
                BoundedValue::new(1.0, ErrorMargin::absolute(0.01)),
                BoundedValue::new(2.0, ErrorMargin::absolute(0.01)),
            ],
            vec![1, 2],
        );
        let b = BoundedTensor::new(
            vec![
                BoundedValue::new(3.0, ErrorMargin::absolute(0.02)),
                BoundedValue::new(4.0, ErrorMargin::absolute(0.02)),
            ],
            vec![2, 1],
        );

        let result = matmul(&a, &b, Precision::F32).unwrap();
        // Result should be [[11.0]] with non-zero error
        assert!((result.values()[0] - 11.0).abs() < 1e-10);
        assert!(result.data()[0].absolute_error() > 0.0);
    }
}
