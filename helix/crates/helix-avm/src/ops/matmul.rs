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

    let k = k1;
    let precision_error = precision.max_relative_error();

    // Compute result
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
}
