//! Softmax operation with numerical stability and error tracking.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};
use thiserror::Error;

/// Errors from softmax.
#[derive(Error, Debug)]
pub enum SoftmaxError {
    #[error("Empty input tensor")]
    EmptyInput,
}

/// Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
pub fn softmax(input: &BoundedTensor, precision: Precision) -> Result<BoundedTensor, SoftmaxError> {
    if input.is_empty() {
        return Err(SoftmaxError::EmptyInput);
    }

    let precision_error = precision.max_relative_error();

    // Find max for numerical stability
    let max_val = input
        .data()
        .iter()
        .map(|v| v.value())
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(x - max)
    let exp_values: Vec<f64> = input
        .data()
        .iter()
        .map(|v| (v.value() - max_val).exp())
        .collect();

    // Compute sum of exp values
    let sum_exp: f64 = exp_values.iter().sum();

    // Compute softmax with error
    let data: Vec<_> = input
        .data()
        .iter()
        .zip(&exp_values)
        .map(|(v, &exp_v)| {
            let softmax_val = exp_v / sum_exp;

            // Error for softmax is complex, use conservative estimate
            // d(softmax_i)/d(x_j) = softmax_i * (delta_ij - softmax_j)
            // For single element: max partial derivative is softmax * (1 - softmax)
            let derivative_bound = softmax_val * (1.0 - softmax_val);
            let error = derivative_bound * v.absolute_error()
                + softmax_val * precision_error
                + precision_error; // numerical stability overhead

            BoundedValue::new(softmax_val, ErrorMargin::absolute(error))
        })
        .collect();

    Ok(BoundedTensor::new(data, input.shape().clone()))
}

/// Log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max)))
pub fn log_softmax(
    input: &BoundedTensor,
    precision: Precision,
) -> Result<BoundedTensor, SoftmaxError> {
    if input.is_empty() {
        return Err(SoftmaxError::EmptyInput);
    }

    let precision_error = precision.max_relative_error();

    // Find max for numerical stability
    let max_val = input
        .data()
        .iter()
        .map(|v| v.value())
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(x - max) and sum
    let exp_values: Vec<f64> = input
        .data()
        .iter()
        .map(|v| (v.value() - max_val).exp())
        .collect();
    let sum_exp: f64 = exp_values.iter().sum();
    let log_sum_exp = sum_exp.ln();

    // Compute log-softmax
    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let log_softmax_val = v.value() - max_val - log_sum_exp;

            // Error propagation for log-softmax
            let error = v.absolute_error() + 2.0 * precision_error;

            BoundedValue::new(log_softmax_val, ErrorMargin::absolute(error))
        })
        .collect();

    Ok(BoundedTensor::new(data, input.shape().clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_uniform() {
        let input = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0], vec![3]);
        let output = softmax(&input, Precision::F32).unwrap();
        let values = output.values();

        // All equal inputs should give equal outputs
        let expected = 1.0 / 3.0;
        for v in values {
            assert!((v - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let output = softmax(&input, Precision::F32).unwrap();
        let sum: f64 = output.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without stability
        let input = BoundedTensor::from_exact(vec![1000.0, 1001.0, 1002.0], vec![3]);
        let result = softmax(&input, Precision::F32);
        assert!(result.is_ok());

        let sum: f64 = result.unwrap().values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
