//! Normalization operations.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};

/// Layer normalization.
/// Normalizes across features: (x - mean) / sqrt(var + eps)
pub fn layer_norm(
    input: &BoundedTensor,
    eps: f64,
    precision: Precision,
) -> BoundedTensor {
    if input.is_empty() {
        return input.clone();
    }

    let precision_error = precision.max_relative_error();
    let n = input.len() as f64;

    // Compute mean
    let mean: f64 = input.data().iter().map(|v| v.value()).sum::<f64>() / n;

    // Compute variance
    let variance: f64 = input
        .data()
        .iter()
        .map(|v| (v.value() - mean).powi(2))
        .sum::<f64>()
        / n;

    let std = (variance + eps).sqrt();

    // Normalize
    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let normalized = (v.value() - mean) / std;

            // Error propagation for normalization
            let error = v.absolute_error() / std
                + normalized.abs() * precision_error;

            BoundedValue::new(normalized, ErrorMargin::absolute(error))
        })
        .collect();

    BoundedTensor::new(data, input.shape().clone())
}

/// RMS normalization (Root Mean Square Layer Normalization).
/// Used in LLaMA and other modern architectures.
pub fn rms_norm(
    input: &BoundedTensor,
    eps: f64,
    precision: Precision,
) -> BoundedTensor {
    if input.is_empty() {
        return input.clone();
    }

    let precision_error = precision.max_relative_error();
    let n = input.len() as f64;

    // Compute RMS
    let sum_squares: f64 = input.data().iter().map(|v| v.value().powi(2)).sum();
    let rms = (sum_squares / n + eps).sqrt();

    // Normalize
    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let normalized = v.value() / rms;
            let error = v.absolute_error() / rms + normalized.abs() * precision_error;
            BoundedValue::new(normalized, ErrorMargin::absolute(error))
        })
        .collect();

    BoundedTensor::new(data, input.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let output = layer_norm(&input, 1e-5, Precision::F32);

        // Mean should be 0 after normalization (approximately)
        let mean: f64 = output.values().iter().sum::<f64>() / 5.0;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_rms_norm() {
        let input = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let output = rms_norm(&input, 1e-5, Precision::F32);

        // All values should be ~1 for constant input
        for v in output.values() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }
}
