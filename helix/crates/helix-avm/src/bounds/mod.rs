//! Error Bound Algebra for HELIX.
//!
//! Provides rigorous error propagation analysis for all neural network
//! operations, enabling provable bounds on computation accuracy.

pub mod attention;
pub mod clipping;
pub mod residual;
pub mod rms_norm;
pub mod softmax;
pub mod stochastic;

// Re-export key types
pub use attention::{AttentionBounds, MultiHeadAttentionError};
pub use clipping::{ClippingBounds, GradientClipError};
pub use residual::{ResidualBounds, SkipConnectionError};
pub use rms_norm::{LayerNormBounds, RMSNormBounds};
pub use softmax::{SoftmaxBounds, SoftmaxError};
pub use stochastic::{RoundingBounds, StochasticRoundingError};

use helix_core::types::BoundedTensor;

/// Accumulated error from a set of tensors produced during one training step.
///
/// Sums the maximum absolute error from each tensor in the step (forward
/// intermediates, gradients, weight updates) into a single f64 value that
/// can then be quantized to Fr for the circuit's `total_error` public input.
///
/// # Arguments
/// * `tensors` — slice of references to every `BoundedTensor` whose error
///   should contribute (e.g. h_pre, h, y, dy, dw2, dw1, …).
///
/// # Returns
/// The saturating sum of each tensor's `max_error()`.
pub fn accumulated_circuit_error(tensors: &[&BoundedTensor]) -> f64 {
    let mut total: f64 = 0.0;
    for tensor in tensors {
        total += tensor.max_error();
    }
    total
}

/// Same as [`accumulated_circuit_error`] but takes individual per-element
/// error values (useful when errors are tracked separately from values).
pub fn accumulated_circuit_error_from_f64(errors: &[f64]) -> f64 {
    errors.iter().copied().fold(0.0f64, |acc, e| acc + e.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulated_circuit_error_basic() {
        let t1 = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![2], 0.01);
        let t2 = BoundedTensor::from_approximate(vec![3.0], vec![1], 0.05);
        let total = accumulated_circuit_error(&[&t1, &t2]);
        assert!((total - 0.06).abs() < 1e-10);
    }

    #[test]
    fn test_accumulated_circuit_error_zero() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let total = accumulated_circuit_error(&[&t]);
        assert!((total - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_accumulated_from_f64() {
        let errors = vec![0.01, 0.02, 0.005];
        let total = accumulated_circuit_error_from_f64(&errors);
        assert!((total - 0.035).abs() < 1e-10);
    }
}
