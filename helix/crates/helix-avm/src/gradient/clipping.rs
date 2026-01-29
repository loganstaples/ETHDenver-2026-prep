//! Gradient clipping utilities.
//!
//! Provides gradient clipping to prevent exploding gradients during training.

use std::collections::HashMap;
use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin};

use super::autodiff::NodeIndex;

/// Clips gradients by value.
///
/// Each gradient element is clipped to [-clip_value, clip_value].
pub fn clip_grad_value(
    grads: &mut HashMap<NodeIndex, BoundedTensor>,
    clip_value: f64,
) {
    for grad in grads.values_mut() {
        let clipped_data: Vec<BoundedValue<f64>> = grad
            .data()
            .iter()
            .map(|v| {
                let clipped = v.value().clamp(-clip_value, clip_value);
                // If clipped, error becomes 0 at the boundary (we know exact value)
                // If not clipped, preserve error
                if clipped != v.value() {
                    BoundedValue::exact(clipped)
                } else {
                    *v
                }
            })
            .collect();
        *grad = BoundedTensor::new(clipped_data, grad.shape().clone());
    }
}

/// Clips gradients by global L2 norm.
///
/// If the total L2 norm of all gradients exceeds max_norm,
/// gradients are scaled down proportionally.
///
/// Returns the original global norm.
pub fn clip_grad_norm(
    grads: &mut HashMap<NodeIndex, BoundedTensor>,
    max_norm: f64,
) -> f64 {
    // Compute global norm
    let mut total_norm_sq = 0.0;
    for grad in grads.values() {
        for v in grad.data() {
            total_norm_sq += v.value() * v.value();
        }
    }
    let total_norm = total_norm_sq.sqrt();

    // Clip if necessary
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-10);
        for grad in grads.values_mut() {
            let scaled_data: Vec<BoundedValue<f64>> = grad
                .data()
                .iter()
                .map(|v| {
                    let scaled = v.value() * scale;
                    // Error also scales
                    let error = v.absolute_error() * scale;
                    BoundedValue::new(scaled, ErrorMargin::absolute(error))
                })
                .collect();
            *grad = BoundedTensor::new(scaled_data, grad.shape().clone());
        }
    }

    total_norm
}

/// Computes the global L2 norm of gradients.
pub fn grad_norm(grads: &HashMap<NodeIndex, BoundedTensor>) -> f64 {
    let mut total_norm_sq = 0.0;
    for grad in grads.values() {
        for v in grad.data() {
            total_norm_sq += v.value() * v.value();
        }
    }
    total_norm_sq.sqrt()
}

/// Computes the maximum absolute gradient value.
pub fn grad_max_abs(grads: &HashMap<NodeIndex, BoundedTensor>) -> f64 {
    let mut max_abs: f64 = 0.0;
    for grad in grads.values() {
        for v in grad.data() {
            max_abs = max_abs.max(v.value().abs());
        }
    }
    max_abs
}

/// Configuration for gradient clipping.
#[derive(Debug, Clone)]
pub enum GradientClipConfig {
    /// No clipping.
    None,
    /// Clip by value: [-value, value].
    Value(f64),
    /// Clip by global L2 norm.
    Norm(f64),
}

impl GradientClipConfig {
    /// Applies the configured clipping to gradients.
    pub fn apply(&self, grads: &mut HashMap<NodeIndex, BoundedTensor>) -> f64 {
        match self {
            GradientClipConfig::None => 0.0,
            GradientClipConfig::Value(v) => {
                clip_grad_value(grads, *v);
                0.0
            }
            GradientClipConfig::Norm(max_norm) => {
                clip_grad_norm(grads, *max_norm)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_grad_value() {
        let mut grads: HashMap<NodeIndex, BoundedTensor> = HashMap::new();
        grads.insert(
            0,
            BoundedTensor::from_exact(vec![-2.0, 0.5, 1.5], vec![3]),
        );

        clip_grad_value(&mut grads, 1.0);

        let values = grads.get(&0).unwrap().values();
        assert_eq!(values, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn test_clip_grad_norm() {
        let mut grads: HashMap<NodeIndex, BoundedTensor> = HashMap::new();
        grads.insert(
            0,
            BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]),
        );

        // Norm is 5, max_norm is 5, should not clip
        let norm = clip_grad_norm(&mut grads, 5.0);
        assert!((norm - 5.0).abs() < 1e-10);

        // Norm is 5, max_norm is 2.5, should clip to half
        grads.insert(
            0,
            BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]),
        );
        let norm = clip_grad_norm(&mut grads, 2.5);
        assert!((norm - 5.0).abs() < 1e-10);

        let values = grads.get(&0).unwrap().values();
        assert!((values[0] - 1.5).abs() < 1e-10);
        assert!((values[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_norm() {
        let mut grads: HashMap<NodeIndex, BoundedTensor> = HashMap::new();
        grads.insert(
            0,
            BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]),
        );

        let norm = grad_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-10);
    }
}
