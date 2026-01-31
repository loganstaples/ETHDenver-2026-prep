//! Circuit Bridge Module.
//!
//! Provides conversion between AVM error tracking (f64-based) and circuit
//! error tracking (field element-based). This enables seamless integration
//! between the high-level AVM operations and the low-level ZK circuit proofs.
//!
//! # Scaling
//!
//! Circuit field elements represent errors in fixed-point format:
//! - Scale: 2^SCALE_BITS (e.g., 2^16 = 65536)
//! - Value: error * scale
//!
//! For example, an error of 0.001 with scale 2^16 becomes 65.536 ≈ 66.

/// Number of bits for fixed-point scaling.
pub const SCALE_BITS: u32 = 16;

/// Fixed-point scale factor (2^SCALE_BITS).
pub const SCALE_FACTOR: u64 = 1 << SCALE_BITS;

/// Maximum representable error (before overflow).
pub const MAX_ERROR: f64 = (u64::MAX >> 1) as f64 / SCALE_FACTOR as f64;

/// Converts an f64 error to a scaled u64 for circuit use.
///
/// # Arguments
/// * `error` - The error value in f64
///
/// # Returns
/// A scaled u64 suitable for use as a circuit field element
pub fn error_to_field(error: f64) -> u64 {
    if error <= 0.0 {
        return 0;
    }
    if error >= MAX_ERROR {
        return u64::MAX >> 1; // Saturate
    }
    (error * SCALE_FACTOR as f64).round() as u64
}

/// Converts a scaled circuit field element back to f64 error.
pub fn field_to_error(field_value: u64) -> f64 {
    field_value as f64 / SCALE_FACTOR as f64
}

/// Converts error bounds from AVM format to circuit format.
#[derive(Debug, Clone)]
pub struct CircuitErrorBounds {
    /// Scaled error bound for layer 1 matmul.
    pub layer1_matmul_error: u64,
    /// Scaled error bound for layer 1 bias add.
    pub layer1_bias_error: u64,
    /// Scaled error bound for ReLU (typically 0).
    pub relu_error: u64,
    /// Scaled error bound for layer 2 matmul.
    pub layer2_matmul_error: u64,
    /// Scaled error bound for layer 2 bias add.
    pub layer2_bias_error: u64,
    /// Scaled error bound for loss computation.
    pub loss_error: u64,
    /// Total accumulated error.
    pub total_error: u64,
}

impl CircuitErrorBounds {
    /// Creates zero error bounds.
    pub fn zero() -> Self {
        Self {
            layer1_matmul_error: 0,
            layer1_bias_error: 0,
            relu_error: 0,
            layer2_matmul_error: 0,
            layer2_bias_error: 0,
            loss_error: 0,
            total_error: 0,
        }
    }

    /// Creates from individual f64 error components.
    pub fn from_f64(
        layer1_matmul: f64,
        layer1_bias: f64,
        relu: f64,
        layer2_matmul: f64,
        layer2_bias: f64,
        loss: f64,
    ) -> Self {
        let layer1_matmul_error = error_to_field(layer1_matmul);
        let layer1_bias_error = error_to_field(layer1_bias);
        let relu_error = error_to_field(relu);
        let layer2_matmul_error = error_to_field(layer2_matmul);
        let layer2_bias_error = error_to_field(layer2_bias);
        let loss_error = error_to_field(loss);

        let total_error = layer1_matmul_error
            .saturating_add(layer1_bias_error)
            .saturating_add(relu_error)
            .saturating_add(layer2_matmul_error)
            .saturating_add(layer2_bias_error)
            .saturating_add(loss_error);

        Self {
            layer1_matmul_error,
            layer1_bias_error,
            relu_error,
            layer2_matmul_error,
            layer2_bias_error,
            loss_error,
            total_error,
        }
    }

    /// Converts back to f64 total error.
    pub fn total_as_f64(&self) -> f64 {
        field_to_error(self.total_error)
    }
}

/// Configuration for error tracking in training.
#[derive(Debug, Clone)]
pub struct TrainingErrorConfig {
    /// Base quantization error per value.
    pub quantization_error: f64,
    /// Error per multiplication.
    pub mul_error_factor: f64,
    /// Error per addition.
    pub add_error_factor: f64,
    /// Maximum weight magnitude (for error scaling).
    pub max_weight_magnitude: f64,
    /// Maximum activation magnitude.
    pub max_activation_magnitude: f64,
}

impl Default for TrainingErrorConfig {
    fn default() -> Self {
        Self {
            quantization_error: 1.0 / SCALE_FACTOR as f64,
            mul_error_factor: 1.0,
            add_error_factor: 0.5,
            max_weight_magnitude: 10.0,
            max_activation_magnitude: 10.0,
        }
    }
}

impl TrainingErrorConfig {
    /// Estimates error bounds for a 2-layer MLP training step.
    ///
    /// # Arguments
    /// * `d_in` - Input dimension
    /// * `d_hid` - Hidden dimension
    /// * `d_out` - Output dimension
    pub fn estimate_mlp_bounds(
        &self,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> CircuitErrorBounds {
        let q = self.quantization_error;
        let w_max = self.max_weight_magnitude;
        let a_max = self.max_activation_magnitude;

        // Layer 1 matmul: d_in multiplications and additions per output
        let layer1_matmul = (d_in as f64) * (w_max * q + a_max * q + q * q);
        let layer1_bias = q;

        // ReLU has no error (just zeroes negatives)
        let relu = 0.0;

        // Layer 2 matmul: d_hid multiplications and additions per output
        let layer2_matmul = (d_hid as f64) * (w_max * q + a_max * q + q * q);
        let layer2_bias = q;

        // Loss: MSE involves d_out subtractions and squares
        let loss = (d_out as f64) * 2.0 * q;

        CircuitErrorBounds::from_f64(
            layer1_matmul,
            layer1_bias,
            relu,
            layer2_matmul,
            layer2_bias,
            loss,
        )
    }

    /// Estimates gradient error bounds for backpropagation.
    pub fn estimate_gradient_bounds(
        &self,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> CircuitErrorBounds {
        // Gradient computation typically has higher error due to chain rule
        let forward_bounds = self.estimate_mlp_bounds(d_in, d_hid, d_out);

        // Gradient error is roughly proportional to forward error
        // multiplied by number of gradient computations
        CircuitErrorBounds {
            layer1_matmul_error: forward_bounds.layer1_matmul_error * 2,
            layer1_bias_error: forward_bounds.layer1_bias_error * 2,
            relu_error: forward_bounds.relu_error,
            layer2_matmul_error: forward_bounds.layer2_matmul_error * 2,
            layer2_bias_error: forward_bounds.layer2_bias_error * 2,
            loss_error: forward_bounds.loss_error,
            total_error: forward_bounds.total_error * 2,
        }
    }
}

/// Converts quantized weights to field element representation.
///
/// Weights are stored as fixed-point integers for circuit compatibility.
pub fn weights_to_field_elements(
    weights: &[f64],
    scale: f64,
) -> Vec<i64> {
    weights
        .iter()
        .map(|&w| (w * scale).round() as i64)
        .collect()
}

/// Converts field element weights back to f64.
pub fn field_elements_to_weights(
    field_elements: &[i64],
    scale: f64,
) -> Vec<f64> {
    field_elements
        .iter()
        .map(|&f| f as f64 / scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let error = 0.001;
        let field = error_to_field(error);
        let recovered = field_to_error(field);
        assert!((error - recovered).abs() < 0.0001);
    }

    #[test]
    fn test_zero_error() {
        assert_eq!(error_to_field(0.0), 0);
        assert_eq!(error_to_field(-1.0), 0);
    }

    #[test]
    fn test_circuit_error_bounds() {
        let bounds = CircuitErrorBounds::from_f64(0.01, 0.001, 0.0, 0.02, 0.001, 0.005);
        assert!(bounds.total_error > 0);
        assert!(bounds.total_as_f64() > 0.03);
    }

    #[test]
    fn test_mlp_bounds_estimation() {
        let config = TrainingErrorConfig::default();
        let bounds = config.estimate_mlp_bounds(4, 8, 2);
        assert!(bounds.total_error > 0);
    }

    #[test]
    fn test_weights_conversion() {
        let weights = vec![0.5, -0.3, 1.2, 0.0];
        let scale = 1000.0;
        let fields = weights_to_field_elements(&weights, scale);
        let recovered = field_elements_to_weights(&fields, scale);

        for (w, r) in weights.iter().zip(recovered.iter()) {
            assert!((w - r).abs() < 0.001);
        }
    }
}
