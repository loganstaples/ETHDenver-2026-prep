//! Quantization Schemes.
//!
//! Defines different quantization formats (INT8, INT4, FP8) and their parameters.

use std::fmt;

/// Represents different quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Symmetric INT8 quantization (-128 to 127)
    SymmetricInt8,
    /// Asymmetric INT8 quantization (0 to 255)
    AsymmetricInt8,
    /// Symmetric INT4 quantization (-8 to 7)
    SymmetricInt4,
    /// Asymmetric INT4 quantization (0 to 15)
    AsymmetricInt4,
    /// FP8 E4M3 format (for forward pass)
    FP8E4M3,
    /// FP8 E5M2 format (for backward pass)
    FP8E5M2,
    /// No quantization (full precision)
    None,
}

impl QuantScheme {
    /// Returns the number of bits used for this scheme.
    pub fn bits(&self) -> u8 {
        match self {
            QuantScheme::SymmetricInt8 | QuantScheme::AsymmetricInt8 => 8,
            QuantScheme::SymmetricInt4 | QuantScheme::AsymmetricInt4 => 4,
            QuantScheme::FP8E4M3 | QuantScheme::FP8E5M2 => 8,
            QuantScheme::None => 32,
        }
    }

    /// Returns whether this scheme is symmetric (zero_point = 0).
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            QuantScheme::SymmetricInt8 | QuantScheme::SymmetricInt4 | QuantScheme::FP8E4M3 | QuantScheme::FP8E5M2
        )
    }

    /// Returns the representable range for this scheme.
    pub fn range(&self) -> (i64, i64) {
        match self {
            QuantScheme::SymmetricInt8 => (-127, 127), // Symmetric uses -127 to 127
            QuantScheme::AsymmetricInt8 => (0, 255),
            QuantScheme::SymmetricInt4 => (-7, 7),
            QuantScheme::AsymmetricInt4 => (0, 15),
            QuantScheme::FP8E4M3 => (-448, 448), // Approximate max
            QuantScheme::FP8E5M2 => (-57344, 57344), // Approximate max
            QuantScheme::None => (i64::MIN, i64::MAX),
        }
    }

    /// Returns the quantization error bound for this scheme.
    /// This is the maximum error introduced by a single quantization.
    pub fn max_quant_error(&self, scale: f64) -> f64 {
        match self {
            QuantScheme::None => 0.0,
            _ => scale / 2.0, // Half of one quantum
        }
    }
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantScheme::SymmetricInt8 => write!(f, "INT8-symmetric"),
            QuantScheme::AsymmetricInt8 => write!(f, "INT8-asymmetric"),
            QuantScheme::SymmetricInt4 => write!(f, "INT4-symmetric"),
            QuantScheme::AsymmetricInt4 => write!(f, "INT4-asymmetric"),
            QuantScheme::FP8E4M3 => write!(f, "FP8-E4M3"),
            QuantScheme::FP8E5M2 => write!(f, "FP8-E5M2"),
            QuantScheme::None => write!(f, "None"),
        }
    }
}

/// Configuration for a quantization layer.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Scheme for weight quantization.
    pub weight_scheme: QuantScheme,
    /// Scheme for activation quantization.
    pub activation_scheme: QuantScheme,
    /// Whether to use per-channel quantization for weights.
    pub per_channel: bool,
    /// Whether to use dynamic quantization for activations.
    pub dynamic_activations: bool,
    /// Calibration method to use.
    pub calibration_method: CalibrationMethod,
    /// Observer type for collecting statistics.
    pub observer: ObserverType,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            weight_scheme: QuantScheme::SymmetricInt8,
            activation_scheme: QuantScheme::AsymmetricInt8,
            per_channel: true,
            dynamic_activations: false,
            calibration_method: CalibrationMethod::MinMax,
            observer: ObserverType::MinMax,
        }
    }
}

impl QuantConfig {
    /// Creates a new configuration for INT8 static quantization.
    pub fn int8_static() -> Self {
        Self::default()
    }

    /// Creates a new configuration for INT8 dynamic quantization.
    pub fn int8_dynamic() -> Self {
        Self {
            dynamic_activations: true,
            ..Self::default()
        }
    }

    /// Creates a new configuration for INT4 quantization.
    pub fn int4() -> Self {
        Self {
            weight_scheme: QuantScheme::SymmetricInt4,
            activation_scheme: QuantScheme::AsymmetricInt8, // Keep activations at INT8
            per_channel: true,
            dynamic_activations: false,
            calibration_method: CalibrationMethod::Percentile(99.99),
            observer: ObserverType::Histogram { bins: 2048 },
        }
    }

    /// Creates a new configuration for FP8 quantization.
    pub fn fp8() -> Self {
        Self {
            weight_scheme: QuantScheme::FP8E4M3,
            activation_scheme: QuantScheme::FP8E4M3,
            per_channel: false,
            dynamic_activations: true,
            calibration_method: CalibrationMethod::MinMax,
            observer: ObserverType::MinMax,
        }
    }
}

/// Calibration method for determining quantization parameters.
#[derive(Debug, Clone, Copy)]
pub enum CalibrationMethod {
    /// Use min/max observed values.
    MinMax,
    /// Use percentile values (e.g., 99.9th percentile).
    Percentile(f64),
    /// Use mean-squared error minimization.
    MSE,
    /// Use entropy calibration (KL divergence).
    Entropy,
    /// Use ACIQ (Analytical Clipping for Integer Quantization).
    ACIQ,
}

/// Observer type for collecting statistics during calibration.
#[derive(Debug, Clone, Copy)]
pub enum ObserverType {
    /// Track min and max values.
    MinMax,
    /// Track histogram of values.
    Histogram { bins: usize },
    /// Moving average observer.
    MovingAverage { averaging_constant: f64 },
}

/// Quantization parameters for a tensor.
#[derive(Debug, Clone)]
pub struct TensorQuantParams {
    /// Quantization scheme.
    pub scheme: QuantScheme,
    /// Scale factor(s) - one per channel if per_channel, otherwise one global.
    pub scales: Vec<f64>,
    /// Zero point(s) - one per channel if per_channel, otherwise one global.
    pub zero_points: Vec<i64>,
    /// Axis for per-channel quantization (if applicable).
    pub axis: Option<usize>,
}

impl TensorQuantParams {
    /// Creates new tensor quantization parameters.
    pub fn new(scheme: QuantScheme, scale: f64, zero_point: i64) -> Self {
        Self {
            scheme,
            scales: vec![scale],
            zero_points: vec![zero_point],
            axis: None,
        }
    }

    /// Creates new per-channel tensor quantization parameters.
    pub fn per_channel(
        scheme: QuantScheme,
        scales: Vec<f64>,
        zero_points: Vec<i64>,
        axis: usize,
    ) -> Self {
        assert_eq!(scales.len(), zero_points.len());
        Self {
            scheme,
            scales,
            zero_points,
            axis: Some(axis),
        }
    }

    /// Returns whether this is per-channel quantization.
    pub fn is_per_channel(&self) -> bool {
        self.axis.is_some() && self.scales.len() > 1
    }

    /// Computes quantization parameters from observed min/max values.
    pub fn from_range(scheme: QuantScheme, min_val: f64, max_val: f64) -> Self {
        let (qmin, qmax) = scheme.range();
        let qmin = qmin as f64;
        let qmax = qmax as f64;

        let (scale, zero_point) = if scheme.is_symmetric() {
            // Symmetric: zero_point = 0, scale based on max absolute value
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / qmax;
            (scale, 0)
        } else {
            // Asymmetric: use full range
            let scale = (max_val - min_val) / (qmax - qmin);
            let zero_point = (qmin - min_val / scale).round() as i64;
            (scale, zero_point)
        };

        Self::new(scheme, scale.max(1e-10), zero_point)
    }

    /// Returns the quantization error bound.
    pub fn quantization_error(&self) -> f64 {
        if self.scales.is_empty() {
            return 0.0;
        }
        // Maximum error is half a quantum at the largest scale
        self.scales.iter().cloned().fold(0.0_f64, f64::max) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheme_properties() {
        assert_eq!(QuantScheme::SymmetricInt8.bits(), 8);
        assert_eq!(QuantScheme::SymmetricInt4.bits(), 4);
        assert!(QuantScheme::SymmetricInt8.is_symmetric());
        assert!(!QuantScheme::AsymmetricInt8.is_symmetric());
    }

    #[test]
    fn test_params_from_range() {
        // Symmetric INT8
        let params = TensorQuantParams::from_range(QuantScheme::SymmetricInt8, -1.0, 1.0);
        assert_eq!(params.zero_points[0], 0);
        assert!((params.scales[0] - 1.0 / 127.0).abs() < 1e-10);

        // Asymmetric INT8
        let params = TensorQuantParams::from_range(QuantScheme::AsymmetricInt8, 0.0, 1.0);
        assert!((params.scales[0] - 1.0 / 255.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_presets() {
        let static_config = QuantConfig::int8_static();
        assert!(!static_config.dynamic_activations);

        let dynamic_config = QuantConfig::int8_dynamic();
        assert!(dynamic_config.dynamic_activations);

        let int4_config = QuantConfig::int4();
        assert_eq!(int4_config.weight_scheme, QuantScheme::SymmetricInt4);
    }
}
