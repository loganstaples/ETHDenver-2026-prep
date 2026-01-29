//! Precision levels for approximate computation.

use serde::{Deserialize, Serialize};

/// Precision level for approximate computation.
/// Each level defines the bit width and maximum allowable error.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum Precision {
    /// 32-bit floating point (standard precision).
    #[default]
    F32,
    /// 16-bit floating point (half precision).
    F16,
    /// Brain floating point (16-bit, 8-bit exponent).
    BF16,
    /// 8-bit integer quantization.
    INT8,
    /// 4-bit integer quantization.
    INT4,
    /// Custom precision with specified bit width and max relative error.
    Custom { bits: u8, max_relative_error: f64 },
}

impl Precision {
    /// Returns the bit width for this precision level.
    pub fn bits(&self) -> u8 {
        match self {
            Precision::F32 => 32,
            Precision::F16 => 16,
            Precision::BF16 => 16,
            Precision::INT8 => 8,
            Precision::INT4 => 4,
            Precision::Custom { bits, .. } => *bits,
        }
    }

    /// Returns the maximum relative error for this precision level.
    /// This is the machine epsilon or quantization error.
    pub fn max_relative_error(&self) -> f64 {
        match self {
            Precision::F32 => 1.19e-7,   // 2^-23
            Precision::F16 => 9.77e-4,   // 2^-10
            Precision::BF16 => 3.91e-3,  // 2^-8 (7-bit mantissa)
            Precision::INT8 => 1.0 / 256.0,  // Uniform quantization
            Precision::INT4 => 1.0 / 16.0,   // Uniform quantization
            Precision::Custom { max_relative_error, .. } => *max_relative_error,
        }
    }

    /// Returns a human-readable name for this precision.
    pub fn name(&self) -> &'static str {
        match self {
            Precision::F32 => "fp32",
            Precision::F16 => "fp16",
            Precision::BF16 => "bf16",
            Precision::INT8 => "int8",
            Precision::INT4 => "int4",
            Precision::Custom { .. } => "custom",
        }
    }

    /// Returns the number of bytes required to store a single value.
    pub fn bytes_per_element(&self) -> usize {
        self.bits().div_ceil(8) as usize
    }

    /// Returns true if this is a floating-point precision.
    pub fn is_floating_point(&self) -> bool {
        matches!(self, Precision::F32 | Precision::F16 | Precision::BF16)
    }

    /// Returns true if this is an integer/quantized precision.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Precision::INT8 | Precision::INT4)
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_bits() {
        assert_eq!(Precision::F32.bits(), 32);
        assert_eq!(Precision::INT8.bits(), 8);
        assert_eq!(Precision::INT4.bits(), 4);
    }

    #[test]
    fn test_precision_error() {
        assert!(Precision::F32.max_relative_error() < Precision::F16.max_relative_error());
        assert!(Precision::F16.max_relative_error() < Precision::INT8.max_relative_error());
    }

    #[test]
    fn test_custom_precision() {
        let custom = Precision::Custom {
            bits: 6,
            max_relative_error: 0.05,
        };
        assert_eq!(custom.bits(), 6);
        assert_eq!(custom.max_relative_error(), 0.05);
    }
}
