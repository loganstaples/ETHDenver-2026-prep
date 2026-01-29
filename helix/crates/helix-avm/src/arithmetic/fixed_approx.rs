//! Approximate fixed-point operations (placeholder).

use helix_core::types::BoundedValue;

/// Fixed-point number representation.
#[derive(Debug, Clone, Copy)]
pub struct FixedPoint {
    /// Raw integer value.
    pub raw: i64,
    /// Number of fractional bits.
    pub frac_bits: u8,
}

impl FixedPoint {
    /// Creates a new fixed-point number.
    pub fn new(raw: i64, frac_bits: u8) -> Self {
        Self { raw, frac_bits }
    }

    /// Converts from f64.
    pub fn from_f64(value: f64, frac_bits: u8) -> Self {
        let scale = 1i64 << frac_bits;
        let raw = (value * scale as f64).round() as i64;
        Self { raw, frac_bits }
    }

    /// Converts to f64.
    pub fn to_f64(&self) -> f64 {
        let scale = 1i64 << self.frac_bits;
        self.raw as f64 / scale as f64
    }

    /// Returns the maximum representable value for given frac_bits.
    pub fn max_value(frac_bits: u8) -> f64 {
        let scale = 1i64 << frac_bits;
        (i64::MAX / scale) as f64
    }

    /// Returns the quantization error.
    pub fn quantization_error(&self) -> f64 {
        0.5 / (1i64 << self.frac_bits) as f64
    }
}

/// Converts a bounded f64 to fixed-point with error tracking.
pub fn to_fixed_with_error(
    value: BoundedValue<f64>,
    frac_bits: u8,
) -> (FixedPoint, BoundedValue<f64>) {
    let fixed = FixedPoint::from_f64(value.value(), frac_bits);
    let recovered = fixed.to_f64();
    let quant_error = fixed.quantization_error();

    let total_error = value.absolute_error() + quant_error;
    let bounded = BoundedValue::<f64>::with_absolute_error(recovered, total_error);

    (fixed, bounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        let fp = FixedPoint::from_f64(1.5, 16);
        assert!((fp.to_f64() - 1.5).abs() < 1e-4);
    }
}
