//! Approximate integer quantization operations.

use helix_core::types::BoundedValue;

/// Quantization parameters for int8.
#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    /// Scale factor.
    pub scale: f64,
    /// Zero point.
    pub zero_point: i32,
}

impl QuantParams {
    /// Creates new quantization parameters.
    pub fn new(scale: f64, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }

    /// Computes quantization params from min/max values.
    pub fn from_range(min_val: f64, max_val: f64, bits: u8) -> Self {
        let qmax = (1i32 << bits) - 1;
        let scale = (max_val - min_val) / qmax as f64;
        let zero_point = (-min_val / scale).round() as i32;
        Self { scale, zero_point }
    }
}

/// Quantizes a f64 value to int8.
pub fn quantize_i8(value: f64, params: &QuantParams) -> i8 {
    let quantized = (value / params.scale).round() as i32 + params.zero_point;
    quantized.clamp(-128, 127) as i8
}

/// Dequantizes an int8 value to f64.
pub fn dequantize_i8(value: i8, params: &QuantParams) -> f64 {
    (value as i32 - params.zero_point) as f64 * params.scale
}

/// Quantizes a bounded value with error tracking.
pub fn quantize_with_error(
    value: BoundedValue<f64>,
    params: &QuantParams,
) -> (i8, BoundedValue<f64>) {
    let quantized = quantize_i8(value.value(), params);
    let dequantized = dequantize_i8(quantized, params);

    // Quantization error is at most half the scale
    let quant_error = params.scale / 2.0;
    let total_error = value.absolute_error() + quant_error;

    (quantized, BoundedValue::<f64>::with_absolute_error(dequantized, total_error))

}

/// Performs quantized addition.
pub fn quantized_add(a: i8, b: i8) -> i8 {
    let sum = a as i16 + b as i16;
    sum.clamp(-128, 127) as i8
}

/// Performs quantized multiplication with output scaling.
pub fn quantized_mul(a: i8, b: i8, output_scale: i8) -> i8 {
    let prod = a as i32 * b as i32;
    let scaled = prod / output_scale as i32;
    scaled.clamp(-128, 127) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        // Use explicit params for simpler testing: scale=0.01, zero=0
        let params = QuantParams::new(0.01, 0);
        let original = 0.5;
        let quant = quantize_i8(original, &params);
        let recovered = dequantize_i8(quant, &params);
        // With scale=0.01, 0.5 -> round(50) = 50 -> 0.5 exactly
        assert_eq!(quant, 50);
        assert!((recovered - original).abs() < 0.01);
    }
}


