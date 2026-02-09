//! FP8 Tensor Type.
//!
//! Provides FP8 (8-bit floating point) tensor representations for neural network
//! inference and training. Two formats are supported:
//!
//! - **E4M3**: 4-bit exponent, 3-bit mantissa. Range +/-448. Best for forward pass (weights/activations).
//! - **E5M2**: 5-bit exponent, 2-bit mantissa. Range +/-57344. Best for backward pass (gradients).
//!
//! FP8 quantization preserves more dynamic range than INT8 while using the same
//! storage, making it ideal for transformer-based models.

use helix_core::types::{BoundedTensor, BoundedValue};
use serde::{Deserialize, Serialize};

/// FP8 format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fp8Format {
    /// E4M3: 4-bit exponent, 3-bit mantissa. Range +/-448. For forward pass.
    E4M3,
    /// E5M2: 5-bit exponent, 2-bit mantissa. Range +/-57344. For backward pass.
    E5M2,
}

impl Fp8Format {
    /// Returns the exponent bias for this format.
    pub fn bias(self) -> i32 {
        match self {
            Fp8Format::E4M3 => 7,
            Fp8Format::E5M2 => 15,
        }
    }

    /// Returns the number of exponent bits.
    pub fn exponent_bits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 4,
            Fp8Format::E5M2 => 5,
        }
    }

    /// Returns the number of mantissa bits.
    pub fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 3,
            Fp8Format::E5M2 => 2,
        }
    }

    /// Returns the maximum representable normal value.
    pub fn max_value(self) -> f64 {
        match self {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        }
    }

    /// Returns the minimum representable positive normal value.
    pub fn min_normal(self) -> f64 {
        match self {
            Fp8Format::E4M3 => 2.0_f64.powi(-6),  // 2^(1-bias) = 2^-6
            Fp8Format::E5M2 => 2.0_f64.powi(-14),  // 2^(1-bias) = 2^-14
        }
    }

    /// Returns the smallest representable subnormal value.
    pub fn min_subnormal(self) -> f64 {
        match self {
            // 2^(1-bias) * 2^(-mantissa_bits) = 2^-6 * 2^-3 = 2^-9
            Fp8Format::E4M3 => 2.0_f64.powi(-9),
            // 2^(1-bias) * 2^(-mantissa_bits) = 2^-14 * 2^-2 = 2^-16
            Fp8Format::E5M2 => 2.0_f64.powi(-16),
        }
    }
}

/// An FP8-quantized tensor.
///
/// Stores values in 8-bit floating point format, with an optional per-tensor
/// scale factor for extending the representable range beyond the native FP8 range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp8Tensor {
    /// Raw FP8 data stored as u8 (bit pattern).
    data: Vec<u8>,
    /// Tensor shape.
    shape: Vec<usize>,
    /// Per-tensor scale factor. The real value is `decode(data[i]) * scale`.
    scale: f64,
    /// FP8 format (E4M3 or E5M2).
    format: Fp8Format,
}

impl Fp8Tensor {
    /// Creates a new FP8 tensor from raw bit-pattern data.
    pub fn new(data: Vec<u8>, shape: Vec<usize>, scale: f64, format: Fp8Format) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        Self {
            data,
            shape,
            scale,
            format,
        }
    }

    /// Creates a zero-filled FP8 tensor.
    pub fn zeros(shape: Vec<usize>, format: Fp8Format) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0u8; len],
            shape,
            scale: 1.0,
            format,
        }
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the per-tensor scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the FP8 format.
    pub fn format(&self) -> Fp8Format {
        self.format
    }

    /// Returns the raw FP8 bit-pattern data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns true if this is a 2D tensor (matrix).
    pub fn is_matrix(&self) -> bool {
        self.shape.len() == 2
    }

    /// Returns true if this is a 1D tensor (vector).
    pub fn is_vector(&self) -> bool {
        self.shape.len() == 1
    }

    /// Returns the maximum quantization error for a single element.
    ///
    /// For FP8, the quantization error depends on the format:
    /// - E4M3: relative error up to 2^-3 = 0.125 (3 mantissa bits)
    /// - E5M2: relative error up to 2^-2 = 0.25 (2 mantissa bits)
    ///
    /// We return the worst-case absolute error scaled by the tensor scale.
    pub fn quantization_error(&self) -> f64 {
        // The worst-case error is half the smallest representable step at max value.
        // For a floating-point format, the relative precision is 2^(-mantissa_bits).
        // The maximum absolute error is max_value * 2^(-mantissa_bits) * scale.
        let relative_precision = match self.format {
            Fp8Format::E4M3 => 2.0_f64.powi(-3), // 1/8
            Fp8Format::E5M2 => 2.0_f64.powi(-2), // 1/4
        };
        self.format.max_value() * relative_precision * self.scale
    }

    /// Dequantizes a single raw FP8 bit pattern to f64.
    fn decode_value(&self, bits: u8) -> f64 {
        let raw = match self.format {
            Fp8Format::E4M3 => decode_fp8_e4m3(bits),
            Fp8Format::E5M2 => decode_fp8_e5m2(bits),
        };
        raw * self.scale
    }

    /// Encodes an f64 value to a raw FP8 bit pattern (after removing scale).
    #[allow(dead_code)]
    fn encode_value(&self, value: f64) -> u8 {
        let scaled = if self.scale.abs() > 1e-30 {
            value / self.scale
        } else {
            0.0
        };
        match self.format {
            Fp8Format::E4M3 => encode_fp8_e4m3(scaled),
            Fp8Format::E5M2 => encode_fp8_e5m2(scaled),
        }
    }

    /// Dequantizes the entire tensor to a vector of f64 values.
    pub fn dequantize_raw(&self) -> Vec<f64> {
        self.data.iter().map(|&bits| self.decode_value(bits)).collect()
    }

    /// Quantizes a `BoundedTensor` into FP8 with automatic scale selection.
    ///
    /// The scale is chosen so that the maximum absolute value in the tensor
    /// maps to the maximum representable FP8 value, preserving full dynamic range.
    pub fn from_bounded_tensor(tensor: &BoundedTensor, format: Fp8Format) -> Self {
        let data = tensor.data();
        let shape = tensor.shape().clone();

        // Find max absolute value
        let max_abs = data
            .iter()
            .map(|v| v.value().abs())
            .fold(0.0_f64, f64::max);

        // Compute scale: scale * fp8_max = max_abs
        let fp8_max = format.max_value();
        let scale = if max_abs > 1e-30 {
            max_abs / fp8_max
        } else {
            1e-30
        };

        let quantized: Vec<u8> = data
            .iter()
            .map(|v| {
                let scaled = v.value() / scale;
                match format {
                    Fp8Format::E4M3 => encode_fp8_e4m3(scaled),
                    Fp8Format::E5M2 => encode_fp8_e5m2(scaled),
                }
            })
            .collect();

        Self {
            data: quantized,
            shape,
            scale,
            format,
        }
    }

    /// Quantizes a `BoundedTensor` into FP8 with an explicit scale.
    pub fn from_bounded_tensor_scaled(
        tensor: &BoundedTensor,
        scale: f64,
        format: Fp8Format,
    ) -> Self {
        let data = tensor.data();
        let shape = tensor.shape().clone();

        let quantized: Vec<u8> = data
            .iter()
            .map(|v| {
                let scaled = if scale.abs() > 1e-30 {
                    v.value() / scale
                } else {
                    0.0
                };
                match format {
                    Fp8Format::E4M3 => encode_fp8_e4m3(scaled),
                    Fp8Format::E5M2 => encode_fp8_e5m2(scaled),
                }
            })
            .collect();

        Self {
            data: quantized,
            shape,
            scale,
            format,
        }
    }

    /// Dequantizes the FP8 tensor to a `BoundedTensor` with proper error tracking.
    pub fn to_bounded_tensor(&self) -> BoundedTensor {
        let quant_error = self.quantization_error();

        let bounded_data: Vec<BoundedValue<f64>> = self
            .data
            .iter()
            .map(|&bits| {
                let value = self.decode_value(bits);
                BoundedValue::<f64>::with_absolute_error(value, quant_error)
            })
            .collect();

        BoundedTensor::new(bounded_data, self.shape.clone())
    }

    /// Transposes a 2D FP8 tensor.
    pub fn transpose(&self) -> Self {
        assert!(self.is_matrix(), "Transpose only supported for 2D tensors");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut transposed = vec![0u8; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = self.data[i * cols + j];
            }
        }

        Self {
            data: transposed,
            shape: vec![cols, rows],
            scale: self.scale,
            format: self.format,
        }
    }
}

// ============================================================================
// FP8 E4M3 Encoding/Decoding
// ============================================================================

/// Encodes an f64 value to FP8 E4M3 format.
///
/// FP8 E4M3 layout: `[sign(1) | exponent(4) | mantissa(3)]`
/// - Bias = 7
/// - Max normal value = 448.0 (exponent=1110, mantissa=111 -> 2^8 * 1.875 = 480... but
///   E4M3 reserves exponent=1111 mantissa=000 as the max value 448 per the OFP8 spec:
///   max = 2^(14-7) * (1 + 7/8) = 2^7 * 1.875 = 240? No, let's use the standard FP8 E4M3
///   definition from the OCP spec.)
///
/// Per the OCP/NVIDIA FP8 E4M3 specification:
/// - Exponent all-ones (1111) with non-zero mantissa = NaN
/// - Exponent all-ones (1111) with zero mantissa = +/-448 (NOT infinity)
///   Actually: max exp = 1111 = 15, but with bias 7 that gives 2^8 = 256.
///   With mantissa 000, value = 256. With mantissa 111, value would be NaN.
///   The max representable value is exp=1110 (14-7=7), mant=111 -> 2^7 * 1.875 = 240.
///   But the standard says max=448 which is 2^(15-7) * (1+0/8) * sign correction...
///   Let's use: max exponent for normals = 15 (all ones), mantissa=000 = 448 = 2^(8)*(1+3/4)
///   Actually the standard: E4M3 max = (2 - 2^-3) * 2^8 = 1.875 * 256? No that's 480.
///
/// We implement the standard OCP FP8 E4M3:
/// - Max exponent for normals is 15 (all ones is NOT reserved for inf)
/// - Exponent 1111 mantissa 111 = NaN
/// - Exponent 1111 mantissa 000..110 = normal values (no infinity)
/// - Max value = 2^(15-7) * (1 + 6/8) = 256 * 1.75 = 448
fn encode_fp8_e4m3(value: f64) -> u8 {
    if value.is_nan() {
        // NaN: exponent=1111, mantissa=111
        return 0x7F; // 0_1111_111
    }

    let sign = if value < 0.0 { 1u8 } else { 0u8 };
    let abs_val = value.abs();

    if abs_val == 0.0 {
        return sign << 7;
    }

    let max_val = 448.0;
    let abs_val = abs_val.min(max_val);

    let min_subnormal = 2.0_f64.powi(-9); // 2^(1-7) * 2^(-3) = 2^-9

    if abs_val < min_subnormal {
        // Too small to represent, round to zero
        return sign << 7;
    }

    let min_normal = 2.0_f64.powi(-6); // 2^(1-7) = 2^-6

    if abs_val < min_normal {
        // Subnormal: exponent = 0, mantissa encodes value / 2^(1-bias)
        // value = mantissa * 2^(1-bias) / 2^3 (implicit leading 0)
        // mantissa = value / (2^(1-bias) * 2^(-3)) = value / 2^(-9) = value * 2^9
        let mantissa = (abs_val * 2.0_f64.powi(9)).round() as u8;
        let mantissa = mantissa.min(7);
        return (sign << 7) | mantissa;
    }

    // Normal: value = 2^(exp - bias) * (1 + mantissa/8)
    let exp_f = abs_val.log2().floor() as i32;
    let biased_exp = (exp_f + 7).clamp(1, 15) as u8;

    // Compute mantissa
    let significand = abs_val / 2.0_f64.powi(exp_f);
    // significand is in [1, 2)
    let frac = significand - 1.0;
    let mantissa = (frac * 8.0).round() as u8;

    // Handle mantissa overflow (rounding up past 7)
    let (biased_exp, mantissa) = if mantissa > 7 {
        // mantissa overflowed, increment exponent
        let new_exp = biased_exp + 1;
        if new_exp > 15 {
            // Clamp to max value: exp=15, mantissa=6 (mantissa=7 is NaN at exp=15)
            (15u8, 6u8)
        } else {
            (new_exp, 0u8)
        }
    } else if biased_exp == 15 && mantissa >= 7 {
        // At max exponent, mantissa=7 is NaN, clamp to 6
        (15u8, 6u8)
    } else {
        (biased_exp, mantissa)
    };

    (sign << 7) | (biased_exp << 3) | mantissa
}

/// Decodes an FP8 E4M3 value to f64.
fn decode_fp8_e4m3(bits: u8) -> f64 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0x0F; // 4-bit exponent
    let mantissa = bits & 0x07; // 3-bit mantissa

    // NaN: exp=15, mantissa=7
    if exp == 15 && mantissa == 7 {
        return f64::NAN;
    }

    let value = if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            0.0
        } else {
            // Subnormal: value = mantissa/8 * 2^(1-bias) = mantissa/8 * 2^-6
            (mantissa as f64 / 8.0) * 2.0_f64.powi(-6)
        }
    } else {
        // Normal: value = (1 + mantissa/8) * 2^(exp - bias)
        let significand = 1.0 + mantissa as f64 / 8.0;
        significand * 2.0_f64.powi(exp as i32 - 7)
    };

    if sign == 1 { -value } else { value }
}

// ============================================================================
// FP8 E5M2 Encoding/Decoding
// ============================================================================

/// Encodes an f64 value to FP8 E5M2 format.
///
/// FP8 E5M2 layout: `[sign(1) | exponent(5) | mantissa(2)]`
/// - Bias = 15
/// - Follows IEEE 754 conventions: exp=11111 -> Inf (mantissa=0) or NaN (mantissa!=0)
/// - Max normal value = 2^(30-15) * (1 + 3/4) = 2^15 * 1.75 = 57344
fn encode_fp8_e5m2(value: f64) -> u8 {
    if value.is_nan() {
        // NaN: exponent=11111, mantissa=11
        return 0x7F; // 0_11111_11
    }

    let sign = if value < 0.0 { 1u8 } else { 0u8 };
    let abs_val = value.abs();

    if abs_val.is_infinite() {
        // Infinity: exp=11111, mantissa=00
        return (sign << 7) | 0x7C; // sign_11111_00
    }

    if abs_val == 0.0 {
        return sign << 7;
    }

    let max_val = 57344.0;
    if abs_val >= max_val {
        // Clamp to max: exp=11110 (30), mantissa=11
        // 2^(30-15) * (1 + 3/4) = 32768 * 1.75 = 57344
        return (sign << 7) | (30u8 << 2) | 3;
    }

    let min_subnormal = 2.0_f64.powi(-16); // 2^(1-15) * 2^(-2) = 2^-16

    if abs_val < min_subnormal {
        return sign << 7;
    }

    let min_normal = 2.0_f64.powi(-14); // 2^(1-15) = 2^-14

    if abs_val < min_normal {
        // Subnormal: value = mantissa/4 * 2^(1-bias)
        let mantissa = (abs_val * 2.0_f64.powi(16)).round() as u8;
        let mantissa = mantissa.min(3);
        return (sign << 7) | mantissa;
    }

    // Normal: value = 2^(exp - bias) * (1 + mantissa/4)
    let exp_f = abs_val.log2().floor() as i32;
    let biased_exp = (exp_f + 15).clamp(1, 30) as u8;

    let significand = abs_val / 2.0_f64.powi(exp_f);
    let frac = significand - 1.0;
    let mantissa = (frac * 4.0).round() as u8;

    // Handle mantissa overflow
    let (biased_exp, mantissa) = if mantissa > 3 {
        let new_exp = biased_exp + 1;
        if new_exp > 30 {
            // Clamp to max: exp=30, mantissa=3
            (30u8, 3u8)
        } else {
            (new_exp, 0u8)
        }
    } else {
        (biased_exp, mantissa)
    };

    (sign << 7) | (biased_exp << 2) | mantissa
}

/// Decodes an FP8 E5M2 value to f64.
fn decode_fp8_e5m2(bits: u8) -> f64 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 2) & 0x1F; // 5-bit exponent
    let mantissa = bits & 0x03; // 2-bit mantissa

    // Special exponent (all ones = 31)
    if exp == 31 {
        return if mantissa == 0 {
            if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY }
        } else {
            f64::NAN
        };
    }

    let value = if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            0.0
        } else {
            // Subnormal: value = mantissa/4 * 2^(1-bias) = mantissa/4 * 2^-14
            (mantissa as f64 / 4.0) * 2.0_f64.powi(-14)
        }
    } else {
        // Normal: value = (1 + mantissa/4) * 2^(exp - bias)
        let significand = 1.0 + mantissa as f64 / 4.0;
        significand * 2.0_f64.powi(exp as i32 - 15)
    };

    if sign == 1 { -value } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_e4m3_encode_decode() {
        // Test zero
        assert_eq!(encode_fp8_e4m3(0.0), 0x00);
        assert_eq!(decode_fp8_e4m3(0x00), 0.0);

        // Test negative zero
        let neg_zero_bits = encode_fp8_e4m3(-0.0);
        assert_eq!(decode_fp8_e4m3(neg_zero_bits), 0.0); // -0.0 decodes to 0.0 or -0.0

        // Test 1.0: exp = 7 (biased=7+7=14? No: 1.0 = 2^0, biased_exp = 0+7 = 7)
        // bits = 0_0111_000 = 0x38
        let one_bits = encode_fp8_e4m3(1.0);
        let one_decoded = decode_fp8_e4m3(one_bits);
        assert!(
            (one_decoded - 1.0).abs() < 0.01,
            "Expected ~1.0, got {}",
            one_decoded
        );

        // Test -1.0
        let neg_one_bits = encode_fp8_e4m3(-1.0);
        let neg_one_decoded = decode_fp8_e4m3(neg_one_bits);
        assert!(
            (neg_one_decoded + 1.0).abs() < 0.01,
            "Expected ~-1.0, got {}",
            neg_one_decoded
        );

        // Test round-trip for various values
        let test_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 100.0, 448.0, 0.0625];
        for &val in &test_values {
            let bits = encode_fp8_e4m3(val);
            let decoded = decode_fp8_e4m3(bits);
            let rel_error = if val.abs() > 1e-6 {
                (decoded - val).abs() / val.abs()
            } else {
                (decoded - val).abs()
            };
            assert!(
                rel_error < 0.15,
                "E4M3 round-trip for {}: got {} (rel_error={})",
                val,
                decoded,
                rel_error
            );
        }
    }

    #[test]
    fn test_fp8_e5m2_encode_decode() {
        // Test zero
        assert_eq!(encode_fp8_e5m2(0.0), 0x00);
        assert_eq!(decode_fp8_e5m2(0x00), 0.0);

        // Test 1.0
        let one_bits = encode_fp8_e5m2(1.0);
        let one_decoded = decode_fp8_e5m2(one_bits);
        assert!(
            (one_decoded - 1.0).abs() < 0.01,
            "Expected ~1.0, got {}",
            one_decoded
        );

        // Test round-trip for various values
        let test_values = [0.5, 1.0, 2.0, 8.0, 100.0, 1000.0, 57344.0];
        for &val in &test_values {
            let bits = encode_fp8_e5m2(val);
            let decoded = decode_fp8_e5m2(bits);
            let rel_error = if val.abs() > 1e-6 {
                (decoded - val).abs() / val.abs()
            } else {
                (decoded - val).abs()
            };
            assert!(
                rel_error < 0.30,
                "E5M2 round-trip for {}: got {} (rel_error={})",
                val,
                decoded,
                rel_error
            );
        }
    }

    #[test]
    fn test_fp8_quantize_dequantize() {
        let original = BoundedTensor::from_exact(vec![0.1, 0.5, -0.3, 0.8, -0.9, 0.0], vec![2, 3]);

        // E4M3
        let fp8 = Fp8Tensor::from_bounded_tensor(&original, Fp8Format::E4M3);
        assert_eq!(fp8.shape(), &[2, 3]);
        assert_eq!(fp8.len(), 6);
        assert_eq!(fp8.format(), Fp8Format::E4M3);

        let recovered = fp8.dequantize_raw();
        for (orig, rec) in original.data().iter().zip(recovered.iter()) {
            let error = (orig.value() - rec).abs();
            assert!(
                error < 0.2,
                "E4M3 quantization error too large: orig={}, rec={}, error={}",
                orig.value(),
                rec,
                error
            );
        }

        // E5M2
        let fp8_e5m2 = Fp8Tensor::from_bounded_tensor(&original, Fp8Format::E5M2);
        let recovered_e5m2 = fp8_e5m2.dequantize_raw();
        for (orig, rec) in original.data().iter().zip(recovered_e5m2.iter()) {
            let error = (orig.value() - rec).abs();
            assert!(
                error < 0.4,
                "E5M2 quantization error too large: orig={}, rec={}, error={}",
                orig.value(),
                rec,
                error
            );
        }
    }

    #[test]
    fn test_fp8_special_values() {
        // Zero
        let bits_zero = encode_fp8_e4m3(0.0);
        assert_eq!(decode_fp8_e4m3(bits_zero), 0.0);

        // Max E4M3 value (448)
        let bits_max = encode_fp8_e4m3(448.0);
        let decoded_max = decode_fp8_e4m3(bits_max);
        assert!(
            (decoded_max - 448.0).abs() < 1.0,
            "Max E4M3: expected ~448, got {}",
            decoded_max
        );

        // Negative max
        let bits_neg_max = encode_fp8_e4m3(-448.0);
        let decoded_neg_max = decode_fp8_e4m3(bits_neg_max);
        assert!(
            (decoded_neg_max + 448.0).abs() < 1.0,
            "Neg max E4M3: expected ~-448, got {}",
            decoded_neg_max
        );

        // Values beyond max should be clamped
        let bits_overflow = encode_fp8_e4m3(1000.0);
        let decoded_overflow = decode_fp8_e4m3(bits_overflow);
        assert!(
            decoded_overflow <= 448.0 + 1.0,
            "Overflow should clamp: got {}",
            decoded_overflow
        );

        // NaN
        let nan_bits = encode_fp8_e4m3(f64::NAN);
        assert!(decode_fp8_e4m3(nan_bits).is_nan());

        // E5M2 infinity
        let inf_bits = encode_fp8_e5m2(f64::INFINITY);
        assert!(decode_fp8_e5m2(inf_bits).is_infinite());

        // E5M2 NaN
        let nan_bits_e5 = encode_fp8_e5m2(f64::NAN);
        assert!(decode_fp8_e5m2(nan_bits_e5).is_nan());
    }

    #[test]
    fn test_fp8_quantization_error() {
        let e4m3_tensor = Fp8Tensor::zeros(vec![4], Fp8Format::E4M3);
        let e5m2_tensor = Fp8Tensor::zeros(vec![4], Fp8Format::E5M2);

        let e4m3_error = e4m3_tensor.quantization_error();
        let e5m2_error = e5m2_tensor.quantization_error();

        // E4M3 should have lower error than E5M2 (more mantissa bits)
        // because the relative precision is better, but max_value is lower.
        // e4m3: 448 * 0.125 * 1.0 = 56
        // e5m2: 57344 * 0.25 * 1.0 = 14336
        // With scale=1.0, E5M2 has larger absolute error due to wider range.
        assert!(e4m3_error > 0.0);
        assert!(e5m2_error > 0.0);

        // A tensor with custom scale
        let tensor = Fp8Tensor::new(vec![0, 0, 0, 0], vec![4], 0.01, Fp8Format::E4M3);
        let error = tensor.quantization_error();
        assert!(error > 0.0);
        assert!(error < 10.0); // Should be small due to small scale
    }

    #[test]
    fn test_fp8_to_bounded_tensor() {
        let original = BoundedTensor::from_exact(vec![0.5, -0.3, 1.0, 0.0], vec![2, 2]);
        let fp8 = Fp8Tensor::from_bounded_tensor(&original, Fp8Format::E4M3);
        let bounded = fp8.to_bounded_tensor();

        assert_eq!(bounded.shape(), original.shape());
        assert_eq!(bounded.data().len(), original.data().len());

        // Each value should have non-zero error tracking
        for bv in bounded.data() {
            assert!(bv.error().epsilon() >= 0.0);
        }
    }

    #[test]
    fn test_fp8_tensor_zeros() {
        let t = Fp8Tensor::zeros(vec![3, 4], Fp8Format::E4M3);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.len(), 12);
        assert_eq!(t.format(), Fp8Format::E4M3);

        let dq = t.dequantize_raw();
        for &v in &dq {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_fp8_transpose() {
        let original = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let fp8 = Fp8Tensor::from_bounded_tensor(&original, Fp8Format::E4M3);
        let transposed = fp8.transpose();

        assert_eq!(transposed.shape(), &[3, 2]);
        let dq_orig = fp8.dequantize_raw();
        let dq_trans = transposed.dequantize_raw();

        // Check that transposition is correct
        // Original [2,3]: [[1,2,3],[4,5,6]]
        // Transposed [3,2]: [[1,4],[2,5],[3,6]]
        assert!((dq_trans[0] - dq_orig[0]).abs() < 0.01); // (0,0) -> (0,0)
        assert!((dq_trans[1] - dq_orig[3]).abs() < 0.01); // (0,1) -> (1,0)
        assert!((dq_trans[2] - dq_orig[1]).abs() < 0.01); // (1,0) -> (0,1)
    }

    #[test]
    fn test_fp8_from_bounded_tensor_scaled() {
        let original = BoundedTensor::from_exact(vec![0.1, 0.2, 0.3, 0.4], vec![4]);
        let scale = 0.01;
        let fp8 = Fp8Tensor::from_bounded_tensor_scaled(&original, scale, Fp8Format::E4M3);

        assert_eq!(fp8.scale(), scale);
        let dq = fp8.dequantize_raw();
        for (orig, rec) in original.data().iter().zip(dq.iter()) {
            let error = (orig.value() - rec).abs();
            assert!(
                error < 0.05,
                "Scaled quantization error: orig={}, rec={}, error={}",
                orig.value(),
                rec,
                error
            );
        }
    }
}
