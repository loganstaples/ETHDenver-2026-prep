//! Circuit Quantizer — maps AVM f64 values to BN254 Fr field elements.
//!
//! Uses the same `QUANTIZATION_SCALE = 1000` as `MLTrainingStepV2Circuit`
//! so that witnesses produced here are directly compatible with the circuit.
//!
//! Negative values are represented as `p - |v|` in the field, matching the
//! circuit's convention for ReLU and subtraction gates.

use helix_core::types::BoundedTensor;

/// The fixed-point scale factor matching `MLTrainingStepV2Circuit::QUANTIZATION_SCALE`.
pub const QUANTIZATION_SCALE: u64 = 1000;

/// Quantizes AVM f64 values into circuit-compatible field elements.
///
/// The quantizer scales every f64 by `QUANTIZATION_SCALE`, rounds to the
/// nearest integer, then maps the result into Fr:
///   - non-negative → `Fr::from(v)`
///   - negative     → `Fr::ZERO - Fr::from(|v|)`  (i.e. `p - |v|`)
#[derive(Debug, Clone)]
pub struct CircuitQuantizer {
    scale: u64,
}

impl Default for CircuitQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitQuantizer {
    /// Creates a quantizer with the default `QUANTIZATION_SCALE`.
    pub fn new() -> Self {
        Self {
            scale: QUANTIZATION_SCALE,
        }
    }

    /// Creates a quantizer with a custom scale.
    pub fn with_scale(scale: u64) -> Self {
        Self { scale }
    }

    /// Returns the scale factor.
    pub fn scale(&self) -> u64 {
        self.scale
    }

    /// Quantizes a single f64 value to a scaled i64.
    pub fn quantize_to_i64(&self, value: f64) -> i64 {
        (value * self.scale as f64).round() as i64
    }

    /// Quantizes a slice of f64 values to scaled i64 values.
    pub fn quantize_slice_to_i64(&self, values: &[f64]) -> Vec<i64> {
        values.iter().map(|&v| self.quantize_to_i64(v)).collect()
    }

    /// Dequantizes a scaled i64 back to f64.
    pub fn dequantize_i64(&self, scaled: i64) -> f64 {
        scaled as f64 / self.scale as f64
    }

    /// Quantizes a `BoundedTensor` and returns flat f64 values and their
    /// per-element absolute error bounds.
    pub fn extract_values_and_errors(
        &self,
        tensor: &BoundedTensor,
    ) -> (Vec<f64>, Vec<f64>) {
        let values: Vec<f64> = tensor.values();
        let errors: Vec<f64> = tensor
            .data()
            .iter()
            .map(|bv| bv.absolute_error())
            .collect();
        (values, errors)
    }

    /// Quantization error introduced per value (half of one scale unit).
    pub fn quantization_error(&self) -> f64 {
        0.5 / self.scale as f64
    }
}

/// Trait for converting to/from circuit field elements.
///
/// This is implemented conditionally when the `circuit-bridge` feature (or
/// dev-dependencies) makes `halo2curves` available.
#[cfg(any(feature = "circuit-bridge", test))]
pub mod fr_ops {
    use super::*;
    use helix_circuits::halo2curves::bn256::Fr;
    use helix_circuits::halo2curves::ff::{Field, PrimeField};

    impl CircuitQuantizer {
        /// Quantizes a single f64 → Fr.
        ///
        /// Negative values map to `p - |scaled|`.
        pub fn quantize_to_fr(&self, value: f64) -> Fr {
            let scaled = self.quantize_to_i64(value);
            i64_to_fr(scaled)
        }

        /// Quantizes a slice of f64 → Vec<Fr>.
        pub fn quantize_slice_to_fr(&self, values: &[f64]) -> Vec<Fr> {
            values.iter().map(|&v| self.quantize_to_fr(v)).collect()
        }

        /// Quantizes a `BoundedTensor`'s values → Vec<Fr>.
        pub fn quantize_tensor_values(&self, tensor: &BoundedTensor) -> Vec<Fr> {
            let values = tensor.values();
            self.quantize_slice_to_fr(&values)
        }

        /// Quantizes a `BoundedTensor`'s absolute errors → Vec<Fr>.
        ///
        /// Errors are always non-negative, so they map directly to `Fr::from(scaled)`.
        pub fn quantize_tensor_errors(&self, tensor: &BoundedTensor) -> Vec<Fr> {
            tensor
                .data()
                .iter()
                .map(|bv| {
                    let err = bv.absolute_error();
                    let scaled = (err * self.scale as f64).round().max(0.0) as u64;
                    Fr::from(scaled)
                })
                .collect()
        }

        /// Dequantizes an Fr back to f64.
        ///
        /// Detects negative field elements (those > p/2) and returns the
        /// corresponding negative f64.
        pub fn dequantize_fr(&self, fr: Fr) -> f64 {
            let scaled = fr_to_i64(fr);
            self.dequantize_i64(scaled)
        }
    }

    /// Converts a signed i64 to Fr.
    ///
    /// Negative values use `p - |v|` representation.
    pub fn i64_to_fr(value: i64) -> Fr {
        if value >= 0 {
            Fr::from(value as u64)
        } else {
            Fr::ZERO - Fr::from((-value) as u64)
        }
    }

    /// Converts an Fr back to i64.
    ///
    /// For small values used in circuit quantization, negative field elements
    /// (p - |v|) have non-zero upper bytes while small positive values have
    /// zero upper bytes. We detect negatives by checking bytes 8..32.
    pub fn fr_to_i64(fr: Fr) -> i64 {
        if fr == Fr::ZERO {
            return 0;
        }
        let repr = fr.to_repr();
        let bytes = repr.as_ref();
        // If any byte above the first u64 is non-zero, this is a large
        // field element — treat it as negative (p - |v|).
        let upper_nonzero = bytes[8..].iter().any(|&b| b != 0);
        if upper_nonzero {
            let neg = Fr::ZERO - fr;
            let neg_repr = neg.to_repr();
            let neg_bytes = neg_repr.as_ref();
            let abs_val = u64::from_le_bytes(neg_bytes[0..8].try_into().unwrap());
            -(abs_val as i64)
        } else {
            let val = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
            val as i64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_positive() {
        let q = CircuitQuantizer::new();
        assert_eq!(q.quantize_to_i64(1.5), 1500);
        assert_eq!(q.quantize_to_i64(0.001), 1);
        assert_eq!(q.quantize_to_i64(0.0), 0);
    }

    #[test]
    fn test_quantize_negative() {
        let q = CircuitQuantizer::new();
        assert_eq!(q.quantize_to_i64(-0.5), -500);
        assert_eq!(q.quantize_to_i64(-1.0), -1000);
    }

    #[test]
    fn test_roundtrip_i64() {
        let q = CircuitQuantizer::new();
        for &v in &[0.0, 1.0, -1.0, 0.123, -0.456, 3.14159] {
            let scaled = q.quantize_to_i64(v);
            let recovered = q.dequantize_i64(scaled);
            assert!(
                (v - recovered).abs() < 0.001,
                "roundtrip failed for {}: got {}",
                v,
                recovered
            );
        }
    }

    #[test]
    fn test_extract_values_and_errors() {
        let tensor = BoundedTensor::from_approximate(vec![1.0, -2.0, 3.0], vec![3], 0.01);
        let q = CircuitQuantizer::new();
        let (vals, errs) = q.extract_values_and_errors(&tensor);
        assert_eq!(vals.len(), 3);
        assert_eq!(errs.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!(errs[0] > 0.0);
    }

    #[test]
    fn test_quantization_error() {
        let q = CircuitQuantizer::new();
        assert!((q.quantization_error() - 0.0005).abs() < 1e-10);
    }

    // Fr-dependent tests
    #[cfg(any(feature = "circuit-bridge", test))]
    mod fr_tests {
        use super::super::fr_ops::*;
        use super::*;
        use helix_circuits::halo2curves::bn256::Fr;
        use helix_circuits::halo2curves::ff::Field;

        #[test]
        fn test_i64_to_fr_positive() {
            let fr = i64_to_fr(42);
            assert_eq!(fr, Fr::from(42u64));
        }

        #[test]
        fn test_i64_to_fr_negative() {
            let fr = i64_to_fr(-42);
            assert_eq!(fr, Fr::ZERO - Fr::from(42u64));
        }

        #[test]
        fn test_i64_to_fr_zero() {
            let fr = i64_to_fr(0);
            assert_eq!(fr, Fr::ZERO);
        }

        #[test]
        fn test_fr_roundtrip() {
            for v in &[0i64, 1, -1, 1000, -1000, 31415, -27182] {
                let fr = i64_to_fr(*v);
                let recovered = fr_to_i64(fr);
                assert_eq!(*v, recovered, "roundtrip failed for {}", v);
            }
        }

        #[test]
        fn test_quantize_to_fr() {
            let q = CircuitQuantizer::new();

            // Positive
            let fr = q.quantize_to_fr(1.5);
            assert_eq!(fr, Fr::from(1500u64));

            // Negative
            let fr = q.quantize_to_fr(-0.5);
            assert_eq!(fr, Fr::ZERO - Fr::from(500u64));

            // Zero
            let fr = q.quantize_to_fr(0.0);
            assert_eq!(fr, Fr::ZERO);
        }

        #[test]
        fn test_dequantize_fr() {
            let q = CircuitQuantizer::new();

            let fr = Fr::from(1500u64);
            assert!((q.dequantize_fr(fr) - 1.5).abs() < 0.001);

            let fr = Fr::ZERO - Fr::from(500u64);
            assert!((q.dequantize_fr(fr) - (-0.5)).abs() < 0.001);
        }

        #[test]
        fn test_quantize_tensor_values() {
            let tensor = BoundedTensor::from_exact(vec![1.0, -0.5, 0.0], vec![3]);
            let q = CircuitQuantizer::new();
            let frs = q.quantize_tensor_values(&tensor);
            assert_eq!(frs.len(), 3);
            assert_eq!(frs[0], Fr::from(1000u64));
            assert_eq!(frs[1], Fr::ZERO - Fr::from(500u64));
            assert_eq!(frs[2], Fr::ZERO);
        }

        #[test]
        fn test_quantize_tensor_errors() {
            let tensor = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![2], 0.01);
            let q = CircuitQuantizer::new();
            let errs = q.quantize_tensor_errors(&tensor);
            assert_eq!(errs.len(), 2);
            // 0.01 * 1000 = 10
            assert_eq!(errs[0], Fr::from(10u64));
        }
    }
}
