//! Model Output Validation with PyTorch-compatible reference testing.
//!
//! This module provides comprehensive validation infrastructure for neural network
//! computations in HELIX AVM. It validates:
//!
//! - Forward pass correctness against known reference outputs
//! - Layer-by-layer output validation
//! - Numerical stability of attention/softmax operations
//! - Error bound accuracy
//!
//! # Design Philosophy
//!
//! All validation uses analytically computed expected values or pre-computed
//! PyTorch reference vectors. This ensures the AVM computes mathematically
//! correct results within specified error tolerances.

use helix_core::types::{BoundedTensor, Precision};
use std::collections::HashMap;
use thiserror::Error;

use crate::ops;

/// Errors during validation.
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Value mismatch at index {index}: expected {expected}, got {actual} (diff: {diff})")]
    ValueMismatch {
        index: usize,
        expected: f64,
        actual: f64,
        diff: f64,
    },

    #[error("Error bound exceeded: computed error {computed} exceeds tolerance {tolerance}")]
    ErrorBoundExceeded { computed: f64, tolerance: f64 },

    #[error("Numerical instability detected: {message}")]
    NumericalInstability { message: String },

    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/// Result of a validation check.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub passed: bool,
    /// Maximum absolute error observed.
    pub max_absolute_error: f64,
    /// Mean absolute error.
    pub mean_absolute_error: f64,
    /// Maximum relative error (for non-zero values).
    pub max_relative_error: f64,
    /// Number of elements compared.
    pub num_elements: usize,
    /// Detailed per-element errors (optional).
    pub element_errors: Option<Vec<f64>>,
}

impl ValidationResult {
    /// Creates a passing result with metrics.
    pub fn pass(
        max_abs_err: f64,
        mean_abs_err: f64,
        max_rel_err: f64,
        num_elements: usize,
    ) -> Self {
        Self {
            passed: true,
            max_absolute_error: max_abs_err,
            mean_absolute_error: mean_abs_err,
            max_relative_error: max_rel_err,
            num_elements,
            element_errors: None,
        }
    }

    /// Creates a failing result.
    pub fn fail(
        max_abs_err: f64,
        mean_abs_err: f64,
        max_rel_err: f64,
        num_elements: usize,
        element_errors: Vec<f64>,
    ) -> Self {
        Self {
            passed: false,
            max_absolute_error: max_abs_err,
            mean_absolute_error: mean_abs_err,
            max_relative_error: max_rel_err,
            num_elements,
            element_errors: Some(element_errors),
        }
    }
}

/// Configuration for validation tolerances.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Absolute tolerance for value comparison.
    pub absolute_tolerance: f64,
    /// Relative tolerance for value comparison.
    pub relative_tolerance: f64,
    /// Tolerance for softmax sum-to-one check.
    pub softmax_sum_tolerance: f64,
    /// Tolerance for attention numerical stability.
    pub attention_stability_tolerance: f64,
    /// Whether to collect per-element errors.
    pub collect_element_errors: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            softmax_sum_tolerance: 1e-6,
            attention_stability_tolerance: 1e-5,
            collect_element_errors: false,
        }
    }
}

impl ValidationConfig {
    /// Creates config for FP32 precision.
    pub fn fp32() -> Self {
        Self {
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            softmax_sum_tolerance: 1e-6,
            attention_stability_tolerance: 1e-5,
            collect_element_errors: false,
        }
    }

    /// Creates config for INT8 quantized comparisons.
    pub fn int8() -> Self {
        Self {
            absolute_tolerance: 0.05, // 5% of scale
            relative_tolerance: 0.1,  // 10% relative
            softmax_sum_tolerance: 0.01,
            attention_stability_tolerance: 0.01,
            collect_element_errors: true,
        }
    }

    /// Creates strict config for reference comparisons.
    pub fn strict() -> Self {
        Self {
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-9,
            softmax_sum_tolerance: 1e-10,
            attention_stability_tolerance: 1e-10,
            collect_element_errors: true,
        }
    }
}

/// Output validator for neural network computations.
pub struct OutputValidator {
    config: ValidationConfig,
}

impl OutputValidator {
    /// Creates a new validator with default config.
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Creates a validator with custom config.
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validates that actual tensor matches expected values.
    pub fn validate_tensor(
        &self,
        actual: &BoundedTensor,
        expected: &[f64],
        expected_shape: &[usize],
    ) -> Result<ValidationResult, ValidationError> {
        // Check shape
        if actual.shape().as_slice() != expected_shape {
            return Err(ValidationError::ShapeMismatch {
                expected: expected_shape.to_vec(),
                actual: actual.shape().clone(),
            });
        }

        // Check values
        let actual_values = actual.values();
        if actual_values.len() != expected.len() {
            return Err(ValidationError::ShapeMismatch {
                expected: vec![expected.len()],
                actual: vec![actual_values.len()],
            });
        }

        let mut max_abs_err = 0.0_f64;
        let mut sum_abs_err = 0.0_f64;
        let mut max_rel_err = 0.0_f64;
        let mut element_errors = Vec::new();

        for (i, (&act, &exp)) in actual_values.iter().zip(expected.iter()).enumerate() {
            let abs_err = (act - exp).abs();
            max_abs_err = max_abs_err.max(abs_err);
            sum_abs_err += abs_err;

            if self.config.collect_element_errors {
                element_errors.push(abs_err);
            }

            if exp.abs() > 1e-10 {
                let rel_err = abs_err / exp.abs();
                max_rel_err = max_rel_err.max(rel_err);
            }

            // Check tolerance
            let tolerance = self.config.absolute_tolerance
                + self.config.relative_tolerance * exp.abs();
            if abs_err > tolerance {
                if self.config.collect_element_errors {
                    return Ok(ValidationResult::fail(
                        max_abs_err,
                        sum_abs_err / expected.len() as f64,
                        max_rel_err,
                        expected.len(),
                        element_errors,
                    ));
                } else {
                    return Err(ValidationError::ValueMismatch {
                        index: i,
                        expected: exp,
                        actual: act,
                        diff: abs_err,
                    });
                }
            }
        }

        Ok(ValidationResult::pass(
            max_abs_err,
            sum_abs_err / expected.len() as f64,
            max_rel_err,
            expected.len(),
        ))
    }

    /// Validates that two tensors are approximately equal.
    pub fn validate_tensors_equal(
        &self,
        actual: &BoundedTensor,
        expected: &BoundedTensor,
    ) -> Result<ValidationResult, ValidationError> {
        let expected_values = expected.values();
        self.validate_tensor(actual, &expected_values, expected.shape())
    }

    /// Validates softmax numerical stability.
    ///
    /// Checks:
    /// 1. Output sums to 1.0 (within tolerance)
    /// 2. All values are in [0, 1]
    /// 3. No NaN or Inf values
    pub fn validate_softmax(&self, output: &BoundedTensor) -> Result<(), ValidationError> {
        let values = output.values();

        // Check for NaN/Inf
        for (i, &v) in values.iter().enumerate() {
            if v.is_nan() {
                return Err(ValidationError::NumericalInstability {
                    message: format!("NaN detected at index {}", i),
                });
            }
            if v.is_infinite() {
                return Err(ValidationError::NumericalInstability {
                    message: format!("Infinity detected at index {}", i),
                });
            }
        }

        // Check bounds [0, 1]
        for (i, &v) in values.iter().enumerate() {
            if v < -self.config.softmax_sum_tolerance {
                return Err(ValidationError::NumericalInstability {
                    message: format!(
                        "Softmax value {} at index {} is negative",
                        v, i
                    ),
                });
            }
            if v > 1.0 + self.config.softmax_sum_tolerance {
                return Err(ValidationError::NumericalInstability {
                    message: format!(
                        "Softmax value {} at index {} exceeds 1.0",
                        v, i
                    ),
                });
            }
        }

        // Check sum-to-one (for 1D softmax or per-row for 2D)
        if output.ndim() == 1 {
            let sum: f64 = values.iter().sum();
            if (sum - 1.0).abs() > self.config.softmax_sum_tolerance {
                return Err(ValidationError::NumericalInstability {
                    message: format!(
                        "Softmax sum {} differs from 1.0 by {}",
                        sum,
                        (sum - 1.0).abs()
                    ),
                });
            }
        } else if output.ndim() == 2 {
            let rows = output.shape()[0];
            let cols = output.shape()[1];
            for r in 0..rows {
                let row_sum: f64 = (0..cols)
                    .map(|c| values[r * cols + c])
                    .sum();
                if (row_sum - 1.0).abs() > self.config.softmax_sum_tolerance {
                    return Err(ValidationError::NumericalInstability {
                        message: format!(
                            "Softmax row {} sum {} differs from 1.0",
                            r, row_sum
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validates attention output for numerical stability.
    ///
    /// Checks:
    /// 1. No NaN or Inf
    /// 2. Values are bounded
    /// 3. Attention weights sum to 1 per query
    pub fn validate_attention(
        &self,
        attention_weights: &BoundedTensor,
        attention_output: &BoundedTensor,
    ) -> Result<(), ValidationError> {
        // Validate attention weights
        self.validate_softmax(attention_weights)?;

        // Validate output for NaN/Inf
        let output_values = attention_output.values();
        for (i, &v) in output_values.iter().enumerate() {
            if v.is_nan() {
                return Err(ValidationError::NumericalInstability {
                    message: format!("NaN in attention output at index {}", i),
                });
            }
            if v.is_infinite() {
                return Err(ValidationError::NumericalInstability {
                    message: format!("Infinity in attention output at index {}", i),
                });
            }
        }

        Ok(())
    }

    /// Validates that error bounds are respected.
    pub fn validate_error_bounds(
        &self,
        tensor: &BoundedTensor,
        max_expected_error: f64,
    ) -> Result<(), ValidationError> {
        let max_error = tensor.max_error();
        if max_error > max_expected_error {
            return Err(ValidationError::ErrorBoundExceeded {
                computed: max_error,
                tolerance: max_expected_error,
            });
        }
        Ok(())
    }
}

impl Default for OutputValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer-by-layer validation for model forward pass.
pub struct LayerValidator {
    /// Expected outputs for each layer by name.
    expected_outputs: HashMap<String, (Vec<f64>, Vec<usize>)>,
    /// Validation config.
    config: ValidationConfig,
    /// Collected results.
    results: HashMap<String, ValidationResult>,
}

impl LayerValidator {
    /// Creates a new layer validator.
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            expected_outputs: HashMap::new(),
            config,
            results: HashMap::new(),
        }
    }

    /// Registers expected output for a layer.
    pub fn register_expected(
        &mut self,
        layer_name: &str,
        values: Vec<f64>,
        shape: Vec<usize>,
    ) {
        self.expected_outputs
            .insert(layer_name.to_string(), (values, shape));
    }

    /// Validates output for a layer.
    pub fn validate_layer(
        &mut self,
        layer_name: &str,
        actual: &BoundedTensor,
    ) -> Result<ValidationResult, ValidationError> {
        let (expected_values, expected_shape) = self
            .expected_outputs
            .get(layer_name)
            .ok_or_else(|| {
                ValidationError::ValidationFailed(format!(
                    "No expected output registered for layer '{}'",
                    layer_name
                ))
            })?;

        let validator = OutputValidator::with_config(self.config.clone());
        let result = validator.validate_tensor(actual, expected_values, expected_shape)?;

        self.results.insert(layer_name.to_string(), result.clone());
        Ok(result)
    }

    /// Returns all collected validation results.
    pub fn results(&self) -> &HashMap<String, ValidationResult> {
        &self.results
    }

    /// Checks if all validations passed.
    pub fn all_passed(&self) -> bool {
        self.results.values().all(|r| r.passed)
    }
}

/// Numerical stability checker for softmax and attention operations.
pub struct StabilityChecker {
    config: ValidationConfig,
}

impl StabilityChecker {
    /// Creates a new stability checker.
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Checks softmax stability with extreme inputs.
    pub fn check_softmax_stability(
        &self,
        input: &BoundedTensor,
    ) -> Result<StabilityReport, ValidationError> {
        let values = input.values();
        let mut report = StabilityReport::new();

        // Check input range
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        report.input_range = (min_val, max_val);

        // Compute softmax
        let softmax_result = ops::softmax::softmax(input, Precision::F32)
            .map_err(|e| ValidationError::NumericalInstability {
                message: format!("Softmax computation failed: {:?}", e),
            })?;

        // Validate result
        let validator = OutputValidator::with_config(self.config.clone());
        validator.validate_softmax(&softmax_result)?;

        // Check output range
        let out_values = softmax_result.values();
        report.output_range = (
            out_values.iter().cloned().fold(f64::INFINITY, f64::min),
            out_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );

        // Check sum deviation
        let sum: f64 = out_values.iter().sum();
        report.sum_deviation = (sum - 1.0).abs();

        report.passed = true;
        Ok(report)
    }

    /// Checks attention stability with large sequence lengths.
    pub fn check_attention_stability(
        &self,
        query: &BoundedTensor,
        key: &BoundedTensor,
        value: &BoundedTensor,
    ) -> Result<StabilityReport, ValidationError> {
        use crate::nn::attention::scaled_dot_product_attention;

        let mut report = StabilityReport::new();

        // Check input ranges
        let q_range = get_tensor_range(query);
        let k_range = get_tensor_range(key);
        let v_range = get_tensor_range(value);
        report.input_range = (
            q_range.0.min(k_range.0).min(v_range.0),
            q_range.1.max(k_range.1).max(v_range.1),
        );

        // Compute attention
        let output = scaled_dot_product_attention(query, key, value, Precision::F32)
            .map_err(|e| ValidationError::NumericalInstability {
                message: format!("Attention computation failed: {:?}", e),
            })?;

        // Check output
        let out_values = output.values();
        for (i, &v) in out_values.iter().enumerate() {
            if v.is_nan() {
                return Err(ValidationError::NumericalInstability {
                    message: format!("NaN in attention output at index {}", i),
                });
            }
            if v.is_infinite() {
                return Err(ValidationError::NumericalInstability {
                    message: format!("Infinity in attention output at index {}", i),
                });
            }
        }

        report.output_range = get_tensor_range(&output);
        report.passed = true;
        Ok(report)
    }
}

impl Default for StabilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Report from stability checking.
#[derive(Debug, Clone)]
pub struct StabilityReport {
    /// Whether the check passed.
    pub passed: bool,
    /// Input value range (min, max).
    pub input_range: (f64, f64),
    /// Output value range (min, max).
    pub output_range: (f64, f64),
    /// Sum deviation from 1.0 (for softmax).
    pub sum_deviation: f64,
}

impl StabilityReport {
    fn new() -> Self {
        Self {
            passed: false,
            input_range: (0.0, 0.0),
            output_range: (0.0, 0.0),
            sum_deviation: 0.0,
        }
    }
}

/// Helper to get min/max range of tensor values.
fn get_tensor_range(tensor: &BoundedTensor) -> (f64, f64) {
    let values = tensor.values();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// Validates linear layer forward pass.
pub fn validate_linear_forward(
    input: &BoundedTensor,
    weights: &BoundedTensor,
    bias: Option<&BoundedTensor>,
    expected_output: &[f64],
    config: &ValidationConfig,
) -> Result<ValidationResult, ValidationError> {
    use crate::nn::Linear;

    let layer = Linear::new(
        weights.clone(),
        bias.cloned(),
        Precision::F32,
    ).map_err(|e| ValidationError::ValidationFailed(format!("Layer creation failed: {:?}", e)))?;

    let actual = layer.forward(input)
        .map_err(|e| ValidationError::ValidationFailed(format!("Forward pass failed: {:?}", e)))?;

    let validator = OutputValidator::with_config(config.clone());
    validator.validate_tensor(&actual, expected_output, actual.shape())
}

/// Validates attention layer forward pass.
pub fn validate_attention_forward(
    query: &BoundedTensor,
    key: &BoundedTensor,
    value: &BoundedTensor,
    expected_output: &[f64],
    config: &ValidationConfig,
) -> Result<ValidationResult, ValidationError> {
    use crate::nn::attention::scaled_dot_product_attention;

    let actual = scaled_dot_product_attention(query, key, value, Precision::F32)
        .map_err(|e| ValidationError::ValidationFailed(format!("Attention failed: {:?}", e)))?;

    let validator = OutputValidator::with_config(config.clone());
    validator.validate_tensor(&actual, expected_output, actual.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_validator_exact() {
        let validator = OutputValidator::new();
        let tensor = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let expected = vec![1.0, 2.0, 3.0];

        let result = validator.validate_tensor(&tensor, &expected, &[3]).unwrap();
        assert!(result.passed);
        assert!(result.max_absolute_error < 1e-10);
    }

    #[test]
    fn test_output_validator_within_tolerance() {
        let config = ValidationConfig {
            absolute_tolerance: 0.01,
            relative_tolerance: 0.01,
            ..Default::default()
        };
        let validator = OutputValidator::with_config(config);

        let tensor = BoundedTensor::from_exact(vec![1.001, 2.002, 3.003], vec![3]);
        let expected = vec![1.0, 2.0, 3.0];

        let result = validator.validate_tensor(&tensor, &expected, &[3]).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_softmax_validation() {
        let validator = OutputValidator::new();

        // Valid softmax output
        let valid = BoundedTensor::from_exact(vec![0.7, 0.2, 0.1], vec![3]);
        assert!(validator.validate_softmax(&valid).is_ok());

        // Invalid: doesn't sum to 1
        let invalid = BoundedTensor::from_exact(vec![0.5, 0.5, 0.5], vec![3]);
        assert!(validator.validate_softmax(&invalid).is_err());
    }

    #[test]
    fn test_stability_checker_softmax() {
        let checker = StabilityChecker::new();

        // Normal input
        let normal = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let report = checker.check_softmax_stability(&normal).unwrap();
        assert!(report.passed);
        assert!(report.sum_deviation < 1e-6);

        // Large values (should still be stable due to max subtraction)
        let large = BoundedTensor::from_exact(vec![1000.0, 1001.0, 1002.0], vec![3]);
        let report = checker.check_softmax_stability(&large).unwrap();
        assert!(report.passed);
    }

    #[test]
    fn test_layer_validator() {
        let mut validator = LayerValidator::new(ValidationConfig::default());

        // Register expected output
        validator.register_expected("layer1", vec![1.0, 2.0], vec![2]);

        // Validate matching output
        let actual = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let result = validator.validate_layer("layer1", &actual).unwrap();
        assert!(result.passed);
        assert!(validator.all_passed());
    }

    #[test]
    fn test_validate_linear_forward() {
        // Identity transformation: W = I, b = 0
        let weights = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let input = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);
        let expected = vec![3.0, 4.0];

        let result = validate_linear_forward(
            &input,
            &weights,
            None,
            &expected,
            &ValidationConfig::default(),
        ).unwrap();

        assert!(result.passed);
    }

    #[test]
    fn test_validation_shape_mismatch() {
        let validator = OutputValidator::new();
        let tensor = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        let result = validator.validate_tensor(&tensor, &expected, &[4]);
        assert!(matches!(result, Err(ValidationError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_error_bound_validation() {
        let validator = OutputValidator::new();

        // Tensor with small error
        let small_error = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![2], 0.001);
        assert!(validator.validate_error_bounds(&small_error, 0.01).is_ok());

        // Tensor with large error
        let large_error = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![2], 0.1);
        assert!(validator.validate_error_bounds(&large_error, 0.01).is_err());
    }
}
