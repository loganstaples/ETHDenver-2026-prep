//! Input validation utilities for HELIX.
//!
//! This module provides comprehensive validation for:
//! - Configuration parameters with defensive checks
//! - External data input sanitization
//! - Serialization round-trip verification
//! - Numeric value bounds checking
//! - Recovery strategies for non-fatal errors

use crate::config::{HelixConfig, ProverConfig, TrainingConfig, VMConfig};
use crate::error::{HelixError, HelixResult, SerializationError, ValidationError};
use crate::types::{BoundedTensor, BoundedValue, Precision, Shape};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

// ============================================================================
// Configuration Validation
// ============================================================================

/// Validates a complete HelixConfig.
pub fn validate_config(config: &HelixConfig) -> HelixResult<()> {
    let mut errors = Vec::new();

    if let Err(e) = validate_vm_config(&config.vm) {
        errors.push(extract_validation_error(e));
    }
    if let Err(e) = validate_prover_config(&config.prover) {
        errors.push(extract_validation_error(e));
    }
    if let Err(e) = validate_training_config(&config.training) {
        errors.push(extract_validation_error(e));
    }

    if errors.is_empty() {
        Ok(())
    } else if errors.len() == 1 {
        Err(errors.remove(0).into())
    } else {
        Err(ValidationError::multiple(errors).into())
    }
}

/// Validates VM configuration.
pub fn validate_vm_config(config: &VMConfig) -> HelixResult<()> {
    let mut errors = Vec::new();

    // max_error_accumulation must be positive and reasonable
    if config.max_error_accumulation <= 0.0 {
        errors.push(ValidationError::out_of_range(
            "max_error_accumulation",
            config.max_error_accumulation,
            1e-10,
            1.0,
        ));
    }
    if config.max_error_accumulation > 1.0 {
        errors.push(ValidationError::out_of_range(
            "max_error_accumulation",
            config.max_error_accumulation,
            1e-10,
            1.0,
        ));
    }

    // memory_limit must be reasonable (at least 1MB, at most 1TB)
    if config.memory_limit < 1_000_000 {
        errors.push(ValidationError::out_of_range(
            "memory_limit",
            config.memory_limit as f64,
            1_000_000.0,
            1e15,
        ));
    }

    // max_operations must be positive
    if config.max_operations == 0 {
        errors.push(ValidationError::invalid_value(
            "max_operations",
            "0",
            "must be positive",
        ));
    }

    finish_validation(errors)
}

/// Validates prover configuration.
pub fn validate_prover_config(config: &ProverConfig) -> HelixResult<()> {
    let mut errors = Vec::new();

    // num_threads must be at least 1
    if config.num_threads == 0 {
        errors.push(ValidationError::invalid_value(
            "num_threads",
            "0",
            "must be at least 1",
        ));
    }

    // max_chunk_size must be reasonable
    if config.max_chunk_size == 0 {
        errors.push(ValidationError::invalid_value(
            "max_chunk_size",
            "0",
            "must be positive",
        ));
    }
    if config.max_chunk_size > 1_000_000 {
        errors.push(ValidationError::out_of_range(
            "max_chunk_size",
            config.max_chunk_size as f64,
            1.0,
            1_000_000.0,
        ));
    }

    finish_validation(errors)
}

/// Validates training configuration.
pub fn validate_training_config(config: &TrainingConfig) -> HelixResult<()> {
    let mut errors = Vec::new();

    // batch_size must be positive
    if config.batch_size == 0 {
        errors.push(ValidationError::invalid_value(
            "batch_size",
            "0",
            "must be positive",
        ));
    }

    // learning_rate must be positive and reasonable
    if config.learning_rate <= 0.0 {
        errors.push(ValidationError::out_of_range(
            "learning_rate",
            config.learning_rate,
            1e-10,
            10.0,
        ));
    }
    if config.learning_rate > 10.0 {
        errors.push(ValidationError::out_of_range(
            "learning_rate",
            config.learning_rate,
            1e-10,
            10.0,
        ));
    }
    if !config.learning_rate.is_finite() {
        errors.push(ValidationError::invalid_value(
            "learning_rate",
            format!("{}", config.learning_rate),
            "must be finite",
        ));
    }

    // num_rounds must be positive
    if config.num_rounds == 0 {
        errors.push(ValidationError::invalid_value(
            "num_rounds",
            "0",
            "must be positive",
        ));
    }

    // max_gradient_error must be positive
    if config.max_gradient_error <= 0.0 {
        errors.push(ValidationError::out_of_range(
            "max_gradient_error",
            config.max_gradient_error,
            1e-10,
            1.0,
        ));
    }
    if config.max_gradient_error > 1.0 {
        errors.push(ValidationError::out_of_range(
            "max_gradient_error",
            config.max_gradient_error,
            1e-10,
            1.0,
        ));
    }

    finish_validation(errors)
}

fn extract_validation_error(error: HelixError) -> ValidationError {
    match error {
        HelixError::Validation(v) => v,
        other => ValidationError::invalid_config(other.to_string()),
    }
}

fn finish_validation(errors: Vec<ValidationError>) -> HelixResult<()> {
    if errors.is_empty() {
        Ok(())
    } else if errors.len() == 1 {
        Err(errors.into_iter().next().unwrap().into())
    } else {
        Err(ValidationError::multiple(errors).into())
    }
}

// ============================================================================
// Numeric Validation
// ============================================================================

/// Validates that a value is finite (not NaN or Inf).
pub fn validate_finite(value: f64, field: &str) -> HelixResult<()> {
    if value.is_nan() {
        return Err(ValidationError::invalid_value(field, "NaN", "must be a finite number").into());
    }
    if value.is_infinite() {
        return Err(ValidationError::invalid_value(
            field,
            format!("{}", value),
            "must be a finite number",
        )
        .into());
    }
    Ok(())
}

/// Validates that a value is in the given range [min, max].
pub fn validate_range(value: f64, field: &str, min: f64, max: f64) -> HelixResult<()> {
    validate_finite(value, field)?;
    if value < min || value > max {
        return Err(ValidationError::out_of_range(field, value, min, max).into());
    }
    Ok(())
}

/// Validates that a value is positive.
pub fn validate_positive(value: f64, field: &str) -> HelixResult<()> {
    validate_finite(value, field)?;
    if value <= 0.0 {
        return Err(ValidationError::out_of_range(field, value, 0.0, f64::INFINITY).into());
    }
    Ok(())
}

/// Validates that a value is non-negative.
pub fn validate_non_negative(value: f64, field: &str) -> HelixResult<()> {
    validate_finite(value, field)?;
    if value < 0.0 {
        return Err(ValidationError::out_of_range(field, value, 0.0, f64::INFINITY).into());
    }
    Ok(())
}

/// Validates that a usize value is non-zero.
pub fn validate_non_zero(value: usize, field: &str) -> HelixResult<()> {
    if value == 0 {
        return Err(ValidationError::invalid_value(field, "0", "must be non-zero").into());
    }
    Ok(())
}

/// Validates a vector of f64 values for NaN/Inf.
pub fn validate_vector(data: &[f64], field: &str) -> HelixResult<()> {
    for (i, &v) in data.iter().enumerate() {
        if v.is_nan() {
            return Err(ValidationError::invalid_value(
                format!("{}[{}]", field, i),
                "NaN",
                "must be finite",
            )
            .into());
        }
        if v.is_infinite() {
            return Err(ValidationError::invalid_value(
                format!("{}[{}]", field, i),
                format!("{}", v),
                "must be finite",
            )
            .into());
        }
    }
    Ok(())
}

/// Validates a vector is non-empty.
pub fn validate_non_empty<T>(data: &[T], field: &str) -> HelixResult<()> {
    if data.is_empty() {
        return Err(ValidationError::invalid_value(field, "[]", "must not be empty").into());
    }
    Ok(())
}

// ============================================================================
// Input Sanitization
// ============================================================================

/// Sanitizes a f64 value by clamping NaN/Inf to safe defaults.
///
/// - NaN -> 0.0
/// - +Inf -> f64::MAX
/// - -Inf -> f64::MIN
pub fn sanitize_f64(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if value == f64::INFINITY {
        f64::MAX
    } else if value == f64::NEG_INFINITY {
        f64::MIN
    } else {
        value
    }
}

/// Sanitizes a vector of f64 values.
pub fn sanitize_vector(data: &mut [f64]) {
    for v in data.iter_mut() {
        *v = sanitize_f64(*v);
    }
}

/// Sanitizes a f64 value with a custom replacement for invalid values.
pub fn sanitize_f64_with(value: f64, replacement: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        replacement
    }
}

/// Clamps a value to the given range.
pub fn clamp_to_range(value: f64, min: f64, max: f64) -> f64 {
    sanitize_f64(value).clamp(min, max)
}

/// Sanitizes and validates external input data.
pub struct InputSanitizer {
    /// Whether to replace NaN/Inf or return an error.
    replace_invalid: bool,
    /// Replacement value for NaN.
    nan_replacement: f64,
    /// Replacement value for positive infinity.
    inf_replacement: f64,
    /// Replacement value for negative infinity.
    neg_inf_replacement: f64,
    /// Optional range for clamping.
    clamp_range: Option<(f64, f64)>,
}

impl Default for InputSanitizer {
    fn default() -> Self {
        Self {
            replace_invalid: true,
            nan_replacement: 0.0,
            inf_replacement: f64::MAX,
            neg_inf_replacement: f64::MIN,
            clamp_range: None,
        }
    }
}

impl InputSanitizer {
    /// Creates a new sanitizer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures to return errors instead of replacing invalid values.
    pub fn strict(mut self) -> Self {
        self.replace_invalid = false;
        self
    }

    /// Sets the replacement for NaN values.
    pub fn nan_replacement(mut self, value: f64) -> Self {
        self.nan_replacement = value;
        self
    }

    /// Sets a clamp range for all values.
    pub fn clamp(mut self, min: f64, max: f64) -> Self {
        self.clamp_range = Some((min, max));
        self
    }

    /// Sanitizes a single value.
    pub fn sanitize_value(&self, value: f64, field: &str) -> HelixResult<f64> {
        let result = if value.is_nan() {
            if self.replace_invalid {
                self.nan_replacement
            } else {
                return Err(
                    ValidationError::invalid_value(field, "NaN", "NaN not allowed").into()
                );
            }
        } else if value == f64::INFINITY {
            if self.replace_invalid {
                self.inf_replacement
            } else {
                return Err(
                    ValidationError::invalid_value(field, "Infinity", "Infinity not allowed").into()
                );
            }
        } else if value == f64::NEG_INFINITY {
            if self.replace_invalid {
                self.neg_inf_replacement
            } else {
                return Err(ValidationError::invalid_value(
                    field,
                    "-Infinity",
                    "negative infinity not allowed",
                )
                .into());
            }
        } else {
            value
        };

        Ok(if let Some((min, max)) = self.clamp_range {
            result.clamp(min, max)
        } else {
            result
        })
    }

    /// Sanitizes a vector of values.
    pub fn sanitize_vec(&self, data: &[f64], field: &str) -> HelixResult<Vec<f64>> {
        let mut result = Vec::with_capacity(data.len());
        for (i, &v) in data.iter().enumerate() {
            result.push(self.sanitize_value(v, &format!("{}[{}]", field, i))?);
        }
        Ok(result)
    }

    /// Sanitizes tensor data.
    pub fn sanitize_tensor_data(
        &self,
        data: Vec<f64>,
        shape: Shape,
        field: &str,
    ) -> HelixResult<BoundedTensor> {
        let sanitized = self.sanitize_vec(&data, field)?;
        BoundedTensor::try_from_exact(sanitized, shape)
    }
}

// ============================================================================
// Serialization Round-Trip Validation
// ============================================================================

/// Validates that a value can be serialized and deserialized without data loss.
pub fn validate_serialization_roundtrip<T>(value: &T) -> HelixResult<()>
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    // Serialize to JSON
    let json = serde_json::to_string(value)
        .map_err(|e| SerializationError::json(format!("serialization failed: {}", e)))?;

    // Deserialize back
    let restored: T = serde_json::from_str(&json)
        .map_err(|e| SerializationError::deserialization(format!("deserialization failed: {}", e)))?;

    // Compare
    if *value != restored {
        return Err(ValidationError::serialization_roundtrip(
            "deserialized value doesn't match original",
        )
        .into());
    }

    Ok(())
}

/// Validates binary serialization round-trip.
pub fn validate_binary_roundtrip<T>(value: &T) -> HelixResult<()>
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    // Serialize to bincode (using JSON as fallback since bincode might not be available)
    let bytes = serde_json::to_vec(value)
        .map_err(|e| SerializationError::binary(format!("serialization failed: {}", e)))?;

    // Deserialize back
    let restored: T = serde_json::from_slice(&bytes)
        .map_err(|e| SerializationError::deserialization(format!("deserialization failed: {}", e)))?;

    // Compare
    if *value != restored {
        return Err(ValidationError::serialization_roundtrip(
            "deserialized value doesn't match original",
        )
        .into());
    }

    Ok(())
}

/// Validates tensor serialization round-trip including error preservation.
pub fn validate_tensor_roundtrip(tensor: &BoundedTensor) -> HelixResult<()> {
    let json = serde_json::to_string(tensor)
        .map_err(|e| SerializationError::json(format!("tensor serialization failed: {}", e)))?;

    let restored: BoundedTensor = serde_json::from_str(&json)
        .map_err(|e| SerializationError::deserialization(format!("tensor deserialization failed: {}", e)))?;

    // Check shape
    if tensor.shape() != restored.shape() {
        return Err(ValidationError::serialization_roundtrip(format!(
            "shape mismatch: {:?} vs {:?}",
            tensor.shape(),
            restored.shape()
        ))
        .into());
    }

    // Check values and errors
    for (i, (orig, rest)) in tensor.data().iter().zip(restored.data().iter()).enumerate() {
        let val_diff = (orig.value() - rest.value()).abs();
        if val_diff > 1e-10 {
            return Err(ValidationError::serialization_roundtrip(format!(
                "value mismatch at index {}: {} vs {}",
                i,
                orig.value(),
                rest.value()
            ))
            .into());
        }

        let err_diff = (orig.absolute_error() - rest.absolute_error()).abs();
        if err_diff > 1e-10 {
            return Err(ValidationError::serialization_roundtrip(format!(
                "error mismatch at index {}: {} vs {}",
                i,
                orig.absolute_error(),
                rest.absolute_error()
            ))
            .into());
        }
    }

    Ok(())
}

// ============================================================================
// Recovery Strategies
// ============================================================================

/// Strategy for recovering from non-fatal errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Stop immediately on any error.
    FailFast,
    /// Replace invalid values with defaults.
    ReplaceWithDefault,
    /// Clamp values to valid range.
    ClampToRange,
    /// Skip invalid items (for collections).
    SkipInvalid,
    /// Use last known good value.
    UseLastGood,
    /// Retry with exponential backoff.
    RetryWithBackoff,
}

/// Context for recovery operations.
pub struct RecoveryContext {
    /// The strategy to use.
    pub strategy: RecoveryStrategy,
    /// Maximum number of retries.
    pub max_retries: usize,
    /// Default value for replacements.
    pub default_value: f64,
    /// Clamp range (min, max).
    pub clamp_range: (f64, f64),
    /// Number of errors encountered.
    pub error_count: usize,
    /// Whether to log recovery actions.
    pub log_recoveries: bool,
    /// Recovery log.
    recovery_log: Vec<RecoveryAction>,
}

/// A logged recovery action.
#[derive(Debug, Clone)]
pub struct RecoveryAction {
    /// Field that was recovered.
    pub field: String,
    /// Original value (if available).
    pub original: Option<f64>,
    /// Recovered value.
    pub recovered: f64,
    /// Strategy used.
    pub strategy: RecoveryStrategy,
    /// Reason for recovery.
    pub reason: String,
}

impl Default for RecoveryContext {
    fn default() -> Self {
        Self {
            strategy: RecoveryStrategy::ReplaceWithDefault,
            max_retries: 3,
            default_value: 0.0,
            clamp_range: (f64::MIN, f64::MAX),
            error_count: 0,
            log_recoveries: true,
            recovery_log: Vec::new(),
        }
    }
}

impl RecoveryContext {
    /// Creates a new recovery context.
    pub fn new(strategy: RecoveryStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    /// Sets the default replacement value.
    pub fn with_default(mut self, default: f64) -> Self {
        self.default_value = default;
        self
    }

    /// Sets the clamp range.
    pub fn with_clamp_range(mut self, min: f64, max: f64) -> Self {
        self.clamp_range = (min, max);
        self
    }

    /// Sets the maximum retries.
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Recovers an invalid value.
    pub fn recover_value(&mut self, value: f64, field: &str) -> HelixResult<f64> {
        if value.is_finite() {
            return Ok(value);
        }

        self.error_count += 1;

        let (recovered, reason) = match self.strategy {
            RecoveryStrategy::FailFast => {
                return Err(ValidationError::invalid_value(
                    field,
                    format!("{}", value),
                    "fail-fast strategy rejects invalid values",
                )
                .into());
            }
            RecoveryStrategy::ReplaceWithDefault => {
                (self.default_value, "replaced with default".to_string())
            }
            RecoveryStrategy::ClampToRange => {
                let clamped = if value.is_nan() {
                    self.default_value
                } else if value == f64::INFINITY {
                    self.clamp_range.1
                } else {
                    self.clamp_range.0
                };
                (clamped, format!("clamped to range {:?}", self.clamp_range))
            }
            RecoveryStrategy::SkipInvalid => {
                // For skip strategy, we need to signal the caller
                return Err(ValidationError::invalid_value(
                    field,
                    format!("{}", value),
                    "skip-invalid strategy",
                )
                .into());
            }
            RecoveryStrategy::UseLastGood => {
                // In a simple context, fall back to default
                (self.default_value, "used default (no last good value)".to_string())
            }
            RecoveryStrategy::RetryWithBackoff => {
                // Retry doesn't make sense for value recovery
                (self.default_value, "replaced with default (retry not applicable)".to_string())
            }
        };

        if self.log_recoveries {
            self.recovery_log.push(RecoveryAction {
                field: field.to_string(),
                original: Some(value),
                recovered,
                strategy: self.strategy,
                reason,
            });
        }

        Ok(recovered)
    }

    /// Recovers a vector of values.
    pub fn recover_vector(&mut self, data: &mut Vec<f64>, field: &str) -> HelixResult<()> {
        if self.strategy == RecoveryStrategy::SkipInvalid {
            // Filter out invalid values
            let original_len = data.len();
            data.retain(|v| v.is_finite());
            if data.len() < original_len && self.log_recoveries {
                self.recovery_log.push(RecoveryAction {
                    field: field.to_string(),
                    original: None,
                    recovered: f64::NAN, // placeholder
                    strategy: self.strategy,
                    reason: format!("skipped {} invalid values", original_len - data.len()),
                });
            }
        } else {
            for (i, v) in data.iter_mut().enumerate() {
                *v = self.recover_value(*v, &format!("{}[{}]", field, i))?;
            }
        }
        Ok(())
    }

    /// Returns the recovery log.
    pub fn recovery_log(&self) -> &[RecoveryAction] {
        &self.recovery_log
    }

    /// Returns the number of errors encountered.
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Clears the recovery log and error count.
    pub fn reset(&mut self) {
        self.recovery_log.clear();
        self.error_count = 0;
    }
}

// ============================================================================
// Validation Result Aggregation
// ============================================================================

/// Collects multiple validation results and reports all errors.
pub struct ValidationCollector {
    errors: Vec<ValidationError>,
}

impl Default for ValidationCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationCollector {
    /// Creates a new validation collector.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Adds a validation result.
    pub fn add<T>(&mut self, result: HelixResult<T>) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                if let HelixError::Validation(v) = e {
                    self.errors.push(v);
                } else {
                    self.errors
                        .push(ValidationError::invalid_config(e.to_string()));
                }
                None
            }
        }
    }

    /// Adds a validation error directly.
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    /// Validates a condition and adds an error if false.
    pub fn require(&mut self, condition: bool, error: ValidationError) {
        if !condition {
            self.errors.push(error);
        }
    }

    /// Returns true if no errors were collected.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Finishes collection and returns the result.
    pub fn finish(self) -> HelixResult<()> {
        finish_validation(self.errors)
    }

    /// Returns collected errors.
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }
}

// ============================================================================
// Type-Specific Validators
// ============================================================================

/// Validates a Shape.
pub fn validate_shape(shape: &Shape, field: &str) -> HelixResult<()> {
    if shape.is_empty() {
        return Err(
            ValidationError::invalid_shape(field, shape.clone(), "shape cannot be empty").into(),
        );
    }

    for (i, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(ValidationError::invalid_shape(
                field,
                shape.clone(),
                format!("dimension {} is zero", i),
            )
            .into());
        }
    }

    // Check for overflow in total size
    let mut total: usize = 1;
    for &dim in shape {
        total = total
            .checked_mul(dim)
            .ok_or_else(|| ValidationError::invalid_shape(field, shape.clone(), "size overflow"))?;
    }

    Ok(())
}

/// Validates a Precision value.
pub fn validate_precision(precision: &Precision, field: &str) -> HelixResult<()> {
    if let Precision::Custom {
        bits,
        max_relative_error,
    } = precision
    {
        if *bits == 0 || *bits > 64 {
            return Err(ValidationError::out_of_range(field, *bits as f64, 1.0, 64.0).into());
        }
        if *max_relative_error <= 0.0 || *max_relative_error > 1.0 {
            return Err(
                ValidationError::out_of_range(field, *max_relative_error, 0.0, 1.0).into(),
            );
        }
    }
    Ok(())
}

/// Validates a BoundedValue.
pub fn validate_bounded_value(
    value: &BoundedValue<f64>,
    field: &str,
    max_error: Option<f64>,
) -> HelixResult<()> {
    if value.value().is_nan() {
        return Err(ValidationError::invalid_value(field, "NaN", "value is NaN").into());
    }
    if value.value().is_infinite() {
        return Err(ValidationError::invalid_value(
            field,
            format!("{}", value.value()),
            "value is infinite",
        )
        .into());
    }

    let abs_err = value.absolute_error();
    if abs_err.is_nan() {
        return Err(
            ValidationError::invalid_value(field, "error=NaN", "error bound is NaN").into()
        );
    }
    if abs_err < 0.0 {
        return Err(ValidationError::invalid_value(
            field,
            format!("error={}", abs_err),
            "error bound is negative",
        )
        .into());
    }

    if let Some(max) = max_error {
        if abs_err > max {
            return Err(ValidationError::out_of_range(
                &format!("{}.error", field),
                abs_err,
                0.0,
                max,
            )
            .into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_finite() {
        assert!(validate_finite(1.0, "test").is_ok());
        assert!(validate_finite(f64::NAN, "test").is_err());
        assert!(validate_finite(f64::INFINITY, "test").is_err());
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range(0.5, "test", 0.0, 1.0).is_ok());
        assert!(validate_range(-0.1, "test", 0.0, 1.0).is_err());
        assert!(validate_range(1.5, "test", 0.0, 1.0).is_err());
    }

    #[test]
    fn test_sanitize_f64() {
        assert_eq!(sanitize_f64(1.0), 1.0);
        assert_eq!(sanitize_f64(f64::NAN), 0.0);
        assert_eq!(sanitize_f64(f64::INFINITY), f64::MAX);
        assert_eq!(sanitize_f64(f64::NEG_INFINITY), f64::MIN);
    }

    #[test]
    fn test_input_sanitizer() {
        let sanitizer = InputSanitizer::new().clamp(-1.0, 1.0);

        assert_eq!(sanitizer.sanitize_value(0.5, "test").unwrap(), 0.5);
        assert_eq!(sanitizer.sanitize_value(f64::NAN, "test").unwrap(), 0.0);
        assert_eq!(sanitizer.sanitize_value(100.0, "test").unwrap(), 1.0);
        assert_eq!(sanitizer.sanitize_value(-100.0, "test").unwrap(), -1.0);
    }

    #[test]
    fn test_strict_sanitizer() {
        let sanitizer = InputSanitizer::new().strict();

        assert!(sanitizer.sanitize_value(1.0, "test").is_ok());
        assert!(sanitizer.sanitize_value(f64::NAN, "test").is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let value = vec![1.0, 2.0, 3.0];
        assert!(validate_serialization_roundtrip(&value).is_ok());
    }

    #[test]
    fn test_tensor_roundtrip() {
        let tensor = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0.01);
        assert!(validate_tensor_roundtrip(&tensor).is_ok());
    }

    #[test]
    fn test_recovery_context() {
        let mut ctx = RecoveryContext::new(RecoveryStrategy::ReplaceWithDefault)
            .with_default(0.0);

        assert_eq!(ctx.recover_value(1.0, "test").unwrap(), 1.0);
        assert_eq!(ctx.recover_value(f64::NAN, "test").unwrap(), 0.0);
        assert_eq!(ctx.error_count(), 1);
    }

    #[test]
    fn test_recovery_fail_fast() {
        let mut ctx = RecoveryContext::new(RecoveryStrategy::FailFast);

        assert!(ctx.recover_value(1.0, "test").is_ok());
        assert!(ctx.recover_value(f64::NAN, "test").is_err());
    }

    #[test]
    fn test_recovery_clamp() {
        let mut ctx = RecoveryContext::new(RecoveryStrategy::ClampToRange)
            .with_clamp_range(-1.0, 1.0);

        assert_eq!(ctx.recover_value(0.5, "test").unwrap(), 0.5);
        assert_eq!(ctx.recover_value(f64::INFINITY, "test").unwrap(), 1.0);
    }

    #[test]
    fn test_validation_collector() {
        let mut collector = ValidationCollector::new();

        collector.add(validate_finite(1.0, "a"));
        collector.add::<()>(Err(ValidationError::missing_field("b").into()));
        collector.add::<()>(Err(ValidationError::missing_field("c").into()));

        assert!(!collector.is_valid());
        assert_eq!(collector.errors().len(), 2);
    }

    #[test]
    fn test_validate_vm_config() {
        let valid = VMConfig::default();
        assert!(validate_vm_config(&valid).is_ok());

        let invalid = VMConfig {
            max_error_accumulation: -0.1,
            ..Default::default()
        };
        assert!(validate_vm_config(&invalid).is_err());
    }

    #[test]
    fn test_validate_training_config() {
        let valid = TrainingConfig::default();
        assert!(validate_training_config(&valid).is_ok());

        let invalid = TrainingConfig {
            learning_rate: -0.1,
            ..Default::default()
        };
        assert!(validate_training_config(&invalid).is_err());
    }

    #[test]
    fn test_validate_shape() {
        assert!(validate_shape(&vec![2, 3, 4], "test").is_ok());
        assert!(validate_shape(&vec![], "test").is_err());
        assert!(validate_shape(&vec![2, 0, 4], "test").is_err());
    }

    #[test]
    fn test_validate_bounded_value() {
        let valid = BoundedValue::<f64>::with_absolute_error(1.0, 0.01);
        assert!(validate_bounded_value(&valid, "test", Some(0.1)).is_ok());

        let exceeds = BoundedValue::<f64>::with_absolute_error(1.0, 0.5);
        assert!(validate_bounded_value(&exceeds, "test", Some(0.1)).is_err());

        let nan = BoundedValue::exact(f64::NAN);
        assert!(validate_bounded_value(&nan, "test", None).is_err());
    }
}
