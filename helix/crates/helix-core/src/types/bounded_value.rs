//! Bounded value type for approximate computation.
//!
//! A `BoundedValue<T>` represents a computed value together with its error bounds.
//! This module provides comprehensive bounds checking, NaN/Inf detection, and
//! graceful overflow handling for robust numerical computations.

use super::error_margin::ErrorMargin;
use crate::error::{ArithmeticError, BoundsError, HelixResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use tracing;

/// Maximum allowed error bound to prevent overflow in subsequent computations.
/// Note: This is a permissive upper limit for clamping. For practical error tracking,
/// use `MAX_SAFE_ERROR` which triggers warnings/errors before reaching this ceiling.
pub const MAX_ERROR_BOUND: f64 = 1e100;

/// Maximum safe error bound for practical computations.
/// If error exceeds this threshold, the computation is likely numerically unstable
/// and should be flagged. This is much more conservative than MAX_ERROR_BOUND.
pub const MAX_SAFE_ERROR: f64 = 1e6;

/// Minimum representable positive value for underflow detection.
pub const MIN_POSITIVE_VALUE: f64 = 1e-300;

/// Threshold for division-by-zero detection.
pub const DIVISION_THRESHOLD: f64 = 1e-15;

/// A value with tracked error bounds.
///
/// The true value is guaranteed to be within `[value - error, value + error]`.
///
/// # Invariants
///
/// - Error margin is always non-negative
/// - Neither value nor error bound should be NaN
/// - Infinity in error bound indicates unbounded uncertainty
///
/// # Example
///
/// ```
/// use helix_core::types::{BoundedValue, ErrorMargin};
///
/// // Create a bounded value with absolute error
/// let x = BoundedValue::<f64>::with_absolute_error(10.0, 0.1);
/// assert_eq!(x.value(), 10.0);
/// assert!((x.absolute_error() - 0.1).abs() < 1e-10);
///
/// // True value is in [9.9, 10.1]
/// assert!((x.lower_bound() - 9.9).abs() < 1e-10);
/// assert!((x.upper_bound() - 10.1).abs() < 1e-10);
///
/// // Arithmetic propagates errors
/// let y = BoundedValue::<f64>::with_absolute_error(5.0, 0.05);
/// let sum = x + y;  // Error adds: 0.1 + 0.05 = 0.15
/// assert_eq!(sum.value(), 15.0);
/// assert!((sum.absolute_error() - 0.15).abs() < 1e-10);
/// ```
#[must_use = "BoundedValue tracks error margins; discarding it silently drops error tracking"]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundedValue<T> {
    /// The computed/approximate value.
    value: T,
    /// The error margin (bounds on approximation error).
    error: ErrorMargin,
}

impl<T: Copy> BoundedValue<T> {
    /// Creates a new bounded value with the given error margin.
    pub fn new(value: T, error: ErrorMargin) -> Self {
        Self { value, error }
    }

    /// Creates an exact value with zero error.
    pub fn exact(value: T) -> Self {
        Self {
            value,
            error: ErrorMargin::ZERO,
        }
    }

    /// Returns the computed value.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the error margin.
    pub fn error(&self) -> ErrorMargin {
        self.error
    }

    /// Returns the error margin as a mutable reference.
    pub fn error_mut(&mut self) -> &mut ErrorMargin {
        &mut self.error
    }

    /// Sets the error margin.
    pub fn set_error(&mut self, error: ErrorMargin) {
        self.error = error;
    }
}

impl BoundedValue<f64> {
    /// Creates a bounded value with absolute error, with validation.
    ///
    /// Returns an error if the value or epsilon is NaN/Inf or if epsilon is negative.
    pub fn try_with_absolute_error(value: f64, epsilon: f64) -> HelixResult<Self> {
        Self::validate_finite(value, "value")?;
        Self::validate_non_negative(epsilon, "epsilon")?;
        Ok(Self::new(value, ErrorMargin::absolute(epsilon.min(MAX_ERROR_BOUND))))
    }

    /// Creates a bounded value with relative error, with validation.
    ///
    /// Returns an error if the value or epsilon is NaN/Inf or if epsilon is negative.
    pub fn try_with_relative_error(value: f64, epsilon: f64) -> HelixResult<Self> {
        Self::validate_finite(value, "value")?;
        Self::validate_non_negative(epsilon, "epsilon")?;
        Ok(Self::new(value, ErrorMargin::relative(epsilon.min(1e10))))
    }

    /// Creates a bounded value with absolute error.
    pub fn with_absolute_error(value: f64, epsilon: f64) -> Self {
        Self::new(value, ErrorMargin::absolute(epsilon))
    }

    /// Creates a bounded value with relative error.
    pub fn with_relative_error(value: f64, epsilon: f64) -> Self {
        Self::new(value, ErrorMargin::relative(epsilon))
    }

    /// Returns the absolute error bound.
    pub fn absolute_error(&self) -> f64 {
        self.error.to_absolute(self.value)
    }

    /// Returns the lower bound of the true value.
    pub fn lower_bound(&self) -> f64 {
        self.value - self.absolute_error()
    }

    /// Returns the upper bound of the true value.
    pub fn upper_bound(&self) -> f64 {
        self.value + self.absolute_error()
    }

    /// Returns true if this value's bounds contain the given value.
    pub fn contains(&self, val: f64) -> bool {
        val >= self.lower_bound() && val <= self.upper_bound()
    }

    /// Returns true if this value's bounds overlap with another.
    pub fn overlaps(&self, other: &BoundedValue<f64>) -> bool {
        self.lower_bound() <= other.upper_bound() && other.lower_bound() <= self.upper_bound()
    }

    /// Returns the width of the error interval.
    pub fn interval_width(&self) -> f64 {
        2.0 * self.absolute_error()
    }

    /// Checks if the value is finite (not NaN or Inf).
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    /// Checks if the value is NaN.
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    /// Checks if the value is infinite.
    pub fn is_infinite(&self) -> bool {
        self.value.is_infinite()
    }

    /// Checks if the error bound is finite.
    pub fn has_finite_error(&self) -> bool {
        let abs_err = self.absolute_error();
        abs_err.is_finite()
    }

    /// Checks if the value has valid bounds (error is finite and non-negative).
    pub fn has_valid_bounds(&self) -> bool {
        let abs_err = self.absolute_error();
        abs_err.is_finite() && abs_err >= 0.0
    }

    /// Returns true if this bounded value is in a valid state.
    pub fn is_valid(&self) -> bool {
        self.is_finite() && self.has_valid_bounds()
    }

    /// Validates that a value is finite.
    fn validate_finite(value: f64, name: &str) -> HelixResult<()> {
        if value.is_nan() {
            return Err(ArithmeticError::nan_detected(format!("{} creation", name)).into());
        }
        if value.is_infinite() {
            return Err(ArithmeticError::infinity_detected(
                format!("{} creation", name),
                value,
            )
            .into());
        }
        Ok(())
    }

    /// Validates that a value is non-negative.
    fn validate_non_negative(value: f64, name: &str) -> HelixResult<()> {
        if value.is_nan() {
            return Err(ArithmeticError::nan_detected(format!("{} validation", name)).into());
        }
        if value < 0.0 {
            return Err(BoundsError::NegativeMargin(value).into());
        }
        Ok(())
    }

    /// Clamps the error bound to prevent overflow.
    pub fn clamp_error(&mut self) {
        let abs_err = self.absolute_error();
        if abs_err > MAX_ERROR_BOUND || abs_err.is_infinite() {
            tracing::warn!(
                original_error = abs_err,
                clamped_to = MAX_ERROR_BOUND,
                "Error bound exceeds MAX_ERROR_BOUND, clamping"
            );
            self.error = ErrorMargin::absolute(MAX_ERROR_BOUND);
        }
    }

    /// Returns a version with clamped error bounds.
    pub fn with_clamped_error(mut self) -> Self {
        self.clamp_error();
        self
    }

    /// Sanitizes the value by replacing NaN with zero and clamping infinities.
    pub fn sanitize(&mut self) {
        if self.value.is_nan() {
            tracing::warn!("Sanitizing NaN value to 0.0 with maximum error bound");
            self.value = 0.0;
            self.error = ErrorMargin::absolute(MAX_ERROR_BOUND);
        } else if self.value.is_infinite() {
            tracing::warn!(
                original_value = self.value,
                "Sanitizing infinite value to f64::MAX with maximum error bound"
            );
            self.value = self.value.signum() * f64::MAX;
            self.error = ErrorMargin::absolute(MAX_ERROR_BOUND);
        }
        self.clamp_error();
    }

    /// Returns a sanitized version of this value.
    pub fn sanitized(mut self) -> Self {
        self.sanitize();
        self
    }

    // === Checked arithmetic operations ===

    /// Checked addition that returns a Result.
    #[must_use]
    pub fn checked_add(self, rhs: Self) -> HelixResult<Self> {
        // Check for NaN/Inf in inputs
        if self.is_nan() || rhs.is_nan() {
            return Err(ArithmeticError::nan_detected("addition").into());
        }

        let value = self.value + rhs.value;

        // Check for overflow in result
        if value.is_nan() {
            return Err(ArithmeticError::nan_detected("addition result").into());
        }
        if value.is_infinite() {
            return Err(ArithmeticError::infinity_detected("addition", value).into());
        }

        // Compute error with overflow protection
        let error = self.error.add(&rhs.error, self.value, rhs.value);
        let abs_error = error.epsilon();

        if abs_error.is_infinite() || abs_error > MAX_ERROR_BOUND {
            return Err(BoundsError::overflow("addition", abs_error).into());
        }

        Ok(BoundedValue { value, error })
    }

    /// Checked subtraction that returns a Result.
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> HelixResult<Self> {
        if self.is_nan() || rhs.is_nan() {
            return Err(ArithmeticError::nan_detected("subtraction").into());
        }

        let value = self.value - rhs.value;

        if value.is_nan() {
            return Err(ArithmeticError::nan_detected("subtraction result").into());
        }
        if value.is_infinite() {
            return Err(ArithmeticError::infinity_detected("subtraction", value).into());
        }

        let error = self.error.add(&rhs.error, self.value, rhs.value);
        let abs_error = error.epsilon();

        if abs_error.is_infinite() || abs_error > MAX_ERROR_BOUND {
            return Err(BoundsError::overflow("subtraction", abs_error).into());
        }

        Ok(BoundedValue { value, error })
    }

    /// Checked multiplication that returns a Result.
    #[must_use]
    pub fn checked_mul(self, rhs: Self) -> HelixResult<Self> {
        if self.is_nan() || rhs.is_nan() {
            return Err(ArithmeticError::nan_detected("multiplication").into());
        }

        let value = self.value * rhs.value;

        if value.is_nan() {
            return Err(ArithmeticError::nan_detected("multiplication result").into());
        }
        if value.is_infinite() {
            return Err(ArithmeticError::infinity_detected("multiplication", value).into());
        }

        let error = self.error.multiply(&rhs.error, self.value, rhs.value);
        let abs_error = error.epsilon();

        if abs_error.is_infinite() || abs_error > MAX_ERROR_BOUND {
            return Err(BoundsError::overflow("multiplication", abs_error).into());
        }

        Ok(BoundedValue { value, error })
    }

    /// Checked division that returns a Result.
    #[must_use]
    pub fn checked_div(self, rhs: Self) -> HelixResult<Self> {
        if self.is_nan() || rhs.is_nan() {
            return Err(ArithmeticError::nan_detected("division").into());
        }

        // Check for division by zero
        let b_abs = rhs.value.abs();
        if b_abs < DIVISION_THRESHOLD {
            return Err(ArithmeticError::DivisionByZero.into());
        }

        let value = self.value / rhs.value;

        if value.is_nan() {
            return Err(ArithmeticError::nan_detected("division result").into());
        }
        if value.is_infinite() {
            return Err(ArithmeticError::infinity_detected("division", value).into());
        }

        // Compute division error
        let eps_a = self.error.to_absolute(self.value);
        let eps_b = rhs.error.to_absolute(rhs.value);

        let rel_error = eps_a / self.value.abs().max(DIVISION_THRESHOLD) + eps_b / b_abs;
        let abs_error = value.abs() * rel_error + eps_a / b_abs;

        if abs_error.is_infinite() || abs_error > MAX_ERROR_BOUND {
            return Err(BoundsError::overflow("division", abs_error).into());
        }

        Ok(BoundedValue::new(value, ErrorMargin::absolute(abs_error)))
    }

    /// Saturating addition that clamps overflow instead of returning an error.
    #[must_use]
    pub fn saturating_add(self, rhs: Self) -> Self {
        let mut result = self + rhs;
        if !result.is_finite() || !result.has_finite_error() {
            tracing::debug!(
                op = "saturating_add",
                lhs = self.value,
                rhs = rhs.value,
                "Saturating arithmetic required sanitization"
            );
        }
        result.sanitize();
        result
    }

    /// Saturating subtraction that clamps overflow instead of returning an error.
    #[must_use]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        let mut result = self - rhs;
        if !result.is_finite() || !result.has_finite_error() {
            tracing::debug!(
                op = "saturating_sub",
                lhs = self.value,
                rhs = rhs.value,
                "Saturating arithmetic required sanitization"
            );
        }
        result.sanitize();
        result
    }

    /// Saturating multiplication that clamps overflow instead of returning an error.
    #[must_use]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let mut result = self * rhs;
        if !result.is_finite() || !result.has_finite_error() {
            tracing::debug!(
                op = "saturating_mul",
                lhs = self.value,
                rhs = rhs.value,
                "Saturating arithmetic required sanitization"
            );
        }
        result.sanitize();
        result
    }

    /// Saturating division that clamps overflow instead of returning an error.
    /// Returns a value with maximum error if dividing by near-zero.
    #[must_use]
    pub fn saturating_div(self, rhs: Self) -> Self {
        let mut result = self / rhs;
        if !result.is_finite() || !result.has_finite_error() {
            tracing::debug!(
                op = "saturating_div",
                lhs = self.value,
                rhs = rhs.value,
                "Saturating arithmetic required sanitization"
            );
        }
        result.sanitize();
        result
    }

    // === Error accumulation helpers ===

    /// Accumulates error from this operation with graceful overflow handling.
    ///
    /// Returns the accumulated error if successful, or the clamped maximum if overflow.
    pub fn accumulate_error(current: f64, additional: f64) -> Result<f64, f64> {
        let sum = current + additional;
        if sum.is_infinite() || sum > MAX_ERROR_BOUND {
            Err(MAX_ERROR_BOUND)
        } else {
            Ok(sum)
        }
    }

    /// Creates a value representing a failed computation with maximum uncertainty.
    pub fn failed_computation() -> Self {
        Self::new(0.0, ErrorMargin::absolute(MAX_ERROR_BOUND))
    }

    /// Creates a value representing positive infinity with maximum uncertainty.
    pub fn positive_overflow() -> Self {
        Self::new(f64::MAX, ErrorMargin::absolute(MAX_ERROR_BOUND))
    }

    /// Creates a value representing negative infinity with maximum uncertainty.
    pub fn negative_overflow() -> Self {
        Self::new(f64::MIN, ErrorMargin::absolute(MAX_ERROR_BOUND))
    }

    // === Validation helpers ===

    /// Validates that this value can be used in subsequent computations.
    pub fn validate(&self) -> HelixResult<()> {
        if self.is_nan() {
            return Err(ArithmeticError::nan_detected("bounded value").into());
        }
        if self.is_infinite() {
            return Err(ArithmeticError::infinity_detected("bounded value", self.value).into());
        }
        let abs_err = self.absolute_error();
        if abs_err.is_nan() {
            return Err(BoundsError::invalid_propagation("error margin is NaN").into());
        }
        if abs_err < 0.0 {
            return Err(BoundsError::NegativeMargin(abs_err).into());
        }
        Ok(())
    }

    /// Validates and returns self if valid.
    pub fn validated(self) -> HelixResult<Self> {
        self.validate()?;
        Ok(self)
    }

    /// Checks if this value exceeds an error threshold.
    pub fn exceeds_threshold(&self, threshold: f64) -> bool {
        self.absolute_error() > threshold
    }

    /// Returns an error if this value exceeds the given threshold.
    pub fn check_threshold(&self, threshold: f64) -> HelixResult<()> {
        let abs_err = self.absolute_error();
        if abs_err > threshold {
            return Err(BoundsError::ExceedsMaximum {
                computed: abs_err,
                max_allowed: threshold,
            }
            .into());
        }
        Ok(())
    }

    /// Checks if error has exceeded the safe threshold, indicating potential numerical instability.
    /// This is more conservative than `MAX_ERROR_BOUND` and should be used to detect
    /// error explosion early in training runs.
    pub fn check_safe_error(&self) -> HelixResult<()> {
        let abs_err = self.absolute_error();
        if abs_err > MAX_SAFE_ERROR {
            return Err(BoundsError::ErrorExplosion {
                accumulated: abs_err,
                safe_limit: MAX_SAFE_ERROR,
            }
            .into());
        }
        Ok(())
    }

    /// Checks if error has exceeded a custom safe threshold.
    /// Use this for precision-aware error checking (e.g., BF16/INT8 have
    /// different safe limits than F32).
    pub fn check_safe_error_with_limit(&self, limit: f64) -> HelixResult<()> {
        let abs_err = self.absolute_error();
        if abs_err > limit {
            return Err(BoundsError::ErrorExplosion {
                accumulated: abs_err,
                safe_limit: limit,
            }
            .into());
        }
        Ok(())
    }

    /// Returns true if the error has exceeded the safe threshold.
    pub fn has_error_explosion(&self) -> bool {
        self.absolute_error() > MAX_SAFE_ERROR
    }
}

impl BoundedValue<f32> {
    /// Creates a bounded value with absolute error.
    pub fn with_absolute_error(value: f32, epsilon: f32) -> Self {
        Self::new(value, ErrorMargin::absolute(epsilon as f64))
    }

    /// Returns the absolute error bound.
    pub fn absolute_error(&self) -> f32 {
        self.error.to_absolute(self.value as f64) as f32
    }

    /// Returns the lower bound.
    pub fn lower_bound(&self) -> f32 {
        self.value - self.absolute_error()
    }

    /// Returns the upper bound.
    pub fn upper_bound(&self) -> f32 {
        self.value + self.absolute_error()
    }

    /// Checks if the value is finite.
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    /// Checks if the value is NaN.
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    /// Sanitizes the value.
    pub fn sanitize(&mut self) {
        if self.value.is_nan() {
            self.value = 0.0;
            self.error = ErrorMargin::absolute(MAX_ERROR_BOUND);
        } else if self.value.is_infinite() {
            self.value = self.value.signum() * f32::MAX;
            self.error = ErrorMargin::absolute(MAX_ERROR_BOUND);
        }
    }
}

// Arithmetic operations with error propagation

impl Add for BoundedValue<f64> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.value + rhs.value;
        let error = self.error.add(&rhs.error, self.value, rhs.value);
        BoundedValue { value, error }
    }
}

impl Sub for BoundedValue<f64> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.value - rhs.value;
        // Subtraction has same error propagation as addition
        let error = self.error.add(&rhs.error, self.value, rhs.value);
        BoundedValue { value, error }
    }
}

impl Mul for BoundedValue<f64> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let value = self.value * rhs.value;
        let error = self.error.multiply(&rhs.error, self.value, rhs.value);
        BoundedValue { value, error }
    }
}

impl Div for BoundedValue<f64> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Guard against division by near-zero — cap error at MAX_ERROR_BOUND
        // instead of producing Infinity which corrupts downstream computations
        let b_abs = rhs.value.abs();
        if b_abs < DIVISION_THRESHOLD {
            let value = if rhs.value == 0.0 { 0.0 } else { self.value / rhs.value };
            let value = if value.is_finite() { value } else { 0.0 };
            return BoundedValue::new(value, ErrorMargin::absolute(MAX_ERROR_BOUND));
        }

        let value = self.value / rhs.value;
        // Handle NaN/Inf results from the division itself
        if !value.is_finite() {
            return BoundedValue::new(0.0, ErrorMargin::absolute(MAX_ERROR_BOUND));
        }

        // Division error: ε(a/b) ≈ |a/b| * (εa/|a| + εb/|b|) for small relative errors
        let eps_a = self.error.to_absolute(self.value);
        let eps_b = rhs.error.to_absolute(rhs.value);

        let rel_error = eps_a / self.value.abs().max(DIVISION_THRESHOLD) + eps_b / b_abs;
        let abs_error = value.abs() * rel_error + eps_a / b_abs;

        // Cap at MAX_ERROR_BOUND to prevent infinite error propagation
        let capped_error = if abs_error.is_finite() { abs_error.min(MAX_ERROR_BOUND) } else { MAX_ERROR_BOUND };
        BoundedValue::new(value, ErrorMargin::absolute(capped_error))
    }
}

impl Neg for BoundedValue<f64> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        BoundedValue {
            value: -self.value,
            error: self.error,
        }
    }
}

impl<T: Default + Copy> Default for BoundedValue<T> {
    fn default() -> Self {
        Self::exact(T::default())
    }
}

impl<T: fmt::Display + Copy> fmt::Display for BoundedValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.error)
    }
}

impl<T: PartialEq + Copy> PartialEq for BoundedValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.error == other.error
    }
}

/// Result type for bounded value operations.
pub type BoundedValueResult = HelixResult<BoundedValue<f64>>;

/// Trait for types that can be converted to bounded values.
pub trait IntoBounded {
    /// Converts this value to a bounded value with zero error.
    fn into_exact(self) -> BoundedValue<f64>;

    /// Converts this value to a bounded value with the given absolute error.
    fn with_error(self, error: f64) -> BoundedValue<f64>;
}

impl IntoBounded for f64 {
    fn into_exact(self) -> BoundedValue<f64> {
        BoundedValue::exact(self)
    }

    fn with_error(self, error: f64) -> BoundedValue<f64> {
        BoundedValue::<f64>::with_absolute_error(self, error)
    }
}

impl IntoBounded for f32 {
    fn into_exact(self) -> BoundedValue<f64> {
        BoundedValue::exact(self as f64)
    }

    fn with_error(self, error: f64) -> BoundedValue<f64> {
        BoundedValue::<f64>::with_absolute_error(self as f64, error)
    }
}

impl IntoBounded for i32 {
    fn into_exact(self) -> BoundedValue<f64> {
        BoundedValue::exact(self as f64)
    }

    fn with_error(self, error: f64) -> BoundedValue<f64> {
        BoundedValue::<f64>::with_absolute_error(self as f64, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_value() {
        let v: BoundedValue<f64> = BoundedValue::exact(5.0_f64);
        assert_eq!(v.value(), 5.0);
        assert_eq!(v.absolute_error(), 0.0);
        assert!(v.is_valid());
    }

    #[test]
    fn test_bounded_value() {
        let v = BoundedValue::<f64>::with_absolute_error(10.0, 0.1);
        assert_eq!(v.lower_bound(), 9.9);
        assert_eq!(v.upper_bound(), 10.1);
        assert!(v.contains(10.0));
        assert!(v.contains(9.95));
        assert!(!v.contains(9.8));
    }

    #[test]
    fn test_add() {
        let a = BoundedValue::<f64>::with_absolute_error(10.0, 0.1);
        let b = BoundedValue::<f64>::with_absolute_error(5.0, 0.05);
        let c = a + b;
        assert_eq!(c.value(), 15.0);
        assert!((c.absolute_error() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_mul() {
        let a = BoundedValue::<f64>::with_absolute_error(2.0, 0.01);
        let b = BoundedValue::<f64>::with_absolute_error(3.0, 0.02);
        let c = a * b;
        assert_eq!(c.value(), 6.0);
        // Error: |2|*0.02 + |3|*0.01 + 0.01*0.02 = 0.04 + 0.03 + 0.0002 = 0.0702
        assert!((c.absolute_error() - 0.0702).abs() < 1e-10);
    }

    #[test]
    fn test_overlaps() {
        let a = BoundedValue::<f64>::with_absolute_error(10.0, 0.5);
        let b = BoundedValue::<f64>::with_absolute_error(10.8, 0.5);
        assert!(a.overlaps(&b));

        let c = BoundedValue::<f64>::with_absolute_error(12.0, 0.5);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_nan_detection() {
        let nan_val = BoundedValue::exact(f64::NAN);
        assert!(nan_val.is_nan());
        assert!(!nan_val.is_finite());
        assert!(!nan_val.is_valid());

        // Checked operations should fail on NaN
        let normal = BoundedValue::exact(1.0);
        assert!(normal.checked_add(nan_val).is_err());
        assert!(normal.checked_mul(nan_val).is_err());
    }

    #[test]
    fn test_infinity_detection() {
        let inf_val = BoundedValue::exact(f64::INFINITY);
        assert!(inf_val.is_infinite());
        assert!(!inf_val.is_finite());
        assert!(!inf_val.is_valid());

        let normal = BoundedValue::exact(1.0);
        assert!(normal.checked_add(inf_val).is_err());
    }

    #[test]
    fn test_division_by_zero() {
        let a = BoundedValue::exact(10.0);
        let b = BoundedValue::exact(0.0);
        assert!(a.checked_div(b).is_err());

        // Regular division caps error at MAX_ERROR_BOUND instead of Infinity
        let result = a / b;
        assert_eq!(result.absolute_error(), MAX_ERROR_BOUND);
    }

    #[test]
    fn test_sanitize() {
        let mut nan_val = BoundedValue::exact(f64::NAN);
        nan_val.sanitize();
        assert_eq!(nan_val.value(), 0.0);
        assert!(nan_val.is_valid());

        let mut inf_val = BoundedValue::exact(f64::INFINITY);
        inf_val.sanitize();
        assert!(inf_val.is_finite());
        assert!(inf_val.is_valid());
    }

    #[test]
    fn test_saturating_operations() {
        let a = BoundedValue::exact(f64::MAX / 2.0);
        let b = BoundedValue::exact(f64::MAX / 2.0);

        // Saturating add should not panic or return NaN
        let result = a.saturating_add(b);
        assert!(result.is_finite());

        // Saturating with NaN should sanitize
        let nan = BoundedValue::exact(f64::NAN);
        let result = a.saturating_add(nan);
        assert!(result.is_finite());
    }

    #[test]
    fn test_checked_operations() {
        let a = BoundedValue::exact(10.0);
        let b = BoundedValue::exact(5.0);

        assert!(a.checked_add(b).is_ok());
        assert!(a.checked_sub(b).is_ok());
        assert!(a.checked_mul(b).is_ok());
        assert!(a.checked_div(b).is_ok());

        let zero = BoundedValue::exact(0.0);
        assert!(a.checked_div(zero).is_err());
    }

    #[test]
    fn test_error_accumulation() {
        let result = BoundedValue::accumulate_error(0.5, 0.3);
        assert_eq!(result, Ok(0.8));

        // MAX_ERROR_BOUND + MAX_ERROR_BOUND exceeds MAX_ERROR_BOUND
        let overflow = BoundedValue::accumulate_error(MAX_ERROR_BOUND, MAX_ERROR_BOUND);
        assert_eq!(overflow, Err(MAX_ERROR_BOUND));

        // Test infinity case
        let inf_overflow = BoundedValue::accumulate_error(f64::MAX, f64::MAX);
        assert_eq!(inf_overflow, Err(MAX_ERROR_BOUND));
    }

    #[test]
    fn test_threshold_check() {
        let v = BoundedValue::<f64>::with_absolute_error(10.0, 0.5);
        assert!(v.check_threshold(1.0).is_ok());
        assert!(v.check_threshold(0.1).is_err());
    }

    #[test]
    fn test_validated() {
        let valid = BoundedValue::exact(10.0);
        assert!(valid.validated().is_ok());

        let invalid = BoundedValue::exact(f64::NAN);
        assert!(invalid.validated().is_err());
    }

    #[test]
    fn test_try_with_error() {
        assert!(BoundedValue::try_with_absolute_error(10.0, 0.1).is_ok());
        assert!(BoundedValue::try_with_absolute_error(f64::NAN, 0.1).is_err());
        assert!(BoundedValue::try_with_absolute_error(10.0, -0.1).is_err());
    }

    #[test]
    fn test_into_bounded() {
        let v: BoundedValue<f64> = 5.0_f64.into_exact();
        assert_eq!(v.value(), 5.0);
        assert_eq!(v.absolute_error(), 0.0);

        let v: BoundedValue<f64> = 5.0_f64.with_error(0.1);
        assert_eq!(v.value(), 5.0);
        assert_eq!(v.absolute_error(), 0.1);
    }

    #[test]
    fn test_error_bound_clamping() {
        let mut v = BoundedValue::new(10.0, ErrorMargin::absolute(f64::INFINITY));
        v.clamp_error();
        assert_eq!(v.absolute_error(), MAX_ERROR_BOUND);
    }

    #[test]
    fn test_failed_computation() {
        let v = BoundedValue::failed_computation();
        assert_eq!(v.value(), 0.0);
        assert_eq!(v.absolute_error(), MAX_ERROR_BOUND);
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid BoundedValue<f64> instances.
    fn bounded_value_strategy() -> impl Strategy<Value = BoundedValue<f64>> {
        (-1e6..1e6f64, 0.0..1e3f64).prop_map(|(value, error)| {
            BoundedValue::new(value, ErrorMargin::absolute(error))
        })
    }

    /// Strategy for non-zero BoundedValue (for division).
    fn nonzero_bounded_value_strategy() -> impl Strategy<Value = BoundedValue<f64>> {
        prop_oneof![
            (0.001..1e6f64, 0.0..1e3f64).prop_map(|(v, e)| BoundedValue::new(v, ErrorMargin::absolute(e))),
            (-1e6..-0.001f64, 0.0..1e3f64).prop_map(|(v, e)| BoundedValue::new(v, ErrorMargin::absolute(e))),
        ]
    }

    proptest! {
        /// Error monotonicity: addition error >= max of individual errors.
        #[test]
        fn prop_addition_error_monotonicity(
            a in bounded_value_strategy(),
            b in bounded_value_strategy()
        ) {
            let sum = a.saturating_add(b);
            let sum_error = sum.absolute_error();
            // Addition error should be at least as large as either input error
            // (absolute errors add: εa + εb >= max(εa, εb))
            prop_assert!(
                sum_error >= a.absolute_error().min(b.absolute_error()) - 1e-10,
                "Sum error {} should be >= min({}, {})",
                sum_error, a.absolute_error(), b.absolute_error()
            );
        }

        /// Commutativity of addition: a + b ≈ b + a in both value and error.
        #[test]
        fn prop_addition_commutativity(
            a in bounded_value_strategy(),
            b in bounded_value_strategy()
        ) {
            let ab = a.saturating_add(b);
            let ba = b.saturating_add(a);
            prop_assert!(
                (ab.value() - ba.value()).abs() < 1e-10,
                "Values differ: {} vs {}", ab.value(), ba.value()
            );
            prop_assert!(
                (ab.absolute_error() - ba.absolute_error()).abs() < 1e-10,
                "Errors differ: {} vs {}", ab.absolute_error(), ba.absolute_error()
            );
        }

        /// Error bound validity: error is never negative or NaN.
        #[test]
        fn prop_error_never_negative_or_nan(
            a in bounded_value_strategy(),
            b in bounded_value_strategy()
        ) {
            let sum = a.saturating_add(b);
            prop_assert!(!sum.absolute_error().is_nan(), "Sum error is NaN");
            prop_assert!(sum.absolute_error() >= 0.0, "Sum error is negative: {}", sum.absolute_error());

            let product = a.saturating_mul(b);
            prop_assert!(!product.absolute_error().is_nan(), "Product error is NaN");
            prop_assert!(product.absolute_error() >= 0.0, "Product error is negative: {}", product.absolute_error());

            let diff = a.saturating_sub(b);
            prop_assert!(!diff.absolute_error().is_nan(), "Diff error is NaN");
            prop_assert!(diff.absolute_error() >= 0.0, "Diff error is negative: {}", diff.absolute_error());
        }

        /// Multiplication commutativity.
        #[test]
        fn prop_multiplication_commutativity(
            a in bounded_value_strategy(),
            b in bounded_value_strategy()
        ) {
            let ab = a.saturating_mul(b);
            let ba = b.saturating_mul(a);
            let value_diff = (ab.value() - ba.value()).abs();
            prop_assert!(
                value_diff < 1e-6,
                "Multiplication not commutative: {} vs {} (diff={})",
                ab.value(), ba.value(), value_diff
            );
        }

        /// Division error never negative or NaN for non-zero divisors.
        #[test]
        fn prop_division_error_valid(
            a in bounded_value_strategy(),
            b in nonzero_bounded_value_strategy()
        ) {
            let result = a.saturating_div(b);
            prop_assert!(!result.absolute_error().is_nan(), "Division error is NaN");
            prop_assert!(result.absolute_error() >= 0.0, "Division error is negative");
            prop_assert!(result.value().is_finite(), "Division value not finite");
        }

        /// Saturating operations clamp rather than overflow.
        #[test]
        fn prop_saturating_ops_clamp(
            a in bounded_value_strategy(),
            b in bounded_value_strategy()
        ) {
            let sum = a.saturating_add(b);
            prop_assert!(
                sum.absolute_error() <= MAX_ERROR_BOUND,
                "Sum error {} exceeds MAX_ERROR_BOUND {}", sum.absolute_error(), MAX_ERROR_BOUND
            );

            let product = a.saturating_mul(b);
            prop_assert!(
                product.absolute_error() <= MAX_ERROR_BOUND,
                "Product error {} exceeds MAX_ERROR_BOUND {}", product.absolute_error(), MAX_ERROR_BOUND
            );
        }
    }
}
