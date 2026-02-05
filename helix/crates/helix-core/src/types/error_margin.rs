//! Error margin types for approximate computation.
//!
//! Supports both absolute error (±0.001) and relative error (±0.1%) representations.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents the error margin of a computed value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ErrorMargin {
    /// Absolute error: the true value is within ±epsilon of the computed value.
    Absolute(f64),
    /// Relative error: the true value is within ±(epsilon * |value|) of the computed value.
    Relative(f64),
}

impl ErrorMargin {
    /// Creates a new absolute error margin.
    ///
    /// Negative values are clamped to 0. NaN values are treated as 0.
    pub fn absolute(epsilon: f64) -> Self {
        // Handle NaN and negative values gracefully
        let safe_epsilon = if epsilon.is_nan() || epsilon < 0.0 {
            0.0
        } else {
            epsilon
        };
        ErrorMargin::Absolute(safe_epsilon)
    }

    /// Creates a new relative error margin.
    ///
    /// Negative values are clamped to 0. NaN values are treated as 0.
    pub fn relative(epsilon: f64) -> Self {
        // Handle NaN and negative values gracefully
        let safe_epsilon = if epsilon.is_nan() || epsilon < 0.0 {
            0.0
        } else {
            epsilon
        };
        ErrorMargin::Relative(safe_epsilon)
    }

    /// Creates a new absolute error margin with strict validation.
    ///
    /// Returns None if epsilon is negative or NaN.
    pub fn try_absolute(epsilon: f64) -> Option<Self> {
        if epsilon.is_nan() || epsilon < 0.0 {
            None
        } else {
            Some(ErrorMargin::Absolute(epsilon))
        }
    }

    /// Creates a new relative error margin with strict validation.
    ///
    /// Returns None if epsilon is negative or NaN.
    pub fn try_relative(epsilon: f64) -> Option<Self> {
        if epsilon.is_nan() || epsilon < 0.0 {
            None
        } else {
            Some(ErrorMargin::Relative(epsilon))
        }
    }

    /// Converts this error margin to an absolute value given the reference value.
    pub fn to_absolute(&self, reference_value: f64) -> f64 {
        match self {
            ErrorMargin::Absolute(eps) => *eps,
            ErrorMargin::Relative(eps) => eps * reference_value.abs(),
        }
    }

    /// Converts this error margin to a relative value given the reference value.
    /// Returns None if reference_value is zero (relative error undefined).
    pub fn to_relative(&self, reference_value: f64) -> Option<f64> {
        if reference_value == 0.0 {
            return None;
        }
        match self {
            ErrorMargin::Absolute(eps) => Some(eps / reference_value.abs()),
            ErrorMargin::Relative(eps) => Some(*eps),
        }
    }

    /// Returns the raw epsilon value.
    pub fn epsilon(&self) -> f64 {
        match self {
            ErrorMargin::Absolute(eps) | ErrorMargin::Relative(eps) => *eps,
        }
    }

    /// Returns true if this is an absolute error margin.
    pub fn is_absolute(&self) -> bool {
        matches!(self, ErrorMargin::Absolute(_))
    }

    /// Returns true if this is a relative error margin.
    pub fn is_relative(&self) -> bool {
        matches!(self, ErrorMargin::Relative(_))
    }

    /// Combines two error margins (for addition/subtraction).
    /// Result is always absolute.
    pub fn add(&self, other: &ErrorMargin, self_ref: f64, other_ref: f64) -> ErrorMargin {
        let abs1 = self.to_absolute(self_ref);
        let abs2 = other.to_absolute(other_ref);
        ErrorMargin::Absolute(abs1 + abs2)
    }

    /// Combines error margins for multiplication: ε(a*b) = |a|*εb + |b|*εa + εa*εb
    ///
    /// This uses the full error formula including the second-order term εa*εb,
    /// which provides tighter bounds for operations with significant error margins.
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::ErrorMargin;
    ///
    /// let a_err = ErrorMargin::absolute(0.01);
    /// let b_err = ErrorMargin::absolute(0.02);
    /// let product_err = a_err.multiply(&b_err, 10.0, 5.0);
    /// // Result: |10| * 0.02 + |5| * 0.01 + 0.01 * 0.02 = 0.2502
    /// ```
    pub fn multiply(
        &self,
        other: &ErrorMargin,
        self_val: f64,
        other_val: f64,
    ) -> ErrorMargin {
        let eps_a = self.to_absolute(self_val);
        let eps_b = other.to_absolute(other_val);
        // For multiplication: ε(ab) = |a|εb + |b|εa + εaεb
        // Includes the second-order term for conservative error estimation
        let combined = self_val.abs() * eps_b + other_val.abs() * eps_a + eps_a * eps_b;
        ErrorMargin::Absolute(combined)
    }

    /// Combines error margins for division: ε(a/b) ≈ |a/b| * (εa/|a| + εb/|b|) + εa/|b|
    ///
    /// For a/b with errors εa and εb, the propagated error accounts for:
    /// - The relative error contribution from numerator: εa/|a|
    /// - The relative error contribution from denominator: εb/|b|
    /// - The absolute error scaled by the result: |a/b| * relative_error
    /// - Direct contribution from numerator error: εa/|b|
    ///
    /// This is used in softmax, layer normalization, and other operations involving division.
    ///
    /// # Arguments
    ///
    /// * `other` - Error margin of the denominator
    /// * `self_val` - Value of the numerator
    /// * `other_val` - Value of the denominator (must be non-zero)
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::ErrorMargin;
    ///
    /// let num_err = ErrorMargin::absolute(0.01);
    /// let den_err = ErrorMargin::absolute(0.005);
    /// let div_err = num_err.divide(&den_err, 10.0, 2.0);
    /// // For a=10, b=2: result = 5.0
    /// // εa = 0.01, εb = 0.005
    /// // relative_error ≈ 0.01/10 + 0.005/2 = 0.001 + 0.0025 = 0.0035
    /// // absolute_error ≈ 5.0 * 0.0035 + 0.01/2 = 0.0175 + 0.005 = 0.0225
    /// ```
    pub fn divide(
        &self,
        other: &ErrorMargin,
        self_val: f64,
        other_val: f64,
    ) -> ErrorMargin {
        // Guard against division by zero
        let b_abs = other_val.abs();
        if b_abs < 1e-15 {
            return ErrorMargin::Absolute(f64::INFINITY);
        }

        let eps_a = self.to_absolute(self_val);
        let eps_b = other.to_absolute(other_val);

        // Result value
        let result = self_val / other_val;

        // Relative error: εa/|a| + εb/|b|
        let a_abs = self_val.abs().max(1e-15);
        let rel_error = eps_a / a_abs + eps_b / b_abs;

        // Absolute error: |result| * rel_error + εa/|b|
        let abs_error = result.abs() * rel_error + eps_a / b_abs;

        ErrorMargin::Absolute(abs_error)
    }

    /// Zero error margin (exact value).
    pub const ZERO: ErrorMargin = ErrorMargin::Absolute(0.0);
}

impl Default for ErrorMargin {
    fn default() -> Self {
        ErrorMargin::ZERO
    }
}

impl fmt::Display for ErrorMargin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorMargin::Absolute(eps) => write!(f, "±{:.2e}", eps),
            ErrorMargin::Relative(eps) => write!(f, "±{:.2}%", eps * 100.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absolute_error() {
        let err = ErrorMargin::absolute(0.01);
        assert_eq!(err.to_absolute(100.0), 0.01);
        assert_eq!(err.to_absolute(1.0), 0.01);
    }

    #[test]
    fn test_relative_error() {
        let err = ErrorMargin::relative(0.01); // 1%
        assert!((err.to_absolute(100.0) - 1.0).abs() < 1e-10);
        assert!((err.to_absolute(50.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_error_add() {
        let e1 = ErrorMargin::absolute(0.1);
        let e2 = ErrorMargin::absolute(0.2);
        let combined = e1.add(&e2, 1.0, 1.0);
        assert!((combined.epsilon() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_error_multiply() {
        let e1 = ErrorMargin::absolute(0.01);
        let e2 = ErrorMargin::absolute(0.02);
        let combined = e1.multiply(&e2, 10.0, 5.0);
        // |10| * 0.02 + |5| * 0.01 + 0.01 * 0.02 = 0.2 + 0.05 + 0.0002 = 0.2502
        assert!((combined.epsilon() - 0.2502).abs() < 1e-10);
    }

    #[test]
    fn test_error_divide() {
        let e1 = ErrorMargin::absolute(0.01);
        let e2 = ErrorMargin::absolute(0.005);
        let combined = e1.divide(&e2, 10.0, 2.0);
        // a=10, b=2, result=5
        // εa=0.01, εb=0.005
        // rel_error = 0.01/10 + 0.005/2 = 0.001 + 0.0025 = 0.0035
        // abs_error = 5 * 0.0035 + 0.01/2 = 0.0175 + 0.005 = 0.0225
        assert!((combined.epsilon() - 0.0225).abs() < 1e-10);
    }

    #[test]
    fn test_error_divide_by_near_zero() {
        let e1 = ErrorMargin::absolute(0.01);
        let e2 = ErrorMargin::absolute(0.001);
        let combined = e1.divide(&e2, 1.0, 1e-20);
        // Division by near-zero should return infinite error
        assert!(combined.epsilon().is_infinite());
    }

    #[test]
    fn test_error_divide_exact_values() {
        // When both values have zero error, result should also have zero error
        let e1 = ErrorMargin::absolute(0.0);
        let e2 = ErrorMargin::absolute(0.0);
        let combined = e1.divide(&e2, 10.0, 2.0);
        assert!((combined.epsilon() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_error_divide_relative() {
        // Test with relative errors
        let e1 = ErrorMargin::relative(0.01); // 1% relative error
        let e2 = ErrorMargin::relative(0.02); // 2% relative error
        let combined = e1.divide(&e2, 100.0, 10.0);
        // For relative errors, the errors scale with the values
        // a=100, b=10, result=10
        // εa = 0.01 * 100 = 1.0, εb = 0.02 * 10 = 0.2
        // rel_error = 1.0/100 + 0.2/10 = 0.01 + 0.02 = 0.03
        // abs_error = 10 * 0.03 + 1.0/10 = 0.3 + 0.1 = 0.4
        assert!((combined.epsilon() - 0.4).abs() < 1e-10);
    }
}
