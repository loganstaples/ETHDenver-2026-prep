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

    /// Combines error margins for multiplication: ε(a*b) ≈ |a|*εb + |b|*εa
    pub fn multiply(
        &self,
        other: &ErrorMargin,
        self_val: f64,
        other_val: f64,
    ) -> ErrorMargin {
        let eps_a = self.to_absolute(self_val);
        let eps_b = other.to_absolute(other_val);
        // For multiplication: ε(ab) ≈ |a|εb + |b|εa + εaεb
        // We use the approximation without the second-order term for simplicity
        let combined = self_val.abs() * eps_b + other_val.abs() * eps_a + eps_a * eps_b;
        ErrorMargin::Absolute(combined)
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
}
