//! Bounded value type for approximate computation.
//!
//! A `BoundedValue<T>` represents a computed value together with its error bounds.

use super::error_margin::ErrorMargin;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A value with tracked error bounds.
///
/// The true value is guaranteed to be within `[value - error, value + error]`.
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
        let value = self.value / rhs.value;
        // Division error: ε(a/b) ≈ |a/b| * (εa/|a| + εb/|b|) for small relative errors
        let eps_a = self.error.to_absolute(self.value);
        let eps_b = rhs.error.to_absolute(rhs.value);
        
        // Guard against division by near-zero
        let b_abs = rhs.value.abs();
        if b_abs < 1e-15 {
            return BoundedValue::new(value, ErrorMargin::absolute(f64::INFINITY));
        }
        
        let rel_error = eps_a / self.value.abs().max(1e-15) + eps_b / b_abs;
        let abs_error = value.abs() * rel_error + eps_a / b_abs;
        
        BoundedValue::new(value, ErrorMargin::absolute(abs_error))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_value() {
        let v: BoundedValue<f64> = BoundedValue::exact(5.0_f64);
        assert_eq!(v.value(), 5.0);
        assert_eq!(v.absolute_error(), 0.0);
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
}
