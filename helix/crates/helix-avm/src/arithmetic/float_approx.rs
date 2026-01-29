//! Approximate floating point operations with error tracking.

use helix_core::types::{BoundedValue, ErrorMargin, Precision};

/// Approximate addition with error tracking.
pub fn approx_add(a: BoundedValue<f64>, b: BoundedValue<f64>, precision: Precision) -> BoundedValue<f64> {
    let result = a.value() + b.value();
    let precision_error = result.abs() * precision.max_relative_error();
    let total_error = a.absolute_error() + b.absolute_error() + precision_error;
    BoundedValue::new(result, ErrorMargin::absolute(total_error))
}

/// Approximate subtraction with error tracking.
pub fn approx_sub(a: BoundedValue<f64>, b: BoundedValue<f64>, precision: Precision) -> BoundedValue<f64> {
    let result = a.value() - b.value();
    let precision_error = result.abs() * precision.max_relative_error();
    let total_error = a.absolute_error() + b.absolute_error() + precision_error;
    BoundedValue::new(result, ErrorMargin::absolute(total_error))
}

/// Approximate multiplication with error tracking.
pub fn approx_mul(a: BoundedValue<f64>, b: BoundedValue<f64>, precision: Precision) -> BoundedValue<f64> {
    let result = a.value() * b.value();

    // Error propagation: |a|*Δb + |b|*Δa + Δa*Δb + round_err
    let base_error = a.value().abs() * b.absolute_error()
        + b.value().abs() * a.absolute_error()
        + a.absolute_error() * b.absolute_error();
    let precision_error = result.abs() * precision.max_relative_error();

    BoundedValue::new(result, ErrorMargin::absolute(base_error + precision_error))
}

/// Approximate division with error tracking.
pub fn approx_div(
    a: BoundedValue<f64>,
    b: BoundedValue<f64>,
    precision: Precision,
) -> Option<BoundedValue<f64>> {
    // Check for division by zero (considering error bounds)
    if b.contains(0.0) {
        return None;
    }

    let result = a.value() / b.value();

    // Error: (Δa*|b| + |a|*Δb) / b² + round_err
    let b_squared = b.value() * b.value();
    let base_error = (a.absolute_error() * b.value().abs() + a.value().abs() * b.absolute_error())
        / b_squared;
    let precision_error = result.abs() * precision.max_relative_error();

    Some(BoundedValue::new(result, ErrorMargin::absolute(base_error + precision_error)))
}

/// Approximate square root with error tracking.
pub fn approx_sqrt(a: BoundedValue<f64>, precision: Precision) -> Option<BoundedValue<f64>> {
    if a.value() < 0.0 {
        return None;
    }

    let result = a.value().sqrt();

    // d(sqrt(x))/dx = 1/(2*sqrt(x))
    let derivative = if result > 0.0 { 0.5 / result } else { 0.0 };
    let base_error = derivative * a.absolute_error();
    let precision_error = result * precision.max_relative_error();

    Some(BoundedValue::new(result, ErrorMargin::absolute(base_error + precision_error)))
}

/// Approximate exponential with error tracking.
pub fn approx_exp(a: BoundedValue<f64>, precision: Precision) -> BoundedValue<f64> {
    let result = a.value().exp();

    // d(exp(x))/dx = exp(x)
    let base_error = result * a.absolute_error();
    let precision_error = result * precision.max_relative_error();

    BoundedValue::new(result, ErrorMargin::absolute(base_error + precision_error))
}

/// Approximate natural logarithm with error tracking.
pub fn approx_ln(a: BoundedValue<f64>, precision: Precision) -> Option<BoundedValue<f64>> {
    if a.value() <= 0.0 {
        return None;
    }

    let result = a.value().ln();

    // d(ln(x))/dx = 1/x
    let base_error = a.absolute_error() / a.value();
    let precision_error = result.abs() * precision.max_relative_error();

    Some(BoundedValue::new(result, ErrorMargin::absolute(base_error + precision_error)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_add() {
        let a = BoundedValue::<f64>::exact(1.0);
        let b = BoundedValue::<f64>::exact(2.0);
        let c = approx_add(a, b, Precision::F32);
        assert_eq!(c.value(), 3.0);
        assert!(c.absolute_error() < 1e-6);
    }

    #[test]
    fn test_approx_mul() {
        let a = BoundedValue::<f64>::exact(2.0);
        let b = BoundedValue::<f64>::exact(3.0);
        let c = approx_mul(a, b, Precision::F32);
        assert_eq!(c.value(), 6.0);
    }

    #[test]
    fn test_approx_div_zero() {
        let a = BoundedValue::<f64>::exact(1.0);
        let b = BoundedValue::<f64>::exact(0.0);
        assert!(approx_div(a, b, Precision::F32).is_none());
    }
}
