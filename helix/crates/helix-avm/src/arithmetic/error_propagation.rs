//! Error propagation rules for compound operations.

use helix_core::types::ErrorMargin;

/// Minimum safe denominator to prevent error explosion from near-zero division.
/// Approximately sqrt(f64::EPSILON).
const MIN_SAFE_DENOMINATOR: f64 = 1e-15;

/// Maximum propagated error to prevent unbounded growth.
const MAX_PROPAGATED_ERROR: f64 = 1e10;

/// Propagates error through addition: Δ(a+b) = Δa + Δb
pub fn propagate_add(a_error: f64, b_error: f64) -> f64 {
    a_error + b_error
}

/// Propagates error through subtraction: Δ(a-b) = Δa + Δb
pub fn propagate_sub(a_error: f64, b_error: f64) -> f64 {
    a_error + b_error
}

/// Propagates error through multiplication: Δ(ab) ≈ |a|Δb + |b|Δa
pub fn propagate_mul(a: f64, a_error: f64, b: f64, b_error: f64) -> f64 {
    a.abs() * b_error + b.abs() * a_error + a_error * b_error
}

/// Propagates error through division: Δ(a/b) ≈ (|b|Δa + |a|Δb) / b²
///
/// Near-zero denominators are clamped to `MIN_SAFE_DENOMINATOR` to prevent
/// error explosion. The result is saturated to `MAX_PROPAGATED_ERROR`.
/// Exact zero still returns `INFINITY`.
pub fn propagate_div(a: f64, a_error: f64, b: f64, b_error: f64) -> f64 {
    if b == 0.0 {
        return f64::INFINITY;
    }
    // Clamp denominator magnitude to prevent explosion from near-zero values
    let b_clamped = if b.abs() < MIN_SAFE_DENOMINATOR {
        b.signum() * MIN_SAFE_DENOMINATOR
    } else {
        b
    };
    let result = (b_clamped.abs() * a_error + a.abs() * b_error) / (b_clamped * b_clamped);
    // Saturate to prevent unbounded growth
    result.min(MAX_PROPAGATED_ERROR)
}

/// Propagates error through a chain of N additions.
/// Total error grows linearly: sum of all individual errors.
pub fn propagate_sum_chain(errors: &[f64]) -> f64 {
    errors.iter().sum()
}

/// Propagates error for dot product of two vectors.
/// Each element i contributes: |a_i|*Δb_i + |b_i|*Δa_i
pub fn propagate_dot_product(
    a_values: &[f64],
    a_errors: &[f64],
    b_values: &[f64],
    b_errors: &[f64],
) -> f64 {
    a_values
        .iter()
        .zip(a_errors)
        .zip(b_values.iter().zip(b_errors))
        .map(|((a, ae), (b, be))| propagate_mul(*a, *ae, *b, *be))
        .sum()
}

/// Computes the error bound for matrix multiplication.
/// For C = A @ B where A is (m x k) and B is (k x n):
/// Each element C[i,j] has error from k multiply-accumulates.
pub fn propagate_matmul(
    k: usize,
    max_a_value: f64,
    max_a_error: f64,
    max_b_value: f64,
    max_b_error: f64,
    precision_error: f64,
) -> f64 {
    let k = k as f64;
    let mul_error = propagate_mul(max_a_value, max_a_error, max_b_value, max_b_error);
    let accumulation_error = k * mul_error;
    let rounding_error = k * max_a_value * max_b_value * precision_error;
    accumulation_error + rounding_error
}

/// Combines multiple error margins.
pub fn combine_errors(errors: &[ErrorMargin]) -> ErrorMargin {
    let total = errors.iter().map(|e| e.to_absolute(1.0)).sum();
    ErrorMargin::absolute(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_propagation() {
        let result = propagate_add(0.1, 0.2);
        assert!((result - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_mul_propagation() {
        // (2 ± 0.1) * (3 ± 0.2) = 6 ± (2*0.2 + 3*0.1 + 0.1*0.2)
        let error = propagate_mul(2.0, 0.1, 3.0, 0.2);
        assert!((error - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_div_exact_zero_returns_infinity() {
        let result = propagate_div(1.0, 0.01, 0.0, 0.01);
        assert!(result.is_infinite());
    }

    #[test]
    fn test_div_near_zero_positive() {
        // Near-zero positive denominator should not explode
        let result = propagate_div(1.0, 0.01, 1e-300, 0.01);
        assert!(result.is_finite());
        assert!(result <= MAX_PROPAGATED_ERROR);
        assert!(result > 0.0);
    }

    #[test]
    fn test_div_near_zero_negative() {
        // Near-zero negative denominator should also be clamped
        let result = propagate_div(1.0, 0.01, -1e-300, 0.01);
        assert!(result.is_finite());
        assert!(result <= MAX_PROPAGATED_ERROR);
        assert!(result > 0.0);
    }

    #[test]
    fn test_div_normal_values_unchanged() {
        // For normal denominators, result should match the standard formula
        let a = 6.0;
        let a_err = 0.1;
        let b = 3.0;
        let b_err = 0.05;
        let result = propagate_div(a, a_err, b, b_err);
        let expected = (b.abs() * a_err + a.abs() * b_err) / (b * b);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_div_saturation() {
        // Very small denominator with large numerator error should saturate
        let result = propagate_div(1e20, 1e20, 1e-14, 1e-14);
        assert!(result <= MAX_PROPAGATED_ERROR);
    }
}
