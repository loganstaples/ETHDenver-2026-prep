//! Probabilistic error bounds with confidence intervals.
//!
//! Unlike worst-case error bounds that track only the maximum possible deviation,
//! probabilistic error bounds model the statistical distribution of errors. This enables:
//! - Confidence intervals (e.g., 95% of values within this range)
//! - Tighter bounds for common cases while still tracking worst-case
//! - Monte Carlo error estimation
//! - Stochastic rounding analysis

use serde::{Deserialize, Serialize};
use std::fmt;

/// A probabilistic error bound that tracks both statistical and worst-case properties.
///
/// Models error as a distribution, typically assumed to be approximately Gaussian
/// for accumulated errors (by Central Limit Theorem) or uniform for quantization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProbabilisticError {
    /// Mean of the error distribution (typically 0 for unbiased operations).
    pub mean: f64,
    /// Standard deviation of the error distribution.
    pub std_dev: f64,
    /// Worst-case absolute bound (for hard guarantees).
    pub worst_case: f64,
    /// Number of samples/operations this error accumulated over.
    pub sample_count: usize,
    /// Distribution type for correct propagation rules.
    pub distribution: ErrorDistribution,
}

/// Type of error distribution for correct propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorDistribution {
    /// Uniform distribution (e.g., quantization error).
    Uniform,
    /// Gaussian/Normal distribution (accumulated errors).
    Gaussian,
    /// Unknown/worst-case (use conservative bounds).
    Unknown,
    /// Bounded uniform (known min/max).
    BoundedUniform { min_bound: i64, max_bound: i64 },
}

impl Default for ErrorDistribution {
    fn default() -> Self {
        ErrorDistribution::Unknown
    }
}

/// Confidence level for interval computation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// 68.27% confidence (1 standard deviation).
    OneStdDev,
    /// 90% confidence.
    Ninety,
    /// 95% confidence.
    NinetyFive,
    /// 99% confidence.
    NinetyNine,
    /// 99.7% confidence (3 standard deviations).
    ThreeStdDev,
    /// Custom confidence level (0.0 to 1.0).
    Custom(f64),
}

impl ConfidenceLevel {
    /// Returns the z-score for this confidence level (assuming Gaussian).
    pub fn z_score(&self) -> f64 {
        match self {
            ConfidenceLevel::OneStdDev => 1.0,
            ConfidenceLevel::Ninety => 1.645,
            ConfidenceLevel::NinetyFive => 1.96,
            ConfidenceLevel::NinetyNine => 2.576,
            ConfidenceLevel::ThreeStdDev => 3.0,
            ConfidenceLevel::Custom(p) => {
                // Approximate inverse normal CDF using Acklam's algorithm
                inverse_normal_cdf(0.5 + *p / 2.0)
            }
        }
    }

    /// Returns the confidence probability (0.0 to 1.0).
    pub fn probability(&self) -> f64 {
        match self {
            ConfidenceLevel::OneStdDev => 0.6827,
            ConfidenceLevel::Ninety => 0.90,
            ConfidenceLevel::NinetyFive => 0.95,
            ConfidenceLevel::NinetyNine => 0.99,
            ConfidenceLevel::ThreeStdDev => 0.9973,
            ConfidenceLevel::Custom(p) => *p,
        }
    }
}

/// Confidence interval for a value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound of the interval.
    pub lower: f64,
    /// Upper bound of the interval.
    pub upper: f64,
    /// Center value.
    pub center: f64,
    /// Confidence level.
    pub confidence: f64,
}

impl ConfidenceInterval {
    /// Creates a new confidence interval.
    pub fn new(center: f64, lower: f64, upper: f64, confidence: f64) -> Self {
        Self {
            lower,
            upper,
            center,
            confidence,
        }
    }

    /// Returns the width of the interval.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Returns the half-width (radius) of the interval.
    pub fn half_width(&self) -> f64 {
        self.width() / 2.0
    }

    /// Checks if a value falls within this interval.
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Checks if this interval overlaps with another.
    pub fn overlaps(&self, other: &ConfidenceInterval) -> bool {
        self.lower <= other.upper && other.lower <= self.upper
    }
}

impl fmt::Display for ConfidenceInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.6}, {:.6}] ({:.1}% confidence)",
            self.lower,
            self.upper,
            self.confidence * 100.0
        )
    }
}

impl ProbabilisticError {
    /// Creates a new probabilistic error bound.
    pub fn new(mean: f64, std_dev: f64, worst_case: f64, distribution: ErrorDistribution) -> Self {
        Self {
            mean,
            std_dev: std_dev.abs(),
            worst_case: worst_case.abs(),
            sample_count: 1,
            distribution,
        }
    }

    /// Creates a zero error bound (exact value).
    pub fn zero() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            worst_case: 0.0,
            sample_count: 0,
            distribution: ErrorDistribution::Gaussian,
        }
    }

    /// Creates an error bound from uniform quantization.
    ///
    /// For uniform quantization with step size `delta`, the error is uniformly
    /// distributed in [-delta/2, delta/2].
    pub fn from_quantization(delta: f64) -> Self {
        let half_delta = delta.abs() / 2.0;
        // For uniform distribution: variance = (b-a)^2/12 = delta^2/12
        let variance = delta * delta / 12.0;
        Self {
            mean: 0.0,
            std_dev: variance.sqrt(),
            worst_case: half_delta,
            sample_count: 1,
            distribution: ErrorDistribution::Uniform,
        }
    }

    /// Creates an error bound from machine epsilon.
    pub fn from_relative_epsilon(value: f64, epsilon: f64) -> Self {
        let abs_error = value.abs() * epsilon;
        // For floating-point rounding, model as uniform in [-epsilon*|x|, epsilon*|x|]
        Self::from_quantization(2.0 * abs_error)
    }

    /// Creates an error from an absolute bound.
    pub fn from_absolute(bound: f64) -> Self {
        // Assume uniform distribution within bounds
        Self::from_quantization(2.0 * bound)
    }

    /// Returns the variance of the error distribution.
    pub fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }

    /// Returns the coefficient of variation (relative std dev).
    /// Returns None if mean is zero.
    pub fn coefficient_of_variation(&self) -> Option<f64> {
        if self.mean.abs() < f64::EPSILON {
            None
        } else {
            Some(self.std_dev / self.mean.abs())
        }
    }

    /// Computes a confidence interval at the specified confidence level.
    pub fn confidence_interval(&self, value: f64, level: ConfidenceLevel) -> ConfidenceInterval {
        let z = level.z_score();
        let half_width = z * self.std_dev;

        ConfidenceInterval::new(
            value + self.mean,
            value + self.mean - half_width,
            value + self.mean + half_width,
            level.probability(),
        )
    }

    /// Returns the probability that the true value is within `threshold` of the computed value.
    pub fn probability_within(&self, threshold: f64) -> f64 {
        if threshold <= 0.0 {
            return 0.0;
        }
        if threshold >= self.worst_case {
            return 1.0;
        }

        match self.distribution {
            ErrorDistribution::Uniform | ErrorDistribution::BoundedUniform { .. } => {
                // For uniform: P(|X| < t) = min(1, t / worst_case)
                (threshold / self.worst_case).min(1.0)
            }
            ErrorDistribution::Gaussian | ErrorDistribution::Unknown => {
                // For Gaussian: use error function
                if self.std_dev == 0.0 {
                    return if threshold > self.mean.abs() { 1.0 } else { 0.0 };
                }
                erf((threshold - self.mean) / (self.std_dev * std::f64::consts::SQRT_2))
            }
        }
    }

    /// Combines this error with another for addition/subtraction.
    ///
    /// For independent random variables:
    /// - Variances add
    /// - Means add
    /// - Worst cases add
    pub fn add(&self, other: &ProbabilisticError) -> ProbabilisticError {
        let mean = self.mean + other.mean;
        // For independent variables: Var(X + Y) = Var(X) + Var(Y)
        let variance = self.variance() + other.variance();
        let std_dev = variance.sqrt();
        let worst_case = self.worst_case + other.worst_case;

        // Result distribution: if both Gaussian, result is Gaussian
        // Otherwise, by CLT with enough samples, tends toward Gaussian
        let distribution = match (&self.distribution, &other.distribution) {
            (ErrorDistribution::Gaussian, ErrorDistribution::Gaussian) => {
                ErrorDistribution::Gaussian
            }
            _ => {
                if self.sample_count + other.sample_count > 10 {
                    // CLT approximation
                    ErrorDistribution::Gaussian
                } else {
                    ErrorDistribution::Unknown
                }
            }
        };

        ProbabilisticError {
            mean,
            std_dev,
            worst_case,
            sample_count: self.sample_count + other.sample_count,
            distribution,
        }
    }

    /// Combines error for multiplication: error(a*b) given values a, b and their errors.
    ///
    /// For small relative errors: Var(XY) ≈ a²Var(Y) + b²Var(X)
    pub fn multiply(
        &self,
        other: &ProbabilisticError,
        self_value: f64,
        other_value: f64,
    ) -> ProbabilisticError {
        let a = self_value;
        let b = other_value;

        // Mean of product (for independent variables with zero-mean errors):
        // E[(a + εa)(b + εb)] = ab + E[εa]b + E[εb]a + E[εaεb]
        // With zero correlation: E[εaεb] = E[εa]E[εb]
        let mean = self.mean * other_value + other.mean * self_value + self.mean * other.mean;

        // Variance using delta method for products:
        // Var(XY) ≈ b²Var(X) + a²Var(Y) + Var(X)Var(Y) for small errors
        let var_x = self.variance();
        let var_y = other.variance();
        let variance = b * b * var_x + a * a * var_y + var_x * var_y;
        let std_dev = variance.sqrt();

        // Worst case: |(a ± εa)(b ± εb) - ab| ≤ |a|εb + |b|εa + εaεb
        let worst_case =
            a.abs() * other.worst_case + b.abs() * self.worst_case
            + self.worst_case * other.worst_case;

        ProbabilisticError {
            mean,
            std_dev,
            worst_case,
            sample_count: self.sample_count.max(other.sample_count),
            distribution: ErrorDistribution::Gaussian, // Product of many terms → Gaussian
        }
    }

    /// Combines error for division: error(a/b) given values a, b and their errors.
    ///
    /// Requires b to be sufficiently far from zero.
    pub fn divide(
        &self,
        other: &ProbabilisticError,
        self_value: f64,
        other_value: f64,
    ) -> ProbabilisticError {
        let a = self_value;
        let b = other_value;

        if b.abs() < f64::EPSILON {
            // Division by near-zero: return infinite error
            return ProbabilisticError {
                mean: 0.0,
                std_dev: f64::INFINITY,
                worst_case: f64::INFINITY,
                sample_count: 1,
                distribution: ErrorDistribution::Unknown,
            };
        }

        // For a/b where a, b have errors:
        // Using delta method: Var(a/b) ≈ (1/b²)Var(a) + (a²/b⁴)Var(b)
        let var_a = self.variance();
        let var_b = other.variance();
        let b2 = b * b;
        let b4 = b2 * b2;
        let variance = var_a / b2 + (a * a * var_b) / b4;
        let std_dev = variance.sqrt();

        // Worst case: complex, use conservative bound
        // |a/b - (a+εa)/(b+εb)| with Taylor expansion
        let eps_a = self.worst_case;
        let eps_b = other.worst_case;
        let worst_case = if b.abs() > eps_b {
            (eps_a * b.abs() + eps_b * a.abs()) / (b.abs() * (b.abs() - eps_b))
        } else {
            f64::INFINITY
        };

        ProbabilisticError {
            mean: self.mean / b - (a * other.mean) / b2,
            std_dev,
            worst_case,
            sample_count: self.sample_count.max(other.sample_count),
            distribution: ErrorDistribution::Gaussian,
        }
    }

    /// Scales the error by a constant factor.
    pub fn scale(&self, factor: f64) -> ProbabilisticError {
        ProbabilisticError {
            mean: self.mean * factor,
            std_dev: self.std_dev * factor.abs(),
            worst_case: self.worst_case * factor.abs(),
            sample_count: self.sample_count,
            distribution: self.distribution,
        }
    }

    /// Accumulates n identical independent errors.
    ///
    /// By CLT, sum of n iid random variables has:
    /// - Mean = n * mean
    /// - Variance = n * variance
    /// - Worst case = n * worst_case
    pub fn accumulate(&self, n: usize) -> ProbabilisticError {
        if n == 0 {
            return ProbabilisticError::zero();
        }

        let n_f = n as f64;
        ProbabilisticError {
            mean: self.mean * n_f,
            // Std dev of sum = sqrt(n) * std_dev
            std_dev: self.std_dev * n_f.sqrt(),
            worst_case: self.worst_case * n_f,
            sample_count: self.sample_count * n,
            distribution: if n > 10 {
                ErrorDistribution::Gaussian // CLT kicks in
            } else {
                self.distribution
            },
        }
    }

    /// Computes error propagation through a nonlinear function using linearization.
    ///
    /// For f(x) where x has error ε:
    /// error(f(x)) ≈ |f'(x)| * ε
    pub fn through_function(&self, value: f64, derivative: f64) -> ProbabilisticError {
        let d_abs = derivative.abs();
        ProbabilisticError {
            mean: self.mean * derivative,
            std_dev: self.std_dev * d_abs,
            worst_case: self.worst_case * d_abs,
            sample_count: self.sample_count,
            distribution: self.distribution,
        }
    }

    /// Creates error bound for sum of n identical operations.
    pub fn sum_n(base_error: &ProbabilisticError, n: usize) -> ProbabilisticError {
        base_error.accumulate(n)
    }

    /// Creates error bound for average of n identical operations.
    ///
    /// Average reduces variance by factor of n.
    pub fn average_n(base_error: &ProbabilisticError, n: usize) -> ProbabilisticError {
        if n == 0 {
            return ProbabilisticError::zero();
        }

        let n_f = n as f64;
        ProbabilisticError {
            mean: base_error.mean,
            std_dev: base_error.std_dev / n_f.sqrt(),
            worst_case: base_error.worst_case, // Worst case doesn't improve for average
            sample_count: base_error.sample_count * n,
            distribution: if n > 10 {
                ErrorDistribution::Gaussian
            } else {
                base_error.distribution
            },
        }
    }

    /// Checks if this error is within an acceptable budget.
    pub fn within_budget(&self, max_std_dev: f64, max_worst_case: f64) -> bool {
        self.std_dev <= max_std_dev && self.worst_case <= max_worst_case
    }

    /// Checks if the error is negligible (effectively zero).
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.worst_case < threshold && self.std_dev < threshold
    }
}

impl Default for ProbabilisticError {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for ProbabilisticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Error[μ={:.2e}, σ={:.2e}, max={:.2e}]",
            self.mean, self.std_dev, self.worst_case
        )
    }
}

// Helper functions for statistical computations

/// Error function (erf) approximation.
fn erf(x: f64) -> f64 {
    // Horner form coefficients for erf approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp();
    sign * y
}

/// Inverse normal CDF (Acklam's algorithm).
fn inverse_normal_cdf(p: f64) -> f64 {
    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Chi-squared distribution quantile (for variance testing).
pub fn chi_squared_quantile(df: usize, p: f64) -> f64 {
    // Wilson-Hilferty approximation
    let df_f = df as f64;
    let z = inverse_normal_cdf(p);
    let h = 2.0 / (9.0 * df_f);
    let term = 1.0 - h + z * h.sqrt();
    df_f * term * term * term
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_error() {
        let err = ProbabilisticError::zero();
        assert_eq!(err.mean, 0.0);
        assert_eq!(err.std_dev, 0.0);
        assert_eq!(err.worst_case, 0.0);
    }

    #[test]
    fn test_quantization_error() {
        let delta = 1.0 / 256.0; // INT8 quantization
        let err = ProbabilisticError::from_quantization(delta);

        // Uniform distribution: variance = delta^2 / 12
        let expected_std_dev = delta / (12.0_f64).sqrt();
        assert!((err.std_dev - expected_std_dev).abs() < 1e-10);
        assert!((err.worst_case - delta / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_interval() {
        let err = ProbabilisticError::new(0.0, 0.1, 0.3, ErrorDistribution::Gaussian);
        let interval = err.confidence_interval(10.0, ConfidenceLevel::NinetyFive);

        // 95% CI: mean ± 1.96 * std_dev
        assert!(interval.contains(10.0));
        assert!((interval.half_width() - 1.96 * 0.1).abs() < 0.01);
    }

    #[test]
    fn test_error_addition() {
        let e1 = ProbabilisticError::new(0.0, 0.1, 0.2, ErrorDistribution::Gaussian);
        let e2 = ProbabilisticError::new(0.0, 0.1, 0.2, ErrorDistribution::Gaussian);
        let sum = e1.add(&e2);

        // Variance adds, std_dev = sqrt(0.01 + 0.01) = sqrt(0.02)
        assert!((sum.std_dev - (0.02_f64).sqrt()).abs() < 1e-10);
        assert!((sum.worst_case - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_error_multiplication() {
        let e1 = ProbabilisticError::new(0.0, 0.01, 0.02, ErrorDistribution::Gaussian);
        let e2 = ProbabilisticError::new(0.0, 0.01, 0.02, ErrorDistribution::Gaussian);

        let prod = e1.multiply(&e2, 5.0, 4.0);

        // Worst case: |5| * 0.02 + |4| * 0.02 + 0.02 * 0.02 = 0.1 + 0.08 + 0.0004 = 0.1804
        // Note: both errors are 0.02
        assert!((prod.worst_case - 0.1804).abs() < 1e-10);
    }

    #[test]
    fn test_accumulation() {
        let err = ProbabilisticError::from_quantization(0.001);
        let acc = err.accumulate(100);

        // Std dev of sum = sqrt(n) * individual std dev
        assert!((acc.std_dev - 10.0 * err.std_dev).abs() < 1e-10);
        // Worst case is linear
        assert!((acc.worst_case - 100.0 * err.worst_case).abs() < 1e-10);
    }

    #[test]
    fn test_probability_within() {
        let err = ProbabilisticError::new(0.0, 0.1, 0.3, ErrorDistribution::Gaussian);

        // For Gaussian, ~68% within 1 std dev
        let p = err.probability_within(0.1);
        assert!(p > 0.6 && p < 0.7);

        // 100% within worst case
        let p_max = err.probability_within(0.31);
        assert!((p_max - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_erf() {
        // erf(0) should be very close to 0 (allow some floating point error)
        assert!(erf(0.0).abs() < 1e-6);
        // erf(∞) → 1
        assert!((erf(5.0) - 1.0).abs() < 1e-5);
        // erf(-x) = -erf(x)
        assert!((erf(1.0) + erf(-1.0)).abs() < 1e-6);
    }
}
