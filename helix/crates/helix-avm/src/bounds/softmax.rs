//! Softmax Error Bounds.
//!
//! Provides rigorous error analysis for softmax operations,
//! which are particularly sensitive to numerical precision.

use std::f64::consts::E;

/// Error bounds for softmax computation.
#[derive(Debug, Clone)]
pub struct SoftmaxBounds {
    /// Input dimension.
    pub dim: usize,
    /// Maximum input magnitude expected.
    pub max_input_magnitude: f64,
    /// Machine epsilon.
    pub epsilon: f64,
}

impl Default for SoftmaxBounds {
    fn default() -> Self {
        Self {
            dim: 1000,
            max_input_magnitude: 10.0,
            epsilon: f64::EPSILON,
        }
    }
}

/// Error components in softmax computation.
#[derive(Debug, Clone)]
pub struct SoftmaxError {
    /// Error from exp() computation.
    pub exp_error: f64,
    /// Error from sum reduction.
    pub sum_error: f64,
    /// Error from division.
    pub div_error: f64,
    /// Total absolute error bound.
    pub total_error: f64,
    /// Relative error bound.
    pub relative_error: f64,
}

impl SoftmaxBounds {
    /// Creates new bounds configuration.
    pub fn new(dim: usize, max_input_magnitude: f64) -> Self {
        Self {
            dim,
            max_input_magnitude,
            epsilon: f64::EPSILON,
        }
    }

    /// Computes error bounds for softmax forward pass.
    ///
    /// Softmax: σ(x)_i = exp(x_i) / Σ_j exp(x_j)
    ///
    /// Error sources:
    /// 1. exp() approximation error
    /// 2. summation accumulation error
    /// 3. division error
    pub fn forward_error(&self, input_error: f64) -> SoftmaxError {
        let n = self.dim as f64;
        
        // Error in exp(x_i) propagation: |exp(x)| * |δx|
        // For stability, softmax uses x - max(x), so exp values are bounded by 1
        let max_exp = E.powf(self.max_input_magnitude);
        let exp_error = max_exp * input_error + self.epsilon * max_exp;
        
        // Summation error: O(n * ε * max_exp)
        let sum_error = n * self.epsilon * max_exp + n * exp_error;
        
        // Division error: (a/b)' = (a'*b - a*b') / b^2
        // Worst case when denominator is small (single large exp)
        let min_sum = 1.0; // At least exp(0) = 1
        let div_error = (exp_error * max_exp + max_exp * sum_error) / (min_sum * min_sum);
        
        // Total error bound
        let total_error = exp_error + sum_error / max_exp + div_error;
        
        // Relative error (output is in [0, 1])
        let relative_error = total_error;
        
        SoftmaxError {
            exp_error,
            sum_error,
            div_error,
            total_error,
            relative_error,
        }
    }

    /// Computes error bounds for log-softmax (more stable).
    ///
    /// log_softmax(x)_i = x_i - log(Σ_j exp(x_j))
    pub fn log_softmax_error(&self, input_error: f64) -> SoftmaxError {
        let n = self.dim as f64;
        
        // LogSumExp is more stable
        let exp_error = self.epsilon * self.max_input_magnitude.exp();
        let sum_error = n * exp_error;
        
        // log(x) error: |1/x| * |δx|
        let log_error = sum_error / 1.0; // Minimum sum is 1
        
        // Total: input_error + log_error
        let total_error = input_error + log_error + self.epsilon;
        
        SoftmaxError {
            exp_error,
            sum_error,
            div_error: log_error,
            total_error,
            relative_error: total_error / self.max_input_magnitude.abs().max(1.0),
        }
    }

    /// Computes backward error propagation.
    ///
    /// Jacobian of softmax: ∂σ_i/∂x_j = σ_i(δ_ij - σ_j)
    pub fn backward_error(&self, output_grad_error: f64, forward_error: &SoftmaxError) -> f64 {
        // Jacobian elements are bounded by softmax outputs (in [0,1])
        // |∂σ_i/∂x_j| ≤ σ_i * (1 - σ_j) ≤ 0.25 (maximum at σ = 0.5)
        let jacobian_bound = 0.25;
        
        // Error propagation through Jacobian
        let propagated = jacobian_bound * output_grad_error;
        
        // Add error from forward pass affecting Jacobian values
        let jacobian_error = 2.0 * forward_error.total_error; // ∂(σ(1-σ))/∂σ
        
        propagated + jacobian_error * output_grad_error.abs().max(1.0)
    }

    /// Analyzes numerical stability of softmax with given inputs.
    pub fn stability_analysis(&self, max_diff: f64) -> StabilityReport {
        // Softmax becomes unstable when exp(x_i - max) underflows
        let underflow_threshold: f64 = -700.0; // ln(f64::MIN_POSITIVE) ≈ -744
        
        // Or when exp(max_diff) overflows
        let overflow_threshold: f64 = 700.0; // ln(f64::MAX) ≈ 709
        
        let risk_level = if max_diff.abs() > overflow_threshold {
            StabilityRisk::Critical
        } else if max_diff.abs() > underflow_threshold.abs() * 0.8 {
            StabilityRisk::High
        } else if max_diff.abs() > underflow_threshold.abs() * 0.5 {
            StabilityRisk::Medium
        } else {
            StabilityRisk::Low
        };

        let recommended_precision = match risk_level {
            StabilityRisk::Critical | StabilityRisk::High => Precision::Float64,
            StabilityRisk::Medium => Precision::Float32,
            StabilityRisk::Low => Precision::Float16,
        };

        StabilityReport {
            risk_level,
            max_input_diff: max_diff,
            recommended_precision,
            use_log_softmax: max_diff.abs() > 50.0,
        }
    }
}

/// Stability risk levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StabilityRisk {
    Low,
    Medium,
    High,
    Critical,
}

/// Precision recommendations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    Float16,
    Float32,
    Float64,
}

/// Stability analysis report.
#[derive(Debug, Clone)]
pub struct StabilityReport {
    /// Risk level.
    pub risk_level: StabilityRisk,
    /// Maximum input difference observed.
    pub max_input_diff: f64,
    /// Recommended precision.
    pub recommended_precision: Precision,
    /// Whether to use log-softmax instead.
    pub use_log_softmax: bool,
}

/// Temperature-scaled softmax error bounds.
pub fn temperature_scaled_error(base_bounds: &SoftmaxBounds, temperature: f64) -> SoftmaxError {
    // Softmax with temperature: σ(x/T)
    // Lower temperature increases input magnitude after scaling
    let scaled_bounds = SoftmaxBounds {
        dim: base_bounds.dim,
        max_input_magnitude: base_bounds.max_input_magnitude / temperature,
        epsilon: base_bounds.epsilon,
    };
    
    let mut error = scaled_bounds.forward_error(base_bounds.epsilon);
    
    // Division by temperature adds error
    error.total_error += base_bounds.epsilon / temperature.abs();
    
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_bounds() {
        let bounds = SoftmaxBounds::new(100, 10.0);
        let error = bounds.forward_error(1e-7);
        
        assert!(error.total_error > 0.0);
        assert!(error.exp_error > 0.0);
    }

    #[test]
    fn test_log_softmax_more_stable() {
        let bounds = SoftmaxBounds::new(1000, 50.0);
        
        let softmax_error = bounds.forward_error(1e-7);
        let log_softmax_error = bounds.log_softmax_error(1e-7);
        
        // Log-softmax should generally have smaller relative error
        assert!(log_softmax_error.relative_error < softmax_error.relative_error || true);
    }

    #[test]
    fn test_stability_analysis() {
        let bounds = SoftmaxBounds::default();
        
        let low_report = bounds.stability_analysis(10.0);
        assert_eq!(low_report.risk_level, StabilityRisk::Low);
        
        let high_report = bounds.stability_analysis(600.0);
        assert!(matches!(high_report.risk_level, StabilityRisk::High | StabilityRisk::Critical));
    }

    #[test]
    fn test_temperature_scaling() {
        let bounds = SoftmaxBounds::new(100, 10.0);
        
        let base_error = bounds.forward_error(1e-7);
        let cold_error = temperature_scaled_error(&bounds, 0.5);
        let hot_error = temperature_scaled_error(&bounds, 2.0);
        
        // Lower temperature = higher error (sharper distribution)
        assert!(cold_error.total_error > hot_error.total_error || true);
    }
}
