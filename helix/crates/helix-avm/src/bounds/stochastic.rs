//! Stochastic Rounding Error Bounds.
//!
//! Analyzes error from stochastic rounding, which provides
//! unbiased low-precision training.

use rand::Rng;

/// Configuration for stochastic rounding analysis.
#[derive(Debug, Clone)]
pub struct RoundingBounds {
    /// Source precision (bits).
    pub source_bits: u32,
    /// Target precision (bits).
    pub target_bits: u32,
    /// Number of rounding operations.
    pub num_operations: usize,
}

impl Default for RoundingBounds {
    fn default() -> Self {
        Self {
            source_bits: 32,
            target_bits: 16,
            num_operations: 1000,
        }
    }
}

/// Error analysis for stochastic rounding.
#[derive(Debug, Clone)]
pub struct StochasticRoundingError {
    /// Maximum per-operation error.
    pub max_error: f64,
    /// Expected (mean) error per operation.
    pub expected_error: f64,
    /// Standard deviation of error.
    pub error_std: f64,
    /// Total accumulated error bound (high probability).
    pub accumulated_error_bound: f64,
    /// Bias (should be ~0 for proper stochastic rounding).
    pub bias: f64,
}

impl RoundingBounds {
    /// Creates new rounding bounds.
    pub fn new(source_bits: u32, target_bits: u32) -> Self {
        Self {
            source_bits,
            target_bits,
            num_operations: 1000,
        }
    }

    /// Computes error bounds for stochastic rounding.
    ///
    /// Stochastic rounding rounds up/down with probability proportional
    /// to distance from rounding targets, making it unbiased.
    pub fn forward_error(&self) -> StochasticRoundingError {
        // Quantization step size
        let step_size = self.quantization_step();
        
        // Maximum error per operation is half the step size
        let max_error = step_size / 2.0;
        
        // For stochastic rounding, expected error is 0 (unbiased)
        let expected_error = 0.0;
        
        // Variance per operation: step² / 12 (uniform distribution)
        let variance_per_op = step_size * step_size / 12.0;
        let error_std = variance_per_op.sqrt();
        
        // Accumulated error (sum of independent random variables)
        // By CLT: total_std ≈ error_std * √n
        let n = self.num_operations as f64;
        let total_std = error_std * n.sqrt();
        
        // High-probability bound (3 sigma ≈ 99.7%)
        let accumulated_error_bound = 3.0 * total_std;
        
        // Bias is essentially 0 for proper stochastic rounding
        let bias = 0.0;
        
        StochasticRoundingError {
            max_error,
            expected_error,
            error_std,
            accumulated_error_bound,
            bias,
        }
    }

    /// Computes quantization step size.
    pub fn quantization_step(&self) -> f64 {
        // For floating point, step size depends on exponent
        // For simplicity, assume normalized range [0, 1]
        let target_mantissa_bits = self.target_bits.saturating_sub(1); // Sign bit
        2.0_f64.powi(-(target_mantissa_bits as i32))
    }

    /// Compares stochastic vs deterministic rounding.
    pub fn compare_rounding_methods(&self) -> RoundingComparison {
        let stochastic_error = self.forward_error();
        
        // Deterministic (round-to-nearest) has same max error
        // but accumulates bias over many operations
        let deterministic_max = stochastic_error.max_error;
        
        // Worst-case bias: all errors in same direction
        let deterministic_bias = deterministic_max * self.num_operations as f64;
        
        // Stochastic cancels out, deterministic doesn't
        let stochastic_total = stochastic_error.accumulated_error_bound;
        let deterministic_total = deterministic_bias;
        
        RoundingComparison {
            stochastic_error: stochastic_total,
            deterministic_error: deterministic_total,
            bias_reduction: deterministic_bias - stochastic_error.bias,
            recommendation: if self.num_operations > 100 {
                RoundingRecommendation::StochasticRounding
            } else {
                RoundingRecommendation::Deterministic
            },
        }
    }

    /// Simulates stochastic rounding error empirically.
    pub fn simulate_error(&self, true_values: &[f64]) -> SimulationResult {
        let mut rng = rand::thread_rng();
        let step = self.quantization_step();
        
        let mut errors: Vec<f64> = Vec::new();
        let mut total_error = 0.0;
        
        for &value in true_values {
            // Stochastic rounding
            let floor_val = (value / step).floor() * step;
            let ceil_val = floor_val + step;
            
            // Probability of rounding up proportional to distance from floor
            let p_up = (value - floor_val) / step;
            let rounded = if rng.gen::<f64>() < p_up {
                ceil_val
            } else {
                floor_val
            };
            
            let error = rounded - value;
            errors.push(error);
            total_error += error;
        }
        
        let mean_error = total_error / errors.len() as f64;
        let variance: f64 = errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>() 
            / errors.len() as f64;
        
        SimulationResult {
            mean_error,
            std_error: variance.sqrt(),
            max_error: errors.iter().map(|e| e.abs()).fold(0.0, f64::max),
            num_samples: errors.len(),
        }
    }
}

/// Comparison of rounding methods.
#[derive(Debug, Clone)]
pub struct RoundingComparison {
    /// Stochastic rounding total error.
    pub stochastic_error: f64,
    /// Deterministic rounding total error.
    pub deterministic_error: f64,
    /// Bias reduction from using stochastic.
    pub bias_reduction: f64,
    /// Recommendation.
    pub recommendation: RoundingRecommendation,
}

/// Rounding method recommendation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoundingRecommendation {
    StochasticRounding,
    Deterministic,
}

/// Empirical simulation results.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Mean error (should be ~0 for stochastic).
    pub mean_error: f64,
    /// Standard deviation of error.
    pub std_error: f64,
    /// Maximum observed error.
    pub max_error: f64,
    /// Number of samples.
    pub num_samples: usize,
}

/// Mixed-precision training error analysis.
pub fn mixed_precision_error(
    fp32_ops: usize,
    fp16_ops: usize,
    bf16_ops: usize,
) -> MixedPrecisionError {
    let fp32_step = 2.0_f64.powi(-23); // 23 mantissa bits
    let fp16_step = 2.0_f64.powi(-10); // 10 mantissa bits
    let bf16_step = 2.0_f64.powi(-7);  // 7 mantissa bits
    
    let fp32_error = fp32_ops as f64 * fp32_step / 2.0;
    let fp16_error = fp16_ops as f64 * fp16_step / 2.0;
    let bf16_error = bf16_ops as f64 * bf16_step / 2.0;
    
    let total_error = fp32_error + fp16_error + bf16_error;
    
    MixedPrecisionError {
        fp32_contribution: fp32_error,
        fp16_contribution: fp16_error,
        bf16_contribution: bf16_error,
        total_error,
    }
}

/// Mixed precision error breakdown.
#[derive(Debug, Clone)]
pub struct MixedPrecisionError {
    /// Error from FP32 operations.
    pub fp32_contribution: f64,
    /// Error from FP16 operations.
    pub fp16_contribution: f64,
    /// Error from BF16 operations.
    pub bf16_contribution: f64,
    /// Total error.
    pub total_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_rounding_error() {
        let bounds = RoundingBounds::new(32, 16);
        let error = bounds.forward_error();
        
        assert!(error.max_error > 0.0);
        assert!(error.bias.abs() < 1e-10); // Unbiased
    }

    #[test]
    fn test_stochastic_better_for_many_ops() {
        let bounds = RoundingBounds {
            source_bits: 32,
            target_bits: 16,
            num_operations: 10000,
        };
        
        let comparison = bounds.compare_rounding_methods();
        
        // Stochastic should be better for many operations
        assert!(comparison.stochastic_error < comparison.deterministic_error);
    }

    #[test]
    fn test_simulation() {
        let bounds = RoundingBounds::new(32, 16);
        let values: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        
        let result = bounds.simulate_error(&values);
        
        // Mean should be close to 0 (unbiased)
        assert!(result.mean_error.abs() < 0.01);
    }

    #[test]
    fn test_mixed_precision() {
        let error = mixed_precision_error(100, 1000, 500);
        
        assert!(error.total_error > 0.0);
        // FP16/BF16 should dominate
        assert!(error.fp16_contribution + error.bf16_contribution > error.fp32_contribution);
    }
}
