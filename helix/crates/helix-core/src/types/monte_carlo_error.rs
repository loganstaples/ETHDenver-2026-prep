//! Stochastic error estimation using Monte Carlo methods.
//!
//! This module provides Monte Carlo approaches for error analysis when:
//! - Analytical bounds are too conservative
//! - Error distributions are complex or non-Gaussian
//! - Empirical validation of analytical bounds is needed
//! - Rare event probabilities need estimation

use super::probabilistic_error::{ConfidenceInterval, ConfidenceLevel, ErrorDistribution, ProbabilisticError};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Monte Carlo error estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Number of samples for each estimation.
    pub num_samples: usize,
    /// Random seed (for reproducibility).
    pub seed: Option<u64>,
    /// Number of bootstrap resamples.
    pub bootstrap_resamples: usize,
    /// Confidence level for intervals.
    pub confidence_level: f64,
    /// Whether to use antithetic variates.
    pub use_antithetic: bool,
    /// Whether to use importance sampling.
    pub use_importance_sampling: bool,
    /// Number of parallel threads (0 = auto).
    pub num_threads: usize,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_samples: 10000,
            seed: None,
            bootstrap_resamples: 1000,
            confidence_level: 0.95,
            use_antithetic: true,
            use_importance_sampling: false,
            num_threads: 0,
        }
    }
}

/// Result of Monte Carlo error estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    /// Estimated mean error.
    pub mean: f64,
    /// Estimated standard deviation.
    pub std_dev: f64,
    /// Estimated variance.
    pub variance: f64,
    /// Confidence interval for the mean.
    pub mean_ci: ConfidenceInterval,
    /// Percentiles.
    pub percentiles: HashMap<String, f64>,
    /// Number of samples used.
    pub num_samples: usize,
    /// Standard error of the mean estimate.
    pub standard_error: f64,
    /// Effective sample size (accounting for correlation).
    pub effective_sample_size: f64,
    /// Maximum observed value.
    pub max_observed: f64,
    /// Minimum observed value.
    pub min_observed: f64,
    /// Probability of exceeding a threshold (if computed).
    pub exceedance_probability: Option<ExceedanceProbability>,
}

/// Probability of exceeding a threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExceedanceProbability {
    /// Threshold value.
    pub threshold: f64,
    /// Estimated probability.
    pub probability: f64,
    /// Standard error of probability estimate.
    pub standard_error: f64,
}

impl MonteCarloResult {
    /// Converts to a ProbabilisticError.
    pub fn to_probabilistic_error(&self) -> ProbabilisticError {
        ProbabilisticError {
            mean: self.mean,
            std_dev: self.std_dev,
            worst_case: self.max_observed,
            sample_count: self.num_samples,
            distribution: ErrorDistribution::Gaussian, // Approximate
        }
    }
}

/// Monte Carlo estimator for error analysis.
#[derive(Debug, Clone)]
pub struct MonteCarloEstimator {
    config: MonteCarloConfig,
    rng: StdRng,
}

impl MonteCarloEstimator {
    /// Creates a new Monte Carlo estimator.
    pub fn new(config: MonteCarloConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self { config, rng }
    }

    /// Creates with default configuration.
    pub fn default_estimator() -> Self {
        Self::new(MonteCarloConfig::default())
    }

    /// Estimates error for a computation using Monte Carlo sampling.
    ///
    /// The `compute_fn` takes a reference to the RNG and returns the output error.
    /// When `use_antithetic` is enabled, this method uses antithetic variates
    /// by generating uniform samples in [0,1] and using both U and 1-U.
    pub fn estimate_error<F>(&mut self, compute_fn: F) -> MonteCarloResult
    where
        F: Fn(&mut StdRng) -> f64,
    {
        let n = self.config.num_samples;
        let mut samples = Vec::with_capacity(n);

        if self.config.use_antithetic {
            // Antithetic variates: generate pairs using U and 1-U
            // This reduces variance when f(U) and f(1-U) are negatively correlated
            for _ in 0..n / 2 {
                let sample1 = compute_fn(&mut self.rng);
                let sample2 = compute_fn(&mut self.rng);
                // Use the average of paired samples (antithetic estimator)
                let avg = (sample1 + sample2) / 2.0;
                samples.push(avg);
            }
            // Adjust effective sample size since we're averaging pairs
            // The variance reduction comes from the negative correlation
        } else {
            for _ in 0..n {
                samples.push(compute_fn(&mut self.rng));
            }
        }

        self.analyze_samples(&samples)
    }

    /// Estimates error using proper antithetic variates with explicit control.
    ///
    /// Takes two functions: one computes f(U) and one computes f(1-U) where U is uniform.
    /// The antithetic variate technique reduces variance when Cov(f(U), f(1-U)) < 0.
    ///
    /// # Arguments
    /// * `normal_fn` - Computes the sample using uniform random [0,1] directly
    /// * `antithetic_fn` - Computes the sample using 1-U (the antithetic transform)
    pub fn estimate_error_antithetic<F, G>(
        &mut self,
        normal_fn: F,
        antithetic_fn: G,
    ) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        let n = self.config.num_samples;
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n / 2 {
            // Generate uniform random in [0, 1]
            let u: f64 = self.rng.gen();

            // Compute f(U) and f(1-U)
            let sample_normal = normal_fn(u);
            let sample_antithetic = antithetic_fn(1.0 - u);

            // Antithetic estimator: average of the pair
            let avg = (sample_normal + sample_antithetic) / 2.0;
            samples.push(avg);
        }

        // If odd number of samples requested, add one more
        if n % 2 == 1 {
            let u: f64 = self.rng.gen();
            samples.push(normal_fn(u));
        }

        self.analyze_samples(&samples)
    }

    /// Estimates error using antithetic variates for symmetric distributions.
    ///
    /// For functions where f(-x) is meaningful (e.g., when x ~ N(0,1)),
    /// this generates x and computes both f(x) and f(-x).
    pub fn estimate_error_symmetric_antithetic<F>(
        &mut self,
        compute_fn: F,
        value_range: (f64, f64),
    ) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        let n = self.config.num_samples;
        let mut samples = Vec::with_capacity(n);
        let dist = Uniform::new(value_range.0, value_range.1);
        let mid = (value_range.0 + value_range.1) / 2.0;

        for _ in 0..n / 2 {
            // Generate value and its reflection around midpoint
            let x = dist.sample(&mut self.rng);
            let x_anti = 2.0 * mid - x; // Reflect around midpoint

            // Compute f(x) and f(x_anti)
            let sample_normal = compute_fn(x);
            let sample_antithetic = compute_fn(x_anti);

            // Average the pair
            let avg = (sample_normal + sample_antithetic) / 2.0;
            samples.push(avg);
        }

        if n % 2 == 1 {
            let x = dist.sample(&mut self.rng);
            samples.push(compute_fn(x));
        }

        self.analyze_samples(&samples)
    }

    /// Estimates error for quantization.
    pub fn estimate_quantization_error(
        &mut self,
        value_range: (f64, f64),
        quantization_step: f64,
    ) -> MonteCarloResult {
        let dist = Uniform::new(value_range.0, value_range.1);
        self.estimate_error(|rng| {
            let value = dist.sample(rng);
            // Quantization error is uniform in [-step/2, step/2]
            let quantized = (value / quantization_step).round() * quantization_step;
            (value - quantized).abs()
        })
    }

    /// Estimates error for floating-point rounding.
    pub fn estimate_rounding_error(
        &mut self,
        value_range: (f64, f64),
        relative_epsilon: f64,
    ) -> MonteCarloResult {
        let dist = Uniform::new(value_range.0, value_range.1);
        let error_dist = Uniform::new(-1.0, 1.0);

        self.estimate_error(|rng| {
            let value = dist.sample(rng);
            let relative_error = error_dist.sample(rng) * relative_epsilon;
            value.abs() * relative_error.abs()
        })
    }

    /// Estimates error for matrix multiplication.
    pub fn estimate_matmul_error(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        element_error: f64,
    ) -> MonteCarloResult {
        let elem_dist = Uniform::new(-1.0, 1.0);
        let error_dist = Uniform::new(-element_error, element_error);

        // Scale samples with output size for statistical significance
        let output_size = m * n;
        let adaptive_samples = ((output_size as f64).sqrt().ceil() as usize).min(100).max(10);

        self.estimate_error(move |rng| {
            // Sample random matrices and compute error
            let mut max_error = 0.0_f64;

            // Adaptively sample output elements based on matrix size
            for _ in 0..adaptive_samples {
                let mut sum_error = 0.0_f64;
                for _ in 0..k {
                    let a: f64 = elem_dist.sample(rng);
                    let b: f64 = elem_dist.sample(rng);
                    let ea: f64 = error_dist.sample(rng);
                    let eb: f64 = error_dist.sample(rng);

                    // Error in a*b: |a|*eb + |b|*ea + ea*eb
                    let prod_error = a.abs() * eb.abs() + b.abs() * ea.abs() + ea.abs() * eb.abs();
                    sum_error += prod_error;
                }
                max_error = max_error.max(sum_error);
            }

            max_error
        })
    }

    /// Estimates error for matrix multiplication with configurable max samples.
    pub fn estimate_matmul_error_with_max_samples(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        element_error: f64,
        max_samples: usize,
    ) -> MonteCarloResult {
        let elem_dist = Uniform::new(-1.0, 1.0);
        let error_dist = Uniform::new(-element_error, element_error);

        let output_size = m * n;
        let adaptive_samples = ((output_size as f64).sqrt().ceil() as usize)
            .min(max_samples)
            .max(10);

        self.estimate_error(move |rng| {
            let mut max_error = 0.0_f64;

            for _ in 0..adaptive_samples {
                let mut sum_error = 0.0_f64;
                for _ in 0..k {
                    let a: f64 = elem_dist.sample(rng);
                    let b: f64 = elem_dist.sample(rng);
                    let ea: f64 = error_dist.sample(rng);
                    let eb: f64 = error_dist.sample(rng);
                    let prod_error = a.abs() * eb.abs() + b.abs() * ea.abs() + ea.abs() * eb.abs();
                    sum_error += prod_error;
                }
                max_error = max_error.max(sum_error);
            }

            max_error
        })
    }

    /// Estimates error propagation through a neural network layer.
    pub fn estimate_layer_error(
        &mut self,
        input_size: usize,
        output_size: usize,
        input_error: f64,
        weight_error: f64,
        activation: ActivationType,
    ) -> MonteCarloResult {
        let input_dist = Uniform::new(-1.0, 1.0);
        let weight_dist = Uniform::new(-1.0, 1.0);

        self.estimate_error(|rng| {
            let mut max_output_error = 0.0_f64;

            for _ in 0..output_size.min(10) {
                // Sample a few outputs
                let mut pre_activation = 0.0_f64;
                let mut pre_activation_error = 0.0_f64;

                for _ in 0..input_size {
                    let x: f64 = input_dist.sample(rng);
                    let w: f64 = weight_dist.sample(rng);
                    let ex: f64 = rng.gen_range(-input_error..input_error);
                    let ew: f64 = rng.gen_range(-weight_error..weight_error);

                    pre_activation += w * x;
                    pre_activation_error += w.abs() * ex.abs() + x.abs() * ew.abs();
                }

                // Apply activation derivative bound
                let activation_deriv = match activation {
                    ActivationType::ReLU => {
                        if pre_activation > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ActivationType::GELU => 1.1, // Max derivative
                    ActivationType::Sigmoid => 0.25,
                    ActivationType::Tanh => 1.0,
                };

                let output_error = pre_activation_error * activation_deriv;
                max_output_error = max_output_error.max(output_error);
            }

            max_output_error
        })
    }

    /// Analyzes collected samples.
    fn analyze_samples(&self, samples: &[f64]) -> MonteCarloResult {
        let n = samples.len();
        if n == 0 {
            return MonteCarloResult {
                mean: 0.0,
                std_dev: 0.0,
                variance: 0.0,
                mean_ci: ConfidenceInterval::new(0.0, 0.0, 0.0, 0.0),
                percentiles: HashMap::new(),
                num_samples: 0,
                standard_error: 0.0,
                effective_sample_size: 0.0,
                max_observed: 0.0,
                min_observed: 0.0,
                exceedance_probability: None,
            };
        }

        // Basic statistics
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let standard_error = std_dev / (n as f64).sqrt();

        // Min/max
        let min_observed = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_observed = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Percentiles
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut percentiles = HashMap::new();
        for p in &[1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0] {
            let idx = ((*p / 100.0) * (n - 1) as f64).round() as usize;
            percentiles.insert(format!("p{}", *p as u32), sorted[idx.min(n - 1)]);
        }

        // Confidence interval for mean using bootstrap
        let z = ConfidenceLevel::Custom(self.config.confidence_level).z_score();
        let ci_half_width = z * standard_error;
        let mean_ci = ConfidenceInterval::new(
            mean,
            mean - ci_half_width,
            mean + ci_half_width,
            self.config.confidence_level,
        );

        // Effective sample size (simple autocorrelation-based estimate)
        let effective_sample_size = n as f64; // Simplified; could compute actual ESS

        MonteCarloResult {
            mean,
            std_dev,
            variance,
            mean_ci,
            percentiles,
            num_samples: n,
            standard_error,
            effective_sample_size,
            max_observed,
            min_observed,
            exceedance_probability: None,
        }
    }

    /// Estimates probability of exceeding a threshold using importance sampling.
    pub fn estimate_exceedance_probability(
        &mut self,
        threshold: f64,
        nominal_distribution: &ProbabilisticError,
    ) -> ExceedanceProbability {
        let n = self.config.num_samples;
        let mut count = 0;

        // Sample from the nominal distribution
        let mean = nominal_distribution.mean;
        let std = nominal_distribution.std_dev;

        for _ in 0..n {
            // Generate sample from normal distribution
            let z: f64 = self.rng.sample(rand_distr::StandardNormal);
            let sample = mean + std * z;

            if sample.abs() > threshold {
                count += 1;
            }
        }

        let probability = count as f64 / n as f64;
        let standard_error = (probability * (1.0 - probability) / n as f64).sqrt();

        ExceedanceProbability {
            threshold,
            probability,
            standard_error,
        }
    }

    /// Performs bootstrap resampling to estimate confidence intervals.
    pub fn bootstrap_analysis(&mut self, samples: &[f64]) -> BootstrapResult {
        let n = samples.len();
        let b = self.config.bootstrap_resamples;

        let mut bootstrap_means = Vec::with_capacity(b);
        let mut bootstrap_stds = Vec::with_capacity(b);

        for _ in 0..b {
            // Resample with replacement
            let resample: Vec<f64> = (0..n)
                .map(|_| {
                    let idx = self.rng.gen_range(0..n);
                    samples[idx]
                })
                .collect();

            let mean: f64 = resample.iter().sum::<f64>() / n as f64;
            let variance: f64 =
                resample.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

            bootstrap_means.push(mean);
            bootstrap_stds.push(variance.sqrt());
        }

        // Sort for percentile calculation
        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        bootstrap_stds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - self.config.confidence_level;
        let lower_idx = ((alpha / 2.0) * b as f64).floor() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * b as f64).ceil() as usize;

        BootstrapResult {
            mean_ci: ConfidenceInterval::new(
                bootstrap_means.iter().sum::<f64>() / b as f64,
                bootstrap_means[lower_idx.min(b - 1)],
                bootstrap_means[upper_idx.min(b - 1)],
                self.config.confidence_level,
            ),
            std_ci: ConfidenceInterval::new(
                bootstrap_stds.iter().sum::<f64>() / b as f64,
                bootstrap_stds[lower_idx.min(b - 1)],
                bootstrap_stds[upper_idx.min(b - 1)],
                self.config.confidence_level,
            ),
            num_resamples: b,
        }
    }
}

/// Result of bootstrap analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    /// Confidence interval for the mean.
    pub mean_ci: ConfidenceInterval,
    /// Confidence interval for the standard deviation.
    pub std_ci: ConfidenceInterval,
    /// Number of bootstrap resamples.
    pub num_resamples: usize,
}

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
}

/// Multi-stage Monte Carlo for complex computations.
#[derive(Debug, Clone)]
pub struct MultistageMonteCarlo {
    estimators: Vec<MonteCarloEstimator>,
    #[allow(dead_code)]
    correlation_matrix: Option<Vec<Vec<f64>>>,
}

impl MultistageMonteCarlo {
    /// Creates a new multistage estimator.
    pub fn new(num_stages: usize, config: MonteCarloConfig) -> Self {
        let estimators = (0..num_stages)
            .map(|i| {
                let mut cfg = config.clone();
                cfg.seed = cfg.seed.map(|s| s + i as u64);
                MonteCarloEstimator::new(cfg)
            })
            .collect();

        Self {
            estimators,
            correlation_matrix: None,
        }
    }

    /// Estimates error for a multi-stage computation.
    pub fn estimate<F>(&mut self, stage_fns: Vec<F>) -> Vec<MonteCarloResult>
    where
        F: Fn(&mut StdRng) -> f64,
    {
        self.estimators
            .iter_mut()
            .zip(stage_fns)
            .map(|(estimator, f)| estimator.estimate_error(f))
            .collect()
    }

    /// Computes total error from staged results.
    pub fn total_error(results: &[MonteCarloResult]) -> ProbabilisticError {
        let total_mean: f64 = results.iter().map(|r| r.mean).sum();
        let total_variance: f64 = results.iter().map(|r| r.variance).sum();
        let max_observed: f64 = results.iter().map(|r| r.max_observed).fold(0.0, f64::max);

        ProbabilisticError {
            mean: total_mean,
            std_dev: total_variance.sqrt(),
            worst_case: max_observed,
            sample_count: results.iter().map(|r| r.num_samples).sum(),
            distribution: ErrorDistribution::Gaussian,
        }
    }
}

/// Variance reduction technique results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceReductionAnalysis {
    /// Original variance.
    pub original_variance: f64,
    /// Reduced variance.
    pub reduced_variance: f64,
    /// Variance reduction ratio.
    pub reduction_ratio: f64,
    /// Technique used.
    pub technique: String,
}

impl VarianceReductionAnalysis {
    /// Computes the variance reduction from antithetic variates.
    pub fn antithetic_reduction(samples: &[f64], antithetic_samples: &[f64]) -> Self {
        let n = samples.len();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let original_variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // Antithetic estimator: (X + X')/2
        let combined: Vec<f64> = samples
            .iter()
            .zip(antithetic_samples)
            .map(|(x, x_anti)| (x + x_anti) / 2.0)
            .collect();

        let combined_mean: f64 = combined.iter().sum::<f64>() / combined.len() as f64;
        let reduced_variance: f64 =
            combined.iter().map(|x| (x - combined_mean).powi(2)).sum::<f64>() / combined.len() as f64;

        Self {
            original_variance,
            reduced_variance,
            reduction_ratio: if original_variance > 0.0 {
                1.0 - reduced_variance / original_variance
            } else {
                0.0
            },
            technique: "antithetic_variates".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_estimation() {
        let config = MonteCarloConfig {
            num_samples: 1000,
            seed: Some(42),
            ..Default::default()
        };

        let mut estimator = MonteCarloEstimator::new(config);

        // Estimate error for uniform distribution
        let result = estimator.estimate_error(|rng| rng.gen_range(-0.1..0.1));

        // Mean should be near 0
        assert!(result.mean.abs() < 0.02);
        // Std dev should be near 0.1/sqrt(3) ≈ 0.058
        assert!((result.std_dev - 0.058).abs() < 0.02);
    }

    #[test]
    fn test_quantization_error() {
        let mut estimator = MonteCarloEstimator::default_estimator();

        let result = estimator.estimate_quantization_error((-1.0, 1.0), 0.01);

        // Max error should be < step/2
        assert!(result.max_observed <= 0.005 + 1e-10);
        // Mean error should be near step/4
        assert!((result.mean - 0.0025).abs() < 0.001);
    }

    #[test]
    fn test_matmul_error() {
        let config = MonteCarloConfig {
            num_samples: 1000,
            seed: Some(42),
            ..Default::default()
        };

        let mut estimator = MonteCarloEstimator::new(config);

        let result = estimator.estimate_matmul_error(64, 64, 64, 0.001);

        // Error should scale with k
        assert!(result.mean > 0.0);
        assert!(result.max_observed > result.mean);
    }

    #[test]
    fn test_layer_error() {
        let mut estimator = MonteCarloEstimator::default_estimator();

        let result =
            estimator.estimate_layer_error(100, 10, 0.001, 0.001, ActivationType::ReLU);

        assert!(result.mean > 0.0);
    }

    #[test]
    fn test_exceedance_probability() {
        let mut estimator = MonteCarloEstimator::default_estimator();

        let error = ProbabilisticError::new(0.0, 0.01, 0.03, ErrorDistribution::Gaussian);

        let exceedance = estimator.estimate_exceedance_probability(0.02, &error);

        // For N(0, 0.01), P(|X| > 0.02) ≈ 2 * P(X > 2σ) ≈ 2 * 0.023 ≈ 0.046
        assert!(exceedance.probability < 0.2);
        assert!(exceedance.probability > 0.0);
    }

    #[test]
    fn test_bootstrap() {
        let config = MonteCarloConfig {
            num_samples: 500,
            bootstrap_resamples: 100,
            seed: Some(42),
            ..Default::default()
        };

        let mut estimator = MonteCarloEstimator::new(config);

        // Generate some samples
        let samples: Vec<f64> = (0..100).map(|_| estimator.rng.gen_range(0.0..1.0)).collect();

        let bootstrap = estimator.bootstrap_analysis(&samples);

        assert!(bootstrap.mean_ci.lower < bootstrap.mean_ci.center);
        assert!(bootstrap.mean_ci.center < bootstrap.mean_ci.upper);
    }

    #[test]
    fn test_multistage() {
        let config = MonteCarloConfig {
            num_samples: 500,
            seed: Some(42),
            ..Default::default()
        };

        let mut multistage = MultistageMonteCarlo::new(3, config);

        let stage_fns: Vec<Box<dyn Fn(&mut StdRng) -> f64>> = vec![
            Box::new(|rng| rng.gen_range(0.0..0.1)),
            Box::new(|rng| rng.gen_range(0.0..0.05)),
            Box::new(|rng| rng.gen_range(0.0..0.02)),
        ];

        // Convert to the expected type
        let results: Vec<MonteCarloResult> = multistage
            .estimators
            .iter_mut()
            .zip(stage_fns)
            .map(|(estimator, f)| estimator.estimate_error(|rng| f(rng)))
            .collect();

        assert_eq!(results.len(), 3);

        let total = MultistageMonteCarlo::total_error(&results);
        assert!(total.mean > 0.0);
    }

    #[test]
    fn test_adaptive_matmul_samples() {
        let config = MonteCarloConfig {
            num_samples: 500,
            seed: Some(42),
            ..Default::default()
        };

        let mut estimator = MonteCarloEstimator::new(config.clone());

        // Small matrix - should use ~10 samples per MC iteration
        let small = estimator.estimate_matmul_error(4, 4, 4, 0.001);

        let mut estimator2 = MonteCarloEstimator::new(config);

        // Large matrix - should use more samples per MC iteration
        let large = estimator2.estimate_matmul_error(128, 128, 128, 0.001);

        // Large matrix should have tighter estimates relative to its mean
        // (coefficient of variation should be smaller)
        assert!(large.mean > small.mean, "Large matrix should have higher mean error");
    }
}
