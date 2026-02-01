//! HELIX Performance Benchmark Suite
//!
//! This module provides comprehensive benchmarking infrastructure for measuring
//! and proving the overhead claims of the HELIX trustless AI training system.
//!
//! # Components
//!
//! - `harness` - Standardized benchmark harness with JSON/markdown reporting
//! - `native_baseline` - Native computation baselines (no proving overhead)
//! - `gkr_prover` - GKR prover benchmarks at multiple model sizes
//! - `halo2_prover` - Halo2 prover benchmarks for comparison
//! - `overhead_report` - Automated overhead calculation and reporting
//! - `metal_vs_cpu` - Metal GPU vs CPU comparison benchmarks
//! - `memory_profile` - Memory usage profiling
//! - `scaling` - Scaling tests showing overhead vs model size
//!
//! # Usage
//!
//! Run the full benchmark suite:
//! ```bash
//! cargo bench --package helix-integration-tests --bench comprehensive_benchmarks
//! ```
//!
//! Run with baseline comparison (CI mode):
//! ```bash
//! cargo bench --package helix-integration-tests -- --save-baseline main
//! cargo bench --package helix-integration-tests -- --baseline main
//! ```
//!
//! Generate overhead report:
//! ```bash
//! cargo bench --package helix-integration-tests --bench overhead_report
//! ```
//!
//! # Success Criteria
//!
//! The HELIX system targets:
//! - **30x overhead**: Total proof generation time should be ≤30x native computation
//! - **<500ms per step**: Proof generation for demo model sizes
//! - **Regression detection**: CI fails if overhead increases by >10%

pub mod chart_generator;
pub mod ci_runner;
pub mod gkr_prover;
pub mod halo2_prover;
pub mod harness;
pub mod memory_profile;
pub mod metal_vs_cpu;
pub mod native_baseline;
pub mod overhead_report;
pub mod prover_comparison;
pub mod regression;
pub mod scaling;

// Re-exports for convenience
pub use chart_generator::{ChartConfig, ChartDataPoint, ChartGenerator};
pub use harness::{
    BenchmarkConfig, BenchmarkHarness, BenchmarkMetrics, BenchmarkReport, BenchmarkResult,
    OutputFormat, RegressionStatus,
};
pub use native_baseline::{ComputationType, NativeBaseline};
pub use overhead_report::{OverheadMetrics, OverheadReport};
pub use prover_comparison::{ComparisonReport, ProverComparisonBenchmarks, ProverResult};
pub use regression::{RegressionDetector, RegressionReport, RegressionSeverity};
pub use scaling::{ScalingAnalysis, ScalingPoint};

use std::time::Duration;

/// Target overhead multiple for HELIX (ZK time / native time).
pub const TARGET_OVERHEAD_MULTIPLE: f64 = 30.0;

/// Alert threshold for regression detection - alerts if overhead exceeds this.
pub const REGRESSION_ALERT_THRESHOLD: f64 = 35.0;

/// Maximum acceptable overhead regression percentage for CI.
pub const MAX_REGRESSION_PERCENT: f64 = 10.0;

/// Target proof generation time for demo model sizes.
pub const TARGET_PROOF_TIME_MS: u64 = 500;

/// Standard model parameter sizes for benchmarking.
/// These match the requirements: 10K, 100K, 500K, 1M, 2M params.
pub const MODEL_PARAM_SIZES: &[usize] = &[10_000, 100_000, 500_000, 1_000_000, 2_000_000];

/// Model size categories for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSize {
    /// ~500 parameters (2x2x1 MLP) - for quick tests
    Tiny,
    /// ~5K parameters (8x16x4 MLP)
    Small,
    /// ~50K parameters (32x64x16 MLP)
    Medium,
    /// ~500K parameters (128x256x64 MLP)
    Large,
    /// ~2M parameters (256x512x128 MLP)
    XLarge,
    /// Exactly 10K parameters for standardized benchmarking
    Params10K,
    /// Exactly 100K parameters for standardized benchmarking
    Params100K,
    /// Exactly 500K parameters for standardized benchmarking
    Params500K,
    /// Exactly 1M parameters for standardized benchmarking
    Params1M,
    /// Exactly 2M parameters for standardized benchmarking
    Params2M,
}

impl ModelSize {
    /// Returns the model dimensions (d_in, d_hid, d_out).
    /// Dimensions are calculated to achieve approximately the target parameter count.
    /// Params = d_in * d_hid + d_hid + d_hid * d_out + d_out ≈ d_in * d_hid + d_hid * d_out
    pub fn dimensions(&self) -> (usize, usize, usize) {
        match self {
            ModelSize::Tiny => (2, 2, 1),
            ModelSize::Small => (8, 16, 4),
            ModelSize::Medium => (32, 64, 16),
            ModelSize::Large => (128, 256, 64),
            ModelSize::XLarge => (256, 512, 128),
            // Standardized sizes: dimensions chosen to achieve target param count
            // For 2-layer MLP: params ≈ d_in * d_hid + d_hid * d_out
            ModelSize::Params10K => (32, 128, 64), // ~10,272 params
            ModelSize::Params100K => (128, 384, 192), // ~110,016 params
            ModelSize::Params500K => (256, 768, 384), // ~491,520 params
            ModelSize::Params1M => (384, 1024, 512), // ~917,504 params
            ModelSize::Params2M => (512, 1536, 768), // ~1,966,080 params
        }
    }

    /// Returns the approximate parameter count.
    pub fn param_count(&self) -> usize {
        let (d_in, d_hid, d_out) = self.dimensions();
        d_in * d_hid + d_hid + d_hid * d_out + d_out
    }

    /// Returns all model sizes in order.
    pub fn all() -> &'static [ModelSize] {
        &[
            ModelSize::Tiny,
            ModelSize::Small,
            ModelSize::Medium,
            ModelSize::Large,
            ModelSize::XLarge,
        ]
    }

    /// Returns standardized benchmark model sizes (10K, 100K, 500K, 1M, 2M).
    pub fn benchmark_sizes() -> &'static [ModelSize] {
        &[
            ModelSize::Params10K,
            ModelSize::Params100K,
            ModelSize::Params500K,
            ModelSize::Params1M,
            ModelSize::Params2M,
        ]
    }

    /// Returns quick benchmark sizes (for CI).
    pub fn quick_sizes() -> &'static [ModelSize] {
        &[ModelSize::Tiny, ModelSize::Small, ModelSize::Params10K]
    }

    /// Returns the appropriate K value for Halo2 circuits.
    pub fn halo2_k(&self) -> u32 {
        match self {
            ModelSize::Tiny => 10,
            ModelSize::Small => 12,
            ModelSize::Medium => 14,
            ModelSize::Large => 16,
            ModelSize::XLarge => 18,
            ModelSize::Params10K => 13,
            ModelSize::Params100K => 16,
            ModelSize::Params500K => 18,
            ModelSize::Params1M => 19,
            ModelSize::Params2M => 20,
        }
    }

    /// Returns string name for this size.
    pub fn name(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "tiny",
            ModelSize::Small => "small",
            ModelSize::Medium => "medium",
            ModelSize::Large => "large",
            ModelSize::XLarge => "xlarge",
            ModelSize::Params10K => "10k_params",
            ModelSize::Params100K => "100k_params",
            ModelSize::Params500K => "500k_params",
            ModelSize::Params1M => "1m_params",
            ModelSize::Params2M => "2m_params",
        }
    }

    /// Returns the target parameter count category for human display.
    pub fn target_params_display(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "~500",
            ModelSize::Small => "~5K",
            ModelSize::Medium => "~50K",
            ModelSize::Large => "~500K",
            ModelSize::XLarge => "~2M",
            ModelSize::Params10K => "10K",
            ModelSize::Params100K => "100K",
            ModelSize::Params500K => "500K",
            ModelSize::Params1M => "1M",
            ModelSize::Params2M => "2M",
        }
    }
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (d_in, d_hid, d_out) = self.dimensions();
        write!(
            f,
            "{} ({}x{}x{}, ~{}params)",
            self.name(),
            d_in,
            d_hid,
            d_out,
            self.param_count()
        )
    }
}

/// Circuit size categories for GKR benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CircuitSize {
    /// 100 gates
    XSmall,
    /// 1,000 gates
    Small,
    /// 10,000 gates
    Medium,
    /// 100,000 gates
    Large,
    /// 1,000,000 gates
    XLarge,
}

impl CircuitSize {
    pub fn gate_count(&self) -> usize {
        match self {
            CircuitSize::XSmall => 100,
            CircuitSize::Small => 1_000,
            CircuitSize::Medium => 10_000,
            CircuitSize::Large => 100_000,
            CircuitSize::XLarge => 1_000_000,
        }
    }

    pub fn all() -> &'static [CircuitSize] {
        &[
            CircuitSize::XSmall,
            CircuitSize::Small,
            CircuitSize::Medium,
            CircuitSize::Large,
            CircuitSize::XLarge,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            CircuitSize::XSmall => "100_gates",
            CircuitSize::Small => "1k_gates",
            CircuitSize::Medium => "10k_gates",
            CircuitSize::Large => "100k_gates",
            CircuitSize::XLarge => "1m_gates",
        }
    }
}

/// Benchmark timing helper that measures with warm-up and multiple iterations.
pub struct TimingHelper {
    warmup_iterations: usize,
    measure_iterations: usize,
}

impl TimingHelper {
    pub fn new(warmup: usize, measure: usize) -> Self {
        Self {
            warmup_iterations: warmup,
            measure_iterations: measure,
        }
    }

    /// Measures the execution time of a function with warm-up.
    pub fn measure<F, R>(&self, mut f: F) -> TimingResult
    where
        F: FnMut() -> R,
    {
        // Warm-up
        for _ in 0..self.warmup_iterations {
            let _ = std::hint::black_box(f());
        }

        // Measure
        let mut times = Vec::with_capacity(self.measure_iterations);
        for _ in 0..self.measure_iterations {
            let start = std::time::Instant::now();
            let _ = std::hint::black_box(f());
            times.push(start.elapsed());
        }

        TimingResult::from_samples(&times)
    }

    /// Quick measurement with default settings.
    pub fn quick() -> Self {
        Self::new(2, 5)
    }

    /// Precise measurement with more iterations.
    pub fn precise() -> Self {
        Self::new(5, 20)
    }
}

impl Default for TimingHelper {
    fn default() -> Self {
        Self::new(3, 10)
    }
}

/// Result of timing measurements with statistics.
#[derive(Debug, Clone)]
pub struct TimingResult {
    pub mean: Duration,
    pub min: Duration,
    pub max: Duration,
    pub stddev: Duration,
    pub samples: Vec<Duration>,
}

impl TimingResult {
    pub fn from_samples(samples: &[Duration]) -> Self {
        let n = samples.len() as f64;
        let mean_ns: f64 = samples.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / n;
        let min = *samples.iter().min().unwrap_or(&Duration::ZERO);
        let max = *samples.iter().max().unwrap_or(&Duration::ZERO);

        let variance: f64 = samples
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_ns;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let stddev = Duration::from_nanos(variance.sqrt() as u64);

        Self {
            mean: Duration::from_nanos(mean_ns as u64),
            min,
            max,
            stddev,
            samples: samples.to_vec(),
        }
    }

    /// Returns mean time in milliseconds.
    pub fn mean_ms(&self) -> f64 {
        self.mean.as_secs_f64() * 1000.0
    }

    /// Returns mean time in microseconds.
    pub fn mean_us(&self) -> f64 {
        self.mean.as_secs_f64() * 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_sizes() {
        for size in ModelSize::all() {
            let (d_in, d_hid, d_out) = size.dimensions();
            assert!(d_in > 0);
            assert!(d_hid > 0);
            assert!(d_out > 0);
            assert!(size.param_count() > 0);
        }
    }

    #[test]
    fn test_circuit_sizes() {
        for size in CircuitSize::all() {
            assert!(size.gate_count() > 0);
        }
    }

    #[test]
    fn test_timing_helper() {
        let helper = TimingHelper::quick();
        let result = helper.measure(|| {
            std::thread::sleep(Duration::from_micros(100));
            42
        });
        assert!(result.mean >= Duration::from_micros(50));
        assert!(result.samples.len() == 5);
    }
}
