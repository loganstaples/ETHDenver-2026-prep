//! Scaling Tests
//!
//! This module analyzes how overhead changes with model size, providing
//! crucial insights for understanding HELIX performance characteristics.
//!
//! # Key Questions Answered
//!
//! - How does overhead scale with model parameters?
//! - At what model size does overhead become prohibitive?
//! - What is the asymptotic behavior of proving cost?
//! - How do GKR and Halo2 compare at different scales?
//!
//! # Analysis Types
//!
//! - Linear scaling analysis
//! - Log-log scaling (power law detection)
//! - Crossover point analysis (where one prover beats another)
//! - Projected overhead for production model sizes

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_prover::gkr::{GKRConfig, GKRProver};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::{
    gkr_prover::create_mlp_circuit, native_baseline::NativeBaseline, BenchmarkHarness,
    BenchmarkMetrics, CircuitSize, ModelSize, TimingHelper, TARGET_OVERHEAD_MULTIPLE,
};

/// A single data point in scaling analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPoint {
    /// Model/circuit size parameter.
    pub size: usize,
    /// Native computation time.
    pub native_time_us: f64,
    /// Proof generation time.
    pub proof_time_us: f64,
    /// Computed overhead.
    pub overhead: f64,
    /// Proof size in bytes.
    pub proof_size: usize,
    /// Memory usage (if measured).
    pub memory_bytes: Option<usize>,
}

impl ScalingPoint {
    pub fn new(size: usize, native: Duration, proof: Duration, proof_size: usize) -> Self {
        let native_us = native.as_secs_f64() * 1_000_000.0;
        let proof_us = proof.as_secs_f64() * 1_000_000.0;
        let overhead = if native_us > 0.0 {
            proof_us / native_us
        } else {
            f64::INFINITY
        };

        Self {
            size,
            native_time_us: native_us,
            proof_time_us: proof_us,
            overhead,
            proof_size,
            memory_bytes: None,
        }
    }

    pub fn with_memory(mut self, memory: usize) -> Self {
        self.memory_bytes = Some(memory);
        self
    }
}

/// Complete scaling analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    /// Analysis name.
    pub name: String,
    /// Data points collected.
    pub points: Vec<ScalingPoint>,
    /// Fitted power law exponent (if applicable).
    pub power_law_exponent: Option<f64>,
    /// Linear fit slope (overhead per param).
    pub linear_slope: Option<f64>,
    /// Estimated max model size meeting target overhead.
    pub max_feasible_size: Option<usize>,
    /// Summary statistics.
    pub summary: ScalingSummary,
}

/// Summary statistics for scaling analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScalingSummary {
    /// Smallest size tested.
    pub min_size: usize,
    /// Largest size tested.
    pub max_size: usize,
    /// Minimum overhead observed.
    pub min_overhead: f64,
    /// Maximum overhead observed.
    pub max_overhead: f64,
    /// Whether overhead stays within target across all sizes.
    pub all_within_target: bool,
    /// Scaling behavior description.
    pub scaling_description: String,
}

impl ScalingAnalysis {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            points: Vec::new(),
            power_law_exponent: None,
            linear_slope: None,
            max_feasible_size: None,
            summary: ScalingSummary::default(),
        }
    }

    /// Adds a data point.
    pub fn add_point(&mut self, point: ScalingPoint) {
        self.points.push(point);
    }

    /// Analyzes the collected data.
    pub fn analyze(&mut self) {
        if self.points.is_empty() {
            return;
        }

        // Sort by size
        self.points.sort_by_key(|p| p.size);

        // Basic summary
        let min_size = self.points.first().map(|p| p.size).unwrap_or(0);
        let max_size = self.points.last().map(|p| p.size).unwrap_or(0);
        let min_overhead = self
            .points
            .iter()
            .map(|p| p.overhead)
            .fold(f64::INFINITY, f64::min);
        let max_overhead = self
            .points
            .iter()
            .map(|p| p.overhead)
            .fold(0.0f64, f64::max);
        let all_within = self
            .points
            .iter()
            .all(|p| p.overhead <= TARGET_OVERHEAD_MULTIPLE);

        // Fit power law: overhead = a * size^b
        // log(overhead) = log(a) + b * log(size)
        if self.points.len() >= 3 {
            let log_sizes: Vec<f64> = self.points.iter().map(|p| (p.size as f64).ln()).collect();
            let log_overheads: Vec<f64> = self.points.iter().map(|p| p.overhead.ln()).collect();

            // Simple linear regression on log-log
            if let Some(exponent) = fit_linear_slope(&log_sizes, &log_overheads) {
                self.power_law_exponent = Some(exponent);
            }
        }

        // Fit linear slope: overhead = a + b * size
        if self.points.len() >= 2 {
            let sizes: Vec<f64> = self.points.iter().map(|p| p.size as f64).collect();
            let overheads: Vec<f64> = self.points.iter().map(|p| p.overhead).collect();

            if let Some(slope) = fit_linear_slope(&sizes, &overheads) {
                self.linear_slope = Some(slope);
            }
        }

        // Estimate max feasible size
        // If overhead = a + b * size <= TARGET, then size <= (TARGET - a) / b
        if let Some(slope) = self.linear_slope {
            if slope > 0.0 && self.points.len() > 0 {
                let intercept = min_overhead; // Approximate
                if intercept < TARGET_OVERHEAD_MULTIPLE {
                    let max_feasible = ((TARGET_OVERHEAD_MULTIPLE - intercept) / slope) as usize;
                    if max_feasible > max_size {
                        self.max_feasible_size = Some(max_feasible);
                    }
                }
            }
        }

        // Determine scaling description
        let scaling_desc = match self.power_law_exponent {
            Some(exp) if exp < 0.1 => "constant (O(1))".to_string(),
            Some(exp) if exp < 0.6 => "sub-linear (O(n^{:.2}))"
                .to_string()
                .replace("{:.2}", &format!("{:.2}", exp)),
            Some(exp) if exp < 1.1 => "linear (O(n))".to_string(),
            Some(exp) if exp < 1.6 => "slightly super-linear (O(n^{:.2}))"
                .to_string()
                .replace("{:.2}", &format!("{:.2}", exp)),
            Some(exp) if exp < 2.1 => "quadratic (O(n^2))".to_string(),
            Some(exp) => format!("polynomial (O(n^{:.2}))", exp),
            None => "unknown".to_string(),
        };

        self.summary = ScalingSummary {
            min_size,
            max_size,
            min_overhead,
            max_overhead,
            all_within_target: all_within,
            scaling_description: scaling_desc,
        };
    }

    /// Generates a markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("## {}\n\n", self.name));

        // Summary table
        md.push_str("### Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Size Range | {} - {} |\n",
            self.summary.min_size, self.summary.max_size
        ));
        md.push_str(&format!(
            "| Overhead Range | {:.1}x - {:.1}x |\n",
            self.summary.min_overhead, self.summary.max_overhead
        ));
        md.push_str(&format!(
            "| Scaling | {} |\n",
            self.summary.scaling_description
        ));
        md.push_str(&format!(
            "| All Within Target | {} |\n",
            if self.summary.all_within_target {
                "Yes"
            } else {
                "No"
            }
        ));

        if let Some(exponent) = self.power_law_exponent {
            md.push_str(&format!("| Power Law Exponent | {:.3} |\n", exponent));
        }

        if let Some(max_feasible) = self.max_feasible_size {
            md.push_str(&format!("| Est. Max Feasible Size | {} |\n", max_feasible));
        }

        md.push_str("\n### Data Points\n\n");
        md.push_str("| Size | Native (μs) | Proof (μs) | Overhead | Proof Size |\n");
        md.push_str("|------|-------------|------------|----------|------------|\n");

        for point in &self.points {
            md.push_str(&format!(
                "| {} | {:.1} | {:.1} | {:.1}x | {} |\n",
                point.size,
                point.native_time_us,
                point.proof_time_us,
                point.overhead,
                point.proof_size,
            ));
        }

        md
    }

    /// Prints to terminal.
    pub fn print_terminal(&self) {
        println!("\n{}", "=".repeat(70));
        println!("Scaling Analysis: {}", self.name);
        println!("{}", "=".repeat(70));

        println!("\nSummary:");
        println!(
            "  Size range: {} - {}",
            self.summary.min_size, self.summary.max_size
        );
        println!(
            "  Overhead range: {:.1}x - {:.1}x",
            self.summary.min_overhead, self.summary.max_overhead
        );
        println!("  Scaling behavior: {}", self.summary.scaling_description);
        println!(
            "  All within {:.0}x target: {}",
            TARGET_OVERHEAD_MULTIPLE,
            if self.summary.all_within_target {
                "Yes"
            } else {
                "No"
            }
        );

        if let Some(exp) = self.power_law_exponent {
            println!("  Power law exponent: {:.3}", exp);
        }

        println!("\nData Points:");
        println!(
            "{:<12} {:>12} {:>12} {:>10} {:>12}",
            "Size", "Native(μs)", "Proof(μs)", "Overhead", "Proof Size"
        );
        println!("{}", "-".repeat(60));

        for point in &self.points {
            println!(
                "{:<12} {:>12.1} {:>12.1} {:>9.1}x {:>12}",
                point.size,
                point.native_time_us,
                point.proof_time_us,
                point.overhead,
                point.proof_size,
            );
        }
    }
}

/// Fits a linear slope using least squares.
fn fit_linear_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator.abs() < 1e-10 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    Some(slope)
}

/// Runs scaling analysis for GKR proving on model sizes.
/// Uses the standard benchmark sizes (10K, 100K, 500K, 1M, 2M params).
pub fn analyze_model_size_scaling() -> ScalingAnalysis {
    analyze_model_size_scaling_with_sizes(ModelSize::benchmark_sizes())
}

/// Runs scaling analysis for GKR proving on specified model sizes.
pub fn analyze_model_size_scaling_with_sizes(sizes: &[ModelSize]) -> ScalingAnalysis {
    let mut analysis = ScalingAnalysis::new("GKR Proving vs Model Size");
    let timing = TimingHelper::quick();

    for &size in sizes {
        let baseline = NativeBaseline::new(size);
        let (d_in, d_hid, d_out) = baseline.dimensions();
        let input = baseline.random_input();
        let target = baseline.random_target();

        // Measure native time
        let native_result = timing.measure(|| {
            let mut b = NativeBaseline::new(size);
            b.training_step(&input, &target)
        });

        // Create circuit and measure proving
        let circuit = create_mlp_circuit(d_in, d_hid, d_out);
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
        let config = GKRConfig::for_testing();

        let proof_result = timing.measure(|| {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });

        // Get proof size
        let mut prover = GKRProver::new(config);
        let proof = prover.prove(&circuit, &inputs).unwrap();

        analysis.add_point(ScalingPoint::new(
            size.param_count(),
            native_result.mean,
            proof_result.mean,
            proof.size_bytes(),
        ));
    }

    analysis.analyze();
    analysis
}

/// Runs scaling analysis for GKR on circuit sizes.
pub fn analyze_circuit_size_scaling() -> ScalingAnalysis {
    let mut analysis = ScalingAnalysis::new("GKR Proving vs Circuit Size");
    let timing = TimingHelper::quick();

    use super::gkr_prover::create_benchmark_circuit;

    for &size in CircuitSize::all() {
        let gates = size.gate_count();

        // Skip very large circuits for quick analysis
        if gates > 100_000 {
            continue;
        }

        let circuit = create_benchmark_circuit(gates);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];
        let config = GKRConfig::for_testing();

        // Measure native evaluation
        let native_result = timing.measure(|| circuit.evaluate(&inputs));

        // Measure proving
        let proof_result = timing.measure(|| {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });

        // Get proof size
        let mut prover = GKRProver::new(config);
        let proof = prover.prove(&circuit, &inputs).unwrap();

        analysis.add_point(ScalingPoint::new(
            gates,
            native_result.mean,
            proof_result.mean,
            proof.size_bytes(),
        ));
    }

    analysis.analyze();
    analysis
}

/// Runs scaling analysis for proof sizes.
pub fn analyze_proof_size_scaling() -> ScalingAnalysis {
    let mut analysis = ScalingAnalysis::new("Proof Size vs Model Size");

    for &size in ModelSize::all() {
        let (d_in, d_hid, d_out) = size.dimensions();
        let circuit = create_mlp_circuit(d_in, d_hid, d_out);
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
        let config = GKRConfig::for_testing();

        let mut prover = GKRProver::new(config);
        let proof = prover.prove(&circuit, &inputs).unwrap();

        // For proof size analysis, use proof size as the "native time" analog
        analysis.add_point(ScalingPoint {
            size: size.param_count(),
            native_time_us: size.param_count() as f64, // Just for scaling reference
            proof_time_us: proof.size_bytes() as f64,
            overhead: proof.size_bytes() as f64 / size.param_count() as f64, // bytes per param
            proof_size: proof.size_bytes(),
            memory_bytes: None,
        });
    }

    analysis.analyze();
    analysis
}

/// Runs all scaling tests.
pub fn run_scaling_tests(harness: &mut BenchmarkHarness) {
    println!("\nRunning scaling analysis...\n");

    // Model size scaling
    let model_analysis = analyze_model_size_scaling();
    model_analysis.print_terminal();

    // Circuit size scaling
    let circuit_analysis = analyze_circuit_size_scaling();
    circuit_analysis.print_terminal();

    // Proof size scaling
    let proof_analysis = analyze_proof_size_scaling();
    proof_analysis.print_terminal();

    // Record key metrics
    for point in &model_analysis.points {
        let metrics = BenchmarkMetrics {
            mean_ns: point.proof_time_us * 1000.0,
            stddev_ns: 0.0,
            min_ns: point.proof_time_us * 1000.0,
            max_ns: point.proof_time_us * 1000.0,
            iterations: 1,
            throughput_ops: None,
            memory_bytes: point.memory_bytes,
            tags: BTreeMap::new(),
        };
        harness.record_benchmark(&format!("scaling_model_{}", point.size), "scaling", metrics);
    }
}

/// Generates a combined scaling report.
pub fn generate_scaling_report() -> String {
    let model_analysis = analyze_model_size_scaling();
    let circuit_analysis = analyze_circuit_size_scaling();
    let proof_analysis = analyze_proof_size_scaling();

    let mut report = String::new();
    report.push_str("# HELIX Scaling Analysis Report\n\n");

    report.push_str(&format!(
        "**Target Overhead:** ≤{:.0}x\n\n",
        TARGET_OVERHEAD_MULTIPLE
    ));

    report.push_str(&model_analysis.to_markdown());
    report.push_str("\n");
    report.push_str(&circuit_analysis.to_markdown());
    report.push_str("\n");
    report.push_str(&proof_analysis.to_markdown());

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_point_creation() {
        let point = ScalingPoint::new(
            100,
            Duration::from_micros(10),
            Duration::from_micros(200),
            1024,
        );

        assert_eq!(point.size, 100);
        assert!((point.overhead - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_fit_linear_slope() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let slope = fit_linear_slope(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_scaling_analysis() {
        let mut analysis = ScalingAnalysis::new("Test");

        analysis.add_point(ScalingPoint::new(
            10,
            Duration::from_micros(1),
            Duration::from_micros(10),
            100,
        ));
        analysis.add_point(ScalingPoint::new(
            100,
            Duration::from_micros(10),
            Duration::from_micros(150),
            500,
        ));
        analysis.add_point(ScalingPoint::new(
            1000,
            Duration::from_micros(100),
            Duration::from_micros(2000),
            2000,
        ));

        analysis.analyze();

        assert_eq!(analysis.summary.min_size, 10);
        assert_eq!(analysis.summary.max_size, 1000);
        assert!(analysis.points.len() == 3);
    }
}
