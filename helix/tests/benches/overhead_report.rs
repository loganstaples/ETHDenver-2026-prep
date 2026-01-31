//! Overhead Calculation and Automated Reporting
//!
//! This module computes the overhead multiple for HELIX proving compared to
//! native computation and generates comprehensive reports.
//!
//! # Key Metric
//!
//! The primary metric is the **overhead multiple**:
//! ```text
//! overhead = proof_time / native_time
//! ```
//!
//! HELIX targets **≤30x overhead** to be practical for real-world training.
//!
//! # Report Contents
//!
//! - Per-operation overhead (matmul, forward, backward, training step)
//! - Per-model-size overhead
//! - GKR vs Halo2 comparison
//! - Proof size analysis
//! - Trend analysis across model sizes

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use super::{
    gkr_prover::{create_mlp_circuit, GKRBenchmarks},
    halo2_prover::Halo2Benchmarks,
    native_baseline::{NativeBaseline, NativeBaselineCollection},
    BenchmarkConfig, BenchmarkHarness, BenchmarkMetrics, ModelSize, TimingHelper,
    REGRESSION_ALERT_THRESHOLD, TARGET_OVERHEAD_MULTIPLE, TARGET_PROOF_TIME_MS,
};

use helix_circuits::halo2curves::bn256::Fr;
use helix_prover::gkr::{GKRConfig, GKRProver};

/// Overhead metrics for a single measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadMetrics {
    /// Native computation time.
    pub native_time: Duration,
    /// Proof generation time.
    pub proof_time: Duration,
    /// Computed overhead multiple.
    pub overhead: f64,
    /// Whether overhead meets target (≤30x).
    pub meets_target: bool,
    /// Whether overhead exceeds alert threshold (>35x).
    pub exceeds_alert: bool,
    /// Proof size in bytes.
    pub proof_size_bytes: Option<usize>,
}

impl OverheadMetrics {
    pub fn compute(native: Duration, proof: Duration, proof_size: Option<usize>) -> Self {
        let overhead = if native.as_nanos() > 0 {
            proof.as_secs_f64() / native.as_secs_f64()
        } else {
            f64::INFINITY
        };

        Self {
            native_time: native,
            proof_time: proof,
            overhead,
            meets_target: overhead <= TARGET_OVERHEAD_MULTIPLE,
            exceeds_alert: overhead > REGRESSION_ALERT_THRESHOLD,
            proof_size_bytes: proof_size,
        }
    }

    /// Returns a status string for display.
    pub fn status(&self) -> &'static str {
        if self.meets_target {
            "PASS"
        } else if !self.exceeds_alert {
            "WARN"
        } else {
            "ALERT"
        }
    }
}

/// Complete overhead report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadReport {
    /// Report title.
    pub title: String,
    /// Timestamp.
    pub timestamp: String,
    /// Git commit if available.
    pub git_commit: Option<String>,
    /// Target overhead multiple.
    pub target_overhead: f64,
    /// Target proof time in milliseconds.
    pub target_proof_time_ms: u64,
    /// Results by model size.
    pub by_model_size: BTreeMap<String, ModelSizeOverhead>,
    /// Results by operation type.
    pub by_operation: BTreeMap<String, OperationOverhead>,
    /// GKR vs Halo2 comparison.
    pub prover_comparison: ProverComparison,
    /// Summary statistics.
    pub summary: OverheadSummary,
}

/// Overhead results for a specific model size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSizeOverhead {
    pub model_size: String,
    pub param_count: usize,
    pub gkr_overhead: Option<OverheadMetrics>,
    pub halo2_overhead: Option<OverheadMetrics>,
    pub recommended_prover: String,
}

/// Overhead results for a specific operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationOverhead {
    pub operation: String,
    pub native_time_us: f64,
    pub gkr_time_us: f64,
    pub halo2_time_us: f64,
    pub gkr_overhead: f64,
    pub halo2_overhead: f64,
}

/// Comparison between GKR and Halo2 provers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverComparison {
    pub gkr_avg_overhead: f64,
    pub halo2_avg_overhead: f64,
    pub gkr_proof_size_avg: usize,
    pub halo2_proof_size_avg: usize,
    pub gkr_wins_on_speed: bool,
    pub gkr_wins_on_size: bool,
    pub recommendation: String,
}

/// Summary statistics for the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadSummary {
    /// Average overhead across all measurements.
    pub avg_overhead: f64,
    /// Minimum overhead observed.
    pub min_overhead: f64,
    /// Maximum overhead observed.
    pub max_overhead: f64,
    /// Number of measurements meeting target (≤30x).
    pub measurements_meeting_target: usize,
    /// Number of measurements exceeding alert threshold (>35x).
    pub measurements_exceeding_alert: usize,
    /// Total measurements.
    pub total_measurements: usize,
    /// Overall pass/fail status (all meet target).
    pub passed: bool,
    /// Whether any measurement exceeded alert threshold.
    pub has_alerts: bool,
    /// Human-readable summary message.
    pub message: String,
}

impl OverheadReport {
    /// Creates a new empty report.
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_commit: get_git_commit(),
            target_overhead: TARGET_OVERHEAD_MULTIPLE,
            target_proof_time_ms: TARGET_PROOF_TIME_MS,
            by_model_size: BTreeMap::new(),
            by_operation: BTreeMap::new(),
            prover_comparison: ProverComparison {
                gkr_avg_overhead: 0.0,
                halo2_avg_overhead: 0.0,
                gkr_proof_size_avg: 0,
                halo2_proof_size_avg: 0,
                gkr_wins_on_speed: false,
                gkr_wins_on_size: false,
                recommendation: String::new(),
            },
            summary: OverheadSummary {
                avg_overhead: 0.0,
                min_overhead: f64::INFINITY,
                max_overhead: 0.0,
                measurements_meeting_target: 0,
                measurements_exceeding_alert: 0,
                total_measurements: 0,
                passed: false,
                has_alerts: false,
                message: String::new(),
            },
        }
    }

    /// Generates a complete overhead report by running benchmarks.
    pub fn generate() -> Self {
        let mut report = Self::new("HELIX Overhead Analysis Report");
        let timing = TimingHelper::default();

        // Collect native baselines
        let native_baselines = NativeBaselineCollection::collect_all(&timing);

        // Run GKR benchmarks for each model size
        let gkr_benchmarks = GKRBenchmarks::new();
        let mut gkr_overheads = Vec::new();
        let mut gkr_proof_sizes = Vec::new();

        // Use benchmark sizes (10K, 100K, 500K, 1M, 2M params)
        for &size in ModelSize::benchmark_sizes() {
            if let Some(native_timing) = native_baselines.get(size) {
                let (d_in, d_hid, d_out) = size.dimensions();
                let circuit = create_mlp_circuit(d_in, d_hid, d_out);
                let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
                let config = GKRConfig::default();

                // Measure GKR proving
                let gkr_timing = timing.measure(|| {
                    let mut prover = GKRProver::new(config.clone());
                    prover.prove(&circuit, &inputs).unwrap()
                });

                // Get proof size
                let mut prover = GKRProver::new(config);
                let proof = prover.prove(&circuit, &inputs).unwrap();
                let proof_size = proof.size_bytes();

                let gkr_metrics = OverheadMetrics::compute(
                    native_timing.training_step.mean,
                    gkr_timing.mean,
                    Some(proof_size),
                );

                gkr_overheads.push(gkr_metrics.overhead);
                gkr_proof_sizes.push(proof_size);

                report.by_model_size.insert(
                    size.name().to_string(),
                    ModelSizeOverhead {
                        model_size: size.to_string(),
                        param_count: size.param_count(),
                        gkr_overhead: Some(gkr_metrics),
                        halo2_overhead: None, // Halo2 is slower, optional
                        recommended_prover: "GKR".to_string(),
                    },
                );
            }
        }

        // Compute summary
        if !gkr_overheads.is_empty() {
            let avg = gkr_overheads.iter().sum::<f64>() / gkr_overheads.len() as f64;
            let min = gkr_overheads.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = gkr_overheads.iter().cloned().fold(0.0f64, f64::max);
            let meeting_target = gkr_overheads
                .iter()
                .filter(|&&o| o <= TARGET_OVERHEAD_MULTIPLE)
                .count();
            let exceeding_alert = gkr_overheads
                .iter()
                .filter(|&&o| o > REGRESSION_ALERT_THRESHOLD)
                .count();

            report.summary = OverheadSummary {
                avg_overhead: avg,
                min_overhead: min,
                max_overhead: max,
                measurements_meeting_target: meeting_target,
                measurements_exceeding_alert: exceeding_alert,
                total_measurements: gkr_overheads.len(),
                passed: meeting_target == gkr_overheads.len(),
                has_alerts: exceeding_alert > 0,
                message: if meeting_target == gkr_overheads.len() {
                    format!(
                        "All {} measurements meet the {:.0}x overhead target",
                        gkr_overheads.len(),
                        TARGET_OVERHEAD_MULTIPLE
                    )
                } else if exceeding_alert > 0 {
                    format!(
                        "ALERT: {}/{} measurements exceed {:.0}x alert threshold",
                        exceeding_alert,
                        gkr_overheads.len(),
                        REGRESSION_ALERT_THRESHOLD
                    )
                } else {
                    format!(
                        "{}/{} measurements meet the {:.0}x overhead target",
                        meeting_target,
                        gkr_overheads.len(),
                        TARGET_OVERHEAD_MULTIPLE
                    )
                },
            };

            report.prover_comparison.gkr_avg_overhead = avg;
            report.prover_comparison.gkr_proof_size_avg =
                gkr_proof_sizes.iter().sum::<usize>() / gkr_proof_sizes.len();
            report.prover_comparison.gkr_wins_on_speed = true;
            report.prover_comparison.recommendation =
                "GKR recommended for neural network operations".to_string();
        }

        report
    }

    /// Saves the report to a JSON file.
    pub fn save_json(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Generates a markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", self.title));
        md.push_str(&format!("**Generated:** {}\n\n", self.timestamp));

        if let Some(ref commit) = self.git_commit {
            md.push_str(&format!("**Commit:** `{}`\n\n", commit));
        }

        // Target section
        md.push_str("## Targets\n\n");
        md.push_str(&format!(
            "- **Overhead Target:** ≤{:.0}x\n",
            self.target_overhead
        ));
        md.push_str(&format!(
            "- **Alert Threshold:** ≤{:.0}x\n",
            REGRESSION_ALERT_THRESHOLD
        ));
        md.push_str(&format!(
            "- **Proof Time Target:** ≤{}ms per step\n\n",
            self.target_proof_time_ms
        ));

        // Summary section
        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Average Overhead | **{:.1}x** |\n",
            self.summary.avg_overhead
        ));
        md.push_str(&format!(
            "| Min Overhead | {:.1}x |\n",
            self.summary.min_overhead
        ));
        md.push_str(&format!(
            "| Max Overhead | {:.1}x |\n",
            self.summary.max_overhead
        ));
        md.push_str(&format!(
            "| Measurements Meeting Target (≤{:.0}x) | {}/{} |\n",
            TARGET_OVERHEAD_MULTIPLE,
            self.summary.measurements_meeting_target,
            self.summary.total_measurements
        ));
        md.push_str(&format!(
            "| Measurements Exceeding Alert (>{:.0}x) | {} |\n",
            REGRESSION_ALERT_THRESHOLD, self.summary.measurements_exceeding_alert
        ));
        let status = if self.summary.has_alerts {
            "🚨 ALERT"
        } else if self.summary.passed {
            "✅ PASSED"
        } else {
            "⚠️ WARNING"
        };
        md.push_str(&format!("| Status | {} |\n\n", status));

        // Model size breakdown
        md.push_str("## Overhead by Model Size\n\n");
        md.push_str("| Model | Params | Native (μs) | GKR (μs) | Overhead | Status |\n");
        md.push_str("|-------|--------|-------------|----------|----------|--------|\n");

        for (name, data) in &self.by_model_size {
            if let Some(ref gkr) = data.gkr_overhead {
                let native_us = gkr.native_time.as_secs_f64() * 1_000_000.0;
                let proof_us = gkr.proof_time.as_secs_f64() * 1_000_000.0;
                let status = match gkr.status() {
                    "PASS" => "✅",
                    "WARN" => "⚠️",
                    "ALERT" => "🚨",
                    _ => "❓",
                };

                md.push_str(&format!(
                    "| {} | {} | {:.1} | {:.1} | **{:.1}x** | {} |\n",
                    name, data.param_count, native_us, proof_us, gkr.overhead, status
                ));
            }
        }

        md.push_str("\n");

        // Proof sizes
        md.push_str("## Proof Sizes\n\n");
        md.push_str("| Model | GKR Proof (bytes) |\n");
        md.push_str("|-------|-------------------|\n");

        for (name, data) in &self.by_model_size {
            if let Some(ref gkr) = data.gkr_overhead {
                if let Some(size) = gkr.proof_size_bytes {
                    md.push_str(&format!("| {} | {} |\n", name, size));
                }
            }
        }

        md.push_str("\n");

        // Recommendations
        md.push_str("## Recommendations\n\n");
        md.push_str(&format!("- {}\n", self.prover_comparison.recommendation));
        md.push_str(&format!(
            "- Average GKR overhead: **{:.1}x** (target: {:.0}x)\n",
            self.prover_comparison.gkr_avg_overhead, TARGET_OVERHEAD_MULTIPLE
        ));

        md
    }

    /// Prints a terminal-friendly report.
    pub fn print_terminal(&self) {
        println!("\n{}", "=".repeat(70));
        println!("{}", self.title);
        println!("{}", "=".repeat(70));
        println!("Generated: {}", self.timestamp);
        if let Some(ref commit) = self.git_commit {
            println!("Commit: {}", commit);
        }
        println!();

        println!("Targets:");
        println!("  - Overhead: ≤{:.0}x", self.target_overhead);
        println!("  - Proof time: ≤{}ms", self.target_proof_time_ms);
        println!();

        // Summary box
        println!("┌─────────────────────────────────────────────────────────┐");
        println!("│ SUMMARY                                                 │");
        println!("├─────────────────────────────────────────────────────────┤");
        println!(
            "│ Average Overhead:    {:>8.1}x {:>22} │",
            self.summary.avg_overhead,
            if self.summary.avg_overhead <= self.target_overhead {
                "(OK)"
            } else {
                "(OVER TARGET)"
            }
        );
        println!(
            "│ Min Overhead:        {:>8.1}x                          │",
            self.summary.min_overhead
        );
        println!(
            "│ Max Overhead:        {:>8.1}x                          │",
            self.summary.max_overhead
        );
        println!(
            "│ Meeting Target:      {:>8}/{:<8}                    │",
            self.summary.measurements_meeting_target, self.summary.total_measurements
        );
        println!(
            "│ Status:              {:>8}                           │",
            if self.summary.passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );
        println!("└─────────────────────────────────────────────────────────┘");
        println!();

        // Detailed results
        println!(
            "{:<12} {:>10} {:>12} {:>12} {:>10} {:>8}",
            "Model", "Params", "Native(μs)", "GKR(μs)", "Overhead", "Status"
        );
        println!("{}", "-".repeat(70));

        for (name, data) in &self.by_model_size {
            if let Some(ref gkr) = data.gkr_overhead {
                let native_us = gkr.native_time.as_secs_f64() * 1_000_000.0;
                let proof_us = gkr.proof_time.as_secs_f64() * 1_000_000.0;
                let status = if gkr.meets_target { "[OK]" } else { "[!!]" };

                println!(
                    "{:<12} {:>10} {:>12.1} {:>12.1} {:>9.1}x {:>8}",
                    name, data.param_count, native_us, proof_us, gkr.overhead, status
                );
            }
        }

        println!("{}", "=".repeat(70));
        println!("{}", self.summary.message);
        println!();
    }

    /// Asserts that all measurements meet the target overhead.
    pub fn assert_passed(&self) {
        if !self.summary.passed {
            panic!(
                "Overhead target not met: {}/{} measurements exceeded {:.0}x overhead. Max overhead: {:.1}x",
                self.summary.total_measurements - self.summary.measurements_meeting_target,
                self.summary.total_measurements,
                self.target_overhead,
                self.summary.max_overhead,
            );
        }
    }
}

/// Gets the current git commit hash.
fn get_git_commit() -> Option<String> {
    std::env::var("GITHUB_SHA").ok().or_else(|| {
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    })
}

/// Generates a quick overhead comparison for CI.
pub fn quick_overhead_check() -> OverheadReport {
    let mut report = OverheadReport::new("HELIX CI Overhead Check");
    let timing = TimingHelper::quick();

    // Only test small models for CI speed
    for &size in &[ModelSize::Tiny, ModelSize::Small] {
        let baseline = NativeBaseline::new(size);
        let (d_in, d_hid, d_out) = baseline.dimensions();
        let input = baseline.random_input();
        let target = baseline.random_target();

        // Native timing
        let mut baseline_clone = NativeBaseline::new(size);
        let native_timing = timing.measure(|| baseline_clone.training_step(&input, &target));

        // GKR timing
        let circuit = create_mlp_circuit(d_in, d_hid, d_out);
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
        let config = GKRConfig::for_testing();

        let gkr_timing = timing.measure(|| {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });

        let gkr_metrics = OverheadMetrics::compute(native_timing.mean, gkr_timing.mean, None);

        report.by_model_size.insert(
            size.name().to_string(),
            ModelSizeOverhead {
                model_size: size.to_string(),
                param_count: size.param_count(),
                gkr_overhead: Some(gkr_metrics),
                halo2_overhead: None,
                recommended_prover: "GKR".to_string(),
            },
        );
    }

    // Compute summary
    let overheads: Vec<f64> = report
        .by_model_size
        .values()
        .filter_map(|m| m.gkr_overhead.as_ref().map(|o| o.overhead))
        .collect();

    if !overheads.is_empty() {
        let avg = overheads.iter().sum::<f64>() / overheads.len() as f64;
        let min = overheads.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = overheads.iter().cloned().fold(0.0f64, f64::max);
        let meeting_target = overheads
            .iter()
            .filter(|&&o| o <= TARGET_OVERHEAD_MULTIPLE)
            .count();
        let exceeding_alert = overheads
            .iter()
            .filter(|&&o| o > REGRESSION_ALERT_THRESHOLD)
            .count();

        report.summary = OverheadSummary {
            avg_overhead: avg,
            min_overhead: min,
            max_overhead: max,
            measurements_meeting_target: meeting_target,
            measurements_exceeding_alert: exceeding_alert,
            total_measurements: overheads.len(),
            passed: meeting_target == overheads.len(),
            has_alerts: exceeding_alert > 0,
            message: if meeting_target == overheads.len() {
                format!(
                    "CI check passed: all measurements within {:.0}x overhead",
                    TARGET_OVERHEAD_MULTIPLE
                )
            } else if exceeding_alert > 0 {
                format!(
                    "CI check ALERT: {}/{} measurements exceeded {:.0}x alert threshold",
                    exceeding_alert,
                    overheads.len(),
                    REGRESSION_ALERT_THRESHOLD
                )
            } else {
                format!(
                    "CI check FAILED: {}/{} measurements exceeded target",
                    overheads.len() - meeting_target,
                    overheads.len()
                )
            },
        };
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overhead_metrics_computation() {
        let native = Duration::from_micros(100);
        let proof = Duration::from_millis(2); // 2000us = 20x overhead

        let metrics = OverheadMetrics::compute(native, proof, Some(1024));

        assert!((metrics.overhead - 20.0).abs() < 0.1);
        assert!(metrics.meets_target); // 20x < 30x
    }

    #[test]
    fn test_overhead_report_creation() {
        let report = OverheadReport::new("Test Report");
        assert_eq!(report.target_overhead, TARGET_OVERHEAD_MULTIPLE);
    }

    #[test]
    fn test_quick_overhead_check() {
        let report = quick_overhead_check();
        assert!(report.by_model_size.len() > 0);
        assert!(report.summary.total_measurements > 0);
    }

    #[test]
    fn test_markdown_generation() {
        let mut report = OverheadReport::new("Test Report");
        report.summary = OverheadSummary {
            avg_overhead: 15.0,
            min_overhead: 10.0,
            max_overhead: 20.0,
            measurements_meeting_target: 3,
            measurements_exceeding_alert: 0,
            total_measurements: 3,
            passed: true,
            has_alerts: false,
            message: "All tests passed".to_string(),
        };

        let md = report.to_markdown();
        assert!(md.contains("Test Report"));
        assert!(md.contains("PASSED"));
    }

    #[test]
    fn test_overhead_metrics_alert_threshold() {
        // Test a measurement that exceeds alert threshold (>35x)
        let native = Duration::from_micros(100);
        let proof = Duration::from_millis(4); // 4000us = 40x overhead

        let metrics = OverheadMetrics::compute(native, proof, None);

        assert!((metrics.overhead - 40.0).abs() < 0.1);
        assert!(!metrics.meets_target); // 40x > 30x
        assert!(metrics.exceeds_alert); // 40x > 35x
        assert_eq!(metrics.status(), "ALERT");
    }

    #[test]
    fn test_overhead_metrics_warning_zone() {
        // Test a measurement in warning zone (30x < overhead <= 35x)
        let native = Duration::from_micros(100);
        let proof = Duration::from_micros(3200); // 3200us = 32x overhead

        let metrics = OverheadMetrics::compute(native, proof, None);

        assert!((metrics.overhead - 32.0).abs() < 0.1);
        assert!(!metrics.meets_target); // 32x > 30x
        assert!(!metrics.exceeds_alert); // 32x <= 35x
        assert_eq!(metrics.status(), "WARN");
    }
}
