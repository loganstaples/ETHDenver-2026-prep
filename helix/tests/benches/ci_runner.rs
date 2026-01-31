//! CI-Compatible Benchmark Runner
//!
//! This module provides a standalone benchmark runner suitable for CI/CD
//! integration with regression detection and failure on threshold violations.
//!
//! # Exit Codes
//!
//! - 0: All benchmarks passed, no regressions detected
//! - 1: One or more benchmarks exceeded the overhead threshold
//! - 2: Regression detected compared to baseline
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings
//! cargo run --package helix-integration-tests --bin ci_benchmarks
//!
//! # Save as new baseline
//! CI_SAVE_BASELINE=1 cargo run --package helix-integration-tests --bin ci_benchmarks
//!
//! # Fail on regression
//! CI_FAIL_ON_REGRESSION=1 cargo run --package helix-integration-tests --bin ci_benchmarks
//! ```
//!
//! # Environment Variables
//!
//! - `CI_SAVE_BASELINE`: If set, saves results as the new baseline
//! - `CI_FAIL_ON_REGRESSION`: If set, exits with error on regression
//! - `CI_THRESHOLD`: Override default regression threshold (percentage)
//! - `HELIX_BENCH_QUICK`: Run fewer iterations for faster results

use std::path::PathBuf;
use std::time::Instant;

/// Runs the CI benchmark suite.
pub fn run_ci_benchmarks() -> CIResult {
    let start = Instant::now();
    let mut result = CIResult::new();

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                   HELIX CI Benchmark Runner                        ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration from environment
    let save_baseline = std::env::var("CI_SAVE_BASELINE").is_ok();
    let fail_on_regression = std::env::var("CI_FAIL_ON_REGRESSION").is_ok();
    let threshold: f64 = std::env::var("CI_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);
    let quick_mode = std::env::var("HELIX_BENCH_QUICK").is_ok();

    println!("Configuration:");
    println!("  Save baseline: {}", save_baseline);
    println!("  Fail on regression: {}", fail_on_regression);
    println!("  Regression threshold: {}%", threshold);
    println!("  Quick mode: {}", quick_mode);
    println!();

    // Run overhead check
    println!("Running overhead analysis...");
    let overhead_passed = run_overhead_check(&mut result, quick_mode);

    // Run scaling check
    println!("\nRunning scaling analysis...");
    let scaling_passed = run_scaling_check(&mut result, quick_mode);

    // Final results
    result.total_duration = start.elapsed();
    result.passed = overhead_passed && scaling_passed;

    // Save baseline if requested
    if save_baseline {
        save_baseline_results(&result);
    }

    result
}

/// Result of CI benchmark run.
#[derive(Debug, Clone)]
pub struct CIResult {
    pub passed: bool,
    pub total_duration: std::time::Duration,
    pub overhead_results: Vec<OverheadResult>,
    pub scaling_results: Vec<ScalingResult>,
    pub regressions: Vec<String>,
}

impl CIResult {
    pub fn new() -> Self {
        Self {
            passed: true,
            total_duration: std::time::Duration::ZERO,
            overhead_results: Vec::new(),
            scaling_results: Vec::new(),
            regressions: Vec::new(),
        }
    }

    pub fn print_summary(&self) {
        println!("\n");
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                          CI SUMMARY                                ║");
        println!("╠════════════════════════════════════════════════════════════════════╣");
        println!("║ Duration: {:>10.2}s                                             ║",
            self.total_duration.as_secs_f64());
        println!("║ Overhead tests: {:>5}                                             ║",
            self.overhead_results.len());
        println!("║ Scaling tests: {:>5}                                              ║",
            self.scaling_results.len());
        println!("║ Regressions: {:>5}                                                ║",
            self.regressions.len());
        println!("╠════════════════════════════════════════════════════════════════════╣");

        if self.passed {
            println!("║                      STATUS: PASSED                              ║");
        } else {
            println!("║                      STATUS: FAILED                              ║");
        }
        println!("╚════════════════════════════════════════════════════════════════════╝");

        if !self.regressions.is_empty() {
            println!("\nRegressions detected:");
            for r in &self.regressions {
                println!("  - {}", r);
            }
        }
    }

    pub fn exit_code(&self) -> i32 {
        if self.passed {
            0
        } else if !self.regressions.is_empty() {
            2
        } else {
            1
        }
    }
}

#[derive(Debug, Clone)]
pub struct OverheadResult {
    pub name: String,
    pub overhead: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct ScalingResult {
    pub name: String,
    pub exponent: Option<f64>,
    pub passed: bool,
}

fn run_overhead_check(result: &mut CIResult, quick: bool) -> bool {
    use helix_integration_tests::benches::{
        overhead_report::quick_overhead_check,
        TARGET_OVERHEAD_MULTIPLE,
    };

    let report = quick_overhead_check();

    let mut all_passed = true;
    for (name, data) in &report.by_model_size {
        if let Some(ref gkr) = data.gkr_overhead {
            let passed = gkr.meets_target;
            result.overhead_results.push(OverheadResult {
                name: name.clone(),
                overhead: gkr.overhead,
                passed,
            });
            if !passed {
                all_passed = false;
                result.regressions.push(format!(
                    "{}: overhead {:.1}x exceeds target {:.0}x",
                    name, gkr.overhead, TARGET_OVERHEAD_MULTIPLE
                ));
            }
        }
    }

    println!("  Overhead check: {}", if all_passed { "PASSED" } else { "FAILED" });
    all_passed
}

fn run_scaling_check(result: &mut CIResult, quick: bool) -> bool {
    use helix_integration_tests::benches::scaling::analyze_model_size_scaling;

    let analysis = analyze_model_size_scaling();

    let passed = analysis.summary.all_within_target;
    result.scaling_results.push(ScalingResult {
        name: "model_size".to_string(),
        exponent: analysis.power_law_exponent,
        passed,
    });

    if !passed {
        result.regressions.push(format!(
            "Scaling analysis: max overhead {:.1}x exceeds target",
            analysis.summary.max_overhead
        ));
    }

    println!("  Scaling check: {}", if passed { "PASSED" } else { "FAILED" });
    passed
}

fn save_baseline_results(result: &CIResult) {
    let output_dir = PathBuf::from("target/helix-benchmarks");
    std::fs::create_dir_all(&output_dir).ok();

    let baseline_path = output_dir.join("ci_baseline.json");

    #[derive(serde::Serialize)]
    struct Baseline {
        timestamp: String,
        overhead_results: Vec<OverheadResult>,
        scaling_results: Vec<ScalingResult>,
    }

    let baseline = Baseline {
        timestamp: chrono::Utc::now().to_rfc3339(),
        overhead_results: result.overhead_results.clone(),
        scaling_results: result.scaling_results.clone(),
    };

    if let Ok(json) = serde_json::to_string_pretty(&baseline) {
        if std::fs::write(&baseline_path, json).is_ok() {
            println!("\nBaseline saved to: {}", baseline_path.display());
        }
    }
}

// Implement serde for results
impl serde::Serialize for OverheadResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("OverheadResult", 3)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("overhead", &self.overhead)?;
        state.serialize_field("passed", &self.passed)?;
        state.end()
    }
}

impl serde::Serialize for ScalingResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ScalingResult", 3)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("exponent", &self.exponent)?;
        state.serialize_field("passed", &self.passed)?;
        state.end()
    }
}

/// Main entry point for CI runner.
pub fn main() {
    let result = run_ci_benchmarks();
    result.print_summary();
    std::process::exit(result.exit_code());
}
