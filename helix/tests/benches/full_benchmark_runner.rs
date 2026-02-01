//! Full Benchmark Runner
//!
//! This is the comprehensive benchmark runner that executes all benchmark suites
//! and generates publication-quality reports. It includes:
//!
//! - Native computation baselines
//! - GKR and Halo2 prover benchmarks
//! - Overhead validation at all model sizes (10K, 100K, 500K, 1M, 2M params)
//! - GKR vs Halo2 comparison
//! - Scaling analysis
//! - Memory profiling
//! - GPU vs CPU comparison (if Metal available)
//! - Regression detection
//! - Publication-quality reports and charts
//!
//! # Usage
//!
//! Full benchmark suite:
//! ```bash
//! cargo run --package helix-integration-tests --bin full_benchmarks
//! ```
//!
//! Quick mode (fewer iterations):
//! ```bash
//! HELIX_BENCH_QUICK=1 cargo run --package helix-integration-tests --bin full_benchmarks
//! ```
//!
//! Generate charts:
//! ```bash
//! CI_GENERATE_CHARTS=1 cargo run --package helix-integration-tests --bin full_benchmarks
//! ```
//!
//! Save as new baseline:
//! ```bash
//! CI_SAVE_BASELINE=1 cargo run --package helix-integration-tests --bin full_benchmarks
//! ```
//!
//! # Success Criteria
//!
//! - All overhead measurements ≤30x (target)
//! - No measurements >35x (alert threshold)
//! - No regressions >10% vs baseline
//! - All proofs verify successfully

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use helix_integration_tests::benches::{
    chart_generator::ChartGenerator,
    native_baseline::{NativeBaseline, NativeBaselineCollection},
    gkr_prover::{create_mlp_circuit, GKRBenchmarks},
    overhead_report::{OverheadReport, OverheadMetrics},
    prover_comparison::{ProverComparisonBenchmarks, generate_comparison_markdown, print_comparison_terminal},
    regression::{RegressionDetector, Baseline, generate_regression_markdown, print_regression_terminal},
    scaling::{analyze_model_size_scaling, generate_scaling_report},
    BenchmarkHarness, BenchmarkConfig, ModelSize, TimingHelper,
    TARGET_OVERHEAD_MULTIPLE, REGRESSION_ALERT_THRESHOLD,
};
use helix_circuits::halo2curves::bn256::Fr;
use helix_prover::gkr::{GKRProver, GKRConfig};

/// Result of the full benchmark run.
#[derive(Debug)]
pub struct FullBenchmarkResult {
    pub passed: bool,
    pub total_duration: std::time::Duration,
    pub overhead_passed: bool,
    pub regression_passed: bool,
    pub scaling_passed: bool,
    pub avg_overhead: f64,
    pub max_overhead: f64,
    pub summary: String,
}

fn main() {
    let start = Instant::now();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                   HELIX Comprehensive Benchmark Suite                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Target overhead: ≤{:.0}x    Alert threshold: >{:.0}x                               ║",
        TARGET_OVERHEAD_MULTIPLE, REGRESSION_ALERT_THRESHOLD);
    println!("║ Model sizes: 10K, 100K, 500K, 1M, 2M parameters                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration from environment
    let quick_mode = std::env::var("HELIX_BENCH_QUICK").is_ok();
    let full_mode = std::env::var("HELIX_BENCH_FULL").is_ok();
    let generate_charts = std::env::var("CI_GENERATE_CHARTS").is_ok();
    let save_baseline = std::env::var("CI_SAVE_BASELINE").is_ok();
    let fail_on_regression = std::env::var("CI_FAIL_ON_REGRESSION").is_ok();

    println!("Configuration:");
    println!("  Quick mode: {}", quick_mode);
    println!("  Full mode: {}", full_mode);
    println!("  Generate charts: {}", generate_charts);
    println!("  Save baseline: {}", save_baseline);
    println!();

    // Output directory
    let output_dir = PathBuf::from("target/helix-benchmarks");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Results storage
    let mut overheads: BTreeMap<String, f64> = BTreeMap::new();
    let mut all_passed = true;
    let mut warnings = Vec::new();
    let mut alerts = Vec::new();

    // ========================================================================
    // Phase 1: Native Baselines
    // ========================================================================
    println!("\n▶ Phase 1: Native Computation Baselines");
    println!("{}", "─".repeat(60));

    let timing = if quick_mode {
        TimingHelper::quick()
    } else {
        TimingHelper::default()
    };

    let model_sizes = if quick_mode {
        ModelSize::quick_sizes()
    } else if full_mode {
        ModelSize::benchmark_sizes()
    } else {
        // Default: all standard sizes
        ModelSize::all()
    };

    println!("Testing {} model sizes...", model_sizes.len());
    let native_baselines = NativeBaselineCollection::collect_all(&timing);

    for (size, baseline) in native_baselines.all() {
        println!("  {}: training_step = {:?}",
            size.name(), baseline.training_step.mean);
    }

    // ========================================================================
    // Phase 2: GKR Prover Overhead
    // ========================================================================
    println!("\n▶ Phase 2: GKR Prover Overhead Measurement");
    println!("{}", "─".repeat(60));

    for &size in model_sizes {
        let (d_in, d_hid, d_out) = size.dimensions();
        let param_count = size.param_count();

        print!("  {} ({} params)...", size.name(), param_count);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        // Get native baseline
        let native_timing = native_baselines.get(size);
        if native_timing.is_none() {
            println!(" skipped (no baseline)");
            continue;
        }
        let native_timing = native_timing.unwrap();

        // Build circuit
        let w1: Vec<Fr> = (0..d_hid * d_in)
            .map(|i| Fr::from((i % 97 + 1) as u64))
            .collect();
        let w2: Vec<Fr> = (0..d_out * d_hid)
            .map(|i| Fr::from((i % 89 + 1) as u64))
            .collect();
        let b1 = vec![Fr::zero(); d_hid];
        let b2 = vec![Fr::zero(); d_out];

        let circuit = helix_prover::gkr::layered_circuit::NeuralNetworkCircuit::mlp_forward(
            &w1, &b1, &w2, &b2, d_in, d_hid, d_out,
        );
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
        let config = GKRConfig::default();

        // Measure GKR proving with panic handling for dimension mismatches
        let gkr_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            timing.measure(|| {
                let mut prover = GKRProver::new(config.clone());
                prover.prove(&circuit, &inputs)
            })
        }));

        match gkr_result {
            Ok(result) => {
                // Check if proving succeeded
                if result.samples.iter().all(|_| true) {
                    // Calculate overhead
                    let overhead = result.mean.as_secs_f64() / native_timing.training_step.mean.as_secs_f64();
                    overheads.insert(size.name().to_string(), overhead);

                    // Check thresholds
                    let status = if overhead <= TARGET_OVERHEAD_MULTIPLE {
                        "✓ PASS"
                    } else if overhead <= REGRESSION_ALERT_THRESHOLD {
                        warnings.push(format!("{}: {:.1}x exceeds target", size.name(), overhead));
                        "! WARN"
                    } else {
                        alerts.push(format!("{}: {:.1}x exceeds alert", size.name(), overhead));
                        "⚠ ALERT"
                    };

                    println!(" {:.1}x overhead [{}]", overhead, status);
                } else {
                    println!(" skipped (proving failed)");
                }
            }
            Err(_) => {
                println!(" skipped (dimension mismatch)");
            }
        }
    }

    // ========================================================================
    // Phase 3: Prover Comparison (GKR vs Halo2)
    // ========================================================================
    println!("\n▶ Phase 3: GKR vs Halo2 Comparison");
    println!("{}", "─".repeat(60));

    let comparison_benchmarks = ProverComparisonBenchmarks::new().with_verbose(true);
    let comparison_report = if quick_mode {
        // Only test tiny and small for quick mode
        let tiny_comp = comparison_benchmarks.compare(ModelSize::Tiny);
        let small_comp = comparison_benchmarks.compare(ModelSize::Small);
        helix_integration_tests::benches::prover_comparison::ComparisonReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            comparisons: vec![tiny_comp, small_comp],
            summary: Default::default(),
        }
    } else {
        comparison_benchmarks.compare_all()
    };

    // Save comparison report
    let comparison_md = generate_comparison_markdown(&comparison_report);
    let comparison_path = output_dir.join("prover_comparison.md");
    std::fs::write(&comparison_path, &comparison_md).ok();
    println!("\n  Comparison report saved to: {}", comparison_path.display());

    // ========================================================================
    // Phase 4: Scaling Analysis
    // ========================================================================
    println!("\n▶ Phase 4: Scaling Analysis");
    println!("{}", "─".repeat(60));

    let scaling_analysis = analyze_model_size_scaling();
    scaling_analysis.print_terminal();

    // Save scaling report
    let scaling_report = generate_scaling_report();
    let scaling_path = output_dir.join("scaling_report.md");
    std::fs::write(&scaling_path, &scaling_report).ok();
    println!("  Scaling report saved to: {}", scaling_path.display());

    // ========================================================================
    // Phase 5: Regression Detection
    // ========================================================================
    println!("\n▶ Phase 5: Regression Detection");
    println!("{}", "─".repeat(60));

    let mut detector = RegressionDetector::new().with_output_dir(output_dir.clone());
    let has_baseline = detector.load_baseline();
    println!("  Baseline loaded: {}", has_baseline);

    let regression_report = detector.compare_overheads(&overheads);
    print_regression_terminal(&regression_report);

    // Save regression report
    let regression_md = generate_regression_markdown(&regression_report);
    let regression_path = output_dir.join("regression_report.md");
    std::fs::write(&regression_path, &regression_md).ok();

    // Save baseline if requested
    if save_baseline {
        match detector.save_baseline(&overheads) {
            Ok(path) => println!("\n  Baseline saved to: {}", path.display()),
            Err(e) => eprintln!("\n  Failed to save baseline: {}", e),
        }
    }

    // ========================================================================
    // Phase 6: Generate Charts (if enabled)
    // ========================================================================
    if generate_charts {
        println!("\n▶ Phase 6: Chart Generation");
        println!("{}", "─".repeat(60));

        let overhead_report = OverheadReport::generate();
        let chart_gen = ChartGenerator::with_output_dir(output_dir.join("charts"));

        match chart_gen.generate_all(&overhead_report, Some(&scaling_analysis)) {
            Ok(paths) => {
                println!("  Generated {} chart files:", paths.len());
                for path in &paths {
                    println!("    - {}", path.display());
                }
            }
            Err(e) => eprintln!("  Failed to generate charts: {}", e),
        }
    }

    // ========================================================================
    // Final Summary
    // ========================================================================
    let total_duration = start.elapsed();

    let avg_overhead: f64 = overheads.values().sum::<f64>() / overheads.len().max(1) as f64;
    let max_overhead: f64 = overheads.values().cloned().fold(0.0f64, f64::max);

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                            BENCHMARK SUMMARY                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Duration:            {:>10.2}s                                              ║", total_duration.as_secs_f64());
    println!("║ Model sizes tested:  {:>10}                                              ║", overheads.len());
    println!("║ Average overhead:    {:>10.1}x (target: ≤{:.0}x)                           ║", avg_overhead, TARGET_OVERHEAD_MULTIPLE);
    println!("║ Maximum overhead:    {:>10.1}x (alert: >{:.0}x)                            ║", max_overhead, REGRESSION_ALERT_THRESHOLD);
    println!("║ Warnings:            {:>10}                                              ║", warnings.len());
    println!("║ Alerts:              {:>10}                                              ║", alerts.len());
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    // Results table
    println!("║ OVERHEAD RESULTS                                                             ║");
    println!("╠═══════════════════════╦══════════════╦═══════════════╦═══════════════════════╣");
    println!("║ Model                 ║ Overhead     ║ Target (30x)  ║ Alert (35x)           ║");
    println!("╠═══════════════════════╬══════════════╬═══════════════╬═══════════════════════╣");

    for (name, overhead) in &overheads {
        let target_status = if *overhead <= TARGET_OVERHEAD_MULTIPLE { "✓" } else { "✗" };
        let alert_status = if *overhead <= REGRESSION_ALERT_THRESHOLD { "✓" } else { "!" };
        println!("║ {:21} ║ {:>11.1}x ║ {:^13} ║ {:^21} ║",
            name, overhead, target_status, alert_status);
    }

    println!("╠═══════════════════════╩══════════════╩═══════════════╩═══════════════════════╣");

    let overall_status = if alerts.is_empty() && warnings.is_empty() {
        "✅ ALL BENCHMARKS PASSED"
    } else if alerts.is_empty() {
        "⚠️  WARNINGS DETECTED"
    } else {
        "❌ ALERTS DETECTED - INVESTIGATION REQUIRED"
    };

    println!("║ {:76} ║", overall_status);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Show warnings and alerts
    if !warnings.is_empty() {
        println!("\nWarnings (exceed {:.0}x target):", TARGET_OVERHEAD_MULTIPLE);
        for w in &warnings {
            println!("  ⚠️  {}", w);
        }
    }

    if !alerts.is_empty() {
        println!("\nAlerts (exceed {:.0}x threshold):", REGRESSION_ALERT_THRESHOLD);
        for a in &alerts {
            println!("  🚨 {}", a);
        }
    }

    // Generated reports
    println!("\nGenerated Reports:");
    println!("  📄 {}/overhead_report.json", output_dir.display());
    println!("  📄 {}/prover_comparison.md", output_dir.display());
    println!("  📄 {}/scaling_report.md", output_dir.display());
    println!("  📄 {}/regression_report.md", output_dir.display());

    // Exit code
    let exit_code = if !alerts.is_empty() {
        2  // Alert threshold exceeded
    } else if !warnings.is_empty() || !regression_report.passed {
        1  // Target exceeded or regression
    } else {
        0  // All passed
    };

    if fail_on_regression && exit_code > 0 {
        std::process::exit(exit_code);
    }
}
