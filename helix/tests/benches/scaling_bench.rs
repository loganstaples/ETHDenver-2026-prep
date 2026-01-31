//! Scaling Analysis Benchmark
//!
//! Analyzes how overhead scales with model size, providing insights into:
//! - Asymptotic complexity of the proving system
//! - Maximum feasible model sizes
//! - Scaling characteristics (linear, quadratic, etc.)
//!
//! # Usage
//!
//! ```bash
//! cargo bench --package helix-integration-tests --bench scaling_analysis
//! ```

use helix_integration_tests::benches::{
    BenchmarkHarness, BenchmarkConfig, ModelSize, CircuitSize,
    scaling::{
        ScalingAnalysis, analyze_model_size_scaling, analyze_circuit_size_scaling,
        analyze_proof_size_scaling, generate_scaling_report,
    },
    TARGET_OVERHEAD_MULTIPLE,
};
use std::path::PathBuf;

fn main() {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                  HELIX Scaling Analysis Benchmark                  ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Analyzing how overhead scales with model/circuit size              ║");
    println!("║ Target: ≤{:.0}x overhead at all tested sizes                         ║", TARGET_OVERHEAD_MULTIPLE);
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!("\n");

    // Model sizes to analyze
    println!("Model sizes being analyzed:");
    for &size in ModelSize::all() {
        println!("  - {} ({} params)", size.name(), size.param_count());
    }
    println!();

    // Run model size scaling analysis
    println!("Running model size scaling analysis...");
    let model_analysis = analyze_model_size_scaling();
    model_analysis.print_terminal();

    // Run circuit size scaling analysis
    println!("\nRunning circuit size scaling analysis...");
    let circuit_analysis = analyze_circuit_size_scaling();
    circuit_analysis.print_terminal();

    // Run proof size scaling analysis
    println!("\nRunning proof size scaling analysis...");
    let proof_analysis = analyze_proof_size_scaling();
    proof_analysis.print_terminal();

    // Generate combined report
    let output_dir = PathBuf::from("target/helix-benchmarks");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let report = generate_scaling_report();
    let report_path = output_dir.join("scaling_report.md");
    if let Err(e) = std::fs::write(&report_path, &report) {
        eprintln!("Warning: Failed to save scaling report: {}", e);
    } else {
        println!("\nScaling report saved to: {}", report_path.display());
    }

    // Summary
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                       SCALING ANALYSIS SUMMARY                     ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Model Size Scaling:                                                ║");
    println!("║   Behavior: {:<52} ║", model_analysis.summary.scaling_description);
    println!("║   Overhead range: {:.1}x - {:.1}x {:>30} ║",
        model_analysis.summary.min_overhead,
        model_analysis.summary.max_overhead,
        if model_analysis.summary.all_within_target { "(OK)" } else { "(EXCEEDS TARGET)" });
    if let Some(exp) = model_analysis.power_law_exponent {
        println!("║   Power law exponent: {:.3} {:>36} ║", exp, "");
    }
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Circuit Size Scaling:                                              ║");
    println!("║   Behavior: {:<52} ║", circuit_analysis.summary.scaling_description);
    println!("║   Overhead range: {:.1}x - {:.1}x {:>30} ║",
        circuit_analysis.summary.min_overhead,
        circuit_analysis.summary.max_overhead,
        if circuit_analysis.summary.all_within_target { "(OK)" } else { "(EXCEEDS TARGET)" });
    if let Some(exp) = circuit_analysis.power_law_exponent {
        println!("║   Power law exponent: {:.3} {:>36} ║", exp, "");
    }
    println!("╚════════════════════════════════════════════════════════════════════╝");

    // Exit status for CI
    let all_passed = model_analysis.summary.all_within_target
        && circuit_analysis.summary.all_within_target;

    if !all_passed && std::env::var("CI").is_ok() {
        std::process::exit(1);
    }
}
