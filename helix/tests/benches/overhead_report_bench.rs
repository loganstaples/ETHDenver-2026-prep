//! Overhead Report Benchmark
//!
//! Generates a comprehensive overhead analysis report comparing native
//! computation to ZK proving, measuring the exact overhead multiple.
//!
//! # Usage
//!
//! Generate full report:
//! ```bash
//! cargo bench --package helix-integration-tests --bench overhead_report
//! ```
//!
//! The report is saved to `target/helix-benchmarks/overhead_report.{json,md}`

use helix_integration_tests::benches::{
    BenchmarkHarness, BenchmarkConfig,
    overhead_report::{OverheadReport, quick_overhead_check},
    TARGET_OVERHEAD_MULTIPLE, TARGET_PROOF_TIME_MS,
};
use std::path::PathBuf;

fn main() {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║               HELIX Overhead Analysis Benchmark                    ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Target overhead: ≤{:.0}x                                            ║", TARGET_OVERHEAD_MULTIPLE);
    println!("║ Target proof time: ≤{}ms per step                                 ║", TARGET_PROOF_TIME_MS);
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!("\n");

    // Check for quick mode via environment variable
    let quick_mode = std::env::var("HELIX_BENCH_QUICK").is_ok();

    let report = if quick_mode {
        println!("Running in QUICK mode (fewer iterations)...\n");
        quick_overhead_check()
    } else {
        println!("Running FULL overhead analysis...\n");
        println!("(Set HELIX_BENCH_QUICK=1 for faster, less precise results)\n");
        OverheadReport::generate()
    };

    // Print terminal report
    report.print_terminal();

    // Save reports
    let output_dir = PathBuf::from("target/helix-benchmarks");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // JSON report
    let json_path = output_dir.join("overhead_report.json");
    if let Err(e) = report.save_json(&json_path) {
        eprintln!("Warning: Failed to save JSON report: {}", e);
    } else {
        println!("\nJSON report saved to: {}", json_path.display());
    }

    // Markdown report
    let md_path = output_dir.join("overhead_report.md");
    let md_content = report.to_markdown();
    if let Err(e) = std::fs::write(&md_path, &md_content) {
        eprintln!("Warning: Failed to save Markdown report: {}", e);
    } else {
        println!("Markdown report saved to: {}", md_path.display());
    }

    // Final status
    println!("\n");
    if report.summary.passed {
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                         BENCHMARK PASSED                           ║");
        println!("║   All measurements within {:.0}x overhead target                      ║", TARGET_OVERHEAD_MULTIPLE);
        println!("╚════════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                         BENCHMARK FAILED                           ║");
        println!("║   {}/{} measurements exceeded {:.0}x overhead target                  ║",
            report.summary.total_measurements - report.summary.measurements_meeting_target,
            report.summary.total_measurements,
            TARGET_OVERHEAD_MULTIPLE);
        println!("║   Maximum overhead observed: {:.1}x                                   ║", report.summary.max_overhead);
        println!("╚════════════════════════════════════════════════════════════════════╝");
    }

    // For CI integration - exit with error if failed
    if std::env::var("CI").is_ok() && !report.summary.passed {
        std::process::exit(1);
    }
}
