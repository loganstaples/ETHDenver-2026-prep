//! Memory Usage Benchmark
//!
//! Profiles memory usage for HELIX proving operations to understand:
//! - Peak memory requirements
//! - Memory scaling with model size
//! - Memory optimization opportunities
//!
//! # Usage
//!
//! ```bash
//! cargo bench --package helix-integration-tests --bench memory_benchmarks
//! ```

use helix_integration_tests::benches::{
    BenchmarkHarness, BenchmarkConfig, ModelSize, CircuitSize,
    memory_profile::{
        MemoryProfiler, MemoryProfile, MemoryEstimate,
        profile_gkr_memory, profile_mlp_memory, get_current_memory,
    },
    gkr_prover::{create_benchmark_circuit, create_mlp_circuit},
};
use helix_circuits::halo2curves::bn256::Fr;
use helix_prover::gkr::{GKRProver, GKRConfig};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                   HELIX Memory Usage Benchmark                     ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Profiling memory usage for proving operations                      ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!("\n");

    // Initial memory state
    let initial_mem = get_current_memory();
    println!("Initial memory state:");
    println!("  Allocated: {:.2} MB", initial_mem.allocated_mb());
    println!();

    // Create profiler
    let mut profiler = MemoryProfiler::new();

    // Profile circuit creation at different sizes
    println!("=== Circuit Creation Memory ===\n");

    for &size in &[CircuitSize::Small, CircuitSize::Medium, CircuitSize::Large] {
        let gates = size.gate_count();
        if gates > 100_000 {
            continue; // Skip very large for this benchmark
        }

        profiler.profile(&format!("circuit_creation_{}", size.name()), || {
            create_benchmark_circuit(gates)
        });
    }

    // Profile MLP circuit creation
    println!("=== MLP Circuit Creation Memory ===\n");

    for &size in &[ModelSize::Tiny, ModelSize::Small, ModelSize::Medium] {
        let (d_in, d_hid, d_out) = size.dimensions();

        profiler.profile(&format!("mlp_circuit_{}", size.name()), || {
            create_mlp_circuit(d_in, d_hid, d_out)
        });
    }

    // Profile GKR proving
    println!("=== GKR Proving Memory ===\n");

    for &size in &[CircuitSize::Small, CircuitSize::Medium] {
        let gates = size.gate_count();
        let circuit = create_benchmark_circuit(gates);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];
        let config = GKRConfig::for_testing();

        profiler.profile(&format!("gkr_prove_{}", size.name()), || {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });
    }

    // Profile MLP proving
    println!("=== MLP Proving Memory ===\n");

    for &size in &[ModelSize::Tiny, ModelSize::Small] {
        let (d_in, d_hid, d_out) = size.dimensions();
        let circuit = create_mlp_circuit(d_in, d_hid, d_out);
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
        let config = GKRConfig::for_testing();

        profiler.profile(&format!("mlp_prove_{}", size.name()), || {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });
    }

    // Print summary
    profiler.print_summary();

    // Memory estimates for all sizes
    println!("\n=== Memory Requirements Estimate ===\n");
    let estimates = MemoryEstimate::estimate_all();
    MemoryEstimate::print_table(&estimates);

    // Save results
    let output_dir = PathBuf::from("target/helix-benchmarks");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Save as CSV
    let csv_path = output_dir.join("memory_profile.csv");
    let mut csv_content = String::from("operation,before_bytes,peak_bytes,after_bytes,duration_us\n");
    for profile in profiler.profiles() {
        csv_content.push_str(&format!(
            "{},{},{},{},{}\n",
            profile.operation,
            profile.before_bytes,
            profile.peak_bytes,
            profile.after_bytes,
            profile.duration_us,
        ));
    }
    if let Err(e) = std::fs::write(&csv_path, &csv_content) {
        eprintln!("Warning: Failed to save CSV: {}", e);
    } else {
        println!("\nMemory profile saved to: {}", csv_path.display());
    }

    // Save estimates as JSON
    let estimates_path = output_dir.join("memory_estimates.json");
    let estimates_json = serde_json::to_string_pretty(&estimates).unwrap_or_default();
    if let Err(e) = std::fs::write(&estimates_path, &estimates_json) {
        eprintln!("Warning: Failed to save estimates: {}", e);
    } else {
        println!("Memory estimates saved to: {}", estimates_path.display());
    }

    // Final summary
    let summary = profiler.summary();
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                         MEMORY SUMMARY                             ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Operations profiled: {:>5}                                         ║", summary.num_operations);
    println!("║ Peak memory usage: {:>10}                                     ║",
        MemoryProfile::format_bytes(summary.total_peak_bytes));
    println!("║ Average peak: {:>15}                                     ║",
        MemoryProfile::format_bytes(summary.avg_peak_bytes));
    println!("╚════════════════════════════════════════════════════════════════════╝");

    // Hardware recommendations
    println!("\n=== Hardware Recommendations ===\n");

    let large_est = MemoryEstimate::estimate(ModelSize::Large);
    let xlarge_est = MemoryEstimate::estimate(ModelSize::XLarge);

    println!("For Large models (~{}K params):", large_est.param_count / 1000);
    println!("  Recommended RAM: {} minimum",
        MemoryProfile::format_bytes(large_est.estimated_total_memory * 2)); // 2x safety margin

    println!("\nFor XLarge models (~{}M params):", xlarge_est.param_count / 1_000_000);
    println!("  Recommended RAM: {} minimum",
        MemoryProfile::format_bytes(xlarge_est.estimated_total_memory * 2));
}

use serde::{Deserialize, Serialize};
