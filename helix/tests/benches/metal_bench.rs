//! Metal vs CPU Comparison Benchmark
//!
//! Compares Metal GPU acceleration (on macOS) with CPU-only implementations
//! for field operations critical to the proving system.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --package helix-integration-tests --bench metal_comparison
//! ```
//!
//! Note: Metal acceleration is only available on macOS with a Metal-capable GPU.

use helix_integration_tests::benches::{
    BenchmarkHarness, BenchmarkConfig, OutputFormat,
    metal_vs_cpu::{
        MetalVsCpuBenchmarks, CpuFieldOps, generate_test_data, generate_metal_summary,
    },
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn format_throughput(elements: usize, duration: Duration) -> String {
    let elements_per_sec = elements as f64 / duration.as_secs_f64();
    if elements_per_sec >= 1_000_000_000.0 {
        format!("{:.2} G/s", elements_per_sec / 1_000_000_000.0)
    } else if elements_per_sec >= 1_000_000.0 {
        format!("{:.2} M/s", elements_per_sec / 1_000_000.0)
    } else if elements_per_sec >= 1_000.0 {
        format!("{:.2} K/s", elements_per_sec / 1_000.0)
    } else {
        format!("{:.2}/s", elements_per_sec)
    }
}

fn main() {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                Metal vs CPU Comparison Benchmark                   ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");

    let metal_available = MetalVsCpuBenchmarks::is_metal_available();
    if metal_available {
        println!("║ Metal GPU acceleration: AVAILABLE                                ║");
    } else {
        println!("║ Metal GPU acceleration: NOT AVAILABLE                            ║");
        println!("║ (Running CPU-only benchmarks)                                    ║");
    }
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!("\n");

    // Batch sizes to test
    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    // Warm-up
    println!("Warming up...");
    let (warm_a, warm_b) = generate_test_data(1000);
    for _ in 0..5 {
        let _ = CpuFieldOps::batch_add(&warm_a, &warm_b);
        let _ = CpuFieldOps::batch_mul(&warm_a, &warm_b);
    }
    println!();

    // Results storage
    let mut results = Vec::new();

    // Batch Addition
    println!("=== Batch Field Addition ===\n");
    println!("{:<12} {:>12} {:>12} {:>15}", "Size", "CPU (μs)", "Throughput", "");
    println!("{}", "-".repeat(55));

    for &size in &sizes {
        let (a, b) = generate_test_data(size);

        // CPU timing
        let start = Instant::now();
        for _ in 0..10 {
            let _ = CpuFieldOps::batch_add(&a, &b);
        }
        let cpu_time = start.elapsed() / 10;
        let cpu_us = cpu_time.as_secs_f64() * 1_000_000.0;
        let throughput = format_throughput(size, cpu_time);

        println!("{:<12} {:>12.1} {:>15}", size, cpu_us, throughput);
        results.push(("add", size, cpu_us));
    }
    println!();

    // Batch Multiplication
    println!("=== Batch Field Multiplication ===\n");
    println!("{:<12} {:>12} {:>12} {:>15}", "Size", "CPU (μs)", "Throughput", "");
    println!("{}", "-".repeat(55));

    for &size in &sizes {
        let (a, b) = generate_test_data(size);

        let start = Instant::now();
        for _ in 0..10 {
            let _ = CpuFieldOps::batch_mul(&a, &b);
        }
        let cpu_time = start.elapsed() / 10;
        let cpu_us = cpu_time.as_secs_f64() * 1_000_000.0;
        let throughput = format_throughput(size, cpu_time);

        println!("{:<12} {:>12.1} {:>15}", size, cpu_us, throughput);
        results.push(("mul", size, cpu_us));
    }
    println!();

    // Batch Inversion
    println!("=== Batch Field Inversion ===\n");
    println!("{:<12} {:>12} {:>12} {:>15}", "Size", "CPU (μs)", "Throughput", "");
    println!("{}", "-".repeat(55));

    for &size in &[1_000, 10_000, 100_000] {
        let (a, _) = generate_test_data(size);

        let start = Instant::now();
        for _ in 0..5 {
            let _ = CpuFieldOps::batch_inv(&a);
        }
        let cpu_time = start.elapsed() / 5;
        let cpu_us = cpu_time.as_secs_f64() * 1_000_000.0;
        let throughput = format_throughput(size, cpu_time);

        println!("{:<12} {:>12.1} {:>15}", size, cpu_us, throughput);
        results.push(("inv", size, cpu_us));
    }
    println!();

    // Polynomial Evaluation
    println!("=== Polynomial Evaluation (Horner's Method) ===\n");
    println!("{:<12} {:>12} {:>12} {:>15}", "Degree", "CPU (μs)", "Throughput", "");
    println!("{}", "-".repeat(55));

    use helix_circuits::halo2curves::bn256::Fr;
    for degree in [256, 1024, 4096, 16384] {
        let coeffs: Vec<Fr> = (0..degree).map(|i| Fr::from(i as u64)).collect();
        let point = Fr::from(42u64);

        let start = Instant::now();
        for _ in 0..100 {
            let _ = CpuFieldOps::poly_eval(&coeffs, &point);
        }
        let cpu_time = start.elapsed() / 100;
        let cpu_us = cpu_time.as_secs_f64() * 1_000_000.0;
        let throughput = format_throughput(degree, cpu_time);

        println!("{:<12} {:>12.1} {:>15}", degree, cpu_us, throughput);
        results.push(("poly_eval", degree, cpu_us));
    }
    println!();

    // Matrix-Vector Multiplication
    println!("=== Matrix-Vector Multiplication ===\n");
    println!("{:<12} {:>12} {:>12} {:>15}", "Dimension", "CPU (μs)", "Throughput", "");
    println!("{}", "-".repeat(55));

    for dim in [16, 32, 64, 128, 256] {
        let matrix: Vec<Fr> = (0..dim * dim).map(|i| Fr::from(i as u64)).collect();
        let vector: Vec<Fr> = (0..dim).map(|i| Fr::from(i as u64)).collect();

        let start = Instant::now();
        for _ in 0..50 {
            let _ = CpuFieldOps::matmul(&matrix, &vector, dim, dim);
        }
        let cpu_time = start.elapsed() / 50;
        let cpu_us = cpu_time.as_secs_f64() * 1_000_000.0;
        let ops = dim * dim; // multiply-add operations
        let throughput = format_throughput(ops, cpu_time);

        println!("{:<12} {:>12.1} {:>15}", format!("{}x{}", dim, dim), cpu_us, throughput);
        results.push(("matmul", dim * dim, cpu_us));
    }
    println!();

    // Save results
    let output_dir = PathBuf::from("target/helix-benchmarks");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let csv_path = output_dir.join("metal_vs_cpu.csv");
    let mut csv_content = String::from("operation,size,cpu_us\n");
    for (op, size, cpu_us) in &results {
        csv_content.push_str(&format!("{},{},{:.2}\n", op, size, cpu_us));
    }
    if let Err(e) = std::fs::write(&csv_path, &csv_content) {
        eprintln!("Warning: Failed to save CSV: {}", e);
    } else {
        println!("Results saved to: {}", csv_path.display());
    }

    // Summary
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                           SUMMARY                                  ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    if metal_available {
        println!("║ Metal acceleration is available on this system.                  ║");
        println!("║ For optimal performance, compile with: --features metal          ║");
    } else {
        println!("║ Metal acceleration is not available.                             ║");
        println!("║ Using CPU-only field operations.                                 ║");
        #[cfg(target_os = "macos")]
        println!("║ Note: Metal requires macOS 10.14+ and a compatible GPU.          ║");
        #[cfg(not(target_os = "macos"))]
        println!("║ Note: Metal is only available on macOS.                          ║");
    }
    println!("╚════════════════════════════════════════════════════════════════════╝");
}
