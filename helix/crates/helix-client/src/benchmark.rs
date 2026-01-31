//! HELIX Benchmark Module
//!
//! Performance benchmarking for proof generation, verification, and aggregation.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

/// Benchmark types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkType {
    /// Proof generation benchmark
    ProofGen,
    /// Proof verification benchmark
    ProofVerify,
    /// Gradient aggregation benchmark
    Aggregation,
    /// Network throughput benchmark
    Network,
    /// End-to-end training round
    E2e,
}

impl BenchmarkType {
    /// Get the name of the benchmark
    pub fn name(&self) -> &'static str {
        match self {
            BenchmarkType::ProofGen => "Proof Generation",
            BenchmarkType::ProofVerify => "Proof Verification",
            BenchmarkType::Aggregation => "Gradient Aggregation",
            BenchmarkType::Network => "Network Throughput",
            BenchmarkType::E2e => "End-to-End Round",
        }
    }

    /// Get expected duration for simulation
    pub fn expected_duration(&self) -> Duration {
        match self {
            BenchmarkType::ProofGen => Duration::from_millis(10),
            BenchmarkType::ProofVerify => Duration::from_millis(5),
            BenchmarkType::Aggregation => Duration::from_millis(3),
            BenchmarkType::Network => Duration::from_millis(2),
            BenchmarkType::E2e => Duration::from_millis(50),
        }
    }
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub benchmark_type: BenchmarkType,
    pub iterations: u32,
    pub warmup: u32,
    pub output: Option<PathBuf>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            benchmark_type: BenchmarkType::ProofGen,
            iterations: 100,
            warmup: 10,
            output: None,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub benchmark: String,
    pub iterations: u32,
    pub warmup: u32,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_dev_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub throughput_ops_s: f64,
    pub samples: Vec<f64>,
}

impl BenchmarkResults {
    /// Calculate statistics from samples
    pub fn from_samples(benchmark_type: BenchmarkType, iterations: u32, warmup: u32, samples: Vec<f64>) -> Self {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;

        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted.first().cloned().unwrap_or(0.0);
        let max = sorted.last().cloned().unwrap_or(0.0);
        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        Self {
            benchmark: benchmark_type.name().to_string(),
            iterations,
            warmup,
            mean_ms: mean,
            min_ms: min,
            max_ms: max,
            std_dev_ms: std_dev,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            throughput_ops_s: 1000.0 / mean,
            samples,
        }
    }

    /// Save results to file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Calculate percentile
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64) * p / 100.0) as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

/// Benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run the benchmark
    pub async fn run(&self) -> Result<BenchmarkResults> {
        println!("{}", "Running HELIX Benchmarks".cyan().bold());
        println!("  Type:       {}", self.config.benchmark_type.name());
        println!("  Iterations: {}", self.config.iterations);
        println!("  Warmup:     {}", self.config.warmup);
        println!();

        // Warmup
        self.run_warmup().await?;

        // Run benchmark
        let samples = self.run_iterations().await?;

        // Calculate results
        let results = BenchmarkResults::from_samples(
            self.config.benchmark_type,
            self.config.iterations,
            self.config.warmup,
            samples,
        );

        // Display results
        self.display_results(&results);

        // Save if output specified
        if let Some(path) = &self.config.output {
            results.save(path)?;
            println!("\nResults saved to: {}", path.display());
        }

        Ok(results)
    }

    /// Run warmup iterations
    async fn run_warmup(&self) -> Result<()> {
        let pb = ProgressBar::new(self.config.warmup as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.blue} Warmup [{bar:40.blue/cyan}] {pos}/{len}")
                .unwrap()
                .progress_chars("#>-"),
        );

        for _ in 0..self.config.warmup {
            self.run_single_iteration().await;
            pb.inc(1);
        }

        pb.finish_with_message("Warmup complete");
        Ok(())
    }

    /// Run benchmark iterations
    async fn run_iterations(&self) -> Result<Vec<f64>> {
        let pb = ProgressBar::new(self.config.iterations as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut samples = Vec::with_capacity(self.config.iterations as usize);

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            self.run_single_iteration().await;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            samples.push(elapsed);
            pb.inc(1);
        }

        pb.finish_and_clear();
        Ok(samples)
    }

    /// Run a single iteration
    async fn run_single_iteration(&self) {
        // Simulate work based on benchmark type
        tokio::time::sleep(self.config.benchmark_type.expected_duration()).await;
    }

    /// Display benchmark results
    fn display_results(&self, results: &BenchmarkResults) {
        println!("\n{}", "Results:".yellow().bold());
        println!("  ┌────────────┬────────────────┐");
        println!("  │ Metric     │ Value          │");
        println!("  ├────────────┼────────────────┤");
        println!("  │ Mean       │ {:>10.3} ms  │", results.mean_ms);
        println!("  │ Std Dev    │ {:>10.3} ms  │", results.std_dev_ms);
        println!("  │ Min        │ {:>10.3} ms  │", results.min_ms);
        println!("  │ Max        │ {:>10.3} ms  │", results.max_ms);
        println!("  │ P50        │ {:>10.3} ms  │", results.p50_ms);
        println!("  │ P95        │ {:>10.3} ms  │", results.p95_ms);
        println!("  │ P99        │ {:>10.3} ms  │", results.p99_ms);
        println!("  │ Throughput │ {:>10.1} op/s│", results.throughput_ops_s);
        println!("  └────────────┴────────────────┘");
    }
}

/// Quick benchmark for a specific operation
pub async fn quick_benchmark<F, Fut>(name: &str, iterations: u32, f: F) -> BenchmarkResults
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut samples = Vec::with_capacity(iterations as usize);

    for _ in 0..iterations {
        let start = Instant::now();
        f().await;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        samples.push(elapsed);
    }

    BenchmarkResults {
        benchmark: name.to_string(),
        iterations,
        warmup: 0,
        mean_ms: samples.iter().sum::<f64>() / samples.len() as f64,
        min_ms: samples.iter().cloned().fold(f64::INFINITY, f64::min),
        max_ms: samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        std_dev_ms: 0.0, // Simplified
        p50_ms: percentile(&samples, 50.0),
        p95_ms: percentile(&samples, 95.0),
        p99_ms: percentile(&samples, 99.0),
        throughput_ops_s: 1000.0 / (samples.iter().sum::<f64>() / samples.len() as f64),
        samples,
    }
}

/// Comparison benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub name: String,
    pub baseline: BenchmarkResults,
    pub current: BenchmarkResults,
    pub mean_diff_percent: f64,
    pub p99_diff_percent: f64,
    pub throughput_diff_percent: f64,
}

impl BenchmarkComparison {
    /// Create a comparison from two results
    pub fn compare(name: &str, baseline: BenchmarkResults, current: BenchmarkResults) -> Self {
        let mean_diff = ((current.mean_ms - baseline.mean_ms) / baseline.mean_ms) * 100.0;
        let p99_diff = ((current.p99_ms - baseline.p99_ms) / baseline.p99_ms) * 100.0;
        let throughput_diff = ((current.throughput_ops_s - baseline.throughput_ops_s) / baseline.throughput_ops_s) * 100.0;

        Self {
            name: name.to_string(),
            baseline,
            current,
            mean_diff_percent: mean_diff,
            p99_diff_percent: p99_diff,
            throughput_diff_percent: throughput_diff,
        }
    }

    /// Display comparison
    pub fn display(&self) {
        println!("{} Benchmark Comparison: {}", "📊".cyan(), self.name.bold());
        println!();

        let format_diff = |diff: f64| {
            if diff > 0.0 {
                format!("+{:.1}%", diff).red()
            } else {
                format!("{:.1}%", diff).green()
            }
        };

        println!("  Mean:       {:.3} ms → {:.3} ms ({})",
            self.baseline.mean_ms,
            self.current.mean_ms,
            format_diff(self.mean_diff_percent)
        );
        println!("  P99:        {:.3} ms → {:.3} ms ({})",
            self.baseline.p99_ms,
            self.current.p99_ms,
            format_diff(self.p99_diff_percent)
        );
        println!("  Throughput: {:.1} op/s → {:.1} op/s ({})",
            self.baseline.throughput_ops_s,
            self.current.throughput_ops_s,
            format_diff(-self.throughput_diff_percent) // Invert for throughput (higher is better)
        );
    }
}
