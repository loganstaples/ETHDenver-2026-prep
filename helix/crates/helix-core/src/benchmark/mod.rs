//! Benchmark Suite for HELIX.
//!
//! Comprehensive performance benchmarking framework for all major components including:
//! - Timing with statistical analysis (mean, std dev, percentiles)
//! - Memory tracking (allocations, peak usage)
//! - Circuit-specific metrics (constraints, proof size)
//! - Gas cost estimation for on-chain verification
//! - JSON export for CI/CD integration
//! - Standardized model architectures for consistent benchmarking

use std::time::{Duration, Instant};
use std::collections::HashMap;

// Standard model benchmarks
pub mod standard_models;
pub use standard_models::{
    Architecture, BenchmarkComparison, ErrorStats, ModelBenchmarkResult,
    ModelConfig, ModelSize, StandardBenchmarkSuite, compare_results,
};

// =============================================================================
// Statistical Analysis
// =============================================================================

/// Statistical summary of benchmark measurements.
#[derive(Debug, Clone)]
pub struct Statistics {
    /// Number of samples.
    pub count: usize,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

impl Statistics {
    /// Computes statistics from a slice of values.
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let count = samples.len();
        let mean = samples.iter().sum::<f64>() / count as f64;

        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Sort for percentiles
        let mut sorted: Vec<f64> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        Some(Self {
            count,
            mean,
            std_dev,
            min,
            max,
            median,
            p95,
            p99,
        })
    }

    /// Returns coefficient of variation (std_dev / mean).
    pub fn cv(&self) -> f64 {
        if self.mean == 0.0 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }
}

/// Computes the percentile value from a sorted array.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// =============================================================================
// Memory Tracking
// =============================================================================

/// Memory usage snapshot.
#[derive(Debug, Clone, Default)]
pub struct MemorySnapshot {
    /// Allocated bytes.
    pub allocated_bytes: usize,
    /// Peak allocated bytes.
    pub peak_bytes: usize,
    /// Number of allocations.
    pub allocation_count: usize,
}

/// Memory tracker for benchmarks.
/// Note: Full allocation tracking requires a custom allocator.
/// This provides basic heap estimation.
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    /// Starting memory estimate.
    start_bytes: usize,
    /// Peak memory during tracking.
    peak_bytes: usize,
}

impl MemoryTracker {
    /// Creates a new memory tracker.
    pub fn new() -> Self {
        let current = Self::estimate_current_memory();
        Self {
            start_bytes: current,
            peak_bytes: current,
        }
    }

    /// Estimates current memory usage via mach task_info on macOS.
    #[cfg(target_os = "macos")]
    fn estimate_current_memory() -> usize {
        use std::mem;

        // mach_task_basic_info flavor constant
        const MACH_TASK_BASIC_INFO: u32 = 20;

        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u64; 2],    // time_value_t (seconds, microseconds)
            system_time: [u64; 2],  // time_value_t
            policy: i32,
            suspend_count: i32,
        }

        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_count: *mut u32,
            ) -> i32;
        }

        unsafe {
            let mut info: MachTaskBasicInfo = mem::zeroed();
            let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
            let ret = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _,
                &mut count,
            );
            if ret == 0 {
                info.resident_size as usize
            } else {
                0
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn estimate_current_memory() -> usize {
        // Linux: Parse /proc/self/statm
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1))
            .and_then(|s| s.parse::<usize>().ok())
            .map(|pages| pages * 4096)
            .unwrap_or(0)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn estimate_current_memory() -> usize {
        0
    }

    /// Updates peak memory.
    pub fn update(&mut self) {
        let current = Self::estimate_current_memory();
        self.peak_bytes = self.peak_bytes.max(current);
    }

    /// Returns the memory delta since start.
    pub fn delta(&self) -> usize {
        let current = Self::estimate_current_memory();
        current.saturating_sub(self.start_bytes)
    }

    /// Returns peak memory usage.
    pub fn peak(&self) -> usize {
        self.peak_bytes.saturating_sub(self.start_bytes)
    }
}

// =============================================================================
// Gas Cost Estimation
// =============================================================================

/// Gas cost estimates for on-chain operations.
#[derive(Debug, Clone)]
pub struct GasCosts {
    /// Base gas for transaction.
    pub base_tx_gas: u64,
    /// Gas per byte of calldata.
    pub calldata_gas_per_byte: u64,
    /// Gas for BN254 pairing check.
    pub bn254_pairing_gas: u64,
    /// Gas for BN254 scalar multiplication.
    pub bn254_scalar_mul_gas: u64,
    /// Gas for Keccak256 per word.
    pub keccak256_gas_per_word: u64,
    /// Gas for SSTORE (cold).
    pub sstore_cold_gas: u64,
    /// Gas for SSTORE (warm).
    pub sstore_warm_gas: u64,
}

impl Default for GasCosts {
    fn default() -> Self {
        // Based on EIP-2929 / EIP-2930 costs
        Self {
            base_tx_gas: 21_000,
            calldata_gas_per_byte: 16, // non-zero byte
            bn254_pairing_gas: 45_000, // per pairing
            bn254_scalar_mul_gas: 6_000,
            keccak256_gas_per_word: 6,
            sstore_cold_gas: 22_100,
            sstore_warm_gas: 2_900,
        }
    }
}

impl GasCosts {
    /// Estimates gas cost for verifying a KZG proof.
    pub fn estimate_kzg_verification(&self, num_advice_columns: usize) -> u64 {
        // Base verification cost
        let base = self.base_tx_gas;

        // Pairing check (typically 2 pairings for single proof)
        let pairing = self.bn254_pairing_gas * 2;

        // Scalar multiplications (roughly num_columns + some overhead)
        let scalar_muls = self.bn254_scalar_mul_gas * (num_advice_columns as u64 + 10);

        // Calldata for proof (~1KB typical)
        let calldata = self.calldata_gas_per_byte * 1024;

        // State writes (commitment update)
        let storage = self.sstore_cold_gas;

        base + pairing + scalar_muls + calldata + storage
    }

    /// Estimates gas cost for submitting a training proof.
    pub fn estimate_training_proof_submission(&self, proof_size: usize) -> u64 {
        let base = self.base_tx_gas;
        let verification = self.estimate_kzg_verification(4);
        let calldata = self.calldata_gas_per_byte * proof_size as u64;
        let storage = self.sstore_cold_gas * 3; // Update commitment, round, error bound

        base + verification + calldata + storage
    }

    /// Estimates gas cost in USD at a given gas price and ETH price.
    pub fn gas_to_usd(&self, gas: u64, gas_price_gwei: f64, eth_price_usd: f64) -> f64 {
        let eth_cost = (gas as f64 * gas_price_gwei) / 1e9;
        eth_cost * eth_price_usd
    }
}

// =============================================================================
// Benchmark Result Types
// =============================================================================

/// Comprehensive benchmark result with statistics.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name.
    pub name: String,
    /// Number of iterations.
    pub iterations: usize,
    /// Total duration.
    pub total_duration: Duration,
    /// Average duration per iteration.
    pub avg_duration: Duration,
    /// Minimum duration.
    pub min_duration: Duration,
    /// Maximum duration.
    pub max_duration: Duration,
    /// Throughput (iterations per second).
    pub throughput: f64,
    /// Statistical analysis of durations.
    pub stats: Option<Statistics>,
    /// Memory usage (if tracked).
    pub memory: Option<MemorySnapshot>,
    /// Custom metrics.
    pub metrics: HashMap<String, f64>,
}

impl BenchmarkResult {
    /// Formats the result as a string.
    pub fn format(&self) -> String {
        let mut result = format!(
            "{}: avg={:?} (min={:?}, max={:?}), throughput={:.2}/s",
            self.name,
            self.avg_duration,
            self.min_duration,
            self.max_duration,
            self.throughput
        );

        if let Some(stats) = &self.stats {
            result.push_str(&format!(
                "\n  std_dev={:.2}µs, p95={:.2}µs, p99={:.2}µs",
                stats.std_dev / 1000.0,
                stats.p95 / 1000.0,
                stats.p99 / 1000.0
            ));
        }

        if let Some(mem) = &self.memory {
            result.push_str(&format!(
                "\n  memory: peak={}KB, allocations={}",
                mem.peak_bytes / 1024,
                mem.allocation_count
            ));
        }

        result
    }

    /// Converts to JSON string for CI/CD export.
    pub fn to_json(&self) -> String {
        let mut json = format!(
            r#"{{"name":"{}","iterations":{},"avg_ns":{},"min_ns":{},"max_ns":{},"throughput":{:.2}"#,
            self.name,
            self.iterations,
            self.avg_duration.as_nanos(),
            self.min_duration.as_nanos(),
            self.max_duration.as_nanos(),
            self.throughput
        );

        if let Some(stats) = &self.stats {
            json.push_str(&format!(
                r#","stats":{{"mean":{:.2},"std_dev":{:.2},"p95":{:.2},"p99":{:.2}}}"#,
                stats.mean, stats.std_dev, stats.p95, stats.p99
            ));
        }

        if let Some(mem) = &self.memory {
            json.push_str(&format!(
                r#","memory":{{"peak_bytes":{},"allocations":{}}}"#,
                mem.peak_bytes, mem.allocation_count
            ));
        }

        json.push('}');
        json
    }
}

// =============================================================================
// Circuit Benchmark Result
// =============================================================================

/// Circuit-specific benchmark result.
#[derive(Debug, Clone)]
pub struct CircuitBenchmarkResult {
    /// Name of the circuit.
    pub name: String,
    /// K parameter (circuit size = 2^k rows).
    pub k: u32,
    /// Number of constraints.
    pub num_constraints: usize,
    /// Number of advice columns.
    pub num_advice_columns: usize,
    /// Key generation time.
    pub keygen_time: Duration,
    /// Proof generation time.
    pub prove_time: Duration,
    /// Verification time.
    pub verify_time: Duration,
    /// Proof size in bytes.
    pub proof_size: usize,
    /// Estimated gas cost for on-chain verification.
    pub estimated_gas: u64,
    /// Overhead ratio vs direct computation.
    pub overhead_ratio: Option<f64>,
}

impl CircuitBenchmarkResult {
    /// Returns a formatted summary.
    pub fn summary(&self) -> String {
        format!(
            "Circuit: {}\n\
             K: {} (2^{} = {} rows)\n\
             Constraints: {}\n\
             Keygen: {:?}\n\
             Prove: {:?}\n\
             Verify: {:?}\n\
             Proof size: {} bytes ({:.2} KB)\n\
             Estimated gas: {} (~${:.2} at 30 gwei, $3000 ETH)\n{}",
            self.name,
            self.k,
            self.k,
            1 << self.k,
            self.num_constraints,
            self.keygen_time,
            self.prove_time,
            self.verify_time,
            self.proof_size,
            self.proof_size as f64 / 1024.0,
            self.estimated_gas,
            GasCosts::default().gas_to_usd(self.estimated_gas, 30.0, 3000.0),
            self.overhead_ratio.map(|r| format!("Overhead: {:.1}x\n", r)).unwrap_or_default()
        )
    }

    /// Converts to JSON for export.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"name":"{}","k":{},"constraints":{},"keygen_ns":{},"prove_ns":{},"verify_ns":{},"proof_size":{},"estimated_gas":{}}}"#,
            self.name,
            self.k,
            self.num_constraints,
            self.keygen_time.as_nanos(),
            self.prove_time.as_nanos(),
            self.verify_time.as_nanos(),
            self.proof_size,
            self.estimated_gas
        )
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================

/// Configuration for the benchmark runner.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Whether to track memory.
    pub track_memory: bool,
    /// Whether to compute full statistics.
    pub compute_stats: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            track_memory: false,
            compute_stats: true,
        }
    }
}

/// Benchmark runner with enhanced capabilities.
pub struct BenchmarkRunner {
    results: Vec<BenchmarkResult>,
    circuit_results: Vec<CircuitBenchmarkResult>,
    config: BenchmarkConfig,
    gas_costs: GasCosts,
}

impl BenchmarkRunner {
    /// Creates a new benchmark runner.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            circuit_results: Vec::new(),
            config: BenchmarkConfig::default(),
            gas_costs: GasCosts::default(),
        }
    }

    /// Sets warmup iterations.
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.config.warmup_iterations = iterations;
        self
    }

    /// Enables memory tracking.
    pub fn with_memory_tracking(mut self) -> Self {
        self.config.track_memory = true;
        self
    }

    /// Enables full statistics computation.
    pub fn with_stats(mut self) -> Self {
        self.config.compute_stats = true;
        self
    }

    /// Runs a benchmark.
    pub fn bench<F>(&mut self, name: &str, iterations: usize, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            f();
        }

        // Track memory if enabled
        let mut memory_tracker = if self.config.track_memory {
            Some(MemoryTracker::new())
        } else {
            None
        };

        let mut durations = Vec::with_capacity(iterations);
        let total_start = Instant::now();

        for _ in 0..iterations {
            let start = Instant::now();
            f();
            durations.push(start.elapsed());

            if let Some(ref mut tracker) = memory_tracker {
                tracker.update();
            }
        }

        let total_duration = total_start.elapsed();
        let avg_duration = total_duration / iterations as u32;
        let min_duration = *durations.iter().min().unwrap_or(&Duration::ZERO);
        let max_duration = *durations.iter().max().unwrap_or(&Duration::ZERO);
        let throughput = iterations as f64 / total_duration.as_secs_f64();

        // Compute statistics
        let stats = if self.config.compute_stats {
            let nanos: Vec<f64> = durations.iter().map(|d| d.as_nanos() as f64).collect();
            Statistics::from_samples(&nanos)
        } else {
            None
        };

        // Memory snapshot
        let memory = memory_tracker.map(|t| MemorySnapshot {
            allocated_bytes: t.delta(),
            peak_bytes: t.peak(),
            allocation_count: 0, // Would need custom allocator
        });

        let result = BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_duration,
            avg_duration,
            min_duration,
            max_duration,
            throughput,
            stats,
            memory,
            metrics: HashMap::new(),
        };

        self.results.push(result.clone());
        result
    }

    /// Runs matrix multiplication benchmark.
    pub fn bench_matmul(&mut self, m: usize, k: usize, n: usize, iterations: usize) -> BenchmarkResult {
        let a: Vec<f64> = (0..m * k).map(|i| (i % 100) as f64 / 100.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 100) as f64 / 100.0).collect();
        let mut c = vec![0.0; m * n];

        self.bench(&format!("matmul_{}x{}x{}", m, k, n), iterations, || {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            std::hint::black_box(&c);
        })
    }

    /// Runs softmax benchmark.
    pub fn bench_softmax(&mut self, size: usize, iterations: usize) -> BenchmarkResult {
        let input: Vec<f64> = (0..size).map(|i| (i % 100) as f64 / 10.0).collect();
        let mut output = vec![0.0; size];

        self.bench(&format!("softmax_{}", size), iterations, || {
            let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = input.iter().map(|x| (x - max_val).exp()).sum();
            for (i, x) in input.iter().enumerate() {
                output[i] = (x - max_val).exp() / sum;
            }
            std::hint::black_box(&output);
        })
    }

    /// Runs attention benchmark (Q*K^T scaled).
    pub fn bench_attention(&mut self, seq_len: usize, head_dim: usize, iterations: usize) -> BenchmarkResult {
        let q: Vec<f64> = (0..seq_len * head_dim).map(|i| (i % 100) as f64 / 100.0).collect();
        let k: Vec<f64> = (0..seq_len * head_dim).map(|i| (i % 100) as f64 / 100.0).collect();
        let mut output = vec![0.0; seq_len * seq_len];

        self.bench(&format!("attention_{}x{}", seq_len, head_dim), iterations, || {
            let scale = 1.0 / (head_dim as f64).sqrt();

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0;
                    for d in 0..head_dim {
                        sum += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    output[i * seq_len + j] = sum * scale;
                }
            }
            std::hint::black_box(&output);
        })
    }

    /// Runs ReLU benchmark.
    pub fn bench_relu(&mut self, size: usize, iterations: usize) -> BenchmarkResult {
        let input: Vec<f64> = (0..size).map(|i| ((i % 200) as f64 - 100.0) / 10.0).collect();
        let mut output = vec![0.0; size];

        self.bench(&format!("relu_{}", size), iterations, || {
            for (i, &x) in input.iter().enumerate() {
                output[i] = if x > 0.0 { x } else { 0.0 };
            }
            std::hint::black_box(&output);
        })
    }

    /// Runs layer normalization benchmark.
    pub fn bench_layer_norm(&mut self, size: usize, iterations: usize) -> BenchmarkResult {
        let input: Vec<f64> = (0..size).map(|i| (i % 100) as f64 / 10.0).collect();
        let mut output = vec![0.0; size];
        let eps = 1e-5;

        self.bench(&format!("layer_norm_{}", size), iterations, || {
            let mean = input.iter().sum::<f64>() / size as f64;
            let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / size as f64;
            let std = (var + eps).sqrt();
            for (i, &x) in input.iter().enumerate() {
                output[i] = (x - mean) / std;
            }
            std::hint::black_box(&output);
        })
    }

    /// Adds a circuit benchmark result.
    pub fn add_circuit_result(&mut self, result: CircuitBenchmarkResult) {
        self.circuit_results.push(result);
    }

    /// Estimates gas cost for a proof submission.
    pub fn estimate_gas(&self, proof_size: usize) -> u64 {
        self.gas_costs.estimate_training_proof_submission(proof_size)
    }

    /// Generates a full benchmark report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# HELIX Benchmark Report\n\n");
        report.push_str(&format!("Generated at: {:?}\n\n", std::time::SystemTime::now()));

        // Compute benchmarks
        if !self.results.is_empty() {
            report.push_str("## Compute Benchmarks\n\n");
            report.push_str("| Benchmark | Iterations | Avg | Min | Max | p95 | p99 | Throughput |\n");
            report.push_str("|-----------|------------|-----|-----|-----|-----|-----|------------|\n");

            for result in &self.results {
                let p95 = result.stats.as_ref().map(|s| format!("{:.0}ns", s.p95)).unwrap_or("-".into());
                let p99 = result.stats.as_ref().map(|s| format!("{:.0}ns", s.p99)).unwrap_or("-".into());
                report.push_str(&format!(
                    "| {} | {} | {:?} | {:?} | {:?} | {} | {} | {:.2}/s |\n",
                    result.name,
                    result.iterations,
                    result.avg_duration,
                    result.min_duration,
                    result.max_duration,
                    p95,
                    p99,
                    result.throughput
                ));
            }
            report.push('\n');
        }

        // Circuit benchmarks
        if !self.circuit_results.is_empty() {
            report.push_str("## Circuit Benchmarks\n\n");
            report.push_str("| Circuit | K | Constraints | Keygen | Prove | Verify | Proof Size | Gas |\n");
            report.push_str("|---------|---|-------------|--------|-------|--------|------------|-----|\n");

            for result in &self.circuit_results {
                report.push_str(&format!(
                    "| {} | {} | {} | {:?} | {:?} | {:?} | {} B | {} |\n",
                    result.name,
                    result.k,
                    result.num_constraints,
                    result.keygen_time,
                    result.prove_time,
                    result.verify_time,
                    result.proof_size,
                    result.estimated_gas
                ));
            }
            report.push('\n');
        }

        // Summary statistics
        if !self.results.is_empty() {
            let total_iterations: usize = self.results.iter().map(|r| r.iterations).sum();
            let avg_throughput: f64 = self.results.iter().map(|r| r.throughput).sum::<f64>() / self.results.len() as f64;

            report.push_str("## Summary\n\n");
            report.push_str(&format!("- Compute benchmarks: {}\n", self.results.len()));
            report.push_str(&format!("- Circuit benchmarks: {}\n", self.circuit_results.len()));
            report.push_str(&format!("- Total iterations: {}\n", total_iterations));
            report.push_str(&format!("- Average throughput: {:.2}/s\n", avg_throughput));
        }

        report
    }

    /// Exports results to JSON for CI/CD.
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\"compute\":[");

        for (i, result) in self.results.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&result.to_json());
        }

        json.push_str("],\"circuits\":[");

        for (i, result) in self.circuit_results.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&result.to_json());
        }

        json.push_str("]}");
        json
    }

    /// Gets all compute results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Gets all circuit results.
    pub fn circuit_results(&self) -> &[CircuitBenchmarkResult] {
        &self.circuit_results
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Standard Benchmark Suite
// =============================================================================

/// Runs the standard benchmark suite.
pub fn run_standard_suite() -> BenchmarkRunner {
    let mut runner = BenchmarkRunner::new()
        .with_warmup(5)
        .with_stats();

    // Matrix multiplication benchmarks
    runner.bench_matmul(64, 64, 64, 100);
    runner.bench_matmul(128, 128, 128, 50);
    runner.bench_matmul(256, 256, 256, 20);

    // Softmax benchmarks
    runner.bench_softmax(128, 1000);
    runner.bench_softmax(512, 500);
    runner.bench_softmax(2048, 100);

    // Attention benchmarks
    runner.bench_attention(64, 64, 100);
    runner.bench_attention(128, 64, 50);

    // ReLU benchmarks
    runner.bench_relu(1024, 1000);
    runner.bench_relu(4096, 500);

    // Layer norm benchmarks
    runner.bench_layer_norm(512, 500);
    runner.bench_layer_norm(2048, 100);

    runner
}

/// Runs a quick benchmark suite for CI.
pub fn run_quick_suite() -> BenchmarkRunner {
    let mut runner = BenchmarkRunner::new()
        .with_warmup(2)
        .with_stats();

    runner.bench_matmul(64, 64, 64, 20);
    runner.bench_softmax(128, 100);
    runner.bench_attention(32, 32, 50);
    runner.bench_relu(512, 100);

    runner
}

// =============================================================================
// Overhead Analysis
// =============================================================================

/// Compares overhead between direct computation and ZK proof generation.
#[derive(Debug, Clone)]
pub struct OverheadAnalysis {
    /// Name of the computation.
    pub name: String,
    /// Time for direct computation (no proof).
    pub compute_time: Duration,
    /// Time for proof generation.
    pub prove_time: Duration,
    /// Overhead ratio (prove_time / compute_time).
    pub overhead_ratio: f64,
    /// Whether this meets the target (e.g., 30x).
    pub meets_target: bool,
    /// Target overhead ratio.
    pub target_ratio: f64,
}

impl OverheadAnalysis {
    pub fn new(
        name: &str,
        compute_time: Duration,
        prove_time: Duration,
        target_ratio: f64,
    ) -> Self {
        let overhead_ratio = if compute_time.as_nanos() > 0 {
            prove_time.as_nanos() as f64 / compute_time.as_nanos() as f64
        } else {
            f64::INFINITY
        };

        Self {
            name: name.to_string(),
            compute_time,
            prove_time,
            overhead_ratio,
            meets_target: overhead_ratio <= target_ratio,
            target_ratio,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Overhead Analysis: {}\n\
             Direct compute: {:?}\n\
             Proof generation: {:?}\n\
             Overhead: {:.1}x (target: {:.1}x)\n\
             Status: {}\n",
            self.name,
            self.compute_time,
            self.prove_time,
            self.overhead_ratio,
            self.target_ratio,
            if self.meets_target { "MEETS TARGET" } else { "EXCEEDS TARGET" },
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = Statistics::from_samples(&samples).unwrap();

        assert_eq!(stats.count, 10);
        assert!((stats.mean - 5.5).abs() < 0.001);
        assert!(stats.min == 1.0);
        assert!(stats.max == 10.0);
        // Median uses index selection, so for 10 elements it picks index 4 or 5
        assert!(stats.median >= 5.0 && stats.median <= 6.0);
        assert!(stats.p95 >= 9.0 && stats.p95 <= 10.0);
        assert!(stats.p99 >= 9.0 && stats.p99 <= 10.0);
    }

    #[test]
    fn test_benchmark_runner() {
        let mut runner = BenchmarkRunner::new().with_warmup(2).with_stats();

        let result = runner.bench("simple_add", 100, || {
            let x = 1 + 1;
            std::hint::black_box(x);
        });

        assert_eq!(result.iterations, 100);
        assert!(result.throughput > 0.0);
        assert!(result.stats.is_some());
    }

    #[test]
    fn test_matmul_bench() {
        let mut runner = BenchmarkRunner::new().with_warmup(1);
        let result = runner.bench_matmul(16, 16, 16, 10);

        assert!(result.avg_duration > Duration::ZERO);
    }

    #[test]
    fn test_gas_estimation() {
        let costs = GasCosts::default();
        let gas = costs.estimate_training_proof_submission(1024);

        // Should be > 100k gas for a typical proof submission
        assert!(gas > 100_000);

        // Estimate USD cost
        let usd = costs.gas_to_usd(gas, 30.0, 3000.0);
        println!("Estimated cost: ${:.2}", usd);
        assert!(usd > 0.0);
    }

    #[test]
    fn test_quick_suite() {
        let runner = run_quick_suite();
        assert!(runner.results().len() >= 4);
    }

    #[test]
    fn test_report_generation() {
        let mut runner = BenchmarkRunner::new();
        runner.bench("test", 10, || {});

        let report = runner.generate_report();
        assert!(report.contains("Benchmark Report"));
    }

    #[test]
    fn test_json_export() {
        let mut runner = BenchmarkRunner::new().with_stats();
        runner.bench("test_op", 10, || {
            std::hint::black_box(1 + 1);
        });

        let json = runner.to_json();
        assert!(json.contains("test_op"));
        assert!(json.contains("compute"));
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn test_memory_tracker_reports_nonzero() {
        let tracker_before = MemoryTracker::new();

        // Allocate a known-size chunk of memory
        let allocation_size = 10 * 1024 * 1024; // 10 MB
        let large_vec: Vec<u8> = vec![42u8; allocation_size];

        // Force the allocation to not be optimized away
        std::hint::black_box(&large_vec);

        let mut tracker_after = MemoryTracker::new();
        tracker_after.update();

        // The current estimate should be non-zero on macOS/Linux
        let current = MemoryTracker::estimate_current_memory();
        assert!(
            current > 0,
            "MemoryTracker should report non-zero memory on this platform"
        );

        // After allocating 10MB, the delta should be measurable
        // (allowing for some variance due to OS memory management)
        let delta = current.saturating_sub(tracker_before.start_bytes);
        println!("Memory delta after 10MB allocation: {} bytes", delta);

        // Drop the allocation
        drop(large_vec);
    }

    #[test]
    fn test_overhead_analysis() {
        let compute_time = Duration::from_millis(10);
        let prove_time = Duration::from_millis(300);

        let analysis = OverheadAnalysis::new("test", compute_time, prove_time, 30.0);

        assert_eq!(analysis.overhead_ratio, 30.0);
        assert!(analysis.meets_target);

        println!("{}", analysis.summary());
    }

    #[test]
    fn test_circuit_result_json() {
        let result = CircuitBenchmarkResult {
            name: "test_circuit".to_string(),
            k: 14,
            num_constraints: 1000,
            num_advice_columns: 4,
            keygen_time: Duration::from_secs(1),
            prove_time: Duration::from_secs(5),
            verify_time: Duration::from_millis(50),
            proof_size: 1024,
            estimated_gas: 200_000,
            overhead_ratio: Some(25.0),
        };

        let json = result.to_json();
        assert!(json.contains("test_circuit"));
        assert!(json.contains("14"));
    }
}
