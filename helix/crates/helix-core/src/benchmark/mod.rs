//! Benchmark Suite for HELIX.
//!
//! Performance benchmarks for all major components.

use std::time::{Duration, Instant};

/// Benchmark result.
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
}

impl BenchmarkResult {
    /// Formats the result as a string.
    pub fn format(&self) -> String {
        format!(
            "{}: avg={:?} (min={:?}, max={:?}), throughput={:.2}/s",
            self.name,
            self.avg_duration,
            self.min_duration,
            self.max_duration,
            self.throughput
        )
    }
}

/// Benchmark runner.
pub struct BenchmarkRunner {
    results: Vec<BenchmarkResult>,
    warmup_iterations: usize,
}

impl BenchmarkRunner {
    /// Creates a new benchmark runner.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            warmup_iterations: 10,
        }
    }

    /// Sets warmup iterations.
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Runs a benchmark.
    pub fn bench<F>(&mut self, name: &str, iterations: usize, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }

        let mut durations = Vec::with_capacity(iterations);
        let total_start = Instant::now();

        for _ in 0..iterations {
            let start = Instant::now();
            f();
            durations.push(start.elapsed());
        }

        let total_duration = total_start.elapsed();
        let avg_duration = total_duration / iterations as u32;
        let min_duration = *durations.iter().min().unwrap_or(&Duration::ZERO);
        let max_duration = *durations.iter().max().unwrap_or(&Duration::ZERO);
        let throughput = iterations as f64 / total_duration.as_secs_f64();

        let result = BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_duration,
            avg_duration,
            min_duration,
            max_duration,
            throughput,
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
        })
    }

    /// Runs attention benchmark.
    pub fn bench_attention(&mut self, seq_len: usize, head_dim: usize, iterations: usize) -> BenchmarkResult {
        let q: Vec<f64> = (0..seq_len * head_dim).map(|i| (i % 100) as f64 / 100.0).collect();
        let k: Vec<f64> = (0..seq_len * head_dim).map(|i| (i % 100) as f64 / 100.0).collect();
        let _v: Vec<f64> = (0..seq_len * head_dim).map(|i| (i % 100) as f64 / 100.0).collect();
        let mut output = vec![0.0; seq_len * seq_len]; // Attention scores: seq_len x seq_len

        self.bench(&format!("attention_{}x{}", seq_len, head_dim), iterations, || {
            let scale = 1.0 / (head_dim as f64).sqrt();
            
            // QK^T
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0;
                    for d in 0..head_dim {
                        sum += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    output[i * seq_len + j] = sum * scale;
                }
            }
        })
    }

    /// Runs proof generation benchmark (simulated).
    pub fn bench_proof_generation(&mut self, circuit_size: usize, iterations: usize) -> BenchmarkResult {
        self.bench(&format!("proof_gen_{}", circuit_size), iterations, || {
            // Simulate proof generation with computation
            let mut state = vec![0u64; circuit_size / 64];
            for i in 0..state.len() {
                state[i] = state[i].wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            }
            std::hint::black_box(state);
        })
    }

    /// Generates a full benchmark report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# HELIX Benchmark Report\n\n");
        report.push_str(&format!("Benchmarks run: {}\n\n", self.results.len()));
        
        report.push_str("| Benchmark | Iterations | Avg | Min | Max | Throughput |\n");
        report.push_str("|-----------|------------|-----|-----|-----|------------|\n");
        
        for result in &self.results {
            report.push_str(&format!(
                "| {} | {} | {:?} | {:?} | {:?} | {:.2}/s |\n",
                result.name,
                result.iterations,
                result.avg_duration,
                result.min_duration,
                result.max_duration,
                result.throughput
            ));
        }
        
        report
    }

    /// Gets all results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the standard benchmark suite.
pub fn run_standard_suite() -> BenchmarkRunner {
    let mut runner = BenchmarkRunner::new().with_warmup(5);
    
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
    
    // Proof generation benchmarks
    runner.bench_proof_generation(1024, 100);
    runner.bench_proof_generation(4096, 50);
    
    runner
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner() {
        let mut runner = BenchmarkRunner::new().with_warmup(2);
        
        let result = runner.bench("simple_add", 100, || {
            let x = 1 + 1;
            std::hint::black_box(x);
        });
        
        assert_eq!(result.iterations, 100);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_matmul_bench() {
        let mut runner = BenchmarkRunner::new().with_warmup(1);
        let result = runner.bench_matmul(16, 16, 16, 10);
        
        assert!(result.avg_duration > Duration::ZERO);
    }

    #[test]
    fn test_full_suite() {
        let runner = run_standard_suite();
        assert!(runner.results().len() > 5);
    }

    #[test]
    fn test_report_generation() {
        let mut runner = BenchmarkRunner::new();
        runner.bench("test", 10, || {});
        
        let report = runner.generate_report();
        assert!(report.contains("Benchmark Report"));
    }
}
