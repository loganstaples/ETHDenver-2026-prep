//! Metal vs CPU Comparison Benchmarks
//!
//! Benchmarks comparing Metal GPU acceleration (on macOS) with CPU-only
//! implementations for field operations critical to the proving system.
//!
//! # Operations Benchmarked
//!
//! - Batch field addition
//! - Batch field multiplication
//! - Batch field inversion
//! - Polynomial evaluation
//! - Sumcheck partial sums
//! - Matrix operations

use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use std::time::{Duration, Instant};

use super::{BenchmarkHarness, BenchmarkMetrics, TimingHelper};

/// Field operation type for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOperation {
    Add,
    Mul,
    Inv,
    PolyEval,
    MatMul,
}

impl FieldOperation {
    pub fn name(&self) -> &'static str {
        match self {
            FieldOperation::Add => "add",
            FieldOperation::Mul => "mul",
            FieldOperation::Inv => "inv",
            FieldOperation::PolyEval => "poly_eval",
            FieldOperation::MatMul => "matmul",
        }
    }
}

/// CPU implementation of batch field operations.
pub struct CpuFieldOps;

impl CpuFieldOps {
    /// Batch field addition.
    pub fn batch_add(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Batch field multiplication.
    pub fn batch_mul(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    /// Batch field inversion (using batched Montgomery trick).
    pub fn batch_inv(a: &[Fr]) -> Vec<Fr> {
        if a.is_empty() {
            return Vec::new();
        }

        let n = a.len();

        // Compute prefix products
        let mut prefix = Vec::with_capacity(n);
        prefix.push(a[0]);
        for i in 1..n {
            prefix.push(prefix[i - 1] * a[i]);
        }

        // Invert the final product
        let mut inv = prefix[n - 1].invert().unwrap_or(Fr::zero());

        // Compute inverses backwards
        let mut result = vec![Fr::zero(); n];
        for i in (1..n).rev() {
            result[i] = inv * prefix[i - 1];
            inv = inv * a[i];
        }
        result[0] = inv;

        result
    }

    /// Polynomial evaluation at a point.
    pub fn poly_eval(coeffs: &[Fr], point: &Fr) -> Fr {
        if coeffs.is_empty() {
            return Fr::zero();
        }

        // Horner's method
        let mut result = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            result = result * point + coeffs[i];
        }
        result
    }

    /// Matrix-vector multiplication.
    pub fn matmul(matrix: &[Fr], vector: &[Fr], rows: usize, cols: usize) -> Vec<Fr> {
        assert_eq!(matrix.len(), rows * cols);
        assert_eq!(vector.len(), cols);

        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| matrix[i * cols + j] * vector[j])
                    .fold(Fr::zero(), |a, b| a + b)
            })
            .collect()
    }

    /// Parallel batch addition (using rayon if available, otherwise sequential).
    #[cfg(feature = "parallel")]
    pub fn parallel_batch_add(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        use rayon::prelude::*;
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| x + y)
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn parallel_batch_add(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        Self::batch_add(a, b)
    }

    /// Parallel batch multiplication.
    #[cfg(feature = "parallel")]
    pub fn parallel_batch_mul(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        use rayon::prelude::*;
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| x * y)
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn parallel_batch_mul(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        Self::batch_mul(a, b)
    }
}

/// Generates test data for field operations.
pub fn generate_test_data(size: usize) -> (Vec<Fr>, Vec<Fr>) {
    let a: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64 + 1)).collect();
    let b: Vec<Fr> = (0..size).map(|i| Fr::from((i * 7 + 13) as u64)).collect();
    (a, b)
}

/// Metal vs CPU benchmark results.
#[derive(Debug, Clone)]
pub struct MetalVsCpuResult {
    pub operation: String,
    pub size: usize,
    pub cpu_time: Duration,
    pub metal_time: Option<Duration>,
    pub speedup: Option<f64>,
    pub metal_available: bool,
}

impl MetalVsCpuResult {
    pub fn cpu_only(operation: &str, size: usize, cpu_time: Duration) -> Self {
        Self {
            operation: operation.to_string(),
            size,
            cpu_time,
            metal_time: None,
            speedup: None,
            metal_available: false,
        }
    }

    pub fn with_metal(
        operation: &str,
        size: usize,
        cpu_time: Duration,
        metal_time: Duration,
    ) -> Self {
        let speedup = cpu_time.as_secs_f64() / metal_time.as_secs_f64();
        Self {
            operation: operation.to_string(),
            size,
            cpu_time,
            metal_time: Some(metal_time),
            speedup: Some(speedup),
            metal_available: true,
        }
    }
}

/// Runs Metal vs CPU benchmarks.
pub struct MetalVsCpuBenchmarks {
    timing: TimingHelper,
}

impl MetalVsCpuBenchmarks {
    pub fn new() -> Self {
        Self {
            timing: TimingHelper::default(),
        }
    }

    /// Checks if Metal is available on this platform.
    pub fn is_metal_available() -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            helix_prover::metal::is_metal_available()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            false
        }
    }

    /// Benchmarks batch addition.
    pub fn bench_add(&self, harness: &mut BenchmarkHarness) {
        for size in [1_000, 10_000, 100_000, 1_000_000] {
            let (a, b) = generate_test_data(size);

            // CPU benchmark
            harness.run_benchmark(
                &format!("cpu_add_{}", size),
                "cpu",
                || CpuFieldOps::batch_add(&a, &b),
            );

            // Metal benchmark (if available)
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if Self::is_metal_available() {
                use helix_prover::metal::{MetalFieldOps, BatchFieldOperation};

                let mut ops = MetalFieldOps::new();
                let op = BatchFieldOperation::add(a.clone(), b.clone());

                harness.run_benchmark(
                    &format!("metal_add_{}", size),
                    "metal",
                    || ops.execute(&op).unwrap(),
                );
            }
        }
    }

    /// Benchmarks batch multiplication.
    pub fn bench_mul(&self, harness: &mut BenchmarkHarness) {
        for size in [1_000, 10_000, 100_000, 1_000_000] {
            let (a, b) = generate_test_data(size);

            harness.run_benchmark(
                &format!("cpu_mul_{}", size),
                "cpu",
                || CpuFieldOps::batch_mul(&a, &b),
            );

            #[cfg(all(target_os = "macos", feature = "metal"))]
            if Self::is_metal_available() {
                use helix_prover::metal::{MetalFieldOps, BatchFieldOperation};

                let mut ops = MetalFieldOps::new();
                let op = BatchFieldOperation::mul(a.clone(), b.clone());

                harness.run_benchmark(
                    &format!("metal_mul_{}", size),
                    "metal",
                    || ops.execute(&op).unwrap(),
                );
            }
        }
    }

    /// Benchmarks batch inversion.
    pub fn bench_inv(&self, harness: &mut BenchmarkHarness) {
        for size in [1_000, 10_000, 100_000] {
            let (a, _) = generate_test_data(size);

            harness.run_benchmark(
                &format!("cpu_inv_{}", size),
                "cpu",
                || CpuFieldOps::batch_inv(&a),
            );

            #[cfg(all(target_os = "macos", feature = "metal"))]
            if Self::is_metal_available() {
                use helix_prover::metal::{MetalFieldOps, BatchFieldOperation};

                let mut ops = MetalFieldOps::new();
                let op = BatchFieldOperation::inv(a.clone());

                harness.run_benchmark(
                    &format!("metal_inv_{}", size),
                    "metal",
                    || ops.execute(&op).unwrap(),
                );
            }
        }
    }

    /// Benchmarks polynomial evaluation.
    pub fn bench_poly_eval(&self, harness: &mut BenchmarkHarness) {
        for degree in [256, 1024, 4096, 16384] {
            let coeffs: Vec<Fr> = (0..degree).map(|i| Fr::from(i as u64)).collect();
            let point = Fr::from(42u64);

            harness.run_benchmark(
                &format!("cpu_poly_eval_{}", degree),
                "cpu",
                || CpuFieldOps::poly_eval(&coeffs, &point),
            );
        }
    }

    /// Benchmarks matrix-vector multiplication.
    pub fn bench_matmul(&self, harness: &mut BenchmarkHarness) {
        for dim in [16, 32, 64, 128, 256] {
            let matrix: Vec<Fr> = (0..dim * dim).map(|i| Fr::from(i as u64)).collect();
            let vector: Vec<Fr> = (0..dim).map(|i| Fr::from(i as u64)).collect();

            harness.run_benchmark(
                &format!("cpu_matmul_{}x{}", dim, dim),
                "cpu",
                || CpuFieldOps::matmul(&matrix, &vector, dim, dim),
            );
        }
    }

    /// Benchmarks parallel vs sequential CPU operations.
    pub fn bench_parallel(&self, harness: &mut BenchmarkHarness) {
        for size in [10_000, 100_000, 1_000_000] {
            let (a, b) = generate_test_data(size);

            harness.run_benchmark(
                &format!("cpu_seq_add_{}", size),
                "cpu_seq",
                || CpuFieldOps::batch_add(&a, &b),
            );

            harness.run_benchmark(
                &format!("cpu_par_add_{}", size),
                "cpu_par",
                || CpuFieldOps::parallel_batch_add(&a, &b),
            );
        }
    }

    /// Runs all Metal vs CPU benchmarks.
    pub fn run_all(&self, harness: &mut BenchmarkHarness) {
        println!("Metal available: {}", Self::is_metal_available());

        self.bench_add(harness);
        self.bench_mul(harness);
        self.bench_inv(harness);
        self.bench_poly_eval(harness);
        self.bench_matmul(harness);
        self.bench_parallel(harness);
    }
}

impl Default for MetalVsCpuBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates a summary report of Metal vs CPU performance.
pub fn generate_metal_summary(harness: &BenchmarkHarness) -> String {
    let report = harness.report();
    let mut summary = String::new();

    summary.push_str("\n");
    summary.push_str("Metal vs CPU Performance Summary\n");
    summary.push_str("================================\n\n");

    if !MetalVsCpuBenchmarks::is_metal_available() {
        summary.push_str("Metal GPU acceleration: NOT AVAILABLE\n");
        summary.push_str("(Only available on macOS with Metal-capable GPU)\n\n");
        summary.push_str("CPU-only results:\n\n");
    } else {
        summary.push_str("Metal GPU acceleration: AVAILABLE\n\n");
    }

    // Group results by operation
    let cpu_results: Vec<_> = report.results.iter()
        .filter(|r| r.category == "cpu")
        .collect();
    let metal_results: Vec<_> = report.results.iter()
        .filter(|r| r.category == "metal")
        .collect();

    summary.push_str(&format!("{:<25} {:>12} {:>12} {:>12}\n",
        "Operation", "CPU (μs)", "Metal (μs)", "Speedup"));
    summary.push_str(&format!("{}\n", "-".repeat(65)));

    for cpu_result in &cpu_results {
        let cpu_us = cpu_result.metrics.mean_us();

        // Find matching metal result
        let metal_name = cpu_result.name.replace("cpu_", "metal_");
        let metal_result = metal_results.iter().find(|r| r.name == metal_name);

        if let Some(metal) = metal_result {
            let metal_us = metal.metrics.mean_us();
            let speedup = cpu_us / metal_us;
            summary.push_str(&format!("{:<25} {:>12.1} {:>12.1} {:>11.2}x\n",
                cpu_result.name, cpu_us, metal_us, speedup));
        } else {
            summary.push_str(&format!("{:<25} {:>12.1} {:>12} {:>12}\n",
                cpu_result.name, cpu_us, "N/A", "N/A"));
        }
    }

    summary
}

/// Runs Metal vs CPU benchmarks.
pub fn run_metal_vs_cpu_benchmarks(harness: &mut BenchmarkHarness) {
    let benchmarks = MetalVsCpuBenchmarks::new();
    benchmarks.run_all(harness);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_batch_add() {
        let (a, b) = generate_test_data(100);
        let result = CpuFieldOps::batch_add(&a, &b);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_cpu_batch_mul() {
        let (a, b) = generate_test_data(100);
        let result = CpuFieldOps::batch_mul(&a, &b);
        assert_eq!(result.len(), 100);

        // Verify correctness
        for i in 0..100 {
            assert_eq!(result[i], a[i] * b[i]);
        }
    }

    #[test]
    fn test_cpu_batch_inv() {
        let a: Vec<Fr> = (1..101).map(|i| Fr::from(i as u64)).collect();
        let inv = CpuFieldOps::batch_inv(&a);
        assert_eq!(inv.len(), 100);

        // Verify inverses
        for i in 0..100 {
            let product = a[i] * inv[i];
            assert_eq!(product, Fr::one());
        }
    }

    #[test]
    fn test_cpu_poly_eval() {
        // p(x) = 1 + 2x + 3x^2
        let coeffs = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let point = Fr::from(2u64);

        let result = CpuFieldOps::poly_eval(&coeffs, &point);
        // p(2) = 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        assert_eq!(result, Fr::from(17u64));
    }

    #[test]
    fn test_cpu_matmul() {
        // 2x2 identity matrix
        let matrix = vec![
            Fr::from(1u64), Fr::from(0u64),
            Fr::from(0u64), Fr::from(1u64),
        ];
        let vector = vec![Fr::from(5u64), Fr::from(7u64)];

        let result = CpuFieldOps::matmul(&matrix, &vector, 2, 2);
        assert_eq!(result, vector);
    }
}
