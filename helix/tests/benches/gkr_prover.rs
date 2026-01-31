//! GKR Prover Benchmarks
//!
//! Comprehensive benchmarks for the GKR (sumcheck-based) prover at multiple
//! model sizes. GKR is optimal for matrix multiplication and has O(n) prover
//! complexity compared to Halo2's O(n log n).
//!
//! # Benchmarks
//!
//! - Circuit creation from neural network operations
//! - Sumcheck proving rounds
//! - Complete GKR proving
//! - GKR verification
//! - Proof size measurements
//! - Parallel vs sequential performance

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_prover::gkr::{
    layered_circuit::NeuralNetworkCircuit,
    multilinear::{DenseMultilinear, MultilinearPolynomial},
    sumcheck::{Blake3Transcript, SumcheckProver},
    CircuitBuilder, GKRConfig, GKRProof, GKRProver, GKRVerifier, Gate, GateType, LayeredCircuit,
    Wire,
};

use super::{BenchmarkHarness, BenchmarkMetrics, CircuitSize, ModelSize, TimingHelper};

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Creates a layered circuit for benchmarking with the given gate count.
pub fn create_benchmark_circuit(num_gates: usize) -> LayeredCircuit {
    let mut builder = CircuitBuilder::new(2);

    // Ensure at least some gates
    let gates_per_layer = (num_gates / 2).max(1);

    // First layer: additions
    for i in 0..gates_per_layer {
        if i < 2 {
            builder.add_gate(Gate::add(Wire::input(0), Wire::input(1)));
        } else {
            builder.add_gate(Gate::add(
                Wire::internal(0, i - 2),
                Wire::internal(0, (i - 1) % gates_per_layer),
            ));
        }
    }
    builder.finish_layer();

    // Second layer: multiplications
    for i in 0..gates_per_layer {
        builder.add_gate(Gate::mul(
            Wire::internal(0, i % gates_per_layer),
            Wire::internal(0, (i + 1) % gates_per_layer),
        ));
    }
    builder.finish_layer();

    builder.build()
}

/// Creates a matrix multiplication circuit for neural network benchmarks.
pub fn create_matmul_circuit(rows: usize, cols: usize) -> LayeredCircuit {
    let weights: Vec<Fr> = (0..rows * cols)
        .map(|i| Fr::from((i as u64 * 7 + 13) % 100))
        .collect();

    NeuralNetworkCircuit::matmul(&weights, rows, cols)
}

/// Creates a 2-layer MLP circuit.
pub fn create_mlp_circuit(d_in: usize, d_hid: usize, d_out: usize) -> LayeredCircuit {
    // Weight matrices
    let w1: Vec<Fr> = (0..d_hid * d_in)
        .map(|i| Fr::from((i as u64 * 7 + 13) % 100))
        .collect();
    let w2: Vec<Fr> = (0..d_out * d_hid)
        .map(|i| Fr::from((i as u64 * 11 + 17) % 100))
        .collect();
    let b1 = vec![Fr::zero(); d_hid];
    let b2 = vec![Fr::zero(); d_out];

    NeuralNetworkCircuit::mlp_forward(&w1, &b1, &w2, &b2, d_in, d_hid, d_out)
}

/// GKR prover benchmark suite.
pub struct GKRBenchmarks {
    config: GKRConfig,
}

impl GKRBenchmarks {
    pub fn new() -> Self {
        Self {
            config: GKRConfig::default(),
        }
    }

    /// Creates a prover with the given configuration.
    pub fn with_config(config: GKRConfig) -> Self {
        Self { config }
    }

    /// Benchmarks circuit creation.
    pub fn bench_circuit_creation(&self, harness: &mut BenchmarkHarness) {
        for &size in CircuitSize::all() {
            let gates = size.gate_count();
            harness.run_benchmark(
                &format!("gkr_circuit_creation_{}", size.name()),
                "gkr_setup",
                || create_benchmark_circuit(gates),
            );
        }
    }

    /// Benchmarks GKR proving at different circuit sizes.
    pub fn bench_proving(&self, harness: &mut BenchmarkHarness) {
        for &size in &[CircuitSize::XSmall, CircuitSize::Small, CircuitSize::Medium] {
            let gates = size.gate_count();
            let circuit = create_benchmark_circuit(gates);
            let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

            let config = self.config.clone();

            harness.run_benchmark(&format!("gkr_prove_{}", size.name()), "gkr_prove", || {
                let mut prover = GKRProver::new(config.clone());
                prover.prove(&circuit, &inputs).unwrap()
            });
        }
    }

    /// Benchmarks GKR verification.
    pub fn bench_verification(&self, harness: &mut BenchmarkHarness) {
        for &size in &[CircuitSize::XSmall, CircuitSize::Small, CircuitSize::Medium] {
            let gates = size.gate_count();
            let circuit = create_benchmark_circuit(gates);
            let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

            // Generate proof first
            let mut prover = GKRProver::new(self.config.clone());
            let proof = prover.prove(&circuit, &inputs).unwrap();
            let config = self.config.clone();

            harness.run_benchmark(&format!("gkr_verify_{}", size.name()), "gkr_verify", || {
                let verifier = GKRVerifier::new(config.clone());
                verifier.verify_structure(&proof).unwrap()
            });
        }
    }

    /// Benchmarks matrix multiplication circuits.
    pub fn bench_matmul(&self, harness: &mut BenchmarkHarness) {
        for dim in [4, 8, 16, 32] {
            let circuit = create_matmul_circuit(dim, dim);
            let inputs: Vec<Fr> = (0..dim).map(|i| Fr::from(i as u64)).collect();
            let config = self.config.clone();

            harness.run_benchmark(&format!("gkr_matmul_{}x{}", dim, dim), "gkr_matmul", || {
                let mut prover = GKRProver::new(config.clone());
                prover.prove(&circuit, &inputs).unwrap()
            });
        }
    }

    /// Benchmarks 2-layer MLP proving.
    pub fn bench_mlp(&self, harness: &mut BenchmarkHarness) {
        for &size in &[ModelSize::Tiny, ModelSize::Small, ModelSize::Medium] {
            let (d_in, d_hid, d_out) = size.dimensions();
            let circuit = create_mlp_circuit(d_in, d_hid, d_out);
            let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
            let config = self.config.clone();

            harness.run_benchmark(&format!("gkr_mlp_{}", size.name()), "gkr_mlp", || {
                let mut prover = GKRProver::new(config.clone());
                prover.prove(&circuit, &inputs).unwrap()
            });
        }
    }

    /// Benchmarks sumcheck protocol specifically.
    pub fn bench_sumcheck(&self, harness: &mut BenchmarkHarness) {
        for num_vars in [8, 10, 12, 14] {
            let size = 1 << num_vars;
            let evals: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
            let poly = DenseMultilinear::from_evaluations(evals.clone());
            let claimed_sum: Fr = evals.iter().fold(Fr::zero(), |a, b| a + b);

            harness.run_benchmark(
                &format!("gkr_sumcheck_{}_vars", num_vars),
                "gkr_sumcheck",
                || {
                    let mut transcript = Blake3Transcript::new(b"bench");
                    let mut prover = SumcheckProver::new(&poly);
                    prover
                        .prove_with_transcript(&mut transcript, claimed_sum)
                        .unwrap()
                },
            );
        }
    }

    /// Benchmarks polynomial evaluation.
    pub fn bench_poly_eval(&self, harness: &mut BenchmarkHarness) {
        for num_vars in [10, 12, 14, 16] {
            let size = 1 << num_vars;
            let evals: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
            let poly = DenseMultilinear::from_evaluations(evals);
            let point: Vec<Fr> = (0..num_vars).map(|i| Fr::from(i as u64)).collect();

            harness.run_benchmark(
                &format!("gkr_poly_eval_{}_vars", num_vars),
                "gkr_poly",
                || poly.evaluate(&point),
            );
        }
    }

    /// Measures proof sizes.
    pub fn measure_proof_sizes(&self) -> ProofSizeMeasurements {
        let mut measurements = ProofSizeMeasurements::new();

        for &size in CircuitSize::all() {
            let gates = size.gate_count();
            let circuit = create_benchmark_circuit(gates);
            let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

            let mut prover = GKRProver::new(self.config.clone());
            let proof = prover.prove(&circuit, &inputs).unwrap();

            measurements.add(size.name(), gates, proof.size_bytes());
        }

        measurements
    }
}

impl Default for GKRBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof size measurements.
#[derive(Debug, Clone)]
pub struct ProofSizeMeasurements {
    pub measurements: Vec<(String, usize, usize)>, // (name, gates, bytes)
}

impl ProofSizeMeasurements {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    pub fn add(&mut self, name: &str, gates: usize, bytes: usize) {
        self.measurements.push((name.to_string(), gates, bytes));
    }

    /// Returns proof size in bytes for a given name.
    pub fn get(&self, name: &str) -> Option<usize> {
        self.measurements
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, _, b)| *b)
    }

    /// Prints a summary table.
    pub fn print_summary(&self) {
        println!("\nGKR Proof Size Summary:");
        println!(
            "{:<15} {:>12} {:>15} {:>15}",
            "Circuit", "Gates", "Proof (bytes)", "Bytes/Gate"
        );
        println!("{}", "-".repeat(60));

        for (name, gates, bytes) in &self.measurements {
            let bytes_per_gate = *bytes as f64 / *gates as f64;
            println!(
                "{:<15} {:>12} {:>15} {:>15.2}",
                name, gates, bytes, bytes_per_gate
            );
        }
    }
}

impl Default for ProofSizeMeasurements {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs all GKR benchmarks.
pub fn run_gkr_benchmarks(harness: &mut BenchmarkHarness) {
    let benchmarks = GKRBenchmarks::new();

    benchmarks.bench_circuit_creation(harness);
    benchmarks.bench_proving(harness);
    benchmarks.bench_verification(harness);
    benchmarks.bench_matmul(harness);
    benchmarks.bench_mlp(harness);
    benchmarks.bench_sumcheck(harness);
    benchmarks.bench_poly_eval(harness);
}

/// Runs GKR benchmarks for a specific model size (for overhead comparison).
pub fn run_gkr_for_model_size(harness: &mut BenchmarkHarness, size: ModelSize) -> BenchmarkMetrics {
    let (d_in, d_hid, d_out) = size.dimensions();
    let circuit = create_mlp_circuit(d_in, d_hid, d_out);
    let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
    let config = GKRConfig::default();

    let result = harness.run_benchmark(&format!("gkr_full_{}", size.name()), "gkr", || {
        let mut prover = GKRProver::new(config.clone());
        prover.prove(&circuit, &inputs).unwrap()
    });

    result.metrics.clone()
}

/// Benchmarks GKR with different thread counts (parallelism test).
pub fn bench_gkr_parallelism(harness: &mut BenchmarkHarness) {
    let circuit = create_benchmark_circuit(10_000);
    let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

    for threads in [1, 2, 4, 8] {
        let config = GKRConfig {
            parallel: threads > 1,
            num_threads: threads,
            ..GKRConfig::default()
        };

        harness.run_benchmark(
            &format!("gkr_10k_gates_{}threads", threads),
            "gkr_parallel",
            || {
                let mut prover = GKRProver::new(config.clone());
                prover.prove(&circuit, &inputs).unwrap()
            },
        );
    }
}

/// Benchmarks GKR with and without zero-knowledge.
pub fn bench_gkr_zk_overhead(harness: &mut BenchmarkHarness) {
    let circuit = create_benchmark_circuit(1_000);
    let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

    // Without ZK
    {
        let config = GKRConfig {
            zero_knowledge: false,
            ..GKRConfig::default()
        };

        harness.run_benchmark("gkr_1k_no_zk", "gkr_zk", || {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });
    }

    // With ZK
    {
        let config = GKRConfig {
            zero_knowledge: true,
            ..GKRConfig::default()
        };

        harness.run_benchmark("gkr_1k_with_zk", "gkr_zk", || {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_circuit_creation() {
        for &size in CircuitSize::all() {
            let circuit = create_benchmark_circuit(size.gate_count());
            // Basic sanity check
            assert!(circuit.layers.len() >= 2);
        }
    }

    #[test]
    fn test_matmul_circuit_creation() {
        for dim in [4, 8, 16] {
            let circuit = create_matmul_circuit(dim, dim);
            assert!(circuit.layers.len() > 0);
        }
    }

    #[test]
    fn test_mlp_circuit_creation() {
        for &size in &[ModelSize::Tiny, ModelSize::Small] {
            let (d_in, d_hid, d_out) = size.dimensions();
            let circuit = create_mlp_circuit(d_in, d_hid, d_out);
            assert!(circuit.layers.len() > 0);
        }
    }

    #[test]
    fn test_gkr_proving() {
        let circuit = create_benchmark_circuit(100);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        let config = GKRConfig::for_testing();
        let mut prover = GKRProver::new(config);
        let proof = prover.prove(&circuit, &inputs).unwrap();

        assert!(proof.size_bytes() > 0);
    }

    #[test]
    fn test_proof_size_measurements() {
        let benchmarks = GKRBenchmarks::new();
        let measurements = benchmarks.measure_proof_sizes();

        assert!(measurements.measurements.len() > 0);
        for (_, gates, bytes) in &measurements.measurements {
            assert!(*gates > 0);
            assert!(*bytes > 0);
        }
    }
}
