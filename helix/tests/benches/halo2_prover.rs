//! Halo2 Prover Benchmarks
//!
//! Benchmarks for the Halo2-based (PLONK) proving system using MockProver.
//! This provides timing estimates for Halo2 circuit operations without
//! requiring full KZG setup.
//!
//! # Benchmarks
//!
//! - Circuit synthesis
//! - MockProver execution
//! - Constraint verification
//! - Proof size estimates

use helix_circuits::halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Advice, Circuit, Column, ConstraintSystem, ErrorFront, Instance, Selector},
    poly::Rotation,
};
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::{Duration, Instant};

use super::{BenchmarkHarness, BenchmarkMetrics, ModelSize, TimingHelper};

/// Simple benchmark circuit for Halo2 measurements.
#[derive(Clone)]
pub struct Halo2BenchCircuit {
    input: Vec<Fr>,
    weights: Vec<Fr>,
    rows: usize,
}

#[derive(Clone)]
pub struct Halo2BenchConfig {
    advice: [Column<Advice>; 3],
    instance: Column<Instance>,
    selector: Selector,
}

impl Circuit<Fr> for Halo2BenchCircuit {
    type Config = Halo2BenchConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            input: vec![Fr::zero(); self.input.len()],
            weights: vec![Fr::zero(); self.weights.len()],
            rows: self.rows,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let instance = meta.instance_column();
        let selector = meta.selector();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);

        // Simple constraint: a * b = c
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(selector);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        Halo2BenchConfig {
            advice,
            instance,
            selector,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        layouter.assign_region(
            || "benchmark region",
            |mut region| {
                for i in 0..self.rows {
                    config.selector.enable(&mut region, i)?;

                    let a = if i < self.input.len() {
                        self.input[i]
                    } else {
                        Fr::one()
                    };
                    let b = if i < self.weights.len() {
                        self.weights[i]
                    } else {
                        Fr::one()
                    };
                    let c = a * b;

                    region.assign_advice(|| "a", config.advice[0], i, || Value::known(a))?;
                    region.assign_advice(|| "b", config.advice[1], i, || Value::known(b))?;
                    region.assign_advice(|| "c", config.advice[2], i, || Value::known(c))?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }
}

impl Halo2BenchCircuit {
    /// Creates a new benchmark circuit with the given size.
    pub fn new(size: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let input: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();
        let weights: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();

        Self {
            input,
            weights,
            rows: size,
        }
    }

    /// Creates a circuit sized for the given K value (2^K rows).
    pub fn for_k(k: u32) -> Self {
        let size = 1 << (k - 2); // Leave room for blinding factors
        Self::new(size)
    }
}

/// Halo2 benchmark suite using MockProver.
pub struct Halo2Benchmarks;

impl Halo2Benchmarks {
    pub fn new() -> Self {
        Self
    }

    /// Benchmarks MockProver for different K values.
    pub fn bench_mock_prover(&self, harness: &mut BenchmarkHarness) {
        for k in [8, 10, 12] {
            let circuit = Halo2BenchCircuit::for_k(k);

            harness.run_benchmark(&format!("halo2_mock_k{}", k), "halo2_mock", || {
                let prover = MockProver::run(k, &circuit, vec![vec![]]).expect("mock prover");
                prover.verify()
            });
        }
    }

    /// Benchmarks circuit synthesis timing.
    pub fn bench_synthesis(&self, harness: &mut BenchmarkHarness) {
        for k in [8, 10, 12] {
            harness.run_benchmark(&format!("halo2_synthesis_k{}", k), "halo2_synthesis", || {
                Halo2BenchCircuit::for_k(k)
            });
        }
    }

    /// Estimates proof sizes (based on typical Halo2/KZG proofs).
    pub fn estimate_proof_sizes(&self) -> Halo2ProofSizes {
        let mut sizes = Halo2ProofSizes::new();

        // Halo2 with KZG proofs are typically constant size regardless of circuit
        // Typical sizes: ~1KB for basic proofs, ~2KB with additional columns
        for k in [8, 10, 12, 14, 16] {
            // Base size plus small increment per column
            let estimated_bytes = 1024 + (k as usize * 32);
            sizes.add(k, estimated_bytes);
        }

        sizes
    }
}

impl Default for Halo2Benchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Halo2 proof size estimates.
#[derive(Debug, Clone)]
pub struct Halo2ProofSizes {
    pub sizes: Vec<(u32, usize)>, // (k, bytes)
}

impl Halo2ProofSizes {
    pub fn new() -> Self {
        Self { sizes: Vec::new() }
    }

    pub fn add(&mut self, k: u32, bytes: usize) {
        self.sizes.push((k, bytes));
    }

    pub fn get(&self, k: u32) -> Option<usize> {
        self.sizes.iter().find(|(kv, _)| *kv == k).map(|(_, b)| *b)
    }

    pub fn print_summary(&self) {
        println!("\nHalo2 Proof Size Estimates:");
        println!("{:<10} {:>12} {:>15}", "K", "Rows (2^K)", "Est. Proof (bytes)");
        println!("{}", "-".repeat(40));

        for (k, bytes) in &self.sizes {
            let rows = 1u64 << k;
            println!("{:<10} {:>12} {:>15}", k, rows, bytes);
        }
    }
}

impl Default for Halo2ProofSizes {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs all Halo2 benchmarks using MockProver.
pub fn run_halo2_benchmarks(harness: &mut BenchmarkHarness) {
    let benchmarks = Halo2Benchmarks::new();

    benchmarks.bench_synthesis(harness);
    benchmarks.bench_mock_prover(harness);
}

/// Runs Halo2 MockProver for a specific K value (for overhead comparison).
pub fn run_halo2_for_k(harness: &mut BenchmarkHarness, k: u32) -> BenchmarkMetrics {
    let circuit = Halo2BenchCircuit::for_k(k);

    let result = harness.run_benchmark(&format!("halo2_mock_k{}", k), "halo2", || {
        let prover = MockProver::run(k, &circuit, vec![vec![]]).expect("mock prover");
        prover.verify()
    });

    result.metrics.clone()
}

/// Measures MockProver timing for overhead calculations.
pub fn measure_mock_prover_time(k: u32) -> Duration {
    let circuit = Halo2BenchCircuit::for_k(k);

    let timing = TimingHelper::quick();
    let result = timing.measure(|| {
        let prover = MockProver::run(k, &circuit, vec![vec![]]).expect("mock prover");
        prover.verify()
    });

    result.mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_circuit_creation() {
        for k in [8, 10, 12] {
            let circuit = Halo2BenchCircuit::for_k(k);
            assert!(circuit.rows > 0);
        }
    }

    #[test]
    fn test_mock_prover() {
        let circuit = Halo2BenchCircuit::for_k(8);
        // Pass empty instance values for the instance column
        let prover = MockProver::run(8, &circuit, vec![vec![]]).expect("mock prover");
        prover.verify().expect("verification");
    }

    #[test]
    fn test_proof_size_estimates() {
        let benchmarks = Halo2Benchmarks::new();
        let sizes = benchmarks.estimate_proof_sizes();

        assert!(sizes.get(8).is_some());
        assert!(sizes.get(10).is_some());
    }
}
