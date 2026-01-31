//! Halo2 Prover Benchmarks
//!
//! Benchmarks for the Halo2-based (PLONK) proving system. Halo2 uses FFT-based
//! polynomial operations and KZG commitments, providing different tradeoffs
//! compared to GKR.
//!
//! # Benchmarks
//!
//! - Parameter generation (trusted setup)
//! - Verifying key generation
//! - Proving key generation
//! - Proof generation
//! - Proof verification
//! - Proof size measurements

use helix_circuits::halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, Instance, ProvingKey, Selector, VerifyingKey,
    },
    poly::{
        commitment::Params,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::SingleStrategy,
        },
        Rotation,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use helix_circuits::halo2curves::bn256::{Bn256, Fr, G1Affine};
use helix_prover::provers::training_prover::MLTrainingProver;

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
    ) -> Result<(), Error> {
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
        use helix_circuits::halo2_proofs::arithmetic::Field;

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

/// Halo2 benchmark suite.
pub struct Halo2Benchmarks {
    /// Pre-generated parameters for each K value to avoid regenerating.
    params_cache: std::collections::HashMap<u32, ParamsKZG<Bn256>>,
}

impl Halo2Benchmarks {
    pub fn new() -> Self {
        Self {
            params_cache: std::collections::HashMap::new(),
        }
    }

    /// Gets or generates parameters for a given K.
    fn get_params(&mut self, k: u32) -> &ParamsKZG<Bn256> {
        if !self.params_cache.contains_key(&k) {
            let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
            self.params_cache.insert(k, params);
        }
        self.params_cache.get(&k).unwrap()
    }

    /// Benchmarks parameter generation (trusted setup).
    pub fn bench_params_generation(&self, harness: &mut BenchmarkHarness) {
        for k in [8, 10, 12] {
            harness.run_benchmark(&format!("halo2_params_k{}", k), "halo2_setup", || {
                let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
                params
            });
        }
    }

    /// Benchmarks verifying key generation.
    pub fn bench_vk_generation(&mut self, harness: &mut BenchmarkHarness) {
        for k in [8, 10, 12] {
            let circuit = Halo2BenchCircuit::for_k(k);
            let params = self.get_params(k).clone();

            harness.run_benchmark(&format!("halo2_vk_k{}", k), "halo2_setup", || {
                keygen_vk(&params, &circuit).expect("vk generation")
            });
        }
    }

    /// Benchmarks proving key generation.
    pub fn bench_pk_generation(&mut self, harness: &mut BenchmarkHarness) {
        for k in [8, 10, 12] {
            let circuit = Halo2BenchCircuit::for_k(k);
            let params = self.get_params(k).clone();
            let vk = keygen_vk(&params, &circuit).expect("vk generation");

            harness.run_benchmark(&format!("halo2_pk_k{}", k), "halo2_setup", || {
                keygen_pk(&params, vk.clone(), &circuit).expect("pk generation")
            });
        }
    }

    /// Benchmarks proof generation.
    pub fn bench_proving(&mut self, harness: &mut BenchmarkHarness) {
        for k in [8, 10] {
            let circuit = Halo2BenchCircuit::for_k(k);
            let params = self.get_params(k).clone();
            let vk = keygen_vk(&params, &circuit).expect("vk generation");
            let pk = keygen_pk(&params, vk, &circuit).expect("pk generation");

            harness.run_benchmark(&format!("halo2_prove_k{}", k), "halo2_prove", || {
                let mut rng = ChaCha20Rng::seed_from_u64(42);
                let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
                create_proof::<
                    KZGCommitmentScheme<Bn256>,
                    ProverGWC<'_, Bn256>,
                    Challenge255<G1Affine>,
                    _,
                    Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
                    _,
                >(
                    &params,
                    &pk,
                    &[circuit.clone()],
                    &[&[]],
                    &mut rng,
                    &mut transcript,
                )
                .expect("proof creation");
                transcript.finalize()
            });
        }
    }

    /// Benchmarks proof verification.
    pub fn bench_verification(&mut self, harness: &mut BenchmarkHarness) {
        for k in [8, 10] {
            let circuit = Halo2BenchCircuit::for_k(k);
            let params = self.get_params(k).clone();
            let vk = keygen_vk(&params, &circuit).expect("vk generation");
            let pk = keygen_pk(&params, vk.clone(), &circuit).expect("pk generation");

            // Generate a proof to verify
            let mut rng = ChaCha20Rng::seed_from_u64(42);
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
            create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverGWC<'_, Bn256>,
                Challenge255<G1Affine>,
                _,
                Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
                _,
            >(
                &params,
                &pk,
                &[circuit.clone()],
                &[&[]],
                &mut rng,
                &mut transcript,
            )
            .expect("proof creation");
            let proof = transcript.finalize();

            harness.run_benchmark(&format!("halo2_verify_k{}", k), "halo2_verify", || {
                let mut transcript =
                    Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
                let strategy = SingleStrategy::new(&params);
                verify_proof::<
                    KZGCommitmentScheme<Bn256>,
                    VerifierGWC<'_, Bn256>,
                    Challenge255<G1Affine>,
                    Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
                    SingleStrategy<'_, Bn256>,
                >(&params, &vk, strategy, &[&[]], &mut transcript)
            });
        }
    }

    /// Benchmarks MockProver (useful for development iteration).
    pub fn bench_mock_prover(&self, harness: &mut BenchmarkHarness) {
        for k in [8, 10, 12] {
            let circuit = Halo2BenchCircuit::for_k(k);

            harness.run_benchmark(&format!("halo2_mock_k{}", k), "halo2_mock", || {
                let prover = MockProver::run(k, &circuit, vec![]).expect("mock prover");
                prover.verify()
            });
        }
    }

    /// Measures proof sizes.
    pub fn measure_proof_sizes(&mut self) -> Halo2ProofSizes {
        let mut sizes = Halo2ProofSizes::new();

        for k in [8, 10, 12] {
            let circuit = Halo2BenchCircuit::for_k(k);
            let params = self.get_params(k).clone();
            let vk = keygen_vk(&params, &circuit).expect("vk generation");
            let pk = keygen_pk(&params, vk, &circuit).expect("pk generation");

            let mut rng = ChaCha20Rng::seed_from_u64(42);
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
            create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverGWC<'_, Bn256>,
                Challenge255<G1Affine>,
                _,
                Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
                _,
            >(&params, &pk, &[circuit], &[&[]], &mut rng, &mut transcript)
            .expect("proof creation");
            let proof = transcript.finalize();

            sizes.add(k, proof.len());
        }

        sizes
    }
}

impl Default for Halo2Benchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Halo2 proof size measurements.
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
        println!("\nHalo2 Proof Size Summary:");
        println!("{:<10} {:>12} {:>15}", "K", "Rows (2^K)", "Proof (bytes)");
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

/// ML Training Step circuit benchmarks (more realistic for HELIX).
pub struct MLTrainingBenchmarks;

impl MLTrainingBenchmarks {
    /// Benchmarks ML training prover creation.
    pub fn bench_prover_creation(harness: &mut BenchmarkHarness) {
        for &size in &[ModelSize::Tiny, ModelSize::Small] {
            let k = size.halo2_k();
            let (d_in, d_hid, d_out) = size.dimensions();

            harness.run_benchmark(
                &format!("halo2_ml_prover_creation_{}", size.name()),
                "halo2_ml_setup",
                || MLTrainingProver::new(k, d_in, d_hid, d_out),
            );
        }
    }
}

/// Runs all Halo2 benchmarks.
pub fn run_halo2_benchmarks(harness: &mut BenchmarkHarness) {
    let mut benchmarks = Halo2Benchmarks::new();

    benchmarks.bench_params_generation(harness);
    benchmarks.bench_vk_generation(harness);
    benchmarks.bench_pk_generation(harness);
    benchmarks.bench_proving(harness);
    benchmarks.bench_verification(harness);
    benchmarks.bench_mock_prover(harness);
}

/// Runs Halo2 benchmarks for a specific K value (for overhead comparison).
pub fn run_halo2_for_k(harness: &mut BenchmarkHarness, k: u32) -> BenchmarkMetrics {
    let circuit = Halo2BenchCircuit::for_k(k);
    let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
    let vk = keygen_vk(&params, &circuit).expect("vk generation");
    let pk = keygen_pk(&params, vk, &circuit).expect("pk generation");

    let result = harness.run_benchmark(&format!("halo2_full_k{}", k), "halo2", || {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverGWC<'_, Bn256>,
            Challenge255<G1Affine>,
            _,
            Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
            _,
        >(
            &params,
            &pk,
            &[circuit.clone()],
            &[&[]],
            &mut rng,
            &mut transcript,
        )
        .expect("proof creation");
        transcript.finalize()
    });

    result.metrics.clone()
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
        let prover = MockProver::run(8, &circuit, vec![]).expect("mock prover");
        prover.verify().expect("verification");
    }

    #[test]
    fn test_proof_generation() {
        let k = 8;
        let circuit = Halo2BenchCircuit::for_k(k);
        let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
        let vk = keygen_vk(&params, &circuit).expect("vk generation");
        let pk = keygen_pk(&params, vk, &circuit).expect("pk generation");

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverGWC<'_, Bn256>,
            Challenge255<G1Affine>,
            _,
            Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
            _,
        >(&params, &pk, &[circuit], &[&[]], &mut rng, &mut transcript)
        .expect("proof creation");

        let proof = transcript.finalize();
        assert!(proof.len() > 0);
    }
}
