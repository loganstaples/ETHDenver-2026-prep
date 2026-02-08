//! Performance Regression Benchmarks for HELIX
//!
//! This module provides comprehensive benchmarking for CI/CD integration with
//! performance regression tracking. It measures critical path operations including:
//! - Circuit setup and proof generation
//! - Native and EVM verification
//! - MPC secret sharing operations
//! - Full training step pipeline
//!
//! # CI/CD Integration
//!
//! Run benchmarks with baseline comparison:
//! ```bash
//! cargo bench --package helix-integration-tests -- --save-baseline ci
//! cargo bench --package helix-integration-tests -- --baseline ci
//! ```
//!
//! For GitHub Actions, use the output format:
//! ```bash
//! cargo bench --package helix-integration-tests -- --noplot --output-format bencher
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BatchSize,
    BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use halo2_proofs::{
    dev::MockProver,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey, VerifyingKey},
    poly::{
        commitment::Params,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

// Re-use test infrastructure from the library crate
use helix_integration_tests::common::fixtures::DeterministicRng;
use helix_integration_tests::common::metrics::{Metric, MetricCollection, PerformanceBaselines};

// ============================================================================
// Baseline Management for CI/CD
// ============================================================================

/// Baseline data stored between CI runs
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CIBaseline {
    /// Version of the baseline format
    pub version: u32,
    /// Git commit hash when baseline was created
    pub commit_hash: Option<String>,
    /// Timestamp of baseline creation
    pub created_at: String,
    /// Benchmark results by name
    pub benchmarks: BTreeMap<String, BenchmarkBaseline>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkBaseline {
    /// Mean time in nanoseconds
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds
    pub std_dev_ns: f64,
    /// Minimum observed time
    pub min_ns: f64,
    /// Maximum observed time
    pub max_ns: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Throughput if applicable (ops/sec)
    pub throughput: Option<f64>,
}

impl CIBaseline {
    pub fn new() -> Self {
        Self {
            version: 1,
            commit_hash: std::env::var("GITHUB_SHA").ok(),
            created_at: chrono::Utc::now().to_rfc3339(),
            benchmarks: BTreeMap::new(),
        }
    }

    pub fn load(path: &PathBuf) -> Option<Self> {
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).ok()
    }

    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn check_regression(&self, name: &str, current_ns: f64) -> RegressionResult {
        match self.benchmarks.get(name) {
            Some(baseline) => {
                let threshold = 1.10; // 10% regression threshold
                let ratio = current_ns / baseline.mean_ns;

                if ratio > threshold {
                    RegressionResult::Regression {
                        baseline_ns: baseline.mean_ns,
                        current_ns,
                        ratio,
                    }
                } else if ratio < 0.9 {
                    RegressionResult::Improvement {
                        baseline_ns: baseline.mean_ns,
                        current_ns,
                        ratio,
                    }
                } else {
                    RegressionResult::Stable {
                        baseline_ns: baseline.mean_ns,
                        current_ns,
                        ratio,
                    }
                }
            }
            None => RegressionResult::NoBaseline,
        }
    }
}

#[derive(Debug)]
pub enum RegressionResult {
    NoBaseline,
    Stable {
        baseline_ns: f64,
        current_ns: f64,
        ratio: f64,
    },
    Improvement {
        baseline_ns: f64,
        current_ns: f64,
        ratio: f64,
    },
    Regression {
        baseline_ns: f64,
        current_ns: f64,
        ratio: f64,
    },
}

// ============================================================================
// Test Circuit for Benchmarking
// ============================================================================

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector},
    poly::Rotation,
};

/// Minimal circuit mimicking MLTrainingStepV2 structure for benchmarking
#[derive(Clone)]
struct BenchmarkCircuit {
    /// Matrix multiplication input
    input: Vec<Fr>,
    /// Weight matrix (flattened)
    weights: Vec<Fr>,
    /// Dimensions
    rows: usize,
    cols: usize,
}

#[derive(Clone)]
struct BenchmarkConfig {
    advice: [Column<Advice>; 3],
    instance: Column<Instance>,
    selector: Selector,
}

impl Circuit<Fr> for BenchmarkCircuit {
    type Config = BenchmarkConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            input: vec![Fr::zero(); self.input.len()],
            weights: vec![Fr::zero(); self.weights.len()],
            rows: self.rows,
            cols: self.cols,
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

        BenchmarkConfig {
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

impl BenchmarkCircuit {
    fn new(size: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let input: Vec<Fr> = (0..size)
            .map(|_| Fr::from(DeterministicRng::next_u64(&mut rng)))
            .collect();
        let weights: Vec<Fr> = (0..size)
            .map(|_| Fr::from(DeterministicRng::next_u64(&mut rng)))
            .collect();

        Self {
            input,
            weights,
            rows: size,
            cols: size,
        }
    }
}

// ============================================================================
// Circuit Setup Benchmarks
// ============================================================================

fn bench_circuit_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_setup");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10); // Setup is slow

    for k in [8, 10, 12] {
        let circuit = BenchmarkCircuit::new(1 << (k - 2));

        group.bench_with_input(
            BenchmarkId::new("params_generation", format!("k={}", k)),
            &k,
            |b, &k| {
                b.iter(|| {
                    let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
                    black_box(params)
                });
            },
        );

        // Pre-generate params for VK/PK benchmarks
        let params: ParamsKZG<Bn256> = ParamsKZG::new(k);

        group.bench_with_input(
            BenchmarkId::new("vk_generation", format!("k={}", k)),
            &(&params, &circuit),
            |b, (params, circuit)| {
                b.iter(|| {
                    let vk = keygen_vk(*params, *circuit).expect("vk generation");
                    black_box(vk)
                });
            },
        );

        let vk = keygen_vk(&params, &circuit).expect("vk generation");

        group.bench_with_input(
            BenchmarkId::new("pk_generation", format!("k={}", k)),
            &(&params, &circuit, &vk),
            |b, (params, circuit, vk)| {
                b.iter(|| {
                    let pk = keygen_pk(*params, (*vk).clone(), *circuit).expect("pk generation");
                    black_box(pk)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Proof Generation Benchmarks
// ============================================================================

fn bench_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for k in [8, 10] {
        let circuit = BenchmarkCircuit::new(1 << (k - 2));
        let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
        let vk = keygen_vk(&params, &circuit).expect("vk generation");
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("pk generation");

        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("create_proof", format!("k={}", k)),
            &(&params, &pk, &circuit),
            |b, (params, pk, circuit)| {
                b.iter_batched(
                    || {
                        let mut rng = ChaCha20Rng::seed_from_u64(42);
                        (rng, circuit.clone())
                    },
                    |(mut rng, circuit)| {
                        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(
                            Vec::new(),
                        );
                        create_proof::<
                            KZGCommitmentScheme<Bn256>,
                            ProverSHPLONK<'_, Bn256>,
                            Challenge255<G1Affine>,
                            _,
                            Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
                            _,
                        >(
                            *params, *pk, &[circuit], &[&[]], &mut rng, &mut transcript
                        )
                        .expect("proof creation");
                        black_box(transcript.finalize())
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Verification Benchmarks
// ============================================================================

fn bench_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification");
    group.sample_size(50);

    for k in [8, 10] {
        let circuit = BenchmarkCircuit::new(1 << (k - 2));
        let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
        let vk = keygen_vk(&params, &circuit).expect("vk generation");
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("pk generation");

        // Generate a proof to verify
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut transcript =
            Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            Challenge255<G1Affine>,
            _,
            Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
            _,
        >(&params, &pk, &[circuit.clone()], &[&[]], &mut rng, &mut transcript)
        .expect("proof creation");
        let proof = transcript.finalize();

        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("native_verify", format!("k={}", k)),
            &(&params, &vk, &proof),
            |b, (params, vk, proof)| {
                b.iter(|| {
                    let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(
                        proof.as_slice(),
                    );
                    let strategy = SingleStrategy::new(*params);
                    let result = verify_proof::<
                        KZGCommitmentScheme<Bn256>,
                        VerifierSHPLONK<'_, Bn256>,
                        Challenge255<G1Affine>,
                        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
                        SingleStrategy<'_, Bn256>,
                    >(
                        *params, *vk, strategy, &[&[]], &mut transcript
                    );
                    black_box(result)
                });
            },
        );

        // Mock prover verification (for development cycle)
        group.bench_with_input(
            BenchmarkId::new("mock_prover", format!("k={}", k)),
            &(k, &circuit),
            |b, (k, circuit)| {
                b.iter(|| {
                    let prover = MockProver::run(*k, *circuit, vec![]).expect("mock prover");
                    black_box(prover.verify())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// MPC Operations Benchmarks
// ============================================================================

fn bench_mpc_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpc_operations");

    // Test different party counts and value sizes
    let party_counts = [3, 5, 7];
    let value_counts = [16, 64, 256];

    for &n_parties in &party_counts {
        for &n_values in &value_counts {
            let id = format!("parties={},values={}", n_parties, n_values);

            // Additive secret sharing
            group.bench_with_input(
                BenchmarkId::new("additive_share", &id),
                &(n_parties, n_values),
                |b, &(n_parties, n_values)| {
                    b.iter_batched(
                        || {
                            let rng = ChaCha20Rng::seed_from_u64(42);
                            let values: Vec<Fr> = (0..n_values).map(|i| Fr::from(i as u64)).collect();
                            (rng, values)
                        },
                        |(mut rng, values)| {
                            let shares: Vec<Vec<Fr>> = values
                                .iter()
                                .map(|v| {
                                    let mut shares: Vec<Fr> = (0..n_parties - 1)
                                        .map(|_| Fr::from(DeterministicRng::next_u64(&mut rng)))
                                        .collect();
                                    let sum: Fr = shares.iter().fold(Fr::zero(), |a, b| a + b);
                                    shares.push(*v - sum);
                                    shares
                                })
                                .collect();
                            black_box(shares)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            // Reconstruction
            group.bench_with_input(
                BenchmarkId::new("additive_reconstruct", &id),
                &(n_parties, n_values),
                |b, &(n_parties, n_values)| {
                    // Pre-generate shares
                    let mut rng = ChaCha20Rng::seed_from_u64(42);
                    let shares: Vec<Vec<Fr>> = (0..n_values)
                        .map(|i| {
                            let v = Fr::from(i as u64);
                            let mut s: Vec<Fr> = (0..n_parties - 1)
                                .map(|_| Fr::from(DeterministicRng::next_u64(&mut rng)))
                                .collect();
                            let sum: Fr = s.iter().fold(Fr::zero(), |a, b| a + b);
                            s.push(v - sum);
                            s
                        })
                        .collect();

                    b.iter(|| {
                        let reconstructed: Vec<Fr> = shares
                            .iter()
                            .map(|s| s.iter().fold(Fr::zero(), |a, b| a + b))
                            .collect();
                        black_box(reconstructed)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Hash Computation Benchmarks
// ============================================================================

fn bench_hash_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_computation");

    let sizes = [64, 256, 1024, 4096]; // bytes

    for &size in &sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("sha256", format!("{}bytes", size)),
            &size,
            |b, &size| {
                let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
                b.iter(|| {
                    let mut hasher = Sha256::new();
                    hasher.update(&data);
                    black_box(hasher.finalize())
                });
            },
        );

        // State hash (Fr elements)
        let n_elements = size / 32;
        group.bench_with_input(
            BenchmarkId::new("state_hash", format!("{}elements", n_elements)),
            &n_elements,
            |b, &n| {
                let elements: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64)).collect();
                b.iter(|| {
                    let mut hasher = Sha256::new();
                    for elem in &elements {
                        let bytes = elem.to_bytes();
                        hasher.update(&bytes);
                    }
                    black_box(hasher.finalize())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Training Step Pipeline Benchmarks
// ============================================================================

fn bench_training_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_pipeline");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    // Simulate full training step overhead
    let model_sizes = [
        ("tiny", 4, 4),    // 4x4 matrix
        ("small", 8, 8),   // 8x8 matrix
        ("medium", 16, 16), // 16x16 matrix
    ];

    for (name, rows, cols) in model_sizes {
        let total_elements = rows * cols;

        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new("forward_pass", name),
            &(rows, cols),
            |b, &(rows, cols)| {
                b.iter_batched(
                    || {
                        let rng = ChaCha20Rng::seed_from_u64(42);
                        let input: Vec<Fr> = (0..cols).map(|i| Fr::from(i as u64)).collect();
                        let weights: Vec<Fr> = (0..rows * cols)
                            .map(|i| Fr::from(i as u64 + 1))
                            .collect();
                        (rng, input, weights, rows, cols)
                    },
                    |(_rng, input, weights, rows, cols)| {
                        // Simulate matrix multiplication
                        let output: Vec<Fr> = (0..rows)
                            .map(|i| {
                                (0..cols)
                                    .map(|j| weights[i * cols + j] * input[j])
                                    .fold(Fr::zero(), |a, b| a + b)
                            })
                            .collect();
                        black_box(output)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gradient_computation", name),
            &(rows, cols),
            |b, &(rows, cols)| {
                b.iter_batched(
                    || {
                        let output: Vec<Fr> = (0..rows).map(|i| Fr::from(i as u64)).collect();
                        let target: Vec<Fr> = (0..rows).map(|i| Fr::from(i as u64 + 1)).collect();
                        let input: Vec<Fr> = (0..cols).map(|i| Fr::from(i as u64)).collect();
                        (output, target, input, rows, cols)
                    },
                    |(output, target, input, rows, cols)| {
                        // Gradient = (output - target) * input^T
                        let error: Vec<Fr> = output
                            .iter()
                            .zip(&target)
                            .map(|(o, t)| *o - *t)
                            .collect();
                        let gradients: Vec<Fr> = (0..rows)
                            .flat_map(|i| (0..cols).map(move |j| error[i] * input[j]))
                            .collect();
                        black_box(gradients)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Serialization Benchmarks
// ============================================================================

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let element_counts = [16, 64, 256, 1024];

    for &count in &element_counts {
        let elements: Vec<Fr> = (0..count).map(|i| Fr::from(i as u64)).collect();

        group.throughput(Throughput::Elements(count as u64));

        // Fr element serialization
        group.bench_with_input(
            BenchmarkId::new("fr_to_bytes", count),
            &elements,
            |b, elements| {
                b.iter(|| {
                    let bytes: Vec<[u8; 32]> = elements.iter().map(|e| e.to_bytes()).collect();
                    black_box(bytes)
                });
            },
        );

        // Pre-serialize for deserialization bench
        let serialized: Vec<[u8; 32]> = elements.iter().map(|e| e.to_bytes()).collect();

        group.bench_with_input(
            BenchmarkId::new("fr_from_bytes", count),
            &serialized,
            |b, serialized| {
                b.iter(|| {
                    let elements: Vec<Fr> = serialized
                        .iter()
                        .map(|bytes| Fr::from_bytes(bytes).unwrap())
                        .collect();
                    black_box(elements)
                });
            },
        );

        // Bincode serialization (full struct)
        #[derive(serde::Serialize, serde::Deserialize)]
        struct BenchData {
            elements: Vec<[u8; 32]>,
        }

        let data = BenchData {
            elements: elements.iter().map(|e| e.to_bytes()).collect(),
        };

        group.bench_with_input(
            BenchmarkId::new("bincode_serialize", count),
            &data,
            |b, data| {
                b.iter(|| {
                    let bytes = bincode::serialize(data).expect("serialize");
                    black_box(bytes)
                });
            },
        );

        let serialized_bincode = bincode::serialize(&data).expect("serialize");

        group.bench_with_input(
            BenchmarkId::new("bincode_deserialize", count),
            &serialized_bincode,
            |b, bytes| {
                b.iter(|| {
                    let data: BenchData = bincode::deserialize(bytes).expect("deserialize");
                    black_box(data)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// CI/CD Performance Report
// ============================================================================

/// Generate a performance report suitable for CI/CD
pub fn generate_ci_report(metrics: &MetricCollection, baseline: Option<&CIBaseline>) -> String {
    let mut report = String::new();
    report.push_str("# HELIX Performance Report\n\n");
    report.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));

    if let Some(baseline) = baseline {
        report.push_str(&format!(
            "Comparing against baseline from: {}\n",
            baseline.created_at
        ));
        if let Some(ref commit) = baseline.commit_hash {
            report.push_str(&format!("Baseline commit: {}\n", commit));
        }
        report.push_str("\n");
    }

    report.push_str("## Results\n\n");
    report.push_str("| Benchmark | Current | Baseline | Change |\n");
    report.push_str("|-----------|---------|----------|--------|\n");

    for (name, metric) in metrics.iter() {
        let current_ms = metric.mean_duration().as_secs_f64() * 1000.0;

        let (baseline_str, change_str) = match baseline {
            Some(bl) => match bl.benchmarks.get(name) {
                Some(bm) => {
                    let baseline_ms = bm.mean_ns / 1_000_000.0;
                    let ratio = current_ms / baseline_ms;
                    let change = if ratio > 1.10 {
                        format!("🔴 +{:.1}%", (ratio - 1.0) * 100.0)
                    } else if ratio < 0.90 {
                        format!("🟢 -{:.1}%", (1.0 - ratio) * 100.0)
                    } else {
                        format!("⚪ {:.1}%", (ratio - 1.0) * 100.0)
                    };
                    (format!("{:.2}ms", baseline_ms), change)
                }
                None => ("N/A".to_string(), "new".to_string()),
            },
            None => ("N/A".to_string(), "N/A".to_string()),
        };

        report.push_str(&format!(
            "| {} | {:.2}ms | {} | {} |\n",
            name, current_ms, baseline_str, change_str
        ));
    }

    report.push_str("\n## Summary\n\n");

    let baselines = PerformanceBaselines::default();
    let acceptable = baselines.max_zk_overhead;
    report.push_str(&format!(
        "- Target ZK overhead: {:.0}x\n",
        acceptable
    ));

    report
}

// ============================================================================
// Custom Measurement for Gas Tracking
// ============================================================================

/// Tracks both time and simulated gas for EVM operations
#[derive(Debug, Clone)]
pub struct GasTracker {
    pub duration: Duration,
    pub gas_used: u64,
}

impl GasTracker {
    pub fn new() -> Self {
        Self {
            duration: Duration::ZERO,
            gas_used: 0,
        }
    }

    pub fn start() -> Instant {
        Instant::now()
    }

    pub fn stop(start: Instant, gas: u64) -> Self {
        Self {
            duration: start.elapsed(),
            gas_used: gas,
        }
    }
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group! {
    name = setup_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .noise_threshold(0.05)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = bench_circuit_setup
}

criterion_group! {
    name = proof_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .noise_threshold(0.05)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10));
    targets = bench_proof_generation, bench_verification
}

criterion_group! {
    name = mpc_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .noise_threshold(0.03)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_mpc_operations
}

criterion_group! {
    name = util_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_hash_computation, bench_serialization
}

criterion_group! {
    name = pipeline_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = bench_training_pipeline
}

criterion_main!(
    setup_benches,
    proof_benches,
    mpc_benches,
    util_benches,
    pipeline_benches
);

// ============================================================================
// Integration with GitHub Actions
// ============================================================================

#[cfg(test)]
mod ci_integration_tests {
    use super::*;

    #[test]
    fn test_baseline_serialization() {
        let mut baseline = CIBaseline::new();
        baseline.benchmarks.insert(
            "test_bench".to_string(),
            BenchmarkBaseline {
                mean_ns: 1_000_000.0,
                std_dev_ns: 50_000.0,
                min_ns: 900_000.0,
                max_ns: 1_100_000.0,
                sample_count: 100,
                throughput: Some(1000.0),
            },
        );

        let json = serde_json::to_string_pretty(&baseline).expect("serialize");
        let parsed: CIBaseline = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.version, 1);
        assert!(parsed.benchmarks.contains_key("test_bench"));
    }

    #[test]
    fn test_regression_detection() {
        let mut baseline = CIBaseline::new();
        baseline.benchmarks.insert(
            "test".to_string(),
            BenchmarkBaseline {
                mean_ns: 1_000_000.0,
                std_dev_ns: 50_000.0,
                min_ns: 900_000.0,
                max_ns: 1_100_000.0,
                sample_count: 100,
                throughput: None,
            },
        );

        // Regression (20% slower)
        match baseline.check_regression("test", 1_200_000.0) {
            RegressionResult::Regression { ratio, .. } => {
                assert!((ratio - 1.2).abs() < 0.01);
            }
            _ => panic!("Expected regression"),
        }

        // Improvement (20% faster)
        match baseline.check_regression("test", 800_000.0) {
            RegressionResult::Improvement { ratio, .. } => {
                assert!((ratio - 0.8).abs() < 0.01);
            }
            _ => panic!("Expected improvement"),
        }

        // Stable (5% variation)
        match baseline.check_regression("test", 1_050_000.0) {
            RegressionResult::Stable { .. } => {}
            _ => panic!("Expected stable"),
        }

        // No baseline
        match baseline.check_regression("unknown", 1_000_000.0) {
            RegressionResult::NoBaseline => {}
            _ => panic!("Expected no baseline"),
        }
    }

    #[test]
    fn test_ci_report_generation() {
        let mut metrics = MetricCollection::new("test".to_string());
        metrics.record(
            "proof_generation",
            Metric::from_duration(Duration::from_millis(100)),
        );
        metrics.record(
            "verification",
            Metric::from_duration(Duration::from_millis(10)),
        );

        let report = generate_ci_report(&metrics, None);
        assert!(report.contains("Performance Report"));
        assert!(report.contains("proof_generation"));
        assert!(report.contains("verification"));
    }
}

/// GitHub Actions workflow example (to be placed in .github/workflows/benchmarks.yml):
/// ```yaml
/// name: Performance Benchmarks
///
/// on:
///   push:
///     branches: [main]
///   pull_request:
///     branches: [main]
///
/// jobs:
///   benchmark:
///     runs-on: ubuntu-latest
///     steps:
///       - uses: actions/checkout@v4
///
///       - name: Install Rust
///         uses: dtolnay/rust-action@stable
///
///       - name: Run benchmarks
///         run: |
///           cargo bench --package helix-integration-tests \
///             -- --noplot --save-baseline pr-${{ github.event.pull_request.number || 'main' }}
///
///       - name: Compare with main
///         if: github.event_name == 'pull_request'
///         run: |
///           cargo bench --package helix-integration-tests \
///             -- --noplot --baseline main --load-baseline pr-${{ github.event.pull_request.number }}
///
///       - name: Upload benchmark results
///         uses: actions/upload-artifact@v4
///         with:
///           name: benchmark-results
///           path: target/criterion
/// ```
