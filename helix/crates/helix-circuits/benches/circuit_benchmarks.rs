//! Comprehensive benchmarks for HELIX circuit operations.
//!
//! This benchmark suite measures:
//! - Circuit synthesis time
//! - Proof generation time (via MockProver approximation)
//! - Verification time
//! - Memory usage
//!
//! Target: 30x overhead compared to native computation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use halo2_proofs::dev::MockProver;
use halo2curves::bn256::Fr;
use helix_circuits::{
    params::{
        HelixSRS, ParameterProfile, CircuitBenchmarkSuite, ModelConfig,
        ProofSizeEstimator, CircuitOptimizer, CircuitAnalysis,
    },
    ivc::IVCStepCircuit,
    approximate::{
        ErrorAccumulationCircuit, OperationData, OpType,
    },
};
use std::time::Duration;

/// Benchmark SRS generation for different K values.
fn bench_srs_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("SRS Generation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for k in [10, 12, 14] {
        group.bench_with_input(
            BenchmarkId::new("generate", k),
            &k,
            |b, &k| {
                b.iter(|| {
                    let _srs = HelixSRS::new(black_box(k)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark IVC step circuit operations.
fn bench_ivc_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("IVC Circuit");
    group.sample_size(10);

    let circuit = IVCStepCircuit::default();

    for k in [10, 12] {
        group.bench_with_input(
            BenchmarkId::new("mock_prove", k),
            &k,
            |b, &k| {
                b.iter(|| {
                    let pi = circuit.public_inputs();
                    let _prover = MockProver::run(black_box(k), &circuit, vec![pi]).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mock_verify", k),
            &k,
            |b, &k| {
                let pi = circuit.public_inputs();
                let prover = MockProver::run(k, &circuit, vec![pi]).unwrap();
                b.iter(|| {
                    let _ = prover.verify();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark error accumulation circuits.
fn bench_error_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Error Accumulation");
    group.sample_size(10);

    // Create test operations
    let operations: Vec<OperationData<Fr>> = (0..10)
        .map(|i| OperationData {
            op_type: OpType::Add,
            input_val_a: Fr::from(i as u64),
            input_err_a: Fr::from(1u64),
            input_val_b: Fr::from(i as u64 + 1),
            input_err_b: Fr::from(1u64),
            output_val: Fr::from(2 * i as u64 + 1),
            output_err: Fr::from(2u64),
        })
        .collect();

    let circuit: ErrorAccumulationCircuit<Fr, 100> = ErrorAccumulationCircuit::new(
        operations,
        Fr::from(50u64),
    );

    group.bench_function("mock_prove_10_ops", |b| {
        b.iter(|| {
            let _prover = MockProver::run(black_box(10), &circuit, vec![vec![]]).unwrap();
        });
    });

    group.finish();
}

/// Benchmark proof size estimation.
fn bench_proof_size_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Proof Size Estimation");

    let estimator = ProofSizeEstimator::default();

    group.bench_function("small_model", |b| {
        let model = ModelConfig::demo_small();
        b.iter(|| {
            let _estimate = estimator.estimate_for_model(black_box(&model));
        });
    });

    group.bench_function("transformer_model", |b| {
        let model = ModelConfig::transformer_small();
        b.iter(|| {
            let _estimate = estimator.estimate_for_model(black_box(&model));
        });
    });

    group.finish();
}

/// Benchmark circuit optimization analysis.
fn bench_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Circuit Optimization");

    let mut analysis = CircuitAnalysis::new("test_circuit");
    analysis.advice_columns = 12;
    analysis.total_gates = 5000;
    analysis.max_degree = 6;
    analysis.lookup_tables = 5;
    analysis.required_rows = 32768;
    analysis.compute_min_k();

    group.bench_function("analyze_and_optimize", |b| {
        b.iter(|| {
            let mut optimizer = CircuitOptimizer::new();
            let _summary = optimizer.optimize(black_box(&analysis));
        });
    });

    group.finish();
}

/// Benchmark the full benchmark suite.
fn bench_benchmark_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("Benchmark Suite");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("run_estimates", |b| {
        b.iter(|| {
            let mut suite = CircuitBenchmarkSuite::new();
            let _report = suite.run_estimates();
        });
    });

    group.finish();
}

/// Benchmark key generation for different profiles.
fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("Key Generation");
    group.sample_size(5);
    group.measurement_time(Duration::from_secs(30));

    // Only benchmark small K values to keep runtime reasonable
    for k in [10] {
        group.bench_with_input(
            BenchmarkId::new("keygen", k),
            &k,
            |b, &k| {
                let srs = HelixSRS::new(k).unwrap();
                b.iter(|| {
                    use helix_circuits::params::KeyBundle;
                    let _bundle = KeyBundle::from_srs(black_box(&srs), "IVCStepCircuit");
                });
            },
        );
    }

    group.finish();
}

/// Benchmark profile-based operations.
fn bench_profiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parameter Profiles");

    // Just test profile selection, not actual SRS generation
    group.bench_function("recommend_profile", |b| {
        b.iter(|| {
            for params in [100_000, 1_000_000, 10_000_000, 100_000_000] {
                let _profile = ParameterProfile::recommend_for_params(black_box(params));
            }
        });
    });

    group.bench_function("profile_estimates", |b| {
        b.iter(|| {
            for profile in ParameterProfile::all_standard() {
                let _k = profile.k();
                let _rows = profile.num_rows();
                let _time = profile.estimated_prove_time_ms();
                let _size = profile.estimated_proof_size();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_srs_generation,
    bench_ivc_circuit,
    bench_error_accumulation,
    bench_proof_size_estimation,
    bench_optimization,
    bench_benchmark_suite,
    bench_keygen,
    bench_profiles,
);

criterion_main!(benches);
