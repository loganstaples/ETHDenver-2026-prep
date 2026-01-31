//! Comprehensive Benchmark Suite Runner
//!
//! This benchmark combines all HELIX performance measurements into a single
//! comprehensive suite with automated reporting and regression detection.
//!
//! # Usage
//!
//! Run all benchmarks:
//! ```bash
//! cargo bench --package helix-integration-tests --bench comprehensive_benchmarks
//! ```
//!
//! Run with baseline comparison:
//! ```bash
//! cargo bench --package helix-integration-tests --bench comprehensive_benchmarks -- --save-baseline main
//! cargo bench --package helix-integration-tests --bench comprehensive_benchmarks -- --baseline main
//! ```
//!
//! Generate report only (skip CI checks):
//! ```bash
//! cargo bench --package helix-integration-tests --bench comprehensive_benchmarks -- --report
//! ```

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use helix_integration_tests::benches::{
    BenchmarkHarness, BenchmarkConfig, ModelSize, CircuitSize,
    native_baseline::{NativeBaseline, run_native_baselines},
    gkr_prover::{GKRBenchmarks, run_gkr_benchmarks, create_benchmark_circuit, create_mlp_circuit},
    halo2_prover::{Halo2Benchmarks, run_halo2_benchmarks},
    overhead_report::{OverheadReport, quick_overhead_check},
    metal_vs_cpu::MetalVsCpuBenchmarks,
    scaling::{analyze_model_size_scaling, analyze_circuit_size_scaling},
    TARGET_OVERHEAD_MULTIPLE, TARGET_PROOF_TIME_MS,
};
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_prover::gkr::{GKRProver, GKRConfig};
use std::time::Duration;

/// Native computation baseline benchmarks.
fn bench_native_baselines(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_baselines");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[ModelSize::Tiny, ModelSize::Small, ModelSize::Medium] {
        let baseline = NativeBaseline::new(size);
        let input = baseline.random_input();
        let target = baseline.random_target();
        let (d_in, d_hid, _) = baseline.dimensions();

        // Forward pass
        group.throughput(Throughput::Elements(size.param_count() as u64));
        group.bench_with_input(
            BenchmarkId::new("forward_pass", size.name()),
            &size,
            |b, _| {
                b.iter(|| baseline.forward_pass(&input));
            },
        );

        // Training step
        group.bench_with_input(
            BenchmarkId::new("training_step", size.name()),
            &size,
            |b, _| {
                let mut bl = NativeBaseline::new(size);
                b.iter(|| bl.training_step(&input, &target));
            },
        );
    }

    group.finish();
}

/// GKR prover benchmarks.
fn bench_gkr_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("gkr_prover");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    // Circuit size benchmarks
    for &size in &[CircuitSize::XSmall, CircuitSize::Small, CircuitSize::Medium] {
        let gates = size.gate_count();
        let circuit = create_benchmark_circuit(gates);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        group.throughput(Throughput::Elements(gates as u64));

        group.bench_with_input(
            BenchmarkId::new("prove_circuit", size.name()),
            &size,
            |b, _| {
                let config = GKRConfig::for_testing();
                b.iter(|| {
                    let mut prover = GKRProver::new(config.clone());
                    prover.prove(&circuit, &inputs).unwrap()
                });
            },
        );
    }

    // MLP model benchmarks
    for &size in &[ModelSize::Tiny, ModelSize::Small] {
        let (d_in, d_hid, d_out) = size.dimensions();
        let circuit = create_mlp_circuit(d_in, d_hid, d_out);
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();

        group.throughput(Throughput::Elements(size.param_count() as u64));

        group.bench_with_input(
            BenchmarkId::new("prove_mlp", size.name()),
            &size,
            |b, _| {
                let config = GKRConfig::for_testing();
                b.iter(|| {
                    let mut prover = GKRProver::new(config.clone());
                    prover.prove(&circuit, &inputs).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// GKR verification benchmarks.
fn bench_gkr_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("gkr_verification");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[CircuitSize::Small, CircuitSize::Medium] {
        let gates = size.gate_count();
        let circuit = create_benchmark_circuit(gates);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        let config = GKRConfig::for_testing();
        let mut prover = GKRProver::new(config.clone());
        let proof = prover.prove(&circuit, &inputs).unwrap();

        group.throughput(Throughput::Elements(gates as u64));

        group.bench_with_input(
            BenchmarkId::new("verify", size.name()),
            &size,
            |b, _| {
                use helix_prover::gkr::GKRVerifier;
                let verifier = GKRVerifier::new(config.clone());
                b.iter(|| verifier.verify_structure(&proof).unwrap());
            },
        );
    }

    group.finish();
}

/// Overhead comparison benchmarks.
fn bench_overhead_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("overhead_comparison");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &size in &[ModelSize::Tiny, ModelSize::Small] {
        let baseline = NativeBaseline::new(size);
        let (d_in, d_hid, d_out) = baseline.dimensions();
        let input = baseline.random_input();
        let target = baseline.random_target();

        // Native baseline
        group.bench_with_input(
            BenchmarkId::new("native", size.name()),
            &size,
            |b, _| {
                let mut bl = NativeBaseline::new(size);
                b.iter(|| bl.training_step(&input, &target));
            },
        );

        // GKR proving
        let circuit = create_mlp_circuit(d_in, d_hid, d_out);
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();

        group.bench_with_input(
            BenchmarkId::new("gkr", size.name()),
            &size,
            |b, _| {
                let config = GKRConfig::for_testing();
                b.iter(|| {
                    let mut prover = GKRProver::new(config.clone());
                    prover.prove(&circuit, &inputs).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Sumcheck protocol benchmarks.
fn bench_sumcheck(c: &mut Criterion) {
    use helix_prover::gkr::sumcheck::{SumcheckProver, Blake3Transcript};
    use helix_prover::gkr::multilinear::DenseMultilinear;

    let mut group = c.benchmark_group("sumcheck");
    group.measurement_time(Duration::from_secs(5));

    for num_vars in [8, 10, 12, 14] {
        let size = 1 << num_vars;
        let evals: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
        let poly = DenseMultilinear::from_evaluations(evals.clone());
        let claimed_sum: Fr = evals.iter().fold(Fr::zero(), |a, b| a + b);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("prove", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    let mut transcript = Blake3Transcript::new(b"bench");
                    let mut prover = SumcheckProver::new(&poly);
                    prover.prove_with_transcript(&mut transcript, claimed_sum).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Polynomial evaluation benchmarks.
fn bench_poly_eval(c: &mut Criterion) {
    use helix_prover::gkr::multilinear::{DenseMultilinear, MultilinearPolynomial};

    let mut group = c.benchmark_group("polynomial_evaluation");
    group.measurement_time(Duration::from_secs(3));

    for num_vars in [10, 12, 14, 16] {
        let size = 1 << num_vars;
        let evals: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
        let poly = DenseMultilinear::from_evaluations(evals);
        let point: Vec<Fr> = (0..num_vars).map(|i| Fr::from(i as u64)).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("eval", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| poly.evaluate(&point));
            },
        );
    }

    group.finish();
}

/// Proof size measurements (not timed, just informational).
fn measure_proof_sizes(_c: &mut Criterion) {
    println!("\n=== Proof Size Measurements ===\n");
    println!("{:<20} {:>12} {:>15} {:>15}",
        "Circuit", "Gates", "Proof Size", "Bytes/Gate");
    println!("{}", "-".repeat(65));

    for &size in CircuitSize::all() {
        let gates = size.gate_count();
        if gates > 100_000 {
            continue; // Skip very large for speed
        }

        let circuit = create_benchmark_circuit(gates);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];
        let config = GKRConfig::for_testing();

        let mut prover = GKRProver::new(config);
        let proof = prover.prove(&circuit, &inputs).unwrap();
        let bytes = proof.size_bytes();

        println!("{:<20} {:>12} {:>15} {:>15.2}",
            size.name(), gates, bytes, bytes as f64 / gates as f64);
    }
    println!();
}

criterion_group! {
    name = native_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .noise_threshold(0.03)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = bench_native_baselines
}

criterion_group! {
    name = gkr_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .noise_threshold(0.05)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10));
    targets = bench_gkr_prover, bench_gkr_verification
}

criterion_group! {
    name = overhead_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10));
    targets = bench_overhead_comparison
}

criterion_group! {
    name = sumcheck_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = bench_sumcheck, bench_poly_eval
}

criterion_group! {
    name = proof_sizes;
    config = Criterion::default();
    targets = measure_proof_sizes
}

criterion_main!(
    native_benches,
    gkr_benches,
    overhead_benches,
    sumcheck_benches,
    proof_sizes
);
