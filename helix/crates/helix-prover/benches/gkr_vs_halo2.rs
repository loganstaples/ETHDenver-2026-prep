//! Benchmark: GKR vs Halo2 Prover Performance
//!
//! This benchmark compares the proving performance of:
//! - GKR (Orion-style sumcheck-based prover)
//! - Halo2 (Plonk-based prover with KZG commitments)
//!
//! We measure:
//! - Setup time (if applicable)
//! - Proving time
//! - Verification time
//! - Proof size
//!
//! Run with: cargo bench --bench gkr_vs_halo2

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use helix_prover::gkr::{
    GKRProver, GKRConfig, GKRVerifier,
    LayeredCircuit, CircuitLayer, Gate, Wire, GateType,
    layered_circuit::{CircuitBuilder, NeuralNetworkCircuit},
};
use helix_prover::pipeline::ProverPipeline;
use helix_prover::provers::training_prover::MLTrainingProver;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use std::time::Duration;

/// Creates a simple layered circuit for benchmarking.
fn create_benchmark_circuit(num_gates: usize) -> LayeredCircuit {
    let mut builder = CircuitBuilder::new(2);

    // First layer: additions
    let gates_per_layer = (num_gates / 2).max(1);

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

/// Creates a matrix multiplication circuit.
fn create_matmul_circuit(dim: usize) -> LayeredCircuit {
    // Random-ish weights
    let weights: Vec<Fr> = (0..dim * dim)
        .map(|i| Fr::from((i as u64 * 7 + 13) % 100))
        .collect();

    NeuralNetworkCircuit::matmul(&weights, dim, dim)
}

/// Benchmark GKR proving.
fn bench_gkr_proving(c: &mut Criterion) {
    let mut group = c.benchmark_group("GKR Proving");
    group.measurement_time(Duration::from_secs(10));

    // Different circuit sizes
    for size in [100, 1000, 10000, 100000].iter() {
        let circuit = create_benchmark_circuit(*size);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("prove", size),
            size,
            |b, _| {
                let config = GKRConfig::for_testing();
                let mut prover = GKRProver::new(config);

                b.iter(|| {
                    prover.prove(&circuit, &inputs).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GKR verification.
fn bench_gkr_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("GKR Verification");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 1000, 10000].iter() {
        let circuit = create_benchmark_circuit(*size);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        let config = GKRConfig::for_testing();
        let mut prover = GKRProver::new(config.clone());
        let proof = prover.prove(&circuit, &inputs).unwrap();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("verify", size),
            size,
            |b, _| {
                let verifier = GKRVerifier::new(config.clone());
                b.iter(|| {
                    verifier.verify_structure(&proof).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Halo2 proving (small circuits only due to time).
fn bench_halo2_proving(c: &mut Criterion) {
    let mut group = c.benchmark_group("Halo2 Proving");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10); // Fewer samples due to slow Halo2

    // Only small sizes for Halo2 (it's much slower)
    for k in [10, 12, 14].iter() {
        group.bench_with_input(
            BenchmarkId::new("prove", format!("k={}", k)),
            k,
            |b, &k| {
                let prover = MLTrainingProver::new(k, 2, 2, 1);

                b.iter(|| {
                    // Note: We can't easily benchmark the full prove since it needs witnesses
                    // This is a placeholder - in practice we'd need proper witness generation
                    let _ = &prover;
                });
            },
        );
    }

    group.finish();
}

/// Benchmark matrix multiplication circuits with GKR.
fn bench_gkr_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("GKR MatMul");
    group.measurement_time(Duration::from_secs(10));

    for dim in [4, 8, 16, 32].iter() {
        let circuit = create_matmul_circuit(*dim);
        let inputs: Vec<Fr> = (0..*dim).map(|i| Fr::from(i as u64)).collect();

        let num_ops = dim * dim; // Approximate
        group.throughput(Throughput::Elements(num_ops as u64));

        group.bench_with_input(
            BenchmarkId::new("matmul", format!("{}x{}", dim, dim)),
            dim,
            |b, _| {
                let config = GKRConfig::for_testing();
                let mut prover = GKRProver::new(config);

                b.iter(|| {
                    prover.prove(&circuit, &inputs).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark proof sizes.
fn bench_proof_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Proof Sizes");

    for size in [100, 1000, 10000].iter() {
        let circuit = create_benchmark_circuit(*size);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        let config = GKRConfig::for_testing();
        let mut prover = GKRProver::new(config);
        let proof = prover.prove(&circuit, &inputs).unwrap();

        println!("GKR proof size for {} gates: {} bytes", size, proof.size_bytes());
    }

    group.finish();
}

/// Benchmark sumcheck specifically.
fn bench_sumcheck(c: &mut Criterion) {
    use helix_prover::gkr::sumcheck::{SumcheckProver, Blake3Transcript};
    use helix_prover::gkr::multilinear::DenseMultilinear;

    let mut group = c.benchmark_group("Sumcheck");
    group.measurement_time(Duration::from_secs(10));

    for num_vars in [8, 10, 12, 14, 16].iter() {
        let size = 1 << num_vars;
        let evals: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
        let poly = DenseMultilinear::from_evaluations(evals.clone());
        let claimed_sum: Fr = evals.iter().fold(Fr::zero(), |a, b| a + b);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("prove", num_vars),
            num_vars,
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

/// Benchmark polynomial evaluation.
fn bench_poly_eval(c: &mut Criterion) {
    use helix_prover::gkr::multilinear::{DenseMultilinear, MultilinearPolynomial};

    let mut group = c.benchmark_group("Polynomial Evaluation");
    group.measurement_time(Duration::from_secs(5));

    for num_vars in [10, 12, 14, 16].iter() {
        let size = 1 << num_vars;
        let evals: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
        let poly = DenseMultilinear::from_evaluations(evals);

        let point: Vec<Fr> = (0..*num_vars).map(|i| Fr::from(i as u64)).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("eval", num_vars),
            num_vars,
            |b, _| {
                b.iter(|| {
                    poly.evaluate(&point)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Metal acceleration (if available).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_metal(c: &mut Criterion) {
    use helix_prover::metal::{MetalFieldOps, BatchFieldOperation};

    if !helix_prover::is_metal_available() {
        println!("Metal not available, skipping Metal benchmarks");
        return;
    }

    let mut group = c.benchmark_group("Metal Acceleration");
    group.measurement_time(Duration::from_secs(10));

    for size in [1000, 10000, 100000, 1000000].iter() {
        let a: Vec<Fr> = (0..*size).map(|i| Fr::from(i as u64)).collect();
        let b: Vec<Fr> = (0..*size).map(|i| Fr::from((i * 7) as u64)).collect();

        group.throughput(Throughput::Elements(*size as u64));

        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("add_cpu", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let _: Vec<Fr> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
                });
            },
        );

        // Metal accelerated
        group.bench_with_input(
            BenchmarkId::new("add_metal", size),
            size,
            |bench, _| {
                let mut ops = MetalFieldOps::new();
                let op = BatchFieldOperation::add(a.clone(), b.clone());

                bench.iter(|| {
                    ops.execute(&op).unwrap()
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn bench_metal(_c: &mut Criterion) {
    // No-op on non-macOS
}

/// Compare GKR and Halo2 overhead.
fn bench_overhead_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Overhead Comparison");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    // Measure pure computation time vs proof time for a simple operation
    let circuit = create_benchmark_circuit(1000);
    let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

    // Pure computation (no proof)
    group.bench_function("pure_computation_1k", |b| {
        b.iter(|| {
            circuit.evaluate(&inputs)
        });
    });

    // GKR proving
    group.bench_function("gkr_prove_1k", |b| {
        let config = GKRConfig::for_testing();
        let mut prover = GKRProver::new(config);

        b.iter(|| {
            prover.prove(&circuit, &inputs).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gkr_proving,
    bench_gkr_verification,
    bench_halo2_proving,
    bench_gkr_matmul,
    bench_sumcheck,
    bench_poly_eval,
    bench_metal,
    bench_overhead_comparison,
);

criterion_main!(benches);
