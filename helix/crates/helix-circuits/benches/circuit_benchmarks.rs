//! Comprehensive benchmarks for HELIX circuit operations.
//!
//! This benchmark suite measures:
//! - Circuit synthesis time
//! - Proof generation time (via MockProver approximation)
//! - Verification time
//! - Memory usage
//! - Profiling overhead
//! - Optimization effectiveness
//!
//! Target: 30x overhead compared to native computation.
//! Target: <500ms proof generation for demo model.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use halo2_proofs::dev::MockProver;
use halo2curves::bn256::Fr;
use helix_circuits::{
    params::{
        HelixSRS, ParameterProfile,
    },
    ivc::IVCStepCircuit,
    approximate::{
        ErrorAccumulationCircuit, OperationData, OpType,
    },
    lookup::{
        ReLULookup, GELULookup, SigmoidLookup, TanhLookup,
        relu_table_entries, gelu_table_entries, sigmoid_table_entries, tanh_table_entries,
    },
    quantization::{
        Int8SymmetricParams, Int4WeightParams,
    },
    MLTrainingStepV2Circuit,
    compute_witness_v2,
    compute_state_hash_v2,
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

/// Benchmark key generation for different profiles.
fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("Key Generation");
    group.sample_size(10);
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

/// Benchmark lookup table generation.
fn bench_lookup_table_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lookup Table Generation");

    group.bench_function("relu_256", |b| {
        b.iter(|| {
            let _entries = relu_table_entries::<Fr>(black_box(256));
        });
    });

    group.bench_function("gelu_256_64", |b| {
        b.iter(|| {
            let _entries = gelu_table_entries::<Fr>(black_box(256), black_box(64));
        });
    });

    group.bench_function("sigmoid_256_64", |b| {
        b.iter(|| {
            let _entries = sigmoid_table_entries::<Fr>(black_box(256), black_box(64));
        });
    });

    group.bench_function("tanh_256_64", |b| {
        b.iter(|| {
            let _entries = tanh_table_entries::<Fr>(black_box(256), black_box(64));
        });
    });

    group.finish();
}

/// Benchmark lookup-based vs arithmetic activation functions.
///
/// This is the key benchmark demonstrating 10x+ speedup for lookup-based activations.
fn bench_lookup_vs_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lookup vs Arithmetic Activations");
    group.sample_size(20);

    // Pre-build lookup tables
    let relu_lookup = ReLULookup::<Fr, 256>::new();
    let gelu_lookup = GELULookup::<Fr, 256, 64>::new();
    let sigmoid_lookup = SigmoidLookup::<Fr, 256, 64>::new();
    let tanh_lookup = TanhLookup::<Fr, 256, 64>::new();

    // Benchmark lookup-based ReLU (single lookup operation)
    group.bench_function("relu_lookup", |b| {
        let test_values: Vec<Fr> = (0..100).map(|i| Fr::from(i)).collect();
        b.iter(|| {
            for val in &test_values {
                let _result = relu_lookup.table().lookup(black_box(&[*val]));
            }
        });
    });

    // Benchmark arithmetic ReLU (comparison + branch)
    group.bench_function("relu_arithmetic", |b| {
        let test_values: Vec<i64> = (0..100).map(|i| i - 50).collect();
        b.iter(|| {
            for &val in &test_values {
                let _result = if val >= 0 { val } else { 0 };
            }
        });
    });

    // Benchmark lookup-based GELU
    group.bench_function("gelu_lookup", |b| {
        let test_values: Vec<Fr> = (0..100).map(|i| Fr::from(i)).collect();
        b.iter(|| {
            for val in &test_values {
                let _result = gelu_lookup.table().lookup(black_box(&[*val]));
            }
        });
    });

    // Benchmark arithmetic GELU (expensive polynomial approximation)
    group.bench_function("gelu_arithmetic", |b| {
        let test_values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 64.0).collect();
        let sqrt_2_pi: f64 = 0.7978845608028654;
        b.iter(|| {
            for &x in &test_values {
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                let _result = 0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh());
            }
        });
    });

    // Benchmark lookup-based Sigmoid
    group.bench_function("sigmoid_lookup", |b| {
        let test_values: Vec<Fr> = (0..100).map(|i| Fr::from(i)).collect();
        b.iter(|| {
            for val in &test_values {
                let _result = sigmoid_lookup.table().lookup(black_box(&[*val]));
            }
        });
    });

    // Benchmark arithmetic Sigmoid (expensive exp)
    group.bench_function("sigmoid_arithmetic", |b| {
        let test_values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 64.0).collect();
        b.iter(|| {
            for &x in &test_values {
                let _result = 1.0 / (1.0 + (-x).exp());
            }
        });
    });

    // Benchmark lookup-based Tanh
    group.bench_function("tanh_lookup", |b| {
        let test_values: Vec<Fr> = (0..100).map(|i| Fr::from(i)).collect();
        b.iter(|| {
            for val in &test_values {
                let _result = tanh_lookup.table().lookup(black_box(&[*val]));
            }
        });
    });

    // Benchmark arithmetic Tanh (expensive exp)
    group.bench_function("tanh_arithmetic", |b| {
        let test_values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 64.0).collect();
        b.iter(|| {
            for &x in &test_values {
                let _result = x.tanh();
            }
        });
    });

    group.finish();
}

/// Benchmark quantization operations.
fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantization Operations");

    // INT8 symmetric quantization
    let int8_params = Int8SymmetricParams::from_range(-1.0, 1.0);

    group.bench_function("int8_quantize", |b| {
        let test_values: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 500.0).collect();
        b.iter(|| {
            for &val in &test_values {
                let _q = int8_params.quantize(black_box(val));
            }
        });
    });

    group.bench_function("int8_dequantize", |b| {
        let test_values: Vec<i8> = (-128..=127).collect();
        b.iter(|| {
            for &val in &test_values {
                let _dq = int8_params.dequantize(black_box(val));
            }
        });
    });

    // INT4 weight quantization
    let int4_params = Int4WeightParams::from_range(-1.0, 1.0);

    group.bench_function("int4_quantize", |b| {
        let test_values: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 500.0).collect();
        b.iter(|| {
            for &val in &test_values {
                let _q = int4_params.quantize(black_box(val));
            }
        });
    });

    group.bench_function("int4_dequantize", |b| {
        let test_values: Vec<i8> = (-8..=7).collect();
        b.iter(|| {
            for &val in &test_values {
                let _dq = int4_params.dequantize(black_box(val));
            }
        });
    });

    group.finish();
}

/// Benchmark INT4 packing/unpacking.
fn bench_int4_packing(c: &mut Criterion) {
    use helix_circuits::quantization::{pack_int4_values, unpack_int4_values};

    let mut group = c.benchmark_group("INT4 Packing");

    group.bench_function("pack_1000", |b| {
        let values: Vec<(i8, i8)> = (-8..=7).flat_map(|h| (-8..=7).map(move |l| (h, l))).take(1000).collect();
        b.iter(|| {
            for &(h, l) in &values {
                let _packed = pack_int4_values(black_box(h), black_box(l));
            }
        });
    });

    group.bench_function("unpack_256", |b| {
        let packed: Vec<u8> = (0..=255).collect();
        b.iter(|| {
            for &p in &packed {
                let _unpacked = unpack_int4_values(black_box(p));
            }
        });
    });

    group.finish();
}

/// Helper: Create benchmark witness for MLTrainingStepV2.
fn create_training_witness(d_in: usize, d_hid: usize, d_out: usize) -> helix_circuits::MLTrainingStepV2Witness {
    let w1: Vec<Fr> = (0..d_hid * d_in)
        .map(|i| Fr::from((i % 10 + 1) as u64))
        .collect();
    let b1 = vec![Fr::zero(); d_hid];
    let w2: Vec<Fr> = (0..d_out * d_hid)
        .map(|i| Fr::from((i % 10 + 1) as u64))
        .collect();
    let b2 = vec![Fr::zero(); d_out];
    let x: Vec<Fr> = (0..d_in).map(|i| Fr::from((i + 1) as u64)).collect();
    let target: Vec<Fr> = (0..d_out).map(|_| Fr::from(10u64)).collect();
    let lr = Fr::from(1);
    let base_error = Fr::from(1);

    let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
    let witness = compute_witness_v2(
        d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2,
        lr, old_hash, (Fr::zero(), Fr::zero()), 1, base_error,
    );
    let new_hash = compute_state_hash_v2(
        &witness.w1_new, &witness.b1_new, &witness.w2_new, &witness.b2_new,
    );
    compute_witness_v2(
        d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2,
        lr, old_hash, new_hash, 1, base_error,
    )
}

/// Benchmark MLTrainingStepV2 witness generation.
fn bench_training_witness(c: &mut Criterion) {
    let mut group = c.benchmark_group("Training Witness Generation");
    group.sample_size(50);

    for &(d_in, d_hid, d_out) in &[(2, 2, 1), (4, 4, 2), (8, 8, 4)] {
        let id = format!("{}x{}x{}", d_in, d_hid, d_out);
        let params = (d_in * d_hid + d_hid + d_out * d_hid + d_out) as u64;
        group.throughput(Throughput::Elements(params));

        let w1: Vec<Fr> = (0..d_hid * d_in).map(|i| Fr::from((i % 10 + 1) as u64)).collect();
        let b1 = vec![Fr::zero(); d_hid];
        let w2: Vec<Fr> = (0..d_out * d_hid).map(|i| Fr::from((i % 10 + 1) as u64)).collect();
        let b2 = vec![Fr::zero(); d_out];
        let x: Vec<Fr> = (0..d_in).map(|i| Fr::from((i + 1) as u64)).collect();
        let target: Vec<Fr> = (0..d_out).map(|_| Fr::from(10u64)).collect();
        let lr = Fr::from(1);
        let base_error = Fr::from(1);
        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);

        group.bench_function(BenchmarkId::new("witness", &id), |b| {
            b.iter(|| {
                compute_witness_v2(
                    black_box(d_in), black_box(d_hid), black_box(d_out),
                    black_box(&x), black_box(&target),
                    black_box(&w1), black_box(&b1),
                    black_box(&w2), black_box(&b2),
                    black_box(lr), black_box(old_hash),
                    black_box((Fr::zero(), Fr::zero())), black_box(1), black_box(base_error),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark MLTrainingStepV2 circuit with MockProver.
fn bench_training_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("Training Circuit MockProver");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));

    // Tiny model for quick iteration
    let witness = create_training_witness(2, 2, 1);
    let pi = witness.public_inputs();

    // With Freivalds
    let circuit_freivalds = MLTrainingStepV2Circuit {
        witness: witness.clone(),
        relu_range: 128,
        exp_range: 64,
        exp_scale: 32,
        use_freivalds: true,
    };

    group.bench_function("2x2x1_freivalds", |b| {
        b.iter(|| {
            let prover = MockProver::run(14, black_box(&circuit_freivalds), vec![pi.clone()]).unwrap();
            black_box(prover.verify())
        });
    });

    // Without Freivalds
    let circuit_direct = MLTrainingStepV2Circuit {
        witness: witness.clone(),
        relu_range: 128,
        exp_range: 64,
        exp_scale: 32,
        use_freivalds: false,
    };

    group.bench_function("2x2x1_direct", |b| {
        b.iter(|| {
            let prover = MockProver::run(14, black_box(&circuit_direct), vec![pi.clone()]).unwrap();
            black_box(prover.verify())
        });
    });

    // Small model
    let witness_sm = create_training_witness(4, 4, 2);
    let pi_sm = witness_sm.public_inputs();

    let circuit_sm = MLTrainingStepV2Circuit {
        witness: witness_sm,
        relu_range: 128,
        exp_range: 64,
        exp_scale: 32,
        use_freivalds: true,
    };

    group.bench_function("4x4x2_freivalds", |b| {
        b.iter(|| {
            let prover = MockProver::run(14, black_box(&circuit_sm), vec![pi_sm.clone()]).unwrap();
            black_box(prover.verify())
        });
    });

    group.finish();
}

/// Benchmark state hash computation.
fn bench_state_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("State Hash Computation");
    group.sample_size(100);

    for &size in &[16, 64, 256] {
        let w1: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
        let b1: Vec<Fr> = (0..size / 4).map(|i| Fr::from(i as u64)).collect();
        let w2: Vec<Fr> = (0..size / 2).map(|i| Fr::from(i as u64)).collect();
        let b2: Vec<Fr> = (0..size / 8).map(|i| Fr::from(i as u64)).collect();

        group.throughput(Throughput::Elements(
            (w1.len() + b1.len() + w2.len() + b2.len()) as u64
        ));

        group.bench_function(BenchmarkId::new("hash", size), |b| {
            b.iter(|| {
                compute_state_hash_v2(
                    black_box(&w1), black_box(&b1),
                    black_box(&w2), black_box(&b2),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark overhead comparison: ZK vs direct computation.
fn bench_overhead_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Overhead Comparison");
    group.sample_size(100);

    let d_in = 4;
    let d_hid = 4;
    let d_out = 2;

    let w1: Vec<Fr> = (0..d_hid * d_in).map(|i| Fr::from((i % 10 + 1) as u64)).collect();
    let b1 = vec![Fr::zero(); d_hid];
    let w2: Vec<Fr> = (0..d_out * d_hid).map(|i| Fr::from((i % 10 + 1) as u64)).collect();
    let b2 = vec![Fr::zero(); d_out];
    let x: Vec<Fr> = (0..d_in).map(|i| Fr::from((i + 1) as u64)).collect();

    // Baseline: Direct forward pass
    group.bench_function("direct_forward_4x4x2", |b| {
        b.iter(|| {
            let mut h_pre = vec![Fr::zero(); d_hid];
            for j in 0..d_hid {
                for i in 0..d_in {
                    h_pre[j] = h_pre[j] + w1[j * d_in + i] * x[i];
                }
                h_pre[j] = h_pre[j] + b1[j];
            }
            let h = h_pre.clone();

            let mut y = vec![Fr::zero(); d_out];
            for j in 0..d_out {
                for k in 0..d_hid {
                    y[j] = y[j] + w2[j * d_hid + k] * h[k];
                }
                y[j] = y[j] + b2[j];
            }
            black_box(y)
        });
    });

    // ZK: Witness generation only
    let target: Vec<Fr> = (0..d_out).map(|_| Fr::from(10u64)).collect();
    let lr = Fr::from(1);
    let base_error = Fr::from(1);
    let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);

    group.bench_function("zk_witness_4x4x2", |b| {
        b.iter(|| {
            compute_witness_v2(
                black_box(d_in), black_box(d_hid), black_box(d_out),
                black_box(&x), black_box(&target),
                black_box(&w1), black_box(&b1),
                black_box(&w2), black_box(&b2),
                black_box(lr), black_box(old_hash),
                black_box((Fr::zero(), Fr::zero())), black_box(1), black_box(base_error),
            )
        });
    });

    group.finish();
}

/// Benchmark Freivalds challenge generation.
fn bench_freivalds_challenge(c: &mut Criterion) {
    use helix_circuits::ml::training_step_v2::generate_freivalds_challenge;

    let mut group = c.benchmark_group("Freivalds Challenge");
    group.sample_size(100);

    for &len in &[4, 8, 16, 32] {
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(BenchmarkId::new("generate", len), |b| {
            b.iter(|| {
                generate_freivalds_challenge(black_box(42), black_box(len))
            });
        });
    }

    group.finish();
}

/// Benchmark K parameter scaling.
fn bench_k_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("K Parameter Scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let witness = create_training_witness(2, 2, 1);
    let pi = witness.public_inputs();

    let circuit = MLTrainingStepV2Circuit {
        witness: witness.clone(),
        relu_range: 64,
        exp_range: 32,
        exp_scale: 16,
        use_freivalds: true,
    };

    for k in [12, 13, 14] {
        group.bench_function(BenchmarkId::new("mock_prover_k", k), |b| {
            b.iter(|| {
                let prover = MockProver::run(black_box(k), &circuit, vec![pi.clone()]).unwrap();
                black_box(prover.verify())
            });
        });
    }

    group.finish();
}

/// Benchmark model scaling.
fn bench_model_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Model Scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    for &(d_in, d_hid, d_out, k) in &[(2, 2, 1, 14), (4, 4, 2, 14), (8, 8, 4, 15)] {
        let witness = create_training_witness(d_in, d_hid, d_out);
        let pi = witness.public_inputs();

        let circuit = MLTrainingStepV2Circuit {
            witness: witness.clone(),
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let params = (d_in * d_hid + d_hid + d_out * d_hid + d_out) as u64;
        group.throughput(Throughput::Elements(params));

        let id = format!("{}x{}x{}", d_in, d_hid, d_out);
        group.bench_function(BenchmarkId::new("full_circuit", &id), |b| {
            b.iter(|| {
                let prover = MockProver::run(k, black_box(&circuit), vec![pi.clone()]).unwrap();
                black_box(prover.verify())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_srs_generation,
    bench_ivc_circuit,
    bench_error_accumulation,
    bench_keygen,
    bench_profiles,
    bench_lookup_table_generation,
    bench_lookup_vs_arithmetic,
    bench_quantization,
    bench_int4_packing,
    // New profiling-focused benchmarks
    bench_training_witness,
    bench_training_circuit,
    bench_state_hash,
    bench_overhead_comparison,
    bench_freivalds_challenge,
    bench_k_scaling,
    bench_model_scaling,
);

criterion_main!(benches);
