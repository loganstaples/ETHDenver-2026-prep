//! GPU Acceleration Benchmarks
//!
//! Compares CPU vs GPU performance for MSM and NTT operations.
//! Target: GPU should be 10x+ faster than CPU for large operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use std::time::Duration;

// Import the crate for GPU comparison benchmarks
#[allow(unused_imports)]
use helix_prover;

// ============================================================================
// BN254 Field Constants & Arithmetic (Self-contained for benchmarking)
// ============================================================================

const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

const R: [u64; 4] = [
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
];

const INV: u64 = 0xc2e1f593efffffff;

fn less_than(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

fn field_add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut carry = 0u128;

    for i in 0..4 {
        let sum = (a[i] as u128) + (b[i] as u128) + carry;
        result[i] = sum as u64;
        carry = sum >> 64;
    }

    if carry != 0 || !less_than(&result, &MODULUS) {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (MODULUS[i] as i128) - borrow;
            if diff < 0 {
                result[i] = (diff + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                result[i] = diff as u64;
                borrow = 0;
            }
        }
    }

    result
}

fn field_sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0i128;

    for i in 0..4 {
        let diff = (a[i] as i128) - (b[i] as i128) - borrow;
        if diff < 0 {
            result[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            result[i] = diff as u64;
            borrow = 0;
        }
    }

    if borrow != 0 {
        let mut carry = 0u128;
        for i in 0..4 {
            let sum = (result[i] as u128) + (MODULUS[i] as u128) + carry;
            result[i] = sum as u64;
            carry = sum >> 64;
        }
    }

    result
}

fn field_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut t = [0u64; 8];

    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let product = (a[i] as u128) * (b[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = product as u64;
            carry = product >> 64;
        }
        t[i + 4] = carry as u64;
    }

    for i in 0..4 {
        let m = t[i].wrapping_mul(INV);
        let mut carry = 0u128;

        for j in 0..4 {
            let product = (m as u128) * (MODULUS[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = product as u64;
            carry = product >> 64;
        }

        for j in (i + 4)..8 {
            let sum = (t[j] as u128) + carry;
            t[j] = sum as u64;
            carry = sum >> 64;
            if carry == 0 { break; }
        }
    }

    let mut result = [t[4], t[5], t[6], t[7]];

    if !less_than(&result, &MODULUS) {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (MODULUS[i] as i128) - borrow;
            if diff < 0 {
                result[i] = (diff + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                result[i] = diff as u64;
                borrow = 0;
            }
        }
    }

    result
}

// ============================================================================
// NTT Implementation for Benchmarking
// ============================================================================

const ROOT_OF_UNITY: [u64; 4] = [
    0x3e7e3d9e8b8f1f63,
    0x7ccc637db1fc1cd9,
    0x5f4d5e06718a5d10,
    0x19c6dfb841f39d00,
];

const MAX_LOG_N: usize = 28;

fn field_pow(base: &[u64; 4], exp: &[u64; 4]) -> [u64; 4] {
    let mut result = R;
    let mut base = *base;

    for limb_idx in 0..4 {
        let mut limb = exp[limb_idx];
        for _ in 0..64 {
            if limb & 1 == 1 {
                result = field_mul(&result, &base);
            }
            base = field_mul(&base, &base);
            limb >>= 1;
        }
    }

    result
}

fn field_inv(a: &[u64; 4]) -> [u64; 4] {
    let exp = [
        MODULUS[0].wrapping_sub(2),
        MODULUS[1],
        MODULUS[2],
        MODULUS[3],
    ];
    field_pow(a, &exp)
}

fn get_root_of_unity(log_n: usize) -> [u64; 4] {
    let mut root = ROOT_OF_UNITY;
    for _ in log_n..MAX_LOG_N {
        root = field_mul(&root, &root);
    }
    root
}

fn reverse_bits(x: usize, bits: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn bit_reverse_permutation(data: &mut [[u64; 4]]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;

    for i in 0..n {
        let rev = reverse_bits(i, log_n);
        if i < rev {
            data.swap(i, rev);
        }
    }
}

fn forward_ntt(data: &mut [[u64; 4]]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    bit_reverse_permutation(data);

    let omega_n = get_root_of_unity(log_n);

    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = m / 2;

        let mut omega_m = omega_n;
        for _ in 0..(log_n - stage - 1) {
            omega_m = field_mul(&omega_m, &omega_m);
        }

        for k in (0..n).step_by(m) {
            let mut omega = R;

            for j in 0..half_m {
                let u = data[k + j];
                let t = field_mul(&omega, &data[k + j + half_m]);

                data[k + j] = field_add(&u, &t);
                data[k + j + half_m] = field_sub(&u, &t);

                omega = field_mul(&omega, &omega_m);
            }
        }
    }
}

fn inverse_ntt(data: &mut [[u64; 4]]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    bit_reverse_permutation(data);

    let omega_n = field_inv(&get_root_of_unity(log_n));

    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = m / 2;

        let mut omega_m = omega_n;
        for _ in 0..(log_n - stage - 1) {
            omega_m = field_mul(&omega_m, &omega_m);
        }

        for k in (0..n).step_by(m) {
            let mut omega = R;

            for j in 0..half_m {
                let u = data[k + j];
                let t = field_mul(&omega, &data[k + j + half_m]);

                data[k + j] = field_add(&u, &t);
                data[k + j + half_m] = field_sub(&u, &t);

                omega = field_mul(&omega, &omega_m);
            }
        }
    }

    // Scale by n^{-1}
    let n_mont = to_montgomery(&[n as u64, 0, 0, 0]);
    let n_inv = field_inv(&n_mont);
    for elem in data.iter_mut() {
        *elem = field_mul(elem, &n_inv);
    }
}

fn to_montgomery(a: &[u64; 4]) -> [u64; 4] {
    const R2: [u64; 4] = [
        0xf32cfc5b538afa89,
        0xb5e71911d44501fb,
        0x47ab1eff0a417ff6,
        0x06d89f71cab8351f,
    ];
    field_mul(a, &R2)
}

// ============================================================================
// Helper Functions
// ============================================================================

fn random_field_element() -> [u64; 4] {
    let mut rng = rand::thread_rng();
    [
        rng.gen::<u64>(),
        rng.gen::<u64>(),
        rng.gen::<u64>(),
        rng.gen::<u64>() & 0x3FFFFFFFFFFFFFFF,
    ]
}

// ============================================================================
// NTT Benchmarks
// ============================================================================

fn bench_ntt_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT CPU");
    group.measurement_time(Duration::from_secs(10));

    for log_n in [10, 12, 14, 16].iter() {
        let n = 1 << log_n;
        group.throughput(Throughput::Elements(n as u64));

        let data: Vec<[u64; 4]> = (0..n).map(|_| random_field_element()).collect();

        group.bench_with_input(BenchmarkId::new("forward", log_n), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut working_data| {
                    forward_ntt(&mut working_data);
                    working_data
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("inverse", log_n), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut working_data| {
                    inverse_ntt(&mut working_data);
                    working_data
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_ntt_field_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT Field Operations");

    let a = random_field_element();
    let b = random_field_element();

    group.bench_function("field_add", |bench| {
        bench.iter(|| {
            black_box(field_add(black_box(&a), black_box(&b)))
        });
    });

    group.bench_function("field_sub", |bench| {
        bench.iter(|| {
            black_box(field_sub(black_box(&a), black_box(&b)))
        });
    });

    group.bench_function("field_mul", |bench| {
        bench.iter(|| {
            black_box(field_mul(black_box(&a), black_box(&b)))
        });
    });

    group.finish();
}

// ============================================================================
// GPU vs CPU Comparison Benchmarks (B2 Requirement)
// ============================================================================

/// MSM performance comparison: CPU vs GPU
/// Target: GPU should be 10x+ faster for large operations
fn bench_msm_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("MSM GPU vs CPU");
    group.measurement_time(Duration::from_secs(15));

    // Test various sizes to show scaling
    for log_n in [10, 12, 14].iter() {
        let n = 1 << log_n;
        group.throughput(Throughput::Elements(n as u64));

        // Generate random points and scalars
        let points: Vec<[u64; 8]> = (0..n)
            .map(|_| {
                let mut p = [0u64; 8];
                for i in 0..8 {
                    p[i] = random_field_element()[i % 4];
                }
                p
            })
            .collect();

        let scalars: Vec<[u64; 4]> = (0..n)
            .map(|_| random_field_element())
            .collect();

        // CPU MSM benchmark
        group.bench_with_input(BenchmarkId::new("cpu", log_n), &n, |b, _| {
            b.iter(|| {
                // Use the CPU MSM implementation
                let result = helix_prover::cuda::msm_cpu::compute_msm(&points, &scalars, 16);
                black_box(result)
            });
        });

        // GPU MSM benchmark (if CUDA available)
        #[cfg(feature = "cuda")]
        {
            if helix_prover::cuda::is_cuda_available() {
                group.bench_with_input(BenchmarkId::new("gpu_cuda", log_n), &n, |b, _| {
                    b.iter(|| {
                        let result = unsafe {
                            helix_prover::cuda::bindings::cuda_msm_pippenger(
                                points.as_ptr() as *const u64,
                                scalars.as_ptr() as *const u64,
                                n,
                                16, // window size
                            )
                        };
                        black_box(result)
                    });
                });
            }
        }
    }

    group.finish();
}

/// NTT performance comparison: CPU vs GPU
/// Target: GPU should be 10x+ faster for large transforms
fn bench_ntt_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT GPU vs CPU");
    group.measurement_time(Duration::from_secs(15));

    for log_n in [12, 14, 16].iter() {
        let n = 1 << log_n;
        group.throughput(Throughput::Elements(n as u64));

        let data: Vec<[u64; 4]> = (0..n).map(|_| random_field_element()).collect();

        // CPU NTT benchmark
        group.bench_with_input(BenchmarkId::new("cpu_forward", log_n), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut working_data| {
                    forward_ntt(&mut working_data);
                    working_data
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("cpu_inverse", log_n), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut working_data| {
                    inverse_ntt(&mut working_data);
                    working_data
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // GPU NTT benchmark (if CUDA available)
        #[cfg(feature = "cuda")]
        {
            if helix_prover::cuda::is_cuda_available() {
                group.bench_with_input(BenchmarkId::new("gpu_cuda_forward", log_n), &data, |b, data| {
                    b.iter_batched(
                        || {
                            let mut flat: Vec<u64> = Vec::with_capacity(n * 4);
                            for elem in data.iter() {
                                flat.extend_from_slice(elem);
                            }
                            flat
                        },
                        |mut flat_data| {
                            unsafe {
                                let _ = helix_prover::cuda::bindings::cuda_ntt_forward(
                                    flat_data.as_mut_ptr(),
                                    n,
                                );
                            }
                            flat_data
                        },
                        criterion::BatchSize::SmallInput,
                    );
                });
            }
        }
    }

    group.finish();
}

/// Field operations batch comparison
fn bench_field_ops_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Field Ops GPU vs CPU");
    group.measurement_time(Duration::from_secs(10));

    let batch_size = 100000;
    let a_batch: Vec<[u64; 4]> = (0..batch_size).map(|_| random_field_element()).collect();
    let b_batch: Vec<[u64; 4]> = (0..batch_size).map(|_| random_field_element()).collect();

    group.throughput(Throughput::Elements(batch_size as u64));

    // CPU batch multiplication
    group.bench_function("cpu_batch_mul", |bench| {
        bench.iter(|| {
            let results: Vec<_> = a_batch
                .iter()
                .zip(b_batch.iter())
                .map(|(a, b)| field_mul(a, b))
                .collect();
            black_box(results)
        });
    });

    // GPU batch multiplication (if available)
    #[cfg(feature = "cuda")]
    {
        if helix_prover::cuda::is_cuda_available() {
            group.bench_function("gpu_batch_mul", |bench| {
                bench.iter(|| {
                    let mut result = vec![[0u64; 4]; batch_size];
                    unsafe {
                        let _ = helix_prover::cuda::bindings::cuda_field_mul(
                            a_batch.as_ptr() as *const u64,
                            b_batch.as_ptr() as *const u64,
                            result.as_mut_ptr() as *mut u64,
                            batch_size,
                        );
                    }
                    black_box(result)
                });
            });
        }
    }

    group.finish();
}

fn bench_cpu_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU vs Parallel");
    group.measurement_time(Duration::from_secs(10));

    let n = 1 << 14; // 16K elements
    let data: Vec<[u64; 4]> = (0..n).map(|_| random_field_element()).collect();

    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("ntt_sequential", |b| {
        b.iter_batched(
            || data.clone(),
            |mut working_data| {
                forward_ntt(&mut working_data);
                working_data
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ============================================================================
// Throughput Analysis
// ============================================================================

fn bench_field_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("Field Throughput");

    let batch_size = 10000;
    let a_batch: Vec<[u64; 4]> = (0..batch_size).map(|_| random_field_element()).collect();
    let b_batch: Vec<[u64; 4]> = (0..batch_size).map(|_| random_field_element()).collect();

    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("batch_mul", |bench| {
        bench.iter(|| {
            let results: Vec<_> = a_batch
                .iter()
                .zip(b_batch.iter())
                .map(|(a, b)| field_mul(a, b))
                .collect();
            black_box(results)
        });
    });

    group.bench_function("batch_add", |bench| {
        bench.iter(|| {
            let results: Vec<_> = a_batch
                .iter()
                .zip(b_batch.iter())
                .map(|(a, b)| field_add(a, b))
                .collect();
            black_box(results)
        });
    });

    group.finish();
}

// ============================================================================
// Main Benchmark Entry
// ============================================================================

criterion_group!(
    name = ntt_benches;
    config = Criterion::default().sample_size(50);
    targets = bench_ntt_cpu, bench_ntt_field_ops
);

criterion_group!(
    name = throughput_benches;
    config = Criterion::default();
    targets = bench_cpu_vs_parallel, bench_field_throughput
);

// GPU vs CPU comparison benchmarks (B2 requirement: 10x+ speedup target)
criterion_group!(
    name = gpu_comparison_benches;
    config = Criterion::default().sample_size(30);
    targets = bench_msm_gpu_vs_cpu, bench_ntt_gpu_vs_cpu, bench_field_ops_gpu_vs_cpu
);

criterion_main!(ntt_benches, throughput_benches, gpu_comparison_benches);
