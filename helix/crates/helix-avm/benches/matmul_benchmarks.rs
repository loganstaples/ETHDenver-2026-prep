//! Benchmarks for matrix multiplication comparing BLAS vs naive implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use helix_core::types::{BoundedTensor, Precision};

fn create_random_tensor(rows: usize, cols: usize) -> BoundedTensor {
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i as f64 * 0.7 + 0.3) % 2.0) - 1.0)
        .collect();
    BoundedTensor::from_exact(data, vec![rows, cols])
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for &size in &[16, 32, 64, 128, 256] {
        let a = create_random_tensor(size, size);
        let b = create_random_tensor(size, size);

        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    helix_avm::ops::matmul::matmul(
                        black_box(&a),
                        black_box(&b),
                        black_box(Precision::F32),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
