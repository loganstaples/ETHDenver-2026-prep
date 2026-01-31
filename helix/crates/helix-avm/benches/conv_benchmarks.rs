//! Benchmarks for convolution operations.
//!
//! These benchmarks measure the performance of conv operations with error tracking
//! and compare different configurations.
//!
//! Run with: cargo bench --bench conv_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use helix_avm::ops::{
    conv1d, conv2d, conv2d_with_config, depthwise_conv2d, group_conv2d,
    max_pool2d, avg_pool2d, global_avg_pool2d,
    Conv2dConfig, Pool2dConfig,
};
use helix_core::types::{BoundedTensor, Precision};

/// Creates a random tensor with given shape.
fn random_tensor(shape: Vec<usize>) -> BoundedTensor {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let len: usize = shape.iter().product();
    let data: Vec<f64> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
    BoundedTensor::from_exact(data, shape)
}

/// Creates a random tensor with error bounds.
fn random_tensor_with_error(shape: Vec<usize>, error: f64) -> BoundedTensor {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let len: usize = shape.iter().product();
    let data: Vec<f64> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
    BoundedTensor::from_approximate(data, shape, error)
}

// ============================================================================
// CONV2D BENCHMARKS
// ============================================================================

fn bench_conv2d_basic(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_basic");

    // Benchmark different spatial sizes
    for size in [8, 16, 32, 64].iter() {
        let input = random_tensor(vec![1, 3, *size, *size]);
        let kernel = random_tensor(vec![16, 3, 3, 3]);

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    conv2d(
                        black_box(&input),
                        black_box(&kernel),
                        (1, 1),
                        (1, 1),
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_conv2d_channels(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_channels");

    // Benchmark different channel counts
    for channels in [16, 32, 64, 128].iter() {
        let input = random_tensor(vec![1, *channels, 16, 16]);
        let kernel = random_tensor(vec![*channels, *channels, 3, 3]);

        group.throughput(Throughput::Elements((*channels * 16 * 16) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}ch", channels)),
            channels,
            |b, _| {
                b.iter(|| {
                    conv2d(
                        black_box(&input),
                        black_box(&kernel),
                        (1, 1),
                        (1, 1),
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_conv2d_kernel_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_kernel_sizes");

    let input = random_tensor(vec![1, 32, 32, 32]);

    for kernel_size in [1, 3, 5, 7].iter() {
        let kernel = random_tensor(vec![32, 32, *kernel_size, *kernel_size]);
        let padding = (kernel_size / 2, kernel_size / 2);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", kernel_size, kernel_size)),
            kernel_size,
            |b, _| {
                b.iter(|| {
                    conv2d(
                        black_box(&input),
                        black_box(&kernel),
                        (1, 1),
                        padding,
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_conv2d_strides(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_strides");

    let input = random_tensor(vec![1, 32, 64, 64]);
    let kernel = random_tensor(vec![64, 32, 3, 3]);

    for stride in [1, 2, 4].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("stride_{}", stride)),
            stride,
            |b, stride| {
                b.iter(|| {
                    conv2d(
                        black_box(&input),
                        black_box(&kernel),
                        (*stride, *stride),
                        (1, 1),
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_conv2d_dilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_dilation");

    let input = random_tensor(vec![1, 32, 32, 32]);
    let kernel = random_tensor(vec![32, 32, 3, 3]);

    for dilation in [1, 2, 4].iter() {
        let config = Conv2dConfig::new()
            .stride((1, 1))
            .padding((*dilation, *dilation))
            .dilation((*dilation, *dilation));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("dilation_{}", dilation)),
            dilation,
            |b, _| {
                b.iter(|| {
                    conv2d_with_config(
                        black_box(&input),
                        black_box(&kernel),
                        config,
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// CONV1D BENCHMARKS
// ============================================================================

fn bench_conv1d_basic(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv1d_basic");

    for length in [64, 128, 256, 512].iter() {
        let input = random_tensor(vec![1, 32, *length]);
        let kernel = random_tensor(vec![64, 32, 3]);

        group.throughput(Throughput::Elements(*length as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("len_{}", length)),
            length,
            |b, _| {
                b.iter(|| {
                    conv1d(
                        black_box(&input),
                        black_box(&kernel),
                        1,
                        1,
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// SPECIALIZED CONVOLUTION BENCHMARKS
// ============================================================================

fn bench_depthwise_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("depthwise_conv2d");

    for channels in [32, 64, 128, 256].iter() {
        let input = random_tensor(vec![1, *channels, 32, 32]);
        let kernel = random_tensor(vec![*channels, 1, 3, 3]);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}ch", channels)),
            channels,
            |b, _| {
                b.iter(|| {
                    depthwise_conv2d(
                        black_box(&input),
                        black_box(&kernel),
                        (1, 1),
                        (1, 1),
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_group_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_conv2d");

    let input = random_tensor(vec![1, 64, 32, 32]);

    // groups must divide both in_channels and out_channels
    for groups in [1, 2, 4, 8].iter() {
        let kernel = random_tensor(vec![64, 64 / *groups, 3, 3]);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_groups", groups)),
            groups,
            |b, groups| {
                b.iter(|| {
                    group_conv2d(
                        black_box(&input),
                        black_box(&kernel),
                        *groups,
                        (1, 1),
                        (1, 1),
                        Precision::F32,
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// POOLING BENCHMARKS
// ============================================================================

fn bench_max_pool2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_pool2d");

    for size in [16, 32, 64, 128].iter() {
        let input = random_tensor(vec![1, 64, *size, *size]);
        let config = Pool2dConfig::new((2, 2));

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    max_pool2d(black_box(&input), config)
                })
            },
        );
    }

    group.finish();
}

fn bench_avg_pool2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("avg_pool2d");

    for size in [16, 32, 64, 128].iter() {
        let input = random_tensor(vec![1, 64, *size, *size]);
        let config = Pool2dConfig::new((2, 2));

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    avg_pool2d(black_box(&input), config, Precision::F32)
                })
            },
        );
    }

    group.finish();
}

fn bench_global_avg_pool2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("global_avg_pool2d");

    for size in [7, 14, 28, 56].iter() {
        let input = random_tensor(vec![1, 512, *size, *size]);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    global_avg_pool2d(black_box(&input), Precision::F32)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// ERROR PROPAGATION OVERHEAD BENCHMARKS
// ============================================================================

fn bench_error_propagation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_propagation_overhead");

    // Compare exact values vs values with error bounds
    let input_exact = random_tensor(vec![1, 32, 32, 32]);
    let kernel_exact = random_tensor(vec![64, 32, 3, 3]);

    let input_approx = random_tensor_with_error(vec![1, 32, 32, 32], 1e-6);
    let kernel_approx = random_tensor_with_error(vec![64, 32, 3, 3], 1e-6);

    group.bench_function("exact_values", |b| {
        b.iter(|| {
            conv2d(
                black_box(&input_exact),
                black_box(&kernel_exact),
                (1, 1),
                (1, 1),
                Precision::F32,
            )
        })
    });

    group.bench_function("with_error_bounds", |b| {
        b.iter(|| {
            conv2d(
                black_box(&input_approx),
                black_box(&kernel_approx),
                (1, 1),
                (1, 1),
                Precision::F32,
            )
        })
    });

    group.finish();
}

// ============================================================================
// PRECISION COMPARISON BENCHMARKS
// ============================================================================

fn bench_precision_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_comparison");

    let input = random_tensor(vec![1, 32, 32, 32]);
    let kernel = random_tensor(vec![64, 32, 3, 3]);

    for precision in [Precision::F32, Precision::F16, Precision::INT8].iter() {
        let name = match precision {
            Precision::F32 => "F32",
            Precision::F16 => "F16",
            Precision::INT8 => "INT8",
            _ => "Other",
        };

        group.bench_function(name, |b| {
            b.iter(|| {
                conv2d(
                    black_box(&input),
                    black_box(&kernel),
                    (1, 1),
                    (1, 1),
                    *precision,
                )
            })
        });
    }

    group.finish();
}

// ============================================================================
// REALISTIC WORKLOAD BENCHMARKS
// ============================================================================

fn bench_resnet_block(c: &mut Criterion) {
    // Simulate a ResNet-like bottleneck block
    // 1x1 conv -> 3x3 conv -> 1x1 conv
    let mut group = c.benchmark_group("resnet_bottleneck");

    let input = random_tensor(vec![1, 256, 14, 14]);
    let conv1 = random_tensor(vec![64, 256, 1, 1]);
    let conv2 = random_tensor(vec![64, 64, 3, 3]);
    let conv3 = random_tensor(vec![256, 64, 1, 1]);

    group.bench_function("bottleneck_block", |b| {
        b.iter(|| {
            let out1 = conv2d(
                black_box(&input),
                black_box(&conv1),
                (1, 1),
                (0, 0),
                Precision::F32,
            ).unwrap();

            let out2 = conv2d(
                black_box(&out1),
                black_box(&conv2),
                (1, 1),
                (1, 1),
                Precision::F32,
            ).unwrap();

            conv2d(
                black_box(&out2),
                black_box(&conv3),
                (1, 1),
                (0, 0),
                Precision::F32,
            )
        })
    });

    group.finish();
}

fn bench_mobilenet_block(c: &mut Criterion) {
    // Simulate a MobileNet-like depthwise separable block
    let mut group = c.benchmark_group("mobilenet_block");

    let input = random_tensor(vec![1, 64, 28, 28]);
    let depthwise_kernel = random_tensor(vec![64, 1, 3, 3]);
    let pointwise_kernel = random_tensor(vec![128, 64, 1, 1]);

    group.bench_function("depthwise_separable", |b| {
        b.iter(|| {
            let dw_out = depthwise_conv2d(
                black_box(&input),
                black_box(&depthwise_kernel),
                (1, 1),
                (1, 1),
                Precision::F32,
            ).unwrap();

            conv2d(
                black_box(&dw_out),
                black_box(&pointwise_kernel),
                (1, 1),
                (0, 0),
                Precision::F32,
            )
        })
    });

    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    conv2d_benches,
    bench_conv2d_basic,
    bench_conv2d_channels,
    bench_conv2d_kernel_sizes,
    bench_conv2d_strides,
    bench_conv2d_dilation,
);

criterion_group!(
    conv1d_benches,
    bench_conv1d_basic,
);

criterion_group!(
    specialized_conv_benches,
    bench_depthwise_conv2d,
    bench_group_conv2d,
);

criterion_group!(
    pooling_benches,
    bench_max_pool2d,
    bench_avg_pool2d,
    bench_global_avg_pool2d,
);

criterion_group!(
    overhead_benches,
    bench_error_propagation_overhead,
    bench_precision_comparison,
);

criterion_group!(
    workload_benches,
    bench_resnet_block,
    bench_mobilenet_block,
);

criterion_main!(
    conv2d_benches,
    conv1d_benches,
    specialized_conv_benches,
    pooling_benches,
    overhead_benches,
    workload_benches,
);
