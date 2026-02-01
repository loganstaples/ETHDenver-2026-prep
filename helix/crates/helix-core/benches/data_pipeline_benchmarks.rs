//! Comprehensive benchmarks for HELIX data pipeline.
//!
//! Tests performance of:
//! - Merkle tree construction (including 1M+ elements)
//! - Streaming verification throughput
//! - Shard operations performance
//! - Commitment generation speed
//!
//! Run with: cargo bench --bench data_pipeline_benchmarks
//!
//! Success criteria:
//! - Handles 1M+ elements
//! - Streaming verification <10ms per batch

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use helix_core::data::{
    // Merkle tree types
    Hash, MerkleTree, MerkleTreeBuilder, MerkleProof, Sha256Hasher,
    StreamingMerkleBuilder, StreamingConfig, ChunkedMerkleBuilder,
    IncrementalRootComputer, SparseMerkleTree, ProofBatchVerifier,
    // Commitment types
    DatasetCommitment, SampleCommitment, BatchCommitment,
    // Dataset types
    Sample, Batch, DatasetMetadata, DataType,
    // Sharding types
    DataSharder, ShardingConfig, ShardingStrategy,
};
use helix_core::BinarySerializable;
use std::collections::HashMap;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a random hash for benchmarking.
fn random_hash(seed: u64) -> Hash {
    let mut bytes = [0u8; 32];
    // Simple deterministic "random" for reproducible benchmarks
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = ((seed.wrapping_mul(31).wrapping_add(i as u64)) & 0xFF) as u8;
    }
    Hash(bytes)
}

/// Creates test data of given size.
fn create_test_data(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("sample_data_{}", i).into_bytes())
        .collect()
}

/// Creates test samples for commitment benchmarks.
fn create_test_samples(count: usize) -> Vec<Sample> {
    (0..count)
        .map(|i| Sample {
            id: i,
            features: vec![i as u8; 32],
            labels: vec![(i % 10) as u8; 4],
        })
        .collect()
}

/// Creates test metadata.
fn create_test_metadata() -> DatasetMetadata {
    DatasetMetadata {
        name: "benchmark_dataset".to_string(),
        num_samples: 0,
        feature_dims: vec![32],
        label_dims: vec![1],
        dtype: DataType::Float32,
        extra: HashMap::new(),
    }
}

// =============================================================================
// MERKLE TREE CONSTRUCTION BENCHMARKS
// =============================================================================

fn bench_merkle_tree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree_construction");
    group.sample_size(10);

    // Test various sizes up to 100K
    for size in [1_000, 10_000, 50_000, 100_000].iter() {
        let data = create_test_data(*size);
        let leaf_refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("standard", size),
            &leaf_refs,
            |b, leaves| {
                b.iter(|| {
                    MerkleTree::from_leaves(Sha256Hasher, black_box(leaves)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_merkle_tree_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree_builder");
    group.sample_size(10);

    for size in [1_000, 10_000, 50_000].iter() {
        let data = create_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("incremental", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut builder = MerkleTreeBuilder::with_sha256();
                    for d in data {
                        builder = builder.add_leaf(d);
                    }
                    builder.build().unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_streaming_merkle_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_merkle");
    group.sample_size(10);

    // Test streaming construction for large datasets
    for size in [100_000, 500_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("streaming_root", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut builder = StreamingMerkleBuilder::with_config(
                        Sha256Hasher,
                        StreamingConfig {
                            buffer_size: 65536,
                            ..Default::default()
                        },
                    );
                    for i in 0u64..(size as u64) {
                        builder.add_hash(black_box(random_hash(i)));
                    }
                    builder.finalize_root().unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_incremental_root_computer(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_root");
    group.sample_size(10);

    // The most memory-efficient option for root-only computation
    for size in [100_000, 500_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("incremental", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut computer = IncrementalRootComputer::with_sha256();
                    for i in 0u64..(size as u64) {
                        computer.add_hash(black_box(random_hash(i)));
                    }
                    computer.finalize().unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_chunked_merkle_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_merkle");
    group.sample_size(10);

    for (size, chunk_size) in [(100_000, 10_000), (500_000, 50_000)].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("chunks_{}", chunk_size), size),
            &(*size, *chunk_size),
            |b, &(size, chunk_size)| {
                b.iter(|| {
                    let mut builder = ChunkedMerkleBuilder::with_sha256(chunk_size);
                    for chunk_start in (0..size).step_by(chunk_size) {
                        let chunk_end = (chunk_start + chunk_size).min(size);
                        let hashes: Vec<Hash> = (chunk_start..chunk_end)
                            .map(|i| random_hash(i as u64))
                            .collect();
                        builder.add_hash_chunk(hashes);
                    }
                    builder.compute_root().unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_sparse_merkle_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_merkle");

    // Sparse tree with 1M capacity but varying fill
    for num_entries in [100, 1_000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("set_entries", num_entries),
            num_entries,
            |b, &num_entries| {
                b.iter(|| {
                    let mut tree = SparseMerkleTree::with_sha256(20); // 2^20 = 1M capacity
                    for i in 0..num_entries {
                        let index = (i * 997) % (1 << 20); // Spread entries
                        tree.set(index, &[i as u8; 32]);
                    }
                    tree.root()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// PROOF GENERATION BENCHMARKS
// =============================================================================

fn bench_single_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation");

    for size in [1_000, 10_000, 100_000].iter() {
        let hashes: Vec<Hash> = (0..*size).map(|i| random_hash(i as u64)).collect();
        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();

        group.bench_with_input(
            BenchmarkId::new("single_proof", size),
            &tree,
            |b, tree| {
                b.iter(|| {
                    tree.prove(black_box(*size / 2)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_proof_generation");

    for size in [10_000, 50_000].iter() {
        let hashes: Vec<Hash> = (0..*size).map(|i| random_hash(i as u64)).collect();
        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();

        // Generate proofs for 100 random indices
        let indices: Vec<usize> = (0..100).map(|i| (i * (*size / 100)) % *size).collect();

        group.bench_with_input(
            BenchmarkId::new("batch_100_proofs", size),
            &(&tree, &indices),
            |b, (tree, indices)| {
                b.iter(|| {
                    tree.prove_batch(black_box(indices)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_multi_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_proof_generation");

    for size in [10_000, 50_000].iter() {
        let hashes: Vec<Hash> = (0..*size).map(|i| random_hash(i as u64)).collect();
        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();

        let indices: Vec<usize> = (0..100).map(|i| (i * (*size / 100)) % *size).collect();

        group.bench_with_input(
            BenchmarkId::new("multi_proof_100", size),
            &(&tree, &indices),
            |b, (tree, indices)| {
                b.iter(|| {
                    tree.prove_multi(black_box(indices)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// PROOF VERIFICATION BENCHMARKS
// =============================================================================

fn bench_single_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_verification");

    for size in [1_000, 10_000, 100_000].iter() {
        let hashes: Vec<Hash> = (0..*size).map(|i| random_hash(i as u64)).collect();
        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
        let proof = tree.prove(*size / 2).unwrap();

        group.bench_with_input(
            BenchmarkId::new("single_verify", size),
            &proof,
            |b, proof| {
                b.iter(|| {
                    black_box(proof.verify(&Sha256Hasher))
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_proof_verification");

    for batch_size in [10, 50, 100, 500].iter() {
        let tree_size = 10_000;
        let hashes: Vec<Hash> = (0..tree_size).map(|i| random_hash(i as u64)).collect();
        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
        let root = tree.root().unwrap();

        let indices: Vec<usize> = (0..*batch_size)
            .map(|i| (i * (tree_size / *batch_size)) % tree_size)
            .collect();
        let proofs: Vec<MerkleProof> = indices.iter().map(|&i| tree.prove(i).unwrap()).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_verify", batch_size),
            &(&proofs, root),
            |b, (proofs, root)| {
                b.iter(|| {
                    let mut verifier = ProofBatchVerifier::with_sha256(*root);
                    verifier.verify_batch(black_box(proofs))
                })
            },
        );
    }

    group.finish();
}

fn bench_streaming_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_verification");
    // Target: <10ms per batch

    for batch_size in [100, 500, 1000].iter() {
        let tree_size = 50_000;
        let hashes: Vec<Hash> = (0..tree_size).map(|i| random_hash(i as u64)).collect();
        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
        let root = tree.root().unwrap();

        let indices: Vec<usize> = (0..*batch_size)
            .map(|i| (i * (tree_size / *batch_size)) % tree_size)
            .collect();
        let proofs: Vec<MerkleProof> = indices.iter().map(|&i| tree.prove(i).unwrap()).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("streaming_batch", batch_size),
            &(&proofs, root),
            |b, (proofs, root)| {
                b.iter(|| {
                    // Simulate streaming verification
                    let mut verified = 0;
                    let mut verifier = ProofBatchVerifier::with_sha256(*root);
                    for proof in proofs.iter() {
                        if verifier.verify(black_box(proof)) {
                            verified += 1;
                        }
                    }
                    verified
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// COMMITMENT BENCHMARKS
// =============================================================================

fn bench_sample_commitment(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_commitment");

    for sample_size in [32, 256, 1024, 4096].iter() {
        let sample = Sample {
            id: 0,
            features: vec![0u8; *sample_size],
            labels: vec![0u8; 4],
        };

        group.throughput(Throughput::Bytes(*sample_size as u64));
        group.bench_with_input(
            BenchmarkId::new("hash_sample", sample_size),
            &sample,
            |b, sample| {
                b.iter(|| {
                    SampleCommitment::new(black_box(sample))
                })
            },
        );
    }

    group.finish();
}

fn bench_dataset_commitment(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_commitment");
    group.sample_size(10);

    for num_samples in [100, 1_000, 10_000].iter() {
        let samples = create_test_samples(*num_samples);
        let metadata = create_test_metadata();

        group.throughput(Throughput::Elements(*num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("from_samples", num_samples),
            &(&samples, &metadata),
            |b, (samples, metadata)| {
                b.iter(|| {
                    DatasetCommitment::from_samples(
                        black_box(samples),
                        black_box(metadata),
                        Some("bench_dataset".to_string()),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_commitment(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_commitment");

    for batch_size in [32, 64, 128, 256].iter() {
        let samples = create_test_samples(*batch_size);
        let sample_indices: Vec<usize> = (0..*batch_size).collect();
        let batch = Batch {
            id: 0,
            samples,
        };
        let dataset_id = random_hash(12345);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("from_batch", batch_size),
            &(&batch, sample_indices.clone(), dataset_id),
            |b, (batch, indices, ds_id)| {
                b.iter(|| {
                    BatchCommitment::new(
                        black_box(batch),
                        indices.clone(),
                        *ds_id,
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// SHARDING BENCHMARKS
// =============================================================================

fn bench_data_sharding(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_sharding");
    group.sample_size(10);

    for num_samples in [1_000, 10_000, 50_000].iter() {
        let samples = create_test_samples(*num_samples);

        // Test different shard counts
        for num_shards in [4, 8, 16].iter() {
            let config = ShardingConfig {
                num_shards: *num_shards as u32,
                strategy: ShardingStrategy::RoundRobin,
                min_samples: 100,
                max_samples: None,
            };

            group.throughput(Throughput::Elements(*num_samples as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}shards", num_shards), num_samples),
                &(&samples, config.clone()),
                |b, (samples, config)| {
                    b.iter(|| {
                        let sharder = DataSharder::new(config.clone());
                        sharder.shard(black_box(samples))
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// SERIALIZATION BENCHMARKS
// =============================================================================

fn bench_proof_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    for tree_size in [1_000, 10_000, 100_000].iter() {
        let hashes: Vec<Hash> = (0..*tree_size).map(|i| random_hash(i as u64)).collect();
        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
        let proof = tree.prove(*tree_size / 2).unwrap();

        group.bench_with_input(
            BenchmarkId::new("proof_serialize", tree_size),
            &proof,
            |b, proof| {
                b.iter(|| {
                    proof.to_bytes()
                })
            },
        );

        let bytes = proof.to_bytes();
        group.bench_with_input(
            BenchmarkId::new("proof_deserialize", tree_size),
            &bytes,
            |b, bytes| {
                b.iter(|| {
                    MerkleProof::from_bytes(black_box(bytes)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_commitment_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("commitment_serialization");

    let samples = create_test_samples(1000);
    let metadata = create_test_metadata();
    let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("bench".to_string()));

    group.bench_function("commitment_to_compact", |b| {
        b.iter(|| {
            commitment.to_compact_bytes()
        })
    });

    let compact = commitment.to_compact_bytes();
    group.bench_function("commitment_from_compact", |b| {
        b.iter(|| {
            DatasetCommitment::from_compact_bytes(black_box(&compact)).unwrap()
        })
    });

    group.finish();
}

// =============================================================================
// LARGE SCALE BENCHMARKS (1M+ elements)
// =============================================================================

fn bench_million_elements(c: &mut Criterion) {
    let mut group = c.benchmark_group("million_elements");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    // 1M elements with streaming
    group.bench_function("1m_streaming_root", |b| {
        b.iter(|| {
            let mut builder = StreamingMerkleBuilder::with_config(
                Sha256Hasher,
                StreamingConfig::for_large_dataset(),
            );
            for i in 0u64..1_000_000 {
                builder.add_hash(random_hash(i));
            }
            builder.finalize_root().unwrap()
        })
    });

    // 1M elements with incremental computer
    group.bench_function("1m_incremental_root", |b| {
        b.iter(|| {
            let mut computer = IncrementalRootComputer::with_sha256();
            for i in 0u64..1_000_000 {
                computer.add_hash(random_hash(i));
            }
            computer.finalize().unwrap()
        })
    });

    group.finish();
}

fn bench_throughput_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_targets");

    // Verification throughput target: <10ms per batch
    let tree_size = 50_000;
    let hashes: Vec<Hash> = (0..tree_size).map(|i| random_hash(i as u64)).collect();
    let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
    let root = tree.root().unwrap();

    // Create batch of 100 proofs
    let batch_size = 100;
    let indices: Vec<usize> = (0..batch_size)
        .map(|i| (i * (tree_size / batch_size)) % tree_size)
        .collect();
    let proofs: Vec<MerkleProof> = indices.iter().map(|&i| tree.prove(i).unwrap()).collect();

    group.throughput(Throughput::Elements(batch_size as u64));
    group.bench_function("target_10ms_batch_verify", |b| {
        b.iter(|| {
            let mut verifier = ProofBatchVerifier::with_sha256(root);
            verifier.verify_batch(black_box(&proofs))
        })
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    merkle_construction,
    bench_merkle_tree_construction,
    bench_merkle_tree_builder,
    bench_streaming_merkle_construction,
    bench_incremental_root_computer,
    bench_chunked_merkle_builder,
    bench_sparse_merkle_tree,
);

criterion_group!(
    proof_generation,
    bench_single_proof_generation,
    bench_batch_proof_generation,
    bench_multi_proof_generation,
);

criterion_group!(
    proof_verification,
    bench_single_proof_verification,
    bench_batch_proof_verification,
    bench_streaming_verification,
);

criterion_group!(
    commitments,
    bench_sample_commitment,
    bench_dataset_commitment,
    bench_batch_commitment,
);

criterion_group!(
    sharding,
    bench_data_sharding,
);

criterion_group!(
    serialization,
    bench_proof_serialization,
    bench_commitment_serialization,
);

criterion_group!(
    name = large_scale;
    config = Criterion::default().sample_size(10);
    targets = bench_million_elements, bench_throughput_targets
);

criterion_main!(
    merkle_construction,
    proof_generation,
    proof_verification,
    commitments,
    sharding,
    serialization,
    large_scale,
);
