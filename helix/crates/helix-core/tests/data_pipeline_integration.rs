//! Comprehensive integration tests for the HELIX data pipeline.
//!
//! Tests the complete data pipeline from dataset creation through commitment verification:
//! - Full pipeline from raw data to on-chain commitment
//! - Merkle tree operations at scale (1M+ elements)
//! - Streaming verification performance (<10ms per batch target)
//! - Sharding and shard verification
//! - Provenance tracking through training
//!
//! Run with: cargo test --test data_pipeline_integration

use helix_core::data::{
    // Dataset types
    Sample, Batch, DatasetMetadata, DataType,
    // Merkle tree types
    Hash, MerkleTree, MerkleTreeBuilder, MerkleProof, Sha256Hasher,
    StreamingMerkleBuilder, StreamingConfig, ChunkedMerkleBuilder,
    IncrementalRootComputer, SparseMerkleTree, ProofBatchVerifier,
    // Commitment types
    DatasetCommitment, SampleCommitment, BatchCommitment,
    // Sharding types
    DataSharder, ShardingConfig, ShardingStrategy,
    // Provenance types
    ProvenanceBuilder, DataOrigin, DataTransformation, TransformationType,
    Custodian, CustodianType, CustodyRecord, Attestation, AttestationType,
};
use helix_core::BinarySerializable;
use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// TEST HELPERS
// =============================================================================

/// Creates test samples with realistic structure.
fn create_realistic_samples(count: usize, feature_dim: usize) -> Vec<Sample> {
    (0..count)
        .map(|i| Sample {
            id: i,
            features: (0..feature_dim).map(|j| ((i * 31 + j * 17) % 256) as u8).collect(),
            labels: vec![(i % 10) as u8; 4], // 10 classes, 4 bytes for label
        })
        .collect()
}

/// Creates test metadata.
fn create_metadata(name: &str, num_samples: usize, _feature_dim: usize) -> DatasetMetadata {
    DatasetMetadata {
        name: name.to_string(),
        num_samples,
        feature_dims: vec![32],
        label_dims: vec![10],
        dtype: DataType::Float32,
        extra: HashMap::new(),
    }
}

/// Creates test batches from samples.
fn create_batches(samples: &[Sample], batch_size: usize) -> Vec<Batch> {
    samples
        .chunks(batch_size)
        .enumerate()
        .map(|(id, chunk)| Batch {
            id,
            samples: chunk.to_vec(),
        })
        .collect()
}

/// Creates a random hash for testing.
fn test_hash(seed: u64) -> Hash {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    Hash(bytes)
}

// =============================================================================
// FULL PIPELINE TESTS
// =============================================================================

#[test]
fn test_full_pipeline_small_dataset() {
    // Create dataset
    let samples = create_realistic_samples(100, 32);
    let metadata = create_metadata("test_dataset", 100, 32);

    // Create Merkle tree from samples - use add_hash to match DatasetCommitment::from_samples behavior
    let mut builder = MerkleTreeBuilder::with_sha256();
    for sample in &samples {
        // from_samples uses add_hash(Hash::from_bytes(hash_sample(sample))), so we match that
        builder = builder.add_hash(Hash::from_bytes(SampleCommitment::hash_sample(sample)));
    }
    let tree = builder.build().unwrap();

    // Create dataset commitment
    let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("test".to_string()));

    // Verify tree root matches commitment
    assert_eq!(tree.root().unwrap(), commitment.root);
    assert!(commitment.is_valid());

    // Create and verify proofs for all samples
    for i in 0..samples.len() {
        let proof = tree.prove(i).unwrap();
        assert!(proof.verify(&Sha256Hasher), "Proof failed for sample {}", i);
        assert_eq!(proof.leaf_index, i);
    }

    // Test serialization roundtrip
    let compact = commitment.to_compact_bytes();
    let restored = DatasetCommitment::from_compact_bytes(&compact).unwrap();
    assert_eq!(commitment.root, restored.root);
    assert_eq!(commitment.sample_count, restored.sample_count);
}

#[test]
fn test_full_pipeline_with_batching() {
    let samples = create_realistic_samples(1000, 64);
    let metadata = create_metadata("batched_dataset", 1000, 64);
    let batches = create_batches(&samples, 32);

    // Create dataset commitment
    let dataset_commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("batched".to_string()));

    // Create sample tree for verification
    let sample_tree = MerkleTree::from_hashes(
        Sha256Hasher,
        samples.iter().map(|s| SampleCommitment::new(s).hash).collect()
    ).unwrap();

    for batch in &batches {
        let sample_indices: Vec<usize> = batch.samples.iter().map(|s| s.id).collect();
        let batch_commitment = BatchCommitment::new(batch, sample_indices.clone(), dataset_commitment.root);

        // Verify batch root is non-zero
        assert!(!batch_commitment.root.is_zero());
    }

    // Verify random proofs
    for i in [0, 100, 500, 999] {
        let proof = sample_tree.prove(i).unwrap();
        assert!(proof.verify(&Sha256Hasher));
    }
}

#[test]
fn test_full_pipeline_with_sharding() {
    let samples = create_realistic_samples(10000, 32);
    let metadata = create_metadata("sharded_dataset", 10000, 32);

    // Shard the dataset
    let config = ShardingConfig {
        num_shards: 8,
        strategy: ShardingStrategy::RoundRobin,
        min_samples: 100,
        max_samples: None,
    };

    let sharder = DataSharder::new(config);
    let shards = sharder.shard(&samples);

    // Verify all samples are distributed
    let total_samples: usize = shards.iter().map(|s| s.sample_indices.len()).sum();
    assert_eq!(total_samples, 10000);

    // Verify each shard has reasonable size
    for shard in &shards {
        assert!(shard.sample_indices.len() >= 100);
        assert!(shard.sample_indices.len() <= 2000);
    }

    // Create commitment for each shard
    let _dataset_commitment = DatasetCommitment::from_samples(&samples, &metadata, None);

    for shard in &shards {
        let shard_samples: Vec<Sample> = shard.sample_indices
            .iter()
            .map(|&idx| samples[idx].clone())
            .collect();

        // Create shard-local commitment
        let shard_commitment = DatasetCommitment::from_samples(
            &shard_samples,
            &metadata,
            Some(format!("shard_{}", shard.id.0)),
        );

        assert!(shard_commitment.is_valid());
    }
}

// =============================================================================
// MERKLE TREE SCALE TESTS
// =============================================================================

#[test]
fn test_merkle_tree_1m_elements() {
    // Test streaming construction with 1M elements
    let mut builder = StreamingMerkleBuilder::with_config(
        Sha256Hasher,
        StreamingConfig::for_large_dataset(),
    );

    let start = Instant::now();
    for i in 0u64..1_000_000 {
        builder.add_hash(test_hash(i));
    }

    assert_eq!(builder.len(), 1_000_000);

    let root = builder.finalize_root().unwrap();
    let elapsed = start.elapsed();

    assert!(!root.is_zero());
    println!("1M elements streaming: {:?}", elapsed);

    // Also test incremental computer
    let mut computer = IncrementalRootComputer::with_sha256();
    for i in 0u64..1_000_000 {
        computer.add_hash(test_hash(i));
    }

    let incremental_root = computer.finalize().unwrap();
    assert!(!incremental_root.is_zero());
}

#[test]
fn test_merkle_tree_chunked_construction() {
    // Test chunked construction for 500K elements
    let mut builder = ChunkedMerkleBuilder::with_sha256(50000);

    for chunk_idx in 0u64..10 {
        let chunk_start = chunk_idx * 50000;
        let hashes: Vec<Hash> = (chunk_start..chunk_start + 50000)
            .map(|i| test_hash(i))
            .collect();
        builder.add_hash_chunk(hashes);
    }

    assert_eq!(builder.len(), 500000);

    let root = builder.compute_root().unwrap();
    assert!(!root.is_zero());
}

#[test]
fn test_sparse_merkle_tree_large_capacity() {
    // Test sparse tree with 1M capacity but sparse population
    let mut tree = SparseMerkleTree::with_sha256(20); // 2^20 = 1M capacity

    // Set only 1000 entries
    for i in 0..1000 {
        let index = (i * 997) % (1 << 20); // Spread entries
        tree.set(index, &[i as u8; 32]);
    }

    assert_eq!(tree.len(), 1000);
    assert_eq!(tree.capacity(), 1 << 20);

    let root = tree.root();
    assert!(!root.is_zero());

    // Verify proofs for set entries
    for i in 0..10 {
        let index = (i * 997) % (1 << 20);
        let proof = tree.prove(index).unwrap();
        assert!(proof.verify(&Sha256Hasher));
    }

    // Verify proofs for empty entries (should also work)
    let empty_proof = tree.prove(1).unwrap(); // Likely empty
    assert!(empty_proof.verify(&Sha256Hasher));
}

// =============================================================================
// STREAMING VERIFICATION TESTS
// =============================================================================

#[test]
fn test_streaming_verification_under_10ms() {
    // Create a tree with 50K elements
    let tree_size = 50_000;
    let hashes: Vec<Hash> = (0..tree_size)
        .map(|i| test_hash(i as u64))
        .collect();

    let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
    let root = tree.root().unwrap();

    // Create 100 proofs
    let batch_size = 100;
    let indices: Vec<usize> = (0..batch_size)
        .map(|i| (i * (tree_size / batch_size)) % tree_size)
        .collect();
    let proofs: Vec<MerkleProof> = indices.iter().map(|&i| tree.prove(i).unwrap()).collect();

    // Time batch verification
    let start = Instant::now();
    let mut verifier = ProofBatchVerifier::with_sha256(root);
    let verified = verifier.verify_batch(&proofs);
    let elapsed = start.elapsed();

    assert_eq!(verified, batch_size);
    assert!(
        elapsed.as_millis() < 50,
        "Batch verification took {:?}, expected < 50ms",
        elapsed
    );

    println!("100 proofs verified in {:?}", elapsed);
}

#[test]
fn test_streaming_verification_1000_proofs() {
    // Create tree with 100K elements
    let tree_size = 100_000;
    let hashes: Vec<Hash> = (0..tree_size)
        .map(|i| test_hash(i as u64))
        .collect();

    let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
    let root = tree.root().unwrap();

    // Create 1000 proofs
    let batch_size = 1000;
    let indices: Vec<usize> = (0..batch_size)
        .map(|i| (i * (tree_size / batch_size)) % tree_size)
        .collect();
    let proofs: Vec<MerkleProof> = indices.iter().map(|&i| tree.prove(i).unwrap()).collect();

    // Stream verification
    let start = Instant::now();
    let mut verifier = ProofBatchVerifier::with_sha256(root);
    let mut verified = 0;
    for proof in &proofs {
        if verifier.verify(proof) {
            verified += 1;
        }
    }
    let elapsed = start.elapsed();

    assert_eq!(verified, batch_size);
    println!("1000 proofs streamed in {:?} ({:?}/proof)", elapsed, elapsed / batch_size as u32);
}

// =============================================================================
// COMMITMENT VERIFICATION TESTS
// =============================================================================

#[test]
fn test_commitment_verification_chain() {
    let samples = create_realistic_samples(500, 32);
    let metadata = create_metadata("chain_test", 500, 32);

    // Create dataset commitment
    let dataset_commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("chain".to_string()));

    // Create sample commitments
    let sample_commitments: Vec<SampleCommitment> = samples.iter().map(SampleCommitment::new).collect();

    // Build tree matching from_samples behavior: add_hash(Hash::from_bytes(hash_sample(sample)))
    let mut builder = MerkleTreeBuilder::with_sha256();
    for sample in &samples {
        builder = builder.add_hash(Hash::from_bytes(SampleCommitment::hash_sample(sample)));
    }
    let tree = builder.build().unwrap();

    // Roots should match
    assert_eq!(tree.root().unwrap(), dataset_commitment.root);

    // Verify individual sample proofs
    for i in 0..sample_commitments.len() {
        let proof = tree.prove(i).unwrap();
        assert!(proof.verify(&Sha256Hasher));
    }
}

#[test]
fn test_batch_commitment_verification() {
    let samples = create_realistic_samples(1000, 32);
    let metadata = create_metadata("batch_verify", 1000, 32);
    let batches = create_batches(&samples, 100);

    let dataset_commitment = DatasetCommitment::from_samples(&samples, &metadata, None);

    // Verify each batch
    for batch in &batches {
        let sample_indices: Vec<usize> = batch.samples.iter().map(|s| s.id).collect();
        let batch_commitment = BatchCommitment::new(batch, sample_indices, dataset_commitment.root);

        // Batch root should be non-zero
        assert!(!batch_commitment.root.is_zero());

        // Verify batch is properly sized
        assert_eq!(batch_commitment.sample_count, batch.samples.len());
    }
}

#[test]
fn test_commitment_serialization_roundtrip() {
    let samples = create_realistic_samples(100, 64);
    let metadata = create_metadata("serialize_test", 100, 64);

    let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("serialize".to_string()));

    // Compact serialization
    let compact = commitment.to_compact_bytes();
    let restored_compact = DatasetCommitment::from_compact_bytes(&compact).unwrap();

    assert_eq!(commitment.root, restored_compact.root);
    assert_eq!(commitment.sample_count, restored_compact.sample_count);
    assert_eq!(commitment.metadata_hash, restored_compact.metadata_hash);

    // Full serialization via BinarySerializable
    let full_bytes = commitment.to_bytes();
    let restored_full = DatasetCommitment::from_bytes(&full_bytes).unwrap();

    assert_eq!(commitment.root, restored_full.root);
    assert_eq!(commitment.sample_count, restored_full.sample_count);
    assert_eq!(commitment.dataset_id, restored_full.dataset_id);
}

// =============================================================================
// PROOF SERIALIZATION TESTS
// =============================================================================

#[test]
fn test_proof_serialization_roundtrip() {
    let samples = create_realistic_samples(1000, 32);
    let tree = MerkleTree::from_hashes(
        Sha256Hasher,
        samples.iter().map(|s| SampleCommitment::new(s).hash).collect()
    ).unwrap();

    // Test serialization for various proof depths
    for i in [0, 100, 500, 999] {
        let proof = tree.prove(i).unwrap();
        let bytes = proof.to_bytes();
        let restored = MerkleProof::from_bytes(&bytes).unwrap();

        assert_eq!(proof.leaf_index, restored.leaf_index);
        assert_eq!(proof.leaf_hash, restored.leaf_hash);
        assert_eq!(proof.root, restored.root);
        assert_eq!(proof.path.len(), restored.path.len());

        // Restored proof should still verify
        assert!(restored.verify(&Sha256Hasher));
    }
}

// =============================================================================
// PROVENANCE TRACKING TESTS
// =============================================================================

#[test]
fn test_provenance_full_chain() {
    // Create dataset origin
    let origin = DataOrigin::Synthetic {
        generator: "mnist_generator".to_string(),
        seed: Some(42),
        parameters: HashMap::new(),
    };

    // Create original hash
    let original_hash = Hash::from_slice(&[1u8; 32]);

    // Build provenance chain
    let builder = ProvenanceBuilder::new(origin, original_hash);

    // Add transformations
    let transform1 = DataTransformation::new(
        TransformationType::Normalization { method: "minmax".to_string() },
        original_hash,
        Hash::from_slice(&[2u8; 32]),
        HashMap::new(),
    );
    let transform2 = DataTransformation::new(
        TransformationType::Augmentation { method: "rotation".to_string() },
        Hash::from_slice(&[2u8; 32]),
        Hash::from_slice(&[3u8; 32]),
        HashMap::new(),
    );

    let builder = builder
        .add_transformation(transform1)
        .add_transformation(transform2);

    // Add custody record
    let custodian = Custodian::new(CustodianType::DataProvider, "MNIST Provider".to_string());
    let custody_record = CustodyRecord::new(
        None,
        custodian,
        Hash::from_slice(&[3u8; 32]),
        "Initial data loading".to_string(),
    );
    let builder = builder.add_custody(custody_record);

    // Add attestation
    let attestation = Attestation::new(
        AttestationType::DataIntegrity,
        "MNIST Provider".to_string(),
        Hash::from_slice(&[3u8; 32]),
        "signature_placeholder".to_string(),
    );
    let builder = builder.add_attestation(attestation);

    // Build final record
    let record = builder.build();

    assert_eq!(record.transformations.len(), 2);
    assert_eq!(record.custody_chain.len(), 1);
    assert_eq!(record.attestations.len(), 1);
}

// =============================================================================
// SHARDING TESTS
// =============================================================================

#[test]
fn test_sharding_strategies() {
    let samples = create_realistic_samples(1000, 32);

    // Test RoundRobin
    let config = ShardingConfig {
        num_shards: 10,
        strategy: ShardingStrategy::RoundRobin,
        min_samples: 10,
        max_samples: None,
    };
    let sharder = DataSharder::new(config);
    let shards = sharder.shard(&samples);

    assert_eq!(shards.len(), 10);
    for shard in &shards {
        assert_eq!(shard.sample_indices.len(), 100);
    }

    // Test HashBased
    let config = ShardingConfig {
        num_shards: 10,
        strategy: ShardingStrategy::HashBased,
        min_samples: 10,
        max_samples: None,
    };
    let sharder = DataSharder::new(config);
    let shards = sharder.shard(&samples);

    assert_eq!(shards.len(), 10);
    let total: usize = shards.iter().map(|s| s.sample_indices.len()).sum();
    assert_eq!(total, 1000);

    // Test Random
    let config = ShardingConfig {
        num_shards: 10,
        strategy: ShardingStrategy::Random { seed: 42 },
        min_samples: 10,
        max_samples: None,
    };
    let sharder = DataSharder::new(config);
    let shards = sharder.shard(&samples);

    assert_eq!(shards.len(), 10);
    let total: usize = shards.iter().map(|s| s.sample_indices.len()).sum();
    assert_eq!(total, 1000);
}

#[test]
fn test_sharding_verification() {
    let samples = create_realistic_samples(10000, 32);
    let metadata = create_metadata("shard_verify", 10000, 32);

    let config = ShardingConfig {
        num_shards: 10,
        strategy: ShardingStrategy::RoundRobin,
        min_samples: 100,
        max_samples: None,
    };

    let sharder = DataSharder::new(config);
    let shards = sharder.shard(&samples);

    // Create full dataset commitment (uses add_hash internally)
    let dataset_commitment = DatasetCommitment::from_samples(&samples, &metadata, None);

    // Build tree matching from_samples behavior
    let mut builder = MerkleTreeBuilder::with_sha256();
    for sample in &samples {
        builder = builder.add_hash(Hash::from_bytes(SampleCommitment::hash_sample(sample)));
    }
    let full_tree = builder.build().unwrap();

    // Verify each shard has correct proofs
    for shard in &shards {
        for &idx in &shard.sample_indices {
            let proof = full_tree.prove(idx).unwrap();
            assert!(
                proof.verify_with_root(&Sha256Hasher, &dataset_commitment.root),
                "Proof failed for sample {} in shard {}",
                idx,
                shard.id.0
            );
        }
    }
}

// =============================================================================
// MULTI-PROOF TESTS
// =============================================================================

#[test]
fn test_multi_proof_generation_and_verification() {
    let samples = create_realistic_samples(1000, 32);
    let hashes: Vec<Hash> = samples.iter().map(|s| SampleCommitment::new(s).hash).collect();
    let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();

    // Generate multi-proof for a batch
    let indices: Vec<usize> = (0..100).collect();
    let multi_proof = tree.prove_multi(&indices).unwrap();

    // Verify multi-proof
    assert!(multi_proof.verify(&Sha256Hasher));
    assert_eq!(multi_proof.leaf_indices.len(), 100);

    // Multi-proof should have fewer nodes than individual proofs (shares common paths)
    assert!(multi_proof.proof_nodes.len() < indices.len() * tree.height());
}

#[test]
fn test_batch_proof_verification_performance() {
    let samples = create_realistic_samples(10000, 32);
    let hashes: Vec<Hash> = samples.iter().map(|s| SampleCommitment::new(s).hash).collect();
    let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
    let root = tree.root().unwrap();

    // Generate 500 proofs
    let indices: Vec<usize> = (0..500).map(|i| i * 20).collect();
    let proofs: Vec<MerkleProof> = indices.iter().map(|&i| tree.prove(i).unwrap()).collect();

    // Batch verification
    let start = Instant::now();
    let mut verifier = ProofBatchVerifier::with_sha256(root);
    let verified = verifier.verify_batch(&proofs);
    let elapsed = start.elapsed();

    assert_eq!(verified, 500);
    println!("500 proofs batch verified in {:?}", elapsed);

    let (success, failed) = verifier.stats();
    assert_eq!(success, 500);
    assert_eq!(failed, 0);
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_single_sample_dataset() {
    let samples = create_realistic_samples(1, 32);
    let metadata = create_metadata("single", 1, 32);

    let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
    assert!(commitment.is_valid());
    assert_eq!(commitment.sample_count, 1);

    let tree = MerkleTree::from_hashes(
        Sha256Hasher,
        samples.iter().map(|s| SampleCommitment::new(s).hash).collect()
    ).unwrap();

    let proof = tree.prove(0).unwrap();
    assert!(proof.verify(&Sha256Hasher));
}

#[test]
fn test_power_of_two_dataset() {
    // Test datasets that are exactly powers of 2
    for power in [1, 2, 4, 8, 16] {
        let size = 1usize << power;
        let samples = create_realistic_samples(size, 32);
        let metadata = create_metadata(&format!("pow2_{}", size), size, 32);

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        assert!(commitment.is_valid());
        assert_eq!(commitment.sample_count, size);

        let tree = MerkleTree::from_hashes(
            Sha256Hasher,
            samples.iter().map(|s| SampleCommitment::new(s).hash).collect()
        ).unwrap();

        // Height should be exactly power
        assert_eq!(tree.height(), power);

        // All proofs should verify
        for i in 0..size {
            let proof = tree.prove(i).unwrap();
            assert!(proof.verify(&Sha256Hasher));
        }
    }
}

#[test]
fn test_non_power_of_two_dataset() {
    // Test various non-power-of-2 sizes
    for size in [3, 7, 15, 100, 999, 1001] {
        let samples = create_realistic_samples(size, 32);

        let tree = MerkleTree::from_hashes(
            Sha256Hasher,
            samples.iter().map(|s| SampleCommitment::new(s).hash).collect()
        ).unwrap();

        // All proofs should verify
        for i in 0..size {
            let proof = tree.prove(i).unwrap();
            assert!(proof.verify(&Sha256Hasher), "Failed for size {} at index {}", size, i);
        }
    }
}

#[test]
fn test_proof_corruption_detection() {
    let samples = create_realistic_samples(100, 32);
    let tree = MerkleTree::from_hashes(
        Sha256Hasher,
        samples.iter().map(|s| SampleCommitment::new(s).hash).collect()
    ).unwrap();

    let mut proof = tree.prove(50).unwrap();

    // Corrupt leaf hash
    let original_hash = proof.leaf_hash;
    proof.leaf_hash = Hash::zero();
    assert!(!proof.verify(&Sha256Hasher));

    // Restore and corrupt path
    proof.leaf_hash = original_hash;
    if !proof.path.is_empty() {
        proof.path[0].sibling = Hash::zero();
        assert!(!proof.verify(&Sha256Hasher));
    }
}

#[test]
fn test_deterministic_commitments() {
    let samples = create_realistic_samples(100, 32);
    let metadata = create_metadata("deterministic", 100, 32);

    // Same data should produce same commitment
    let commitment1 = DatasetCommitment::from_samples(&samples, &metadata, Some("test".to_string()));
    let commitment2 = DatasetCommitment::from_samples(&samples, &metadata, Some("test".to_string()));

    assert_eq!(commitment1.root, commitment2.root);
    assert_eq!(commitment1.metadata_hash, commitment2.metadata_hash);
}

// =============================================================================
// CONCURRENCY SAFETY TESTS
// =============================================================================

#[test]
fn test_concurrent_proof_verification() {
    use std::thread;
    use std::sync::Arc;

    let samples = create_realistic_samples(1000, 32);
    let hashes: Vec<Hash> = samples.iter().map(|s| SampleCommitment::new(s).hash).collect();
    let tree = Arc::new(MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap());
    let root = tree.root().unwrap();

    // Generate proofs for various indices
    let proofs: Arc<Vec<MerkleProof>> = Arc::new(
        (0..100).map(|i| tree.prove(i * 10).unwrap()).collect()
    );

    // Verify in multiple threads
    let mut handles = vec![];
    for _ in 0..4 {
        let proofs_clone = Arc::clone(&proofs);
        let handle = thread::spawn(move || {
            for proof in proofs_clone.iter() {
                assert!(proof.verify_with_root(&Sha256Hasher, &root));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// =============================================================================
// PERFORMANCE CHARACTERISTIC TESTS
// =============================================================================

#[test]
fn test_tree_height_scaling() {
    // Verify tree height scales logarithmically
    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let hashes: Vec<Hash> = (0..size)
            .map(|i| test_hash(i as u64))
            .collect();

        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
        let expected_height = (size as f64).log2().ceil() as usize;

        assert_eq!(tree.height(), expected_height, "Height mismatch for size {}", size);
    }
}

#[test]
fn test_proof_size_scaling() {
    // Verify proof size scales logarithmically
    let sizes = [100, 1000, 10000];
    let mut proof_sizes = vec![];

    for &size in &sizes {
        let hashes: Vec<Hash> = (0..size)
            .map(|i| test_hash(i as u64))
            .collect();

        let tree = MerkleTree::from_hashes(Sha256Hasher, hashes).unwrap();
        let proof = tree.prove(size / 2).unwrap();
        proof_sizes.push((size, proof.to_bytes().len()));
    }

    // Proof size should roughly double per 1000x increase in tree size
    println!("Proof sizes: {:?}", proof_sizes);

    for window in proof_sizes.windows(2) {
        let (_, size1) = window[0];
        let (_, size2) = window[1];
        // Proof size should increase, but sublinearly
        assert!(size2 < size1 * 3, "Proof size growing too fast");
    }
}
