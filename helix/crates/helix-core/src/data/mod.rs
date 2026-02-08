//! Data Module for HELIX.
//!
//! Provides comprehensive data management for training:
//! - Dataset loading and batching
//! - Data sharding for federated learning
//! - Model serialization and versioning
//! - IPFS integration for decentralized storage
//! - Model registry
//! - Merkle tree commitments for data verification
//! - Batch membership proofs
//! - Data provenance tracking
//! - Multi-source data fetching (IPFS, Filecoin, S3)

pub mod commitment;
pub mod dataset;
pub mod dataset_registry;
pub mod ipfs;
pub mod membership_proof;
pub mod merkle;
pub mod model;
pub mod provenance;
pub mod registry;
pub mod sharding;
pub mod shuffling;
pub mod sources;
pub mod streaming;

// Re-export dataset types
pub use dataset::{
    Batch, DataType, DatasetConfig, DatasetMetadata, InMemoryDataset, Sample,
    create_synthetic, load_csv,
};

// Re-export IPFS types (legacy module)
pub use ipfs::{Cid, IpfsClient, IpfsConfig, IpfsDatasetRef, IpfsModelRef, IpfsModelRegistry};

// Re-export model types
pub use model::{
    LayerType, LayerWeights, ModelCompression, ModelDiff, ModelFormat, ModelMetadata,
    ModelSerializer, SerializedModel, TensorData,
};

// Re-export registry types
pub use registry::{ModelEntry, ModelId, ModelRegistry, ModelStatus, ModelVersion};

// Re-export sharding types
pub use sharding::{
    DataShard, DataSharder, ShardAssignment, ShardId, ShardingConfig, ShardingStrategy,
    // Worker types for distributed training
    WorkerId, WorkerInfo, WorkerStatus, ShardStats, ShardRegistry, ShardRegistryConfig,
    LocalityAwareAssigner, ShardStreamer, ShardStreamConfig, ShardChunk, ShardProgress,
    OverallProgress, ShardMetadata, ShardVerification,
};

// Re-export Merkle tree types
pub use merkle::{
    Hash, MerkleError, MerkleHasher, MerkleProof, MerkleTree, MerkleTreeBuilder,
    MerkleTreeConfig, MultiProof, ProofDirection, ProofStep, Sha256Hasher, TreePosition,
    HASH_SIZE,
    // Streaming and memory-efficient builders
    StreamingMerkleBuilder, StreamingConfig, StreamingStats,
    ChunkedMerkleBuilder, IncrementalRootComputer,
    ParallelMerkleBuilder, ParallelConfig,
    SparseMerkleTree, ProofBatchVerifier,
};

// Re-export commitment types
pub use commitment::{
    BatchCommitment, CommitmentError, CommitmentManager, DatasetCommitment, SampleCommitment,
};

// Re-export membership proof types
pub use membership_proof::{
    AggregatedBatchProof, BatchMembershipProof, MembershipProofError, MembershipProofGenerator,
    MembershipVerifier, SampleMembershipProof, VerificationStats,
};

// Re-export provenance types
pub use provenance::{
    Attestation, AttestationType, CustodyRecord, Custodian, CustodianType, DataOrigin,
    DataTransformation, ProvenanceBuilder, ProvenanceChainSummary, ProvenanceError,
    ProvenanceId, ProvenanceRecord, ProvenanceRegistry, TransformationType,
};

// Re-export data source types
pub use sources::{
    DataCache, DataChunk, DataSource, DataSourceError, DataSourceResult, DataStream,
    FallbackBehavior, FetchOptions, MultiSourceFetcher, PoolConfig, ResourceMetadata,
    UploadOptions, WritableDataSource,
    // IPFS source
    IpfsDataSource, IpfsSourceConfig,
    // Filecoin source
    FilecoinClient, FilecoinConfig, FilecoinDataSource,
    // S3 source
    S3Config, S3DataSource,
};

// Re-export streaming verification types
pub use streaming::{
    BatchBuilder, BatchVerificationResult, StreamingBatchVerifier,
    StreamingVerificationConfig, VerificationBatch, VerificationCheckpoint,
    VerificationStats as StreamingVerificationStats,
};

// Re-export deterministic shuffling types
pub use shuffling::{
    DeterministicShuffler, SeedCommitment, ShuffleConfig, ShuffleResult,
    ShuffleSeed, ShuffleSchedule, ShuffleVerifier,
};

// Re-export dataset commitment registry types
pub use dataset_registry::{
    AttestationType as DatasetAttestationType, AttesterType, DatasetAttestation,
    DatasetCommitment as DatasetCommitmentEntry, DatasetCommitmentRegistry,
    DatasetEntry, DatasetId, DatasetMetadata as DatasetRegistryMetadata,
    DatasetStatus, RegistryConfig, RegistryError, RegistryResult, RegistryStats,
};

// ============================================================================
// DataLoader trait and implementations
// ============================================================================

use crate::error::{DataError, HelixError, HelixResult};

/// Trait for loading training data batches.
///
/// Implementations provide batches of (inputs, labels) pairs identified by
/// batch ID. This abstraction allows swapping between in-memory datasets,
/// sharded distributed datasets, and remote data sources.
pub trait DataLoader: Send + Sync {
    /// Loads a batch by its ID.
    ///
    /// Returns `(inputs, labels)` where:
    /// - `inputs`: flattened feature values for all samples in the batch
    /// - `labels`: corresponding target values
    ///
    /// # Errors
    /// Returns `DataError` if the batch ID is out of range or data is unavailable.
    fn load_batch(&self, batch_id: u64) -> HelixResult<(Vec<f64>, Vec<f64>)>;

    /// Returns the total number of batches available.
    fn num_batches(&self) -> u64;
}

/// In-memory data loader that stores all data in memory.
///
/// Suitable for small datasets or testing. Data is pre-loaded and batches
/// are sliced from the stored vectors.
#[derive(Debug, Clone)]
pub struct InMemoryDataLoader {
    /// All input features, organized as batches.
    batches: Vec<(Vec<f64>, Vec<f64>)>,
}

impl InMemoryDataLoader {
    /// Creates a new in-memory loader from pre-batched data.
    pub fn new(batches: Vec<(Vec<f64>, Vec<f64>)>) -> Self {
        Self { batches }
    }

    /// Creates a loader from flat data, splitting into batches of the given size.
    ///
    /// `features_per_sample` determines how many f64 values constitute one sample's input.
    pub fn from_flat(
        inputs: Vec<f64>,
        labels: Vec<f64>,
        batch_size: usize,
        features_per_sample: usize,
    ) -> Self {
        let sample_count = labels.len();
        let mut batches = Vec::new();

        for start in (0..sample_count).step_by(batch_size) {
            let end = (start + batch_size).min(sample_count);
            let input_start = start * features_per_sample;
            let input_end = end * features_per_sample;

            if input_end <= inputs.len() && end <= labels.len() {
                batches.push((
                    inputs[input_start..input_end].to_vec(),
                    labels[start..end].to_vec(),
                ));
            }
        }

        Self { batches }
    }
}

impl DataLoader for InMemoryDataLoader {
    fn load_batch(&self, batch_id: u64) -> HelixResult<(Vec<f64>, Vec<f64>)> {
        let idx = batch_id as usize;
        self.batches.get(idx).cloned().ok_or_else(|| {
            HelixError::Data(DataError::BatchNotFound(idx))
        })
    }

    fn num_batches(&self) -> u64 {
        self.batches.len() as u64
    }
}

/// Sharded data loader that distributes data across shards.
///
/// Each shard contains a subset of the full dataset. The loader maps
/// batch IDs to the appropriate shard and local batch index.
#[derive(Debug, Clone)]
pub struct ShardedDataLoader {
    /// Per-shard data, indexed by shard number.
    shards: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    /// Total number of batches across all shards.
    total_batches: u64,
}

impl ShardedDataLoader {
    /// Creates a new sharded loader from pre-organized shard data.
    pub fn new(shards: Vec<Vec<(Vec<f64>, Vec<f64>)>>) -> Self {
        let total_batches: u64 = shards.iter().map(|s| s.len() as u64).sum();
        Self {
            shards,
            total_batches,
        }
    }

    /// Creates a sharded loader by distributing flat data evenly across `num_shards`.
    pub fn from_flat(
        inputs: Vec<f64>,
        labels: Vec<f64>,
        batch_size: usize,
        features_per_sample: usize,
        num_shards: usize,
    ) -> Self {
        let sample_count = labels.len();
        let samples_per_shard = (sample_count + num_shards - 1) / num_shards;
        let mut shards = Vec::with_capacity(num_shards);

        for shard_idx in 0..num_shards {
            let shard_start = shard_idx * samples_per_shard;
            let shard_end = (shard_start + samples_per_shard).min(sample_count);
            if shard_start >= sample_count {
                shards.push(Vec::new());
                continue;
            }

            let mut shard_batches = Vec::new();
            for batch_start in (shard_start..shard_end).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(shard_end);
                let input_start = batch_start * features_per_sample;
                let input_end = batch_end * features_per_sample;

                if input_end <= inputs.len() && batch_end <= labels.len() {
                    shard_batches.push((
                        inputs[input_start..input_end].to_vec(),
                        labels[batch_start..batch_end].to_vec(),
                    ));
                }
            }
            shards.push(shard_batches);
        }

        Self::new(shards)
    }

    /// Returns the number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

impl DataLoader for ShardedDataLoader {
    fn load_batch(&self, batch_id: u64) -> HelixResult<(Vec<f64>, Vec<f64>)> {
        // Map global batch_id to (shard_index, local_batch_index)
        let mut remaining = batch_id;
        for shard in &self.shards {
            let shard_len = shard.len() as u64;
            if remaining < shard_len {
                return Ok(shard[remaining as usize].clone());
            }
            remaining -= shard_len;
        }
        Err(HelixError::Data(DataError::BatchNotFound(batch_id as usize)))
    }

    fn num_batches(&self) -> u64 {
        self.total_batches
    }
}
