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
pub mod ipfs;
pub mod membership_proof;
pub mod merkle;
pub mod model;
pub mod provenance;
pub mod registry;
pub mod sharding;
pub mod sources;

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
