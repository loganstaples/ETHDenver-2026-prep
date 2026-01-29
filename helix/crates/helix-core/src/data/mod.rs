//! Data Module for HELIX.
//!
//! Provides data loading, sharding, model serialization, and storage:
//! - Dataset loading and batching
//! - Data sharding for federated learning
//! - Model serialization and versioning
//! - IPFS integration for decentralized storage
//! - Model registry

pub mod dataset;
pub mod ipfs;
pub mod model;
pub mod registry;
pub mod sharding;

// Re-export commonly used types
pub use dataset::{
    Batch, DataType, DatasetConfig, DatasetMetadata, InMemoryDataset, Sample,
    create_synthetic, load_csv,
};
pub use ipfs::{Cid, IpfsClient, IpfsConfig, IpfsDatasetRef, IpfsModelRef, IpfsModelRegistry};
pub use model::{
    LayerType, LayerWeights, ModelCompression, ModelDiff, ModelFormat, ModelMetadata,
    ModelSerializer, SerializedModel, TensorData,
};
pub use registry::{ModelEntry, ModelId, ModelRegistry, ModelStatus, ModelVersion};
pub use sharding::{DataShard, DataSharder, ShardAssignment, ShardId, ShardingConfig, ShardingStrategy};
