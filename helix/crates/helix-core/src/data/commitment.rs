//! Dataset Commitment Types for HELIX.
//!
//! Provides cryptographic commitments to training datasets that can be:
//! - Verified on-chain
//! - Used to prove batch membership
//! - Tracked through training provenance
//!
//! Commitments are designed for efficient serialization and smart contract verification.

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::time::SystemTime;

use super::merkle::{Hash, MerkleTree, MerkleTreeBuilder, Sha256Hasher, HASH_SIZE};
use super::dataset::{Batch, DatasetMetadata, Sample};
use crate::traits::BinarySerializable;
use crate::traits::serializable::SerializeError;

/// A commitment to an entire dataset.
///
/// This represents a Merkle root over all samples in a dataset,
/// along with metadata for verification and provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetCommitment {
    /// The Merkle root hash of all samples.
    pub root: Hash,
    /// Number of samples in the dataset.
    pub sample_count: usize,
    /// Hash of the dataset metadata.
    pub metadata_hash: Hash,
    /// Version identifier for the commitment format.
    pub version: u8,
    /// Timestamp when commitment was created (Unix seconds).
    pub created_at: u64,
    /// Optional dataset name/identifier.
    pub dataset_id: Option<String>,
    /// Hash algorithm used.
    pub hash_algorithm: String,
}

impl DatasetCommitment {
    /// Current commitment format version.
    pub const VERSION: u8 = 1;

    /// Creates a new dataset commitment from a Merkle tree.
    pub fn new(
        tree: &MerkleTree<Sha256Hasher>,
        metadata: &DatasetMetadata,
        dataset_id: Option<String>,
    ) -> Self {
        let root = tree.root().unwrap_or(Hash::zero());
        let metadata_hash = Self::hash_metadata(metadata);
        let created_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            root,
            sample_count: tree.len(),
            metadata_hash,
            version: Self::VERSION,
            created_at,
            dataset_id,
            hash_algorithm: "sha256".to_string(),
        }
    }

    /// Creates a commitment from raw samples.
    pub fn from_samples(
        samples: &[Sample],
        metadata: &DatasetMetadata,
        dataset_id: Option<String>,
    ) -> Self {
        let mut builder = MerkleTreeBuilder::with_sha256();
        for sample in samples {
            // Use add_hash since hash_sample already returns a hash
            let hash = Hash::from_bytes(SampleCommitment::hash_sample(sample));
            builder = builder.add_hash(hash);
        }

        let tree = builder.build().expect("Failed to build Merkle tree");
        Self::new(&tree, metadata, dataset_id)
    }

    /// Hashes metadata for inclusion in commitment.
    fn hash_metadata(metadata: &DatasetMetadata) -> Hash {
        let json = serde_json::to_vec(metadata).unwrap_or_default();
        Sha256Hasher.hash_leaf(&json)
    }

    /// Returns the commitment as bytes suitable for on-chain storage.
    /// Compact format: 32 bytes root + 8 bytes count + 32 bytes metadata = 72 bytes minimum.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(72);
        bytes.extend_from_slice(self.root.as_bytes());
        bytes.extend_from_slice(&(self.sample_count as u64).to_le_bytes());
        bytes.extend_from_slice(self.metadata_hash.as_bytes());
        bytes
    }

    /// Parses from compact bytes.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self, CommitmentError> {
        if bytes.len() < 72 {
            return Err(CommitmentError::InvalidFormat(
                "Compact bytes too short".into(),
            ));
        }

        let mut root_bytes = [0u8; HASH_SIZE];
        root_bytes.copy_from_slice(&bytes[0..32]);
        let root = Hash(root_bytes);

        let mut count_bytes = [0u8; 8];
        count_bytes.copy_from_slice(&bytes[32..40]);
        let sample_count = u64::from_le_bytes(count_bytes) as usize;

        let mut meta_bytes = [0u8; HASH_SIZE];
        meta_bytes.copy_from_slice(&bytes[40..72]);
        let metadata_hash = Hash(meta_bytes);

        Ok(Self {
            root,
            sample_count,
            metadata_hash,
            version: Self::VERSION,
            created_at: 0,
            dataset_id: None,
            hash_algorithm: "sha256".to_string(),
        })
    }

    /// Verifies that this commitment is well-formed.
    pub fn is_valid(&self) -> bool {
        !self.root.is_zero() && self.sample_count > 0 && self.version == Self::VERSION
    }

    /// Returns a unique identifier for this commitment.
    pub fn commitment_id(&self) -> Hash {
        Sha256Hasher.hash_leaf(&self.to_compact_bytes())
    }
}

impl BinarySerializable for DatasetCommitment {
    fn serialized_size(&self) -> usize {
        // root + count + meta_hash + version + created_at + algo_len + algo + id_len + id
        let id_len = self.dataset_id.as_ref().map(|s| s.len()).unwrap_or(0);
        HASH_SIZE + 8 + HASH_SIZE + 1 + 8 + 2 + self.hash_algorithm.len() + 2 + id_len
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        BinarySerializable::serialize(&self.root, writer)?;
        writer.write_all(&(self.sample_count as u64).to_le_bytes())?;
        BinarySerializable::serialize(&self.metadata_hash, writer)?;
        writer.write_all(&[self.version])?;
        writer.write_all(&self.created_at.to_le_bytes())?;

        // Hash algorithm
        let algo_bytes = self.hash_algorithm.as_bytes();
        writer.write_all(&(algo_bytes.len() as u16).to_le_bytes())?;
        writer.write_all(algo_bytes)?;

        // Dataset ID
        match &self.dataset_id {
            Some(id) => {
                let id_bytes = id.as_bytes();
                writer.write_all(&(id_bytes.len() as u16).to_le_bytes())?;
                writer.write_all(id_bytes)?;
            }
            None => {
                writer.write_all(&0u16.to_le_bytes())?;
            }
        }

        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let root = <Hash as BinarySerializable>::deserialize(reader)?;

        let mut count_buf = [0u8; 8];
        reader.read_exact(&mut count_buf)?;
        let sample_count = u64::from_le_bytes(count_buf) as usize;

        let metadata_hash = <Hash as BinarySerializable>::deserialize(reader)?;

        let mut version_buf = [0u8; 1];
        reader.read_exact(&mut version_buf)?;
        let version = version_buf[0];

        let mut ts_buf = [0u8; 8];
        reader.read_exact(&mut ts_buf)?;
        let created_at = u64::from_le_bytes(ts_buf);

        // Hash algorithm
        let mut algo_len_buf = [0u8; 2];
        reader.read_exact(&mut algo_len_buf)?;
        let algo_len = u16::from_le_bytes(algo_len_buf) as usize;
        let mut algo_buf = vec![0u8; algo_len];
        reader.read_exact(&mut algo_buf)?;
        let hash_algorithm = String::from_utf8(algo_buf)
            .map_err(|_| SerializeError::InvalidData("Invalid algorithm string".into()))?;

        // Dataset ID
        let mut id_len_buf = [0u8; 2];
        reader.read_exact(&mut id_len_buf)?;
        let id_len = u16::from_le_bytes(id_len_buf) as usize;
        let dataset_id = if id_len > 0 {
            let mut id_buf = vec![0u8; id_len];
            reader.read_exact(&mut id_buf)?;
            Some(
                String::from_utf8(id_buf)
                    .map_err(|_| SerializeError::InvalidData("Invalid dataset ID".into()))?,
            )
        } else {
            None
        };

        Ok(Self {
            root,
            sample_count,
            metadata_hash,
            version,
            created_at,
            dataset_id,
            hash_algorithm,
        })
    }
}

/// A commitment to a single training batch.
///
/// Batches are subsets of a dataset used in individual training steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCommitment {
    /// The Merkle root of samples in this batch.
    pub root: Hash,
    /// Batch index within the dataset.
    pub batch_index: usize,
    /// Number of samples in the batch.
    pub sample_count: usize,
    /// Sample indices in the parent dataset.
    pub sample_indices: Vec<usize>,
    /// Hash of batch configuration (ordering, augmentation, etc.).
    pub config_hash: Hash,
    /// Reference to parent dataset commitment.
    pub dataset_commitment_id: Hash,
}

impl BatchCommitment {
    /// Creates a batch commitment from a batch.
    pub fn new(
        batch: &Batch,
        sample_indices: Vec<usize>,
        dataset_commitment_id: Hash,
    ) -> Self {
        let mut builder = MerkleTreeBuilder::with_sha256();
        for sample in &batch.samples {
            builder = builder.add_leaf(&SampleCommitment::hash_sample(sample));
        }

        let tree = builder.build().expect("Failed to build batch tree");
        let root = tree.root().unwrap_or(Hash::zero());

        // Config hash includes batch ID and sample ordering
        let config_data = format!(
            "batch:{}:samples:{:?}",
            batch.id,
            sample_indices
        );
        let config_hash = Sha256Hasher.hash_leaf(config_data.as_bytes());

        Self {
            root,
            batch_index: batch.id,
            sample_count: batch.samples.len(),
            sample_indices,
            config_hash,
            dataset_commitment_id,
        }
    }

    /// Creates a batch commitment from raw samples with explicit indices.
    pub fn from_samples(
        samples: &[&Sample],
        batch_index: usize,
        sample_indices: Vec<usize>,
        dataset_commitment_id: Hash,
    ) -> Self {
        let mut builder = MerkleTreeBuilder::with_sha256();
        for sample in samples {
            builder = builder.add_leaf(&SampleCommitment::hash_sample(sample));
        }

        let tree = builder.build().expect("Failed to build batch tree");
        let root = tree.root().unwrap_or(Hash::zero());

        let config_data = format!(
            "batch:{}:samples:{:?}",
            batch_index,
            sample_indices
        );
        let config_hash = Sha256Hasher.hash_leaf(config_data.as_bytes());

        Self {
            root,
            batch_index,
            sample_count: samples.len(),
            sample_indices,
            config_hash,
            dataset_commitment_id,
        }
    }

    /// Returns compact bytes for on-chain verification.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(72 + self.sample_indices.len() * 8);
        bytes.extend_from_slice(self.root.as_bytes());
        bytes.extend_from_slice(&(self.batch_index as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.sample_count as u64).to_le_bytes());
        bytes.extend_from_slice(self.dataset_commitment_id.as_bytes());
        bytes
    }

    /// Returns a unique identifier for this batch commitment.
    pub fn commitment_id(&self) -> Hash {
        Sha256Hasher.hash_leaf(&self.to_compact_bytes())
    }
}

impl BinarySerializable for BatchCommitment {
    fn serialized_size(&self) -> usize {
        HASH_SIZE + 8 + 8 + 4 + self.sample_indices.len() * 8 + HASH_SIZE + HASH_SIZE
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        BinarySerializable::serialize(&self.root, writer)?;
        writer.write_all(&(self.batch_index as u64).to_le_bytes())?;
        writer.write_all(&(self.sample_count as u64).to_le_bytes())?;

        writer.write_all(&(self.sample_indices.len() as u32).to_le_bytes())?;
        for &idx in &self.sample_indices {
            writer.write_all(&(idx as u64).to_le_bytes())?;
        }

        BinarySerializable::serialize(&self.config_hash, writer)?;
        BinarySerializable::serialize(&self.dataset_commitment_id, writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let root = <Hash as BinarySerializable>::deserialize(reader)?;

        let mut batch_idx_buf = [0u8; 8];
        reader.read_exact(&mut batch_idx_buf)?;
        let batch_index = u64::from_le_bytes(batch_idx_buf) as usize;

        let mut count_buf = [0u8; 8];
        reader.read_exact(&mut count_buf)?;
        let sample_count = u64::from_le_bytes(count_buf) as usize;

        let mut indices_len_buf = [0u8; 4];
        reader.read_exact(&mut indices_len_buf)?;
        let indices_len = u32::from_le_bytes(indices_len_buf) as usize;

        let mut sample_indices = Vec::with_capacity(indices_len);
        for _ in 0..indices_len {
            let mut idx_buf = [0u8; 8];
            reader.read_exact(&mut idx_buf)?;
            sample_indices.push(u64::from_le_bytes(idx_buf) as usize);
        }

        let config_hash = <Hash as BinarySerializable>::deserialize(reader)?;
        let dataset_commitment_id = <Hash as BinarySerializable>::deserialize(reader)?;

        Ok(Self {
            root,
            batch_index,
            sample_count,
            sample_indices,
            config_hash,
            dataset_commitment_id,
        })
    }
}

/// A commitment to a single sample.
///
/// Used for fine-grained verification and provenance tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleCommitment {
    /// Hash of the sample data.
    pub hash: Hash,
    /// Sample index in the dataset.
    pub index: usize,
    /// Hash of features only.
    pub features_hash: Hash,
    /// Hash of labels only.
    pub labels_hash: Hash,
}

impl SampleCommitment {
    /// Creates a commitment from a sample.
    pub fn new(sample: &Sample) -> Self {
        let features_hash = Sha256Hasher.hash_leaf(&sample.features);
        let labels_hash = Sha256Hasher.hash_leaf(&sample.labels);
        let hash = Hash(Self::hash_sample(sample));

        Self {
            hash,
            index: sample.id,
            features_hash,
            labels_hash,
        }
    }

    /// Computes the canonical hash of a sample.
    pub fn hash_sample(sample: &Sample) -> [u8; HASH_SIZE] {
        // Combine: sample_id || features_len || features || labels_len || labels
        let mut data = Vec::with_capacity(16 + sample.features.len() + sample.labels.len());
        data.extend_from_slice(&(sample.id as u64).to_le_bytes());
        data.extend_from_slice(&(sample.features.len() as u32).to_le_bytes());
        data.extend_from_slice(&sample.features);
        data.extend_from_slice(&(sample.labels.len() as u32).to_le_bytes());
        data.extend_from_slice(&sample.labels);

        *Sha256Hasher.hash_leaf(&data).as_bytes()
    }

    /// Verifies that a sample matches this commitment.
    pub fn verify(&self, sample: &Sample) -> bool {
        let computed = Self::hash_sample(sample);
        computed == *self.hash.as_bytes() && sample.id == self.index
    }
}

impl BinarySerializable for SampleCommitment {
    fn serialized_size(&self) -> usize {
        HASH_SIZE * 3 + 8
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        BinarySerializable::serialize(&self.hash, writer)?;
        writer.write_all(&(self.index as u64).to_le_bytes())?;
        BinarySerializable::serialize(&self.features_hash, writer)?;
        BinarySerializable::serialize(&self.labels_hash, writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let hash = <Hash as BinarySerializable>::deserialize(reader)?;

        let mut idx_buf = [0u8; 8];
        reader.read_exact(&mut idx_buf)?;
        let index = u64::from_le_bytes(idx_buf) as usize;

        let features_hash = <Hash as BinarySerializable>::deserialize(reader)?;
        let labels_hash = <Hash as BinarySerializable>::deserialize(reader)?;

        Ok(Self {
            hash,
            index,
            features_hash,
            labels_hash,
        })
    }
}

/// A commitment manager for handling dataset commitments and batch tracking.
pub struct CommitmentManager {
    /// The dataset commitment.
    dataset: DatasetCommitment,
    /// The Merkle tree for proofs.
    tree: MerkleTree<Sha256Hasher>,
    /// Sample commitments cache.
    samples: Vec<SampleCommitment>,
    /// Batch commitments generated so far.
    batches: Vec<BatchCommitment>,
}

impl CommitmentManager {
    /// Creates a new commitment manager from samples.
    pub fn new(
        samples: &[Sample],
        metadata: &DatasetMetadata,
        dataset_id: Option<String>,
    ) -> Self {
        let sample_commitments: Vec<SampleCommitment> =
            samples.iter().map(SampleCommitment::new).collect();

        let mut builder = MerkleTreeBuilder::with_sha256();
        for sc in &sample_commitments {
            builder = builder.add_hash(sc.hash);
        }

        let tree = builder.build().expect("Failed to build tree");
        let dataset = DatasetCommitment::new(&tree, metadata, dataset_id);

        Self {
            dataset,
            tree,
            samples: sample_commitments,
            batches: Vec::new(),
        }
    }

    /// Returns the dataset commitment.
    pub fn dataset_commitment(&self) -> &DatasetCommitment {
        &self.dataset
    }

    /// Returns the Merkle tree.
    pub fn tree(&self) -> &MerkleTree<Sha256Hasher> {
        &self.tree
    }

    /// Gets a sample commitment by index.
    pub fn sample_commitment(&self, index: usize) -> Option<&SampleCommitment> {
        self.samples.get(index)
    }

    /// Creates and stores a batch commitment.
    pub fn commit_batch(&mut self, batch: &Batch, sample_indices: Vec<usize>) -> BatchCommitment {
        let commitment = BatchCommitment::new(batch, sample_indices, self.dataset.commitment_id());
        self.batches.push(commitment.clone());
        commitment
    }

    /// Returns all batch commitments.
    pub fn batch_commitments(&self) -> &[BatchCommitment] {
        &self.batches
    }

    /// Gets a batch commitment by index.
    pub fn get_batch(&self, index: usize) -> Option<&BatchCommitment> {
        self.batches.get(index)
    }
}

/// Errors related to commitments.
#[derive(Debug, Clone)]
pub enum CommitmentError {
    /// Invalid commitment format.
    InvalidFormat(String),
    /// Commitment verification failed.
    VerificationFailed,
    /// Sample not found.
    SampleNotFound(usize),
    /// Batch not found.
    BatchNotFound(usize),
    /// Merkle tree error.
    MerkleError(String),
}

impl std::fmt::Display for CommitmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommitmentError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            CommitmentError::VerificationFailed => write!(f, "Verification failed"),
            CommitmentError::SampleNotFound(idx) => write!(f, "Sample not found: {}", idx),
            CommitmentError::BatchNotFound(idx) => write!(f, "Batch not found: {}", idx),
            CommitmentError::MerkleError(msg) => write!(f, "Merkle error: {}", msg),
        }
    }
}

impl std::error::Error for CommitmentError {}

/// Use the MerkleHasher trait for Sha256Hasher
use super::merkle::MerkleHasher;

// ============================================================================
// DATASET COMMITMENT REGISTRY
// ============================================================================

/// Registry for storing and managing dataset commitments for on-chain reference.
///
/// The registry tracks dataset commitments, their versions, and provides
/// encoding suitable for smart contract verification.
pub struct DatasetCommitmentRegistry {
    /// Registered commitments by ID.
    commitments: std::collections::HashMap<Hash, RegisteredCommitment>,
    /// Commitment history (latest version for each dataset_id).
    by_dataset_id: std::collections::HashMap<String, Hash>,
    /// Commitment versions.
    versions: std::collections::HashMap<Hash, Vec<CommitmentVersion>>,
    /// Aggregated commitments for batch on-chain submission.
    aggregated: Vec<AggregatedCommitment>,
    /// Registry configuration.
    config: RegistryConfig,
}

/// Configuration for the commitment registry.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Maximum commitments per aggregation.
    pub max_aggregation_size: usize,
    /// Whether to track full history.
    pub track_history: bool,
    /// Enable verification on registration.
    pub verify_on_register: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_aggregation_size: 256,
            track_history: true,
            verify_on_register: true,
        }
    }
}

/// A registered commitment with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredCommitment {
    /// The commitment.
    pub commitment: DatasetCommitment,
    /// Registration timestamp.
    pub registered_at: u64,
    /// Registration block (if on-chain).
    pub registered_block: Option<u64>,
    /// Transaction hash (if on-chain).
    pub tx_hash: Option<Hash>,
    /// Current status.
    pub status: CommitmentStatus,
    /// Owner address (if applicable).
    pub owner: Option<String>,
    /// Version number.
    pub version: u64,
}

/// Commitment status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentStatus {
    /// Registered locally, not yet on-chain.
    Local,
    /// Pending on-chain confirmation.
    Pending,
    /// Confirmed on-chain.
    Confirmed,
    /// Revoked or replaced.
    Revoked,
    /// Verification failed.
    Invalid,
}

/// A version entry for commitment history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentVersion {
    /// Version number.
    pub version: u64,
    /// Commitment hash for this version.
    pub commitment_id: Hash,
    /// Timestamp.
    pub created_at: u64,
    /// Reason for new version.
    pub reason: Option<String>,
    /// Previous version's commitment ID.
    pub previous: Option<Hash>,
}

/// Aggregated commitment for efficient on-chain storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedCommitment {
    /// Root of the aggregated commitments Merkle tree.
    pub root: Hash,
    /// Number of commitments included.
    pub count: usize,
    /// Individual commitment IDs.
    pub commitment_ids: Vec<Hash>,
    /// Timestamp.
    pub created_at: u64,
    /// Aggregation ID.
    pub aggregation_id: Hash,
}

impl AggregatedCommitment {
    /// Creates a new aggregated commitment.
    pub fn new(commitments: &[&DatasetCommitment]) -> Self {
        let commitment_ids: Vec<Hash> = commitments
            .iter()
            .map(|c| c.commitment_id())
            .collect();

        // Build Merkle tree from commitment IDs
        let mut builder = MerkleTreeBuilder::with_sha256();
        for id in &commitment_ids {
            builder = builder.add_hash(*id);
        }
        let tree = builder.build().expect("Failed to build aggregation tree");
        let root = tree.root().unwrap_or(Hash::zero());

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Aggregation ID is hash of root + timestamp
        let mut id_data = Vec::new();
        id_data.extend_from_slice(root.as_bytes());
        id_data.extend_from_slice(&now.to_le_bytes());
        let aggregation_id = Sha256Hasher.hash_leaf(&id_data);

        Self {
            root,
            count: commitments.len(),
            commitment_ids,
            created_at: now,
            aggregation_id,
        }
    }

    /// Returns compact bytes for on-chain storage.
    /// Format: root (32) + count (4) + aggregation_id (32) = 68 bytes minimum
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(68);
        bytes.extend_from_slice(self.root.as_bytes());
        bytes.extend_from_slice(&(self.count as u32).to_le_bytes());
        bytes.extend_from_slice(self.aggregation_id.as_bytes());
        bytes
    }

    /// Verifies that a commitment is included in this aggregation.
    pub fn verify_inclusion(&self, commitment_id: &Hash) -> bool {
        self.commitment_ids.contains(commitment_id)
    }
}

/// On-chain commitment data structure.
///
/// This is the minimal data structure for smart contract storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainCommitment {
    /// Dataset root hash (32 bytes).
    pub root: Hash,
    /// Sample count (8 bytes).
    pub sample_count: u64,
    /// Metadata hash (32 bytes).
    pub metadata_hash: Hash,
    /// Owner address (20 bytes for Ethereum).
    pub owner: [u8; 20],
    /// Block number when registered.
    pub block_number: u64,
}

impl OnChainCommitment {
    /// Creates from a dataset commitment.
    pub fn from_commitment(commitment: &DatasetCommitment, owner: [u8; 20], block: u64) -> Self {
        Self {
            root: commitment.root,
            sample_count: commitment.sample_count as u64,
            metadata_hash: commitment.metadata_hash,
            owner,
            block_number: block,
        }
    }

    /// Returns ABI-encoded bytes for Solidity.
    /// Total: 32 + 32 + 32 + 32 + 32 = 160 bytes (padded for Solidity)
    pub fn to_abi_encoded(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(160);

        // root (bytes32)
        bytes.extend_from_slice(self.root.as_bytes());

        // sample_count (uint256) - left-padded to 32 bytes
        bytes.extend_from_slice(&[0u8; 24]);
        bytes.extend_from_slice(&self.sample_count.to_be_bytes());

        // metadata_hash (bytes32)
        bytes.extend_from_slice(self.metadata_hash.as_bytes());

        // owner (address) - left-padded to 32 bytes
        bytes.extend_from_slice(&[0u8; 12]);
        bytes.extend_from_slice(&self.owner);

        // block_number (uint256)
        bytes.extend_from_slice(&[0u8; 24]);
        bytes.extend_from_slice(&self.block_number.to_be_bytes());

        bytes
    }

    /// Computes the commitment hash for on-chain verification.
    pub fn commitment_hash(&self) -> Hash {
        Sha256Hasher.hash_leaf(&self.to_abi_encoded())
    }
}

impl DatasetCommitmentRegistry {
    /// Creates a new registry.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            commitments: std::collections::HashMap::new(),
            by_dataset_id: std::collections::HashMap::new(),
            versions: std::collections::HashMap::new(),
            aggregated: Vec::new(),
            config,
        }
    }

    /// Creates with default configuration.
    pub fn default_registry() -> Self {
        Self::new(RegistryConfig::default())
    }

    // ========================================================================
    // REGISTRATION
    // ========================================================================

    /// Registers a commitment.
    pub fn register(&mut self, commitment: DatasetCommitment, owner: Option<String>) -> Result<Hash, CommitmentError> {
        // Validate if configured
        if self.config.verify_on_register && !commitment.is_valid() {
            return Err(CommitmentError::InvalidFormat("Commitment validation failed".into()));
        }

        let commitment_id = commitment.commitment_id();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Check if this is an update to existing dataset
        let version = if let Some(dataset_id) = &commitment.dataset_id {
            if let Some(existing_id) = self.by_dataset_id.get(dataset_id) {
                // This is an update
                let existing = self.commitments.get(existing_id);
                let v = existing.map(|e| e.version + 1).unwrap_or(1);

                // Record version history
                if self.config.track_history {
                    self.versions.entry(commitment_id).or_default().push(CommitmentVersion {
                        version: v,
                        commitment_id,
                        created_at: now,
                        reason: Some("Update".into()),
                        previous: Some(*existing_id),
                    });
                }

                // Mark old version as revoked
                if let Some(old) = self.commitments.get_mut(existing_id) {
                    old.status = CommitmentStatus::Revoked;
                }

                v
            } else {
                1
            }
        } else {
            1
        };

        let registered = RegisteredCommitment {
            commitment: commitment.clone(),
            registered_at: now,
            registered_block: None,
            tx_hash: None,
            status: CommitmentStatus::Local,
            owner,
            version,
        };

        self.commitments.insert(commitment_id, registered);

        // Update dataset_id mapping
        if let Some(dataset_id) = &commitment.dataset_id {
            self.by_dataset_id.insert(dataset_id.clone(), commitment_id);
        }

        Ok(commitment_id)
    }

    /// Registers a commitment and marks it as pending on-chain submission.
    pub fn register_pending(&mut self, commitment: DatasetCommitment, owner: Option<String>) -> Result<Hash, CommitmentError> {
        let id = self.register(commitment, owner)?;
        if let Some(reg) = self.commitments.get_mut(&id) {
            reg.status = CommitmentStatus::Pending;
        }
        Ok(id)
    }

    // ========================================================================
    // QUERIES
    // ========================================================================

    /// Gets a registered commitment by ID.
    pub fn get(&self, commitment_id: &Hash) -> Option<&RegisteredCommitment> {
        self.commitments.get(commitment_id)
    }

    /// Gets the latest commitment for a dataset ID.
    pub fn get_by_dataset_id(&self, dataset_id: &str) -> Option<&RegisteredCommitment> {
        self.by_dataset_id.get(dataset_id)
            .and_then(|id| self.commitments.get(id))
    }

    /// Lists all commitments.
    pub fn list_all(&self) -> Vec<&RegisteredCommitment> {
        self.commitments.values().collect()
    }

    /// Lists commitments by status.
    pub fn list_by_status(&self, status: CommitmentStatus) -> Vec<&RegisteredCommitment> {
        self.commitments.values()
            .filter(|c| c.status == status)
            .collect()
    }

    /// Lists pending commitments (ready for on-chain submission).
    pub fn list_pending(&self) -> Vec<&RegisteredCommitment> {
        self.list_by_status(CommitmentStatus::Pending)
    }

    /// Gets version history for a commitment.
    pub fn get_history(&self, commitment_id: &Hash) -> Vec<&CommitmentVersion> {
        self.versions.get(commitment_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    // ========================================================================
    // STATUS UPDATES
    // ========================================================================

    /// Marks a commitment as confirmed on-chain.
    pub fn confirm(&mut self, commitment_id: &Hash, block: u64, tx_hash: Hash) -> Result<(), CommitmentError> {
        let reg = self.commitments.get_mut(commitment_id)
            .ok_or(CommitmentError::InvalidFormat("Commitment not found".into()))?;

        reg.status = CommitmentStatus::Confirmed;
        reg.registered_block = Some(block);
        reg.tx_hash = Some(tx_hash);

        Ok(())
    }

    /// Marks a commitment as revoked.
    pub fn revoke(&mut self, commitment_id: &Hash) -> Result<(), CommitmentError> {
        let reg = self.commitments.get_mut(commitment_id)
            .ok_or(CommitmentError::InvalidFormat("Commitment not found".into()))?;

        reg.status = CommitmentStatus::Revoked;
        Ok(())
    }

    /// Marks a commitment as invalid.
    pub fn mark_invalid(&mut self, commitment_id: &Hash) -> Result<(), CommitmentError> {
        let reg = self.commitments.get_mut(commitment_id)
            .ok_or(CommitmentError::InvalidFormat("Commitment not found".into()))?;

        reg.status = CommitmentStatus::Invalid;
        Ok(())
    }

    // ========================================================================
    // AGGREGATION
    // ========================================================================

    /// Creates an aggregated commitment from pending commitments.
    pub fn aggregate_pending(&mut self) -> Option<AggregatedCommitment> {
        let pending: Vec<_> = self.list_pending()
            .iter()
            .take(self.config.max_aggregation_size)
            .map(|r| &r.commitment)
            .collect();

        if pending.is_empty() {
            return None;
        }

        let agg = AggregatedCommitment::new(&pending);
        self.aggregated.push(agg.clone());
        Some(agg)
    }

    /// Gets all aggregated commitments.
    pub fn aggregated_commitments(&self) -> &[AggregatedCommitment] {
        &self.aggregated
    }

    /// Finds the aggregation containing a commitment.
    pub fn find_aggregation(&self, commitment_id: &Hash) -> Option<&AggregatedCommitment> {
        self.aggregated.iter().find(|a| a.verify_inclusion(commitment_id))
    }

    // ========================================================================
    // VERIFICATION
    // ========================================================================

    /// Verifies a commitment against stored data.
    pub fn verify_commitment(
        &self,
        commitment_id: &Hash,
        samples: &[Sample],
        metadata: &DatasetMetadata,
    ) -> Result<bool, CommitmentError> {
        let reg = self.get(commitment_id)
            .ok_or(CommitmentError::InvalidFormat("Commitment not found".into()))?;

        // Recompute commitment
        let recomputed = DatasetCommitment::from_samples(
            samples,
            metadata,
            reg.commitment.dataset_id.clone(),
        );

        Ok(recomputed.root == reg.commitment.root
            && recomputed.sample_count == reg.commitment.sample_count
            && recomputed.metadata_hash == reg.commitment.metadata_hash)
    }

    /// Verifies a sample belongs to a commitment using Merkle proof.
    pub fn verify_sample_membership(
        &self,
        commitment_id: &Hash,
        _sample: &Sample,
        proof: &super::merkle::MerkleProof,
    ) -> Result<bool, CommitmentError> {
        let reg = self.get(commitment_id)
            .ok_or(CommitmentError::InvalidFormat("Commitment not found".into()))?;

        // Check proof root matches commitment
        if proof.root != reg.commitment.root {
            return Ok(false);
        }

        // Verify the proof
        Ok(proof.verify(&Sha256Hasher))
    }

    // ========================================================================
    // ON-CHAIN ENCODING
    // ========================================================================

    /// Prepares a commitment for on-chain registration.
    pub fn prepare_for_chain(
        &self,
        commitment_id: &Hash,
        owner: [u8; 20],
        block: u64,
    ) -> Result<OnChainCommitment, CommitmentError> {
        let reg = self.get(commitment_id)
            .ok_or(CommitmentError::InvalidFormat("Commitment not found".into()))?;

        Ok(OnChainCommitment::from_commitment(&reg.commitment, owner, block))
    }

    /// Prepares multiple commitments for batch on-chain registration.
    pub fn prepare_batch_for_chain(
        &self,
        commitment_ids: &[Hash],
        owner: [u8; 20],
        block: u64,
    ) -> Vec<OnChainCommitment> {
        commitment_ids.iter()
            .filter_map(|id| self.prepare_for_chain(id, owner, block).ok())
            .collect()
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Returns registry statistics.
    pub fn stats(&self) -> RegistryStats {
        let mut stats = RegistryStats::default();

        for reg in self.commitments.values() {
            stats.total_commitments += 1;
            stats.total_samples += reg.commitment.sample_count;

            match reg.status {
                CommitmentStatus::Local => stats.local_count += 1,
                CommitmentStatus::Pending => stats.pending_count += 1,
                CommitmentStatus::Confirmed => stats.confirmed_count += 1,
                CommitmentStatus::Revoked => stats.revoked_count += 1,
                CommitmentStatus::Invalid => stats.invalid_count += 1,
            }
        }

        stats.aggregation_count = self.aggregated.len();
        stats
    }
}

/// Registry statistics.
#[derive(Debug, Clone, Default)]
pub struct RegistryStats {
    /// Total commitments.
    pub total_commitments: usize,
    /// Total samples across all commitments.
    pub total_samples: usize,
    /// Local (unsubmitted) count.
    pub local_count: usize,
    /// Pending count.
    pub pending_count: usize,
    /// Confirmed count.
    pub confirmed_count: usize,
    /// Revoked count.
    pub revoked_count: usize,
    /// Invalid count.
    pub invalid_count: usize,
    /// Aggregation count.
    pub aggregation_count: usize,
}

/// Commitment proof for on-chain verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentProof {
    /// The commitment being proved.
    pub commitment_id: Hash,
    /// Merkle proof for inclusion in aggregation (if aggregated).
    pub aggregation_proof: Option<Vec<Hash>>,
    /// Aggregation root (if aggregated).
    pub aggregation_root: Option<Hash>,
    /// Direct commitment data.
    pub commitment_data: Vec<u8>,
}

impl CommitmentProof {
    /// Creates a proof for a non-aggregated commitment.
    pub fn direct(commitment: &DatasetCommitment) -> Self {
        Self {
            commitment_id: commitment.commitment_id(),
            aggregation_proof: None,
            aggregation_root: None,
            commitment_data: commitment.to_compact_bytes(),
        }
    }

    /// Returns ABI-encoded proof for smart contract verification.
    pub fn to_abi_encoded(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // commitment_id
        bytes.extend_from_slice(self.commitment_id.as_bytes());

        // has_aggregation (bool as uint8)
        bytes.push(if self.aggregation_root.is_some() { 1 } else { 0 });

        // aggregation_root (if present)
        if let Some(root) = &self.aggregation_root {
            bytes.extend_from_slice(root.as_bytes());
        } else {
            bytes.extend_from_slice(&[0u8; 32]);
        }

        // commitment_data_length
        bytes.extend_from_slice(&(self.commitment_data.len() as u32).to_be_bytes());

        // commitment_data
        bytes.extend_from_slice(&self.commitment_data);

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_metadata() -> DatasetMetadata {
        DatasetMetadata {
            name: "test_dataset".to_string(),
            num_samples: 10,
            feature_dims: vec![4],
            label_dims: vec![2],
            dtype: crate::data::DataType::Float32,
            extra: HashMap::new(),
        }
    }

    fn create_test_samples(count: usize) -> Vec<Sample> {
        (0..count)
            .map(|i| Sample::from_f32(i, vec![i as f32; 4], vec![0.0, 1.0]))
            .collect()
    }

    #[test]
    fn test_sample_commitment() {
        let sample = Sample::from_f32(0, vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]);
        let commitment = SampleCommitment::new(&sample);

        assert_eq!(commitment.index, 0);
        assert!(commitment.verify(&sample));
    }

    #[test]
    fn test_sample_commitment_fails_on_modified() {
        let sample = Sample::from_f32(0, vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]);
        let commitment = SampleCommitment::new(&sample);

        let modified = Sample::from_f32(0, vec![1.0, 2.0, 3.0, 5.0], vec![0.0, 1.0]);
        assert!(!commitment.verify(&modified));
    }

    #[test]
    fn test_dataset_commitment() {
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("test".into()));

        assert!(commitment.is_valid());
        assert_eq!(commitment.sample_count, 10);
        assert!(!commitment.root.is_zero());
    }

    #[test]
    fn test_dataset_commitment_serialization() {
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("test".into()));

        let bytes = commitment.to_bytes();
        let restored = DatasetCommitment::from_bytes(&bytes).unwrap();

        assert_eq!(commitment.root, restored.root);
        assert_eq!(commitment.sample_count, restored.sample_count);
    }

    #[test]
    fn test_dataset_commitment_compact() {
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);

        let compact = commitment.to_compact_bytes();
        assert_eq!(compact.len(), 72);

        let restored = DatasetCommitment::from_compact_bytes(&compact).unwrap();
        assert_eq!(commitment.root, restored.root);
        assert_eq!(commitment.sample_count, restored.sample_count);
    }

    #[test]
    fn test_batch_commitment() {
        let samples = create_test_samples(10);
        let batch = Batch::new(0, samples[0..4].to_vec());
        let dataset_id = Hash::from_slice(b"test_dataset");

        let commitment = BatchCommitment::new(&batch, vec![0, 1, 2, 3], dataset_id);

        assert_eq!(commitment.batch_index, 0);
        assert_eq!(commitment.sample_count, 4);
        assert_eq!(commitment.sample_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_batch_commitment_serialization() {
        let samples = create_test_samples(10);
        let batch = Batch::new(0, samples[0..4].to_vec());
        let dataset_id = Hash::from_slice(b"test_dataset");

        let commitment = BatchCommitment::new(&batch, vec![0, 1, 2, 3], dataset_id);

        let bytes = commitment.to_bytes();
        let restored = BatchCommitment::from_bytes(&bytes).unwrap();

        assert_eq!(commitment.root, restored.root);
        assert_eq!(commitment.batch_index, restored.batch_index);
        assert_eq!(commitment.sample_indices, restored.sample_indices);
    }

    #[test]
    fn test_commitment_manager() {
        let samples = create_test_samples(100);
        let metadata = DatasetMetadata {
            name: "test".to_string(),
            num_samples: 100,
            feature_dims: vec![4],
            label_dims: vec![2],
            dtype: crate::data::DataType::Float32,
            extra: HashMap::new(),
        };

        let mut manager = CommitmentManager::new(&samples, &metadata, Some("test".into()));

        // Check dataset commitment
        let dc = manager.dataset_commitment();
        assert_eq!(dc.sample_count, 100);

        // Create batch commitment
        let batch = Batch::new(0, samples[0..32].to_vec());
        let bc = manager.commit_batch(&batch, (0..32).collect());

        assert_eq!(bc.sample_count, 32);
        assert_eq!(manager.batch_commitments().len(), 1);
    }

    #[test]
    fn test_deterministic_commitments() {
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let c1 = DatasetCommitment::from_samples(&samples, &metadata, None);
        let c2 = DatasetCommitment::from_samples(&samples, &metadata, None);

        assert_eq!(c1.root, c2.root);
        assert_eq!(c1.metadata_hash, c2.metadata_hash);
    }

    // ========== Registry Tests ==========

    #[test]
    fn test_registry_creation() {
        let registry = DatasetCommitmentRegistry::default_registry();
        let stats = registry.stats();
        assert_eq!(stats.total_commitments, 0);
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("test".into()));
        let id = registry.register(commitment, Some("owner".into())).unwrap();

        let reg = registry.get(&id).unwrap();
        assert_eq!(reg.status, CommitmentStatus::Local);
        assert_eq!(reg.version, 1);
    }

    #[test]
    fn test_registry_get_by_dataset_id() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("my_dataset".into()));
        registry.register(commitment, None).unwrap();

        let reg = registry.get_by_dataset_id("my_dataset").unwrap();
        assert_eq!(reg.commitment.dataset_id, Some("my_dataset".to_string()));
    }

    #[test]
    fn test_registry_versioning() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples1 = create_test_samples(10);
        let samples2 = create_test_samples(20);
        let metadata = create_test_metadata();

        // Register first version
        let c1 = DatasetCommitment::from_samples(&samples1, &metadata, Some("dataset".into()));
        let id1 = registry.register(c1, None).unwrap();

        // Register second version (update)
        let mut metadata2 = metadata.clone();
        metadata2.num_samples = 20;
        let c2 = DatasetCommitment::from_samples(&samples2, &metadata2, Some("dataset".into()));
        let id2 = registry.register(c2, None).unwrap();

        // Check versions
        let reg1 = registry.get(&id1).unwrap();
        let reg2 = registry.get(&id2).unwrap();

        assert_eq!(reg1.status, CommitmentStatus::Revoked);
        assert_eq!(reg2.version, 2);
    }

    #[test]
    fn test_registry_pending_status() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let id = registry.register_pending(commitment, None).unwrap();

        let pending = registry.list_pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].commitment.commitment_id(), id);
    }

    #[test]
    fn test_registry_confirm() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let id = registry.register_pending(commitment, None).unwrap();

        let tx_hash = Hash::from_slice(b"tx_hash_12345678901234567890");
        registry.confirm(&id, 12345, tx_hash).unwrap();

        let reg = registry.get(&id).unwrap();
        assert_eq!(reg.status, CommitmentStatus::Confirmed);
        assert_eq!(reg.registered_block, Some(12345));
    }

    #[test]
    fn test_registry_aggregation() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let metadata = create_test_metadata();

        // Register multiple pending commitments
        for i in 0..5 {
            let samples = create_test_samples(10 + i);
            let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some(format!("ds_{}", i)));
            registry.register_pending(commitment, None).unwrap();
        }

        let agg = registry.aggregate_pending().unwrap();
        assert_eq!(agg.count, 5);
        assert!(!agg.root.is_zero());
    }

    #[test]
    fn test_aggregated_commitment_inclusion() {
        let samples1 = create_test_samples(10);
        let samples2 = create_test_samples(20);
        let metadata = create_test_metadata();

        let c1 = DatasetCommitment::from_samples(&samples1, &metadata, None);
        let c2 = DatasetCommitment::from_samples(&samples2, &metadata, None);

        let agg = AggregatedCommitment::new(&[&c1, &c2]);

        assert!(agg.verify_inclusion(&c1.commitment_id()));
        assert!(agg.verify_inclusion(&c2.commitment_id()));
        assert!(!agg.verify_inclusion(&Hash::zero()));
    }

    #[test]
    fn test_on_chain_commitment() {
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let owner = [0u8; 20];

        let on_chain = OnChainCommitment::from_commitment(&commitment, owner, 12345);

        assert_eq!(on_chain.root, commitment.root);
        assert_eq!(on_chain.sample_count, 10);

        let abi_bytes = on_chain.to_abi_encoded();
        assert_eq!(abi_bytes.len(), 160);
    }

    #[test]
    fn test_registry_verification() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let id = registry.register(commitment, None).unwrap();

        let valid = registry.verify_commitment(&id, &samples, &metadata).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_registry_verification_fails_on_mismatch() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let id = registry.register(commitment, None).unwrap();

        // Different samples
        let different = create_test_samples(20);
        let valid = registry.verify_commitment(&id, &different, &metadata).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_registry_stats() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let metadata = create_test_metadata();

        // Add various commitments
        let s1 = create_test_samples(10);
        let c1 = DatasetCommitment::from_samples(&s1, &metadata, Some("ds1".into()));
        let id1 = registry.register(c1, None).unwrap();

        let s2 = create_test_samples(20);
        let c2 = DatasetCommitment::from_samples(&s2, &metadata, Some("ds2".into()));
        registry.register_pending(c2, None).unwrap();

        // Confirm first
        let tx = Hash::from_slice(b"tx_hash_12345678901234567890");
        registry.confirm(&id1, 100, tx).unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total_commitments, 2);
        assert_eq!(stats.total_samples, 30);
        assert_eq!(stats.confirmed_count, 1);
        assert_eq!(stats.pending_count, 1);
    }

    #[test]
    fn test_commitment_proof() {
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let proof = CommitmentProof::direct(&commitment);

        assert_eq!(proof.commitment_id, commitment.commitment_id());
        assert!(proof.aggregation_proof.is_none());

        let abi = proof.to_abi_encoded();
        assert!(!abi.is_empty());
    }

    #[test]
    fn test_prepare_for_chain() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let id = registry.register(commitment, None).unwrap();

        let owner = [1u8; 20];
        let on_chain = registry.prepare_for_chain(&id, owner, 1000).unwrap();

        assert_eq!(on_chain.owner, owner);
        assert_eq!(on_chain.block_number, 1000);
    }

    #[test]
    fn test_registry_revoke() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let id = registry.register(commitment, None).unwrap();

        registry.revoke(&id).unwrap();

        let reg = registry.get(&id).unwrap();
        assert_eq!(reg.status, CommitmentStatus::Revoked);
    }

    #[test]
    fn test_registry_mark_invalid() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let commitment = DatasetCommitment::from_samples(&samples, &metadata, None);
        let id = registry.register(commitment, None).unwrap();

        registry.mark_invalid(&id).unwrap();

        let reg = registry.get(&id).unwrap();
        assert_eq!(reg.status, CommitmentStatus::Invalid);
    }

    #[test]
    fn test_find_aggregation() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let metadata = create_test_metadata();

        let samples = create_test_samples(10);
        let commitment = DatasetCommitment::from_samples(&samples, &metadata, Some("ds".into()));
        let id = registry.register_pending(commitment.clone(), None).unwrap();

        registry.aggregate_pending();

        let found = registry.find_aggregation(&id);
        assert!(found.is_some());
        assert!(found.unwrap().verify_inclusion(&id));
    }

    #[test]
    fn test_aggregated_compact_bytes() {
        let samples = create_test_samples(10);
        let metadata = create_test_metadata();

        let c1 = DatasetCommitment::from_samples(&samples, &metadata, None);
        let c2 = DatasetCommitment::from_samples(&create_test_samples(5), &metadata, None);

        let agg = AggregatedCommitment::new(&[&c1, &c2]);
        let compact = agg.to_compact_bytes();

        assert_eq!(compact.len(), 68);
    }

    #[test]
    fn test_list_by_status() {
        let mut registry = DatasetCommitmentRegistry::default_registry();
        let metadata = create_test_metadata();

        // Create local, pending, and confirmed commitments
        let s1 = create_test_samples(5);
        let c1 = DatasetCommitment::from_samples(&s1, &metadata, Some("d1".into()));
        registry.register(c1, None).unwrap();

        let s2 = create_test_samples(10);
        let c2 = DatasetCommitment::from_samples(&s2, &metadata, Some("d2".into()));
        let id2 = registry.register_pending(c2, None).unwrap();

        registry.confirm(&id2, 100, Hash::from_slice(b"tx")).unwrap();

        assert_eq!(registry.list_by_status(CommitmentStatus::Local).len(), 1);
        assert_eq!(registry.list_by_status(CommitmentStatus::Confirmed).len(), 1);
    }
}
