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
            builder = builder.add_leaf(&SampleCommitment::hash_sample(sample));
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
        Ok::<(), ()>(()).unwrap();
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
}
