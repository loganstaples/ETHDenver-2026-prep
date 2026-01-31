//! Batch Membership Proofs for HELIX.
//!
//! Provides efficient proof generation and verification for batch membership
//! in committed datasets. Supports:
//! - Single sample membership proofs
//! - Batch membership proofs (aggregated)
//! - Multi-batch verification
//! - Proof compression and serialization
//! - On-chain verification optimization

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};

use super::commitment::{BatchCommitment, CommitmentError, DatasetCommitment, SampleCommitment};
use super::dataset::{Batch, Sample};
use super::merkle::{
    Hash, MerkleProof, MerkleTree, MultiProof, Sha256Hasher, TreePosition, HASH_SIZE,
};
use crate::traits::BinarySerializable;
use crate::traits::serializable::SerializeError;

/// A proof that a sample belongs to a committed dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMembershipProof {
    /// The sample being proven.
    pub sample_commitment: SampleCommitment,
    /// Index of the sample in the dataset.
    pub sample_index: usize,
    /// Merkle proof from leaf to root.
    pub merkle_proof: MerkleProof,
    /// The dataset commitment this proves membership in.
    pub dataset_root: Hash,
}

impl SampleMembershipProof {
    /// Creates a membership proof for a sample.
    pub fn new(
        sample: &Sample,
        tree: &MerkleTree<Sha256Hasher>,
        dataset_commitment: &DatasetCommitment,
    ) -> Result<Self, MembershipProofError> {
        let sample_commitment = SampleCommitment::new(sample);

        // Verify sample is in tree
        let stored_hash = tree
            .get_leaf(sample.id)
            .ok_or_else(|| MembershipProofError::SampleNotInDataset(sample.id))?;

        if stored_hash != sample_commitment.hash {
            return Err(MembershipProofError::SampleMismatch {
                expected: stored_hash,
                actual: sample_commitment.hash,
            });
        }

        let merkle_proof = tree
            .prove(sample.id)
            .map_err(|e| MembershipProofError::MerkleError(format!("{}", e)))?;

        Ok(Self {
            sample_commitment,
            sample_index: sample.id,
            merkle_proof,
            dataset_root: dataset_commitment.root,
        })
    }

    /// Verifies this proof against a dataset commitment.
    pub fn verify(&self, dataset_commitment: &DatasetCommitment) -> bool {
        // Check root matches
        if self.dataset_root != dataset_commitment.root {
            return false;
        }

        // Check Merkle proof
        self.merkle_proof.verify_with_root(&Sha256Hasher, &dataset_commitment.root)
    }

    /// Verifies this proof against a root hash.
    pub fn verify_with_root(&self, root: &Hash) -> bool {
        if self.dataset_root != *root {
            return false;
        }
        self.merkle_proof.verify_with_root(&Sha256Hasher, root)
    }

    /// Returns the sample hash being proven.
    pub fn sample_hash(&self) -> Hash {
        self.sample_commitment.hash
    }
}

impl BinarySerializable for SampleMembershipProof {
    fn serialized_size(&self) -> usize {
        self.sample_commitment.serialized_size()
            + 8
            + self.merkle_proof.serialized_size()
            + HASH_SIZE
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        BinarySerializable::serialize(&self.sample_commitment, writer)?;
        writer.write_all(&(self.sample_index as u64).to_le_bytes())?;
        BinarySerializable::serialize(&self.merkle_proof, writer)?;
        BinarySerializable::serialize(&self.dataset_root, writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let sample_commitment = <SampleCommitment as BinarySerializable>::deserialize(reader)?;

        let mut idx_buf = [0u8; 8];
        reader.read_exact(&mut idx_buf)?;
        let sample_index = u64::from_le_bytes(idx_buf) as usize;

        let merkle_proof = <MerkleProof as BinarySerializable>::deserialize(reader)?;
        let dataset_root = <Hash as BinarySerializable>::deserialize(reader)?;

        Ok(Self {
            sample_commitment,
            sample_index,
            merkle_proof,
            dataset_root,
        })
    }
}

/// A proof that multiple samples belong to a committed dataset.
///
/// Uses optimized multi-proof structure to reduce redundant sibling nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMembershipProof {
    /// Sample commitments for all samples in the batch.
    pub sample_commitments: Vec<SampleCommitment>,
    /// Sample indices in the dataset.
    pub sample_indices: Vec<usize>,
    /// Aggregated proof nodes (deduped siblings).
    pub proof_nodes: HashMap<TreePosition, Hash>,
    /// Tree height.
    pub tree_height: usize,
    /// The dataset root hash.
    pub dataset_root: Hash,
    /// Optional batch commitment for additional verification.
    pub batch_commitment: Option<BatchCommitment>,
}

impl BatchMembershipProof {
    /// Creates a batch membership proof.
    pub fn new(
        samples: &[Sample],
        tree: &MerkleTree<Sha256Hasher>,
        dataset_commitment: &DatasetCommitment,
    ) -> Result<Self, MembershipProofError> {
        if samples.is_empty() {
            return Err(MembershipProofError::EmptyBatch);
        }

        let sample_commitments: Vec<SampleCommitment> =
            samples.iter().map(SampleCommitment::new).collect();
        let sample_indices: Vec<usize> = samples.iter().map(|s| s.id).collect();

        // Verify all samples are in tree
        for (i, sc) in sample_commitments.iter().enumerate() {
            let stored_hash = tree.get_leaf(sample_indices[i]).ok_or_else(|| {
                MembershipProofError::SampleNotInDataset(sample_indices[i])
            })?;

            if stored_hash != sc.hash {
                return Err(MembershipProofError::SampleMismatch {
                    expected: stored_hash,
                    actual: sc.hash,
                });
            }
        }

        // Generate multi-proof
        let multi_proof = tree
            .prove_multi(&sample_indices)
            .map_err(|e| MembershipProofError::MerkleError(format!("{}", e)))?;

        Ok(Self {
            sample_commitments,
            sample_indices,
            proof_nodes: multi_proof.proof_nodes,
            tree_height: tree.height(),
            dataset_root: dataset_commitment.root,
            batch_commitment: None,
        })
    }

    /// Creates a batch membership proof with batch commitment.
    pub fn with_batch_commitment(
        batch: &Batch,
        tree: &MerkleTree<Sha256Hasher>,
        dataset_commitment: &DatasetCommitment,
    ) -> Result<Self, MembershipProofError> {
        let sample_indices: Vec<usize> = batch.samples.iter().map(|s| s.id).collect();
        let batch_commitment =
            BatchCommitment::new(batch, sample_indices.clone(), dataset_commitment.commitment_id());

        let mut proof = Self::new(&batch.samples, tree, dataset_commitment)?;
        proof.batch_commitment = Some(batch_commitment);
        Ok(proof)
    }

    /// Verifies this proof against a dataset commitment.
    pub fn verify(&self, dataset_commitment: &DatasetCommitment) -> bool {
        if self.dataset_root != dataset_commitment.root {
            return false;
        }

        // Reconstruct the multi-proof and verify
        let multi = MultiProof {
            leaf_indices: self.sample_indices.clone(),
            leaf_hashes: self.sample_commitments.iter().map(|sc| sc.hash).collect(),
            proof_nodes: self.proof_nodes.clone(),
            root: dataset_commitment.root,
            height: self.tree_height,
        };

        multi.verify(&Sha256Hasher)
    }

    /// Returns the number of samples in this proof.
    pub fn sample_count(&self) -> usize {
        self.sample_indices.len()
    }

    /// Calculates the compression ratio compared to individual proofs.
    pub fn compression_ratio(&self) -> f64 {
        let individual_nodes = self.sample_indices.len() * self.tree_height;
        let compressed_nodes = self.proof_nodes.len();
        if compressed_nodes == 0 {
            return 0.0;
        }
        1.0 - (compressed_nodes as f64 / individual_nodes as f64)
    }
}

impl BinarySerializable for BatchMembershipProof {
    fn serialized_size(&self) -> usize {
        // sample_commitments length + commitments + indices length + indices +
        // proof_nodes length + nodes + tree_height + dataset_root + batch_commitment flag + optional batch
        let commitment_size: usize = self
            .sample_commitments
            .iter()
            .map(|c| c.serialized_size())
            .sum();
        let node_size = self.proof_nodes.len() * (8 + 8 + HASH_SIZE);
        let batch_size = self
            .batch_commitment
            .as_ref()
            .map(|b| b.serialized_size())
            .unwrap_or(0);

        4 + commitment_size
            + 4
            + self.sample_indices.len() * 8
            + 4
            + node_size
            + 8
            + HASH_SIZE
            + 1
            + batch_size
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        // Sample commitments
        writer.write_all(&(self.sample_commitments.len() as u32).to_le_bytes())?;
        for sc in &self.sample_commitments {
            BinarySerializable::serialize(sc, writer)?;
        }

        // Sample indices
        writer.write_all(&(self.sample_indices.len() as u32).to_le_bytes())?;
        for &idx in &self.sample_indices {
            writer.write_all(&(idx as u64).to_le_bytes())?;
        }

        // Proof nodes
        writer.write_all(&(self.proof_nodes.len() as u32).to_le_bytes())?;
        for (pos, hash) in &self.proof_nodes {
            writer.write_all(&(pos.level as u64).to_le_bytes())?;
            writer.write_all(&(pos.index as u64).to_le_bytes())?;
            BinarySerializable::serialize(hash, writer)?;
        }

        // Tree height and root
        writer.write_all(&(self.tree_height as u64).to_le_bytes())?;
        BinarySerializable::serialize(&self.dataset_root, writer)?;

        // Batch commitment
        match &self.batch_commitment {
            Some(bc) => {
                writer.write_all(&[1u8])?;
                BinarySerializable::serialize(bc, writer)?;
            }
            None => {
                writer.write_all(&[0u8])?;
            }
        }

        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        // Sample commitments
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let sc_len = u32::from_le_bytes(len_buf) as usize;
        let mut sample_commitments = Vec::with_capacity(sc_len);
        for _ in 0..sc_len {
            sample_commitments.push(<SampleCommitment as BinarySerializable>::deserialize(reader)?);
        }

        // Sample indices
        reader.read_exact(&mut len_buf)?;
        let si_len = u32::from_le_bytes(len_buf) as usize;
        let mut sample_indices = Vec::with_capacity(si_len);
        for _ in 0..si_len {
            let mut idx_buf = [0u8; 8];
            reader.read_exact(&mut idx_buf)?;
            sample_indices.push(u64::from_le_bytes(idx_buf) as usize);
        }

        // Proof nodes
        reader.read_exact(&mut len_buf)?;
        let pn_len = u32::from_le_bytes(len_buf) as usize;
        let mut proof_nodes = HashMap::with_capacity(pn_len);
        for _ in 0..pn_len {
            let mut level_buf = [0u8; 8];
            reader.read_exact(&mut level_buf)?;
            let level = u64::from_le_bytes(level_buf) as usize;

            let mut index_buf = [0u8; 8];
            reader.read_exact(&mut index_buf)?;
            let index = u64::from_le_bytes(index_buf) as usize;

            let hash = <Hash as BinarySerializable>::deserialize(reader)?;
            proof_nodes.insert(TreePosition { level, index }, hash);
        }

        // Tree height and root
        let mut height_buf = [0u8; 8];
        reader.read_exact(&mut height_buf)?;
        let tree_height = u64::from_le_bytes(height_buf) as usize;

        let dataset_root = <Hash as BinarySerializable>::deserialize(reader)?;

        // Batch commitment
        let mut flag_buf = [0u8; 1];
        reader.read_exact(&mut flag_buf)?;
        let batch_commitment = if flag_buf[0] == 1 {
            Some(<BatchCommitment as BinarySerializable>::deserialize(reader)?)
        } else {
            None
        };

        Ok(Self {
            sample_commitments,
            sample_indices,
            proof_nodes,
            tree_height,
            dataset_root,
            batch_commitment,
        })
    }
}

/// Aggregated proof for multiple batches.
///
/// Optimized for verifying many training steps against the same dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedBatchProof {
    /// All batch proofs.
    pub batch_proofs: Vec<BatchMembershipProof>,
    /// Combined unique proof nodes across all batches.
    pub shared_proof_nodes: HashMap<TreePosition, Hash>,
    /// The dataset root.
    pub dataset_root: Hash,
    /// Tree height.
    pub tree_height: usize,
    /// Total unique samples proven.
    pub unique_sample_count: usize,
}

impl AggregatedBatchProof {
    /// Creates an aggregated proof from multiple batch proofs.
    pub fn aggregate(proofs: Vec<BatchMembershipProof>) -> Result<Self, MembershipProofError> {
        if proofs.is_empty() {
            return Err(MembershipProofError::EmptyBatch);
        }

        let dataset_root = proofs[0].dataset_root;
        let tree_height = proofs[0].tree_height;

        // Verify all proofs are for the same dataset
        for proof in &proofs {
            if proof.dataset_root != dataset_root {
                return Err(MembershipProofError::InconsistentRoots);
            }
        }

        // Collect unique proof nodes
        let mut shared_proof_nodes: HashMap<TreePosition, Hash> = HashMap::new();
        let mut unique_samples: HashSet<usize> = HashSet::new();

        for proof in &proofs {
            for (&pos, &hash) in &proof.proof_nodes {
                shared_proof_nodes.entry(pos).or_insert(hash);
            }
            for &idx in &proof.sample_indices {
                unique_samples.insert(idx);
            }
        }

        Ok(Self {
            batch_proofs: proofs,
            shared_proof_nodes,
            dataset_root,
            tree_height,
            unique_sample_count: unique_samples.len(),
        })
    }

    /// Verifies all batches.
    pub fn verify(&self, dataset_commitment: &DatasetCommitment) -> bool {
        if self.dataset_root != dataset_commitment.root {
            return false;
        }

        for proof in &self.batch_proofs {
            if !proof.verify(dataset_commitment) {
                return false;
            }
        }

        true
    }

    /// Returns the number of batches in this aggregated proof.
    pub fn batch_count(&self) -> usize {
        self.batch_proofs.len()
    }

    /// Returns overall compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        let individual_nodes: usize = self
            .batch_proofs
            .iter()
            .map(|p| p.sample_indices.len() * self.tree_height)
            .sum();
        let compressed_nodes = self.shared_proof_nodes.len();
        if compressed_nodes == 0 {
            return 0.0;
        }
        1.0 - (compressed_nodes as f64 / individual_nodes as f64)
    }
}

/// A verifier for batch membership proofs.
///
/// Caches dataset commitments and tree structures for efficient repeated verification.
pub struct MembershipVerifier {
    /// Cached dataset commitments.
    commitments: HashMap<Hash, DatasetCommitment>,
    /// Cached trees (for proof generation).
    trees: HashMap<Hash, MerkleTree<Sha256Hasher>>,
    /// Verification statistics.
    stats: VerificationStats,
}

/// Statistics for verification operations.
#[derive(Debug, Clone, Default)]
pub struct VerificationStats {
    /// Total proofs verified.
    pub proofs_verified: u64,
    /// Successful verifications.
    pub successful: u64,
    /// Failed verifications.
    pub failed: u64,
    /// Total samples verified.
    pub samples_verified: u64,
}

impl MembershipVerifier {
    /// Creates a new verifier.
    pub fn new() -> Self {
        Self {
            commitments: HashMap::new(),
            trees: HashMap::new(),
            stats: VerificationStats::default(),
        }
    }

    /// Registers a dataset for verification.
    pub fn register_dataset(
        &mut self,
        commitment: DatasetCommitment,
        tree: MerkleTree<Sha256Hasher>,
    ) {
        let id = commitment.commitment_id();
        self.commitments.insert(id, commitment);
        self.trees.insert(id, tree);
    }

    /// Verifies a sample membership proof.
    pub fn verify_sample(&mut self, proof: &SampleMembershipProof) -> Result<bool, MembershipProofError> {
        let commitment = self
            .commitments
            .values()
            .find(|c| c.root == proof.dataset_root)
            .ok_or(MembershipProofError::UnknownDataset(proof.dataset_root))?;

        self.stats.proofs_verified += 1;
        self.stats.samples_verified += 1;

        let valid = proof.verify(commitment);
        if valid {
            self.stats.successful += 1;
        } else {
            self.stats.failed += 1;
        }

        Ok(valid)
    }

    /// Verifies a batch membership proof.
    pub fn verify_batch(
        &mut self,
        proof: &BatchMembershipProof,
    ) -> Result<bool, MembershipProofError> {
        let commitment = self
            .commitments
            .values()
            .find(|c| c.root == proof.dataset_root)
            .ok_or(MembershipProofError::UnknownDataset(proof.dataset_root))?;

        self.stats.proofs_verified += 1;
        self.stats.samples_verified += proof.sample_count() as u64;

        let valid = proof.verify(commitment);
        if valid {
            self.stats.successful += 1;
        } else {
            self.stats.failed += 1;
        }

        Ok(valid)
    }

    /// Verifies a sample against a specific root (no registration needed).
    pub fn verify_sample_with_root(
        &mut self,
        proof: &SampleMembershipProof,
        root: &Hash,
    ) -> bool {
        self.stats.proofs_verified += 1;
        self.stats.samples_verified += 1;

        let valid = proof.verify_with_root(root);
        if valid {
            self.stats.successful += 1;
        } else {
            self.stats.failed += 1;
        }

        valid
    }

    /// Returns verification statistics.
    pub fn stats(&self) -> &VerificationStats {
        &self.stats
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.stats = VerificationStats::default();
    }
}

impl Default for MembershipVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof generator for efficient batch proof creation.
pub struct MembershipProofGenerator {
    /// The Merkle tree.
    tree: MerkleTree<Sha256Hasher>,
    /// The dataset commitment.
    commitment: DatasetCommitment,
    /// Cached sample commitments.
    sample_commitments: Vec<SampleCommitment>,
}

impl MembershipProofGenerator {
    /// Creates a new proof generator.
    pub fn new(
        tree: MerkleTree<Sha256Hasher>,
        commitment: DatasetCommitment,
        samples: &[Sample],
    ) -> Self {
        let sample_commitments = samples.iter().map(SampleCommitment::new).collect();

        Self {
            tree,
            commitment,
            sample_commitments,
        }
    }

    /// Generates a proof for a single sample by index.
    pub fn prove_sample(&self, index: usize) -> Result<SampleMembershipProof, MembershipProofError> {
        let sc = self
            .sample_commitments
            .get(index)
            .ok_or_else(|| MembershipProofError::SampleNotInDataset(index))?
            .clone();

        let merkle_proof = self
            .tree
            .prove(index)
            .map_err(|e| MembershipProofError::MerkleError(format!("{}", e)))?;

        Ok(SampleMembershipProof {
            sample_commitment: sc,
            sample_index: index,
            merkle_proof,
            dataset_root: self.commitment.root,
        })
    }

    /// Generates proofs for multiple samples.
    pub fn prove_samples(
        &self,
        indices: &[usize],
    ) -> Result<Vec<SampleMembershipProof>, MembershipProofError> {
        indices.iter().map(|&i| self.prove_sample(i)).collect()
    }

    /// Generates a batch proof.
    pub fn prove_batch(&self, indices: &[usize]) -> Result<BatchMembershipProof, MembershipProofError> {
        if indices.is_empty() {
            return Err(MembershipProofError::EmptyBatch);
        }

        let sample_commitments: Result<Vec<_>, _> = indices
            .iter()
            .map(|&i| {
                self.sample_commitments
                    .get(i)
                    .cloned()
                    .ok_or_else(|| MembershipProofError::SampleNotInDataset(i))
            })
            .collect();

        let sample_commitments = sample_commitments?;

        let multi_proof = self
            .tree
            .prove_multi(indices)
            .map_err(|e| MembershipProofError::MerkleError(format!("{}", e)))?;

        Ok(BatchMembershipProof {
            sample_commitments,
            sample_indices: indices.to_vec(),
            proof_nodes: multi_proof.proof_nodes,
            tree_height: self.tree.height(),
            dataset_root: self.commitment.root,
            batch_commitment: None,
        })
    }

    /// Returns the dataset commitment.
    pub fn commitment(&self) -> &DatasetCommitment {
        &self.commitment
    }

    /// Returns the tree.
    pub fn tree(&self) -> &MerkleTree<Sha256Hasher> {
        &self.tree
    }
}

/// Errors for membership proof operations.
#[derive(Debug, Clone)]
pub enum MembershipProofError {
    /// Sample not found in dataset.
    SampleNotInDataset(usize),
    /// Sample hash mismatch.
    SampleMismatch { expected: Hash, actual: Hash },
    /// Empty batch provided.
    EmptyBatch,
    /// Merkle tree error.
    MerkleError(String),
    /// Unknown dataset.
    UnknownDataset(Hash),
    /// Inconsistent roots across proofs.
    InconsistentRoots,
    /// Verification failed.
    VerificationFailed,
}

impl std::fmt::Display for MembershipProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MembershipProofError::SampleNotInDataset(idx) => {
                write!(f, "Sample {} not in dataset", idx)
            }
            MembershipProofError::SampleMismatch { expected, actual } => {
                write!(f, "Sample mismatch: expected {}, got {}", expected, actual)
            }
            MembershipProofError::EmptyBatch => write!(f, "Empty batch"),
            MembershipProofError::MerkleError(e) => write!(f, "Merkle error: {}", e),
            MembershipProofError::UnknownDataset(h) => write!(f, "Unknown dataset: {}", h),
            MembershipProofError::InconsistentRoots => write!(f, "Inconsistent roots"),
            MembershipProofError::VerificationFailed => write!(f, "Verification failed"),
        }
    }
}

impl std::error::Error for MembershipProofError {}

impl From<CommitmentError> for MembershipProofError {
    fn from(e: CommitmentError) -> Self {
        match e {
            CommitmentError::MerkleError(s) => MembershipProofError::MerkleError(s),
            CommitmentError::SampleNotFound(i) => MembershipProofError::SampleNotInDataset(i),
            _ => MembershipProofError::VerificationFailed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataType, DatasetMetadata};
    use std::collections::HashMap as StdHashMap;

    fn create_test_samples(count: usize) -> Vec<Sample> {
        (0..count)
            .map(|i| Sample::from_f32(i, vec![i as f32; 4], vec![0.0, 1.0]))
            .collect()
    }

    fn create_test_metadata(count: usize) -> DatasetMetadata {
        DatasetMetadata {
            name: "test".to_string(),
            num_samples: count,
            feature_dims: vec![4],
            label_dims: vec![2],
            dtype: DataType::Float32,
            extra: StdHashMap::new(),
        }
    }

    fn setup_test_tree(samples: &[Sample]) -> (MerkleTree<Sha256Hasher>, DatasetCommitment) {
        use super::super::merkle::{Hash, MerkleTreeBuilder};

        let mut builder = MerkleTreeBuilder::with_sha256();
        for sample in samples {
            // Use add_hash since hash_sample returns a hash already
            let hash = Hash::from_bytes(SampleCommitment::hash_sample(sample));
            builder = builder.add_hash(hash);
        }
        let tree = builder.build().unwrap();
        let metadata = create_test_metadata(samples.len());
        let commitment = DatasetCommitment::new(&tree, &metadata, None);

        (tree, commitment)
    }

    #[test]
    fn test_sample_membership_proof() {
        let samples = create_test_samples(16);
        let (tree, commitment) = setup_test_tree(&samples);

        let proof = SampleMembershipProof::new(&samples[5], &tree, &commitment).unwrap();

        assert!(proof.verify(&commitment));
        assert_eq!(proof.sample_index, 5);
    }

    #[test]
    fn test_sample_membership_proof_serialization() {
        let samples = create_test_samples(16);
        let (tree, commitment) = setup_test_tree(&samples);

        let proof = SampleMembershipProof::new(&samples[5], &tree, &commitment).unwrap();
        let bytes = proof.to_bytes();
        let restored = SampleMembershipProof::from_bytes(&bytes).unwrap();

        assert!(restored.verify(&commitment));
        assert_eq!(restored.sample_index, proof.sample_index);
    }

    #[test]
    fn test_batch_membership_proof() {
        let samples = create_test_samples(64);
        let (tree, commitment) = setup_test_tree(&samples);

        let batch_samples: Vec<Sample> = (0..8).map(|i| samples[i].clone()).collect();
        let proof = BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap();

        assert!(proof.verify(&commitment));
        assert_eq!(proof.sample_count(), 8);
    }

    #[test]
    fn test_batch_membership_proof_compression() {
        let samples = create_test_samples(256);
        let (tree, commitment) = setup_test_tree(&samples);

        // Contiguous batch should have good compression
        let batch_samples: Vec<Sample> = (0..32).map(|i| samples[i].clone()).collect();
        let proof = BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap();

        let ratio = proof.compression_ratio();
        assert!(ratio > 0.0, "Should have some compression");
    }

    #[test]
    fn test_batch_membership_proof_serialization() {
        let samples = create_test_samples(64);
        let (tree, commitment) = setup_test_tree(&samples);

        let batch_samples: Vec<Sample> = (0..8).map(|i| samples[i].clone()).collect();
        let proof = BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap();

        let bytes = proof.to_bytes();
        let restored = BatchMembershipProof::from_bytes(&bytes).unwrap();

        assert!(restored.verify(&commitment));
        assert_eq!(restored.sample_count(), proof.sample_count());
    }

    #[test]
    fn test_aggregated_batch_proof() {
        let samples = create_test_samples(256);
        let (tree, commitment) = setup_test_tree(&samples);

        // Create multiple batch proofs
        let batch1: Vec<Sample> = (0..32).map(|i| samples[i].clone()).collect();
        let batch2: Vec<Sample> = (32..64).map(|i| samples[i].clone()).collect();
        let batch3: Vec<Sample> = (64..96).map(|i| samples[i].clone()).collect();

        let proof1 = BatchMembershipProof::new(&batch1, &tree, &commitment).unwrap();
        let proof2 = BatchMembershipProof::new(&batch2, &tree, &commitment).unwrap();
        let proof3 = BatchMembershipProof::new(&batch3, &tree, &commitment).unwrap();

        let aggregated =
            AggregatedBatchProof::aggregate(vec![proof1, proof2, proof3]).unwrap();

        assert!(aggregated.verify(&commitment));
        assert_eq!(aggregated.batch_count(), 3);
        assert_eq!(aggregated.unique_sample_count, 96);
    }

    #[test]
    fn test_membership_verifier() {
        let samples = create_test_samples(64);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut verifier = MembershipVerifier::new();
        verifier.register_dataset(commitment.clone(), tree.clone());

        let proof = SampleMembershipProof::new(&samples[10], &tree, &commitment).unwrap();

        let result = verifier.verify_sample(&proof);
        assert!(result.unwrap());
        assert_eq!(verifier.stats().successful, 1);
    }

    #[test]
    fn test_membership_proof_generator() {
        let samples = create_test_samples(128);
        let (tree, commitment) = setup_test_tree(&samples);

        let generator = MembershipProofGenerator::new(tree, commitment.clone(), &samples);

        // Single proof
        let proof = generator.prove_sample(50).unwrap();
        assert!(proof.verify(&commitment));

        // Batch proof
        let batch_proof = generator.prove_batch(&[10, 20, 30, 40, 50]).unwrap();
        assert!(batch_proof.verify(&commitment));
    }

    #[test]
    fn test_invalid_sample_proof() {
        let samples = create_test_samples(16);
        let (tree, commitment) = setup_test_tree(&samples);

        // Modify a sample
        let mut fake_sample = samples[5].clone();
        fake_sample.features = vec![0u8; 16];

        let result = SampleMembershipProof::new(&fake_sample, &tree, &commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_against_wrong_root() {
        let samples = create_test_samples(16);
        let (tree, commitment) = setup_test_tree(&samples);

        let proof = SampleMembershipProof::new(&samples[5], &tree, &commitment).unwrap();

        // Create a different dataset with different values
        let other_samples: Vec<Sample> = (0..16)
            .map(|i| Sample::from_f32(i, vec![(i + 100) as f32; 4], vec![1.0, 0.0]))
            .collect();
        let (_, other_commitment) = setup_test_tree(&other_samples);

        // Should fail verification against different root
        assert!(!proof.verify(&other_commitment));
    }
}
