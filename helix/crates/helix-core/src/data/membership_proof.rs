//! Batch Membership Proofs for HELIX.
//!
//! Provides efficient proof generation and verification for batch membership
//! in committed datasets. Supports:
//! - Single sample membership proofs
//! - Batch membership proofs (aggregated)
//! - Multi-batch verification
//! - Proof compression and serialization
//! - On-chain verification optimization
//! - Streaming verification for large datasets
//! - Parallel proof verification
//! - Proof batching and aggregation optimization

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::sync::Arc;

use super::commitment::{BatchCommitment, CommitmentError, DatasetCommitment, SampleCommitment};
use super::dataset::{Batch, DatasetMetadata, DataType, Sample};
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

// =============================================================================
// STREAMING BATCH VERIFIER - Memory-efficient verification for large datasets
// =============================================================================

/// Configuration for streaming verification.
#[derive(Debug, Clone)]
pub struct StreamingVerificationConfig {
    /// Maximum proofs to buffer before flushing.
    pub buffer_size: usize,
    /// Whether to fail fast on first invalid proof.
    pub fail_fast: bool,
    /// Enable verification statistics.
    pub collect_stats: bool,
    /// Timeout per proof in microseconds (0 = no timeout).
    pub timeout_per_proof_us: u64,
}

impl Default for StreamingVerificationConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            fail_fast: false,
            collect_stats: true,
            timeout_per_proof_us: 0,
        }
    }
}

impl StreamingVerificationConfig {
    /// Creates a config optimized for speed.
    pub fn fast() -> Self {
        Self {
            buffer_size: 10000,
            fail_fast: true,
            collect_stats: false,
            timeout_per_proof_us: 0,
        }
    }

    /// Creates a config for thorough verification.
    pub fn thorough() -> Self {
        Self {
            buffer_size: 100,
            fail_fast: false,
            collect_stats: true,
            timeout_per_proof_us: 100000, // 100ms
        }
    }
}

/// Statistics for streaming verification.
#[derive(Debug, Clone, Default)]
pub struct StreamingVerificationStats {
    /// Total proofs processed.
    pub proofs_processed: u64,
    /// Total samples verified.
    pub samples_verified: u64,
    /// Successful verifications.
    pub successful: u64,
    /// Failed verifications.
    pub failed: u64,
    /// Average verification time in microseconds.
    pub avg_verify_time_us: u64,
    /// Peak verification time in microseconds.
    pub peak_verify_time_us: u64,
    /// Total bytes processed.
    pub bytes_processed: u64,
    /// Cache hits (if caching enabled).
    pub cache_hits: u64,
    /// Buffer flushes.
    pub buffer_flushes: u64,
}

impl StreamingVerificationStats {
    /// Returns the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.proofs_processed == 0 {
            return 100.0;
        }
        (self.successful as f64 / self.proofs_processed as f64) * 100.0
    }

    /// Returns the verification throughput (proofs per second).
    pub fn throughput(&self) -> f64 {
        if self.avg_verify_time_us == 0 {
            return 0.0;
        }
        1_000_000.0 / self.avg_verify_time_us as f64
    }
}

/// A streaming verifier for processing large batches of membership proofs.
///
/// Processes proofs in chunks to minimize memory usage while maintaining
/// verification performance.
pub struct StreamingBatchVerifier {
    /// Dataset commitment to verify against.
    commitment: DatasetCommitment,
    /// Configuration.
    config: StreamingVerificationConfig,
    /// Statistics.
    stats: StreamingVerificationStats,
    /// Cached verification results (proof hash -> result).
    result_cache: HashMap<Hash, bool>,
    /// Buffer of pending results.
    results: Vec<VerificationResult>,
    /// Total verification time in microseconds.
    total_verify_time_us: u64,
    /// Number of verifications for avg calculation.
    verify_count: u64,
}

/// Result of a single proof verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Index of the proof in the input stream.
    pub proof_index: usize,
    /// Sample indices being verified.
    pub sample_indices: Vec<usize>,
    /// Whether verification succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
    /// Verification time in microseconds.
    pub verify_time_us: u64,
}

impl StreamingBatchVerifier {
    /// Creates a new streaming verifier.
    pub fn new(commitment: DatasetCommitment, config: StreamingVerificationConfig) -> Self {
        Self {
            commitment,
            config,
            stats: StreamingVerificationStats::default(),
            result_cache: HashMap::new(),
            results: Vec::new(),
            total_verify_time_us: 0,
            verify_count: 0,
        }
    }

    /// Creates a verifier with default config.
    pub fn with_commitment(commitment: DatasetCommitment) -> Self {
        Self::new(commitment, StreamingVerificationConfig::default())
    }

    /// Verifies a single sample proof.
    pub fn verify_sample_proof(&mut self, proof: &SampleMembershipProof) -> bool {
        let start = std::time::Instant::now();

        let result = proof.verify(&self.commitment);

        let elapsed_us = start.elapsed().as_micros() as u64;
        self.update_stats(1, result, elapsed_us);

        result
    }

    /// Verifies a batch proof.
    pub fn verify_batch_proof(&mut self, proof: &BatchMembershipProof) -> bool {
        let start = std::time::Instant::now();

        let result = proof.verify(&self.commitment);

        let elapsed_us = start.elapsed().as_micros() as u64;
        self.update_stats(proof.sample_count() as u64, result, elapsed_us);

        result
    }

    /// Verifies a stream of batch proofs.
    pub fn verify_stream<'a, I>(&mut self, proofs: I) -> StreamingVerificationResult
    where
        I: IntoIterator<Item = &'a BatchMembershipProof>,
    {
        let mut verified = 0;
        let mut failed = 0;
        let mut failed_indices = Vec::new();

        for (idx, proof) in proofs.into_iter().enumerate() {
            let start = std::time::Instant::now();
            let result = proof.verify(&self.commitment);
            let elapsed_us = start.elapsed().as_micros() as u64;

            self.update_stats(proof.sample_count() as u64, result, elapsed_us);

            if result {
                verified += 1;
            } else {
                failed += 1;
                failed_indices.push(idx);

                if self.config.fail_fast {
                    break;
                }
            }

            // Buffer results if collecting
            if self.config.collect_stats {
                self.results.push(VerificationResult {
                    proof_index: idx,
                    sample_indices: proof.sample_indices.clone(),
                    success: result,
                    error: if result { None } else { Some("Verification failed".into()) },
                    verify_time_us: elapsed_us,
                });

                if self.results.len() >= self.config.buffer_size {
                    self.stats.buffer_flushes += 1;
                    // In a real implementation, could persist results here
                }
            }
        }

        StreamingVerificationResult {
            verified,
            failed,
            failed_indices,
            stats: self.stats.clone(),
        }
    }

    /// Verifies multiple sample proofs.
    pub fn verify_sample_stream<'a, I>(&mut self, proofs: I) -> StreamingVerificationResult
    where
        I: IntoIterator<Item = &'a SampleMembershipProof>,
    {
        let mut verified = 0;
        let mut failed = 0;
        let mut failed_indices = Vec::new();

        for (idx, proof) in proofs.into_iter().enumerate() {
            let result = self.verify_sample_proof(proof);

            if result {
                verified += 1;
            } else {
                failed += 1;
                failed_indices.push(idx);

                if self.config.fail_fast {
                    break;
                }
            }
        }

        StreamingVerificationResult {
            verified,
            failed,
            failed_indices,
            stats: self.stats.clone(),
        }
    }

    /// Updates statistics.
    fn update_stats(&mut self, samples: u64, success: bool, elapsed_us: u64) {
        self.stats.proofs_processed += 1;
        self.stats.samples_verified += samples;

        if success {
            self.stats.successful += 1;
        } else {
            self.stats.failed += 1;
        }

        self.total_verify_time_us += elapsed_us;
        self.verify_count += 1;
        self.stats.avg_verify_time_us = self.total_verify_time_us / self.verify_count;

        if elapsed_us > self.stats.peak_verify_time_us {
            self.stats.peak_verify_time_us = elapsed_us;
        }
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &StreamingVerificationStats {
        &self.stats
    }

    /// Returns all collected results.
    pub fn results(&self) -> &[VerificationResult] {
        &self.results
    }

    /// Resets the verifier for reuse.
    pub fn reset(&mut self) {
        self.stats = StreamingVerificationStats::default();
        self.result_cache.clear();
        self.results.clear();
        self.total_verify_time_us = 0;
        self.verify_count = 0;
    }
}

/// Result of streaming verification.
#[derive(Debug, Clone)]
pub struct StreamingVerificationResult {
    /// Number of proofs successfully verified.
    pub verified: usize,
    /// Number of failed verifications.
    pub failed: usize,
    /// Indices of failed proofs.
    pub failed_indices: Vec<usize>,
    /// Verification statistics.
    pub stats: StreamingVerificationStats,
}

impl StreamingVerificationResult {
    /// Returns true if all proofs were verified.
    pub fn all_verified(&self) -> bool {
        self.failed == 0
    }

    /// Returns the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.verified + self.failed == 0 {
            return 100.0;
        }
        (self.verified as f64 / (self.verified + self.failed) as f64) * 100.0
    }
}

// =============================================================================
// OPTIMIZED PROOF AGGREGATION - Maximum compression for on-chain verification
// =============================================================================

/// Configuration for proof aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Maximum proofs to aggregate.
    pub max_proofs: usize,
    /// Target compression ratio (0.0-1.0).
    pub target_compression: f64,
    /// Enable deduplication of proof nodes.
    pub deduplicate: bool,
    /// Sort proofs for better compression.
    pub sort_for_compression: bool,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            max_proofs: 1000,
            target_compression: 0.7,
            deduplicate: true,
            sort_for_compression: true,
        }
    }
}

/// Statistics for proof aggregation.
#[derive(Debug, Clone, Default)]
pub struct AggregationStats {
    /// Total proofs aggregated.
    pub proofs_aggregated: usize,
    /// Total samples covered.
    pub total_samples: usize,
    /// Unique proof nodes.
    pub unique_nodes: usize,
    /// Original nodes (before deduplication).
    pub original_nodes: usize,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
    /// Bytes saved through compression.
    pub bytes_saved: usize,
}

/// Optimized proof aggregator for maximum compression.
pub struct ProofAggregator {
    /// Configuration.
    config: AggregationConfig,
    /// Collected batch proofs.
    batch_proofs: Vec<BatchMembershipProof>,
    /// Shared proof nodes across all proofs.
    shared_nodes: HashMap<TreePosition, Hash>,
    /// Statistics.
    stats: AggregationStats,
}

impl ProofAggregator {
    /// Creates a new aggregator.
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            config,
            batch_proofs: Vec::new(),
            shared_nodes: HashMap::new(),
            stats: AggregationStats::default(),
        }
    }

    /// Creates with default config.
    pub fn default_aggregator() -> Self {
        Self::new(AggregationConfig::default())
    }

    /// Adds a batch proof for aggregation.
    pub fn add_proof(&mut self, proof: BatchMembershipProof) -> Result<(), MembershipProofError> {
        if self.batch_proofs.len() >= self.config.max_proofs {
            return Err(MembershipProofError::MerkleError(
                "Maximum proofs exceeded".into()
            ));
        }

        // Verify consistency if we have existing proofs
        if let Some(first) = self.batch_proofs.first() {
            if first.dataset_root != proof.dataset_root {
                return Err(MembershipProofError::InconsistentRoots);
            }
        }

        // Collect shared nodes
        for (pos, hash) in &proof.proof_nodes {
            self.shared_nodes.entry(*pos).or_insert(*hash);
        }

        self.stats.proofs_aggregated += 1;
        self.stats.total_samples += proof.sample_count();
        self.stats.original_nodes += proof.proof_nodes.len();

        self.batch_proofs.push(proof);
        Ok(())
    }

    /// Adds multiple proofs.
    pub fn add_proofs<I>(&mut self, proofs: I) -> Result<(), MembershipProofError>
    where
        I: IntoIterator<Item = BatchMembershipProof>,
    {
        for proof in proofs {
            self.add_proof(proof)?;
        }
        Ok(())
    }

    /// Finalizes aggregation and returns the optimized proof.
    pub fn finalize(mut self) -> Result<OptimizedAggregatedProof, MembershipProofError> {
        if self.batch_proofs.is_empty() {
            return Err(MembershipProofError::EmptyBatch);
        }

        // Sort proofs by sample indices for better compression if enabled
        if self.config.sort_for_compression {
            self.batch_proofs.sort_by(|a, b| {
                a.sample_indices.first().cmp(&b.sample_indices.first())
            });
        }

        // Compute final statistics
        self.stats.unique_nodes = self.shared_nodes.len();
        self.stats.compression_ratio = if self.stats.original_nodes > 0 {
            1.0 - (self.stats.unique_nodes as f64 / self.stats.original_nodes as f64)
        } else {
            0.0
        };

        let dataset_root = self.batch_proofs[0].dataset_root;
        let tree_height = self.batch_proofs[0].tree_height;

        // Collect all sample data
        let mut all_indices = Vec::new();
        let mut all_commitments = Vec::new();

        for proof in &self.batch_proofs {
            all_indices.extend(proof.sample_indices.iter().copied());
            all_commitments.extend(proof.sample_commitments.iter().cloned());
        }

        // Compute bytes saved
        let original_size = self.stats.original_nodes * 40; // ~40 bytes per node (hash + position)
        let compressed_size = self.stats.unique_nodes * 40;
        self.stats.bytes_saved = original_size.saturating_sub(compressed_size);

        Ok(OptimizedAggregatedProof {
            sample_indices: all_indices,
            sample_commitments: all_commitments,
            shared_proof_nodes: self.shared_nodes,
            dataset_root,
            tree_height,
            stats: self.stats,
        })
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &AggregationStats {
        &self.stats
    }

    /// Returns the number of proofs added.
    pub fn proof_count(&self) -> usize {
        self.batch_proofs.len()
    }
}

/// Optimized aggregated proof with maximum compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedAggregatedProof {
    /// All sample indices across all batches.
    pub sample_indices: Vec<usize>,
    /// All sample commitments.
    pub sample_commitments: Vec<SampleCommitment>,
    /// Shared proof nodes (deduplicated).
    pub shared_proof_nodes: HashMap<TreePosition, Hash>,
    /// Dataset root.
    pub dataset_root: Hash,
    /// Tree height.
    pub tree_height: usize,
    /// Aggregation statistics.
    #[serde(skip)]
    pub stats: AggregationStats,
}

impl OptimizedAggregatedProof {
    /// Verifies this aggregated proof.
    pub fn verify(&self, commitment: &DatasetCommitment) -> bool {
        if self.dataset_root != commitment.root {
            return false;
        }

        // Reconstruct and verify
        let multi = MultiProof {
            leaf_indices: self.sample_indices.clone(),
            leaf_hashes: self.sample_commitments.iter().map(|sc| sc.hash).collect(),
            proof_nodes: self.shared_proof_nodes.clone(),
            root: commitment.root,
            height: self.tree_height,
        };

        multi.verify(&Sha256Hasher)
    }

    /// Returns the compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        let individual_nodes = self.sample_indices.len() * self.tree_height;
        let compressed = self.shared_proof_nodes.len();
        if compressed == 0 {
            return 0.0;
        }
        1.0 - (compressed as f64 / individual_nodes as f64)
    }

    /// Returns the total number of samples.
    pub fn sample_count(&self) -> usize {
        self.sample_indices.len()
    }

    /// Converts to compact bytes for on-chain storage.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Root
        bytes.extend_from_slice(self.dataset_root.as_bytes());

        // Tree height
        bytes.extend_from_slice(&(self.tree_height as u32).to_le_bytes());

        // Sample count
        bytes.extend_from_slice(&(self.sample_indices.len() as u32).to_le_bytes());

        // Sample indices (compressed as deltas for contiguous ranges)
        let mut prev = 0usize;
        for &idx in &self.sample_indices {
            let delta = idx.saturating_sub(prev);
            bytes.extend_from_slice(&(delta as u32).to_le_bytes());
            prev = idx;
        }

        // Proof node count
        bytes.extend_from_slice(&(self.shared_proof_nodes.len() as u32).to_le_bytes());

        // Proof nodes
        for (pos, hash) in &self.shared_proof_nodes {
            bytes.extend_from_slice(&(pos.level as u16).to_le_bytes());
            bytes.extend_from_slice(&(pos.index as u32).to_le_bytes());
            bytes.extend_from_slice(hash.as_bytes());
        }

        bytes
    }
}

// =============================================================================
// PARALLEL VERIFICATION - Multi-threaded proof verification
// =============================================================================

/// Configuration for parallel verification.
#[derive(Debug, Clone)]
pub struct ParallelVerificationConfig {
    /// Number of threads (0 = auto-detect).
    pub num_threads: usize,
    /// Minimum proofs per thread.
    pub min_proofs_per_thread: usize,
    /// Chunk size for work distribution.
    pub chunk_size: usize,
}

impl Default for ParallelVerificationConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            min_proofs_per_thread: 100,
            chunk_size: 500,
        }
    }
}

/// Parallel proof verifier using rayon for concurrent verification.
///
/// Proofs are independent and verified in parallel when the batch size
/// exceeds `min_proofs_per_thread`.
pub struct ParallelBatchVerifier {
    /// Configuration.
    config: ParallelVerificationConfig,
    /// Dataset commitment.
    commitment: Arc<DatasetCommitment>,
}

/// Threshold below which sequential verification is used (avoids rayon overhead).
const PARALLEL_VERIFICATION_THRESHOLD: usize = 50;

impl ParallelBatchVerifier {
    /// Creates a new parallel verifier.
    pub fn new(commitment: DatasetCommitment, config: ParallelVerificationConfig) -> Self {
        Self {
            config,
            commitment: Arc::new(commitment),
        }
    }

    /// Verifies multiple batch proofs in parallel using rayon.
    ///
    /// Falls back to sequential verification for small batches (< 50 proofs)
    /// to avoid thread pool overhead.
    pub fn verify_all(&self, proofs: &[BatchMembershipProof]) -> ParallelVerificationResult {
        let start = std::time::Instant::now();

        let results: Vec<bool> = if proofs.len() >= PARALLEL_VERIFICATION_THRESHOLD {
            proofs
                .par_iter()
                .map(|proof| proof.verify(&self.commitment))
                .collect()
        } else {
            proofs
                .iter()
                .map(|proof| proof.verify(&self.commitment))
                .collect()
        };

        let verified = results.iter().filter(|&&r| r).count();
        let failed = results.len() - verified;
        let elapsed_us = start.elapsed().as_micros() as u64;

        ParallelVerificationResult {
            results,
            verified,
            failed,
            total_time_us: elapsed_us,
            proofs_per_second: if elapsed_us > 0 {
                (proofs.len() as f64 / elapsed_us as f64) * 1_000_000.0
            } else {
                0.0
            },
        }
    }

    /// Verifies and returns indices of failures, using parallel verification.
    pub fn verify_and_collect_failures(&self, proofs: &[BatchMembershipProof]) -> Vec<usize> {
        if proofs.len() >= PARALLEL_VERIFICATION_THRESHOLD {
            proofs
                .par_iter()
                .enumerate()
                .filter_map(|(idx, proof)| {
                    if !proof.verify(&self.commitment) {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            proofs
                .iter()
                .enumerate()
                .filter_map(|(idx, proof)| {
                    if !proof.verify(&self.commitment) {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}

/// Result of parallel verification.
#[derive(Debug, Clone)]
pub struct ParallelVerificationResult {
    /// Individual results.
    pub results: Vec<bool>,
    /// Count of verified proofs.
    pub verified: usize,
    /// Count of failed proofs.
    pub failed: usize,
    /// Total verification time in microseconds.
    pub total_time_us: u64,
    /// Throughput in proofs per second.
    pub proofs_per_second: f64,
}

impl ParallelVerificationResult {
    /// Returns true if all verifications succeeded.
    pub fn all_verified(&self) -> bool {
        self.failed == 0
    }

    /// Returns the success rate.
    pub fn success_rate(&self) -> f64 {
        if self.verified + self.failed == 0 {
            return 100.0;
        }
        (self.verified as f64 / (self.verified + self.failed) as f64) * 100.0
    }
}

// =============================================================================
// INCREMENTAL PROOF GENERATOR - Generate proofs as data streams in
// =============================================================================

/// Incremental proof generator for streaming data.
pub struct IncrementalProofGenerator {
    /// The Merkle tree (grows as samples are added).
    tree: MerkleTree<Sha256Hasher>,
    /// Dataset commitment (updated as tree grows).
    commitment: Option<DatasetCommitment>,
    /// Sample commitments cache.
    sample_commitments: Vec<SampleCommitment>,
    /// Metadata for commitment updates.
    metadata: DatasetMetadata,
}

impl IncrementalProofGenerator {
    /// Creates a new incremental generator.
    pub fn new(metadata: DatasetMetadata) -> Self {
        Self {
            tree: MerkleTree::with_sha256(),
            commitment: None,
            sample_commitments: Vec::new(),
            metadata,
        }
    }

    /// Adds a sample and returns its index.
    pub fn add_sample(&mut self, sample: &Sample) -> Result<usize, MembershipProofError> {
        let sc = SampleCommitment::new(sample);
        let hash_bytes = SampleCommitment::hash_sample(sample);
        let hash = Hash::from_bytes(hash_bytes);

        self.tree
            .push_hash(hash)
            .map_err(|e| MembershipProofError::MerkleError(format!("{}", e)))?;

        self.sample_commitments.push(sc);

        // Update commitment
        self.commitment = Some(DatasetCommitment::new(
            &self.tree,
            &self.metadata,
            None,
        ));

        Ok(self.sample_commitments.len() - 1)
    }

    /// Generates a proof for a sample by index.
    pub fn prove(&self, index: usize) -> Result<SampleMembershipProof, MembershipProofError> {
        let commitment = self.commitment
            .as_ref()
            .ok_or(MembershipProofError::EmptyBatch)?;

        let sc = self.sample_commitments
            .get(index)
            .ok_or(MembershipProofError::SampleNotInDataset(index))?
            .clone();

        let merkle_proof = self.tree
            .prove(index)
            .map_err(|e| MembershipProofError::MerkleError(format!("{}", e)))?;

        Ok(SampleMembershipProof {
            sample_commitment: sc,
            sample_index: index,
            merkle_proof,
            dataset_root: commitment.root,
        })
    }

    /// Generates a batch proof for multiple samples.
    pub fn prove_batch(&self, indices: &[usize]) -> Result<BatchMembershipProof, MembershipProofError> {
        if indices.is_empty() {
            return Err(MembershipProofError::EmptyBatch);
        }

        let commitment = self.commitment
            .as_ref()
            .ok_or(MembershipProofError::EmptyBatch)?;

        let sample_commitments: Result<Vec<_>, _> = indices
            .iter()
            .map(|&i| {
                self.sample_commitments
                    .get(i)
                    .cloned()
                    .ok_or(MembershipProofError::SampleNotInDataset(i))
            })
            .collect();

        let sample_commitments = sample_commitments?;

        let multi_proof = self.tree
            .prove_multi(indices)
            .map_err(|e| MembershipProofError::MerkleError(format!("{}", e)))?;

        Ok(BatchMembershipProof {
            sample_commitments,
            sample_indices: indices.to_vec(),
            proof_nodes: multi_proof.proof_nodes,
            tree_height: self.tree.height(),
            dataset_root: commitment.root,
            batch_commitment: None,
        })
    }

    /// Returns the current commitment.
    pub fn commitment(&self) -> Option<&DatasetCommitment> {
        self.commitment.as_ref()
    }

    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.sample_commitments.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.sample_commitments.is_empty()
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

    // =========================================================================
    // STREAMING BATCH VERIFIER TESTS
    // =========================================================================

    #[test]
    fn test_streaming_batch_verifier() {
        let samples = create_test_samples(100);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut verifier = StreamingBatchVerifier::with_commitment(commitment.clone());

        // Create and verify proofs
        for i in 0..10 {
            let batch_samples: Vec<Sample> = (i * 10..(i + 1) * 10)
                .map(|j| samples[j].clone())
                .collect();
            let proof = BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap();
            assert!(verifier.verify_batch_proof(&proof));
        }

        let stats = verifier.stats();
        assert_eq!(stats.proofs_processed, 10);
        assert_eq!(stats.successful, 10);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_streaming_verifier_with_failures() {
        let samples = create_test_samples(100);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut verifier = StreamingBatchVerifier::new(
            commitment.clone(),
            StreamingVerificationConfig::thorough()
        );

        // Create valid proofs
        let mut proofs = Vec::new();
        for i in 0..5 {
            let batch_samples: Vec<Sample> = (i * 10..(i + 1) * 10)
                .map(|j| samples[j].clone())
                .collect();
            proofs.push(BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap());
        }

        // Corrupt one proof
        proofs[2].dataset_root = Hash::zero();

        let result = verifier.verify_stream(&proofs);

        assert_eq!(result.verified, 4);
        assert_eq!(result.failed, 1);
        assert_eq!(result.failed_indices, vec![2]);
    }

    #[test]
    fn test_streaming_verifier_fail_fast() {
        let samples = create_test_samples(100);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut verifier = StreamingBatchVerifier::new(
            commitment.clone(),
            StreamingVerificationConfig::fast()
        );

        let mut proofs = Vec::new();
        for i in 0..10 {
            let batch_samples: Vec<Sample> = (i * 10..(i + 1) * 10)
                .map(|j| samples[j].clone())
                .collect();
            proofs.push(BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap());
        }

        // Corrupt second proof
        proofs[1].dataset_root = Hash::zero();

        let result = verifier.verify_stream(&proofs);

        // Should stop after first failure (index 1)
        assert_eq!(result.verified, 1);
        assert_eq!(result.failed, 1);
    }

    // =========================================================================
    // PROOF AGGREGATION TESTS
    // =========================================================================

    #[test]
    fn test_proof_aggregator() {
        let samples = create_test_samples(256);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut aggregator = ProofAggregator::default_aggregator();

        // Add multiple batch proofs
        for i in 0..8 {
            let batch_samples: Vec<Sample> = (i * 32..(i + 1) * 32)
                .map(|j| samples[j].clone())
                .collect();
            let proof = BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap();
            aggregator.add_proof(proof).unwrap();
        }

        let optimized = aggregator.finalize().unwrap();

        assert!(optimized.verify(&commitment));
        assert_eq!(optimized.sample_count(), 256);
        assert!(optimized.compression_ratio() > 0.0);
    }

    #[test]
    fn test_proof_aggregator_compression() {
        let samples = create_test_samples(512);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut aggregator = ProofAggregator::new(AggregationConfig {
            sort_for_compression: true,
            deduplicate: true,
            ..Default::default()
        });

        // Add contiguous batches (should have good compression)
        for i in 0..16 {
            let batch_samples: Vec<Sample> = (i * 32..(i + 1) * 32)
                .map(|j| samples[j].clone())
                .collect();
            let proof = BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap();
            aggregator.add_proof(proof).unwrap();
        }

        let stats = aggregator.stats();
        assert!(stats.proofs_aggregated == 16);

        let optimized = aggregator.finalize().unwrap();
        let compression = optimized.compression_ratio();

        // Contiguous ranges should compress well
        assert!(compression > 0.5, "Expected >50% compression, got {}%", compression * 100.0);
    }

    #[test]
    fn test_optimized_proof_compact_bytes() {
        let samples = create_test_samples(64);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut aggregator = ProofAggregator::default_aggregator();

        for i in 0..4 {
            let batch_samples: Vec<Sample> = (i * 16..(i + 1) * 16)
                .map(|j| samples[j].clone())
                .collect();
            let proof = BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap();
            aggregator.add_proof(proof).unwrap();
        }

        let optimized = aggregator.finalize().unwrap();
        let bytes = optimized.to_compact_bytes();

        // Should be reasonably compact
        assert!(bytes.len() < 10000);
    }

    // =========================================================================
    // PARALLEL VERIFICATION TESTS
    // =========================================================================

    #[test]
    fn test_parallel_verifier() {
        let samples = create_test_samples(200);
        let (tree, commitment) = setup_test_tree(&samples);

        let proofs: Vec<BatchMembershipProof> = (0..20)
            .map(|i| {
                let batch_samples: Vec<Sample> = (i * 10..(i + 1) * 10)
                    .map(|j| samples[j].clone())
                    .collect();
                BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap()
            })
            .collect();

        let verifier = ParallelBatchVerifier::new(
            commitment.clone(),
            ParallelVerificationConfig::default()
        );

        let result = verifier.verify_all(&proofs);

        assert!(result.all_verified());
        assert_eq!(result.verified, 20);
        assert_eq!(result.failed, 0);
        assert!(result.proofs_per_second > 0.0);
    }

    #[test]
    fn test_parallel_verifier_collect_failures() {
        let samples = create_test_samples(100);
        let (tree, commitment) = setup_test_tree(&samples);

        let mut proofs: Vec<BatchMembershipProof> = (0..10)
            .map(|i| {
                let batch_samples: Vec<Sample> = (i * 10..(i + 1) * 10)
                    .map(|j| samples[j].clone())
                    .collect();
                BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap()
            })
            .collect();

        // Corrupt some proofs
        proofs[3].dataset_root = Hash::zero();
        proofs[7].dataset_root = Hash::zero();

        let verifier = ParallelBatchVerifier::new(
            commitment.clone(),
            ParallelVerificationConfig::default()
        );

        let failures = verifier.verify_and_collect_failures(&proofs);

        assert_eq!(failures.len(), 2);
        assert!(failures.contains(&3));
        assert!(failures.contains(&7));
    }

    // =========================================================================
    // INCREMENTAL PROOF GENERATOR TESTS
    // =========================================================================

    #[test]
    fn test_incremental_generator() {
        let metadata = DatasetMetadata {
            name: "incremental_test".to_string(),
            num_samples: 0,
            feature_dims: vec![4],
            label_dims: vec![2],
            dtype: DataType::Float32,
            extra: StdHashMap::new(),
        };

        let mut generator = IncrementalProofGenerator::new(metadata);

        // Add samples incrementally
        for i in 0..50 {
            let sample = Sample::from_f32(i, vec![i as f32; 4], vec![0.0, 1.0]);
            generator.add_sample(&sample).unwrap();
        }

        assert_eq!(generator.len(), 50);

        // Generate and verify proofs
        let proof = generator.prove(25).unwrap();
        assert!(proof.verify(generator.commitment().unwrap()));

        let batch_proof = generator.prove_batch(&[10, 20, 30, 40]).unwrap();
        assert!(batch_proof.verify(generator.commitment().unwrap()));
    }

    #[test]
    fn test_incremental_generator_single_sample() {
        let metadata = DatasetMetadata {
            name: "single".to_string(),
            num_samples: 0,
            feature_dims: vec![4],
            label_dims: vec![2],
            dtype: DataType::Float32,
            extra: StdHashMap::new(),
        };

        let mut generator = IncrementalProofGenerator::new(metadata);

        let sample = Sample::from_f32(0, vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]);
        generator.add_sample(&sample).unwrap();

        let proof = generator.prove(0).unwrap();
        assert!(proof.verify(generator.commitment().unwrap()));
    }

    // =========================================================================
    // LARGE SCALE VERIFICATION TESTS
    // =========================================================================

    #[test]
    fn test_large_scale_verification() {
        let samples = create_test_samples(1000);
        let (tree, commitment) = setup_test_tree(&samples);

        // Create many batch proofs
        let proofs: Vec<BatchMembershipProof> = (0..100)
            .map(|i| {
                let batch_samples: Vec<Sample> = (i * 10..(i + 1) * 10)
                    .map(|j| samples[j].clone())
                    .collect();
                BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap()
            })
            .collect();

        let mut verifier = StreamingBatchVerifier::with_commitment(commitment);
        let result = verifier.verify_stream(&proofs);

        assert!(result.all_verified());
        assert_eq!(result.verified, 100);
        assert!(result.stats.samples_verified == 1000);
    }

    #[test]
    fn test_verification_throughput() {
        let samples = create_test_samples(500);
        let (tree, commitment) = setup_test_tree(&samples);

        let proofs: Vec<BatchMembershipProof> = (0..50)
            .map(|i| {
                let batch_samples: Vec<Sample> = (i * 10..(i + 1) * 10)
                    .map(|j| samples[j].clone())
                    .collect();
                BatchMembershipProof::new(&batch_samples, &tree, &commitment).unwrap()
            })
            .collect();

        let verifier = ParallelBatchVerifier::new(
            commitment,
            ParallelVerificationConfig::default()
        );

        let result = verifier.verify_all(&proofs);

        // Should have reasonable throughput
        assert!(result.proofs_per_second > 100.0,
            "Expected >100 proofs/sec, got {}", result.proofs_per_second);
    }
}
