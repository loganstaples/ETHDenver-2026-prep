//! Verified Data Loader for HELIX.
//!
//! Provides cryptographically verified data loading that checks Merkle proofs
//! before accepting any training data. Ensures data integrity and prevents
//! data poisoning attacks in distributed training.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use helix_core::data::{
    Batch, BatchCommitment, BatchMembershipProof, DatasetCommitment, DatasetMetadata,
    Hash, MembershipProofError, MembershipProofGenerator, MembershipVerifier, MerkleProof,
    MerkleTree, Sample, SampleCommitment, SampleMembershipProof, Sha256Hasher,
};

use serde::{Deserialize, Serialize};

/// Configuration for verified data loading.
#[derive(Debug, Clone)]
pub struct VerifiedDataLoaderConfig {
    /// Batch size for training.
    pub batch_size: usize,
    /// Maximum number of batches to prefetch.
    pub prefetch_count: usize,
    /// Whether to verify proofs synchronously (blocking) or cache failures.
    pub strict_verification: bool,
    /// Maximum time to wait for proof verification.
    pub verification_timeout: Duration,
    /// Cache size for verified batches.
    pub verification_cache_size: usize,
    /// Whether to drop batches that fail verification.
    pub drop_unverified: bool,
    /// Maximum consecutive verification failures before halting.
    pub max_consecutive_failures: usize,
    /// Whether to generate proofs for batches (if not provided).
    pub auto_generate_proofs: bool,
}

impl Default for VerifiedDataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            prefetch_count: 4,
            strict_verification: true,
            verification_timeout: Duration::from_secs(5),
            verification_cache_size: 1000,
            drop_unverified: true,
            max_consecutive_failures: 3,
            auto_generate_proofs: true,
        }
    }
}

/// A verified training batch with proof.
#[derive(Debug, Clone)]
pub struct VerifiedBatch {
    /// The underlying batch.
    pub batch: Batch,
    /// Sample indices in the dataset.
    pub sample_indices: Vec<usize>,
    /// Batch membership proof.
    pub proof: BatchMembershipProof,
    /// Verification timestamp.
    pub verified_at: Instant,
    /// Whether verification passed.
    pub is_verified: bool,
    /// Batch commitment.
    pub commitment: BatchCommitment,
}

impl VerifiedBatch {
    /// Creates a new verified batch.
    pub fn new(
        batch: Batch,
        sample_indices: Vec<usize>,
        proof: BatchMembershipProof,
        commitment: BatchCommitment,
    ) -> Self {
        Self {
            batch,
            sample_indices,
            proof,
            verified_at: Instant::now(),
            is_verified: false,
            commitment,
        }
    }

    /// Returns the batch size.
    pub fn len(&self) -> usize {
        self.batch.samples.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.batch.samples.is_empty()
    }

    /// Returns the batch ID.
    pub fn batch_id(&self) -> usize {
        self.batch.id
    }

    /// Serializes the batch proof for network transmission.
    pub fn serialize_proof(&self) -> Vec<u8> {
        use helix_core::traits::BinarySerializable;
        self.proof.to_bytes()
    }

    /// Returns the dataset root from the proof.
    pub fn dataset_root(&self) -> Hash {
        self.proof.dataset_root
    }
}

/// Result of batch verification.
#[derive(Debug, Clone)]
pub struct BatchVerificationResult {
    /// Batch index.
    pub batch_index: usize,
    /// Whether verification passed.
    pub is_valid: bool,
    /// Verification time in milliseconds.
    pub verification_time_ms: u64,
    /// Error message if invalid.
    pub error: Option<String>,
    /// Number of samples verified.
    pub samples_verified: usize,
    /// Proof compression ratio achieved.
    pub compression_ratio: f64,
}

/// Statistics for verification operations.
#[derive(Debug, Clone, Default)]
pub struct VerificationStats {
    /// Total batches processed.
    pub batches_processed: u64,
    /// Batches that passed verification.
    pub batches_verified: u64,
    /// Batches that failed verification.
    pub batches_failed: u64,
    /// Total samples verified.
    pub samples_verified: u64,
    /// Total verification time in milliseconds.
    pub total_verification_time_ms: u64,
    /// Average verification time per batch.
    pub avg_verification_time_ms: f64,
    /// Average compression ratio.
    pub avg_compression_ratio: f64,
    /// Consecutive failures count.
    pub consecutive_failures: usize,
    /// Peak consecutive failures.
    pub peak_consecutive_failures: usize,
}

impl VerificationStats {
    /// Updates statistics with a verification result.
    pub fn update(&mut self, result: &BatchVerificationResult) {
        self.batches_processed += 1;
        self.total_verification_time_ms += result.verification_time_ms;
        self.samples_verified += result.samples_verified as u64;

        if result.is_valid {
            self.batches_verified += 1;
            self.consecutive_failures = 0;
        } else {
            self.batches_failed += 1;
            self.consecutive_failures += 1;
            if self.consecutive_failures > self.peak_consecutive_failures {
                self.peak_consecutive_failures = self.consecutive_failures;
            }
        }

        // Update averages
        if self.batches_processed > 0 {
            self.avg_verification_time_ms =
                self.total_verification_time_ms as f64 / self.batches_processed as f64;
        }

        // Update compression ratio (running average)
        if result.compression_ratio > 0.0 {
            let n = self.batches_processed as f64;
            self.avg_compression_ratio =
                (self.avg_compression_ratio * (n - 1.0) + result.compression_ratio) / n;
        }
    }

    /// Returns the verification success rate.
    pub fn success_rate(&self) -> f64 {
        if self.batches_processed == 0 {
            return 1.0;
        }
        self.batches_verified as f64 / self.batches_processed as f64
    }
}

/// Errors during data verification.
#[derive(Debug, Clone)]
pub enum DataVerificationError {
    /// Dataset commitment not found.
    CommitmentNotFound,
    /// Dataset commitment mismatch.
    CommitmentMismatch { expected: Hash, actual: Hash },
    /// Merkle proof verification failed.
    ProofVerificationFailed(String),
    /// Sample not in dataset.
    SampleNotInDataset(usize),
    /// Batch proof invalid.
    InvalidBatchProof(String),
    /// Too many consecutive failures.
    TooManyFailures { consecutive: usize, max: usize },
    /// Verification timeout.
    Timeout(Duration),
    /// Merkle tree error.
    MerkleError(String),
    /// Dataset not initialized.
    DatasetNotInitialized,
    /// Proof generation failed.
    ProofGenerationFailed(String),
}

impl std::fmt::Display for DataVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommitmentNotFound => write!(f, "Dataset commitment not found"),
            Self::CommitmentMismatch { expected, actual } => {
                write!(f, "Commitment mismatch: expected {}, got {}", expected, actual)
            }
            Self::ProofVerificationFailed(msg) => write!(f, "Proof verification failed: {}", msg),
            Self::SampleNotInDataset(idx) => write!(f, "Sample {} not in dataset", idx),
            Self::InvalidBatchProof(msg) => write!(f, "Invalid batch proof: {}", msg),
            Self::TooManyFailures { consecutive, max } => {
                write!(f, "Too many consecutive failures: {} > {}", consecutive, max)
            }
            Self::Timeout(d) => write!(f, "Verification timeout after {:?}", d),
            Self::MerkleError(msg) => write!(f, "Merkle error: {}", msg),
            Self::DatasetNotInitialized => write!(f, "Dataset not initialized"),
            Self::ProofGenerationFailed(msg) => write!(f, "Proof generation failed: {}", msg),
        }
    }
}

impl std::error::Error for DataVerificationError {}

impl From<MembershipProofError> for DataVerificationError {
    fn from(e: MembershipProofError) -> Self {
        DataVerificationError::ProofVerificationFailed(e.to_string())
    }
}

/// Verified data loader with Merkle proof verification.
///
/// Wraps a standard data source and ensures all batches are cryptographically
/// verified against the dataset commitment before use in training.
// Manual Debug impl to avoid requiring Debug on all fields
pub struct VerifiedDataLoader {
    /// Configuration.
    config: VerifiedDataLoaderConfig,
    /// Dataset metadata.
    metadata: Option<DatasetMetadata>,
    /// Dataset commitment.
    commitment: Option<DatasetCommitment>,
    /// Merkle tree for the dataset.
    tree: Option<MerkleTree<Sha256Hasher>>,
    /// Proof generator.
    proof_generator: Option<MembershipProofGenerator>,
    /// Membership verifier.
    verifier: MembershipVerifier,
    /// Cached samples.
    samples: Vec<Sample>,
    /// Current position in dataset.
    position: usize,
    /// Shuffled indices.
    indices: Vec<usize>,
    /// Prefetched verified batches.
    prefetch_queue: VecDeque<VerifiedBatch>,
    /// Verification statistics.
    stats: VerificationStats,
    /// Verification cache (batch commitment -> result).
    verification_cache: HashMap<Hash, BatchVerificationResult>,
    /// Current epoch.
    current_epoch: usize,
    /// Current batch index.
    current_batch_idx: usize,
    /// Whether loader is exhausted for this epoch.
    exhausted: bool,
}

impl std::fmt::Debug for VerifiedDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedDataLoader")
            .field("config", &self.config)
            .field("metadata", &self.metadata.is_some())
            .field("commitment", &self.commitment.is_some())
            .field("tree", &self.tree.is_some())
            .field("samples_count", &self.samples.len())
            .field("position", &self.position)
            .field("stats", &self.stats)
            .field("current_epoch", &self.current_epoch)
            .field("current_batch_idx", &self.current_batch_idx)
            .field("exhausted", &self.exhausted)
            .finish()
    }
}

impl VerifiedDataLoader {
    /// Creates a new verified data loader.
    pub fn new(config: VerifiedDataLoaderConfig) -> Self {
        Self {
            config,
            metadata: None,
            commitment: None,
            tree: None,
            proof_generator: None,
            verifier: MembershipVerifier::new(),
            samples: Vec::new(),
            position: 0,
            indices: Vec::new(),
            prefetch_queue: VecDeque::new(),
            stats: VerificationStats::default(),
            verification_cache: HashMap::new(),
            current_epoch: 0,
            current_batch_idx: 0,
            exhausted: false,
        }
    }

    /// Initializes the loader with a dataset and commitment.
    ///
    /// This verifies that the samples match the commitment and builds
    /// the necessary verification structures.
    pub fn initialize(
        &mut self,
        samples: Vec<Sample>,
        metadata: DatasetMetadata,
        commitment: DatasetCommitment,
    ) -> Result<(), DataVerificationError> {
        // Build Merkle tree from samples
        let leaf_hashes: Vec<Hash> = samples
            .iter()
            .map(|s| Hash::from_bytes(SampleCommitment::hash_sample(s)))
            .collect();

        let tree = MerkleTree::from_hashes(Sha256Hasher, leaf_hashes)
            .map_err(|e| DataVerificationError::MerkleError(e.to_string()))?;

        // Verify commitment matches
        let computed_root = tree.root().ok_or(DataVerificationError::MerkleError(
            "Empty tree".to_string(),
        ))?;

        if computed_root != commitment.root {
            return Err(DataVerificationError::CommitmentMismatch {
                expected: commitment.root,
                actual: computed_root,
            });
        }

        // Register dataset with verifier
        self.verifier.register_dataset(commitment.clone(), tree.clone());

        // Create proof generator
        let proof_generator = MembershipProofGenerator::new(tree.clone(), commitment.clone(), &samples);

        // Initialize indices
        self.indices = (0..samples.len()).collect();

        self.samples = samples;
        self.metadata = Some(metadata);
        self.commitment = Some(commitment);
        self.tree = Some(tree);
        self.proof_generator = Some(proof_generator);
        self.position = 0;
        self.exhausted = false;

        Ok(())
    }

    /// Returns the dataset commitment.
    pub fn commitment(&self) -> Option<&DatasetCommitment> {
        self.commitment.as_ref()
    }

    /// Returns the dataset root hash.
    pub fn dataset_root(&self) -> Option<Hash> {
        self.commitment.as_ref().map(|c| c.root)
    }

    /// Returns verification statistics.
    pub fn stats(&self) -> &VerificationStats {
        &self.stats
    }

    /// Resets the loader for a new epoch.
    pub fn reset_epoch(&mut self) {
        self.position = 0;
        self.current_batch_idx = 0;
        self.current_epoch += 1;
        self.exhausted = false;
        self.prefetch_queue.clear();

        // Shuffle indices for new epoch
        self.shuffle_indices();
    }

    /// Shuffles the sample indices.
    fn shuffle_indices(&mut self) {
        use rand::seq::SliceRandom;
        self.indices.shuffle(&mut rand::thread_rng());
    }

    /// Returns the current epoch.
    pub fn epoch(&self) -> usize {
        self.current_epoch
    }

    /// Returns the number of samples.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Returns the number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        self.samples.len() / self.config.batch_size
    }

    /// Gets the next verified batch.
    ///
    /// Returns None if exhausted or if verification fails and drop_unverified is true.
    pub fn next_batch(&mut self) -> Result<Option<VerifiedBatch>, DataVerificationError> {
        // Check for too many consecutive failures
        if self.stats.consecutive_failures >= self.config.max_consecutive_failures {
            return Err(DataVerificationError::TooManyFailures {
                consecutive: self.stats.consecutive_failures,
                max: self.config.max_consecutive_failures,
            });
        }

        // Try prefetch queue first
        if let Some(batch) = self.prefetch_queue.pop_front() {
            self.prefetch_batches()?;
            return Ok(Some(batch));
        }

        // Build and verify a new batch
        self.build_verified_batch()
    }

    /// Prefetches batches into the queue.
    fn prefetch_batches(&mut self) -> Result<(), DataVerificationError> {
        while self.prefetch_queue.len() < self.config.prefetch_count {
            match self.build_verified_batch()? {
                Some(batch) => self.prefetch_queue.push_back(batch),
                None => break,
            }
        }
        Ok(())
    }

    /// Builds and verifies a single batch.
    fn build_verified_batch(&mut self) -> Result<Option<VerifiedBatch>, DataVerificationError> {
        if self.exhausted {
            return Ok(None);
        }

        if self.proof_generator.is_none() {
            return Err(DataVerificationError::DatasetNotInitialized);
        }

        let commitment = self.commitment.as_ref()
            .ok_or(DataVerificationError::DatasetNotInitialized)?;

        // Calculate batch boundaries
        let start = self.position;
        let end = (start + self.config.batch_size).min(self.samples.len());

        if start >= self.samples.len() {
            self.exhausted = true;
            return Ok(None);
        }

        // Check if we have enough for a full batch
        if end - start < self.config.batch_size {
            self.exhausted = true;
            return Ok(None);
        }

        // Get sample indices for this batch
        let batch_indices: Vec<usize> = self.indices[start..end].to_vec();

        // Collect samples
        let batch_samples: Vec<Sample> = batch_indices
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();

        // Create batch
        let batch = Batch::new(self.current_batch_idx, batch_samples);

        // Generate batch proof
        let start_time = Instant::now();
        let proof = self.proof_generator
            .as_ref()
            .ok_or(DataVerificationError::DatasetNotInitialized)?
            .prove_batch(&batch_indices)
            .map_err(|e| DataVerificationError::ProofGenerationFailed(e.to_string()))?;

        // Create batch commitment
        let batch_commitment = BatchCommitment::from_samples(
            &batch.samples.iter().collect::<Vec<_>>(),
            self.current_batch_idx,
            batch_indices.clone(),
            commitment.commitment_id(),
        );

        // Verify the proof
        let verification_start = Instant::now();
        let is_valid = proof.verify(commitment);
        let verification_time = verification_start.elapsed();

        // Record verification result
        let result = BatchVerificationResult {
            batch_index: self.current_batch_idx,
            is_valid,
            verification_time_ms: verification_time.as_millis() as u64,
            error: if is_valid { None } else { Some("Proof verification failed".to_string()) },
            samples_verified: batch.samples.len(),
            compression_ratio: proof.compression_ratio(),
        };

        // Update statistics
        self.stats.update(&result);

        // Cache result
        self.verification_cache.insert(batch_commitment.commitment_id(), result.clone());

        // Handle verification failure
        if !is_valid {
            if self.config.drop_unverified {
                // Skip this batch and try next
                self.position = end;
                self.current_batch_idx += 1;
                return self.build_verified_batch();
            } else if self.config.strict_verification {
                return Err(DataVerificationError::ProofVerificationFailed(
                    "Batch proof verification failed".to_string(),
                ));
            }
        }

        // Create verified batch
        let mut verified_batch = VerifiedBatch::new(batch, batch_indices, proof, batch_commitment);
        verified_batch.is_verified = is_valid;
        verified_batch.verified_at = Instant::now();

        // Update position
        self.position = end;
        self.current_batch_idx += 1;

        Ok(Some(verified_batch))
    }

    /// Verifies a batch with an external proof.
    ///
    /// Used when receiving batches from the network with attached proofs.
    pub fn verify_external_batch(
        &mut self,
        batch: &Batch,
        sample_indices: &[usize],
        proof: &BatchMembershipProof,
    ) -> Result<BatchVerificationResult, DataVerificationError> {
        let commitment = self.commitment.as_ref()
            .ok_or(DataVerificationError::DatasetNotInitialized)?;

        // Verify the proof matches our commitment
        if proof.dataset_root != commitment.root {
            return Err(DataVerificationError::CommitmentMismatch {
                expected: commitment.root,
                actual: proof.dataset_root,
            });
        }

        // Verify sample commitments match the batch
        for (i, (sample, &idx)) in batch.samples.iter().zip(sample_indices.iter()).enumerate() {
            if i >= proof.sample_commitments.len() {
                return Err(DataVerificationError::InvalidBatchProof(
                    format!("Missing sample commitment for index {}", i),
                ));
            }

            let expected_hash = Hash::from_bytes(SampleCommitment::hash_sample(sample));
            if proof.sample_commitments[i].hash != expected_hash {
                return Err(DataVerificationError::InvalidBatchProof(
                    format!("Sample {} hash mismatch", idx),
                ));
            }
        }

        // Verify the Merkle proof
        let start = Instant::now();
        let is_valid = proof.verify(commitment);
        let verification_time = start.elapsed();

        let result = BatchVerificationResult {
            batch_index: batch.id,
            is_valid,
            verification_time_ms: verification_time.as_millis() as u64,
            error: if is_valid { None } else { Some("External proof verification failed".to_string()) },
            samples_verified: batch.samples.len(),
            compression_ratio: proof.compression_ratio(),
        };

        self.stats.update(&result);

        Ok(result)
    }

    /// Generates a membership proof for specific sample indices.
    pub fn generate_batch_proof(
        &self,
        sample_indices: &[usize],
    ) -> Result<BatchMembershipProof, DataVerificationError> {
        let generator = self.proof_generator.as_ref()
            .ok_or(DataVerificationError::DatasetNotInitialized)?;

        generator
            .prove_batch(sample_indices)
            .map_err(|e| DataVerificationError::ProofGenerationFailed(e.to_string()))
    }

    /// Generates a single sample membership proof.
    pub fn generate_sample_proof(
        &self,
        sample_index: usize,
    ) -> Result<SampleMembershipProof, DataVerificationError> {
        let generator = self.proof_generator.as_ref()
            .ok_or(DataVerificationError::DatasetNotInitialized)?;

        generator
            .prove_sample(sample_index)
            .map_err(|e| DataVerificationError::ProofGenerationFailed(e.to_string()))
    }

    /// Verifies a single sample membership proof.
    pub fn verify_sample_proof(
        &mut self,
        proof: &SampleMembershipProof,
    ) -> Result<bool, DataVerificationError> {
        self.verifier
            .verify_sample(proof)
            .map_err(|e| DataVerificationError::ProofVerificationFailed(e.to_string()))
    }

    /// Clears the verification cache.
    pub fn clear_cache(&mut self) {
        self.verification_cache.clear();
    }

    /// Returns cached verification result for a batch commitment.
    pub fn get_cached_verification(&self, commitment_id: &Hash) -> Option<&BatchVerificationResult> {
        self.verification_cache.get(commitment_id)
    }
}

/// Iterator for verified batches.
impl Iterator for VerifiedDataLoader {
    type Item = Result<VerifiedBatch, DataVerificationError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            dtype: helix_core::data::DataType::Float32,
            extra: StdHashMap::new(),
        }
    }

    fn create_test_commitment(samples: &[Sample], metadata: &DatasetMetadata) -> DatasetCommitment {
        DatasetCommitment::from_samples(samples, metadata, Some("test".to_string()))
    }

    #[test]
    fn test_verified_loader_initialization() {
        let samples = create_test_samples(100);
        let metadata = create_test_metadata(100);
        let commitment = create_test_commitment(&samples, &metadata);

        let mut loader = VerifiedDataLoader::new(VerifiedDataLoaderConfig::default());
        let result = loader.initialize(samples, metadata, commitment);

        assert!(result.is_ok());
        assert!(loader.commitment().is_some());
        assert_eq!(loader.num_samples(), 100);
    }

    #[test]
    fn test_verified_loader_batch_verification() {
        let samples = create_test_samples(100);
        let metadata = create_test_metadata(100);
        let commitment = create_test_commitment(&samples, &metadata);

        let config = VerifiedDataLoaderConfig {
            batch_size: 10,
            ..Default::default()
        };
        let mut loader = VerifiedDataLoader::new(config);
        loader.initialize(samples, metadata, commitment).unwrap();

        // Get first batch
        let batch = loader.next_batch().unwrap().unwrap();
        assert!(batch.is_verified);
        assert_eq!(batch.len(), 10);

        // Check stats
        assert_eq!(loader.stats().batches_verified, 1);
        assert_eq!(loader.stats().batches_failed, 0);
    }

    #[test]
    fn test_verified_loader_all_batches() {
        let samples = create_test_samples(100);
        let metadata = create_test_metadata(100);
        let commitment = create_test_commitment(&samples, &metadata);

        let config = VerifiedDataLoaderConfig {
            batch_size: 10,
            ..Default::default()
        };
        let mut loader = VerifiedDataLoader::new(config);
        loader.initialize(samples, metadata, commitment).unwrap();

        let mut batch_count = 0;
        while let Ok(Some(batch)) = loader.next_batch() {
            assert!(batch.is_verified);
            batch_count += 1;
        }

        assert_eq!(batch_count, 10);
        assert_eq!(loader.stats().batches_verified, 10);
    }

    #[test]
    fn test_verified_loader_epoch_reset() {
        let samples = create_test_samples(32);
        let metadata = create_test_metadata(32);
        let commitment = create_test_commitment(&samples, &metadata);

        let config = VerifiedDataLoaderConfig {
            batch_size: 8,
            ..Default::default()
        };
        let mut loader = VerifiedDataLoader::new(config);
        loader.initialize(samples, metadata, commitment).unwrap();

        // Consume all batches
        while let Ok(Some(_)) = loader.next_batch() {}

        assert_eq!(loader.epoch(), 0);

        // Reset for new epoch
        loader.reset_epoch();
        assert_eq!(loader.epoch(), 1);

        // Should be able to get batches again
        let batch = loader.next_batch().unwrap();
        assert!(batch.is_some());
    }

    #[test]
    fn test_commitment_mismatch_detection() {
        let samples = create_test_samples(100);
        let metadata = create_test_metadata(100);

        // Create commitment from different samples
        let other_samples = create_test_samples(100);
        // Modify samples to have different values
        let mut modified_samples = other_samples;
        for s in &mut modified_samples {
            s.features = vec![99u8; s.features.len()];
        }
        let wrong_commitment = create_test_commitment(&modified_samples, &metadata);

        let mut loader = VerifiedDataLoader::new(VerifiedDataLoaderConfig::default());
        let result = loader.initialize(samples, metadata, wrong_commitment);

        assert!(matches!(result, Err(DataVerificationError::CommitmentMismatch { .. })));
    }

    #[test]
    fn test_proof_generation() {
        let samples = create_test_samples(64);
        let metadata = create_test_metadata(64);
        let commitment = create_test_commitment(&samples, &metadata);

        let mut loader = VerifiedDataLoader::new(VerifiedDataLoaderConfig::default());
        loader.initialize(samples.clone(), metadata, commitment.clone()).unwrap();

        // Generate proof for specific indices
        let indices = vec![0, 10, 20, 30];
        let proof = loader.generate_batch_proof(&indices).unwrap();

        // Verify proof
        assert!(proof.verify(&commitment));
        assert_eq!(proof.sample_count(), 4);
    }

    #[test]
    fn test_verification_stats() {
        let samples = create_test_samples(50);
        let metadata = create_test_metadata(50);
        let commitment = create_test_commitment(&samples, &metadata);

        let config = VerifiedDataLoaderConfig {
            batch_size: 10,
            ..Default::default()
        };
        let mut loader = VerifiedDataLoader::new(config);
        loader.initialize(samples, metadata, commitment).unwrap();

        // Process all batches
        while let Ok(Some(_)) = loader.next_batch() {}

        let stats = loader.stats();
        assert_eq!(stats.batches_processed, 5);
        assert_eq!(stats.batches_verified, 5);
        assert_eq!(stats.batches_failed, 0);
        assert_eq!(stats.samples_verified, 50);
        assert!(stats.success_rate() > 0.99);
    }
}
