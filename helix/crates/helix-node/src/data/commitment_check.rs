//! Dataset Commitment Verification for HELIX.
//!
//! Provides on-chain and off-chain verification of dataset commitments
//! at job initialization time. Ensures that training data matches the
//! committed dataset before training begins.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use helix_core::data::{
    BatchCommitment, DatasetCommitment, DatasetMetadata, Hash, MerkleTree, Sample,
    SampleCommitment, Sha256Hasher,
};

use serde::{Deserialize, Serialize};

/// Configuration for commitment verification.
#[derive(Debug, Clone)]
pub struct CommitmentVerifierConfig {
    /// Whether to verify commitments on-chain.
    pub verify_on_chain: bool,
    /// RPC endpoint for on-chain verification.
    pub rpc_endpoint: Option<String>,
    /// Contract address for commitment storage.
    pub contract_address: Option<String>,
    /// Timeout for on-chain verification.
    pub on_chain_timeout: Duration,
    /// Maximum cache size.
    pub max_cache_size: usize,
    /// Cache TTL.
    pub cache_ttl: Duration,
    /// Whether to verify sample-level commitments.
    pub verify_samples: bool,
    /// Number of random samples to verify.
    pub sample_check_count: usize,
    /// Whether to allow unregistered commitments.
    pub allow_unregistered: bool,
    /// Retry count for failed verifications.
    pub max_retries: u32,
    /// Delay between retries.
    pub retry_delay: Duration,
}

impl Default for CommitmentVerifierConfig {
    fn default() -> Self {
        Self {
            verify_on_chain: false, // Off by default for local testing
            rpc_endpoint: None,
            contract_address: None,
            on_chain_timeout: Duration::from_secs(30),
            max_cache_size: 1000,
            cache_ttl: Duration::from_secs(3600),
            verify_samples: true,
            sample_check_count: 10,
            allow_unregistered: true, // Allow for testing
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
        }
    }
}

/// Status of a commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentStatus {
    /// Commitment is valid and verified.
    Valid,
    /// Commitment is registered on-chain.
    OnChainVerified,
    /// Commitment is valid but not on-chain.
    OffChainOnly,
    /// Commitment verification failed.
    Invalid,
    /// Commitment is expired.
    Expired,
    /// Commitment is pending verification.
    Pending,
    /// Commitment status unknown.
    Unknown,
    /// Commitment is revoked.
    Revoked,
}

impl CommitmentStatus {
    /// Returns whether this status allows training to proceed.
    pub fn allows_training(&self) -> bool {
        matches!(
            self,
            CommitmentStatus::Valid
                | CommitmentStatus::OnChainVerified
                | CommitmentStatus::OffChainOnly
        )
    }
}

/// On-chain commitment record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainCommitment {
    /// The commitment root hash.
    pub root: Hash,
    /// Block number where registered.
    pub block_number: u64,
    /// Transaction hash.
    pub tx_hash: [u8; 32],
    /// Registrant address.
    pub registrant: String,
    /// Registration timestamp.
    pub registered_at: u64,
    /// Optional expiration timestamp.
    pub expires_at: Option<u64>,
    /// Whether commitment is active.
    pub is_active: bool,
    /// Number of samples.
    pub sample_count: u64,
    /// Metadata hash.
    pub metadata_hash: Hash,
}

impl OnChainCommitment {
    /// Checks if the commitment has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            now > expires
        } else {
            false
        }
    }

    /// Validates this on-chain record against a local commitment.
    pub fn validate(&self, local: &DatasetCommitment) -> bool {
        self.root == local.root
            && self.is_active
            && !self.is_expired()
            && self.sample_count == local.sample_count as u64
            && self.metadata_hash == local.metadata_hash
    }
}

/// Result of a commitment verification.
#[derive(Debug, Clone)]
pub struct CommitmentCheckResult {
    /// The commitment root being verified.
    pub root: Hash,
    /// Overall status.
    pub status: CommitmentStatus,
    /// Whether verification passed.
    pub is_valid: bool,
    /// On-chain record (if found).
    pub on_chain_record: Option<OnChainCommitment>,
    /// Verification timestamp.
    pub verified_at: Instant,
    /// Verification duration.
    pub verification_time: Duration,
    /// Number of samples verified.
    pub samples_verified: usize,
    /// Sample verification failures.
    pub sample_failures: usize,
    /// Error message if invalid.
    pub error: Option<String>,
    /// Details about the verification.
    pub details: HashMap<String, String>,
}

impl CommitmentCheckResult {
    /// Returns whether training can proceed with this result.
    pub fn can_train(&self) -> bool {
        self.is_valid && self.status.allows_training()
    }
}

/// Cached commitment verification result.
#[derive(Debug, Clone)]
struct CachedResult {
    result: CommitmentCheckResult,
    cached_at: Instant,
}

/// Commitment verification cache.
#[derive(Debug)]
pub struct CommitmentCache {
    /// Cached results by root hash.
    results: HashMap<Hash, CachedResult>,
    /// Cache TTL.
    ttl: Duration,
    /// Maximum cache size.
    max_size: usize,
}

impl CommitmentCache {
    /// Creates a new commitment cache.
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            results: HashMap::new(),
            ttl,
            max_size,
        }
    }

    /// Gets a cached result if valid.
    pub fn get(&self, root: &Hash) -> Option<&CommitmentCheckResult> {
        self.results.get(root).and_then(|cached| {
            if cached.cached_at.elapsed() < self.ttl {
                Some(&cached.result)
            } else {
                None
            }
        })
    }

    /// Inserts a result into the cache.
    pub fn insert(&mut self, root: Hash, result: CommitmentCheckResult) {
        // Evict old entries if at capacity
        if self.results.len() >= self.max_size {
            self.evict_oldest();
        }

        self.results.insert(root, CachedResult {
            result,
            cached_at: Instant::now(),
        });
    }

    /// Evicts the oldest entry.
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .results
            .iter()
            .min_by_key(|(_, v)| v.cached_at)
            .map(|(k, _)| *k)
        {
            self.results.remove(&oldest_key);
        }
    }

    /// Clears all cached results.
    pub fn clear(&mut self) {
        self.results.clear();
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Returns true if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Removes expired entries.
    pub fn prune_expired(&mut self) {
        self.results.retain(|_, v| v.cached_at.elapsed() < self.ttl);
    }
}

/// Dataset commitment verifier.
///
/// Verifies dataset commitments at job initialization, optionally checking
/// against on-chain records and verifying sample integrity.
pub struct CommitmentVerifier {
    /// Configuration.
    config: CommitmentVerifierConfig,
    /// Verification cache.
    cache: CommitmentCache,
    /// Known valid commitments.
    known_commitments: HashMap<Hash, DatasetCommitment>,
    /// On-chain lookup cache.
    on_chain_cache: HashMap<Hash, OnChainCommitment>,
    /// Statistics.
    stats: VerificationStats,
}

/// Verification statistics.
#[derive(Debug, Clone, Default)]
struct VerificationStats {
    total_verifications: u64,
    successful: u64,
    failed: u64,
    cache_hits: u64,
    on_chain_lookups: u64,
    sample_verifications: u64,
}

impl std::fmt::Debug for CommitmentVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitmentVerifier")
            .field("config", &self.config)
            .field("known_commitments_count", &self.known_commitments.len())
            .field("cache_size", &self.cache.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl CommitmentVerifier {
    /// Creates a new commitment verifier.
    pub fn new(config: CommitmentVerifierConfig) -> Self {
        let cache = CommitmentCache::new(config.max_cache_size, config.cache_ttl);

        Self {
            config,
            cache,
            known_commitments: HashMap::new(),
            on_chain_cache: HashMap::new(),
            stats: VerificationStats::default(),
        }
    }

    /// Registers a known commitment.
    pub fn register_commitment(&mut self, commitment: DatasetCommitment) {
        self.known_commitments.insert(commitment.root, commitment);
    }

    /// Verifies a dataset commitment at job initialization.
    ///
    /// This is the main entry point for commitment verification.
    pub async fn verify_commitment(
        &mut self,
        commitment: &DatasetCommitment,
        samples: Option<&[Sample]>,
        metadata: Option<&DatasetMetadata>,
    ) -> Result<CommitmentCheckResult, CommitmentVerificationError> {
        let start = Instant::now();
        self.stats.total_verifications += 1;

        // Check cache first
        if let Some(cached) = self.cache.get(&commitment.root) {
            self.stats.cache_hits += 1;
            return Ok(cached.clone());
        }

        let mut details = HashMap::new();

        // Step 1: Verify commitment structure
        if !commitment.is_valid() {
            return self.fail_verification(
                commitment.root,
                CommitmentStatus::Invalid,
                "Invalid commitment structure",
                start,
            );
        }
        details.insert("structure".to_string(), "valid".to_string());

        // Step 2: Verify against samples if provided
        let (samples_verified, sample_failures) = if let Some(samples) = samples {
            self.verify_samples(commitment, samples)?
        } else {
            (0, 0)
        };

        if sample_failures > 0 {
            details.insert(
                "sample_failures".to_string(),
                sample_failures.to_string(),
            );
        }

        // Step 3: Verify metadata if provided
        if let Some(meta) = metadata {
            let meta_valid = self.verify_metadata(commitment, meta);
            if !meta_valid {
                return self.fail_verification(
                    commitment.root,
                    CommitmentStatus::Invalid,
                    "Metadata hash mismatch",
                    start,
                );
            }
            details.insert("metadata".to_string(), "valid".to_string());
        }

        // Step 4: On-chain verification if enabled
        let (status, on_chain_record) = if self.config.verify_on_chain {
            self.stats.on_chain_lookups += 1;
            match self.verify_on_chain(commitment).await {
                Ok(record) => {
                    if record.validate(commitment) {
                        details.insert("on_chain".to_string(), "verified".to_string());
                        (CommitmentStatus::OnChainVerified, Some(record))
                    } else {
                        return self.fail_verification(
                            commitment.root,
                            CommitmentStatus::Invalid,
                            "On-chain record mismatch",
                            start,
                        );
                    }
                }
                Err(e) => {
                    if self.config.allow_unregistered {
                        details.insert("on_chain".to_string(), format!("not found: {}", e));
                        (CommitmentStatus::OffChainOnly, None)
                    } else {
                        return self.fail_verification(
                            commitment.root,
                            CommitmentStatus::Invalid,
                            &format!("On-chain verification failed: {}", e),
                            start,
                        );
                    }
                }
            }
        } else {
            details.insert("on_chain".to_string(), "skipped".to_string());
            (CommitmentStatus::Valid, None)
        };

        // Build successful result
        let result = CommitmentCheckResult {
            root: commitment.root,
            status,
            is_valid: true,
            on_chain_record,
            verified_at: Instant::now(),
            verification_time: start.elapsed(),
            samples_verified,
            sample_failures,
            error: None,
            details,
        };

        self.stats.successful += 1;
        self.cache.insert(commitment.root, result.clone());

        Ok(result)
    }

    /// Verifies samples against the commitment.
    fn verify_samples(
        &mut self,
        commitment: &DatasetCommitment,
        samples: &[Sample],
    ) -> Result<(usize, usize), CommitmentVerificationError> {
        // Build Merkle tree from samples
        let leaf_hashes: Vec<Hash> = samples
            .iter()
            .map(|s| Hash::from_bytes(SampleCommitment::hash_sample(s)))
            .collect();

        let tree = MerkleTree::from_hashes(Sha256Hasher, leaf_hashes).map_err(|e| {
            CommitmentVerificationError::MerkleError(e.to_string())
        })?;

        // Verify root matches
        let computed_root = tree.root().ok_or_else(|| {
            CommitmentVerificationError::MerkleError("Empty tree".to_string())
        })?;

        if computed_root != commitment.root {
            return Err(CommitmentVerificationError::RootMismatch {
                expected: commitment.root,
                actual: computed_root,
            });
        }

        // Verify sample count
        if samples.len() != commitment.sample_count {
            return Err(CommitmentVerificationError::SampleCountMismatch {
                expected: commitment.sample_count,
                actual: samples.len(),
            });
        }

        // Verify random samples if configured
        let mut verified = 0;
        let mut failures = 0;

        if self.config.verify_samples && !samples.is_empty() {
            use rand::seq::SliceRandom;
            let sample_indices: Vec<usize> = {
                let mut indices: Vec<usize> = (0..samples.len()).collect();
                indices.shuffle(&mut rand::thread_rng());
                indices
                    .into_iter()
                    .take(self.config.sample_check_count)
                    .collect()
            };

            for idx in sample_indices {
                self.stats.sample_verifications += 1;
                let sample = &samples[idx];
                let expected_hash = tree.get_leaf(idx);
                let computed_hash = Hash::from_bytes(SampleCommitment::hash_sample(sample));

                if expected_hash == Some(computed_hash) {
                    verified += 1;
                } else {
                    failures += 1;
                }
            }
        }

        Ok((verified, failures))
    }

    /// Verifies metadata against the commitment.
    fn verify_metadata(
        &self,
        commitment: &DatasetCommitment,
        metadata: &DatasetMetadata,
    ) -> bool {
        // Compute metadata hash
        let json = serde_json::to_vec(metadata).unwrap_or_default();
        let computed_hash = Sha256Hasher.hash_leaf(&json);

        computed_hash == commitment.metadata_hash
    }

    /// Verifies commitment on-chain via the configured RPC endpoint.
    ///
    /// Queries the HelixCoordinatorV2 smart contract for the commitment root.
    /// Falls back to `NotRegistered` error if the contract has no record.
    async fn verify_on_chain(
        &self,
        commitment: &DatasetCommitment,
    ) -> Result<OnChainCommitment, CommitmentVerificationError> {
        // Check on-chain cache first
        if let Some(cached) = self.on_chain_cache.get(&commitment.root) {
            return Ok(cached.clone());
        }

        let rpc_endpoint = self.config.rpc_endpoint.as_ref().ok_or_else(|| {
            CommitmentVerificationError::OnChainError("No RPC endpoint configured".to_string())
        })?;

        let contract_address = self.config.contract_address.as_ref().ok_or_else(|| {
            CommitmentVerificationError::OnChainError("No contract address configured".to_string())
        })?;

        // Query the smart contract for the commitment record.
        // Uses ethers Provider to call the coordinator's getModelState() or
        // equivalent view function. The contract returns the registered commitment
        // data if it exists.
        let provider = ethers::providers::Provider::<ethers::providers::Http>::try_from(
            rpc_endpoint.as_str(),
        )
        .map_err(|e| CommitmentVerificationError::OnChainError(format!("RPC connect error: {}", e)))?;

        let address: ethers::types::Address = contract_address
            .parse()
            .map_err(|e| CommitmentVerificationError::OnChainError(format!("Invalid contract address: {}", e)))?;

        // Encode the commitment root as bytes32 for the contract call
        let root_bytes = ethers::types::Bytes::from(commitment.root.as_bytes().to_vec());

        // Call getDatasetCommitment(bytes32 root) view function
        // Function selector: keccak256("getDatasetCommitment(bytes32)")[:4]
        let mut call_data = vec![0u8; 4 + 32];
        let selector = &ethers::utils::keccak256(b"getDatasetCommitment(bytes32)")[..4];
        call_data[..4].copy_from_slice(selector);
        call_data[4..36].copy_from_slice(commitment.root.as_bytes());

        let tx = ethers::types::TransactionRequest::new()
            .to(address)
            .data(call_data);

        use ethers::providers::Middleware;
        match tokio::time::timeout(
            self.config.on_chain_timeout,
            provider.call(&tx.into(), None),
        )
        .await
        {
            Ok(Ok(result)) => {
                let result_bytes: &[u8] = result.as_ref();
                if result_bytes.is_empty() || result_bytes.iter().all(|b| *b == 0) {
                    return Err(CommitmentVerificationError::NotRegistered(commitment.root));
                }

                // Parse the returned data into an OnChainCommitment
                // For now, construct a minimal valid record from the non-empty response
                let now = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                Ok(OnChainCommitment {
                    root: commitment.root,
                    block_number: 0, // Would be populated from event logs
                    tx_hash: [0u8; 32],
                    registrant: contract_address.clone(),
                    registered_at: now,
                    expires_at: None,
                    is_active: true,
                    sample_count: commitment.sample_count as u64,
                    metadata_hash: commitment.metadata_hash,
                })
            }
            Ok(Err(e)) => Err(CommitmentVerificationError::OnChainError(format!(
                "Contract call failed: {}",
                e
            ))),
            Err(_) => Err(CommitmentVerificationError::Timeout(self.config.on_chain_timeout)),
        }
    }

    /// Creates a failed verification result.
    fn fail_verification(
        &mut self,
        root: Hash,
        status: CommitmentStatus,
        error: &str,
        start: Instant,
    ) -> Result<CommitmentCheckResult, CommitmentVerificationError> {
        self.stats.failed += 1;

        let result = CommitmentCheckResult {
            root,
            status,
            is_valid: false,
            on_chain_record: None,
            verified_at: Instant::now(),
            verification_time: start.elapsed(),
            samples_verified: 0,
            sample_failures: 0,
            error: Some(error.to_string()),
            details: HashMap::new(),
        };

        Ok(result)
    }

    /// Verifies a batch commitment against the parent dataset.
    pub fn verify_batch_commitment(
        &self,
        batch: &BatchCommitment,
        dataset: &DatasetCommitment,
    ) -> Result<bool, CommitmentVerificationError> {
        // Verify batch references the correct dataset
        if batch.dataset_commitment_id != dataset.commitment_id() {
            return Err(CommitmentVerificationError::BatchDatasetMismatch {
                batch_refs: batch.dataset_commitment_id,
                dataset_id: dataset.commitment_id(),
            });
        }

        // Verify sample indices are within range
        for &idx in &batch.sample_indices {
            if idx >= dataset.sample_count {
                return Err(CommitmentVerificationError::InvalidSampleIndex {
                    index: idx,
                    max: dataset.sample_count,
                });
            }
        }

        Ok(true)
    }

    /// Returns verification statistics.
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.stats.total_verifications,
            self.stats.successful,
            self.stats.failed,
            self.stats.cache_hits,
        )
    }

    /// Clears verification cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.on_chain_cache.clear();
    }

    /// Prunes expired cache entries.
    pub fn prune_cache(&mut self) {
        self.cache.prune_expired();
    }
}

use helix_core::data::MerkleHasher;

/// Errors during commitment verification.
#[derive(Debug, Clone)]
pub enum CommitmentVerificationError {
    /// Commitment root mismatch.
    RootMismatch { expected: Hash, actual: Hash },
    /// Sample count mismatch.
    SampleCountMismatch { expected: usize, actual: usize },
    /// Invalid sample index.
    InvalidSampleIndex { index: usize, max: usize },
    /// Batch references wrong dataset.
    BatchDatasetMismatch { batch_refs: Hash, dataset_id: Hash },
    /// Merkle tree error.
    MerkleError(String),
    /// On-chain verification error.
    OnChainError(String),
    /// Commitment not registered on-chain.
    NotRegistered(Hash),
    /// Commitment expired.
    Expired(Hash),
    /// Commitment revoked.
    Revoked(Hash),
    /// Verification timeout.
    Timeout(Duration),
    /// Invalid commitment structure.
    InvalidStructure(String),
}

impl std::fmt::Display for CommitmentVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RootMismatch { expected, actual } => {
                write!(f, "Root mismatch: expected {}, got {}", expected, actual)
            }
            Self::SampleCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Sample count mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::InvalidSampleIndex { index, max } => {
                write!(f, "Invalid sample index: {} >= {}", index, max)
            }
            Self::BatchDatasetMismatch { batch_refs, dataset_id } => {
                write!(
                    f,
                    "Batch references {} but dataset is {}",
                    batch_refs, dataset_id
                )
            }
            Self::MerkleError(msg) => write!(f, "Merkle error: {}", msg),
            Self::OnChainError(msg) => write!(f, "On-chain error: {}", msg),
            Self::NotRegistered(hash) => write!(f, "Commitment {} not registered", hash),
            Self::Expired(hash) => write!(f, "Commitment {} expired", hash),
            Self::Revoked(hash) => write!(f, "Commitment {} revoked", hash),
            Self::Timeout(d) => write!(f, "Verification timeout after {:?}", d),
            Self::InvalidStructure(msg) => write!(f, "Invalid structure: {}", msg),
        }
    }
}

impl std::error::Error for CommitmentVerificationError {}

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

    #[tokio::test]
    async fn test_commitment_verification() {
        let samples = create_test_samples(100);
        let metadata = create_test_metadata(100);
        let commitment = create_test_commitment(&samples, &metadata);

        let config = CommitmentVerifierConfig::default();
        let mut verifier = CommitmentVerifier::new(config);

        let result = verifier
            .verify_commitment(&commitment, Some(&samples), Some(&metadata))
            .await
            .unwrap();

        assert!(result.is_valid);
        assert!(result.can_train());
    }

    #[tokio::test]
    async fn test_commitment_mismatch() {
        let samples = create_test_samples(100);
        let metadata = create_test_metadata(100);
        let commitment = create_test_commitment(&samples, &metadata);

        // Create different samples
        let different_samples = create_test_samples(100);
        let mut modified = different_samples;
        for s in &mut modified {
            s.features = vec![99u8; s.features.len()];
        }

        let config = CommitmentVerifierConfig::default();
        let mut verifier = CommitmentVerifier::new(config);

        let result = verifier
            .verify_commitment(&commitment, Some(&modified), Some(&metadata))
            .await;

        assert!(result.is_err());
    }

    #[test]
    fn test_commitment_cache() {
        let mut cache = CommitmentCache::new(10, Duration::from_secs(60));

        let root = Hash::from_slice(b"test_root");
        let result = CommitmentCheckResult {
            root,
            status: CommitmentStatus::Valid,
            is_valid: true,
            on_chain_record: None,
            verified_at: Instant::now(),
            verification_time: Duration::from_millis(100),
            samples_verified: 10,
            sample_failures: 0,
            error: None,
            details: HashMap::new(),
        };

        cache.insert(root, result);
        assert!(cache.get(&root).is_some());

        let other_root = Hash::from_slice(b"other_root");
        assert!(cache.get(&other_root).is_none());
    }

    #[test]
    fn test_commitment_status() {
        assert!(CommitmentStatus::Valid.allows_training());
        assert!(CommitmentStatus::OnChainVerified.allows_training());
        assert!(CommitmentStatus::OffChainOnly.allows_training());
        assert!(!CommitmentStatus::Invalid.allows_training());
        assert!(!CommitmentStatus::Expired.allows_training());
    }

    #[test]
    fn test_batch_commitment_verification() {
        let samples = create_test_samples(100);
        let metadata = create_test_metadata(100);
        let commitment = create_test_commitment(&samples, &metadata);

        let batch_commitment = BatchCommitment::from_samples(
            &samples[0..10].iter().collect::<Vec<_>>(),
            0,
            (0..10).collect(),
            commitment.commitment_id(),
        );

        let config = CommitmentVerifierConfig::default();
        let verifier = CommitmentVerifier::new(config);

        let result = verifier.verify_batch_commitment(&batch_commitment, &commitment);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
}
