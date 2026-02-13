//! Dataset Distribution & Training Synchronization.
//!
//! Ensures all workers in a training round train on identical data in identical order:
//!
//! 1. **DatasetSpec** — Full dataset specification embedded in round config
//! 2. **DatasetManager** — Fetch, cache, and verify datasets by reference
//! 3. **DeterministicBatcher** — Seeded Fisher-Yates shuffle for identical batch ordering
//! 4. **StepAlignmentValidator** — Validates proofs correspond to correct (step, batch) pairs

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// Dataset Specification
// ============================================================================

/// Normalization method for preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NormalizationMethod {
    /// No normalization.
    None,
    /// Scale all features by a fixed divisor (e.g., 255.0 for pixel data).
    Scale { divisor: f64 },
    /// Z-score normalization: (x - mean) / std, computed per-feature.
    ZScore,
    /// Min-max normalization to [0, 1], computed per-feature.
    MinMax,
}

impl Default for NormalizationMethod {
    fn default() -> Self {
        Self::None
    }
}

/// Preprocessing configuration for the dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreprocessingConfig {
    /// Normalization method to apply to features.
    pub normalization: NormalizationMethod,
    /// Column indices to use as input features (empty = all except label columns).
    pub feature_columns: Vec<usize>,
    /// Column indices to use as labels/targets.
    pub label_columns: Vec<usize>,
    /// Whether the CSV has a header row.
    pub has_header: bool,
    /// CSV delimiter character.
    pub delimiter: u8,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalization: NormalizationMethod::None,
            feature_columns: Vec::new(),
            label_columns: Vec::new(),
            has_header: true,
            delimiter: b',',
        }
    }
}

/// Complete dataset specification for a training round.
///
/// Sent as part of `RoundConfigure` so all workers load, preprocess,
/// and batch the same data in the same order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetSpec {
    /// Reference to the dataset (IPFS CID, HTTP URL, S3 path, or local path).
    pub dataset_ref: String,
    /// SHA-256 hash of the raw dataset bytes (for integrity verification).
    pub dataset_hash: [u8; 32],
    /// Preprocessing configuration.
    pub preprocessing: PreprocessingConfig,
    /// Batch size for training.
    pub batch_size: u32,
    /// Deterministic shuffle seed so all workers see the same batch order.
    pub shuffle_seed: u64,
    /// Number of training steps per round (each step = one batch).
    pub steps_per_round: u32,
}

impl DatasetSpec {
    /// Creates a new DatasetSpec with default preprocessing.
    pub fn new(
        dataset_ref: impl Into<String>,
        dataset_hash: [u8; 32],
        batch_size: u32,
        shuffle_seed: u64,
        steps_per_round: u32,
    ) -> Self {
        Self {
            dataset_ref: dataset_ref.into(),
            dataset_hash,
            preprocessing: PreprocessingConfig::default(),
            batch_size,
            shuffle_seed,
            steps_per_round,
        }
    }

    /// Sets the preprocessing configuration.
    pub fn with_preprocessing(mut self, preprocessing: PreprocessingConfig) -> Self {
        self.preprocessing = preprocessing;
        self
    }
}

// ============================================================================
// Dataset Manager — Fetch, Cache, Verify
// ============================================================================

/// Manages dataset downloading, caching, and verification.
pub struct DatasetManager {
    /// Local cache directory.
    cache_dir: PathBuf,
    /// Cached datasets: dataset_hash → local file path.
    cache_index: HashMap<[u8; 32], PathBuf>,
}

impl DatasetManager {
    /// Creates a new DatasetManager with the given cache directory.
    pub fn new(cache_dir: impl AsRef<Path>) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        Self {
            cache_dir,
            cache_index: HashMap::new(),
        }
    }

    /// Creates a DatasetManager using the default cache location (~/.cache/helix/datasets/).
    pub fn default_cache() -> Self {
        let cache_dir = dirs_default();
        Self::new(cache_dir)
    }

    /// Returns the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Checks if a dataset is already cached and verified.
    pub fn is_cached(&self, dataset_hash: &[u8; 32]) -> bool {
        if let Some(path) = self.cache_index.get(dataset_hash) {
            path.exists()
        } else {
            // Check disk
            let path = self.hash_to_path(dataset_hash);
            path.exists()
        }
    }

    /// Returns the cached dataset bytes if available and verified.
    pub fn get_cached(&self, dataset_hash: &[u8; 32]) -> Option<Vec<u8>> {
        let path = self.cache_index
            .get(dataset_hash)
            .cloned()
            .unwrap_or_else(|| self.hash_to_path(dataset_hash));

        if !path.exists() {
            return None;
        }

        let data = std::fs::read(&path).ok()?;

        // Verify hash
        if compute_dataset_hash(&data) != *dataset_hash {
            // Corrupted cache entry — remove it
            let _ = std::fs::remove_file(&path);
            return None;
        }

        Some(data)
    }

    /// Stores dataset bytes in the cache after verifying the hash.
    pub fn cache_dataset(
        &mut self,
        data: &[u8],
        expected_hash: &[u8; 32],
    ) -> Result<PathBuf, DatasetError> {
        let actual_hash = compute_dataset_hash(data);
        if actual_hash != *expected_hash {
            return Err(DatasetError::HashMismatch {
                expected: hex::encode(expected_hash),
                actual: hex::encode(actual_hash),
            });
        }

        let path = self.hash_to_path(expected_hash);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| DatasetError::Io(e.to_string()))?;
        }
        std::fs::write(&path, data).map_err(|e| DatasetError::Io(e.to_string()))?;
        self.cache_index.insert(*expected_hash, path.clone());
        Ok(path)
    }

    /// Resolves a dataset reference to raw bytes, using cache if available.
    ///
    /// For local file paths, reads from disk directly.
    /// For IPFS/HTTP/S3, the caller should fetch the data externally
    /// and pass it through `cache_dataset`.
    pub fn resolve_local(&mut self, spec: &DatasetSpec) -> Result<Vec<u8>, DatasetError> {
        // Check cache first
        if let Some(data) = self.get_cached(&spec.dataset_hash) {
            return Ok(data);
        }

        // Try as a local file path
        let path = Path::new(&spec.dataset_ref);
        if path.exists() {
            let data = std::fs::read(path).map_err(|e| DatasetError::Io(e.to_string()))?;
            let actual_hash = compute_dataset_hash(&data);
            if actual_hash != spec.dataset_hash {
                return Err(DatasetError::HashMismatch {
                    expected: hex::encode(spec.dataset_hash),
                    actual: hex::encode(actual_hash),
                });
            }
            self.cache_dataset(&data, &spec.dataset_hash)?;
            return Ok(data);
        }

        Err(DatasetError::NotFound {
            reference: spec.dataset_ref.clone(),
        })
    }

    /// Converts a dataset hash to a cache file path.
    fn hash_to_path(&self, hash: &[u8; 32]) -> PathBuf {
        let hex_hash = hex::encode(hash);
        self.cache_dir.join(format!("{}.dat", &hex_hash[..16]))
    }
}

/// Returns the default cache directory.
fn dirs_default() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".cache").join("helix").join("datasets")
}

// ============================================================================
// Deterministic Batcher
// ============================================================================

/// A sample in the dataset: (features, labels).
pub type Sample = (Vec<f64>, Vec<f64>);

/// Deterministic batch iterator that produces identical batches on all workers.
///
/// Uses Fisher-Yates shuffle with a fixed seed so that given the same dataset
/// and seed, every worker produces the same sequence of batches.
pub struct DeterministicBatcher {
    /// All samples, shuffled deterministically.
    samples: Vec<Sample>,
    /// Batch size.
    batch_size: usize,
    /// Current position in the sample list.
    cursor: usize,
    /// Total steps to produce (may cycle through data).
    total_steps: usize,
    /// Steps produced so far.
    steps_produced: usize,
}

impl DeterministicBatcher {
    /// Creates a new DeterministicBatcher.
    ///
    /// Shuffles `samples` in-place using `shuffle_seed` for deterministic ordering.
    /// `total_steps` is the number of batches to produce (may cycle if steps > batches).
    pub fn new(
        mut samples: Vec<Sample>,
        batch_size: usize,
        shuffle_seed: u64,
        total_steps: usize,
    ) -> Self {
        // Fisher-Yates shuffle with deterministic LCG PRNG
        let n = samples.len();
        if n > 1 {
            let mut rng = shuffle_seed;
            for i in (1..n).rev() {
                rng = lcg_next(rng);
                let j = (rng >> 16) as usize % (i + 1);
                samples.swap(i, j);
            }
        }

        Self {
            samples,
            batch_size,
            cursor: 0,
            total_steps,
            steps_produced: 0,
        }
    }

    /// Returns the next batch, or None if all steps have been produced.
    pub fn next_batch(&mut self) -> Option<Vec<Sample>> {
        if self.steps_produced >= self.total_steps || self.samples.is_empty() {
            return None;
        }

        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            // Cycle through dataset if we reach the end
            if self.cursor >= self.samples.len() {
                self.cursor = 0;
            }
            batch.push(self.samples[self.cursor].clone());
            self.cursor += 1;
        }

        self.steps_produced += 1;
        Some(batch)
    }

    /// Returns the current step number (0-indexed).
    pub fn current_step(&self) -> usize {
        self.steps_produced
    }

    /// Returns total steps configured.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Returns the number of samples in the dataset.
    pub fn dataset_size(&self) -> usize {
        self.samples.len()
    }

    /// Resets the batcher to the beginning (same order, same shuffle).
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.steps_produced = 0;
    }
}

impl Iterator for DeterministicBatcher {
    type Item = Vec<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

/// Simple LCG PRNG for deterministic shuffling.
/// Uses the same constants as the trainer's model init for consistency.
fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1)
}

// ============================================================================
// Step Alignment Validator
// ============================================================================

/// Validates that worker proofs correspond to the correct training steps.
///
/// The aggregator uses this to ensure all proofs it collects are for
/// the same (round_id, step_number) before aggregating.
pub struct StepAlignmentValidator {
    /// Round ID.
    round_id: u64,
    /// Expected steps per worker.
    expected_steps: u32,
}

impl StepAlignmentValidator {
    /// Creates a new validator for the given round.
    pub fn new(round_id: u64, expected_steps: u32) -> Self {
        Self {
            round_id,
            expected_steps,
        }
    }

    /// Validates a proof submission's step count.
    pub fn validate_submission(
        &self,
        round_id: u64,
        steps_completed: u32,
    ) -> Result<(), StepAlignmentError> {
        if round_id != self.round_id {
            return Err(StepAlignmentError::WrongRound {
                expected: self.round_id,
                got: round_id,
            });
        }

        if steps_completed != self.expected_steps {
            return Err(StepAlignmentError::StepCountMismatch {
                expected: self.expected_steps,
                got: steps_completed,
            });
        }

        Ok(())
    }

    /// Validates that all submissions completed the same number of steps.
    pub fn validate_alignment(
        &self,
        submissions: &[(String, u32)], // (worker_id, steps_completed)
    ) -> Result<(), StepAlignmentError> {
        for (worker_id, steps) in submissions {
            if *steps != self.expected_steps {
                return Err(StepAlignmentError::WorkerMisaligned {
                    worker_id: worker_id.clone(),
                    expected: self.expected_steps,
                    got: *steps,
                });
            }
        }
        Ok(())
    }
}

// ============================================================================
// Dataset Parsing Helpers
// ============================================================================

/// Parses raw CSV bytes into (features, labels) samples using the preprocessing config.
pub fn parse_csv_dataset(
    data: &[u8],
    config: &PreprocessingConfig,
) -> Result<Vec<Sample>, DatasetError> {
    let text = std::str::from_utf8(data)
        .map_err(|e| DatasetError::ParseError(format!("Invalid UTF-8: {}", e)))?;

    let mut lines = text.lines();

    // Skip header if configured
    if config.has_header {
        lines.next();
    }

    let delimiter = config.delimiter as char;
    let mut samples = Vec::new();

    for (line_num, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(delimiter).collect();

        // Determine which columns to use
        let feature_cols = if config.feature_columns.is_empty() {
            // Use all columns except label columns
            (0..fields.len())
                .filter(|c| !config.label_columns.contains(c))
                .collect::<Vec<_>>()
        } else {
            config.feature_columns.clone()
        };

        let label_cols = &config.label_columns;

        // Parse features
        let features: Result<Vec<f64>, _> = feature_cols
            .iter()
            .map(|&col| {
                fields
                    .get(col)
                    .ok_or_else(|| {
                        DatasetError::ParseError(format!(
                            "Line {}: column {} out of range (have {})",
                            line_num + 1,
                            col,
                            fields.len()
                        ))
                    })?
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| {
                        DatasetError::ParseError(format!(
                            "Line {}, col {}: {}",
                            line_num + 1,
                            col,
                            e
                        ))
                    })
            })
            .collect();
        let features = features?;

        // Parse labels
        let labels: Result<Vec<f64>, _> = label_cols
            .iter()
            .map(|&col| {
                fields
                    .get(col)
                    .ok_or_else(|| {
                        DatasetError::ParseError(format!(
                            "Line {}: label column {} out of range",
                            line_num + 1,
                            col
                        ))
                    })?
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| {
                        DatasetError::ParseError(format!(
                            "Line {}, label col {}: {}",
                            line_num + 1,
                            col,
                            e
                        ))
                    })
            })
            .collect();
        let labels = labels?;

        samples.push((features, labels));
    }

    if samples.is_empty() {
        return Err(DatasetError::ParseError("No valid samples found".to_string()));
    }

    // Apply normalization
    apply_normalization(&mut samples, &config.normalization)?;

    Ok(samples)
}

/// Applies normalization to feature vectors in-place.
fn apply_normalization(
    samples: &mut [Sample],
    method: &NormalizationMethod,
) -> Result<(), DatasetError> {
    match method {
        NormalizationMethod::None => {}
        NormalizationMethod::Scale { divisor } => {
            if *divisor == 0.0 {
                return Err(DatasetError::ParseError(
                    "Scale divisor cannot be zero".to_string(),
                ));
            }
            for (features, _) in samples.iter_mut() {
                for f in features.iter_mut() {
                    *f /= divisor;
                }
            }
        }
        NormalizationMethod::ZScore => {
            if samples.is_empty() {
                return Ok(());
            }
            let n = samples.len() as f64;
            let dim = samples[0].0.len();

            for col in 0..dim {
                let mean: f64 = samples.iter().map(|(f, _)| f[col]).sum::<f64>() / n;
                let variance: f64 =
                    samples.iter().map(|(f, _)| (f[col] - mean).powi(2)).sum::<f64>() / n;
                let std = variance.sqrt().max(1e-8); // avoid division by zero

                for (features, _) in samples.iter_mut() {
                    features[col] = (features[col] - mean) / std;
                }
            }
        }
        NormalizationMethod::MinMax => {
            if samples.is_empty() {
                return Ok(());
            }
            let dim = samples[0].0.len();

            for col in 0..dim {
                let min = samples
                    .iter()
                    .map(|(f, _)| f[col])
                    .fold(f64::INFINITY, f64::min);
                let max = samples
                    .iter()
                    .map(|(f, _)| f[col])
                    .fold(f64::NEG_INFINITY, f64::max);
                let range = (max - min).max(1e-8);

                for (features, _) in samples.iter_mut() {
                    features[col] = (features[col] - min) / range;
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Hash Helpers
// ============================================================================

/// Computes the SHA-256 hash of dataset bytes.
pub fn compute_dataset_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

// ============================================================================
// Errors
// ============================================================================

/// Dataset-related errors.
#[derive(Debug, Clone)]
pub enum DatasetError {
    /// Dataset hash doesn't match expected value.
    HashMismatch { expected: String, actual: String },
    /// Dataset reference could not be resolved.
    NotFound { reference: String },
    /// IO error.
    Io(String),
    /// Parse error.
    ParseError(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HashMismatch { expected, actual } => {
                write!(
                    f,
                    "Dataset hash mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::NotFound { reference } => {
                write!(f, "Dataset not found: {}", reference)
            }
            Self::Io(msg) => write!(f, "Dataset IO error: {}", msg),
            Self::ParseError(msg) => write!(f, "Dataset parse error: {}", msg),
        }
    }
}

impl std::error::Error for DatasetError {}

/// Step alignment errors.
#[derive(Debug, Clone)]
pub enum StepAlignmentError {
    /// Proof is for the wrong round.
    WrongRound { expected: u64, got: u64 },
    /// Worker completed wrong number of steps.
    StepCountMismatch { expected: u32, got: u32 },
    /// Specific worker is misaligned.
    WorkerMisaligned {
        worker_id: String,
        expected: u32,
        got: u32,
    },
}

impl std::fmt::Display for StepAlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongRound { expected, got } => {
                write!(f, "Wrong round: expected {}, got {}", expected, got)
            }
            Self::StepCountMismatch { expected, got } => {
                write!(
                    f,
                    "Step count mismatch: expected {}, got {}",
                    expected, got
                )
            }
            Self::WorkerMisaligned {
                worker_id,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Worker {} misaligned: expected {} steps, got {}",
                    worker_id, expected, got
                )
            }
        }
    }
}

impl std::error::Error for StepAlignmentError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_csv_data() -> Vec<u8> {
        b"x1,x2,y\n1.0,2.0,3.0\n4.0,5.0,9.0\n7.0,8.0,15.0\n2.0,3.0,5.0\n".to_vec()
    }

    fn make_samples() -> Vec<Sample> {
        vec![
            (vec![1.0, 2.0], vec![3.0]),
            (vec![4.0, 5.0], vec![9.0]),
            (vec![7.0, 8.0], vec![15.0]),
            (vec![2.0, 3.0], vec![5.0]),
        ]
    }

    // ---- DatasetSpec tests ----

    #[test]
    fn test_dataset_spec_creation() {
        let hash = [0xAA; 32];
        let spec = DatasetSpec::new("ipfs://QmTest", hash, 2, 42, 10);
        assert_eq!(spec.dataset_ref, "ipfs://QmTest");
        assert_eq!(spec.dataset_hash, hash);
        assert_eq!(spec.batch_size, 2);
        assert_eq!(spec.shuffle_seed, 42);
        assert_eq!(spec.steps_per_round, 10);
    }

    #[test]
    fn test_dataset_spec_serialization() {
        let hash = [0xBB; 32];
        let spec = DatasetSpec::new("https://example.com/data.csv", hash, 32, 123, 100);
        let json = serde_json::to_string(&spec).unwrap();
        let deserialized: DatasetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, deserialized);
    }

    // ---- DatasetManager tests ----

    #[test]
    fn test_dataset_manager_cache_and_retrieve() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = DatasetManager::new(dir.path());

        let data = b"test dataset content";
        let hash = compute_dataset_hash(data);

        // Not cached initially
        assert!(!mgr.is_cached(&hash));

        // Cache it
        let path = mgr.cache_dataset(data, &hash).unwrap();
        assert!(path.exists());
        assert!(mgr.is_cached(&hash));

        // Retrieve it
        let retrieved = mgr.get_cached(&hash).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_dataset_manager_rejects_wrong_hash() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = DatasetManager::new(dir.path());

        let data = b"some data";
        let wrong_hash = [0xFF; 32];

        let result = mgr.cache_dataset(data, &wrong_hash);
        assert!(result.is_err());
        match result.unwrap_err() {
            DatasetError::HashMismatch { .. } => {}
            other => panic!("Expected HashMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_dataset_manager_corrupted_cache_evicted() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = DatasetManager::new(dir.path());

        let data = b"original data";
        let hash = compute_dataset_hash(data);

        // Cache it
        let path = mgr.cache_dataset(data, &hash).unwrap();

        // Corrupt the cached file
        std::fs::write(&path, b"corrupted").unwrap();

        // Should return None (and remove corrupted file)
        assert!(mgr.get_cached(&hash).is_none());
        assert!(!path.exists());
    }

    #[test]
    fn test_dataset_manager_resolve_local_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("data.csv");
        let data = b"x,y\n1,2\n3,4\n";
        std::fs::write(&file_path, data).unwrap();

        let hash = compute_dataset_hash(data);
        let spec = DatasetSpec::new(file_path.to_str().unwrap(), hash, 2, 42, 1);

        let cache_dir = dir.path().join("cache");
        let mut mgr = DatasetManager::new(&cache_dir);

        let resolved = mgr.resolve_local(&spec).unwrap();
        assert_eq!(resolved, data);

        // Should now be cached
        assert!(mgr.is_cached(&hash));
    }

    // ---- DeterministicBatcher tests ----

    #[test]
    fn test_deterministic_batcher_produces_correct_count() {
        let samples = make_samples();
        let mut batcher = DeterministicBatcher::new(samples, 2, 42, 3);

        let mut count = 0;
        while batcher.next_batch().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_deterministic_batcher_same_seed_same_order() {
        let samples1 = make_samples();
        let samples2 = make_samples();

        let mut batcher1 = DeterministicBatcher::new(samples1, 2, 42, 3);
        let mut batcher2 = DeterministicBatcher::new(samples2, 2, 42, 3);

        for _ in 0..3 {
            let batch1 = batcher1.next_batch().unwrap();
            let batch2 = batcher2.next_batch().unwrap();
            assert_eq!(batch1.len(), batch2.len());
            for (s1, s2) in batch1.iter().zip(batch2.iter()) {
                assert_eq!(s1.0, s2.0, "Features should be identical");
                assert_eq!(s1.1, s2.1, "Labels should be identical");
            }
        }
    }

    #[test]
    fn test_deterministic_batcher_different_seed_different_order() {
        let samples1 = make_samples();
        let samples2 = make_samples();

        let mut batcher1 = DeterministicBatcher::new(samples1, 4, 42, 1);
        let mut batcher2 = DeterministicBatcher::new(samples2, 4, 99, 1);

        let batch1 = batcher1.next_batch().unwrap();
        let batch2 = batcher2.next_batch().unwrap();

        // With different seeds and enough samples, order should differ
        let same = batch1
            .iter()
            .zip(batch2.iter())
            .all(|(s1, s2)| s1.0 == s2.0);
        assert!(!same, "Different seeds should produce different orderings");
    }

    #[test]
    fn test_deterministic_batcher_cycles_data() {
        // 2 samples, batch_size=2, 3 steps → must cycle
        let samples = vec![
            (vec![1.0], vec![10.0]),
            (vec![2.0], vec![20.0]),
        ];
        let mut batcher = DeterministicBatcher::new(samples, 2, 0, 3);

        let b1 = batcher.next_batch().unwrap();
        assert_eq!(b1.len(), 2);

        let b2 = batcher.next_batch().unwrap();
        assert_eq!(b2.len(), 2);

        let b3 = batcher.next_batch().unwrap();
        assert_eq!(b3.len(), 2);

        assert!(batcher.next_batch().is_none());
    }

    #[test]
    fn test_deterministic_batcher_reset() {
        let samples = make_samples();
        let mut batcher = DeterministicBatcher::new(samples, 2, 42, 2);

        let first_run: Vec<_> = (0..2).map(|_| batcher.next_batch().unwrap()).collect();
        assert!(batcher.next_batch().is_none());

        batcher.reset();
        let second_run: Vec<_> = (0..2).map(|_| batcher.next_batch().unwrap()).collect();

        // Same order after reset
        for (b1, b2) in first_run.iter().zip(second_run.iter()) {
            for (s1, s2) in b1.iter().zip(b2.iter()) {
                assert_eq!(s1.0, s2.0);
                assert_eq!(s1.1, s2.1);
            }
        }
    }

    // ---- CSV Parsing tests ----

    #[test]
    fn test_parse_csv_dataset_basic() {
        let data = make_csv_data();
        let config = PreprocessingConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            delimiter: b',',
            normalization: NormalizationMethod::None,
        };

        let samples = parse_csv_dataset(&data, &config).unwrap();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].0, vec![1.0, 2.0]);
        assert_eq!(samples[0].1, vec![3.0]);
        assert_eq!(samples[2].0, vec![7.0, 8.0]);
        assert_eq!(samples[2].1, vec![15.0]);
    }

    #[test]
    fn test_parse_csv_with_zscore_normalization() {
        let data = make_csv_data();
        let config = PreprocessingConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            delimiter: b',',
            normalization: NormalizationMethod::ZScore,
        };

        let samples = parse_csv_dataset(&data, &config).unwrap();
        assert_eq!(samples.len(), 4);

        // After z-score, mean should be ~0 and std ~1
        let mean_col0: f64 = samples.iter().map(|(f, _)| f[0]).sum::<f64>() / 4.0;
        assert!(mean_col0.abs() < 1e-10, "Mean should be ~0, got {}", mean_col0);
    }

    #[test]
    fn test_parse_csv_with_minmax_normalization() {
        let data = make_csv_data();
        let config = PreprocessingConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            delimiter: b',',
            normalization: NormalizationMethod::MinMax,
        };

        let samples = parse_csv_dataset(&data, &config).unwrap();

        // After min-max, values should be in [0, 1]
        for (features, _) in &samples {
            for &f in features {
                assert!(f >= 0.0 && f <= 1.0, "Value {} not in [0,1]", f);
            }
        }
    }

    #[test]
    fn test_parse_csv_auto_feature_columns() {
        let data = make_csv_data();
        let config = PreprocessingConfig {
            feature_columns: vec![], // auto: all except labels
            label_columns: vec![2],
            has_header: true,
            delimiter: b',',
            normalization: NormalizationMethod::None,
        };

        let samples = parse_csv_dataset(&data, &config).unwrap();
        assert_eq!(samples[0].0.len(), 2); // columns 0 and 1
        assert_eq!(samples[0].1.len(), 1); // column 2
    }

    #[test]
    fn test_parse_csv_empty_data() {
        let data = b"x,y\n";
        let config = PreprocessingConfig {
            label_columns: vec![1],
            ..PreprocessingConfig::default()
        };

        let result = parse_csv_dataset(data, &config);
        assert!(result.is_err());
    }

    // ---- StepAlignmentValidator tests ----

    #[test]
    fn test_step_alignment_valid() {
        let validator = StepAlignmentValidator::new(1, 10);
        assert!(validator.validate_submission(1, 10).is_ok());
    }

    #[test]
    fn test_step_alignment_wrong_round() {
        let validator = StepAlignmentValidator::new(1, 10);
        let result = validator.validate_submission(2, 10);
        assert!(matches!(result, Err(StepAlignmentError::WrongRound { .. })));
    }

    #[test]
    fn test_step_alignment_wrong_steps() {
        let validator = StepAlignmentValidator::new(1, 10);
        let result = validator.validate_submission(1, 5);
        assert!(matches!(
            result,
            Err(StepAlignmentError::StepCountMismatch { .. })
        ));
    }

    #[test]
    fn test_step_alignment_batch_validation() {
        let validator = StepAlignmentValidator::new(1, 10);
        let submissions = vec![
            ("w1".to_string(), 10),
            ("w2".to_string(), 10),
            ("w3".to_string(), 10),
        ];
        assert!(validator.validate_alignment(&submissions).is_ok());
    }

    #[test]
    fn test_step_alignment_batch_misaligned() {
        let validator = StepAlignmentValidator::new(1, 10);
        let submissions = vec![
            ("w1".to_string(), 10),
            ("w2".to_string(), 8), // misaligned
            ("w3".to_string(), 10),
        ];
        let result = validator.validate_alignment(&submissions);
        assert!(matches!(
            result,
            Err(StepAlignmentError::WorkerMisaligned { .. })
        ));
    }

    // ---- Hash tests ----

    #[test]
    fn test_compute_dataset_hash_deterministic() {
        let data = b"hello world";
        let h1 = compute_dataset_hash(data);
        let h2 = compute_dataset_hash(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compute_dataset_hash_different_data() {
        let h1 = compute_dataset_hash(b"data1");
        let h2 = compute_dataset_hash(b"data2");
        assert_ne!(h1, h2);
    }

    // ---- End-to-end integration test ----

    #[test]
    fn test_e2e_dataset_sync_flow() {
        // Simulate: aggregator creates spec, two workers load and batch identically

        // 1. Aggregator creates dataset and spec
        let csv_data = make_csv_data();
        let dataset_hash = compute_dataset_hash(&csv_data);
        let spec = DatasetSpec::new(
            "test://dataset",
            dataset_hash,
            2,    // batch_size
            42,   // shuffle_seed
            3,    // steps_per_round
        ).with_preprocessing(PreprocessingConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            delimiter: b',',
            normalization: NormalizationMethod::None,
        });

        // 2. Both workers parse the dataset
        let samples_w1 = parse_csv_dataset(&csv_data, &spec.preprocessing).unwrap();
        let samples_w2 = parse_csv_dataset(&csv_data, &spec.preprocessing).unwrap();

        // 3. Both workers create batchers with the same seed
        let mut batcher_w1 = DeterministicBatcher::new(
            samples_w1,
            spec.batch_size as usize,
            spec.shuffle_seed,
            spec.steps_per_round as usize,
        );
        let mut batcher_w2 = DeterministicBatcher::new(
            samples_w2,
            spec.batch_size as usize,
            spec.shuffle_seed,
            spec.steps_per_round as usize,
        );

        // 4. Verify all batches are identical across workers
        for step in 0..spec.steps_per_round {
            let b1 = batcher_w1.next_batch().unwrap();
            let b2 = batcher_w2.next_batch().unwrap();

            assert_eq!(b1.len(), b2.len(), "Step {}: batch sizes differ", step);
            for (s1, s2) in b1.iter().zip(b2.iter()) {
                assert_eq!(
                    s1.0, s2.0,
                    "Step {}: feature mismatch across workers",
                    step
                );
                assert_eq!(
                    s1.1, s2.1,
                    "Step {}: label mismatch across workers",
                    step
                );
            }
        }

        // 5. Verify step alignment
        let validator = StepAlignmentValidator::new(1, spec.steps_per_round);
        assert!(validator
            .validate_alignment(&[
                ("worker-1".to_string(), spec.steps_per_round),
                ("worker-2".to_string(), spec.steps_per_round),
            ])
            .is_ok());
    }
}
