//! Deterministic Shuffling with Seed Commitment for HELIX.
//!
//! Provides cryptographically-sound deterministic shuffling of datasets
//! that enables:
//! - Reproducible shuffling across distributed workers
//! - Verifiable shuffling through seed commitments
//! - Epoch-based shuffling with committed schedules
//! - Fisher-Yates shuffle with committed randomness
//! - Multi-phase shuffling for curriculum learning
//!
//! The shuffling is deterministic given a seed, allowing verification
//! that all workers used the same shuffle order.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::data::merkle::{Hash, MerkleHasher, Sha256Hasher};

/// Seed for deterministic shuffling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShuffleSeed(pub [u8; 32]);

impl ShuffleSeed {
    /// Creates a seed from bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Creates a seed from a slice, padding or truncating.
    pub fn from_slice(slice: &[u8]) -> Self {
        let mut bytes = [0u8; 32];
        let len = slice.len().min(32);
        bytes[..len].copy_from_slice(&slice[..len]);
        Self(bytes)
    }

    /// Creates a seed from a u64.
    pub fn from_u64(value: u64) -> Self {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&value.to_le_bytes());
        Self(bytes)
    }

    /// Creates a seed from combining epoch and base seed.
    pub fn for_epoch(base_seed: &ShuffleSeed, epoch: u64) -> Self {
        let hasher = Sha256Hasher;
        let mut combined = base_seed.0.to_vec();
        combined.extend_from_slice(&epoch.to_le_bytes());
        let hash = hasher.hash_leaf(&combined);
        Self(hash.0)
    }

    /// Combines with another seed.
    pub fn combine(&self, other: &ShuffleSeed) -> Self {
        let hasher = Sha256Hasher;
        let mut combined = self.0.to_vec();
        combined.extend_from_slice(&other.0);
        let hash = hasher.hash_leaf(&combined);
        Self(hash.0)
    }

    /// Derives a sub-seed for a specific purpose.
    pub fn derive(&self, purpose: &[u8]) -> Self {
        let hasher = Sha256Hasher;
        let mut data = self.0.to_vec();
        data.extend_from_slice(purpose);
        let hash = hasher.hash_leaf(&data);
        Self(hash.0)
    }

    /// Returns as bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Converts to hex string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Parses from hex string.
    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 64 {
            return None;
        }
        let mut bytes = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk).ok()?;
            bytes[i] = u8::from_str_radix(hex_str, 16).ok()?;
        }
        Some(Self(bytes))
    }
}

impl Default for ShuffleSeed {
    fn default() -> Self {
        Self([0u8; 32])
    }
}

impl From<Hash> for ShuffleSeed {
    fn from(hash: Hash) -> Self {
        Self(hash.0)
    }
}

impl From<ShuffleSeed> for Hash {
    fn from(seed: ShuffleSeed) -> Self {
        Hash(seed.0)
    }
}

/// Commitment to a shuffle seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedCommitment {
    /// Hash of the seed (commitment).
    pub commitment: Hash,
    /// The actual seed (only revealed after commitment).
    pub seed: Option<ShuffleSeed>,
    /// Timestamp of commitment.
    pub committed_at: u64,
    /// Timestamp of reveal (if revealed).
    pub revealed_at: Option<u64>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl SeedCommitment {
    /// Creates a new commitment without revealing the seed.
    pub fn commit(seed: &ShuffleSeed) -> Self {
        let hasher = Sha256Hasher;
        let commitment = hasher.hash_leaf(seed.as_bytes());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            commitment,
            seed: None,
            committed_at: now,
            revealed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a commitment with the seed already revealed.
    pub fn commit_and_reveal(seed: ShuffleSeed) -> Self {
        let hasher = Sha256Hasher;
        let commitment = hasher.hash_leaf(seed.as_bytes());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            commitment,
            seed: Some(seed),
            committed_at: now,
            revealed_at: Some(now),
            metadata: HashMap::new(),
        }
    }

    /// Reveals the seed and verifies it matches the commitment.
    pub fn reveal(&mut self, seed: ShuffleSeed) -> bool {
        let hasher = Sha256Hasher;
        let computed = hasher.hash_leaf(seed.as_bytes());

        if computed == self.commitment {
            self.seed = Some(seed);
            self.revealed_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            );
            true
        } else {
            false
        }
    }

    /// Verifies a seed against this commitment.
    pub fn verify(&self, seed: &ShuffleSeed) -> bool {
        let hasher = Sha256Hasher;
        let computed = hasher.hash_leaf(seed.as_bytes());
        computed == self.commitment
    }

    /// Returns whether the seed has been revealed.
    pub fn is_revealed(&self) -> bool {
        self.seed.is_some()
    }

    /// Gets the revealed seed.
    pub fn get_seed(&self) -> Option<&ShuffleSeed> {
        self.seed.as_ref()
    }
}

/// Configuration for shuffling.
#[derive(Debug, Clone)]
pub struct ShuffleConfig {
    /// Base seed for the shuffle.
    pub seed: ShuffleSeed,
    /// Number of epochs to pre-compute.
    pub num_epochs: usize,
    /// Whether to use epoch-varying seeds.
    pub epoch_varying: bool,
    /// Buffer size for large dataset shuffling.
    pub buffer_size: usize,
    /// Whether to track shuffle history.
    pub track_history: bool,
}

impl Default for ShuffleConfig {
    fn default() -> Self {
        Self {
            seed: ShuffleSeed::default(),
            num_epochs: 1,
            epoch_varying: true,
            buffer_size: 10000,
            track_history: false,
        }
    }
}

/// Result of a shuffle operation.
#[derive(Debug, Clone)]
pub struct ShuffleResult {
    /// The shuffled indices.
    pub indices: Vec<usize>,
    /// The seed used.
    pub seed: ShuffleSeed,
    /// Epoch number (if epoch-varying).
    pub epoch: Option<u64>,
    /// Hash of the shuffle result (for verification).
    pub result_hash: Hash,
}

impl ShuffleResult {
    /// Verifies this shuffle against a commitment.
    pub fn verify_against_commitment(&self, commitment: &SeedCommitment) -> bool {
        commitment.verify(&self.seed)
    }

    /// Creates a commitment for this shuffle result.
    pub fn create_commitment(&self) -> Hash {
        self.result_hash
    }
}

/// Deterministic PRNG based on seed.
struct DeterministicRng {
    state: [u8; 32],
    counter: u64,
}

impl DeterministicRng {
    fn new(seed: &ShuffleSeed) -> Self {
        Self {
            state: seed.0,
            counter: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // Use counter-based construction for determinism
        let hasher = Sha256Hasher;
        let mut data = self.state.to_vec();
        data.extend_from_slice(&self.counter.to_le_bytes());
        self.counter += 1;

        let hash = hasher.hash_leaf(&data);

        // Extract u64 from hash
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&hash.0[..8]);
        u64::from_le_bytes(bytes)
    }

    fn next_bounded(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }

        // Rejection sampling for uniformity
        let bound_u64 = bound as u64;
        let threshold = u64::MAX - (u64::MAX % bound_u64);

        loop {
            let r = self.next_u64();
            if r < threshold {
                return (r % bound_u64) as usize;
            }
        }
    }
}

/// Deterministic shuffler using Fisher-Yates algorithm.
pub struct DeterministicShuffler {
    /// Configuration.
    config: ShuffleConfig,
    /// Seed commitment.
    commitment: SeedCommitment,
    /// Pre-computed epoch seeds.
    epoch_seeds: Vec<ShuffleSeed>,
    /// Shuffle history (if tracking enabled).
    history: Vec<ShuffleResult>,
}

impl DeterministicShuffler {
    /// Creates a new shuffler with a seed.
    pub fn new(seed: ShuffleSeed) -> Self {
        let config = ShuffleConfig {
            seed: seed.clone(),
            ..Default::default()
        };

        let commitment = SeedCommitment::commit_and_reveal(seed);

        Self {
            config,
            commitment,
            epoch_seeds: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Creates with full configuration.
    pub fn with_config(config: ShuffleConfig) -> Self {
        let commitment = SeedCommitment::commit_and_reveal(config.seed.clone());

        let mut shuffler = Self {
            config,
            commitment,
            epoch_seeds: Vec::new(),
            history: Vec::new(),
        };

        // Pre-compute epoch seeds if configured
        shuffler.precompute_epoch_seeds();

        shuffler
    }

    /// Pre-computes seeds for all epochs.
    fn precompute_epoch_seeds(&mut self) {
        self.epoch_seeds.clear();
        for epoch in 0..self.config.num_epochs {
            let seed = if self.config.epoch_varying {
                ShuffleSeed::for_epoch(&self.config.seed, epoch as u64)
            } else {
                self.config.seed.clone()
            };
            self.epoch_seeds.push(seed);
        }
    }

    /// Returns the seed commitment.
    pub fn commitment(&self) -> &SeedCommitment {
        &self.commitment
    }

    /// Shuffles indices from 0 to n-1.
    pub fn shuffle(&mut self, n: usize) -> ShuffleResult {
        self.shuffle_with_epoch(n, 0)
    }

    /// Shuffles with a specific epoch.
    pub fn shuffle_with_epoch(&mut self, n: usize, epoch: u64) -> ShuffleResult {
        let seed = if self.config.epoch_varying {
            if (epoch as usize) < self.epoch_seeds.len() {
                self.epoch_seeds[epoch as usize].clone()
            } else {
                ShuffleSeed::for_epoch(&self.config.seed, epoch)
            }
        } else {
            self.config.seed.clone()
        };

        let indices = self.fisher_yates_shuffle(n, &seed);
        let result_hash = self.compute_shuffle_hash(&indices, &seed);

        let result = ShuffleResult {
            indices,
            seed,
            epoch: Some(epoch),
            result_hash,
        };

        if self.config.track_history {
            self.history.push(result.clone());
        }

        result
    }

    /// Shuffles a slice in place.
    pub fn shuffle_slice<T>(&self, slice: &mut [T], seed: &ShuffleSeed) {
        let n = slice.len();
        if n <= 1 {
            return;
        }

        let mut rng = DeterministicRng::new(seed);

        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.next_bounded(i + 1);
            slice.swap(i, j);
        }
    }

    /// Internal Fisher-Yates shuffle implementation.
    fn fisher_yates_shuffle(&self, n: usize, seed: &ShuffleSeed) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = DeterministicRng::new(seed);

        // Fisher-Yates: iterate from end to beginning
        for i in (1..n).rev() {
            let j = rng.next_bounded(i + 1);
            indices.swap(i, j);
        }

        indices
    }

    /// Computes a hash of the shuffle result.
    fn compute_shuffle_hash(&self, indices: &[usize], seed: &ShuffleSeed) -> Hash {
        let hasher = Sha256Hasher;

        // Hash the seed first
        let mut data = seed.0.to_vec();

        // Add indices (efficiently - just sample for large arrays)
        if indices.len() <= 1000 {
            for &idx in indices {
                data.extend_from_slice(&(idx as u64).to_le_bytes());
            }
        } else {
            // Sample for large arrays
            data.extend_from_slice(&(indices.len() as u64).to_le_bytes());
            for i in (0..indices.len()).step_by(indices.len() / 100) {
                data.extend_from_slice(&(indices[i] as u64).to_le_bytes());
            }
        }

        hasher.hash_leaf(&data)
    }

    /// Gets the inverse permutation (for unshuffling).
    pub fn inverse_permutation(indices: &[usize]) -> Vec<usize> {
        let n = indices.len();
        let mut inverse = vec![0; n];
        for (i, &idx) in indices.iter().enumerate() {
            if idx < n {
                inverse[idx] = i;
            }
        }
        inverse
    }

    /// Verifies that a shuffle matches expected indices.
    pub fn verify_shuffle(&self, indices: &[usize], seed: &ShuffleSeed) -> bool {
        let expected = self.fisher_yates_shuffle(indices.len(), seed);
        indices == expected
    }

    /// Returns shuffle history.
    pub fn history(&self) -> &[ShuffleResult] {
        &self.history
    }

    /// Gets the seed for a specific epoch.
    pub fn epoch_seed(&self, epoch: u64) -> ShuffleSeed {
        if self.config.epoch_varying {
            if (epoch as usize) < self.epoch_seeds.len() {
                self.epoch_seeds[epoch as usize].clone()
            } else {
                ShuffleSeed::for_epoch(&self.config.seed, epoch)
            }
        } else {
            self.config.seed.clone()
        }
    }
}

/// Shuffle schedule for multi-epoch training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShuffleSchedule {
    /// Base seed commitment.
    pub base_commitment: Hash,
    /// Epoch seeds (committed but not revealed until needed).
    pub epoch_commitments: Vec<Hash>,
    /// Number of samples in the dataset.
    pub dataset_size: usize,
    /// Number of epochs in the schedule.
    pub num_epochs: usize,
    /// Creation timestamp.
    pub created_at: u64,
}

impl ShuffleSchedule {
    /// Creates a new shuffle schedule.
    pub fn new(base_seed: &ShuffleSeed, dataset_size: usize, num_epochs: usize) -> Self {
        let hasher = Sha256Hasher;

        // Commit to base seed
        let base_commitment = hasher.hash_leaf(base_seed.as_bytes());

        // Commit to each epoch's seed
        let epoch_commitments: Vec<Hash> = (0..num_epochs)
            .map(|epoch| {
                let epoch_seed = ShuffleSeed::for_epoch(base_seed, epoch as u64);
                hasher.hash_leaf(epoch_seed.as_bytes())
            })
            .collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            base_commitment,
            epoch_commitments,
            dataset_size,
            num_epochs,
            created_at: now,
        }
    }

    /// Verifies a seed against this schedule.
    pub fn verify_base_seed(&self, seed: &ShuffleSeed) -> bool {
        let hasher = Sha256Hasher;
        let computed = hasher.hash_leaf(seed.as_bytes());
        computed == self.base_commitment
    }

    /// Verifies an epoch seed.
    pub fn verify_epoch_seed(&self, seed: &ShuffleSeed, epoch: usize) -> bool {
        if epoch >= self.epoch_commitments.len() {
            return false;
        }

        let hasher = Sha256Hasher;
        let computed = hasher.hash_leaf(seed.as_bytes());
        computed == self.epoch_commitments[epoch]
    }

    /// Gets the commitment for an epoch.
    pub fn epoch_commitment(&self, epoch: usize) -> Option<Hash> {
        self.epoch_commitments.get(epoch).copied()
    }
}

/// Verifier for shuffle commitments.
pub struct ShuffleVerifier {
    /// The schedule to verify against.
    schedule: ShuffleSchedule,
}

impl ShuffleVerifier {
    /// Creates a new verifier.
    pub fn new(schedule: ShuffleSchedule) -> Self {
        Self { schedule }
    }

    /// Verifies a complete shuffle.
    pub fn verify_shuffle(&self, result: &ShuffleResult, epoch: usize) -> bool {
        // Verify seed matches commitment
        if !self.schedule.verify_epoch_seed(&result.seed, epoch) {
            return false;
        }

        // Verify indices are valid
        if result.indices.len() != self.schedule.dataset_size {
            return false;
        }

        // Verify indices form a valid permutation
        let mut seen = vec![false; self.schedule.dataset_size];
        for &idx in &result.indices {
            if idx >= self.schedule.dataset_size || seen[idx] {
                return false;
            }
            seen[idx] = true;
        }

        // Verify shuffle is deterministic (recompute and compare)
        let expected = {
            let shuffler = DeterministicShuffler::new(result.seed.clone());
            shuffler.fisher_yates_shuffle(self.schedule.dataset_size, &result.seed)
        };

        result.indices == expected
    }

    /// Verifies just the seed against the schedule.
    pub fn verify_seed(&self, seed: &ShuffleSeed, epoch: usize) -> bool {
        self.schedule.verify_epoch_seed(seed, epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_creation() {
        let seed = ShuffleSeed::from_u64(12345);
        assert_ne!(seed.0, [0u8; 32]);

        let seed2 = ShuffleSeed::from_u64(12345);
        assert_eq!(seed, seed2);
    }

    #[test]
    fn test_seed_hex_roundtrip() {
        let seed = ShuffleSeed::from_u64(999);
        let hex = seed.to_hex();
        let restored = ShuffleSeed::from_hex(&hex).unwrap();
        assert_eq!(seed, restored);
    }

    #[test]
    fn test_epoch_seed_derivation() {
        let base = ShuffleSeed::from_u64(42);
        let epoch0 = ShuffleSeed::for_epoch(&base, 0);
        let epoch1 = ShuffleSeed::for_epoch(&base, 1);
        let epoch0_again = ShuffleSeed::for_epoch(&base, 0);

        // Different epochs should produce different seeds
        assert_ne!(epoch0, epoch1);

        // Same epoch should produce same seed
        assert_eq!(epoch0, epoch0_again);
    }

    #[test]
    fn test_seed_commitment() {
        let seed = ShuffleSeed::from_u64(12345);
        let mut commitment = SeedCommitment::commit(&seed);

        // Seed should not be revealed yet
        assert!(!commitment.is_revealed());

        // Wrong seed should not verify
        let wrong_seed = ShuffleSeed::from_u64(99999);
        assert!(!commitment.reveal(wrong_seed));

        // Correct seed should verify
        assert!(commitment.reveal(seed));
        assert!(commitment.is_revealed());
    }

    #[test]
    fn test_deterministic_shuffle() {
        let seed = ShuffleSeed::from_u64(42);
        let mut shuffler = DeterministicShuffler::new(seed.clone());

        let result1 = shuffler.shuffle(100);
        let _result2 = shuffler.shuffle(100);

        // Same seed should produce same shuffle (when same epoch)
        let mut shuffler2 = DeterministicShuffler::new(seed);
        let result3 = shuffler2.shuffle(100);

        assert_eq!(result1.indices, result3.indices);
    }

    #[test]
    fn test_shuffle_is_valid_permutation() {
        let seed = ShuffleSeed::from_u64(123);
        let mut shuffler = DeterministicShuffler::new(seed);

        let result = shuffler.shuffle(1000);

        // Check it's a valid permutation
        let mut sorted = result.indices.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..1000).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_shuffle_verification() {
        let seed = ShuffleSeed::from_u64(789);
        let shuffler = DeterministicShuffler::new(seed.clone());

        let indices = shuffler.fisher_yates_shuffle(100, &seed);

        assert!(shuffler.verify_shuffle(&indices, &seed));

        // Wrong indices should fail
        let mut wrong = indices.clone();
        wrong.swap(0, 1);
        assert!(!shuffler.verify_shuffle(&wrong, &seed));
    }

    #[test]
    fn test_inverse_permutation() {
        let seed = ShuffleSeed::from_u64(42);
        let mut shuffler = DeterministicShuffler::new(seed);

        let result = shuffler.shuffle(100);
        let inverse = DeterministicShuffler::inverse_permutation(&result.indices);

        // Applying inverse should recover original order
        let mut recovered: Vec<usize> = vec![0; 100];
        for (new_pos, &old_pos) in result.indices.iter().enumerate() {
            recovered[old_pos] = new_pos;
        }

        assert_eq!(inverse, recovered);
    }

    #[test]
    fn test_shuffle_schedule() {
        let seed = ShuffleSeed::from_u64(42);
        let schedule = ShuffleSchedule::new(&seed, 1000, 10);

        assert_eq!(schedule.num_epochs, 10);
        assert_eq!(schedule.dataset_size, 1000);
        assert!(schedule.verify_base_seed(&seed));

        // Verify epoch seeds
        for epoch in 0..10 {
            let epoch_seed = ShuffleSeed::for_epoch(&seed, epoch as u64);
            assert!(schedule.verify_epoch_seed(&epoch_seed, epoch));
        }
    }

    #[test]
    fn test_shuffle_verifier() {
        let seed = ShuffleSeed::from_u64(42);
        let schedule = ShuffleSchedule::new(&seed, 100, 5);
        let verifier = ShuffleVerifier::new(schedule);

        let config = ShuffleConfig {
            seed: seed.clone(),
            epoch_varying: true,
            ..Default::default()
        };
        let mut shuffler = DeterministicShuffler::with_config(config);

        let result = shuffler.shuffle_with_epoch(100, 2);

        assert!(verifier.verify_shuffle(&result, 2));
    }

    #[test]
    fn test_epoch_varying_shuffles() {
        let seed = ShuffleSeed::from_u64(42);
        let config = ShuffleConfig {
            seed,
            epoch_varying: true,
            num_epochs: 5,
            ..Default::default()
        };
        let mut shuffler = DeterministicShuffler::with_config(config);

        let epoch0 = shuffler.shuffle_with_epoch(100, 0);
        let epoch1 = shuffler.shuffle_with_epoch(100, 1);
        let epoch0_again = shuffler.shuffle_with_epoch(100, 0);

        // Different epochs should have different orders
        assert_ne!(epoch0.indices, epoch1.indices);

        // Same epoch should have same order
        assert_eq!(epoch0.indices, epoch0_again.indices);
    }

    #[test]
    fn test_shuffle_slice() {
        let seed = ShuffleSeed::from_u64(42);
        let shuffler = DeterministicShuffler::new(seed.clone());

        let mut data1: Vec<i32> = (0..100).collect();
        let mut data2: Vec<i32> = (0..100).collect();

        shuffler.shuffle_slice(&mut data1, &seed);
        shuffler.shuffle_slice(&mut data2, &seed);

        // Same seed should produce same shuffle
        assert_eq!(data1, data2);

        // Should be shuffled (not in original order)
        let original: Vec<i32> = (0..100).collect();
        assert_ne!(data1, original);
    }
}
