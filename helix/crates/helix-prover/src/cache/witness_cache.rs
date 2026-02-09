//! Content-Addressed Witness Cache.
//!
//! Provides deduplication for proof generation by caching proofs keyed by
//! a deterministic hash of the witness. Identical witnesses always produce
//! the same proof, so a cache hit means we can skip the expensive proving
//! step entirely.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// WitnessHash — content-addressed key
// ---------------------------------------------------------------------------

/// A 32-byte content hash of a witness, used as a cache key.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct WitnessHash(pub [u8; 32]);

impl fmt::Display for WitnessHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0[..8] {
            write!(f, "{:02x}", byte)?;
        }
        write!(f, "...")
    }
}

// ---------------------------------------------------------------------------
// WitnessHashBuilder — incremental hasher
// ---------------------------------------------------------------------------

/// Builder for computing a deterministic [`WitnessHash`] from witness components.
pub struct WitnessHashBuilder {
    hasher: Sha256,
}

impl WitnessHashBuilder {
    pub fn new() -> Self {
        Self {
            hasher: Sha256::new(),
        }
    }

    /// Adds model dimensions to the hash.
    pub fn add_dims(mut self, d_in: usize, d_hid: usize, d_out: usize) -> Self {
        self.hasher.update(d_in.to_le_bytes());
        self.hasher.update(d_hid.to_le_bytes());
        self.hasher.update(d_out.to_le_bytes());
        self
    }

    /// Adds field elements to the hash (each element's repr bytes).
    pub fn add_field_elements<F: halo2curves::ff::PrimeField>(mut self, elements: &[F]) -> Self {
        for el in elements {
            self.hasher.update(el.to_repr().as_ref());
        }
        self
    }

    /// Adds a u64 value to the hash.
    pub fn add_u64(mut self, val: u64) -> Self {
        self.hasher.update(val.to_le_bytes());
        self
    }

    /// Finalises the hash and returns a [`WitnessHash`].
    pub fn finish(self) -> WitnessHash {
        let result = self.hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        WitnessHash(hash)
    }
}

impl Default for WitnessHashBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WitnessCachedProof — a cached proof result
// ---------------------------------------------------------------------------

/// A proof stored in the witness cache.
#[derive(Clone)]
pub struct WitnessCachedProof {
    /// Witness hash that produced this proof.
    pub witness_hash: WitnessHash,
    /// Serialised proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs (as 32-byte field element representations).
    pub public_inputs: Vec<[u8; 32]>,
    /// How long the original proof generation took (ms).
    pub generation_time_ms: u64,
    /// Whether this proof has been verified.
    pub verified: bool,
}

impl WitnessCachedProof {
    pub fn new(
        witness_hash: WitnessHash,
        proof: Vec<u8>,
        public_inputs: Vec<[u8; 32]>,
        generation_time_ms: u64,
    ) -> Self {
        Self {
            witness_hash,
            proof,
            public_inputs,
            generation_time_ms,
            verified: false,
        }
    }
}

// ---------------------------------------------------------------------------
// EvictionStrategy
// ---------------------------------------------------------------------------

/// Strategy for evicting entries when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Evict the oldest entry.
    FIFO,
    /// Evict the least recently used entry.
    LRU,
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        Self::LRU
    }
}

// ---------------------------------------------------------------------------
// WitnessCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the witness cache.
#[derive(Debug, Clone)]
pub struct WitnessCacheConfig {
    /// Maximum number of cached proofs.
    pub max_entries: usize,
    /// Eviction strategy.
    pub eviction_strategy: EvictionStrategy,
}

impl Default for WitnessCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            eviction_strategy: EvictionStrategy::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// WitnessCacheStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the witness cache.
#[derive(Debug, Clone, Default)]
pub struct WitnessCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted.
    pub evictions: u64,
    /// Current number of entries.
    pub size: usize,
}

// ---------------------------------------------------------------------------
// WitnessCache (inner implementation)
// ---------------------------------------------------------------------------

struct CacheEntry {
    proof: WitnessCachedProof,
    last_used: u64,
    insert_order: u64,
}

/// Content-addressed witness cache.
pub struct WitnessCache {
    entries: HashMap<WitnessHash, CacheEntry>,
    config: WitnessCacheConfig,
    stats: WitnessCacheStats,
    insert_counter: u64,
}

impl WitnessCache {
    pub fn new(config: WitnessCacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: WitnessCacheStats::default(),
            insert_counter: 0,
        }
    }

    pub fn get(&mut self, hash: &WitnessHash) -> Option<WitnessCachedProof> {
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.last_used = Self::now();
            self.stats.hits += 1;
            Some(entry.proof.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, proof: WitnessCachedProof) {
        let hash = proof.witness_hash;

        if self.entries.len() >= self.config.max_entries && !self.entries.contains_key(&hash) {
            self.evict_one();
        }

        self.insert_counter += 1;
        self.entries.insert(hash, CacheEntry {
            proof,
            last_used: Self::now(),
            insert_order: self.insert_counter,
        });
        self.stats.size = self.entries.len();
    }

    pub fn mark_verified(&mut self, hash: &WitnessHash) {
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.proof.verified = true;
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.size = 0;
    }

    pub fn stats(&self) -> WitnessCacheStats {
        self.stats.clone()
    }

    fn evict_one(&mut self) {
        let victim = match self.config.eviction_strategy {
            EvictionStrategy::LRU => {
                self.entries.iter()
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(k, _)| *k)
            }
            EvictionStrategy::FIFO => {
                self.entries.iter()
                    .min_by_key(|(_, e)| e.insert_order)
                    .map(|(k, _)| *k)
            }
        };

        if let Some(key) = victim {
            self.entries.remove(&key);
            self.stats.evictions += 1;
        }
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

// ---------------------------------------------------------------------------
// SharedWitnessCache — thread-safe wrapper
// ---------------------------------------------------------------------------

/// Thread-safe shared witness cache.
#[derive(Clone)]
pub struct SharedWitnessCache {
    inner: Arc<Mutex<WitnessCache>>,
}

impl SharedWitnessCache {
    pub fn new(config: WitnessCacheConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WitnessCache::new(config))),
        }
    }

    pub fn get(&self, hash: &WitnessHash) -> Option<WitnessCachedProof> {
        self.inner.lock().ok()?.get(hash)
    }

    pub fn insert(&self, proof: WitnessCachedProof) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.insert(proof);
        }
    }

    pub fn mark_verified(&self, hash: &WitnessHash) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.mark_verified(hash);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.clear();
        }
    }

    pub fn stats(&self) -> WitnessCacheStats {
        self.inner.lock()
            .map(|c| c.stats())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Global shared witness cache
// ---------------------------------------------------------------------------

static GLOBAL_WITNESS_CACHE: OnceLock<SharedWitnessCache> = OnceLock::new();

/// Returns a reference to the process-wide shared witness cache (default config).
pub fn shared_witness_cache() -> SharedWitnessCache {
    GLOBAL_WITNESS_CACHE
        .get_or_init(|| SharedWitnessCache::new(WitnessCacheConfig::default()))
        .clone()
}

/// Returns a shared witness cache with custom configuration.
/// If the global cache already exists, this returns a new independent cache.
pub fn shared_witness_cache_with_config(config: WitnessCacheConfig) -> SharedWitnessCache {
    SharedWitnessCache::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hash(val: u8) -> WitnessHash {
        let mut h = [0u8; 32];
        h[0] = val;
        WitnessHash(h)
    }

    #[test]
    fn test_insert_and_get() {
        let cache = SharedWitnessCache::new(WitnessCacheConfig::default());
        let hash = test_hash(1);
        let proof = WitnessCachedProof::new(hash, vec![1, 2, 3], vec![], 100);

        cache.insert(proof);
        let retrieved = cache.get(&hash).unwrap();
        assert_eq!(retrieved.proof, vec![1, 2, 3]);
    }

    #[test]
    fn test_miss() {
        let cache = SharedWitnessCache::new(WitnessCacheConfig::default());
        assert!(cache.get(&test_hash(99)).is_none());
    }

    #[test]
    fn test_mark_verified() {
        let cache = SharedWitnessCache::new(WitnessCacheConfig::default());
        let hash = test_hash(1);
        cache.insert(WitnessCachedProof::new(hash, vec![], vec![], 0));
        assert!(!cache.get(&hash).unwrap().verified);

        cache.mark_verified(&hash);
        assert!(cache.get(&hash).unwrap().verified);
    }

    #[test]
    fn test_eviction() {
        let config = WitnessCacheConfig {
            max_entries: 2,
            eviction_strategy: EvictionStrategy::FIFO,
        };
        let cache = SharedWitnessCache::new(config);

        cache.insert(WitnessCachedProof::new(test_hash(1), vec![1], vec![], 0));
        cache.insert(WitnessCachedProof::new(test_hash(2), vec![2], vec![], 0));
        cache.insert(WitnessCachedProof::new(test_hash(3), vec![3], vec![], 0));

        assert!(cache.get(&test_hash(1)).is_none()); // evicted
        assert!(cache.get(&test_hash(3)).is_some());
    }

    #[test]
    fn test_hash_display() {
        let hash = test_hash(0xAB);
        let s = format!("{}", hash);
        assert!(s.starts_with("ab"));
        assert!(s.ends_with("..."));
    }
}
