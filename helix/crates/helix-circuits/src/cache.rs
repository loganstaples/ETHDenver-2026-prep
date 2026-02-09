//! Circuit parameter caching.
//!
//! Provides caching for SRS parameters, proving/verification keys, circuit
//! structures, witnesses, and lookup tables to avoid expensive regeneration
//! on repeated circuit operations.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Eviction policy
// ---------------------------------------------------------------------------

/// Cache eviction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least-recently-used.
    LRU,
    /// First-in-first-out.
    FIFO,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

// ---------------------------------------------------------------------------
// Generic cache configuration
// ---------------------------------------------------------------------------

/// Generic cache configuration shared across all cache types.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries before eviction kicks in.
    pub max_entries: usize,
    /// Eviction policy to apply when at capacity.
    pub eviction_policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 16,
            eviction_policy: EvictionPolicy::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Runtime cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted.
    pub evictions: u64,
    /// Current number of entries.
    pub size: usize,
}

impl CacheStats {
    /// Hit ratio (0.0 – 1.0). Returns 0.0 if no lookups have occurred.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// CacheKey (generic circuit parameter key)
// ---------------------------------------------------------------------------

/// Cache key identifying a specific circuit configuration.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct CacheKey {
    /// Circuit type identifier.
    pub circuit_type: String,
    /// Parameter hash (e.g. model dimensions).
    pub param_hash: [u8; 32],
}

impl CacheKey {
    pub fn new(circuit_type: impl Into<String>, param_hash: [u8; 32]) -> Self {
        Self {
            circuit_type: circuit_type.into(),
            param_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// CachedParams
// ---------------------------------------------------------------------------

/// Cached circuit parameters (proving key, verification key, etc.).
#[derive(Clone)]
pub struct CachedParams {
    /// Serialized proving key bytes.
    pub pk_bytes: Vec<u8>,
    /// Serialized verification key bytes.
    pub vk_bytes: Vec<u8>,
    /// Timestamp of last use (for LRU eviction).
    pub last_used: u64,
}

// ---------------------------------------------------------------------------
// CircuitCacheConfig / CircuitCacheBuilder / CircuitCache
// ---------------------------------------------------------------------------

/// Configuration specific to [`CircuitCache`].
#[derive(Debug, Clone)]
pub struct CircuitCacheConfig {
    /// Base cache settings.
    pub base: CacheConfig,
}

impl Default for CircuitCacheConfig {
    fn default() -> Self {
        Self {
            base: CacheConfig::default(),
        }
    }
}

/// Builder for [`CircuitCache`].
pub struct CircuitCacheBuilder {
    config: CircuitCacheConfig,
}

impl CircuitCacheBuilder {
    pub fn new() -> Self {
        Self {
            config: CircuitCacheConfig::default(),
        }
    }

    pub fn max_entries(mut self, n: usize) -> Self {
        self.config.base.max_entries = n;
        self
    }

    pub fn eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.config.base.eviction_policy = policy;
        self
    }

    pub fn build(self) -> CircuitCache {
        CircuitCache {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: self.config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
}

impl Default for CircuitCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe circuit parameter cache.
pub struct CircuitCache {
    entries: Arc<RwLock<HashMap<CacheKey, CachedParams>>>,
    config: CircuitCacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl CircuitCache {
    /// Creates a new circuit cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        CircuitCacheBuilder::new()
            .max_entries(max_entries)
            .build()
    }

    /// Returns a builder for more detailed configuration.
    pub fn builder() -> CircuitCacheBuilder {
        CircuitCacheBuilder::new()
    }

    /// Gets cached parameters for a circuit configuration.
    pub fn get(&self, key: &CacheKey) -> Option<CachedParams> {
        let mut cache = self.entries.write();
        if let Some(entry) = cache.get_mut(key) {
            entry.last_used = now_secs();
            self.stats.write().hits += 1;
            Some(entry.clone())
        } else {
            self.stats.write().misses += 1;
            None
        }
    }

    /// Stores parameters in the cache, evicting the oldest entry if at capacity.
    pub fn put(&self, key: CacheKey, params: CachedParams) {
        let mut cache = self.entries.write();

        if cache.len() >= self.config.base.max_entries {
            let oldest: Option<CacheKey> = cache
                .iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone());
            if let Some(old_key) = oldest {
                cache.remove(&old_key);
                self.stats.write().evictions += 1;
            }
        }

        cache.insert(key, params);
        self.stats.write().size = cache.len();
    }

    /// Removes a cache entry.
    pub fn remove(&self, key: &CacheKey) -> bool {
        let removed = self.entries.write().remove(key).is_some();
        if removed {
            self.stats.write().size = self.entries.read().len();
        }
        removed
    }

    /// Clears all cached entries.
    pub fn clear(&self) {
        self.entries.write().clear();
        self.stats.write().size = 0;
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Returns a snapshot of the current cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }
}

impl Default for CircuitCache {
    fn default() -> Self {
        Self::new(16)
    }
}

// ---------------------------------------------------------------------------
// StructureCache — caches compiled circuit structures
// ---------------------------------------------------------------------------

/// Key for looking up a cached circuit structure.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct StructureKey {
    /// Identifier for the circuit variant.
    pub circuit_id: String,
    /// Hash of the circuit parameters that affect structure.
    pub config_hash: [u8; 32],
}

impl StructureKey {
    pub fn new(circuit_id: impl Into<String>, config_hash: [u8; 32]) -> Self {
        Self {
            circuit_id: circuit_id.into(),
            config_hash,
        }
    }
}

/// A cached, pre-compiled circuit structure (column assignments, gate layout).
#[derive(Clone)]
pub struct CachedStructure {
    /// Serialised structure bytes.
    pub data: Vec<u8>,
    /// Number of advice columns.
    pub num_advice: usize,
    /// Degree (k) used for this structure.
    pub k: u32,
    /// Timestamp of last use.
    pub last_used: u64,
}

/// Cache for pre-compiled circuit structures.
pub struct StructureCache {
    entries: Arc<RwLock<HashMap<StructureKey, CachedStructure>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl StructureCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: CacheConfig {
                max_entries,
                ..Default::default()
            },
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    pub fn get(&self, key: &StructureKey) -> Option<CachedStructure> {
        let mut cache = self.entries.write();
        if let Some(entry) = cache.get_mut(key) {
            entry.last_used = now_secs();
            self.stats.write().hits += 1;
            Some(entry.clone())
        } else {
            self.stats.write().misses += 1;
            None
        }
    }

    pub fn put(&self, key: StructureKey, structure: CachedStructure) {
        let mut cache = self.entries.write();
        if cache.len() >= self.config.max_entries {
            let oldest: Option<StructureKey> = cache
                .iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone());
            if let Some(old_key) = oldest {
                cache.remove(&old_key);
                self.stats.write().evictions += 1;
            }
        }
        cache.insert(key, structure);
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

impl Default for StructureCache {
    fn default() -> Self {
        Self::new(32)
    }
}

// ---------------------------------------------------------------------------
// WitnessCache — caches witness generation intermediates
// ---------------------------------------------------------------------------

/// Cache for witness generation intermediates (e.g. evaluated polynomials).
pub struct WitnessCache {
    entries: Arc<RwLock<HashMap<[u8; 32], Vec<u8>>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl WitnessCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: CacheConfig {
                max_entries,
                ..Default::default()
            },
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    pub fn get(&self, hash: &[u8; 32]) -> Option<Vec<u8>> {
        let cache = self.entries.read();
        if let Some(data) = cache.get(hash) {
            self.stats.write().hits += 1;
            Some(data.clone())
        } else {
            self.stats.write().misses += 1;
            None
        }
    }

    pub fn put(&self, hash: [u8; 32], data: Vec<u8>) {
        let mut cache = self.entries.write();
        if cache.len() >= self.config.max_entries && !cache.contains_key(&hash) {
            // FIFO eviction for witness cache — order doesn't matter much
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
                self.stats.write().evictions += 1;
            }
        }
        cache.insert(hash, data);
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

impl Default for WitnessCache {
    fn default() -> Self {
        Self::new(64)
    }
}

// ---------------------------------------------------------------------------
// TableCache — caches lookup tables
// ---------------------------------------------------------------------------

/// Cache for precomputed lookup tables (ReLU, GELU, sigmoid, etc.).
pub struct TableCache {
    entries: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl TableCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: CacheConfig {
                max_entries,
                ..Default::default()
            },
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    pub fn get(&self, name: &str) -> Option<Vec<u8>> {
        let cache = self.entries.read();
        if let Some(data) = cache.get(name) {
            self.stats.write().hits += 1;
            Some(data.clone())
        } else {
            self.stats.write().misses += 1;
            None
        }
    }

    pub fn put(&self, name: impl Into<String>, data: Vec<u8>) {
        let mut cache = self.entries.write();
        if cache.len() >= self.config.max_entries {
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
                self.stats.write().evictions += 1;
            }
        }
        cache.insert(name.into(), data);
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

impl Default for TableCache {
    fn default() -> Self {
        Self::new(32)
    }
}

// ---------------------------------------------------------------------------
// KeyCache — caches proving/verification key pairs
// ---------------------------------------------------------------------------

/// Cache for serialised proving/verification keys.
pub struct KeyCache {
    entries: Arc<RwLock<HashMap<CacheKey, CachedParams>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl KeyCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: CacheConfig {
                max_entries,
                ..Default::default()
            },
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    pub fn get(&self, key: &CacheKey) -> Option<CachedParams> {
        let mut cache = self.entries.write();
        if let Some(entry) = cache.get_mut(key) {
            entry.last_used = now_secs();
            self.stats.write().hits += 1;
            Some(entry.clone())
        } else {
            self.stats.write().misses += 1;
            None
        }
    }

    pub fn put(&self, key: CacheKey, params: CachedParams) {
        let mut cache = self.entries.write();
        if cache.len() >= self.config.max_entries {
            let oldest: Option<CacheKey> = cache
                .iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone());
            if let Some(old_key) = oldest {
                cache.remove(&old_key);
                self.stats.write().evictions += 1;
            }
        }
        cache.insert(key, params);
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

impl Default for KeyCache {
    fn default() -> Self {
        Self::new(8)
    }
}

// ---------------------------------------------------------------------------
// Global cache singleton
// ---------------------------------------------------------------------------

static GLOBAL_CACHE: std::sync::OnceLock<CircuitCache> = std::sync::OnceLock::new();

/// Returns a reference to the process-wide global circuit cache.
pub fn global_cache() -> &'static CircuitCache {
    GLOBAL_CACHE.get_or_init(CircuitCache::default)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_put_get() {
        let cache = CircuitCache::new(10);
        let key = CacheKey::new("ml_training", [0u8; 32]);
        let params = CachedParams {
            pk_bytes: vec![1, 2, 3],
            vk_bytes: vec![4, 5, 6],
            last_used: 0,
        };

        cache.put(key.clone(), params);

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.pk_bytes, vec![1, 2, 3]);
        assert_eq!(retrieved.vk_bytes, vec![4, 5, 6]);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = CircuitCache::new(2);

        for i in 0..3u8 {
            let key = CacheKey::new("test", [i; 32]);
            let params = CachedParams {
                pk_bytes: vec![i],
                vk_bytes: vec![],
                last_used: i as u64,
            };
            cache.put(key, params);
        }

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cache_stats() {
        let cache = CircuitCache::new(10);
        let key = CacheKey::new("test", [0u8; 32]);

        // Miss
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Put + hit
        cache.put(
            key.clone(),
            CachedParams {
                pk_bytes: vec![],
                vk_bytes: vec![],
                last_used: 0,
            },
        );
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_builder() {
        let cache = CircuitCache::builder()
            .max_entries(4)
            .eviction_policy(EvictionPolicy::FIFO)
            .build();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_structure_cache() {
        let cache = StructureCache::new(4);
        let key = StructureKey::new("matmul", [1u8; 32]);
        let structure = CachedStructure {
            data: vec![10, 20, 30],
            num_advice: 4,
            k: 12,
            last_used: 0,
        };

        cache.put(key.clone(), structure);
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.data, vec![10, 20, 30]);
        assert_eq!(retrieved.k, 12);
    }

    #[test]
    fn test_witness_cache() {
        let cache = WitnessCache::new(4);
        let hash = [42u8; 32];
        cache.put(hash, vec![1, 2, 3]);
        assert_eq!(cache.get(&hash).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_table_cache() {
        let cache = TableCache::new(4);
        cache.put("relu_int8", vec![0; 256]);
        assert!(cache.get("relu_int8").is_some());
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_key_cache() {
        let cache = KeyCache::new(4);
        let key = CacheKey::new("training_v2", [5u8; 32]);
        let params = CachedParams {
            pk_bytes: vec![1],
            vk_bytes: vec![2],
            last_used: 0,
        };
        cache.put(key.clone(), params);
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.pk_bytes, vec![1]);
    }

    #[test]
    fn test_global_cache() {
        let gc = global_cache();
        assert!(gc.is_empty() || gc.len() > 0); // just ensure it initializes
    }
}
