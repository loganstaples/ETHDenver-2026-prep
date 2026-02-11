//! Circuit caching for performance optimization.
//!
//! Caches circuit structures, witnesses, and lookup tables to avoid
//! redundant computation during repeated proof generation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Configuration for the circuit cache.
#[derive(Debug, Clone)]
pub struct CircuitCacheConfig {
    /// Maximum number of cached circuit structures.
    pub max_structures: usize,
    /// Maximum number of cached witnesses.
    pub max_witnesses: usize,
    /// Eviction policy.
    pub eviction_policy: EvictionPolicy,
}

impl Default for CircuitCacheConfig {
    fn default() -> Self {
        Self {
            max_structures: 100,
            max_witnesses: 50,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Cache eviction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used.
    LRU,
    /// First In First Out.
    FIFO,
}

/// General cache configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum entries.
    pub max_entries: usize,
    /// Eviction policy.
    pub eviction_policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Statistics for cache usage.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Current number of cached entries.
    pub entries: usize,
}

/// Key for identifying cached structures.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StructureKey {
    /// Circuit identifier.
    pub circuit_id: String,
    /// Parameter hash.
    pub param_hash: u64,
}

/// A cached circuit structure.
#[derive(Debug, Clone)]
pub struct CachedStructure {
    /// Structure key.
    pub key: StructureKey,
    /// Serialized structure data.
    pub data: Vec<u8>,
}

/// Cache for circuit structures.
#[derive(Debug)]
pub struct StructureCache {
    entries: HashMap<StructureKey, CachedStructure>,
    config: CacheConfig,
    stats: CacheStats,
}

impl StructureCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: CacheStats::default(),
        }
    }

    pub fn get(&mut self, key: &StructureKey) -> Option<&CachedStructure> {
        if self.entries.contains_key(key) {
            self.stats.hits += 1;
            self.entries.get(key)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, structure: CachedStructure) {
        if self.entries.len() >= self.config.max_entries {
            // Simple eviction: remove first entry
            if let Some(key) = self.entries.keys().next().cloned() {
                self.entries.remove(&key);
            }
        }
        self.stats.entries = self.entries.len() + 1;
        self.entries.insert(structure.key.clone(), structure);
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// Cache for witness data.
#[derive(Debug)]
pub struct WitnessCache {
    entries: HashMap<String, Vec<u8>>,
    config: CacheConfig,
    stats: CacheStats,
}

impl WitnessCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: CacheStats::default(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&Vec<u8>> {
        if self.entries.contains_key(key) {
            self.stats.hits += 1;
            self.entries.get(key)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: String, data: Vec<u8>) {
        if self.entries.len() >= self.config.max_entries {
            if let Some(k) = self.entries.keys().next().cloned() {
                self.entries.remove(&k);
            }
        }
        self.stats.entries = self.entries.len() + 1;
        self.entries.insert(key, data);
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// Cache for lookup tables.
#[derive(Debug)]
pub struct TableCache {
    entries: HashMap<String, Vec<Vec<u8>>>,
    stats: CacheStats,
}

impl TableCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&Vec<Vec<u8>>> {
        if self.entries.contains_key(key) {
            self.stats.hits += 1;
            self.entries.get(key)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: String, data: Vec<Vec<u8>>) {
        self.stats.entries = self.entries.len() + 1;
        self.entries.insert(key, data);
    }
}

/// Key cache for proving/verification keys.
#[derive(Debug)]
pub struct KeyCache {
    entries: HashMap<String, Vec<u8>>,
    stats: CacheStats,
}

impl KeyCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&Vec<u8>> {
        if self.entries.contains_key(key) {
            self.stats.hits += 1;
            self.entries.get(key)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: String, data: Vec<u8>) {
        self.stats.entries = self.entries.len() + 1;
        self.entries.insert(key, data);
    }
}

/// Main circuit cache combining all sub-caches.
pub struct CircuitCache {
    pub structures: StructureCache,
    pub witnesses: WitnessCache,
    pub tables: TableCache,
    pub keys: KeyCache,
}

impl CircuitCache {
    pub fn new(config: CircuitCacheConfig) -> Self {
        Self {
            structures: StructureCache::new(CacheConfig {
                max_entries: config.max_structures,
                eviction_policy: config.eviction_policy,
            }),
            witnesses: WitnessCache::new(CacheConfig {
                max_entries: config.max_witnesses,
                eviction_policy: config.eviction_policy,
            }),
            tables: TableCache::new(),
            keys: KeyCache::new(),
        }
    }
}

/// Builder for CircuitCache.
pub struct CircuitCacheBuilder {
    config: CircuitCacheConfig,
}

impl CircuitCacheBuilder {
    pub fn new() -> Self {
        Self {
            config: CircuitCacheConfig::default(),
        }
    }

    pub fn max_structures(mut self, max: usize) -> Self {
        self.config.max_structures = max;
        self
    }

    pub fn max_witnesses(mut self, max: usize) -> Self {
        self.config.max_witnesses = max;
        self
    }

    pub fn eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.config.eviction_policy = policy;
        self
    }

    pub fn build(self) -> CircuitCache {
        CircuitCache::new(self.config)
    }
}

/// Global circuit cache singleton.
static GLOBAL_CACHE: OnceLock<Arc<Mutex<CircuitCache>>> = OnceLock::new();

/// Returns the global circuit cache.
pub fn global_cache() -> Arc<Mutex<CircuitCache>> {
    GLOBAL_CACHE
        .get_or_init(|| {
            Arc::new(Mutex::new(CircuitCache::new(CircuitCacheConfig::default())))
        })
        .clone()
}
