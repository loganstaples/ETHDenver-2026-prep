//! LRU Cache for Proving Keys.
//!
//! Provides an efficient, thread-safe LRU cache for proving and verification keys
//! with configurable memory limits and automatic eviction.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::keys::{KeyId, KeyMetadata, KeyError, KeyResult};

/// Configuration for the key cache.
#[derive(Debug, Clone)]
pub struct KeyCacheConfig {
    /// Maximum number of key pairs to cache.
    pub max_entries: usize,
    /// Maximum total memory usage (bytes).
    pub max_memory_bytes: usize,
    /// Time-to-live for cached entries (None = infinite).
    pub ttl: Option<Duration>,
    /// Whether to preload keys on startup.
    pub preload: bool,
    /// Background cleanup interval.
    pub cleanup_interval: Duration,
}

impl Default for KeyCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 32,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            ttl: Some(Duration::from_secs(3600)), // 1 hour
            preload: false,
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// A cached key entry with metadata.
#[derive(Clone)]
struct CachedKeyEntry {
    /// Proving key bytes.
    pk: Arc<Vec<u8>>,
    /// Verification key bytes.
    vk: Arc<Vec<u8>>,
    /// Key metadata.
    metadata: KeyMetadata,
    /// When this entry was created.
    created_at: Instant,
    /// When this entry was last accessed.
    last_accessed: Instant,
    /// Access count for statistics.
    access_count: u64,
    /// Size in bytes.
    size_bytes: usize,
}

impl CachedKeyEntry {
    fn new(pk: Vec<u8>, vk: Vec<u8>, metadata: KeyMetadata) -> Self {
        let size = pk.len() + vk.len() + std::mem::size_of::<KeyMetadata>();
        let now = Instant::now();
        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            metadata,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes: size,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    fn is_expired(&self, ttl: Option<Duration>) -> bool {
        match ttl {
            Some(ttl) => self.created_at.elapsed() > ttl,
            None => false,
        }
    }
}

/// Statistics about cache usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyCacheStats {
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Total number of evictions.
    pub evictions: u64,
    /// Current number of entries.
    pub entry_count: usize,
    /// Current memory usage in bytes.
    pub memory_bytes: usize,
    /// Total bytes served from cache.
    pub bytes_served: u64,
}

impl KeyCacheStats {
    /// Returns the cache hit ratio.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Thread-safe LRU cache for proving keys.
pub struct KeyCache {
    /// Configuration.
    config: KeyCacheConfig,
    /// Cached entries (key_id -> entry).
    entries: RwLock<HashMap<KeyId, CachedKeyEntry>>,
    /// LRU order (front = oldest, back = newest).
    lru_order: Mutex<VecDeque<KeyId>>,
    /// Cache statistics.
    stats: Mutex<KeyCacheStats>,
    /// Current memory usage.
    current_memory: Mutex<usize>,
}

impl KeyCache {
    /// Creates a new key cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(KeyCacheConfig::default())
    }

    /// Creates a key cache with custom configuration.
    pub fn with_config(config: KeyCacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            lru_order: Mutex::new(VecDeque::new()),
            stats: Mutex::new(KeyCacheStats::default()),
            current_memory: Mutex::new(0),
        }
    }

    /// Inserts a key pair into the cache.
    pub fn insert(&self, id: KeyId, pk: Vec<u8>, vk: Vec<u8>, metadata: KeyMetadata) -> KeyResult<()> {
        let entry = CachedKeyEntry::new(pk, vk, metadata);
        let entry_size = entry.size_bytes;

        // Check if we need to evict entries
        self.ensure_capacity(entry_size);

        // Insert the entry
        {
            let mut entries = self.entries.write().unwrap();
            let mut lru = self.lru_order.lock().unwrap();
            let mut memory = self.current_memory.lock().unwrap();

            // Remove existing entry if present
            if let Some(old) = entries.remove(&id) {
                *memory = memory.saturating_sub(old.size_bytes);
                lru.retain(|k| k != &id);
            }

            entries.insert(id.clone(), entry);
            lru.push_back(id);
            *memory += entry_size;
        }

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            let entries = self.entries.read().unwrap();
            let memory = self.current_memory.lock().unwrap();
            stats.entry_count = entries.len();
            stats.memory_bytes = *memory;
        }

        Ok(())
    }

    /// Gets the proving key for a given ID.
    pub fn get_pk(&self, id: &KeyId) -> Option<Arc<Vec<u8>>> {
        self.get_entry(id).map(|e| e.pk.clone())
    }

    /// Gets the verification key for a given ID.
    pub fn get_vk(&self, id: &KeyId) -> Option<Arc<Vec<u8>>> {
        self.get_entry(id).map(|e| e.vk.clone())
    }

    /// Gets key metadata for a given ID.
    pub fn get_metadata(&self, id: &KeyId) -> Option<KeyMetadata> {
        self.get_entry(id).map(|e| e.metadata.clone())
    }

    /// Checks if a key is cached.
    pub fn contains(&self, id: &KeyId) -> bool {
        let entries = self.entries.read().unwrap();
        entries.contains_key(id)
    }

    /// Removes a key from the cache.
    pub fn remove(&self, id: &KeyId) -> bool {
        let mut entries = self.entries.write().unwrap();
        let mut lru = self.lru_order.lock().unwrap();
        let mut memory = self.current_memory.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(entry) = entries.remove(id) {
            *memory = memory.saturating_sub(entry.size_bytes);
            lru.retain(|k| k != id);
            stats.entry_count = entries.len();
            stats.memory_bytes = *memory;
            true
        } else {
            false
        }
    }

    /// Clears all entries from the cache.
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap();
        let mut lru = self.lru_order.lock().unwrap();
        let mut memory = self.current_memory.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        entries.clear();
        lru.clear();
        *memory = 0;
        stats.entry_count = 0;
        stats.memory_bytes = 0;
    }

    /// Returns current cache statistics.
    pub fn stats(&self) -> KeyCacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
    }

    /// Returns all cached key IDs.
    pub fn keys(&self) -> Vec<KeyId> {
        self.entries.read().unwrap().keys().cloned().collect()
    }

    /// Removes expired entries based on TTL.
    pub fn cleanup_expired(&self) {
        let mut entries = self.entries.write().unwrap();
        let mut lru = self.lru_order.lock().unwrap();
        let mut memory = self.current_memory.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let expired: Vec<KeyId> = entries
            .iter()
            .filter(|(_, e)| e.is_expired(self.config.ttl))
            .map(|(k, _)| k.clone())
            .collect();

        for id in expired {
            if let Some(entry) = entries.remove(&id) {
                *memory = memory.saturating_sub(entry.size_bytes);
                lru.retain(|k| k != &id);
                stats.evictions += 1;
            }
        }

        stats.entry_count = entries.len();
        stats.memory_bytes = *memory;
    }

    /// Warms the cache by preloading keys.
    pub fn warm(&self, keys: Vec<(KeyId, Vec<u8>, Vec<u8>, KeyMetadata)>) {
        for (id, pk, vk, metadata) in keys {
            let _ = self.insert(id, pk, vk, metadata);
        }
    }

    /// Returns detailed information about cache entries.
    pub fn entry_info(&self) -> Vec<KeyCacheEntryInfo> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .map(|(id, entry)| KeyCacheEntryInfo {
                id: id.clone(),
                size_bytes: entry.size_bytes,
                access_count: entry.access_count,
                age_secs: entry.created_at.elapsed().as_secs(),
                last_access_secs: entry.last_accessed.elapsed().as_secs(),
            })
            .collect()
    }

    fn get_entry(&self, id: &KeyId) -> Option<CachedKeyEntry> {
        // First check with read lock
        {
            let entries = self.entries.read().unwrap();
            if !entries.contains_key(id) {
                let mut stats = self.stats.lock().unwrap();
                stats.misses += 1;
                return None;
            }
        }

        // Upgrade to write lock to update access time
        {
            let mut entries = self.entries.write().unwrap();
            let mut lru = self.lru_order.lock().unwrap();
            let mut stats = self.stats.lock().unwrap();

            if let Some(entry) = entries.get_mut(id) {
                // Check if expired
                if entry.is_expired(self.config.ttl) {
                    let size = entry.size_bytes;
                    entries.remove(id);
                    lru.retain(|k| k != id);
                    let mut memory = self.current_memory.lock().unwrap();
                    *memory = memory.saturating_sub(size);
                    stats.misses += 1;
                    stats.evictions += 1;
                    return None;
                }

                entry.touch();
                stats.hits += 1;
                stats.bytes_served += entry.size_bytes as u64;

                // Move to back of LRU
                lru.retain(|k| k != id);
                lru.push_back(id.clone());

                return Some(entry.clone());
            }

            stats.misses += 1;
            None
        }
    }

    fn ensure_capacity(&self, needed_bytes: usize) {
        let mut entries = self.entries.write().unwrap();
        let mut lru = self.lru_order.lock().unwrap();
        let mut memory = self.current_memory.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Evict until we have space
        while (entries.len() >= self.config.max_entries
            || *memory + needed_bytes > self.config.max_memory_bytes)
            && !lru.is_empty()
        {
            if let Some(oldest_id) = lru.pop_front() {
                if let Some(entry) = entries.remove(&oldest_id) {
                    *memory = memory.saturating_sub(entry.size_bytes);
                    stats.evictions += 1;
                }
            }
        }
    }
}

impl Default for KeyCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a cached entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyCacheEntryInfo {
    /// Key identifier.
    pub id: KeyId,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Number of times accessed.
    pub access_count: u64,
    /// Age in seconds.
    pub age_secs: u64,
    /// Seconds since last access.
    pub last_access_secs: u64,
}

/// A shared, thread-safe key cache.
pub type SharedKeyCache = Arc<KeyCache>;

/// Creates a new shared key cache.
pub fn shared_cache() -> SharedKeyCache {
    Arc::new(KeyCache::new())
}

/// Creates a shared key cache with custom configuration.
pub fn shared_cache_with_config(config: KeyCacheConfig) -> SharedKeyCache {
    Arc::new(KeyCache::with_config(config))
}

/// LRU cache with frequency-based eviction (LFU hybrid).
pub struct LFUKeyCache {
    /// Underlying cache.
    cache: KeyCache,
    /// Frequency counts for LFU.
    frequency: Mutex<HashMap<KeyId, u64>>,
}

impl LFUKeyCache {
    /// Creates a new LFU key cache.
    pub fn new() -> Self {
        Self {
            cache: KeyCache::new(),
            frequency: Mutex::new(HashMap::new()),
        }
    }

    /// Creates an LFU cache with custom configuration.
    pub fn with_config(config: KeyCacheConfig) -> Self {
        Self {
            cache: KeyCache::with_config(config),
            frequency: Mutex::new(HashMap::new()),
        }
    }

    /// Inserts a key pair into the cache.
    pub fn insert(&self, id: KeyId, pk: Vec<u8>, vk: Vec<u8>, metadata: KeyMetadata) -> KeyResult<()> {
        let mut freq = self.frequency.lock().unwrap();
        freq.insert(id.clone(), 0);
        self.cache.insert(id, pk, vk, metadata)
    }

    /// Gets the proving key with frequency tracking.
    pub fn get_pk(&self, id: &KeyId) -> Option<Arc<Vec<u8>>> {
        let result = self.cache.get_pk(id);
        if result.is_some() {
            let mut freq = self.frequency.lock().unwrap();
            *freq.entry(id.clone()).or_insert(0) += 1;
        }
        result
    }

    /// Gets the verification key with frequency tracking.
    pub fn get_vk(&self, id: &KeyId) -> Option<Arc<Vec<u8>>> {
        let result = self.cache.get_vk(id);
        if result.is_some() {
            let mut freq = self.frequency.lock().unwrap();
            *freq.entry(id.clone()).or_insert(0) += 1;
        }
        result
    }

    /// Returns the frequency count for a key.
    pub fn frequency(&self, id: &KeyId) -> u64 {
        self.frequency.lock().unwrap().get(id).copied().unwrap_or(0)
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> KeyCacheStats {
        self.cache.stats()
    }
}

impl Default for LFUKeyCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Disk-backed key cache with LRU memory cache.
pub struct DiskBackedKeyCache {
    /// Memory cache for hot keys.
    memory_cache: KeyCache,
    /// Base directory for disk storage.
    base_dir: std::path::PathBuf,
    /// Index of keys on disk.
    disk_index: RwLock<HashMap<KeyId, std::path::PathBuf>>,
}

impl DiskBackedKeyCache {
    /// Creates a new disk-backed cache.
    pub fn new(base_dir: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;

        let mut disk_index = HashMap::new();

        // Scan existing keys on disk
        if let Ok(entries) = std::fs::read_dir(&base_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        let id = KeyId(name.to_string());
                        disk_index.insert(id, entry.path());
                    }
                }
            }
        }

        Ok(Self {
            memory_cache: KeyCache::with_config(KeyCacheConfig {
                max_entries: 8,
                max_memory_bytes: 256 * 1024 * 1024, // 256MB memory cache
                ..Default::default()
            }),
            base_dir,
            disk_index: RwLock::new(disk_index),
        })
    }

    /// Stores a key pair to disk and memory cache.
    pub fn store(&self, id: KeyId, pk: Vec<u8>, vk: Vec<u8>, metadata: KeyMetadata) -> std::io::Result<()> {
        let key_dir = self.base_dir.join(&id.0);
        std::fs::create_dir_all(&key_dir)?;

        // Write to disk
        std::fs::write(key_dir.join("pk.bin"), &pk)?;
        std::fs::write(key_dir.join("vk.bin"), &vk)?;
        std::fs::write(
            key_dir.join("metadata.json"),
            serde_json::to_string_pretty(&metadata).unwrap_or_default(),
        )?;

        // Update disk index
        {
            let mut index = self.disk_index.write().unwrap();
            index.insert(id.clone(), key_dir);
        }

        // Add to memory cache
        let _ = self.memory_cache.insert(id, pk, vk, metadata);

        Ok(())
    }

    /// Loads a proving key, checking memory first then disk.
    pub fn load_pk(&self, id: &KeyId) -> Option<Vec<u8>> {
        // Check memory cache first
        if let Some(pk) = self.memory_cache.get_pk(id) {
            return Some((*pk).clone());
        }

        // Load from disk
        self.load_from_disk(id).map(|(pk, _, _)| pk)
    }

    /// Loads a verification key, checking memory first then disk.
    pub fn load_vk(&self, id: &KeyId) -> Option<Vec<u8>> {
        // Check memory cache first
        if let Some(vk) = self.memory_cache.get_vk(id) {
            return Some((*vk).clone());
        }

        // Load from disk
        self.load_from_disk(id).map(|(_, vk, _)| vk)
    }

    /// Checks if a key exists (in memory or on disk).
    pub fn contains(&self, id: &KeyId) -> bool {
        self.memory_cache.contains(id) || self.disk_index.read().unwrap().contains_key(id)
    }

    /// Lists all available key IDs.
    pub fn list(&self) -> Vec<KeyId> {
        self.disk_index.read().unwrap().keys().cloned().collect()
    }

    /// Removes a key from memory and disk.
    pub fn remove(&self, id: &KeyId) -> std::io::Result<()> {
        self.memory_cache.remove(id);

        let mut index = self.disk_index.write().unwrap();
        if let Some(path) = index.remove(id) {
            std::fs::remove_dir_all(path)?;
        }

        Ok(())
    }

    fn load_from_disk(&self, id: &KeyId) -> Option<(Vec<u8>, Vec<u8>, KeyMetadata)> {
        let index = self.disk_index.read().unwrap();
        let key_dir = index.get(id)?;

        let pk = std::fs::read(key_dir.join("pk.bin")).ok()?;
        let vk = std::fs::read(key_dir.join("vk.bin")).ok()?;
        let metadata: KeyMetadata = serde_json::from_str(
            &std::fs::read_to_string(key_dir.join("metadata.json")).ok()?,
        ).ok()?;

        // Promote to memory cache
        let _ = self.memory_cache.insert(id.clone(), pk.clone(), vk.clone(), metadata.clone());

        Some((pk, vk, metadata))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_metadata(id: &KeyId) -> KeyMetadata {
        KeyMetadata {
            id: id.clone(),
            circuit_name: "test".to_string(),
            version: 1,
            created_at: 0,
            k: 10,
            pk_hash: [0; 32],
            vk_hash: [0; 32],
            pk_size: 100,
            vk_size: 50,
        }
    }

    #[test]
    fn test_basic_operations() {
        let cache = KeyCache::new();
        let id = KeyId::new("test", 1);
        let pk = vec![1, 2, 3, 4];
        let vk = vec![5, 6, 7, 8];
        let metadata = make_test_metadata(&id);

        cache.insert(id.clone(), pk.clone(), vk.clone(), metadata).unwrap();

        assert!(cache.contains(&id));
        assert_eq!(*cache.get_pk(&id).unwrap(), pk);
        assert_eq!(*cache.get_vk(&id).unwrap(), vk);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = KeyCache::with_config(KeyCacheConfig {
            max_entries: 2,
            ..Default::default()
        });

        for i in 0..3 {
            let id = KeyId::new("test", i);
            let metadata = make_test_metadata(&id);
            cache.insert(id, vec![i as u8], vec![i as u8], metadata).unwrap();
        }

        // First entry should have been evicted
        assert!(!cache.contains(&KeyId::new("test", 0)));
        assert!(cache.contains(&KeyId::new("test", 1)));
        assert!(cache.contains(&KeyId::new("test", 2)));
    }

    #[test]
    fn test_memory_limit() {
        let cache = KeyCache::with_config(KeyCacheConfig {
            max_entries: 100,
            max_memory_bytes: 500,
            ..Default::default()
        });

        for i in 0..10 {
            let id = KeyId::new("test", i);
            let metadata = make_test_metadata(&id);
            let pk = vec![0u8; 100];
            let vk = vec![0u8; 50];
            cache.insert(id, pk, vk, metadata).unwrap();
        }

        // Should have evicted entries to stay under memory limit
        assert!(cache.len() < 10);
    }

    #[test]
    fn test_stats() {
        let cache = KeyCache::new();
        let id = KeyId::new("test", 1);
        let metadata = make_test_metadata(&id);

        cache.insert(id.clone(), vec![1], vec![2], metadata).unwrap();

        // Cache hit
        cache.get_pk(&id);
        cache.get_pk(&id);

        // Cache miss
        cache.get_pk(&KeyId::new("nonexistent", 1));

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let cache = Arc::new(KeyCache::new());
        let mut handles = vec![];

        for i in 0..10 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                let id = KeyId::new("test", i);
                let metadata = KeyMetadata {
                    id: id.clone(),
                    circuit_name: "test".to_string(),
                    version: i,
                    created_at: 0,
                    k: 10,
                    pk_hash: [0; 32],
                    vk_hash: [0; 32],
                    pk_size: 100,
                    vk_size: 50,
                };
                cache.insert(id, vec![i as u8], vec![i as u8], metadata).unwrap();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_lfu_cache() {
        let cache = LFUKeyCache::new();
        let id = KeyId::new("test", 1);
        let metadata = make_test_metadata(&id);

        cache.insert(id.clone(), vec![1], vec![2], metadata).unwrap();

        // Access multiple times
        for _ in 0..5 {
            cache.get_pk(&id);
        }

        assert_eq!(cache.frequency(&id), 5);
    }
}
