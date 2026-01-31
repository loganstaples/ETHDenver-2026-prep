//! Incremental Proof Cache.
//!
//! Provides caching for intermediate proofs to enable resumable proving sessions,
//! efficient re-proving on failure, and proof deduplication.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::chunking::ChunkId;
use crate::parallel::ChunkProof;
use crate::aggregation::AggregatedProof;
use crate::ivc::IVCState;

/// Configuration for the proof cache.
#[derive(Debug, Clone)]
pub struct ProofCacheConfig {
    /// Maximum number of cached proofs.
    pub max_proofs: usize,
    /// Maximum total cache size in bytes.
    pub max_size_bytes: usize,
    /// Enable disk persistence.
    pub persist_to_disk: bool,
    /// Base directory for disk persistence.
    pub cache_dir: Option<PathBuf>,
    /// Proof TTL (None = infinite).
    pub ttl: Option<Duration>,
    /// Enable proof deduplication.
    pub deduplicate: bool,
    /// Checkpoint interval (proofs between checkpoints).
    pub checkpoint_interval: usize,
}

impl Default for ProofCacheConfig {
    fn default() -> Self {
        Self {
            max_proofs: 10000,
            max_size_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            persist_to_disk: false,
            cache_dir: None,
            ttl: Some(Duration::from_secs(3600 * 24)), // 24 hours
            deduplicate: true,
            checkpoint_interval: 100,
        }
    }
}

/// A cached proof entry.
#[derive(Clone, Serialize, Deserialize)]
pub struct CachedProof {
    /// The chunk ID this proof belongs to.
    pub chunk_id: ChunkId,
    /// Serialized proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs hash for deduplication.
    pub input_hash: [u8; 32],
    /// Proof size in bytes.
    pub size_bytes: usize,
    /// When this proof was generated.
    pub created_at: u64,
    /// Error bound of the proof.
    pub error_bound: f64,
    /// Generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Whether this proof has been verified.
    pub verified: bool,
}

impl CachedProof {
    /// Creates a new cached proof from a ChunkProof.
    pub fn from_chunk_proof(proof: &ChunkProof) -> Self {
        let input_hash = Self::compute_input_hash(&proof.public_inputs);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            chunk_id: proof.chunk_id,
            proof: proof.proof.clone(),
            input_hash,
            size_bytes: proof.proof.len(),
            created_at: now,
            error_bound: proof.error_bound,
            generation_time_ms: proof.generation_time_ms,
            verified: false,
        }
    }

    /// Computes a hash of the public inputs for deduplication.
    fn compute_input_hash(inputs: &[[u8; 32]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for input in inputs {
            hasher.update(input);
        }
        hasher.finalize().into()
    }
}

/// Statistics about cache usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of proofs stored.
    pub proofs_stored: u64,
    /// Number of proofs evicted.
    pub proofs_evicted: u64,
    /// Total bytes stored.
    pub total_bytes: usize,
    /// Number of duplicates avoided.
    pub duplicates_avoided: u64,
    /// Number of checkpoints saved.
    pub checkpoints_saved: u64,
    /// Number of successful resumes.
    pub resumes: u64,
}

impl ProofCacheStats {
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

/// Thread-safe proof cache with incremental caching support.
pub struct ProofCache {
    /// Configuration.
    config: ProofCacheConfig,
    /// Cached proofs by chunk ID.
    proofs: RwLock<HashMap<ChunkId, CachedProof>>,
    /// Input hash to chunk ID map for deduplication.
    dedup_index: RwLock<HashMap<[u8; 32], ChunkId>>,
    /// Cache statistics.
    stats: Mutex<ProofCacheStats>,
    /// Current total size.
    current_size: Mutex<usize>,
    /// Proofs since last checkpoint.
    proofs_since_checkpoint: Mutex<usize>,
}

impl ProofCache {
    /// Creates a new proof cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(ProofCacheConfig::default())
    }

    /// Creates a proof cache with custom configuration.
    pub fn with_config(config: ProofCacheConfig) -> Self {
        // Create cache directory if persistence enabled
        if config.persist_to_disk {
            if let Some(ref dir) = config.cache_dir {
                let _ = fs::create_dir_all(dir);
            }
        }

        Self {
            config,
            proofs: RwLock::new(HashMap::new()),
            dedup_index: RwLock::new(HashMap::new()),
            stats: Mutex::new(ProofCacheStats::default()),
            current_size: Mutex::new(0),
            proofs_since_checkpoint: Mutex::new(0),
        }
    }

    /// Inserts a chunk proof into the cache.
    pub fn insert(&self, proof: &ChunkProof) -> bool {
        let cached = CachedProof::from_chunk_proof(proof);

        // Check for duplicate
        if self.config.deduplicate {
            let dedup = self.dedup_index.read().unwrap();
            if dedup.contains_key(&cached.input_hash) {
                let mut stats = self.stats.lock().unwrap();
                stats.duplicates_avoided += 1;
                return false;
            }
        }

        self.ensure_capacity(cached.size_bytes);

        let chunk_id = cached.chunk_id;
        let input_hash = cached.input_hash;
        let size = cached.size_bytes;

        {
            let mut proofs = self.proofs.write().unwrap();
            let mut dedup = self.dedup_index.write().unwrap();
            let mut current_size = self.current_size.lock().unwrap();
            let mut stats = self.stats.lock().unwrap();

            // Remove existing if present
            if let Some(old) = proofs.remove(&chunk_id) {
                *current_size = current_size.saturating_sub(old.size_bytes);
                dedup.remove(&old.input_hash);
            }

            proofs.insert(chunk_id, cached);
            dedup.insert(input_hash, chunk_id);
            *current_size += size;
            stats.proofs_stored += 1;
            stats.total_bytes = *current_size;
        }

        // Check if we need to checkpoint
        {
            let mut checkpoint_count = self.proofs_since_checkpoint.lock().unwrap();
            *checkpoint_count += 1;
            if *checkpoint_count >= self.config.checkpoint_interval {
                *checkpoint_count = 0;
                self.maybe_checkpoint();
            }
        }

        true
    }

    /// Gets a cached proof by chunk ID.
    pub fn get(&self, chunk_id: ChunkId) -> Option<CachedProof> {
        let proofs = self.proofs.read().unwrap();
        let result = proofs.get(&chunk_id).cloned();

        let mut stats = self.stats.lock().unwrap();
        if result.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }

        result
    }

    /// Gets a proof by input hash (for deduplication).
    pub fn get_by_input_hash(&self, input_hash: &[u8; 32]) -> Option<CachedProof> {
        let dedup = self.dedup_index.read().unwrap();
        let chunk_id = dedup.get(input_hash)?;

        let proofs = self.proofs.read().unwrap();
        proofs.get(chunk_id).cloned()
    }

    /// Checks if a proof is cached.
    pub fn contains(&self, chunk_id: ChunkId) -> bool {
        self.proofs.read().unwrap().contains_key(&chunk_id)
    }

    /// Marks a proof as verified.
    pub fn mark_verified(&self, chunk_id: ChunkId) -> bool {
        let mut proofs = self.proofs.write().unwrap();
        if let Some(proof) = proofs.get_mut(&chunk_id) {
            proof.verified = true;
            true
        } else {
            false
        }
    }

    /// Removes a proof from the cache.
    pub fn remove(&self, chunk_id: ChunkId) -> Option<CachedProof> {
        let mut proofs = self.proofs.write().unwrap();
        let mut dedup = self.dedup_index.write().unwrap();
        let mut current_size = self.current_size.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(proof) = proofs.remove(&chunk_id) {
            dedup.remove(&proof.input_hash);
            *current_size = current_size.saturating_sub(proof.size_bytes);
            stats.total_bytes = *current_size;
            stats.proofs_evicted += 1;
            Some(proof)
        } else {
            None
        }
    }

    /// Returns all cached chunk IDs.
    pub fn chunk_ids(&self) -> Vec<ChunkId> {
        self.proofs.read().unwrap().keys().copied().collect()
    }

    /// Returns the number of cached proofs.
    pub fn len(&self) -> usize {
        self.proofs.read().unwrap().len()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.proofs.read().unwrap().is_empty()
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> ProofCacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clears all cached proofs.
    pub fn clear(&self) {
        let mut proofs = self.proofs.write().unwrap();
        let mut dedup = self.dedup_index.write().unwrap();
        let mut current_size = self.current_size.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.proofs_evicted += proofs.len() as u64;
        proofs.clear();
        dedup.clear();
        *current_size = 0;
        stats.total_bytes = 0;
    }

    /// Saves a checkpoint to disk.
    pub fn save_checkpoint(&self, name: &str) -> std::io::Result<PathBuf> {
        let cache_dir = self.config.cache_dir.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "No cache directory configured"))?;

        let checkpoint_path = cache_dir.join(format!("checkpoint_{}.bin", name));
        let proofs = self.proofs.read().unwrap();
        let checkpoint = ProofCheckpoint {
            proofs: proofs.values().cloned().collect(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let data = serde_json::to_vec(&checkpoint)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        fs::write(&checkpoint_path, data)?;

        let mut stats = self.stats.lock().unwrap();
        stats.checkpoints_saved += 1;

        Ok(checkpoint_path)
    }

    /// Loads a checkpoint from disk.
    pub fn load_checkpoint(&self, path: impl AsRef<Path>) -> std::io::Result<usize> {
        let data = fs::read(path)?;
        let checkpoint: ProofCheckpoint = serde_json::from_slice(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut proofs = self.proofs.write().unwrap();
        let mut dedup = self.dedup_index.write().unwrap();
        let mut current_size = self.current_size.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let mut loaded = 0;
        for cached in checkpoint.proofs {
            let chunk_id = cached.chunk_id;
            let input_hash = cached.input_hash;
            let size = cached.size_bytes;

            if !proofs.contains_key(&chunk_id) {
                proofs.insert(chunk_id, cached);
                dedup.insert(input_hash, chunk_id);
                *current_size += size;
                loaded += 1;
            }
        }

        stats.total_bytes = *current_size;
        stats.resumes += 1;

        Ok(loaded)
    }

    /// Lists available checkpoints.
    pub fn list_checkpoints(&self) -> Vec<PathBuf> {
        let cache_dir = match &self.config.cache_dir {
            Some(dir) => dir,
            None => return Vec::new(),
        };

        fs::read_dir(cache_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.starts_with("checkpoint_") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect()
    }

    fn ensure_capacity(&self, needed_bytes: usize) {
        let mut proofs = self.proofs.write().unwrap();
        let mut dedup = self.dedup_index.write().unwrap();
        let mut current_size = self.current_size.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Simple eviction: remove oldest proofs
        while (proofs.len() >= self.config.max_proofs
            || *current_size + needed_bytes > self.config.max_size_bytes)
            && !proofs.is_empty()
        {
            // Find oldest proof
            let oldest_id = proofs
                .iter()
                .min_by_key(|(_, p)| p.created_at)
                .map(|(id, _)| *id);

            if let Some(id) = oldest_id {
                if let Some(proof) = proofs.remove(&id) {
                    dedup.remove(&proof.input_hash);
                    *current_size = current_size.saturating_sub(proof.size_bytes);
                    stats.proofs_evicted += 1;
                }
            } else {
                break;
            }
        }
    }

    fn maybe_checkpoint(&self) {
        if !self.config.persist_to_disk {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let _ = self.save_checkpoint(&format!("{}", timestamp));
    }
}

impl Default for ProofCache {
    fn default() -> Self {
        Self::new()
    }
}

/// A checkpoint containing cached proofs.
#[derive(Serialize, Deserialize)]
struct ProofCheckpoint {
    proofs: Vec<CachedProof>,
    timestamp: u64,
}

/// Shared proof cache type.
pub type SharedProofCache = Arc<ProofCache>;

/// Creates a new shared proof cache.
pub fn shared_cache() -> SharedProofCache {
    Arc::new(ProofCache::new())
}

/// Creates a shared proof cache with custom configuration.
pub fn shared_cache_with_config(config: ProofCacheConfig) -> SharedProofCache {
    Arc::new(ProofCache::with_config(config))
}

/// Session-based proof cache for resumable proving.
pub struct ProvingSession {
    /// Session ID.
    pub id: String,
    /// Associated proof cache.
    cache: SharedProofCache,
    /// Session state.
    state: Mutex<SessionState>,
    /// Completed chunk IDs.
    completed: RwLock<HashSet<ChunkId>>,
    /// Failed chunk IDs with error messages.
    failed: RwLock<HashMap<ChunkId, String>>,
}

/// State of a proving session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Session ID.
    pub id: String,
    /// When the session was created.
    pub created_at: u64,
    /// When the session was last updated.
    pub updated_at: u64,
    /// Total chunks to prove.
    pub total_chunks: usize,
    /// Completed chunks.
    pub completed_chunks: usize,
    /// Failed chunks.
    pub failed_chunks: usize,
    /// Whether the session is complete.
    pub is_complete: bool,
    /// Current IVC state (if using IVC).
    pub ivc_state: Option<IVCState>,
}

impl ProvingSession {
    /// Creates a new proving session.
    pub fn new(total_chunks: usize) -> Self {
        let id = Self::generate_id();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: id.clone(),
            cache: shared_cache(),
            state: Mutex::new(SessionState {
                id,
                created_at: now,
                updated_at: now,
                total_chunks,
                completed_chunks: 0,
                failed_chunks: 0,
                is_complete: false,
                ivc_state: None,
            }),
            completed: RwLock::new(HashSet::new()),
            failed: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a session with a specific cache.
    pub fn with_cache(cache: SharedProofCache, total_chunks: usize) -> Self {
        let id = Self::generate_id();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: id.clone(),
            cache,
            state: Mutex::new(SessionState {
                id,
                created_at: now,
                updated_at: now,
                total_chunks,
                completed_chunks: 0,
                failed_chunks: 0,
                is_complete: false,
                ivc_state: None,
            }),
            completed: RwLock::new(HashSet::new()),
            failed: RwLock::new(HashMap::new()),
        }
    }

    /// Records a successful proof.
    pub fn record_success(&self, proof: &ChunkProof) {
        self.cache.insert(proof);

        let mut completed = self.completed.write().unwrap();
        completed.insert(proof.chunk_id);

        let mut state = self.state.lock().unwrap();
        state.completed_chunks = completed.len();
        state.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        state.is_complete = state.completed_chunks + state.failed_chunks >= state.total_chunks;
    }

    /// Records a failed proof attempt.
    pub fn record_failure(&self, chunk_id: ChunkId, error: String) {
        let mut failed = self.failed.write().unwrap();
        failed.insert(chunk_id, error);

        let mut state = self.state.lock().unwrap();
        state.failed_chunks = failed.len();
        state.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        state.is_complete = state.completed_chunks + state.failed_chunks >= state.total_chunks;
    }

    /// Checks if a chunk has been completed.
    pub fn is_completed(&self, chunk_id: ChunkId) -> bool {
        self.completed.read().unwrap().contains(&chunk_id)
    }

    /// Checks if a chunk has failed.
    pub fn is_failed(&self, chunk_id: ChunkId) -> bool {
        self.failed.read().unwrap().contains_key(&chunk_id)
    }

    /// Gets the proof for a completed chunk.
    pub fn get_proof(&self, chunk_id: ChunkId) -> Option<CachedProof> {
        if self.is_completed(chunk_id) {
            self.cache.get(chunk_id)
        } else {
            None
        }
    }

    /// Gets all completed proofs.
    pub fn get_completed_proofs(&self) -> Vec<CachedProof> {
        let completed = self.completed.read().unwrap();
        completed
            .iter()
            .filter_map(|id| self.cache.get(*id))
            .collect()
    }

    /// Gets pending (not completed, not failed) chunk IDs.
    pub fn get_pending(&self, all_chunks: &[ChunkId]) -> Vec<ChunkId> {
        let completed = self.completed.read().unwrap();
        let failed = self.failed.read().unwrap();

        all_chunks
            .iter()
            .filter(|id| !completed.contains(id) && !failed.contains_key(id))
            .copied()
            .collect()
    }

    /// Returns the current session state.
    pub fn state(&self) -> SessionState {
        self.state.lock().unwrap().clone()
    }

    /// Updates the IVC state.
    pub fn update_ivc_state(&self, ivc: IVCState) {
        let mut state = self.state.lock().unwrap();
        state.ivc_state = Some(ivc);
    }

    /// Saves the session to disk.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let state = self.state.lock().unwrap();
        let completed = self.completed.read().unwrap();
        let failed = self.failed.read().unwrap();

        let session_data = SavedSession {
            state: state.clone(),
            completed: completed.iter().copied().collect(),
            failed: failed.clone(),
            proofs: completed
                .iter()
                .filter_map(|id| self.cache.get(*id))
                .collect(),
        };

        let data = serde_json::to_vec_pretty(&session_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        fs::write(path, data)
    }

    /// Loads a session from disk.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let data = fs::read(path)?;
        let saved: SavedSession = serde_json::from_slice(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let cache = shared_cache();

        // Restore proofs to cache
        for proof in saved.proofs {
            let chunk_proof = ChunkProof {
                chunk_id: proof.chunk_id,
                proof: proof.proof,
                public_inputs: Vec::new(), // Not stored in CachedProof
                error_bound: proof.error_bound,
                generation_time_ms: proof.generation_time_ms,
            };
            cache.insert(&chunk_proof);
        }

        Ok(Self {
            id: saved.state.id.clone(),
            cache,
            state: Mutex::new(saved.state),
            completed: RwLock::new(saved.completed.into_iter().collect()),
            failed: RwLock::new(saved.failed),
        })
    }

    fn generate_id() -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: [u8; 16] = rng.gen();
        hex::encode(bytes)
    }
}

/// Saved session data for persistence.
#[derive(Serialize, Deserialize)]
struct SavedSession {
    state: SessionState,
    completed: Vec<ChunkId>,
    failed: HashMap<ChunkId, String>,
    proofs: Vec<CachedProof>,
}

/// Simple hex encoding for session IDs.
mod hex {
    pub fn encode(bytes: [u8; 16]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// Aggregated proof cache for storing aggregation results.
pub struct AggregatedProofCache {
    /// Cached aggregated proofs.
    proofs: RwLock<HashMap<u64, CachedAggregatedProof>>,
    /// Total size in bytes.
    total_size: Mutex<usize>,
    /// Maximum size.
    max_size: usize,
}

/// A cached aggregated proof.
#[derive(Clone, Serialize, Deserialize)]
pub struct CachedAggregatedProof {
    /// The aggregated proof.
    pub proof: AggregatedProof,
    /// When it was created.
    pub created_at: u64,
    /// Size in bytes.
    pub size_bytes: usize,
}

impl AggregatedProofCache {
    /// Creates a new aggregated proof cache.
    pub fn new(max_size: usize) -> Self {
        Self {
            proofs: RwLock::new(HashMap::new()),
            total_size: Mutex::new(0),
            max_size,
        }
    }

    /// Inserts an aggregated proof.
    pub fn insert(&self, proof: AggregatedProof) {
        let size = proof.proof.len() + proof.public_inputs.len() * 32;
        let cached = CachedAggregatedProof {
            proof: proof.clone(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            size_bytes: size,
        };

        let mut proofs = self.proofs.write().unwrap();
        let mut total = self.total_size.lock().unwrap();

        // Simple size-based eviction
        while *total + size > self.max_size && !proofs.is_empty() {
            let oldest = proofs
                .iter()
                .min_by_key(|(_, p)| p.created_at)
                .map(|(id, p)| (*id, p.size_bytes));

            if let Some((id, old_size)) = oldest {
                proofs.remove(&id);
                *total = total.saturating_sub(old_size);
            } else {
                break;
            }
        }

        proofs.insert(proof.id.0, cached);
        *total += size;
    }

    /// Gets an aggregated proof by ID.
    pub fn get(&self, id: u64) -> Option<AggregatedProof> {
        self.proofs.read().unwrap().get(&id).map(|c| c.proof.clone())
    }

    /// Returns the number of cached proofs.
    pub fn len(&self) -> usize {
        self.proofs.read().unwrap().len()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.proofs.read().unwrap().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_proof(id: u64) -> ChunkProof {
        ChunkProof {
            chunk_id: ChunkId(id),
            proof: vec![1, 2, 3, 4, 5],
            public_inputs: vec![[id as u8; 32]],
            error_bound: 0.01,
            generation_time_ms: 100,
        }
    }

    #[test]
    fn test_basic_cache_operations() {
        let cache = ProofCache::new();
        let proof = make_test_proof(1);

        assert!(cache.insert(&proof));
        assert!(cache.contains(ChunkId(1)));

        let cached = cache.get(ChunkId(1)).unwrap();
        assert_eq!(cached.chunk_id.0, 1);
        assert_eq!(cached.proof, proof.proof);
    }

    #[test]
    fn test_deduplication() {
        let cache = ProofCache::with_config(ProofCacheConfig {
            deduplicate: true,
            ..Default::default()
        });

        let proof1 = make_test_proof(1);
        let mut proof2 = make_test_proof(2);
        proof2.public_inputs = proof1.public_inputs.clone(); // Same inputs

        assert!(cache.insert(&proof1));
        assert!(!cache.insert(&proof2)); // Should be rejected as duplicate

        let stats = cache.stats();
        assert_eq!(stats.duplicates_avoided, 1);
    }

    #[test]
    fn test_eviction() {
        let cache = ProofCache::with_config(ProofCacheConfig {
            max_proofs: 2,
            ..Default::default()
        });

        for i in 0..3 {
            cache.insert(&make_test_proof(i));
        }

        // Oldest proof should be evicted
        assert_eq!(cache.len(), 2);
        let stats = cache.stats();
        assert!(stats.proofs_evicted > 0);
    }

    #[test]
    fn test_session() {
        let session = ProvingSession::new(3);

        // Record successes and failures
        session.record_success(&make_test_proof(1));
        session.record_success(&make_test_proof(2));
        session.record_failure(ChunkId(3), "Test error".to_string());

        assert!(session.is_completed(ChunkId(1)));
        assert!(session.is_completed(ChunkId(2)));
        assert!(session.is_failed(ChunkId(3)));

        let state = session.state();
        assert_eq!(state.completed_chunks, 2);
        assert_eq!(state.failed_chunks, 1);
        assert!(state.is_complete);
    }

    #[test]
    fn test_get_pending() {
        let session = ProvingSession::new(5);

        session.record_success(&make_test_proof(1));
        session.record_failure(ChunkId(3), "Error".to_string());

        let all: Vec<ChunkId> = (1..=5).map(ChunkId).collect();
        let pending = session.get_pending(&all);

        assert_eq!(pending.len(), 3);
        assert!(pending.contains(&ChunkId(2)));
        assert!(pending.contains(&ChunkId(4)));
        assert!(pending.contains(&ChunkId(5)));
    }

    #[test]
    fn test_mark_verified() {
        let cache = ProofCache::new();
        let proof = make_test_proof(1);

        cache.insert(&proof);
        assert!(!cache.get(ChunkId(1)).unwrap().verified);

        cache.mark_verified(ChunkId(1));
        assert!(cache.get(ChunkId(1)).unwrap().verified);
    }

    #[test]
    fn test_stats() {
        let cache = ProofCache::new();

        // Insert and access
        cache.insert(&make_test_proof(1));
        cache.get(ChunkId(1)); // Hit
        cache.get(ChunkId(2)); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.proofs_stored, 1);
    }
}
