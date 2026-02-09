//! Data Sharding Module.
//!
//! Provides functionality for partitioning datasets across multiple nodes
//! in federated learning scenarios. Supports:
//!
//! - Multiple sharding strategies (IID, non-IID, Dirichlet)
//! - Worker-aware shard assignment with load balancing
//! - Dynamic rebalancing when workers join/leave
//! - Locality-aware sharding for efficient data transfer
//! - Shard replication for fault tolerance
//! - Progress tracking for distributed training
//! - Checksum verification for data integrity

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::SystemTime;

use super::dataset::Sample;
use super::merkle::{Hash, MerkleHasher, Sha256Hasher};

/// Shard identifier.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard_{}", self.0)
    }
}

/// Sharding strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Random assignment with uniform distribution.
    Random { seed: u64 },
    /// Round-robin assignment.
    RoundRobin,
    /// Hash-based assignment using sample ID.
    HashBased,
    /// IID (independent and identically distributed) - uniform label distribution.
    IID { seed: u64 },
    /// Non-IID with label skew.
    NonIID {
        /// Number of classes per shard.
        classes_per_shard: usize,
        seed: u64,
    },
    /// Dirichlet distribution for label imbalance.
    Dirichlet {
        /// Concentration parameter (lower = more imbalanced).
        alpha: f64,
        seed: u64,
    },
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        Self::Random { seed: 42 }
    }
}

/// Configuration for data sharding.
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Number of shards.
    pub num_shards: u32,
    /// Sharding strategy.
    pub strategy: ShardingStrategy,
    /// Minimum samples per shard.
    pub min_samples: usize,
    /// Maximum samples per shard (None for no limit).
    pub max_samples: Option<usize>,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            num_shards: 10,
            strategy: ShardingStrategy::default(),
            min_samples: 1,
            max_samples: None,
        }
    }
}

/// A data shard containing a subset of samples.
#[derive(Debug, Clone)]
pub struct DataShard {
    /// Shard ID.
    pub id: ShardId,
    /// Sample indices in the original dataset.
    pub sample_indices: Vec<usize>,
    /// Shard statistics.
    pub stats: ShardStats,
}

/// Statistics about a shard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardStats {
    /// Number of samples.
    pub num_samples: usize,
    /// Label distribution (label -> count).
    pub label_distribution: HashMap<usize, usize>,
    /// Total data size in bytes.
    pub size_bytes: usize,
}

impl DataShard {
    /// Creates a new shard.
    pub fn new(id: ShardId, sample_indices: Vec<usize>) -> Self {
        let stats = ShardStats {
            num_samples: sample_indices.len(),
            ..Default::default()
        };

        Self {
            id,
            sample_indices,
            stats,
        }
    }

    /// Creates a shard with full statistics.
    pub fn with_stats(id: ShardId, sample_indices: Vec<usize>, stats: ShardStats) -> Self {
        Self {
            id,
            sample_indices,
            stats,
        }
    }

    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.sample_indices.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.sample_indices.is_empty()
    }

    /// Computes statistics from samples.
    pub fn compute_stats(&mut self, samples: &[Sample]) {
        self.stats.num_samples = self.sample_indices.len();
        self.stats.label_distribution.clear();
        self.stats.size_bytes = 0;

        for &idx in &self.sample_indices {
            if let Some(sample) = samples.get(idx) {
                let label = sample.labels.first().copied().unwrap_or(0) as usize;
                *self.stats.label_distribution.entry(label).or_insert(0) += 1;
                self.stats.size_bytes += sample.features.len() * std::mem::size_of::<f32>();
                self.stats.size_bytes += sample.labels.len();
            }
        }
    }
}

// ============================================================================
// WORKER MANAGEMENT
// ============================================================================

/// Unique worker identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct WorkerId(pub String);

impl WorkerId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Worker capabilities and status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID.
    pub id: WorkerId,
    /// Available memory in bytes.
    pub memory_bytes: u64,
    /// Number of CPU cores.
    pub cpu_cores: u32,
    /// Whether GPU is available.
    pub has_gpu: bool,
    /// GPU memory in bytes (if available).
    pub gpu_memory_bytes: Option<u64>,
    /// Network bandwidth estimate (bytes/sec).
    pub bandwidth_bps: u64,
    /// Worker's location/zone for locality.
    pub zone: Option<String>,
    /// Current status.
    pub status: WorkerStatus,
    /// Last heartbeat timestamp.
    pub last_heartbeat: u64,
    /// Processing capacity (samples/sec estimate).
    pub processing_rate: f64,
}

impl WorkerInfo {
    /// Creates a new worker info.
    pub fn new(id: WorkerId) -> Self {
        Self {
            id,
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB default
            cpu_cores: 4,
            has_gpu: false,
            gpu_memory_bytes: None,
            bandwidth_bps: 100 * 1024 * 1024, // 100 MB/s
            zone: None,
            status: WorkerStatus::Active,
            last_heartbeat: current_timestamp(),
            processing_rate: 1000.0,
        }
    }

    /// Creates a GPU worker.
    pub fn with_gpu(id: WorkerId, gpu_memory: u64) -> Self {
        Self {
            has_gpu: true,
            gpu_memory_bytes: Some(gpu_memory),
            ..Self::new(id)
        }
    }

    /// Checks if the worker is healthy.
    pub fn is_healthy(&self) -> bool {
        self.status == WorkerStatus::Active
    }

    /// Updates the heartbeat.
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = current_timestamp();
    }

    /// Checks if the worker is stale (no heartbeat in given seconds).
    pub fn is_stale(&self, timeout_secs: u64) -> bool {
        current_timestamp().saturating_sub(self.last_heartbeat) > timeout_secs
    }
}

/// Worker status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Worker is active and available.
    Active,
    /// Worker is busy processing.
    Busy,
    /// Worker is paused.
    Paused,
    /// Worker has failed.
    Failed,
    /// Worker is draining (not accepting new work).
    Draining,
}

// ============================================================================
// SHARD METADATA AND VERIFICATION
// ============================================================================

/// Complete shard metadata for distributed verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    /// Shard ID.
    pub shard_id: ShardId,
    /// Number of samples.
    pub num_samples: usize,
    /// Total size in bytes.
    pub size_bytes: u64,
    /// Content hash for verification.
    pub content_hash: Hash,
    /// Version number.
    pub version: u64,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last modified timestamp.
    pub modified_at: u64,
    /// Sample index range (start, end exclusive).
    pub index_range: (usize, usize),
    /// Label distribution.
    pub label_distribution: HashMap<usize, usize>,
    /// Replication factor.
    pub replication_factor: u32,
    /// Assigned workers.
    pub assigned_workers: Vec<WorkerId>,
}

impl ShardMetadata {
    /// Creates new shard metadata.
    pub fn new(shard_id: ShardId, num_samples: usize, size_bytes: u64) -> Self {
        let now = current_timestamp();
        Self {
            shard_id,
            num_samples,
            size_bytes,
            content_hash: Hash::zero(),
            version: 1,
            created_at: now,
            modified_at: now,
            index_range: (0, num_samples),
            label_distribution: HashMap::new(),
            replication_factor: 1,
            assigned_workers: Vec::new(),
        }
    }

    /// Computes content hash from samples.
    pub fn compute_hash(&mut self, samples: &[Sample], indices: &[usize]) {
        let mut data = Vec::new();
        for &idx in indices {
            if let Some(sample) = samples.get(idx) {
                for f in &sample.features {
                    data.extend_from_slice(&f.to_le_bytes());
                }
                data.extend_from_slice(&sample.labels);
            }
        }
        self.content_hash = Sha256Hasher.hash_leaf(&data);
        self.modified_at = current_timestamp();
    }
}

/// Shard verification result.
#[derive(Debug, Clone)]
pub struct ShardVerification {
    /// Shard ID.
    pub shard_id: ShardId,
    /// Whether verification passed.
    pub valid: bool,
    /// Expected hash.
    pub expected_hash: Hash,
    /// Actual hash.
    pub actual_hash: Hash,
    /// Verification timestamp.
    pub verified_at: u64,
    /// Missing samples (if any).
    pub missing_samples: Vec<usize>,
}

// ============================================================================
// DISTRIBUTED SHARD REGISTRY
// ============================================================================

/// Registry for tracking shards across distributed workers.
pub struct ShardRegistry {
    /// Shard metadata.
    shards: HashMap<ShardId, ShardMetadata>,
    /// Worker information.
    workers: HashMap<WorkerId, WorkerInfo>,
    /// Shard to worker assignments.
    assignments: HashMap<ShardId, Vec<WorkerId>>,
    /// Worker to shard assignments.
    worker_shards: HashMap<WorkerId, Vec<ShardId>>,
    /// Processing progress.
    progress: HashMap<ShardId, ShardProgress>,
    /// Configuration.
    config: ShardRegistryConfig,
}

/// Configuration for shard registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRegistryConfig {
    /// Default replication factor.
    pub default_replication: u32,
    /// Maximum shards per worker.
    pub max_shards_per_worker: u32,
    /// Worker heartbeat timeout in seconds.
    pub heartbeat_timeout_secs: u64,
    /// Enable locality-aware assignment.
    pub locality_aware: bool,
    /// Enable automatic rebalancing.
    pub auto_rebalance: bool,
}

impl Default for ShardRegistryConfig {
    fn default() -> Self {
        Self {
            default_replication: 1,
            max_shards_per_worker: 100,
            heartbeat_timeout_secs: 60,
            locality_aware: true,
            auto_rebalance: true,
        }
    }
}

/// Progress tracking for a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardProgress {
    /// Shard ID.
    pub shard_id: ShardId,
    /// Samples processed.
    pub samples_processed: usize,
    /// Total samples.
    pub total_samples: usize,
    /// Current epoch.
    pub current_epoch: u32,
    /// Total epochs.
    pub total_epochs: u32,
    /// Processing worker.
    pub worker: Option<WorkerId>,
    /// Start timestamp.
    pub started_at: u64,
    /// Last update timestamp.
    pub updated_at: u64,
    /// Completion timestamp (if done).
    pub completed_at: Option<u64>,
}

impl ShardProgress {
    /// Creates new progress tracker.
    pub fn new(shard_id: ShardId, total_samples: usize, total_epochs: u32) -> Self {
        Self {
            shard_id,
            samples_processed: 0,
            total_samples,
            current_epoch: 0,
            total_epochs,
            worker: None,
            started_at: current_timestamp(),
            updated_at: current_timestamp(),
            completed_at: None,
        }
    }

    /// Returns progress as a percentage.
    pub fn percentage(&self) -> f64 {
        if self.total_samples == 0 || self.total_epochs == 0 {
            return 0.0;
        }
        let total = (self.total_samples * self.total_epochs as usize) as f64;
        let done = (self.current_epoch as usize * self.total_samples + self.samples_processed) as f64;
        (done / total * 100.0).min(100.0)
    }

    /// Returns true if complete.
    pub fn is_complete(&self) -> bool {
        self.completed_at.is_some()
    }

    /// Updates progress.
    pub fn update(&mut self, samples_processed: usize, current_epoch: u32) {
        self.samples_processed = samples_processed;
        self.current_epoch = current_epoch;
        self.updated_at = current_timestamp();

        if current_epoch >= self.total_epochs && samples_processed >= self.total_samples {
            self.completed_at = Some(current_timestamp());
        }
    }
}

impl ShardRegistry {
    /// Creates a new shard registry.
    pub fn new(config: ShardRegistryConfig) -> Self {
        Self {
            shards: HashMap::new(),
            workers: HashMap::new(),
            assignments: HashMap::new(),
            worker_shards: HashMap::new(),
            progress: HashMap::new(),
            config,
        }
    }

    /// Creates with default configuration.
    pub fn default_registry() -> Self {
        Self::new(ShardRegistryConfig::default())
    }

    // ========================================================================
    // SHARD MANAGEMENT
    // ========================================================================

    /// Registers a shard.
    pub fn register_shard(&mut self, metadata: ShardMetadata) {
        let shard_id = metadata.shard_id;
        self.shards.insert(shard_id, metadata);
        self.assignments.entry(shard_id).or_default();
    }

    /// Gets shard metadata.
    pub fn get_shard(&self, shard_id: ShardId) -> Option<&ShardMetadata> {
        self.shards.get(&shard_id)
    }

    /// Gets mutable shard metadata.
    pub fn get_shard_mut(&mut self, shard_id: ShardId) -> Option<&mut ShardMetadata> {
        self.shards.get_mut(&shard_id)
    }

    /// Lists all shards.
    pub fn list_shards(&self) -> Vec<&ShardMetadata> {
        self.shards.values().collect()
    }

    /// Removes a shard.
    pub fn remove_shard(&mut self, shard_id: ShardId) -> Option<ShardMetadata> {
        // Remove from worker assignments
        if let Some(workers) = self.assignments.remove(&shard_id) {
            for worker_id in workers {
                if let Some(worker_shards) = self.worker_shards.get_mut(&worker_id) {
                    worker_shards.retain(|s| *s != shard_id);
                }
            }
        }
        self.progress.remove(&shard_id);
        self.shards.remove(&shard_id)
    }

    // ========================================================================
    // WORKER MANAGEMENT
    // ========================================================================

    /// Registers a worker.
    pub fn register_worker(&mut self, info: WorkerInfo) {
        let worker_id = info.id.clone();
        self.workers.insert(worker_id.clone(), info);
        self.worker_shards.entry(worker_id).or_default();
    }

    /// Gets worker info.
    pub fn get_worker(&self, worker_id: &WorkerId) -> Option<&WorkerInfo> {
        self.workers.get(worker_id)
    }

    /// Updates worker heartbeat.
    pub fn worker_heartbeat(&mut self, worker_id: &WorkerId) -> bool {
        if let Some(worker) = self.workers.get_mut(worker_id) {
            worker.heartbeat();
            true
        } else {
            false
        }
    }

    /// Lists all workers.
    pub fn list_workers(&self) -> Vec<&WorkerInfo> {
        self.workers.values().collect()
    }

    /// Lists active workers.
    pub fn list_active_workers(&self) -> Vec<&WorkerInfo> {
        self.workers.values()
            .filter(|w| w.is_healthy())
            .collect()
    }

    /// Removes a worker.
    pub fn remove_worker(&mut self, worker_id: &WorkerId) -> Option<WorkerInfo> {
        // Reassign shards if auto-rebalance is enabled
        if self.config.auto_rebalance {
            if let Some(shards) = self.worker_shards.remove(worker_id) {
                for shard_id in shards {
                    if let Some(workers) = self.assignments.get_mut(&shard_id) {
                        workers.retain(|w| w != worker_id);
                    }
                }
            }
        }
        self.workers.remove(worker_id)
    }

    /// Sets a worker's status.
    pub fn set_worker_status(&mut self, worker_id: &WorkerId, status: WorkerStatus) -> bool {
        if let Some(worker) = self.workers.get_mut(worker_id) {
            worker.status = status;
            true
        } else {
            false
        }
    }

    /// Marks stale workers as failed.
    pub fn check_worker_health(&mut self) -> Vec<WorkerId> {
        let timeout = self.config.heartbeat_timeout_secs;
        let stale: Vec<WorkerId> = self.workers.iter()
            .filter(|(_, w)| w.is_stale(timeout) && w.status == WorkerStatus::Active)
            .map(|(id, _)| id.clone())
            .collect();

        for worker_id in &stale {
            if let Some(worker) = self.workers.get_mut(worker_id) {
                worker.status = WorkerStatus::Failed;
            }
        }

        stale
    }

    // ========================================================================
    // ASSIGNMENT MANAGEMENT
    // ========================================================================

    /// Assigns a shard to a worker.
    pub fn assign_shard(&mut self, shard_id: ShardId, worker_id: WorkerId) -> bool {
        // Check limits
        if let Some(worker_shards) = self.worker_shards.get(&worker_id) {
            if worker_shards.len() >= self.config.max_shards_per_worker as usize {
                return false;
            }
        }

        // Update assignments
        self.assignments.entry(shard_id).or_default().push(worker_id.clone());
        self.worker_shards.entry(worker_id.clone()).or_default().push(shard_id);

        // Update shard metadata
        if let Some(shard) = self.shards.get_mut(&shard_id) {
            if !shard.assigned_workers.contains(&worker_id) {
                shard.assigned_workers.push(worker_id);
            }
        }

        true
    }

    /// Unassigns a shard from a worker.
    pub fn unassign_shard(&mut self, shard_id: ShardId, worker_id: &WorkerId) {
        if let Some(workers) = self.assignments.get_mut(&shard_id) {
            workers.retain(|w| w != worker_id);
        }
        if let Some(shards) = self.worker_shards.get_mut(worker_id) {
            shards.retain(|s| *s != shard_id);
        }
        if let Some(shard) = self.shards.get_mut(&shard_id) {
            shard.assigned_workers.retain(|w| w != worker_id);
        }
    }

    /// Gets workers assigned to a shard.
    pub fn get_shard_workers(&self, shard_id: ShardId) -> Vec<&WorkerInfo> {
        self.assignments.get(&shard_id)
            .map(|workers| {
                workers.iter()
                    .filter_map(|w| self.workers.get(w))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Gets shards assigned to a worker.
    pub fn get_worker_shards(&self, worker_id: &WorkerId) -> Vec<&ShardMetadata> {
        self.worker_shards.get(worker_id)
            .map(|shards| {
                shards.iter()
                    .filter_map(|s| self.shards.get(s))
                    .collect()
            })
            .unwrap_or_default()
    }

    // ========================================================================
    // LOAD BALANCING
    // ========================================================================

    /// Performs automatic shard assignment with load balancing.
    pub fn auto_assign_shards(&mut self) -> Vec<(ShardId, WorkerId)> {
        let mut assignments = Vec::new();
        let active_workers: Vec<WorkerId> = self.list_active_workers()
            .iter()
            .map(|w| w.id.clone())
            .collect();

        if active_workers.is_empty() {
            return assignments;
        }

        // Find unassigned shards
        let unassigned: Vec<ShardId> = self.shards.keys()
            .filter(|s| {
                self.assignments.get(s)
                    .map(|w| w.is_empty())
                    .unwrap_or(true)
            })
            .copied()
            .collect();

        // Calculate current load per worker
        let mut worker_load: HashMap<&WorkerId, usize> = active_workers.iter()
            .map(|w| (w, self.worker_shards.get(w).map(|s| s.len()).unwrap_or(0)))
            .collect();

        // Sort workers by load (ascending)
        let mut sorted_workers: Vec<&WorkerId> = active_workers.iter().collect();

        for shard_id in unassigned {
            // Re-sort by current load
            sorted_workers.sort_by_key(|w| worker_load.get(w).unwrap_or(&0));

            // Assign to least loaded worker
            if let Some(worker_id) = sorted_workers.first() {
                if self.assign_shard(shard_id, (*worker_id).clone()) {
                    assignments.push((shard_id, (*worker_id).clone()));
                    *worker_load.entry(*worker_id).or_insert(0) += 1;
                }
            }
        }

        assignments
    }

    /// Rebalances shards across workers.
    pub fn rebalance(&mut self) -> Vec<(ShardId, WorkerId, WorkerId)> {
        let mut moves = Vec::new();
        let active_workers: Vec<WorkerId> = self.list_active_workers()
            .iter()
            .map(|w| w.id.clone())
            .collect();

        if active_workers.len() < 2 {
            return moves;
        }

        // Calculate current load
        let mut worker_load: BTreeMap<WorkerId, Vec<ShardId>> = BTreeMap::new();
        for worker_id in &active_workers {
            let shards = self.worker_shards.get(worker_id)
                .cloned()
                .unwrap_or_default();
            worker_load.insert(worker_id.clone(), shards);
        }

        // Calculate average and threshold
        let total_shards: usize = worker_load.values().map(|s| s.len()).sum();
        let avg = total_shards / active_workers.len();
        let threshold = (avg as f64 * 0.2).ceil() as usize; // 20% threshold

        // Find overloaded and underloaded workers
        let overloaded: Vec<_> = worker_load.iter()
            .filter(|(_, shards)| shards.len() > avg + threshold)
            .map(|(w, s)| (w.clone(), s.clone()))
            .collect();

        let mut underloaded: Vec<_> = worker_load.iter()
            .filter(|(_, shards)| shards.len() < avg.saturating_sub(threshold))
            .map(|(w, _)| w.clone())
            .collect();

        // Move shards from overloaded to underloaded
        for (from_worker, shards) in overloaded {
            let excess = shards.len().saturating_sub(avg);
            for shard_id in shards.into_iter().take(excess) {
                if let Some(to_worker) = underloaded.pop() {
                    self.unassign_shard(shard_id, &from_worker);
                    if self.assign_shard(shard_id, to_worker.clone()) {
                        moves.push((shard_id, from_worker.clone(), to_worker.clone()));
                    }
                    // Re-add if still underloaded
                    let new_load = self.worker_shards.get(&to_worker).map(|s| s.len()).unwrap_or(0);
                    if new_load < avg.saturating_sub(threshold) {
                        underloaded.push(to_worker);
                    }
                }
            }
        }

        moves
    }

    // ========================================================================
    // PROGRESS TRACKING
    // ========================================================================

    /// Initializes progress tracking for a shard.
    pub fn init_progress(&mut self, shard_id: ShardId, total_samples: usize, total_epochs: u32) {
        let progress = ShardProgress::new(shard_id, total_samples, total_epochs);
        self.progress.insert(shard_id, progress);
    }

    /// Updates shard progress.
    pub fn update_progress(
        &mut self,
        shard_id: ShardId,
        samples_processed: usize,
        current_epoch: u32,
        worker: Option<WorkerId>,
    ) {
        if let Some(progress) = self.progress.get_mut(&shard_id) {
            progress.update(samples_processed, current_epoch);
            progress.worker = worker;
        }
    }

    /// Gets shard progress.
    pub fn get_progress(&self, shard_id: ShardId) -> Option<&ShardProgress> {
        self.progress.get(&shard_id)
    }

    /// Gets overall progress.
    pub fn overall_progress(&self) -> OverallProgress {
        let total_shards = self.shards.len();
        let completed_shards = self.progress.values()
            .filter(|p| p.is_complete())
            .count();

        let total_samples: usize = self.progress.values()
            .map(|p| p.total_samples * p.total_epochs as usize)
            .sum();
        let processed_samples: usize = self.progress.values()
            .map(|p| p.current_epoch as usize * p.total_samples + p.samples_processed)
            .sum();

        let avg_percentage = if !self.progress.is_empty() {
            self.progress.values().map(|p| p.percentage()).sum::<f64>() / self.progress.len() as f64
        } else {
            0.0
        };

        OverallProgress {
            total_shards,
            completed_shards,
            total_samples,
            processed_samples,
            average_percentage: avg_percentage,
        }
    }

    // ========================================================================
    // VERIFICATION
    // ========================================================================

    /// Verifies a shard's data integrity.
    pub fn verify_shard(&self, shard_id: ShardId, samples: &[Sample], indices: &[usize]) -> ShardVerification {
        let shard = self.shards.get(&shard_id);
        let expected_hash = shard.map(|s| s.content_hash).unwrap_or_else(Hash::zero);

        // Compute actual hash
        let mut data = Vec::new();
        let mut missing = Vec::new();
        for &idx in indices {
            if let Some(sample) = samples.get(idx) {
                for f in &sample.features {
                    data.extend_from_slice(&f.to_le_bytes());
                }
                data.extend_from_slice(&sample.labels);
            } else {
                missing.push(idx);
            }
        }
        let actual_hash = Sha256Hasher.hash_leaf(&data);

        ShardVerification {
            shard_id,
            valid: actual_hash == expected_hash && missing.is_empty(),
            expected_hash,
            actual_hash,
            verified_at: current_timestamp(),
            missing_samples: missing,
        }
    }

    /// Saves the registry to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let snapshot = ShardRegistrySnapshot {
            shards: self.shards.values().cloned().collect(),
            workers: self.workers.values().cloned().collect(),
            assignments: self.assignments.clone(),
            progress: self.progress.values().cloned().collect(),
            config: self.config.clone(),
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Loads the registry from a JSON file, rebuilding indices.
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: ShardRegistrySnapshot = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut registry = Self::new(snapshot.config);
        for worker in snapshot.workers {
            registry.register_worker(worker);
        }
        for shard in snapshot.shards {
            let shard_id = shard.shard_id;
            let assigned = shard.assigned_workers.clone();
            registry.register_shard(shard);
            for worker_id in assigned {
                registry.assign_shard(shard_id, worker_id);
            }
        }
        for (shard_id, progress) in snapshot.assignments.keys().zip(snapshot.progress.iter()) {
            registry.progress.insert(*shard_id, progress.clone());
        }
        // Restore any remaining progress entries
        for p in snapshot.progress {
            registry.progress.entry(p.shard_id).or_insert(p);
        }
        Ok(registry)
    }
}

/// Serializable snapshot of a ShardRegistry for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShardRegistrySnapshot {
    shards: Vec<ShardMetadata>,
    workers: Vec<WorkerInfo>,
    assignments: HashMap<ShardId, Vec<WorkerId>>,
    progress: Vec<ShardProgress>,
    config: ShardRegistryConfig,
}

/// Overall progress summary.
#[derive(Debug, Clone)]
pub struct OverallProgress {
    /// Total number of shards.
    pub total_shards: usize,
    /// Completed shards.
    pub completed_shards: usize,
    /// Total samples across all shards.
    pub total_samples: usize,
    /// Processed samples.
    pub processed_samples: usize,
    /// Average progress percentage.
    pub average_percentage: f64,
}

// ============================================================================
// LOCALITY-AWARE SHARDING
// ============================================================================

/// Locality-aware shard assigner.
pub struct LocalityAwareAssigner {
    /// Zone preferences (zone -> preferred workers).
    zone_workers: HashMap<String, Vec<WorkerId>>,
    /// Data locality hints (shard -> preferred zone).
    shard_zones: HashMap<ShardId, String>,
}

impl LocalityAwareAssigner {
    /// Creates a new locality-aware assigner.
    pub fn new() -> Self {
        Self {
            zone_workers: HashMap::new(),
            shard_zones: HashMap::new(),
        }
    }

    /// Registers a worker in a zone.
    pub fn register_zone_worker(&mut self, zone: &str, worker_id: WorkerId) {
        self.zone_workers.entry(zone.to_string()).or_default().push(worker_id);
    }

    /// Sets a shard's preferred zone.
    pub fn set_shard_zone(&mut self, shard_id: ShardId, zone: &str) {
        self.shard_zones.insert(shard_id, zone.to_string());
    }

    /// Gets workers in a shard's preferred zone.
    pub fn get_preferred_workers(&self, shard_id: ShardId) -> Vec<&WorkerId> {
        self.shard_zones.get(&shard_id)
            .and_then(|zone| self.zone_workers.get(zone))
            .map(|workers| workers.iter().collect())
            .unwrap_or_default()
    }

    /// Performs locality-aware assignment.
    pub fn assign_with_locality(
        &self,
        registry: &mut ShardRegistry,
        shard_id: ShardId,
    ) -> Option<WorkerId> {
        let preferred = self.get_preferred_workers(shard_id);

        // Try preferred workers first
        for worker_id in preferred {
            if let Some(worker) = registry.get_worker(worker_id) {
                if worker.is_healthy() {
                    let current_shards = registry.worker_shards
                        .get(worker_id)
                        .map(|s| s.len())
                        .unwrap_or(0);
                    if current_shards < registry.config.max_shards_per_worker as usize {
                        let id = worker_id.clone();
                        registry.assign_shard(shard_id, id.clone());
                        return Some(id);
                    }
                }
            }
        }

        // Collect active worker IDs first to avoid borrow issues
        let active_worker_ids: Vec<(WorkerId, usize)> = registry.list_active_workers()
            .iter()
            .map(|w| {
                let current = registry.worker_shards
                    .get(&w.id)
                    .map(|s| s.len())
                    .unwrap_or(0);
                (w.id.clone(), current)
            })
            .collect();

        let max_shards = registry.config.max_shards_per_worker as usize;

        // Fall back to any available worker
        for (worker_id, current_shards) in active_worker_ids {
            if current_shards < max_shards {
                registry.assign_shard(shard_id, worker_id.clone());
                return Some(worker_id);
            }
        }

        None
    }
}

impl Default for LocalityAwareAssigner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SHARD STREAMING
// ============================================================================

/// Configuration for shard streaming.
#[derive(Debug, Clone)]
pub struct ShardStreamConfig {
    /// Chunk size in samples.
    pub chunk_size: usize,
    /// Enable compression.
    pub compress: bool,
    /// Checksum each chunk.
    pub checksum_chunks: bool,
}

impl Default for ShardStreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            compress: true,
            checksum_chunks: true,
        }
    }
}

/// A chunk of shard data for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardChunk {
    /// Shard ID.
    pub shard_id: ShardId,
    /// Chunk number.
    pub chunk_num: u32,
    /// Total chunks.
    pub total_chunks: u32,
    /// Sample indices in this chunk.
    pub sample_indices: Vec<usize>,
    /// Chunk checksum.
    pub checksum: Option<Hash>,
    /// Is this the last chunk?
    pub is_last: bool,
}

/// Shard streamer for efficient data transfer.
pub struct ShardStreamer {
    /// Configuration.
    config: ShardStreamConfig,
}

impl ShardStreamer {
    /// Creates a new shard streamer.
    pub fn new(config: ShardStreamConfig) -> Self {
        Self { config }
    }

    /// Creates with default configuration.
    pub fn default_streamer() -> Self {
        Self::new(ShardStreamConfig::default())
    }

    /// Chunks a shard for streaming.
    pub fn create_chunks(&self, shard: &DataShard) -> Vec<ShardChunk> {
        let total_chunks = (shard.sample_indices.len() + self.config.chunk_size - 1)
            / self.config.chunk_size;

        shard.sample_indices
            .chunks(self.config.chunk_size)
            .enumerate()
            .map(|(i, indices)| {
                let checksum = if self.config.checksum_chunks {
                    let data: Vec<u8> = indices.iter()
                        .flat_map(|&idx| idx.to_le_bytes())
                        .collect();
                    Some(Sha256Hasher.hash_leaf(&data))
                } else {
                    None
                };

                ShardChunk {
                    shard_id: shard.id,
                    chunk_num: i as u32,
                    total_chunks: total_chunks as u32,
                    sample_indices: indices.to_vec(),
                    checksum,
                    is_last: i == total_chunks - 1,
                }
            })
            .collect()
    }

    /// Reassembles chunks into a shard.
    pub fn reassemble_chunks(&self, chunks: &[ShardChunk]) -> Option<DataShard> {
        if chunks.is_empty() {
            return None;
        }

        let shard_id = chunks[0].shard_id;
        let expected_total = chunks[0].total_chunks;

        // Verify all chunks present
        let mut seen: HashSet<u32> = HashSet::new();
        for chunk in chunks {
            if chunk.shard_id != shard_id {
                return None;
            }
            seen.insert(chunk.chunk_num);
        }

        if seen.len() != expected_total as usize {
            return None;
        }

        // Sort and combine
        let mut sorted_chunks = chunks.to_vec();
        sorted_chunks.sort_by_key(|c| c.chunk_num);

        let indices: Vec<usize> = sorted_chunks
            .iter()
            .flat_map(|c| c.sample_indices.iter().copied())
            .collect();

        Some(DataShard::new(shard_id, indices))
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Data sharder for partitioning datasets.
pub struct DataSharder {
    /// Configuration.
    config: ShardingConfig,
}

impl DataSharder {
    /// Creates a new sharder with the given configuration.
    pub fn new(config: ShardingConfig) -> Self {
        Self { config }
    }

    /// Shards a dataset according to the strategy.
    pub fn shard(&self, samples: &[Sample]) -> Vec<DataShard> {
        match &self.config.strategy {
            ShardingStrategy::Random { seed } => self.shard_random(samples, *seed),
            ShardingStrategy::RoundRobin => self.shard_round_robin(samples),
            ShardingStrategy::HashBased => self.shard_hash_based(samples),
            ShardingStrategy::IID { seed } => self.shard_iid(samples, *seed),
            ShardingStrategy::NonIID { classes_per_shard, seed } => {
                self.shard_non_iid(samples, *classes_per_shard, *seed)
            }
            ShardingStrategy::Dirichlet { alpha, seed } => {
                self.shard_dirichlet(samples, *alpha, *seed)
            }
        }
    }

    fn shard_random(&self, samples: &[Sample], seed: u64) -> Vec<DataShard> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        indices.shuffle(&mut rng);

        self.distribute_indices(indices)
    }

    fn shard_round_robin(&self, samples: &[Sample]) -> Vec<DataShard> {
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        for (i, _) in samples.iter().enumerate() {
            let shard_idx = i % self.config.num_shards as usize;
            shards[shard_idx].push(i);
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn shard_hash_based(&self, samples: &[Sample]) -> Vec<DataShard> {
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        for (i, sample) in samples.iter().enumerate() {
            let hash = self.simple_hash(sample.id);
            let shard_idx = (hash % self.config.num_shards as u64) as usize;
            shards[shard_idx].push(i);
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn shard_iid(&self, samples: &[Sample], seed: u64) -> Vec<DataShard> {
        // For IID, we want each shard to have similar class distribution
        // This is essentially random sharding with balancing
        self.shard_random(samples, seed)
    }

    fn shard_non_iid(&self, samples: &[Sample], classes_per_shard: usize, seed: u64) -> Vec<DataShard> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Group samples by their class (using first label byte as class indicator)
        let mut class_samples: HashMap<u8, Vec<usize>> = HashMap::new();
        for (i, sample) in samples.iter().enumerate() {
            let class = sample.labels.first().copied().unwrap_or(0);
            class_samples.entry(class).or_default().push(i);
        }

        let _num_classes = class_samples.len();
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        // Assign classes to shards
        let classes: Vec<u8> = class_samples.keys().copied().collect();
        for (shard_idx, chunk) in classes.chunks(classes_per_shard.max(1)).enumerate() {
            if shard_idx >= self.config.num_shards as usize {
                break;
            }
            for &class in chunk {
                if let Some(indices) = class_samples.get_mut(&class) {
                    indices.shuffle(&mut rng);
                    shards[shard_idx].extend(indices.drain(..));
                }
            }
        }

        // Distribute remaining samples
        let remaining: Vec<usize> = class_samples.values().flatten().copied().collect();
        for (i, idx) in remaining.into_iter().enumerate() {
            let shard_idx = i % self.config.num_shards as usize;
            shards[shard_idx].push(idx);
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn shard_dirichlet(&self, samples: &[Sample], alpha: f64, seed: u64) -> Vec<DataShard> {
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Simplified Dirichlet-like distribution
        // In practice, would use proper Dirichlet sampling
        let mut weights: Vec<f64> = (0..self.config.num_shards)
            .map(|_| {
                // Gamma approximation
                let u: f64 = rng.gen();
                (-u.ln()).powf(alpha)
            })
            .collect();

        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        // Assign samples according to weights
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        for (i, _) in samples.iter().enumerate() {
            let p: f64 = rng.gen();
            let mut cumsum = 0.0;
            for (shard_idx, &w) in weights.iter().enumerate() {
                cumsum += w;
                if p <= cumsum {
                    shards[shard_idx].push(i);
                    break;
                }
            }
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn distribute_indices(&self, indices: Vec<usize>) -> Vec<DataShard> {
        let chunk_size = (indices.len() + self.config.num_shards as usize - 1) 
            / self.config.num_shards as usize;

        indices
            .chunks(chunk_size)
            .enumerate()
            .map(|(id, chunk)| DataShard::new(ShardId(id as u32), chunk.to_vec()))
            .collect()
    }

    fn simple_hash(&self, value: usize) -> u64 {
        let mut h = value as u64;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }
}

/// Shard assignment for distributed training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    /// Node ID to shard mapping.
    pub node_shards: HashMap<String, Vec<ShardId>>,
    /// Total number of shards.
    pub total_shards: u32,
}

impl ShardAssignment {
    /// Creates a new assignment.
    pub fn new(num_shards: u32) -> Self {
        Self {
            node_shards: HashMap::new(),
            total_shards: num_shards,
        }
    }

    /// Assigns a shard to a node.
    pub fn assign(&mut self, node_id: String, shard_id: ShardId) {
        self.node_shards.entry(node_id).or_default().push(shard_id);
    }

    /// Gets shards for a node.
    pub fn get_shards(&self, node_id: &str) -> Vec<ShardId> {
        self.node_shards.get(node_id).cloned().unwrap_or_default()
    }

    /// Returns the number of assigned nodes.
    pub fn num_nodes(&self) -> usize {
        self.node_shards.len()
    }
}

// ============================================================================
// WORKER DATA ASSIGNER — Full pipeline: dataset → shards → worker assignment → DataLoaders
// ============================================================================

use crate::error::{HelixError, DataError, HelixResult};
use super::DataLoader;

/// An assignment plan mapping workers to their shards and sample counts.
#[derive(Debug, Clone)]
pub struct AssignmentPlan {
    /// Maps WorkerId to list of ShardIds assigned to that worker.
    pub worker_shards: HashMap<WorkerId, Vec<ShardId>>,
    /// Per-worker sample counts.
    pub worker_sample_counts: HashMap<WorkerId, usize>,
    /// Total samples assigned across all workers.
    pub total_assigned: usize,
    /// Whether any samples were unassigned (e.g., remainder from uneven division).
    pub has_remainder: bool,
}

/// Orchestrates the full pipeline: dataset -> shards -> worker assignment -> per-worker DataLoaders.
pub struct WorkerDataAssigner {
    /// Sharding configuration used to partition the dataset.
    sharding_config: ShardingConfig,
    /// Registry configuration for worker/shard management.
    #[allow(dead_code)]
    registry_config: ShardRegistryConfig,
    /// Internal shard registry tracking workers, shards, and assignments.
    registry: ShardRegistry,
    /// Cached shards from the most recent `assign_dataset` call.
    /// Needed because `ShardMetadata` only stores an index range,
    /// but non-contiguous strategies (e.g., RoundRobin) produce
    /// non-contiguous sample index lists.
    cached_shards: HashMap<ShardId, DataShard>,
}

impl WorkerDataAssigner {
    /// Creates a new `WorkerDataAssigner` with the given sharding and registry configurations.
    pub fn new(sharding_config: ShardingConfig, registry_config: ShardRegistryConfig) -> Self {
        let registry = ShardRegistry::new(registry_config.clone());
        Self {
            sharding_config,
            registry_config,
            registry,
            cached_shards: HashMap::new(),
        }
    }

    /// Registers a list of workers as available for shard assignment.
    pub fn register_workers(&mut self, workers: Vec<WorkerInfo>) {
        for worker in workers {
            self.registry.register_worker(worker);
        }
    }

    /// Shards the dataset and assigns shards to registered workers.
    ///
    /// The `epochs` parameter is used to initialize progress tracking for each shard.
    /// Returns an `AssignmentPlan` summarizing the mapping.
    pub fn assign_dataset(&mut self, samples: &[Sample], epochs: u32) -> AssignmentPlan {
        // Step 1: Create shards from samples using the configured strategy.
        let sharder = DataSharder::new(self.sharding_config.clone());
        let shards = sharder.shard(samples);

        // Step 2: Register each shard in the registry with metadata and cache the shard.
        self.cached_shards.clear();
        for shard in shards {
            let metadata = ShardMetadata::new(
                shard.id,
                shard.len(),
                (shard.len() * std::mem::size_of::<f32>()) as u64,
            );
            self.registry.register_shard(metadata);
            self.registry.init_progress(shard.id, shard.len(), epochs);
            self.cached_shards.insert(shard.id, shard);
        }

        // Step 3: Auto-assign shards to workers (load-balanced).
        self.registry.auto_assign_shards();

        // Step 4: Build the assignment plan.
        let active_workers: Vec<WorkerId> = self
            .registry
            .list_active_workers()
            .iter()
            .map(|w| w.id.clone())
            .collect();

        let mut worker_shard_map: HashMap<WorkerId, Vec<ShardId>> = HashMap::new();
        let mut worker_sample_counts: HashMap<WorkerId, usize> = HashMap::new();
        let mut total_assigned: usize = 0;

        for worker_id in &active_workers {
            let assigned_shards = self.registry.get_worker_shards(worker_id);
            let shard_ids: Vec<ShardId> = assigned_shards.iter().map(|s| s.shard_id).collect();
            let sample_count: usize = assigned_shards.iter().map(|s| s.num_samples).sum();

            worker_shard_map.insert(worker_id.clone(), shard_ids);
            worker_sample_counts.insert(worker_id.clone(), sample_count);
            total_assigned += sample_count;
        }

        let has_remainder = total_assigned < samples.len();

        AssignmentPlan {
            worker_shards: worker_shard_map,
            worker_sample_counts,
            total_assigned,
            has_remainder,
        }
    }

    /// Builds a `WorkerDataLoader` for a specific worker based on its assigned shards.
    ///
    /// `features_per_sample` indicates the number of f64 feature values per sample,
    /// used to partition the flattened features into batches. Each sample becomes one
    /// batch entry.
    ///
    /// Returns `None` if the worker has no shard assignments.
    pub fn build_worker_loader(
        &self,
        worker_id: &WorkerId,
        samples: &[Sample],
        _features_per_sample: usize,
    ) -> Option<WorkerDataLoader> {
        let assigned_shards = self.registry.get_worker_shards(worker_id);
        if assigned_shards.is_empty() {
            return None;
        }

        // Collect all sample indices from the worker's cached shards.
        let shard_ids: Vec<ShardId> = assigned_shards.iter().map(|s| s.shard_id).collect();
        let all_indices: Vec<usize> = shard_ids
            .iter()
            .flat_map(|sid| {
                self.cached_shards
                    .get(sid)
                    .map(|shard| shard.sample_indices.as_slice())
                    .unwrap_or(&[])
            })
            .copied()
            .filter(|&idx| idx < samples.len())
            .collect();

        if all_indices.is_empty() {
            return None;
        }

        // Convert each sample to (inputs_f64, labels_f64) forming one batch entry.
        let mut batches: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        // Group samples into batches of `features_per_sample`-sized chunks.
        // Each batch contains one sample (granular batching).
        for &idx in &all_indices {
            let sample = &samples[idx];
            let features_f64: Vec<f64> = sample
                .features_as_f32()
                .into_iter()
                .map(|f| f as f64)
                .collect();
            let labels_f64: Vec<f64> = sample
                .labels_as_f32()
                .into_iter()
                .map(|f| f as f64)
                .collect();
            batches.push((features_f64, labels_f64));
        }

        Some(WorkerDataLoader {
            worker_id: worker_id.clone(),
            batches,
        })
    }

    /// Reassigns shards from a failed worker to other active workers.
    ///
    /// Marks the failed worker as `Failed`, collects its shards, unassigns them,
    /// then redistributes to the remaining active workers using load balancing.
    ///
    /// Returns a list of `(ShardId, WorkerId)` pairs describing the new assignments.
    pub fn reassign_on_failure(&mut self, failed_worker: &WorkerId) -> Vec<(ShardId, WorkerId)> {
        // Mark the worker as failed.
        self.registry.set_worker_status(failed_worker, WorkerStatus::Failed);

        // Collect the shards that were assigned to the failed worker.
        let orphaned_shards: Vec<ShardId> = self
            .registry
            .get_worker_shards(failed_worker)
            .iter()
            .map(|s| s.shard_id)
            .collect();

        // Unassign all shards from the failed worker.
        for shard_id in &orphaned_shards {
            self.registry.unassign_shard(*shard_id, failed_worker);
        }

        // Re-assign orphaned shards using auto_assign (they are now unassigned).
        let new_assignments = self.registry.auto_assign_shards();

        // Filter to only the shards that were orphaned (auto_assign may pick up others too).
        new_assignments
            .into_iter()
            .filter(|(sid, _)| orphaned_shards.contains(sid))
            .collect()
    }

    /// Returns a reference to the inner `ShardRegistry`.
    pub fn registry(&self) -> &ShardRegistry {
        &self.registry
    }

    /// Returns a mutable reference to the inner `ShardRegistry`.
    pub fn registry_mut(&mut self) -> &mut ShardRegistry {
        &mut self.registry
    }
}

/// A per-worker data loader that implements the `DataLoader` trait.
///
/// Contains the subset of training data assigned to a single worker,
/// pre-converted to `(Vec<f64>, Vec<f64>)` batches.
pub struct WorkerDataLoader {
    /// The worker this loader serves.
    pub worker_id: WorkerId,
    /// Pre-built batches of (inputs, labels).
    batches: Vec<(Vec<f64>, Vec<f64>)>,
}

impl DataLoader for WorkerDataLoader {
    fn load_batch(&self, batch_id: u64) -> HelixResult<(Vec<f64>, Vec<f64>)> {
        let idx = batch_id as usize;
        self.batches.get(idx).cloned().ok_or_else(|| {
            HelixError::Data(DataError::BatchNotFound(idx))
        })
    }

    fn num_batches(&self) -> u64 {
        self.batches.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataset::create_synthetic;

    #[test]
    fn test_round_robin_sharding() {
        let (_, samples) = create_synthetic(100, 10, 5);

        let sharder = DataSharder::new(ShardingConfig {
            num_shards: 4,
            strategy: ShardingStrategy::RoundRobin,
            ..Default::default()
        });

        let shards = sharder.shard(&samples);

        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].len(), 25);
        assert_eq!(shards[1].len(), 25);
    }

    #[test]
    fn test_random_sharding() {
        let (_, samples) = create_synthetic(100, 10, 5);

        let sharder = DataSharder::new(ShardingConfig {
            num_shards: 5,
            strategy: ShardingStrategy::Random { seed: 42 },
            ..Default::default()
        });

        let shards = sharder.shard(&samples);

        assert_eq!(shards.len(), 5);

        // All samples should be assigned
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_shard_assignment() {
        let mut assignment = ShardAssignment::new(4);

        assignment.assign("node1".to_string(), ShardId(0));
        assignment.assign("node1".to_string(), ShardId(1));
        assignment.assign("node2".to_string(), ShardId(2));

        assert_eq!(assignment.get_shards("node1").len(), 2);
        assert_eq!(assignment.get_shards("node2").len(), 1);
        assert_eq!(assignment.num_nodes(), 2);
    }

    // ========== Worker Tests ==========

    #[test]
    fn test_worker_info() {
        let worker = WorkerInfo::new(WorkerId::new("worker-1"));
        assert!(worker.is_healthy());
        assert_eq!(worker.status, WorkerStatus::Active);
    }

    #[test]
    fn test_worker_with_gpu() {
        let worker = WorkerInfo::with_gpu(
            WorkerId::new("gpu-worker"),
            16 * 1024 * 1024 * 1024 // 16 GB
        );
        assert!(worker.has_gpu);
        assert_eq!(worker.gpu_memory_bytes, Some(16 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_worker_heartbeat() {
        let mut worker = WorkerInfo::new(WorkerId::new("worker-1"));
        let initial = worker.last_heartbeat;
        std::thread::sleep(std::time::Duration::from_millis(10));
        worker.heartbeat();
        assert!(worker.last_heartbeat >= initial);
    }

    // ========== Registry Tests ==========

    #[test]
    fn test_shard_registry_basic() {
        let mut registry = ShardRegistry::default_registry();

        // Register workers
        registry.register_worker(WorkerInfo::new(WorkerId::new("w1")));
        registry.register_worker(WorkerInfo::new(WorkerId::new("w2")));

        assert_eq!(registry.list_workers().len(), 2);
        assert_eq!(registry.list_active_workers().len(), 2);
    }

    #[test]
    fn test_shard_registration() {
        let mut registry = ShardRegistry::default_registry();

        let metadata = ShardMetadata::new(ShardId(0), 1000, 1024 * 1024);
        registry.register_shard(metadata);

        assert!(registry.get_shard(ShardId(0)).is_some());
        assert_eq!(registry.list_shards().len(), 1);
    }

    #[test]
    fn test_shard_assignment_in_registry() {
        let mut registry = ShardRegistry::default_registry();

        registry.register_worker(WorkerInfo::new(WorkerId::new("w1")));
        registry.register_shard(ShardMetadata::new(ShardId(0), 100, 1024));

        assert!(registry.assign_shard(ShardId(0), WorkerId::new("w1")));

        let workers = registry.get_shard_workers(ShardId(0));
        assert_eq!(workers.len(), 1);

        let shards = registry.get_worker_shards(&WorkerId::new("w1"));
        assert_eq!(shards.len(), 1);
    }

    #[test]
    fn test_auto_assign_shards() {
        let mut registry = ShardRegistry::default_registry();

        // Register workers
        for i in 0..3 {
            registry.register_worker(WorkerInfo::new(WorkerId::new(format!("w{}", i))));
        }

        // Register unassigned shards
        for i in 0..9 {
            registry.register_shard(ShardMetadata::new(ShardId(i), 100, 1024));
        }

        let assignments = registry.auto_assign_shards();
        assert_eq!(assignments.len(), 9);

        // Each worker should have ~3 shards
        for i in 0..3 {
            let shards = registry.get_worker_shards(&WorkerId::new(format!("w{}", i)));
            assert!(shards.len() >= 2 && shards.len() <= 4);
        }
    }

    #[test]
    fn test_remove_worker() {
        let mut registry = ShardRegistry::default_registry();

        registry.register_worker(WorkerInfo::new(WorkerId::new("w1")));
        registry.register_shard(ShardMetadata::new(ShardId(0), 100, 1024));
        registry.assign_shard(ShardId(0), WorkerId::new("w1"));

        registry.remove_worker(&WorkerId::new("w1"));

        assert!(registry.get_worker(&WorkerId::new("w1")).is_none());
        // Shard should be unassigned
        let workers = registry.get_shard_workers(ShardId(0));
        assert!(workers.is_empty());
    }

    // ========== Progress Tracking Tests ==========

    #[test]
    fn test_progress_tracking() {
        let mut registry = ShardRegistry::default_registry();

        registry.register_shard(ShardMetadata::new(ShardId(0), 1000, 1024));
        registry.init_progress(ShardId(0), 1000, 10);

        let progress = registry.get_progress(ShardId(0)).unwrap();
        assert_eq!(progress.percentage(), 0.0);
        assert!(!progress.is_complete());
    }

    #[test]
    fn test_progress_update() {
        let mut registry = ShardRegistry::default_registry();

        registry.register_shard(ShardMetadata::new(ShardId(0), 100, 1024));
        registry.init_progress(ShardId(0), 100, 2);

        // Complete first epoch (epoch 1 done, starting epoch 2 with 0 samples)
        registry.update_progress(ShardId(0), 0, 1, None);
        let progress = registry.get_progress(ShardId(0)).unwrap();
        assert_eq!(progress.percentage(), 50.0);

        // Complete second epoch
        registry.update_progress(ShardId(0), 100, 2, None);
        let progress = registry.get_progress(ShardId(0)).unwrap();
        assert!(progress.is_complete());
    }

    #[test]
    fn test_overall_progress() {
        let mut registry = ShardRegistry::default_registry();

        for i in 0..4 {
            registry.register_shard(ShardMetadata::new(ShardId(i), 100, 1024));
            registry.init_progress(ShardId(i), 100, 1);
        }

        // Complete two shards
        registry.update_progress(ShardId(0), 100, 1, None);
        registry.update_progress(ShardId(1), 100, 1, None);

        let overall = registry.overall_progress();
        assert_eq!(overall.total_shards, 4);
        assert_eq!(overall.completed_shards, 2);
        assert_eq!(overall.average_percentage, 50.0);
    }

    // ========== Locality Tests ==========

    #[test]
    fn test_locality_aware_assigner() {
        let mut assigner = LocalityAwareAssigner::new();

        assigner.register_zone_worker("zone-a", WorkerId::new("w1"));
        assigner.register_zone_worker("zone-a", WorkerId::new("w2"));
        assigner.register_zone_worker("zone-b", WorkerId::new("w3"));

        assigner.set_shard_zone(ShardId(0), "zone-a");
        assigner.set_shard_zone(ShardId(1), "zone-b");

        let preferred = assigner.get_preferred_workers(ShardId(0));
        assert_eq!(preferred.len(), 2);
    }

    #[test]
    fn test_locality_assignment() {
        let mut registry = ShardRegistry::default_registry();
        let mut assigner = LocalityAwareAssigner::new();

        // Setup
        let mut w1 = WorkerInfo::new(WorkerId::new("w1"));
        w1.zone = Some("zone-a".to_string());
        registry.register_worker(w1);

        let mut w2 = WorkerInfo::new(WorkerId::new("w2"));
        w2.zone = Some("zone-b".to_string());
        registry.register_worker(w2);

        assigner.register_zone_worker("zone-a", WorkerId::new("w1"));
        assigner.register_zone_worker("zone-b", WorkerId::new("w2"));

        registry.register_shard(ShardMetadata::new(ShardId(0), 100, 1024));
        assigner.set_shard_zone(ShardId(0), "zone-a");

        // Assign with locality
        let assigned = assigner.assign_with_locality(&mut registry, ShardId(0));
        assert_eq!(assigned, Some(WorkerId::new("w1")));
    }

    // ========== Streaming Tests ==========

    #[test]
    fn test_shard_streaming() {
        let shard = DataShard::new(ShardId(0), (0..1000).collect());
        let streamer = ShardStreamer::new(ShardStreamConfig {
            chunk_size: 100,
            compress: false,
            checksum_chunks: true,
        });

        let chunks = streamer.create_chunks(&shard);

        assert_eq!(chunks.len(), 10);
        assert!(chunks[0].checksum.is_some());
        assert!(!chunks[0].is_last);
        assert!(chunks[9].is_last);
    }

    #[test]
    fn test_chunk_reassembly() {
        let original = DataShard::new(ShardId(0), (0..250).collect());
        let streamer = ShardStreamer::new(ShardStreamConfig {
            chunk_size: 100,
            ..Default::default()
        });

        let chunks = streamer.create_chunks(&original);
        let reassembled = streamer.reassemble_chunks(&chunks).unwrap();

        assert_eq!(reassembled.id, original.id);
        assert_eq!(reassembled.sample_indices, original.sample_indices);
    }

    #[test]
    fn test_chunk_reassembly_missing_chunk() {
        // Use 3000 samples with 1000 chunk_size = 3 chunks
        let original = DataShard::new(ShardId(0), (0..3000).collect());
        let streamer = ShardStreamer::default_streamer();

        let mut chunks = streamer.create_chunks(&original);
        assert!(chunks.len() >= 3, "Need at least 3 chunks for this test");
        chunks.remove(1); // Remove middle chunk

        let result = streamer.reassemble_chunks(&chunks);
        assert!(result.is_none());
    }

    // ========== Verification Tests ==========

    #[test]
    fn test_shard_verification() {
        let (_, samples) = create_synthetic(100, 10, 5);
        let mut registry = ShardRegistry::default_registry();

        let indices: Vec<usize> = (0..50).collect();
        let mut metadata = ShardMetadata::new(ShardId(0), 50, 1024);
        metadata.compute_hash(&samples, &indices);
        registry.register_shard(metadata);

        let verification = registry.verify_shard(ShardId(0), &samples, &indices);
        assert!(verification.valid);
        assert!(verification.missing_samples.is_empty());
    }

    #[test]
    fn test_shard_verification_mismatch() {
        let (_, samples) = create_synthetic(100, 10, 5);
        let mut registry = ShardRegistry::default_registry();

        // Create metadata with different indices
        let metadata_indices: Vec<usize> = (0..50).collect();
        let verify_indices: Vec<usize> = (50..100).collect();

        let mut metadata = ShardMetadata::new(ShardId(0), 50, 1024);
        metadata.compute_hash(&samples, &metadata_indices);
        registry.register_shard(metadata);

        let verification = registry.verify_shard(ShardId(0), &samples, &verify_indices);
        assert!(!verification.valid);
    }

    // ========== Rebalancing Tests ==========

    #[test]
    fn test_rebalancing() {
        let mut registry = ShardRegistry::default_registry();

        // Register workers
        for i in 0..3 {
            registry.register_worker(WorkerInfo::new(WorkerId::new(format!("w{}", i))));
        }

        // Assign all shards to one worker
        for i in 0..9 {
            registry.register_shard(ShardMetadata::new(ShardId(i), 100, 1024));
            registry.assign_shard(ShardId(i), WorkerId::new("w0"));
        }

        // Rebalance
        let moves = registry.rebalance();

        // Should have moved some shards
        assert!(!moves.is_empty());

        // Load should be more balanced now
        for i in 0..3 {
            let shards = registry.get_worker_shards(&WorkerId::new(format!("w{}", i)));
            assert!(shards.len() <= 5); // No worker should have more than ~5 shards
        }
    }

    // ========== Shard Stats Tests ==========

    #[test]
    fn test_compute_shard_stats() {
        let (_, samples) = create_synthetic(100, 10, 5);

        let mut shard = DataShard::new(ShardId(0), (0..50).collect());
        shard.compute_stats(&samples);

        assert_eq!(shard.stats.num_samples, 50);
        assert!(shard.stats.size_bytes > 0);
        assert!(!shard.stats.label_distribution.is_empty());
    }

    // ========== Worker Health Tests ==========

    #[test]
    fn test_worker_health_check() {
        let mut registry = ShardRegistry::new(ShardRegistryConfig {
            heartbeat_timeout_secs: 0, // Immediate timeout for testing
            ..Default::default()
        });

        let mut worker = WorkerInfo::new(WorkerId::new("w1"));
        worker.last_heartbeat = 0; // Very old heartbeat
        registry.register_worker(worker);

        let stale = registry.check_worker_health();
        assert_eq!(stale.len(), 1);

        let worker = registry.get_worker(&WorkerId::new("w1")).unwrap();
        assert_eq!(worker.status, WorkerStatus::Failed);
    }

    // ========== Shard Unassignment Tests ==========

    #[test]
    fn test_unassign_shard() {
        let mut registry = ShardRegistry::default_registry();

        registry.register_worker(WorkerInfo::new(WorkerId::new("w1")));
        registry.register_shard(ShardMetadata::new(ShardId(0), 100, 1024));
        registry.assign_shard(ShardId(0), WorkerId::new("w1"));

        registry.unassign_shard(ShardId(0), &WorkerId::new("w1"));

        assert!(registry.get_shard_workers(ShardId(0)).is_empty());
        assert!(registry.get_worker_shards(&WorkerId::new("w1")).is_empty());
    }

    // ========== Shard Removal Tests ==========

    #[test]
    fn test_remove_shard() {
        let mut registry = ShardRegistry::default_registry();

        registry.register_worker(WorkerInfo::new(WorkerId::new("w1")));
        registry.register_shard(ShardMetadata::new(ShardId(0), 100, 1024));
        registry.assign_shard(ShardId(0), WorkerId::new("w1"));

        let removed = registry.remove_shard(ShardId(0));
        assert!(removed.is_some());
        assert!(registry.get_shard(ShardId(0)).is_none());
        assert!(registry.get_worker_shards(&WorkerId::new("w1")).is_empty());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut registry = ShardRegistry::default_registry();
        registry.register_worker(WorkerInfo::new(WorkerId::new("w1")));
        registry.register_worker(WorkerInfo::new(WorkerId::new("w2")));
        registry.register_shard(ShardMetadata::new(ShardId(0), 100, 1024));
        registry.register_shard(ShardMetadata::new(ShardId(1), 200, 2048));
        registry.assign_shard(ShardId(0), WorkerId::new("w1"));
        registry.assign_shard(ShardId(1), WorkerId::new("w2"));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard_registry.json");

        registry.save_to_file(&path).unwrap();
        let loaded = ShardRegistry::load_from_file(&path).unwrap();

        assert!(loaded.get_shard(ShardId(0)).is_some());
        assert!(loaded.get_shard(ShardId(1)).is_some());
        assert_eq!(loaded.get_shard(ShardId(0)).unwrap().num_samples, 100);
        assert_eq!(loaded.get_shard(ShardId(1)).unwrap().num_samples, 200);
        assert!(loaded.get_worker(&WorkerId::new("w1")).is_some());
        assert!(loaded.get_worker(&WorkerId::new("w2")).is_some());
    }

    // ========== WorkerDataAssigner Tests ==========

    fn make_workers(n: usize) -> Vec<WorkerInfo> {
        (0..n)
            .map(|i| WorkerInfo::new(WorkerId::new(format!("worker-{}", i))))
            .collect()
    }

    #[test]
    fn test_worker_data_assigner_assign_and_verify_coverage() {
        let (_, samples) = create_synthetic(100, 10, 5);
        let workers = make_workers(3);

        let mut assigner = WorkerDataAssigner::new(
            ShardingConfig {
                num_shards: 6,
                strategy: ShardingStrategy::RoundRobin,
                ..Default::default()
            },
            ShardRegistryConfig::default(),
        );

        assigner.register_workers(workers);
        let plan = assigner.assign_dataset(&samples, 1);

        // Every worker should have at least one shard.
        assert_eq!(plan.worker_shards.len(), 3);
        for (_, shards) in &plan.worker_shards {
            assert!(!shards.is_empty(), "every worker must have at least one shard");
        }

        // Total assigned samples should equal the full dataset.
        assert_eq!(plan.total_assigned, 100);
        assert!(!plan.has_remainder);

        // Sum of per-worker sample counts must equal total.
        let sum: usize = plan.worker_sample_counts.values().sum();
        assert_eq!(sum, plan.total_assigned);
    }

    #[test]
    fn test_worker_data_assigner_each_worker_gets_data() {
        let (_, samples) = create_synthetic(60, 4, 2);
        let workers = make_workers(3);

        let mut assigner = WorkerDataAssigner::new(
            ShardingConfig {
                num_shards: 3,
                strategy: ShardingStrategy::RoundRobin,
                ..Default::default()
            },
            ShardRegistryConfig::default(),
        );

        assigner.register_workers(workers.clone());
        let plan = assigner.assign_dataset(&samples, 2);

        // Each worker should have a non-zero sample count.
        for worker in &workers {
            let count = plan.worker_sample_counts.get(&worker.id).copied().unwrap_or(0);
            assert!(count > 0, "worker {} must have samples", worker.id);
        }
    }

    #[test]
    fn test_worker_data_loader_returns_correct_data() {
        let (_, samples) = create_synthetic(20, 4, 2);
        let workers = make_workers(2);

        let mut assigner = WorkerDataAssigner::new(
            ShardingConfig {
                num_shards: 2,
                strategy: ShardingStrategy::RoundRobin,
                ..Default::default()
            },
            ShardRegistryConfig::default(),
        );

        assigner.register_workers(workers.clone());
        let _plan = assigner.assign_dataset(&samples, 1);

        // Build loaders for each worker.
        let features_per_sample = 4;
        let mut total_batches = 0u64;

        for worker in &workers {
            let loader = assigner
                .build_worker_loader(&worker.id, &samples, features_per_sample)
                .expect("loader should be Some for assigned worker");

            assert!(loader.num_batches() > 0, "loader must have batches");
            total_batches += loader.num_batches();

            // Verify each batch is loadable and has the right feature dimension.
            for batch_id in 0..loader.num_batches() {
                let (inputs, labels) = loader.load_batch(batch_id).unwrap();
                assert_eq!(inputs.len(), features_per_sample, "features length mismatch");
                assert!(!labels.is_empty(), "labels must not be empty");
            }

            // Out-of-range batch should error.
            assert!(loader.load_batch(loader.num_batches()).is_err());
        }

        // Combined batches should cover all samples.
        assert_eq!(total_batches, 20);
    }

    #[test]
    fn test_reassign_on_failure() {
        let (_, samples) = create_synthetic(30, 4, 2);
        let workers = make_workers(3);

        let mut assigner = WorkerDataAssigner::new(
            ShardingConfig {
                num_shards: 6,
                strategy: ShardingStrategy::RoundRobin,
                ..Default::default()
            },
            ShardRegistryConfig::default(),
        );

        assigner.register_workers(workers.clone());
        let plan_before = assigner.assign_dataset(&samples, 1);

        // Pick the first worker and record how many shards it had.
        let failed_id = &workers[0].id;
        let failed_shard_count = plan_before.worker_shards[failed_id].len();
        assert!(failed_shard_count > 0, "worker must have shards to fail");

        // Simulate failure and reassignment.
        let reassigned = assigner.reassign_on_failure(failed_id);

        // All orphaned shards should have been reassigned.
        assert_eq!(
            reassigned.len(),
            failed_shard_count,
            "all orphaned shards must be reassigned"
        );

        // None of the reassigned shards should go back to the failed worker.
        for (_, new_worker) in &reassigned {
            assert_ne!(new_worker, failed_id, "shard must not be reassigned to failed worker");
        }

        // The failed worker should now have no shards.
        let remaining = assigner.registry().get_worker_shards(failed_id);
        assert!(remaining.is_empty(), "failed worker must have 0 shards");
    }

    #[test]
    fn test_worker_data_loader_trait_impl() {
        // Verify WorkerDataLoader satisfies the DataLoader trait.
        let (_, samples) = create_synthetic(10, 3, 2);
        let workers = make_workers(1);

        let mut assigner = WorkerDataAssigner::new(
            ShardingConfig {
                num_shards: 1,
                strategy: ShardingStrategy::RoundRobin,
                ..Default::default()
            },
            ShardRegistryConfig::default(),
        );

        assigner.register_workers(workers.clone());
        assigner.assign_dataset(&samples, 1);

        let loader = assigner
            .build_worker_loader(&workers[0].id, &samples, 3)
            .unwrap();

        // Use the loader through the trait interface.
        let dl: &dyn DataLoader = &loader;
        assert_eq!(dl.num_batches(), 10);
        let (inputs, labels) = dl.load_batch(0).unwrap();
        assert_eq!(inputs.len(), 3);
        assert!(!labels.is_empty());
    }

    #[test]
    fn test_worker_data_assigner_registry_access() {
        let workers = make_workers(2);

        let mut assigner = WorkerDataAssigner::new(
            ShardingConfig::default(),
            ShardRegistryConfig::default(),
        );

        assigner.register_workers(workers.clone());

        // Immutable access.
        assert_eq!(assigner.registry().list_workers().len(), 2);

        // Mutable access.
        assigner
            .registry_mut()
            .set_worker_status(&workers[0].id, WorkerStatus::Paused);

        let w = assigner.registry().get_worker(&workers[0].id).unwrap();
        assert_eq!(w.status, WorkerStatus::Paused);
    }

    #[test]
    fn test_build_loader_for_unknown_worker_returns_none() {
        let (_, samples) = create_synthetic(10, 4, 2);
        let workers = make_workers(1);

        let mut assigner = WorkerDataAssigner::new(
            ShardingConfig {
                num_shards: 1,
                strategy: ShardingStrategy::RoundRobin,
                ..Default::default()
            },
            ShardRegistryConfig::default(),
        );

        assigner.register_workers(workers);
        assigner.assign_dataset(&samples, 1);

        // Worker that was never registered should return None.
        let unknown = WorkerId::new("unknown-worker");
        assert!(assigner.build_worker_loader(&unknown, &samples, 4).is_none());
    }
}
