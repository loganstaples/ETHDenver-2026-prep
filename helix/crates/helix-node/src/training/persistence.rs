//! State Persistence for Crash Recovery.
//!
//! Provides serializable snapshots for aggregator and worker nodes, enabling
//! recovery from crashes without losing training progress.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Per-worker state stored in the aggregator snapshot for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistryEntry {
    /// Worker's peer ID string.
    pub peer_id: String,
    /// Worker health status string (Healthy, Degraded, Failed, etc.).
    pub health_status: String,
    /// Number of rounds this worker has participated in.
    pub rounds_participated: u64,
    /// Number of proofs submitted by this worker.
    pub proofs_submitted: u64,
    /// Last heartbeat timestamp (epoch seconds).
    pub last_heartbeat_ts: u64,
    /// Whether the worker is currently excluded.
    pub excluded: bool,
    /// Number of failures since joining.
    pub failure_count: u32,
}

/// Active round state stored in the aggregator snapshot for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveRoundState {
    /// Round ID.
    pub round_id: u64,
    /// Round phase (e.g., "Collecting", "Aggregating").
    pub phase: String,
    /// Workers assigned to this round.
    pub assigned_workers: Vec<String>,
    /// Workers that have submitted proofs.
    pub submitted_workers: Vec<String>,
    /// Collected proof hashes for this round.
    pub collected_proof_hashes: Vec<[u8; 32]>,
    /// Round start timestamp (epoch seconds).
    pub started_at: u64,
    /// Error bounds collected so far in this round: worker_id -> error_bound.
    pub worker_error_bounds: HashMap<String, f64>,
}

/// Aggregator state snapshot — everything needed to resume after a crash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorSnapshot {
    /// Serialized model checkpoint bytes (helix-core ModelCheckpoint format).
    pub model_checkpoint_bytes: Vec<u8>,
    /// Last completed round number.
    pub last_completed_round: u64,
    /// Total rounds completed so far.
    pub total_rounds_completed: u64,
    /// Known participant peer IDs from the last round.
    pub participants: Vec<String>,
    /// Accumulated error bounds per round: round_id -> error_bound.
    pub error_bounds: HashMap<u64, f64>,
    /// Proof hashes from submitted proofs (not full proofs): round_id -> [hash].
    pub proof_hashes: HashMap<u64, Vec<[u8; 32]>>,
    /// On-chain state: last submitted proof hash (if any).
    pub last_on_chain_proof_hash: Option<[u8; 32]>,
    /// On-chain model ID.
    pub on_chain_model_id: Option<u64>,
    /// Model dimensions.
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub model_seed: u64,
    pub learning_rate: f64,
    /// Timestamp of this snapshot.
    pub timestamp: u64,
    /// Snapshot version for forward compatibility.
    pub version: u32,
    /// URI where current model weights are stored (e.g. "file://..." or "ipfs://...").
    #[serde(default)]
    pub current_weight_uri: Option<String>,
    /// Completed round results: (round_number, final_commitment, weights_uri, loss, error).
    #[serde(default)]
    pub completed_round_results: Vec<RoundResultEntry>,
    /// Current model weight commitment (SHA-256).
    #[serde(default)]
    pub current_weight_commitment: Option<[u8; 32]>,
    /// Active round state (if a round was in progress when snapshot was taken).
    #[serde(default)]
    pub active_round: Option<ActiveRoundState>,
    /// Worker registry — per-worker health and participation state.
    #[serde(default)]
    pub worker_registry: Vec<WorkerRegistryEntry>,
    /// MPC session state (session ID if active).
    #[serde(default)]
    pub mpc_session_id: Option<String>,
}

/// Compact record of a completed round for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResultEntry {
    /// Round number.
    pub round_number: u64,
    /// Model commitment after this round.
    pub final_commitment: [u8; 32],
    /// URI where weights are stored.
    pub weights_uri: String,
    /// Final loss value.
    pub final_loss: f64,
    /// Accumulated error bound.
    pub total_error: f64,
    /// Number of training steps.
    pub steps_completed: u64,
    /// Number of workers that contributed.
    pub num_contributors: u32,
    /// On-chain tx hash (if submitted).
    pub tx_hash: Option<[u8; 32]>,
    /// Completion timestamp.
    pub completed_at: u64,
}

impl AggregatorSnapshot {
    /// Creates a new snapshot with the current timestamp.
    pub fn new(
        model_checkpoint_bytes: Vec<u8>,
        last_completed_round: u64,
        total_rounds_completed: u64,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        model_seed: u64,
        learning_rate: f64,
    ) -> Self {
        Self {
            model_checkpoint_bytes,
            last_completed_round,
            total_rounds_completed,
            participants: Vec::new(),
            error_bounds: HashMap::new(),
            proof_hashes: HashMap::new(),
            last_on_chain_proof_hash: None,
            on_chain_model_id: None,
            d_in,
            d_hid,
            d_out,
            model_seed,
            learning_rate,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            version: 1,
            current_weight_uri: None,
            completed_round_results: Vec::new(),
            current_weight_commitment: None,
            active_round: None,
            worker_registry: Vec::new(),
            mpc_session_id: None,
        }
    }

    /// Serializes to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PersistenceError> {
        // Use bincode for the outer envelope, but the model_checkpoint_bytes
        // is already in helix-core binary format.
        bincode::serialize(self).map_err(|e| PersistenceError::SerializeFailed(e.to_string()))
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, PersistenceError> {
        bincode::deserialize(data).map_err(|e| PersistenceError::DeserializeFailed(e.to_string()))
    }
}

/// Worker state snapshot — local model and training progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSnapshot {
    /// Serialized model checkpoint bytes.
    pub model_checkpoint_bytes: Option<Vec<u8>>,
    /// Last completed round number.
    pub last_completed_round: u64,
    /// Total training steps completed.
    pub steps_completed: u64,
    /// Worker's peer ID string.
    pub peer_id: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Snapshot version.
    pub version: u32,
    /// Optimizer state bytes (Adam moments, etc.) for resume.
    #[serde(default)]
    pub optimizer_state: Option<Vec<u8>>,
    /// Last verified on-chain commitment for this worker.
    #[serde(default)]
    pub last_verified_commitment: Option<[u8; 32]>,
    /// URI of the weights this worker was trained on.
    #[serde(default)]
    pub weights_uri: Option<String>,
    /// Accumulated error from training.
    #[serde(default)]
    pub accumulated_error: f64,
    /// Current round ID this worker is participating in (if any).
    #[serde(default)]
    pub current_round_id: Option<u64>,
    /// MPC party index (if MPC is enabled).
    #[serde(default)]
    pub mpc_party_index: Option<usize>,
    /// MPC session ID (if in an active MPC session).
    #[serde(default)]
    pub mpc_session_id: Option<String>,
}

impl WorkerSnapshot {
    pub fn new(peer_id: String, last_completed_round: u64, steps_completed: u64) -> Self {
        Self {
            model_checkpoint_bytes: None,
            last_completed_round,
            steps_completed,
            peer_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            version: 1,
            optimizer_state: None,
            last_verified_commitment: None,
            weights_uri: None,
            accumulated_error: 0.0,
            current_round_id: None,
            mpc_party_index: None,
            mpc_session_id: None,
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, PersistenceError> {
        bincode::serialize(self).map_err(|e| PersistenceError::SerializeFailed(e.to_string()))
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, PersistenceError> {
        bincode::deserialize(data).map_err(|e| PersistenceError::DeserializeFailed(e.to_string()))
    }
}

/// Manages checkpoint persistence on disk with automatic pruning.
pub struct StatePersistence {
    /// Base directory for checkpoints.
    checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep.
    max_checkpoints: usize,
    /// Prefix for checkpoint files (e.g., "aggregator" or "worker").
    prefix: String,
}

impl StatePersistence {
    /// Creates a new persistence manager.
    ///
    /// `checkpoint_dir` defaults to `~/.helix/checkpoints/` if not specified.
    /// `max_checkpoints` defaults to 5.
    pub fn new(
        checkpoint_dir: Option<PathBuf>,
        max_checkpoints: usize,
        prefix: &str,
    ) -> Result<Self, PersistenceError> {
        let dir = checkpoint_dir.unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".helix").join("checkpoints")
        });

        std::fs::create_dir_all(&dir)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(Self {
            checkpoint_dir: dir,
            max_checkpoints,
            prefix: prefix.to_string(),
        })
    }

    /// Saves an aggregator snapshot, returning the file path.
    pub fn save_aggregator(&self, snapshot: &AggregatorSnapshot) -> Result<PathBuf, PersistenceError> {
        let bytes = snapshot.to_bytes()?;
        let filename = format!(
            "{}_round_{:06}_{}.bin",
            self.prefix,
            snapshot.last_completed_round,
            snapshot.timestamp,
        );
        let path = self.checkpoint_dir.join(&filename);

        std::fs::write(&path, &bytes)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        // Also write a "latest" symlink/copy for quick recovery
        let latest_path = self.checkpoint_dir.join(format!("{}_latest.bin", self.prefix));
        std::fs::write(&latest_path, &bytes)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        // Prune old checkpoints
        self.prune()?;

        tracing::info!(
            "Aggregator snapshot saved: round={}, {} bytes -> {}",
            snapshot.last_completed_round,
            bytes.len(),
            path.display(),
        );

        Ok(path)
    }

    /// Saves a worker snapshot, returning the file path.
    pub fn save_worker(&self, snapshot: &WorkerSnapshot) -> Result<PathBuf, PersistenceError> {
        let bytes = snapshot.to_bytes()?;
        let filename = format!(
            "{}_step_{:06}_{}.bin",
            self.prefix,
            snapshot.steps_completed,
            snapshot.timestamp,
        );
        let path = self.checkpoint_dir.join(&filename);

        std::fs::write(&path, &bytes)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let latest_path = self.checkpoint_dir.join(format!("{}_latest.bin", self.prefix));
        std::fs::write(&latest_path, &bytes)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        self.prune()?;

        tracing::info!(
            "Worker snapshot saved: step={}, {} bytes -> {}",
            snapshot.steps_completed,
            bytes.len(),
            path.display(),
        );

        Ok(path)
    }

    /// Loads the latest aggregator snapshot, if one exists.
    pub fn load_latest_aggregator(&self) -> Result<Option<AggregatorSnapshot>, PersistenceError> {
        let latest_path = self.checkpoint_dir.join(format!("{}_latest.bin", self.prefix));
        if !latest_path.exists() {
            return Ok(None);
        }

        let bytes = std::fs::read(&latest_path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        if bytes.is_empty() {
            return Ok(None);
        }

        let snapshot = AggregatorSnapshot::from_bytes(&bytes)?;
        tracing::info!(
            "Loaded aggregator snapshot: round={}, timestamp={}",
            snapshot.last_completed_round,
            snapshot.timestamp,
        );
        Ok(Some(snapshot))
    }

    /// Loads the latest worker snapshot, if one exists.
    pub fn load_latest_worker(&self) -> Result<Option<WorkerSnapshot>, PersistenceError> {
        let latest_path = self.checkpoint_dir.join(format!("{}_latest.bin", self.prefix));
        if !latest_path.exists() {
            return Ok(None);
        }

        let bytes = std::fs::read(&latest_path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        if bytes.is_empty() {
            return Ok(None);
        }

        let snapshot = WorkerSnapshot::from_bytes(&bytes)?;
        tracing::info!(
            "Loaded worker snapshot: step={}, peer={}",
            snapshot.steps_completed,
            snapshot.peer_id,
        );
        Ok(Some(snapshot))
    }

    /// Prunes old checkpoint files, keeping only `max_checkpoints` most recent.
    fn prune(&self) -> Result<(), PersistenceError> {
        let pattern = format!("{}_round_", self.prefix);
        let worker_pattern = format!("{}_step_", self.prefix);

        let mut checkpoint_files: Vec<(PathBuf, u64)> = Vec::new();

        let entries = std::fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        for entry in entries {
            let entry = entry.map_err(|e| PersistenceError::IoError(e.to_string()))?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if (name.starts_with(&pattern) || name.starts_with(&worker_pattern))
                    && name.ends_with(".bin")
                    && !name.ends_with("_latest.bin")
                {
                    // Extract timestamp from filename (last segment before .bin)
                    let ts = name
                        .trim_end_matches(".bin")
                        .rsplit('_')
                        .next()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                    checkpoint_files.push((path, ts));
                }
            }
        }

        // Sort by timestamp ascending (oldest first)
        checkpoint_files.sort_by_key(|(_, ts)| *ts);

        // Remove oldest files beyond max_checkpoints
        while checkpoint_files.len() > self.max_checkpoints {
            let (path, _) = checkpoint_files.remove(0);
            if let Err(e) = std::fs::remove_file(&path) {
                tracing::warn!("Failed to prune checkpoint {}: {}", path.display(), e);
            } else {
                tracing::debug!("Pruned old checkpoint: {}", path.display());
            }
        }

        Ok(())
    }

    /// Returns the checkpoint directory path.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Lists all checkpoint files for this prefix, sorted by timestamp.
    pub fn list_checkpoints(&self) -> Result<Vec<PathBuf>, PersistenceError> {
        let pattern = format!("{}_round_", self.prefix);
        let worker_pattern = format!("{}_step_", self.prefix);
        let mut files: Vec<(PathBuf, u64)> = Vec::new();

        let entries = std::fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        for entry in entries {
            let entry = entry.map_err(|e| PersistenceError::IoError(e.to_string()))?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if (name.starts_with(&pattern) || name.starts_with(&worker_pattern))
                    && name.ends_with(".bin")
                    && !name.ends_with("_latest.bin")
                {
                    let ts = name
                        .trim_end_matches(".bin")
                        .rsplit('_')
                        .next()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                    files.push((path, ts));
                }
            }
        }

        files.sort_by_key(|(_, ts)| *ts);
        Ok(files.into_iter().map(|(p, _)| p).collect())
    }
}

/// Persistence errors.
#[derive(Debug)]
pub enum PersistenceError {
    SerializeFailed(String),
    DeserializeFailed(String),
    IoError(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializeFailed(msg) => write!(f, "Serialization failed: {}", msg),
            Self::DeserializeFailed(msg) => write!(f, "Deserialization failed: {}", msg),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for PersistenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_aggregator_snapshot_roundtrip() {
        let snapshot = AggregatorSnapshot::new(
            vec![1, 2, 3, 4],
            5,
            5,
            4, 8, 2,
            42,
            0.01,
        );

        let bytes = snapshot.to_bytes().unwrap();
        let restored = AggregatorSnapshot::from_bytes(&bytes).unwrap();

        assert_eq!(restored.last_completed_round, 5);
        assert_eq!(restored.total_rounds_completed, 5);
        assert_eq!(restored.model_checkpoint_bytes, vec![1, 2, 3, 4]);
        assert_eq!(restored.d_in, 4);
        assert_eq!(restored.d_hid, 8);
        assert_eq!(restored.d_out, 2);
        assert_eq!(restored.model_seed, 42);
        assert_eq!(restored.version, 1);
    }

    #[test]
    fn test_worker_snapshot_roundtrip() {
        let mut snapshot = WorkerSnapshot::new("worker-1".to_string(), 3, 10);
        snapshot.model_checkpoint_bytes = Some(vec![5, 6, 7]);

        let bytes = snapshot.to_bytes().unwrap();
        let restored = WorkerSnapshot::from_bytes(&bytes).unwrap();

        assert_eq!(restored.last_completed_round, 3);
        assert_eq!(restored.steps_completed, 10);
        assert_eq!(restored.peer_id, "worker-1");
        assert_eq!(restored.model_checkpoint_bytes, Some(vec![5, 6, 7]));
    }

    #[test]
    fn test_aggregator_snapshot_with_proof_hashes() {
        let mut snapshot = AggregatorSnapshot::new(
            vec![1, 2, 3],
            10,
            10,
            4, 8, 2,
            42,
            0.01,
        );
        snapshot.participants = vec!["w1".to_string(), "w2".to_string()];
        snapshot.error_bounds.insert(1, 0.05);
        snapshot.error_bounds.insert(2, 0.03);
        snapshot.proof_hashes.insert(1, vec![[0xAA; 32]]);
        snapshot.last_on_chain_proof_hash = Some([0xBB; 32]);

        let bytes = snapshot.to_bytes().unwrap();
        let restored = AggregatorSnapshot::from_bytes(&bytes).unwrap();

        assert_eq!(restored.participants.len(), 2);
        assert_eq!(restored.error_bounds.len(), 2);
        assert_eq!(restored.proof_hashes[&1].len(), 1);
        assert_eq!(restored.last_on_chain_proof_hash, Some([0xBB; 32]));
    }

    #[test]
    fn test_state_persistence_save_load_aggregator() {
        let dir = tempdir().unwrap();
        let persistence =
            StatePersistence::new(Some(dir.path().to_path_buf()), 5, "aggregator").unwrap();

        let snapshot = AggregatorSnapshot::new(
            vec![10, 20, 30],
            3,
            3,
            4, 8, 2,
            42,
            0.01,
        );

        let path = persistence.save_aggregator(&snapshot).unwrap();
        assert!(path.exists());

        let loaded = persistence.load_latest_aggregator().unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.last_completed_round, 3);
        assert_eq!(loaded.model_checkpoint_bytes, vec![10, 20, 30]);
    }

    #[test]
    fn test_state_persistence_save_load_worker() {
        let dir = tempdir().unwrap();
        let persistence =
            StatePersistence::new(Some(dir.path().to_path_buf()), 5, "worker").unwrap();

        let mut snapshot = WorkerSnapshot::new("w1".to_string(), 2, 8);
        snapshot.model_checkpoint_bytes = Some(vec![99]);

        let path = persistence.save_worker(&snapshot).unwrap();
        assert!(path.exists());

        let loaded = persistence.load_latest_worker().unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.steps_completed, 8);
        assert_eq!(loaded.peer_id, "w1");
    }

    #[test]
    fn test_state_persistence_pruning() {
        let dir = tempdir().unwrap();
        let persistence =
            StatePersistence::new(Some(dir.path().to_path_buf()), 3, "aggregator").unwrap();

        // Save 5 snapshots (only 3 should remain after pruning)
        for i in 1..=5 {
            let mut snapshot = AggregatorSnapshot::new(
                vec![i as u8],
                i,
                i,
                4, 8, 2,
                42,
                0.01,
            );
            // Override timestamp to ensure unique filenames
            snapshot.timestamp += i;
            persistence.save_aggregator(&snapshot).unwrap();
        }

        let files = persistence.list_checkpoints().unwrap();
        assert_eq!(files.len(), 3, "Should have pruned to 3 checkpoints, got {}", files.len());

        // Latest should still be round 5
        let latest = persistence.load_latest_aggregator().unwrap().unwrap();
        assert_eq!(latest.last_completed_round, 5);
    }

    #[test]
    fn test_state_persistence_no_existing_checkpoint() {
        let dir = tempdir().unwrap();
        let persistence =
            StatePersistence::new(Some(dir.path().to_path_buf()), 5, "aggregator").unwrap();

        let loaded = persistence.load_latest_aggregator().unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_state_persistence_default_dir() {
        // Just verify it doesn't panic with None dir
        let persistence = StatePersistence::new(None, 5, "test_default");
        assert!(persistence.is_ok());
    }
}
