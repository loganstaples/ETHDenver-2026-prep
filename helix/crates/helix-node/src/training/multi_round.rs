//! Multi-Round Training Controller.
//!
//! Drives consecutive training rounds where each round builds on the previous
//! round's final weights. Handles:
//!
//! - Weight persistence after each round (local filesystem or IPFS)
//! - Round transition with commitment chaining
//! - Worker weight download and verification
//! - Crash recovery from persisted state
//! - On-chain round finalization
//!
//! # Architecture
//!
//! ```text
//! Round N completes
//!   ├─ Collect final weights from best worker / aggregate
//!   ├─ Serialize via ModelCheckpoint
//!   ├─ Store to WeightStore (local / IPFS)
//!   ├─ Record RoundResult (commitment, URI, loss, error)
//!   ├─ Persist AggregatorSnapshot for crash recovery
//!   └─ Start Round N+1
//!       ├─ initial_commitment = Round N's final_commitment
//!       ├─ Workers download weights from stored URI
//!       ├─ Workers verify weights match commitment
//!       └─ Training begins
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::persistence::{AggregatorSnapshot, StatePersistence, PersistenceError};

// ============================================================================
// Round Result
// ============================================================================

/// Result of a completed training round — everything needed to start the next.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResult {
    /// Round number (0-indexed).
    pub round_number: u64,
    /// Model commitment at round start.
    pub initial_commitment: [u8; 32],
    /// Model commitment after training + aggregation.
    pub final_commitment: [u8; 32],
    /// URI where the final weights are stored (e.g. "file:///path" or "ipfs://Qm...").
    pub weights_uri: String,
    /// Final loss value.
    pub final_loss: f64,
    /// Accumulated error bound.
    pub total_error: f64,
    /// Number of training steps completed.
    pub steps_completed: u64,
    /// Number of workers that contributed.
    pub num_contributors: u32,
    /// On-chain transaction hash (if submitted).
    pub tx_hash: Option<[u8; 32]>,
    /// Unix timestamp of round completion.
    pub completed_at: u64,
    /// Worker reputations at round end: peer_id -> score.
    pub worker_scores: HashMap<String, f64>,
}

// ============================================================================
// Weight Store Trait
// ============================================================================

/// Abstraction over weight storage backends.
///
/// Implementations must guarantee that stored weights can be retrieved by URI
/// and that the SHA-256 hash of stored bytes matches the returned commitment.
pub trait WeightStore: Send + Sync {
    /// Stores serialized model weights. Returns (URI, commitment_hash).
    ///
    /// The commitment is SHA-256(weight_bytes), which must match the on-chain
    /// model commitment for the round.
    fn store_weights(
        &self,
        round: u64,
        weight_bytes: &[u8],
    ) -> Result<(String, [u8; 32]), WeightStoreError>;

    /// Loads serialized model weights by URI.
    fn load_weights(&self, uri: &str) -> Result<Vec<u8>, WeightStoreError>;

    /// Verifies that stored weights match the expected commitment.
    fn verify_commitment(
        &self,
        uri: &str,
        expected_commitment: [u8; 32],
    ) -> Result<bool, WeightStoreError> {
        let bytes = self.load_weights(uri)?;
        let actual = compute_weight_commitment(&bytes);
        Ok(actual == expected_commitment)
    }
}

/// Errors from weight storage operations.
#[derive(Debug)]
pub enum WeightStoreError {
    /// IO error.
    Io(std::io::Error),
    /// Weight data not found at URI.
    NotFound(String),
    /// Commitment mismatch.
    CommitmentMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// Storage backend error.
    Backend(String),
}

impl std::fmt::Display for WeightStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Weight store IO error: {}", e),
            Self::NotFound(uri) => write!(f, "Weights not found: {}", uri),
            Self::CommitmentMismatch { expected, actual } => {
                write!(
                    f,
                    "Commitment mismatch: expected {}, got {}",
                    hex::encode(&expected[..8]),
                    hex::encode(&actual[..8]),
                )
            }
            Self::Backend(msg) => write!(f, "Weight store error: {}", msg),
        }
    }
}

impl std::error::Error for WeightStoreError {}

impl From<std::io::Error> for WeightStoreError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================================
// Local Weight Store
// ============================================================================

/// Stores weights on the local filesystem with SHA-256 integrity checks.
///
/// Directory layout:
/// ```text
/// {base_dir}/
///   round_000000.weights     — serialized model weights
///   round_000000.sha256      — hex-encoded SHA-256 checksum
///   round_000001.weights
///   round_000001.sha256
///   ...
/// ```
pub struct LocalWeightStore {
    /// Base directory for weight files.
    base_dir: PathBuf,
    /// Maximum weight files to retain (0 = unlimited).
    max_retained: usize,
}

impl LocalWeightStore {
    /// Creates a new local weight store, creating the directory if needed.
    pub fn new(base_dir: impl Into<PathBuf>, max_retained: usize) -> Result<Self, WeightStoreError> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir, max_retained })
    }

    /// Returns the file path for a given round's weights.
    fn weight_path(&self, round: u64) -> PathBuf {
        self.base_dir.join(format!("round_{:06}.weights", round))
    }

    /// Returns the checksum path for a given round.
    fn checksum_path(&self, round: u64) -> PathBuf {
        self.base_dir.join(format!("round_{:06}.sha256", round))
    }

    /// Extracts the round number from a weight file URI.
    fn round_from_uri(uri: &str) -> Option<u64> {
        // URI format: "file:///path/to/round_000042.weights"
        let filename = uri.rsplit('/').next()?;
        let round_str = filename.strip_prefix("round_")?.strip_suffix(".weights")?;
        round_str.parse().ok()
    }

    /// Garbage-collects old weight files.
    fn gc(&self) -> Result<(), WeightStoreError> {
        if self.max_retained == 0 {
            return Ok(());
        }

        let mut rounds: Vec<u64> = Vec::new();
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(round_str) = name.strip_prefix("round_") {
                if let Some(round_str) = round_str.strip_suffix(".weights") {
                    if let Ok(round) = round_str.parse::<u64>() {
                        rounds.push(round);
                    }
                }
            }
        }

        if rounds.len() <= self.max_retained {
            return Ok(());
        }

        rounds.sort();
        let to_remove = rounds.len() - self.max_retained;
        for round in rounds.into_iter().take(to_remove) {
            let _ = std::fs::remove_file(self.weight_path(round));
            let _ = std::fs::remove_file(self.checksum_path(round));
        }

        Ok(())
    }
}

impl WeightStore for LocalWeightStore {
    fn store_weights(
        &self,
        round: u64,
        weight_bytes: &[u8],
    ) -> Result<(String, [u8; 32]), WeightStoreError> {
        let commitment = compute_weight_commitment(weight_bytes);

        let path = self.weight_path(round);
        std::fs::write(&path, weight_bytes)?;

        // Write checksum sidecar
        let checksum_path = self.checksum_path(round);
        std::fs::write(&checksum_path, hex::encode(commitment))?;

        // GC old files
        self.gc()?;

        let uri = format!("file://{}", path.display());
        log::info!(
            "Stored weights for round {}: {} bytes, commitment={}",
            round,
            weight_bytes.len(),
            hex::encode(&commitment[..8]),
        );

        Ok((uri, commitment))
    }

    fn load_weights(&self, uri: &str) -> Result<Vec<u8>, WeightStoreError> {
        let path = if let Some(file_path) = uri.strip_prefix("file://") {
            PathBuf::from(file_path)
        } else if let Some(round) = Self::round_from_uri(uri) {
            self.weight_path(round)
        } else {
            return Err(WeightStoreError::NotFound(uri.to_string()));
        };

        if !path.exists() {
            return Err(WeightStoreError::NotFound(uri.to_string()));
        }

        let bytes = std::fs::read(&path)?;

        // Verify checksum if available
        let checksum_path = path.with_extension("sha256");
        if checksum_path.exists() {
            let expected_hex = std::fs::read_to_string(&checksum_path)?;
            let actual_hex = hex::encode(compute_weight_commitment(&bytes));
            if expected_hex.trim() != actual_hex {
                let mut expected = [0u8; 32];
                let mut actual = [0u8; 32];
                if let Ok(e) = hex::decode(expected_hex.trim()) {
                    expected[..e.len().min(32)].copy_from_slice(&e[..e.len().min(32)]);
                }
                if let Ok(a) = hex::decode(&actual_hex) {
                    actual[..a.len().min(32)].copy_from_slice(&a[..a.len().min(32)]);
                }
                return Err(WeightStoreError::CommitmentMismatch { expected, actual });
            }
        }

        Ok(bytes)
    }
}

// ============================================================================
// Multi-Round Configuration
// ============================================================================

/// Configuration for multi-round training.
#[derive(Debug, Clone)]
pub struct MultiRoundConfig {
    /// Maximum number of rounds to run (0 = unlimited).
    pub max_rounds: u64,
    /// Directory for weight persistence.
    pub weight_store_dir: PathBuf,
    /// Maximum weight files to retain.
    pub max_weight_files: usize,
    /// Directory for aggregator snapshots.
    pub snapshot_dir: Option<PathBuf>,
    /// Maximum snapshots to retain.
    pub max_snapshots: usize,
    /// Whether to verify weight commitments before starting each round.
    pub verify_commitments: bool,
    /// Model dimensions for snapshot metadata.
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub model_seed: u64,
    pub learning_rate: f64,
}

impl Default for MultiRoundConfig {
    fn default() -> Self {
        Self {
            max_rounds: 0,
            weight_store_dir: PathBuf::from("./helix_weights"),
            max_weight_files: 10,
            snapshot_dir: None,
            max_snapshots: 5,
            verify_commitments: true,
            d_in: 2,
            d_hid: 4,
            d_out: 1,
            model_seed: 42,
            learning_rate: 0.01,
        }
    }
}

// ============================================================================
// Multi-Round Controller
// ============================================================================

/// Drives consecutive training rounds with weight persistence and commitment chaining.
///
/// After each round completes:
/// 1. Collects final weights from best/aggregated worker result
/// 2. Serializes and stores weights via `WeightStore`
/// 3. Records `RoundResult` with commitment, URI, loss, error
/// 4. Persists `AggregatorSnapshot` for crash recovery
/// 5. Uses final commitment as next round's initial commitment
pub struct MultiRoundController {
    /// Configuration.
    config: MultiRoundConfig,
    /// Weight storage backend.
    weight_store: Box<dyn WeightStore>,
    /// State persistence for crash recovery.
    persistence: StatePersistence,
    /// Completed round results (in order).
    round_history: Vec<RoundResult>,
    /// Current round number.
    current_round: u64,
    /// Current model commitment (initial commitment for next round).
    current_commitment: [u8; 32],
    /// Current model weights (serialized bytes).
    current_weights: Option<Vec<u8>>,
    /// On-chain model ID.
    model_id: Option<u64>,
}

impl MultiRoundController {
    /// Creates a new multi-round controller.
    pub fn new(
        config: MultiRoundConfig,
        initial_commitment: [u8; 32],
        initial_weights: Option<Vec<u8>>,
    ) -> Result<Self, MultiRoundError> {
        let weight_store = Box::new(LocalWeightStore::new(
            &config.weight_store_dir,
            config.max_weight_files,
        ).map_err(|e| MultiRoundError::WeightStore(e.to_string()))?);

        let persistence = StatePersistence::new(
            config.snapshot_dir.clone(),
            config.max_snapshots,
            "multi_round",
        ).map_err(|e| MultiRoundError::Persistence(e.to_string()))?;

        Ok(Self {
            config,
            weight_store,
            persistence,
            round_history: Vec::new(),
            current_round: 0,
            current_commitment: initial_commitment,
            current_weights: initial_weights,
            model_id: None,
        })
    }

    /// Creates a controller with a custom weight store backend.
    pub fn with_weight_store(
        config: MultiRoundConfig,
        initial_commitment: [u8; 32],
        initial_weights: Option<Vec<u8>>,
        weight_store: Box<dyn WeightStore>,
    ) -> Result<Self, MultiRoundError> {
        let persistence = StatePersistence::new(
            config.snapshot_dir.clone(),
            config.max_snapshots,
            "multi_round",
        ).map_err(|e| MultiRoundError::Persistence(e.to_string()))?;

        Ok(Self {
            config,
            weight_store,
            persistence,
            round_history: Vec::new(),
            current_round: 0,
            current_commitment: initial_commitment,
            current_weights: initial_weights,
            model_id: None,
        })
    }

    /// Sets the on-chain model ID for snapshot metadata.
    pub fn set_model_id(&mut self, model_id: u64) {
        self.model_id = Some(model_id);
    }

    /// Returns the current round number.
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Returns the current model commitment.
    pub fn current_commitment(&self) -> [u8; 32] {
        self.current_commitment
    }

    /// Returns the current model weights (if loaded).
    pub fn current_weights(&self) -> Option<&[u8]> {
        self.current_weights.as_deref()
    }

    /// Returns the history of completed rounds.
    pub fn round_history(&self) -> &[RoundResult] {
        &self.round_history
    }

    /// Returns the total number of completed rounds.
    pub fn completed_rounds(&self) -> u64 {
        self.round_history.len() as u64
    }

    /// Returns whether we've reached the maximum number of rounds.
    pub fn is_complete(&self) -> bool {
        self.config.max_rounds > 0 && self.completed_rounds() >= self.config.max_rounds
    }

    /// Returns the next round's initial commitment (= current commitment).
    ///
    /// This is the commitment that workers must verify their downloaded
    /// weights against before starting training.
    pub fn next_round_initial_commitment(&self) -> [u8; 32] {
        self.current_commitment
    }

    /// Records a completed round and transitions to the next.
    ///
    /// This is the core method called after a round finishes:
    /// 1. Stores the final weights
    /// 2. Records the round result
    /// 3. Updates current commitment for the next round
    /// 4. Persists aggregator snapshot for crash recovery
    pub fn complete_round(
        &mut self,
        final_weights: Vec<u8>,
        final_loss: f64,
        total_error: f64,
        steps_completed: u64,
        num_contributors: u32,
        tx_hash: Option<[u8; 32]>,
        worker_scores: HashMap<String, f64>,
    ) -> Result<RoundResult, MultiRoundError> {
        let round_number = self.current_round;

        // 1. Store final weights
        let (weights_uri, final_commitment) = self.weight_store
            .store_weights(round_number, &final_weights)
            .map_err(|e| MultiRoundError::WeightStore(e.to_string()))?;

        // 2. Build round result
        let result = RoundResult {
            round_number,
            initial_commitment: self.current_commitment,
            final_commitment,
            weights_uri: weights_uri.clone(),
            final_loss,
            total_error,
            steps_completed,
            num_contributors,
            tx_hash,
            completed_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            worker_scores,
        };

        log::info!(
            "Round {} complete: commitment {} -> {}, loss={:.6}, error={:.6}, steps={}, workers={}",
            round_number,
            hex::encode(&result.initial_commitment[..8]),
            hex::encode(&result.final_commitment[..8]),
            final_loss,
            total_error,
            steps_completed,
            num_contributors,
        );

        // 3. Update state for next round
        self.round_history.push(result.clone());
        self.current_commitment = final_commitment;
        self.current_weights = Some(final_weights.clone());
        self.current_round += 1;

        // 4. Persist aggregator snapshot
        self.save_snapshot(&final_weights)?;

        Ok(result)
    }

    /// Loads weights for the current round, verifying commitment.
    ///
    /// Workers call this to download weights before starting a round.
    /// Returns the weight bytes after verifying they match the expected commitment.
    pub fn load_and_verify_weights(
        &self,
        uri: &str,
        expected_commitment: [u8; 32],
    ) -> Result<Vec<u8>, MultiRoundError> {
        let bytes = self.weight_store
            .load_weights(uri)
            .map_err(|e| MultiRoundError::WeightStore(e.to_string()))?;

        if self.config.verify_commitments {
            let actual = compute_weight_commitment(&bytes);
            if actual != expected_commitment {
                return Err(MultiRoundError::CommitmentMismatch {
                    round: self.current_round,
                    expected: expected_commitment,
                    actual,
                });
            }
        }

        Ok(bytes)
    }

    /// Loads weights from the last completed round's URI.
    ///
    /// Convenience method for workers joining a new round.
    pub fn load_latest_weights(&self) -> Result<Option<Vec<u8>>, MultiRoundError> {
        if let Some(last_result) = self.round_history.last() {
            let bytes = self.load_and_verify_weights(
                &last_result.weights_uri,
                last_result.final_commitment,
            )?;
            Ok(Some(bytes))
        } else {
            Ok(self.current_weights.clone())
        }
    }

    /// Persists the aggregator snapshot for crash recovery.
    fn save_snapshot(&self, weight_bytes: &[u8]) -> Result<(), MultiRoundError> {
        let mut snapshot = AggregatorSnapshot::new(
            weight_bytes.to_vec(),
            self.current_round.saturating_sub(1),
            self.completed_rounds(),
            self.config.d_in,
            self.config.d_hid,
            self.config.d_out,
            self.config.model_seed,
            self.config.learning_rate,
        );

        snapshot.on_chain_model_id = self.model_id;

        // Store error bounds from history
        for result in &self.round_history {
            snapshot.error_bounds.insert(result.round_number, result.total_error);
        }

        self.persistence
            .save_aggregator(&snapshot)
            .map_err(|e| MultiRoundError::Persistence(e.to_string()))?;

        Ok(())
    }

    /// Attempts to recover from the latest aggregator snapshot.
    ///
    /// Returns the recovered state including round number, commitment,
    /// and weight bytes. The controller's state is updated to resume
    /// from where it left off.
    pub fn recover_from_snapshot(&mut self) -> Result<Option<RecoveredState>, MultiRoundError> {
        let snapshot = self.persistence
            .load_latest_aggregator()
            .map_err(|e| MultiRoundError::Persistence(e.to_string()))?;

        let snapshot = match snapshot {
            Some(s) => s,
            None => return Ok(None),
        };

        // Restore state
        let weight_commitment = compute_weight_commitment(&snapshot.model_checkpoint_bytes);
        self.current_round = snapshot.last_completed_round + 1;
        self.current_commitment = weight_commitment;
        self.current_weights = Some(snapshot.model_checkpoint_bytes.clone());
        self.model_id = snapshot.on_chain_model_id;

        log::info!(
            "Recovered from snapshot: round={}, commitment={}, {} bytes",
            self.current_round,
            hex::encode(&weight_commitment[..8]),
            snapshot.model_checkpoint_bytes.len(),
        );

        Ok(Some(RecoveredState {
            round_number: self.current_round,
            commitment: weight_commitment,
            weight_bytes: snapshot.model_checkpoint_bytes,
            total_rounds_completed: snapshot.total_rounds_completed,
            model_id: snapshot.on_chain_model_id,
        }))
    }

    /// Validates that the current state is consistent for starting the next round.
    ///
    /// Checks:
    /// - Commitment chain is unbroken (each round's initial = previous final)
    /// - Current weights match current commitment
    /// - We haven't exceeded max rounds
    pub fn validate_state(&self) -> Result<(), MultiRoundError> {
        // Check max rounds
        if self.is_complete() {
            return Err(MultiRoundError::MaxRoundsReached {
                max: self.config.max_rounds,
            });
        }

        // Verify commitment chain
        for i in 1..self.round_history.len() {
            let prev = &self.round_history[i - 1];
            let curr = &self.round_history[i];
            if prev.final_commitment != curr.initial_commitment {
                return Err(MultiRoundError::BrokenCommitmentChain {
                    round: curr.round_number,
                    expected: prev.final_commitment,
                    actual: curr.initial_commitment,
                });
            }
        }

        // Verify current weights match commitment
        if let Some(ref weights) = self.current_weights {
            let actual = compute_weight_commitment(weights);
            if actual != self.current_commitment {
                return Err(MultiRoundError::CommitmentMismatch {
                    round: self.current_round,
                    expected: self.current_commitment,
                    actual,
                });
            }
        }

        Ok(())
    }

    /// Returns a summary of training progress across all rounds.
    pub fn training_summary(&self) -> TrainingSummary {
        let total_steps: u64 = self.round_history.iter().map(|r| r.steps_completed).sum();
        let total_contributors: u32 = self.round_history.iter().map(|r| r.num_contributors).sum();
        let losses: Vec<f64> = self.round_history.iter().map(|r| r.final_loss).collect();
        let errors: Vec<f64> = self.round_history.iter().map(|r| r.total_error).collect();

        let avg_loss = if losses.is_empty() { 0.0 } else {
            losses.iter().sum::<f64>() / losses.len() as f64
        };
        let avg_error = if errors.is_empty() { 0.0 } else {
            errors.iter().sum::<f64>() / errors.len() as f64
        };

        let initial_commitment = self.round_history
            .first()
            .map(|r| r.initial_commitment)
            .unwrap_or(self.current_commitment);

        TrainingSummary {
            rounds_completed: self.completed_rounds(),
            total_steps,
            total_contributors,
            initial_commitment,
            current_commitment: self.current_commitment,
            average_loss: avg_loss,
            average_error: avg_error,
            loss_history: losses,
            error_history: errors,
        }
    }
}

// ============================================================================
// Worker State Manager
// ============================================================================

/// Manages worker-side state across rounds.
///
/// Workers persist their training state after each round and can recover
/// from crashes by loading the last snapshot and validating against
/// the on-chain commitment.
pub struct WorkerStateManager {
    /// Worker's peer ID.
    peer_id: String,
    /// State persistence.
    persistence: StatePersistence,
    /// Weight store for downloading initial weights.
    weight_store: Box<dyn WeightStore>,
    /// Last completed round.
    last_completed_round: u64,
    /// Total steps completed across all rounds.
    total_steps: u64,
    /// Current model weights.
    current_weights: Option<Vec<u8>>,
    /// Current verified commitment.
    current_commitment: Option<[u8; 32]>,
    /// Optimizer state bytes (Adam moments, etc.).
    optimizer_state: Option<Vec<u8>>,
}

impl WorkerStateManager {
    /// Creates a new worker state manager.
    pub fn new(
        peer_id: String,
        persistence_dir: Option<PathBuf>,
        weight_store: Box<dyn WeightStore>,
    ) -> Result<Self, MultiRoundError> {
        let persistence = StatePersistence::new(
            persistence_dir,
            5,
            &format!("worker_{}", &peer_id[..peer_id.len().min(8)]),
        ).map_err(|e| MultiRoundError::Persistence(e.to_string()))?;

        Ok(Self {
            peer_id,
            persistence,
            weight_store,
            last_completed_round: 0,
            total_steps: 0,
            current_weights: None,
            current_commitment: None,
            optimizer_state: None,
        })
    }

    /// Downloads and verifies weights for a new round.
    pub fn prepare_for_round(
        &mut self,
        weights_uri: &str,
        expected_commitment: [u8; 32],
    ) -> Result<Vec<u8>, MultiRoundError> {
        let bytes = self.weight_store
            .load_weights(weights_uri)
            .map_err(|e| MultiRoundError::WeightStore(e.to_string()))?;

        let actual = compute_weight_commitment(&bytes);
        if actual != expected_commitment {
            return Err(MultiRoundError::CommitmentMismatch {
                round: self.last_completed_round + 1,
                expected: expected_commitment,
                actual,
            });
        }

        self.current_weights = Some(bytes.clone());
        self.current_commitment = Some(expected_commitment);

        log::info!(
            "Worker {} prepared for round {}: {} bytes, commitment={}",
            self.peer_id,
            self.last_completed_round + 1,
            bytes.len(),
            hex::encode(&expected_commitment[..8]),
        );

        Ok(bytes)
    }

    /// Records that a round has been completed and persists state.
    pub fn complete_round(
        &mut self,
        final_weights: Vec<u8>,
        steps_in_round: u64,
        optimizer_state: Option<Vec<u8>>,
    ) -> Result<(), MultiRoundError> {
        self.last_completed_round += 1;
        self.total_steps += steps_in_round;
        self.current_weights = Some(final_weights.clone());
        self.current_commitment = Some(compute_weight_commitment(&final_weights));
        self.optimizer_state = optimizer_state;

        // Persist snapshot
        let mut snapshot = super::persistence::WorkerSnapshot::new(
            self.peer_id.clone(),
            self.last_completed_round,
            self.total_steps,
        );
        snapshot.model_checkpoint_bytes = Some(final_weights);

        self.persistence
            .save_worker(&snapshot)
            .map_err(|e| MultiRoundError::Persistence(e.to_string()))?;

        Ok(())
    }

    /// Attempts crash recovery from the latest snapshot.
    ///
    /// After recovery, the worker should validate its state against
    /// the on-chain commitment before resuming training.
    pub fn recover(&mut self) -> Result<Option<WorkerRecoveredState>, MultiRoundError> {
        let snapshot = self.persistence
            .load_latest_worker()
            .map_err(|e| MultiRoundError::Persistence(e.to_string()))?;

        let snapshot = match snapshot {
            Some(s) => s,
            None => return Ok(None),
        };

        self.last_completed_round = snapshot.last_completed_round;
        self.total_steps = snapshot.steps_completed;

        let commitment = snapshot.model_checkpoint_bytes.as_ref()
            .map(|b| compute_weight_commitment(b));

        self.current_weights = snapshot.model_checkpoint_bytes.clone();
        self.current_commitment = commitment;

        Ok(Some(WorkerRecoveredState {
            peer_id: snapshot.peer_id,
            last_completed_round: snapshot.last_completed_round,
            total_steps: snapshot.steps_completed,
            weight_bytes: snapshot.model_checkpoint_bytes,
            commitment,
        }))
    }

    /// Validates local state against an on-chain commitment.
    pub fn validate_against_commitment(
        &self,
        on_chain_commitment: [u8; 32],
    ) -> Result<bool, MultiRoundError> {
        match &self.current_commitment {
            Some(local) => Ok(*local == on_chain_commitment),
            None => Ok(false),
        }
    }

    /// Returns the last completed round number.
    pub fn last_completed_round(&self) -> u64 {
        self.last_completed_round
    }

    /// Returns total steps completed.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// State recovered from an aggregator snapshot.
#[derive(Debug, Clone)]
pub struct RecoveredState {
    /// Round number to resume from.
    pub round_number: u64,
    /// Current model commitment.
    pub commitment: [u8; 32],
    /// Current model weights.
    pub weight_bytes: Vec<u8>,
    /// Total rounds completed before crash.
    pub total_rounds_completed: u64,
    /// On-chain model ID.
    pub model_id: Option<u64>,
}

/// State recovered from a worker snapshot.
#[derive(Debug, Clone)]
pub struct WorkerRecoveredState {
    /// Worker's peer ID.
    pub peer_id: String,
    /// Last completed round number.
    pub last_completed_round: u64,
    /// Total steps completed.
    pub total_steps: u64,
    /// Model weights (if stored).
    pub weight_bytes: Option<Vec<u8>>,
    /// Weight commitment (if weights present).
    pub commitment: Option<[u8; 32]>,
}

/// Summary of multi-round training progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// Number of completed rounds.
    pub rounds_completed: u64,
    /// Total training steps across all rounds.
    pub total_steps: u64,
    /// Total number of worker contributions.
    pub total_contributors: u32,
    /// Model commitment at the very start.
    pub initial_commitment: [u8; 32],
    /// Current model commitment.
    pub current_commitment: [u8; 32],
    /// Average loss across rounds.
    pub average_loss: f64,
    /// Average error across rounds.
    pub average_error: f64,
    /// Loss per round.
    pub loss_history: Vec<f64>,
    /// Error per round.
    pub error_history: Vec<f64>,
}

// ============================================================================
// Errors
// ============================================================================

/// Errors from multi-round training.
#[derive(Debug)]
pub enum MultiRoundError {
    /// Weight store error.
    WeightStore(String),
    /// Persistence error.
    Persistence(String),
    /// Commitment mismatch when loading weights.
    CommitmentMismatch {
        round: u64,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// Commitment chain is broken between rounds.
    BrokenCommitmentChain {
        round: u64,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// Maximum rounds reached.
    MaxRoundsReached {
        max: u64,
    },
    /// No weights available.
    NoWeightsAvailable,
}

impl std::fmt::Display for MultiRoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightStore(msg) => write!(f, "Weight store error: {}", msg),
            Self::Persistence(msg) => write!(f, "Persistence error: {}", msg),
            Self::CommitmentMismatch { round, expected, actual } => {
                write!(
                    f,
                    "Commitment mismatch at round {}: expected {}, got {}",
                    round,
                    hex::encode(&expected[..8]),
                    hex::encode(&actual[..8]),
                )
            }
            Self::BrokenCommitmentChain { round, expected, actual } => {
                write!(
                    f,
                    "Broken commitment chain at round {}: expected {}, got {}",
                    round,
                    hex::encode(&expected[..8]),
                    hex::encode(&actual[..8]),
                )
            }
            Self::MaxRoundsReached { max } => {
                write!(f, "Maximum rounds reached: {}", max)
            }
            Self::NoWeightsAvailable => write!(f, "No weights available"),
        }
    }
}

impl std::error::Error for MultiRoundError {}

// ============================================================================
// Helpers
// ============================================================================

/// Computes SHA-256 commitment of serialized weight bytes.
pub fn compute_weight_commitment(weight_bytes: &[u8]) -> [u8; 32] {
    let hash = Sha256::digest(weight_bytes);
    hash.into()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(dir: &Path) -> MultiRoundConfig {
        MultiRoundConfig {
            max_rounds: 10,
            weight_store_dir: dir.join("weights"),
            max_weight_files: 5,
            snapshot_dir: Some(dir.join("snapshots")),
            max_snapshots: 3,
            verify_commitments: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_compute_weight_commitment() {
        let data = vec![1u8, 2, 3, 4, 5];
        let c1 = compute_weight_commitment(&data);
        let c2 = compute_weight_commitment(&data);
        assert_eq!(c1, c2); // deterministic

        let other = vec![5u8, 4, 3, 2, 1];
        let c3 = compute_weight_commitment(&other);
        assert_ne!(c1, c3); // different data → different commitment
    }

    #[test]
    fn test_local_weight_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalWeightStore::new(dir.path(), 5).unwrap();

        let weights = vec![42u8; 100];
        let (uri, commitment) = store.store_weights(0, &weights).unwrap();

        assert!(uri.starts_with("file://"));
        assert_eq!(commitment, compute_weight_commitment(&weights));

        let loaded = store.load_weights(&uri).unwrap();
        assert_eq!(loaded, weights);
    }

    #[test]
    fn test_local_weight_store_verify_commitment() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalWeightStore::new(dir.path(), 5).unwrap();

        let weights = vec![99u8; 50];
        let (uri, commitment) = store.store_weights(0, &weights).unwrap();

        assert!(store.verify_commitment(&uri, commitment).unwrap());
        assert!(!store.verify_commitment(&uri, [0xFF; 32]).unwrap());
    }

    #[test]
    fn test_local_weight_store_gc() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalWeightStore::new(dir.path(), 3).unwrap();

        // Store 5 rounds of weights
        for i in 0..5u64 {
            store.store_weights(i, &vec![i as u8; 20]).unwrap();
        }

        // Only 3 should remain (rounds 2, 3, 4)
        assert!(store.load_weights(&format!("file://{}", store.weight_path(0).display())).is_err());
        assert!(store.load_weights(&format!("file://{}", store.weight_path(1).display())).is_err());
        assert!(store.load_weights(&format!("file://{}", store.weight_path(4).display())).is_ok());
    }

    #[test]
    fn test_local_weight_store_corruption_detected() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalWeightStore::new(dir.path(), 5).unwrap();

        let weights = vec![42u8; 100];
        let (uri, _) = store.store_weights(0, &weights).unwrap();

        // Corrupt the file
        let path = store.weight_path(0);
        std::fs::write(&path, b"corrupted").unwrap();

        let result = store.load_weights(&uri);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_round_controller_single_round() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let initial_weights = vec![1u8; 100];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        let mut controller = MultiRoundController::new(
            config,
            initial_commitment,
            Some(initial_weights),
        ).unwrap();

        assert_eq!(controller.current_round(), 0);
        assert_eq!(controller.current_commitment(), initial_commitment);
        assert_eq!(controller.completed_rounds(), 0);

        // Complete a round
        let final_weights = vec![2u8; 100];
        let result = controller.complete_round(
            final_weights.clone(),
            0.5,
            0.01,
            100,
            3,
            None,
            HashMap::new(),
        ).unwrap();

        assert_eq!(result.round_number, 0);
        assert_eq!(result.initial_commitment, initial_commitment);
        assert_eq!(result.final_commitment, compute_weight_commitment(&final_weights));
        assert_eq!(controller.current_round(), 1);
        assert_eq!(controller.completed_rounds(), 1);
    }

    #[test]
    fn test_multi_round_controller_three_rounds_commitment_chaining() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let initial_weights = vec![10u8; 50];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        let mut controller = MultiRoundController::new(
            config,
            initial_commitment,
            Some(initial_weights),
        ).unwrap();

        let mut prev_commitment = initial_commitment;

        for round in 0..3u64 {
            // Each round produces different weights
            let final_weights = vec![(round as u8 + 1) * 10; 50 + round as usize];
            let expected_final_commitment = compute_weight_commitment(&final_weights);

            // The next round's initial commitment should be the previous final
            assert_eq!(
                controller.next_round_initial_commitment(),
                prev_commitment,
                "Round {} initial commitment mismatch",
                round,
            );

            let result = controller.complete_round(
                final_weights,
                0.5 - (round as f64 * 0.1),
                0.01 + (round as f64 * 0.005),
                100 + round * 10,
                3,
                Some([round as u8; 32]),
                HashMap::new(),
            ).unwrap();

            assert_eq!(result.round_number, round);
            assert_eq!(result.initial_commitment, prev_commitment);
            assert_eq!(result.final_commitment, expected_final_commitment);

            prev_commitment = result.final_commitment;
        }

        // Verify chain integrity
        assert_eq!(controller.completed_rounds(), 3);
        controller.validate_state().unwrap();

        // Verify history
        let history = controller.round_history();
        assert_eq!(history.len(), 3);

        // Check commitment chaining in history
        assert_eq!(history[0].initial_commitment, initial_commitment);
        assert_eq!(history[0].final_commitment, history[1].initial_commitment);
        assert_eq!(history[1].final_commitment, history[2].initial_commitment);
    }

    #[test]
    fn test_multi_round_controller_weight_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let initial_weights = vec![1u8; 80];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        let mut controller = MultiRoundController::new(
            config.clone(),
            initial_commitment,
            Some(initial_weights),
        ).unwrap();

        // Complete a round
        let final_weights = vec![2u8; 80];
        let result = controller.complete_round(
            final_weights.clone(),
            0.5,
            0.01,
            50,
            2,
            None,
            HashMap::new(),
        ).unwrap();

        // Load weights from stored URI
        let loaded = controller.load_and_verify_weights(
            &result.weights_uri,
            result.final_commitment,
        ).unwrap();
        assert_eq!(loaded, final_weights);

        // Load latest weights convenience method
        let latest = controller.load_latest_weights().unwrap().unwrap();
        assert_eq!(latest, final_weights);
    }

    #[test]
    fn test_multi_round_controller_crash_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let initial_weights = vec![1u8; 60];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        // Run 2 rounds, then "crash"
        {
            let mut controller = MultiRoundController::new(
                config.clone(),
                initial_commitment,
                Some(initial_weights),
            ).unwrap();

            let w1 = vec![2u8; 60];
            controller.complete_round(w1, 0.5, 0.01, 50, 2, None, HashMap::new()).unwrap();

            let w2 = vec![3u8; 60];
            controller.complete_round(w2, 0.4, 0.008, 50, 3, None, HashMap::new()).unwrap();
        }

        // Recover in a new controller
        let mut recovered = MultiRoundController::new(
            config,
            [0u8; 32], // dummy, will be overwritten by recovery
            None,
        ).unwrap();

        let state = recovered.recover_from_snapshot().unwrap().unwrap();

        assert_eq!(state.round_number, 2);
        assert_eq!(state.total_rounds_completed, 2);
        assert_eq!(state.weight_bytes, vec![3u8; 60]);
        assert_eq!(state.commitment, compute_weight_commitment(&vec![3u8; 60]));
        assert_eq!(recovered.current_round(), 2);
    }

    #[test]
    fn test_multi_round_controller_max_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.max_rounds = 2;

        let initial_weights = vec![1u8; 30];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        let mut controller = MultiRoundController::new(
            config,
            initial_commitment,
            Some(initial_weights),
        ).unwrap();

        // Complete 2 rounds
        controller.complete_round(vec![2u8; 30], 0.5, 0.01, 50, 2, None, HashMap::new()).unwrap();
        controller.complete_round(vec![3u8; 30], 0.4, 0.008, 50, 2, None, HashMap::new()).unwrap();

        assert!(controller.is_complete());

        // Validate should fail because max reached
        let result = controller.validate_state();
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_round_controller_commitment_verification_failure() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let initial_weights = vec![1u8; 50];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        let mut controller = MultiRoundController::new(
            config,
            initial_commitment,
            Some(initial_weights),
        ).unwrap();

        // Store weights
        let final_weights = vec![2u8; 50];
        let result = controller.complete_round(
            final_weights,
            0.5,
            0.01,
            50,
            2,
            None,
            HashMap::new(),
        ).unwrap();

        // Try to load with wrong commitment
        let wrong_commitment = [0xFF; 32];
        let load_result = controller.load_and_verify_weights(
            &result.weights_uri,
            wrong_commitment,
        );
        assert!(load_result.is_err());
    }

    #[test]
    fn test_multi_round_controller_training_summary() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let initial_weights = vec![1u8; 40];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        let mut controller = MultiRoundController::new(
            config,
            initial_commitment,
            Some(initial_weights),
        ).unwrap();

        controller.complete_round(vec![2u8; 40], 0.5, 0.02, 100, 3, None, HashMap::new()).unwrap();
        controller.complete_round(vec![3u8; 40], 0.3, 0.015, 100, 4, None, HashMap::new()).unwrap();
        controller.complete_round(vec![4u8; 40], 0.1, 0.01, 100, 3, None, HashMap::new()).unwrap();

        let summary = controller.training_summary();
        assert_eq!(summary.rounds_completed, 3);
        assert_eq!(summary.total_steps, 300);
        assert_eq!(summary.total_contributors, 10);
        assert_eq!(summary.initial_commitment, initial_commitment);
        assert_eq!(summary.loss_history.len(), 3);
        assert!((summary.average_loss - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_worker_state_manager_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let weight_store = Box::new(
            LocalWeightStore::new(dir.path().join("weights"), 5).unwrap(),
        );

        let mut worker = WorkerStateManager::new(
            "worker-1".to_string(),
            Some(dir.path().join("snapshots")),
            weight_store,
        ).unwrap();

        assert_eq!(worker.last_completed_round(), 0);
        assert_eq!(worker.total_steps(), 0);

        // Complete a round
        let weights = vec![42u8; 80];
        worker.complete_round(weights.clone(), 50, None).unwrap();

        assert_eq!(worker.last_completed_round(), 1);
        assert_eq!(worker.total_steps(), 50);

        // Complete another round
        let weights2 = vec![43u8; 80];
        worker.complete_round(weights2, 30, None).unwrap();

        assert_eq!(worker.last_completed_round(), 2);
        assert_eq!(worker.total_steps(), 80);
    }

    #[test]
    fn test_worker_state_manager_crash_recovery() {
        let dir = tempfile::tempdir().unwrap();

        // First session
        {
            let weight_store = Box::new(
                LocalWeightStore::new(dir.path().join("weights"), 5).unwrap(),
            );
            let mut worker = WorkerStateManager::new(
                "worker-1".to_string(),
                Some(dir.path().join("snapshots")),
                weight_store,
            ).unwrap();

            worker.complete_round(vec![10u8; 40], 25, None).unwrap();
            worker.complete_round(vec![20u8; 40], 30, None).unwrap();
        }

        // Recovery
        let weight_store = Box::new(
            LocalWeightStore::new(dir.path().join("weights"), 5).unwrap(),
        );
        let mut worker = WorkerStateManager::new(
            "worker-1".to_string(),
            Some(dir.path().join("snapshots")),
            weight_store,
        ).unwrap();

        let recovered = worker.recover().unwrap().unwrap();
        assert_eq!(recovered.peer_id, "worker-1");
        assert_eq!(recovered.last_completed_round, 2);
        assert_eq!(recovered.total_steps, 55);
        assert_eq!(recovered.weight_bytes, Some(vec![20u8; 40]));
    }

    #[test]
    fn test_worker_state_manager_commitment_validation() {
        let dir = tempfile::tempdir().unwrap();

        let weight_store = Box::new(
            LocalWeightStore::new(dir.path().join("weights"), 5).unwrap(),
        );
        let mut worker = WorkerStateManager::new(
            "worker-1".to_string(),
            Some(dir.path().join("snapshots")),
            weight_store,
        ).unwrap();

        let weights = vec![42u8; 80];
        let commitment = compute_weight_commitment(&weights);
        worker.complete_round(weights, 50, None).unwrap();

        // Valid commitment
        assert!(worker.validate_against_commitment(commitment).unwrap());

        // Invalid commitment
        assert!(!worker.validate_against_commitment([0xFF; 32]).unwrap());
    }

    #[test]
    fn test_worker_prepare_for_round_with_valid_weights() {
        let dir = tempfile::tempdir().unwrap();

        // Store some weights first
        let store_for_upload = LocalWeightStore::new(dir.path().join("weights"), 5).unwrap();
        let test_weights = vec![99u8; 60];
        let (uri, commitment) = store_for_upload.store_weights(0, &test_weights).unwrap();

        // Worker downloads and verifies
        let weight_store = Box::new(
            LocalWeightStore::new(dir.path().join("weights"), 5).unwrap(),
        );
        let mut worker = WorkerStateManager::new(
            "worker-2".to_string(),
            Some(dir.path().join("snapshots")),
            weight_store,
        ).unwrap();

        let loaded = worker.prepare_for_round(&uri, commitment).unwrap();
        assert_eq!(loaded, test_weights);
    }

    #[test]
    fn test_worker_prepare_for_round_commitment_mismatch() {
        let dir = tempfile::tempdir().unwrap();

        let store = LocalWeightStore::new(dir.path().join("weights"), 5).unwrap();
        let (uri, _) = store.store_weights(0, &vec![1u8; 30]).unwrap();

        let weight_store = Box::new(
            LocalWeightStore::new(dir.path().join("weights"), 5).unwrap(),
        );
        let mut worker = WorkerStateManager::new(
            "worker-3".to_string(),
            Some(dir.path().join("snapshots")),
            weight_store,
        ).unwrap();

        let result = worker.prepare_for_round(&uri, [0xFF; 32]);
        assert!(result.is_err());
    }

    #[test]
    fn test_three_consecutive_rounds_full_integration() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        // === Setup ===
        let initial_weights = vec![100u8; 200];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        let mut controller = MultiRoundController::new(
            config,
            initial_commitment,
            Some(initial_weights.clone()),
        ).unwrap();

        // Store initial weights so workers can download them
        let weight_store = LocalWeightStore::new(
            dir.path().join("weights"),
            10,
        ).unwrap();
        let (initial_uri, _) = weight_store.store_weights(999, &initial_weights).unwrap();

        // Create 3 workers
        let mut workers: Vec<WorkerStateManager> = (0..3)
            .map(|i| {
                WorkerStateManager::new(
                    format!("worker-{}", i),
                    Some(dir.path().join(format!("worker_{}_snapshots", i))),
                    Box::new(
                        LocalWeightStore::new(dir.path().join("weights"), 10).unwrap(),
                    ),
                ).unwrap()
            })
            .collect();

        // === Run 3 rounds ===
        let mut current_uri = initial_uri;
        let mut current_commitment = initial_commitment;

        for round in 0..3u64 {
            // Workers download and verify weights
            for worker in &mut workers {
                let loaded = worker.prepare_for_round(&current_uri, current_commitment).unwrap();
                assert_eq!(compute_weight_commitment(&loaded), current_commitment);
            }

            // Simulate training: each worker produces slightly different weights
            // In reality, the aggregator would pick best or average them
            let round_weights: Vec<Vec<u8>> = (0..3)
                .map(|w| {
                    let mut weights = vec![(round as u8 + 1) * 10 + w; 200];
                    // Make each worker's output slightly different
                    weights[0] = w;
                    weights
                })
                .collect();

            // Aggregator picks the best (worker 0 for simplicity)
            let aggregated_weights = round_weights[0].clone();

            // Complete the round in the controller
            let result = controller.complete_round(
                aggregated_weights.clone(),
                0.5 - (round as f64 * 0.15),
                0.02 - (round as f64 * 0.005),
                100,
                3,
                Some([(round + 1) as u8; 32]),
                [
                    ("worker-0".to_string(), 95.0),
                    ("worker-1".to_string(), 90.0),
                    ("worker-2".to_string(), 85.0),
                ].into_iter().collect(),
            ).unwrap();

            // Workers record completion
            for (i, worker) in workers.iter_mut().enumerate() {
                worker.complete_round(
                    round_weights[i].clone(),
                    100,
                    None,
                ).unwrap();
            }

            // Update for next round
            current_uri = result.weights_uri;
            current_commitment = result.final_commitment;

            // Verify controller state
            assert_eq!(controller.current_round(), round + 1);
            assert_eq!(controller.current_commitment(), current_commitment);
        }

        // === Verify final state ===
        assert_eq!(controller.completed_rounds(), 3);
        controller.validate_state().unwrap();

        let history = controller.round_history();
        assert_eq!(history.len(), 3);

        // Verify commitment chain
        assert_eq!(history[0].initial_commitment, initial_commitment);
        for i in 1..3 {
            assert_eq!(
                history[i].initial_commitment,
                history[i - 1].final_commitment,
                "Commitment chain broken at round {}",
                i,
            );
        }

        // Verify workers tracked correctly
        for worker in &workers {
            assert_eq!(worker.last_completed_round(), 3);
            assert_eq!(worker.total_steps(), 300);
        }

        // Verify training summary
        let summary = controller.training_summary();
        assert_eq!(summary.rounds_completed, 3);
        assert_eq!(summary.total_steps, 300);
        assert_eq!(summary.total_contributors, 9);
        assert_eq!(summary.loss_history.len(), 3);
        // Loss should be decreasing
        assert!(summary.loss_history[0] > summary.loss_history[2]);
    }

    #[test]
    fn test_three_rounds_with_crash_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let initial_weights = vec![50u8; 100];
        let initial_commitment = compute_weight_commitment(&initial_weights);

        // Run 2 rounds, then simulate crash
        let final_weights_r2;
        {
            let mut controller = MultiRoundController::new(
                config.clone(),
                initial_commitment,
                Some(initial_weights),
            ).unwrap();

            controller.complete_round(
                vec![60u8; 100], 0.5, 0.02, 100, 3, None, HashMap::new(),
            ).unwrap();

            final_weights_r2 = vec![70u8; 100];
            controller.complete_round(
                final_weights_r2.clone(), 0.3, 0.015, 100, 3, None, HashMap::new(),
            ).unwrap();
        }

        // Recover and run round 3
        let mut controller = MultiRoundController::new(
            config,
            [0u8; 32],
            None,
        ).unwrap();

        let recovered = controller.recover_from_snapshot().unwrap().unwrap();
        assert_eq!(recovered.round_number, 2);
        assert_eq!(recovered.weight_bytes, final_weights_r2);

        // Run round 3
        let final_weights_r3 = vec![80u8; 100];
        let result = controller.complete_round(
            final_weights_r3.clone(), 0.1, 0.01, 100, 3, None, HashMap::new(),
        ).unwrap();

        assert_eq!(result.round_number, 2);
        assert_eq!(controller.current_round(), 3);
        assert_eq!(
            result.initial_commitment,
            compute_weight_commitment(&final_weights_r2),
        );
        assert_eq!(
            result.final_commitment,
            compute_weight_commitment(&final_weights_r3),
        );
    }
}
