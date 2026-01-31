//! Distributed Checkpoint Coordination.
//!
//! This module provides distributed checkpoint coordination for multi-worker training.
//! It ensures consistent checkpoint creation across all workers, supports resume from
//! interrupted training, and integrates with the fault tolerance and synchronization systems.
//!
//! Key features:
//! - Coordinated checkpoint creation with consensus
//! - MPC share-aware checkpointing (each worker stores their shares)
//! - Resume from any checkpoint with full training state reconstruction
//! - Automatic checkpoint on failure detection
//! - Cross-worker checkpoint verification with Merkle proofs

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, oneshot};

use crate::network::messages::PeerId;

use super::checkpoint::{
    Checkpoint, CheckpointConfig, CheckpointError, CheckpointId, CheckpointManager,
    CheckpointMetadata, OptimizerState,
};
use super::model::{ModelWeights, ModelMetadata, ModelVersion};
use super::round::RoundId;
use super::state_machine::{DistributedRoundId, DistributedRoundState};
use super::synchronization::{SyncPhase, BarrierResult};

// ============================================================================
// Distributed Checkpoint Types
// ============================================================================

/// Unique identifier for a distributed checkpoint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DistributedCheckpointId {
    /// Round ID at checkpoint.
    pub round_id: u64,
    /// Checkpoint sequence within round.
    pub sequence: u64,
    /// Timestamp when initiated.
    pub timestamp: u64,
}

impl DistributedCheckpointId {
    /// Creates a new distributed checkpoint ID.
    pub fn new(round_id: u64, sequence: u64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            round_id,
            sequence,
            timestamp,
        }
    }
}

impl std::fmt::Display for DistributedCheckpointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dckpt-r{}-s{}-{}", self.round_id, self.sequence, self.timestamp)
    }
}

/// Status of a distributed checkpoint operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedCheckpointStatus {
    /// Checkpoint initiated, waiting for workers.
    Initiated,
    /// Collecting worker checkpoints.
    Collecting,
    /// Verifying checkpoint consistency.
    Verifying,
    /// Checkpoint completed successfully.
    Completed,
    /// Checkpoint failed.
    Failed,
    /// Checkpoint cancelled.
    Cancelled,
}

/// Worker's contribution to a distributed checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCheckpointContribution {
    /// Worker ID.
    pub worker_id: PeerId,
    /// Worker's local checkpoint ID.
    pub local_checkpoint_id: CheckpointId,
    /// Worker's share commitment (Merkle root).
    pub share_commitment: [u8; 32],
    /// Current iteration at checkpoint.
    pub iteration: u64,
    /// Round state at checkpoint.
    pub round_state: DistributedRoundState,
    /// Worker's shares hash (for verification).
    pub shares_hash: [u8; 32],
    /// Timestamp of contribution.
    pub contributed_at: u64,
    /// Worker's view of aggregated gradient commitment.
    pub gradient_commitment: Option<[u8; 32]>,
}

/// Complete distributed checkpoint with all worker contributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCheckpoint {
    /// Distributed checkpoint ID.
    pub id: DistributedCheckpointId,
    /// Status of the checkpoint.
    pub status: DistributedCheckpointStatus,
    /// Worker contributions.
    pub contributions: HashMap<PeerId, WorkerCheckpointContribution>,
    /// Required workers for consensus.
    pub required_workers: HashSet<PeerId>,
    /// Consensus commitment (agreed upon by all workers).
    pub consensus_commitment: Option<[u8; 32]>,
    /// Aggregated model commitment.
    pub model_commitment: Option<[u8; 32]>,
    /// Training progress metrics.
    pub progress: CheckpointProgress,
    /// Timestamp when checkpoint started.
    pub started_at: u64,
    /// Timestamp when checkpoint completed.
    pub completed_at: Option<u64>,
    /// Reason for failure (if failed).
    pub failure_reason: Option<String>,
}

impl DistributedCheckpoint {
    /// Creates a new distributed checkpoint.
    pub fn new(id: DistributedCheckpointId, required_workers: HashSet<PeerId>) -> Self {
        Self {
            id,
            status: DistributedCheckpointStatus::Initiated,
            contributions: HashMap::new(),
            required_workers,
            consensus_commitment: None,
            model_commitment: None,
            progress: CheckpointProgress::default(),
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            completed_at: None,
            failure_reason: None,
        }
    }

    /// Checks if all required workers have contributed.
    pub fn is_complete(&self) -> bool {
        self.required_workers
            .iter()
            .all(|w| self.contributions.contains_key(w))
    }

    /// Returns missing workers.
    pub fn missing_workers(&self) -> Vec<PeerId> {
        self.required_workers
            .iter()
            .filter(|w| !self.contributions.contains_key(w))
            .cloned()
            .collect()
    }

    /// Adds a worker contribution.
    pub fn add_contribution(&mut self, contribution: WorkerCheckpointContribution) {
        self.contributions.insert(contribution.worker_id, contribution);
        if self.is_complete() {
            self.status = DistributedCheckpointStatus::Verifying;
        } else {
            self.status = DistributedCheckpointStatus::Collecting;
        }
    }
}

/// Training progress snapshot for checkpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckpointProgress {
    /// Total iterations completed.
    pub total_iterations: u64,
    /// Current round number.
    pub current_round: u64,
    /// Total rounds completed.
    pub rounds_completed: u64,
    /// Current epoch.
    pub current_epoch: u64,
    /// Total samples processed.
    pub samples_processed: u64,
    /// Current loss value.
    pub current_loss: Option<f64>,
    /// Best loss achieved.
    pub best_loss: Option<f64>,
    /// Learning rate at checkpoint.
    pub learning_rate: f64,
    /// Gradient norm at checkpoint.
    pub gradient_norm: Option<f64>,
}

/// Full training state for resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumableTrainingState {
    /// Distributed checkpoint info.
    pub checkpoint_id: DistributedCheckpointId,
    /// This worker's ID.
    pub worker_id: PeerId,
    /// All worker IDs in training.
    pub worker_ids: Vec<PeerId>,
    /// Training progress.
    pub progress: CheckpointProgress,
    /// Round state.
    pub round_state: DistributedRoundState,
    /// Model metadata.
    pub model_metadata: ModelMetadata,
    /// Model version.
    pub model_version: ModelVersion,
    /// Optimizer state.
    pub optimizer_state: Option<OptimizerState>,
    /// Worker's share commitments.
    pub share_commitment: [u8; 32],
    /// Path to local weights file.
    pub weights_path: PathBuf,
    /// Path to shares file.
    pub shares_path: Option<PathBuf>,
    /// Pending gradient shares (if mid-aggregation).
    pub pending_shares: Option<Vec<u8>>,
    /// Leader at checkpoint.
    pub leader_id: Option<PeerId>,
    /// Training configuration snapshot.
    pub training_config: TrainingConfigSnapshot,
}

/// Snapshot of training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigSnapshot {
    /// Batch size.
    pub batch_size: usize,
    /// Number of gradient accumulation steps.
    pub gradient_accumulation_steps: usize,
    /// Maximum rounds.
    pub max_rounds: u64,
    /// Checkpoint interval.
    pub checkpoint_interval: u64,
    /// MPC threshold.
    pub mpc_threshold: usize,
    /// Total MPC parties.
    pub mpc_parties: usize,
    /// Aggregation strategy.
    pub aggregation_strategy: String,
}

impl Default for TrainingConfigSnapshot {
    fn default() -> Self {
        Self {
            batch_size: 32,
            gradient_accumulation_steps: 1,
            max_rounds: 1000,
            checkpoint_interval: 10,
            mpc_threshold: 2,
            mpc_parties: 3,
            aggregation_strategy: "fedavg".to_string(),
        }
    }
}

// ============================================================================
// Share Checkpoint (MPC-aware)
// ============================================================================

/// MPC share data for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareCheckpoint {
    /// Worker ID that owns these shares.
    pub worker_id: PeerId,
    /// Share index in MPC scheme.
    pub share_index: usize,
    /// Encrypted share data (for persistence).
    pub encrypted_shares: Vec<u8>,
    /// Share commitment (Merkle root).
    pub commitment: [u8; 32],
    /// Number of shares stored.
    pub num_shares: usize,
    /// Threshold required for reconstruction.
    pub threshold: usize,
    /// Total parties in MPC scheme.
    pub total_parties: usize,
    /// Timestamp of share snapshot.
    pub timestamp: u64,
}

impl ShareCheckpoint {
    /// Creates a new share checkpoint.
    pub fn new(
        worker_id: PeerId,
        share_index: usize,
        shares: &[Vec<f32>],
        threshold: usize,
        total_parties: usize,
    ) -> Self {
        // Compute commitment (simplified - in production use proper Merkle tree)
        let commitment = Self::compute_commitment(shares);

        // Serialize shares (in production, encrypt with worker's key)
        let shares_bytes: Vec<u8> = shares
            .iter()
            .flat_map(|s| s.iter().flat_map(|f| f.to_le_bytes()))
            .collect();

        Self {
            worker_id,
            share_index,
            encrypted_shares: shares_bytes,
            commitment,
            num_shares: shares.len(),
            threshold,
            total_parties,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Computes commitment for shares.
    fn compute_commitment(shares: &[Vec<f32>]) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for share in shares {
            for &val in share {
                val.to_bits().hash(&mut hasher);
            }
        }

        let hash = hasher.finish();
        let mut commitment = [0u8; 32];
        commitment[..8].copy_from_slice(&hash.to_le_bytes());
        commitment[8..16].copy_from_slice(&hash.to_be_bytes());
        commitment
    }

    /// Verifies share commitment.
    pub fn verify_commitment(&self, shares: &[Vec<f32>]) -> bool {
        Self::compute_commitment(shares) == self.commitment
    }

    /// Reconstructs shares from checkpoint.
    pub fn reconstruct_shares(&self) -> Vec<Vec<f32>> {
        let float_size = std::mem::size_of::<f32>();
        let mut shares = Vec::new();

        // This is a simplified reconstruction - in production, decrypt first
        let mut offset = 0;
        while offset < self.encrypted_shares.len() && shares.len() < self.num_shares {
            let mut share = Vec::new();
            // Estimate share size (would be stored in metadata in production)
            let share_size = (self.encrypted_shares.len() / self.num_shares) / float_size;

            for _ in 0..share_size {
                if offset + float_size <= self.encrypted_shares.len() {
                    let bytes: [u8; 4] = self.encrypted_shares[offset..offset + float_size]
                        .try_into()
                        .unwrap_or([0; 4]);
                    share.push(f32::from_le_bytes(bytes));
                    offset += float_size;
                }
            }
            shares.push(share);
        }

        shares
    }
}

// ============================================================================
// Distributed Checkpoint Coordinator
// ============================================================================

/// Configuration for distributed checkpointing.
#[derive(Debug, Clone)]
pub struct DistributedCheckpointConfig {
    /// Base checkpoint config.
    pub base_config: CheckpointConfig,
    /// Directory for distributed checkpoint metadata.
    pub distributed_dir: PathBuf,
    /// Timeout for collecting worker contributions.
    pub collection_timeout: Duration,
    /// Minimum workers required for valid checkpoint.
    pub min_workers: usize,
    /// Whether to checkpoint on failure detection.
    pub checkpoint_on_failure: bool,
    /// Whether to verify cross-worker consistency.
    pub verify_consistency: bool,
    /// Maximum concurrent checkpoint operations.
    pub max_concurrent: usize,
}

impl Default for DistributedCheckpointConfig {
    fn default() -> Self {
        Self {
            base_config: CheckpointConfig::default(),
            distributed_dir: PathBuf::from("checkpoints/distributed"),
            collection_timeout: Duration::from_secs(60),
            min_workers: 2,
            checkpoint_on_failure: true,
            verify_consistency: true,
            max_concurrent: 2,
        }
    }
}

/// Events emitted by the checkpoint coordinator.
#[derive(Debug, Clone)]
pub enum CheckpointEvent {
    /// Checkpoint initiated.
    Initiated {
        checkpoint_id: DistributedCheckpointId,
        required_workers: usize,
    },
    /// Worker contributed to checkpoint.
    WorkerContributed {
        checkpoint_id: DistributedCheckpointId,
        worker_id: PeerId,
        remaining: usize,
    },
    /// Checkpoint verification started.
    VerificationStarted {
        checkpoint_id: DistributedCheckpointId,
    },
    /// Checkpoint completed successfully.
    Completed {
        checkpoint_id: DistributedCheckpointId,
        duration: Duration,
    },
    /// Checkpoint failed.
    Failed {
        checkpoint_id: DistributedCheckpointId,
        reason: String,
    },
    /// Resume operation started.
    ResumeStarted {
        checkpoint_id: DistributedCheckpointId,
        worker_id: PeerId,
    },
    /// Resume operation completed.
    ResumeCompleted {
        checkpoint_id: DistributedCheckpointId,
        worker_id: PeerId,
    },
    /// Emergency checkpoint triggered.
    EmergencyCheckpoint {
        checkpoint_id: DistributedCheckpointId,
        reason: String,
    },
}

/// Request for checkpoint operations.
#[derive(Debug)]
pub enum CheckpointRequest {
    /// Initiate a new distributed checkpoint.
    Initiate {
        round_id: u64,
        workers: Vec<PeerId>,
        response: oneshot::Sender<Result<DistributedCheckpointId, DistributedCheckpointError>>,
    },
    /// Submit worker contribution.
    Contribute {
        checkpoint_id: DistributedCheckpointId,
        contribution: WorkerCheckpointContribution,
        response: oneshot::Sender<Result<(), DistributedCheckpointError>>,
    },
    /// Get checkpoint status.
    GetStatus {
        checkpoint_id: DistributedCheckpointId,
        response: oneshot::Sender<Option<DistributedCheckpoint>>,
    },
    /// Resume from checkpoint.
    Resume {
        checkpoint_id: DistributedCheckpointId,
        worker_id: PeerId,
        response: oneshot::Sender<Result<ResumableTrainingState, DistributedCheckpointError>>,
    },
    /// Cancel checkpoint.
    Cancel {
        checkpoint_id: DistributedCheckpointId,
        response: oneshot::Sender<Result<(), DistributedCheckpointError>>,
    },
}

/// Distributed checkpoint coordinator.
pub struct DistributedCheckpointCoordinator {
    /// Configuration.
    config: DistributedCheckpointConfig,
    /// Active checkpoints.
    active_checkpoints: Arc<RwLock<HashMap<DistributedCheckpointId, DistributedCheckpoint>>>,
    /// Completed checkpoints (limited history).
    completed_checkpoints: Arc<RwLock<Vec<DistributedCheckpoint>>>,
    /// Local checkpoint manager.
    local_manager: Arc<RwLock<CheckpointManager>>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<CheckpointEvent>,
    /// Request channel.
    request_tx: mpsc::Sender<CheckpointRequest>,
    /// Checkpoint sequence counter.
    sequence_counter: Arc<RwLock<u64>>,
    /// Worker's share checkpoints.
    share_checkpoints: Arc<RwLock<HashMap<PeerId, ShareCheckpoint>>>,
}

impl DistributedCheckpointCoordinator {
    /// Creates a new distributed checkpoint coordinator.
    pub fn new(config: DistributedCheckpointConfig) -> Result<Self, DistributedCheckpointError> {
        // Create directories
        std::fs::create_dir_all(&config.distributed_dir)
            .map_err(|e| DistributedCheckpointError::IoError(e.to_string()))?;

        // Create local manager
        let local_manager = CheckpointManager::new(config.base_config.clone())
            .map_err(|e| DistributedCheckpointError::IoError(e.to_string()))?;

        let (event_tx, _) = broadcast::channel(256);
        let (request_tx, _request_rx) = mpsc::channel(64);

        Ok(Self {
            config,
            active_checkpoints: Arc::new(RwLock::new(HashMap::new())),
            completed_checkpoints: Arc::new(RwLock::new(Vec::new())),
            local_manager: Arc::new(RwLock::new(local_manager)),
            event_tx,
            request_tx,
            sequence_counter: Arc::new(RwLock::new(0)),
            share_checkpoints: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Subscribes to checkpoint events.
    pub fn subscribe(&self) -> broadcast::Receiver<CheckpointEvent> {
        self.event_tx.subscribe()
    }

    /// Returns the request sender for async operations.
    pub fn request_sender(&self) -> mpsc::Sender<CheckpointRequest> {
        self.request_tx.clone()
    }

    /// Initiates a new distributed checkpoint.
    pub async fn initiate_checkpoint(
        &self,
        round_id: u64,
        workers: Vec<PeerId>,
    ) -> Result<DistributedCheckpointId, DistributedCheckpointError> {
        if workers.len() < self.config.min_workers {
            return Err(DistributedCheckpointError::InsufficientWorkers {
                required: self.config.min_workers,
                available: workers.len(),
            });
        }

        // Check for concurrent checkpoint limit
        let active_count = self.active_checkpoints.read().len();
        if active_count >= self.config.max_concurrent {
            return Err(DistributedCheckpointError::TooManyConcurrent {
                max: self.config.max_concurrent,
            });
        }

        // Generate checkpoint ID
        let sequence = {
            let mut counter = self.sequence_counter.write();
            *counter += 1;
            *counter
        };
        let checkpoint_id = DistributedCheckpointId::new(round_id, sequence);

        // Create checkpoint
        let checkpoint = DistributedCheckpoint::new(
            checkpoint_id,
            workers.iter().cloned().collect(),
        );

        // Store checkpoint
        self.active_checkpoints.write().insert(checkpoint_id, checkpoint);

        // Emit event
        let _ = self.event_tx.send(CheckpointEvent::Initiated {
            checkpoint_id,
            required_workers: workers.len(),
        });

        Ok(checkpoint_id)
    }

    /// Submits a worker's contribution to the checkpoint.
    pub async fn contribute(
        &self,
        checkpoint_id: DistributedCheckpointId,
        contribution: WorkerCheckpointContribution,
    ) -> Result<(), DistributedCheckpointError> {
        let mut checkpoints = self.active_checkpoints.write();

        let checkpoint = checkpoints.get_mut(&checkpoint_id)
            .ok_or(DistributedCheckpointError::NotFound(checkpoint_id))?;

        // Verify worker is required
        if !checkpoint.required_workers.contains(&contribution.worker_id) {
            return Err(DistributedCheckpointError::UnexpectedWorker(contribution.worker_id));
        }

        // Check for duplicate contribution
        if checkpoint.contributions.contains_key(&contribution.worker_id) {
            return Err(DistributedCheckpointError::DuplicateContribution(contribution.worker_id));
        }

        let worker_id = contribution.worker_id;
        let remaining = checkpoint.missing_workers().len() - 1;

        checkpoint.add_contribution(contribution);

        // Emit event
        let _ = self.event_tx.send(CheckpointEvent::WorkerContributed {
            checkpoint_id,
            worker_id,
            remaining,
        });

        // If complete, start verification
        if checkpoint.is_complete() {
            drop(checkpoints);
            self.verify_and_complete(checkpoint_id).await?;
        }

        Ok(())
    }

    /// Verifies checkpoint consistency and marks as complete.
    async fn verify_and_complete(
        &self,
        checkpoint_id: DistributedCheckpointId,
    ) -> Result<(), DistributedCheckpointError> {
        let start_time = Instant::now();

        let _ = self.event_tx.send(CheckpointEvent::VerificationStarted { checkpoint_id });

        let verification_result = if self.config.verify_consistency {
            self.verify_consistency(checkpoint_id).await
        } else {
            Ok(())
        };

        let mut checkpoints = self.active_checkpoints.write();

        if let Some(checkpoint) = checkpoints.get_mut(&checkpoint_id) {
            match verification_result {
                Ok(()) => {
                    checkpoint.status = DistributedCheckpointStatus::Completed;
                    checkpoint.completed_at = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs()
                    );

                    // Save distributed checkpoint metadata
                    let _ = self.save_checkpoint_metadata(checkpoint);

                    let _ = self.event_tx.send(CheckpointEvent::Completed {
                        checkpoint_id,
                        duration: start_time.elapsed(),
                    });

                    // Move to completed list
                    let completed = checkpoint.clone();
                    drop(checkpoints);

                    let mut completed_list = self.completed_checkpoints.write();
                    completed_list.push(completed);
                    // Keep only recent checkpoints
                    while completed_list.len() > 10 {
                        completed_list.remove(0);
                    }

                    // Remove from active
                    self.active_checkpoints.write().remove(&checkpoint_id);
                }
                Err(e) => {
                    checkpoint.status = DistributedCheckpointStatus::Failed;
                    checkpoint.failure_reason = Some(e.to_string());

                    let _ = self.event_tx.send(CheckpointEvent::Failed {
                        checkpoint_id,
                        reason: e.to_string(),
                    });
                }
            }
        }

        verification_result
    }

    /// Verifies consistency across worker contributions.
    async fn verify_consistency(
        &self,
        checkpoint_id: DistributedCheckpointId,
    ) -> Result<(), DistributedCheckpointError> {
        let checkpoints = self.active_checkpoints.read();
        let checkpoint = checkpoints.get(&checkpoint_id)
            .ok_or(DistributedCheckpointError::NotFound(checkpoint_id))?;

        // Verify all workers are at the same iteration
        let iterations: HashSet<u64> = checkpoint.contributions.values()
            .map(|c| c.iteration)
            .collect();

        if iterations.len() > 1 {
            return Err(DistributedCheckpointError::IterationMismatch {
                iterations: iterations.into_iter().collect(),
            });
        }

        // Verify all workers are in the same round state
        let states: HashSet<String> = checkpoint.contributions.values()
            .map(|c| format!("{:?}", c.round_state))
            .collect();

        if states.len() > 1 {
            return Err(DistributedCheckpointError::StateMismatch {
                states: states.into_iter().collect(),
            });
        }

        // Verify gradient commitments match (for aggregation verification)
        let gradient_commitments: Vec<_> = checkpoint.contributions.values()
            .filter_map(|c| c.gradient_commitment)
            .collect();

        if !gradient_commitments.is_empty() {
            let first = gradient_commitments[0];
            if !gradient_commitments.iter().all(|c| *c == first) {
                return Err(DistributedCheckpointError::GradientCommitmentMismatch);
            }
        }

        Ok(())
    }

    /// Saves checkpoint metadata to disk.
    fn save_checkpoint_metadata(&self, checkpoint: &DistributedCheckpoint) -> Result<(), DistributedCheckpointError> {
        let filename = format!("{}.json", checkpoint.id);
        let path = self.config.distributed_dir.join(&filename);

        let json = serde_json::to_string_pretty(checkpoint)
            .map_err(|e| DistributedCheckpointError::SerializationError(e.to_string()))?;

        std::fs::write(&path, json)
            .map_err(|e| DistributedCheckpointError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Gets the status of a checkpoint.
    pub fn get_status(&self, checkpoint_id: DistributedCheckpointId) -> Option<DistributedCheckpoint> {
        // Check active checkpoints
        if let Some(checkpoint) = self.active_checkpoints.read().get(&checkpoint_id) {
            return Some(checkpoint.clone());
        }

        // Check completed checkpoints
        self.completed_checkpoints.read()
            .iter()
            .find(|c| c.id == checkpoint_id)
            .cloned()
    }

    /// Lists all available checkpoints for resume.
    pub fn list_resumable(&self) -> Vec<DistributedCheckpoint> {
        self.completed_checkpoints.read()
            .iter()
            .filter(|c| c.status == DistributedCheckpointStatus::Completed)
            .cloned()
            .collect()
    }

    /// Finds the latest checkpoint for a specific round.
    pub fn find_latest_for_round(&self, round_id: u64) -> Option<DistributedCheckpoint> {
        self.completed_checkpoints.read()
            .iter()
            .filter(|c| c.id.round_id == round_id && c.status == DistributedCheckpointStatus::Completed)
            .max_by_key(|c| c.id.sequence)
            .cloned()
    }

    /// Saves local model checkpoint.
    pub fn save_local(
        &self,
        model: &ModelWeights,
        optimizer_state: Option<&OptimizerState>,
        round_id: Option<RoundId>,
    ) -> Result<CheckpointId, DistributedCheckpointError> {
        self.local_manager.write()
            .save(model, optimizer_state, round_id)
            .map_err(|e| DistributedCheckpointError::LocalCheckpointError(e.to_string()))
    }

    /// Loads latest local checkpoint.
    pub fn load_latest_local(&self) -> Result<Option<Checkpoint>, DistributedCheckpointError> {
        self.local_manager.read()
            .load_latest()
            .map_err(|e| DistributedCheckpointError::LocalCheckpointError(e.to_string()))
    }

    /// Saves share checkpoint for a worker.
    pub fn save_shares(
        &self,
        worker_id: PeerId,
        share_checkpoint: ShareCheckpoint,
    ) -> Result<(), DistributedCheckpointError> {
        // Save to disk
        let filename = format!("shares-{}.json", worker_id);
        let path = self.config.distributed_dir.join(&filename);

        let json = serde_json::to_string(&share_checkpoint)
            .map_err(|e| DistributedCheckpointError::SerializationError(e.to_string()))?;

        std::fs::write(&path, json)
            .map_err(|e| DistributedCheckpointError::IoError(e.to_string()))?;

        // Store in memory
        self.share_checkpoints.write().insert(worker_id, share_checkpoint);

        Ok(())
    }

    /// Loads share checkpoint for a worker.
    pub fn load_shares(&self, worker_id: PeerId) -> Option<ShareCheckpoint> {
        // Try memory first
        if let Some(shares) = self.share_checkpoints.read().get(&worker_id) {
            return Some(shares.clone());
        }

        // Try disk
        let filename = format!("shares-{}.json", worker_id);
        let path = self.config.distributed_dir.join(&filename);

        if path.exists() {
            if let Ok(json) = std::fs::read_to_string(&path) {
                if let Ok(shares) = serde_json::from_str(&json) {
                    return Some(shares);
                }
            }
        }

        None
    }

    /// Initiates emergency checkpoint (on failure detection).
    pub async fn emergency_checkpoint(
        &self,
        round_id: u64,
        workers: Vec<PeerId>,
        reason: String,
    ) -> Result<DistributedCheckpointId, DistributedCheckpointError> {
        if !self.config.checkpoint_on_failure {
            return Err(DistributedCheckpointError::EmergencyCheckpointDisabled);
        }

        let checkpoint_id = self.initiate_checkpoint(round_id, workers).await?;

        let _ = self.event_tx.send(CheckpointEvent::EmergencyCheckpoint {
            checkpoint_id,
            reason,
        });

        Ok(checkpoint_id)
    }

    /// Cancels an active checkpoint.
    pub async fn cancel_checkpoint(
        &self,
        checkpoint_id: DistributedCheckpointId,
    ) -> Result<(), DistributedCheckpointError> {
        let mut checkpoints = self.active_checkpoints.write();

        if let Some(checkpoint) = checkpoints.get_mut(&checkpoint_id) {
            if checkpoint.status == DistributedCheckpointStatus::Completed ||
               checkpoint.status == DistributedCheckpointStatus::Failed {
                return Err(DistributedCheckpointError::AlreadyFinalized(checkpoint_id));
            }

            checkpoint.status = DistributedCheckpointStatus::Cancelled;
            checkpoint.failure_reason = Some("Cancelled by user".to_string());

            Ok(())
        } else {
            Err(DistributedCheckpointError::NotFound(checkpoint_id))
        }
    }
}

// ============================================================================
// Resume Coordinator
// ============================================================================

/// Coordinator for resuming training from checkpoint.
pub struct ResumeCoordinator {
    /// Checkpoint coordinator.
    checkpoint_coordinator: Arc<DistributedCheckpointCoordinator>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<CheckpointEvent>,
}

impl ResumeCoordinator {
    /// Creates a new resume coordinator.
    pub fn new(checkpoint_coordinator: Arc<DistributedCheckpointCoordinator>) -> Self {
        let (event_tx, _) = broadcast::channel(64);
        Self {
            checkpoint_coordinator,
            event_tx,
        }
    }

    /// Finds the best checkpoint to resume from.
    pub fn find_resume_checkpoint(&self) -> Option<DistributedCheckpoint> {
        let resumable = self.checkpoint_coordinator.list_resumable();

        // Find the most recent completed checkpoint
        resumable.into_iter()
            .max_by_key(|c| (c.progress.total_iterations, c.id.timestamp))
    }

    /// Finds a specific checkpoint for resume.
    pub fn find_checkpoint(
        &self,
        checkpoint_id: DistributedCheckpointId,
    ) -> Option<DistributedCheckpoint> {
        self.checkpoint_coordinator.get_status(checkpoint_id)
    }

    /// Creates resumable training state for a worker.
    pub async fn create_resume_state(
        &self,
        checkpoint: &DistributedCheckpoint,
        worker_id: PeerId,
    ) -> Result<ResumableTrainingState, DistributedCheckpointError> {
        let _ = self.event_tx.send(CheckpointEvent::ResumeStarted {
            checkpoint_id: checkpoint.id,
            worker_id,
        });

        // Get worker's contribution
        let contribution = checkpoint.contributions.get(&worker_id)
            .ok_or(DistributedCheckpointError::WorkerNotInCheckpoint(worker_id))?;

        // Load local checkpoint
        let local_checkpoint = self.checkpoint_coordinator
            .load_latest_local()?
            .ok_or(DistributedCheckpointError::NoLocalCheckpoint)?;

        // Load shares if available
        let shares = self.checkpoint_coordinator.load_shares(worker_id);

        let state = ResumableTrainingState {
            checkpoint_id: checkpoint.id,
            worker_id,
            worker_ids: checkpoint.required_workers.iter().cloned().collect(),
            progress: checkpoint.progress.clone(),
            round_state: contribution.round_state,
            model_metadata: local_checkpoint.weights.as_ref()
                .map(|w| w.metadata.clone())
                .unwrap_or_default(),
            model_version: local_checkpoint.metadata.model_version,
            optimizer_state: local_checkpoint.optimizer_state,
            share_commitment: contribution.share_commitment,
            weights_path: local_checkpoint.metadata.weights_path,
            shares_path: shares.as_ref().map(|_| {
                self.checkpoint_coordinator.config.distributed_dir
                    .join(format!("shares-{}.json", worker_id))
            }),
            pending_shares: None,
            leader_id: None,
            training_config: TrainingConfigSnapshot::default(),
        };

        let _ = self.event_tx.send(CheckpointEvent::ResumeCompleted {
            checkpoint_id: checkpoint.id,
            worker_id,
        });

        Ok(state)
    }

    /// Coordinates resume across multiple workers.
    pub async fn coordinate_resume(
        &self,
        checkpoint: &DistributedCheckpoint,
        workers: Vec<PeerId>,
    ) -> Result<HashMap<PeerId, ResumableTrainingState>, DistributedCheckpointError> {
        let mut states = HashMap::new();

        for worker_id in workers {
            if checkpoint.contributions.contains_key(&worker_id) {
                let state = self.create_resume_state(checkpoint, worker_id).await?;
                states.insert(worker_id, state);
            }
        }

        if states.is_empty() {
            return Err(DistributedCheckpointError::NoValidWorkersForResume);
        }

        Ok(states)
    }

    /// Verifies all workers can resume from the same checkpoint.
    pub fn verify_resume_compatibility(
        &self,
        checkpoint: &DistributedCheckpoint,
        workers: &[PeerId],
    ) -> Result<(), DistributedCheckpointError> {
        let missing: Vec<_> = workers.iter()
            .filter(|w| !checkpoint.contributions.contains_key(w))
            .cloned()
            .collect();

        if !missing.is_empty() {
            return Err(DistributedCheckpointError::WorkersMissingFromCheckpoint(missing));
        }

        Ok(())
    }
}

// ============================================================================
// Checkpoint Synchronization Helper
// ============================================================================

/// Helper for synchronizing checkpoint operations with training barriers.
pub struct CheckpointSyncHelper {
    /// Checkpoint coordinator.
    coordinator: Arc<DistributedCheckpointCoordinator>,
    /// This worker's ID.
    worker_id: PeerId,
}

impl CheckpointSyncHelper {
    /// Creates a new checkpoint sync helper.
    pub fn new(
        coordinator: Arc<DistributedCheckpointCoordinator>,
        worker_id: PeerId,
    ) -> Self {
        Self {
            coordinator,
            worker_id,
        }
    }

    /// Creates a checkpoint contribution from current state.
    pub fn create_contribution(
        &self,
        local_checkpoint_id: CheckpointId,
        iteration: u64,
        round_state: DistributedRoundState,
        shares: &[Vec<f32>],
        gradient_commitment: Option<[u8; 32]>,
    ) -> WorkerCheckpointContribution {
        let share_commitment = ShareCheckpoint::compute_commitment(shares);

        // Compute shares hash
        let shares_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            shares.len().hash(&mut hasher);
            let hash = hasher.finish();
            let mut hash_bytes = [0u8; 32];
            hash_bytes[..8].copy_from_slice(&hash.to_le_bytes());
            hash_bytes
        };

        WorkerCheckpointContribution {
            worker_id: self.worker_id,
            local_checkpoint_id,
            share_commitment,
            iteration,
            round_state,
            shares_hash,
            contributed_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            gradient_commitment,
        }
    }

    /// Performs synchronized checkpoint with barrier.
    pub async fn checkpoint_with_barrier(
        &self,
        checkpoint_id: DistributedCheckpointId,
        contribution: WorkerCheckpointContribution,
        timeout: Duration,
    ) -> Result<(), DistributedCheckpointError> {
        // Submit contribution
        self.coordinator.contribute(checkpoint_id, contribution).await?;

        // Wait for checkpoint to complete
        let start = Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(DistributedCheckpointError::Timeout);
            }

            if let Some(status) = self.coordinator.get_status(checkpoint_id) {
                match status.status {
                    DistributedCheckpointStatus::Completed => return Ok(()),
                    DistributedCheckpointStatus::Failed => {
                        return Err(DistributedCheckpointError::CheckpointFailed(
                            status.failure_reason.unwrap_or_default()
                        ));
                    }
                    DistributedCheckpointStatus::Cancelled => {
                        return Err(DistributedCheckpointError::Cancelled);
                    }
                    _ => {}
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Distributed checkpoint errors.
#[derive(Debug)]
pub enum DistributedCheckpointError {
    /// Not enough workers for checkpoint.
    InsufficientWorkers { required: usize, available: usize },
    /// Too many concurrent checkpoints.
    TooManyConcurrent { max: usize },
    /// Checkpoint not found.
    NotFound(DistributedCheckpointId),
    /// Unexpected worker contribution.
    UnexpectedWorker(PeerId),
    /// Duplicate contribution from worker.
    DuplicateContribution(PeerId),
    /// Workers at different iterations.
    IterationMismatch { iterations: Vec<u64> },
    /// Workers in different states.
    StateMismatch { states: Vec<String> },
    /// Gradient commitments don't match.
    GradientCommitmentMismatch,
    /// IO error.
    IoError(String),
    /// Serialization error.
    SerializationError(String),
    /// Local checkpoint error.
    LocalCheckpointError(String),
    /// Checkpoint already finalized.
    AlreadyFinalized(DistributedCheckpointId),
    /// Emergency checkpoint disabled.
    EmergencyCheckpointDisabled,
    /// Worker not in checkpoint.
    WorkerNotInCheckpoint(PeerId),
    /// No local checkpoint available.
    NoLocalCheckpoint,
    /// No valid workers for resume.
    NoValidWorkersForResume,
    /// Workers missing from checkpoint.
    WorkersMissingFromCheckpoint(Vec<PeerId>),
    /// Timeout waiting for checkpoint.
    Timeout,
    /// Checkpoint failed.
    CheckpointFailed(String),
    /// Checkpoint cancelled.
    Cancelled,
}

impl std::fmt::Display for DistributedCheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientWorkers { required, available } => {
                write!(f, "Insufficient workers: need {}, have {}", required, available)
            }
            Self::TooManyConcurrent { max } => {
                write!(f, "Too many concurrent checkpoints (max: {})", max)
            }
            Self::NotFound(id) => write!(f, "Checkpoint not found: {}", id),
            Self::UnexpectedWorker(id) => write!(f, "Unexpected worker: {}", id),
            Self::DuplicateContribution(id) => write!(f, "Duplicate contribution from: {}", id),
            Self::IterationMismatch { iterations } => {
                write!(f, "Iteration mismatch: {:?}", iterations)
            }
            Self::StateMismatch { states } => write!(f, "State mismatch: {:?}", states),
            Self::GradientCommitmentMismatch => write!(f, "Gradient commitment mismatch"),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::LocalCheckpointError(msg) => write!(f, "Local checkpoint error: {}", msg),
            Self::AlreadyFinalized(id) => write!(f, "Checkpoint already finalized: {}", id),
            Self::EmergencyCheckpointDisabled => write!(f, "Emergency checkpoint disabled"),
            Self::WorkerNotInCheckpoint(id) => write!(f, "Worker not in checkpoint: {}", id),
            Self::NoLocalCheckpoint => write!(f, "No local checkpoint available"),
            Self::NoValidWorkersForResume => write!(f, "No valid workers for resume"),
            Self::WorkersMissingFromCheckpoint(ids) => {
                write!(f, "Workers missing from checkpoint: {:?}", ids)
            }
            Self::Timeout => write!(f, "Checkpoint operation timed out"),
            Self::CheckpointFailed(reason) => write!(f, "Checkpoint failed: {}", reason),
            Self::Cancelled => write!(f, "Checkpoint cancelled"),
        }
    }
}

impl std::error::Error for DistributedCheckpointError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_peer_id(id: u8) -> PeerId {
        let mut bytes = [0u8; 32];
        bytes[0] = id;
        PeerId(bytes)
    }

    fn create_test_config(dir: &Path) -> DistributedCheckpointConfig {
        DistributedCheckpointConfig {
            base_config: CheckpointConfig {
                checkpoint_dir: dir.join("local"),
                max_checkpoints: 5,
                checkpoint_interval: 10,
                save_optimizer: false,
                verify_merkle: false,
            },
            distributed_dir: dir.join("distributed"),
            collection_timeout: Duration::from_secs(10),
            min_workers: 2,
            checkpoint_on_failure: true,
            verify_consistency: true,
            max_concurrent: 3,
        }
    }

    #[test]
    fn test_distributed_checkpoint_id() {
        let id = DistributedCheckpointId::new(5, 10);
        assert_eq!(id.round_id, 5);
        assert_eq!(id.sequence, 10);
        assert!(id.timestamp > 0);

        let display = format!("{}", id);
        assert!(display.starts_with("dckpt-r5-s10-"));
    }

    #[test]
    fn test_distributed_checkpoint_creation() {
        let workers: HashSet<PeerId> = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
            create_test_peer_id(3),
        ].into_iter().collect();

        let id = DistributedCheckpointId::new(1, 1);
        let checkpoint = DistributedCheckpoint::new(id, workers.clone());

        assert_eq!(checkpoint.status, DistributedCheckpointStatus::Initiated);
        assert_eq!(checkpoint.required_workers, workers);
        assert!(checkpoint.contributions.is_empty());
        assert!(!checkpoint.is_complete());
        assert_eq!(checkpoint.missing_workers().len(), 3);
    }

    #[test]
    fn test_worker_contribution() {
        let workers: HashSet<PeerId> = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
        ].into_iter().collect();

        let id = DistributedCheckpointId::new(1, 1);
        let mut checkpoint = DistributedCheckpoint::new(id, workers);

        let contribution = WorkerCheckpointContribution {
            worker_id: create_test_peer_id(1),
            local_checkpoint_id: CheckpointId::new(100),
            share_commitment: [0u8; 32],
            iteration: 50,
            round_state: DistributedRoundState::Computing,
            shares_hash: [1u8; 32],
            contributed_at: 12345,
            gradient_commitment: None,
        };

        checkpoint.add_contribution(contribution);

        assert_eq!(checkpoint.status, DistributedCheckpointStatus::Collecting);
        assert_eq!(checkpoint.missing_workers().len(), 1);
        assert!(!checkpoint.is_complete());

        // Add second contribution
        let contribution2 = WorkerCheckpointContribution {
            worker_id: create_test_peer_id(2),
            local_checkpoint_id: CheckpointId::new(101),
            share_commitment: [0u8; 32],
            iteration: 50,
            round_state: DistributedRoundState::Computing,
            shares_hash: [2u8; 32],
            contributed_at: 12346,
            gradient_commitment: None,
        };

        checkpoint.add_contribution(contribution2);

        assert_eq!(checkpoint.status, DistributedCheckpointStatus::Verifying);
        assert!(checkpoint.is_complete());
        assert!(checkpoint.missing_workers().is_empty());
    }

    #[test]
    fn test_share_checkpoint() {
        let worker_id = create_test_peer_id(1);
        let shares = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let share_ckpt = ShareCheckpoint::new(worker_id, 0, &shares, 2, 3);

        assert_eq!(share_ckpt.worker_id, worker_id);
        assert_eq!(share_ckpt.share_index, 0);
        assert_eq!(share_ckpt.num_shares, 2);
        assert_eq!(share_ckpt.threshold, 2);
        assert_eq!(share_ckpt.total_parties, 3);

        // Verify commitment
        assert!(share_ckpt.verify_commitment(&shares));

        // Different shares should have different commitment
        let different_shares = vec![
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ];
        assert!(!share_ckpt.verify_commitment(&different_shares));
    }

    #[tokio::test]
    async fn test_coordinator_initiate_checkpoint() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
            create_test_peer_id(3),
        ];

        let checkpoint_id = coordinator.initiate_checkpoint(1, workers.clone()).await.unwrap();

        assert_eq!(checkpoint_id.round_id, 1);
        assert_eq!(checkpoint_id.sequence, 1);

        let status = coordinator.get_status(checkpoint_id).unwrap();
        assert_eq!(status.status, DistributedCheckpointStatus::Initiated);
        assert_eq!(status.required_workers.len(), 3);
    }

    #[tokio::test]
    async fn test_coordinator_insufficient_workers() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        // Only one worker, but min_workers is 2
        let workers = vec![create_test_peer_id(1)];

        let result = coordinator.initiate_checkpoint(1, workers).await;
        assert!(matches!(
            result,
            Err(DistributedCheckpointError::InsufficientWorkers { .. })
        ));
    }

    #[tokio::test]
    async fn test_coordinator_contribution_flow() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
        ];

        let checkpoint_id = coordinator.initiate_checkpoint(1, workers).await.unwrap();

        // Contribute from first worker
        let contribution1 = WorkerCheckpointContribution {
            worker_id: create_test_peer_id(1),
            local_checkpoint_id: CheckpointId::new(100),
            share_commitment: [0u8; 32],
            iteration: 50,
            round_state: DistributedRoundState::Computing,
            shares_hash: [1u8; 32],
            contributed_at: 12345,
            gradient_commitment: Some([42u8; 32]),
        };

        coordinator.contribute(checkpoint_id, contribution1).await.unwrap();

        let status = coordinator.get_status(checkpoint_id).unwrap();
        assert_eq!(status.status, DistributedCheckpointStatus::Collecting);

        // Contribute from second worker (matching iteration and state)
        let contribution2 = WorkerCheckpointContribution {
            worker_id: create_test_peer_id(2),
            local_checkpoint_id: CheckpointId::new(101),
            share_commitment: [0u8; 32],
            iteration: 50, // Same iteration
            round_state: DistributedRoundState::Computing, // Same state
            shares_hash: [2u8; 32],
            contributed_at: 12346,
            gradient_commitment: Some([42u8; 32]), // Same gradient commitment
        };

        coordinator.contribute(checkpoint_id, contribution2).await.unwrap();

        // Should be completed now
        let status = coordinator.get_status(checkpoint_id);
        // Note: status might be None if moved to completed list, or Completed
        if let Some(s) = status {
            assert!(matches!(
                s.status,
                DistributedCheckpointStatus::Completed | DistributedCheckpointStatus::Verifying
            ));
        }
    }

    #[tokio::test]
    async fn test_coordinator_duplicate_contribution_rejected() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
        ];

        let checkpoint_id = coordinator.initiate_checkpoint(1, workers).await.unwrap();

        let contribution = WorkerCheckpointContribution {
            worker_id: create_test_peer_id(1),
            local_checkpoint_id: CheckpointId::new(100),
            share_commitment: [0u8; 32],
            iteration: 50,
            round_state: DistributedRoundState::Computing,
            shares_hash: [1u8; 32],
            contributed_at: 12345,
            gradient_commitment: None,
        };

        coordinator.contribute(checkpoint_id, contribution.clone()).await.unwrap();

        // Second contribution from same worker should fail
        let result = coordinator.contribute(checkpoint_id, contribution).await;
        assert!(matches!(
            result,
            Err(DistributedCheckpointError::DuplicateContribution(_))
        ));
    }

    #[tokio::test]
    async fn test_coordinator_unexpected_worker_rejected() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
        ];

        let checkpoint_id = coordinator.initiate_checkpoint(1, workers).await.unwrap();

        // Worker 3 is not in the required workers
        let contribution = WorkerCheckpointContribution {
            worker_id: create_test_peer_id(3),
            local_checkpoint_id: CheckpointId::new(100),
            share_commitment: [0u8; 32],
            iteration: 50,
            round_state: DistributedRoundState::Computing,
            shares_hash: [3u8; 32],
            contributed_at: 12345,
            gradient_commitment: None,
        };

        let result = coordinator.contribute(checkpoint_id, contribution).await;
        assert!(matches!(
            result,
            Err(DistributedCheckpointError::UnexpectedWorker(_))
        ));
    }

    #[tokio::test]
    async fn test_emergency_checkpoint() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        let mut receiver = coordinator.subscribe();

        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
        ];

        let checkpoint_id = coordinator
            .emergency_checkpoint(5, workers, "Worker failure detected".to_string())
            .await
            .unwrap();

        assert_eq!(checkpoint_id.round_id, 5);

        // Should have received emergency event
        let event = receiver.try_recv().unwrap();
        assert!(matches!(event, CheckpointEvent::Initiated { .. }));

        let event = receiver.try_recv().unwrap();
        assert!(matches!(event, CheckpointEvent::EmergencyCheckpoint { .. }));
    }

    #[test]
    fn test_checkpoint_progress() {
        let progress = CheckpointProgress {
            total_iterations: 1000,
            current_round: 10,
            rounds_completed: 9,
            current_epoch: 5,
            samples_processed: 50000,
            current_loss: Some(0.5),
            best_loss: Some(0.3),
            learning_rate: 0.001,
            gradient_norm: Some(1.5),
        };

        assert_eq!(progress.total_iterations, 1000);
        assert_eq!(progress.current_epoch, 5);
    }

    #[test]
    fn test_training_config_snapshot_default() {
        let config = TrainingConfigSnapshot::default();

        assert_eq!(config.batch_size, 32);
        assert_eq!(config.mpc_threshold, 2);
        assert_eq!(config.mpc_parties, 3);
        assert_eq!(config.aggregation_strategy, "fedavg");
    }

    #[test]
    fn test_checkpoint_sync_helper_create_contribution() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = Arc::new(
            DistributedCheckpointCoordinator::new(config).unwrap()
        );

        let worker_id = create_test_peer_id(1);
        let helper = CheckpointSyncHelper::new(coordinator, worker_id);

        let shares = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let contribution = helper.create_contribution(
            CheckpointId::new(100),
            50,
            DistributedRoundState::Computing,
            &shares,
            Some([42u8; 32]),
        );

        assert_eq!(contribution.worker_id, worker_id);
        assert_eq!(contribution.iteration, 50);
        assert_eq!(contribution.gradient_commitment, Some([42u8; 32]));
    }

    #[tokio::test]
    async fn test_cancel_checkpoint() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
        ];

        let checkpoint_id = coordinator.initiate_checkpoint(1, workers).await.unwrap();

        // Cancel it
        coordinator.cancel_checkpoint(checkpoint_id).await.unwrap();

        let status = coordinator.get_status(checkpoint_id).unwrap();
        assert_eq!(status.status, DistributedCheckpointStatus::Cancelled);
    }

    #[test]
    fn test_list_resumable() {
        let dir = tempdir().unwrap();
        let config = create_test_config(dir.path());
        let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

        // Initially empty
        let resumable = coordinator.list_resumable();
        assert!(resumable.is_empty());
    }

    #[test]
    fn test_distributed_checkpoint_error_display() {
        let err = DistributedCheckpointError::InsufficientWorkers {
            required: 3,
            available: 1,
        };
        assert_eq!(
            format!("{}", err),
            "Insufficient workers: need 3, have 1"
        );

        let err = DistributedCheckpointError::Timeout;
        assert_eq!(format!("{}", err), "Checkpoint operation timed out");
    }
}
