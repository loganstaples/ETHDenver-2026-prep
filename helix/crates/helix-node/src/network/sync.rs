//! State Synchronization for HELIX Network.
//!
//! Implements state sync between nodes, including model checkpoints,
//! training progress, and gradient history.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

use super::messages::{MessagePayload, NetworkMessage, PeerId, SyncMessage};

/// State being synchronized.
#[derive(Debug, Clone)]
pub struct NetworkState {
    /// Latest completed training round.
    pub latest_round: u64,
    /// Current model hash.
    pub model_hash: [u8; 32],
    /// Accumulated error bound.
    pub accumulated_error: f64,
    /// Last update timestamp.
    pub last_updated: u64,
    /// Active participants in current round.
    pub active_participants: Vec<PeerId>,
}

impl Default for NetworkState {
    fn default() -> Self {
        Self {
            latest_round: 0,
            model_hash: [0; 32],
            accumulated_error: 0.0,
            last_updated: 0,
            active_participants: Vec::new(),
        }
    }
}

/// Configuration for state synchronization.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Sync interval (seconds).
    pub sync_interval_secs: u64,
    /// Maximum rounds to sync at once.
    pub max_sync_batch: u32,
    /// Checkpoint retention count.
    pub checkpoint_retention: usize,
    /// Enable checkpoint compression.
    pub compress_checkpoints: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            sync_interval_secs: 30,
            max_sync_batch: 10,
            checkpoint_retention: 100,
            compress_checkpoints: true,
        }
    }
}

/// Checkpoint data for a training round.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Round number.
    pub round: u64,
    /// Model state hash.
    pub model_hash: [u8; 32],
    /// Serialized model weights (or reference).
    pub weights: CheckpointData,
    /// Error bound at this checkpoint.
    pub error_bound: f64,
    /// Timestamp.
    pub timestamp: u64,
    /// Proof of validity.
    pub proof: Vec<u8>,
}

/// Checkpoint data storage.
#[derive(Debug, Clone)]
pub enum CheckpointData {
    /// Full weights embedded.
    Embedded(Vec<u8>),
    /// Reference to external storage (IPFS, etc).
    Reference(String),
}

/// State synchronization manager.
pub struct StateSync {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: SyncConfig,
    /// Current network state.
    state: Arc<RwLock<NetworkState>>,
    /// Stored checkpoints.
    checkpoints: Arc<RwLock<HashMap<u64, Checkpoint>>>,
    /// Sync status.
    sync_status: Arc<RwLock<SyncStatus>>,
}

/// Synchronization status.
#[derive(Debug, Clone, Default)]
pub struct SyncStatus {
    /// Whether currently syncing.
    pub is_syncing: bool,
    /// Target round to sync to.
    pub target_round: Option<u64>,
    /// Current sync progress (round).
    pub current_round: u64,
    /// Peer we're syncing from.
    pub sync_peer: Option<PeerId>,
    /// Last successful sync time.
    pub last_sync_time: u64,
}

impl StateSync {
    /// Creates a new state sync manager.
    pub fn new(local_id: PeerId, config: SyncConfig) -> Self {
        Self {
            local_id,
            config,
            state: Arc::new(RwLock::new(NetworkState::default())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            sync_status: Arc::new(RwLock::new(SyncStatus::default())),
        }
    }

    /// Gets the current network state.
    pub async fn get_state(&self) -> NetworkState {
        self.state.read().await.clone()
    }

    /// Updates the network state.
    pub async fn update_state(&self, state: NetworkState) {
        let mut current = self.state.write().await;
        *current = state;
        current.last_updated = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Advances to a new round.
    pub async fn advance_round(&self, model_hash: [u8; 32], error: f64) {
        let mut state = self.state.write().await;
        state.latest_round += 1;
        state.model_hash = model_hash;
        state.accumulated_error += error;
        state.last_updated = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Stores a checkpoint.
    pub async fn store_checkpoint(&self, checkpoint: Checkpoint) {
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(checkpoint.round, checkpoint);

        // Prune old checkpoints
        while checkpoints.len() > self.config.checkpoint_retention {
            if let Some(min_round) = checkpoints.keys().min().copied() {
                checkpoints.remove(&min_round);
            }
        }
    }

    /// Gets a checkpoint.
    pub async fn get_checkpoint(&self, round: u64) -> Option<Checkpoint> {
        self.checkpoints.read().await.get(&round).cloned()
    }

    /// Gets latest checkpoint.
    pub async fn get_latest_checkpoint(&self) -> Option<Checkpoint> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.values().max_by_key(|c| c.round).cloned()
    }

    /// Creates a state request message.
    pub fn create_state_request(&self) -> NetworkMessage {
        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Sync(SyncMessage::GetState),
        )
    }

    /// Creates a checkpoint request message.
    pub fn create_checkpoint_request(&self, round: u64) -> NetworkMessage {
        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Sync(SyncMessage::GetCheckpoint { round }),
        )
    }

    /// Handles a sync message.
    pub async fn handle_sync_message(
        &self,
        _sender: PeerId,
        msg: SyncMessage,
    ) -> Option<NetworkMessage> {
        match msg {
            SyncMessage::GetState => {
                let state = self.get_state().await;
                Some(NetworkMessage::new(
                    self.local_id.clone(),
                    MessagePayload::Sync(SyncMessage::State {
                        latest_round: state.latest_round,
                        model_hash: state.model_hash,
                        accumulated_error: state.accumulated_error,
                    }),
                ))
            }
            SyncMessage::State {
                latest_round,
                model_hash,
                accumulated_error,
            } => {
                let current = self.get_state().await;
                
                // If remote is ahead, we need to sync
                if latest_round > current.latest_round {
                    let mut status = self.sync_status.write().await;
                    status.target_round = Some(latest_round);
                    // Could trigger sync here
                }
                None
            }
            SyncMessage::GetCheckpoint { round } => {
                if let Some(checkpoint) = self.get_checkpoint(round).await {
                    let data = match checkpoint.weights {
                        CheckpointData::Embedded(d) => d,
                        CheckpointData::Reference(r) => r.into_bytes(),
                    };
                    Some(NetworkMessage::new(
                        self.local_id.clone(),
                        MessagePayload::Sync(SyncMessage::Checkpoint { round, data }),
                    ))
                } else {
                    None
                }
            }
            SyncMessage::Checkpoint { round, data } => {
                // Store received checkpoint
                let checkpoint = Checkpoint {
                    round,
                    model_hash: [0; 32], // Would parse from data
                    weights: CheckpointData::Embedded(data),
                    error_bound: 0.0,
                    timestamp: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    proof: Vec::new(),
                };
                self.store_checkpoint(checkpoint).await;
                
                // Update sync status
                let mut status = self.sync_status.write().await;
                status.current_round = round;
                if status.target_round == Some(round) {
                    status.is_syncing = false;
                    status.last_sync_time = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                }
                None
            }
        }
    }

    /// Gets sync status.
    pub async fn get_sync_status(&self) -> SyncStatus {
        self.sync_status.read().await.clone()
    }

    /// Starts syncing to a target round.
    pub async fn start_sync(&self, target_round: u64, peer: PeerId) {
        let mut status = self.sync_status.write().await;
        status.is_syncing = true;
        status.target_round = Some(target_round);
        status.sync_peer = Some(peer);
    }

    /// Checks if we're behind and need to sync.
    pub async fn needs_sync(&self, network_round: u64) -> bool {
        let state = self.state.read().await;
        network_round > state.latest_round
    }

    /// Gets the number of stored checkpoints.
    pub async fn checkpoint_count(&self) -> usize {
        self.checkpoints.read().await.len()
    }
}

/// Diff between two states for efficient sync.
#[derive(Debug, Clone)]
pub struct StateDiff {
    /// Rounds included in diff.
    pub rounds: Vec<u64>,
    /// Model diffs (layer-wise).
    pub model_diffs: HashMap<String, Vec<u8>>,
    /// Error deltas.
    pub error_delta: f64,
}

impl StateDiff {
    /// Creates a new empty diff.
    pub fn empty() -> Self {
        Self {
            rounds: Vec::new(),
            model_diffs: HashMap::new(),
            error_delta: 0.0,
        }
    }

    /// Whether the diff is empty (no changes).
    pub fn is_empty(&self) -> bool {
        self.rounds.is_empty() && self.model_diffs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_state_sync_init() {
        let local_id = PeerId::random();
        let sync = StateSync::new(local_id, SyncConfig::default());
        
        let state = sync.get_state().await;
        assert_eq!(state.latest_round, 0);
    }

    #[tokio::test]
    async fn test_advance_round() {
        let local_id = PeerId::random();
        let sync = StateSync::new(local_id, SyncConfig::default());
        
        sync.advance_round([1; 32], 0.01).await;
        
        let state = sync.get_state().await;
        assert_eq!(state.latest_round, 1);
        assert_eq!(state.model_hash, [1; 32]);
    }

    #[tokio::test]
    async fn test_checkpoint_storage() {
        let local_id = PeerId::random();
        let sync = StateSync::new(local_id, SyncConfig::default());
        
        let checkpoint = Checkpoint {
            round: 5,
            model_hash: [5; 32],
            weights: CheckpointData::Embedded(vec![1, 2, 3]),
            error_bound: 0.01,
            timestamp: 12345,
            proof: vec![],
        };
        
        sync.store_checkpoint(checkpoint).await;
        
        let retrieved = sync.get_checkpoint(5).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().round, 5);
    }

    #[tokio::test]
    async fn test_needs_sync() {
        let local_id = PeerId::random();
        let sync = StateSync::new(local_id, SyncConfig::default());
        
        assert!(sync.needs_sync(5).await);
        
        sync.advance_round([1; 32], 0.01).await;
        sync.advance_round([2; 32], 0.01).await;
        
        assert!(sync.needs_sync(5).await);
        assert!(!sync.needs_sync(2).await);
    }
}
