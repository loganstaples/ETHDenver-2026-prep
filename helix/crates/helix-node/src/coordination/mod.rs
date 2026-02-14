//! Distributed Training Coordination Layer.
//!
//! Wires together the aggregator, trainer, and P2P messaging into a
//! complete distributed training lifecycle:
//!
//! 1. Coordinator (aggregator side) broadcasts `RoundConfigure` + model weights
//! 2. Workers receive the round config, load the model, signal `WorkerReady`
//! 3. Coordinator waits for minimum workers, sends `BeginTraining`
//! 4. Workers run `Trainer::train()` with real ZK proof generation
//! 5. Workers send `ProofSubmission` with proof bytes + public inputs
//! 6. Coordinator feeds submissions into `AggregatorNode` for aggregation
//! 7. Coordinator broadcasts `RoundCompleted` with final commitment
//!
//! Communication uses `tokio::mpsc` channels, which can be backed by
//! either in-process connections (for testing) or bridged to TCP/P2P transport.

pub mod coordinator;
pub mod worker_handler;

pub use coordinator::TrainingJobCoordinator;
pub use worker_handler::TrainingWorkerHandler;

use thiserror::Error;

use crate::network::messages::{NetworkMessage, PeerId};

/// Errors that can occur during distributed training coordination.
#[derive(Debug, Error)]
pub enum CoordinationError {
    #[error("not enough workers: need {needed}, got {got}")]
    InsufficientWorkers { needed: usize, got: usize },

    #[error("round timed out after {elapsed_secs}s (deadline was {deadline_secs}s)")]
    RoundTimeout { elapsed_secs: u64, deadline_secs: u64 },

    #[error("worker {peer_id} failed: {reason}")]
    WorkerFailed { peer_id: String, reason: String },

    #[error("aggregation failed: {0}")]
    AggregationFailed(String),

    #[error("proof generation failed: {0}")]
    ProofFailed(String),

    #[error("model checkpoint error: {0}")]
    CheckpointError(String),

    #[error("channel closed: {0}")]
    ChannelClosed(String),

    #[error("round {round_id} not active")]
    RoundNotActive { round_id: u64 },
}

/// A message routed to/from a specific peer.
pub type RoutedMessage = (PeerId, NetworkMessage);

/// Creates a connected pair of channels for coordinator ↔ worker communication.
///
/// Returns `(coordinator_side, worker_side)` where:
/// - `coordinator_side.0` = send messages TO this worker
/// - `coordinator_side.1` = receive messages FROM this worker
/// - `worker_side.0` = send messages TO the coordinator
/// - `worker_side.1` = receive messages FROM the coordinator
pub fn create_channel_pair(
    buffer: usize,
) -> (
    (tokio::sync::mpsc::Sender<NetworkMessage>, tokio::sync::mpsc::Receiver<NetworkMessage>),
    (tokio::sync::mpsc::Sender<NetworkMessage>, tokio::sync::mpsc::Receiver<NetworkMessage>),
) {
    // Coordinator → Worker
    let (coord_tx, worker_rx) = tokio::sync::mpsc::channel(buffer);
    // Worker → Coordinator
    let (worker_tx, coord_rx) = tokio::sync::mpsc::channel(buffer);

    ((coord_tx, coord_rx), (worker_tx, worker_rx))
}
