//! Resilient MPC training session at the node layer.
//!
//! This module handles network-level resilience for MPC training:
//!
//! 1. **Worker disconnect detection**: Uses P2P heartbeat timeouts from
//!    the connection manager to detect worker failures.
//! 2. **Reconnection grace period**: When a worker disconnects, the session
//!    waits for a configurable grace period before removing them. If the
//!    worker reconnects within the grace period, they rejoin seamlessly.
//! 3. **Session state persistence**: Periodic snapshots of session state
//!    to disk for crash recovery and resumability.
//! 4. **Graceful session shutdown**: Coordinated shutdown that notifies
//!    all peers, saves state, and exits cleanly.
//! 5. **Health monitoring integration**: Bridges the ConnectionManager's
//!    health events into the MPC training flow.
//!
//! # Architecture
//!
//! The [`ResilientSession`] sits between the P2P `ConnectionManager` and
//! the MPC `ResilientTrainingLoop`. It translates P2P events into MPC
//! events and vice versa.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::network::messages::PeerId;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a resilient MPC training session.
#[derive(Debug, Clone)]
pub struct ResilientSessionConfig {
    /// Grace period before declaring a worker permanently disconnected.
    pub disconnect_grace_period: Duration,
    /// Maximum reconnection attempts for a disconnected worker.
    pub max_reconnect_attempts: u32,
    /// Interval between session state snapshots.
    pub snapshot_interval: Duration,
    /// Directory for session state persistence.
    pub persistence_dir: PathBuf,
    /// Minimum workers required to continue training.
    pub min_workers: usize,
    /// Heartbeat interval for liveness checking.
    pub heartbeat_interval: Duration,
    /// How long after shutdown signal to wait for current step.
    pub shutdown_timeout: Duration,
}

impl Default for ResilientSessionConfig {
    fn default() -> Self {
        Self {
            disconnect_grace_period: Duration::from_secs(30),
            max_reconnect_attempts: 5,
            snapshot_interval: Duration::from_secs(60),
            persistence_dir: PathBuf::from("/tmp/helix-session"),
            min_workers: 2,
            heartbeat_interval: Duration::from_secs(5),
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

impl ResilientSessionConfig {
    /// Creates a configuration for testing.
    pub fn for_testing() -> Self {
        Self {
            disconnect_grace_period: Duration::from_millis(100),
            max_reconnect_attempts: 2,
            snapshot_interval: Duration::from_millis(500),
            persistence_dir: std::env::temp_dir().join("helix-test-session"),
            min_workers: 2,
            heartbeat_interval: Duration::from_millis(50),
            shutdown_timeout: Duration::from_secs(5),
        }
    }
}

// ============================================================================
// Worker Tracking
// ============================================================================

/// Health state of a worker in the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerSessionState {
    /// Worker is connected and participating.
    Active,
    /// Worker missed heartbeats, in grace period.
    GracePeriod {
        /// When the disconnect was first detected.
        detected_at: Instant,
    },
    /// Worker reconnected after grace period.
    Reconnected,
    /// Worker permanently removed from session.
    Removed {
        /// Reason for removal.
        reason: String,
    },
}

/// Per-worker tracking information.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    /// P2P peer ID.
    pub peer_id: PeerId,
    /// MPC party index.
    pub party_index: usize,
    /// Current state.
    pub state: WorkerSessionState,
    /// Last heartbeat received.
    pub last_heartbeat: Instant,
    /// Reconnection attempts so far.
    pub reconnect_attempts: u32,
    /// Total messages received from this worker.
    pub messages_received: u64,
    /// Number of training steps completed by this worker.
    pub steps_completed: u64,
}

impl WorkerInfo {
    /// Creates a new active worker.
    pub fn new(peer_id: PeerId, party_index: usize) -> Self {
        Self {
            peer_id,
            party_index,
            state: WorkerSessionState::Active,
            last_heartbeat: Instant::now(),
            reconnect_attempts: 0,
            messages_received: 0,
            steps_completed: 0,
        }
    }

    /// Returns whether the worker is available for computation.
    pub fn is_available(&self) -> bool {
        matches!(
            self.state,
            WorkerSessionState::Active | WorkerSessionState::Reconnected,
        )
    }
}

// ============================================================================
// Session Events
// ============================================================================

/// Events emitted by the resilient session.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// A worker's heartbeat was received.
    HeartbeatReceived {
        party_index: usize,
    },
    /// A worker entered the grace period (stopped responding).
    WorkerGracePeriodStarted {
        party_index: usize,
        peer_id: PeerId,
    },
    /// A worker reconnected during the grace period.
    WorkerReconnected {
        party_index: usize,
        peer_id: PeerId,
    },
    /// A worker was permanently removed.
    WorkerRemoved {
        party_index: usize,
        peer_id: PeerId,
        reason: String,
    },
    /// Not enough workers remain — training must pause.
    InsufficientWorkers {
        active: usize,
        required: usize,
    },
    /// Session state was saved to disk.
    StateSaved {
        path: PathBuf,
    },
    /// Graceful shutdown initiated.
    ShutdownStarted,
    /// Graceful shutdown completed.
    ShutdownCompleted,
}

// ============================================================================
// Session State Snapshot
// ============================================================================

/// Serializable snapshot of the session state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Session identifier.
    pub session_id: String,
    /// Current training step.
    pub current_step: u64,
    /// Active worker party indices.
    pub active_workers: Vec<usize>,
    /// Removed worker party indices with reasons.
    pub removed_workers: Vec<(usize, String)>,
    /// Timestamp of the snapshot (unix millis).
    pub timestamp_ms: u64,
    /// Number of checkpoints submitted.
    pub checkpoints_submitted: u64,
    /// Total steps completed.
    pub total_steps_completed: u64,
}

// ============================================================================
// Resilient Session
// ============================================================================

/// Manages the network-level resilience of an MPC training session.
///
/// Tracks worker health, manages disconnections with grace periods,
/// persists session state for crash recovery, and coordinates graceful
/// shutdown across all workers.
pub struct ResilientSession {
    /// Configuration.
    config: ResilientSessionConfig,
    /// Session identifier.
    session_id: String,
    /// Worker tracking state.
    workers: HashMap<usize, WorkerInfo>,
    /// Event channel sender.
    event_tx: mpsc::Sender<SessionEvent>,
    /// Event channel receiver.
    event_rx: mpsc::Receiver<SessionEvent>,
    /// Whether shutdown has been requested.
    shutdown_requested: bool,
    /// Current training step (tracked for snapshots).
    current_step: u64,
    /// Total checkpoints submitted.
    checkpoints_submitted: u64,
}

impl ResilientSession {
    /// Creates a new resilient session.
    pub fn new(
        session_id: String,
        config: ResilientSessionConfig,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1000);

        Self {
            config,
            session_id,
            workers: HashMap::new(),
            event_tx,
            event_rx,
            shutdown_requested: false,
            current_step: 0,
            checkpoints_submitted: 0,
        }
    }

    /// Registers a worker in the session.
    pub fn register_worker(&mut self, peer_id: PeerId, party_index: usize) {
        info!(
            party = party_index,
            peer = %peer_id,
            "Worker registered in session"
        );
        self.workers.insert(party_index, WorkerInfo::new(peer_id, party_index));
    }

    /// Records a heartbeat from a worker.
    pub fn record_heartbeat(&mut self, party_index: usize) {
        if let Some(worker) = self.workers.get_mut(&party_index) {
            worker.last_heartbeat = Instant::now();
            worker.messages_received += 1;

            // If the worker was in grace period, it has reconnected.
            if matches!(worker.state, WorkerSessionState::GracePeriod { .. }) {
                info!(
                    party = party_index,
                    "Worker reconnected during grace period"
                );
                worker.state = WorkerSessionState::Reconnected;
                worker.reconnect_attempts = 0;

                let _ = self.event_tx.try_send(SessionEvent::WorkerReconnected {
                    party_index,
                    peer_id: worker.peer_id.clone(),
                });
            }
        }
    }

    /// Updates the current training step.
    pub fn update_step(&mut self, step: u64) {
        self.current_step = step;
        for worker in self.workers.values_mut() {
            if worker.is_available() {
                worker.steps_completed += 1;
            }
        }
    }

    /// Checks for disconnected workers and manages grace periods.
    ///
    /// Returns the indices of workers that were permanently removed
    /// during this check.
    pub fn check_worker_health(&mut self) -> Vec<usize> {
        let now = Instant::now();
        let grace_period = self.config.disconnect_grace_period;
        let max_attempts = self.config.max_reconnect_attempts;
        let mut removed = Vec::new();

        let mut events_to_send = Vec::new();

        for (party_index, worker) in self.workers.iter_mut() {
            match &worker.state {
                WorkerSessionState::Active | WorkerSessionState::Reconnected => {
                    let elapsed = now.duration_since(worker.last_heartbeat);
                    if elapsed > grace_period {
                        // Enter grace period.
                        worker.state = WorkerSessionState::GracePeriod {
                            detected_at: now,
                        };
                        worker.reconnect_attempts += 1;

                        warn!(
                            party = party_index,
                            elapsed_ms = elapsed.as_millis() as u64,
                            "Worker entered grace period"
                        );

                        events_to_send.push(SessionEvent::WorkerGracePeriodStarted {
                            party_index: *party_index,
                            peer_id: worker.peer_id.clone(),
                        });
                    }
                }
                WorkerSessionState::GracePeriod { detected_at } => {
                    let grace_elapsed = now.duration_since(*detected_at);
                    if grace_elapsed > grace_period {
                        if worker.reconnect_attempts >= max_attempts {
                            // Permanently remove.
                            let reason = format!(
                                "exceeded {} reconnect attempts after {}s grace period",
                                max_attempts,
                                grace_elapsed.as_secs(),
                            );
                            worker.state = WorkerSessionState::Removed {
                                reason: reason.clone(),
                            };
                            removed.push(*party_index);

                            warn!(
                                party = party_index,
                                reason = %reason,
                                "Worker permanently removed"
                            );

                            events_to_send.push(SessionEvent::WorkerRemoved {
                                party_index: *party_index,
                                peer_id: worker.peer_id.clone(),
                                reason,
                            });
                        } else {
                            // Give another grace period.
                            worker.state = WorkerSessionState::GracePeriod {
                                detected_at: now,
                            };
                            worker.reconnect_attempts += 1;

                            debug!(
                                party = party_index,
                                attempt = worker.reconnect_attempts,
                                "Worker grace period extended"
                            );
                        }
                    }
                }
                WorkerSessionState::Removed { .. } => {
                    // Already removed, nothing to do.
                }
            }
        }

        // Send events.
        for event in events_to_send {
            let _ = self.event_tx.try_send(event);
        }

        // Check if enough workers remain.
        let active_count = self.active_worker_count();
        if active_count < self.config.min_workers {
            let _ = self.event_tx.try_send(SessionEvent::InsufficientWorkers {
                active: active_count,
                required: self.config.min_workers,
            });
        }

        removed
    }

    /// Returns the number of active (available) workers.
    pub fn active_worker_count(&self) -> usize {
        self.workers.values().filter(|w| w.is_available()).count()
    }

    /// Returns whether enough workers remain for training to continue.
    pub fn has_sufficient_workers(&self) -> bool {
        self.active_worker_count() >= self.config.min_workers
    }

    /// Returns active worker party indices.
    pub fn active_party_indices(&self) -> Vec<usize> {
        self.workers
            .values()
            .filter(|w| w.is_available())
            .map(|w| w.party_index)
            .collect()
    }

    /// Returns removed worker party indices.
    pub fn removed_party_indices(&self) -> Vec<usize> {
        self.workers
            .values()
            .filter(|w| matches!(w.state, WorkerSessionState::Removed { .. }))
            .map(|w| w.party_index)
            .collect()
    }

    /// Explicitly removes a worker (e.g., after cheater identification).
    pub fn remove_worker(&mut self, party_index: usize, reason: &str) {
        if let Some(worker) = self.workers.get_mut(&party_index) {
            worker.state = WorkerSessionState::Removed {
                reason: reason.to_string(),
            };

            let _ = self.event_tx.try_send(SessionEvent::WorkerRemoved {
                party_index,
                peer_id: worker.peer_id.clone(),
                reason: reason.to_string(),
            });
        }
    }

    /// Saves the session state to disk.
    pub fn save_snapshot(&self) -> Result<PathBuf, String> {
        let snapshot = SessionSnapshot {
            session_id: self.session_id.clone(),
            current_step: self.current_step,
            active_workers: self.active_party_indices(),
            removed_workers: self
                .workers
                .values()
                .filter_map(|w| {
                    if let WorkerSessionState::Removed { reason } = &w.state {
                        Some((w.party_index, reason.clone()))
                    } else {
                        None
                    }
                })
                .collect(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            checkpoints_submitted: self.checkpoints_submitted,
            total_steps_completed: self.workers.values().map(|w| w.steps_completed).sum(),
        };

        std::fs::create_dir_all(&self.config.persistence_dir)
            .map_err(|e| format!("failed to create persistence dir: {}", e))?;

        let filename = format!(
            "session_{}_step_{}.json",
            self.session_id,
            self.current_step,
        );
        let path = self.config.persistence_dir.join(&filename);

        let data = serde_json::to_vec_pretty(&snapshot)
            .map_err(|e| format!("serialization failed: {}", e))?;

        std::fs::write(&path, &data)
            .map_err(|e| format!("write failed: {}", e))?;

        info!(
            path = %path.display(),
            step = self.current_step,
            "Session snapshot saved"
        );

        let _ = self.event_tx.try_send(SessionEvent::StateSaved {
            path: path.clone(),
        });

        Ok(path)
    }

    /// Loads a session snapshot from disk.
    pub fn load_snapshot(path: &Path) -> Result<SessionSnapshot, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

        serde_json::from_slice(&data)
            .map_err(|e| format!("deserialization failed: {}", e))
    }

    /// Finds the most recent session snapshot in the persistence directory.
    pub fn find_latest_snapshot(dir: &Path, session_id: &str) -> Option<PathBuf> {
        let prefix = format!("session_{}_step_", session_id);

        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(&prefix)
            })
            .collect();

        entries.sort_by(|a, b| {
            let ta = a.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            let tb = b.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            tb.cmp(&ta)
        });

        entries.first().map(|e| e.path())
    }

    /// Initiates graceful shutdown.
    pub fn request_shutdown(&mut self) {
        if !self.shutdown_requested {
            info!(session = %self.session_id, "Graceful shutdown requested");
            self.shutdown_requested = true;
            let _ = self.event_tx.try_send(SessionEvent::ShutdownStarted);
        }
    }

    /// Returns whether shutdown has been requested.
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    /// Performs the shutdown procedure: save state, return final snapshot.
    pub fn execute_shutdown(&self) -> Result<PathBuf, String> {
        info!(session = %self.session_id, "Executing session shutdown");

        let path = self.save_snapshot()?;
        let _ = self.event_tx.try_send(SessionEvent::ShutdownCompleted);

        info!(
            session = %self.session_id,
            path = %path.display(),
            "Session shutdown complete"
        );

        Ok(path)
    }

    /// Returns the next event from the event channel (non-blocking).
    pub fn try_next_event(&mut self) -> Option<SessionEvent> {
        self.event_rx.try_recv().ok()
    }

    /// Returns worker info for a specific party.
    pub fn worker(&self, party_index: usize) -> Option<&WorkerInfo> {
        self.workers.get(&party_index)
    }

    /// Returns all worker info.
    pub fn all_workers(&self) -> &HashMap<usize, WorkerInfo> {
        &self.workers
    }

    /// Increments the checkpoint counter.
    pub fn record_checkpoint_submitted(&mut self) {
        self.checkpoints_submitted += 1;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_peer_id(index: usize) -> PeerId {
        PeerId::random()
    }

    #[test]
    fn test_session_creation() {
        let config = ResilientSessionConfig::for_testing();
        let session = ResilientSession::new("test-1".to_string(), config);

        assert_eq!(session.active_worker_count(), 0);
        assert!(!session.is_shutdown_requested());
    }

    #[test]
    fn test_register_workers() {
        let config = ResilientSessionConfig::for_testing();
        let mut session = ResilientSession::new("test-1".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);
        session.register_worker(test_peer_id(2), 2);

        assert_eq!(session.active_worker_count(), 3);
        assert!(session.has_sufficient_workers());
    }

    #[test]
    fn test_heartbeat_recording() {
        let config = ResilientSessionConfig::for_testing();
        let mut session = ResilientSession::new("test-1".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.record_heartbeat(0);

        let worker = session.worker(0).unwrap();
        assert_eq!(worker.messages_received, 1);
        assert!(worker.is_available());
    }

    #[test]
    fn test_worker_disconnect_and_grace_period() {
        let config = ResilientSessionConfig {
            disconnect_grace_period: Duration::from_millis(10),
            max_reconnect_attempts: 1,
            ..ResilientSessionConfig::for_testing()
        };
        let mut session = ResilientSession::new("test-1".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);
        session.register_worker(test_peer_id(2), 2);

        // Let grace period expire for worker 2.
        std::thread::sleep(Duration::from_millis(20));
        session.record_heartbeat(0);
        session.record_heartbeat(1);

        let removed = session.check_worker_health();
        // Worker 2 should be in grace period first.
        // First check moves to grace period (not immediately removed).
        assert_eq!(session.active_worker_count(), 2);

        // Wait again for the grace period to fully expire and allow removal.
        std::thread::sleep(Duration::from_millis(20));
        let removed = session.check_worker_health();
        // After max_reconnect_attempts=1 and another grace period, worker should be removed.
        assert!(removed.contains(&2));
        assert_eq!(session.active_worker_count(), 2);
    }

    #[test]
    fn test_worker_reconnection() {
        let config = ResilientSessionConfig {
            disconnect_grace_period: Duration::from_millis(10),
            max_reconnect_attempts: 5,
            ..ResilientSessionConfig::for_testing()
        };
        let mut session = ResilientSession::new("test-1".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);

        // Let worker 1 enter grace period.
        std::thread::sleep(Duration::from_millis(20));
        session.record_heartbeat(0);
        session.check_worker_health();

        // Worker 1 should be in grace period.
        let w1 = session.worker(1).unwrap();
        assert!(matches!(w1.state, WorkerSessionState::GracePeriod { .. }));

        // Worker 1 reconnects.
        session.record_heartbeat(1);
        let w1 = session.worker(1).unwrap();
        assert!(matches!(w1.state, WorkerSessionState::Reconnected));
        assert!(w1.is_available());
    }

    #[test]
    fn test_explicit_removal() {
        let config = ResilientSessionConfig::for_testing();
        let mut session = ResilientSession::new("test-1".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);
        session.register_worker(test_peer_id(2), 2);

        session.remove_worker(1, "cheater identified");

        assert_eq!(session.active_worker_count(), 2);
        let w = session.worker(1).unwrap();
        assert!(matches!(w.state, WorkerSessionState::Removed { .. }));
    }

    #[test]
    fn test_insufficient_workers() {
        let config = ResilientSessionConfig {
            min_workers: 3,
            ..ResilientSessionConfig::for_testing()
        };
        let mut session = ResilientSession::new("test-1".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);
        session.register_worker(test_peer_id(2), 2);

        assert!(session.has_sufficient_workers());

        session.remove_worker(2, "test");
        assert!(!session.has_sufficient_workers());
    }

    #[test]
    fn test_session_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let config = ResilientSessionConfig {
            persistence_dir: dir.path().to_path_buf(),
            ..ResilientSessionConfig::for_testing()
        };
        let mut session = ResilientSession::new("test-snap".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);
        session.update_step(42);

        let path = session.save_snapshot().unwrap();
        assert!(path.exists());

        let snapshot = ResilientSession::load_snapshot(&path).unwrap();
        assert_eq!(snapshot.session_id, "test-snap");
        assert_eq!(snapshot.current_step, 42);
        assert_eq!(snapshot.active_workers.len(), 2);
    }

    #[test]
    fn test_find_latest_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let config = ResilientSessionConfig {
            persistence_dir: dir.path().to_path_buf(),
            ..ResilientSessionConfig::for_testing()
        };
        let mut session = ResilientSession::new("test-latest".to_string(), config);

        session.register_worker(test_peer_id(0), 0);

        session.update_step(10);
        session.save_snapshot().unwrap();

        std::thread::sleep(Duration::from_millis(10));

        session.update_step(20);
        session.save_snapshot().unwrap();

        let latest = ResilientSession::find_latest_snapshot(dir.path(), "test-latest");
        assert!(latest.is_some());
        let path = latest.unwrap();
        assert!(path.to_string_lossy().contains("step_20"));
    }

    #[test]
    fn test_graceful_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let config = ResilientSessionConfig {
            persistence_dir: dir.path().to_path_buf(),
            ..ResilientSessionConfig::for_testing()
        };
        let mut session = ResilientSession::new("test-shutdown".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.update_step(50);

        assert!(!session.is_shutdown_requested());

        session.request_shutdown();
        assert!(session.is_shutdown_requested());

        let path = session.execute_shutdown().unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_update_step() {
        let config = ResilientSessionConfig::for_testing();
        let mut session = ResilientSession::new("test-step".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);

        session.update_step(1);
        session.update_step(2);
        session.update_step(3);

        let w = session.worker(0).unwrap();
        assert_eq!(w.steps_completed, 3);
    }

    #[test]
    fn test_checkpoint_counter() {
        let config = ResilientSessionConfig::for_testing();
        let mut session = ResilientSession::new("test-cp".to_string(), config);

        assert_eq!(session.checkpoints_submitted, 0);
        session.record_checkpoint_submitted();
        session.record_checkpoint_submitted();
        assert_eq!(session.checkpoints_submitted, 2);
    }

    #[test]
    fn test_active_party_indices() {
        let config = ResilientSessionConfig::for_testing();
        let mut session = ResilientSession::new("test-idx".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);
        session.register_worker(test_peer_id(2), 2);
        session.remove_worker(1, "test");

        let active = session.active_party_indices();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&0));
        assert!(active.contains(&2));
        assert!(!active.contains(&1));
    }

    #[test]
    fn test_removed_party_indices() {
        let config = ResilientSessionConfig::for_testing();
        let mut session = ResilientSession::new("test-rem".to_string(), config);

        session.register_worker(test_peer_id(0), 0);
        session.register_worker(test_peer_id(1), 1);
        session.remove_worker(1, "cheater");

        let removed = session.removed_party_indices();
        assert_eq!(removed, vec![1]);
    }
}
