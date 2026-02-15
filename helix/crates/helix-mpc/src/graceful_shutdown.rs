//! Graceful shutdown for MPC training sessions.
//!
//! When a SIGTERM or SIGINT is received, the system should:
//!
//! 1. Stop accepting new training steps.
//! 2. Finish the current in-progress step (if any).
//! 3. Save an emergency checkpoint to disk.
//! 4. Notify all peers that this party is shutting down.
//! 5. Clean up resources and exit.
//!
//! The [`GracefulShutdown`] uses a `tokio::sync::watch` channel to broadcast
//! the shutdown signal to all components.
//!
//! # Usage
//!
//! ```ignore
//! let shutdown = GracefulShutdown::new();
//! let rx = shutdown.subscribe();
//!
//! // In the training loop:
//! loop {
//!     if shutdown.is_shutting_down() {
//!         break;
//!     }
//!     trainer.training_step_with_mac(...).await?;
//! }
//!
//! // On shutdown:
//! shutdown.emergency_checkpoint(&trainer, "/tmp/checkpoint").await?;
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tracing::{info, warn};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;

// ============================================================================
// Graceful Shutdown
// ============================================================================

/// Manages graceful shutdown for MPC training sessions.
///
/// Uses a `watch` channel to broadcast shutdown signals to all subscribers.
/// The shutdown can be triggered by:
/// - SIGINT/SIGTERM (via `listen_for_signals()`)
/// - Explicit call to `trigger()`
/// - MAC failure in the training loop
#[derive(Clone)]
pub struct GracefulShutdown {
    /// Watch channel sender (triggers shutdown).
    tx: Arc<watch::Sender<bool>>,
    /// Whether shutdown has been triggered.
    shutting_down: Arc<AtomicBool>,
}

impl GracefulShutdown {
    /// Creates a new graceful shutdown coordinator.
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(false);
        Self {
            tx: Arc::new(tx),
            shutting_down: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Subscribes to the shutdown signal.
    ///
    /// Returns a `watch::Receiver<bool>` that becomes `true` when shutdown
    /// is triggered. Components should check this receiver periodically.
    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.tx.subscribe()
    }

    /// Triggers the shutdown signal.
    ///
    /// All subscribers will see `true` on their receivers.
    pub fn trigger(&self) {
        if !self.shutting_down.swap(true, Ordering::SeqCst) {
            info!("Graceful shutdown triggered");
            let _ = self.tx.send(true);
        }
    }

    /// Returns whether shutdown has been triggered.
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::SeqCst)
    }

    /// Listens for SIGINT (Ctrl+C) and triggers shutdown.
    ///
    /// This spawns a background task that waits for Ctrl+C and then calls
    /// `trigger()`. Returns immediately.
    pub fn listen_for_signals(&self) {
        let shutdown = self.clone();
        tokio::spawn(async move {
            match tokio::signal::ctrl_c().await {
                Ok(()) => {
                    warn!("Received SIGINT/SIGTERM — initiating graceful shutdown");
                    shutdown.trigger();
                }
                Err(e) => {
                    warn!("Failed to listen for ctrl_c signal: {}", e);
                }
            }
        });
    }

    /// Saves an emergency checkpoint to disk.
    ///
    /// This serializes the current training state (weight shares, step number,
    /// MAC state) to a JSON file at the given path. The checkpoint does NOT
    /// contain secret values in a way that compromises security — it stores
    /// the individual party's shares only.
    ///
    /// # Arguments
    ///
    /// * `state` - The emergency checkpoint state to save.
    /// * `checkpoint_dir` - Directory to save the checkpoint file in.
    ///
    /// # Returns
    ///
    /// The path to the saved checkpoint file.
    pub async fn emergency_checkpoint(
        &self,
        state: &EmergencyCheckpointState,
        checkpoint_dir: &Path,
    ) -> MPCResult<PathBuf> {
        info!(
            step = state.step,
            party = state.party_index,
            "Saving emergency checkpoint"
        );

        // Ensure directory exists.
        std::fs::create_dir_all(checkpoint_dir).map_err(|e| {
            MPCError::SessionError(format!(
                "failed to create checkpoint directory {}: {}",
                checkpoint_dir.display(),
                e,
            ))
        })?;

        let filename = format!(
            "emergency_party{}_step{}.json",
            state.party_index,
            state.step,
        );
        let path = checkpoint_dir.join(&filename);

        let data = serde_json::to_vec_pretty(state).map_err(|e| {
            MPCError::SessionError(format!("emergency checkpoint serialization failed: {}", e))
        })?;

        std::fs::write(&path, &data).map_err(|e| {
            MPCError::SessionError(format!(
                "failed to write emergency checkpoint to {}: {}",
                path.display(),
                e,
            ))
        })?;

        info!(
            path = %path.display(),
            "Emergency checkpoint saved"
        );

        Ok(path)
    }
}

impl Default for GracefulShutdown {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Emergency Checkpoint State
// ============================================================================

/// Serializable state for an emergency checkpoint.
///
/// Contains this party's shares and metadata. Does NOT reconstruct the
/// full weights — each party saves only its own shares.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyCheckpointState {
    /// Session identifier.
    pub session_id: String,
    /// This party's index.
    pub party_index: usize,
    /// Current training step.
    pub step: u64,
    /// Number of weight elements per layer.
    pub w1_len: usize,
    pub b1_len: usize,
    pub w2_len: usize,
    pub b2_len: usize,
    /// Serialized weight shares (as byte vectors for portability).
    pub w1_bytes: Vec<u8>,
    pub b1_bytes: Vec<u8>,
    pub w2_bytes: Vec<u8>,
    pub b2_bytes: Vec<u8>,
    /// Whether MAC state was active.
    pub mac_active: bool,
    /// Serialized MAC shares (empty if MAC not active).
    pub w1_mac_bytes: Vec<u8>,
    pub b1_mac_bytes: Vec<u8>,
    pub w2_mac_bytes: Vec<u8>,
    pub b2_mac_bytes: Vec<u8>,
    /// Beaver triple cursor position.
    pub beaver_cursor: usize,
    /// Authenticated Beaver triple cursor position.
    pub auth_beaver_cursor: usize,
    /// Reason for the emergency checkpoint.
    pub reason: String,
}

impl EmergencyCheckpointState {
    /// Creates an emergency checkpoint state from raw field element vectors.
    pub fn from_shares(
        session_id: &str,
        party_index: usize,
        step: u64,
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        w1_macs: &[Fr],
        b1_macs: &[Fr],
        w2_macs: &[Fr],
        b2_macs: &[Fr],
        beaver_cursor: usize,
        auth_beaver_cursor: usize,
        reason: &str,
    ) -> Self {
        Self {
            session_id: session_id.to_string(),
            party_index,
            step,
            w1_len: w1.len(),
            b1_len: b1.len(),
            w2_len: w2.len(),
            b2_len: b2.len(),
            w1_bytes: serialize_fr_vec(w1),
            b1_bytes: serialize_fr_vec(b1),
            w2_bytes: serialize_fr_vec(w2),
            b2_bytes: serialize_fr_vec(b2),
            mac_active: !w1_macs.is_empty(),
            w1_mac_bytes: serialize_fr_vec(w1_macs),
            b1_mac_bytes: serialize_fr_vec(b1_macs),
            w2_mac_bytes: serialize_fr_vec(w2_macs),
            b2_mac_bytes: serialize_fr_vec(b2_macs),
            beaver_cursor,
            auth_beaver_cursor,
            reason: reason.to_string(),
        }
    }

    /// Deserializes the weight shares back to Fr vectors.
    pub fn restore_shares(&self) -> MPCResult<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> {
        Ok((
            deserialize_fr_vec(&self.w1_bytes, self.w1_len)?,
            deserialize_fr_vec(&self.b1_bytes, self.b1_len)?,
            deserialize_fr_vec(&self.w2_bytes, self.w2_len)?,
            deserialize_fr_vec(&self.b2_bytes, self.b2_len)?,
        ))
    }

    /// Deserializes the MAC shares back to Fr vectors.
    pub fn restore_macs(&self) -> MPCResult<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> {
        if !self.mac_active {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        Ok((
            deserialize_fr_vec(&self.w1_mac_bytes, self.w1_len)?,
            deserialize_fr_vec(&self.b1_mac_bytes, self.b1_len)?,
            deserialize_fr_vec(&self.w2_mac_bytes, self.w2_len)?,
            deserialize_fr_vec(&self.b2_mac_bytes, self.b2_len)?,
        ))
    }

    /// Returns a summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "EmergencyCheckpoint[session={}, party={}, step={}, mac={}, reason={}]",
            self.session_id,
            self.party_index,
            self.step,
            self.mac_active,
            self.reason,
        )
    }
}

// ============================================================================
// Serialization Helpers
// ============================================================================

/// Serializes a vector of Fr elements to bytes.
fn serialize_fr_vec(values: &[Fr]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 32);
    for v in values {
        bytes.extend_from_slice(&v.to_bytes_le());
    }
    bytes
}

/// Deserializes a vector of Fr elements from bytes.
fn deserialize_fr_vec(bytes: &[u8], expected_len: usize) -> MPCResult<Vec<Fr>> {
    if bytes.len() != expected_len * 32 {
        return Err(MPCError::ProtocolError(format!(
            "expected {} bytes for {} Fr elements, got {}",
            expected_len * 32,
            expected_len,
            bytes.len(),
        )));
    }

    let mut result = Vec::with_capacity(expected_len);
    for i in 0..expected_len {
        let chunk = &bytes[i * 32..(i + 1) * 32];
        let mut arr = [0u8; 32];
        arr.copy_from_slice(chunk);
        result.push(Fr::from_bytes_le(&arr));
    }
    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graceful_shutdown_creation() {
        let shutdown = GracefulShutdown::new();
        assert!(!shutdown.is_shutting_down());
    }

    #[test]
    fn test_graceful_shutdown_trigger() {
        let shutdown = GracefulShutdown::new();
        let mut rx = shutdown.subscribe();

        assert!(!shutdown.is_shutting_down());
        shutdown.trigger();
        assert!(shutdown.is_shutting_down());

        // The watch receiver should have the new value.
        assert_eq!(*rx.borrow(), true);
    }

    #[test]
    fn test_graceful_shutdown_multiple_triggers() {
        let shutdown = GracefulShutdown::new();

        shutdown.trigger();
        shutdown.trigger(); // Idempotent.
        assert!(shutdown.is_shutting_down());
    }

    #[test]
    fn test_graceful_shutdown_multiple_subscribers() {
        let shutdown = GracefulShutdown::new();
        let rx1 = shutdown.subscribe();
        let rx2 = shutdown.subscribe();

        shutdown.trigger();
        assert_eq!(*rx1.borrow(), true);
        assert_eq!(*rx2.borrow(), true);
    }

    #[test]
    fn test_graceful_shutdown_clone() {
        let shutdown = GracefulShutdown::new();
        let shutdown2 = shutdown.clone();

        shutdown2.trigger();
        assert!(shutdown.is_shutting_down());
    }

    #[tokio::test]
    async fn test_emergency_checkpoint_save_load() {
        let shutdown = GracefulShutdown::new();
        let dir = tempfile::tempdir().unwrap();

        let w1 = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
        let b1 = vec![Fr::from_f64(0.1)];
        let w2 = vec![Fr::from_f64(0.5)];
        let b2 = vec![Fr::from_f64(0.01)];
        let w1_macs = vec![Fr::from_f64(10.0), Fr::from_f64(20.0)];
        let b1_macs = vec![Fr::from_f64(1.0)];
        let w2_macs = vec![Fr::from_f64(5.0)];
        let b2_macs = vec![Fr::from_f64(0.1)];

        let state = EmergencyCheckpointState::from_shares(
            "test-session",
            0,
            42,
            &w1, &b1, &w2, &b2,
            &w1_macs, &b1_macs, &w2_macs, &b2_macs,
            100, 50,
            "test shutdown",
        );

        let path = shutdown.emergency_checkpoint(&state, dir.path()).await.unwrap();
        assert!(path.exists());

        // Read it back.
        let data = std::fs::read(&path).unwrap();
        let restored: EmergencyCheckpointState = serde_json::from_slice(&data).unwrap();

        assert_eq!(restored.session_id, "test-session");
        assert_eq!(restored.party_index, 0);
        assert_eq!(restored.step, 42);
        assert!(restored.mac_active);
        assert_eq!(restored.beaver_cursor, 100);
        assert_eq!(restored.auth_beaver_cursor, 50);
        assert_eq!(restored.reason, "test shutdown");

        // Verify weight share deserialization.
        let (rw1, rb1, rw2, rb2) = restored.restore_shares().unwrap();
        assert_eq!(rw1.len(), 2);
        assert!((rw1[0].to_f64() - 1.0).abs() < 1e-6);
        assert!((rw1[1].to_f64() - 2.0).abs() < 1e-6);
        assert!((rb1[0].to_f64() - 0.1).abs() < 1e-6);

        // Verify MAC share deserialization.
        let (rm1, rmb1, rm2, rmb2) = restored.restore_macs().unwrap();
        assert_eq!(rm1.len(), 2);
        assert!((rm1[0].to_f64() - 10.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_emergency_checkpoint_no_macs() {
        let shutdown = GracefulShutdown::new();
        let dir = tempfile::tempdir().unwrap();

        let state = EmergencyCheckpointState::from_shares(
            "no-mac-session",
            1,
            5,
            &[Fr::from_f64(1.0)],
            &[Fr::from_f64(0.1)],
            &[Fr::from_f64(0.5)],
            &[Fr::from_f64(0.01)],
            &[], &[], &[], &[],
            10, 0,
            "planned shutdown",
        );

        assert!(!state.mac_active);

        let path = shutdown.emergency_checkpoint(&state, dir.path()).await.unwrap();
        assert!(path.exists());

        let data = std::fs::read(&path).unwrap();
        let restored: EmergencyCheckpointState = serde_json::from_slice(&data).unwrap();
        assert!(!restored.mac_active);

        let (m1, m2, m3, m4) = restored.restore_macs().unwrap();
        assert!(m1.is_empty());
    }

    #[test]
    fn test_serialize_deserialize_fr_vec() {
        let values = vec![Fr::from_f64(1.5), Fr::from_f64(-2.3), Fr::from_f64(0.0)];
        let bytes = serialize_fr_vec(&values);
        assert_eq!(bytes.len(), 96); // 3 * 32 bytes

        let restored = deserialize_fr_vec(&bytes, 3).unwrap();
        assert_eq!(restored.len(), 3);
        for (orig, rest) in values.iter().zip(restored.iter()) {
            let diff = (orig.to_f64() - rest.to_f64()).abs();
            assert!(diff < 1e-6, "Fr roundtrip failed");
        }
    }

    #[test]
    fn test_deserialize_fr_vec_wrong_length() {
        let result = deserialize_fr_vec(&[0u8; 64], 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_emergency_checkpoint_summary() {
        let state = EmergencyCheckpointState::from_shares(
            "s1", 0, 10,
            &[Fr::ZERO], &[], &[], &[],
            &[], &[], &[], &[],
            0, 0, "test",
        );
        let summary = state.summary();
        assert!(summary.contains("s1"));
        assert!(summary.contains("step=10"));
    }
}
