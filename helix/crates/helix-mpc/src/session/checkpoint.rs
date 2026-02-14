//! MPC session checkpoint/resume and disk persistence.
//!
//! Enables saving MPC session state to disk so sessions can survive process
//! restarts. Checkpoints capture:
//!
//! - Session metadata (ID, config, phase, step number)
//! - Participant list with roles and status
//! - Beaver triple pool levels (scalar/vector/matrix counts)
//! - Model share existence flags (not the actual secret values, which would
//!   compromise security — shares are re-distributed on resume)
//!
//! # Security
//!
//! Actual secret shares and Beaver triples are NOT persisted to disk.
//! Persisting secret-shared values would make the on-disk data a single point
//! of compromise. On resume, shares must be re-distributed and triples
//! re-generated.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use crate::error::{MPCError, MPCResult};
use crate::types::{MPCConfig, MPCPhase, PartyId, PartyRole};

/// Serializable snapshot of an MPC session (no secrets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    /// Session ID.
    pub session_id: String,
    /// MPC configuration.
    pub config: MPCConfig,
    /// Current phase at checkpoint time.
    pub phase: MPCPhase,
    /// Current training step.
    pub current_step: u64,
    /// Participant state.
    pub participants: Vec<ParticipantCheckpoint>,
    /// Beaver triple pool levels per party (scalar count only — triples
    /// themselves are not persisted for security).
    pub pool_levels: Vec<PoolLevelCheckpoint>,
    /// Which parties had model shares at checkpoint time.
    pub has_model_shares: Vec<bool>,
    /// Unix timestamp (seconds) when checkpoint was created.
    pub created_at: u64,
    /// Checkpoint version for forward compatibility.
    pub version: u32,
}

/// Participant state in a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantCheckpoint {
    pub party_id: PartyId,
    pub index: usize,
    pub role: PartyRole,
    pub stake: u64,
    pub is_active: bool,
}

/// Beaver pool levels in a checkpoint (counts only, no secret values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolLevelCheckpoint {
    pub party_index: usize,
    pub scalar_count: usize,
    pub vector_dims: Vec<(usize, usize)>,   // (dim, count)
    pub matrix_dims: Vec<(usize, usize, usize, usize)>, // (m, k, n, count)
}

/// Current checkpoint version.
pub const CHECKPOINT_VERSION: u32 = 1;

impl SessionCheckpoint {
    /// Creates a checkpoint from the current session state.
    pub fn from_session(session: &super::manager::MPCSession) -> Self {
        let participants = session
            .participants
            .values()
            .map(|p| ParticipantCheckpoint {
                party_id: p.party_id.clone(),
                index: p.index,
                role: p.role,
                stake: p.stake,
                is_active: p.is_active,
            })
            .collect();

        let pool_levels = session
            .pools
            .iter()
            .enumerate()
            .map(|(i, pool)| PoolLevelCheckpoint {
                party_index: i,
                scalar_count: pool.scalar_available(),
                vector_dims: vec![], // Simplified — vector counts not tracked externally
                matrix_dims: vec![], // Simplified — matrix counts not tracked externally
            })
            .collect();

        let has_model_shares = session
            .model_shares
            .iter()
            .map(|s| s.is_some())
            .collect();

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            session_id: session.session_id.clone(),
            config: session.config.clone(),
            phase: session.phase,
            current_step: session.current_step,
            participants,
            pool_levels,
            has_model_shares,
            created_at,
            version: CHECKPOINT_VERSION,
        }
    }

    /// Serializes the checkpoint to JSON bytes.
    pub fn to_bytes(&self) -> MPCResult<Vec<u8>> {
        serde_json::to_vec_pretty(self).map_err(|e| {
            MPCError::SessionError(format!("checkpoint serialization failed: {}", e))
        })
    }

    /// Deserializes a checkpoint from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> MPCResult<Self> {
        let checkpoint: Self = serde_json::from_slice(data).map_err(|e| {
            MPCError::SessionError(format!("checkpoint deserialization failed: {}", e))
        })?;

        if checkpoint.version > CHECKPOINT_VERSION {
            return Err(MPCError::SessionError(format!(
                "checkpoint version {} is newer than supported version {}",
                checkpoint.version, CHECKPOINT_VERSION,
            )));
        }

        Ok(checkpoint)
    }

    /// Returns a summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "Checkpoint[session={}, phase={}, step={}, parties={}, created={}]",
            self.session_id,
            self.phase,
            self.current_step,
            self.participants.len(),
            self.created_at,
        )
    }
}

/// Manages session persistence to disk.
///
/// Stores checkpoints as JSON files in a configurable directory.
/// File naming: `{session_id}_step{step}.json` with a `latest` symlink.
pub struct SessionPersistence {
    base_dir: PathBuf,
    /// Maximum number of checkpoint files to retain per session.
    max_checkpoints: usize,
}

impl SessionPersistence {
    /// Creates a new persistence manager. Creates the directory if it doesn't exist.
    pub fn new(base_dir: impl Into<PathBuf>, max_checkpoints: usize) -> MPCResult<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir).map_err(|e| {
            MPCError::SessionError(format!(
                "failed to create checkpoint directory {}: {}",
                base_dir.display(),
                e,
            ))
        })?;

        Ok(Self {
            base_dir,
            max_checkpoints,
        })
    }

    /// Saves a checkpoint to disk.
    #[instrument(skip_all, level = "info", fields(
        session_id = %checkpoint.session_id,
        step = checkpoint.current_step,
    ))]
    pub fn save(&self, checkpoint: &SessionCheckpoint) -> MPCResult<PathBuf> {
        let filename = format!(
            "{}_step{}.json",
            sanitize_filename(&checkpoint.session_id),
            checkpoint.current_step,
        );
        let path = self.base_dir.join(&filename);

        let data = checkpoint.to_bytes()?;
        std::fs::write(&path, &data).map_err(|e| {
            MPCError::SessionError(format!("failed to write checkpoint to {}: {}", path.display(), e))
        })?;

        // Update the "latest" pointer.
        let latest_path = self.latest_path(&checkpoint.session_id);
        // Write latest file as a plain text file containing the checkpoint filename.
        let _ = std::fs::write(&latest_path, &filename);

        // Prune old checkpoints.
        self.prune_old_checkpoints(&checkpoint.session_id);

        tracing::info!("Saved checkpoint: {} -> {}", checkpoint.summary(), path.display());

        Ok(path)
    }

    /// Loads the latest checkpoint for a session.
    #[instrument(skip_all, level = "info", fields(session_id = session_id))]
    pub fn load_latest(&self, session_id: &str) -> MPCResult<Option<SessionCheckpoint>> {
        let latest_path = self.latest_path(session_id);

        let filename = match std::fs::read_to_string(&latest_path) {
            Ok(f) => f.trim().to_string(),
            Err(_) => {
                debug!(session_id = session_id, "No checkpoint found for session");
                return Ok(None);
            }
        };

        let path = self.base_dir.join(&filename);
        let checkpoint = self.load_from_path(&path)?;
        info!(
            session_id = session_id,
            step = checkpoint.current_step,
            "Checkpoint loaded"
        );
        Ok(Some(checkpoint))
    }

    /// Loads a checkpoint from a specific path.
    #[instrument(skip_all, level = "debug", fields(path = %path.display()))]
    pub fn load_from_path(&self, path: &Path) -> MPCResult<SessionCheckpoint> {
        let data = std::fs::read(path).map_err(|e| {
            MPCError::SessionError(format!(
                "failed to read checkpoint from {}: {}",
                path.display(),
                e,
            ))
        })?;

        SessionCheckpoint::from_bytes(&data)
    }

    /// Lists all checkpoint files for a session.
    pub fn list_checkpoints(&self, session_id: &str) -> MPCResult<Vec<PathBuf>> {
        let prefix = sanitize_filename(session_id);
        let mut checkpoints = Vec::new();

        let entries = std::fs::read_dir(&self.base_dir).map_err(|e| {
            MPCError::SessionError(format!(
                "failed to read checkpoint directory: {}",
                e,
            ))
        })?;

        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(".json") && !name.ends_with("_latest.json") {
                checkpoints.push(entry.path());
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }

    /// Deletes all checkpoints for a session.
    pub fn delete_session(&self, session_id: &str) -> MPCResult<usize> {
        let checkpoints = self.list_checkpoints(session_id)?;
        let count = checkpoints.len();

        for path in &checkpoints {
            let _ = std::fs::remove_file(path);
        }

        let latest = self.latest_path(session_id);
        let _ = std::fs::remove_file(latest);

        Ok(count)
    }

    /// Path for the "latest" pointer file.
    fn latest_path(&self, session_id: &str) -> PathBuf {
        self.base_dir
            .join(format!("{}_latest.json", sanitize_filename(session_id)))
    }

    /// Removes old checkpoints beyond the retention limit.
    fn prune_old_checkpoints(&self, session_id: &str) {
        if let Ok(mut checkpoints) = self.list_checkpoints(session_id) {
            if checkpoints.len() > self.max_checkpoints {
                // Sort by name (which includes step number) so oldest are first.
                checkpoints.sort();
                let to_remove = checkpoints.len() - self.max_checkpoints;
                for path in &checkpoints[..to_remove] {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
    }
}

/// Sanitizes a session ID for use as a filename component.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::manager::MPCSession;

    #[test]
    fn test_checkpoint_roundtrip() {
        let config = MPCConfig::three_party();
        let session = MPCSession::new(config, "test-session").unwrap();

        let checkpoint = SessionCheckpoint::from_session(&session);

        assert_eq!(checkpoint.session_id, "test-session");
        assert_eq!(checkpoint.phase, MPCPhase::Preprocessing);
        assert_eq!(checkpoint.current_step, 0);
        assert_eq!(checkpoint.version, CHECKPOINT_VERSION);

        // Serialize and deserialize.
        let bytes = checkpoint.to_bytes().unwrap();
        let restored = SessionCheckpoint::from_bytes(&bytes).unwrap();

        assert_eq!(restored.session_id, "test-session");
        assert_eq!(restored.phase, MPCPhase::Preprocessing);
        assert_eq!(restored.current_step, 0);
        assert_eq!(restored.config.num_parties, 3);
    }

    #[test]
    fn test_persistence_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let persistence = SessionPersistence::new(dir.path(), 5).unwrap();

        let config = MPCConfig::three_party();
        let session = MPCSession::new(config, "persist-test").unwrap();
        let checkpoint = SessionCheckpoint::from_session(&session);

        persistence.save(&checkpoint).unwrap();

        let loaded = persistence.load_latest("persist-test").unwrap().unwrap();
        assert_eq!(loaded.session_id, "persist-test");
        assert_eq!(loaded.current_step, 0);
    }

    #[test]
    fn test_persistence_multiple_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let persistence = SessionPersistence::new(dir.path(), 3).unwrap();

        let config = MPCConfig::three_party();
        let mut session = MPCSession::new(config, "multi-ckpt").unwrap();

        // Save checkpoints at steps 0, 1, 2, 3, 4.
        for _ in 0..5 {
            let checkpoint = SessionCheckpoint::from_session(&session);
            persistence.save(&checkpoint).unwrap();
            session.current_step += 1;
        }

        // Should only have 3 checkpoints (pruned).
        let checkpoints = persistence.list_checkpoints("multi-ckpt").unwrap();
        assert_eq!(checkpoints.len(), 3);

        // Latest should be step 4.
        let latest = persistence.load_latest("multi-ckpt").unwrap().unwrap();
        assert_eq!(latest.current_step, 4);
    }

    #[test]
    fn test_persistence_delete_session() {
        let dir = tempfile::tempdir().unwrap();
        let persistence = SessionPersistence::new(dir.path(), 10).unwrap();

        let config = MPCConfig::three_party();
        let session = MPCSession::new(config, "delete-me").unwrap();
        let checkpoint = SessionCheckpoint::from_session(&session);
        persistence.save(&checkpoint).unwrap();

        let deleted = persistence.delete_session("delete-me").unwrap();
        assert_eq!(deleted, 1);

        let loaded = persistence.load_latest("delete-me").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let persistence = SessionPersistence::new(dir.path(), 5).unwrap();

        let loaded = persistence.load_latest("nonexistent").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("session-123"), "session-123");
        assert_eq!(sanitize_filename("ses/sion:bad"), "ses_sion_bad");
        assert_eq!(sanitize_filename("test_ok"), "test_ok");
    }

    #[test]
    fn test_checkpoint_summary() {
        let config = MPCConfig::three_party();
        let session = MPCSession::new(config, "summary-test").unwrap();
        let checkpoint = SessionCheckpoint::from_session(&session);

        let summary = checkpoint.summary();
        assert!(summary.contains("summary-test"));
        assert!(summary.contains("Preprocessing"));
    }

    #[test]
    fn test_version_check() {
        let mut checkpoint = SessionCheckpoint {
            session_id: "test".into(),
            config: MPCConfig::three_party(),
            phase: MPCPhase::Preprocessing,
            current_step: 0,
            participants: vec![],
            pool_levels: vec![],
            has_model_shares: vec![],
            created_at: 0,
            version: 999,
        };

        let bytes = serde_json::to_vec(&checkpoint).unwrap();
        let result = SessionCheckpoint::from_bytes(&bytes);
        assert!(result.is_err());

        // Valid version works.
        checkpoint.version = CHECKPOINT_VERSION;
        let bytes = serde_json::to_vec(&checkpoint).unwrap();
        let result = SessionCheckpoint::from_bytes(&bytes);
        assert!(result.is_ok());
    }
}
