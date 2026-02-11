//! Training Checkpoint Persistence
//!
//! Saves and restores [`TrainingState`] to disk so that training can resume
//! after a crash without replaying every step from scratch.
//!
//! Checkpoints are stored as JSON files under a configurable directory
//! (default: `~/.helix/checkpoints/{model_id}/`).  Each checkpoint records
//! the current weights, step number, model ID, last proof hash, and
//! cumulative error bound.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Serializable checkpoint written to disk after each successful training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Model identifier.
    pub model_id: u64,
    /// Training step that produced this checkpoint.
    pub step: u64,
    /// Current loss value.
    pub loss: f64,
    /// Cumulative error bound.
    pub error_bound: f64,
    /// Loss history up to this point.
    pub loss_history: Vec<f64>,
    /// Hash of the last submitted proof (hex string, may be empty).
    pub last_proof_hash: String,
    /// First-layer weights serialized as little-endian `[u8; 32]` chunks.
    pub w1_bytes: Vec<Vec<u8>>,
    /// First-layer biases.
    pub b1_bytes: Vec<Vec<u8>>,
    /// Second-layer weights.
    pub w2_bytes: Vec<Vec<u8>>,
    /// Second-layer biases.
    pub b2_bytes: Vec<Vec<u8>>,
}

/// Configuration for checkpoint storage.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Root directory for checkpoints.
    pub dir: PathBuf,
    /// Whether checkpointing is enabled.
    pub enabled: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        let dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".helix")
            .join("checkpoints");
        Self {
            dir,
            enabled: true,
        }
    }
}

impl CheckpointConfig {
    /// Build the path for a specific model's checkpoint file.
    pub fn checkpoint_path(&self, model_id: u64) -> PathBuf {
        self.dir.join(format!("{}", model_id)).join("checkpoint.json")
    }
}

/// Convert a field element to its canonical little-endian byte representation.
fn fr_to_bytes(fr: &helix_prover::halo2curves::bn256::Fr) -> Vec<u8> {
    use ff::PrimeField;
    fr.to_repr().as_ref().to_vec()
}

/// Reconstruct a field element from its little-endian byte representation.
fn fr_from_bytes(bytes: &[u8]) -> Result<helix_prover::halo2curves::bn256::Fr> {
    use ff::PrimeField;
    use helix_prover::halo2curves::bn256::Fr;

    let mut repr = <Fr as PrimeField>::Repr::default();
    let dst: &mut [u8] = repr.as_mut();
    if bytes.len() != dst.len() {
        anyhow::bail!(
            "invalid field element length: expected {}, got {}",
            dst.len(),
            bytes.len()
        );
    }
    dst.copy_from_slice(bytes);
    Option::from(Fr::from_repr(repr))
        .ok_or_else(|| anyhow::anyhow!("invalid field element encoding"))
}

impl TrainingCheckpoint {
    /// Create a checkpoint from the current [`super::real_training::TrainingState`].
    pub fn from_state(
        state: &super::real_training::TrainingState,
        model_id: u64,
        last_proof_hash: &str,
    ) -> Self {
        Self {
            model_id,
            step: state.step,
            loss: state.loss,
            error_bound: state.error_bound,
            loss_history: state.loss_history.clone(),
            last_proof_hash: last_proof_hash.to_string(),
            w1_bytes: state.w1.iter().map(fr_to_bytes).collect(),
            b1_bytes: state.b1.iter().map(fr_to_bytes).collect(),
            w2_bytes: state.w2.iter().map(fr_to_bytes).collect(),
            b2_bytes: state.b2.iter().map(fr_to_bytes).collect(),
        }
    }

    /// Reconstruct a [`super::real_training::TrainingState`] from this checkpoint.
    pub fn to_state(&self) -> Result<super::real_training::TrainingState> {
        let w1 = self
            .w1_bytes
            .iter()
            .map(|b| fr_from_bytes(b))
            .collect::<Result<Vec<_>>>()
            .context("failed to decode w1")?;
        let b1 = self
            .b1_bytes
            .iter()
            .map(|b| fr_from_bytes(b))
            .collect::<Result<Vec<_>>>()
            .context("failed to decode b1")?;
        let w2 = self
            .w2_bytes
            .iter()
            .map(|b| fr_from_bytes(b))
            .collect::<Result<Vec<_>>>()
            .context("failed to decode w2")?;
        let b2 = self
            .b2_bytes
            .iter()
            .map(|b| fr_from_bytes(b))
            .collect::<Result<Vec<_>>>()
            .context("failed to decode b2")?;

        Ok(super::real_training::TrainingState {
            w1,
            b1,
            w2,
            b2,
            step: self.step,
            loss: self.loss,
            error_bound: self.error_bound,
            loss_history: self.loss_history.clone(),
        })
    }

    /// Save this checkpoint to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create checkpoint dir {}", parent.display()))?;
        }

        let json = serde_json::to_string_pretty(self)
            .context("failed to serialize checkpoint")?;

        // Write to a temp file first, then rename for atomicity.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)
            .with_context(|| format!("failed to write checkpoint {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("failed to rename checkpoint to {}", path.display()))?;

        tracing::info!(
            "Checkpoint saved: step={}, model={}, path={}",
            self.step,
            self.model_id,
            path.display()
        );

        Ok(())
    }

    /// Load a checkpoint from disk.  Returns `Ok(None)` if the file does not
    /// exist.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }

        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read checkpoint {}", path.display()))?;

        let ckpt: Self = serde_json::from_str(&json)
            .with_context(|| format!("failed to parse checkpoint {}", path.display()))?;

        tracing::info!(
            "Checkpoint loaded: step={}, model={}, path={}",
            ckpt.step,
            ckpt.model_id,
            path.display()
        );

        Ok(Some(ckpt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demo::real_training::TrainingState;

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let state = TrainingState::new_random(8, 16, 4);
        let ckpt = TrainingCheckpoint::from_state(&state, 42, "0xdeadbeef");

        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test_checkpoint.json");

        ckpt.save(&path).unwrap();

        let loaded = TrainingCheckpoint::load(&path).unwrap().unwrap();
        assert_eq!(loaded.model_id, 42);
        assert_eq!(loaded.step, state.step);
        assert_eq!(loaded.last_proof_hash, "0xdeadbeef");
        assert_eq!(loaded.w1_bytes.len(), state.w1.len());
        assert_eq!(loaded.b1_bytes.len(), state.b1.len());
        assert_eq!(loaded.w2_bytes.len(), state.w2.len());
        assert_eq!(loaded.b2_bytes.len(), state.b2.len());

        // Verify field elements roundtrip correctly.
        let restored = loaded.to_state().unwrap();
        assert_eq!(restored.w1, state.w1);
        assert_eq!(restored.b1, state.b1);
        assert_eq!(restored.w2, state.w2);
        assert_eq!(restored.b2, state.b2);
        assert_eq!(restored.step, state.step);
    }

    #[test]
    fn test_checkpoint_load_nonexistent_returns_none() {
        let path = PathBuf::from("/tmp/nonexistent_helix_checkpoint_xyzzy.json");
        let result = TrainingCheckpoint::load(&path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_checkpoint_resume_from_correct_step() {
        // Simulate training for 3 steps, checkpointing at each step.
        let mut state = TrainingState::new_random(4, 8, 2);
        state.step = 3;
        state.loss = 1.5;
        state.error_bound = 12.3;
        state.loss_history = vec![2.5, 2.0, 1.5];

        let ckpt = TrainingCheckpoint::from_state(&state, 7, "0xcafe");

        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("resume_test.json");
        ckpt.save(&path).unwrap();

        // "Crash" and reload.
        let loaded = TrainingCheckpoint::load(&path).unwrap().unwrap();
        let restored = loaded.to_state().unwrap();

        // Training should resume from step 3.
        assert_eq!(restored.step, 3);
        assert!((restored.loss - 1.5).abs() < f64::EPSILON);
        assert!((restored.error_bound - 12.3).abs() < f64::EPSILON);
        assert_eq!(restored.loss_history, vec![2.5, 2.0, 1.5]);
    }
}
