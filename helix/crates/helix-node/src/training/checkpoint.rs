//! Checkpoint Management.
//!
//! Automatic checkpointing with local storage, resumption, and Merkle verification.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use super::model::{ModelWeights, ModelGradient, ModelMetadata, ModelVersion};
use super::round::{RoundId, RoundSummary};

/// Checkpoint identifier.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CheckpointId(pub u64);

impl CheckpointId {
    /// Creates a new checkpoint ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Creates a checkpoint ID from timestamp.
    pub fn from_timestamp() -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self(ts)
    }
}

impl std::fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ckpt-{}", self.0)
    }
}

/// Checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint ID.
    pub id: CheckpointId,
    /// Model version at checkpoint.
    pub model_version: ModelVersion,
    /// Training iteration count.
    pub iteration: u64,
    /// Round ID at checkpoint.
    pub round_id: Option<RoundId>,
    /// Timestamp of creation.
    pub created_at: u64,
    /// Model commitment hash.
    pub commitment: [u8; 32],
    /// Path to weights file.
    pub weights_path: PathBuf,
    /// Path to optimizer state (if saved).
    pub optimizer_path: Option<PathBuf>,
    /// Additional metadata.
    pub extra: HashMap<String, String>,
}

/// Optimizer state for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Optimizer type (sgd, adam, etc.).
    pub optimizer_type: String,
    /// Learning rate.
    pub learning_rate: f64,
    /// First moment estimates (for Adam).
    pub m: Option<Vec<Vec<f32>>>,
    /// Second moment estimates (for Adam).
    pub v: Option<Vec<Vec<f32>>>,
    /// Momentum (for SGD with momentum).
    pub momentum_buffer: Option<Vec<Vec<f32>>>,
    /// Training step count.
    pub step: u64,
    /// Beta1 for Adam.
    pub beta1: Option<f64>,
    /// Beta2 for Adam.
    pub beta2: Option<f64>,
    /// Epsilon for Adam.
    pub eps: Option<f64>,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self {
            optimizer_type: "sgd".to_string(),
            learning_rate: 0.001,
            m: None,
            v: None,
            momentum_buffer: None,
            step: 0,
            beta1: None,
            beta2: None,
            eps: None,
        }
    }
}

/// Complete checkpoint data.
#[derive(Debug)]
pub struct Checkpoint {
    /// Checkpoint metadata.
    pub metadata: CheckpointMetadata,
    /// Model weights (loaded if present).
    pub weights: Option<ModelWeights>,
    /// Optimizer state (loaded if present).
    pub optimizer_state: Option<OptimizerState>,
}

/// Checkpoint manager configuration.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to store checkpoints.
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep.
    pub max_checkpoints: usize,
    /// Interval between checkpoints (in iterations).
    pub checkpoint_interval: u64,
    /// Whether to save optimizer state.
    pub save_optimizer: bool,
    /// Whether to verify checkpoints with Merkle proofs.
    pub verify_merkle: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            max_checkpoints: 5,
            checkpoint_interval: 100,
            save_optimizer: true,
            verify_merkle: true,
        }
    }
}

/// Checkpoint manager.
#[derive(Debug)]
pub struct CheckpointManager {
    /// Configuration.
    config: CheckpointConfig,
    /// Known checkpoints (metadata only).
    checkpoints: Vec<CheckpointMetadata>,
    /// Last checkpoint iteration.
    last_checkpoint_iter: u64,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager.
    pub fn new(config: CheckpointConfig) -> std::io::Result<Self> {
        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&config.checkpoint_dir)?;
        
        // Load existing checkpoints
        let checkpoints = Self::scan_checkpoints(&config.checkpoint_dir)?;
        
        Ok(Self {
            config,
            checkpoints,
            last_checkpoint_iter: 0,
        })
    }

    /// Scans directory for existing checkpoints.
    fn scan_checkpoints(dir: &Path) -> std::io::Result<Vec<CheckpointMetadata>> {
        let mut checkpoints = Vec::new();
        
        if !dir.exists() {
            return Ok(checkpoints);
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && path.extension().map_or(false, |e| e == "meta") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&content) {
                        checkpoints.push(meta);
                    }
                }
            }
        }

        // Sort by iteration
        checkpoints.sort_by_key(|c| c.iteration);
        
        Ok(checkpoints)
    }

    /// Checks if a checkpoint should be created at this iteration.
    pub fn should_checkpoint(&self, iteration: u64) -> bool {
        iteration > 0 && 
        iteration - self.last_checkpoint_iter >= self.config.checkpoint_interval
    }

    /// Creates a checkpoint.
    pub fn save(
        &mut self,
        model: &ModelWeights,
        optimizer_state: Option<&OptimizerState>,
        round_id: Option<RoundId>,
    ) -> Result<CheckpointId, CheckpointError> {
        let checkpoint_id = CheckpointId::from_timestamp();
        let iteration = model.metadata.training_iterations;

        // Create file paths
        let weights_filename = format!("{}.weights.json", checkpoint_id);
        let weights_path = self.config.checkpoint_dir.join(&weights_filename);
        
        let optimizer_path = if self.config.save_optimizer && optimizer_state.is_some() {
            let opt_filename = format!("{}.optimizer.json", checkpoint_id);
            Some(self.config.checkpoint_dir.join(&opt_filename))
        } else {
            None
        };

        // Save weights
        model.save(&weights_path)
            .map_err(|e| CheckpointError::SaveFailed(e.to_string()))?;

        // Save optimizer state
        if let (Some(path), Some(state)) = (&optimizer_path, optimizer_state) {
            let json = serde_json::to_string(state)
                .map_err(|e| CheckpointError::SaveFailed(e.to_string()))?;
            std::fs::write(path, json)
                .map_err(|e| CheckpointError::SaveFailed(e.to_string()))?;
        }

        // Create metadata
        let metadata = CheckpointMetadata {
            id: checkpoint_id,
            model_version: model.metadata.version,
            iteration,
            round_id,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            commitment: model.metadata.commitment,
            weights_path: weights_path.clone(),
            optimizer_path: optimizer_path.clone(),
            extra: HashMap::new(),
        };

        // Save metadata
        let meta_filename = format!("{}.meta", checkpoint_id);
        let meta_path = self.config.checkpoint_dir.join(&meta_filename);
        let meta_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| CheckpointError::SaveFailed(e.to_string()))?;
        std::fs::write(&meta_path, meta_json)
            .map_err(|e| CheckpointError::SaveFailed(e.to_string()))?;

        // Add to list and prune old checkpoints
        self.checkpoints.push(metadata);
        self.last_checkpoint_iter = iteration;
        self.prune_old_checkpoints()?;

        Ok(checkpoint_id)
    }

    /// Loads the latest checkpoint.
    pub fn load_latest(&self) -> Result<Option<Checkpoint>, CheckpointError> {
        if let Some(meta) = self.checkpoints.last() {
            self.load(meta.id)
        } else {
            Ok(None)
        }
    }

    /// Loads a specific checkpoint.
    pub fn load(&self, id: CheckpointId) -> Result<Option<Checkpoint>, CheckpointError> {
        let meta = self.checkpoints.iter().find(|c| c.id == id);
        
        let meta = match meta {
            Some(m) => m.clone(),
            None => return Ok(None),
        };

        // Load weights
        let weights = ModelWeights::load(&meta.weights_path)
            .map_err(|e| CheckpointError::LoadFailed(e.to_string()))?;

        // Verify commitment
        if self.config.verify_merkle && weights.metadata.commitment != meta.commitment {
            return Err(CheckpointError::CommitmentMismatch {
                expected: meta.commitment,
                got: weights.metadata.commitment,
            });
        }

        // Load optimizer state
        let optimizer_state = if let Some(opt_path) = &meta.optimizer_path {
            if opt_path.exists() {
                let json = std::fs::read_to_string(opt_path)
                    .map_err(|e| CheckpointError::LoadFailed(e.to_string()))?;
                Some(serde_json::from_str(&json)
                    .map_err(|e| CheckpointError::LoadFailed(e.to_string()))?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Some(Checkpoint {
            metadata: meta,
            weights: Some(weights),
            optimizer_state,
        }))
    }

    /// Lists all available checkpoints.
    pub fn list(&self) -> &[CheckpointMetadata] {
        &self.checkpoints
    }

    /// Returns the latest checkpoint metadata.
    pub fn latest(&self) -> Option<&CheckpointMetadata> {
        self.checkpoints.last()
    }

    /// Prunes old checkpoints to stay within max_checkpoints limit.
    fn prune_old_checkpoints(&mut self) -> Result<(), CheckpointError> {
        while self.checkpoints.len() > self.config.max_checkpoints {
            let oldest = self.checkpoints.remove(0);
            
            // Delete files
            let _ = std::fs::remove_file(&oldest.weights_path);
            if let Some(opt_path) = &oldest.optimizer_path {
                let _ = std::fs::remove_file(opt_path);
            }
            
            // Delete metadata file
            let meta_filename = format!("{}.meta", oldest.id);
            let meta_path = self.config.checkpoint_dir.join(&meta_filename);
            let _ = std::fs::remove_file(meta_path);
        }
        
        Ok(())
    }

    /// Deletes a specific checkpoint.
    pub fn delete(&mut self, id: CheckpointId) -> Result<(), CheckpointError> {
        let idx = self.checkpoints.iter().position(|c| c.id == id);
        
        if let Some(idx) = idx {
            let meta = self.checkpoints.remove(idx);
            
            let _ = std::fs::remove_file(&meta.weights_path);
            if let Some(opt_path) = &meta.optimizer_path {
                let _ = std::fs::remove_file(opt_path);
            }
            
            let meta_filename = format!("{}.meta", meta.id);
            let meta_path = self.config.checkpoint_dir.join(&meta_filename);
            let _ = std::fs::remove_file(meta_path);
            
            Ok(())
        } else {
            Err(CheckpointError::NotFound(id))
        }
    }
}

/// Checkpoint errors.
#[derive(Debug)]
pub enum CheckpointError {
    /// Failed to save checkpoint.
    SaveFailed(String),
    /// Failed to load checkpoint.
    LoadFailed(String),
    /// Checkpoint not found.
    NotFound(CheckpointId),
    /// Commitment mismatch (possible corruption).
    CommitmentMismatch {
        expected: [u8; 32],
        got: [u8; 32],
    },
    /// IO error.
    IoError(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SaveFailed(msg) => write!(f, "Failed to save checkpoint: {}", msg),
            Self::LoadFailed(msg) => write!(f, "Failed to load checkpoint: {}", msg),
            Self::NotFound(id) => write!(f, "Checkpoint not found: {}", id),
            Self::CommitmentMismatch { expected, got } => {
                write!(f, "Commitment mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for CheckpointError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_id() {
        let id = CheckpointId::new(12345);
        assert_eq!(format!("{}", id), "ckpt-12345");
    }

    #[test]
    fn test_checkpoint_manager() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: dir.path().to_path_buf(),
            max_checkpoints: 3,
            checkpoint_interval: 10,
            save_optimizer: false,
            verify_merkle: true,
        };

        let mut manager = CheckpointManager::new(config).unwrap();

        // Create a model
        let metadata = super::super::model::ModelMetadata {
            name: "test".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 1,
            vocab_size: 10,
            ..Default::default()
        };
        let model = super::super::model::ModelWeights::random(metadata, 42);

        // Save checkpoint
        let id = manager.save(&model, None, None).unwrap();
        assert!(manager.list().len() == 1);

        // Load checkpoint
        let loaded = manager.load(id).unwrap().unwrap();
        assert!(loaded.weights.is_some());
        assert_eq!(loaded.metadata.id, id);
    }
}
