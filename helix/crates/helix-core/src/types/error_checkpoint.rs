//! Error bound checkpointing for long training runs.
//!
//! This module provides mechanisms to save and restore error state during training,
//! enabling:
//! - Recovery from failures without losing error tracking
//! - Periodic snapshots of error budget consumption
//! - Analysis of error evolution over training
//! - Distributed error state synchronization

use super::probabilistic_error::ProbabilisticError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A checkpoint of error state at a point in training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCheckpoint {
    /// Unique checkpoint identifier.
    pub checkpoint_id: String,
    /// Training step when checkpoint was created.
    pub step: usize,
    /// Timestamp (Unix milliseconds).
    pub timestamp_ms: u64,
    /// Total accumulated error.
    pub total_error: ProbabilisticError,
    /// Error by layer/component.
    pub layer_errors: HashMap<String, LayerErrorState>,
    /// Error budget state.
    pub budget_state: ErrorBudgetState,
    /// Precision state at checkpoint.
    pub precision_state: PrecisionCheckpointState,
    /// Training statistics.
    pub training_stats: TrainingCheckpointStats,
    /// Metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Version for forward compatibility.
    pub version: u32,
}

/// Error state for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerErrorState {
    /// Layer name.
    pub name: String,
    /// Layer type (e.g., "attention", "feedforward").
    pub layer_type: String,
    /// Accumulated error for forward pass.
    pub forward_error: ProbabilisticError,
    /// Accumulated error for backward pass.
    pub backward_error: ProbabilisticError,
    /// Precision used for this layer.
    pub precision: String,
    /// Number of operations in this layer.
    pub operation_count: usize,
}

/// Error budget state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBudgetState {
    /// Total budget allocated.
    pub total_budget: f64,
    /// Budget consumed so far.
    pub consumed: f64,
    /// Budget remaining.
    pub remaining: f64,
    /// Budget consumed this epoch.
    pub epoch_consumed: f64,
    /// Average budget consumption per step.
    pub avg_per_step: f64,
    /// Projected total consumption at current rate.
    pub projected_total: f64,
}

impl ErrorBudgetState {
    /// Creates a new budget state.
    pub fn new(total_budget: f64) -> Self {
        Self {
            total_budget,
            consumed: 0.0,
            remaining: total_budget,
            epoch_consumed: 0.0,
            avg_per_step: 0.0,
            projected_total: 0.0,
        }
    }

    /// Updates the budget state after a step.
    pub fn consume(&mut self, amount: f64, current_step: usize, total_steps: usize) {
        self.consumed += amount;
        self.remaining = self.total_budget - self.consumed;
        self.epoch_consumed += amount;

        if current_step > 0 {
            self.avg_per_step = self.consumed / current_step as f64;
            self.projected_total = self.avg_per_step * total_steps as f64;
        }
    }

    /// Resets epoch consumption.
    pub fn new_epoch(&mut self) {
        self.epoch_consumed = 0.0;
    }

    /// Checks if budget is exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.remaining < 0.0
    }

    /// Returns the fraction of budget consumed.
    pub fn consumed_fraction(&self) -> f64 {
        if self.total_budget == 0.0 {
            return 0.0;
        }
        self.consumed / self.total_budget
    }
}

/// Precision state at checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionCheckpointState {
    /// Current forward precision.
    pub forward_precision: String,
    /// Current backward precision.
    pub backward_precision: String,
    /// Current weight update precision.
    pub update_precision: String,
    /// Number of precision increases.
    pub precision_increases: usize,
    /// Number of precision decreases.
    pub precision_decreases: usize,
    /// Layer-specific precision overrides.
    pub layer_precisions: HashMap<String, String>,
}

/// Training statistics at checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpointStats {
    /// Current epoch.
    pub epoch: usize,
    /// Steps in current epoch.
    pub steps_in_epoch: usize,
    /// Current loss.
    pub loss: f64,
    /// Moving average loss.
    pub moving_avg_loss: f64,
    /// Gradient norm.
    pub grad_norm: f64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of unstable steps.
    pub unstable_steps: usize,
}

impl ErrorCheckpoint {
    /// Creates a new checkpoint.
    pub fn new(checkpoint_id: &str, step: usize) -> Self {
        Self {
            checkpoint_id: checkpoint_id.to_string(),
            step,
            timestamp_ms: current_timestamp_ms(),
            total_error: ProbabilisticError::zero(),
            layer_errors: HashMap::new(),
            budget_state: ErrorBudgetState::new(1.0),
            precision_state: PrecisionCheckpointState {
                forward_precision: "fp32".to_string(),
                backward_precision: "fp32".to_string(),
                update_precision: "fp32".to_string(),
                precision_increases: 0,
                precision_decreases: 0,
                layer_precisions: HashMap::new(),
            },
            training_stats: TrainingCheckpointStats {
                epoch: 0,
                steps_in_epoch: 0,
                loss: 0.0,
                moving_avg_loss: 0.0,
                grad_norm: 0.0,
                learning_rate: 0.0,
                unstable_steps: 0,
            },
            metadata: HashMap::new(),
            version: 1,
        }
    }

    /// Adds a layer error state.
    pub fn add_layer_error(&mut self, layer: LayerErrorState) {
        self.layer_errors.insert(layer.name.clone(), layer);
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Serializes to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serializes to pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Saves to a file.
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let json = self.to_json_pretty().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        std::fs::write(path, json)
    }

    /// Loads from a file.
    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }

    /// Computes the diff between this checkpoint and another.
    pub fn diff(&self, other: &ErrorCheckpoint) -> CheckpointDiff {
        CheckpointDiff {
            step_delta: other.step as i64 - self.step as i64,
            time_delta_ms: other.timestamp_ms as i64 - self.timestamp_ms as i64,
            error_delta: other.total_error.worst_case - self.total_error.worst_case,
            budget_consumed_delta: other.budget_state.consumed - self.budget_state.consumed,
            loss_delta: other.training_stats.loss - self.training_stats.loss,
            precision_changes: other.precision_state.precision_increases
                + other.precision_state.precision_decreases
                - self.precision_state.precision_increases
                - self.precision_state.precision_decreases,
        }
    }
}

/// Difference between two checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointDiff {
    /// Steps between checkpoints.
    pub step_delta: i64,
    /// Time between checkpoints (ms).
    pub time_delta_ms: i64,
    /// Error change.
    pub error_delta: f64,
    /// Budget consumed between checkpoints.
    pub budget_consumed_delta: f64,
    /// Loss change.
    pub loss_delta: f64,
    /// Number of precision changes.
    pub precision_changes: usize,
}

/// Manager for error checkpoints.
#[derive(Debug)]
pub struct CheckpointManager {
    /// Directory for storing checkpoints.
    checkpoint_dir: std::path::PathBuf,
    /// Checkpoint interval (steps).
    checkpoint_interval: usize,
    /// Maximum checkpoints to keep.
    max_checkpoints: usize,
    /// List of checkpoint IDs in order.
    checkpoint_ids: Vec<String>,
    /// Current checkpoint.
    current: Option<ErrorCheckpoint>,
    /// Prefix for checkpoint files.
    prefix: String,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager.
    pub fn new(checkpoint_dir: &Path, checkpoint_interval: usize, max_checkpoints: usize) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            checkpoint_interval,
            max_checkpoints,
            checkpoint_ids: Vec::new(),
            current: None,
            prefix: "error_checkpoint".to_string(),
        }
    }

    /// Sets the filename prefix.
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Checks if a checkpoint should be saved at this step.
    pub fn should_checkpoint(&self, step: usize) -> bool {
        step > 0 && step % self.checkpoint_interval == 0
    }

    /// Generates a checkpoint ID.
    fn generate_id(&self, step: usize) -> String {
        format!("{}_{:08}", self.prefix, step)
    }

    /// Gets the file path for a checkpoint.
    fn checkpoint_path(&self, id: &str) -> std::path::PathBuf {
        self.checkpoint_dir.join(format!("{}.json", id))
    }

    /// Saves a checkpoint.
    pub fn save(&mut self, checkpoint: ErrorCheckpoint) -> std::io::Result<()> {
        // Ensure directory exists
        std::fs::create_dir_all(&self.checkpoint_dir)?;

        let path = self.checkpoint_path(&checkpoint.checkpoint_id);
        checkpoint.save_to_file(&path)?;

        self.checkpoint_ids.push(checkpoint.checkpoint_id.clone());
        self.current = Some(checkpoint);

        // Prune old checkpoints
        self.prune()?;

        Ok(())
    }

    /// Prunes old checkpoints beyond the maximum.
    fn prune(&mut self) -> std::io::Result<()> {
        while self.checkpoint_ids.len() > self.max_checkpoints {
            if let Some(oldest_id) = self.checkpoint_ids.first().cloned() {
                let path = self.checkpoint_path(&oldest_id);
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }
                self.checkpoint_ids.remove(0);
            }
        }
        Ok(())
    }

    /// Loads the latest checkpoint.
    pub fn load_latest(&mut self) -> std::io::Result<Option<ErrorCheckpoint>> {
        if let Some(id) = self.checkpoint_ids.last() {
            let path = self.checkpoint_path(id);
            if path.exists() {
                let checkpoint = ErrorCheckpoint::load_from_file(&path)?;
                self.current = Some(checkpoint.clone());
                return Ok(Some(checkpoint));
            }
        }
        Ok(None)
    }

    /// Loads a specific checkpoint by ID.
    pub fn load(&self, checkpoint_id: &str) -> std::io::Result<Option<ErrorCheckpoint>> {
        let path = self.checkpoint_path(checkpoint_id);
        if path.exists() {
            Ok(Some(ErrorCheckpoint::load_from_file(&path)?))
        } else {
            Ok(None)
        }
    }

    /// Lists all checkpoint IDs.
    pub fn list_checkpoints(&self) -> &[String] {
        &self.checkpoint_ids
    }

    /// Scans the checkpoint directory for existing checkpoints.
    pub fn scan_existing(&mut self) -> std::io::Result<()> {
        if !self.checkpoint_dir.exists() {
            return Ok(());
        }

        let mut checkpoints: Vec<(usize, String)> = Vec::new();

        for entry in std::fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem.starts_with(&self.prefix) {
                        // Extract step number
                        if let Some(step_str) = stem.strip_prefix(&format!("{}_", self.prefix)) {
                            if let Ok(step) = step_str.parse::<usize>() {
                                checkpoints.push((step, stem.to_string()));
                            }
                        }
                    }
                }
            }
        }

        // Sort by step
        checkpoints.sort_by_key(|(step, _)| *step);
        self.checkpoint_ids = checkpoints.into_iter().map(|(_, id)| id).collect();

        Ok(())
    }

    /// Gets the current checkpoint.
    pub fn current(&self) -> Option<&ErrorCheckpoint> {
        self.current.as_ref()
    }

    /// Creates a new checkpoint at the given step.
    pub fn create_checkpoint(
        &self,
        step: usize,
        total_error: ProbabilisticError,
        budget_state: ErrorBudgetState,
    ) -> ErrorCheckpoint {
        let id = self.generate_id(step);
        let mut checkpoint = ErrorCheckpoint::new(&id, step);
        checkpoint.total_error = total_error;
        checkpoint.budget_state = budget_state;
        checkpoint
    }
}

/// Recovery options from a checkpoint.
#[derive(Debug, Clone)]
pub struct RecoveryOptions {
    /// Whether to restore exact error state.
    pub restore_error_state: bool,
    /// Whether to restore precision settings.
    pub restore_precision: bool,
    /// Whether to adjust budget based on consumed.
    pub adjust_budget: bool,
    /// Whether to log recovery details.
    pub verbose: bool,
}

impl Default for RecoveryOptions {
    fn default() -> Self {
        Self {
            restore_error_state: true,
            restore_precision: true,
            adjust_budget: true,
            verbose: true,
        }
    }
}

/// Result of checkpoint recovery.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery was successful.
    pub success: bool,
    /// Step resumed from.
    pub resumed_step: usize,
    /// Error state restored.
    pub restored_error: ProbabilisticError,
    /// Budget remaining after recovery.
    pub remaining_budget: f64,
    /// Messages/warnings during recovery.
    pub messages: Vec<String>,
}

/// Performs recovery from a checkpoint.
pub fn recover_from_checkpoint(
    checkpoint: &ErrorCheckpoint,
    options: &RecoveryOptions,
) -> RecoveryResult {
    let mut messages = Vec::new();

    if options.verbose {
        messages.push(format!(
            "Recovering from checkpoint {} at step {}",
            checkpoint.checkpoint_id, checkpoint.step
        ));
    }

    let restored_error = if options.restore_error_state {
        checkpoint.total_error.clone()
    } else {
        ProbabilisticError::zero()
    };

    let remaining_budget = if options.adjust_budget {
        checkpoint.budget_state.remaining
    } else {
        checkpoint.budget_state.total_budget
    };

    if options.verbose {
        messages.push(format!(
            "Restored error: {:.6}, remaining budget: {:.6}",
            restored_error.worst_case, remaining_budget
        ));
    }

    RecoveryResult {
        success: true,
        resumed_step: checkpoint.step,
        restored_error,
        remaining_budget,
        messages,
    }
}

/// Helper to get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = ErrorCheckpoint::new("test_001", 100);

        assert_eq!(checkpoint.checkpoint_id, "test_001");
        assert_eq!(checkpoint.step, 100);
        assert_eq!(checkpoint.version, 1);
    }

    #[test]
    fn test_checkpoint_serialization() {
        let mut checkpoint = ErrorCheckpoint::new("test_002", 200);
        checkpoint.total_error = ProbabilisticError::from_absolute(0.01);
        checkpoint.budget_state = ErrorBudgetState {
            total_budget: 0.1,
            consumed: 0.05,
            remaining: 0.05,
            epoch_consumed: 0.02,
            avg_per_step: 0.00025,
            projected_total: 0.05,
        };

        let json = checkpoint.to_json().unwrap();
        let restored = ErrorCheckpoint::from_json(&json).unwrap();

        assert_eq!(restored.checkpoint_id, "test_002");
        assert_eq!(restored.step, 200);
        assert!((restored.budget_state.consumed - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_checkpoint_diff() {
        let mut cp1 = ErrorCheckpoint::new("test_003", 100);
        cp1.total_error = ProbabilisticError::from_absolute(0.01);
        cp1.training_stats.loss = 1.0;

        let mut cp2 = ErrorCheckpoint::new("test_004", 200);
        cp2.total_error = ProbabilisticError::from_absolute(0.02);
        cp2.training_stats.loss = 0.5;

        let diff = cp1.diff(&cp2);

        assert_eq!(diff.step_delta, 100);
        assert!((diff.error_delta - 0.01).abs() < 1e-10);
        assert!((diff.loss_delta - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_budget_state() {
        let mut budget = ErrorBudgetState::new(1.0);

        budget.consume(0.1, 100, 1000);
        assert!((budget.consumed - 0.1).abs() < 1e-10);
        assert!((budget.remaining - 0.9).abs() < 1e-10);
        assert!((budget.projected_total - 1.0).abs() < 0.01);

        budget.new_epoch();
        assert_eq!(budget.epoch_consumed, 0.0);
    }

    #[test]
    fn test_checkpoint_manager() {
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path(), 100, 3);

        // Create and save checkpoints
        for step in [100, 200, 300, 400] {
            let checkpoint = manager.create_checkpoint(
                step,
                ProbabilisticError::from_absolute(0.01),
                ErrorBudgetState::new(1.0),
            );
            manager.save(checkpoint).unwrap();
        }

        // Should only keep 3
        assert_eq!(manager.list_checkpoints().len(), 3);

        // Latest should be step 400
        let latest = manager.load_latest().unwrap().unwrap();
        assert_eq!(latest.step, 400);
    }

    #[test]
    fn test_recovery() {
        let mut checkpoint = ErrorCheckpoint::new("test_recovery", 500);
        checkpoint.total_error = ProbabilisticError::from_absolute(0.05);
        checkpoint.budget_state = ErrorBudgetState {
            total_budget: 0.1,
            consumed: 0.05,
            remaining: 0.05,
            epoch_consumed: 0.01,
            avg_per_step: 0.0001,
            projected_total: 0.1,
        };

        let options = RecoveryOptions::default();
        let result = recover_from_checkpoint(&checkpoint, &options);

        assert!(result.success);
        assert_eq!(result.resumed_step, 500);
        assert!((result.remaining_budget - 0.05).abs() < 1e-10);
    }
}
