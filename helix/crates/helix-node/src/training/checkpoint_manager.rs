//! Checkpoint Manager — tracks checkpoint state and determines when proofs should be generated.
//!
//! Between checkpoints, training proceeds via MPC consensus only (no ZK proofs).
//! At checkpoint boundaries, the manager signals that a proof should be generated.
//! An early checkpoint can be forced if the accumulated error exceeds a threshold.

use halo2curves::bn256::Fr;
use halo2curves::ff::Field;
use tracing::{debug, info, warn};

/// Configuration for checkpoint-based proving.
#[derive(Debug, Clone)]
pub struct CheckpointProvingConfig {
    /// Checkpoint interval: generate proof every N steps.
    /// Must match the on-chain `modelCheckpointInterval`.
    pub checkpoint_interval: u64,
    /// Maximum accumulated error between checkpoints before forcing an early proof.
    /// Set to 0.0 to disable error-based early checkpoints.
    pub max_inter_checkpoint_error: f64,
}

impl Default for CheckpointProvingConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 1,
            max_inter_checkpoint_error: 0.0,
        }
    }
}

impl CheckpointProvingConfig {
    /// Creates a config with the given interval and no error threshold.
    pub fn with_interval(interval: u64) -> Self {
        Self {
            checkpoint_interval: interval.max(1),
            ..Default::default()
        }
    }

    /// Creates a config with both interval and error threshold.
    pub fn with_interval_and_error(interval: u64, max_error: f64) -> Self {
        Self {
            checkpoint_interval: interval.max(1),
            max_inter_checkpoint_error: max_error,
        }
    }
}

/// Saved weights at a checkpoint boundary.
#[derive(Debug, Clone)]
pub struct CheckpointWeights {
    /// Layer 1 weights.
    pub w1: Vec<Fr>,
    /// Layer 1 biases.
    pub b1: Vec<Fr>,
    /// Layer 2 weights.
    pub w2: Vec<Fr>,
    /// Layer 2 biases.
    pub b2: Vec<Fr>,
    /// Step number when this checkpoint was saved.
    pub saved_at_step: u64,
}

/// Action to take after recording a training step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepAction {
    /// Continue training without generating a proof.
    ContinueTraining,
    /// Generate a checkpoint proof covering the interval.
    GenerateProof {
        /// Step number when the checkpoint interval started.
        checkpoint_start_step: u64,
        /// Number of steps in this interval.
        steps_in_interval: u64,
        /// Total error accumulated during this interval.
        accumulated_error: f64,
    },
}

/// Manages checkpoint state for an aggregator.
///
/// Tracks:
/// - Weights at the last checkpoint boundary
/// - Steps since last checkpoint
/// - Accumulated error since last checkpoint
/// - Global step counter
///
/// Note: This is distinct from `training::CheckpointManager` which handles
/// model weight snapshots for resumption. This type manages the proving
/// schedule: when to generate ZK proofs vs continue training without proofs.
pub struct CheckpointProvingManager {
    /// Configuration.
    config: CheckpointProvingConfig,
    /// Weights saved at the last checkpoint boundary.
    checkpoint_weights: Option<CheckpointWeights>,
    /// Number of steps since the last checkpoint proof.
    steps_since_checkpoint: u64,
    /// Accumulated error since the last checkpoint.
    error_since_checkpoint: f64,
    /// Global step counter.
    global_step: u64,
}

impl CheckpointProvingManager {
    /// Creates a new checkpoint proving manager with the given config.
    pub fn new(config: CheckpointProvingConfig) -> Self {
        Self {
            config,
            checkpoint_weights: None,
            steps_since_checkpoint: 0,
            error_since_checkpoint: 0.0,
            global_step: 0,
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CheckpointProvingConfig {
        &self.config
    }

    /// Returns the current global step.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Returns steps since last checkpoint.
    pub fn steps_since_checkpoint(&self) -> u64 {
        self.steps_since_checkpoint
    }

    /// Returns accumulated error since last checkpoint.
    pub fn error_since_checkpoint(&self) -> f64 {
        self.error_since_checkpoint
    }

    /// Returns a reference to the checkpoint weights, if saved.
    pub fn checkpoint_weights(&self) -> Option<&CheckpointWeights> {
        self.checkpoint_weights.as_ref()
    }

    /// Saves checkpoint weights at the current step.
    /// Call this when starting a new checkpoint interval.
    pub fn save_checkpoint_weights(&mut self, w1: Vec<Fr>, b1: Vec<Fr>, w2: Vec<Fr>, b2: Vec<Fr>) {
        self.checkpoint_weights = Some(CheckpointWeights {
            w1,
            b1,
            w2,
            b2,
            saved_at_step: self.global_step,
        });
        self.steps_since_checkpoint = 0;
        self.error_since_checkpoint = 0.0;
        debug!(
            step = self.global_step,
            interval = self.config.checkpoint_interval,
            "Checkpoint weights saved"
        );
    }

    /// Records a completed training step and returns the action to take.
    ///
    /// Returns `StepAction::GenerateProof` when:
    /// - `steps_since_checkpoint >= checkpoint_interval`, OR
    /// - `error_since_checkpoint > max_inter_checkpoint_error` (if threshold is set)
    ///
    /// Returns `StepAction::ContinueTraining` otherwise.
    pub fn record_step(&mut self, step_error: f64) -> StepAction {
        self.global_step += 1;
        self.steps_since_checkpoint += 1;
        self.error_since_checkpoint += step_error;

        let interval_reached = self.steps_since_checkpoint >= self.config.checkpoint_interval;
        let error_exceeded = self.config.max_inter_checkpoint_error > 0.0
            && self.error_since_checkpoint > self.config.max_inter_checkpoint_error;

        if interval_reached || error_exceeded {
            if error_exceeded && !interval_reached {
                warn!(
                    step = self.global_step,
                    error = self.error_since_checkpoint,
                    threshold = self.config.max_inter_checkpoint_error,
                    "Early checkpoint triggered by error threshold"
                );
            }

            let checkpoint_start = self
                .checkpoint_weights
                .as_ref()
                .map(|cw| cw.saved_at_step)
                .unwrap_or(0);

            let action = StepAction::GenerateProof {
                checkpoint_start_step: checkpoint_start,
                steps_in_interval: self.steps_since_checkpoint,
                accumulated_error: self.error_since_checkpoint,
            };

            info!(
                step = self.global_step,
                interval_steps = self.steps_since_checkpoint,
                accumulated_error = self.error_since_checkpoint,
                "Checkpoint proof required"
            );

            action
        } else {
            debug!(
                step = self.global_step,
                steps_remaining = self.config.checkpoint_interval - self.steps_since_checkpoint,
                "Continue training (no checkpoint yet)"
            );
            StepAction::ContinueTraining
        }
    }

    /// Resets the checkpoint counter after a successful proof submission.
    /// Call this after the proof has been generated and submitted.
    pub fn checkpoint_completed(&mut self) {
        info!(
            step = self.global_step,
            "Checkpoint completed, resetting counters"
        );
        self.steps_since_checkpoint = 0;
        self.error_since_checkpoint = 0.0;
    }

    /// Returns true if this is a checkpoint boundary (proof should be generated).
    pub fn is_checkpoint_boundary(&self) -> bool {
        self.steps_since_checkpoint >= self.config.checkpoint_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CheckpointProvingConfig::default();
        assert_eq!(config.checkpoint_interval, 1);
        assert_eq!(config.max_inter_checkpoint_error, 0.0);
    }

    #[test]
    fn test_interval_one_every_step() {
        let mut mgr = CheckpointProvingManager::new(CheckpointProvingConfig::default());
        // With interval=1, every step should trigger a proof
        let action = mgr.record_step(0.01);
        assert!(matches!(action, StepAction::GenerateProof { .. }));
    }

    #[test]
    fn test_interval_five() {
        let mut mgr =
            CheckpointProvingManager::new(CheckpointProvingConfig::with_interval(5));

        // Steps 1-4 should continue
        for _ in 0..4 {
            assert_eq!(mgr.record_step(0.01), StepAction::ContinueTraining);
        }

        // Step 5 should trigger proof
        let action = mgr.record_step(0.01);
        match action {
            StepAction::GenerateProof {
                steps_in_interval,
                accumulated_error,
                ..
            } => {
                assert_eq!(steps_in_interval, 5);
                assert!((accumulated_error - 0.05).abs() < 1e-10);
            }
            _ => panic!("Expected GenerateProof"),
        }
    }

    #[test]
    fn test_error_threshold_early_checkpoint() {
        let mut mgr = CheckpointProvingManager::new(
            CheckpointProvingConfig::with_interval_and_error(10, 0.05),
        );

        // Large error should trigger early checkpoint before interval
        for _ in 0..3 {
            assert_eq!(mgr.record_step(0.01), StepAction::ContinueTraining);
        }

        // This step pushes error over 0.05 threshold
        let action = mgr.record_step(0.03);
        assert!(matches!(action, StepAction::GenerateProof { .. }));
    }

    #[test]
    fn test_checkpoint_reset() {
        let mut mgr =
            CheckpointProvingManager::new(CheckpointProvingConfig::with_interval(3));

        // First interval
        mgr.record_step(0.01);
        mgr.record_step(0.01);
        let action = mgr.record_step(0.01);
        assert!(matches!(action, StepAction::GenerateProof { .. }));

        // Reset after checkpoint
        mgr.checkpoint_completed();
        assert_eq!(mgr.steps_since_checkpoint(), 0);
        assert_eq!(mgr.error_since_checkpoint(), 0.0);

        // Second interval
        assert_eq!(mgr.record_step(0.01), StepAction::ContinueTraining);
        assert_eq!(mgr.record_step(0.01), StepAction::ContinueTraining);
        let action = mgr.record_step(0.01);
        assert!(matches!(action, StepAction::GenerateProof { .. }));
    }

    #[test]
    fn test_save_checkpoint_weights() {
        let mut mgr =
            CheckpointProvingManager::new(CheckpointProvingConfig::with_interval(5));
        mgr.record_step(0.01);
        mgr.record_step(0.01);

        let w1 = vec![Fr::from(1u64)];
        let b1 = vec![Fr::from(2u64)];
        let w2 = vec![Fr::from(3u64)];
        let b2 = vec![Fr::from(4u64)];

        mgr.save_checkpoint_weights(w1.clone(), b1.clone(), w2.clone(), b2.clone());

        // Saving checkpoint resets step and error counters
        assert_eq!(mgr.steps_since_checkpoint(), 0);
        assert_eq!(mgr.error_since_checkpoint(), 0.0);

        let cw = mgr.checkpoint_weights().unwrap();
        assert_eq!(cw.w1, w1);
        assert_eq!(cw.b1, b1);
    }

    #[test]
    fn test_global_step_counter() {
        let mut mgr =
            CheckpointProvingManager::new(CheckpointProvingConfig::with_interval(3));

        for i in 0..9 {
            mgr.record_step(0.01);
            assert_eq!(mgr.global_step(), i + 1);

            if mgr.is_checkpoint_boundary() {
                mgr.checkpoint_completed();
            }
        }

        assert_eq!(mgr.global_step(), 9);
    }

    #[test]
    fn test_is_checkpoint_boundary() {
        let mut mgr =
            CheckpointProvingManager::new(CheckpointProvingConfig::with_interval(3));

        assert!(!mgr.is_checkpoint_boundary());
        mgr.record_step(0.01);
        assert!(!mgr.is_checkpoint_boundary());
        mgr.record_step(0.01);
        assert!(!mgr.is_checkpoint_boundary());
        mgr.record_step(0.01);
        assert!(mgr.is_checkpoint_boundary());
    }

    #[test]
    fn test_with_interval_clamps_to_one() {
        let config = CheckpointProvingConfig::with_interval(0);
        assert_eq!(config.checkpoint_interval, 1);
    }

    #[test]
    fn test_error_threshold_disabled_when_zero() {
        let mut mgr = CheckpointProvingManager::new(
            CheckpointProvingConfig::with_interval_and_error(10, 0.0),
        );

        // Even with massive error, should not trigger early checkpoint
        // because threshold is 0.0 (disabled)
        for _ in 0..9 {
            assert_eq!(mgr.record_step(100.0), StepAction::ContinueTraining);
        }

        // Step 10 triggers by interval, not error
        let action = mgr.record_step(100.0);
        assert!(matches!(action, StepAction::GenerateProof { .. }));
    }

    #[test]
    fn test_checkpoint_start_step_tracking() {
        let mut mgr =
            CheckpointProvingManager::new(CheckpointProvingConfig::with_interval(3));

        // First interval: no saved weights, checkpoint_start_step = 0
        mgr.record_step(0.01);
        mgr.record_step(0.01);
        let action = mgr.record_step(0.01);
        match action {
            StepAction::GenerateProof {
                checkpoint_start_step,
                ..
            } => {
                assert_eq!(checkpoint_start_step, 0);
            }
            _ => panic!("Expected GenerateProof"),
        }

        // Save weights at step 3 and reset
        mgr.save_checkpoint_weights(
            vec![Fr::from(1u64)],
            vec![Fr::from(2u64)],
            vec![Fr::from(3u64)],
            vec![Fr::from(4u64)],
        );

        // Second interval: checkpoint_start_step should be 3
        mgr.record_step(0.01);
        mgr.record_step(0.01);
        let action = mgr.record_step(0.01);
        match action {
            StepAction::GenerateProof {
                checkpoint_start_step,
                ..
            } => {
                assert_eq!(checkpoint_start_step, 3);
            }
            _ => panic!("Expected GenerateProof"),
        }
    }
}
