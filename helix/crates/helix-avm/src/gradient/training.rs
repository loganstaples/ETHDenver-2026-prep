//! Training Loop Orchestration.
//!
//! Provides a complete training loop with support for:
//! - Batch processing
//! - Gradient accumulation
//! - Learning rate scheduling
//! - Checkpointing
//! - Metrics tracking

use std::collections::HashMap;
use helix_core::types::{BoundedTensor, Precision};

use super::autodiff::{NodeIndex, Variable};
use super::backward::backward;
use super::accumulator::GradientAccumulator;
use super::clipping::GradientClipConfig;
use super::optimizer::Optimizer;

use crate::memory::gradient_checkpoint::{
    CheckpointError, CheckpointStrategy, GradientCheckpointer,
};

/// Errors that can occur during training.
#[derive(Debug, Clone)]
pub enum TrainingError {
    /// Loss became NaN or Inf for too many consecutive steps.
    DivergentLoss {
        consecutive_nan_steps: usize,
        last_loss: f64,
    },
    /// Gradient norm exceeded the threshold.
    ExplodingGradients {
        grad_norm: f64,
        threshold: f64,
    },
    /// Backward pass failed.
    BackwardError(String),
    /// Accumulated error exceeded the allowed budget.
    ErrorBudgetExceeded {
        accumulated_error: f64,
        error_budget: f64,
    },
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::DivergentLoss { consecutive_nan_steps, last_loss } =>
                write!(f, "Divergent loss: {} consecutive NaN/Inf steps, last loss: {}", consecutive_nan_steps, last_loss),
            TrainingError::ExplodingGradients { grad_norm, threshold } =>
                write!(f, "Exploding gradients: norm {} exceeds threshold {}", grad_norm, threshold),
            TrainingError::BackwardError(msg) => write!(f, "Backward error: {}", msg),
            TrainingError::ErrorBudgetExceeded { accumulated_error, error_budget } =>
                write!(f, "Error budget exceeded: accumulated {:.6} > budget {:.6}", accumulated_error, error_budget),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Gradient accumulation steps.
    pub accumulation_steps: usize,
    /// Gradient clipping configuration.
    pub grad_clip: GradientClipConfig,
    /// Precision for computations.
    pub precision: Precision,
    /// Whether to log training progress.
    pub verbose: bool,
    /// Logging frequency (steps between logs).
    pub log_frequency: usize,
    /// Optional gradient checkpointing strategy.
    pub checkpoint_strategy: Option<CheckpointStrategy>,
    /// Maximum allowed consecutive NaN/Inf loss steps before halting (default: 3).
    pub max_divergent_steps: usize,
    /// Maximum gradient norm before halting (0.0 = no limit).
    pub max_grad_norm: f64,
    /// Maximum allowed accumulated error before halting (0.0 = no limit, default: 0.0).
    pub error_budget: f64,
    /// Warning threshold as fraction of budget (default: 0.8 = warn at 80%).
    pub error_budget_warn_threshold: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 32,
            accumulation_steps: 1,
            grad_clip: GradientClipConfig::None,
            precision: Precision::F32,
            verbose: true,
            log_frequency: 100,
            checkpoint_strategy: None,
            max_divergent_steps: 3,
            max_grad_norm: 1e6,
            error_budget: 0.0,
            error_budget_warn_threshold: 0.8,
        }
    }
}

impl TrainingConfig {
    /// Creates a new training config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of epochs.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets gradient accumulation steps.
    pub fn with_accumulation_steps(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps;
        self
    }

    /// Sets gradient clipping.
    pub fn with_grad_clip(mut self, clip: GradientClipConfig) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Sets gradient checkpointing strategy.
    pub fn with_checkpoint_strategy(mut self, strategy: CheckpointStrategy) -> Self {
        self.checkpoint_strategy = Some(strategy);
        self
    }

    /// Sets the maximum allowed accumulated error budget.
    pub fn with_error_budget(mut self, budget: f64) -> Self {
        self.error_budget = budget;
        self
    }
}

/// Training metrics for a single step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Loss value.
    pub loss: f64,
    /// Loss error bound.
    pub loss_error: f64,
    /// Gradient norm before clipping.
    pub grad_norm: f64,
    /// Learning rate used.
    pub learning_rate: f64,
    /// Step number.
    pub step: usize,
}

/// Training metrics for an epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Epoch number.
    pub epoch: usize,
    /// Average loss.
    pub avg_loss: f64,
    /// Average gradient norm.
    pub avg_grad_norm: f64,
    /// Number of steps.
    pub num_steps: usize,
    /// Total error accumulation.
    pub total_error: f64,
}

/// Training state tracker.
#[derive(Debug)]
pub struct TrainingState {
    /// Current epoch.
    pub epoch: usize,
    /// Current global step.
    pub global_step: usize,
    /// Running loss sum for current epoch.
    running_loss: f64,
    /// Running grad norm sum.
    running_grad_norm: f64,
    /// Steps in current epoch.
    epoch_steps: usize,
    /// Accumulated error from BoundedValue operations.
    running_error: f64,
    /// Count of consecutive NaN/Inf loss steps.
    consecutive_divergent_steps: usize,
}

impl TrainingState {
    /// Creates a new training state.
    pub fn new() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            running_loss: 0.0,
            running_grad_norm: 0.0,
            epoch_steps: 0,
            running_error: 0.0,
            consecutive_divergent_steps: 0,
        }
    }

    /// Records metrics for a step. Returns the number of consecutive divergent steps.
    pub fn record_step(&mut self, loss: f64, grad_norm: f64, loss_error: f64) -> usize {
        if loss.is_finite() {
            self.running_loss += loss;
            self.consecutive_divergent_steps = 0;
        } else {
            self.consecutive_divergent_steps += 1;
        }
        self.running_grad_norm += grad_norm;
        self.running_error += loss_error;
        self.epoch_steps += 1;
        self.global_step += 1;
        self.consecutive_divergent_steps
    }

    /// Returns the number of consecutive divergent (NaN/Inf) loss steps.
    pub fn consecutive_divergent_steps(&self) -> usize {
        self.consecutive_divergent_steps
    }

    /// Returns the accumulated error for the current epoch.
    pub fn running_error(&self) -> f64 {
        self.running_error
    }

    /// Finalizes epoch and returns metrics.
    pub fn finalize_epoch(&mut self) -> EpochMetrics {
        let metrics = EpochMetrics {
            epoch: self.epoch,
            avg_loss: if self.epoch_steps > 0 { self.running_loss / self.epoch_steps as f64 } else { 0.0 },
            avg_grad_norm: if self.epoch_steps > 0 { self.running_grad_norm / self.epoch_steps as f64 } else { 0.0 },
            num_steps: self.epoch_steps,
            total_error: self.running_error,
        };

        // Reset for next epoch
        self.epoch += 1;
        self.running_loss = 0.0;
        self.running_grad_norm = 0.0;
        self.epoch_steps = 0;
        self.running_error = 0.0;

        metrics
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Training loop handler.
pub struct Trainer<O: Optimizer> {
    /// Optimizer.
    optimizer: O,
    /// Training configuration.
    config: TrainingConfig,
    /// Gradient accumulator.
    accumulator: GradientAccumulator,
    /// Training state.
    state: TrainingState,
    /// Current model parameters (by name).
    /// Maps name -> (current_tape_node_index, tensor).
    parameters: HashMap<String, (NodeIndex, BoundedTensor)>,
    /// Stable canonical indices for gradient accumulation across tape boundaries.
    /// Maps parameter name -> canonical index (assigned at registration time).
    canonical_indices: HashMap<String, NodeIndex>,
    /// Next canonical index to assign.
    next_canonical: NodeIndex,
    /// Gradient checkpointer (if checkpointing is enabled).
    checkpointer: Option<GradientCheckpointer>,
}

impl<O: Optimizer> Trainer<O> {
    /// Creates a new trainer.
    pub fn new(optimizer: O, config: TrainingConfig) -> Self {
        let checkpointer = config
            .checkpoint_strategy
            .map(GradientCheckpointer::new);

        Self {
            optimizer,
            config,
            accumulator: GradientAccumulator::new(),
            state: TrainingState::new(),
            parameters: HashMap::new(),
            canonical_indices: HashMap::new(),
            next_canonical: 0,
            checkpointer,
        }
    }

    /// Registers model parameters.
    ///
    /// Each parameter is assigned a stable canonical index that persists across
    /// tape boundaries, enabling gradient accumulation across mini-batches.
    pub fn register_parameter(&mut self, name: &str, idx: NodeIndex, param: BoundedTensor) {
        self.parameters.insert(name.to_string(), (idx, param));
        if !self.canonical_indices.contains_key(name) {
            self.canonical_indices.insert(name.to_string(), self.next_canonical);
            self.next_canonical += 1;
        }
    }

    /// Updates the tape node index for a named parameter.
    ///
    /// Call this when a fresh tape is created for each training step,
    /// so the trainer knows which tape node corresponds to which parameter.
    pub fn update_param_index(&mut self, name: &str, new_idx: NodeIndex) {
        if let Some((idx, _)) = self.parameters.get_mut(name) {
            *idx = new_idx;
        }
    }

    /// Gets a parameter tensor by name.
    pub fn get_param(&self, name: &str) -> Option<&BoundedTensor> {
        self.parameters.get(name).map(|(_, t)| t)
    }

    /// Gets parameter tensors as a HashMap by NodeIndex.
    pub fn get_param_tensors(&self) -> HashMap<NodeIndex, BoundedTensor> {
        self.parameters.values().map(|(idx, t)| (*idx, t.clone())).collect()
    }

    /// Initializes checkpointing for a model with the given number of layers.
    ///
    /// Must be called before using checkpoint methods if checkpointing is enabled.
    pub fn init_checkpointing(&mut self, num_layers: usize) {
        if let Some(ref mut cp) = self.checkpointer {
            cp.init(num_layers);
        }
    }

    /// Returns whether the given layer should have its activation checkpointed.
    pub fn should_checkpoint(&self, layer: usize) -> bool {
        self.checkpointer
            .as_ref()
            .map(|cp| cp.should_checkpoint(layer))
            .unwrap_or(false)
    }

    /// Saves an activation for the given layer (if checkpointing is enabled).
    pub fn save_activation(&mut self, layer: usize, activation: &BoundedTensor) {
        if let Some(ref mut cp) = self.checkpointer {
            cp.save_activation(layer, activation);
        }
    }

    /// Gets a cached activation or recomputes it from the nearest checkpoint.
    ///
    /// The `recompute_fn` takes `(layer_index, input_activation)` and returns
    /// the output activation for that layer.
    pub fn get_or_recompute<F>(
        &mut self,
        layer: usize,
        recompute_fn: F,
    ) -> Result<BoundedTensor, CheckpointError>
    where
        F: Fn(usize, &BoundedTensor) -> Result<BoundedTensor, String>,
    {
        match self.checkpointer {
            Some(ref mut cp) => cp.get_or_recompute(layer, recompute_fn),
            None => Err(CheckpointError::InvalidConfig(
                "Checkpointing not enabled".to_string(),
            )),
        }
    }

    /// Signals the start of a forward pass to the checkpointer.
    pub fn begin_forward(&mut self) {
        if let Some(ref mut cp) = self.checkpointer {
            cp.begin_forward();
        }
    }

    /// Signals the start of a backward pass to the checkpointer.
    pub fn begin_backward(&mut self) {
        if let Some(ref mut cp) = self.checkpointer {
            cp.begin_backward();
        }
    }

    /// Clears checkpointed activations (typically between training steps).
    pub fn clear_checkpoints(&mut self) {
        if let Some(ref mut cp) = self.checkpointer {
            cp.clear();
        }
    }

    /// Returns the checkpointing strategy, if any.
    pub fn checkpoint_strategy(&self) -> Option<CheckpointStrategy> {
        self.checkpointer.as_ref().map(|cp| cp.strategy())
    }

    /// Performs a single training step.
    ///
    /// Uses canonical parameter indices for gradient accumulation across tape
    /// boundaries. When parameters are registered via `register_parameter()`,
    /// gradients from `backward()` are remapped from tape-specific NodeIndices
    /// to stable canonical indices, enabling correct gradient accumulation
    /// even when a fresh tape is created for each forward pass.
    ///
    /// Returns the step metrics, or a TrainingError if training should halt.
    pub fn step(
        &mut self,
        loss_var: &Variable,
    ) -> Result<StepMetrics, TrainingError> {
        // Compute gradients
        let grads = backward(loss_var).map_err(TrainingError::BackwardError)?;

        // Remap gradients from tape NodeIndex to canonical indices.
        // This enables stable gradient accumulation across tape boundaries.
        let use_canonical = !self.canonical_indices.is_empty();
        let mut remapped_grads: HashMap<NodeIndex, BoundedTensor> = if use_canonical {
            let mut remapped = HashMap::new();
            for (name, (tape_idx, _)) in &self.parameters {
                if let Some(grad) = grads.get(tape_idx) {
                    if let Some(&canonical_idx) = self.canonical_indices.get(name) {
                        remapped.insert(canonical_idx, grad.clone());
                    }
                }
            }
            remapped
        } else {
            grads
        };

        // Apply gradient clipping
        let grad_norm = self.config.grad_clip.apply(&mut remapped_grads);

        // Check for exploding gradients
        if self.config.max_grad_norm > 0.0 && grad_norm > self.config.max_grad_norm {
            return Err(TrainingError::ExplodingGradients {
                grad_norm,
                threshold: self.config.max_grad_norm,
            });
        }

        // Accumulate gradients (canonical indices are stable across tapes)
        self.accumulator.accumulate(remapped_grads);

        // Check if we should update
        if self.accumulator.step_count() >= self.config.accumulation_steps {
            // Get averaged gradients
            let avg_grads = self.accumulator.average();

            // Build param tensors keyed by canonical or tape index
            let mut param_tensors: HashMap<NodeIndex, BoundedTensor> = if use_canonical {
                self.canonical_indices
                    .iter()
                    .filter_map(|(name, &canonical_idx)| {
                        self.parameters
                            .get(name)
                            .map(|(_, t)| (canonical_idx, t.clone()))
                    })
                    .collect()
            } else {
                self.get_param_tensors()
            };

            // Optimizer step
            self.optimizer.step(&mut param_tensors, &avg_grads);

            // Update stored parameters
            if use_canonical {
                for (name, &canonical_idx) in &self.canonical_indices {
                    if let Some(new_tensor) = param_tensors.get(&canonical_idx) {
                        if let Some((_, stored_tensor)) = self.parameters.get_mut(name) {
                            *stored_tensor = new_tensor.clone();
                        }
                    }
                }
            } else {
                for (_, (idx, param)) in self.parameters.iter_mut() {
                    if let Some(new_tensor) = param_tensors.get(idx) {
                        *param = new_tensor.clone();
                    }
                }
            }

            // Clear accumulator
            self.accumulator.clear();
        }

        // Clear checkpoints after each step
        self.clear_checkpoints();

        let loss_value = if !loss_var.tensor.is_empty() {
            loss_var.tensor.data()[0].value()
        } else {
            0.0
        };

        let loss_error = if !loss_var.tensor.is_empty() {
            loss_var.tensor.data()[0].absolute_error()
        } else {
            0.0
        };

        // Update state with error tracking
        let divergent_count = self.state.record_step(loss_value, grad_norm, loss_error);

        // Check for divergent loss (NaN/Inf for consecutive steps)
        if divergent_count >= self.config.max_divergent_steps {
            return Err(TrainingError::DivergentLoss {
                consecutive_nan_steps: divergent_count,
                last_loss: loss_value,
            });
        }

        // Check error budget
        if self.config.error_budget > 0.0 {
            let accumulated = self.state.running_error();
            if accumulated > self.config.error_budget {
                return Err(TrainingError::ErrorBudgetExceeded {
                    accumulated_error: accumulated,
                    error_budget: self.config.error_budget,
                });
            }
            // Warn at threshold
            let threshold = self.config.error_budget * self.config.error_budget_warn_threshold;
            if accumulated > threshold {
                eprintln!(
                    "[helix-avm] WARNING: Error budget {:.1}% consumed ({:.6} / {:.6})",
                    (accumulated / self.config.error_budget) * 100.0,
                    accumulated,
                    self.config.error_budget,
                );
            }
        }

        Ok(StepMetrics {
            loss: loss_value,
            loss_error,
            grad_norm,
            learning_rate: self.optimizer.learning_rate(),
            step: self.state.global_step,
        })
    }

    /// Training state accessor.
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Returns the optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Returns mutable optimizer.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

/// Early stopping based on validation loss.
///
/// Monitors a metric (typically validation loss) and signals when training
/// should stop if the metric hasn't improved for `patience` evaluations.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of evaluations with no improvement before stopping.
    patience: usize,
    /// Minimum change to qualify as an improvement.
    min_delta: f64,
    /// Best metric value seen so far.
    best_value: f64,
    /// Number of evaluations since last improvement.
    counter: usize,
    /// Whether to look for lower (true) or higher (false) values.
    minimize: bool,
    /// Best epoch number.
    best_epoch: usize,
}

impl EarlyStopping {
    /// Creates a new early stopping monitor.
    ///
    /// - `patience`: Number of evaluations with no improvement before stopping.
    /// - `min_delta`: Minimum change to qualify as an improvement.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_value: f64::INFINITY,
            counter: 0,
            minimize: true,
            best_epoch: 0,
        }
    }

    /// Sets whether to minimize (default) or maximize the metric.
    pub fn maximize(mut self) -> Self {
        self.minimize = false;
        self.best_value = f64::NEG_INFINITY;
        self
    }

    /// Checks if training should stop. Call this once per evaluation (e.g., end of epoch).
    ///
    /// Returns `true` if training should stop (no improvement for `patience` evaluations).
    pub fn should_stop(&mut self, metric: f64, epoch: usize) -> bool {
        let improved = if self.minimize {
            metric < self.best_value - self.min_delta
        } else {
            metric > self.best_value + self.min_delta
        };

        if improved {
            self.best_value = metric;
            self.counter = 0;
            self.best_epoch = epoch;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    /// Returns the best metric value seen.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Returns the epoch of the best metric value.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Returns the number of evaluations since last improvement.
    pub fn evaluations_without_improvement(&self) -> usize {
        self.counter
    }
}

/// Simple training step function for one-off use.
pub fn train_step(
    loss: &Variable,
    params: &mut HashMap<NodeIndex, BoundedTensor>,
    optimizer: &mut dyn Optimizer,
    grad_clip: &GradientClipConfig,
) -> Result<f64, TrainingError> {
    let mut grads = backward(loss).map_err(TrainingError::BackwardError)?;
    let grad_norm = grad_clip.apply(&mut grads);

    // Check for NaN/Inf gradients
    if !grad_norm.is_finite() {
        return Err(TrainingError::ExplodingGradients {
            grad_norm,
            threshold: 0.0,
        });
    }

    optimizer.step(params, &grads);

    let loss_value = if !loss.tensor.is_empty() {
        loss.tensor.data()[0].value()
    } else {
        0.0
    };

    if !loss_value.is_finite() {
        return Err(TrainingError::DivergentLoss {
            consecutive_nan_steps: 1,
            last_loss: loss_value,
        });
    }

    Ok(loss_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::optimizer::SGD;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new()
            .with_epochs(10)
            .with_batch_size(64)
            .with_accumulation_steps(4);

        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.accumulation_steps, 4);
    }

    #[test]
    fn test_training_config_with_checkpointing() {
        let config = TrainingConfig::new()
            .with_checkpoint_strategy(CheckpointStrategy::SqrtN);

        assert_eq!(config.checkpoint_strategy, Some(CheckpointStrategy::SqrtN));
    }

    #[test]
    fn test_training_state() {
        let mut state = TrainingState::new();

        state.record_step(0.5, 1.0, 0.01);
        state.record_step(0.4, 0.8, 0.02);
        state.record_step(0.3, 0.6, 0.01);

        let metrics = state.finalize_epoch();

        assert_eq!(metrics.epoch, 0);
        assert_eq!(metrics.num_steps, 3);
        assert!((metrics.avg_loss - 0.4).abs() < 1e-10);
        assert!((metrics.total_error - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_divergent_loss_detection() {
        let mut state = TrainingState::new();

        // Normal step
        let div = state.record_step(0.5, 1.0, 0.01);
        assert_eq!(div, 0);

        // NaN steps
        let div = state.record_step(f64::NAN, 1.0, 0.0);
        assert_eq!(div, 1);
        let div = state.record_step(f64::INFINITY, 1.0, 0.0);
        assert_eq!(div, 2);
        let div = state.record_step(f64::NAN, 1.0, 0.0);
        assert_eq!(div, 3);

        // Recovery
        let div = state.record_step(0.3, 0.5, 0.01);
        assert_eq!(div, 0);
    }

    #[test]
    fn test_trainer_creation() {
        let optimizer = SGD::new(0.01);
        let config = TrainingConfig::default();
        let trainer = Trainer::new(optimizer, config);

        assert_eq!(trainer.state().global_step, 0);
        assert!(trainer.checkpoint_strategy().is_none());
    }

    #[test]
    fn test_trainer_with_checkpointing() {
        let optimizer = SGD::new(0.01);
        let config = TrainingConfig::new()
            .with_checkpoint_strategy(CheckpointStrategy::SqrtN);
        let mut trainer = Trainer::new(optimizer, config);

        assert_eq!(trainer.checkpoint_strategy(), Some(CheckpointStrategy::SqrtN));

        // Init checkpointing for 16 layers
        trainer.init_checkpointing(16);

        // SqrtN with 16 layers should checkpoint some layers
        assert!(trainer.should_checkpoint(0));
    }

    #[test]
    fn test_trainer_checkpoint_save_and_clear() {
        let optimizer = SGD::new(0.01);
        let config = TrainingConfig::new()
            .with_checkpoint_strategy(CheckpointStrategy::None); // Store all
        let mut trainer = Trainer::new(optimizer, config);
        trainer.init_checkpointing(4);

        let tensor = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        trainer.save_activation(0, &tensor);

        // After clear, saved activations should be gone
        trainer.clear_checkpoints();
    }

    #[test]
    fn test_trainer_no_checkpoint_should_return_false() {
        let optimizer = SGD::new(0.01);
        let config = TrainingConfig::default(); // No checkpointing
        let trainer = Trainer::new(optimizer, config);

        // Without checkpointing, should_checkpoint always returns false
        assert!(!trainer.should_checkpoint(0));
        assert!(!trainer.should_checkpoint(5));
    }

    #[test]
    fn test_error_budget_exceeded() {
        let mut state = TrainingState::new();

        // Accumulate error over budget
        for _ in 0..100 {
            state.record_step(0.5, 1.0, 0.1); // 0.1 error per step
        }

        // total_error should be ~10.0
        let metrics = state.finalize_epoch();
        assert!(metrics.total_error > 9.9);
    }

    #[test]
    fn test_error_budget_config() {
        let config = TrainingConfig::new()
            .with_error_budget(1.0);

        assert_eq!(config.error_budget, 1.0);
        assert_eq!(config.error_budget_warn_threshold, 0.8);
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut es = EarlyStopping::new(3, 0.001);

        // Improving losses
        assert!(!es.should_stop(1.0, 0));
        assert!(!es.should_stop(0.8, 1));
        assert!(!es.should_stop(0.6, 2));

        // Stagnating losses
        assert!(!es.should_stop(0.6, 3)); // counter = 1
        assert!(!es.should_stop(0.601, 4)); // counter = 2 (within min_delta)
        assert!(es.should_stop(0.7, 5)); // counter = 3 -> stop!

        assert_eq!(es.best_epoch(), 2);
        assert!((es.best_value() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.0);

        assert!(!es.should_stop(1.0, 0));
        assert!(!es.should_stop(1.0, 1)); // no improvement, counter = 1
        assert!(!es.should_stop(1.0, 2)); // counter = 2
        assert!(!es.should_stop(0.5, 3)); // improvement! counter resets
        assert_eq!(es.evaluations_without_improvement(), 0);
        assert_eq!(es.best_epoch(), 3);
    }

    #[test]
    fn test_early_stopping_maximize() {
        let mut es = EarlyStopping::new(2, 0.0).maximize();

        assert!(!es.should_stop(0.5, 0)); // best = 0.5
        assert!(!es.should_stop(0.7, 1)); // improved
        assert!(!es.should_stop(0.6, 2)); // counter = 1
        assert!(es.should_stop(0.65, 3)); // counter = 2 -> stop
        assert!((es.best_value() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_running_error_accessor() {
        let mut state = TrainingState::new();
        state.record_step(0.5, 1.0, 0.05);
        assert!((state.running_error() - 0.05).abs() < 1e-10);
    }
}
