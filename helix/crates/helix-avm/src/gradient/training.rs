//! Training Loop Orchestration.
//!
//! Provides a complete training loop with support for:
//! - Batch processing
//! - Gradient accumulation
//! - Learning rate scheduling
//! - Checkpointing
//! - Metrics tracking

use std::collections::HashMap;
use helix_core::types::{BoundedTensor, BoundedValue, Precision};

use super::autodiff::{GradientTape, NodeIndex, Variable};
use super::backward::backward;
use super::accumulator::GradientAccumulator;
use super::clipping::GradientClipConfig;
use super::optimizer::Optimizer;
use super::loss;

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
        }
    }

    /// Records metrics for a step.
    pub fn record_step(&mut self, loss: f64, grad_norm: f64) {
        self.running_loss += loss;
        self.running_grad_norm += grad_norm;
        self.epoch_steps += 1;
        self.global_step += 1;
    }

    /// Finalizes epoch and returns metrics.
    pub fn finalize_epoch(&mut self) -> EpochMetrics {
        let metrics = EpochMetrics {
            epoch: self.epoch,
            avg_loss: if self.epoch_steps > 0 { self.running_loss / self.epoch_steps as f64 } else { 0.0 },
            avg_grad_norm: if self.epoch_steps > 0 { self.running_grad_norm / self.epoch_steps as f64 } else { 0.0 },
            num_steps: self.epoch_steps,
            total_error: 0.0, // Placeholder
        };

        // Reset for next epoch
        self.epoch += 1;
        self.running_loss = 0.0;
        self.running_grad_norm = 0.0;
        self.epoch_steps = 0;

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
    parameters: HashMap<String, (NodeIndex, BoundedTensor)>,
}

impl<O: Optimizer> Trainer<O> {
    /// Creates a new trainer.
    pub fn new(optimizer: O, config: TrainingConfig) -> Self {
        Self {
            optimizer,
            config,
            accumulator: GradientAccumulator::new(),
            state: TrainingState::new(),
            parameters: HashMap::new(),
        }
    }

    /// Registers model parameters.
    pub fn register_parameter(&mut self, name: &str, idx: NodeIndex, param: BoundedTensor) {
        self.parameters.insert(name.to_string(), (idx, param));
    }

    /// Gets parameter tensors as a HashMap by NodeIndex.
    pub fn get_param_tensors(&self) -> HashMap<NodeIndex, BoundedTensor> {
        self.parameters.values().map(|(idx, t)| (*idx, t.clone())).collect()
    }

    /// Performs a single training step.
    ///
    /// Returns the step metrics.
    pub fn step(
        &mut self,
        loss_var: &Variable,
    ) -> Result<StepMetrics, String> {
        // Compute gradients
        let mut grads = backward(loss_var)?;

        // Apply gradient clipping
        let grad_norm = self.config.grad_clip.apply(&mut grads);

        // Accumulate gradients
        self.accumulator.accumulate(grads);

        // Check if we should update
        if self.accumulator.step_count() >= self.config.accumulation_steps {
            // Get averaged gradients
            let avg_grads = self.accumulator.average();

            // Get mutable params
            let mut param_tensors = self.get_param_tensors();

            // Optimizer step
            self.optimizer.step(&mut param_tensors, &avg_grads);

            // Update stored parameters
            for (_, (idx, param)) in self.parameters.iter_mut() {
                if let Some(new_tensor) = param_tensors.get(idx) {
                    *param = new_tensor.clone();
                }
            }

            // Clear accumulator
            self.accumulator.clear();
        }

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

        // Update state
        self.state.record_step(loss_value, grad_norm);

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

/// Simple training step function for one-off use.
pub fn train_step(
    loss: &Variable,
    params: &mut HashMap<NodeIndex, BoundedTensor>,
    optimizer: &mut dyn Optimizer,
    grad_clip: &GradientClipConfig,
) -> Result<f64, String> {
    let mut grads = backward(loss)?;
    grad_clip.apply(&mut grads);
    optimizer.step(params, &grads);
    
    let loss_value = if !loss.tensor.is_empty() {
        loss.tensor.data()[0].value()
    } else {
        0.0
    };
    
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
    fn test_training_state() {
        let mut state = TrainingState::new();

        state.record_step(0.5, 1.0);
        state.record_step(0.4, 0.8);
        state.record_step(0.3, 0.6);

        let metrics = state.finalize_epoch();

        assert_eq!(metrics.epoch, 0);
        assert_eq!(metrics.num_steps, 3);
        assert!((metrics.avg_loss - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_creation() {
        let optimizer = SGD::new(0.01);
        let config = TrainingConfig::default();
        let trainer = Trainer::new(optimizer, config);

        assert_eq!(trainer.state().global_step, 0);
    }
}
