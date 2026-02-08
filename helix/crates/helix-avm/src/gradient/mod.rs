//! Gradient Computation Module.
//!
//! Provides complete automatic differentiation, optimization, and training
//! capabilities with error bound tracking.
//!
//! # Overview
//!
//! - **autodiff**: Tape-based automatic differentiation
//! - **backward**: Backward pass implementation for all operations
//! - **loss**: Loss functions (MSE, cross-entropy, etc.)
//! - **optimizer**: SGD, Adam, AdamW with LR schedulers
//! - **clipping**: Gradient clipping by value and norm
//! - **training**: Training loop orchestration
//! - **accumulator**: Gradient accumulation for large batches

pub mod accumulator;
pub mod autodiff;
pub mod backward;
pub mod chain_rule;
pub mod clipping;
pub mod loss;
pub mod optimizer;
pub mod training;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use accumulator::GradientAccumulator;
pub use autodiff::{GradientTape, NodeIndex, Operation, Variable};
pub use backward::backward;
pub use clipping::{clip_grad_norm, clip_grad_value, GradientClipConfig};
pub use loss::{
    binary_cross_entropy_loss, cross_entropy_grad, cross_entropy_loss,
    huber_loss, mae_loss, mse_grad, mse_loss, softmax_cross_entropy_grad,
    softmax_cross_entropy_loss,
};
pub use optimizer::{
    Adam, CosineAnnealingLR, ExponentialLR, LRScheduler, LinearWarmupCosineDecay, OneCycleLR,
    Optimizer, PolynomialLR, SGD, StepLR, WarmupScheduler,
};
pub use training::{EpochMetrics, StepMetrics, Trainer, TrainingConfig, TrainingState};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        backward, Adam, GradientAccumulator, GradientClipConfig, GradientTape,
        Optimizer, SGD, Trainer, TrainingConfig, Variable,
    };
}
