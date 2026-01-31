//! Adaptive precision scheduling for training phases.
//!
//! This module provides intelligent precision management across training,
//! adapting precision based on:
//! - Training phase (warmup, main, cooldown)
//! - Loss stability
//! - Gradient statistics
//! - Error budget consumption

use super::precision::Precision;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Training phase for precision scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingPhase {
    /// Initial warmup phase (learning rate ramp-up).
    Warmup,
    /// Main training phase.
    Main,
    /// Fine-tuning phase (lower learning rate).
    FineTune,
    /// Final cooldown phase.
    Cooldown,
    /// Evaluation phase (no gradients).
    Evaluation,
}

impl TrainingPhase {
    /// Returns the recommended minimum precision for this phase.
    pub fn recommended_precision(&self) -> Precision {
        match self {
            TrainingPhase::Warmup => Precision::F32,     // High precision during warmup
            TrainingPhase::Main => Precision::BF16,      // Mixed precision main phase
            TrainingPhase::FineTune => Precision::BF16,  // Medium precision fine-tuning
            TrainingPhase::Cooldown => Precision::F32,   // High precision for final steps
            TrainingPhase::Evaluation => Precision::F16, // Can use lower for eval
        }
    }

    /// Returns the error tolerance multiplier for this phase.
    pub fn error_tolerance_multiplier(&self) -> f64 {
        match self {
            TrainingPhase::Warmup => 0.5,    // Lower tolerance
            TrainingPhase::Main => 1.0,      // Normal tolerance
            TrainingPhase::FineTune => 0.7,  // Medium tolerance
            TrainingPhase::Cooldown => 0.5,  // Lower tolerance
            TrainingPhase::Evaluation => 1.5, // Higher tolerance ok
        }
    }
}

/// Configuration for precision scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSchedulerConfig {
    /// Total number of training steps.
    pub total_steps: usize,
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Number of cooldown steps.
    pub cooldown_steps: usize,
    /// Base precision for main phase.
    pub base_precision: Precision,
    /// Maximum allowed precision (highest quality).
    pub max_precision: Precision,
    /// Minimum allowed precision (fastest).
    pub min_precision: Precision,
    /// Loss smoothing window size.
    pub loss_window_size: usize,
    /// Gradient norm window size.
    pub grad_norm_window_size: usize,
    /// Loss instability threshold for precision increase.
    pub loss_instability_threshold: f64,
    /// Gradient explosion threshold.
    pub grad_explosion_threshold: f64,
    /// Error budget for entire training.
    pub total_error_budget: f64,
    /// Enable automatic adjustment.
    pub auto_adjust: bool,
    /// Number of stable steps before decreasing precision.
    pub stable_steps_threshold: usize,
}

impl Default for PrecisionSchedulerConfig {
    fn default() -> Self {
        Self {
            total_steps: 10000,
            warmup_steps: 500,
            cooldown_steps: 200,
            base_precision: Precision::BF16,
            max_precision: Precision::F32,
            min_precision: Precision::INT8,
            loss_window_size: 100,
            grad_norm_window_size: 50,
            loss_instability_threshold: 0.1,
            grad_explosion_threshold: 100.0,
            total_error_budget: 0.01,
            auto_adjust: true,
            stable_steps_threshold: 100,
        }
    }
}

/// Statistics tracked for adaptive scheduling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Recent loss values.
    pub loss_history: VecDeque<f64>,
    /// Recent gradient norms.
    pub grad_norm_history: VecDeque<f64>,
    /// Recent error magnitudes.
    pub error_history: VecDeque<f64>,
    /// Number of precision increases.
    pub precision_increases: usize,
    /// Number of precision decreases.
    pub precision_decreases: usize,
    /// Steps with unstable loss.
    pub unstable_steps: usize,
    /// Steps with gradient explosion.
    pub explosion_steps: usize,
}

impl TrainingStats {
    /// Creates new training stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a loss value.
    pub fn record_loss(&mut self, loss: f64, window_size: usize) {
        self.loss_history.push_back(loss);
        while self.loss_history.len() > window_size {
            self.loss_history.pop_front();
        }
    }

    /// Records a gradient norm.
    pub fn record_grad_norm(&mut self, norm: f64, window_size: usize) {
        self.grad_norm_history.push_back(norm);
        while self.grad_norm_history.len() > window_size {
            self.grad_norm_history.pop_front();
        }
    }

    /// Records an error magnitude.
    pub fn record_error(&mut self, error: f64, window_size: usize) {
        self.error_history.push_back(error);
        while self.error_history.len() > window_size {
            self.error_history.pop_front();
        }
    }

    /// Computes loss variance over recent window.
    pub fn loss_variance(&self) -> f64 {
        if self.loss_history.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.loss_history.iter().sum::<f64>() / self.loss_history.len() as f64;
        let variance: f64 = self
            .loss_history
            .iter()
            .map(|l| (l - mean).powi(2))
            .sum::<f64>()
            / self.loss_history.len() as f64;

        variance
    }

    /// Computes loss trend (positive = increasing, negative = decreasing).
    pub fn loss_trend(&self) -> f64 {
        if self.loss_history.len() < 2 {
            return 0.0;
        }

        let n = self.loss_history.len();
        let half = n / 2;

        let first_half_mean: f64 = self.loss_history.iter().take(half).sum::<f64>() / half as f64;
        let second_half_mean: f64 = self.loss_history.iter().skip(half).sum::<f64>() / (n - half) as f64;

        second_half_mean - first_half_mean
    }

    /// Computes gradient norm statistics.
    pub fn grad_norm_stats(&self) -> (f64, f64, f64) {
        if self.grad_norm_history.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mean: f64 =
            self.grad_norm_history.iter().sum::<f64>() / self.grad_norm_history.len() as f64;
        let max = self
            .grad_norm_history
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        let variance: f64 = self
            .grad_norm_history
            .iter()
            .map(|g| (g - mean).powi(2))
            .sum::<f64>()
            / self.grad_norm_history.len() as f64;

        (mean, variance.sqrt(), max)
    }

    /// Computes accumulated error.
    pub fn accumulated_error(&self) -> f64 {
        self.error_history.iter().sum()
    }
}

/// Decision made by the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerDecision {
    /// Recommended precision for forward pass.
    pub forward_precision: Precision,
    /// Recommended precision for backward pass.
    pub backward_precision: Precision,
    /// Recommended precision for weight updates.
    pub update_precision: Precision,
    /// Current training phase.
    pub phase: TrainingPhase,
    /// Reason for the decision.
    pub reason: String,
    /// Confidence in the decision (0-1).
    pub confidence: f64,
    /// Recommended loss scale for mixed precision.
    pub loss_scale: f64,
    /// Whether gradient clipping is recommended.
    pub recommend_grad_clip: bool,
    /// Recommended gradient clip value.
    pub grad_clip_value: f64,
}

/// Adaptive precision scheduler for training.
#[derive(Debug, Clone)]
pub struct PrecisionScheduler {
    config: PrecisionSchedulerConfig,
    stats: TrainingStats,
    current_step: usize,
    current_phase: TrainingPhase,
    current_forward_precision: Precision,
    current_backward_precision: Precision,
    current_update_precision: Precision,
    current_loss_scale: f64,
    consecutive_stable_steps: usize,
    consumed_error_budget: f64,
}

impl PrecisionScheduler {
    /// Creates a new precision scheduler.
    pub fn new(config: PrecisionSchedulerConfig) -> Self {
        Self {
            current_forward_precision: config.base_precision,
            current_backward_precision: config.base_precision,
            current_update_precision: Precision::F32, // Always high for updates
            current_phase: TrainingPhase::Warmup,
            current_loss_scale: 1.0,
            consecutive_stable_steps: 0,
            consumed_error_budget: 0.0,
            config,
            stats: TrainingStats::new(),
            current_step: 0,
        }
    }

    /// Gets the decision for the current step.
    pub fn get_decision(&self) -> SchedulerDecision {
        let (grad_mean, grad_std, grad_max) = self.stats.grad_norm_stats();

        SchedulerDecision {
            forward_precision: self.current_forward_precision,
            backward_precision: self.current_backward_precision,
            update_precision: self.current_update_precision,
            phase: self.current_phase,
            reason: self.explain_decision(),
            confidence: self.compute_confidence(),
            loss_scale: self.current_loss_scale,
            recommend_grad_clip: grad_max > self.config.grad_explosion_threshold * 0.5,
            grad_clip_value: grad_mean + 3.0 * grad_std,
        }
    }

    /// Steps the scheduler forward after a training step.
    pub fn step(
        &mut self,
        loss: f64,
        grad_norm: f64,
        step_error: f64,
    ) -> SchedulerDecision {
        self.current_step += 1;

        // Record statistics
        self.stats.record_loss(loss, self.config.loss_window_size);
        self.stats.record_grad_norm(grad_norm, self.config.grad_norm_window_size);
        self.stats.record_error(step_error, self.config.loss_window_size);
        self.consumed_error_budget += step_error;

        // Update phase
        self.update_phase();

        // Check for issues and adjust
        if self.config.auto_adjust {
            self.auto_adjust();
        }

        self.get_decision()
    }

    /// Updates the training phase based on current step.
    fn update_phase(&mut self) {
        if self.current_step < self.config.warmup_steps {
            self.current_phase = TrainingPhase::Warmup;
        } else if self.current_step
            >= self.config.total_steps.saturating_sub(self.config.cooldown_steps)
        {
            self.current_phase = TrainingPhase::Cooldown;
        } else if self.current_step >= self.config.total_steps * 9 / 10 {
            self.current_phase = TrainingPhase::FineTune;
        } else {
            self.current_phase = TrainingPhase::Main;
        }
    }

    /// Automatically adjusts precision based on training dynamics.
    fn auto_adjust(&mut self) {
        // Check for loss instability
        let loss_var = self.stats.loss_variance();
        let loss_trend = self.stats.loss_trend();

        let is_unstable = loss_var.sqrt()
            > self.config.loss_instability_threshold
                * self.current_phase.error_tolerance_multiplier();

        let is_diverging = loss_trend > 0.0 && self.stats.loss_history.len() >= 10;

        // Check for gradient explosion
        let (_, grad_std, grad_max) = self.stats.grad_norm_stats();
        let has_gradient_explosion = grad_max > self.config.grad_explosion_threshold;

        // Check error budget
        let budget_fraction =
            self.consumed_error_budget / self.config.total_error_budget;
        let is_over_budget = budget_fraction > 0.8;

        // Decision logic
        if has_gradient_explosion {
            // Increase precision and reduce loss scale
            self.increase_precision("gradient explosion");
            self.current_loss_scale /= 2.0;
            self.stats.explosion_steps += 1;
            self.consecutive_stable_steps = 0;
        } else if is_unstable || is_diverging {
            // Increase precision
            self.increase_precision("loss instability");
            self.stats.unstable_steps += 1;
            self.consecutive_stable_steps = 0;
        } else if is_over_budget {
            // Need higher precision to reduce error
            self.increase_precision("error budget");
            self.consecutive_stable_steps = 0;
        } else {
            self.consecutive_stable_steps += 1;

            // If stable for a while, try decreasing precision
            if self.consecutive_stable_steps > self.config.stable_steps_threshold {
                self.decrease_precision("stable training");
                self.consecutive_stable_steps = 0;
            }
        }

        // Phase-specific adjustments
        match self.current_phase {
            TrainingPhase::Warmup => {
                // Always use high precision during warmup
                self.current_forward_precision = self.config.max_precision;
                self.current_backward_precision = self.config.max_precision;
            }
            TrainingPhase::Cooldown => {
                // High precision for final steps
                self.current_forward_precision = self.config.max_precision;
                self.current_backward_precision = self.config.max_precision;
            }
            _ => {}
        }
    }

    /// Increases precision.
    fn increase_precision(&mut self, reason: &str) {
        let increased = match self.current_forward_precision {
            Precision::INT4 => Precision::INT8,
            Precision::INT8 => Precision::F16,
            Precision::F16 => Precision::BF16,
            Precision::BF16 => Precision::F32,
            Precision::F32 => Precision::F32, // Already max
            Precision::Custom { bits, max_relative_error } => {
                if bits < 32 {
                    Precision::F32
                } else {
                    self.current_forward_precision
                }
            }
        };

        if increased != self.current_forward_precision {
            self.current_forward_precision = increased;
            self.current_backward_precision = increased;
            self.stats.precision_increases += 1;
        }
    }

    /// Decreases precision.
    fn decrease_precision(&mut self, reason: &str) {
        let decreased = match self.current_forward_precision {
            Precision::F32 => Precision::BF16,
            Precision::BF16 => Precision::F16,
            Precision::F16 => Precision::INT8,
            Precision::INT8 => Precision::INT8, // Don't go lower
            Precision::INT4 => Precision::INT4, // Already min
            Precision::Custom { bits, max_relative_error } => {
                if bits > 8 {
                    Precision::INT8
                } else {
                    self.current_forward_precision
                }
            }
        };

        // Don't go below min precision
        if decreased.bits() >= self.config.min_precision.bits()
            && decreased != self.current_forward_precision
        {
            self.current_forward_precision = decreased;
            self.current_backward_precision = decreased;
            self.stats.precision_decreases += 1;
        }
    }

    /// Explains the current decision.
    fn explain_decision(&self) -> String {
        let budget_used = self.consumed_error_budget / self.config.total_error_budget * 100.0;
        format!(
            "Phase: {:?}, Step: {}/{}, Budget used: {:.1}%, Stable steps: {}",
            self.current_phase,
            self.current_step,
            self.config.total_steps,
            budget_used,
            self.consecutive_stable_steps
        )
    }

    /// Computes confidence in current decision.
    fn compute_confidence(&self) -> f64 {
        let base_confidence = 0.8;

        // Reduce confidence if recently adjusted
        let stability_bonus = (self.consecutive_stable_steps as f64 / 100.0).min(0.15);

        // Reduce confidence if near budget limit
        let budget_fraction =
            self.consumed_error_budget / self.config.total_error_budget;
        let budget_penalty = (budget_fraction - 0.5).max(0.0) * 0.2;

        (base_confidence + stability_bonus - budget_penalty).clamp(0.5, 1.0)
    }

    /// Returns the current statistics.
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Returns the remaining error budget.
    pub fn remaining_budget(&self) -> f64 {
        self.config.total_error_budget - self.consumed_error_budget
    }

    /// Returns the current step.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Returns the current phase.
    pub fn current_phase(&self) -> TrainingPhase {
        self.current_phase
    }

    /// Resets the scheduler for a new training run.
    pub fn reset(&mut self) {
        self.stats = TrainingStats::new();
        self.current_step = 0;
        self.current_phase = TrainingPhase::Warmup;
        self.current_forward_precision = self.config.base_precision;
        self.current_backward_precision = self.config.base_precision;
        self.current_loss_scale = 1.0;
        self.consecutive_stable_steps = 0;
        self.consumed_error_budget = 0.0;
    }

    /// Generates a summary report.
    pub fn summary(&self) -> PrecisionSchedulerSummary {
        let (grad_mean, grad_std, grad_max) = self.stats.grad_norm_stats();

        PrecisionSchedulerSummary {
            total_steps: self.current_step,
            final_phase: self.current_phase,
            final_forward_precision: self.current_forward_precision,
            final_backward_precision: self.current_backward_precision,
            precision_increases: self.stats.precision_increases,
            precision_decreases: self.stats.precision_decreases,
            unstable_steps: self.stats.unstable_steps,
            explosion_steps: self.stats.explosion_steps,
            consumed_budget_fraction: self.consumed_error_budget / self.config.total_error_budget,
            average_grad_norm: grad_mean,
            max_grad_norm: grad_max,
            final_loss_variance: self.stats.loss_variance(),
        }
    }
}

/// Summary of precision scheduler operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSchedulerSummary {
    pub total_steps: usize,
    pub final_phase: TrainingPhase,
    pub final_forward_precision: Precision,
    pub final_backward_precision: Precision,
    pub precision_increases: usize,
    pub precision_decreases: usize,
    pub unstable_steps: usize,
    pub explosion_steps: usize,
    pub consumed_budget_fraction: f64,
    pub average_grad_norm: f64,
    pub max_grad_norm: f64,
    pub final_loss_variance: f64,
}

impl std::fmt::Display for PrecisionSchedulerSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Precision Scheduler Summary")?;
        writeln!(f, "  Total steps: {}", self.total_steps)?;
        writeln!(f, "  Final phase: {:?}", self.final_phase)?;
        writeln!(
            f,
            "  Final precision: {} (fwd) / {} (bwd)",
            self.final_forward_precision.name(),
            self.final_backward_precision.name()
        )?;
        writeln!(
            f,
            "  Precision changes: {} increases, {} decreases",
            self.precision_increases, self.precision_decreases
        )?;
        writeln!(
            f,
            "  Issues: {} unstable, {} explosion",
            self.unstable_steps, self.explosion_steps
        )?;
        writeln!(
            f,
            "  Budget used: {:.1}%",
            self.consumed_budget_fraction * 100.0
        )?;
        writeln!(
            f,
            "  Gradient norm: avg={:.4}, max={:.4}",
            self.average_grad_norm, self.max_grad_norm
        )?;
        writeln!(f, "  Final loss variance: {:.6}", self.final_loss_variance)?;
        Ok(())
    }
}

/// Precision schedule for different layer types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPrecisionSchedule {
    /// Precision for embedding layers.
    pub embedding: Precision,
    /// Precision for attention layers.
    pub attention: Precision,
    /// Precision for feed-forward layers.
    pub feedforward: Precision,
    /// Precision for normalization layers.
    pub normalization: Precision,
    /// Precision for output layers.
    pub output: Precision,
}

impl Default for LayerPrecisionSchedule {
    fn default() -> Self {
        Self {
            embedding: Precision::F32,     // Lookup tables need full precision
            attention: Precision::BF16,    // Can use mixed
            feedforward: Precision::BF16,  // Can use mixed
            normalization: Precision::F32, // Statistics need precision
            output: Precision::F32,        // Output layer needs precision
        }
    }
}

impl LayerPrecisionSchedule {
    /// Creates a high-precision schedule.
    pub fn high_precision() -> Self {
        Self {
            embedding: Precision::F32,
            attention: Precision::F32,
            feedforward: Precision::F32,
            normalization: Precision::F32,
            output: Precision::F32,
        }
    }

    /// Creates a low-precision schedule for inference.
    pub fn low_precision() -> Self {
        Self {
            embedding: Precision::INT8,
            attention: Precision::INT8,
            feedforward: Precision::INT8,
            normalization: Precision::F16,
            output: Precision::F16,
        }
    }

    /// Creates an adaptive schedule based on phase.
    pub fn for_phase(phase: TrainingPhase) -> Self {
        match phase {
            TrainingPhase::Warmup | TrainingPhase::Cooldown => Self::high_precision(),
            TrainingPhase::Main | TrainingPhase::FineTune => Self::default(),
            TrainingPhase::Evaluation => Self::low_precision(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_detection() {
        let config = PrecisionSchedulerConfig {
            total_steps: 1000,
            warmup_steps: 100,
            cooldown_steps: 50,
            ..Default::default()
        };

        let mut scheduler = PrecisionScheduler::new(config);

        // Start in warmup
        assert_eq!(scheduler.current_phase(), TrainingPhase::Warmup);

        // Step through warmup
        for _ in 0..100 {
            scheduler.step(1.0, 1.0, 0.0001);
        }
        assert_eq!(scheduler.current_phase(), TrainingPhase::Main);

        // Step to near end
        for _ in 0..800 {
            scheduler.step(0.5, 1.0, 0.0001);
        }
        // Should be in finetune or cooldown
        assert!(matches!(
            scheduler.current_phase(),
            TrainingPhase::FineTune | TrainingPhase::Cooldown
        ));
    }

    #[test]
    fn test_precision_adjustment() {
        let config = PrecisionSchedulerConfig {
            total_steps: 1000,
            warmup_steps: 0, // Skip warmup for test
            auto_adjust: true,
            grad_explosion_threshold: 10.0,
            ..Default::default()
        };

        let mut scheduler = PrecisionScheduler::new(config);

        // Simulate gradient explosion
        for _ in 0..5 {
            scheduler.step(1.0, 100.0, 0.0001); // Very high gradient norm
        }

        // Should have increased precision
        assert!(scheduler.stats.explosion_steps > 0);
        assert_eq!(scheduler.current_forward_precision, Precision::F32);
    }

    #[test]
    fn test_stable_training_precision_decrease() {
        let config = PrecisionSchedulerConfig {
            total_steps: 10000,
            warmup_steps: 0,
            auto_adjust: true,
            base_precision: Precision::F32,
            stable_steps_threshold: 50, // Lower threshold for test
            ..Default::default()
        };

        let mut scheduler = PrecisionScheduler::new(config);

        // Simulate very stable training
        for _ in 0..150 {
            scheduler.step(0.1, 1.0, 0.00001);
        }

        // Should have tried to decrease precision (after 51 stable steps)
        // With 150 iterations and threshold of 50, we expect at least 2 decreases
        assert!(scheduler.stats().precision_decreases > 0);
        // Verify precision actually changed via decision
        let decision = scheduler.get_decision();
        assert_ne!(decision.forward_precision, Precision::F32);
    }

    #[test]
    fn test_budget_tracking() {
        let config = PrecisionSchedulerConfig {
            total_steps: 100,
            total_error_budget: 0.1,
            ..Default::default()
        };

        let mut scheduler = PrecisionScheduler::new(config);

        for _ in 0..50 {
            scheduler.step(1.0, 1.0, 0.001);
        }

        // Should have consumed ~0.05 of budget
        let remaining = scheduler.remaining_budget();
        assert!((remaining - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let config = PrecisionSchedulerConfig {
            total_steps: 100,
            warmup_steps: 10,
            ..Default::default()
        };

        let mut scheduler = PrecisionScheduler::new(config);

        for i in 0..50 {
            let loss = 1.0 - i as f64 * 0.01;
            scheduler.step(loss, 1.0, 0.0001);
        }

        let summary = scheduler.summary();
        assert_eq!(summary.total_steps, 50);
        assert!(summary.consumed_budget_fraction > 0.0);
    }

    #[test]
    fn test_layer_precision_schedule() {
        let schedule = LayerPrecisionSchedule::for_phase(TrainingPhase::Main);
        assert_eq!(schedule.attention, Precision::BF16);
        assert_eq!(schedule.normalization, Precision::F32);

        let eval_schedule = LayerPrecisionSchedule::for_phase(TrainingPhase::Evaluation);
        assert_eq!(eval_schedule.attention, Precision::INT8);
    }
}
