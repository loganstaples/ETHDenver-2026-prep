//! Training Metrics and Convergence Tracking.
//!
//! Provides comprehensive metrics collection for training runs including loss,
//! gradient statistics, and convergence detection.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Training metrics over time.
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    /// Configuration for metrics tracking.
    config: MetricsConfig,
    /// Loss history.
    loss_window: VecDeque<f64>,
    /// Gradient norm history.
    gradient_norm_window: VecDeque<f64>,
    /// Error bound history.
    error_bound_window: VecDeque<f64>,
    /// Per-iteration metrics.
    iteration_metrics: VecDeque<IterationMetrics>,
    /// Current iteration.
    current_iteration: u64,
    /// Training start time.
    start_time: Option<Instant>,
    /// Convergence detector.
    convergence: ConvergenceDetector,
}

/// Configuration for metrics tracking.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Window size for moving averages.
    pub window_size: usize,
    /// Maximum history to keep.
    pub max_history: usize,
    /// Convergence threshold (relative change).
    pub convergence_threshold: f64,
    /// Number of iterations to check for convergence.
    pub convergence_window: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            max_history: 10000,
            convergence_threshold: 0.001,
            convergence_window: 50,
        }
    }
}

/// Metrics for a single training iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationMetrics {
    /// Iteration number.
    pub iteration: u64,
    /// Training loss.
    pub loss: f64,
    /// Gradient L2 norm.
    pub gradient_norm: f64,
    /// Accumulated error bound.
    pub error_bound: f64,
    /// Learning rate used.
    pub learning_rate: f64,
    /// Number of samples processed.
    pub batch_size: usize,
    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
    /// Memory usage in bytes (if available).
    pub memory_bytes: Option<u64>,
    /// Perplexity (for language models).
    pub perplexity: Option<f64>,
    /// Accuracy (for classification).
    pub accuracy: Option<f64>,
    /// Custom metrics.
    pub custom: std::collections::HashMap<String, f64>,
}

impl IterationMetrics {
    /// Creates new iteration metrics.
    pub fn new(iteration: u64) -> Self {
        Self {
            iteration,
            loss: 0.0,
            gradient_norm: 0.0,
            error_bound: 0.0,
            learning_rate: 0.0,
            batch_size: 0,
            processing_time_ms: 0,
            memory_bytes: None,
            perplexity: None,
            accuracy: None,
            custom: std::collections::HashMap::new(),
        }
    }

    /// Sets the loss value.
    pub fn with_loss(mut self, loss: f64) -> Self {
        self.loss = loss;
        // Compute perplexity for language models: exp(loss) for cross-entropy
        if loss.is_finite() && loss >= 0.0 && loss < 20.0 {
            self.perplexity = Some(loss.exp());
        }
        self
    }

    /// Sets the gradient norm.
    pub fn with_gradient_norm(mut self, norm: f64) -> Self {
        self.gradient_norm = norm;
        self
    }

    /// Sets the error bound.
    pub fn with_error_bound(mut self, bound: f64) -> Self {
        self.error_bound = bound;
        self
    }

    /// Sets the learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Sets processing time.
    pub fn with_processing_time(mut self, duration: Duration) -> Self {
        self.processing_time_ms = duration.as_millis() as u64;
        self
    }

    /// Adds a custom metric.
    pub fn with_custom_metric(mut self, name: &str, value: f64) -> Self {
        self.custom.insert(name.to_string(), value);
        self
    }
}

impl MetricsTracker {
    /// Creates a new metrics tracker.
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            convergence: ConvergenceDetector::new(config.convergence_threshold, config.convergence_window),
            config,
            loss_window: VecDeque::new(),
            gradient_norm_window: VecDeque::new(),
            error_bound_window: VecDeque::new(),
            iteration_metrics: VecDeque::new(),
            current_iteration: 0,
            start_time: None,
        }
    }

    /// Starts the training timer.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Records metrics for an iteration.
    pub fn record(&mut self, metrics: IterationMetrics) {
        // Update iteration counter
        self.current_iteration = metrics.iteration;

        // Add to windows
        self.loss_window.push_back(metrics.loss);
        if self.loss_window.len() > self.config.window_size {
            self.loss_window.pop_front();
        }

        self.gradient_norm_window.push_back(metrics.gradient_norm);
        if self.gradient_norm_window.len() > self.config.window_size {
            self.gradient_norm_window.pop_front();
        }

        self.error_bound_window.push_back(metrics.error_bound);
        if self.error_bound_window.len() > self.config.window_size {
            self.error_bound_window.pop_front();
        }

        // Update convergence detector
        self.convergence.add_sample(metrics.loss);

        // Add to history
        self.iteration_metrics.push_back(metrics);
        if self.iteration_metrics.len() > self.config.max_history {
            self.iteration_metrics.pop_front();
        }
    }

    /// Returns the current iteration.
    pub fn current_iteration(&self) -> u64 {
        self.current_iteration
    }

    /// Returns the most recent loss.
    pub fn current_loss(&self) -> Option<f64> {
        self.loss_window.back().copied()
    }

    /// Returns the moving average of loss.
    pub fn average_loss(&self) -> f64 {
        if self.loss_window.is_empty() {
            return 0.0;
        }
        self.loss_window.iter().sum::<f64>() / self.loss_window.len() as f64
    }

    /// Returns the moving average of gradient norm.
    pub fn average_gradient_norm(&self) -> f64 {
        if self.gradient_norm_window.is_empty() {
            return 0.0;
        }
        self.gradient_norm_window.iter().sum::<f64>() / self.gradient_norm_window.len() as f64
    }

    /// Returns the moving average of error bound.
    pub fn average_error_bound(&self) -> f64 {
        if self.error_bound_window.is_empty() {
            return 0.0;
        }
        self.error_bound_window.iter().sum::<f64>() / self.error_bound_window.len() as f64
    }

    /// Returns whether training has converged.
    pub fn has_converged(&self) -> bool {
        self.convergence.has_converged()
    }

    /// Returns the total training time.
    pub fn elapsed(&self) -> Duration {
        self.start_time.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Returns samples per second.
    pub fn samples_per_second(&self) -> f64 {
        let total_samples: usize = self.iteration_metrics.iter().map(|m| m.batch_size).sum();
        let elapsed_secs = self.elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            total_samples as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    /// Returns a summary of training progress.
    pub fn summary(&self) -> TrainingSummary {
        let min_loss = self.loss_window.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_loss = self.loss_window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        TrainingSummary {
            total_iterations: self.current_iteration,
            elapsed_seconds: self.elapsed().as_secs(),
            samples_per_second: self.samples_per_second(),
            current_loss: self.current_loss().unwrap_or(0.0),
            average_loss: self.average_loss(),
            min_loss,
            max_loss,
            average_gradient_norm: self.average_gradient_norm(),
            average_error_bound: self.average_error_bound(),
            has_converged: self.has_converged(),
        }
    }

    /// Returns the last N iteration metrics.
    pub fn last_n(&self, n: usize) -> Vec<&IterationMetrics> {
        self.iteration_metrics.iter().rev().take(n).collect()
    }

    /// Returns all iteration metrics.
    pub fn all_metrics(&self) -> impl Iterator<Item = &IterationMetrics> {
        self.iteration_metrics.iter()
    }
}

/// Summary of training progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// Total iterations completed.
    pub total_iterations: u64,
    /// Total elapsed time in seconds.
    pub elapsed_seconds: u64,
    /// Throughput (samples per second).
    pub samples_per_second: f64,
    /// Latest loss value.
    pub current_loss: f64,
    /// Moving average of loss.
    pub average_loss: f64,
    /// Minimum loss in window.
    pub min_loss: f64,
    /// Maximum loss in window.
    pub max_loss: f64,
    /// Average gradient norm.
    pub average_gradient_norm: f64,
    /// Average error bound.
    pub average_error_bound: f64,
    /// Whether training has converged.
    pub has_converged: bool,
}

impl std::fmt::Display for TrainingSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Iter: {} | Loss: {:.4} (avg: {:.4}) | Grad: {:.4} | Error: {:.6} | {:.1} samples/s{}",
            self.total_iterations,
            self.current_loss,
            self.average_loss,
            self.average_gradient_norm,
            self.average_error_bound,
            self.samples_per_second,
            if self.has_converged { " [CONVERGED]" } else { "" }
        )
    }
}

/// Detects convergence based on loss plateau.
#[derive(Debug, Clone)]
pub struct ConvergenceDetector {
    /// Threshold for relative change.
    threshold: f64,
    /// Window size to check.
    window_size: usize,
    /// Recent loss values.
    samples: VecDeque<f64>,
    /// Whether converged flag was set.
    converged: bool,
}

impl ConvergenceDetector {
    /// Creates a new convergence detector.
    pub fn new(threshold: f64, window_size: usize) -> Self {
        Self {
            threshold,
            window_size,
            samples: VecDeque::new(),
            converged: false,
        }
    }

    /// Adds a sample and updates convergence status.
    pub fn add_sample(&mut self, value: f64) {
        self.samples.push_back(value);
        if self.samples.len() > self.window_size {
            self.samples.pop_front();
        }

        // Check for convergence only if we have enough samples
        if self.samples.len() >= self.window_size {
            self.converged = self.check_convergence();
        }
    }

    /// Returns whether training has converged.
    pub fn has_converged(&self) -> bool {
        self.converged
    }

    /// Checks if the loss has plateaued.
    fn check_convergence(&self) -> bool {
        if self.samples.len() < 2 {
            return false;
        }

        let mean: f64 = self.samples.iter().sum::<f64>() / self.samples.len() as f64;
        if mean.abs() < 1e-10 {
            return true; // Loss is essentially zero
        }

        // Compute coefficient of variation
        let variance: f64 = self.samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.samples.len() as f64;
        let cv = variance.sqrt() / mean.abs();

        cv < self.threshold
    }

    /// Resets the detector.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.converged = false;
    }
}

/// Tracks learning rate schedules.
#[derive(Debug, Clone)]
pub struct LearningRateSchedule {
    /// Initial learning rate.
    initial_lr: f64,
    /// Current learning rate.
    current_lr: f64,
    /// Schedule type.
    schedule_type: ScheduleType,
    /// Warmup steps.
    warmup_steps: u64,
    /// Total training steps.
    total_steps: u64,
    /// Current step.
    current_step: u64,
}

/// Types of learning rate schedules.
#[derive(Debug, Clone, Copy)]
pub enum ScheduleType {
    /// Constant learning rate.
    Constant,
    /// Linear warmup then constant.
    LinearWarmup,
    /// Cosine annealing.
    CosineAnnealing,
    /// Linear decay.
    LinearDecay,
    /// Exponential decay with given rate.
    ExponentialDecay { decay_rate: f64 },
    /// Step decay (reduce by factor every N steps).
    StepDecay { step_size: u64, gamma: f64 },
}

impl LearningRateSchedule {
    /// Creates a new learning rate schedule.
    pub fn new(
        initial_lr: f64,
        schedule_type: ScheduleType,
        warmup_steps: u64,
        total_steps: u64,
    ) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            schedule_type,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Steps the schedule and returns the current learning rate.
    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.current_lr = self.compute_lr();
        self.current_lr
    }

    /// Returns the current learning rate without stepping.
    pub fn current(&self) -> f64 {
        self.current_lr
    }

    /// Computes the learning rate for the current step.
    fn compute_lr(&self) -> f64 {
        // Handle warmup
        if self.current_step < self.warmup_steps {
            return self.initial_lr * (self.current_step as f64 / self.warmup_steps as f64);
        }

        let step_after_warmup = self.current_step - self.warmup_steps;
        let steps_after_warmup = self.total_steps.saturating_sub(self.warmup_steps);

        match self.schedule_type {
            ScheduleType::Constant => self.initial_lr,
            ScheduleType::LinearWarmup => self.initial_lr,
            ScheduleType::CosineAnnealing => {
                let progress = step_after_warmup as f64 / steps_after_warmup.max(1) as f64;
                let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                self.initial_lr * cosine
            }
            ScheduleType::LinearDecay => {
                let progress = step_after_warmup as f64 / steps_after_warmup.max(1) as f64;
                self.initial_lr * (1.0 - progress).max(0.0)
            }
            ScheduleType::ExponentialDecay { decay_rate } => {
                self.initial_lr * decay_rate.powi(step_after_warmup as i32)
            }
            ScheduleType::StepDecay { step_size, gamma } => {
                let num_decays = step_after_warmup / step_size;
                self.initial_lr * gamma.powi(num_decays as i32)
            }
        }
    }

    /// Resets the schedule.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.initial_lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_tracker() {
        let config = MetricsConfig::default();
        let mut tracker = MetricsTracker::new(config);
        tracker.start();

        for i in 0..10 {
            let metrics = IterationMetrics::new(i)
                .with_loss(1.0 / (i as f64 + 1.0))
                .with_gradient_norm(0.1)
                .with_batch_size(8);
            tracker.record(metrics);
        }

        assert_eq!(tracker.current_iteration(), 9);
        assert!(tracker.average_loss() > 0.0);
    }

    #[test]
    fn test_convergence_detection() {
        let mut detector = ConvergenceDetector::new(0.001, 10);

        // Add converging samples (constant)
        for _ in 0..20 {
            detector.add_sample(0.5);
        }

        assert!(detector.has_converged());
    }

    #[test]
    fn test_learning_rate_schedule() {
        let mut schedule = LearningRateSchedule::new(
            0.1,
            ScheduleType::CosineAnnealing,
            10,
            100,
        );

        // Warmup phase
        for _ in 0..10 {
            let lr = schedule.step();
            assert!(lr <= schedule.initial_lr);
        }

        // Decay phase
        let lr_at_warmup = schedule.current();
        for _ in 0..50 {
            schedule.step();
        }
        assert!(schedule.current() < lr_at_warmup);
    }
}
