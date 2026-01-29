//! Optimizer implementations with error tracking.
//!
//! Provides SGD, Adam, and AdamW optimizers that update model parameters
//! based on computed gradients while tracking error bounds.

use std::collections::HashMap;
use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin};

use super::autodiff::NodeIndex;

/// Trait for optimizers.
pub trait Optimizer {
    /// Performs a single optimization step.
    fn step(
        &mut self,
        params: &mut HashMap<NodeIndex, BoundedTensor>,
        grads: &HashMap<NodeIndex, BoundedTensor>,
    );
    
    /// Resets optimizer state (e.g., momentum buffers).
    fn reset(&mut self);
    
    /// Returns the current learning rate.
    fn learning_rate(&self) -> f64;
    
    /// Sets the learning rate.
    fn set_learning_rate(&mut self, lr: f64);
}

/// Stochastic Gradient Descent (SGD) optimizer.
#[derive(Debug)]
pub struct SGD {
    /// Learning rate.
    lr: f64,
    /// Momentum factor.
    momentum: f64,
    /// Weight decay (L2 regularization).
    weight_decay: f64,
    /// Dampening for momentum.
    dampening: f64,
    /// Nesterov momentum.
    nesterov: bool,
    /// Velocity buffers for momentum.
    velocity: HashMap<NodeIndex, BoundedTensor>,
    /// Number of steps taken.
    step_count: usize,
}

impl SGD {
    /// Creates a new SGD optimizer.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
            step_count: 0,
        }
    }

    /// Sets momentum.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enables Nesterov momentum.
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Sets dampening.
    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        params: &mut HashMap<NodeIndex, BoundedTensor>,
        grads: &HashMap<NodeIndex, BoundedTensor>,
    ) {
        self.step_count += 1;

        for (idx, param) in params.iter_mut() {
            if let Some(grad) = grads.get(idx) {
                // Apply weight decay
                let grad_with_decay = if self.weight_decay > 0.0 {
                    let decay_term = param.scale(BoundedValue::exact(self.weight_decay));
                    grad.add(&decay_term)
                } else {
                    grad.clone()
                };

                // Apply momentum
                let update = if self.momentum > 0.0 {
                    let v = self.velocity.entry(*idx).or_insert_with(|| {
                        BoundedTensor::zeros(param.shape().clone())
                    });

                    // v = momentum * v + (1 - dampening) * grad
                    let momentum_term = v.scale(BoundedValue::exact(self.momentum));
                    let grad_term = grad_with_decay.scale(BoundedValue::exact(1.0 - self.dampening));
                    let new_v = momentum_term.add(&grad_term);
                    *v = new_v.clone();

                    if self.nesterov {
                        // Nesterov: grad + momentum * v
                        grad_with_decay.add(&v.scale(BoundedValue::exact(self.momentum)))
                    } else {
                        new_v
                    }
                } else {
                    grad_with_decay
                };

                // param = param - lr * update
                let scaled_update = update.scale(BoundedValue::exact(-self.lr));
                *param = param.add(&scaled_update);
            }
        }
    }

    fn reset(&mut self) {
        self.velocity.clear();
        self.step_count = 0;
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// Adam optimizer (Adaptive Moment Estimation).
#[derive(Debug)]
pub struct Adam {
    /// Learning rate.
    lr: f64,
    /// Exponential decay rate for first moment (beta1).
    beta1: f64,
    /// Exponential decay rate for second moment (beta2).
    beta2: f64,
    /// Numerical stability epsilon.
    eps: f64,
    /// Weight decay (L2 regularization).
    weight_decay: f64,
    /// Whether to use AdamW (decoupled weight decay).
    adamw: bool,
    /// First moment estimates.
    m: HashMap<NodeIndex, BoundedTensor>,
    /// Second moment estimates.
    v: HashMap<NodeIndex, BoundedTensor>,
    /// Number of steps taken.
    step_count: usize,
}

impl Adam {
    /// Creates a new Adam optimizer with default parameters.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            adamw: false,
            m: HashMap::new(),
            v: HashMap::new(),
            step_count: 0,
        }
    }

    /// Sets beta1 (first moment decay).
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets beta2 (second moment decay).
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enables AdamW (decoupled weight decay).
    pub fn with_adamw(mut self, adamw: bool) -> Self {
        self.adamw = adamw;
        self
    }

    /// Creates an AdamW optimizer.
    pub fn adamw(lr: f64, weight_decay: f64) -> Self {
        Self::new(lr).with_weight_decay(weight_decay).with_adamw(true)
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        params: &mut HashMap<NodeIndex, BoundedTensor>,
        grads: &HashMap<NodeIndex, BoundedTensor>,
    ) {
        self.step_count += 1;
        let t = self.step_count as f64;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        for (idx, param) in params.iter_mut() {
            if let Some(grad) = grads.get(idx) {
                // For AdamW, apply weight decay directly to params (decoupled)
                if self.adamw && self.weight_decay > 0.0 {
                    let decay = param.scale(BoundedValue::exact(-self.lr * self.weight_decay));
                    *param = param.add(&decay);
                }

                // For vanilla Adam, add weight decay to gradient
                let grad_wd = if !self.adamw && self.weight_decay > 0.0 {
                    let decay_term = param.scale(BoundedValue::exact(self.weight_decay));
                    grad.add(&decay_term)
                } else {
                    grad.clone()
                };

                // Initialize or update first moment estimate
                let m = self.m.entry(*idx).or_insert_with(|| {
                    BoundedTensor::zeros(param.shape().clone())
                });
                // m = beta1 * m + (1 - beta1) * grad
                let m_new = m
                    .scale(BoundedValue::exact(self.beta1))
                    .add(&grad_wd.scale(BoundedValue::exact(1.0 - self.beta1)));
                *m = m_new;

                // Initialize or update second moment estimate
                let v = self.v.entry(*idx).or_insert_with(|| {
                    BoundedTensor::zeros(param.shape().clone())
                });
                // v = beta2 * v + (1 - beta2) * grad^2
                let grad_sq = grad_wd.hadamard(&grad_wd);
                let v_new = v
                    .scale(BoundedValue::exact(self.beta2))
                    .add(&grad_sq.scale(BoundedValue::exact(1.0 - self.beta2)));
                *v = v_new;

                // Bias-corrected estimates
                let m_hat = self.m.get(idx).unwrap().scale(BoundedValue::exact(1.0 / bias_correction1));
                let v_hat = self.v.get(idx).unwrap().scale(BoundedValue::exact(1.0 / bias_correction2));

                // Compute update: m_hat / (sqrt(v_hat) + eps)
                let update_data: Vec<BoundedValue<f64>> = m_hat
                    .data()
                    .iter()
                    .zip(v_hat.data().iter())
                    .map(|(m_val, v_val)| {
                        let update = m_val.value() / (v_val.value().sqrt() + self.eps);
                        // Error propagation for division
                        let error = m_val.absolute_error() / (v_val.value().sqrt() + self.eps)
                            + m_val.value().abs() * v_val.absolute_error() / (2.0 * v_val.value().sqrt().powi(3) + self.eps);
                        BoundedValue::new(update, ErrorMargin::absolute(error.abs()))
                    })
                    .collect();

                let update = BoundedTensor::new(update_data, param.shape().clone());

                // param = param - lr * update
                let scaled_update = update.scale(BoundedValue::exact(-self.lr));
                *param = param.add(&scaled_update);
            }
        }
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.step_count = 0;
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// Learning rate scheduler trait.
pub trait LRScheduler {
    /// Computes the learning rate for the given step.
    fn get_lr(&self, step: usize) -> f64;
}

/// Constant learning rate (no scheduling).
pub struct ConstantLR {
    lr: f64,
}

impl ConstantLR {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

/// Step decay learning rate scheduler.
pub struct StepLR {
    initial_lr: f64,
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self { initial_lr, step_size, gamma }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self, step: usize) -> f64 {
        let num_decays = step / self.step_size;
        self.initial_lr * self.gamma.powi(num_decays as i32)
    }
}

/// Cosine annealing learning rate scheduler.
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    total_steps: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f64, min_lr: f64, total_steps: usize) -> Self {
        Self { initial_lr, min_lr, total_steps }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f64 {
        let step = step.min(self.total_steps);
        let progress = step as f64 / self.total_steps as f64;
        let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
    }
}

/// Warmup then decay scheduler.
pub struct WarmupScheduler {
    initial_lr: f64,
    warmup_steps: usize,
    inner: Box<dyn LRScheduler>,
}

impl WarmupScheduler {
    pub fn new(initial_lr: f64, warmup_steps: usize, inner: Box<dyn LRScheduler>) -> Self {
        Self { initial_lr, warmup_steps, inner }
    }
}

impl LRScheduler for WarmupScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.initial_lr * (step + 1) as f64 / self.warmup_steps as f64
        } else {
            self.inner.get_lr(step - self.warmup_steps)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_basic() {
        let mut optimizer = SGD::new(0.1);
        
        let mut params: HashMap<NodeIndex, BoundedTensor> = HashMap::new();
        params.insert(0, BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]));
        
        let mut grads: HashMap<NodeIndex, BoundedTensor> = HashMap::new();
        grads.insert(0, BoundedTensor::from_exact(vec![0.1, 0.2, 0.3], vec![3]));
        
        optimizer.step(&mut params, &grads);
        
        let new_params = params.get(&0).unwrap().values();
        // param = param - lr * grad = [1 - 0.01, 2 - 0.02, 3 - 0.03]
        assert!((new_params[0] - 0.99).abs() < 1e-10);
        assert!((new_params[1] - 1.98).abs() < 1e-10);
        assert!((new_params[2] - 2.97).abs() < 1e-10);
    }

    #[test]
    fn test_adam_basic() {
        let mut optimizer = Adam::new(0.001);
        
        let mut params: HashMap<NodeIndex, BoundedTensor> = HashMap::new();
        params.insert(0, BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]));
        
        let mut grads: HashMap<NodeIndex, BoundedTensor> = HashMap::new();
        grads.insert(0, BoundedTensor::from_exact(vec![0.5, 0.5], vec![2]));
        
        // Run a few steps
        for _ in 0..3 {
            optimizer.step(&mut params, &grads);
        }
        
        // Params should have decreased
        let new_params = params.get(&0).unwrap().values();
        assert!(new_params[0] < 1.0);
        assert!(new_params[1] < 2.0);
    }

    #[test]
    fn test_step_lr_scheduler() {
        let scheduler = StepLR::new(0.1, 10, 0.1);
        
        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(9) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(10) - 0.01).abs() < 1e-10);
        assert!((scheduler.get_lr(20) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_scheduler() {
        let scheduler = CosineAnnealingLR::new(0.1, 0.0, 100);
        
        // At step 0, should be initial_lr
        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);
        
        // At step 50, should be around half
        let mid_lr = scheduler.get_lr(50);
        assert!(mid_lr > 0.04 && mid_lr < 0.06);
        
        // At step 100, should be min_lr
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-10);
    }
}
