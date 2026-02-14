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

/// Combined linear warmup followed by cosine decay to min_lr.
///
/// This is the most commonly used scheduler in modern ML training:
/// 1. Linear warmup from 0 to `peak_lr` over `warmup_steps`
/// 2. Cosine decay from `peak_lr` to `min_lr` over remaining steps
pub struct LinearWarmupCosineDecay {
    peak_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl LinearWarmupCosineDecay {
    pub fn new(peak_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            peak_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LRScheduler for LinearWarmupCosineDecay {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.peak_lr * (step + 1) as f64 / self.warmup_steps.max(1) as f64
        } else {
            // Cosine decay
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
            let progress = (step - self.warmup_steps).min(decay_steps) as f64 / decay_steps as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        }
    }
}

/// Exponential learning rate decay.
///
/// lr = initial_lr * gamma^step
pub struct ExponentialLR {
    initial_lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self, step: usize) -> f64 {
        self.initial_lr * self.gamma.powi(step as i32)
    }
}

/// Polynomial learning rate decay.
///
/// lr = (initial_lr - end_lr) * (1 - step/total_steps)^power + end_lr
pub struct PolynomialLR {
    initial_lr: f64,
    end_lr: f64,
    total_steps: usize,
    power: f64,
}

impl PolynomialLR {
    pub fn new(initial_lr: f64, end_lr: f64, total_steps: usize, power: f64) -> Self {
        Self {
            initial_lr,
            end_lr,
            total_steps,
            power,
        }
    }

    /// Creates a linear decay (power=1.0).
    pub fn linear(initial_lr: f64, end_lr: f64, total_steps: usize) -> Self {
        Self::new(initial_lr, end_lr, total_steps, 1.0)
    }
}

impl LRScheduler for PolynomialLR {
    fn get_lr(&self, step: usize) -> f64 {
        let step = step.min(self.total_steps);
        let ratio = 1.0 - step as f64 / self.total_steps.max(1) as f64;
        (self.initial_lr - self.end_lr) * ratio.powf(self.power) + self.end_lr
    }
}

/// One-cycle learning rate policy (Smith, 2018).
///
/// Phase 1 (0 → pct_start): Linear warmup from div_factor*max_lr to max_lr
/// Phase 2 (pct_start → 1.0): Cosine decay from max_lr to max_lr/final_div_factor
pub struct OneCycleLR {
    max_lr: f64,
    div_factor: f64,
    final_div_factor: f64,
    total_steps: usize,
    pct_start: f64,
}

impl OneCycleLR {
    pub fn new(max_lr: f64, total_steps: usize) -> Self {
        Self {
            max_lr,
            div_factor: 25.0,
            final_div_factor: 1e4,
            total_steps,
            pct_start: 0.3,
        }
    }

    /// Sets the initial lr divisor (initial_lr = max_lr / div_factor).
    pub fn with_div_factor(mut self, div_factor: f64) -> Self {
        self.div_factor = div_factor;
        self
    }

    /// Sets the final lr divisor (final_lr = max_lr / final_div_factor).
    pub fn with_final_div_factor(mut self, final_div_factor: f64) -> Self {
        self.final_div_factor = final_div_factor;
        self
    }

    /// Sets the percentage of training spent in warmup phase.
    pub fn with_pct_start(mut self, pct_start: f64) -> Self {
        self.pct_start = pct_start;
        self
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self, step: usize) -> f64 {
        let total = self.total_steps.max(1) as f64;
        let progress = (step as f64 / total).min(1.0);
        let initial_lr = self.max_lr / self.div_factor;
        let final_lr = self.max_lr / self.final_div_factor;

        if progress <= self.pct_start {
            // Phase 1: warmup
            let phase_progress = progress / self.pct_start.max(1e-10);
            initial_lr + (self.max_lr - initial_lr) * phase_progress
        } else {
            // Phase 2: cosine annealing
            let phase_progress =
                (progress - self.pct_start) / (1.0 - self.pct_start).max(1e-10);
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * phase_progress).cos());
            final_lr + (self.max_lr - final_lr) * cosine
        }
    }
}

/// Cosine annealing with warm restarts (SGDR, Loshchilov & Hutter 2016).
///
/// After each restart, the period is multiplied by `t_mult`.
/// lr(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * T_cur / T_i))
///
/// where T_cur is steps since last restart, T_i is the current period length.
pub struct CosineAnnealingWarmRestarts {
    /// Maximum (initial) learning rate.
    eta_max: f64,
    /// Minimum learning rate.
    eta_min: f64,
    /// Initial restart period (in steps).
    t_0: usize,
    /// Period multiplier after each restart.
    t_mult: f64,
}

impl CosineAnnealingWarmRestarts {
    /// Creates a new warm restarts scheduler.
    ///
    /// - `eta_max`: Peak learning rate at each restart.
    /// - `t_0`: Number of steps in the first cycle.
    /// - `t_mult`: Multiplier for the period after each restart (1.0 = constant period).
    /// - `eta_min`: Minimum learning rate (default: 0.0).
    pub fn new(eta_max: f64, t_0: usize, t_mult: f64, eta_min: f64) -> Self {
        Self {
            eta_max,
            eta_min,
            t_0: t_0.max(1),
            t_mult: t_mult.max(1.0),
        }
    }
}

impl LRScheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&self, step: usize) -> f64 {
        if self.t_mult == 1.0 {
            // Constant period: simple modular arithmetic
            let t_cur = step % self.t_0;
            let progress = t_cur as f64 / self.t_0 as f64;
            self.eta_min + 0.5 * (self.eta_max - self.eta_min)
                * (1.0 + (std::f64::consts::PI * progress).cos())
        } else {
            // Geometric series of periods: T_0, T_0*T_mult, T_0*T_mult^2, ...
            // Find which cycle we're in: sum of geometric series = T_0 * (T_mult^n - 1) / (T_mult - 1)
            let mut t_remaining = step as f64;
            let mut t_i = self.t_0 as f64;

            loop {
                if t_remaining < t_i {
                    let progress = t_remaining / t_i;
                    return self.eta_min + 0.5 * (self.eta_max - self.eta_min)
                        * (1.0 + (std::f64::consts::PI * progress).cos());
                }
                t_remaining -= t_i;
                t_i *= self.t_mult;
            }
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

    #[test]
    fn test_linear_warmup_cosine_decay() {
        let scheduler = LinearWarmupCosineDecay::new(0.001, 1e-6, 100, 1000);

        // During warmup: linear increase
        assert!((scheduler.get_lr(0) - 0.001 / 100.0).abs() < 1e-8);
        assert!((scheduler.get_lr(49) - 0.001 * 50.0 / 100.0).abs() < 1e-8);
        assert!((scheduler.get_lr(99) - 0.001).abs() < 1e-8);

        // After warmup: cosine decay
        let at_warmup_end = scheduler.get_lr(100);
        assert!((at_warmup_end - 0.001).abs() < 1e-6);

        // Midpoint of decay
        let mid = scheduler.get_lr(550);
        assert!(mid > 1e-6 && mid < 0.001);

        // End: should approach min_lr
        let end = scheduler.get_lr(1000);
        assert!((end - 1e-6).abs() < 1e-7);
    }

    #[test]
    fn test_exponential_lr() {
        let scheduler = ExponentialLR::new(0.1, 0.9);

        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(1) - 0.09).abs() < 1e-10);
        assert!((scheduler.get_lr(2) - 0.081).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_lr_linear() {
        let scheduler = PolynomialLR::linear(0.1, 0.0, 100);

        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(50) - 0.05).abs() < 1e-10);
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_lr_quadratic() {
        let scheduler = PolynomialLR::new(0.1, 0.0, 100, 2.0);

        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);
        // (1 - 50/100)^2 = 0.25
        assert!((scheduler.get_lr(50) - 0.025).abs() < 1e-10);
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_warm_restarts_constant_period() {
        let scheduler = CosineAnnealingWarmRestarts::new(0.1, 10, 1.0, 0.0);

        // At start of each cycle: eta_max
        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(10) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(20) - 0.1).abs() < 1e-10);

        // At midpoint of cycle: ~eta_min (cosine at pi = -1, so (1+(-1))/2 = 0)
        // Actually at step 5: cos(pi * 5/10) = cos(pi/2) = 0, so lr = 0.05
        let mid = scheduler.get_lr(5);
        assert!((mid - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_warm_restarts_increasing_period() {
        let scheduler = CosineAnnealingWarmRestarts::new(0.1, 10, 2.0, 0.001);

        // Start: eta_max
        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);

        // After first cycle (10 steps), restart at step 10
        assert!((scheduler.get_lr(10) - 0.1).abs() < 1e-6);

        // Second cycle is 20 steps (10 * 2), restart at step 30
        assert!((scheduler.get_lr(30) - 0.1).abs() < 1e-6);

        // Third cycle is 40 steps (20 * 2), so step 70 is restart
        assert!((scheduler.get_lr(70) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_one_cycle_lr() {
        let scheduler = OneCycleLR::new(0.01, 1000).with_pct_start(0.3);

        // Start: max_lr / div_factor = 0.01 / 25 = 0.0004
        let start = scheduler.get_lr(0);
        assert!((start - 0.0004).abs() < 1e-6);

        // At pct_start (step 300): should be at max_lr
        let peak = scheduler.get_lr(300);
        assert!((peak - 0.01).abs() < 1e-4);

        // End: max_lr / final_div_factor = 0.01 / 10000 = 1e-6
        let end = scheduler.get_lr(1000);
        assert!((end - 1e-6).abs() < 1e-7);

        // Should be monotonically decreasing after peak
        let lr_400 = scheduler.get_lr(400);
        let lr_600 = scheduler.get_lr(600);
        let lr_800 = scheduler.get_lr(800);
        assert!(lr_400 > lr_600);
        assert!(lr_600 > lr_800);
    }
}
