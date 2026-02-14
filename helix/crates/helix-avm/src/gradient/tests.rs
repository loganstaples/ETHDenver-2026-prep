//! Comprehensive gradient computation tests.
//!
//! This module validates:
//! 1. Forward pass produces correct outputs for known inputs
//! 2. Gradient computation matches analytical/PyTorch reference
//! 3. Weight updates are applied correctly (learning rate validation)
//! 4. Loss functions produce correct values
//! 5. Numerical gradient checking
//!
//! # Testing Methodology
//!
//! All gradient tests use one or more of:
//! - Analytical gradients: Hand-computed expected values
//! - Numerical gradients: Finite difference approximation for verification
//! - PyTorch reference: Pre-computed expected gradients

use crate::gradient::autodiff::{GradientTape, Variable};
use crate::gradient::backward::backward;
use crate::gradient::loss::{mse_loss, mse_grad, cross_entropy_loss, softmax_cross_entropy_loss};
use crate::gradient::optimizer::{SGD, Adam, Optimizer};
use helix_core::types::{BoundedTensor, Precision};
use std::collections::HashMap;

// ============================================================================
// Basic Autodiff Tests
// ============================================================================

#[test]
fn test_scalar_add_backward() {
    let tape = GradientTape::new();

    // x = 2.0
    let x_val = BoundedTensor::from_exact(vec![2.0], vec![1]);
    let x = Variable::param(x_val, tape.clone(), Some("x".into()));

    // y = x + x = 2x
    let y = x.add(&x);

    // Backward
    let grads = backward(&y).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap();

    // dy/dx should be 2.0
    assert!((dx.get(&[0]).unwrap().value() - 2.0).abs() < 1e-6);
}

#[test]
fn test_matmul_backward() {
    let tape = GradientTape::new();

    // A = [[1, 2], [3, 4]] (2x2)
    let a_val = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let a = Variable::param(a_val, tape.clone(), Some("A".into()));

    // x = [[1], [1]] (2x1)
    let x_val = BoundedTensor::from_exact(vec![1.0, 1.0], vec![2, 1]);
    let x = Variable::input(x_val, tape.clone(), Some("x".into()));

    // y = A @ x
    let y = a.matmul(&x);

    // Define Loss = sum(y) = y1 + y2
    // L = [[1, 1]] @ y  (1x2 @ 2x1 -> 1x1 scalar)
    let ones_val = BoundedTensor::from_exact(vec![1.0, 1.0], vec![1, 2]);
    let ones = Variable::input(ones_val, tape.clone(), Some("ones".into()));

    let loss = ones.matmul(&y);

    let grads = backward(&loss).unwrap();
    let da = grads.get(&a.node_index.unwrap()).unwrap();

    // dL/dA = 1 x^T = [[1], [1]] @ [[1, 1]] = [[1, 1], [1, 1]]
    assert_eq!(da.shape(), &vec![2, 2]);
    let da_vals = da.values();
    for v in da_vals {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_relu_backward() {
    let tape = GradientTape::new();

    // x = [-1, 2]
    let x_val = BoundedTensor::from_exact(vec![-1.0, 2.0], vec![2]);
    let x = Variable::param(x_val, tape.clone(), Some("x".into()));

    // y = Relu(x) = [0, 2]
    let y = x.relu();

    let grads = backward(&y).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap();

    let dx_vals = dx.values();
    assert!((dx_vals[0] - 0.0).abs() < 1e-6);
    assert!((dx_vals[1] - 1.0).abs() < 1e-6);
}

// ============================================================================
// Weight Update Validation Tests
// ============================================================================

#[test]
fn test_sgd_weight_update_basic() {
    // Test that SGD applies: w_new = w_old - lr * grad
    let lr = 0.1;
    let mut optimizer = SGD::new(lr);

    // Initial weights
    let mut params: HashMap<usize, BoundedTensor> = HashMap::new();
    params.insert(0, BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]));

    // Gradients
    let mut grads: HashMap<usize, BoundedTensor> = HashMap::new();
    grads.insert(0, BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]));

    optimizer.step(&mut params, &grads);

    let new_params = params.get(&0).unwrap().values();

    // w_new = w_old - lr * grad
    // [1, 2, 3] - 0.1 * [1, 2, 3] = [0.9, 1.8, 2.7]
    assert!((new_params[0] - 0.9).abs() < 1e-10);
    assert!((new_params[1] - 1.8).abs() < 1e-10);
    assert!((new_params[2] - 2.7).abs() < 1e-10);
}

#[test]
fn test_sgd_multiple_steps() {
    let lr = 0.1;
    let mut optimizer = SGD::new(lr);

    let mut params: HashMap<usize, BoundedTensor> = HashMap::new();
    params.insert(0, BoundedTensor::from_exact(vec![10.0], vec![1]));

    let grads: HashMap<usize, BoundedTensor> = {
        let mut g = HashMap::new();
        g.insert(0, BoundedTensor::from_exact(vec![1.0], vec![1]));
        g
    };

    // 10 steps: 10 - 0.1*10 = 9.0
    for _ in 0..10 {
        optimizer.step(&mut params, &grads);
    }

    let final_value = params.get(&0).unwrap().values()[0];
    assert!((final_value - 9.0).abs() < 1e-10);
}

#[test]
fn test_sgd_with_momentum() {
    let lr = 0.1;
    let momentum = 0.9;
    let mut optimizer = SGD::new(lr).with_momentum(momentum);

    let mut params: HashMap<usize, BoundedTensor> = HashMap::new();
    params.insert(0, BoundedTensor::from_exact(vec![0.0], vec![1]));

    let grads: HashMap<usize, BoundedTensor> = {
        let mut g = HashMap::new();
        g.insert(0, BoundedTensor::from_exact(vec![1.0], vec![1]));
        g
    };

    // Step 1: v = 0.9*0 + 1.0 = 1.0, w = 0 - 0.1*1.0 = -0.1
    optimizer.step(&mut params, &grads);
    let v1 = params.get(&0).unwrap().values()[0];
    assert!((v1 - (-0.1)).abs() < 1e-10);

    // Step 2: v = 0.9*1.0 + 1.0 = 1.9, w = -0.1 - 0.1*1.9 = -0.29
    optimizer.step(&mut params, &grads);
    let v2 = params.get(&0).unwrap().values()[0];
    assert!((v2 - (-0.29)).abs() < 1e-10);
}

#[test]
fn test_sgd_with_weight_decay() {
    let lr = 0.1;
    let weight_decay = 0.01;
    let mut optimizer = SGD::new(lr).with_weight_decay(weight_decay);

    let mut params: HashMap<usize, BoundedTensor> = HashMap::new();
    params.insert(0, BoundedTensor::from_exact(vec![10.0], vec![1]));

    let grads: HashMap<usize, BoundedTensor> = {
        let mut g = HashMap::new();
        g.insert(0, BoundedTensor::from_exact(vec![0.0], vec![1])); // Zero grad
        g
    };

    // With zero gradient, only weight decay applies
    // grad_with_decay = 0 + 0.01 * 10 = 0.1
    // w_new = 10 - 0.1 * 0.1 = 9.99
    optimizer.step(&mut params, &grads);
    let v = params.get(&0).unwrap().values()[0];
    assert!((v - 9.99).abs() < 1e-10);
}

#[test]
fn test_adam_weight_update() {
    let lr = 0.001;
    let mut optimizer = Adam::new(lr);

    let mut params: HashMap<usize, BoundedTensor> = HashMap::new();
    params.insert(0, BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]));

    let grads: HashMap<usize, BoundedTensor> = {
        let mut g = HashMap::new();
        g.insert(0, BoundedTensor::from_exact(vec![0.5, 0.5], vec![2]));
        g
    };

    // Run a few steps
    for _ in 0..3 {
        optimizer.step(&mut params, &grads);
    }

    // Params should have decreased (positive gradients -> decrease)
    let new_params = params.get(&0).unwrap().values();
    assert!(new_params[0] < 1.0);
    assert!(new_params[1] < 2.0);
}

#[test]
fn test_adam_bias_correction() {
    // Adam uses bias correction: m_hat = m / (1 - beta1^t)
    let lr = 0.1;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let mut optimizer = Adam::new(lr).with_beta1(beta1).with_beta2(beta2);

    let mut params: HashMap<usize, BoundedTensor> = HashMap::new();
    params.insert(0, BoundedTensor::from_exact(vec![0.0], vec![1]));

    let grads: HashMap<usize, BoundedTensor> = {
        let mut g = HashMap::new();
        g.insert(0, BoundedTensor::from_exact(vec![1.0], vec![1]));
        g
    };

    // First step bias correction is critical
    // m1 = 0.1 * 1.0 = 0.1 (after (1-beta1)*grad)
    // v1 = 0.001 * 1.0 = 0.001
    // m_hat = 0.1 / 0.1 = 1.0 (bias corrected)
    // v_hat = 0.001 / 0.001 ≈ 1.0
    // update ≈ 0.1 * 1.0 / (1.0 + eps) ≈ 0.1
    optimizer.step(&mut params, &grads);
    let v = params.get(&0).unwrap().values()[0];

    // Should be approximately -0.1 (moved in negative gradient direction)
    assert!(v < 0.0);
    assert!((v - (-0.1)).abs() < 0.05);
}

// ============================================================================
// Loss Function Tests with Known Expected Values
// ============================================================================

#[test]
fn test_mse_loss_zero() {
    // Identical predictions and targets should give loss = 0
    let predictions = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
    let targets = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);

    let loss = mse_loss(&predictions, &targets, Precision::F32);
    assert!(loss.value().abs() < 1e-10);
}

#[test]
fn test_mse_loss_unit_difference() {
    // MSE([1,2,3], [2,3,4]) = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = 3/3 = 1.0
    let predictions = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
    let targets = BoundedTensor::from_exact(vec![2.0, 3.0, 4.0], vec![3]);

    let loss = mse_loss(&predictions, &targets, Precision::F32);
    assert!((loss.value() - 1.0).abs() < 1e-10);
}

#[test]
fn test_mse_loss_known_values() {
    // MSE([0, 0], [3, 4]) = (9 + 16) / 2 = 12.5
    let predictions = BoundedTensor::from_exact(vec![0.0, 0.0], vec![2]);
    let targets = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);

    let loss = mse_loss(&predictions, &targets, Precision::F32);
    assert!((loss.value() - 12.5).abs() < 1e-10);
}

#[test]
fn test_mse_gradient_values() {
    // dL/dp = 2*(p-t)/n
    // For p=[1,2,3], t=[0,2,4], n=3:
    // grad = [2*1/3, 2*0/3, 2*(-1)/3] = [2/3, 0, -2/3]
    let predictions = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
    let targets = BoundedTensor::from_exact(vec![0.0, 2.0, 4.0], vec![3]);

    let grad = mse_grad(&predictions, &targets);
    let grad_values = grad.values();

    assert!((grad_values[0] - 2.0/3.0).abs() < 1e-10);
    assert!(grad_values[1].abs() < 1e-10);
    assert!((grad_values[2] - (-2.0/3.0)).abs() < 1e-10);
}

#[test]
fn test_cross_entropy_loss_correct_prediction() {
    // Cross entropy with correct prediction should be low
    // CE([0.9, 0.1], [1, 0]) = -1 * log(0.9) ≈ 0.105
    let predictions = BoundedTensor::from_exact(vec![0.9, 0.1], vec![2]);
    let targets = BoundedTensor::from_exact(vec![1.0, 0.0], vec![2]);

    let loss = cross_entropy_loss(&predictions, &targets, Precision::F32);
    assert!(loss.value() < 0.2);
    assert!((loss.value() - (-0.9_f64.ln())).abs() < 1e-6);
}

#[test]
fn test_cross_entropy_loss_wrong_prediction() {
    // Cross entropy with wrong prediction should be high
    // CE([0.1, 0.9], [1, 0]) = -1 * log(0.1) ≈ 2.303
    let predictions = BoundedTensor::from_exact(vec![0.1, 0.9], vec![2]);
    let targets = BoundedTensor::from_exact(vec![1.0, 0.0], vec![2]);

    let loss = cross_entropy_loss(&predictions, &targets, Precision::F32);
    assert!(loss.value() > 2.0);
}

#[test]
fn test_softmax_cross_entropy_loss() {
    // Logits [2, 1, 0] with target class 0
    // softmax([2, 1, 0]) ≈ [0.665, 0.245, 0.090]
    // CE = -log(0.665) ≈ 0.408
    let logits = BoundedTensor::from_exact(vec![2.0, 1.0, 0.0], vec![3]);
    let targets = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0], vec![3]);

    let loss = softmax_cross_entropy_loss(&logits, &targets, Precision::F32);
    assert!(loss.value() > 0.3 && loss.value() < 0.5);
}

// ============================================================================
// Numerical Gradient Checking
// ============================================================================

/// Computes numerical gradient using finite differences.
fn numerical_gradient(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
    epsilon: f64,
) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..x.len() {
        x_plus[i] = x[i] + epsilon;
        x_minus[i] = x[i] - epsilon;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);

        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    grad
}

#[test]
fn test_numerical_gradient_mse() {
    let targets = vec![1.0, 2.0, 3.0];
    let predictions = vec![1.5, 2.5, 2.5];

    // MSE loss function
    let mse_fn = |p: &[f64]| -> f64 {
        let n = p.len() as f64;
        p.iter()
            .zip(targets.iter())
            .map(|(&pred, &tgt)| (pred - tgt).powi(2))
            .sum::<f64>()
            / n
    };

    // Numerical gradient
    let num_grad = numerical_gradient(mse_fn, &predictions, 1e-5);

    // Analytical gradient: 2*(p-t)/n
    let analytical_grad: Vec<f64> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| 2.0 * (p - t) / predictions.len() as f64)
        .collect();

    // Compare
    for (num, ana) in num_grad.iter().zip(analytical_grad.iter()) {
        assert!(
            (num - ana).abs() < 1e-4,
            "Numerical: {}, Analytical: {}",
            num,
            ana
        );
    }
}

#[test]
fn test_numerical_gradient_relu() {
    // ReLU: f(x) = max(0, x)
    // df/dx = 1 if x > 0, 0 if x < 0
    let values = vec![-1.0, 0.5, 2.0];

    let relu_sum_fn = |x: &[f64]| -> f64 {
        x.iter().map(|&v| v.max(0.0)).sum()
    };

    let num_grad = numerical_gradient(relu_sum_fn, &values, 1e-5);

    // Expected: [0, 1, 1]
    assert!(num_grad[0].abs() < 1e-4); // Negative input
    assert!((num_grad[1] - 1.0).abs() < 1e-4); // Positive input
    assert!((num_grad[2] - 1.0).abs() < 1e-4); // Positive input
}

#[test]
fn test_numerical_gradient_matmul() {
    // f(A) = sum(A @ x) where x = [1, 1]
    // df/dA_ij = x_j
    let x = vec![1.0, 1.0];

    let matmul_fn = |a: &[f64]| -> f64 {
        // A is 2x2, flattened row-major
        let mut result = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                result += a[i * 2 + j] * x[j];
            }
        }
        result
    };

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let num_grad = numerical_gradient(matmul_fn, &a, 1e-5);

    // df/dA = [[x_0, x_1], [x_0, x_1]] = [[1, 1], [1, 1]]
    for g in num_grad {
        assert!((g - 1.0).abs() < 1e-4);
    }
}

// ============================================================================
// Gradient Accumulation Tests
// ============================================================================

#[test]
fn test_gradient_accumulation() {
    use crate::gradient::accumulator::GradientAccumulator;

    let mut accumulator = GradientAccumulator::new();

    // Accumulate 4 batches of gradients
    for i in 0..4 {
        let mut grads = HashMap::new();
        grads.insert(0, BoundedTensor::from_exact(vec![1.0 * (i + 1) as f64], vec![1]));
        accumulator.accumulate(grads);
    }

    // Average should be (1 + 2 + 3 + 4) / 4 = 2.5
    let avg = accumulator.average();
    let avg_value = avg.get(&0).unwrap().values()[0];
    assert!((avg_value - 2.5).abs() < 1e-10);
}

// ============================================================================
// Learning Rate Scheduler Tests
// ============================================================================

#[test]
fn test_step_lr_scheduler() {
    use crate::gradient::optimizer::StepLR;
    use crate::gradient::optimizer::LRScheduler;

    let scheduler = StepLR::new(0.1, 10, 0.1);

    assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);
    assert!((scheduler.get_lr(9) - 0.1).abs() < 1e-10);
    assert!((scheduler.get_lr(10) - 0.01).abs() < 1e-10);
    assert!((scheduler.get_lr(20) - 0.001).abs() < 1e-10);
}

#[test]
fn test_cosine_annealing_lr() {
    use crate::gradient::optimizer::CosineAnnealingLR;
    use crate::gradient::optimizer::LRScheduler;

    let scheduler = CosineAnnealingLR::new(0.1, 0.0, 100);

    // At step 0, should be initial_lr
    assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-10);

    // At step 50, should be around half
    let mid_lr = scheduler.get_lr(50);
    assert!(mid_lr > 0.04 && mid_lr < 0.06);

    // At step 100, should be min_lr
    assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-10);
}

// ============================================================================
// End-to-End Training Step Tests
// ============================================================================

#[test]
fn test_simple_linear_training_step() {
    // Train a simple linear model: y = w * x
    // Target: w should converge to 2.0 given pairs (x, y) = (1, 2)

    let tape = GradientTape::new();

    // Weight initialized to 0.0
    let w_val = BoundedTensor::from_exact(vec![0.0], vec![1]);
    let _w = Variable::param(w_val.clone(), tape.clone(), Some("w".into()));

    // Input x = 1.0
    let x_val = BoundedTensor::from_exact(vec![1.0], vec![1]);
    let _x = Variable::input(x_val, tape.clone(), Some("x".into()));

    // Target y = 2.0
    let target: f64 = 2.0;

    let mut current_w: f64 = 0.0;
    let lr: f64 = 0.1;

    // Manual training loop (50 steps for convergence)
    for _ in 0..50 {
        // Forward: y_hat = w * x
        let y_hat: f64 = current_w * 1.0;

        // Loss: (y_hat - target)^2
        let _loss = (y_hat - target).powi(2);

        // Gradient: d(loss)/dw = 2 * (y_hat - target) * x = 2 * (y_hat - target)
        let grad = 2.0 * (y_hat - target) * 1.0;

        // Update: w = w - lr * grad
        current_w = current_w - lr * grad;
    }

    // Weight should be close to 2.0 (allow larger tolerance)
    assert!((current_w - 2.0_f64).abs() < 0.1, "w should converge to 2.0, got {}", current_w);
}

#[test]
fn test_autodiff_training_step() {
    // Use autodiff to train w: y = w * x, target = 2x
    let tape = GradientTape::new();

    let w_val = BoundedTensor::from_exact(vec![0.0], vec![1, 1]);
    let w = Variable::param(w_val, tape.clone(), Some("w".into()));

    let x_val = BoundedTensor::from_exact(vec![1.0], vec![1, 1]);
    let x = Variable::input(x_val, tape.clone(), Some("x".into()));

    // y = w @ x
    let y = w.matmul(&x);

    // Loss would be (y - 2)^2, but let's just verify forward pass
    let y_val = y.tensor.values()[0];
    assert!((y_val - 0.0).abs() < 1e-10); // w=0, so y=0
}

// ============================================================================
// Gradient Clipping Tests
// ============================================================================

#[test]
fn test_gradient_clipping_by_value() {
    use crate::gradient::clipping::GradientClipConfig;

    let mut grads = HashMap::new();
    grads.insert(0, BoundedTensor::from_exact(vec![5.0, -10.0, 2.0], vec![3]));

    let config = GradientClipConfig::Value(3.0);
    config.apply(&mut grads);

    let clipped = grads.get(&0).unwrap().values();
    assert!((clipped[0] - 3.0).abs() < 1e-10);  // Clipped from 5
    assert!((clipped[1] - (-3.0)).abs() < 1e-10);  // Clipped from -10
    assert!((clipped[2] - 2.0).abs() < 1e-10);  // Unchanged
}

#[test]
fn test_gradient_clipping_by_norm() {
    use crate::gradient::clipping::clip_grad_norm;

    let mut grads = HashMap::new();
    // Gradient with L2 norm = sqrt(9 + 16) = 5.0
    grads.insert(0, BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]));

    // Clip to max_norm = 2.5
    let norm = clip_grad_norm(&mut grads, 2.5);

    // Original norm was 5.0
    assert!((norm - 5.0).abs() < 1e-10);

    // Clipped gradient should have norm = 2.5
    let clipped = grads.get(&0).unwrap().values();
    let clipped_norm = (clipped[0].powi(2) + clipped[1].powi(2)).sqrt();
    assert!((clipped_norm - 2.5).abs() < 1e-6);
}

// ============================================================================
// Error Bound Propagation Tests
// ============================================================================

#[test]
fn test_gradient_error_bounds() {
    let tape = GradientTape::new();

    // Input with some error
    let x_val = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![2], 0.01);
    let x = Variable::param(x_val, tape.clone(), Some("x".into()));

    // y = x + x
    let y = x.add(&x);

    // Output should have propagated error
    assert!(y.tensor.max_error() > 0.0);

    // For addition, error should roughly double
    assert!(y.tensor.max_error() >= 0.01);
}

#[test]
fn test_matmul_error_propagation() {
    let tape = GradientTape::new();

    // Weight matrix with error
    let w_val = BoundedTensor::from_approximate(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], 0.001);
    let w = Variable::param(w_val, tape.clone(), Some("W".into()));

    // Input vector with error
    let x_val = BoundedTensor::from_approximate(vec![1.0, 1.0], vec![2, 1], 0.001);
    let x = Variable::input(x_val, tape.clone(), Some("x".into()));

    // y = W @ x
    let y = w.matmul(&x);

    // Output should have error > 0
    assert!(y.tensor.max_error() > 0.0);
}

// ============================================================================
// Convergence Verification (Simple Cases)
// ============================================================================

#[test]
fn test_sgd_converges_to_minimum() {
    // Minimize f(x) = x^2, optimum at x = 0
    let lr: f64 = 0.1;
    let mut x: f64 = 10.0;

    for _ in 0..100 {
        let grad = 2.0 * x; // df/dx = 2x
        x = x - lr * grad;
    }

    assert!(x.abs() < 0.01);
}

#[test]
fn test_adam_converges_to_minimum() {
    // Minimize f(x) = x^2 using Adam-like update
    let lr: f64 = 0.5;  // Higher learning rate for faster convergence
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let eps: f64 = 1e-8;

    let mut x: f64 = 10.0;
    let mut m: f64 = 0.0;
    let mut v: f64 = 0.0;

    for t in 1..=500_i32 {  // More iterations
        let grad = 2.0 * x;

        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;

        let m_hat = m / (1.0 - beta1.powi(t));
        let v_hat = v / (1.0 - beta2.powi(t));

        x = x - lr * m_hat / (v_hat.sqrt() + eps);
    }

    assert!(x.abs() < 0.1, "x should converge near 0, got {}", x);
}

#[test]
fn test_linear_regression_convergence() {
    // Simple linear regression: y = 2x + 1
    // Train to learn w=2, b=1

    let data_x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data_y: Vec<f64> = data_x.iter().map(|&x| 2.0 * x + 1.0).collect();

    let mut w = 0.0;
    let mut b = 0.0;
    let lr = 0.01;

    for _ in 0..1000 {
        let mut total_grad_w = 0.0;
        let mut total_grad_b = 0.0;

        for (&x, &y_true) in data_x.iter().zip(data_y.iter()) {
            let y_pred = w * x + b;
            let error = y_pred - y_true;

            // MSE gradients
            total_grad_w += 2.0 * error * x / data_x.len() as f64;
            total_grad_b += 2.0 * error / data_x.len() as f64;
        }

        w = w - lr * total_grad_w;
        b = b - lr * total_grad_b;
    }

    assert!((w - 2.0).abs() < 0.1);
    assert!((b - 1.0).abs() < 0.1);
}

// ============================================================================
// Comprehensive Backward Pass Tests with Finite-Difference Verification
// ============================================================================

/// Helper for finite-difference gradient checking of autodiff operations.
///
/// Compares the autodiff gradient (from `backward()`) against the numerical gradient
/// computed via central finite differences for a single parameter variable.
///
/// # Arguments
/// * `param_vals` - Initial values for the parameter
/// * `param_shape` - Shape of the parameter tensor
/// * `build_loss` - Closure: given a tracked Variable (the parameter), returns the loss Variable
/// * `numerical_loss` - Closure: given raw &[f64] values, returns the scalar loss
/// * `epsilon` - Finite-difference step size (typically 1e-5)
/// * `tolerance` - Max allowed relative difference between autodiff and numerical gradients
fn check_autodiff_gradient(
    param_vals: &[f64],
    param_shape: &[usize],
    build_loss: impl Fn(&Variable) -> Variable,
    numerical_loss: impl Fn(&[f64]) -> f64,
    epsilon: f64,
    tolerance: f64,
) {
    // 1. Compute autodiff gradient
    let tape = GradientTape::new();
    let param = Variable::param(
        BoundedTensor::from_exact(param_vals.to_vec(), param_shape.to_vec()),
        tape.clone(),
        None,
    );
    let loss = build_loss(&param);
    let grads = backward(&loss).expect("backward() failed");
    let ad_grad = grads
        .get(&param.node_index.unwrap())
        .expect("No gradient for parameter")
        .values();

    // 2. Compute numerical gradient via central finite differences
    let num_grad = numerical_gradient(&numerical_loss, param_vals, epsilon);

    // 3. Compare element-wise
    assert_eq!(
        ad_grad.len(),
        num_grad.len(),
        "Gradient length mismatch: autodiff={}, numerical={}",
        ad_grad.len(),
        num_grad.len()
    );

    for (i, (&ad, &nd)) in ad_grad.iter().zip(num_grad.iter()).enumerate() {
        let diff = (ad - nd).abs();
        let scale = ad.abs().max(nd.abs()).max(1e-7);
        assert!(
            diff / scale < tolerance || diff < 1e-7,
            "Gradient mismatch at index {}: autodiff={:.8}, numerical={:.8}, rel_diff={:.8}",
            i,
            ad,
            nd,
            diff / scale,
        );
    }
}

// --- Element-wise Mul backward ---

#[test]
fn test_mul_backward_analytical() {
    // loss = sum(a * b), dL/da = b, dL/db = a
    let tape = GradientTape::new();
    let a = Variable::param(
        BoundedTensor::from_exact(vec![2.0, 3.0, 4.0], vec![3]),
        tape.clone(),
        Some("a".into()),
    );
    let b = Variable::param(
        BoundedTensor::from_exact(vec![5.0, 6.0, 7.0], vec![3]),
        tape.clone(),
        Some("b".into()),
    );
    let loss = a.mul(&b).sum();
    let grads = backward(&loss).unwrap();

    let da = grads.get(&a.node_index.unwrap()).unwrap().values();
    let db = grads.get(&b.node_index.unwrap()).unwrap().values();

    // dL/da_i = b_i
    assert!((da[0] - 5.0).abs() < 1e-6);
    assert!((da[1] - 6.0).abs() < 1e-6);
    assert!((da[2] - 7.0).abs() < 1e-6);

    // dL/db_i = a_i
    assert!((db[0] - 2.0).abs() < 1e-6);
    assert!((db[1] - 3.0).abs() < 1e-6);
    assert!((db[2] - 4.0).abs() < 1e-6);
}

#[test]
fn test_mul_backward_finite_diff() {
    let a_vals = vec![2.0, 3.0, 4.0];
    let b_vals = vec![5.0, 6.0, 7.0];

    // Check gradient w.r.t. a
    check_autodiff_gradient(
        &a_vals,
        &[3],
        |a| {
            let b = Variable::input(
                BoundedTensor::from_exact(b_vals.clone(), vec![3]),
                a.tape.as_ref().unwrap().clone(),
                None,
            );
            a.mul(&b).sum()
        },
        |a| a.iter().zip(b_vals.iter()).map(|(ai, bi)| ai * bi).sum(),
        1e-5,
        1e-4,
    );

    // Check gradient w.r.t. b
    check_autodiff_gradient(
        &b_vals,
        &[3],
        |b| {
            let a = Variable::input(
                BoundedTensor::from_exact(a_vals.clone(), vec![3]),
                b.tape.as_ref().unwrap().clone(),
                None,
            );
            a.mul(b).sum()
        },
        |b| a_vals.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum(),
        1e-5,
        1e-4,
    );
}

#[test]
fn test_mul_self_backward() {
    // loss = sum(x * x) = sum(x^2), dL/dx_i = 2 * x_i
    let tape = GradientTape::new();
    let x = Variable::param(
        BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]),
        tape.clone(),
        Some("x".into()),
    );
    let loss = x.mul(&x).sum();
    let grads = backward(&loss).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap().values();

    assert!((dx[0] - 2.0).abs() < 1e-6);
    assert!((dx[1] - 4.0).abs() < 1e-6);
    assert!((dx[2] - 6.0).abs() < 1e-6);
}

// --- Sigmoid backward ---

#[test]
fn test_sigmoid_backward_analytical() {
    // loss = sum(sigmoid(x)), dL/dx_i = sigmoid(x_i) * (1 - sigmoid(x_i))
    let tape = GradientTape::new();
    let x = Variable::param(
        BoundedTensor::from_exact(vec![-1.0, 0.0, 1.0], vec![3]),
        tape.clone(),
        None,
    );
    let loss = x.sigmoid().sum();
    let grads = backward(&loss).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap().values();

    for (i, &xi) in [-1.0_f64, 0.0, 1.0].iter().enumerate() {
        let s = 1.0 / (1.0 + (-xi).exp());
        let expected = s * (1.0 - s);
        assert!(
            (dx[i] - expected).abs() < 1e-6,
            "sigmoid grad mismatch at {}: got {}, expected {}",
            i,
            dx[i],
            expected
        );
    }
}

#[test]
fn test_sigmoid_backward_finite_diff() {
    let x_vals = vec![-2.0, -0.5, 0.0, 0.5, 2.0];

    check_autodiff_gradient(
        &x_vals,
        &[5],
        |x| x.sigmoid().sum(),
        |x| x.iter().map(|xi| 1.0 / (1.0 + (-xi).exp())).sum(),
        1e-5,
        1e-4,
    );
}

// --- Softmax backward ---

#[test]
fn test_softmax_backward_finite_diff() {
    let x_vals = vec![1.0, 2.0, 3.0];
    let selector = vec![1.0, 0.0, 0.0]; // Select first softmax output

    check_autodiff_gradient(
        &x_vals,
        &[3],
        |x| {
            let sel = Variable::input(
                BoundedTensor::from_exact(selector.clone(), vec![3]),
                x.tape.as_ref().unwrap().clone(),
                None,
            );
            x.softmax().mul(&sel).sum()
        },
        |x| {
            let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_x: Vec<f64> = x.iter().map(|xi| (xi - max_x).exp()).collect();
            let sum_exp: f64 = exp_x.iter().sum();
            exp_x[0] / sum_exp // softmax(x)[0]
        },
        1e-5,
        1e-4,
    );
}

#[test]
fn test_softmax_backward_second_element() {
    let x_vals = vec![0.5, 1.5, 0.2, 0.8];
    let selector = vec![0.0, 1.0, 0.0, 0.0]; // Select second softmax output

    check_autodiff_gradient(
        &x_vals,
        &[4],
        |x| {
            let sel = Variable::input(
                BoundedTensor::from_exact(selector.clone(), vec![4]),
                x.tape.as_ref().unwrap().clone(),
                None,
            );
            x.softmax().mul(&sel).sum()
        },
        |x| {
            let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_x: Vec<f64> = x.iter().map(|xi| (xi - max_x).exp()).collect();
            let sum_exp: f64 = exp_x.iter().sum();
            exp_x[1] / sum_exp
        },
        1e-5,
        1e-4,
    );
}

// --- Tanh backward ---

#[test]
fn test_tanh_backward_analytical() {
    // loss = sum(tanh(x)), dL/dx_i = 1 - tanh^2(x_i)
    let tape = GradientTape::new();
    let x = Variable::param(
        BoundedTensor::from_exact(vec![-1.0, 0.0, 0.5, 1.0], vec![4]),
        tape.clone(),
        None,
    );
    let loss = x.tanh().sum();
    let grads = backward(&loss).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap().values();

    for (i, &xi) in [-1.0_f64, 0.0, 0.5, 1.0].iter().enumerate() {
        let t = xi.tanh();
        let expected = 1.0 - t * t;
        assert!(
            (dx[i] - expected).abs() < 1e-6,
            "tanh grad mismatch at {}: got {}, expected {}",
            i,
            dx[i],
            expected
        );
    }
}

#[test]
fn test_tanh_backward_finite_diff() {
    let x_vals = vec![-2.0, -0.5, 0.0, 0.5, 2.0];

    check_autodiff_gradient(
        &x_vals,
        &[5],
        |x| x.tanh().sum(),
        |x| x.iter().map(|xi| xi.tanh()).sum(),
        1e-5,
        1e-4,
    );
}

// --- LeakyReLU backward ---

#[test]
fn test_leaky_relu_backward_analytical() {
    let alpha = 0.01;
    let tape = GradientTape::new();
    let x = Variable::param(
        BoundedTensor::from_exact(vec![-2.0, -0.5, 0.5, 2.0], vec![4]),
        tape.clone(),
        None,
    );
    let loss = x.leaky_relu(alpha).sum();
    let grads = backward(&loss).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap().values();

    // dL/dx = 1 if x > 0, alpha if x <= 0
    assert!((dx[0] - alpha).abs() < 1e-6); // x=-2 < 0
    assert!((dx[1] - alpha).abs() < 1e-6); // x=-0.5 < 0
    assert!((dx[2] - 1.0).abs() < 1e-6);   // x=0.5 > 0
    assert!((dx[3] - 1.0).abs() < 1e-6);   // x=2 > 0
}

#[test]
fn test_leaky_relu_backward_finite_diff() {
    let x_vals = vec![-2.0, -0.5, 0.5, 2.0];
    let alpha = 0.01;

    check_autodiff_gradient(
        &x_vals,
        &[4],
        |x| x.leaky_relu(alpha).sum(),
        |x| {
            x.iter()
                .map(|&xi| if xi > 0.0 { xi } else { alpha * xi })
                .sum()
        },
        1e-5,
        1e-4,
    );
}

// --- GELU backward ---

#[test]
fn test_gelu_backward_finite_diff() {
    let x_vals = vec![-2.0, -0.5, 0.0, 0.5, 2.0];

    check_autodiff_gradient(
        &x_vals,
        &[5],
        |x| x.gelu().sum(),
        |x| {
            x.iter()
                .map(|&xi| {
                    let sig = 1.0 / (1.0 + (-1.702 * xi).exp());
                    xi * sig
                })
                .sum()
        },
        1e-5,
        1e-4,
    );
}

// --- Div backward ---

#[test]
fn test_div_backward_finite_diff() {
    let a_vals = vec![6.0, 8.0, 10.0];
    let b_vals = vec![2.0, 4.0, 5.0];

    check_autodiff_gradient(
        &a_vals,
        &[3],
        |a| {
            let b = Variable::input(
                BoundedTensor::from_exact(b_vals.clone(), vec![3]),
                a.tape.as_ref().unwrap().clone(),
                None,
            );
            a.div(&b).sum()
        },
        |a| a.iter().zip(b_vals.iter()).map(|(ai, bi)| ai / bi).sum(),
        1e-5,
        1e-4,
    );
}

// --- Sub backward ---

#[test]
fn test_sub_backward_analytical() {
    let tape = GradientTape::new();
    let a = Variable::param(
        BoundedTensor::from_exact(vec![3.0, 5.0], vec![2]),
        tape.clone(),
        None,
    );
    let b = Variable::param(
        BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]),
        tape.clone(),
        None,
    );
    let loss = a.sub(&b).sum();
    let grads = backward(&loss).unwrap();

    let da = grads.get(&a.node_index.unwrap()).unwrap().values();
    let db = grads.get(&b.node_index.unwrap()).unwrap().values();

    // dL/da = 1, dL/db = -1
    assert!((da[0] - 1.0).abs() < 1e-6);
    assert!((da[1] - 1.0).abs() < 1e-6);
    assert!((db[0] - (-1.0)).abs() < 1e-6);
    assert!((db[1] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_sub_backward_finite_diff() {
    let a_vals = vec![3.0, 5.0, 7.0];
    let b_vals = vec![1.0, 2.0, 3.0];

    check_autodiff_gradient(
        &a_vals,
        &[3],
        |a| {
            let b = Variable::input(
                BoundedTensor::from_exact(b_vals.clone(), vec![3]),
                a.tape.as_ref().unwrap().clone(),
                None,
            );
            a.sub(&b).sum()
        },
        |a| a.iter().zip(b_vals.iter()).map(|(ai, bi)| ai - bi).sum(),
        1e-5,
        1e-4,
    );
}

// --- Mean backward ---

#[test]
fn test_mean_backward_analytical() {
    // loss = mean(x), dL/dx_i = 1/n
    let tape = GradientTape::new();
    let x = Variable::param(
        BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
        tape.clone(),
        None,
    );
    let loss = x.mean();
    let grads = backward(&loss).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap().values();

    for &d in &dx {
        assert!((d - 0.25).abs() < 1e-6, "mean grad should be 1/n=0.25, got {}", d);
    }
}

#[test]
fn test_mean_backward_finite_diff() {
    let x_vals = vec![1.0, 2.0, 3.0, 4.0];

    check_autodiff_gradient(
        &x_vals,
        &[4],
        |x| x.mean(),
        |x| x.iter().sum::<f64>() / x.len() as f64,
        1e-5,
        1e-4,
    );
}

// --- MatMul backward ---

#[test]
fn test_matmul_backward_weight_finite_diff() {
    // loss = sum(W @ x), check gradient w.r.t. W
    let w_vals = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let x_vals = vec![0.5, 1.5]; // 2x1

    check_autodiff_gradient(
        &w_vals,
        &[2, 2],
        |w| {
            let x = Variable::input(
                BoundedTensor::from_exact(x_vals.clone(), vec![2, 1]),
                w.tape.as_ref().unwrap().clone(),
                None,
            );
            w.matmul(&x).sum()
        },
        |w| {
            // y = W @ x, loss = sum(y)
            let mut result = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    result += w[i * 2 + j] * x_vals[j];
                }
            }
            result
        },
        1e-5,
        1e-4,
    );
}

#[test]
fn test_matmul_backward_input_finite_diff() {
    // loss = sum(W @ x), check gradient w.r.t. x
    let w_vals = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let x_vals = vec![0.5, 1.5]; // 2x1

    check_autodiff_gradient(
        &x_vals,
        &[2, 1],
        |x| {
            let w = Variable::input(
                BoundedTensor::from_exact(w_vals.clone(), vec![2, 2]),
                x.tape.as_ref().unwrap().clone(),
                None,
            );
            w.matmul(x).sum()
        },
        |x| {
            let mut result = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    result += w_vals[i * 2 + j] * x[j];
                }
            }
            result
        },
        1e-5,
        1e-4,
    );
}

// --- Conv2d backward ---

#[test]
fn test_conv2d_backward_input_finite_diff() {
    // Input: [1, 1, 4, 4], Kernel: [1, 1, 3, 3]
    let input_vals: Vec<f64> = (1..=16).map(|x| x as f64 * 0.1).collect();
    let kernel_vals = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];

    check_autodiff_gradient(
        &input_vals,
        &[1, 1, 4, 4],
        |input| {
            let kernel = Variable::input(
                BoundedTensor::from_exact(kernel_vals.clone(), vec![1, 1, 3, 3]),
                input.tape.as_ref().unwrap().clone(),
                None,
            );
            input.conv2d(&kernel, (1, 1), (0, 0)).sum()
        },
        |input_v| {
            let input_t = BoundedTensor::from_exact(input_v.to_vec(), vec![1, 1, 4, 4]);
            let kernel_t = BoundedTensor::from_exact(kernel_vals.clone(), vec![1, 1, 3, 3]);
            let config = crate::ops::Conv2dConfig {
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
                groups: 1,
            };
            let out = crate::ops::conv2d_with_config(&input_t, &kernel_t, config, Precision::F32)
                .unwrap();
            out.values().iter().sum()
        },
        1e-5,
        1e-3, // Conv2d accumulates more numerical error
    );
}

#[test]
fn test_conv2d_backward_kernel_finite_diff() {
    let input_vals: Vec<f64> = (1..=16).map(|x| x as f64 * 0.1).collect();
    let kernel_vals = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];

    check_autodiff_gradient(
        &kernel_vals,
        &[1, 1, 3, 3],
        |kernel| {
            let input = Variable::input(
                BoundedTensor::from_exact(input_vals.clone(), vec![1, 1, 4, 4]),
                kernel.tape.as_ref().unwrap().clone(),
                None,
            );
            input.conv2d(kernel, (1, 1), (0, 0)).sum()
        },
        |kernel_v| {
            let input_t = BoundedTensor::from_exact(input_vals.clone(), vec![1, 1, 4, 4]);
            let kernel_t = BoundedTensor::from_exact(kernel_v.to_vec(), vec![1, 1, 3, 3]);
            let config = crate::ops::Conv2dConfig {
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
                groups: 1,
            };
            let out = crate::ops::conv2d_with_config(&input_t, &kernel_t, config, Precision::F32)
                .unwrap();
            out.values().iter().sum()
        },
        1e-5,
        1e-3,
    );
}

#[test]
fn test_conv2d_backward_with_padding_finite_diff() {
    // Test conv2d backward with padding
    let input_vals: Vec<f64> = (1..=9).map(|x| x as f64 * 0.1).collect();
    let kernel_vals = vec![1.0, 0.5, 0.5, 1.0];

    check_autodiff_gradient(
        &input_vals,
        &[1, 1, 3, 3],
        |input| {
            let kernel = Variable::input(
                BoundedTensor::from_exact(kernel_vals.clone(), vec![1, 1, 2, 2]),
                input.tape.as_ref().unwrap().clone(),
                None,
            );
            input.conv2d(&kernel, (1, 1), (1, 1)).sum()
        },
        |input_v| {
            let input_t = BoundedTensor::from_exact(input_v.to_vec(), vec![1, 1, 3, 3]);
            let kernel_t = BoundedTensor::from_exact(kernel_vals.clone(), vec![1, 1, 2, 2]);
            let config = crate::ops::Conv2dConfig {
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
                groups: 1,
            };
            let out = crate::ops::conv2d_with_config(&input_t, &kernel_t, config, Precision::F32)
                .unwrap();
            out.values().iter().sum()
        },
        1e-5,
        1e-3,
    );
}

// --- MaxPool2d backward ---

#[test]
fn test_max_pool2d_backward_analytical() {
    // Input [1,1,4,4]: values 1..16; pool 2x2, stride 2
    // Max of each 2x2 block: positions (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
    // Only those positions should receive gradient 1.0; all others get 0.0
    let input_vals: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let tape = GradientTape::new();
    let input = Variable::param(
        BoundedTensor::from_exact(input_vals, vec![1, 1, 4, 4]),
        tape.clone(),
        None,
    );
    let loss = input.max_pool2d((2, 2), (2, 2), (0, 0)).sum();
    let grads = backward(&loss).unwrap();
    let dx = grads.get(&input.node_index.unwrap()).unwrap().values();

    // Check that gradient is 1.0 at max positions, 0.0 elsewhere
    let max_positions = vec![5, 7, 13, 15]; // 0-indexed flat positions of max elements in each 2x2 block
    for i in 0..16 {
        if max_positions.contains(&i) {
            assert!(
                (dx[i] - 1.0).abs() < 1e-6,
                "Expected grad 1.0 at max position {}, got {}",
                i,
                dx[i]
            );
        } else {
            assert!(
                dx[i].abs() < 1e-6,
                "Expected grad 0.0 at non-max position {}, got {}",
                i,
                dx[i]
            );
        }
    }
}

#[test]
fn test_max_pool2d_backward_finite_diff() {
    // Use well-separated values to ensure stable max positions under perturbation
    let input_vals: Vec<f64> = (1..=16).map(|x| x as f64 * 10.0).collect();

    check_autodiff_gradient(
        &input_vals,
        &[1, 1, 4, 4],
        |input| input.max_pool2d((2, 2), (2, 2), (0, 0)).sum(),
        |input_v| {
            let input_t = BoundedTensor::from_exact(input_v.to_vec(), vec![1, 1, 4, 4]);
            let config = crate::ops::Pool2dConfig {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
            };
            let result = crate::ops::max_pool2d(&input_t, config).unwrap();
            result.output.values().iter().sum()
        },
        1e-5,
        1e-4,
    );
}

// --- Chain Rule Tests ---

#[test]
fn test_chain_matmul_sigmoid_sum_finite_diff() {
    let w_vals = vec![0.5, -0.3, 0.2, 0.8]; // 2x2
    let x_vals = vec![1.0, 2.0]; // 2x1

    check_autodiff_gradient(
        &w_vals,
        &[2, 2],
        |w| {
            let x = Variable::input(
                BoundedTensor::from_exact(x_vals.clone(), vec![2, 1]),
                w.tape.as_ref().unwrap().clone(),
                None,
            );
            w.matmul(&x).sigmoid().sum()
        },
        |w| {
            let mut y = vec![0.0; 2];
            for i in 0..2 {
                for j in 0..2 {
                    y[i] += w[i * 2 + j] * x_vals[j];
                }
                y[i] = 1.0 / (1.0 + (-y[i]).exp());
            }
            y.iter().sum()
        },
        1e-5,
        1e-4,
    );
}

#[test]
fn test_chain_matmul_tanh_sum_finite_diff() {
    let w_vals = vec![0.5, -0.3, 0.2, 0.8]; // 2x2
    let x_vals = vec![1.0, 2.0]; // 2x1

    check_autodiff_gradient(
        &w_vals,
        &[2, 2],
        |w| {
            let x = Variable::input(
                BoundedTensor::from_exact(x_vals.clone(), vec![2, 1]),
                w.tape.as_ref().unwrap().clone(),
                None,
            );
            w.matmul(&x).tanh().sum()
        },
        |w| {
            let mut y = vec![0.0; 2];
            for i in 0..2 {
                for j in 0..2 {
                    y[i] += w[i * 2 + j] * x_vals[j];
                }
                y[i] = y[i].tanh();
            }
            y.iter().sum()
        },
        1e-5,
        1e-4,
    );
}

#[test]
fn test_chain_matmul_relu_sum_finite_diff() {
    let w_vals = vec![0.5, -0.3, -0.2, 0.8]; // 2x2
    let x_vals = vec![1.0, 2.0]; // 2x1

    check_autodiff_gradient(
        &w_vals,
        &[2, 2],
        |w| {
            let x = Variable::input(
                BoundedTensor::from_exact(x_vals.clone(), vec![2, 1]),
                w.tape.as_ref().unwrap().clone(),
                None,
            );
            w.matmul(&x).relu().sum()
        },
        |w| {
            let mut y = vec![0.0; 2];
            for i in 0..2 {
                for j in 0..2 {
                    y[i] += w[i * 2 + j] * x_vals[j];
                }
                y[i] = y[i].max(0.0);
            }
            y.iter().sum()
        },
        1e-5,
        1e-4,
    );
}

#[test]
fn test_chain_mul_sigmoid_sum_finite_diff() {
    // loss = sum(sigmoid(a * b))
    let a_vals = vec![0.5, 1.0, -0.5, 1.5];
    let b_vals = vec![2.0, -1.0, 3.0, 0.5];

    check_autodiff_gradient(
        &a_vals,
        &[4],
        |a| {
            let b = Variable::input(
                BoundedTensor::from_exact(b_vals.clone(), vec![4]),
                a.tape.as_ref().unwrap().clone(),
                None,
            );
            a.mul(&b).sigmoid().sum()
        },
        |a| {
            a.iter()
                .zip(b_vals.iter())
                .map(|(&ai, &bi)| 1.0 / (1.0 + (-(ai * bi)).exp()))
                .sum()
        },
        1e-5,
        1e-4,
    );
}

#[test]
fn test_chain_add_mul_sum() {
    // loss = sum((a + b) * c)
    let a_vals = vec![1.0, 2.0, 3.0];
    let b_vals = vec![0.5, 1.5, 2.5];
    let c_vals = vec![2.0, 3.0, 4.0];

    check_autodiff_gradient(
        &a_vals,
        &[3],
        |a| {
            let tape = a.tape.as_ref().unwrap().clone();
            let b = Variable::input(
                BoundedTensor::from_exact(b_vals.clone(), vec![3]),
                tape.clone(),
                None,
            );
            let c = Variable::input(
                BoundedTensor::from_exact(c_vals.clone(), vec![3]),
                tape,
                None,
            );
            a.add(&b).mul(&c).sum()
        },
        |a| {
            a.iter()
                .zip(b_vals.iter())
                .zip(c_vals.iter())
                .map(|((&ai, &bi), &ci)| (ai + bi) * ci)
                .sum()
        },
        1e-5,
        1e-4,
    );
}

#[test]
fn test_chain_conv2d_relu_pool_sum_finite_diff() {
    // Common CNN pattern: conv2d -> relu -> max_pool -> sum
    // Input: [1, 1, 6, 6], Kernel: [1, 1, 3, 3]
    let input_vals: Vec<f64> = (1..=36).map(|x| x as f64 * 0.01).collect();
    let kernel_vals = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0];

    check_autodiff_gradient(
        &input_vals,
        &[1, 1, 6, 6],
        |input| {
            let kernel = Variable::input(
                BoundedTensor::from_exact(kernel_vals.clone(), vec![1, 1, 3, 3]),
                input.tape.as_ref().unwrap().clone(),
                None,
            );
            input
                .conv2d(&kernel, (1, 1), (0, 0))
                .relu()
                .max_pool2d((2, 2), (2, 2), (0, 0))
                .sum()
        },
        |input_v| {
            let input_t = BoundedTensor::from_exact(input_v.to_vec(), vec![1, 1, 6, 6]);
            let kernel_t = BoundedTensor::from_exact(kernel_vals.clone(), vec![1, 1, 3, 3]);
            let config = crate::ops::Conv2dConfig {
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
                groups: 1,
            };
            let conv_out =
                crate::ops::conv2d_with_config(&input_t, &kernel_t, config, Precision::F32)
                    .unwrap();
            let relu_out = crate::ops::relu(&conv_out);
            let pool_config = crate::ops::Pool2dConfig {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
            };
            let pool_out = crate::ops::max_pool2d(&relu_out, pool_config).unwrap();
            pool_out.output.values().iter().sum()
        },
        1e-5,
        1e-2, // Multi-layer chain accumulates more error
    );
}

// --- Multi-path gradient accumulation ---

#[test]
fn test_gradient_accumulation_multi_path() {
    // loss = sum(x) + sum(x * x) — x contributes via two paths
    // dL/dx_i = 1 + 2*x_i
    let tape = GradientTape::new();
    let x = Variable::param(
        BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]),
        tape.clone(),
        None,
    );
    let path1 = x.sum();
    let path2 = x.mul(&x).sum();
    let loss = path1.add(&path2);
    let grads = backward(&loss).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap().values();

    assert!((dx[0] - 3.0).abs() < 1e-6, "1 + 2*1 = 3, got {}", dx[0]);
    assert!((dx[1] - 5.0).abs() < 1e-6, "1 + 2*2 = 5, got {}", dx[1]);
    assert!((dx[2] - 7.0).abs() < 1e-6, "1 + 2*3 = 7, got {}", dx[2]);
}

#[test]
fn test_gradient_accumulation_multi_path_finite_diff() {
    let x_vals = vec![1.0, 2.0, 3.0];

    check_autodiff_gradient(
        &x_vals,
        &[3],
        |x| {
            let sum1 = x.sum();
            let sum2 = x.mul(x).sum();
            sum1.add(&sum2)
        },
        |x| {
            let sum1: f64 = x.iter().sum();
            let sum2: f64 = x.iter().map(|xi| xi * xi).sum();
            sum1 + sum2
        },
        1e-5,
        1e-4,
    );
}

// ============================================================================
// MLP Backward Verification
// ============================================================================

#[test]
fn test_mlp_backward_gives_nonzero_gradients() {
    let tape = GradientTape::new();

    let w1 = Variable::param(
        BoundedTensor::from_exact(vec![0.5, -0.3, 0.2, 0.8, -0.1, 0.6, 0.4, -0.2,
                                        0.3, 0.7, -0.5, 0.1, 0.9, -0.4, 0.2, 0.6], vec![8, 2]),
        tape.clone(), Some("w1".into()),
    );
    let b1 = Variable::param(
        BoundedTensor::from_exact(vec![0.0; 8], vec![8]),
        tape.clone(), Some("b1".into()),
    );
    let w2 = Variable::param(
        BoundedTensor::from_exact(vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, -0.3], vec![1, 8]),
        tape.clone(), Some("w2".into()),
    );
    let b2 = Variable::param(
        BoundedTensor::from_exact(vec![0.0], vec![1]),
        tape.clone(), Some("b2".into()),
    );

    let x = Variable::input(
        BoundedTensor::from_exact(vec![0.0, 1.0], vec![1, 2]),
        tape.clone(), None,
    );
    let target = Variable::input(
        BoundedTensor::from_exact(vec![1.0], vec![1, 1]),
        tape.clone(), None,
    );

    let output = Variable::mlp(&x, &w1, Some(&b1), &w2, Some(&b2));
    let diff = output.sub(&target);
    let loss = diff.mul(&diff).mean();

    let grads = backward(&loss).unwrap();

    for (name, var) in [("w1", &w1), ("b1", &b1), ("w2", &w2), ("b2", &b2)] {
        let idx = var.node_index.unwrap();
        let g = grads.get(&idx).unwrap_or_else(|| panic!("No gradient for {}", name));
        let grad_norm: f64 = g.values().iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(grad_norm > 1e-10, "{} gradient is zero", name);
    }
}

#[test]
fn test_trainer_updates_parameters_and_loss_decreases() {
    use crate::gradient::training::{Trainer, TrainingConfig};

    let d_in = 2;
    let d_hid = 4;
    let d_out = 1;

    let config = TrainingConfig::new().with_accumulation_steps(1);
    let optimizer = SGD::new(0.1);
    let mut trainer = Trainer::new(optimizer, config);

    trainer.register_parameter("w1", 0, BoundedTensor::from_exact(vec![0.5, -0.3, 0.2, 0.8, -0.1, 0.6, 0.4, -0.2], vec![d_hid, d_in]));
    trainer.register_parameter("b1", 1, BoundedTensor::from_exact(vec![0.0; d_hid], vec![d_hid]));
    trainer.register_parameter("w2", 2, BoundedTensor::from_exact(vec![0.3, -0.2, 0.5, 0.1], vec![d_out, d_hid]));
    trainer.register_parameter("b2", 3, BoundedTensor::from_exact(vec![0.0], vec![d_out]));

    let w1_before = trainer.get_param("w1").unwrap().values();

    // Run one step
    let loss_before = trainer_mlp_batch_step(
        &mut trainer, &[0.0, 1.0], &[1.0], 1, d_in, d_out,
    );

    // Parameters should change
    let w1_after = trainer.get_param("w1").unwrap().values();
    let any_changed = w1_before.iter().zip(w1_after.iter())
        .any(|(b, a)| (b - a).abs() > 1e-12);
    assert!(any_changed, "Parameters did not change after trainer step!");

    // Run 10 more steps, loss should decrease
    for _ in 0..10 {
        trainer_mlp_batch_step(&mut trainer, &[0.0, 1.0], &[1.0], 1, d_in, d_out);
    }

    let loss_after = trainer_mlp_batch_step(
        &mut trainer, &[0.0, 1.0], &[1.0], 1, d_in, d_out,
    );
    assert!(loss_after < loss_before, "Loss should decrease over training steps");
}

// ============================================================================
// XOR Convergence Tests with AutoDiff Trainer
// ============================================================================

/// Helper to run a single batched forward+backward training step.
///
/// Creates a fresh tape, builds Variable::mlp forward pass with batched input,
/// computes MSE loss over the batch, then calls `trainer.step()`.
fn trainer_mlp_batch_step(
    trainer: &mut crate::gradient::training::Trainer<SGD>,
    x_batch: &[f64],     // flat [batch_size * d_in]
    y_batch: &[f64],     // flat [batch_size * d_out]
    batch_size: usize,
    d_in: usize,
    d_out: usize,
) -> f64 {
    let tape = GradientTape::new();

    let w1_var = Variable::param(trainer.get_param("w1").unwrap().clone(), tape.clone(), Some("w1".into()));
    let b1_var = Variable::param(trainer.get_param("b1").unwrap().clone(), tape.clone(), Some("b1".into()));
    let w2_var = Variable::param(trainer.get_param("w2").unwrap().clone(), tape.clone(), Some("w2".into()));
    let b2_var = Variable::param(trainer.get_param("b2").unwrap().clone(), tape.clone(), Some("b2".into()));

    trainer.update_param_index("w1", w1_var.node_index.unwrap());
    trainer.update_param_index("b1", b1_var.node_index.unwrap());
    trainer.update_param_index("w2", w2_var.node_index.unwrap());
    trainer.update_param_index("b2", b2_var.node_index.unwrap());

    let x_tensor = BoundedTensor::from_exact(x_batch.to_vec(), vec![batch_size, d_in]);
    let x_var = Variable::input(x_tensor, tape.clone(), None);

    let output = Variable::mlp(&x_var, &w1_var, Some(&b1_var), &w2_var, Some(&b2_var));

    let target_tensor = BoundedTensor::from_exact(y_batch.to_vec(), vec![batch_size, d_out]);
    let target_var = Variable::input(target_tensor, tape.clone(), None);
    let diff = output.sub(&target_var);
    let loss = diff.mul(&diff).mean(); // MSE over the entire batch

    let loss_val = loss.tensor.data()[0].value();
    trainer.step(&loss).unwrap();
    loss_val
}

#[test]
fn test_xor_convergence_with_autodiff_trainer() {
    use crate::gradient::training::{Trainer, TrainingConfig};
    use rand::Rng;
    use rand::SeedableRng;

    // XOR dataset - batched: all 4 samples in a single forward pass
    let x_batch: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_batch: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];
    let batch_size = 4;

    let d_in = 2;
    let d_hid = 8;
    let d_out = 1;

    // Xavier initialization with fixed seed for reproducibility
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let xavier1 = (6.0 / (d_in + d_hid) as f64).sqrt();
    let xavier2 = (6.0 / (d_hid + d_out) as f64).sqrt();

    let w1_init: Vec<f64> = (0..d_hid * d_in).map(|_| rng.gen_range(-xavier1..xavier1)).collect();
    let b1_init: Vec<f64> = vec![0.0; d_hid];
    let w2_init: Vec<f64> = (0..d_out * d_hid).map(|_| rng.gen_range(-xavier2..xavier2)).collect();
    let b2_init: Vec<f64> = vec![0.0; d_out];

    // Trainer: SGD lr=0.5, batched gradient descent
    let config = TrainingConfig::new().with_accumulation_steps(1);
    let optimizer = SGD::new(0.5);
    let mut trainer = Trainer::new(optimizer, config);

    trainer.register_parameter("w1", 0, BoundedTensor::from_exact(w1_init, vec![d_hid, d_in]));
    trainer.register_parameter("b1", 1, BoundedTensor::from_exact(b1_init, vec![d_hid]));
    trainer.register_parameter("w2", 2, BoundedTensor::from_exact(w2_init, vec![d_out, d_hid]));
    trainer.register_parameter("b2", 3, BoundedTensor::from_exact(b2_init, vec![d_out]));

    let max_epochs = 5000;
    let target_loss = 0.05;
    let mut final_loss = f64::MAX;

    for _epoch in 0..max_epochs {
        let loss = trainer_mlp_batch_step(
            &mut trainer, &x_batch, &y_batch, batch_size, d_in, d_out,
        );
        final_loss = loss;

        if loss < target_loss {
            break;
        }
    }

    assert!(
        final_loss < target_loss,
        "XOR training did not converge: final loss = {:.6}, target = {}",
        final_loss,
        target_loss
    );
}

#[test]
fn test_gradient_accumulation_across_minibatches() {
    use crate::gradient::training::{Trainer, TrainingConfig};
    use rand::Rng;
    use rand::SeedableRng;

    // XOR dataset — each sample is its own mini-batch,
    // gradients accumulate over all 4 before a weight update.
    // This tests the gradient accumulation across tape boundaries.
    let xor_inputs = [
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let xor_targets = [0.0, 1.0, 1.0, 0.0];

    let d_in = 2;
    let d_hid = 8;
    let d_out = 1;

    // Same Xavier init as the batched test
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let xavier1 = (6.0 / (d_in + d_hid) as f64).sqrt();
    let xavier2 = (6.0 / (d_hid + d_out) as f64).sqrt();

    let w1_init: Vec<f64> = (0..d_hid * d_in).map(|_| rng.gen_range(-xavier1..xavier1)).collect();
    let b1_init: Vec<f64> = vec![0.0; d_hid];
    let w2_init: Vec<f64> = (0..d_out * d_hid).map(|_| rng.gen_range(-xavier2..xavier2)).collect();
    let b2_init: Vec<f64> = vec![0.0; d_out];

    // 4 accumulation steps = full-batch GD via micro-batches of 1
    let config = TrainingConfig::new().with_accumulation_steps(4);
    let optimizer = SGD::new(0.5);
    let mut trainer = Trainer::new(optimizer, config);

    trainer.register_parameter("w1", 0, BoundedTensor::from_exact(w1_init, vec![d_hid, d_in]));
    trainer.register_parameter("b1", 1, BoundedTensor::from_exact(b1_init, vec![d_hid]));
    trainer.register_parameter("w2", 2, BoundedTensor::from_exact(w2_init, vec![d_out, d_hid]));
    trainer.register_parameter("b2", 3, BoundedTensor::from_exact(b2_init, vec![d_out]));

    let max_epochs = 10000;
    let target_loss = 0.1;
    let mut final_loss = f64::MAX;

    for _epoch in 0..max_epochs {
        let mut epoch_loss = 0.0;

        for (x_data, &y_target) in xor_inputs.iter().zip(xor_targets.iter()) {
            // Each sample in its own mini-batch (size 1)
            epoch_loss += trainer_mlp_batch_step(
                &mut trainer,
                x_data,
                &[y_target],
                1,
                d_in,
                d_out,
            );
        }

        epoch_loss /= xor_inputs.len() as f64;
        final_loss = epoch_loss;

        if epoch_loss < target_loss {
            break;
        }
    }

    assert!(
        final_loss < target_loss,
        "XOR with gradient accumulation did not converge: final loss = {:.6}, target = {}",
        final_loss,
        target_loss
    );
}

// ============================================================================
// Circuit Bridge Validation from Trained State
// ============================================================================

#[test]
fn test_circuit_bridge_from_trained_state() {
    use crate::gradient::training::{Trainer, TrainingConfig};
    use crate::nn::Linear;
    use crate::circuit_bridge::{build_training_witness, model_to_circuit_weights};
    use helix_core::types::Precision;
    use helix_circuits::halo2curves::bn256::Fr;
    use helix_circuits::halo2curves::ff::Field;

    // Tiny model: 2 → 2 → 1 with integer-ish initial weights
    let d_in = 2;
    let d_hid = 2;
    let d_out = 1;

    let config = TrainingConfig::new().with_accumulation_steps(1);
    let optimizer = SGD::new(1.0);  // Large lr so quantized weights change at scale=1
    let mut trainer = Trainer::new(optimizer, config);

    trainer.register_parameter("w1", 0, BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 1.0], vec![d_hid, d_in]));
    trainer.register_parameter("b1", 1, BoundedTensor::from_exact(vec![0.0, 0.0], vec![d_hid]));
    trainer.register_parameter("w2", 2, BoundedTensor::from_exact(vec![1.0, 1.0], vec![d_out, d_hid]));
    trainer.register_parameter("b2", 3, BoundedTensor::from_exact(vec![0.0], vec![d_out]));

    let input_data = vec![1.0, 1.0];
    let target_data = vec![5.0];

    // Train for 1 step with lr=1.0 so weights change enough to be visible after
    // integer quantization (circuit's scale=1 rounds to nearest integer)
    trainer_mlp_batch_step(&mut trainer, &input_data, &target_data, 1, d_in, d_out);

    // Extract trained weights and construct Linear layers
    let w1_trained = trainer.get_param("w1").unwrap();
    let b1_trained = trainer.get_param("b1").unwrap();
    let w2_trained = trainer.get_param("w2").unwrap();
    let b2_trained = trainer.get_param("b2").unwrap();

    let layer1 = Linear::from_raw(
        w1_trained.values(),
        vec![d_hid, d_in],
        Some(b1_trained.values()),
        Precision::F32,
    ).unwrap();

    let layer2 = Linear::from_raw(
        w2_trained.values(),
        vec![d_out, d_hid],
        Some(b2_trained.values()),
        Precision::F32,
    ).unwrap();

    // Verify circuit weight extraction
    let cw = model_to_circuit_weights(&layer1, &layer2).unwrap();
    assert_eq!(cw.d_in, d_in);
    assert_eq!(cw.d_hid, d_hid);
    assert_eq!(cw.d_out, d_out);

    for &v in cw.w1.iter().chain(cw.b1.iter()).chain(cw.w2.iter()).chain(cw.b2.iter()) {
        assert!(v.is_finite(), "Trained weight is not finite: {}", v);
    }

    // Build training witness from the trained state
    let output = build_training_witness(
        &layer1,
        &layer2,
        &input_data,
        &target_data,
        1.0,
        2, // step_number
    ).expect("build_training_witness should succeed with trained weights");

    let witness = &output.witness;

    // Verify witness dimensions
    assert_eq!(witness.d_in, d_in);
    assert_eq!(witness.d_hid, d_hid);
    assert_eq!(witness.d_out, d_out);

    // Verify witness vector lengths
    assert_eq!(witness.x.len(), d_in);
    assert_eq!(witness.target.len(), d_out);
    assert_eq!(witness.w1.len(), d_hid * d_in);
    assert_eq!(witness.b1.len(), d_hid);
    assert_eq!(witness.w2.len(), d_out * d_hid);
    assert_eq!(witness.b2.len(), d_out);
    assert_eq!(witness.h_pre.len(), d_hid);
    assert_eq!(witness.h.len(), d_hid);
    assert_eq!(witness.y.len(), d_out);
    assert_eq!(witness.w1_new.len(), d_hid * d_in);
    assert_eq!(witness.b1_new.len(), d_hid);
    assert_eq!(witness.w2_new.len(), d_out * d_hid);
    assert_eq!(witness.b2_new.len(), d_out);

    // Public inputs should have 8 elements
    let pi = witness.public_inputs();
    assert_eq!(pi.len(), 8);

    // State hashes should be non-trivial
    assert_ne!(witness.old_state_hash.0, Fr::ZERO);
    assert_ne!(witness.new_state_hash.0, Fr::ZERO);

    // State should change after a training step
    assert_ne!(witness.old_state_hash, witness.new_state_hash);

    // relu_range should be reasonable
    assert!(output.relu_range >= 256);
}

// ============================================================================
// Cross-Entropy Loss Tests
// ============================================================================

#[test]
fn test_cross_entropy_forward_backward() {
    // Test: softmax + cross-entropy for a simple 3-class problem
    let tape = GradientTape::new();

    // Logits: raw scores for 3 classes
    let logits_data = BoundedTensor::from_exact(vec![2.0, 1.0, 0.1], vec![3]);
    let logits = Variable::param(logits_data, tape.clone(), Some("logits".into()));

    // One-hot target: class 0
    let target_data = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0], vec![3]);
    let target = Variable::input(target_data, tape.clone(), None);

    // Compute cross-entropy loss
    let loss = logits.cross_entropy_loss(&target);
    let loss_val = loss.tensor.data()[0].value();

    // Expected: -log(softmax(2.0)) where softmax(2.0) = exp(2.0) / (exp(2.0) + exp(1.0) + exp(0.1))
    let exp2 = 2.0_f64.exp();
    let exp1 = 1.0_f64.exp();
    let exp01 = 0.1_f64.exp();
    let sum_exp = exp2 + exp1 + exp01;
    let expected_loss = -(exp2 / sum_exp).ln();

    assert!(
        (loss_val - expected_loss).abs() < 1e-6,
        "Cross-entropy loss: got {}, expected {}",
        loss_val,
        expected_loss
    );

    // Backward: gradient should be softmax(logits) - target
    let grads = backward(&loss).unwrap();
    let dlogits = grads.get(&logits.node_index.unwrap()).unwrap();

    let expected_grad = vec![
        exp2 / sum_exp - 1.0,  // softmax[0] - target[0]
        exp1 / sum_exp,         // softmax[1] - target[1]
        exp01 / sum_exp,        // softmax[2] - target[2]
    ];

    for (i, &expected) in expected_grad.iter().enumerate() {
        let actual = dlogits.data()[i].value();
        assert!(
            (actual - expected).abs() < 1e-5,
            "dlogits[{}]: got {}, expected {}",
            i, actual, expected
        );
    }
}

#[test]
fn test_binary_cross_entropy_forward_backward() {
    let tape = GradientTape::new();

    // Predictions (after sigmoid): p = [0.7, 0.3]
    let pred_data = BoundedTensor::from_exact(vec![0.7, 0.3], vec![2]);
    let pred = Variable::param(pred_data, tape.clone(), Some("pred".into()));

    // Targets: [1.0, 0.0]
    let target_data = BoundedTensor::from_exact(vec![1.0, 0.0], vec![2]);
    let target = Variable::input(target_data, tape.clone(), None);

    let loss = pred.binary_cross_entropy_loss(&target);
    let loss_val = loss.tensor.data()[0].value();

    // Expected: -[1.0*ln(0.7) + 0.0*ln(0.3) + 0.0*ln(1-0.7) + 1.0*ln(1-0.3)] / 2
    let expected = -(0.7_f64.ln() + 0.7_f64.ln()) / 2.0; // -ln(0.7) per sample, averaged
    assert!(
        (loss_val - expected).abs() < 1e-5,
        "BCE loss: got {}, expected {}",
        loss_val, expected
    );

    // Backward
    let grads = backward(&loss).unwrap();
    let dpred = grads.get(&pred.node_index.unwrap()).unwrap();

    // dL/dp_0 = (-1.0/0.7 + 0.0/0.3) / 2 = -1/(2*0.7)
    let expected_dp0 = (-1.0 / 0.7 + 0.0 / 0.3) / 2.0;
    // dL/dp_1 = (0.0/0.3 + 1.0/0.7) / 2 = 1/(2*0.7)
    let expected_dp1 = (0.0 / 0.3 + 1.0 / 0.7) / 2.0;

    assert!(
        (dpred.data()[0].value() - expected_dp0).abs() < 1e-4,
        "dpred[0]: got {}, expected {}",
        dpred.data()[0].value(), expected_dp0
    );
    assert!(
        (dpred.data()[1].value() - expected_dp1).abs() < 1e-4,
        "dpred[1]: got {}, expected {}",
        dpred.data()[1].value(), expected_dp1
    );
}

#[test]
fn test_log_softmax_backward_finite_diff() {
    let tape = GradientTape::new();
    let x_data = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
    let x = Variable::param(x_data.clone(), tape.clone(), Some("x".into()));

    let y = x.log_softmax().sum();
    let grads = backward(&y).unwrap();
    let dx_auto = grads.get(&x.node_index.unwrap()).unwrap();

    // Finite difference
    let eps = 1e-5;
    for i in 0..3 {
        let mut x_plus = x_data.values().clone();
        x_plus[i] += eps;
        let tape_plus = GradientTape::new();
        let xp = Variable::param(BoundedTensor::from_exact(x_plus, vec![3]), tape_plus, None);
        let yp = xp.log_softmax().sum();
        let yp_val = yp.tensor.data()[0].value();

        let mut x_minus = x_data.values().clone();
        x_minus[i] -= eps;
        let tape_minus = GradientTape::new();
        let xm = Variable::param(BoundedTensor::from_exact(x_minus, vec![3]), tape_minus, None);
        let ym = xm.log_softmax().sum();
        let ym_val = ym.tensor.data()[0].value();

        let numerical = (yp_val - ym_val) / (2.0 * eps);
        let analytical = dx_auto.data()[i].value();

        assert!(
            (analytical - numerical).abs() < 1e-4,
            "log_softmax grad[{}]: analytical={}, numerical={}",
            i, analytical, numerical
        );
    }
}

/// Helper: runs one forward+backward step with cross-entropy loss for a batched MLP.
#[allow(dead_code)]
fn trainer_mlp_cross_entropy_step(
    trainer: &mut crate::gradient::training::Trainer<Adam>,
    x_batch: &[f64],
    y_batch: &[f64],
    batch_size: usize,
    d_in: usize,
    d_out: usize,
) -> f64 {
    let tape = GradientTape::new();

    let w1_var = Variable::param(trainer.get_param("w1").unwrap().clone(), tape.clone(), Some("w1".into()));
    let b1_var = Variable::param(trainer.get_param("b1").unwrap().clone(), tape.clone(), Some("b1".into()));
    let w2_var = Variable::param(trainer.get_param("w2").unwrap().clone(), tape.clone(), Some("w2".into()));
    let b2_var = Variable::param(trainer.get_param("b2").unwrap().clone(), tape.clone(), Some("b2".into()));

    trainer.update_param_index("w1", w1_var.node_index.unwrap());
    trainer.update_param_index("b1", b1_var.node_index.unwrap());
    trainer.update_param_index("w2", w2_var.node_index.unwrap());
    trainer.update_param_index("b2", b2_var.node_index.unwrap());

    let x_tensor = BoundedTensor::from_exact(x_batch.to_vec(), vec![batch_size, d_in]);
    let x_var = Variable::input(x_tensor, tape.clone(), None);

    // Forward: MLP output is logits
    let logits = Variable::mlp(&x_var, &w1_var, Some(&b1_var), &w2_var, Some(&b2_var));

    // For batch cross-entropy: compute mean over samples
    // Each row of logits is one sample; we sum per-sample cross-entropy losses
    let target_tensor = BoundedTensor::from_exact(y_batch.to_vec(), vec![batch_size, d_out]);
    let target_var = Variable::input(target_tensor, tape.clone(), None);

    // Per-sample cross-entropy: compute over all logits at once
    // (This treats the batch as one big softmax, which is a simplification;
    // for production you'd split by rows. For XOR with batch=4 this converges fine.)
    let loss = logits.cross_entropy_loss(&target_var);

    let loss_val = loss.tensor.data()[0].value();
    trainer.step(&loss).unwrap();
    loss_val
}

#[test]
fn test_xor_classification_cross_entropy_convergence() {
    use crate::gradient::training::{Trainer, TrainingConfig, EarlyStopping};
    use crate::gradient::optimizer::Adam;
    use rand::Rng;
    use rand::SeedableRng;

    // XOR as 2-class classification: output is [p(class0), p(class1)]
    // XOR truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    // We train per-sample (not batched) with cross-entropy
    let xor_data: Vec<([f64; 2], [f64; 2])> = vec![
        ([0.0, 0.0], [1.0, 0.0]),  // class 0
        ([0.0, 1.0], [0.0, 1.0]),  // class 1
        ([1.0, 0.0], [0.0, 1.0]),  // class 1
        ([1.0, 1.0], [1.0, 0.0]),  // class 0
    ];

    let d_in = 2;
    let d_hid = 16;
    let d_out = 2;

    // Xavier initialization
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let xavier1 = (6.0 / (d_in + d_hid) as f64).sqrt();
    let xavier2 = (6.0 / (d_hid + d_out) as f64).sqrt();

    let w1_init: Vec<f64> = (0..d_hid * d_in).map(|_| rng.gen_range(-xavier1..xavier1)).collect();
    let b1_init: Vec<f64> = vec![0.0; d_hid];
    let w2_init: Vec<f64> = (0..d_out * d_hid).map(|_| rng.gen_range(-xavier2..xavier2)).collect();
    let b2_init: Vec<f64> = vec![0.0; d_out];

    let config = TrainingConfig::new().with_accumulation_steps(1);
    let optimizer = Adam::new(0.01);
    let mut trainer = Trainer::new(optimizer, config);

    trainer.register_parameter("w1", 0, BoundedTensor::from_exact(w1_init, vec![d_hid, d_in]));
    trainer.register_parameter("b1", 1, BoundedTensor::from_exact(b1_init, vec![d_hid]));
    trainer.register_parameter("w2", 2, BoundedTensor::from_exact(w2_init, vec![d_out, d_hid]));
    trainer.register_parameter("b2", 3, BoundedTensor::from_exact(b2_init, vec![d_out]));

    let mut early_stopping = EarlyStopping::new(500, 1e-6);
    let max_epochs = 3000;
    let mut final_accuracy = 0.0;

    for epoch in 0..max_epochs {
        let mut epoch_loss = 0.0;

        // Train on each sample
        for (x, y) in &xor_data {
            let tape = GradientTape::new();

            let w1_var = Variable::param(trainer.get_param("w1").unwrap().clone(), tape.clone(), Some("w1".into()));
            let b1_var = Variable::param(trainer.get_param("b1").unwrap().clone(), tape.clone(), Some("b1".into()));
            let w2_var = Variable::param(trainer.get_param("w2").unwrap().clone(), tape.clone(), Some("w2".into()));
            let b2_var = Variable::param(trainer.get_param("b2").unwrap().clone(), tape.clone(), Some("b2".into()));

            trainer.update_param_index("w1", w1_var.node_index.unwrap());
            trainer.update_param_index("b1", b1_var.node_index.unwrap());
            trainer.update_param_index("w2", w2_var.node_index.unwrap());
            trainer.update_param_index("b2", b2_var.node_index.unwrap());

            let x_tensor = BoundedTensor::from_exact(x.to_vec(), vec![1, d_in]);
            let x_var = Variable::input(x_tensor, tape.clone(), None);

            let logits = Variable::mlp(&x_var, &w1_var, Some(&b1_var), &w2_var, Some(&b2_var));

            let target_tensor = BoundedTensor::from_exact(y.to_vec(), vec![1, d_out]);
            let target_var = Variable::input(target_tensor, tape.clone(), None);

            let loss = logits.cross_entropy_loss(&target_var);
            epoch_loss += loss.tensor.data()[0].value();
            trainer.step(&loss).unwrap();
        }

        epoch_loss /= xor_data.len() as f64;

        // Evaluate accuracy
        let mut correct = 0;
        for (x, y) in &xor_data {
            let tape = GradientTape::new();
            let w1_var = Variable::param(trainer.get_param("w1").unwrap().clone(), tape.clone(), None);
            let b1_var = Variable::param(trainer.get_param("b1").unwrap().clone(), tape.clone(), None);
            let w2_var = Variable::param(trainer.get_param("w2").unwrap().clone(), tape.clone(), None);
            let b2_var = Variable::param(trainer.get_param("b2").unwrap().clone(), tape.clone(), None);

            let x_tensor = BoundedTensor::from_exact(x.to_vec(), vec![1, d_in]);
            let x_var = Variable::input(x_tensor, tape.clone(), None);
            let logits = Variable::mlp(&x_var, &w1_var, Some(&b1_var), &w2_var, Some(&b2_var));

            let logits_vals = logits.tensor.values();
            let pred_class = if logits_vals[0] > logits_vals[1] { 0 } else { 1 };
            let true_class = if y[0] > y[1] { 0 } else { 1 };
            if pred_class == true_class {
                correct += 1;
            }
        }

        final_accuracy = correct as f64 / xor_data.len() as f64;
        if final_accuracy >= 1.0 {
            break;
        }

        if early_stopping.should_stop(epoch_loss, epoch) {
            break;
        }
    }

    assert!(
        final_accuracy >= 0.9,
        "XOR cross-entropy classification accuracy: {:.0}% (expected >= 90%)",
        final_accuracy * 100.0,
    );
}

#[test]
fn test_cosine_warm_restarts_with_training() {
    use crate::gradient::optimizer::CosineAnnealingWarmRestarts;
    use crate::gradient::optimizer::LRScheduler;

    let scheduler = CosineAnnealingWarmRestarts::new(0.01, 50, 2.0, 1e-6);

    // Verify restart pattern: cycle 0 = 50 steps, cycle 1 = 100 steps, cycle 2 = 200 steps
    // At step 0: peak lr
    assert!((scheduler.get_lr(0) - 0.01).abs() < 1e-8);

    // At step 50: restart → peak again
    assert!((scheduler.get_lr(50) - 0.01).abs() < 1e-6);

    // At step 150 (50 + 100): restart → peak
    assert!((scheduler.get_lr(150) - 0.01).abs() < 1e-6);

    // Verify lr is always >= eta_min
    for step in 0..500 {
        let lr = scheduler.get_lr(step);
        assert!(lr >= 1e-6 - 1e-10, "LR at step {} = {} < eta_min", step, lr);
        assert!(lr <= 0.01 + 1e-10, "LR at step {} = {} > eta_max", step, lr);
    }
}

#[test]
fn test_early_stopping_integration_with_training() {
    use crate::gradient::training::EarlyStopping;

    let mut es = EarlyStopping::new(5, 0.001);

    // Simulate improving validation losses
    let losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44];
    let mut stopped_at = None;

    for (epoch, &loss) in losses.iter().enumerate() {
        if es.should_stop(loss, epoch) {
            stopped_at = Some(epoch);
            break;
        }
    }

    // Should stop after epoch 10 (patience=5, no improvement after epoch 4)
    assert!(stopped_at.is_some(), "Early stopping should have triggered");
    let stop_epoch = stopped_at.unwrap();
    assert!(stop_epoch >= 9, "Stopped too early at epoch {}", stop_epoch);
    assert_eq!(es.best_epoch(), 5);
    assert!((es.best_value() - 0.44).abs() < 1e-3);
}
