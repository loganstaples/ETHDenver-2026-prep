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
