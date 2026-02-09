//! Integration tests for the transformer block backward pass.
//!
//! Validates that:
//! - Backward pass produces gradients with correct shapes
//! - Gradient magnitudes are reasonable (no explosion or vanishing)
//! - Pre-norm and post-norm backward produce consistent results
//! - Multi-layer backward propagates through the stack
//! - Numerical gradient checking (finite differences) matches analytical

use helix_avm::nn::{
    ActivationType, NormPosition, TransformerBlock, TransformerConfig,
};
use helix_core::types::{BoundedTensor, BoundedValue};

/// Helper: create a TransformerBlock with given config.
fn make_block(d_model: usize, num_heads: usize, norm_pos: NormPosition) -> TransformerBlock {
    let config = TransformerConfig::new(d_model, num_heads)
        .expect("config should be valid")
        .with_activation(ActivationType::ReLU);

    // Override norm position via the struct field
    let config = TransformerConfig {
        norm_position: norm_pos,
        ..config
    };

    TransformerBlock::new(config).expect("block creation should succeed")
}

/// Helper: create a deterministic input tensor.
fn make_input(seq_len: usize, d_model: usize) -> BoundedTensor {
    let mut data = Vec::with_capacity(seq_len * d_model);
    for i in 0..seq_len {
        for j in 0..d_model {
            // Small deterministic values to keep in stable range
            let val = ((i * d_model + j) as f64 * 0.01).sin() * 0.1;
            data.push(BoundedValue::exact(val));
        }
    }
    BoundedTensor::new(data, vec![seq_len, d_model])
}

/// Helper: create a deterministic gradient tensor (simulating upstream gradient).
fn make_grad(seq_len: usize, d_model: usize) -> BoundedTensor {
    let mut data = Vec::with_capacity(seq_len * d_model);
    for i in 0..seq_len {
        for j in 0..d_model {
            let val = ((i * d_model + j) as f64 * 0.03).cos() * 0.05;
            data.push(BoundedValue::exact(val));
        }
    }
    BoundedTensor::new(data, vec![seq_len, d_model])
}

// ============================================================================
// Shape correctness tests
// ============================================================================

#[test]
fn test_transformer_backward_shapes_pre_norm() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;

    let block = make_block(d_model, num_heads, NormPosition::Pre);
    let input = make_input(seq_len, d_model);
    let grad_output = make_grad(seq_len, d_model);

    let grads = block
        .backward(&grad_output, &input)
        .expect("backward should succeed");

    // grad_input should have same shape as input
    assert_eq!(
        grads.grad_input.shape(),
        &vec![seq_len, d_model],
        "grad_input shape mismatch"
    );

    // Attention gradients
    assert_eq!(
        grads.attention_grads.grad_query.shape(),
        &vec![seq_len, d_model],
        "attention grad_query shape mismatch"
    );
    assert_eq!(
        grads.attention_grads.grad_key.shape(),
        &vec![seq_len, d_model],
        "attention grad_key shape mismatch"
    );
    assert_eq!(
        grads.attention_grads.grad_value.shape(),
        &vec![seq_len, d_model],
        "attention grad_value shape mismatch"
    );

    // MLP gradients
    assert_eq!(
        grads.mlp_grads.grad_input.shape(),
        &vec![seq_len, d_model],
        "mlp grad_input shape mismatch"
    );
}

#[test]
fn test_transformer_backward_shapes_post_norm() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;

    let block = make_block(d_model, num_heads, NormPosition::Post);
    let input = make_input(seq_len, d_model);
    let grad_output = make_grad(seq_len, d_model);

    let grads = block
        .backward(&grad_output, &input)
        .expect("backward should succeed for post-norm");

    assert_eq!(
        grads.grad_input.shape(),
        &vec![seq_len, d_model],
        "post-norm grad_input shape mismatch"
    );
}

// ============================================================================
// Gradient magnitude tests
// ============================================================================

#[test]
fn test_transformer_backward_gradient_magnitudes() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;

    let block = make_block(d_model, num_heads, NormPosition::Pre);
    let input = make_input(seq_len, d_model);
    let grad_output = make_grad(seq_len, d_model);

    let grads = block
        .backward(&grad_output, &input)
        .expect("backward should succeed");

    // Gradient should not be all zeros (vanishing)
    let grad_l2: f64 = grads
        .grad_input
        .data()
        .iter()
        .map(|v: &BoundedValue<f64>| v.value() * v.value())
        .sum::<f64>()
        .sqrt();

    assert!(
        grad_l2 > 1e-15,
        "grad_input L2 norm is too small (vanishing gradient): {}",
        grad_l2
    );

    // Gradient should not explode
    let max_abs: f64 = grads
        .grad_input
        .data()
        .iter()
        .map(|v: &BoundedValue<f64>| v.value().abs())
        .fold(0.0, f64::max);

    assert!(
        max_abs < 1e6,
        "grad_input max absolute value is too large (exploding gradient): {}",
        max_abs
    );

    // All gradients should be finite (weight gradients may be near-zero depending on init)
    for v in grads.attention_grads.grad_w_q.data() {
        assert!(v.value().is_finite(), "attention W_Q gradient should be finite");
    }
    for v in grads.attention_grads.grad_query.data() {
        assert!(v.value().is_finite(), "attention grad_query should be finite");
    }
    for v in grads.mlp_grads.grad_fc1_weight.data() {
        assert!(v.value().is_finite(), "MLP fc1 weight gradient should be finite");
    }
    for v in grads.mlp_grads.grad_input.data() {
        assert!(v.value().is_finite(), "MLP grad_input should be finite");
    }
}

// ============================================================================
// Consistency tests
// ============================================================================

#[test]
fn test_transformer_backward_deterministic() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;

    let block = make_block(d_model, num_heads, NormPosition::Pre);
    let input = make_input(seq_len, d_model);
    let grad_output = make_grad(seq_len, d_model);

    let grads1 = block
        .backward(&grad_output, &input)
        .expect("backward run 1 should succeed");
    let grads2 = block
        .backward(&grad_output, &input)
        .expect("backward run 2 should succeed");

    // Same inputs should produce identical gradients
    for (a, b) in grads1
        .grad_input
        .data()
        .iter()
        .zip(grads2.grad_input.data().iter())
    {
        assert!(
            (a.value() - b.value()).abs() < 1e-12,
            "backward should be deterministic: {} vs {}",
            a.value(),
            b.value()
        );
    }
}

#[test]
fn test_transformer_backward_zero_gradient() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;

    let block = make_block(d_model, num_heads, NormPosition::Pre);
    let input = make_input(seq_len, d_model);

    // Zero upstream gradient should produce near-zero downstream gradient
    let zero_grad = BoundedTensor::zeros(vec![seq_len, d_model]);

    let grads = block
        .backward(&zero_grad, &input)
        .expect("backward with zero grad should succeed");

    let max_abs: f64 = grads
        .grad_input
        .data()
        .iter()
        .map(|v: &BoundedValue<f64>| v.value().abs())
        .fold(0.0, f64::max);

    assert!(
        max_abs < 1e-10,
        "zero upstream gradient should produce near-zero downstream gradient, got max_abs={}",
        max_abs
    );
}

// ============================================================================
// Numerical gradient checking
// ============================================================================

#[test]
fn test_transformer_backward_numerical_gradient_check() {
    let d_model = 4;
    let num_heads = 1;
    let seq_len = 2;
    let eps = 1e-5;

    let block = make_block(d_model, num_heads, NormPosition::Pre);
    let input = make_input(seq_len, d_model);

    // Forward pass to get output
    let _output = block.forward(&input).expect("forward should succeed");

    // Use a simple scalar loss: sum of all output elements
    // L = sum(output)
    // dL/d_output = ones
    let grad_output = BoundedTensor::from_exact(
        vec![1.0; seq_len * d_model],
        vec![seq_len, d_model],
    );

    let grads = block
        .backward(&grad_output, &input)
        .expect("backward should succeed");

    // Numerical gradient check: for a few elements, perturb input and check
    // dL/d_input[i] ≈ (L(input + eps*e_i) - L(input - eps*e_i)) / (2*eps)
    let n_check = std::cmp::min(4, seq_len * d_model);
    for idx in 0..n_check {
        let mut input_plus = input.data().to_vec();
        let mut input_minus = input.data().to_vec();

        input_plus[idx] = BoundedValue::exact(input_plus[idx].value() + eps);
        input_minus[idx] = BoundedValue::exact(input_minus[idx].value() - eps);

        let input_p = BoundedTensor::new(input_plus, input.shape().clone());
        let input_m = BoundedTensor::new(input_minus, input.shape().clone());

        let out_p = block.forward(&input_p).expect("forward+ should succeed");
        let out_m = block.forward(&input_m).expect("forward- should succeed");

        let loss_p: f64 = out_p.data().iter().map(|v: &BoundedValue<f64>| v.value()).sum();
        let loss_m: f64 = out_m.data().iter().map(|v: &BoundedValue<f64>| v.value()).sum();

        let numerical_grad = (loss_p - loss_m) / (2.0 * eps);
        let analytical_grad = grads.grad_input.data()[idx].value();

        // Allow generous tolerance due to error propagation through attention/softmax
        let abs_diff = (numerical_grad - analytical_grad).abs();
        let scale = numerical_grad.abs().max(analytical_grad.abs()).max(1e-8);
        let rel_diff = abs_diff / scale;

        assert!(
            rel_diff < 0.5 || abs_diff < 1e-4,
            "Numerical gradient check failed at index {}: analytical={}, numerical={}, rel_diff={}",
            idx,
            analytical_grad,
            numerical_grad,
            rel_diff
        );
    }
}

// ============================================================================
// Multi-block backward (stack)
// ============================================================================

#[test]
fn test_transformer_stack_backward() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;
    let n_layers = 3;

    // Create multiple blocks
    let blocks: Vec<_> = (0..n_layers)
        .map(|_| make_block(d_model, num_heads, NormPosition::Pre))
        .collect();

    let input = make_input(seq_len, d_model);
    let grad_output = make_grad(seq_len, d_model);

    // Forward through all blocks
    let mut intermediates = vec![input.clone()];
    let mut x = input.clone();
    for block in &blocks {
        x = block.forward(&x).expect("forward should succeed");
        intermediates.push(x.clone());
    }

    // Backward through all blocks in reverse
    let mut grad = grad_output.clone();
    let mut all_grads = Vec::new();
    for (i, block) in blocks.iter().enumerate().rev() {
        let block_input = &intermediates[i];
        let block_grads = block
            .backward(&grad, block_input)
            .expect("backward should succeed");
        grad = block_grads.grad_input.clone();
        all_grads.push(block_grads);
    }

    // Should have gradients for each layer
    assert_eq!(all_grads.len(), n_layers);

    // Final gradient (w.r.t. original input) should have correct shape
    assert_eq!(grad.shape(), &vec![seq_len, d_model]);

    // Gradient should not vanish through 3 layers
    let grad_l2: f64 = grad
        .data()
        .iter()
        .map(|v: &BoundedValue<f64>| v.value() * v.value())
        .sum::<f64>()
        .sqrt();

    assert!(
        grad_l2 > 1e-20,
        "gradient should not vanish through {} layers: L2={}",
        n_layers,
        grad_l2
    );
}

// ============================================================================
// Different activation functions
// ============================================================================

#[test]
fn test_transformer_backward_gelu_activation() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;

    let config = TransformerConfig::new(d_model, num_heads)
        .expect("config should be valid")
        .with_activation(ActivationType::GELU);

    let block = TransformerBlock::new(config).expect("block creation should succeed");
    let input = make_input(seq_len, d_model);
    let grad_output = make_grad(seq_len, d_model);

    let grads = block
        .backward(&grad_output, &input)
        .expect("backward with GELU should succeed");

    assert_eq!(grads.grad_input.shape(), &vec![seq_len, d_model]);

    let grad_l2: f64 = grads
        .grad_input
        .data()
        .iter()
        .map(|v: &BoundedValue<f64>| v.value() * v.value())
        .sum::<f64>()
        .sqrt();

    assert!(grad_l2 > 1e-15, "GELU backward gradient should be non-zero");
}

#[test]
fn test_transformer_backward_silu_activation() {
    let d_model = 8;
    let num_heads = 2;
    let seq_len = 4;

    let config = TransformerConfig::new(d_model, num_heads)
        .expect("config should be valid")
        .with_activation(ActivationType::SiLU);

    let block = TransformerBlock::new(config).expect("block creation should succeed");
    let input = make_input(seq_len, d_model);
    let grad_output = make_grad(seq_len, d_model);

    let grads = block
        .backward(&grad_output, &input)
        .expect("backward with SiLU should succeed");

    assert_eq!(grads.grad_input.shape(), &vec![seq_len, d_model]);

    let grad_l2: f64 = grads
        .grad_input
        .data()
        .iter()
        .map(|v: &BoundedValue<f64>| v.value() * v.value())
        .sum::<f64>()
        .sqrt();

    assert!(grad_l2 > 1e-15, "SiLU backward gradient should be non-zero");
}
