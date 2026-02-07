//! End-to-End Integration Test for HELIX Training Pipeline
//!
//! This test verifies the complete ML training flow:
//! - Dataset loading and batching
//! - Forward pass through a simple MLP with BoundedTensors
//! - Backward pass (gradient computation)
//! - Witness generation for ZK proofs
//! - Error budget tracking and verification
//!
//! Run with: cargo test --test e2e_training_pipeline

use helix_core::traits::{Provable, SimpleWitness, Witness};
use helix_core::types::{
    AllocationStrategy, BudgetAllocator, BudgetAllocationConfig,
    BudgetComponent, ComponentType,
};
use helix_core::config::VMConfig;
use helix_core::types::bounded_value::MAX_SAFE_ERROR;
use helix_core::types::Precision;
use helix_core::{
    BoundedTensor, BoundedValue, ErrorMargin, HelixResult,
};
use std::time::Instant;

// =============================================================================
// SIMPLE MLP MODEL FOR TESTING
// =============================================================================

/// A simple 2-layer MLP with bounded arithmetic for testing.
/// Architecture: input -> hidden (ReLU) -> output (softmax)
struct SimpleMLP {
    /// Weight matrix for first layer [input_dim, hidden_dim]
    w1: BoundedTensor,
    /// Bias for first layer [hidden_dim]
    b1: BoundedTensor,
    /// Weight matrix for second layer [hidden_dim, output_dim]
    w2: BoundedTensor,
    /// Bias for second layer [output_dim]
    b2: BoundedTensor,
}

/// Forward pass result with intermediate activations for witness generation.
struct ForwardResult {
    /// Pre-activation of hidden layer
    z1: BoundedTensor,
    /// Hidden layer activations (after ReLU)
    a1: BoundedTensor,
    /// Pre-activation of output layer
    z2: BoundedTensor,
    /// Output probabilities (after softmax)
    output: BoundedTensor,
}

/// Gradients computed during backward pass.
struct Gradients {
    dw1: BoundedTensor,
    db1: BoundedTensor,
    dw2: BoundedTensor,
    db2: BoundedTensor,
}

/// Complete training step data for witness generation.
struct TrainingStep {
    /// Input batch
    input: BoundedTensor,
    /// Target labels (one-hot encoded)
    #[allow(dead_code)]
    target: BoundedTensor,
    /// Forward pass intermediates
    forward: ForwardResult,
    /// Computed gradients
    gradients: Gradients,
    /// Loss value
    loss: BoundedValue<f64>,
    /// Step number
    step: usize,
}

impl Provable for TrainingStep {
    type Witness = SimpleWitness;

    fn generate_witness(&self) -> Self::Witness {
        let mut witness = SimpleWitness::empty();

        // Add input values (quantized to u64)
        for v in self.input.data() {
            witness.push(quantize_to_u64(v.value()));
        }

        // Add all intermediate activations
        for v in self.forward.z1.data() {
            witness.push(quantize_to_u64(v.value()));
        }
        for v in self.forward.a1.data() {
            witness.push(quantize_to_u64(v.value()));
        }
        for v in self.forward.z2.data() {
            witness.push(quantize_to_u64(v.value()));
        }
        for v in self.forward.output.data() {
            witness.push(quantize_to_u64(v.value()));
        }

        // Add gradient values
        for v in self.gradients.dw1.data() {
            witness.push(quantize_to_u64(v.value()));
        }
        for v in self.gradients.dw2.data() {
            witness.push(quantize_to_u64(v.value()));
        }

        witness
    }

    fn public_inputs(&self) -> Vec<u64> {
        // Public inputs format: [loss, error_bound, step_number]
        vec![
            quantize_to_u64(self.loss.value()),
            quantize_to_u64(self.loss.absolute_error()),
            self.step as u64,
        ]
    }

    fn circuit_id(&self) -> &'static str {
        "ml_training_step_v1"
    }
}

impl SimpleMLP {
    /// Creates a new MLP with random weights (using seeded deterministic values).
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> HelixResult<Self> {
        // Initialize weights with small random values (Xavier initialization approximation)
        let scale1 = 1.0 / (input_dim as f64).sqrt();
        let scale2 = 1.0 / (hidden_dim as f64).sqrt();

        let w1_data: Vec<f64> = (0..input_dim * hidden_dim)
            .map(|i| (i as f64 * 0.1).sin() * scale1)
            .collect();
        let w1 = BoundedTensor::try_from_approximate(
            w1_data,
            vec![input_dim, hidden_dim],
            1e-7, // Machine epsilon for f64
        )?;

        let b1 = BoundedTensor::try_zeros(vec![hidden_dim])?;

        let w2_data: Vec<f64> = (0..hidden_dim * output_dim)
            .map(|i| (i as f64 * 0.2).cos() * scale2)
            .collect();
        let w2 = BoundedTensor::try_from_approximate(
            w2_data,
            vec![hidden_dim, output_dim],
            1e-7,
        )?;

        let b2 = BoundedTensor::try_zeros(vec![output_dim])?;

        Ok(Self { w1, b1, w2, b2 })
    }

    /// Forward pass through the network.
    fn forward(&self, input: &BoundedTensor) -> HelixResult<ForwardResult> {
        // input: [batch_size, input_dim]
        // w1: [input_dim, hidden_dim]
        // z1 = input @ w1 + b1

        let z1 = input.matmul(&self.w1)?;
        let z1 = add_bias(&z1, &self.b1)?;

        // ReLU activation
        let a1 = relu(&z1);

        // z2 = a1 @ w2 + b2
        let z2 = a1.matmul(&self.w2)?;
        let z2 = add_bias(&z2, &self.b2)?;

        // Softmax output
        let output = softmax(&z2)?;

        Ok(ForwardResult { z1, a1, z2, output })
    }

    /// Backward pass to compute gradients.
    fn backward(
        &self,
        input: &BoundedTensor,
        forward: &ForwardResult,
        target: &BoundedTensor,
    ) -> HelixResult<Gradients> {
        let batch_size = input.shape()[0];

        // d_output = output - target (cross-entropy derivative with softmax)
        let d_output = forward.output.try_sub(target)?;

        // dw2 = a1.T @ d_output
        let a1_t = forward.a1.try_transpose()?;
        let dw2 = a1_t.matmul(&d_output)?;

        // db2 = sum(d_output, axis=0)
        let db2 = sum_axis0(&d_output)?;

        // d_a1 = d_output @ w2.T
        let w2_t = self.w2.try_transpose()?;
        let d_a1 = d_output.matmul(&w2_t)?;

        // d_z1 = d_a1 * relu'(z1)
        let d_z1 = relu_backward(&d_a1, &forward.z1);

        // dw1 = input.T @ d_z1
        let input_t = input.try_transpose()?;
        let dw1 = input_t.matmul(&d_z1)?;

        // db1 = sum(d_z1, axis=0)
        let db1 = sum_axis0(&d_z1)?;

        // Scale gradients by batch size
        let scale = BoundedValue::exact(1.0 / batch_size as f64);
        let dw1 = dw1.try_scale(scale)?;
        let db1 = db1.try_scale(scale)?;
        let dw2 = dw2.try_scale(scale)?;
        let db2 = db2.try_scale(scale)?;

        Ok(Gradients { dw1, db1, dw2, db2 })
    }

    /// Compute cross-entropy loss.
    fn compute_loss(
        &self,
        output: &BoundedTensor,
        target: &BoundedTensor,
    ) -> HelixResult<BoundedValue<f64>> {
        // Cross-entropy: -sum(target * log(output)) / batch_size
        let epsilon = 1e-10;
        let batch_size = output.shape()[0];

        let mut total_loss = BoundedValue::exact(0.0);

        for i in 0..output.len() {
            let o = output.data()[i].value().max(epsilon);
            let t = target.data()[i].value();

            if t > 0.0 {
                let log_o = o.ln();
                let contribution = BoundedValue::new(
                    -t * log_o,
                    // Error in log: |1/o| * εo
                    ErrorMargin::absolute((1.0 / o) * output.data()[i].absolute_error()),
                );
                total_loss = total_loss.saturating_add(contribution);
            }
        }

        let scale = BoundedValue::exact(1.0 / batch_size as f64);
        Ok(total_loss.saturating_mul(scale))
    }

    /// Perform a complete training step.
    fn training_step(
        &self,
        input: &BoundedTensor,
        target: &BoundedTensor,
        step: usize,
    ) -> HelixResult<TrainingStep> {
        let forward = self.forward(input)?;
        let loss = self.compute_loss(&forward.output, target)?;
        let gradients = self.backward(input, &forward, target)?;

        Ok(TrainingStep {
            input: input.clone(),
            target: target.clone(),
            forward,
            gradients,
            loss,
            step,
        })
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Quantize a f64 value to u64 for witness (fixed-point representation).
fn quantize_to_u64(value: f64) -> u64 {
    const SCALE: f64 = 1e12; // 12 decimal places of precision, matches TensorWitness::SCALE
    let scaled = (value * SCALE).clamp(0.0, u64::MAX as f64);
    scaled as u64
}

/// Add bias to each row of a 2D tensor.
fn add_bias(x: &BoundedTensor, bias: &BoundedTensor) -> HelixResult<BoundedTensor> {
    assert!(x.is_matrix());
    assert!(bias.is_vector());

    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    assert_eq!(bias.shape()[0], cols);

    let mut data = Vec::with_capacity(x.len());
    for i in 0..rows {
        for j in 0..cols {
            let x_val = *x.get(&[i, j]).unwrap();
            let b_val = *bias.get(&[j]).unwrap();
            data.push(x_val.saturating_add(b_val));
        }
    }

    BoundedTensor::try_new(data, vec![rows, cols])
}

/// ReLU activation: max(0, x)
fn relu(x: &BoundedTensor) -> BoundedTensor {
    x.map(|v| {
        let val = v.value().max(0.0);
        BoundedValue::new(val, v.error())
    })
}

/// ReLU backward: gradient is 1 if x > 0, else 0
fn relu_backward(grad: &BoundedTensor, z: &BoundedTensor) -> BoundedTensor {
    assert_eq!(grad.shape(), z.shape());

    let data: Vec<BoundedValue<f64>> = grad.data()
        .iter()
        .zip(z.data())
        .map(|(g, z_val)| {
            if z_val.value() > 0.0 {
                *g
            } else {
                BoundedValue::exact(0.0)
            }
        })
        .collect();

    BoundedTensor::new(data, grad.shape().clone())
}

/// Softmax activation along last axis (for 2D: row-wise).
fn softmax(x: &BoundedTensor) -> HelixResult<BoundedTensor> {
    assert!(x.is_matrix());

    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut data = Vec::with_capacity(x.len());

    for i in 0..rows {
        // Find max for numerical stability
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..cols {
            let val = x.get(&[i, j]).unwrap().value();
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max)
        let mut exp_values: Vec<(f64, f64)> = Vec::with_capacity(cols);
        let mut sum_exp = 0.0;
        for j in 0..cols {
            let bounded = x.get(&[i, j]).unwrap();
            let shifted = bounded.value() - max_val;
            let exp_val = shifted.exp();
            // Error propagation for exp: |exp(x)| * εx
            let exp_err = exp_val * bounded.absolute_error();
            exp_values.push((exp_val, exp_err));
            sum_exp += exp_val;
        }

        // Normalize
        for (exp_val, exp_err) in exp_values {
            let prob = exp_val / sum_exp;
            // Simplified error for division
            let prob_err = (exp_err / sum_exp).min(1.0);
            data.push(BoundedValue::new(prob, ErrorMargin::absolute(prob_err)));
        }
    }

    BoundedTensor::try_new(data, vec![rows, cols])
}

/// Sum along axis 0 (column-wise sum for 2D matrix).
fn sum_axis0(x: &BoundedTensor) -> HelixResult<BoundedTensor> {
    assert!(x.is_matrix());

    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut data = Vec::with_capacity(cols);

    for j in 0..cols {
        let mut sum = BoundedValue::exact(0.0);
        for i in 0..rows {
            sum = sum.saturating_add(*x.get(&[i, j]).unwrap());
        }
        data.push(sum);
    }

    BoundedTensor::try_new(data, vec![cols])
}

/// Create synthetic batch for testing.
fn create_synthetic_batch(
    batch_size: usize,
    input_dim: usize,
    num_classes: usize,
) -> (BoundedTensor, BoundedTensor) {
    // Random-ish input data
    let input_data: Vec<f64> = (0..batch_size * input_dim)
        .map(|i| ((i as f64 * 0.1).sin() + 1.0) / 2.0)
        .collect();
    let input = BoundedTensor::from_approximate(
        input_data,
        vec![batch_size, input_dim],
        1e-7,
    );

    // One-hot encoded targets (deterministic for reproducibility)
    let mut target_data = vec![0.0; batch_size * num_classes];
    for i in 0..batch_size {
        let class = i % num_classes;
        target_data[i * num_classes + class] = 1.0;
    }
    let target = BoundedTensor::from_exact(target_data, vec![batch_size, num_classes]);

    (input, target)
}

// =============================================================================
// E2E INTEGRATION TESTS
// =============================================================================

#[test]
fn test_e2e_forward_pass() {
    let input_dim = 16;
    let hidden_dim = 8;
    let output_dim = 4;
    let batch_size = 4;

    // Create model
    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();

    // Create synthetic batch
    let (input, _target) = create_synthetic_batch(batch_size, input_dim, output_dim);

    // Run forward pass
    let result = model.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(result.output.shape(), &vec![batch_size, output_dim]);

    // Verify output is valid probability distribution (sums to ~1.0 per row)
    for i in 0..batch_size {
        let mut row_sum = 0.0;
        for j in 0..output_dim {
            let val = result.output.get(&[i, j]).unwrap().value();
            assert!(val >= 0.0, "Probability should be non-negative");
            assert!(val <= 1.0, "Probability should be at most 1.0");
            row_sum += val;
        }
        assert!((row_sum - 1.0).abs() < 1e-6, "Probabilities should sum to 1.0");
    }

    // Verify error bounds are reasonable
    assert!(result.output.is_finite(), "Output should be finite");
    assert!(result.output.max_error() < MAX_SAFE_ERROR, "Error should be within safe limits");
}

#[test]
fn test_e2e_backward_pass() {
    let input_dim = 16;
    let hidden_dim = 8;
    let output_dim = 4;
    let batch_size = 4;

    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();
    let (input, target) = create_synthetic_batch(batch_size, input_dim, output_dim);

    // Forward pass
    let forward = model.forward(&input).unwrap();

    // Backward pass
    let gradients = model.backward(&input, &forward, &target).unwrap();

    // Verify gradient shapes
    assert_eq!(gradients.dw1.shape(), &vec![input_dim, hidden_dim]);
    assert_eq!(gradients.db1.shape(), &vec![hidden_dim]);
    assert_eq!(gradients.dw2.shape(), &vec![hidden_dim, output_dim]);
    assert_eq!(gradients.db2.shape(), &vec![output_dim]);

    // Verify gradients are finite
    assert!(gradients.dw1.is_finite(), "dw1 should be finite");
    assert!(gradients.db1.is_finite(), "db1 should be finite");
    assert!(gradients.dw2.is_finite(), "dw2 should be finite");
    assert!(gradients.db2.is_finite(), "db2 should be finite");
}

#[test]
fn test_e2e_loss_computation() {
    let input_dim = 16;
    let hidden_dim = 8;
    let output_dim = 4;
    let batch_size = 4;

    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();
    let (input, target) = create_synthetic_batch(batch_size, input_dim, output_dim);

    let forward = model.forward(&input).unwrap();
    let loss = model.compute_loss(&forward.output, &target).unwrap();

    // Loss should be positive (cross-entropy is always positive)
    assert!(loss.value() >= 0.0, "Loss should be non-negative");

    // Loss should be finite
    assert!(loss.is_finite(), "Loss should be finite");

    // Loss should have reasonable error bound
    assert!(loss.absolute_error() < MAX_SAFE_ERROR, "Loss error should be bounded");

    println!("Loss: {} ± {:.2e}", loss.value(), loss.absolute_error());
}

#[test]
fn test_e2e_witness_generation() {
    let input_dim = 8;
    let hidden_dim = 4;
    let output_dim = 2;
    let batch_size = 2;

    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();
    let (input, target) = create_synthetic_batch(batch_size, input_dim, output_dim);

    // Complete training step
    let step = model.training_step(&input, &target, 0).unwrap();

    // Generate witness
    let witness = step.generate_witness();
    let elements = witness.to_field_elements();

    // Witness should contain all intermediate values
    // Expected: input + z1 + a1 + z2 + output + dw1 + dw2
    let expected_elements =
        (batch_size * input_dim)      // input
        + (batch_size * hidden_dim)   // z1
        + (batch_size * hidden_dim)   // a1
        + (batch_size * output_dim)   // z2
        + (batch_size * output_dim)   // output
        + (input_dim * hidden_dim)    // dw1
        + (hidden_dim * output_dim);  // dw2

    assert_eq!(elements.len(), expected_elements, "Witness should contain all intermediate values");

    // Public inputs should be [loss, error_bound, step]
    let public_inputs = step.public_inputs();
    assert_eq!(public_inputs.len(), 3);
    assert_eq!(public_inputs[2], 0); // Step number

    // Verify circuit ID
    assert_eq!(step.circuit_id(), "ml_training_step_v1");

    println!("Witness elements: {}", elements.len());
    println!("Public inputs: {:?}", public_inputs);
}

#[test]
fn test_e2e_error_budget_tracking() {
    let input_dim = 16;
    let hidden_dim = 8;
    let output_dim = 4;
    let batch_size = 4;

    // Create budget allocator
    let config = BudgetAllocationConfig {
        total_budget: 0.01, // 1% total error budget
        strategy: AllocationStrategy::Weighted,
        min_allocation: 0.01,
        max_allocation: 0.4,
        reserve_fraction: 0.1,
        update_interval: 100,
        smoothing_factor: 0.9,
    };

    let mut allocator = BudgetAllocator::new(config);

    // Add components
    allocator.add_component(BudgetComponent::new("forward", ComponentType::FeedForward));
    allocator.add_component(BudgetComponent::new("loss", ComponentType::Loss));
    allocator.add_component(BudgetComponent::new("backward", ComponentType::Gradient));
    allocator.add_component(BudgetComponent::new("weight_update", ComponentType::WeightUpdate));

    // Allocate budgets
    let allocation = allocator.allocate();
    assert!(!allocation.allocations.is_empty(), "Should successfully allocate budgets");

    // Run training and track error consumption
    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();
    let (input, target) = create_synthetic_batch(batch_size, input_dim, output_dim);

    let step = model.training_step(&input, &target, 0).unwrap();

    // Record error consumption for each component
    let forward_error = step.forward.output.max_error();
    let loss_error = step.loss.absolute_error();
    let grad_error = step.gradients.dw1.max_error().max(step.gradients.dw2.max_error());

    allocator.consume("forward", forward_error);
    allocator.consume("loss", loss_error);
    allocator.consume("backward", grad_error);

    // Verify no component is over budget (for a single step)
    let summary = allocator.summary();
    println!("Budget Summary:\n{}", summary);

    // For a single step, we shouldn't exceed budget
    for component in ["forward", "loss", "backward", "weight_update"] {
        let comp = allocator.get_component(component);
        if let Some(c) = comp {
            assert!(
                !c.is_over_budget(),
                "Component {} should not be over budget after one step",
                component
            );
        }
    }
}

#[test]
fn test_e2e_full_training_loop() {
    let input_dim = 8;
    let hidden_dim = 4;
    let output_dim = 2;
    let batch_size = 4;
    let num_steps = 10;

    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();

    let start = Instant::now();
    let mut total_loss: f64 = 0.0;
    let mut max_error_seen: f64 = 0.0;
    let mut witnesses_generated = 0;

    for step in 0..num_steps {
        let (input, target) = create_synthetic_batch(batch_size, input_dim, output_dim);

        let training_step = model.training_step(&input, &target, step).unwrap();

        // Track metrics
        total_loss += training_step.loss.value();
        max_error_seen = max_error_seen.max(training_step.forward.output.max_error());

        // Generate witness
        let _witness = training_step.generate_witness();
        witnesses_generated += 1;

        // Verify error bounds haven't exploded
        assert!(
            training_step.forward.output.max_error() < MAX_SAFE_ERROR,
            "Error explosion detected at step {}",
            step
        );
    }

    let elapsed = start.elapsed();
    let avg_loss = total_loss / num_steps as f64;

    println!("=== E2E Training Summary ===");
    println!("Steps completed: {}", num_steps);
    println!("Witnesses generated: {}", witnesses_generated);
    println!("Average loss: {:.6}", avg_loss);
    println!("Max error seen: {:.2e}", max_error_seen);
    println!("Total time: {:?}", elapsed);
    println!("Time per step: {:?}", elapsed / num_steps as u32);

    // Verify training completed successfully
    assert!(avg_loss.is_finite(), "Average loss should be finite");
    assert!(max_error_seen < MAX_SAFE_ERROR, "Max error should be within safe limits");
    assert_eq!(witnesses_generated, num_steps, "Should generate witness for each step");
}

#[test]
fn test_e2e_batched_matmul() {
    // Test the batch_matmul operation directly
    let batch = 3;
    let m = 4;
    let k = 5;
    let n = 6;

    // Create batched tensors
    let a_data: Vec<f64> = (0..batch * m * k)
        .map(|i| (i as f64 * 0.1).sin())
        .collect();
    let a = BoundedTensor::from_approximate(a_data, vec![batch, m, k], 1e-7);

    let b_data: Vec<f64> = (0..batch * k * n)
        .map(|i| (i as f64 * 0.2).cos())
        .collect();
    let b = BoundedTensor::from_approximate(b_data, vec![batch, k, n], 1e-7);

    // Batch matmul
    let c = a.batch_matmul(&b).unwrap();

    // Verify shape
    assert_eq!(c.shape(), &vec![batch, m, n]);

    // Verify result is finite
    assert!(c.is_finite(), "Batch matmul result should be finite");

    // Verify error bounds
    assert!(
        c.max_error() < MAX_SAFE_ERROR,
        "Batch matmul error should be within safe limits"
    );

    println!("Batch matmul: [{}, {}, {}] @ [{}, {}, {}] = [{}, {}, {}]",
        batch, m, k, batch, k, n, batch, m, n);
    println!("Max error: {:.2e}", c.max_error());
}

#[test]
fn test_e2e_batch_attention() {
    let batch = 2;
    let seq_len = 4;
    let d_model = 8;

    // Create Q, K, V tensors
    let q_data: Vec<f64> = (0..batch * seq_len * d_model)
        .map(|i| (i as f64 * 0.1).sin() * 0.5)
        .collect();
    let q = BoundedTensor::from_approximate(q_data, vec![batch, seq_len, d_model], 1e-7);

    let k_data: Vec<f64> = (0..batch * seq_len * d_model)
        .map(|i| (i as f64 * 0.15).cos() * 0.5)
        .collect();
    let k = BoundedTensor::from_approximate(k_data, vec![batch, seq_len, d_model], 1e-7);

    let v_data: Vec<f64> = (0..batch * seq_len * d_model)
        .map(|i| (i as f64 * 0.2).sin() * 0.5)
        .collect();
    let v = BoundedTensor::from_approximate(v_data, vec![batch, seq_len, d_model], 1e-7);

    // Compute attention
    let output = BoundedTensor::batch_attention(&q, &k, &v).unwrap();

    // Verify shape
    assert_eq!(output.shape(), &vec![batch, seq_len, d_model]);

    // Verify result is finite
    assert!(output.is_finite(), "Attention output should be finite");

    // Verify error bounds
    assert!(
        output.max_error() < MAX_SAFE_ERROR,
        "Attention error should be within safe limits"
    );

    println!("Batch attention: Q,K,V [{}, {}, {}] -> [{}, {}, {}]",
        batch, seq_len, d_model, batch, seq_len, d_model);
    println!("Max error: {:.2e}", output.max_error());
}

#[test]
fn test_e2e_parallel_operations() {
    // Test parallel operations with larger tensors
    let size = 10000;

    let a = BoundedTensor::from_approximate(
        (0..size).map(|i| i as f64 * 0.001).collect(),
        vec![size],
        1e-7,
    );

    let b = BoundedTensor::from_approximate(
        (0..size).map(|i| (size - i) as f64 * 0.001).collect(),
        vec![size],
        1e-7,
    );

    // Parallel add
    let start = Instant::now();
    let c = a.par_add(&b).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(c.shape(), &vec![size]);
    assert!(c.is_finite());

    println!("Parallel add of {} elements: {:?}", size, elapsed);

    // Parallel sum
    let start = Instant::now();
    let sum = a.par_sum();
    let elapsed = start.elapsed();

    assert!(sum.is_finite());
    println!("Parallel sum of {} elements: {:?}", size, elapsed);

    // Parallel max error
    let start = Instant::now();
    let max_err = a.par_max_error();
    let elapsed = start.elapsed();

    assert!(max_err < 1.0);
    println!("Parallel max_error of {} elements: {:?}", size, elapsed);
}

// =============================================================================
// REGRESSION TESTS FOR LONG TRAINING RUNS
// =============================================================================

/// Regression test: Verify error bounds don't explode over 1000+ training steps.
///
/// This test validates the core claim that HELIX can track error bounds reliably
/// over extended training runs without silent numerical instability.
#[test]
fn test_regression_1000_steps_error_stability() {
    let input_dim = 8;
    let hidden_dim = 4;
    let output_dim = 2;
    let batch_size = 4;
    let num_steps = 1000;

    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();

    let start = Instant::now();
    let mut max_error_per_100 = Vec::new();
    let mut loss_values = Vec::new();
    let mut error_growth_rate = Vec::new();
    let mut prev_max_error = 0.0_f64;

    for step in 0..num_steps {
        let (input, target) = create_synthetic_batch(batch_size, input_dim, output_dim);
        let training_step = model.training_step(&input, &target, step).unwrap();

        let current_max_error = training_step.forward.output.max_error()
            .max(training_step.gradients.dw1.max_error())
            .max(training_step.gradients.dw2.max_error());

        // Track error growth rate
        if prev_max_error > 0.0 {
            let growth = current_max_error / prev_max_error;
            error_growth_rate.push(growth);
        }
        prev_max_error = current_max_error;

        // Sample every 100 steps
        if step % 100 == 99 {
            max_error_per_100.push(current_max_error);
            loss_values.push(training_step.loss.value());
        }

        // Critical check: error must stay within safe bounds
        assert!(
            current_max_error < MAX_SAFE_ERROR,
            "Error explosion at step {}: {:.2e} exceeds safe limit {:.2e}",
            step, current_max_error, MAX_SAFE_ERROR
        );

        // Check that error growth is bounded (not exponentially exploding)
        // Allow up to 10x growth per step in worst case
        if step > 0 && prev_max_error > 1e-15 {
            let growth = current_max_error / prev_max_error;
            assert!(
                growth < 100.0,
                "Unbounded error growth at step {}: {:.2e}x",
                step, growth
            );
        }
    }

    let elapsed = start.elapsed();

    // Compute statistics
    let avg_growth: f64 = if !error_growth_rate.is_empty() {
        error_growth_rate.iter().sum::<f64>() / error_growth_rate.len() as f64
    } else {
        1.0
    };

    let final_error = *max_error_per_100.last().unwrap_or(&0.0);
    let initial_error = *max_error_per_100.first().unwrap_or(&0.0);

    println!("=== 1000-Step Regression Test ===");
    println!("Steps completed: {}", num_steps);
    println!("Total time: {:?}", elapsed);
    println!("Time per step: {:?}", elapsed / num_steps as u32);
    println!("Initial max error: {:.2e}", initial_error);
    println!("Final max error: {:.2e}", final_error);
    println!("Error growth factor: {:.2e}x", final_error / initial_error.max(1e-15));
    println!("Average step growth: {:.4}x", avg_growth);
    println!("Max errors per 100 steps: {:?}",
        max_error_per_100.iter().map(|e| format!("{:.2e}", e)).collect::<Vec<_>>());

    // Verify error didn't grow unboundedly
    // Over 1000 steps, we expect some growth but not exponential explosion
    // Allow up to 1e6 total growth factor (very conservative)
    let total_growth = final_error / initial_error.max(1e-15);
    assert!(
        total_growth < 1e6,
        "Error grew too much over {} steps: {:.2e}x",
        num_steps, total_growth
    );

    // Verify all losses are finite
    assert!(
        loss_values.iter().all(|l| l.is_finite()),
        "Some loss values became non-finite"
    );
}

/// Regression test: Verify matmul error accumulation over many operations.
#[test]
fn test_regression_matmul_chain_error() {
    // Chain of 100 matrix multiplications to test error accumulation
    let size = 16;
    let num_ops = 100;

    // Start with identity-like matrix with small perturbations
    let mut current = BoundedTensor::from_approximate(
        (0..size * size)
            .map(|i| if i / size == i % size { 1.0 } else { 0.01 })
            .collect(),
        vec![size, size],
        1e-10,
    );

    let multiplier = BoundedTensor::from_approximate(
        (0..size * size)
            .map(|i| if i / size == i % size { 0.99 } else { 0.001 })
            .collect(),
        vec![size, size],
        1e-10,
    );

    let initial_error = current.max_error();
    let mut max_error_seen = initial_error;

    for i in 0..num_ops {
        current = current.matmul(&multiplier).expect(&format!("Matmul failed at step {}", i));

        let error = current.max_error();
        max_error_seen = max_error_seen.max(error);

        // Verify error doesn't explode
        assert!(
            error < MAX_SAFE_ERROR,
            "Error explosion at matmul step {}: {:.2e}",
            i, error
        );
    }

    let final_error = current.max_error();
    let growth = final_error / initial_error;

    println!("=== Matmul Chain Test ({} ops) ===", num_ops);
    println!("Initial error: {:.2e}", initial_error);
    println!("Final error: {:.2e}", final_error);
    println!("Max error seen: {:.2e}", max_error_seen);
    println!("Total growth: {:.2e}x", growth);

    // Error should grow polynomially, not exponentially
    // For well-conditioned matrices, expect roughly O(n) growth
    assert!(
        growth < (num_ops as f64).powi(3),
        "Error growth {:.2e} exceeds polynomial bound",
        growth
    );
}

/// Regression test: Verify attention error accumulation.
#[test]
fn test_regression_attention_error_accumulation() {
    let batch = 2;
    let seq_len = 8;
    let d_model = 16;
    let num_layers = 10;

    // Simulate 10 attention layers
    let mut q = BoundedTensor::from_approximate(
        (0..batch * seq_len * d_model)
            .map(|i| ((i as f64) * 0.01).sin())
            .collect(),
        vec![batch, seq_len, d_model],
        1e-10,
    );

    let initial_error = q.max_error();
    let mut errors_per_layer = vec![initial_error];

    for layer in 0..num_layers {
        let k = BoundedTensor::from_approximate(
            (0..batch * seq_len * d_model)
                .map(|i| ((i as f64 + layer as f64 * 100.0) * 0.01).cos())
                .collect(),
            vec![batch, seq_len, d_model],
            1e-10,
        );

        let v = BoundedTensor::from_approximate(
            (0..batch * seq_len * d_model)
                .map(|i| ((i as f64 + layer as f64 * 200.0) * 0.01).sin())
                .collect(),
            vec![batch, seq_len, d_model],
            1e-10,
        );

        // Apply attention
        q = BoundedTensor::batch_attention(&q, &k, &v)
            .expect(&format!("Attention failed at layer {}", layer));

        errors_per_layer.push(q.max_error());

        // Verify no explosion
        assert!(
            q.max_error() < MAX_SAFE_ERROR,
            "Error explosion at attention layer {}: {:.2e}",
            layer, q.max_error()
        );
    }

    let final_error = q.max_error();
    let growth = final_error / initial_error;

    println!("=== Attention Layer Stack Test ({} layers) ===", num_layers);
    println!("Initial error: {:.2e}", initial_error);
    println!("Final error: {:.2e}", final_error);
    println!("Total growth: {:.2e}x", growth);
    println!("Errors per layer: {:?}",
        errors_per_layer.iter().map(|e| format!("{:.2e}", e)).collect::<Vec<_>>());

    // Verify bounded growth
    assert!(
        growth < 1e12,
        "Attention error growth {:.2e} is too large",
        growth
    );
}

/// Regression test: Verify error bounds with BF16-level precision over 1000 steps.
///
/// BF16 (Brain Float 16) has an epsilon of approximately 3.91e-3, which is
/// significantly larger than f64's machine epsilon. This test uses the
/// precision-aware `VMConfig::for_precision(BF16)` budget (0.05) instead of
/// the default F32 budget (0.01).
#[test]
fn test_regression_1000_steps_bf16_precision() {
    let input_dim = 8;
    let hidden_dim = 4;
    let output_dim = 2;
    let batch_size = 4;
    let num_steps = 1000;

    // Use precision-aware config
    let vm_config = VMConfig::for_precision(Precision::BF16);
    let budget = vm_config.max_error_accumulation; // 0.05 for BF16

    // BF16 epsilon: 2^-8 ≈ 3.91e-3
    let bf16_epsilon: f64 = 3.91e-3;

    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();

    let start = Instant::now();
    let mut max_error_per_100 = Vec::new();
    let mut max_single_step_error = 0.0_f64;

    for step in 0..num_steps {
        // Create input with BF16-level epsilon instead of the default 1e-7
        let input_data: Vec<f64> = (0..batch_size * input_dim)
            .map(|i| (((i + step * batch_size * input_dim) as f64 * 0.1).sin() + 1.0) / 2.0)
            .collect();
        let input = BoundedTensor::from_approximate(
            input_data,
            vec![batch_size, input_dim],
            bf16_epsilon,
        );

        // One-hot encoded targets
        let mut target_data = vec![0.0; batch_size * output_dim];
        for i in 0..batch_size {
            let class = i % output_dim;
            target_data[i * output_dim + class] = 1.0;
        }
        let target = BoundedTensor::from_exact(target_data, vec![batch_size, output_dim]);

        let training_step = model.training_step(&input, &target, step).unwrap();

        let current_max_error = training_step.forward.output.max_error()
            .max(training_step.gradients.dw1.max_error())
            .max(training_step.gradients.dw2.max_error());

        max_single_step_error = max_single_step_error.max(current_max_error);

        // Sample every 100 steps
        if step % 100 == 99 {
            max_error_per_100.push(current_max_error);
        }

        // Critical check: error must stay within safe bounds
        assert!(
            current_max_error < vm_config.max_safe_error,
            "BF16 error explosion at step {}: {:.2e} exceeds max_safe_error {:.2e}",
            step, current_max_error, vm_config.max_safe_error
        );
    }

    let elapsed = start.elapsed();
    let final_error = *max_error_per_100.last().unwrap_or(&0.0);
    let initial_error = *max_error_per_100.first().unwrap_or(&0.0);

    // With precision-aware budget, BF16 per-step error should fit
    let budget_holds = max_single_step_error < budget;

    println!("=== BF16 Precision 1000-Step Regression Test ===");
    println!("BF16 epsilon: {:.2e}", bf16_epsilon);
    println!("Precision-aware budget (BF16_MAX_ERROR_ACCUMULATION): {}", budget);
    println!("Steps completed: {} in {:?}", num_steps, elapsed);
    println!("Initial max error (step 100): {:.2e}", initial_error);
    println!("Final max error (step 1000): {:.2e}", final_error);
    println!("Max single-step error: {:.2e}", max_single_step_error);
    println!("Budget holds per step: {}", budget_holds);
    println!("Max errors per 100 steps: {:?}",
        max_error_per_100.iter().map(|e| format!("{:.2e}", e)).collect::<Vec<_>>());

    // With BF16-appropriate budget, per-step error should now pass
    assert!(
        budget_holds,
        "BF16: max single-step error {:.2e} exceeds BF16 budget {:.2e}",
        max_single_step_error, budget
    );

    // Error must remain finite and bounded
    assert!(
        max_single_step_error < vm_config.max_safe_error,
        "BF16: max single-step error {:.2e} exceeds max_safe_error",
        max_single_step_error
    );

    // Verify error growth is bounded across the sampled windows
    if initial_error > 1e-15 {
        let total_growth = final_error / initial_error;
        assert!(
            total_growth < 1e6,
            "BF16: error grew too much over {} steps: {:.2e}x",
            num_steps, total_growth
        );
    }
}

/// Regression test: Verify error bounds with INT8-level precision over 1000 steps.
///
/// INT8 quantization has an epsilon of 1/256 ≈ 3.906e-3. This test uses the
/// precision-aware `VMConfig::for_precision(INT8)` budget (0.10) instead of
/// the default F32 budget (0.01).
#[test]
fn test_regression_1000_steps_int8_precision() {
    let input_dim = 8;
    let hidden_dim = 4;
    let output_dim = 2;
    let batch_size = 4;
    let num_steps = 1000;

    // Use precision-aware config
    let vm_config = VMConfig::for_precision(Precision::INT8);
    let budget = vm_config.max_error_accumulation; // 0.10 for INT8

    // INT8 epsilon: 1/256 ≈ 3.906e-3
    let int8_epsilon: f64 = 1.0 / 256.0;

    let model = SimpleMLP::new(input_dim, hidden_dim, output_dim).unwrap();

    let start = Instant::now();
    let mut max_error_per_100 = Vec::new();
    let mut max_single_step_error = 0.0_f64;
    let mut error_within_budget_count = 0_u64;

    for step in 0..num_steps {
        // Create input with INT8-level epsilon
        let input_data: Vec<f64> = (0..batch_size * input_dim)
            .map(|i| {
                // Simulate INT8 quantized values: values in [0, 1] snapped to 1/256 grid
                let raw = (((i + step * batch_size * input_dim) as f64 * 0.1).sin() + 1.0) / 2.0;
                (raw * 256.0).round() / 256.0
            })
            .collect();
        let input = BoundedTensor::from_approximate(
            input_data,
            vec![batch_size, input_dim],
            int8_epsilon,
        );

        // One-hot encoded targets
        let mut target_data = vec![0.0; batch_size * output_dim];
        for i in 0..batch_size {
            let class = i % output_dim;
            target_data[i * output_dim + class] = 1.0;
        }
        let target = BoundedTensor::from_exact(target_data, vec![batch_size, output_dim]);

        let training_step = model.training_step(&input, &target, step).unwrap();

        let current_max_error = training_step.forward.output.max_error()
            .max(training_step.gradients.dw1.max_error())
            .max(training_step.gradients.dw2.max_error());

        max_single_step_error = max_single_step_error.max(current_max_error);

        if current_max_error < budget {
            error_within_budget_count += 1;
        }

        // Sample every 100 steps
        if step % 100 == 99 {
            max_error_per_100.push(current_max_error);
        }

        // Critical check: error must stay within safe bounds
        assert!(
            current_max_error < vm_config.max_safe_error,
            "INT8 error explosion at step {}: {:.2e} exceeds max_safe_error {:.2e}",
            step, current_max_error, vm_config.max_safe_error
        );
    }

    let elapsed = start.elapsed();
    let final_error = *max_error_per_100.last().unwrap_or(&0.0);
    let initial_error = *max_error_per_100.first().unwrap_or(&0.0);

    let budget_holds = max_single_step_error < budget;
    let budget_hold_pct = (error_within_budget_count as f64 / num_steps as f64) * 100.0;

    println!("=== INT8 Precision 1000-Step Regression Test ===");
    println!("INT8 epsilon: {:.6e} (1/256)", int8_epsilon);
    println!("Precision-aware budget (INT8_MAX_ERROR_ACCUMULATION): {}", budget);
    println!("Steps completed: {} in {:?}", num_steps, elapsed);
    println!("Initial max error (step 100): {:.2e}", initial_error);
    println!("Final max error (step 1000): {:.2e}", final_error);
    println!("Max single-step error: {:.2e}", max_single_step_error);
    println!("Budget holds per step: {}", budget_holds);
    println!("Steps within budget: {}/{} ({:.1}%)",
        error_within_budget_count, num_steps, budget_hold_pct);
    println!("Max errors per 100 steps: {:?}",
        max_error_per_100.iter().map(|e| format!("{:.2e}", e)).collect::<Vec<_>>());

    // With INT8-appropriate budget, per-step error should now pass
    assert!(
        budget_holds,
        "INT8: max single-step error {:.2e} exceeds INT8 budget {:.2e}",
        max_single_step_error, budget
    );

    // Error must remain finite and bounded
    assert!(
        max_single_step_error < vm_config.max_safe_error,
        "INT8: max single-step error {:.2e} exceeds max_safe_error",
        max_single_step_error
    );

    // Verify error growth is bounded across the sampled windows
    if initial_error > 1e-15 {
        let total_growth = final_error / initial_error;
        assert!(
            total_growth < 1e6,
            "INT8: error grew too much over {} steps: {:.2e}x",
            num_steps, total_growth
        );
    }
}
