//! Backward pass implementation for automatic differentiation.

use super::autodiff::{GradientTape, NodeIndex, Operation, Variable};
use crate::ops::{matmul, conv};
use helix_core::types::{BoundedTensor, BoundedValue, Precision};
use std::collections::HashMap;

/// Computes gradients for all variables in the computation graph with respect to the loss.
///
/// Returns a map from NodeIndex to the gradient tensor (dL/dx).
pub fn backward(loss: &Variable) -> Result<HashMap<NodeIndex, BoundedTensor>, String> {
    // 1. Check if tape exists
    let tape_rc = loss.tape.as_ref().ok_or("Variable is not tracked (no tape)")?;
    let tape = tape_rc.borrow();

    // 2. Initialize gradients map
    // We map NodeIndex -> Gradient Tensor
    let mut grads: HashMap<NodeIndex, BoundedTensor> = HashMap::new();

    // 3. Seed the backward pass with the gradient of loss w.r.t itself (1.0)
    // The loss must be a scalar.
    if !loss.tensor.shape().iter().product::<usize>() == 1 {
        // For now, we only support scalar loss backward.
        // If it's not scalar, we could sum it or error.
        // Let's assume scalar for standard ML training.
        // Actually, let's just create a tensor of ones with same shape as loss if it's not scalar
        // (vector Jacobian product with v=1).
    }
    
    // Gradient of loss is 1.0 with 0.0 error (exact).
    let seed_grad = BoundedTensor::full(
        loss.tensor.shape().clone(), 
        BoundedValue::exact(1.0)
    );
    
    if let Some(idx) = loss.node_index {
        grads.insert(idx, seed_grad);
    } else {
        return Err("Loss variable has no node index".to_string());
    }

    // 4. Iterate backwards through the tape
    // Nodes were added in topological order, so reverse order guarantees we process outputs before inputs.
    for (idx, node) in tape.nodes.iter().enumerate().rev() {
        // If we don't have a gradient for this node, it didn't affect the loss (or we pruned it).
        // However, we should have it if it's connected to loss.
        if !grads.contains_key(&idx) {
            continue;
        }

        let grad_output = grads.get(&idx).unwrap().clone();

        match &node.op {
            Operation::Input | Operation::Parameter => {
                // Leaf nodes, nothing to propagate to.
                // The gradient is already stored in `grads`.
            }
            Operation::Add(lhs_idx, rhs_idx) => {
                // y = a + b
                // dL/da = dL/dy
                // dL/db = dL/dy
                accumulate_grad(&mut grads, *lhs_idx, &grad_output);
                accumulate_grad(&mut grads, *rhs_idx, &grad_output);
            }
            Operation::Sub(lhs_idx, rhs_idx) => {
                // y = a - b
                // dL/da = dL/dy
                // dL/db = -dL/dy
                accumulate_grad(&mut grads, *lhs_idx, &grad_output);
                
                let neg_grad = grad_output.scale(BoundedValue::exact(-1.0));
                accumulate_grad(&mut grads, *rhs_idx, &neg_grad);
            }
            Operation::MatMul(lhs_idx, rhs_idx) => {
                // Y = A @ B
                // dL/dA = dL/dY @ B^T
                // dL/dB = A^T @ dL/dY
                
                // Need to retrieve A and B shapes to know if we need to access their values
                // For MatMul backward we DO need the values of A and B.
                // But the Tape currently only stores metadata (NodeInfo).
                // THIS IS A DESIGN FLAW in the minimal tape. 
                // A Wengert list usually stores references to the inputs or values.
                //
                // FIX: In a real implementation we need access to the forward pass values.
                // Since this is a specialized implementation, let's assume `Variable` kept the values alive
                // or the Tape stores the values.
                //
                // Given the constraints and the provided file structures, `Variable` holds the tensor.
                // But `Variable`s are owned by the user.
                // We need the `GradientTape` to store the *values* (or strong refs) if we want to do backward 
                // without the user keeping all intermediates alive.
                //
                // For this stage, let's assume we can't implement MatMul backward fully without the values.
                // However, I can't easily change `GradientTape` to store BoundedTensor without cloning.
                // cloning BoundedTensor is fine (it's data).
                //
                // Let's modify the plan slightly: I will NOT change `GradientTape` definition in `autodiff.rs` now (too messy).
                // I will add a panic! for now or implement what I can.
                // Wait, if I can't do MatMul backward, I can't train.
                //
                // Actually, I can rely on the fact that I don't have the values.
                // Retaining values is required for MatMul.
                // 
                // Let's look at `helix-avm/src/gradient/autodiff.rs` again.
                // I should assume the `GradientTape` optionally can cache values, OR 
                // I assume the user keeps variables alive? No, intermediate variables are dropped.
                //
                // OK, I will update `autodiff.rs` to store `Option<BoundedTensor>` in `NodeInfo`.
                // This is necessary for non-linear ops and Mul/MatMul.
                //
                // Since I am writing `backward.rs` now, I will write it assuming `NodeInfo` has `cached_value`.
                // I will then go back and update `autodiff.rs`.
                
                // Let's assume `node.cached_value` exists.
                // Wait, I can't assume that if I haven't written it.
                // I must update `autodiff.rs` FIRST or concurrently.
                // But I am in the middle of writing `backward.rs`.
                //
                // Strategy: I will write `backward.rs` assuming `node.output` is available.
                // Then I will update `autodiff.rs` immediately after.
                
                // ... logic continues assuming `node.output` ...
                // But wait, for MatMul backwards: dL/dA = G @ B^T. We need B.
                // `tape.nodes[rhs_idx]` should have the value of B.
                
                 let lhs_node = &tape.nodes[*lhs_idx];
                 let rhs_node = &tape.nodes[*rhs_idx];
                 
                 // We need the VALUES of lhs and rhs.
                 // If the tape stored the *result* of the operation at `idx`, 
                 // it doesn't store the inputs A and B directly, only their indices.
                 // So we can look up `tape.nodes[lhs_idx].cached_value`.
                 
                 if let (Some(lhs_val), Some(rhs_val)) = (&lhs_node.cached_value, &rhs_node.cached_value) {
                     let precision = Precision::F32;
                     
                     // dL/dA = grad @ B^T
                     let b_t = rhs_val.transpose();
                     let grad_a = matmul::matmul(&grad_output, &b_t, precision)
                         .map_err(|e| e.to_string())?;
                     accumulate_grad(&mut grads, *lhs_idx, &grad_a);

                     // dL/dB = A^T @ grad
                     let a_t = lhs_val.transpose();
                     let grad_b = matmul::matmul(&a_t, &grad_output, precision)
                         .map_err(|e| e.to_string())?;
                     accumulate_grad(&mut grads, *rhs_idx, &grad_b);
                 } else {
                     return Err(format!("Missing cached values for MatMul inputs at node {}", idx));
                 }
            }
            Operation::Relu(input_idx) => {
                // y = Relu(x)
                // dL/dx = dL/dy * (1 if x > 0 else 0)
                let input_node = &tape.nodes[*input_idx];
                if let Some(input_val) = &input_node.cached_value {
                    let mask_data: Vec<BoundedValue<f64>> = input_val.data().iter().map(|v| {
                        if v.value() > 0.0 {
                            BoundedValue::exact(1.0)
                        } else {
                            BoundedValue::exact(0.0)
                        }
                    }).collect();
                    
                    let mask = BoundedTensor::new(mask_data, input_val.shape().clone());
                    let grad_input = grad_output.hadamard(&mask);
                    
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::Sigmoid(input_idx) => {
                // y = sigmoid(x)
                // dL/dx = dL/dy * y * (1 - y)
                let current_node = &tape.nodes[idx];
                if let Some(output_val) = &current_node.cached_value {
                    let grad_input_data: Vec<BoundedValue<f64>> = output_val.data()
                        .iter()
                        .zip(grad_output.data().iter())
                        .map(|(y, g)| {
                            let y_val = y.value();
                            let derivative = y_val * (1.0 - y_val);
                            let grad_val = g.value() * derivative;
                            // Error: d(derivative)/dy = 1 - 2y, so error compounds
                            let error = g.absolute_error() * derivative.abs() 
                                + g.value().abs() * (1.0 - 2.0 * y_val).abs() * y.absolute_error();
                            BoundedValue::new(grad_val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();
                    
                    let grad_input = BoundedTensor::new(grad_input_data, output_val.shape().clone());
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::Softmax(input_idx) => {
                // y_i = softmax(x)_i = exp(x_i) / sum(exp(x))
                // dL/dx_i = sum_j(dL/dy_j * dy_j/dx_i)
                // dy_j/dx_i = y_i * (delta_ij - y_j)
                // Simplified: dL/dx = y * (dL/dy - sum(dL/dy * y))
                let current_node = &tape.nodes[idx];
                if let Some(softmax_output) = &current_node.cached_value {
                    // Compute sum(dL/dy * y)
                    let weighted_sum: f64 = grad_output.data()
                        .iter()
                        .zip(softmax_output.data().iter())
                        .map(|(g, y)| g.value() * y.value())
                        .sum();
                    
                    let grad_input_data: Vec<BoundedValue<f64>> = softmax_output.data()
                        .iter()
                        .zip(grad_output.data().iter())
                        .map(|(y, g)| {
                            let y_val = y.value();
                            let grad_val = y_val * (g.value() - weighted_sum);
                            // Softmax gradient error is complex; approximate
                            let error = y.absolute_error() * g.value().abs() 
                                + g.absolute_error() * y_val;
                            BoundedValue::new(grad_val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();
                    
                    let grad_input = BoundedTensor::new(grad_input_data, softmax_output.shape().clone());
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::LayerNorm(input_idx, gamma_idx, beta_idx) => {
                // LayerNorm is complex; simplified backward
                // dL/dx_i ≈ (1/std) * (dL/dy_i - mean(dL/dy) - normalized_i * mean(dL/dy * normalized))
                let input_node = &tape.nodes[*input_idx];
                let current_node = &tape.nodes[idx];
                
                if let (Some(input_val), Some(output_val)) = (&input_node.cached_value, &current_node.cached_value) {
                    let n = input_val.len() as f64;
                    
                    // Compute mean and std of input
                    let mean: f64 = input_val.data().iter().map(|v| v.value()).sum::<f64>() / n;
                    let variance: f64 = input_val.data().iter()
                        .map(|v| (v.value() - mean).powi(2))
                        .sum::<f64>() / n;
                    let std = (variance + 1e-5).sqrt();
                    
                    // Compute normalized values
                    let normalized: Vec<f64> = input_val.data()
                        .iter()
                        .map(|v| (v.value() - mean) / std)
                        .collect();
                    
                    // Compute dL/dy mean and dL/dy * normalized mean
                    let grad_mean: f64 = grad_output.data().iter().map(|g| g.value()).sum::<f64>() / n;
                    let grad_norm_mean: f64 = grad_output.data()
                        .iter()
                        .zip(normalized.iter())
                        .map(|(g, &norm)| g.value() * norm)
                        .sum::<f64>() / n;
                    
                    let grad_input_data: Vec<BoundedValue<f64>> = grad_output.data()
                        .iter()
                        .zip(normalized.iter())
                        .map(|(g, &norm)| {
                            let grad_val = (1.0 / std) * (g.value() - grad_mean - norm * grad_norm_mean);
                            let error = g.absolute_error() / std;
                            BoundedValue::new(grad_val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();
                    
                    let grad_input = BoundedTensor::new(grad_input_data, input_val.shape().clone());
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                    
                    // Gamma gradient (if present): dL/dgamma = sum(dL/dy * normalized)
                    if let Some(g_idx) = gamma_idx {
                        let gamma_grad_data: Vec<BoundedValue<f64>> = grad_output.data()
                            .iter()
                            .zip(normalized.iter())
                            .map(|(g, &norm)| {
                                BoundedValue::exact(g.value() * norm)
                            })
                            .collect();
                        let gamma_grad = BoundedTensor::new(gamma_grad_data, input_val.shape().clone());
                        accumulate_grad(&mut grads, *g_idx, &gamma_grad);
                    }
                    
                    // Beta gradient (if present): dL/dbeta = sum(dL/dy)
                    if let Some(b_idx) = beta_idx {
                        accumulate_grad(&mut grads, *b_idx, &grad_output);
                    }
                }
            }
            Operation::Mul(lhs_idx, rhs_idx) => {
                // y = a * b (element-wise)
                // dL/da = dL/dy * b
                // dL/db = dL/dy * a
                let lhs_node = &tape.nodes[*lhs_idx];
                let rhs_node = &tape.nodes[*rhs_idx];
                
                if let (Some(lhs_val), Some(rhs_val)) = (&lhs_node.cached_value, &rhs_node.cached_value) {
                    let grad_lhs = grad_output.hadamard(rhs_val);
                    let grad_rhs = grad_output.hadamard(lhs_val);
                    
                    accumulate_grad(&mut grads, *lhs_idx, &grad_lhs);
                    accumulate_grad(&mut grads, *rhs_idx, &grad_rhs);
                }
            }
            Operation::Div(lhs_idx, rhs_idx) => {
                // y = a / b
                // dL/da = dL/dy / b
                // dL/db = -dL/dy * a / b^2
                let lhs_node = &tape.nodes[*lhs_idx];
                let rhs_node = &tape.nodes[*rhs_idx];
                
                if let (Some(lhs_val), Some(rhs_val)) = (&lhs_node.cached_value, &rhs_node.cached_value) {
                    let grad_lhs_data: Vec<BoundedValue<f64>> = grad_output.data()
                        .iter()
                        .zip(rhs_val.data().iter())
                        .map(|(g, b)| {
                            let val = g.value() / b.value();
                            let error = g.absolute_error() / b.value().abs();
                            BoundedValue::new(val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();
                    
                    let grad_rhs_data: Vec<BoundedValue<f64>> = grad_output.data()
                        .iter()
                        .zip(lhs_val.data().iter())
                        .zip(rhs_val.data().iter())
                        .map(|((g, a), b)| {
                            let b_sq = b.value() * b.value();
                            let val = -g.value() * a.value() / b_sq;
                            let error = g.absolute_error() * a.value().abs() / b_sq;
                            BoundedValue::new(val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();
                    
                    let grad_lhs = BoundedTensor::new(grad_lhs_data, lhs_val.shape().clone());
                    let grad_rhs = BoundedTensor::new(grad_rhs_data, rhs_val.shape().clone());
                    
                    accumulate_grad(&mut grads, *lhs_idx, &grad_lhs);
                    accumulate_grad(&mut grads, *rhs_idx, &grad_rhs);
                }
            }
            Operation::Sum(input_idx) => {
                // y = sum(x) (scalar output)
                // dL/dx_i = dL/dy (broadcast)
                let input_node = &tape.nodes[*input_idx];
                if let Some(input_val) = &input_node.cached_value {
                    // Broadcast the scalar gradient to input shape
                    let grad_scalar = if !grad_output.is_empty() {
                        grad_output.data()[0].value()
                    } else {
                        1.0
                    };
                    
                    let grad_input = BoundedTensor::full(
                        input_val.shape().clone(),
                        BoundedValue::exact(grad_scalar),
                    );
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::Mean(input_idx) => {
                // y = mean(x)
                // dL/dx_i = dL/dy / n
                let input_node = &tape.nodes[*input_idx];
                if let Some(input_val) = &input_node.cached_value {
                    let n = input_val.len() as f64;
                    let grad_scalar = if !grad_output.is_empty() {
                        grad_output.data()[0].value() / n
                    } else {
                        1.0 / n
                    };

                    let grad_input = BoundedTensor::full(
                        input_val.shape().clone(),
                        BoundedValue::exact(grad_scalar),
                    );
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::Conv1d { input, kernel, stride, padding } => {
                // Backward for 1D convolution
                let input_node = &tape.nodes[*input];
                let kernel_node = &tape.nodes[*kernel];
                let precision = Precision::F32;

                if let (Some(input_val), Some(kernel_val)) = (&input_node.cached_value, &kernel_node.cached_value) {
                    // dL/dInput
                    if let Ok(grad_input) = conv::conv1d_backward_input(
                        &grad_output,
                        kernel_val,
                        input_val.shape(),
                        *stride,
                        *padding,
                        precision,
                    ) {
                        accumulate_grad(&mut grads, *input, &grad_input);
                    }

                    // dL/dKernel
                    if let Ok(grad_kernel) = conv::conv1d_backward_weight(
                        input_val,
                        &grad_output,
                        kernel_val.shape(),
                        *stride,
                        *padding,
                        precision,
                    ) {
                        accumulate_grad(&mut grads, *kernel, &grad_kernel);
                    }
                }
            }
            Operation::Conv2d { input, kernel, stride, padding, dilation: _, groups: _ } => {
                // Backward for 2D convolution
                let input_node = &tape.nodes[*input];
                let kernel_node = &tape.nodes[*kernel];
                let precision = Precision::F32;

                if let (Some(input_val), Some(kernel_val)) = (&input_node.cached_value, &kernel_node.cached_value) {
                    // dL/dInput - use transposed convolution
                    if let Ok(grad_input) = conv::conv2d_backward_input(
                        &grad_output,
                        kernel_val,
                        input_val.shape(),
                        *stride,
                        *padding,
                        precision,
                    ) {
                        accumulate_grad(&mut grads, *input, &grad_input);
                    }

                    // dL/dKernel
                    if let Ok(grad_kernel) = conv::conv2d_backward_weight(
                        input_val,
                        &grad_output,
                        kernel_val.shape(),
                        *stride,
                        *padding,
                        precision,
                    ) {
                        accumulate_grad(&mut grads, *kernel, &grad_kernel);
                    }
                }
            }
            Operation::MaxPool2d { input, kernel_size, stride: _, padding: _, indices } => {
                // Backward for max pooling - gradient flows only to max positions
                let input_node = &tape.nodes[*input];

                if let Some(input_val) = &input_node.cached_value {
                    let grad_input = conv::max_pool2d_backward(
                        &grad_output,
                        indices,
                        input_val.shape(),
                    );
                    accumulate_grad(&mut grads, *input, &grad_input);
                }
            }
            Operation::AvgPool2d { input, kernel_size, stride, padding } => {
                // Backward for average pooling - gradient is distributed evenly
                let input_node = &tape.nodes[*input];

                if let Some(input_val) = &input_node.cached_value {
                    let config = conv::Pool2dConfig {
                        kernel_size: *kernel_size,
                        stride: *stride,
                        padding: *padding,
                    };
                    let grad_input = conv::avg_pool2d_backward(
                        &grad_output,
                        input_val.shape(),
                        config,
                    );
                    accumulate_grad(&mut grads, *input, &grad_input);
                }
            }
            Operation::DepthwiseConv2d { input, kernel, stride, padding } => {
                // Backward for depthwise convolution
                // Depthwise conv is essentially groups=channels, so we can use similar logic
                let input_node = &tape.nodes[*input];
                let kernel_node = &tape.nodes[*kernel];
                let precision = Precision::F32;

                if let (Some(input_val), Some(kernel_val)) = (&input_node.cached_value, &kernel_node.cached_value) {
                    // For depthwise, each channel is independent
                    // dL/dInput
                    if let Ok(grad_input) = conv::conv2d_backward_input(
                        &grad_output,
                        kernel_val,
                        input_val.shape(),
                        *stride,
                        *padding,
                        precision,
                    ) {
                        accumulate_grad(&mut grads, *input, &grad_input);
                    }

                    // dL/dKernel
                    if let Ok(grad_kernel) = conv::conv2d_backward_weight(
                        input_val,
                        &grad_output,
                        kernel_val.shape(),
                        *stride,
                        *padding,
                        precision,
                    ) {
                        accumulate_grad(&mut grads, *kernel, &grad_kernel);
                    }
                }
            }
        }
    }

    Ok(grads)
}

/// Helper to accumulate gradients (summing them if multiple paths lead to same node).
fn accumulate_grad(grads: &mut HashMap<NodeIndex, BoundedTensor>, idx: NodeIndex, grad: &BoundedTensor) {
    if let Some(existing) = grads.get_mut(&idx) {
        *existing = existing.add(grad);
    } else {
        grads.insert(idx, grad.clone());
    }
}
