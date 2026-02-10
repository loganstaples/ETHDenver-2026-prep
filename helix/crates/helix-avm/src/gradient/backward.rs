//! Backward pass implementation for automatic differentiation.

use super::autodiff::{NodeIndex, Operation, Variable};
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
                // Uses cached_value from NodeInfo to retrieve A and B.
                 let lhs_node = &tape.nodes[*lhs_idx];
                 let rhs_node = &tape.nodes[*rhs_idx];
                 
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
                
                if let (Some(input_val), Some(_output_val)) = (&input_node.cached_value, &current_node.cached_value) {
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
            Operation::MaxPool2d { input, kernel_size: _, stride: _, padding: _, indices } => {
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
            Operation::Embedding { table, indices, embedding_dim } => {
                // Embedding backward: gradient only flows to the rows that were looked up.
                // grad_output shape: (seq_len, embedding_dim)
                // grad_table shape: (vocab_size, embedding_dim) - sparse, only indexed rows get gradient
                let table_node = &tape.nodes[*table];
                if let Some(table_val) = &table_node.cached_value {
                    let vocab_size = table_val.shape()[0];
                    let emb_dim = *embedding_dim;
                    let mut grad_table_data = vec![BoundedValue::exact(0.0); vocab_size * emb_dim];

                    // For each position in the sequence, accumulate gradient into the
                    // corresponding embedding table row.
                    let grad_data = grad_output.data();
                    for (pos, &token_id) in indices.iter().enumerate() {
                        if token_id < vocab_size {
                            for j in 0..emb_dim {
                                let src_idx = pos * emb_dim + j;
                                let dst_idx = token_id * emb_dim + j;
                                if src_idx < grad_data.len() {
                                    let existing = grad_table_data[dst_idx];
                                    let incoming = grad_data[src_idx];
                                    grad_table_data[dst_idx] = BoundedValue::new(
                                        existing.value() + incoming.value(),
                                        helix_core::types::ErrorMargin::absolute(
                                            existing.absolute_error() + incoming.absolute_error(),
                                        ),
                                    );
                                }
                            }
                        }
                    }

                    let grad_table = BoundedTensor::new(
                        grad_table_data,
                        vec![vocab_size, emb_dim],
                    );
                    accumulate_grad(&mut grads, *table, &grad_table);
                }
            }
            Operation::Tanh(input_idx) => {
                // y = tanh(x)
                // dL/dx = dL/dy * (1 - tanh^2(x)) = dL/dy * (1 - y^2)
                let current_node = &tape.nodes[idx];
                if let Some(output_val) = &current_node.cached_value {
                    let grad_input_data: Vec<BoundedValue<f64>> = output_val.data()
                        .iter()
                        .zip(grad_output.data().iter())
                        .map(|(y, g)| {
                            let y_val = y.value();
                            let derivative = 1.0 - y_val * y_val;
                            let grad_val = g.value() * derivative;
                            let error = g.absolute_error() * derivative.abs()
                                + g.value().abs() * 2.0 * y_val.abs() * y.absolute_error();
                            BoundedValue::new(grad_val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();

                    let grad_input = BoundedTensor::new(grad_input_data, output_val.shape().clone());
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::LeakyRelu(input_idx, alpha) => {
                // y = x if x > 0, alpha * x if x <= 0
                // dL/dx = dL/dy * (1 if x > 0, alpha if x <= 0)
                let input_node = &tape.nodes[*input_idx];
                if let Some(input_val) = &input_node.cached_value {
                    let grad_input_data: Vec<BoundedValue<f64>> = input_val.data()
                        .iter()
                        .zip(grad_output.data().iter())
                        .map(|(x, g)| {
                            let slope = if x.value() > 0.0 { 1.0 } else { *alpha };
                            let grad_val = g.value() * slope;
                            let error = g.absolute_error() * slope.abs();
                            BoundedValue::new(grad_val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();

                    let grad_input = BoundedTensor::new(grad_input_data, input_val.shape().clone());
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::Gelu(input_idx) => {
                // GELU(x) ~ x * sigmoid(1.702 * x)
                // d/dx GELU(x) ~ sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
                let input_node = &tape.nodes[*input_idx];
                if let Some(input_val) = &input_node.cached_value {
                    let grad_input_data: Vec<BoundedValue<f64>> = input_val.data()
                        .iter()
                        .zip(grad_output.data().iter())
                        .map(|(x_bv, g)| {
                            let x = x_bv.value();
                            let sig_arg = 1.702 * x;
                            let sig = 1.0 / (1.0 + (-sig_arg).exp());
                            let gelu_deriv = sig + x * 1.702 * sig * (1.0 - sig);
                            let grad_val = g.value() * gelu_deriv;
                            // Error propagation: chain rule with both input error and gradient error
                            let error = g.absolute_error() * gelu_deriv.abs()
                                + g.value().abs() * x_bv.absolute_error() * 1.702;
                            BoundedValue::new(grad_val, helix_core::types::ErrorMargin::absolute(error))
                        })
                        .collect();

                    let grad_input = BoundedTensor::new(grad_input_data, input_val.shape().clone());
                    accumulate_grad(&mut grads, *input_idx, &grad_input);
                }
            }
            Operation::Attention { query, key, value, num_heads: _ } => {
                // Simplified multi-head attention backward.
                // Forward: scores = Q @ K^T / sqrt(d_k), attn = softmax(scores), out = attn @ V
                //
                // Backward:
                //   dL/dV = attn^T @ grad_output
                //   dL/d_attn = grad_output @ V^T
                //   dL/d_scores = softmax_backward(dL/d_attn, attn)
                //   dL/dQ = dL/d_scores @ K / sqrt(d_k)
                //   dL/dK = dL/d_scores^T @ Q / sqrt(d_k)
                let q_node = &tape.nodes[*query];
                let k_node = &tape.nodes[*key];
                let v_node = &tape.nodes[*value];
                let precision = Precision::F32;

                if let (Some(q_val), Some(k_val), Some(v_val)) =
                    (&q_node.cached_value, &k_node.cached_value, &v_node.cached_value)
                {
                    let d_k = if q_val.is_matrix() {
                        q_val.shape()[1]
                    } else {
                        q_val.shape().last().copied().unwrap_or(1)
                    };
                    let scale = 1.0 / (d_k as f64).sqrt();

                    // Recompute attention weights
                    let k_t = k_val.transpose();
                    if let Ok(scores) = matmul::matmul(q_val, &k_t, precision) {
                        let scaled_scores = scores.scale(BoundedValue::exact(scale));

                        // Row-wise softmax for attention weights
                        let attn_weights = softmax_rows_backward_helper(&scaled_scores);

                        // dL/dV = attn^T @ grad_output
                        let attn_t = attn_weights.transpose();
                        if let Ok(grad_v) = matmul::matmul(&attn_t, &grad_output, precision) {
                            accumulate_grad(&mut grads, *value, &grad_v);
                        }

                        // dL/d_attn = grad_output @ V^T
                        let v_t = v_val.transpose();
                        if let Ok(grad_attn) = matmul::matmul(&grad_output, &v_t, precision) {
                            // dL/d_scores = softmax_backward(dL/d_attn, attn_weights)
                            let grad_scores = softmax_backward_2d(&grad_attn, &attn_weights);

                            // dL/dQ = grad_scores @ K * scale
                            if let Ok(grad_q) = matmul::matmul(&grad_scores, k_val, precision) {
                                let grad_q = grad_q.scale(BoundedValue::exact(scale));
                                accumulate_grad(&mut grads, *query, &grad_q);
                            }

                            // dL/dK = grad_scores^T @ Q * scale
                            let gs_t = grad_scores.transpose();
                            if let Ok(grad_k) = matmul::matmul(&gs_t, q_val, precision) {
                                let grad_k = grad_k.scale(BoundedValue::exact(scale));
                                accumulate_grad(&mut grads, *key, &grad_k);
                            }
                        }
                    }
                }
            }
            Operation::MLP { input, weights1, bias1, weights2, bias2 } => {
                // MLP forward: hidden = relu(x @ W1^T + b1), output = hidden @ W2^T + b2
                // Backward reverses through W2, activation, then W1.
                let input_node = &tape.nodes[*input];
                let w1_node = &tape.nodes[*weights1];
                let w2_node = &tape.nodes[*weights2];
                let precision = Precision::F32;

                if let (Some(input_val), Some(w1_val), Some(w2_val)) =
                    (&input_node.cached_value, &w1_node.cached_value, &w2_node.cached_value)
                {
                    // Recompute hidden activations
                    let w1_t = w1_val.transpose();
                    if let Ok(mut pre_act) = matmul::matmul(input_val, &w1_t, precision) {
                        if let Some(b1_idx) = bias1 {
                            if let Some(b1_val) = &tape.nodes[*b1_idx].cached_value {
                                pre_act = super::autodiff::broadcast_add_bias(&pre_act, b1_val);
                            }
                        }

                        // ReLU activation
                        let activated = crate::ops::relu(&pre_act);

                        // --- Backward through W2 ---
                        // dL/d_activated = grad_output @ W2
                        if let Ok(grad_activated) = matmul::matmul(&grad_output, w2_val, precision) {
                            // dL/d_W2 = grad_output^T @ activated
                            let go_t = grad_output.transpose();
                            if let Ok(grad_w2) = matmul::matmul(&go_t, &activated, precision) {
                                accumulate_grad(&mut grads, *weights2, &grad_w2);
                            }

                            // dL/d_b2 = sum(grad_output) over batch
                            if let Some(b2_idx) = bias2 {
                                let grad_b2 = sum_rows(&grad_output);
                                accumulate_grad(&mut grads, *b2_idx, &grad_b2);
                            }

                            // --- Backward through ReLU ---
                            let relu_mask_data: Vec<BoundedValue<f64>> = pre_act.data()
                                .iter()
                                .map(|v| {
                                    if v.value() > 0.0 {
                                        BoundedValue::exact(1.0)
                                    } else {
                                        BoundedValue::exact(0.0)
                                    }
                                })
                                .collect();
                            let relu_mask = BoundedTensor::new(relu_mask_data, pre_act.shape().clone());
                            let grad_pre_act = grad_activated.hadamard(&relu_mask);

                            // --- Backward through W1 ---
                            // dL/d_input = grad_pre_act @ W1
                            if let Ok(grad_input) = matmul::matmul(&grad_pre_act, w1_val, precision) {
                                accumulate_grad(&mut grads, *input, &grad_input);
                            }

                            // dL/d_W1 = grad_pre_act^T @ input
                            let gpa_t = grad_pre_act.transpose();
                            if let Ok(grad_w1) = matmul::matmul(&gpa_t, input_val, precision) {
                                accumulate_grad(&mut grads, *weights1, &grad_w1);
                            }

                            // dL/d_b1 = sum(grad_pre_act) over batch
                            if let Some(b1_idx) = bias1 {
                                let grad_b1 = sum_rows(&grad_pre_act);
                                accumulate_grad(&mut grads, *b1_idx, &grad_b1);
                            }
                        }
                    }
                }
            }
            Operation::GroupedConv2d { input, kernel, stride, padding, groups: _ } => {
                // Grouped convolution backward: same structure as Conv2d backward.
                // The conv2d_backward_input and conv2d_backward_weight functions handle
                // the group structure internally through the kernel shape.
                let input_node = &tape.nodes[*input];
                let kernel_node = &tape.nodes[*kernel];
                let precision = Precision::F32;

                if let (Some(input_val), Some(kernel_val)) = (&input_node.cached_value, &kernel_node.cached_value) {
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

/// Row-wise softmax helper for recomputing attention weights during backward.
fn softmax_rows_backward_helper(input: &BoundedTensor) -> BoundedTensor {
    if !input.is_matrix() {
        return input.clone();
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.data();
    let mut result = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        let row_start = i * cols;
        let row_max = (0..cols)
            .map(|j| data[row_start + j].value())
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_vals: Vec<f64> = (0..cols)
            .map(|j| (data[row_start + j].value() - row_max).exp())
            .collect();
        let sum: f64 = exp_vals.iter().sum();

        for exp_val in &exp_vals {
            result.push(BoundedValue::exact(exp_val / sum));
        }
    }

    BoundedTensor::new(result, input.shape().clone())
}

/// Softmax backward for a 2D tensor (applied row-wise).
///
/// For each row: dL/d_scores_i = y_i * (dL/d_y_i - sum_j(dL/d_y_j * y_j))
fn softmax_backward_2d(
    grad_output: &BoundedTensor,
    softmax_output: &BoundedTensor,
) -> BoundedTensor {
    if !grad_output.is_matrix() || !softmax_output.is_matrix() {
        return grad_output.clone();
    }

    let rows = grad_output.shape()[0];
    let cols = grad_output.shape()[1];
    let grad_data = grad_output.data();
    let sm_data = softmax_output.data();
    let mut result = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        let row_start = i * cols;
        let weighted_sum: f64 = (0..cols)
            .map(|j| grad_data[row_start + j].value() * sm_data[row_start + j].value())
            .sum();

        for j in 0..cols {
            let y = sm_data[row_start + j].value();
            let g = grad_data[row_start + j].value();
            let grad_val = y * (g - weighted_sum);
            let error = sm_data[row_start + j].absolute_error() * (g - weighted_sum).abs()
                + grad_data[row_start + j].absolute_error() * y;
            result.push(BoundedValue::new(
                grad_val,
                helix_core::types::ErrorMargin::absolute(error),
            ));
        }
    }

    BoundedTensor::new(result, grad_output.shape().clone())
}

/// Sums a 2D tensor over the row dimension, returning a 1D vector.
fn sum_rows(tensor: &BoundedTensor) -> BoundedTensor {
    if !tensor.is_matrix() {
        return tensor.clone();
    }
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let data = tensor.data();
    let mut result = vec![BoundedValue::exact(0.0); cols];

    for i in 0..rows {
        for j in 0..cols {
            let val = data[i * cols + j];
            let existing = result[j];
            result[j] = BoundedValue::new(
                existing.value() + val.value(),
                helix_core::types::ErrorMargin::absolute(
                    existing.absolute_error() + val.absolute_error(),
                ),
            );
        }
    }

    BoundedTensor::new(result, vec![cols])
}

/// Helper to accumulate gradients (summing them if multiple paths lead to same node).
fn accumulate_grad(grads: &mut HashMap<NodeIndex, BoundedTensor>, idx: NodeIndex, grad: &BoundedTensor) {
    if let Some(existing) = grads.get_mut(&idx) {
        *existing = existing.add(grad);
    } else {
        grads.insert(idx, grad.clone());
    }
}
