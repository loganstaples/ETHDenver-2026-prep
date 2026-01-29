//! Backward pass implementation for automatic differentiation.

use super::autodiff::{GradientTape, NodeIndex, Operation, Variable};
use crate::ops::matmul;
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
                    let zero = BoundedValue::exact(0.0);
                    
                    // Create mask manually since we don't have a generic map on BoundedTensor exposed yet
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
            _ => {
                // Implement other ops
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
