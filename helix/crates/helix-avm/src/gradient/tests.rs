use crate::gradient::autodiff::{GradientTape, Variable};
use crate::gradient::backward::backward;
use helix_core::types::{BoundedTensor, BoundedValue};
use std::rc::Rc;

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
    // If y = Ax, then L = sum(Ax)
    // dL/dA_ij = x_j
    // dL/dx_i = sum(A_ji)
    
    // To implement sum(y), let's cheat and multiply by ones 1x2 since we don't have sum in Variable yet
    // L = [[1, 1]] @ y  (1x2 @ 2x1 -> 1x1 scalar)
    let ones_val = BoundedTensor::from_exact(vec![1.0, 1.0], vec![1, 2]);
    let ones = Variable::input(ones_val, tape.clone(), Some("ones".into()));
    
    let loss = ones.matmul(&y);
    
    let grads = backward(&loss).unwrap();
    let da = grads.get(&a.node_index.unwrap()).unwrap();
    
    // dL/dA should be outer product of (dL/dy)^T and x^T ?
    // L = 1^T y = 1^T A x
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
    
    // L = sum(y) implies gradient of 1.0 coming back
    // dL/dx = [0, 1]
    
    // Again, simulate sum with dot product with ones if strictly needed, 
    // but backward() currently assumes scalar loss and propagates 1.0.
    // However, y is vector (size 2). backward() expects scalar.
    // We added a basic check in backward.rs: "For now, we only support scalar loss".
    // "We assume scalar for standard ML training."
    // So we need to reduce y to scalar.
    // L = y.matmult(ones) or similar? 
    // Variable doesn't have `dot` exposed.
    // But we know backward() initializes with 1.0 tensor of the SAME SHAPE as loss.
    // Wait, my impl of backward:
    /*
    let seed_grad = BoundedTensor::full(
        loss.tensor.shape().clone(), 
        BoundedValue::exact(1.0)
    );
     */
    // This implies element-wise loss gradient is 1.0 everywhere if we treat "loss" variable as the root 
    // even if it's not strictly scalar 1x1. Vector-Jacobian product where v=[1,1,...].
    // So if I pass `y` directly to `backward`, it seeds with [1.0, 1.0].
    // So dL/dx should be [1.0 * (x>0), 1.0 * (x>0)] = [0.0, 1.0].
    
    let grads = backward(&y).unwrap();
    let dx = grads.get(&x.node_index.unwrap()).unwrap();
    
    let dx_vals = dx.values();
    assert!((dx_vals[0] - 0.0).abs() < 1e-6);
    assert!((dx_vals[1] - 1.0).abs() < 1e-6);
}
