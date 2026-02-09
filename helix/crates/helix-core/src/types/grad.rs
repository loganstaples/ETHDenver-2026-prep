//! Backward-pass (gradient) support for BoundedTensor.
//!
//! Provides `GradTensor`, a wrapper around `BoundedTensor` that supports
//! automatic differentiation for training. Implements backward for:
//! - `matmul`: dA = dC * B^T, dB = A^T * dC
//! - `relu`: mask * upstream (where mask = 1 if x > 0, else 0)
//! - `add`: pass-through (dA = dC, dB = dC)
//! - `hadamard` (element-wise mul): dA = dC * B, dB = dC * A
//!
//! This enables helix-avm to use core tensors for training without
//! reimplementing gradient logic.

use super::tensor::BoundedTensor;
use super::bounded_value::BoundedValue;
use super::error_margin::ErrorMargin;
use crate::error::{HelixResult, ValidationError};

/// A tensor that tracks whether gradients should be computed.
///
/// `GradTensor` wraps a `BoundedTensor` with optional gradient storage.
/// Gradients are accumulated via the `backward_*` static methods, which
/// compute upstream gradient contributions for each operand.
///
/// # Example
///
/// ```
/// use helix_core::types::grad::GradTensor;
/// use helix_core::types::BoundedTensor;
///
/// // Create tensors for a simple matmul forward + backward pass
/// let a = GradTensor::with_grad(
///     BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
/// );
/// let b = GradTensor::with_grad(
///     BoundedTensor::from_exact(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])
/// );
///
/// // Forward pass
/// let c = a.data().matmul(b.data()).unwrap();
///
/// // Backward pass: given dC (upstream gradient), compute dA and dB
/// let dc = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
/// let (da, db) = GradTensor::backward_matmul(a.data(), b.data(), &dc).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GradTensor {
    /// The forward-pass tensor data.
    data: BoundedTensor,
    /// Accumulated gradient (same shape as data). None if requires_grad is false.
    grad: Option<BoundedTensor>,
    /// Whether this tensor requires gradient computation.
    requires_grad: bool,
}

impl GradTensor {
    /// Creates a GradTensor that does not require gradients.
    pub fn no_grad(data: BoundedTensor) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: false,
        }
    }

    /// Creates a GradTensor that requires gradient computation.
    /// Gradient is initialized to zeros with the same shape.
    pub fn with_grad(data: BoundedTensor) -> Self {
        let grad = BoundedTensor::zeros(data.shape().clone());
        Self {
            data,
            grad: Some(grad),
            requires_grad: true,
        }
    }

    /// Returns the forward-pass tensor data.
    pub fn data(&self) -> &BoundedTensor {
        &self.data
    }

    /// Returns the accumulated gradient, if this tensor requires grad.
    pub fn grad(&self) -> Option<&BoundedTensor> {
        self.grad.as_ref()
    }

    /// Returns whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Zeros out the accumulated gradient.
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(BoundedTensor::zeros(self.data.shape().clone()));
        }
    }

    /// Accumulates an upstream gradient into this tensor's grad buffer.
    ///
    /// If this tensor doesn't require grad, this is a no-op.
    pub fn accumulate_grad(&mut self, upstream: &BoundedTensor) -> HelixResult<()> {
        if !self.requires_grad {
            return Ok(());
        }

        if upstream.shape() != self.data.shape() {
            return Err(ValidationError::invalid_shape(
                "gradient",
                upstream.shape().clone(),
                format!("gradient shape must match tensor shape {:?}", self.data.shape()),
            ).into());
        }

        if let Some(ref mut grad) = self.grad {
            *grad = grad.try_add(upstream)?;
        }
        Ok(())
    }

    // =========================================================================
    // BACKWARD IMPLEMENTATIONS (static methods)
    // =========================================================================

    /// Backward pass for matrix multiplication: C = A @ B.
    ///
    /// Given upstream gradient dC (shape [M, N]):
    /// - dA = dC @ B^T (shape [M, K])
    /// - dB = A^T @ dC (shape [K, N])
    pub fn backward_matmul(
        a: &BoundedTensor,
        b: &BoundedTensor,
        dc: &BoundedTensor,
    ) -> HelixResult<(BoundedTensor, BoundedTensor)> {
        // dA = dC @ B^T
        let bt = b.try_transpose()?;
        let da = dc.matmul(&bt)?;

        // dB = A^T @ dC
        let at = a.try_transpose()?;
        let db = at.matmul(dc)?;

        Ok((da, db))
    }

    /// Backward pass for ReLU activation: y = max(0, x).
    ///
    /// Given upstream gradient dy and the original input x:
    /// - dx = dy * mask, where mask[i] = 1 if x[i] > 0, else 0
    pub fn backward_relu(
        input: &BoundedTensor,
        upstream: &BoundedTensor,
    ) -> HelixResult<BoundedTensor> {
        if input.shape() != upstream.shape() {
            return Err(ValidationError::invalid_shape(
                "relu backward",
                upstream.shape().clone(),
                format!("upstream shape must match input shape {:?}", input.shape()),
            ).into());
        }

        let data: Vec<BoundedValue<f64>> = input.data()
            .iter()
            .zip(upstream.data().iter())
            .map(|(x, dy)| {
                if x.value() > 0.0 {
                    *dy // Pass gradient through
                } else {
                    BoundedValue::exact(0.0) // Kill gradient
                }
            })
            .collect();

        BoundedTensor::try_new(data, input.shape().clone())
    }

    /// Backward pass for element-wise addition: C = A + B.
    ///
    /// Given upstream gradient dC:
    /// - dA = dC (pass-through)
    /// - dB = dC (pass-through)
    pub fn backward_add(
        dc: &BoundedTensor,
    ) -> (BoundedTensor, BoundedTensor) {
        (dc.clone(), dc.clone())
    }

    /// Backward pass for element-wise multiplication (Hadamard): C = A * B.
    ///
    /// Given upstream gradient dC:
    /// - dA = dC * B (element-wise)
    /// - dB = dC * A (element-wise)
    pub fn backward_hadamard(
        a: &BoundedTensor,
        b: &BoundedTensor,
        dc: &BoundedTensor,
    ) -> HelixResult<(BoundedTensor, BoundedTensor)> {
        // dA = dC * B
        let da = dc.try_hadamard(b)?;

        // dB = dC * A
        let db = dc.try_hadamard(a)?;

        Ok((da, db))
    }

    /// Backward pass for softmax: s = softmax(x).
    ///
    /// Given upstream gradient ds and the softmax output s:
    /// - dx_i = s_i * (ds_i - sum_j(ds_j * s_j))
    ///
    /// This is the standard softmax backward (Jacobian-vector product).
    pub fn backward_softmax(
        softmax_output: &BoundedTensor,
        upstream: &BoundedTensor,
    ) -> HelixResult<BoundedTensor> {
        if softmax_output.shape() != upstream.shape() {
            return Err(ValidationError::invalid_shape(
                "softmax backward",
                upstream.shape().clone(),
                format!("upstream shape must match softmax output shape {:?}", softmax_output.shape()),
            ).into());
        }

        match softmax_output.ndim() {
            1 => {
                // dot = sum(ds * s)
                let dot: f64 = softmax_output.data().iter()
                    .zip(upstream.data().iter())
                    .map(|(s, ds)| s.value() * ds.value())
                    .sum();

                let data: Vec<BoundedValue<f64>> = softmax_output.data().iter()
                    .zip(upstream.data().iter())
                    .map(|(s, ds)| {
                        let val = s.value() * (ds.value() - dot);
                        let err = s.absolute_error() * (ds.absolute_error() + dot.abs());
                        BoundedValue::new(val, ErrorMargin::absolute(err))
                    })
                    .collect();

                BoundedTensor::try_new(data, softmax_output.shape().clone())
            }
            2 => {
                let (rows, cols) = (softmax_output.shape()[0], softmax_output.shape()[1]);
                let mut data = Vec::with_capacity(rows * cols);

                for i in 0..rows {
                    // dot = sum_j(ds[i,j] * s[i,j])
                    let dot: f64 = (0..cols)
                        .map(|j| {
                            let idx = i * cols + j;
                            softmax_output.data()[idx].value() * upstream.data()[idx].value()
                        })
                        .sum();

                    for j in 0..cols {
                        let idx = i * cols + j;
                        let s = softmax_output.data()[idx];
                        let ds = upstream.data()[idx];
                        let val = s.value() * (ds.value() - dot);
                        let err = s.absolute_error() * (ds.absolute_error() + dot.abs());
                        data.push(BoundedValue::new(val, ErrorMargin::absolute(err)));
                    }
                }

                BoundedTensor::try_new(data, softmax_output.shape().clone())
            }
            _ => Err(ValidationError::invalid_shape(
                "softmax backward",
                softmax_output.shape().clone(),
                "softmax backward requires 1D or 2D tensor",
            ).into()),
        }
    }

    /// Backward pass for scalar multiplication: Y = scalar * X.
    ///
    /// Given upstream gradient dY:
    /// - dX = scalar * dY
    pub fn backward_scale(
        scalar: BoundedValue<f64>,
        upstream: &BoundedTensor,
    ) -> BoundedTensor {
        upstream.scale(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_tensor_creation() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let gt = GradTensor::with_grad(t);
        assert!(gt.requires_grad());
        assert!(gt.grad().is_some());
        assert_eq!(gt.grad().unwrap().shape(), &vec![2, 2]);

        // Gradient should be zeros initially
        for v in gt.grad().unwrap().data() {
            assert_eq!(v.value(), 0.0);
        }
    }

    #[test]
    fn test_no_grad_tensor() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let gt = GradTensor::no_grad(t);
        assert!(!gt.requires_grad());
        assert!(gt.grad().is_none());
    }

    #[test]
    fn test_accumulate_grad() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut gt = GradTensor::with_grad(t);

        let g1 = BoundedTensor::from_exact(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        gt.accumulate_grad(&g1).unwrap();

        let g2 = BoundedTensor::from_exact(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        gt.accumulate_grad(&g2).unwrap();

        let grad = gt.grad().unwrap();
        assert!((grad.get(&[0, 0]).unwrap().value() - 0.6).abs() < 1e-10);
        assert!((grad.get(&[0, 1]).unwrap().value() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_zero_grad() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let mut gt = GradTensor::with_grad(t);

        let g = BoundedTensor::from_exact(vec![5.0, 6.0], vec![2]);
        gt.accumulate_grad(&g).unwrap();
        assert_eq!(gt.grad().unwrap().get(&[0]).unwrap().value(), 5.0);

        gt.zero_grad();
        assert_eq!(gt.grad().unwrap().get(&[0]).unwrap().value(), 0.0);
    }

    #[test]
    fn test_backward_matmul() {
        // A: [2, 3], B: [3, 2] -> C: [2, 2]
        let a = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let b = BoundedTensor::from_exact(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
        );

        // dC = ones([2, 2])
        let dc = BoundedTensor::from_exact(vec![1.0; 4], vec![2, 2]);

        let (da, db) = GradTensor::backward_matmul(&a, &b, &dc).unwrap();

        // dA = dC @ B^T, shape [2, 3]
        assert_eq!(da.shape(), &vec![2, 3]);
        // dA[0,0] = dC[0,0]*B[0,0] + dC[0,1]*B[1,0] = 1*7 + 1*8 = 15
        // Wait, B^T is [2, 3], so B^T[j,k] = B[k,j]
        // dA[0,0] = sum_j dC[0,j] * B^T[j,0] = dC[0,0]*B[0,0] + dC[0,1]*B[0,1]
        //         = 1*7 + 1*8 = 15
        assert!((da.get(&[0, 0]).unwrap().value() - 15.0).abs() < 1e-10);
        // dA[0,1] = dC[0,0]*B[1,0] + dC[0,1]*B[1,1] = 1*9 + 1*10 = 19
        assert!((da.get(&[0, 1]).unwrap().value() - 19.0).abs() < 1e-10);

        // dB = A^T @ dC, shape [3, 2]
        assert_eq!(db.shape(), &vec![3, 2]);
        // dB[0,0] = A^T[0,0]*dC[0,0] + A^T[0,1]*dC[1,0] = A[0,0]*1 + A[1,0]*1 = 1+4 = 5
        assert!((db.get(&[0, 0]).unwrap().value() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_relu() {
        let input = BoundedTensor::from_exact(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let upstream = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);

        let dx = GradTensor::backward_relu(&input, &upstream).unwrap();

        // Gradient killed where input <= 0
        assert_eq!(dx.get(&[0]).unwrap().value(), 0.0); // -2.0 <= 0
        assert_eq!(dx.get(&[1]).unwrap().value(), 0.0); // -1.0 <= 0
        assert_eq!(dx.get(&[2]).unwrap().value(), 0.0); //  0.0 <= 0
        // Gradient passed through where input > 0
        assert_eq!(dx.get(&[3]).unwrap().value(), 4.0); //  1.0 > 0
        assert_eq!(dx.get(&[4]).unwrap().value(), 5.0); //  2.0 > 0
    }

    #[test]
    fn test_backward_add() {
        let dc = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let (da, db) = GradTensor::backward_add(&dc);

        // Both gradients should be copies of dc
        for i in 0..3 {
            assert_eq!(da.get(&[i]).unwrap().value(), dc.get(&[i]).unwrap().value());
            assert_eq!(db.get(&[i]).unwrap().value(), dc.get(&[i]).unwrap().value());
        }
    }

    #[test]
    fn test_backward_hadamard() {
        let a = BoundedTensor::from_exact(vec![2.0, 3.0, 4.0], vec![3]);
        let b = BoundedTensor::from_exact(vec![5.0, 6.0, 7.0], vec![3]);
        let dc = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0], vec![3]);

        let (da, db) = GradTensor::backward_hadamard(&a, &b, &dc).unwrap();

        // dA = dC * B -> [5, 6, 7]
        assert!((da.get(&[0]).unwrap().value() - 5.0).abs() < 1e-10);
        assert!((da.get(&[1]).unwrap().value() - 6.0).abs() < 1e-10);
        assert!((da.get(&[2]).unwrap().value() - 7.0).abs() < 1e-10);

        // dB = dC * A -> [2, 3, 4]
        assert!((db.get(&[0]).unwrap().value() - 2.0).abs() < 1e-10);
        assert!((db.get(&[1]).unwrap().value() - 3.0).abs() < 1e-10);
        assert!((db.get(&[2]).unwrap().value() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_softmax() {
        let x = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let s = x.softmax().unwrap();

        // Upstream gradient = ones
        let ds = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0], vec![3]);
        let dx = GradTensor::backward_softmax(&s, &ds).unwrap();

        // When upstream is uniform ones, softmax backward should give ~zeros
        // because sum(ds * s) = sum(s) = 1, so dx_i = s_i * (1 - 1) = 0
        for v in dx.data() {
            assert!(v.value().abs() < 1e-10, "expected ~0, got {}", v.value());
        }
    }

    #[test]
    fn test_backward_softmax_non_uniform() {
        let x = BoundedTensor::from_exact(vec![0.0, 0.0, 0.0], vec![3]);
        let s = x.softmax().unwrap();
        // s = [1/3, 1/3, 1/3]

        // ds = [1, 0, 0] — gradient only on first element
        let ds = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0], vec![3]);
        let dx = GradTensor::backward_softmax(&s, &ds).unwrap();

        // dx_0 = s_0 * (ds_0 - dot) = 1/3 * (1 - 1/3) = 1/3 * 2/3 = 2/9
        // dx_1 = s_1 * (ds_1 - dot) = 1/3 * (0 - 1/3) = -1/9
        // dx_2 = s_2 * (ds_2 - dot) = 1/3 * (0 - 1/3) = -1/9
        assert!((dx.get(&[0]).unwrap().value() - 2.0/9.0).abs() < 1e-10);
        assert!((dx.get(&[1]).unwrap().value() + 1.0/9.0).abs() < 1e-10);
        assert!((dx.get(&[2]).unwrap().value() + 1.0/9.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_forward_backward_pass() {
        // Simple 2-layer forward + backward:
        // x -> matmul(W1) -> relu -> matmul(W2) -> loss

        // Input: [1, 2] (row vector)
        let x = BoundedTensor::from_exact(vec![1.0, 2.0], vec![1, 2]);

        // W1: [2, 3]
        let w1 = BoundedTensor::from_exact(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![2, 3],
        );

        // W2: [3, 1]
        let w2 = BoundedTensor::from_exact(vec![0.7, 0.8, 0.9], vec![3, 1]);

        // Forward: h = x @ W1
        let h = x.matmul(&w1).unwrap();
        assert_eq!(h.shape(), &vec![1, 3]);

        // Forward: a = relu(h)
        let a = h.relu();

        // Forward: y = a @ W2
        let y = a.matmul(&w2).unwrap();
        assert_eq!(y.shape(), &vec![1, 1]);

        // Backward: dY = 1 (scalar loss gradient)
        let dy = BoundedTensor::from_exact(vec![1.0], vec![1, 1]);

        // Backward through matmul(a, W2):
        let (da, dw2) = GradTensor::backward_matmul(&a, &w2, &dy).unwrap();
        assert_eq!(da.shape(), &vec![1, 3]);
        assert_eq!(dw2.shape(), &vec![3, 1]);

        // Backward through relu(h):
        let dh = GradTensor::backward_relu(&h, &da).unwrap();
        assert_eq!(dh.shape(), &vec![1, 3]);

        // Backward through matmul(x, W1):
        let (_dx, dw1) = GradTensor::backward_matmul(&x, &w1, &dh).unwrap();
        assert_eq!(dw1.shape(), &vec![2, 3]);

        // Gradients should be non-zero where ReLU passes through
        // h = [0.9, 1.2, 1.5], all > 0, so relu passes all
        assert!(dw1.max_error() == 0.0); // Exact inputs -> exact gradients
    }

    #[test]
    fn test_backward_matmul_error_propagation() {
        let a = BoundedTensor::from_approximate(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            0.01,
        );
        let b = BoundedTensor::from_approximate(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            0.01,
        );
        let dc = BoundedTensor::from_exact(vec![1.0; 4], vec![2, 2]);

        let (da, db) = GradTensor::backward_matmul(&a, &b, &dc).unwrap();

        // Gradients should have propagated errors from a and b
        assert!(da.max_error() > 0.0);
        assert!(db.max_error() > 0.0);
    }
}
