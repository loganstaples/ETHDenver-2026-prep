//! Basic element-wise tensor operations.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin};
use thiserror::Error;

/// Errors from basic operations.
#[derive(Error, Debug)]
pub enum BasicOpError {
    #[error("Shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),

    #[error("Division by zero detected")]
    DivisionByZero,
}

/// Element-wise addition of two tensors.
pub fn add(a: &BoundedTensor, b: &BoundedTensor) -> Result<BoundedTensor, BasicOpError> {
    if a.shape() != b.shape() {
        return Err(BasicOpError::ShapeMismatch(
            a.shape().clone(),
            b.shape().clone(),
        ));
    }
    Ok(a.add(b))
}

/// Element-wise subtraction of two tensors.
pub fn sub(a: &BoundedTensor, b: &BoundedTensor) -> Result<BoundedTensor, BasicOpError> {
    if a.shape() != b.shape() {
        return Err(BasicOpError::ShapeMismatch(
            a.shape().clone(),
            b.shape().clone(),
        ));
    }
    Ok(a.sub(b))
}

/// Element-wise multiplication of two tensors.
pub fn mul(a: &BoundedTensor, b: &BoundedTensor) -> Result<BoundedTensor, BasicOpError> {
    if a.shape() != b.shape() {
        return Err(BasicOpError::ShapeMismatch(
            a.shape().clone(),
            b.shape().clone(),
        ));
    }
    Ok(a.hadamard(b))
}

/// Element-wise negation.
pub fn neg(a: &BoundedTensor) -> BoundedTensor {
    let data: Vec<_> = a
        .data()
        .iter()
        .map(|v| BoundedValue::new(-v.value(), v.error()))
        .collect();
    BoundedTensor::new(data, a.shape().clone())
}

/// Element-wise division with near-zero clamping.
pub fn div(a: &BoundedTensor, b: &BoundedTensor) -> Result<BoundedTensor, BasicOpError> {
    if a.shape() != b.shape() {
        return Err(BasicOpError::ShapeMismatch(
            a.shape().clone(),
            b.shape().clone(),
        ));
    }

    const NEAR_ZERO: f64 = 1e-12;
    let data: Vec<_> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(av, bv)| {
            let denom = bv.value();
            let clamped_denom = if denom.abs() < NEAR_ZERO {
                if denom >= 0.0 { NEAR_ZERO } else { -NEAR_ZERO }
            } else {
                denom
            };
            let val = av.value() / clamped_denom;
            // Error: |a_err/b| + |a * b_err / b^2|
            let error = av.absolute_error() / clamped_denom.abs()
                + av.value().abs() * bv.absolute_error() / (clamped_denom * clamped_denom);
            BoundedValue::new(val, ErrorMargin::absolute(error))
        })
        .collect();
    Ok(BoundedTensor::new(data, a.shape().clone()))
}

/// Scalar multiplication.
pub fn scale(a: &BoundedTensor, scalar: f64) -> BoundedTensor {
    let scalar_bounded = BoundedValue::<f64>::exact(scalar);
    a.scale(scalar_bounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let b = BoundedTensor::from_exact(vec![4.0, 5.0, 6.0], vec![3]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.values(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = BoundedTensor::zeros(vec![2, 3]);
        let b = BoundedTensor::zeros(vec![3, 2]);
        assert!(add(&a, &b).is_err());
    }

    #[test]
    fn test_neg() {
        let a = BoundedTensor::from_exact(vec![1.0, -2.0, 3.0], vec![3]);
        let b = neg(&a);
        assert_eq!(b.values(), vec![-1.0, 2.0, -3.0]);
    }
}
