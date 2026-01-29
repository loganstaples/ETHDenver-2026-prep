//! Trait for approximate operations.
//!
//! Operations that produce bounded results with tracked error.

use crate::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};

/// Trait for operations that produce approximate (bounded) results.
pub trait ApproximateOp {
    /// The input type for this operation.
    type Input;
    /// The output type for this operation.
    type Output;

    /// Executes the operation with the specified precision.
    fn execute(&self, input: Self::Input, precision: Precision) -> Self::Output;

    /// Returns the maximum possible error for this operation.
    fn max_error(&self, input: &Self::Input, precision: Precision) -> ErrorMargin;
}

/// Trait for element-wise approximate operations on scalars.
pub trait ApproximateScalarOp {
    /// Applies the operation to a bounded value.
    fn apply(&self, value: BoundedValue<f64>, precision: Precision) -> BoundedValue<f64>;
}

/// Trait for approximate tensor operations.
pub trait ApproximateTensorOp {
    /// Applies the operation to a bounded tensor.
    fn apply(&self, tensor: &BoundedTensor, precision: Precision) -> BoundedTensor;

    /// Returns the expected output shape given input shape.
    fn output_shape(&self, input_shape: &[usize]) -> Vec<usize>;
}

/// Trait for binary approximate operations (two inputs).
pub trait ApproximateBinaryOp {
    /// The left operand type.
    type Left;
    /// The right operand type.
    type Right;
    /// The output type.
    type Output;

    /// Executes the binary operation.
    fn execute(&self, left: Self::Left, right: Self::Right, precision: Precision) -> Self::Output;
}
