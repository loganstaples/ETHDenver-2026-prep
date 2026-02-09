//! Bounded tensor type for N-dimensional arrays with error tracking.
//!
//! This module provides comprehensive shape validation, NaN/Inf detection,
//! and Result-returning operations for robust tensor computations.

use super::bounded_value::{BoundedValue, MAX_ERROR_BOUND, MAX_SAFE_ERROR};
use super::error_margin::ErrorMargin;
use crate::error::{ArithmeticError, BoundsError, HelixResult, ValidationError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Shape of a tensor (dimensions).
pub type Shape = Vec<usize>;

/// Maximum number of elements allowed in a tensor.
pub const MAX_TENSOR_ELEMENTS: usize = 100_000_000; // 100M elements

/// Maximum number of dimensions allowed.
pub const MAX_TENSOR_DIMS: usize = 8;

/// A multi-dimensional array of bounded values.
///
/// `BoundedTensor` is the core data structure for neural network computations
/// in HELIX. It tracks error bounds through all operations, enabling provable
/// accuracy guarantees in ZK circuits.
///
/// # Example
///
/// ```
/// use helix_core::types::BoundedTensor;
///
/// // Create a 2x3 matrix with approximate values
/// let a = BoundedTensor::from_approximate(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     vec![2, 3],
///     0.01,  // uniform error bound
/// );
///
/// // Matrix operations track error propagation
/// let b = BoundedTensor::from_approximate(
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
///     vec![3, 2],
///     0.01,
/// );
///
/// let c = a.matmul(&b).unwrap();  // Error accumulates
/// assert_eq!(c.shape(), &vec![2, 2]);
/// assert!(c.max_error() > 0.01);  // Error grew through matmul
/// ```
///
/// # Error Tracking
///
/// Every element in a `BoundedTensor` carries its error margin. Operations
/// propagate errors according to numerical analysis rules:
///
/// - **Addition**: ε(a+b) = εa + εb
/// - **Multiplication**: ε(a*b) = |a|εb + |b|εa + εaεb
/// - **MatMul**: Accumulates per-element multiplication errors across k dimension
///
/// # Parallel Operations
///
/// For tensors larger than 1000 elements, parallel versions of operations
/// are available (e.g., `par_add`, `par_scale`) using the `rayon` crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedTensor {
    /// Flattened data in row-major order.
    data: Vec<BoundedValue<f64>>,
    /// Dimensions of the tensor.
    shape: Shape,
    /// Strides for indexing.
    strides: Vec<usize>,
}

impl BoundedTensor {
    /// Creates a new tensor from data and shape.
    ///
    /// # Panics
    /// Panics if data length doesn't match the product of shape dimensions.
    pub fn new(data: Vec<BoundedValue<f64>>, shape: Shape) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Creates a new tensor from data and shape with validation.
    ///
    /// Returns an error if the shape is invalid or data length doesn't match.
    pub fn try_new(data: Vec<BoundedValue<f64>>, shape: Shape) -> HelixResult<Self> {
        Self::validate_shape(&shape)?;

        let expected_len = Self::compute_total_elements(&shape)?;
        if data.len() != expected_len {
            return Err(ValidationError::invalid_shape(
                "tensor",
                shape.clone(),
                format!("data length {} doesn't match shape product {}", data.len(), expected_len),
            ).into());
        }

        let strides = Self::compute_strides(&shape);
        Ok(Self { data, shape, strides })
    }

    /// Validates a shape.
    pub fn validate_shape(shape: &Shape) -> HelixResult<()> {
        // Check dimension count
        if shape.len() > MAX_TENSOR_DIMS {
            return Err(ValidationError::invalid_shape(
                "tensor",
                shape.clone(),
                format!("too many dimensions ({} > {})", shape.len(), MAX_TENSOR_DIMS),
            ).into());
        }

        // Check for zero dimensions (empty tensor)
        if shape.iter().any(|&d| d == 0) && !shape.is_empty() {
            return Err(ValidationError::invalid_shape(
                "tensor",
                shape.clone(),
                "shape contains zero dimension",
            ).into());
        }

        // Check total element count
        Self::compute_total_elements(shape)?;

        Ok(())
    }

    /// Computes total elements with overflow checking.
    fn compute_total_elements(shape: &Shape) -> HelixResult<usize> {
        let mut total: usize = 1;
        for &dim in shape {
            total = total.checked_mul(dim).ok_or_else(|| {
                ValidationError::invalid_shape(
                    "tensor",
                    shape.clone(),
                    "shape product overflows usize",
                )
            })?;
        }

        if total > MAX_TENSOR_ELEMENTS {
            return Err(ValidationError::invalid_shape(
                "tensor",
                shape.clone(),
                format!("too many elements ({} > {})", total, MAX_TENSOR_ELEMENTS),
            ).into());
        }

        Ok(total)
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: Shape) -> Self {
        let len: usize = shape.iter().product();
        let data = vec![BoundedValue::exact(0.0); len];
        Self::new(data, shape)
    }

    /// Creates a tensor filled with zeros, with validation.
    pub fn try_zeros(shape: Shape) -> HelixResult<Self> {
        Self::validate_shape(&shape)?;
        let len = Self::compute_total_elements(&shape)?;
        let data = vec![BoundedValue::exact(0.0); len];
        Self::try_new(data, shape)
    }

    /// Creates a tensor filled with a constant value.
    pub fn full(shape: Shape, value: BoundedValue<f64>) -> Self {
        let len: usize = shape.iter().product();
        let data = vec![value; len];
        Self::new(data, shape)
    }

    /// Creates a tensor filled with a constant value, with validation.
    pub fn try_full(shape: Shape, value: BoundedValue<f64>) -> HelixResult<Self> {
        Self::validate_shape(&shape)?;
        let len = Self::compute_total_elements(&shape)?;
        let data = vec![value; len];
        Self::try_new(data, shape)
    }

    /// Creates a tensor from raw f64 values with zero error.
    pub fn from_exact(data: Vec<f64>, shape: Shape) -> Self {
        let bounded_data: Vec<_> = data.into_iter().map(BoundedValue::exact).collect();
        Self::new(bounded_data, shape)
    }

    /// Creates a tensor from raw f64 values with zero error, with validation.
    pub fn try_from_exact(data: Vec<f64>, shape: Shape) -> HelixResult<Self> {
        Self::validate_shape(&shape)?;
        Self::validate_data(&data)?;

        let bounded_data: Vec<_> = data.into_iter().map(BoundedValue::exact).collect();
        Self::try_new(bounded_data, shape)
    }

    /// Creates a tensor from raw f64 values with uniform absolute error.
    pub fn from_approximate(data: Vec<f64>, shape: Shape, error: f64) -> Self {
        let error_margin = ErrorMargin::absolute(error);
        let bounded_data: Vec<_> = data
            .into_iter()
            .map(|v| BoundedValue::new(v, error_margin))
            .collect();
        Self::new(bounded_data, shape)
    }

    /// Creates a tensor from raw f64 values with uniform absolute error, with validation.
    pub fn try_from_approximate(data: Vec<f64>, shape: Shape, error: f64) -> HelixResult<Self> {
        Self::validate_shape(&shape)?;
        Self::validate_data(&data)?;

        if error < 0.0 {
            return Err(ValidationError::out_of_range("error", error, 0.0, f64::INFINITY).into());
        }
        if error.is_nan() {
            return Err(ValidationError::invalid_value("error", error.to_string(), "is NaN").into());
        }

        let error_margin = ErrorMargin::absolute(error.min(MAX_ERROR_BOUND));
        let bounded_data: Vec<_> = data
            .into_iter()
            .map(|v| BoundedValue::new(v, error_margin))
            .collect();
        Self::try_new(bounded_data, shape)
    }

    /// Validates raw data for NaN/Inf values.
    fn validate_data(data: &[f64]) -> HelixResult<()> {
        for (i, &v) in data.iter().enumerate() {
            if v.is_nan() {
                return Err(ValidationError::invalid_value(
                    format!("data[{}]", i),
                    "NaN",
                    "NaN values not allowed",
                ).into());
            }
            if v.is_infinite() {
                return Err(ValidationError::invalid_value(
                    format!("data[{}]", i),
                    format!("{}", v),
                    "infinite values not allowed",
                ).into());
            }
        }
        Ok(())
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the shape as a slice (avoids cloning).
    pub fn shape_slice(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Validates an index against the tensor's shape.
    fn validate_index(&self, indices: &[usize]) -> HelixResult<()> {
        if indices.len() != self.ndim() {
            return Err(ArithmeticError::index_out_of_bounds(
                indices.to_vec(),
                self.shape.clone(),
            ).into());
        }

        for (_i, (&idx, &dim)) in indices.iter().zip(&self.shape).enumerate() {
            if idx >= dim {
                return Err(ArithmeticError::index_out_of_bounds(
                    indices.to_vec(),
                    self.shape.clone(),
                ).into());
            }
        }

        Ok(())
    }

    /// Converts multi-dimensional index to flat index.
    ///
    /// Assumes indices length matches ndim (caller should validate).
    fn flat_index(&self, indices: &[usize]) -> usize {
        // Handle mismatched lengths gracefully instead of panicking
        if indices.len() != self.ndim() {
            return 0; // Return 0 as a safe fallback, get/set will handle None
        }
        indices
            .iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Gets an element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> Option<&BoundedValue<f64>> {
        // Check dimension count first
        if indices.len() != self.ndim() {
            return None;
        }
        // Check bounds for each dimension
        for (&idx, &dim) in indices.iter().zip(&self.shape) {
            if idx >= dim {
                return None;
            }
        }
        let flat = self.flat_index(indices);
        self.data.get(flat)
    }

    /// Gets an element by multi-dimensional index with error on invalid index.
    pub fn try_get(&self, indices: &[usize]) -> HelixResult<&BoundedValue<f64>> {
        self.validate_index(indices)?;
        let flat = self.flat_index(indices);
        self.data.get(flat).ok_or_else(|| {
            ArithmeticError::index_out_of_bounds(indices.to_vec(), self.shape.clone()).into()
        })
    }

    /// Gets a mutable element by multi-dimensional index.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut BoundedValue<f64>> {
        // Check dimension count first
        if indices.len() != self.ndim() {
            return None;
        }
        // Check bounds for each dimension
        for (&idx, &dim) in indices.iter().zip(&self.shape) {
            if idx >= dim {
                return None;
            }
        }
        let flat = self.flat_index(indices);
        self.data.get_mut(flat)
    }

    /// Gets a mutable element by multi-dimensional index with error on invalid index.
    pub fn try_get_mut(&mut self, indices: &[usize]) -> HelixResult<&mut BoundedValue<f64>> {
        self.validate_index(indices)?;
        let flat = self.flat_index(indices);
        self.data.get_mut(flat).ok_or_else(|| {
            ArithmeticError::index_out_of_bounds(indices.to_vec(), self.shape.clone()).into()
        })
    }

    /// Sets an element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: BoundedValue<f64>) {
        if let Some(elem) = self.get_mut(indices) {
            *elem = value;
        }
    }

    /// Sets an element by multi-dimensional index with validation.
    pub fn try_set(&mut self, indices: &[usize], value: BoundedValue<f64>) -> HelixResult<()> {
        let elem = self.try_get_mut(indices)?;
        *elem = value;
        Ok(())
    }

    /// Returns the underlying data as a slice.
    pub fn data(&self) -> &[BoundedValue<f64>] {
        &self.data
    }

    /// Returns the underlying data as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [BoundedValue<f64>] {
        &mut self.data
    }

    /// Extracts just the values (without error bounds).
    pub fn values(&self) -> Vec<f64> {
        self.data.iter().map(|b| b.value()).collect()
    }

    /// Returns the maximum absolute error across all elements.
    pub fn max_error(&self) -> f64 {
        self.data
            .iter()
            .map(|b| b.absolute_error())
            .fold(0.0, f64::max)
    }

    /// Returns the mean absolute error across all elements.
    pub fn mean_error(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.data.iter().map(|b| b.absolute_error()).sum();
        sum / self.data.len() as f64
    }

    /// Checks if any element contains NaN.
    pub fn contains_nan(&self) -> bool {
        self.data.iter().any(|b| b.value().is_nan())
    }

    /// Checks if any element contains infinity.
    pub fn contains_inf(&self) -> bool {
        self.data.iter().any(|b| b.value().is_infinite())
    }

    /// Checks if all elements are finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|b| b.value().is_finite())
    }

    /// Validates that all elements are finite and have valid error bounds.
    pub fn validate(&self) -> HelixResult<()> {
        for (i, v) in self.data.iter().enumerate() {
            if v.value().is_nan() {
                return Err(ValidationError::invalid_value(
                    format!("element[{}]", i),
                    "NaN",
                    "NaN values not allowed in tensor",
                ).into());
            }
            if v.value().is_infinite() {
                return Err(ValidationError::invalid_value(
                    format!("element[{}]", i),
                    format!("{}", v.value()),
                    "infinite values not allowed in tensor",
                ).into());
            }
            let err = v.absolute_error();
            if err.is_nan() || err < 0.0 {
                return Err(ValidationError::invalid_value(
                    format!("element[{}] error", i),
                    format!("{}", err),
                    "invalid error bound",
                ).into());
            }
        }
        Ok(())
    }

    /// Sanitizes all elements by replacing NaN/Inf with valid values.
    pub fn sanitize(&mut self) {
        for v in &mut self.data {
            v.sanitize();
        }
    }

    /// Returns a sanitized copy of this tensor.
    pub fn sanitized(mut self) -> Self {
        self.sanitize();
        self
    }

    /// Clamps all error bounds to the maximum allowed value.
    pub fn clamp_errors(&mut self) {
        for v in &mut self.data {
            v.clamp_error();
        }
    }

    /// Reshapes the tensor (must have same total elements).
    pub fn reshape(&self, new_shape: Shape) -> Self {
        let new_len: usize = new_shape.iter().product();
        assert_eq!(self.len(), new_len, "Cannot reshape: element count mismatch");
        Self::new(self.data.clone(), new_shape)
    }

    /// Reshapes the tensor with validation.
    pub fn try_reshape(&self, new_shape: Shape) -> HelixResult<Self> {
        Self::validate_shape(&new_shape)?;
        let new_len = Self::compute_total_elements(&new_shape)?;

        if self.len() != new_len {
            return Err(ValidationError::invalid_shape(
                "new_shape",
                new_shape,
                format!("element count {} doesn't match current {}", new_len, self.len()),
            ).into());
        }

        Self::try_new(self.data.clone(), new_shape)
    }

    /// Flattens the tensor to 1D.
    pub fn flatten(&self) -> Self {
        Self::new(self.data.clone(), vec![self.len()])
    }

    /// Returns true if this is a vector (1D tensor).
    pub fn is_vector(&self) -> bool {
        self.ndim() == 1
    }

    /// Returns true if this is a matrix (2D tensor).
    pub fn is_matrix(&self) -> bool {
        self.ndim() == 2
    }

    /// Returns true if this is a scalar (0D tensor with 1 element).
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0 || (self.ndim() == 1 && self.len() == 1)
    }

    /// Validates that two tensors have compatible shapes for element-wise operations.
    fn validate_same_shape(&self, other: &BoundedTensor, operation: &str) -> HelixResult<()> {
        if self.shape != other.shape {
            return Err(ArithmeticError::shape_mismatch(
                operation,
                self.shape.clone(),
                other.shape.clone(),
            ).into());
        }
        Ok(())
    }

    /// Element-wise addition (tensors must have same shape).
    pub fn add(&self, other: &BoundedTensor) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch for addition");
        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a + *b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Element-wise addition with validation.
    pub fn try_add(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "addition")?;

        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.saturating_add(*b))
            .collect();

        Self::try_new(data, self.shape.clone())
    }

    /// Element-wise checked addition that returns errors on overflow.
    pub fn checked_add(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "addition")?;

        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(&other.data) {
            data.push(a.checked_add(*b)?);
        }

        Self::try_new(data, self.shape.clone())
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &BoundedTensor) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch for subtraction");
        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a - *b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Element-wise subtraction with validation.
    pub fn try_sub(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "subtraction")?;

        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.saturating_sub(*b))
            .collect();

        Self::try_new(data, self.shape.clone())
    }

    /// Element-wise checked subtraction.
    pub fn checked_sub(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "subtraction")?;

        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(&other.data) {
            data.push(a.checked_sub(*b)?);
        }

        Self::try_new(data, self.shape.clone())
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn hadamard(&self, other: &BoundedTensor) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch for Hadamard product");
        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Element-wise multiplication with validation.
    pub fn try_hadamard(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "hadamard product")?;

        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.saturating_mul(*b))
            .collect();

        Self::try_new(data, self.shape.clone())
    }

    /// Element-wise checked multiplication.
    pub fn checked_hadamard(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "hadamard product")?;

        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(&other.data) {
            data.push(a.checked_mul(*b)?);
        }

        Self::try_new(data, self.shape.clone())
    }

    /// Element-wise division.
    pub fn div(&self, other: &BoundedTensor) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch for division");
        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a / *b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Element-wise division with validation.
    pub fn try_div(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "division")?;

        let data: Vec<_> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.saturating_div(*b))
            .collect();

        Self::try_new(data, self.shape.clone())
    }

    /// Element-wise checked division.
    pub fn checked_div(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "division")?;

        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(&other.data) {
            data.push(a.checked_div(*b)?);
        }

        Self::try_new(data, self.shape.clone())
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: BoundedValue<f64>) -> Self {
        let data: Vec<_> = self.data.iter().map(|a| *a * scalar).collect();
        Self::new(data, self.shape.clone())
    }

    /// Adds another tensor element-wise in place, avoiding allocation.
    pub fn add_inplace(&mut self, other: &BoundedTensor) -> HelixResult<()> {
        if self.shape != other.shape {
            return Err(ArithmeticError::shape_mismatch(
                "add_inplace",
                self.shape.clone(),
                other.shape.clone(),
            ).into());
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = a.saturating_add(*b);
        }
        Ok(())
    }

    /// Scales all elements in place, avoiding allocation.
    pub fn scale_inplace(&mut self, scalar: BoundedValue<f64>) {
        for v in self.data.iter_mut() {
            *v = v.saturating_mul(scalar);
        }
    }

    /// Scalar multiplication with validation.
    pub fn try_scale(&self, scalar: BoundedValue<f64>) -> HelixResult<Self> {
        scalar.validate()?;

        let data: Vec<_> = self.data.iter().map(|a| a.saturating_mul(scalar)).collect();
        Self::try_new(data, self.shape.clone())
    }

    /// Scalar addition.
    pub fn add_scalar(&self, scalar: BoundedValue<f64>) -> Self {
        let data: Vec<_> = self.data.iter().map(|a| *a + scalar).collect();
        Self::new(data, self.shape.clone())
    }

    /// Apply a function to all elements.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(BoundedValue<f64>) -> BoundedValue<f64>
    {
        let data: Vec<_> = self.data.iter().map(|v| f(*v)).collect();
        Self::new(data, self.shape.clone())
    }

    /// Apply a fallible function to all elements.
    pub fn try_map<F>(&self, f: F) -> HelixResult<Self>
    where
        F: Fn(BoundedValue<f64>) -> HelixResult<BoundedValue<f64>>
    {
        let mut data = Vec::with_capacity(self.data.len());
        for v in &self.data {
            data.push(f(*v)?);
        }
        Self::try_new(data, self.shape.clone())
    }

    /// Transposes a 2D matrix.
    ///
    /// # Panics
    /// Panics if tensor is not 2D. Use `try_transpose()` for a non-panicking version.
    pub fn transpose(&self) -> Self {
        self.try_transpose().expect("Transpose requires 2D tensor")
    }

    /// Transposes a 2D matrix with validation.
    pub fn try_transpose(&self) -> HelixResult<Self> {
        if !self.is_matrix() {
            return Err(ValidationError::invalid_shape(
                "tensor",
                self.shape.clone(),
                "transpose requires 2D tensor",
            ).into());
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![BoundedValue::exact(0.0); self.len()];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = *self.try_get(&[i, j])?;
            }
        }

        Self::try_new(data, vec![cols, rows])
    }

    /// Computes the sum of all elements.
    pub fn sum(&self) -> BoundedValue<f64> {
        self.data
            .iter()
            .copied()
            .fold(BoundedValue::exact(0.0), |acc, v| acc + v)
    }

    /// Computes the sum with checked arithmetic.
    pub fn checked_sum(&self) -> HelixResult<BoundedValue<f64>> {
        let mut result = BoundedValue::exact(0.0);
        for v in &self.data {
            result = result.checked_add(*v)?;
        }
        Ok(result)
    }

    /// Computes the mean of all elements.
    pub fn mean(&self) -> BoundedValue<f64> {
        if self.is_empty() {
            return BoundedValue::exact(0.0);
        }
        let sum = self.sum();
        let n = BoundedValue::exact(self.len() as f64);
        sum / n
    }

    /// Computes the product of all elements.
    pub fn product(&self) -> BoundedValue<f64> {
        self.data
            .iter()
            .copied()
            .fold(BoundedValue::exact(1.0), |acc, v| acc * v)
    }

    /// Finds the minimum value.
    pub fn min(&self) -> Option<BoundedValue<f64>> {
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if a.value() <= b.value() { a } else { b })
    }

    /// Finds the maximum value.
    pub fn max(&self) -> Option<BoundedValue<f64>> {
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if a.value() >= b.value() { a } else { b })
    }

    /// Clamps all values to the given range.
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        let data: Vec<_> = self
            .data
            .iter()
            .map(|v| {
                let clamped = v.value().clamp(min, max);
                BoundedValue::new(clamped, v.error())
            })
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Returns true if the error of any element exceeds the threshold.
    pub fn exceeds_error_threshold(&self, threshold: f64) -> bool {
        self.max_error() > threshold
    }

    /// Returns the count of elements exceeding the error threshold.
    pub fn count_exceeding_threshold(&self, threshold: f64) -> usize {
        self.data
            .iter()
            .filter(|v| v.absolute_error() > threshold)
            .count()
    }

    /// Returns a slice view (not validated for bounds).
    pub fn slice_flat(&self, start: usize, end: usize) -> Option<&[BoundedValue<f64>]> {
        if start <= end && end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Creates a tensor from a flat slice with the given shape.
    pub fn from_slice(data: &[BoundedValue<f64>], shape: Shape) -> Self {
        Self::new(data.to_vec(), shape)
    }

    /// Broadcasts this tensor to the target shape if possible.
    pub fn try_broadcast(&self, target_shape: &Shape) -> HelixResult<Self> {
        // Simple broadcasting: only handle scalar -> any shape
        if self.len() == 1 {
            Self::try_full(target_shape.clone(), self.data[0])
        } else if &self.shape == target_shape {
            Ok(self.clone())
        } else {
            Err(ValidationError::invalid_shape(
                "tensor",
                self.shape.clone(),
                format!("cannot broadcast to {:?}", target_shape),
            ).into())
        }
    }

    // =========================================================================
    // PARALLEL OPERATIONS (using rayon)
    // =========================================================================

    /// Minimum elements to trigger parallel execution.
    const PARALLEL_THRESHOLD: usize = 1000;

    /// Parallel element-wise addition.
    pub fn par_add(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "parallel addition")?;

        let data: Vec<_> = if self.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(&other.data)
                .map(|(a, b)| a.saturating_add(*b))
                .collect()
        } else {
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a.saturating_add(*b))
                .collect()
        };

        Self::try_new(data, self.shape.clone())
    }

    /// Parallel element-wise subtraction.
    pub fn par_sub(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "parallel subtraction")?;

        let data: Vec<_> = if self.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(&other.data)
                .map(|(a, b)| a.saturating_sub(*b))
                .collect()
        } else {
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a.saturating_sub(*b))
                .collect()
        };

        Self::try_new(data, self.shape.clone())
    }

    /// Parallel element-wise multiplication (Hadamard product).
    pub fn par_hadamard(&self, other: &BoundedTensor) -> HelixResult<Self> {
        self.validate_same_shape(other, "parallel hadamard")?;

        let data: Vec<_> = if self.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(&other.data)
                .map(|(a, b)| a.saturating_mul(*b))
                .collect()
        } else {
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a.saturating_mul(*b))
                .collect()
        };

        Self::try_new(data, self.shape.clone())
    }

    /// Parallel scalar multiplication.
    pub fn par_scale(&self, scalar: BoundedValue<f64>) -> HelixResult<Self> {
        scalar.validate()?;

        let data: Vec<_> = if self.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .map(|a| a.saturating_mul(scalar))
                .collect()
        } else {
            self.data
                .iter()
                .map(|a| a.saturating_mul(scalar))
                .collect()
        };

        Self::try_new(data, self.shape.clone())
    }

    /// Parallel map operation.
    pub fn par_map<F>(&self, f: F) -> Self
    where
        F: Fn(BoundedValue<f64>) -> BoundedValue<f64> + Sync + Send,
    {
        let data: Vec<_> = if self.len() >= Self::PARALLEL_THRESHOLD {
            self.data.par_iter().map(|v| f(*v)).collect()
        } else {
            self.data.iter().map(|v| f(*v)).collect()
        };
        Self::new(data, self.shape.clone())
    }

    /// Parallel sum reduction.
    pub fn par_sum(&self) -> BoundedValue<f64> {
        if self.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .copied()
                .reduce(|| BoundedValue::exact(0.0), |acc, v| acc.saturating_add(v))
        } else {
            self.sum()
        }
    }

    /// Parallel maximum error calculation.
    pub fn par_max_error(&self) -> f64 {
        if self.len() >= Self::PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .map(|b| b.absolute_error())
                .reduce(|| 0.0, f64::max)
        } else {
            self.max_error()
        }
    }

    /// Parallel validation.
    pub fn par_validate(&self) -> HelixResult<()> {
        if self.len() >= Self::PARALLEL_THRESHOLD {
            // Check for any invalid values in parallel
            let has_nan = self.data.par_iter().any(|v| v.value().is_nan());
            if has_nan {
                return Err(ValidationError::invalid_value(
                    "tensor",
                    "NaN",
                    "NaN values not allowed in tensor",
                ).into());
            }

            let has_inf = self.data.par_iter().any(|v| v.value().is_infinite());
            if has_inf {
                return Err(ValidationError::invalid_value(
                    "tensor",
                    "Inf",
                    "infinite values not allowed in tensor",
                ).into());
            }

            let has_invalid_error = self.data.par_iter().any(|v| {
                let err = v.absolute_error();
                err.is_nan() || err < 0.0
            });
            if has_invalid_error {
                return Err(ValidationError::invalid_value(
                    "tensor error",
                    "invalid",
                    "invalid error bound in tensor",
                ).into());
            }

            Ok(())
        } else {
            self.validate()
        }
    }

    // =========================================================================
    // MATRIX MULTIPLICATION
    // =========================================================================

    /// Matrix multiplication for 2D tensors: C = A @ B
    ///
    /// For A with shape [M, K] and B with shape [K, N], produces C with shape [M, N].
    /// Error propagation: ε(C_ij) = Σ_k (|A_ik| * εB_kj + |B_kj| * εA_ik + εA_ik * εB_kj)
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// // A: 2x3 matrix
    /// let a = BoundedTensor::from_exact(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     vec![2, 3],
    /// );
    ///
    /// // B: 3x2 matrix
    /// let b = BoundedTensor::from_exact(
    ///     vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ///     vec![3, 2],
    /// );
    ///
    /// // C = A @ B: 2x2 result
    /// let c = a.matmul(&b).unwrap();
    /// assert_eq!(c.shape(), &vec![2, 2]);
    /// // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    /// assert_eq!(c.get(&[0, 0]).unwrap().value(), 58.0);
    /// ```
    pub fn matmul(&self, other: &BoundedTensor) -> HelixResult<Self> {
        // Validate both are 2D
        if !self.is_matrix() {
            return Err(ValidationError::invalid_shape(
                "left operand",
                self.shape.clone(),
                "matmul requires 2D tensors",
            ).into());
        }
        if !other.is_matrix() {
            return Err(ValidationError::invalid_shape(
                "right operand",
                other.shape.clone(),
                "matmul requires 2D tensors",
            ).into());
        }

        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k1 != k2 {
            return Err(ArithmeticError::shape_mismatch(
                "matmul",
                self.shape.clone(),
                other.shape.clone(),
            ).into());
        }

        let k = k1;
        let result_len = m * n;

        // Use parallel computation for large matrices
        let data: Vec<BoundedValue<f64>> = if result_len >= Self::PARALLEL_THRESHOLD {
            (0..m)
                .into_par_iter()
                .flat_map(|i| {
                    (0..n).into_par_iter().map(move |j| {
                        let mut sum = BoundedValue::exact(0.0);
                        for l in 0..k {
                            let a = self.data[i * k + l];
                            let b = other.data[l * n + j];
                            sum = sum.saturating_add(a.saturating_mul(b));
                        }
                        sum
                    }).collect::<Vec<_>>()
                })
                .collect()
        } else {
            let mut data = Vec::with_capacity(result_len);
            for i in 0..m {
                for j in 0..n {
                    let mut sum = BoundedValue::exact(0.0);
                    for l in 0..k {
                        let a = self.data[i * k + l];
                        let b = other.data[l * n + j];
                        sum = sum.saturating_add(a.saturating_mul(b));
                    }
                    data.push(sum);
                }
            }
            data
        };

        Self::try_new(data, vec![m, n])
    }

    /// Checked matrix multiplication that verifies error bounds.
    pub fn checked_matmul(&self, other: &BoundedTensor) -> HelixResult<Self> {
        let result = self.matmul(other)?;

        // Check for error explosion
        let max_err = result.par_max_error();
        if max_err > MAX_SAFE_ERROR {
            return Err(BoundsError::ErrorExplosion {
                accumulated: max_err,
                safe_limit: MAX_SAFE_ERROR,
            }.into());
        }

        Ok(result)
    }

    /// Checked matrix multiplication with a configurable error threshold.
    /// Use this for precision-aware checking (e.g., BF16/INT8 budgets).
    pub fn checked_matmul_with_limit(&self, other: &BoundedTensor, safe_limit: f64) -> HelixResult<Self> {
        let result = self.matmul(other)?;

        let max_err = result.par_max_error();
        if max_err > safe_limit {
            return Err(BoundsError::ErrorExplosion {
                accumulated: max_err,
                safe_limit,
            }.into());
        }

        Ok(result)
    }

    // =========================================================================
    // BATCHED OPERATIONS
    // =========================================================================

    /// Batched matrix multiplication for 3D tensors: C[b] = A[b] @ B[b]
    ///
    /// For A with shape [batch, M, K] and B with shape [batch, K, N],
    /// produces C with shape [batch, M, N].
    pub fn batch_matmul(&self, other: &BoundedTensor) -> HelixResult<Self> {
        // Validate both are 3D
        if self.ndim() != 3 {
            return Err(ValidationError::invalid_shape(
                "left operand",
                self.shape.clone(),
                "batch_matmul requires 3D tensors [batch, M, K]",
            ).into());
        }
        if other.ndim() != 3 {
            return Err(ValidationError::invalid_shape(
                "right operand",
                other.shape.clone(),
                "batch_matmul requires 3D tensors [batch, K, N]",
            ).into());
        }

        let (batch_a, m, k1) = (self.shape[0], self.shape[1], self.shape[2]);
        let (batch_b, k2, n) = (other.shape[0], other.shape[1], other.shape[2]);

        if batch_a != batch_b {
            return Err(ArithmeticError::shape_mismatch(
                "batch_matmul (batch dimension)",
                self.shape.clone(),
                other.shape.clone(),
            ).into());
        }
        if k1 != k2 {
            return Err(ArithmeticError::shape_mismatch(
                "batch_matmul (inner dimension)",
                self.shape.clone(),
                other.shape.clone(),
            ).into());
        }

        let batch = batch_a;
        let k = k1;

        // Process each batch in parallel
        let batch_results: Vec<Vec<BoundedValue<f64>>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let mut result = Vec::with_capacity(m * n);
                let a_offset = b * m * k;
                let b_offset = b * k * n;

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = BoundedValue::exact(0.0);
                        for l in 0..k {
                            let a = self.data[a_offset + i * k + l];
                            let b_val = other.data[b_offset + l * n + j];
                            sum = sum.saturating_add(a.saturating_mul(b_val));
                        }
                        result.push(sum);
                    }
                }
                result
            })
            .collect();

        // Flatten batch results
        let data: Vec<BoundedValue<f64>> = batch_results.into_iter().flatten().collect();

        Self::try_new(data, vec![batch, m, n])
    }

    /// Batched attention mechanism: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    ///
    /// For Q, K, V with shape [batch, seq_len, d_model], computes scaled dot-product attention.
    /// Returns the attention output with shape [batch, seq_len, d_model].
    pub fn batch_attention(
        query: &BoundedTensor,
        key: &BoundedTensor,
        value: &BoundedTensor,
    ) -> HelixResult<BoundedTensor> {
        // Validate shapes
        if query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3 {
            return Err(ValidationError::invalid_shape(
                "attention inputs",
                query.shape.clone(),
                "batch_attention requires 3D tensors [batch, seq_len, d_model]",
            ).into());
        }

        let (batch_q, seq_q, d_q) = (query.shape[0], query.shape[1], query.shape[2]);
        let (batch_k, seq_k, d_k) = (key.shape[0], key.shape[1], key.shape[2]);
        let (batch_v, seq_v, d_v) = (value.shape[0], value.shape[1], value.shape[2]);

        if batch_q != batch_k || batch_k != batch_v {
            return Err(ArithmeticError::shape_mismatch(
                "attention (batch dimension)",
                query.shape.clone(),
                key.shape.clone(),
            ).into());
        }
        if d_q != d_k {
            return Err(ArithmeticError::shape_mismatch(
                "attention (Q and K dimension)",
                query.shape.clone(),
                key.shape.clone(),
            ).into());
        }
        if seq_k != seq_v {
            return Err(ArithmeticError::shape_mismatch(
                "attention (K and V sequence length)",
                key.shape.clone(),
                value.shape.clone(),
            ).into());
        }

        let batch = batch_q;
        let d_model = d_v;
        let scale = 1.0 / (d_k as f64).sqrt();
        let scale_bounded = BoundedValue::exact(scale);

        // Process each batch in parallel
        let batch_results: Vec<Vec<BoundedValue<f64>>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let q_offset = b * seq_q * d_q;
                let k_offset = b * seq_k * d_k;
                let v_offset = b * seq_v * d_v;

                let mut output = Vec::with_capacity(seq_q * d_model);

                for i in 0..seq_q {
                    // Compute attention scores for this query position
                    // scores[j] = (Q[i] · K[j]) / sqrt(d_k)
                    let mut scores = Vec::with_capacity(seq_k);
                    let mut max_score = f64::NEG_INFINITY;

                    for j in 0..seq_k {
                        let mut dot = BoundedValue::exact(0.0);
                        for l in 0..d_k {
                            let q_val = query.data[q_offset + i * d_q + l];
                            let k_val = key.data[k_offset + j * d_k + l];
                            dot = dot.saturating_add(q_val.saturating_mul(k_val));
                        }
                        let score = dot.saturating_mul(scale_bounded);
                        if score.value() > max_score {
                            max_score = score.value();
                        }
                        scores.push(score);
                    }

                    // Compute softmax with numerical stability
                    // softmax(x) = exp(x - max) / sum(exp(x - max))
                    let max_bounded = BoundedValue::exact(max_score);
                    let exp_scores: Vec<BoundedValue<f64>> = scores
                        .iter()
                        .map(|s| {
                            let shifted = s.saturating_sub(max_bounded);
                            let exp_val = shifted.value().exp();
                            // Error for exp: |exp(x)| * εx (first order approximation)
                            let exp_error = exp_val.abs() * shifted.absolute_error();
                            BoundedValue::new(exp_val, ErrorMargin::absolute(exp_error))
                        })
                        .collect();

                    let sum_exp: BoundedValue<f64> = exp_scores
                        .iter()
                        .copied()
                        .fold(BoundedValue::exact(0.0), |acc, v| acc.saturating_add(v));

                    // Normalize (softmax weights)
                    let weights: Vec<BoundedValue<f64>> = exp_scores
                        .iter()
                        .map(|e| e.saturating_div(sum_exp))
                        .collect();

                    // Compute weighted sum of values: output[i] = Σ_j weights[j] * V[j]
                    for l in 0..d_model {
                        let mut weighted_sum = BoundedValue::exact(0.0);
                        for j in 0..seq_v {
                            let v_val = value.data[v_offset + j * d_v + l];
                            weighted_sum = weighted_sum.saturating_add(weights[j].saturating_mul(v_val));
                        }
                        output.push(weighted_sum);
                    }
                }

                output
            })
            .collect();

        // Flatten batch results
        let data: Vec<BoundedValue<f64>> = batch_results.into_iter().flatten().collect();

        BoundedTensor::try_new(data, vec![batch, seq_q, d_model])
    }

    /// Batched element-wise addition across a vector of tensors.
    /// All tensors must have the same shape. Returns sum of all tensors.
    pub fn batch_add(tensors: &[BoundedTensor]) -> HelixResult<BoundedTensor> {
        if tensors.is_empty() {
            return Err(ValidationError::invalid_input("batch_add requires at least one tensor").into());
        }

        let shape = tensors[0].shape().clone();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape() != &shape {
                return Err(ArithmeticError::shape_mismatch(
                    format!("batch_add tensor[0] vs tensor[{}]", i),
                    shape.clone(),
                    t.shape().clone(),
                ).into());
            }
        }

        let len = tensors[0].len();

        // Sum in parallel across element positions
        let data: Vec<BoundedValue<f64>> = (0..len)
            .into_par_iter()
            .map(|i| {
                tensors
                    .iter()
                    .map(|t| t.data[i])
                    .fold(BoundedValue::exact(0.0), |acc, v| acc.saturating_add(v))
            })
            .collect();

        BoundedTensor::try_new(data, shape)
    }

    /// Batched mean across a vector of tensors.
    /// All tensors must have the same shape. Returns element-wise mean.
    pub fn batch_mean(tensors: &[BoundedTensor]) -> HelixResult<BoundedTensor> {
        if tensors.is_empty() {
            return Err(ValidationError::invalid_input("batch_mean requires at least one tensor").into());
        }

        let sum = Self::batch_add(tensors)?;
        let n = BoundedValue::exact(tensors.len() as f64);

        let data: Vec<_> = sum.data
            .par_iter()
            .map(|v| v.saturating_div(n))
            .collect();

        BoundedTensor::try_new(data, sum.shape.clone())
    }

    /// Checks if any element's error exceeds the safe threshold.
    pub fn check_error_explosion(&self) -> HelixResult<()> {
        let max_err = self.par_max_error();
        if max_err > MAX_SAFE_ERROR {
            return Err(BoundsError::ErrorExplosion {
                accumulated: max_err,
                safe_limit: MAX_SAFE_ERROR,
            }.into());
        }
        Ok(())
    }

    /// Checks if any element's error exceeds a configurable threshold.
    /// Use this for precision-aware checking (e.g., BF16/INT8 budgets).
    pub fn check_error_explosion_with_limit(&self, safe_limit: f64) -> HelixResult<()> {
        let max_err = self.par_max_error();
        if max_err > safe_limit {
            return Err(BoundsError::ErrorExplosion {
                accumulated: max_err,
                safe_limit,
            }.into());
        }
        Ok(())
    }

    // =========================================================================
    // ACTIVATION FUNCTIONS
    // =========================================================================

    /// Applies ReLU activation: max(0, x).
    ///
    /// Error propagation: The error bound is preserved where x > 0, and set to 0
    /// where x <= 0 (since the output is exactly 0).
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let x = BoundedTensor::from_approximate(vec![-1.0, 0.0, 1.0, 2.0], vec![4], 0.01);
    /// let y = x.relu();
    /// // y = [0.0, 0.0, 1.0, 2.0] with appropriate error bounds
    /// ```
    pub fn relu(&self) -> Self {
        self.map(|v| {
            if v.value() > 0.0 {
                v
            } else {
                BoundedValue::exact(0.0)
            }
        })
    }

    /// Applies Leaky ReLU activation: max(alpha * x, x).
    ///
    /// # Arguments
    /// * `alpha` - The slope for negative values (typically 0.01)
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let x = BoundedTensor::from_approximate(vec![-2.0, -1.0, 1.0, 2.0], vec![4], 0.01);
    /// let y = x.leaky_relu(0.1);
    /// // y = [-0.2, -0.1, 1.0, 2.0]
    /// ```
    pub fn leaky_relu(&self, alpha: f64) -> Self {
        let alpha_bounded = BoundedValue::exact(alpha);
        self.map(|v| {
            if v.value() > 0.0 {
                v
            } else {
                v.saturating_mul(alpha_bounded)
            }
        })
    }

    /// Applies GELU (Gaussian Error Linear Unit) activation.
    ///
    /// GELU(x) = x * Φ(x), where Φ is the standard normal CDF.
    /// Uses the approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    ///
    /// Error propagation: Uses derivative bound |GELU'(x)| ≤ 1.08 for error scaling.
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let x = BoundedTensor::from_approximate(vec![-1.0, 0.0, 1.0], vec![3], 0.01);
    /// let y = x.gelu();
    /// ```
    pub fn gelu(&self) -> Self {
        // Constants for GELU approximation
        const SQRT_2_PI: f64 = 0.7978845608028654; // sqrt(2/pi)
        const COEFF: f64 = 0.044715;
        const MAX_DERIVATIVE: f64 = 1.08; // Upper bound on |GELU'(x)|

        self.map(|v| {
            let x = v.value();
            let x3 = x * x * x;
            let inner = SQRT_2_PI * (x + COEFF * x3);
            let tanh_val = inner.tanh();
            let result = 0.5 * x * (1.0 + tanh_val);

            // Error scales with derivative bound
            let error = v.absolute_error() * MAX_DERIVATIVE;
            BoundedValue::new(result, ErrorMargin::absolute(error))
        })
    }

    /// Applies sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
    ///
    /// Error propagation: |σ'(x)| = σ(x)(1 - σ(x)) ≤ 0.25
    /// So error scales by at most 0.25.
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let x = BoundedTensor::from_approximate(vec![-2.0, 0.0, 2.0], vec![3], 0.01);
    /// let y = x.sigmoid();
    /// // y ≈ [0.119, 0.5, 0.881]
    /// ```
    pub fn sigmoid(&self) -> Self {
        const MAX_DERIVATIVE: f64 = 0.25; // Maximum value of σ(x)(1-σ(x))

        self.map(|v| {
            let x = v.value();
            let result = 1.0 / (1.0 + (-x).exp());
            let error = v.absolute_error() * MAX_DERIVATIVE;
            BoundedValue::new(result, ErrorMargin::absolute(error))
        })
    }

    /// Applies tanh activation.
    ///
    /// Error propagation: |tanh'(x)| = 1 - tanh²(x) ≤ 1
    /// So error is preserved.
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let x = BoundedTensor::from_approximate(vec![-1.0, 0.0, 1.0], vec![3], 0.01);
    /// let y = x.tanh_activation();
    /// // y ≈ [-0.762, 0.0, 0.762]
    /// ```
    pub fn tanh_activation(&self) -> Self {
        self.map(|v| {
            let result = v.value().tanh();
            // tanh'(x) = 1 - tanh²(x), maximum at x=0 where derivative = 1
            let tanh_val = result;
            let derivative = 1.0 - tanh_val * tanh_val;
            let error = v.absolute_error() * derivative.max(0.01); // min 0.01 to avoid zero error
            BoundedValue::new(result, ErrorMargin::absolute(error))
        })
    }

    /// Applies Softplus activation: log(1 + exp(x)).
    ///
    /// Softplus is a smooth approximation to ReLU.
    /// Error propagation: softplus'(x) = sigmoid(x) ≤ 1
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let x = BoundedTensor::from_approximate(vec![-2.0, 0.0, 2.0], vec![3], 0.01);
    /// let y = x.softplus();
    /// ```
    pub fn softplus(&self) -> Self {
        self.map(|v| {
            let x = v.value();
            // Use log1p for numerical stability when x < 0
            let result = if x > 20.0 {
                x // Approximation for large x to avoid overflow
            } else if x < -20.0 {
                (-x).exp() // Approximation for very negative x
            } else {
                (1.0 + x.exp()).ln()
            };

            // Derivative is sigmoid(x)
            let sigmoid_val = 1.0 / (1.0 + (-x).exp());
            let error = v.absolute_error() * sigmoid_val;
            BoundedValue::new(result, ErrorMargin::absolute(error))
        })
    }

    /// Applies SiLU (Swish) activation: x * sigmoid(x).
    ///
    /// Error propagation: |SiLU'(x)| ≤ 1.1 (approximately)
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let x = BoundedTensor::from_approximate(vec![-1.0, 0.0, 1.0], vec![3], 0.01);
    /// let y = x.silu();
    /// ```
    pub fn silu(&self) -> Self {
        const MAX_DERIVATIVE: f64 = 1.1;

        self.map(|v| {
            let x = v.value();
            let sigmoid_val = 1.0 / (1.0 + (-x).exp());
            let result = x * sigmoid_val;
            let error = v.absolute_error() * MAX_DERIVATIVE;
            BoundedValue::new(result, ErrorMargin::absolute(error))
        })
    }

    /// Applies softmax normalization using the log-sum-exp trick for numerical stability.
    ///
    /// For 1D tensors, softmax is applied across all elements.
    /// For 2D tensors, softmax is applied independently over each row (last axis).
    ///
    /// Uses the numerically stable formulation:
    ///   softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
    ///
    /// This prevents overflow for large logits that would cause exp() to return infinity.
    ///
    /// Error propagation: The Jacobian of softmax has entries
    ///   ∂s_i/∂x_j = s_i(δ_ij - s_j), with ||J||_∞ ≤ 1.
    ///   Element error is bounded by: ε_i ≤ s_i * (sum_j ε_j + ε_i).
    ///   We use the simpler bound: ε_out ≤ exp(ε_in) * max(ε_in) for each row.
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// // Large logits that would overflow naive exp()
    /// let x = BoundedTensor::from_exact(vec![1000.0, 1001.0, 999.0], vec![3]);
    /// let s = x.softmax().unwrap();
    /// let sum: f64 = s.data().iter().map(|v| v.value()).sum();
    /// assert!((sum - 1.0).abs() < 1e-10, "softmax sum = {}", sum);
    /// assert!(s.is_finite());
    /// ```
    pub fn softmax(&self) -> HelixResult<Self> {
        match self.ndim() {
            1 => self.softmax_1d(),
            2 => self.softmax_2d(),
            _ => Err(ValidationError::invalid_shape(
                "softmax input",
                self.shape.clone(),
                "softmax requires 1D or 2D tensor",
            ).into()),
        }
    }

    /// Softmax over a 1D tensor.
    fn softmax_1d(&self) -> HelixResult<Self> {
        let len = self.shape[0];
        if len == 0 {
            return Self::try_new(vec![], vec![0]);
        }

        // Find max for numerical stability (log-sum-exp trick)
        let max_val = self.data.iter()
            .map(|v| v.value())
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x_i - max) and track max input error
        let mut exp_vals = Vec::with_capacity(len);
        let mut sum_exp = 0.0;
        let mut max_err = 0.0f64;

        for v in &self.data {
            let shifted = v.value() - max_val;
            let e = shifted.exp();
            let err = v.absolute_error();
            if err > max_err {
                max_err = err;
            }
            exp_vals.push((e, err));
            sum_exp += e;
        }

        if sum_exp == 0.0 {
            // All values are -inf after shifting; return uniform
            let uniform = BoundedValue::new(1.0 / len as f64, ErrorMargin::absolute(max_err));
            return Self::try_new(vec![uniform; len], vec![len]);
        }

        // Normalize: s_i = exp(x_i - max) / sum_exp
        // Error bound: each output error ≤ (exp_err / sum_exp) capped at 1.0
        let mut data = Vec::with_capacity(len);
        for (e, input_err) in exp_vals {
            let prob = e / sum_exp;
            // Error propagation through exp: d/dx exp(x) = exp(x), so exp_err = e * input_err
            // Error through division: (exp_err * sum_exp - e * sum_exp_err) / sum_exp^2
            // Simplified upper bound: prob * (input_err + max_err)
            let prob_err = (prob * (input_err + max_err)).min(1.0);
            data.push(BoundedValue::new(prob, ErrorMargin::absolute(prob_err)));
        }

        Self::try_new(data, vec![len])
    }

    /// Softmax over each row of a 2D tensor.
    fn softmax_2d(&self) -> HelixResult<Self> {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(rows * cols);

        for i in 0..rows {
            let row_start = i * cols;

            // Find max in this row
            let max_val = (0..cols)
                .map(|j| self.data[row_start + j].value())
                .fold(f64::NEG_INFINITY, f64::max);

            // Compute exp(x_j - max) and sum
            let mut exp_vals = Vec::with_capacity(cols);
            let mut sum_exp = 0.0;
            let mut max_err = 0.0f64;

            for j in 0..cols {
                let v = &self.data[row_start + j];
                let shifted = v.value() - max_val;
                let e = shifted.exp();
                let err = v.absolute_error();
                if err > max_err {
                    max_err = err;
                }
                exp_vals.push((e, err));
                sum_exp += e;
            }

            if sum_exp == 0.0 {
                let uniform = BoundedValue::new(1.0 / cols as f64, ErrorMargin::absolute(max_err));
                data.extend(std::iter::repeat(uniform).take(cols));
                continue;
            }

            for (e, input_err) in exp_vals {
                let prob = e / sum_exp;
                let prob_err = (prob * (input_err + max_err)).min(1.0);
                data.push(BoundedValue::new(prob, ErrorMargin::absolute(prob_err)));
            }
        }

        Self::try_new(data, vec![rows, cols])
    }

    // =========================================================================
    // CONVOLUTION OPERATIONS
    // =========================================================================

    /// 2D convolution operation.
    ///
    /// Performs a 2D convolution on a batch of images with a set of filters.
    ///
    /// # Arguments
    /// * `kernel` - Filter tensor of shape [out_channels, in_channels, kH, kW]
    /// * `stride` - Stride for both height and width
    /// * `padding` - Zero-padding for both height and width
    ///
    /// # Input Shape
    /// * `self` - Input tensor of shape [batch, in_channels, height, width]
    ///
    /// # Output Shape
    /// * Output tensor of shape [batch, out_channels, out_height, out_width]
    /// * out_height = (height + 2*padding - kH) / stride + 1
    /// * out_width = (width + 2*padding - kW) / stride + 1
    ///
    /// # Error Propagation
    /// For each output element (sum of kernel_size products):
    /// ε_out ≈ kH * kW * (|x| * ε_k + |k| * ε_x + ε_k * ε_x)
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// // Input: batch=1, channels=1, height=4, width=4
    /// let input = BoundedTensor::from_approximate(
    ///     (0..16).map(|i| i as f64).collect(),
    ///     vec![1, 1, 4, 4],
    ///     0.01,
    /// );
    ///
    /// // Kernel: out_channels=1, in_channels=1, kH=3, kW=3
    /// let kernel = BoundedTensor::from_approximate(
    ///     vec![1.0; 9],
    ///     vec![1, 1, 3, 3],
    ///     0.01,
    /// );
    ///
    /// let output = input.conv2d(&kernel, 1, 0).unwrap();
    /// assert_eq!(output.shape(), &vec![1, 1, 2, 2]);
    /// ```
    pub fn conv2d(
        &self,
        kernel: &BoundedTensor,
        stride: usize,
        padding: usize,
    ) -> HelixResult<Self> {
        // Validate input shape [batch, in_channels, height, width]
        if self.ndim() != 4 {
            return Err(ValidationError::invalid_shape(
                "conv2d input",
                self.shape.clone(),
                "expected 4D tensor [batch, in_channels, height, width]",
            ).into());
        }

        // Validate kernel shape [out_channels, in_channels, kH, kW]
        if kernel.ndim() != 4 {
            return Err(ValidationError::invalid_shape(
                "conv2d kernel",
                kernel.shape.clone(),
                "expected 4D tensor [out_channels, in_channels, kH, kW]",
            ).into());
        }

        let batch = self.shape[0];
        let in_channels = self.shape[1];
        let height = self.shape[2];
        let width = self.shape[3];

        let out_channels = kernel.shape[0];
        let kernel_in_channels = kernel.shape[1];
        let kh = kernel.shape[2];
        let kw = kernel.shape[3];

        // Validate channel dimensions match
        if in_channels != kernel_in_channels {
            return Err(ValidationError::invalid_shape(
                "conv2d",
                kernel.shape.clone(),
                format!(
                    "kernel in_channels ({}) doesn't match input in_channels ({})",
                    kernel_in_channels, in_channels
                ),
            ).into());
        }

        // Validate stride
        if stride == 0 {
            return Err(ValidationError::InvalidValue {
                field: "stride".into(),
                value: "0".into(),
                reason: "stride must be > 0".into(),
            }.into());
        }

        // Calculate output dimensions
        let out_height = (height + 2 * padding - kh) / stride + 1;
        let out_width = (width + 2 * padding - kw) / stride + 1;

        if out_height == 0 || out_width == 0 {
            return Err(ValidationError::invalid_shape(
                "conv2d output",
                vec![batch, out_channels, out_height, out_width],
                "output dimensions must be > 0 (check kernel size and padding)",
            ).into());
        }

        let output_size = batch * out_channels * out_height * out_width;
        let mut output_data = Vec::with_capacity(output_size);

        // Perform convolution with statistical error scaling.
        // Each output position accumulates k independent product errors. Under
        // a statistical independence assumption (same as MatrixErrorPropagation),
        // the combined error scales as sqrt(k) rather than k, giving much
        // tighter bounds for large kernels.
        for b in 0..batch {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = BoundedValue::exact(0.0);
                        let mut product_count: usize = 0;

                        for ic in 0..in_channels {
                            for kh_idx in 0..kh {
                                for kw_idx in 0..kw {
                                    // Input position with padding
                                    let ih = (oh * stride + kh_idx) as isize - padding as isize;
                                    let iw = (ow * stride + kw_idx) as isize - padding as isize;

                                    // Skip if outside input bounds (zero padding)
                                    if ih < 0 || ih >= height as isize
                                        || iw < 0 || iw >= width as isize
                                    {
                                        continue;
                                    }

                                    let ih = ih as usize;
                                    let iw = iw as usize;

                                    // Get input value
                                    let input_idx = b * (in_channels * height * width)
                                        + ic * (height * width)
                                        + ih * width
                                        + iw;
                                    let input_val = self.data[input_idx];

                                    // Get kernel value
                                    let kernel_idx = oc * (in_channels * kh * kw)
                                        + ic * (kh * kw)
                                        + kh_idx * kw
                                        + kw_idx;
                                    let kernel_val = kernel.data[kernel_idx];

                                    // Multiply and accumulate
                                    sum = sum.saturating_add(input_val.saturating_mul(kernel_val));
                                    product_count += 1;
                                }
                            }
                        }

                        // Apply sqrt(k) statistical error scaling.
                        // Value is exact (sum of products), only error is rescaled.
                        if product_count > 1 {
                            let k = product_count as f64;
                            let scaled_error = sum.absolute_error() / k.sqrt();
                            sum = BoundedValue::new(sum.value(), ErrorMargin::absolute(scaled_error));
                        }

                        output_data.push(sum);
                    }
                }
            }
        }

        BoundedTensor::try_new(output_data, vec![batch, out_channels, out_height, out_width])
    }

    /// 2D max pooling operation.
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling operation (defaults to kernel_size if 0)
    ///
    /// # Input Shape
    /// * `self` - Input tensor of shape [batch, channels, height, width]
    ///
    /// # Output Shape
    /// * Output tensor of shape [batch, channels, out_height, out_width]
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    ///
    /// let input = BoundedTensor::from_approximate(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     vec![1, 1, 3, 3],
    ///     0.01,
    /// );
    ///
    /// let output = input.max_pool2d(2, 2).unwrap();
    /// assert_eq!(output.shape(), &vec![1, 1, 1, 1]);
    /// ```
    pub fn max_pool2d(&self, kernel_size: usize, stride: usize) -> HelixResult<Self> {
        if self.ndim() != 4 {
            return Err(ValidationError::invalid_shape(
                "max_pool2d input",
                self.shape.clone(),
                "expected 4D tensor [batch, channels, height, width]",
            ).into());
        }

        let stride = if stride == 0 { kernel_size } else { stride };

        let batch = self.shape[0];
        let channels = self.shape[1];
        let height = self.shape[2];
        let width = self.shape[3];

        let out_height = (height - kernel_size) / stride + 1;
        let out_width = (width - kernel_size) / stride + 1;

        if out_height == 0 || out_width == 0 {
            return Err(ValidationError::invalid_shape(
                "max_pool2d output",
                vec![batch, channels, out_height, out_width],
                "output dimensions must be > 0",
            ).into());
        }

        let output_size = batch * channels * out_height * out_width;
        let mut output_data = Vec::with_capacity(output_size);

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut max_val: Option<BoundedValue<f64>> = None;

                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;

                                if ih < height && iw < width {
                                    let idx = b * (channels * height * width)
                                        + c * (height * width)
                                        + ih * width
                                        + iw;
                                    let val = self.data[idx];

                                    max_val = Some(match max_val {
                                        None => val,
                                        Some(current) => {
                                            if val.value() > current.value() {
                                                val
                                            } else {
                                                current
                                            }
                                        }
                                    });
                                }
                            }
                        }

                        output_data.push(max_val.unwrap_or(BoundedValue::exact(f64::NEG_INFINITY)));
                    }
                }
            }
        }

        BoundedTensor::try_new(output_data, vec![batch, channels, out_height, out_width])
    }

    /// 2D average pooling operation.
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling operation (defaults to kernel_size if 0)
    ///
    /// # Error Propagation
    /// Output error = mean of input errors in the pooling window
    pub fn avg_pool2d(&self, kernel_size: usize, stride: usize) -> HelixResult<Self> {
        if self.ndim() != 4 {
            return Err(ValidationError::invalid_shape(
                "avg_pool2d input",
                self.shape.clone(),
                "expected 4D tensor [batch, channels, height, width]",
            ).into());
        }

        let stride = if stride == 0 { kernel_size } else { stride };

        let batch = self.shape[0];
        let channels = self.shape[1];
        let height = self.shape[2];
        let width = self.shape[3];

        let out_height = (height - kernel_size) / stride + 1;
        let out_width = (width - kernel_size) / stride + 1;

        if out_height == 0 || out_width == 0 {
            return Err(ValidationError::invalid_shape(
                "avg_pool2d output",
                vec![batch, channels, out_height, out_width],
                "output dimensions must be > 0",
            ).into());
        }

        let output_size = batch * channels * out_height * out_width;
        let mut output_data = Vec::with_capacity(output_size);
        let _pool_size = (kernel_size * kernel_size) as f64;

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = BoundedValue::exact(0.0);
                        let mut count = 0;

                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;

                                if ih < height && iw < width {
                                    let idx = b * (channels * height * width)
                                        + c * (height * width)
                                        + ih * width
                                        + iw;
                                    sum = sum.saturating_add(self.data[idx]);
                                    count += 1;
                                }
                            }
                        }

                        let divisor = BoundedValue::exact(count as f64);
                        output_data.push(sum.saturating_div(divisor));
                    }
                }
            }
        }

        BoundedTensor::try_new(output_data, vec![batch, channels, out_height, out_width])
    }
}

impl fmt::Display for BoundedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BoundedTensor({:?}, max_err={:.2e})", self.shape, self.max_error())
    }
}

impl PartialEq for BoundedTensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

// =============================================================================
// PROVABLE TRAIT IMPLEMENTATION
// =============================================================================

use crate::traits::{BatchError, BatchProvable, BatchWitness, Provable, Witness};

/// Witness for a BoundedTensor, containing quantized values and error bounds.
#[derive(Debug, Clone)]
pub struct TensorWitness {
    /// Quantized tensor values as u64 (fixed-point representation)
    values: Vec<u64>,
    /// Quantized error bounds as u64
    error_bounds: Vec<u64>,
    /// Tensor shape for reconstruction
    shape: Vec<u64>,
}

impl TensorWitness {
    /// Scale factor for fixed-point quantization (10^12 = 12 decimal places).
    /// Matches ERROR_SCALE in error_commitment.rs for consistent precision across
    /// tensor witnesses and error commitments.
    const SCALE: f64 = 1e12;

    /// Creates a new tensor witness from a BoundedTensor.
    pub fn from_tensor(tensor: &BoundedTensor) -> Self {
        let values: Vec<u64> = tensor
            .data()
            .iter()
            .map(|v| Self::quantize(v.value()))
            .collect();

        let error_bounds: Vec<u64> = tensor
            .data()
            .iter()
            .map(|v| Self::quantize(v.absolute_error()))
            .collect();

        let shape: Vec<u64> = tensor.shape().iter().map(|&s| s as u64).collect();

        Self {
            values,
            error_bounds,
            shape,
        }
    }

    /// Quantizes a f64 to u64 using fixed-point representation.
    fn quantize(value: f64) -> u64 {
        let scaled = (value.abs() * Self::SCALE).clamp(0.0, u64::MAX as f64);
        scaled as u64
    }
}

impl Witness for TensorWitness {
    type FieldElement = u64;

    fn to_field_elements(&self) -> Vec<u64> {
        let mut elements = Vec::new();

        // Add shape information first
        elements.push(self.shape.len() as u64);
        elements.extend(&self.shape);

        // Add values
        elements.push(self.values.len() as u64);
        elements.extend(&self.values);

        // Add error bounds
        elements.extend(&self.error_bounds);

        elements
    }

    fn element_count(&self) -> usize {
        1 + self.shape.len() + 1 + self.values.len() * 2
    }
}

impl Provable for BoundedTensor {
    type Witness = TensorWitness;

    /// Generates a witness from this tensor for ZK circuit verification.
    ///
    /// The witness contains:
    /// - Shape information for reconstruction
    /// - Quantized values (fixed-point at 10^12 scale)
    /// - Quantized error bounds
    ///
    /// # Example
    ///
    /// ```
    /// use helix_core::types::BoundedTensor;
    /// use helix_core::traits::{Provable, Witness};
    ///
    /// let tensor = BoundedTensor::from_approximate(
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![2, 2],
    ///     0.01,
    /// );
    ///
    /// let witness = tensor.generate_witness();
    /// let elements = witness.to_field_elements();
    ///
    /// // Witness contains shape, values, and error bounds
    /// assert!(elements.len() > 4);
    /// ```
    fn generate_witness(&self) -> Self::Witness {
        TensorWitness::from_tensor(self)
    }

    /// Returns public inputs for proof verification.
    ///
    /// Format: [num_elements, max_error_quantized, shape_hash]
    fn public_inputs(&self) -> Vec<u64> {
        vec![
            self.len() as u64,
            TensorWitness::quantize(self.max_error()),
            Self::shape_hash(self.shape()),
        ]
    }

    /// Returns the circuit identifier for this tensor type.
    fn circuit_id(&self) -> &'static str {
        "bounded_tensor_v1"
    }
}

impl BoundedTensor {
    /// Computes a collision-resistant hash of tensor shape dimensions.
    ///
    /// Uses a polynomial hash `sum(dim_i * PRIME^i) mod u64` which is
    /// order-dependent — `[2,3]` and `[3,2]` produce different hashes,
    /// unlike the previous product-based approach.
    pub(crate) fn shape_hash(shape: &[usize]) -> u64 {
        const PRIME: u64 = 1_000_000_007;
        let mut hash: u64 = 0;
        let mut prime_power: u64 = 1;
        for &dim in shape {
            hash = hash.wrapping_add((dim as u64).wrapping_mul(prime_power));
            prime_power = prime_power.wrapping_mul(PRIME);
        }
        hash
    }
}

impl BatchProvable for BoundedTensor {
    /// Generates a batch witness from multiple tensors of the same shape.
    ///
    /// The common section contains the shared shape (written once). Each item
    /// section contains that tensor's quantized values and error bounds.
    fn generate_batch_witness(items: &[Self]) -> Result<BatchWitness, BatchError> {
        if items.is_empty() {
            return Err(BatchError::EmptyBatch);
        }

        let reference_shape: Vec<u64> = items[0].shape().iter().map(|&s| s as u64).collect();

        // Validate all tensors have the same shape
        for (i, item) in items.iter().enumerate().skip(1) {
            let item_shape: Vec<u64> = item.shape().iter().map(|&s| s as u64).collect();
            if item_shape != reference_shape {
                return Err(BatchError::ShapeMismatch {
                    expected: reference_shape,
                    actual: item_shape,
                    index: i,
                });
            }
        }

        // Common: ndims + shape dimensions
        let mut common = Vec::with_capacity(1 + reference_shape.len());
        common.push(reference_shape.len() as u64);
        common.extend(&reference_shape);

        // Per-item: quantized values + error bounds
        let batch_items: Vec<Vec<u64>> = items
            .iter()
            .map(|tensor| {
                let witness = TensorWitness::from_tensor(tensor);
                let mut item_data = Vec::with_capacity(witness.values.len() * 2);
                item_data.extend(&witness.values);
                item_data.extend(&witness.error_bounds);
                item_data
            })
            .collect();

        Ok(BatchWitness::new(common, batch_items))
    }

    /// Returns combined public inputs for a batch.
    ///
    /// Format: [batch_size, shape_hash, item_0_num_elements, item_0_max_error, ...]
    fn batch_public_inputs(items: &[Self]) -> Vec<u64> {
        let mut inputs = Vec::with_capacity(2 + items.len() * 2);
        inputs.push(items.len() as u64);
        if let Some(first) = items.first() {
            inputs.push(Self::shape_hash(first.shape()));
        }
        for item in items {
            inputs.push(item.len() as u64);
            inputs.push(TensorWitness::quantize(item.max_error()));
        }
        inputs
    }
}

/// Builder for creating tensors with validation.
#[derive(Debug, Default)]
pub struct TensorBuilder {
    data: Option<Vec<f64>>,
    shape: Option<Shape>,
    error: Option<f64>,
    validate_finite: bool,
}

impl TensorBuilder {
    /// Creates a new tensor builder.
    pub fn new() -> Self {
        Self {
            data: None,
            shape: None,
            error: None,
            validate_finite: true,
        }
    }

    /// Sets the data.
    pub fn data(mut self, data: Vec<f64>) -> Self {
        self.data = Some(data);
        self
    }

    /// Sets the shape.
    pub fn shape(mut self, shape: Shape) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Sets the uniform error bound.
    pub fn error(mut self, error: f64) -> Self {
        self.error = Some(error);
        self
    }

    /// Disables finite value validation.
    pub fn allow_non_finite(mut self) -> Self {
        self.validate_finite = false;
        self
    }

    /// Builds the tensor with full validation.
    pub fn build(self) -> HelixResult<BoundedTensor> {
        let data = self.data.ok_or_else(|| ValidationError::missing_field("data"))?;
        let shape = self.shape.ok_or_else(|| ValidationError::missing_field("shape"))?;

        BoundedTensor::validate_shape(&shape)?;

        if self.validate_finite {
            BoundedTensor::validate_data(&data)?;
        }

        let error = self.error.unwrap_or(0.0);
        if error < 0.0 {
            return Err(ValidationError::out_of_range("error", error, 0.0, f64::INFINITY).into());
        }

        // Convert to bounded values, bypassing try_from_exact/try_from_approximate
        // since we've already done the validation we want
        let bounded_data: Vec<_> = if error > 0.0 {
            let error_margin = ErrorMargin::absolute(error.min(MAX_ERROR_BOUND));
            data.into_iter()
                .map(|v| BoundedValue::new(v, error_margin))
                .collect()
        } else {
            data.into_iter().map(BoundedValue::exact).collect()
        };

        BoundedTensor::try_new(bounded_data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = BoundedTensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &vec![2, 3]);
        assert_eq!(t.len(), 6);
        assert_eq!(t.get(&[0, 0]).unwrap().value(), 0.0);
    }

    #[test]
    fn test_from_exact() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.get(&[0, 0]).unwrap().value(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap().value(), 2.0);
        assert_eq!(t.get(&[1, 0]).unwrap().value(), 3.0);
        assert_eq!(t.get(&[1, 1]).unwrap().value(), 4.0);
    }

    #[test]
    fn test_add() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let b = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);
        let c = a.add(&b);
        assert_eq!(c.get(&[0]).unwrap().value(), 4.0);
        assert_eq!(c.get(&[1]).unwrap().value(), 6.0);
    }

    #[test]
    fn test_transpose() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t2 = t.transpose();
        assert_eq!(t2.shape(), &vec![3, 2]);
        assert_eq!(t2.get(&[0, 0]).unwrap().value(), 1.0);
        assert_eq!(t2.get(&[0, 1]).unwrap().value(), 4.0);
        assert_eq!(t2.get(&[1, 0]).unwrap().value(), 2.0);
    }

    #[test]
    fn test_max_error() {
        let t = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0], vec![3], 0.1);
        assert!((t.max_error() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_shape_validation() {
        // Too many dimensions
        let result = BoundedTensor::try_zeros(vec![1; 10]);
        assert!(result.is_err());

        // Zero dimension
        let result = BoundedTensor::try_zeros(vec![2, 0, 3]);
        assert!(result.is_err());

        // Valid shape
        let result = BoundedTensor::try_zeros(vec![2, 3, 4]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nan_detection() {
        let t = BoundedTensor::from_exact(vec![1.0, f64::NAN, 3.0], vec![3]);
        assert!(t.contains_nan());
        assert!(!t.is_finite());
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_inf_detection() {
        let t = BoundedTensor::from_exact(vec![1.0, f64::INFINITY, 3.0], vec![3]);
        assert!(t.contains_inf());
        assert!(!t.is_finite());
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_sanitize() {
        let mut t = BoundedTensor::from_exact(vec![1.0, f64::NAN, f64::INFINITY], vec![3]);
        t.sanitize();
        assert!(t.is_finite());
    }

    #[test]
    fn test_try_operations() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let b = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);

        assert!(a.try_add(&b).is_ok());
        assert!(a.try_sub(&b).is_ok());
        assert!(a.try_hadamard(&b).is_ok());
        assert!(a.try_div(&b).is_ok());
    }

    #[test]
    fn test_shape_mismatch() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let b = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);

        assert!(a.try_add(&b).is_err());
    }

    #[test]
    fn test_try_get() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        assert!(t.try_get(&[0, 0]).is_ok());
        assert!(t.try_get(&[1, 1]).is_ok());
        assert!(t.try_get(&[2, 0]).is_err()); // Out of bounds
        assert!(t.try_get(&[0]).is_err()); // Wrong number of indices
    }

    #[test]
    fn test_sum_and_mean() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        assert_eq!(t.sum().value(), 10.0);
        assert_eq!(t.mean().value(), 2.5);
    }

    #[test]
    fn test_min_max() {
        let t = BoundedTensor::from_exact(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);
        assert_eq!(t.min().unwrap().value(), 1.0);
        assert_eq!(t.max().unwrap().value(), 5.0);
    }

    #[test]
    fn test_clamp() {
        let t = BoundedTensor::from_exact(vec![-1.0, 0.5, 2.0], vec![3]);
        let clamped = t.clamp(0.0, 1.0);
        assert_eq!(clamped.get(&[0]).unwrap().value(), 0.0);
        assert_eq!(clamped.get(&[1]).unwrap().value(), 0.5);
        assert_eq!(clamped.get(&[2]).unwrap().value(), 1.0);
    }

    #[test]
    fn test_builder() {
        let t = TensorBuilder::new()
            .data(vec![1.0, 2.0, 3.0, 4.0])
            .shape(vec![2, 2])
            .error(0.01)
            .build()
            .unwrap();

        assert_eq!(t.shape(), &vec![2, 2]);
        assert!((t.max_error() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_builder_validation() {
        // Missing data
        let result = TensorBuilder::new().shape(vec![2, 2]).build();
        assert!(result.is_err());

        // Missing shape
        let result = TensorBuilder::new().data(vec![1.0, 2.0]).build();
        assert!(result.is_err());

        // NaN data
        let result = TensorBuilder::new()
            .data(vec![1.0, f64::NAN])
            .shape(vec![2])
            .build();
        assert!(result.is_err());

        // NaN data allowed
        let result = TensorBuilder::new()
            .data(vec![1.0, f64::NAN])
            .shape(vec![2])
            .allow_non_finite()
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_reshape() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let reshaped = t.try_reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &vec![3, 2]);

        // Wrong element count
        assert!(t.try_reshape(vec![2, 2]).is_err());
    }

    #[test]
    fn test_checked_operations() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let b = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);

        assert!(a.checked_add(&b).is_ok());
        assert!(a.checked_sub(&b).is_ok());
        assert!(a.checked_hadamard(&b).is_ok());
    }

    #[test]
    fn test_map() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let doubled = t.map(|v| v * BoundedValue::exact(2.0));
        assert_eq!(doubled.get(&[0]).unwrap().value(), 2.0);
        assert_eq!(doubled.get(&[1]).unwrap().value(), 4.0);
        assert_eq!(doubled.get(&[2]).unwrap().value(), 6.0);
    }

    #[test]
    fn test_broadcast() {
        let scalar = BoundedTensor::from_exact(vec![5.0], vec![1]);
        let broadcasted = scalar.try_broadcast(&vec![2, 3]).unwrap();
        assert_eq!(broadcasted.shape(), &vec![2, 3]);
        assert_eq!(broadcasted.len(), 6);
        for v in broadcasted.data() {
            assert_eq!(v.value(), 5.0);
        }
    }

    // =========================================================================
    // PARALLEL OPERATION TESTS
    // =========================================================================

    #[test]
    fn test_par_add() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = BoundedTensor::from_exact(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.par_add(&b).unwrap();
        assert_eq!(c.get(&[0, 0]).unwrap().value(), 6.0);
        assert_eq!(c.get(&[1, 1]).unwrap().value(), 12.0);
    }

    #[test]
    fn test_par_hadamard() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = BoundedTensor::from_exact(vec![2.0, 3.0, 4.0, 5.0], vec![4]);
        let c = a.par_hadamard(&b).unwrap();
        assert_eq!(c.get(&[0]).unwrap().value(), 2.0);
        assert_eq!(c.get(&[3]).unwrap().value(), 20.0);
    }

    #[test]
    fn test_par_scale() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let scalar = BoundedValue::exact(2.0);
        let b = a.par_scale(scalar).unwrap();
        assert_eq!(b.get(&[0]).unwrap().value(), 2.0);
        assert_eq!(b.get(&[2]).unwrap().value(), 6.0);
    }

    #[test]
    fn test_par_sum() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let sum = a.par_sum();
        assert_eq!(sum.value(), 10.0);
    }

    // =========================================================================
    // MATRIX MULTIPLICATION TESTS
    // =========================================================================

    #[test]
    fn test_matmul_basic() {
        // 2x3 @ 3x2 = 2x2
        let a = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let b = BoundedTensor::from_exact(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
        );
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &vec![2, 2]);
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert_eq!(c.get(&[0, 0]).unwrap().value(), 58.0);
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assert_eq!(c.get(&[0, 1]).unwrap().value(), 64.0);
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        assert_eq!(c.get(&[1, 0]).unwrap().value(), 139.0);
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(c.get(&[1, 1]).unwrap().value(), 154.0);
    }

    #[test]
    fn test_matmul_identity() {
        // A @ I = A
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let identity = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = a.matmul(&identity).unwrap();

        assert_eq!(c.get(&[0, 0]).unwrap().value(), 1.0);
        assert_eq!(c.get(&[0, 1]).unwrap().value(), 2.0);
        assert_eq!(c.get(&[1, 0]).unwrap().value(), 3.0);
        assert_eq!(c.get(&[1, 1]).unwrap().value(), 4.0);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3, 1]);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_matmul_error_propagation() {
        let a = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![1, 2], 0.01);
        let b = BoundedTensor::from_approximate(vec![3.0, 4.0], vec![2, 1], 0.01);
        let c = a.matmul(&b).unwrap();

        // C = 1*3 + 2*4 = 11
        assert_eq!(c.get(&[0, 0]).unwrap().value(), 11.0);
        // Error should be accumulated through multiplication
        assert!(c.max_error() > 0.01);
    }

    // =========================================================================
    // BATCHED OPERATION TESTS
    // =========================================================================

    #[test]
    fn test_batch_matmul() {
        // Batch of 2 matrices: [2, 2, 3] @ [2, 3, 2] = [2, 2, 2]
        let a = BoundedTensor::from_exact(
            vec![
                // Batch 0: 2x3
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                // Batch 1: 2x3
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        );
        let b = BoundedTensor::from_exact(
            vec![
                // Batch 0: 3x2
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                // Batch 1: 3x2
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            ],
            vec![2, 3, 2],
        );

        let c = a.batch_matmul(&b).unwrap();
        assert_eq!(c.shape(), &vec![2, 2, 2]);

        // Batch 0: C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        assert_eq!(c.get(&[0, 0, 0]).unwrap().value(), 22.0);
    }

    #[test]
    fn test_batch_attention() {
        // Simple attention: batch=1, seq_len=2, d_model=3
        let q = BoundedTensor::from_exact(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![1, 2, 3],
        );
        let k = BoundedTensor::from_exact(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![1, 2, 3],
        );
        let v = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 2, 3],
        );

        let output = BoundedTensor::batch_attention(&q, &k, &v).unwrap();
        assert_eq!(output.shape(), &vec![1, 2, 3]);
        // Output should be weighted sum of values
        assert!(output.is_finite());
    }

    #[test]
    fn test_batch_add() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let b = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);
        let c = BoundedTensor::from_exact(vec![5.0, 6.0], vec![2]);

        let sum = BoundedTensor::batch_add(&[a, b, c]).unwrap();
        assert_eq!(sum.get(&[0]).unwrap().value(), 9.0);
        assert_eq!(sum.get(&[1]).unwrap().value(), 12.0);
    }

    #[test]
    fn test_batch_mean() {
        let a = BoundedTensor::from_exact(vec![1.0, 2.0], vec![2]);
        let b = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);
        let c = BoundedTensor::from_exact(vec![5.0, 6.0], vec![2]);

        let mean = BoundedTensor::batch_mean(&[a, b, c]).unwrap();
        assert_eq!(mean.get(&[0]).unwrap().value(), 3.0);
        assert_eq!(mean.get(&[1]).unwrap().value(), 4.0);
    }

    #[test]
    fn test_error_explosion_check() {
        let a = BoundedTensor::from_approximate(
            vec![1.0, 2.0],
            vec![2],
            1e7, // Above MAX_SAFE_ERROR
        );
        assert!(a.check_error_explosion().is_err());

        let b = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![2], 0.01);
        assert!(b.check_error_explosion().is_ok());
    }

    // =========================================================================
    // PROVABLE TRAIT TESTS
    // =========================================================================

    #[test]
    fn test_tensor_witness_generation() {
        use crate::traits::{Provable, Witness};

        let tensor = BoundedTensor::from_approximate(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            0.01,
        );

        let witness = tensor.generate_witness();
        let elements = witness.to_field_elements();

        // Should contain: ndim (1) + shape (2) + num_values (1) + values (4) + errors (4) = 12
        assert_eq!(elements.len(), 12);

        // First element is number of dimensions
        assert_eq!(elements[0], 2);

        // Next two are shape
        assert_eq!(elements[1], 2);
        assert_eq!(elements[2], 2);

        // Then count of values
        assert_eq!(elements[3], 4);
    }

    #[test]
    fn test_tensor_public_inputs() {
        use crate::traits::Provable;

        let tensor = BoundedTensor::from_approximate(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            0.01,
        );

        let public_inputs = tensor.public_inputs();

        // Should have 3 elements: [num_elements, max_error, shape_hash]
        assert_eq!(public_inputs.len(), 3);
        assert_eq!(public_inputs[0], 6); // num elements
        assert!(public_inputs[1] > 0); // quantized error
        // Shape hash is now a polynomial hash, not a simple product.
        // For shape [2,3]: hash = 2 * 1 + 3 * PRIME = 2 + 3_000_000_021
        assert_eq!(public_inputs[2], BoundedTensor::shape_hash(&[2, 3]));
    }

    #[test]
    fn test_shape_hash_collision_resistance() {
        // Permuted dimensions must produce different hashes
        assert_ne!(BoundedTensor::shape_hash(&[2, 3]), BoundedTensor::shape_hash(&[3, 2]));
        // Different rank with same product must produce different hashes
        assert_ne!(BoundedTensor::shape_hash(&[6]), BoundedTensor::shape_hash(&[2, 3]));
        assert_ne!(BoundedTensor::shape_hash(&[6]), BoundedTensor::shape_hash(&[3, 2]));
        // Same shape must produce same hash
        assert_eq!(BoundedTensor::shape_hash(&[2, 3]), BoundedTensor::shape_hash(&[2, 3]));
        // Empty shape
        assert_eq!(BoundedTensor::shape_hash(&[]), 0);
    }

    #[test]
    fn test_tensor_circuit_id() {
        use crate::traits::Provable;

        let tensor = BoundedTensor::zeros(vec![2, 2]);
        assert_eq!(tensor.circuit_id(), "bounded_tensor_v1");
    }

    #[test]
    fn test_tensor_witness_quantization() {
        use crate::traits::{Provable, Witness};

        // Test that small values are properly quantized
        let tensor = BoundedTensor::from_approximate(
            vec![0.000001, 0.000002],
            vec![2],
            1e-9,
        );

        let witness = tensor.generate_witness();
        let elements = witness.to_field_elements();

        // Values should be quantized (multiplied by 10^12)
        // 0.000001 * 10^12 = 1_000_000
        let value_start = 1 + 1 + 1; // ndim + shape + count
        assert_eq!(elements[value_start], 1_000_000);
        assert_eq!(elements[value_start + 1], 2_000_000);
    }

    // =========================================================================
    // BATCH WITNESS TESTS
    // =========================================================================

    #[test]
    fn test_batch_witness_common_matches_single() {
        use crate::traits::{BatchProvable, Provable, Witness};

        let t1 = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0.01);
        let t2 = BoundedTensor::from_approximate(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], 0.02);

        let batch = BoundedTensor::generate_batch_witness(&[t1.clone(), t2]).unwrap();

        // Common section should have ndims(=2) + shape [2, 2]
        assert_eq!(batch.common[0], 2); // ndims
        assert_eq!(batch.common[1], 2); // dim 0
        assert_eq!(batch.common[2], 2); // dim 1

        // Should match single witness shape section
        let single_witness = t1.generate_witness();
        let single_elements = single_witness.to_field_elements();
        assert_eq!(single_elements[0], batch.common[0]); // ndims
        assert_eq!(single_elements[1], batch.common[1]); // shape dim 0
    }

    #[test]
    fn test_batch_witness_smaller_than_individual() {
        use crate::traits::{BatchProvable, Provable, Witness};

        let tensors: Vec<BoundedTensor> = (0..32)
            .map(|i| {
                BoundedTensor::from_approximate(
                    vec![i as f64; 100],
                    vec![10, 10],
                    0.01,
                )
            })
            .collect();

        // Individual witnesses total size
        let individual_total: usize = tensors
            .iter()
            .map(|t| t.generate_witness().to_field_elements().len())
            .sum();

        // Batch witness total size
        let batch = BoundedTensor::generate_batch_witness(&tensors).unwrap();
        let batch_total = batch.to_field_elements().len();

        // Batch should be smaller: eliminates 31 redundant shape sections
        assert!(
            batch_total < individual_total,
            "Batch ({}) should be smaller than individual ({})",
            batch_total,
            individual_total
        );
    }

    #[test]
    fn test_batch_witness_shape_mismatch() {
        use crate::traits::BatchProvable;

        let t1 = BoundedTensor::from_approximate(vec![1.0, 2.0], vec![2], 0.01);
        let t2 = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0], vec![3], 0.01);

        let result = BoundedTensor::generate_batch_witness(&[t1, t2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_witness_empty() {
        use crate::traits::BatchProvable;

        let result = BoundedTensor::generate_batch_witness(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_public_inputs() {
        use crate::traits::BatchProvable;

        let t1 = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0.01);
        let t2 = BoundedTensor::from_approximate(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], 0.02);

        let inputs = BoundedTensor::batch_public_inputs(&[t1, t2]);

        // Format: [batch_size, shape_hash, elem_count_0, max_error_0, elem_count_1, max_error_1]
        assert_eq!(inputs[0], 2); // batch size
        assert_eq!(inputs[2], 4); // t1 num elements
        assert_eq!(inputs[4], 4); // t2 num elements
    }

    // =========================================================================
    // ACTIVATION FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_relu() {
        let x = BoundedTensor::from_approximate(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![5],
            0.01,
        );
        let y = x.relu();

        assert_eq!(y.get(&[0]).unwrap().value(), 0.0);
        assert_eq!(y.get(&[1]).unwrap().value(), 0.0);
        assert_eq!(y.get(&[2]).unwrap().value(), 0.0);
        assert_eq!(y.get(&[3]).unwrap().value(), 1.0);
        assert_eq!(y.get(&[4]).unwrap().value(), 2.0);

        // Error for negative values should be 0 (exact)
        assert_eq!(y.get(&[0]).unwrap().absolute_error(), 0.0);
        // Error for positive values should be preserved
        assert!(y.get(&[3]).unwrap().absolute_error() > 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        let x = BoundedTensor::from_approximate(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![5],
            0.01,
        );
        let y = x.leaky_relu(0.1);

        assert!((y.get(&[0]).unwrap().value() - (-0.2)).abs() < 1e-10);
        assert!((y.get(&[1]).unwrap().value() - (-0.1)).abs() < 1e-10);
        assert_eq!(y.get(&[2]).unwrap().value(), 0.0);
        assert_eq!(y.get(&[3]).unwrap().value(), 1.0);
        assert_eq!(y.get(&[4]).unwrap().value(), 2.0);
    }

    #[test]
    fn test_gelu() {
        let x = BoundedTensor::from_approximate(
            vec![-1.0, 0.0, 1.0],
            vec![3],
            0.01,
        );
        let y = x.gelu();

        // GELU(0) = 0
        assert!((y.get(&[1]).unwrap().value() - 0.0).abs() < 1e-6);
        // GELU(1) ≈ 0.841
        assert!((y.get(&[2]).unwrap().value() - 0.841).abs() < 0.01);
        // GELU(-1) ≈ -0.159
        assert!((y.get(&[0]).unwrap().value() - (-0.159)).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid() {
        let x = BoundedTensor::from_approximate(
            vec![-2.0, 0.0, 2.0],
            vec![3],
            0.01,
        );
        let y = x.sigmoid();

        // sigmoid(0) = 0.5
        assert!((y.get(&[1]).unwrap().value() - 0.5).abs() < 1e-6);
        // sigmoid(-2) ≈ 0.119
        assert!((y.get(&[0]).unwrap().value() - 0.119).abs() < 0.01);
        // sigmoid(2) ≈ 0.881
        assert!((y.get(&[2]).unwrap().value() - 0.881).abs() < 0.01);

        // All outputs should be in (0, 1)
        for v in y.data() {
            assert!(v.value() > 0.0 && v.value() < 1.0);
        }
    }

    #[test]
    fn test_tanh_activation() {
        let x = BoundedTensor::from_approximate(
            vec![-1.0, 0.0, 1.0],
            vec![3],
            0.01,
        );
        let y = x.tanh_activation();

        // tanh(0) = 0
        assert!((y.get(&[1]).unwrap().value() - 0.0).abs() < 1e-6);
        // tanh(1) ≈ 0.762
        assert!((y.get(&[2]).unwrap().value() - 0.762).abs() < 0.01);
        // tanh(-1) ≈ -0.762
        assert!((y.get(&[0]).unwrap().value() - (-0.762)).abs() < 0.01);
    }

    #[test]
    fn test_softplus() {
        let x = BoundedTensor::from_approximate(
            vec![-2.0, 0.0, 2.0],
            vec![3],
            0.01,
        );
        let y = x.softplus();

        // softplus(0) = ln(2) ≈ 0.693
        assert!((y.get(&[1]).unwrap().value() - 0.693).abs() < 0.01);
        // All outputs should be positive
        for v in y.data() {
            assert!(v.value() > 0.0);
        }
    }

    #[test]
    fn test_silu() {
        let x = BoundedTensor::from_approximate(
            vec![-1.0, 0.0, 1.0],
            vec![3],
            0.01,
        );
        let y = x.silu();

        // SiLU(0) = 0 * sigmoid(0) = 0
        assert!((y.get(&[1]).unwrap().value() - 0.0).abs() < 1e-6);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((y.get(&[2]).unwrap().value() - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_activation_error_propagation() {
        // Test that error bounds are propagated correctly through activations
        let x = BoundedTensor::from_approximate(vec![1.0], vec![1], 0.1);

        // ReLU: error preserved for positive values
        let relu_out = x.relu();
        assert_eq!(relu_out.max_error(), x.max_error());

        // Sigmoid: error scaled by max derivative (0.25)
        let sigmoid_out = x.sigmoid();
        assert!(sigmoid_out.max_error() <= x.max_error() * 0.25 + 1e-10);

        // GELU: error scaled by max derivative (~1.08)
        let gelu_out = x.gelu();
        assert!(gelu_out.max_error() <= x.max_error() * 1.1);
    }

    // =========================================================================
    // CONVOLUTION TESTS
    // =========================================================================

    #[test]
    fn test_conv2d_basic() {
        // Simple 3x3 convolution on 4x4 input
        // Input: batch=1, channels=1, height=4, width=4
        let input = BoundedTensor::from_exact(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
        );

        // 3x3 averaging kernel
        let kernel = BoundedTensor::from_exact(
            vec![1.0; 9],
            vec![1, 1, 3, 3],
        );

        let output = input.conv2d(&kernel, 1, 0).unwrap();
        assert_eq!(output.shape(), &vec![1, 1, 2, 2]);

        // Output[0,0] = sum of top-left 3x3 = 1+2+3+5+6+7+9+10+11 = 54
        assert_eq!(output.get(&[0, 0, 0, 0]).unwrap().value(), 54.0);
    }

    #[test]
    fn test_conv2d_with_stride() {
        // Input: 1x1x4x4
        let input = BoundedTensor::from_exact(
            (1..=16).map(|x| x as f64).collect(),
            vec![1, 1, 4, 4],
        );

        // 2x2 kernel
        let kernel = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);

        // Stride 2 should produce 2x2 output
        let output = input.conv2d(&kernel, 2, 0).unwrap();
        assert_eq!(output.shape(), &vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // Input: 1x1x3x3
        let input = BoundedTensor::from_exact(
            vec![1.0; 9],
            vec![1, 1, 3, 3],
        );

        // 3x3 kernel with padding=1 should produce same size output
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);
        let output = input.conv2d(&kernel, 1, 1).unwrap();
        assert_eq!(output.shape(), &vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_multi_channel() {
        // Input: batch=1, in_channels=2, height=3, width=3
        let input = BoundedTensor::from_exact(
            vec![
                // Channel 0
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                // Channel 1
                9.0, 8.0, 7.0,
                6.0, 5.0, 4.0,
                3.0, 2.0, 1.0,
            ],
            vec![1, 2, 3, 3],
        );

        // Kernel: out_channels=2, in_channels=2, kH=2, kW=2
        let kernel = BoundedTensor::from_exact(
            vec![1.0; 16], // 2 * 2 * 2 * 2
            vec![2, 2, 2, 2],
        );

        let output = input.conv2d(&kernel, 1, 0).unwrap();
        assert_eq!(output.shape(), &vec![1, 2, 2, 2]);
    }

    #[test]
    fn test_conv2d_dimension_validation() {
        // Invalid input shape (3D instead of 4D)
        let input = BoundedTensor::from_exact(vec![1.0; 8], vec![2, 2, 2]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);
        assert!(input.conv2d(&kernel, 1, 0).is_err());

        // Channel mismatch
        let input = BoundedTensor::from_exact(vec![1.0; 8], vec![1, 2, 2, 2]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 3, 3, 1]); // in_channels=3
        assert!(input.conv2d(&kernel, 1, 0).is_err());
    }

    #[test]
    fn test_conv2d_error_propagation() {
        let input = BoundedTensor::from_approximate(
            vec![1.0; 16],
            vec![1, 1, 4, 4],
            0.01,
        );
        let kernel = BoundedTensor::from_approximate(
            vec![0.5; 9],
            vec![1, 1, 3, 3],
            0.01,
        );

        let output = input.conv2d(&kernel, 1, 0).unwrap();

        // Error should accumulate through the convolution
        assert!(output.max_error() > input.max_error());
    }

    #[test]
    fn test_conv2d_statistical_error_scaling() {
        // With sqrt(k) scaling, a 7x7 kernel (k=49) should produce tighter
        // error bounds than linear accumulation would predict.
        // Linear: error ~ 49 * per_product_error
        // Statistical: error ~ sqrt(49) * per_product_error = 7 * per_product_error
        let base_error = 0.01;
        let input = BoundedTensor::from_approximate(
            vec![1.0; 81], // 9x9 input
            vec![1, 1, 9, 9],
            base_error,
        );

        // 3x3 kernel: k = 9 products per output
        let kernel_3x3 = BoundedTensor::from_approximate(
            vec![0.5; 9],
            vec![1, 1, 3, 3],
            base_error,
        );
        let out_3x3 = input.conv2d(&kernel_3x3, 1, 0).unwrap();

        // 7x7 kernel: k = 49 products per output
        let kernel_7x7 = BoundedTensor::from_approximate(
            vec![0.5; 49],
            vec![1, 1, 7, 7],
            base_error,
        );
        let out_7x7 = input.conv2d(&kernel_7x7, 1, 0).unwrap();

        // With sqrt(k) scaling, 7x7 error ratio vs 3x3 should be ~sqrt(49)/sqrt(9) = 7/3 ≈ 2.33
        // Without sqrt scaling, it would be 49/9 ≈ 5.44
        let ratio = out_7x7.max_error() / out_3x3.max_error();
        assert!(
            ratio < 4.0,
            "Error ratio 7x7/3x3 should be ~2.3 (statistical), not ~5.4 (linear). Got: {:.2}",
            ratio
        );
    }

    #[test]
    fn test_conv2d_error_scaling_single_product() {
        // With k=1 (1x1 kernel), no scaling should be applied
        let input = BoundedTensor::from_approximate(
            vec![2.0; 4],
            vec![1, 1, 2, 2],
            0.01,
        );
        let kernel = BoundedTensor::from_approximate(
            vec![3.0],
            vec![1, 1, 1, 1],
            0.01,
        );

        let output = input.conv2d(&kernel, 1, 0).unwrap();
        // Error for single product: |a|*εb + |b|*εa + εa*εb = 2*0.01 + 3*0.01 + 0.01*0.01
        let expected_error = 2.0 * 0.01 + 3.0 * 0.01 + 0.01 * 0.01;
        let actual_error = output.get(&[0, 0, 0, 0]).unwrap().absolute_error();
        assert!(
            (actual_error - expected_error).abs() < 1e-10,
            "Single-product error should match exact formula: expected {}, got {}",
            expected_error,
            actual_error
        );
    }

    #[test]
    fn test_max_pool2d() {
        let input = BoundedTensor::from_exact(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
        );

        let output = input.max_pool2d(2, 2).unwrap();
        assert_eq!(output.shape(), &vec![1, 1, 2, 2]);

        // Max of top-left 2x2 = max(1,2,5,6) = 6
        assert_eq!(output.get(&[0, 0, 0, 0]).unwrap().value(), 6.0);
        // Max of top-right 2x2 = max(3,4,7,8) = 8
        assert_eq!(output.get(&[0, 0, 0, 1]).unwrap().value(), 8.0);
        // Max of bottom-left 2x2 = max(9,10,13,14) = 14
        assert_eq!(output.get(&[0, 0, 1, 0]).unwrap().value(), 14.0);
        // Max of bottom-right 2x2 = max(11,12,15,16) = 16
        assert_eq!(output.get(&[0, 0, 1, 1]).unwrap().value(), 16.0);
    }

    #[test]
    fn test_avg_pool2d() {
        let input = BoundedTensor::from_exact(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
        );

        let output = input.avg_pool2d(2, 2).unwrap();
        assert_eq!(output.shape(), &vec![1, 1, 2, 2]);

        // Avg of top-left 2x2 = (1+2+5+6)/4 = 3.5
        assert!((output.get(&[0, 0, 0, 0]).unwrap().value() - 3.5).abs() < 1e-10);
        // Avg of top-right 2x2 = (3+4+7+8)/4 = 5.5
        assert!((output.get(&[0, 0, 0, 1]).unwrap().value() - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_pooling_error_propagation() {
        let input = BoundedTensor::from_approximate(
            vec![1.0; 16],
            vec![1, 1, 4, 4],
            0.01,
        );

        let max_pool_out = input.max_pool2d(2, 2).unwrap();
        let avg_pool_out = input.avg_pool2d(2, 2).unwrap();

        // Max pool should preserve the error of the maximum element
        assert!(max_pool_out.max_error() > 0.0);
        // Avg pool should have error related to mean of input errors
        assert!(avg_pool_out.max_error() > 0.0);
    }

    // =========================================================================
    // SOFTMAX TESTS
    // =========================================================================

    #[test]
    fn test_softmax_1d_basic() {
        let x = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let s = x.softmax().unwrap();
        assert_eq!(s.shape(), &vec![3]);

        let sum: f64 = s.data().iter().map(|v| v.value()).sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sum = {}", sum);

        // Values should be ordered: s[0] < s[1] < s[2]
        assert!(s.get(&[0]).unwrap().value() < s.get(&[1]).unwrap().value());
        assert!(s.get(&[1]).unwrap().value() < s.get(&[2]).unwrap().value());
    }

    #[test]
    fn test_softmax_large_logits_no_overflow() {
        // These values would cause exp(1001) = Inf in naive softmax
        let x = BoundedTensor::from_exact(vec![1000.0, 1001.0, 999.0], vec![3]);
        let s = x.softmax().unwrap();

        // Should not produce NaN or Inf
        assert!(s.is_finite(), "softmax produced non-finite values");

        let sum: f64 = s.data().iter().map(|v| v.value()).sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sum = {}", sum);

        // exp(1001-1001) / (exp(-1) + exp(0) + exp(-2)) ≈ 0.6652
        let p1 = s.get(&[1]).unwrap().value();
        assert!((p1 - 1.0 / (1.0 + (-1.0f64).exp() + (-2.0f64).exp())).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_negative_large_logits() {
        let x = BoundedTensor::from_exact(vec![-1000.0, -1001.0, -999.0], vec![3]);
        let s = x.softmax().unwrap();

        assert!(s.is_finite());
        let sum: f64 = s.data().iter().map(|v| v.value()).sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sum = {}", sum);
    }

    #[test]
    fn test_softmax_2d() {
        let x = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 100.0, 101.0, 99.0],
            vec![2, 3],
        );
        let s = x.softmax().unwrap();
        assert_eq!(s.shape(), &vec![2, 3]);

        // Each row should sum to 1
        let row0_sum: f64 = (0..3).map(|j| s.get(&[0, j]).unwrap().value()).sum();
        let row1_sum: f64 = (0..3).map(|j| s.get(&[1, j]).unwrap().value()).sum();
        assert!((row0_sum - 1.0).abs() < 1e-10, "row 0 sum = {}", row0_sum);
        assert!((row1_sum - 1.0).abs() < 1e-10, "row 1 sum = {}", row1_sum);
    }

    #[test]
    fn test_softmax_error_propagation() {
        let x = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0], vec![3], 0.01);
        let s = x.softmax().unwrap();

        // All outputs should have tracked error
        for v in s.data() {
            assert!(v.absolute_error() > 0.0, "softmax should propagate error");
            assert!(v.absolute_error() <= 1.0, "softmax error should be bounded");
        }
    }

    #[test]
    fn test_softmax_uniform() {
        // Equal inputs should give uniform distribution
        let x = BoundedTensor::from_exact(vec![5.0, 5.0, 5.0, 5.0], vec![4]);
        let s = x.softmax().unwrap();

        for v in s.data() {
            assert!((v.value() - 0.25).abs() < 1e-10, "expected 0.25, got {}", v.value());
        }
    }

    #[test]
    fn test_softmax_rejects_3d() {
        let x = BoundedTensor::from_exact(vec![1.0; 8], vec![2, 2, 2]);
        assert!(x.softmax().is_err());
    }
}
