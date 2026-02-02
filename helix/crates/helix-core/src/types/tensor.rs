//! Bounded tensor type for N-dimensional arrays with error tracking.
//!
//! This module provides comprehensive shape validation, NaN/Inf detection,
//! and Result-returning operations for robust tensor computations.

use super::bounded_value::{BoundedValue, MAX_ERROR_BOUND};
use super::error_margin::ErrorMargin;
use crate::error::{ArithmeticError, HelixResult, ValidationError};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Shape of a tensor (dimensions).
pub type Shape = Vec<usize>;

/// Maximum number of elements allowed in a tensor.
pub const MAX_TENSOR_ELEMENTS: usize = 100_000_000; // 100M elements

/// Maximum number of dimensions allowed.
pub const MAX_TENSOR_DIMS: usize = 8;

/// A multi-dimensional array of bounded values.
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

        for (i, (&idx, &dim)) in indices.iter().zip(&self.shape).enumerate() {
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
    pub fn transpose(&self) -> Self {
        assert!(self.is_matrix(), "Transpose requires 2D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![BoundedValue::exact(0.0); self.len()];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = *self.get(&[i, j]).unwrap();
            }
        }

        Self::new(data, vec![cols, rows])
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
}
