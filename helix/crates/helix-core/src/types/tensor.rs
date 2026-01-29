//! Bounded tensor type for N-dimensional arrays with error tracking.

use super::bounded_value::BoundedValue;
use super::error_margin::ErrorMargin;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Shape of a tensor (dimensions).
pub type Shape = Vec<usize>;

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

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: Shape) -> Self {
        let len: usize = shape.iter().product();
        let data = vec![BoundedValue::exact(0.0); len];
        Self::new(data, shape)
    }

    /// Creates a tensor filled with a constant value.
    pub fn full(shape: Shape, value: BoundedValue<f64>) -> Self {
        let len: usize = shape.iter().product();
        let data = vec![value; len];
        Self::new(data, shape)
    }

    /// Creates a tensor from raw f64 values with zero error.
    pub fn from_exact(data: Vec<f64>, shape: Shape) -> Self {
        let bounded_data: Vec<_> = data.into_iter().map(BoundedValue::exact).collect();
        Self::new(bounded_data, shape)
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

    /// Converts multi-dimensional index to flat index.
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.ndim());
        indices
            .iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Gets an element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> Option<&BoundedValue<f64>> {
        // Check bounds
        for (i, (&idx, &dim)) in indices.iter().zip(&self.shape).enumerate() {
            if idx >= dim {
                return None;
            }
            if i >= self.ndim() {
                return None;
            }
        }
        let flat = self.flat_index(indices);
        self.data.get(flat)
    }

    /// Gets a mutable element by multi-dimensional index.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut BoundedValue<f64>> {
        for (&idx, &dim) in indices.iter().zip(&self.shape) {
            if idx >= dim {
                return None;
            }
        }
        let flat = self.flat_index(indices);
        self.data.get_mut(flat)
    }

    /// Sets an element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: BoundedValue<f64>) {
        if let Some(elem) = self.get_mut(indices) {
            *elem = value;
        }
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

    /// Reshapes the tensor (must have same total elements).
    pub fn reshape(&self, new_shape: Shape) -> Self {
        let new_len: usize = new_shape.iter().product();
        assert_eq!(self.len(), new_len, "Cannot reshape: element count mismatch");
        Self::new(self.data.clone(), new_shape)
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

    /// Scalar multiplication.
    pub fn scale(&self, scalar: BoundedValue<f64>) -> Self {
        let data: Vec<_> = self.data.iter().map(|a| *a * scalar).collect();
        Self::new(data, self.shape.clone())
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
}
