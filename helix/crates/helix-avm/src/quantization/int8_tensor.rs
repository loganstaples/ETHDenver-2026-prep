//! INT8 Tensor Type with Scale/Zero-Point.
//!
//! Provides a dedicated INT8 tensor representation optimized for quantized neural network
//! inference and training. This implementation is designed for HELIX's approximate computing
//! model with proper error bound tracking.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Shape};
use std::ops::{Add, Mul, Sub};

use super::schemes::{QuantScheme, TensorQuantParams};

/// INT8 quantized tensor with scale and zero-point parameters.
///
/// This is the primary representation for 8-bit quantized data in HELIX.
/// The tensor stores:
/// - Raw INT8 values in the range [-128, 127] or [0, 255] depending on scheme
/// - Scale factor for dequantization: real_value = scale * (quantized - zero_point)
/// - Zero-point for asymmetric quantization
/// - Error bounds tracking for approximate computing
#[derive(Debug, Clone)]
pub struct Int8Tensor {
    /// Raw INT8 quantized data.
    data: Vec<i8>,
    /// Shape of the tensor.
    shape: Shape,
    /// Strides for indexing.
    strides: Vec<usize>,
    /// Scale factor for dequantization.
    scale: f64,
    /// Zero-point offset.
    zero_point: i8,
    /// Whether this uses symmetric quantization.
    symmetric: bool,
    /// Accumulated error from operations.
    accumulated_error: f64,
    /// Per-channel scales (for per-channel quantization).
    per_channel_scales: Option<Vec<f64>>,
    /// Per-channel zero-points.
    per_channel_zero_points: Option<Vec<i8>>,
    /// Axis for per-channel quantization.
    per_channel_axis: Option<usize>,
}

impl Int8Tensor {
    /// Creates a new INT8 tensor with per-tensor quantization.
    pub fn new(data: Vec<i8>, shape: Shape, scale: f64, zero_point: i8, symmetric: bool) -> Self {
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            scale,
            zero_point,
            symmetric,
            accumulated_error: scale / 2.0, // Initial quantization error
            per_channel_scales: None,
            per_channel_zero_points: None,
            per_channel_axis: None,
        }
    }

    /// Creates an INT8 tensor with per-channel quantization.
    pub fn per_channel(
        data: Vec<i8>,
        shape: Shape,
        scales: Vec<f64>,
        zero_points: Vec<i8>,
        axis: usize,
        symmetric: bool,
    ) -> Self {
        let strides = compute_strides(&shape);
        let max_scale = scales.iter().cloned().fold(0.0_f64, f64::max);

        Self {
            data,
            shape,
            strides,
            scale: max_scale,
            zero_point: 0, // Not used in per-channel mode
            symmetric,
            accumulated_error: max_scale / 2.0,
            per_channel_scales: Some(scales),
            per_channel_zero_points: Some(zero_points),
            per_channel_axis: Some(axis),
        }
    }

    /// Creates an INT8 tensor from a BoundedTensor using symmetric quantization.
    pub fn from_bounded_symmetric(tensor: &BoundedTensor) -> Self {
        let data = tensor.data();
        let shape = tensor.shape().clone();

        // Find max absolute value
        let max_abs = data
            .iter()
            .map(|v| v.value().abs())
            .fold(0.0_f64, f64::max);

        // Compute scale for symmetric quantization
        let scale = if max_abs > 1e-10 { max_abs / 127.0 } else { 1e-10 };

        // Quantize values
        let quantized: Vec<i8> = data
            .iter()
            .map(|v| {
                let q = (v.value() / scale).round() as i32;
                q.clamp(-127, 127) as i8
            })
            .collect();

        Self::new(quantized, shape, scale, 0, true)
    }

    /// Creates an INT8 tensor from a BoundedTensor using asymmetric quantization.
    pub fn from_bounded_asymmetric(tensor: &BoundedTensor) -> Self {
        let data = tensor.data();
        let shape = tensor.shape().clone();

        // Find min/max values
        let (min_val, max_val) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |acc, v| {
            (acc.0.min(v.value()), acc.1.max(v.value()))
        });

        // Compute scale and zero-point for asymmetric quantization [0, 255]
        let scale = if (max_val - min_val).abs() > 1e-10 {
            (max_val - min_val) / 255.0
        } else {
            1e-10
        };
        let zero_point = (-min_val / scale).round() as i8;

        // Quantize values
        let quantized: Vec<i8> = data
            .iter()
            .map(|v| {
                let q = (v.value() / scale).round() as i32 + zero_point as i32;
                (q.clamp(0, 255) - 128) as i8 // Store as signed for easier math
            })
            .collect();

        Self::new(quantized, shape, scale, zero_point, false)
    }

    /// Creates an INT8 tensor from raw float data with automatic calibration.
    pub fn from_float_data(data: &[f64], shape: Shape, scheme: QuantScheme) -> Self {
        match scheme {
            QuantScheme::SymmetricInt8 => {
                let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                let scale = if max_abs > 1e-10 { max_abs / 127.0 } else { 1e-10 };
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&v| {
                        let q = (v / scale).round() as i32;
                        q.clamp(-127, 127) as i8
                    })
                    .collect();
                Self::new(quantized, shape, scale, 0, true)
            }
            QuantScheme::AsymmetricInt8 => {
                let (min_val, max_val) = data
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
                        (acc.0.min(v), acc.1.max(v))
                    });
                let scale = if (max_val - min_val).abs() > 1e-10 {
                    (max_val - min_val) / 255.0
                } else {
                    1e-10
                };
                let zero_point = ((-min_val / scale).round() as i32).clamp(0, 255) as i8;
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&v| {
                        let q = (v / scale).round() as i32 + zero_point as i32;
                        q.clamp(0, 255) as i8
                    })
                    .collect();
                Self::new(quantized, shape, scale, zero_point, false)
            }
            _ => Self::from_float_data(data, shape, QuantScheme::SymmetricInt8),
        }
    }

    /// Creates a zero-filled INT8 tensor.
    pub fn zeros(shape: Shape, scale: f64) -> Self {
        let size = shape.iter().product();
        Self::new(vec![0i8; size], shape, scale, 0, true)
    }

    /// Creates an INT8 tensor filled with a specific value.
    pub fn full(shape: Shape, value: i8, scale: f64, zero_point: i8) -> Self {
        let size = shape.iter().product();
        Self::new(vec![value; size], shape, scale, zero_point, true)
    }

    /// Returns the raw INT8 data.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Returns mutable access to raw data.
    pub fn data_mut(&mut self) -> &mut [i8] {
        &mut self.data
    }

    /// Returns the shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the zero-point.
    pub fn zero_point(&self) -> i8 {
        self.zero_point
    }

    /// Returns whether this uses symmetric quantization.
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
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

    /// Returns true if this is a vector (1D).
    pub fn is_vector(&self) -> bool {
        self.shape.len() == 1
    }

    /// Returns true if this is a matrix (2D).
    pub fn is_matrix(&self) -> bool {
        self.shape.len() == 2
    }

    /// Returns whether this uses per-channel quantization.
    pub fn is_per_channel(&self) -> bool {
        self.per_channel_scales.is_some()
    }

    /// Returns per-channel scales if available.
    pub fn per_channel_scales(&self) -> Option<&[f64]> {
        self.per_channel_scales.as_deref()
    }

    /// Returns per-channel zero-points if available.
    pub fn per_channel_zero_points(&self) -> Option<&[i8]> {
        self.per_channel_zero_points.as_deref()
    }

    /// Returns the per-channel axis.
    pub fn per_channel_axis(&self) -> Option<usize> {
        self.per_channel_axis
    }

    /// Gets a value at a specific index.
    pub fn get(&self, index: &[usize]) -> Option<i8> {
        let flat_idx = self.flat_index(index)?;
        self.data.get(flat_idx).copied()
    }

    /// Sets a value at a specific index.
    pub fn set(&mut self, index: &[usize], value: i8) -> Option<()> {
        let flat_idx = self.flat_index(index)?;
        if flat_idx < self.data.len() {
            self.data[flat_idx] = value;
            Some(())
        } else {
            None
        }
    }

    /// Computes the flat index from multi-dimensional index.
    fn flat_index(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.shape.len() {
            return None;
        }
        for (i, &dim) in index.iter().enumerate() {
            if dim >= self.shape[i] {
                return None;
            }
        }
        Some(
            index
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum(),
        )
    }

    /// Dequantizes a single element to f64.
    pub fn dequantize_value(&self, index: &[usize]) -> Option<f64> {
        let flat_idx = self.flat_index(index)?;
        let quantized = self.data[flat_idx];

        if self.is_per_channel() {
            // Per-channel dequantization
            let axis = self.per_channel_axis.unwrap();
            let channel = index[axis];
            let scale = self.per_channel_scales.as_ref().unwrap()[channel];
            let zp = self.per_channel_zero_points.as_ref().unwrap()[channel];
            Some((quantized as i32 - zp as i32) as f64 * scale)
        } else if self.symmetric {
            Some((quantized as i32 - self.zero_point as i32) as f64 * self.scale)
        } else {
            // For asymmetric, reinterpret as unsigned
            let unsigned_val = (quantized as u8) as i32;
            let unsigned_zp = (self.zero_point as u8) as i32;
            Some((unsigned_val - unsigned_zp) as f64 * self.scale)
        }
    }

    /// Dequantizes the entire tensor to a BoundedTensor.
    pub fn dequantize(&self) -> BoundedTensor {
        let quant_error = self.quantization_error();

        let bounded_data: Vec<BoundedValue<f64>> = if self.is_per_channel() {
            let axis = self.per_channel_axis.unwrap();
            let channel_size = self.shape[axis];
            let stride: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);

            self.data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let channel = (i / stride) % channel_size;
                    let scale = self.per_channel_scales.as_ref().unwrap()[channel];
                    let zp = self.per_channel_zero_points.as_ref().unwrap()[channel];
                    let dq = (v as i32 - zp as i32) as f64 * scale;
                    BoundedValue::<f64>::with_absolute_error(dq, scale / 2.0 + self.accumulated_error)
                })
                .collect()
        } else if self.symmetric {
            self.data
                .iter()
                .map(|&v| {
                    let dq = (v as i32 - self.zero_point as i32) as f64 * self.scale;
                    BoundedValue::<f64>::with_absolute_error(dq, quant_error + self.accumulated_error)
                })
                .collect()
        } else {
            // For asymmetric quantization, reinterpret data as unsigned
            self.data
                .iter()
                .map(|&v| {
                    let unsigned_val = (v as u8) as i32;
                    let unsigned_zp = (self.zero_point as u8) as i32;
                    let dq = (unsigned_val - unsigned_zp) as f64 * self.scale;
                    BoundedValue::<f64>::with_absolute_error(dq, quant_error + self.accumulated_error)
                })
                .collect()
        };

        BoundedTensor::new(bounded_data, self.shape.clone())
    }

    /// Dequantizes to raw f64 values without error tracking.
    pub fn dequantize_raw(&self) -> Vec<f64> {
        if self.is_per_channel() {
            let axis = self.per_channel_axis.unwrap();
            let channel_size = self.shape[axis];
            let stride: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);

            self.data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let channel = (i / stride) % channel_size;
                    let scale = self.per_channel_scales.as_ref().unwrap()[channel];
                    let zp = self.per_channel_zero_points.as_ref().unwrap()[channel];
                    (v as i32 - zp as i32) as f64 * scale
                })
                .collect()
        } else if self.symmetric {
            self.data
                .iter()
                .map(|&v| (v as i32 - self.zero_point as i32) as f64 * self.scale)
                .collect()
        } else {
            // For asymmetric quantization, data is stored as unsigned [0,255] cast to i8
            // Need to reinterpret as unsigned for correct dequantization
            self.data
                .iter()
                .map(|&v| {
                    let unsigned_val = (v as u8) as i32;
                    let unsigned_zp = (self.zero_point as u8) as i32;
                    (unsigned_val - unsigned_zp) as f64 * self.scale
                })
                .collect()
        }
    }

    /// Returns the quantization error bound (half of one quantum).
    pub fn quantization_error(&self) -> f64 {
        self.scale / 2.0
    }

    /// Returns the total accumulated error.
    pub fn total_error(&self) -> f64 {
        self.accumulated_error
    }

    /// Adds error from an operation.
    pub fn add_error(&mut self, error: f64) {
        self.accumulated_error += error;
    }

    /// Resets accumulated error to base quantization error.
    pub fn reset_error(&mut self) {
        self.accumulated_error = self.quantization_error();
    }

    /// Creates a copy with updated error.
    pub fn with_error(mut self, error: f64) -> Self {
        self.accumulated_error = error;
        self
    }

    /// Reshapes the tensor (must have same number of elements).
    pub fn reshape(&self, new_shape: Shape) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.data.len(),
            new_size,
            "Cannot reshape: size mismatch"
        );
        Self {
            data: self.data.clone(),
            shape: new_shape.clone(),
            strides: compute_strides(&new_shape),
            scale: self.scale,
            zero_point: self.zero_point,
            symmetric: self.symmetric,
            accumulated_error: self.accumulated_error,
            per_channel_scales: None, // Per-channel loses meaning after reshape
            per_channel_zero_points: None,
            per_channel_axis: None,
        }
    }

    /// Transposes a 2D tensor.
    pub fn transpose(&self) -> Self {
        assert!(self.is_matrix(), "Transpose only supported for 2D tensors");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut transposed = vec![0i8; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = self.data[i * cols + j];
            }
        }

        Self::new(
            transposed,
            vec![cols, rows],
            self.scale,
            self.zero_point,
            self.symmetric,
        )
    }

    /// Converts to TensorQuantParams for compatibility.
    pub fn to_quant_params(&self) -> TensorQuantParams {
        if self.is_per_channel() {
            TensorQuantParams::per_channel(
                if self.symmetric {
                    QuantScheme::SymmetricInt8
                } else {
                    QuantScheme::AsymmetricInt8
                },
                self.per_channel_scales.clone().unwrap(),
                self.per_channel_zero_points
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|&zp| zp as i64)
                    .collect(),
                self.per_channel_axis.unwrap(),
            )
        } else {
            TensorQuantParams::new(
                if self.symmetric {
                    QuantScheme::SymmetricInt8
                } else {
                    QuantScheme::AsymmetricInt8
                },
                self.scale,
                self.zero_point as i64,
            )
        }
    }

    /// Clamps all values to the valid INT8 range.
    pub fn clamp_values(&mut self) {
        let (min_val, max_val) = if self.symmetric { (-127, 127) } else { (0, 255) };
        for v in &mut self.data {
            let clamped = (*v as i32).clamp(min_val, max_val);
            *v = clamped as i8;
        }
    }

    /// Computes the L2 norm of the dequantized values.
    pub fn l2_norm(&self) -> f64 {
        let dq = self.dequantize_raw();
        dq.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Computes the max absolute value in dequantized form.
    pub fn max_abs(&self) -> f64 {
        let dq = self.dequantize_raw();
        dq.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
    }

    /// Returns statistics about the quantized values.
    pub fn statistics(&self) -> Int8TensorStats {
        let min = *self.data.iter().min().unwrap_or(&0);
        let max = *self.data.iter().max().unwrap_or(&0);
        let sum: i64 = self.data.iter().map(|&v| v as i64).sum();
        let mean = sum as f64 / self.data.len().max(1) as f64;

        let dq = self.dequantize_raw();
        let dq_min = dq.iter().cloned().fold(f64::INFINITY, f64::min);
        let dq_max = dq.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Int8TensorStats {
            min_quantized: min,
            max_quantized: max,
            mean_quantized: mean,
            min_dequantized: dq_min,
            max_dequantized: dq_max,
            scale: self.scale,
            zero_point: self.zero_point,
            sparsity: self.data.iter().filter(|&&v| v == self.zero_point).count() as f64
                / self.data.len().max(1) as f64,
        }
    }
}

/// Statistics for an INT8 tensor.
#[derive(Debug, Clone)]
pub struct Int8TensorStats {
    /// Minimum quantized value.
    pub min_quantized: i8,
    /// Maximum quantized value.
    pub max_quantized: i8,
    /// Mean quantized value.
    pub mean_quantized: f64,
    /// Minimum dequantized value.
    pub min_dequantized: f64,
    /// Maximum dequantized value.
    pub max_dequantized: f64,
    /// Scale factor.
    pub scale: f64,
    /// Zero-point.
    pub zero_point: i8,
    /// Fraction of values equal to zero-point.
    pub sparsity: f64,
}

/// Computes strides for row-major layout.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Builder for creating INT8 tensors with various options.
#[derive(Debug, Clone)]
pub struct Int8TensorBuilder {
    data: Option<Vec<i8>>,
    float_data: Option<Vec<f64>>,
    shape: Shape,
    scale: Option<f64>,
    zero_point: i8,
    symmetric: bool,
    per_channel: bool,
    channel_axis: usize,
}

impl Int8TensorBuilder {
    /// Creates a new builder with the given shape.
    pub fn new(shape: Shape) -> Self {
        Self {
            data: None,
            float_data: None,
            shape,
            scale: None,
            zero_point: 0,
            symmetric: true,
            per_channel: false,
            channel_axis: 0,
        }
    }

    /// Sets quantized INT8 data directly.
    pub fn with_data(mut self, data: Vec<i8>) -> Self {
        self.data = Some(data);
        self
    }

    /// Sets float data to be quantized.
    pub fn with_float_data(mut self, data: Vec<f64>) -> Self {
        self.float_data = Some(data);
        self
    }

    /// Sets the scale.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Sets the zero-point.
    pub fn with_zero_point(mut self, zero_point: i8) -> Self {
        self.zero_point = zero_point;
        self
    }

    /// Enables symmetric quantization.
    pub fn symmetric(mut self) -> Self {
        self.symmetric = true;
        self.zero_point = 0;
        self
    }

    /// Enables asymmetric quantization.
    pub fn asymmetric(mut self) -> Self {
        self.symmetric = false;
        self
    }

    /// Enables per-channel quantization.
    pub fn per_channel(mut self, axis: usize) -> Self {
        self.per_channel = true;
        self.channel_axis = axis;
        self
    }

    /// Builds the INT8 tensor.
    pub fn build(self) -> Int8Tensor {
        if let Some(data) = self.data {
            let scale = self.scale.unwrap_or(1.0 / 127.0);
            Int8Tensor::new(data, self.shape, scale, self.zero_point, self.symmetric)
        } else if let Some(float_data) = self.float_data {
            let scheme = if self.symmetric {
                QuantScheme::SymmetricInt8
            } else {
                QuantScheme::AsymmetricInt8
            };

            if self.per_channel {
                // Per-channel quantization
                let num_channels = self.shape[self.channel_axis];
                let channel_size: usize = self.shape[self.channel_axis + 1..].iter().product();
                let channel_size = channel_size.max(1);

                let mut scales = Vec::with_capacity(num_channels);
                let mut zero_points = Vec::with_capacity(num_channels);
                let mut quantized = vec![0i8; float_data.len()];

                for c in 0..num_channels {
                    let start = c * channel_size;
                    let end = start + channel_size;
                    let channel_data = &float_data[start..end];

                    let max_abs = channel_data
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0_f64, f64::max);
                    let scale = if max_abs > 1e-10 { max_abs / 127.0 } else { 1e-10 };

                    for (i, &v) in channel_data.iter().enumerate() {
                        let q = (v / scale).round() as i32;
                        quantized[start + i] = q.clamp(-127, 127) as i8;
                    }

                    scales.push(scale);
                    zero_points.push(0i8);
                }

                Int8Tensor::per_channel(
                    quantized,
                    self.shape,
                    scales,
                    zero_points,
                    self.channel_axis,
                    self.symmetric,
                )
            } else {
                Int8Tensor::from_float_data(&float_data, self.shape, scheme)
            }
        } else {
            Int8Tensor::zeros(self.shape, self.scale.unwrap_or(1.0 / 127.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_quantization() {
        let data = vec![0.5, -0.5, 1.0, -1.0];
        let tensor = Int8Tensor::from_float_data(&data, vec![4], QuantScheme::SymmetricInt8);

        assert!(tensor.is_symmetric());
        assert_eq!(tensor.zero_point(), 0);
        assert_eq!(tensor.len(), 4);

        // Dequantize and check accuracy
        let dq = tensor.dequantize_raw();
        for (orig, recovered) in data.iter().zip(dq.iter()) {
            assert!((orig - recovered).abs() < 0.02); // Within 2% for INT8
        }
    }

    #[test]
    fn test_asymmetric_quantization() {
        let data = vec![0.0, 0.5, 0.75, 1.0];
        let tensor = Int8Tensor::from_float_data(&data, vec![4], QuantScheme::AsymmetricInt8);

        assert!(!tensor.is_symmetric());
        assert_eq!(tensor.len(), 4);

        // Asymmetric quantization has larger error tolerance
        let dq = tensor.dequantize_raw();
        for (orig, recovered) in data.iter().zip(dq.iter()) {
            // Allow up to 0.5 error for asymmetric INT8 with small dynamic range
            assert!((orig - recovered).abs() < 0.5, "orig={}, recovered={}", orig, recovered);
        }
    }

    #[test]
    fn test_builder() {
        let tensor = Int8TensorBuilder::new(vec![2, 2])
            .with_float_data(vec![0.1, 0.2, 0.3, 0.4])
            .symmetric()
            .build();

        assert!(tensor.is_symmetric());
        assert_eq!(tensor.shape(), &vec![2, 2]);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn test_per_channel_quantization() {
        let data = vec![0.1, 0.2, 0.5, 1.0]; // 2 channels of 2 values each
        let tensor = Int8TensorBuilder::new(vec![2, 2])
            .with_float_data(data.clone())
            .per_channel(0)
            .symmetric()
            .build();

        assert!(tensor.is_per_channel());
        assert!(tensor.per_channel_scales().is_some());

        let dq = tensor.dequantize_raw();
        for (orig, recovered) in data.iter().zip(dq.iter()) {
            assert!((orig - recovered).abs() < 0.02);
        }
    }

    #[test]
    fn test_transpose() {
        let data = vec![1, 2, 3, 4];
        let tensor = Int8Tensor::new(data, vec![2, 2], 0.01, 0, true);
        let transposed = tensor.transpose();

        assert_eq!(transposed.shape(), &vec![2, 2]);
        assert_eq!(transposed.get(&[0, 0]), Some(1));
        assert_eq!(transposed.get(&[0, 1]), Some(3));
        assert_eq!(transposed.get(&[1, 0]), Some(2));
        assert_eq!(transposed.get(&[1, 1]), Some(4));
    }

    #[test]
    fn test_error_tracking() {
        let data = vec![0.5, -0.5];
        let mut tensor = Int8Tensor::from_float_data(&data, vec![2], QuantScheme::SymmetricInt8);

        let initial_error = tensor.total_error();
        assert!(initial_error > 0.0);

        tensor.add_error(0.01);
        assert!(tensor.total_error() > initial_error);
    }

    #[test]
    fn test_statistics() {
        let tensor = Int8Tensor::new(vec![-50, 0, 50, 100], vec![4], 0.01, 0, true);
        let stats = tensor.statistics();

        assert_eq!(stats.min_quantized, -50);
        assert_eq!(stats.max_quantized, 100);
        assert!(stats.scale > 0.0);
    }
}
