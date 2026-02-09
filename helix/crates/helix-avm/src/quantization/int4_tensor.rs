//! INT4 Tensor Type with Scale/Zero-Point.
//!
//! Provides a dedicated INT4 tensor representation for extremely memory-efficient
//! quantized inference. INT4 uses packed storage where two 4-bit values share a byte.
//!
//! INT4 quantization is more aggressive than INT8 but can still achieve good accuracy
//! for many neural network operations, especially when combined with INT8 activations.

use helix_core::types::{BoundedTensor, BoundedValue, Shape};

use super::schemes::{QuantScheme, TensorQuantParams};

/// INT4 quantized tensor with scale and zero-point parameters.
///
/// Values are stored in packed format: two 4-bit values per byte.
/// - Low nibble: even indices (0, 2, 4, ...)
/// - High nibble: odd indices (1, 3, 5, ...)
///
/// For symmetric INT4: values range from -8 to 7 (uses -7 to 7 for symmetric)
/// For asymmetric INT4: values range from 0 to 15
#[derive(Debug, Clone)]
pub struct Int4Tensor {
    /// Packed INT4 data (2 values per byte).
    packed_data: Vec<u8>,
    /// Shape of the tensor (in terms of 4-bit values).
    shape: Shape,
    /// Strides for indexing.
    strides: Vec<usize>,
    /// Total number of 4-bit values.
    num_values: usize,
    /// Scale factor for dequantization.
    scale: f64,
    /// Zero-point offset (in 4-bit range).
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
    /// Group size for group-wise quantization (used in LLM quantization).
    group_size: Option<usize>,
    /// Per-group scales if group-wise quantization is used.
    group_scales: Option<Vec<f64>>,
}

impl Int4Tensor {
    /// Creates a new INT4 tensor with per-tensor quantization.
    pub fn new(
        data: Vec<i8>,
        shape: Shape,
        scale: f64,
        zero_point: i8,
        symmetric: bool,
    ) -> Self {
        let num_values = data.len();
        let packed_data = pack_int4(&data);
        let strides = compute_strides(&shape);

        Self {
            packed_data,
            shape,
            strides,
            num_values,
            scale,
            zero_point,
            symmetric,
            accumulated_error: scale / 2.0,
            per_channel_scales: None,
            per_channel_zero_points: None,
            per_channel_axis: None,
            group_size: None,
            group_scales: None,
        }
    }

    /// Creates an INT4 tensor with per-channel quantization.
    pub fn per_channel(
        data: Vec<i8>,
        shape: Shape,
        scales: Vec<f64>,
        zero_points: Vec<i8>,
        axis: usize,
        symmetric: bool,
    ) -> Self {
        let num_values = data.len();
        let packed_data = pack_int4(&data);
        let strides = compute_strides(&shape);
        let max_scale = scales.iter().cloned().fold(0.0_f64, f64::max);

        Self {
            packed_data,
            shape,
            strides,
            num_values,
            scale: max_scale,
            zero_point: 0,
            symmetric,
            accumulated_error: max_scale / 2.0,
            per_channel_scales: Some(scales),
            per_channel_zero_points: Some(zero_points),
            per_channel_axis: Some(axis),
            group_size: None,
            group_scales: None,
        }
    }

    /// Creates an INT4 tensor with group-wise quantization.
    ///
    /// Group-wise quantization divides the tensor into groups along the last dimension
    /// and uses a separate scale for each group. This is common in LLM quantization.
    pub fn group_wise(
        data: Vec<i8>,
        shape: Shape,
        group_size: usize,
        group_scales: Vec<f64>,
        symmetric: bool,
    ) -> Self {
        let num_values = data.len();
        let packed_data = pack_int4(&data);
        let strides = compute_strides(&shape);
        let max_scale = group_scales.iter().cloned().fold(0.0_f64, f64::max);

        Self {
            packed_data,
            shape,
            strides,
            num_values,
            scale: max_scale,
            zero_point: 0,
            symmetric,
            accumulated_error: max_scale / 2.0,
            per_channel_scales: None,
            per_channel_zero_points: None,
            per_channel_axis: None,
            group_size: Some(group_size),
            group_scales: Some(group_scales),
        }
    }

    /// Creates an INT4 tensor from a BoundedTensor using symmetric quantization.
    pub fn from_bounded_symmetric(tensor: &BoundedTensor) -> Self {
        let data = tensor.data();
        let shape = tensor.shape().clone();

        let max_abs = data
            .iter()
            .map(|v| v.value().abs())
            .fold(0.0_f64, f64::max);

        // Scale for symmetric INT4 (range -7 to 7)
        let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1e-10 };

        let quantized: Vec<i8> = data
            .iter()
            .map(|v| {
                let q = (v.value() / scale).round() as i32;
                q.clamp(-7, 7) as i8
            })
            .collect();

        Self::new(quantized, shape, scale, 0, true)
    }

    /// Creates an INT4 tensor from a BoundedTensor using asymmetric quantization.
    pub fn from_bounded_asymmetric(tensor: &BoundedTensor) -> Self {
        let data = tensor.data();
        let shape = tensor.shape().clone();

        let (min_val, max_val) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |acc, v| {
            (acc.0.min(v.value()), acc.1.max(v.value()))
        });

        let scale = if (max_val - min_val).abs() > 1e-10 {
            (max_val - min_val) / 15.0
        } else {
            1e-10
        };
        let zero_point = ((-min_val / scale).round() as i32).clamp(0, 15) as i8;

        let quantized: Vec<i8> = data
            .iter()
            .map(|v| {
                let q = (v.value() / scale).round() as i32 + zero_point as i32;
                q.clamp(0, 15) as i8
            })
            .collect();

        Self::new(quantized, shape, scale, zero_point, false)
    }

    /// Creates an INT4 tensor from raw float data with automatic calibration.
    pub fn from_float_data(data: &[f64], shape: Shape, scheme: QuantScheme) -> Self {
        match scheme {
            QuantScheme::SymmetricInt4 => {
                let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1e-10 };
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&v| {
                        let q = (v / scale).round() as i32;
                        q.clamp(-7, 7) as i8
                    })
                    .collect();
                Self::new(quantized, shape, scale, 0, true)
            }
            QuantScheme::AsymmetricInt4 => {
                let (min_val, max_val) = data
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
                        (acc.0.min(v), acc.1.max(v))
                    });
                let scale = if (max_val - min_val).abs() > 1e-10 {
                    (max_val - min_val) / 15.0
                } else {
                    1e-10
                };
                let zero_point = ((-min_val / scale).round() as i32).clamp(0, 15) as i8;
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&v| {
                        let q = (v / scale).round() as i32 + zero_point as i32;
                        q.clamp(0, 15) as i8
                    })
                    .collect();
                Self::new(quantized, shape, scale, zero_point, false)
            }
            _ => Self::from_float_data(data, shape, QuantScheme::SymmetricInt4),
        }
    }

    /// Creates a zero-filled INT4 tensor.
    pub fn zeros(shape: Shape, scale: f64) -> Self {
        let num_values: usize = shape.iter().product();
        Self::new(vec![0i8; num_values], shape, scale, 0, true)
    }

    /// Returns the packed data.
    pub fn packed_data(&self) -> &[u8] {
        &self.packed_data
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

    /// Returns the number of 4-bit values.
    pub fn len(&self) -> usize {
        self.num_values
    }

    /// Returns true if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.num_values == 0
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

    /// Returns whether this uses group-wise quantization.
    pub fn is_group_wise(&self) -> bool {
        self.group_size.is_some()
    }

    /// Returns the group size if group-wise quantization is used.
    pub fn group_size(&self) -> Option<usize> {
        self.group_size
    }

    /// Gets a single INT4 value at the given flat index.
    pub fn get_flat(&self, index: usize) -> Option<i8> {
        if index >= self.num_values {
            return None;
        }
        Some(unpack_int4_value(&self.packed_data, index))
    }

    /// Gets a value at a multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Option<i8> {
        let flat_idx = self.flat_index(index)?;
        self.get_flat(flat_idx)
    }

    /// Sets a single INT4 value at the given flat index.
    pub fn set_flat(&mut self, index: usize, value: i8) -> Option<()> {
        if index >= self.num_values {
            return None;
        }
        let clamped = if self.symmetric {
            value.clamp(-7, 7)
        } else {
            value.clamp(0, 15)
        };
        pack_int4_value(&mut self.packed_data, index, clamped);
        Some(())
    }

    /// Sets a value at a multi-dimensional index.
    pub fn set(&mut self, index: &[usize], value: i8) -> Option<()> {
        let flat_idx = self.flat_index(index)?;
        self.set_flat(flat_idx, value)
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

    /// Unpacks all INT4 values to a vector.
    pub fn unpack(&self) -> Vec<i8> {
        unpack_int4(&self.packed_data, self.num_values)
    }

    /// Dequantizes the entire tensor to a BoundedTensor.
    pub fn dequantize(&self) -> BoundedTensor {
        let unpacked = self.unpack();
        let quant_error = self.quantization_error();

        let bounded_data: Vec<BoundedValue<f64>> = if self.is_per_channel() {
            let axis = self.per_channel_axis.unwrap();
            let channel_size = self.shape[axis];
            let stride: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);

            unpacked
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
        } else if self.is_group_wise() {
            let group_size = self.group_size.unwrap();
            let group_scales = self.group_scales.as_ref().unwrap();

            unpacked
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let group_idx = i / group_size;
                    let scale = group_scales.get(group_idx).copied().unwrap_or(self.scale);
                    let dq = (v as i32 - self.zero_point as i32) as f64 * scale;
                    BoundedValue::<f64>::with_absolute_error(dq, scale / 2.0 + self.accumulated_error)
                })
                .collect()
        } else {
            unpacked
                .iter()
                .map(|&v| {
                    let dq = (v as i32 - self.zero_point as i32) as f64 * self.scale;
                    BoundedValue::<f64>::with_absolute_error(dq, quant_error + self.accumulated_error)
                })
                .collect()
        };

        BoundedTensor::new(bounded_data, self.shape.clone())
    }

    /// Dequantizes to raw f64 values without error tracking.
    pub fn dequantize_raw(&self) -> Vec<f64> {
        let unpacked = self.unpack();

        if self.is_per_channel() {
            let axis = self.per_channel_axis.unwrap();
            let channel_size = self.shape[axis];
            let stride: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);

            unpacked
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let channel = (i / stride) % channel_size;
                    let scale = self.per_channel_scales.as_ref().unwrap()[channel];
                    let zp = self.per_channel_zero_points.as_ref().unwrap()[channel];
                    (v as i32 - zp as i32) as f64 * scale
                })
                .collect()
        } else if self.is_group_wise() {
            let group_size = self.group_size.unwrap();
            let group_scales = self.group_scales.as_ref().unwrap();

            unpacked
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let group_idx = i / group_size;
                    let scale = group_scales.get(group_idx).copied().unwrap_or(self.scale);
                    (v as i32 - self.zero_point as i32) as f64 * scale
                })
                .collect()
        } else {
            unpacked
                .iter()
                .map(|&v| (v as i32 - self.zero_point as i32) as f64 * self.scale)
                .collect()
        }
    }

    /// Returns the quantization error bound.
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

    /// Reshapes the tensor.
    pub fn reshape(&self, new_shape: Shape) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.num_values, new_size, "Cannot reshape: size mismatch");

        Self {
            packed_data: self.packed_data.clone(),
            shape: new_shape.clone(),
            strides: compute_strides(&new_shape),
            num_values: self.num_values,
            scale: self.scale,
            zero_point: self.zero_point,
            symmetric: self.symmetric,
            accumulated_error: self.accumulated_error,
            per_channel_scales: None,
            per_channel_zero_points: None,
            per_channel_axis: None,
            group_size: None,
            group_scales: None,
        }
    }

    /// Converts to TensorQuantParams for compatibility.
    pub fn to_quant_params(&self) -> TensorQuantParams {
        TensorQuantParams::new(
            if self.symmetric {
                QuantScheme::SymmetricInt4
            } else {
                QuantScheme::AsymmetricInt4
            },
            self.scale,
            self.zero_point as i64,
        )
    }

    /// Returns statistics about the quantized values.
    pub fn statistics(&self) -> Int4TensorStats {
        let unpacked = self.unpack();
        let min = *unpacked.iter().min().unwrap_or(&0);
        let max = *unpacked.iter().max().unwrap_or(&0);
        let sum: i64 = unpacked.iter().map(|&v| v as i64).sum();
        let mean = sum as f64 / unpacked.len().max(1) as f64;

        let dq = self.dequantize_raw();
        let dq_min = dq.iter().cloned().fold(f64::INFINITY, f64::min);
        let dq_max = dq.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Int4TensorStats {
            min_quantized: min,
            max_quantized: max,
            mean_quantized: mean,
            min_dequantized: dq_min,
            max_dequantized: dq_max,
            scale: self.scale,
            zero_point: self.zero_point,
            sparsity: unpacked.iter().filter(|&&v| v == self.zero_point).count() as f64
                / unpacked.len().max(1) as f64,
            compression_ratio: 8.0 / 4.0, // 2x compared to INT8
        }
    }

    /// Computes memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.packed_data.len()
    }

    /// Computes theoretical memory usage for the same data in f32.
    pub fn fp32_equivalent_bytes(&self) -> usize {
        self.num_values * 4
    }

    /// Computes the compression ratio compared to f32.
    pub fn compression_ratio(&self) -> f64 {
        self.fp32_equivalent_bytes() as f64 / self.memory_bytes() as f64
    }
}

/// Statistics for an INT4 tensor.
#[derive(Debug, Clone)]
pub struct Int4TensorStats {
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
    /// Compression ratio vs INT8.
    pub compression_ratio: f64,
}

/// Packs INT4 values into bytes (2 values per byte).
fn pack_int4(data: &[i8]) -> Vec<u8> {
    let num_bytes = (data.len() + 1) / 2;
    let mut packed = vec![0u8; num_bytes];

    for (i, &value) in data.iter().enumerate() {
        // Convert to unsigned 4-bit (0-15)
        let unsigned_val = (value as i32 & 0x0F) as u8;
        let byte_idx = i / 2;
        let nibble_idx = i % 2;

        if nibble_idx == 0 {
            // Low nibble
            packed[byte_idx] = (packed[byte_idx] & 0xF0) | unsigned_val;
        } else {
            // High nibble
            packed[byte_idx] = (packed[byte_idx] & 0x0F) | (unsigned_val << 4);
        }
    }

    packed
}

/// Unpacks bytes to INT4 values.
fn unpack_int4(packed: &[u8], num_values: usize) -> Vec<i8> {
    let mut data = Vec::with_capacity(num_values);

    for i in 0..num_values {
        data.push(unpack_int4_value(packed, i));
    }

    data
}

/// Gets a single INT4 value from packed data.
fn unpack_int4_value(packed: &[u8], index: usize) -> i8 {
    let byte_idx = index / 2;
    let nibble_idx = index % 2;

    let byte = packed.get(byte_idx).copied().unwrap_or(0);
    let unsigned_val = if nibble_idx == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };

    // Sign-extend from 4 bits to 8 bits for symmetric
    if unsigned_val > 7 {
        (unsigned_val as i8) - 16
    } else {
        unsigned_val as i8
    }
}

/// Sets a single INT4 value in packed data.
fn pack_int4_value(packed: &mut [u8], index: usize, value: i8) {
    let byte_idx = index / 2;
    let nibble_idx = index % 2;
    let unsigned_val = (value as i32 & 0x0F) as u8;

    if byte_idx < packed.len() {
        if nibble_idx == 0 {
            packed[byte_idx] = (packed[byte_idx] & 0xF0) | unsigned_val;
        } else {
            packed[byte_idx] = (packed[byte_idx] & 0x0F) | (unsigned_val << 4);
        }
    }
}

/// Computes strides for row-major layout.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Builder for creating INT4 tensors with various options.
#[derive(Debug, Clone)]
pub struct Int4TensorBuilder {
    float_data: Option<Vec<f64>>,
    int4_data: Option<Vec<i8>>,
    shape: Shape,
    scale: Option<f64>,
    zero_point: i8,
    symmetric: bool,
    group_size: Option<usize>,
}

impl Int4TensorBuilder {
    /// Creates a new builder with the given shape.
    pub fn new(shape: Shape) -> Self {
        Self {
            float_data: None,
            int4_data: None,
            shape,
            scale: None,
            zero_point: 0,
            symmetric: true,
            group_size: None,
        }
    }

    /// Sets float data to be quantized.
    pub fn with_float_data(mut self, data: Vec<f64>) -> Self {
        self.float_data = Some(data);
        self
    }

    /// Sets pre-quantized INT4 data.
    pub fn with_int4_data(mut self, data: Vec<i8>) -> Self {
        self.int4_data = Some(data);
        self
    }

    /// Sets the scale.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
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

    /// Enables group-wise quantization with the specified group size.
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = Some(group_size);
        self
    }

    /// Builds the INT4 tensor.
    pub fn build(self) -> Int4Tensor {
        if let Some(int4_data) = self.int4_data {
            let scale = self.scale.unwrap_or(1.0 / 7.0);
            Int4Tensor::new(int4_data, self.shape, scale, self.zero_point, self.symmetric)
        } else if let Some(float_data) = self.float_data {
            if let Some(group_size) = self.group_size {
                // Group-wise quantization
                let num_groups = (float_data.len() + group_size - 1) / group_size;
                let mut quantized = Vec::with_capacity(float_data.len());
                let mut group_scales = Vec::with_capacity(num_groups);

                for g in 0..num_groups {
                    let start = g * group_size;
                    let end = (start + group_size).min(float_data.len());
                    let group_data = &float_data[start..end];

                    let max_abs = group_data
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0_f64, f64::max);
                    let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1e-10 };

                    for &v in group_data {
                        let q = (v / scale).round() as i32;
                        quantized.push(q.clamp(-7, 7) as i8);
                    }

                    group_scales.push(scale);
                }

                Int4Tensor::group_wise(quantized, self.shape, group_size, group_scales, self.symmetric)
            } else {
                let scheme = if self.symmetric {
                    QuantScheme::SymmetricInt4
                } else {
                    QuantScheme::AsymmetricInt4
                };
                Int4Tensor::from_float_data(&float_data, self.shape, scheme)
            }
        } else {
            Int4Tensor::zeros(self.shape, self.scale.unwrap_or(1.0 / 7.0))
        }
    }
}

/// Converts an INT8 tensor to INT4.
pub fn int8_to_int4(int8_tensor: &super::int8_tensor::Int8Tensor) -> Int4Tensor {
    let dq = int8_tensor.dequantize_raw();
    Int4Tensor::from_float_data(&dq, int8_tensor.shape().clone(), QuantScheme::SymmetricInt4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_quantization() {
        let data = vec![0.5, -0.5, 1.0, -1.0];
        let tensor = Int4Tensor::from_float_data(&data, vec![4], QuantScheme::SymmetricInt4);

        assert!(tensor.is_symmetric());
        assert_eq!(tensor.zero_point(), 0);
        assert_eq!(tensor.len(), 4);

        let dq = tensor.dequantize_raw();
        for (orig, recovered) in data.iter().zip(dq.iter()) {
            // INT4 has larger quantization error
            assert!((orig - recovered).abs() < 0.2);
        }
    }

    #[test]
    fn test_packing_unpacking() {
        let original = vec![1, -2, 3, -4, 5, -6, 7];
        let packed = pack_int4(&original);
        let unpacked = unpack_int4(&packed, original.len());

        // Values should survive round-trip within 4-bit range
        for (orig, unpacked_val) in original.iter().zip(unpacked.iter()) {
            assert_eq!(*orig, *unpacked_val);
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let tensor = Int4Tensor::from_float_data(&data, vec![1000], QuantScheme::SymmetricInt4);

        // Should use ~500 bytes (2 values per byte)
        assert_eq!(tensor.memory_bytes(), 500);
        assert!(tensor.compression_ratio() > 7.5); // ~8x compression vs f32
    }

    #[test]
    fn test_group_wise_quantization() {
        let data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0];
        let tensor = Int4TensorBuilder::new(vec![8])
            .with_float_data(data.clone())
            .with_group_size(4)
            .symmetric()
            .build();

        assert!(tensor.is_group_wise());
        assert_eq!(tensor.group_size(), Some(4));

        let dq = tensor.dequantize_raw();
        // Group-wise should give better accuracy than single scale
        for (orig, recovered) in data.iter().zip(dq.iter()) {
            assert!((orig - recovered).abs() < 1.0);
        }
    }

    #[test]
    fn test_builder() {
        let tensor = Int4TensorBuilder::new(vec![2, 2])
            .with_float_data(vec![0.1, 0.2, 0.3, 0.4])
            .symmetric()
            .build();

        assert!(tensor.is_symmetric());
        assert_eq!(tensor.shape(), &vec![2, 2]);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn test_statistics() {
        let tensor = Int4Tensor::new(vec![-3, 0, 3, 7], vec![4], 0.1, 0, true);
        let stats = tensor.statistics();

        assert_eq!(stats.min_quantized, -3);
        assert_eq!(stats.max_quantized, 7);
        assert!(stats.compression_ratio > 1.0);
    }
}
