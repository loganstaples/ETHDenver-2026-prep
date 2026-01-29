//! Quantization Operations.
//!
//! Core functions for converting floating-point values to quantized representations.

use helix_core::types::{BoundedTensor, BoundedValue, Shape};

use super::schemes::{QuantScheme, TensorQuantParams};

/// Result of a quantization operation.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized integer data.
    pub data: Vec<i64>,
    /// Shape of the tensor.
    pub shape: Shape,
    /// Quantization parameters.
    pub params: TensorQuantParams,
    /// Original shape before any reshaping.
    pub original_shape: Shape,
}

impl QuantizedTensor {
    /// Creates a new quantized tensor.
    pub fn new(data: Vec<i64>, shape: Shape, params: TensorQuantParams) -> Self {
        Self {
            original_shape: shape.clone(),
            data,
            shape,
            params,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the quantization scheme.
    pub fn scheme(&self) -> QuantScheme {
        self.params.scheme
    }

    /// Returns the shape dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// Quantizes a single float value to an integer.
pub fn quantize_scalar(value: f64, scale: f64, zero_point: i64, scheme: QuantScheme) -> i64 {
    let (qmin, qmax) = scheme.range();
    let quantized = (value / scale).round() as i64 + zero_point;
    quantized.clamp(qmin, qmax)
}

/// Dequantizes a single integer value to a float.
pub fn dequantize_scalar(value: i64, scale: f64, zero_point: i64) -> f64 {
    (value - zero_point) as f64 * scale
}

/// Quantizes a value with error tracking.
pub fn quantize_with_error(
    value: BoundedValue<f64>,
    scale: f64,
    zero_point: i64,
    scheme: QuantScheme,
) -> (i64, BoundedValue<f64>) {
    let quantized = quantize_scalar(value.value(), scale, zero_point, scheme);
    let dequantized = dequantize_scalar(quantized, scale, zero_point);
    
    // Quantization error is at most half the scale
    let quant_error = scale / 2.0;
    let total_error = value.absolute_error() + quant_error;
    
    (quantized, BoundedValue::<f64>::with_absolute_error(dequantized, total_error))
}

/// Quantizes a tensor of float values.
pub fn quantize_tensor(
    tensor: &BoundedTensor,
    params: &TensorQuantParams,
) -> QuantizedTensor {
    let data = tensor.data();
    let shape = tensor.shape().clone();
    
    let quantized_data: Vec<i64> = if params.is_per_channel() {
        // Per-channel quantization
        let axis = params.axis.unwrap();
        let channel_size = shape[axis];
        let stride: usize = shape[axis + 1..].iter().product();
        
        data.iter()
            .enumerate()
            .map(|(i, v)| {
                let channel = (i / stride) % channel_size;
                quantize_scalar(
                    v.value(),
                    params.scales[channel],
                    params.zero_points[channel],
                    params.scheme,
                )
            })
            .collect()
    } else {
        // Per-tensor quantization
        let scale = params.scales[0];
        let zero_point = params.zero_points[0];
        
        data.iter()
            .map(|v| quantize_scalar(v.value(), scale, zero_point, params.scheme))
            .collect()
    };
    
    QuantizedTensor::new(quantized_data, shape, params.clone())
}

/// Quantizes a tensor given min/max range.
pub fn quantize_tensor_from_range(
    tensor: &BoundedTensor,
    scheme: QuantScheme,
    min_val: f64,
    max_val: f64,
) -> QuantizedTensor {
    let params = TensorQuantParams::from_range(scheme, min_val, max_val);
    quantize_tensor(tensor, &params)
}

/// Computes quantization parameters for a tensor.
pub fn compute_quant_params(
    tensor: &BoundedTensor,
    scheme: QuantScheme,
) -> TensorQuantParams {
    let data = tensor.data();
    
    if data.is_empty() {
        return TensorQuantParams::new(scheme, 1.0, 0);
    }
    
    let min_val = data.iter().map(|v| v.value()).fold(f64::INFINITY, f64::min);
    let max_val = data.iter().map(|v| v.value()).fold(f64::NEG_INFINITY, f64::max);
    
    TensorQuantParams::from_range(scheme, min_val, max_val)
}

/// Computes per-channel quantization parameters.
pub fn compute_per_channel_quant_params(
    tensor: &BoundedTensor,
    scheme: QuantScheme,
    axis: usize,
) -> TensorQuantParams {
    let shape = tensor.shape();
    let data = tensor.data();
    
    if data.is_empty() || axis >= shape.len() {
        return TensorQuantParams::new(scheme, 1.0, 0);
    }
    
    let num_channels = shape[axis];
    let stride: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let outer_stride: usize = shape[..axis].iter().product::<usize>().max(1);
    
    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);
    
    for channel in 0..num_channels {
        // Collect values for this channel
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        
        for outer in 0..outer_stride {
            for inner in 0..stride {
                let idx = outer * (num_channels * stride) + channel * stride + inner;
                if idx < data.len() {
                    let v = data[idx].value();
                    min_val = min_val.min(v);
                    max_val = max_val.max(v);
                }
            }
        }
        
        // Compute params for this channel
        let channel_params = TensorQuantParams::from_range(scheme, min_val, max_val);
        scales.push(channel_params.scales[0]);
        zero_points.push(channel_params.zero_points[0]);
    }
    
    TensorQuantParams::per_channel(scheme, scales, zero_points, axis)
}

/// Quantizes a slice of f64 values directly.
pub fn quantize_slice(
    data: &[f64],
    params: &TensorQuantParams,
) -> Vec<i64> {
    data.iter()
        .map(|&v| quantize_scalar(v, params.scales[0], params.zero_points[0], params.scheme))
        .collect()
}

/// Quantizes to INT8 specifically (for performance).
pub fn quantize_to_int8(value: f64, scale: f64, zero_point: i32) -> i8 {
    let quantized = (value / scale).round() as i32 + zero_point;
    quantized.clamp(-128, 127) as i8
}

/// Quantizes to INT4 specifically.
pub fn quantize_to_int4(value: f64, scale: f64, zero_point: i8) -> i8 {
    let quantized = (value / scale).round() as i8 + zero_point;
    quantized.clamp(-8, 7)
}

/// Quantizes a tensor to INT8.
pub fn quantize_tensor_int8(
    data: &[f64],
    scale: f64,
    zero_point: i32,
) -> Vec<i8> {
    data.iter()
        .map(|&v| quantize_to_int8(v, scale, zero_point))
        .collect()
}

/// Fake quantization (quantize then dequantize in one step).
/// Useful for quantization-aware training.
pub fn fake_quantize(
    value: f64,
    scale: f64,
    zero_point: i64,
    scheme: QuantScheme,
) -> f64 {
    let quantized = quantize_scalar(value, scale, zero_point, scheme);
    dequantize_scalar(quantized, scale, zero_point)
}

/// Fake quantization for a tensor.
pub fn fake_quantize_tensor(
    tensor: &BoundedTensor,
    params: &TensorQuantParams,
) -> BoundedTensor {
    let data = tensor.data();
    let shape = tensor.shape().clone();
    let quant_error = params.quantization_error();
    
    let fake_quantized: Vec<BoundedValue<f64>> = data.iter()
        .map(|v| {
            let fq = fake_quantize(
                v.value(),
                params.scales[0],
                params.zero_points[0],
                params.scheme,
            );
            BoundedValue::<f64>::with_absolute_error(fq, v.absolute_error() + quant_error)
        })
        .collect();
    
    BoundedTensor::new(fake_quantized, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_scalar() {
        // scale = 0.01, zero_point = 0
        let q = quantize_scalar(0.5, 0.01, 0, QuantScheme::SymmetricInt8);
        assert_eq!(q, 50);
        
        let q = quantize_scalar(-0.5, 0.01, 0, QuantScheme::SymmetricInt8);
        assert_eq!(q, -50);
    }

    #[test]
    fn test_dequantize_scalar() {
        let d = dequantize_scalar(50, 0.01, 0);
        assert!((d - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fake_quantize() {
        let value = 0.555;
        let scale = 0.01;
        let zero_point = 0;
        
        let fq = fake_quantize(value, scale, zero_point, QuantScheme::SymmetricInt8);
        // 0.555 / 0.01 = 55.5 -> rounds to 56 -> 56 * 0.01 = 0.56
        assert!((fq - 0.56).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_with_error() {
        let value = BoundedValue::<f64>::with_absolute_error(0.5, 0.01);
        let (q, dq) = quantize_with_error(value, 0.01, 0, QuantScheme::SymmetricInt8);
        
        assert_eq!(q, 50);
        assert!((dq.value() - 0.5).abs() < 1e-10);
        // Error should be original error + quantization error
        assert!(dq.absolute_error() >= 0.01);
    }
}
