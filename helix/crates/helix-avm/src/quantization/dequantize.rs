//! Dequantization Operations.
//!
//! Core functions for converting quantized representations back to floating-point
//! with proper error tracking.

use helix_core::types::{BoundedTensor, BoundedValue, Shape};

use super::quantize::QuantizedTensor;
use super::schemes::TensorQuantParams;

/// Dequantizes a single integer value to a float.
pub fn dequantize_scalar(value: i64, scale: f64, zero_point: i64) -> f64 {
    (value - zero_point) as f64 * scale
}

/// Dequantizes a value with error tracking.
pub fn dequantize_with_error(
    value: i64,
    scale: f64,
    zero_point: i64,
    original_error: f64,
) -> BoundedValue<f64> {
    let dequantized = dequantize_scalar(value, scale, zero_point);
    
    // The dequantized value has quantization error plus any original error
    let quant_error = scale / 2.0;
    let total_error = original_error + quant_error;
    
    BoundedValue::<f64>::with_absolute_error(dequantized, total_error)
}

/// Dequantizes a quantized tensor back to a bounded tensor.
pub fn dequantize_tensor(quantized: &QuantizedTensor) -> BoundedTensor {
    let params = &quantized.params;
    let quant_error = params.quantization_error();
    
    let dequantized_data: Vec<BoundedValue<f64>> = if params.is_per_channel() {
        // Per-channel dequantization
        let axis = params.axis.unwrap();
        let shape = &quantized.shape;
        let channel_size = shape[axis];
        let stride: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        
        quantized.data.iter()
            .enumerate()
            .map(|(i, &v)| {
                let channel = (i / stride) % channel_size;
                let scale = params.scales[channel];
                let zero_point = params.zero_points[channel];
                let dq = dequantize_scalar(v, scale, zero_point);
                // Per-channel error bound
                let channel_error = scale / 2.0;
                BoundedValue::<f64>::with_absolute_error(dq, channel_error)
            })
            .collect()
    } else {
        // Per-tensor dequantization
        let scale = params.scales[0];
        let zero_point = params.zero_points[0];
        
        quantized.data.iter()
            .map(|&v| {
                let dq = dequantize_scalar(v, scale, zero_point);
                BoundedValue::<f64>::with_absolute_error(dq, quant_error)
            })
            .collect()
    };
    
    BoundedTensor::new(dequantized_data, quantized.shape.clone())
}

/// Dequantizes a quantized tensor with additional error information.
pub fn dequantize_tensor_with_accumulated_error(
    quantized: &QuantizedTensor,
    accumulated_errors: &[f64],
) -> BoundedTensor {
    let params = &quantized.params;
    let base_quant_error = params.quantization_error();
    
    let dequantized_data: Vec<BoundedValue<f64>> = quantized.data.iter()
        .zip(accumulated_errors.iter())
        .map(|(&v, &acc_error)| {
            let scale = params.scales[0];
            let zero_point = params.zero_points[0];
            let dq = dequantize_scalar(v, scale, zero_point);
            BoundedValue::<f64>::with_absolute_error(dq, acc_error + base_quant_error)
        })
        .collect();
    
    BoundedTensor::new(dequantized_data, quantized.shape.clone())
}

/// Dequantizes INT8 data directly.
pub fn dequantize_int8(value: i8, scale: f64, zero_point: i32) -> f64 {
    (value as i32 - zero_point) as f64 * scale
}

/// Dequantizes INT4 data directly.
pub fn dequantize_int4(value: i8, scale: f64, zero_point: i8) -> f64 {
    (value - zero_point) as f64 * scale
}

/// Dequantizes an INT8 tensor.
pub fn dequantize_int8_tensor(
    data: &[i8],
    scale: f64,
    zero_point: i32,
) -> Vec<f64> {
    data.iter()
        .map(|&v| dequantize_int8(v, scale, zero_point))
        .collect()
}

/// Dequantizes an INT8 tensor with error tracking.
pub fn dequantize_int8_tensor_with_error(
    data: &[i8],
    scale: f64,
    zero_point: i32,
    shape: Shape,
) -> BoundedTensor {
    let quant_error = scale / 2.0;
    
    let bounded_data: Vec<BoundedValue<f64>> = data.iter()
        .map(|&v| {
            let dq = dequantize_int8(v, scale, zero_point);
            BoundedValue::<f64>::with_absolute_error(dq, quant_error)
        })
        .collect();
    
    BoundedTensor::new(bounded_data, shape)
}

/// Dequantizes with requantization parameters.
/// Used when the output of a quantized operation needs different params.
pub fn dequantize_and_requantize(
    value: i64,
    input_scale: f64,
    input_zero_point: i64,
    output_params: &TensorQuantParams,
) -> i64 {
    // Dequantize with input params
    let float_val = dequantize_scalar(value, input_scale, input_zero_point);
    
    // Requantize with output params
    let (qmin, qmax) = output_params.scheme.range();
    let output_scale = output_params.scales[0];
    let output_zero_point = output_params.zero_points[0];
    
    let requantized = (float_val / output_scale).round() as i64 + output_zero_point;
    requantized.clamp(qmin, qmax)
}

/// Batch dequantization for efficiency.
pub fn batch_dequantize(
    data: &[i64],
    params: &TensorQuantParams,
) -> Vec<f64> {
    let scale = params.scales[0];
    let zero_point = params.zero_points[0];
    
    data.iter()
        .map(|&v| dequantize_scalar(v, scale, zero_point))
        .collect()
}

/// Computes the total error after dequantization.
pub fn compute_dequant_error(params: &TensorQuantParams, input_error: f64) -> f64 {
    params.quantization_error() + input_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schemes::QuantScheme;
    use super::super::quantize::quantize_tensor;

    #[test]
    fn test_dequantize_scalar() {
        let value = dequantize_scalar(50, 0.01, 0);
        assert!((value - 0.5).abs() < 1e-10);
        
        let value = dequantize_scalar(128, 0.01, 128);
        assert!(value.abs() < 1e-10); // Zero point at 128
    }

    #[test]
    fn test_dequantize_with_error() {
        let dq = dequantize_with_error(50, 0.01, 0, 0.001);
        assert!((dq.value() - 0.5).abs() < 1e-10);
        // Error is quantization error (0.005) + input error (0.001)
        assert!((dq.absolute_error() - 0.006).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip() {
        use helix_core::types::BoundedValue;
        
        // Create bounded tensor
        let data: Vec<BoundedValue<f64>> = vec![0.1, 0.5, -0.3, 0.8]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let tensor = BoundedTensor::new(data, vec![4]);
        
        // Quantize
        let params = TensorQuantParams::from_range(QuantScheme::SymmetricInt8, -1.0, 1.0);
        let quantized = quantize_tensor(&tensor, &params);
        
        // Dequantize
        let recovered = dequantize_tensor(&quantized);
        
        // Check values are close
        for i in 0..4 {
            let original = tensor.data()[i].value();
            let recovered_val = recovered.data()[i].value();
            assert!((original - recovered_val).abs() < 0.01);
        }
    }
}
