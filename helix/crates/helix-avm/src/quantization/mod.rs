//! Quantization Module.
//!
//! Provides comprehensive quantization support for neural network operations,
//! enabling efficient inference and training while tracking error bounds.
//!
//! # Overview
//!
//! This module implements:
//! - **Tensor Types**: Dedicated INT8, INT4, and FP8 tensor representations
//! - **Operations**: Specialized quantized operations (matmul, activations, normalization)
//! - **Schemes**: Different quantization formats (INT8, INT4, FP8)
//! - **Static Quantization**: Pre-calibrated quantization with fixed parameters
//! - **Dynamic Quantization**: Runtime quantization with adaptive parameters
//! - **Calibration**: Statistical analysis for optimal quantization parameters
//! - **Mixed Precision**: Support for running different parts at different precisions
//!
//! # Error Tracking
//!
//! All quantization operations properly track error bounds, which is essential
//! for HELIX's approximate computing model. The quantization error at each step
//! is propagated through the computation graph.
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::quantization::{StaticQuantizer, QuantConfig, Int8Tensor};
//!
//! // Create INT8 tensor from float data
//! let tensor = Int8Tensor::from_float_data(&data, shape, QuantScheme::SymmetricInt8);
//!
//! // Create a quantizer
//! let mut quantizer = StaticQuantizer::new(QuantConfig::int8_static());
//!
//! // Calibrate with representative data
//! let calibration_data = calibrator.finalize();
//! quantizer.calibrate(&calibration_data);
//!
//! // Quantize weights and activations
//! let quantized_weights = quantizer.quantize_weights("layer1", &weights);
//! ```

pub mod calibration;
pub mod circuit_quantizer;
pub mod dequantize;
pub mod dynamic;
pub mod fp8_ops;
pub mod fp8_tensor;
pub mod int4_ops;
pub mod int4_tensor;
pub mod int8_ops;
pub mod int8_tensor;
pub mod mixed_precision;
pub mod ops;
pub mod quantize;
pub mod schemes;
pub mod static_quant;

// Re-export circuit quantizer
pub use circuit_quantizer::CircuitQuantizer;

// Re-export commonly used types
pub use calibration::{CalibrationData, Calibrator, Observer};
pub use dequantize::{dequantize_scalar, dequantize_tensor};
pub use dynamic::{DynamicQuantizer, DynamicQuantizedLinear, PerTokenDynamicQuantizer};
pub use fp8_ops::{fp8_add, fp8_cast, fp8_gelu, fp8_linear, fp8_matmul, fp8_relu, fp8_scale};
pub use fp8_tensor::{Fp8Format, Fp8Tensor};
pub use int4_ops::{
    compute_int4_quantization_error, int4_add, int4_gelu, int4_int8_matmul, int4_linear,
    int4_matmul, int4_mul, int4_relu, int4_requantize, int4_scale, Int4QuantConfig,
    Int4QuantizationErrorMetrics,
};
pub use int4_tensor::{Int4Tensor, Int4TensorBuilder, Int4TensorStats};
pub use int8_ops::{
    compute_quantization_error, int8_add, int8_batched_matmul, int8_concat, int8_gelu,
    int8_layer_norm, int8_leaky_relu, int8_linear, int8_matmul, int8_matvec, int8_mean,
    int8_mul, int8_relu, int8_requantize, int8_rms_norm, int8_scale, int8_sigmoid, int8_silu,
    int8_softmax, int8_sub, int8_sum, int8_tanh, QuantizationErrorMetrics, QuantizedOpError,
};
pub use int8_tensor::{Int8Tensor, Int8TensorBuilder, Int8TensorStats};
pub use mixed_precision::{
    ActivationType, ExecutionStats, LayerPrecisionConfig, MemoryFootprint,
    MixedPrecisionConfig, MixedPrecisionExecutor, MixedPrecisionTensor, PrecisionLevel,
};
pub use ops::{
    quantized_add, quantized_gelu, quantized_layer_norm, quantized_linear, quantized_matmul,
    quantized_relu, quantized_sigmoid, quantized_softmax,
};
pub use quantize::{
    compute_per_channel_quant_params, compute_quant_params, fake_quantize, fake_quantize_tensor,
    quantize_scalar, quantize_tensor, quantize_tensor_from_range, quantize_tensor_int8,
    quantize_to_int8, quantize_with_error, QuantizedTensor,
};
pub use schemes::{
    CalibrationMethod, ObserverType, QuantConfig, QuantScheme, TensorQuantParams,
};
pub use static_quant::{
    fake_quantize_weights, quantize_weights_static, StaticQuantizedLayer, StaticQuantizedModel,
    StaticQuantizer,
};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use super::{
        CalibrationData, Calibrator, DynamicQuantizer, Fp8Format, Fp8Tensor, Int4Tensor,
        Int8Tensor, MixedPrecisionConfig, MixedPrecisionExecutor, MixedPrecisionTensor, Observer,
        PrecisionLevel, QuantConfig, QuantScheme, QuantizedTensor, StaticQuantizer,
        TensorQuantParams,
    };
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use helix_core::types::{BoundedTensor, BoundedValue};

    #[test]
    fn test_end_to_end_static_quantization() {
        // Create sample data
        let weights: Vec<BoundedValue<f64>> = vec![0.1, 0.2, -0.1, 0.3]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let weight_tensor = BoundedTensor::new(weights, vec![2, 2]);

        // Calibrate
        let mut calibrator = Calibrator::int8();
        calibrator.record_weights("layer1", &weight_tensor);
        let cal_data = calibrator.finalize();

        // Create quantizer and calibrate
        let mut quantizer = StaticQuantizer::int8();
        quantizer.calibrate(&cal_data);

        // Quantize weights
        let quantized = quantizer.quantize_weights("layer1", &weight_tensor);
        assert!(quantized.is_some());

        // Dequantize and verify
        let recovered = quantizer.dequantize(&quantized.unwrap());
        for (orig, rec) in weight_tensor.data().iter().zip(recovered.data().iter()) {
            let error = (orig.value() - rec.value()).abs();
            assert!(error < 0.01, "Error too large: {}", error);
        }
    }

    #[test]
    fn test_end_to_end_dynamic_quantization() {
        // Create sample activations
        let activations: Vec<BoundedValue<f64>> = vec![0.5, 1.0, 0.0, -0.5, 2.0, 1.5]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let activation_tensor = BoundedTensor::new(activations, vec![2, 3]);

        // Create dynamic quantizer
        let mut quantizer = DynamicQuantizer::int8();

        // Quantize activations (computes params at runtime)
        let quantized = quantizer.quantize_activations(&activation_tensor);
        assert_eq!(quantized.len(), 6);

        // Dequantize
        let recovered = quantizer.dequantize(&quantized);
        assert_eq!(recovered.data().len(), 6);
    }

    #[test]
    fn test_quantized_ops_chain() {
        // Simulate a simple linear layer with ReLU
        let input_data = vec![10, 20, 30, 40];
        let weight_data = vec![1, 0, 0, 1];
        
        let input_params = TensorQuantParams::new(QuantScheme::SymmetricInt8, 0.01, 0);
        let weight_params = TensorQuantParams::new(QuantScheme::SymmetricInt8, 0.01, 0);
        let output_params = TensorQuantParams::new(QuantScheme::SymmetricInt8, 0.01, 0);
        
        let input = QuantizedTensor::new(input_data, vec![2, 2], input_params);
        let weight = QuantizedTensor::new(weight_data, vec![2, 2], weight_params);
        
        // Matmul
        let matmul_result = quantized_matmul(&input, &weight, &output_params);
        
        // ReLU
        let relu_result = quantized_relu(&matmul_result);
        
        // All values should be non-negative
        for &v in &relu_result.data {
            assert!(v >= 0);
        }
    }

    #[test]
    fn test_error_propagation() {
        let params = TensorQuantParams::new(QuantScheme::SymmetricInt8, 0.01, 0);
        
        // Error should be half the scale
        let error = params.quantization_error();
        assert!((error - 0.005).abs() < 1e-10);
        
        // Computation error compounds
        let input_params = TensorQuantParams::from_range(QuantScheme::SymmetricInt8, -1.0, 1.0);
        let weight_params = TensorQuantParams::from_range(QuantScheme::SymmetricInt8, -1.0, 1.0);
        let output_params = TensorQuantParams::from_range(QuantScheme::SymmetricInt8, -1.0, 1.0);
        
        let op_error = ops::compute_op_error(&input_params, &weight_params, &output_params, 32);
        assert!(op_error > 0.0);
    }
}
