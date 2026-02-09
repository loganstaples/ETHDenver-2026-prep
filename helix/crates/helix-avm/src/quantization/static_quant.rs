//! Static Quantization.
//!
//! Pre-calibrated quantization where quantization parameters are determined
//! offline and fixed during inference.

use std::collections::HashMap;

use helix_core::types::BoundedTensor;

use super::calibration::CalibrationData;
use super::quantize::{QuantizedTensor, quantize_tensor, fake_quantize_tensor};
use super::dequantize::dequantize_tensor;
use super::schemes::{QuantConfig, QuantScheme, TensorQuantParams};

/// Static quantizer for a model or layer.
#[derive(Debug)]
pub struct StaticQuantizer {
    /// Configuration for quantization.
    config: QuantConfig,
    /// Weight quantization parameters (per layer name).
    weight_params: HashMap<String, TensorQuantParams>,
    /// Activation quantization parameters (per layer name).
    activation_params: HashMap<String, TensorQuantParams>,
    /// Whether the quantizer has been calibrated.
    calibrated: bool,
}

impl StaticQuantizer {
    /// Creates a new static quantizer.
    pub fn new(config: QuantConfig) -> Self {
        Self {
            config,
            weight_params: HashMap::new(),
            activation_params: HashMap::new(),
            calibrated: false,
        }
    }

    /// Creates a quantizer with INT8 symmetric weights and asymmetric activations.
    pub fn int8() -> Self {
        Self::new(QuantConfig::int8_static())
    }

    /// Creates a quantizer with INT4 weights and INT8 activations.
    pub fn int4() -> Self {
        Self::new(QuantConfig::int4())
    }

    /// Calibrates the quantizer using provided calibration data.
    pub fn calibrate(&mut self, calibration_data: &CalibrationData) {
        // Set weight params from calibration
        for (name, range) in &calibration_data.weight_ranges {
            let params = TensorQuantParams::from_range(
                self.config.weight_scheme,
                range.0,
                range.1,
            );
            self.weight_params.insert(name.clone(), params);
        }

        // Set activation params from calibration
        for (name, range) in &calibration_data.activation_ranges {
            let params = TensorQuantParams::from_range(
                self.config.activation_scheme,
                range.0,
                range.1,
            );
            self.activation_params.insert(name.clone(), params);
        }

        self.calibrated = true;
    }

    /// Sets quantization parameters for weights directly.
    pub fn set_weight_params(&mut self, name: &str, params: TensorQuantParams) {
        self.weight_params.insert(name.to_string(), params);
    }

    /// Sets quantization parameters for activations directly.
    pub fn set_activation_params(&mut self, name: &str, params: TensorQuantParams) {
        self.activation_params.insert(name.to_string(), params);
    }

    /// Gets weight parameters for a layer.
    pub fn get_weight_params(&self, name: &str) -> Option<&TensorQuantParams> {
        self.weight_params.get(name)
    }

    /// Gets activation parameters for a layer.
    pub fn get_activation_params(&self, name: &str) -> Option<&TensorQuantParams> {
        self.activation_params.get(name)
    }

    /// Quantizes weights for a layer.
    pub fn quantize_weights(&self, name: &str, weights: &BoundedTensor) -> Option<QuantizedTensor> {
        self.weight_params.get(name).map(|params| quantize_tensor(weights, params))
    }

    /// Quantizes activations for a layer.
    pub fn quantize_activations(&self, name: &str, activations: &BoundedTensor) -> Option<QuantizedTensor> {
        self.activation_params.get(name).map(|params| quantize_tensor(activations, params))
    }

    /// Dequantizes a tensor.
    pub fn dequantize(&self, quantized: &QuantizedTensor) -> BoundedTensor {
        dequantize_tensor(quantized)
    }

    /// Returns whether the quantizer has been calibrated.
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Returns the weight quantization scheme.
    pub fn weight_scheme(&self) -> QuantScheme {
        self.config.weight_scheme
    }

    /// Returns the activation quantization scheme.
    pub fn activation_scheme(&self) -> QuantScheme {
        self.config.activation_scheme
    }
}

/// Statically quantized layer representation.
#[derive(Debug, Clone)]
pub struct StaticQuantizedLayer {
    /// Layer name/identifier.
    pub name: String,
    /// Quantized weights.
    pub quantized_weights: QuantizedTensor,
    /// Quantized bias (if present).
    pub quantized_bias: Option<QuantizedTensor>,
    /// Input quantization parameters.
    pub input_params: TensorQuantParams,
    /// Output quantization parameters.
    pub output_params: TensorQuantParams,
    /// Scale for the quantized operation.
    pub op_scale: f64,
}

impl StaticQuantizedLayer {
    /// Creates a new statically quantized layer.
    pub fn new(
        name: String,
        quantized_weights: QuantizedTensor,
        quantized_bias: Option<QuantizedTensor>,
        input_params: TensorQuantParams,
        output_params: TensorQuantParams,
    ) -> Self {
        // Compute operation scale for requantization
        let weight_scale = quantized_weights.params.scales[0];
        let input_scale = input_params.scales[0];
        let output_scale = output_params.scales[0];
        
        let op_scale = (weight_scale * input_scale) / output_scale;

        Self {
            name,
            quantized_weights,
            quantized_bias,
            input_params,
            output_params,
            op_scale,
        }
    }

    /// Returns the requantization scale.
    pub fn requant_scale(&self) -> f64 {
        self.op_scale
    }

    /// Computes the total error bound for this layer.
    pub fn total_error_bound(&self) -> f64 {
        let weight_error = self.quantized_weights.params.quantization_error();
        let input_error = self.input_params.quantization_error();
        let output_error = self.output_params.quantization_error();
        
        // Error propagates through matmul
        // Simplified: multiply errors and add output quantization
        weight_error + input_error + output_error
    }
}

/// Quantizes a full model (collection of layers) statically.
pub struct StaticQuantizedModel {
    /// Quantized layers.
    layers: Vec<StaticQuantizedLayer>,
    /// Overall config.
    #[allow(dead_code)]
    config: QuantConfig,
}

impl StaticQuantizedModel {
    /// Creates a new quantized model.
    pub fn new(config: QuantConfig) -> Self {
        Self {
            layers: Vec::new(),
            config,
        }
    }

    /// Adds a quantized layer.
    pub fn add_layer(&mut self, layer: StaticQuantizedLayer) {
        self.layers.push(layer);
    }

    /// Gets a layer by name.
    pub fn get_layer(&self, name: &str) -> Option<&StaticQuantizedLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Returns all layers.
    pub fn layers(&self) -> &[StaticQuantizedLayer] {
        &self.layers
    }

    /// Computes total error bound across all layers.
    pub fn total_error_bound(&self) -> f64 {
        self.layers.iter().map(|l| l.total_error_bound()).sum()
    }
}

/// Converts floating-point weights to statically quantized format.
pub fn quantize_weights_static(
    weights: &BoundedTensor,
    scheme: QuantScheme,
    per_channel: bool,
    axis: usize,
) -> QuantizedTensor {
    let params = if per_channel {
        super::quantize::compute_per_channel_quant_params(weights, scheme, axis)
    } else {
        super::quantize::compute_quant_params(weights, scheme)
    };

    quantize_tensor(weights, &params)
}

/// Fake-quantizes a tensor for simulating quantization during training.
pub fn fake_quantize_weights(
    weights: &BoundedTensor,
    scheme: QuantScheme,
) -> BoundedTensor {
    let params = super::quantize::compute_quant_params(weights, scheme);
    fake_quantize_tensor(weights, &params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_core::types::BoundedValue;

    #[test]
    fn test_static_quantizer_creation() {
        let quantizer = StaticQuantizer::int8();
        assert!(!quantizer.is_calibrated());
        assert_eq!(quantizer.weight_scheme(), QuantScheme::SymmetricInt8);
    }

    #[test]
    fn test_set_params_manually() {
        let mut quantizer = StaticQuantizer::int8();
        
        let params = TensorQuantParams::from_range(QuantScheme::SymmetricInt8, -1.0, 1.0);
        quantizer.set_weight_params("layer1", params);
        
        assert!(quantizer.get_weight_params("layer1").is_some());
        assert!(quantizer.get_weight_params("layer2").is_none());
    }

    #[test]
    fn test_static_quantized_layer() {
        let weight_data: Vec<BoundedValue<f64>> = vec![0.1, 0.2, 0.3, 0.4]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let weights = BoundedTensor::new(weight_data, vec![2, 2]);
        
        let weight_params = TensorQuantParams::from_range(QuantScheme::SymmetricInt8, -0.5, 0.5);
        let quantized_weights = quantize_tensor(&weights, &weight_params);
        
        let input_params = TensorQuantParams::from_range(QuantScheme::AsymmetricInt8, 0.0, 1.0);
        let output_params = TensorQuantParams::from_range(QuantScheme::AsymmetricInt8, 0.0, 1.0);
        
        let layer = StaticQuantizedLayer::new(
            "test_layer".to_string(),
            quantized_weights,
            None,
            input_params,
            output_params,
        );
        
        assert!(layer.total_error_bound() > 0.0);
    }
}
