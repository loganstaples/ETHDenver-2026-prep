//! Dynamic Quantization.
//!
//! Runtime quantization where activation parameters are computed dynamically
//! based on the actual values during inference. Weights are still pre-quantized.

use helix_core::types::{BoundedTensor, BoundedValue, Shape};

use super::quantize::{QuantizedTensor, quantize_tensor, quantize_scalar};
use super::dequantize::dequantize_tensor;
use super::schemes::{QuantScheme, TensorQuantParams};

/// Dynamic quantizer that computes quantization parameters at runtime.
#[derive(Debug)]
pub struct DynamicQuantizer {
    /// Weight quantization scheme.
    weight_scheme: QuantScheme,
    /// Activation quantization scheme.
    activation_scheme: QuantScheme,
    /// Whether to use symmetric quantization for activations.
    symmetric_activations: bool,
    /// Minimum number of samples before using statistics.
    min_samples: usize,
    /// Running statistics for adaptive quantization.
    running_min: f64,
    running_max: f64,
    sample_count: usize,
    /// Smoothing factor for EMA (0.0 to 1.0).
    smoothing: f64,
}

impl DynamicQuantizer {
    /// Creates a new dynamic quantizer.
    pub fn new(weight_scheme: QuantScheme, activation_scheme: QuantScheme) -> Self {
        Self {
            weight_scheme,
            activation_scheme,
            symmetric_activations: false,
            min_samples: 0,
            running_min: f64::INFINITY,
            running_max: f64::NEG_INFINITY,
            sample_count: 0,
            smoothing: 0.1,
        }
    }

    /// Creates an INT8 dynamic quantizer.
    pub fn int8() -> Self {
        Self::new(QuantScheme::SymmetricInt8, QuantScheme::AsymmetricInt8)
    }

    /// Creates an FP8 dynamic quantizer.
    pub fn fp8() -> Self {
        Self::new(QuantScheme::FP8E4M3, QuantScheme::FP8E4M3)
    }

    /// Quantizes activations dynamically (computes params at runtime).
    pub fn quantize_activations(&mut self, activations: &BoundedTensor) -> QuantizedTensor {
        // Compute dynamic quantization parameters
        let data = activations.data();
        
        let (min_val, max_val) = compute_minmax(data);
        
        // Update running statistics
        self.update_statistics(min_val, max_val);
        
        // Use current or running statistics based on config
        let (use_min, use_max) = if self.sample_count >= self.min_samples && self.min_samples > 0 {
            (self.running_min, self.running_max)
        } else {
            (min_val, max_val)
        };
        
        let params = TensorQuantParams::from_range(self.activation_scheme, use_min, use_max);
        quantize_tensor(activations, &params)
    }

    /// Quantizes activations with explicit range.
    pub fn quantize_activations_with_range(
        &self,
        activations: &BoundedTensor,
        min_val: f64,
        max_val: f64,
    ) -> QuantizedTensor {
        let params = TensorQuantParams::from_range(self.activation_scheme, min_val, max_val);
        quantize_tensor(activations, &params)
    }

    /// Computes dynamic quantization parameters for a tensor.
    pub fn compute_activation_params(&self, activations: &BoundedTensor) -> TensorQuantParams {
        let data = activations.data();
        let (min_val, max_val) = compute_minmax(data);
        TensorQuantParams::from_range(self.activation_scheme, min_val, max_val)
    }

    /// Updates running statistics with EMA.
    fn update_statistics(&mut self, min_val: f64, max_val: f64) {
        if self.sample_count == 0 {
            self.running_min = min_val;
            self.running_max = max_val;
        } else {
            // Exponential moving average
            self.running_min = self.smoothing * min_val + (1.0 - self.smoothing) * self.running_min;
            self.running_max = self.smoothing * max_val + (1.0 - self.smoothing) * self.running_max;
        }
        self.sample_count += 1;
    }

    /// Resets running statistics.
    pub fn reset_statistics(&mut self) {
        self.running_min = f64::INFINITY;
        self.running_max = f64::NEG_INFINITY;
        self.sample_count = 0;
    }

    /// Dequantizes a tensor.
    pub fn dequantize(&self, quantized: &QuantizedTensor) -> BoundedTensor {
        dequantize_tensor(quantized)
    }

    /// Sets the minimum samples for using running statistics.
    pub fn set_min_samples(&mut self, min_samples: usize) {
        self.min_samples = min_samples;
    }

    /// Sets the smoothing factor for EMA.
    pub fn set_smoothing(&mut self, smoothing: f64) {
        self.smoothing = smoothing.clamp(0.0, 1.0);
    }

    /// Returns the current running statistics.
    pub fn running_stats(&self) -> (f64, f64, usize) {
        (self.running_min, self.running_max, self.sample_count)
    }
}

/// Computes min and max of a tensor.
fn compute_minmax(data: &[BoundedValue<f64>]) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    
    for v in data {
        let value = v.value();
        min_val = min_val.min(value);
        max_val = max_val.max(value);
    }
    
    // Handle edge case of empty or single-value tensor
    if min_val == max_val {
        if min_val < 0.0 {
            max_val = 0.0;
        } else {
            min_val = 0.0;
        }
    }
    
    (min_val, max_val)
}

/// Dynamically quantized linear layer.
#[derive(Debug)]
pub struct DynamicQuantizedLinear {
    /// Layer name.
    pub name: String,
    /// Pre-quantized weights.
    pub quantized_weights: QuantizedTensor,
    /// Pre-quantized bias (optional).
    pub quantized_bias: Option<QuantizedTensor>,
    /// Dynamic quantizer for activations.
    quantizer: DynamicQuantizer,
}

impl DynamicQuantizedLinear {
    /// Creates a new dynamically quantized linear layer.
    pub fn new(
        name: String,
        weights: &BoundedTensor,
        bias: Option<&BoundedTensor>,
        weight_scheme: QuantScheme,
    ) -> Self {
        // Pre-quantize weights (static)
        let weight_params = super::quantize::compute_quant_params(weights, weight_scheme);
        let quantized_weights = quantize_tensor(weights, &weight_params);
        
        // Pre-quantize bias if present
        let quantized_bias = bias.map(|b| {
            let bias_params = super::quantize::compute_quant_params(b, weight_scheme);
            quantize_tensor(b, &bias_params)
        });
        
        Self {
            name,
            quantized_weights,
            quantized_bias,
            quantizer: DynamicQuantizer::int8(),
        }
    }

    /// Returns the weight quantization error.
    pub fn weight_error(&self) -> f64 {
        self.quantized_weights.params.quantization_error()
    }

    /// Returns the quantizer for runtime modification.
    pub fn quantizer_mut(&mut self) -> &mut DynamicQuantizer {
        &mut self.quantizer
    }

    /// Quantizes input activations dynamically.
    pub fn quantize_input(&mut self, input: &BoundedTensor) -> QuantizedTensor {
        self.quantizer.quantize_activations(input)
    }
}

/// Per-token dynamic quantization (useful for transformers).
pub struct PerTokenDynamicQuantizer {
    /// Base scheme.
    scheme: QuantScheme,
}

impl PerTokenDynamicQuantizer {
    /// Creates a new per-token quantizer.
    pub fn new(scheme: QuantScheme) -> Self {
        Self { scheme }
    }

    /// Quantizes each token (row) independently.
    pub fn quantize(&self, activations: &BoundedTensor) -> (QuantizedTensor, Vec<TensorQuantParams>) {
        let shape = activations.shape();
        let data = activations.data();
        
        // Assume shape is [batch, seq_len, hidden] or [seq_len, hidden]
        let (num_tokens, hidden_size) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else if !shape.is_empty() {
            (shape[0], data.len() / shape[0])
        } else {
            (1, data.len())
        };
        
        let mut all_quantized = Vec::with_capacity(data.len());
        let mut all_params = Vec::with_capacity(num_tokens);
        
        for token_idx in 0..num_tokens {
            let start = token_idx * hidden_size;
            let end = start + hidden_size;
            let token_data = &data[start..end];
            
            // Compute params for this token
            let (min_val, max_val) = compute_minmax(token_data);
            let params = TensorQuantParams::from_range(self.scheme, min_val, max_val);
            
            // Quantize this token
            for v in token_data {
                let q = quantize_scalar(
                    v.value(),
                    params.scales[0],
                    params.zero_points[0],
                    self.scheme,
                );
                all_quantized.push(q);
            }
            
            all_params.push(params);
        }
        
        // Create final quantized tensor (using first params for metadata)
        let final_params = if !all_params.is_empty() {
            all_params[0].clone()
        } else {
            TensorQuantParams::new(self.scheme, 1.0, 0)
        };
        
        let quantized = QuantizedTensor::new(all_quantized, shape.clone(), final_params);
        (quantized, all_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_quantizer() {
        let mut quantizer = DynamicQuantizer::int8();
        
        let data: Vec<BoundedValue<f64>> = vec![0.1, 0.5, -0.3, 0.8]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let tensor = BoundedTensor::new(data, vec![4]);
        
        let quantized = quantizer.quantize_activations(&tensor);
        assert_eq!(quantized.len(), 4);
        
        let (min, max, count) = quantizer.running_stats();
        assert_eq!(count, 1);
        assert!((min - (-0.3)).abs() < 1e-10);
        assert!((max - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_running_statistics() {
        let mut quantizer = DynamicQuantizer::int8();
        quantizer.set_smoothing(0.5);
        
        let data1: Vec<BoundedValue<f64>> = vec![0.0, 1.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let tensor1 = BoundedTensor::new(data1, vec![2]);
        
        quantizer.quantize_activations(&tensor1);
        
        let data2: Vec<BoundedValue<f64>> = vec![-2.0, 2.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let tensor2 = BoundedTensor::new(data2, vec![2]);
        
        quantizer.quantize_activations(&tensor2);
        
        let (min, max, count) = quantizer.running_stats();
        assert_eq!(count, 2);
        // EMA should average
        assert!((min - (-1.0)).abs() < 1e-10); // 0.5 * (-2) + 0.5 * 0 = -1
        assert!((max - 1.5).abs() < 1e-10);   // 0.5 * 2 + 0.5 * 1 = 1.5
    }
}
