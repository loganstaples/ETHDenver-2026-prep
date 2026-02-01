//! Layer Normalization Module.
//!
//! Provides LayerNorm and RMSNorm implementations with learnable parameters
//! and error bound tracking for neural network layers.
//!
//! # Features
//!
//! - **LayerNorm**: Standard layer normalization with learnable gamma and beta
//! - **RMSNorm**: Root Mean Square normalization (no mean subtraction)
//! - **Error Tracking**: Propagates error bounds through normalization
//! - **Memory Efficient**: Supports fused computation to reduce memory
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::nn::layer_norm::{LayerNorm, LayerNormConfig};
//!
//! let config = LayerNormConfig::new(256).with_eps(1e-5);
//! let layer_norm = LayerNorm::new(config);
//!
//! let output = layer_norm.forward(&input)?;
//! ```

use crate::ops::normalization::{layer_norm as layer_norm_fn, rms_norm as rms_norm_fn};
use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};
use thiserror::Error;

/// Errors from layer normalization operations.
#[derive(Error, Debug)]
pub enum LayerNormError {
    #[error("Dimension mismatch: expected normalized_shape {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Empty input tensor")]
    EmptyInput,
}

/// Configuration for layer normalization.
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Size of the normalized dimension.
    pub normalized_shape: usize,
    /// Epsilon for numerical stability.
    pub eps: f64,
    /// Whether to use learnable affine parameters (gamma, beta).
    pub elementwise_affine: bool,
    /// Initial value for gamma (scale).
    pub gamma_init: f64,
    /// Initial value for beta (shift).
    pub beta_init: f64,
    /// Precision for computation.
    pub precision: Precision,
}

impl LayerNormConfig {
    /// Creates a new layer norm configuration.
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
            gamma_init: 1.0,
            beta_init: 0.0,
            precision: Precision::F32,
        }
    }

    /// Sets the epsilon value.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets whether to use learnable parameters.
    pub fn with_elementwise_affine(mut self, affine: bool) -> Self {
        self.elementwise_affine = affine;
        self
    }

    /// Sets the precision.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// Sets initial gamma value.
    pub fn with_gamma_init(mut self, gamma: f64) -> Self {
        self.gamma_init = gamma;
        self
    }

    /// Sets initial beta value.
    pub fn with_beta_init(mut self, beta: f64) -> Self {
        self.beta_init = beta;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), LayerNormError> {
        if self.normalized_shape == 0 {
            return Err(LayerNormError::InvalidConfig(
                "normalized_shape must be positive".to_string(),
            ));
        }
        if self.eps <= 0.0 {
            return Err(LayerNormError::InvalidConfig(
                "eps must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Layer Normalization layer.
///
/// Normalizes inputs across features: `output = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// Where mean and variance are computed across the last dimension.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Configuration.
    config: LayerNormConfig,
    /// Learnable scale parameter (gamma).
    gamma: Option<BoundedTensor>,
    /// Learnable shift parameter (beta).
    beta: Option<BoundedTensor>,
}

impl LayerNorm {
    /// Creates a new LayerNorm layer.
    pub fn new(config: LayerNormConfig) -> Result<Self, LayerNormError> {
        config.validate()?;

        let (gamma, beta) = if config.elementwise_affine {
            let gamma_data: Vec<BoundedValue<f64>> = (0..config.normalized_shape)
                .map(|_| BoundedValue::exact(config.gamma_init))
                .collect();
            let beta_data: Vec<BoundedValue<f64>> = (0..config.normalized_shape)
                .map(|_| BoundedValue::exact(config.beta_init))
                .collect();

            (
                Some(BoundedTensor::new(gamma_data, vec![config.normalized_shape])),
                Some(BoundedTensor::new(beta_data, vec![config.normalized_shape])),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            gamma,
            beta,
        })
    }

    /// Creates a LayerNorm with specified gamma and beta values.
    pub fn with_params(
        config: LayerNormConfig,
        gamma: BoundedTensor,
        beta: BoundedTensor,
    ) -> Result<Self, LayerNormError> {
        config.validate()?;

        if gamma.len() != config.normalized_shape {
            return Err(LayerNormError::DimensionMismatch {
                expected: config.normalized_shape,
                actual: gamma.len(),
            });
        }
        if beta.len() != config.normalized_shape {
            return Err(LayerNormError::DimensionMismatch {
                expected: config.normalized_shape,
                actual: beta.len(),
            });
        }

        Ok(Self {
            config,
            gamma: Some(gamma),
            beta: Some(beta),
        })
    }

    /// Returns the configuration.
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }

    /// Returns the gamma (scale) parameter.
    pub fn gamma(&self) -> Option<&BoundedTensor> {
        self.gamma.as_ref()
    }

    /// Returns the beta (shift) parameter.
    pub fn beta(&self) -> Option<&BoundedTensor> {
        self.beta.as_ref()
    }

    /// Returns the number of parameters.
    pub fn param_count(&self) -> usize {
        if self.config.elementwise_affine {
            self.config.normalized_shape * 2
        } else {
            0
        }
    }

    /// Forward pass.
    ///
    /// Input shape: (..., normalized_shape)
    /// Output shape: same as input
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, LayerNormError> {
        if input.is_empty() {
            return Err(LayerNormError::EmptyInput);
        }

        let shape = input.shape();
        let last_dim = *shape.last().unwrap_or(&0);

        if last_dim != self.config.normalized_shape {
            return Err(LayerNormError::DimensionMismatch {
                expected: self.config.normalized_shape,
                actual: last_dim,
            });
        }

        // Compute layer norm for each group
        let normalized = self.normalize_groups(input)?;

        // Apply affine transformation if enabled
        if let (Some(gamma), Some(beta)) = (&self.gamma, &self.beta) {
            Ok(self.apply_affine(&normalized, gamma, beta))
        } else {
            Ok(normalized)
        }
    }

    /// Normalizes input in groups along the last dimension.
    fn normalize_groups(&self, input: &BoundedTensor) -> Result<BoundedTensor, LayerNormError> {
        let shape = input.shape();
        let norm_size = self.config.normalized_shape;
        let num_groups = input.len() / norm_size;

        let data = input.data();
        let precision_error = self.config.precision.max_relative_error();

        let mut result = Vec::with_capacity(input.len());

        for g in 0..num_groups {
            let start = g * norm_size;
            let end = start + norm_size;
            let group_data = &data[start..end];

            // Compute mean
            let sum: f64 = group_data.iter().map(|v| v.value()).sum();
            let mean = sum / norm_size as f64;

            // Compute variance
            let var_sum: f64 = group_data.iter().map(|v| (v.value() - mean).powi(2)).sum();
            let variance = var_sum / norm_size as f64;
            let std_inv = 1.0 / (variance + self.config.eps).sqrt();

            // Normalize
            for v in group_data {
                let normalized = (v.value() - mean) * std_inv;

                // Error propagation
                let error = v.absolute_error() * std_inv + normalized.abs() * precision_error;

                result.push(BoundedValue::new(normalized, ErrorMargin::absolute(error)));
            }
        }

        Ok(BoundedTensor::new(result, shape.clone()))
    }

    /// Applies affine transformation: output = gamma * normalized + beta.
    fn apply_affine(
        &self,
        normalized: &BoundedTensor,
        gamma: &BoundedTensor,
        beta: &BoundedTensor,
    ) -> BoundedTensor {
        let norm_size = self.config.normalized_shape;
        let gamma_data = gamma.data();
        let beta_data = beta.data();

        let data: Vec<BoundedValue<f64>> = normalized
            .data()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let idx = i % norm_size;
                let g = gamma_data[idx];
                let b = beta_data[idx];

                // y = gamma * x + beta
                let value = g.value() * v.value() + b.value();

                // Error propagation
                let error = g.value().abs() * v.absolute_error()
                    + v.value().abs() * g.absolute_error()
                    + b.absolute_error();

                BoundedValue::new(value, ErrorMargin::absolute(error))
            })
            .collect();

        BoundedTensor::new(data, normalized.shape().clone())
    }
}

/// RMS Normalization layer.
///
/// Normalizes inputs using root mean square: `output = gamma * x / sqrt(mean(x^2) + eps)`
///
/// Used in LLaMA and other modern architectures for improved training stability.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    /// Configuration.
    config: LayerNormConfig,
    /// Learnable scale parameter (gamma).
    gamma: Option<BoundedTensor>,
}

impl RMSNorm {
    /// Creates a new RMSNorm layer.
    pub fn new(config: LayerNormConfig) -> Result<Self, LayerNormError> {
        config.validate()?;

        let gamma = if config.elementwise_affine {
            let gamma_data: Vec<BoundedValue<f64>> = (0..config.normalized_shape)
                .map(|_| BoundedValue::exact(config.gamma_init))
                .collect();
            Some(BoundedTensor::new(gamma_data, vec![config.normalized_shape]))
        } else {
            None
        };

        Ok(Self { config, gamma })
    }

    /// Creates an RMSNorm with specified gamma values.
    pub fn with_params(config: LayerNormConfig, gamma: BoundedTensor) -> Result<Self, LayerNormError> {
        config.validate()?;

        if gamma.len() != config.normalized_shape {
            return Err(LayerNormError::DimensionMismatch {
                expected: config.normalized_shape,
                actual: gamma.len(),
            });
        }

        Ok(Self {
            config,
            gamma: Some(gamma),
        })
    }

    /// Returns the configuration.
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }

    /// Returns the gamma (scale) parameter.
    pub fn gamma(&self) -> Option<&BoundedTensor> {
        self.gamma.as_ref()
    }

    /// Returns the number of parameters.
    pub fn param_count(&self) -> usize {
        if self.config.elementwise_affine {
            self.config.normalized_shape
        } else {
            0
        }
    }

    /// Forward pass.
    ///
    /// Input shape: (..., normalized_shape)
    /// Output shape: same as input
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, LayerNormError> {
        if input.is_empty() {
            return Err(LayerNormError::EmptyInput);
        }

        let shape = input.shape();
        let last_dim = *shape.last().unwrap_or(&0);

        if last_dim != self.config.normalized_shape {
            return Err(LayerNormError::DimensionMismatch {
                expected: self.config.normalized_shape,
                actual: last_dim,
            });
        }

        // Compute RMS norm for each group
        let normalized = self.normalize_rms_groups(input)?;

        // Apply scale if enabled
        if let Some(gamma) = &self.gamma {
            Ok(self.apply_scale(&normalized, gamma))
        } else {
            Ok(normalized)
        }
    }

    /// Normalizes input using RMS for each group along the last dimension.
    fn normalize_rms_groups(&self, input: &BoundedTensor) -> Result<BoundedTensor, LayerNormError> {
        let shape = input.shape();
        let norm_size = self.config.normalized_shape;
        let num_groups = input.len() / norm_size;

        let data = input.data();
        let precision_error = self.config.precision.max_relative_error();

        let mut result = Vec::with_capacity(input.len());

        for g in 0..num_groups {
            let start = g * norm_size;
            let end = start + norm_size;
            let group_data = &data[start..end];

            // Compute RMS
            let sum_squares: f64 = group_data.iter().map(|v| v.value().powi(2)).sum();
            let rms = (sum_squares / norm_size as f64 + self.config.eps).sqrt();
            let rms_inv = 1.0 / rms;

            // Normalize
            for v in group_data {
                let normalized = v.value() * rms_inv;

                // Error propagation
                let error = v.absolute_error() * rms_inv + normalized.abs() * precision_error;

                result.push(BoundedValue::new(normalized, ErrorMargin::absolute(error)));
            }
        }

        Ok(BoundedTensor::new(result, shape.clone()))
    }

    /// Applies scale transformation: output = gamma * normalized.
    fn apply_scale(&self, normalized: &BoundedTensor, gamma: &BoundedTensor) -> BoundedTensor {
        let norm_size = self.config.normalized_shape;
        let gamma_data = gamma.data();

        let data: Vec<BoundedValue<f64>> = normalized
            .data()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let idx = i % norm_size;
                let g = gamma_data[idx];

                // y = gamma * x
                let value = g.value() * v.value();

                // Error propagation
                let error =
                    g.value().abs() * v.absolute_error() + v.value().abs() * g.absolute_error();

                BoundedValue::new(value, ErrorMargin::absolute(error))
            })
            .collect();

        BoundedTensor::new(data, normalized.shape().clone())
    }
}

/// Fused layer normalization for memory efficiency.
///
/// Combines normalization and affine transformation in a single pass
/// to reduce memory allocations.
pub fn fused_layer_norm(
    input: &BoundedTensor,
    gamma: &BoundedTensor,
    beta: &BoundedTensor,
    eps: f64,
    precision: Precision,
) -> Result<BoundedTensor, LayerNormError> {
    let shape = input.shape();
    let norm_size = *shape.last().ok_or(LayerNormError::EmptyInput)?;

    if gamma.len() != norm_size || beta.len() != norm_size {
        return Err(LayerNormError::DimensionMismatch {
            expected: norm_size,
            actual: gamma.len().min(beta.len()),
        });
    }

    let num_groups = input.len() / norm_size;
    let data = input.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();
    let precision_error = precision.max_relative_error();

    let mut result = Vec::with_capacity(input.len());

    for g in 0..num_groups {
        let start = g * norm_size;
        let end = start + norm_size;
        let group_data = &data[start..end];

        // Compute mean and variance
        let sum: f64 = group_data.iter().map(|v| v.value()).sum();
        let mean = sum / norm_size as f64;

        let var_sum: f64 = group_data.iter().map(|v| (v.value() - mean).powi(2)).sum();
        let variance = var_sum / norm_size as f64;
        let std_inv = 1.0 / (variance + eps).sqrt();

        // Fused normalize + affine
        for (i, v) in group_data.iter().enumerate() {
            let normalized = (v.value() - mean) * std_inv;
            let g_val = gamma_data[i];
            let b_val = beta_data[i];

            let value = g_val.value() * normalized + b_val.value();

            let error = g_val.value().abs() * v.absolute_error() * std_inv
                + normalized.abs() * g_val.absolute_error()
                + b_val.absolute_error()
                + value.abs() * precision_error;

            result.push(BoundedValue::new(value, ErrorMargin::absolute(error)));
        }
    }

    Ok(BoundedTensor::new(result, shape.clone()))
}

/// Fused RMS normalization for memory efficiency.
pub fn fused_rms_norm(
    input: &BoundedTensor,
    gamma: &BoundedTensor,
    eps: f64,
    precision: Precision,
) -> Result<BoundedTensor, LayerNormError> {
    let shape = input.shape();
    let norm_size = *shape.last().ok_or(LayerNormError::EmptyInput)?;

    if gamma.len() != norm_size {
        return Err(LayerNormError::DimensionMismatch {
            expected: norm_size,
            actual: gamma.len(),
        });
    }

    let num_groups = input.len() / norm_size;
    let data = input.data();
    let gamma_data = gamma.data();
    let precision_error = precision.max_relative_error();

    let mut result = Vec::with_capacity(input.len());

    for g in 0..num_groups {
        let start = g * norm_size;
        let end = start + norm_size;
        let group_data = &data[start..end];

        // Compute RMS
        let sum_squares: f64 = group_data.iter().map(|v| v.value().powi(2)).sum();
        let rms_inv = 1.0 / (sum_squares / norm_size as f64 + eps).sqrt();

        // Fused normalize + scale
        for (i, v) in group_data.iter().enumerate() {
            let normalized = v.value() * rms_inv;
            let g_val = gamma_data[i];

            let value = g_val.value() * normalized;

            let error = g_val.value().abs() * v.absolute_error() * rms_inv
                + normalized.abs() * g_val.absolute_error()
                + value.abs() * precision_error;

            result.push(BoundedValue::new(value, ErrorMargin::absolute(error)));
        }
    }

    Ok(BoundedTensor::new(result, shape.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_config() {
        let config = LayerNormConfig::new(256).with_eps(1e-6).with_elementwise_affine(true);
        assert_eq!(config.normalized_shape, 256);
        assert_eq!(config.eps, 1e-6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_layer_norm_creation() {
        let config = LayerNormConfig::new(64);
        let layer_norm = LayerNorm::new(config).unwrap();

        assert!(layer_norm.gamma().is_some());
        assert!(layer_norm.beta().is_some());
        assert_eq!(layer_norm.param_count(), 128); // 64 * 2
    }

    #[test]
    fn test_layer_norm_forward() {
        let config = LayerNormConfig::new(4);
        let layer_norm = LayerNorm::new(config).unwrap();

        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let output = layer_norm.forward(&input).unwrap();

        assert_eq!(output.shape(), &vec![4]);

        // Mean should be approximately 0 after normalization (before affine)
        // With gamma=1 and beta=0, output should be close to normalized values
        let mean: f64 = output.values().iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_layer_norm_batched() {
        let config = LayerNormConfig::new(3);
        let layer_norm = LayerNorm::new(config).unwrap();

        // 2 groups of 3 elements each
        let input = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            vec![2, 3],
        );
        let output = layer_norm.forward(&input).unwrap();

        assert_eq!(output.shape(), &vec![2, 3]);

        // Each group should be normalized independently
        let values = output.values();

        // First group mean
        let mean1: f64 = (values[0] + values[1] + values[2]) / 3.0;
        assert!(mean1.abs() < 1e-6);

        // Second group mean
        let mean2: f64 = (values[3] + values[4] + values[5]) / 3.0;
        assert!(mean2.abs() < 1e-6);
    }

    #[test]
    fn test_layer_norm_without_affine() {
        let config = LayerNormConfig::new(4).with_elementwise_affine(false);
        let layer_norm = LayerNorm::new(config).unwrap();

        assert!(layer_norm.gamma().is_none());
        assert!(layer_norm.beta().is_none());
        assert_eq!(layer_norm.param_count(), 0);

        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let output = layer_norm.forward(&input).unwrap();

        assert_eq!(output.shape(), &vec![4]);
    }

    #[test]
    fn test_rms_norm_creation() {
        let config = LayerNormConfig::new(64);
        let rms_norm = RMSNorm::new(config).unwrap();

        assert!(rms_norm.gamma().is_some());
        assert_eq!(rms_norm.param_count(), 64);
    }

    #[test]
    fn test_rms_norm_forward() {
        let config = LayerNormConfig::new(4);
        let rms_norm = RMSNorm::new(config).unwrap();

        let input = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let output = rms_norm.forward(&input).unwrap();

        // For constant input, RMS = 1, so output should be approximately 1
        for v in output.values() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rms_norm_batched() {
        let config = LayerNormConfig::new(2);
        let rms_norm = RMSNorm::new(config).unwrap();

        // 3 groups of 2 elements each
        let input = BoundedTensor::from_exact(
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            vec![3, 2],
        );
        let output = rms_norm.forward(&input).unwrap();

        assert_eq!(output.shape(), &vec![3, 2]);

        // Each group should have RMS ~= 1 after normalization
        let values = output.values();
        let rms1 = ((values[0].powi(2) + values[1].powi(2)) / 2.0).sqrt();
        let rms2 = ((values[2].powi(2) + values[3].powi(2)) / 2.0).sqrt();
        let rms3 = ((values[4].powi(2) + values[5].powi(2)) / 2.0).sqrt();

        assert!((rms1 - 1.0).abs() < 1e-5);
        assert!((rms2 - 1.0).abs() < 1e-5);
        assert!((rms3 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = LayerNormConfig::new(4);
        let layer_norm = LayerNorm::new(config).unwrap();

        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let result = layer_norm.forward(&input);

        assert!(matches!(result, Err(LayerNormError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_fused_layer_norm() {
        let gamma = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let beta = BoundedTensor::from_exact(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let output = fused_layer_norm(&input, &gamma, &beta, 1e-5, Precision::F32).unwrap();

        assert_eq!(output.shape(), &vec![4]);

        let mean: f64 = output.values().iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_fused_rms_norm() {
        let gamma = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let input = BoundedTensor::from_exact(vec![2.0, 2.0, 2.0, 2.0], vec![4]);

        let output = fused_rms_norm(&input, &gamma, 1e-5, Precision::F32).unwrap();

        assert_eq!(output.shape(), &vec![4]);

        // For constant input, all outputs should be approximately 1
        for v in output.values() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_error_propagation() {
        let config = LayerNormConfig::new(4);
        let layer_norm = LayerNorm::new(config).unwrap();

        // Input with error bounds
        let input = BoundedTensor::from_approximate(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            0.001,
        );

        let output = layer_norm.forward(&input).unwrap();

        // Output should have error bounds
        assert!(output.max_error() > 0.0);
    }

    #[test]
    fn test_with_params() {
        let config = LayerNormConfig::new(3);
        let gamma = BoundedTensor::from_exact(vec![2.0, 2.0, 2.0], vec![3]);
        let beta = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0], vec![3]);

        let layer_norm = LayerNorm::with_params(config, gamma, beta).unwrap();
        let input = BoundedTensor::from_exact(vec![0.0, 0.0, 0.0], vec![3]);

        let output = layer_norm.forward(&input).unwrap();

        // With zero mean input and gamma=2, beta=1, output should be close to 1
        for v in output.values() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }
}
