//! Linear layer implementation with error bound tracking.
//!
//! Implements y = Wx + b where W is the weight matrix and b is the bias vector.

use crate::ops::{self, MatMulError};
use helix_core::types::{BoundedTensor, Precision};
use thiserror::Error;

/// Errors from linear layer operations.
#[derive(Error, Debug)]
pub enum LinearError {
    #[error("Matrix multiplication error: {0}")]
    MatMul(#[from] MatMulError),

    #[error("Weight dimensions ({0}, {1}) incompatible with input ({2}, {3})")]
    WeightDimensionMismatch(usize, usize, usize, usize),

    #[error("Bias dimension {0} doesn't match output dimension {1}")]
    BiasDimensionMismatch(usize, usize),

    #[error("Expected 2D input, got {0}D")]
    InvalidInputDimensions(usize),
}

/// Configuration for a linear layer.
#[derive(Debug, Clone)]
pub struct LinearConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Whether to include bias.
    pub bias: bool,
    /// Precision for computation.
    pub precision: Precision,
}

impl LinearConfig {
    /// Creates a new linear layer configuration.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
            precision: Precision::F32,
        }
    }

    /// Sets whether to use bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Sets the precision level.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }
}

/// A linear layer: y = Wx + b
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix (out_features x in_features).
    weights: BoundedTensor,
    /// Optional bias vector (out_features).
    bias: Option<BoundedTensor>,
    /// Precision for computation.
    precision: Precision,
}

impl Linear {
    /// Creates a new linear layer from weights and optional bias.
    pub fn new(
        weights: BoundedTensor,
        bias: Option<BoundedTensor>,
        precision: Precision,
    ) -> Result<Self, LinearError> {
        // Validate weights are 2D
        if !weights.is_matrix() {
            return Err(LinearError::InvalidInputDimensions(weights.ndim()));
        }

        // Validate bias dimension if present
        if let Some(ref b) = bias {
            let out_features = weights.shape()[0];
            if b.len() != out_features {
                return Err(LinearError::BiasDimensionMismatch(b.len(), out_features));
            }
        }

        Ok(Self {
            weights,
            bias,
            precision,
        })
    }

    /// Creates a linear layer from a configuration with zero-initialized weights.
    pub fn from_config(config: &LinearConfig) -> Self {
        let weights = BoundedTensor::zeros(vec![config.out_features, config.in_features]);
        let bias = if config.bias {
            Some(BoundedTensor::zeros(vec![config.out_features]))
        } else {
            None
        };

        Self {
            weights,
            bias,
            precision: config.precision,
        }
    }

    /// Creates a linear layer from raw f64 weights and bias.
    pub fn from_raw(
        weights: Vec<f64>,
        weight_shape: Vec<usize>,
        bias: Option<Vec<f64>>,
        precision: Precision,
    ) -> Result<Self, LinearError> {
        let weight_tensor = BoundedTensor::from_exact(weights, weight_shape);
        let bias_tensor = bias.map(|b| {
            let len = b.len();
            BoundedTensor::from_exact(b, vec![len])
        });

        Self::new(weight_tensor, bias_tensor, precision)
    }

    /// Returns the weight tensor.
    pub fn weights(&self) -> &BoundedTensor {
        &self.weights
    }

    /// Returns the bias tensor if present.
    pub fn bias(&self) -> Option<&BoundedTensor> {
        self.bias.as_ref()
    }

    /// Returns the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.weights.shape()[1]
    }

    /// Returns the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.weights.shape()[0]
    }

    /// Forward pass: y = Wx + b
    ///
    /// Input shape: (batch_size, in_features) or (in_features,) for single sample
    /// Output shape: (batch_size, out_features) or (out_features,)
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, LinearError> {
        // Handle 1D input (single sample)
        let (input_2d, squeeze_output) = if input.is_vector() {
            let len = input.len();
            (input.reshape(vec![1, len]), true)
        } else if input.is_matrix() {
            (input.clone(), false)
        } else {
            return Err(LinearError::InvalidInputDimensions(input.ndim()));
        };

        // Validate dimensions
        let (batch_size, in_features) = (input_2d.shape()[0], input_2d.shape()[1]);
        if in_features != self.in_features() {
            return Err(LinearError::WeightDimensionMismatch(
                self.weights.shape()[0],
                self.weights.shape()[1],
                batch_size,
                in_features,
            ));
        }

        // Compute W^T @ input^T = (input @ W)^T, then transpose
        // Or equivalently: output = input @ W^T
        let weights_t = self.weights.transpose();
        let output = ops::matmul(&input_2d, &weights_t, self.precision)?;

        // Add bias if present
        let output = if let Some(ref bias) = self.bias {
            self.add_bias(&output, bias)?
        } else {
            output
        };

        // Squeeze output back to 1D if input was 1D
        if squeeze_output {
            Ok(output.reshape(vec![self.out_features()]))
        } else {
            Ok(output)
        }
    }

    /// Adds bias to each row of the output.
    fn add_bias(
        &self,
        output: &BoundedTensor,
        bias: &BoundedTensor,
    ) -> Result<BoundedTensor, LinearError> {
        let (batch_size, out_features) = (output.shape()[0], output.shape()[1]);

        let mut result_data = Vec::with_capacity(batch_size * out_features);

        for i in 0..batch_size {
            for j in 0..out_features {
                let out_val = output.get(&[i, j]).unwrap();
                let bias_val = bias.get(&[j]).unwrap();
                result_data.push(*out_val + *bias_val);
            }
        }

        Ok(BoundedTensor::new(result_data, vec![batch_size, out_features]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let config = LinearConfig::new(10, 5);
        let layer = Linear::from_config(&config);
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert!(layer.bias().is_some());
    }

    #[test]
    fn test_linear_forward_1d() {
        // Simple 2x2 weight matrix: [[1, 0], [0, 1]] (identity)
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let layer = Linear::from_raw(weights, vec![2, 2], None, Precision::F32).unwrap();

        let input = BoundedTensor::from_exact(vec![3.0, 4.0], vec![2]);
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &vec![2]);
        let values = output.values();
        assert!((values[0] - 3.0).abs() < 1e-10);
        assert!((values[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_forward_2d() {
        // Weight matrix [[1, 2], [3, 4]] with bias [1, 0]
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![1.0, 0.0];
        let layer = Linear::from_raw(weights, vec![2, 2], Some(bias), Precision::F32).unwrap();

        // Input: [[1, 0], [0, 1]]
        let input = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let output = layer.forward(&input).unwrap();

        // Row 0: [1, 0] @ [[1, 3], [2, 4]] + [1, 0] = [1, 3] + [1, 0] = [2, 3]
        // Row 1: [0, 1] @ [[1, 3], [2, 4]] + [1, 0] = [2, 4] + [1, 0] = [3, 4]
        let values = output.values();
        assert!((values[0] - 2.0).abs() < 1e-10);
        assert!((values[1] - 3.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
        assert!((values[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_error_propagation() {
        let weights = vec![2.0, 0.0, 0.0, 2.0];
        let layer = Linear::from_raw(weights, vec![2, 2], None, Precision::F32).unwrap();

        // Input with error
        let input = BoundedTensor::from_approximate(vec![1.0, 1.0], vec![2], 0.1);
        let output = layer.forward(&input).unwrap();

        // Output should have error > 0
        assert!(output.max_error() > 0.0);
    }
}
