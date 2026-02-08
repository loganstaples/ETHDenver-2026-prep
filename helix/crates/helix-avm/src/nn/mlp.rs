//! MLP (Multi-Layer Perceptron) block implementation with error tracking.
//!
//! Standard transformer MLP: Linear → Activation → Linear

use super::linear::{Linear, LinearError};
use crate::ops::activation;
use helix_core::types::{BoundedTensor, Precision};
use thiserror::Error;

/// Errors from MLP operations.
#[derive(Error, Debug)]
pub enum MLPError {
    #[error("Linear layer error: {0}")]
    Linear(#[from] LinearError),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// ReLU activation.
    ReLU,
    /// GELU activation (used in GPT-2, BERT).
    GELU,
    /// SiLU/Swish activation (used in LLaMA).
    SiLU,
}

/// Configuration for MLP block.
#[derive(Debug, Clone)]
pub struct MLPConfig {
    /// Input/output dimension.
    pub d_model: usize,
    /// Hidden dimension (typically 4 * d_model).
    pub d_ff: usize,
    /// Activation function.
    pub activation: ActivationType,
    /// Whether to use bias.
    pub bias: bool,
    /// Precision for computation.
    pub precision: Precision,
}

impl MLPConfig {
    /// Creates a new MLP configuration with default 4x expansion.
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_ff: d_model * 4,
            activation: ActivationType::GELU,
            bias: true,
            precision: Precision::F32,
        }
    }

    /// Sets the hidden dimension.
    pub fn with_hidden_dim(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    /// Sets the activation function.
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Sets whether to use bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

/// MLP block: Linear → Activation → Linear
#[derive(Debug, Clone)]
pub struct MLP {
    /// First linear layer (d_model → d_ff).
    fc1: Linear,
    /// Second linear layer (d_ff → d_model).
    fc2: Linear,
    /// Activation type.
    activation: ActivationType,
    /// Precision.
    precision: Precision,
}

impl MLP {
    /// Creates a new MLP block from configuration.
    pub fn new(config: &MLPConfig) -> Result<Self, MLPError> {
        let fc1 = Linear::from_raw(
            vec![0.0; config.d_model * config.d_ff],
            vec![config.d_ff, config.d_model],
            if config.bias { Some(vec![0.0; config.d_ff]) } else { None },
            config.precision,
        )?;

        let fc2 = Linear::from_raw(
            vec![0.0; config.d_ff * config.d_model],
            vec![config.d_model, config.d_ff],
            if config.bias { Some(vec![0.0; config.d_model]) } else { None },
            config.precision,
        )?;

        Ok(Self {
            fc1,
            fc2,
            activation: config.activation,
            precision: config.precision,
        })
    }

    /// Creates MLP with specified layers.
    pub fn with_layers(
        fc1: Linear,
        fc2: Linear,
        activation: ActivationType,
        precision: Precision,
    ) -> Result<Self, MLPError> {
        // Validate dimensions
        if fc1.out_features() != fc2.in_features() {
            return Err(MLPError::DimensionMismatch {
                expected: fc1.out_features(),
                actual: fc2.in_features(),
            });
        }

        Ok(Self {
            fc1,
            fc2,
            activation,
            precision,
        })
    }

    /// Returns the input dimension.
    pub fn d_model(&self) -> usize {
        self.fc1.in_features()
    }

    /// Returns the hidden dimension.
    pub fn d_ff(&self) -> usize {
        self.fc1.out_features()
    }

    /// Returns the first linear layer (d_model → d_ff).
    pub fn fc1(&self) -> &Linear {
        &self.fc1
    }

    /// Returns the second linear layer (d_ff → d_model).
    pub fn fc2(&self) -> &Linear {
        &self.fc2
    }

    /// Forward pass.
    ///
    /// Input shape: (seq_len, d_model) or (batch, seq_len, d_model)
    /// Output shape: same as input
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, MLPError> {
        // First linear
        let hidden = self.fc1.forward(input)?;

        // Activation
        let activated = match self.activation {
            ActivationType::ReLU => activation::relu(&hidden),
            ActivationType::GELU => activation::gelu(&hidden, self.precision),
            ActivationType::SiLU => {
                // SiLU(x) = x * sigmoid(x)
                let sig = activation::sigmoid(&hidden, self.precision);
                hidden.hadamard(&sig)
            }
        };

        // Second linear
        let output = self.fc2.forward(&activated)?;

        Ok(output)
    }
}

/// Gated MLP block used in LLaMA (SwiGLU variant).
/// gate = activation(W_gate @ x)
/// up = W_up @ x
/// output = W_down @ (gate * up)
#[derive(Debug, Clone)]
pub struct GatedMLP {
    /// Gate projection.
    w_gate: Linear,
    /// Up projection.
    w_up: Linear,
    /// Down projection.
    w_down: Linear,
    /// Precision.
    precision: Precision,
}

impl GatedMLP {
    /// Creates a new gated MLP.
    pub fn new(d_model: usize, d_ff: usize, bias: bool, precision: Precision) -> Result<Self, MLPError> {
        let w_gate = Linear::from_raw(
            vec![0.0; d_model * d_ff],
            vec![d_ff, d_model],
            if bias { Some(vec![0.0; d_ff]) } else { None },
            precision,
        )?;

        let w_up = Linear::from_raw(
            vec![0.0; d_model * d_ff],
            vec![d_ff, d_model],
            if bias { Some(vec![0.0; d_ff]) } else { None },
            precision,
        )?;

        let w_down = Linear::from_raw(
            vec![0.0; d_ff * d_model],
            vec![d_model, d_ff],
            if bias { Some(vec![0.0; d_model]) } else { None },
            precision,
        )?;

        Ok(Self {
            w_gate,
            w_up,
            w_down,
            precision,
        })
    }

    /// Forward pass with SwiGLU activation.
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, MLPError> {
        // Gate with SiLU activation
        let gate = self.w_gate.forward(input)?;
        let gate_activated = {
            let sig = activation::sigmoid(&gate, self.precision);
            gate.hadamard(&sig)
        };

        // Up projection
        let up = self.w_up.forward(input)?;

        // Element-wise multiply
        let gated = gate_activated.hadamard(&up);

        // Down projection
        let output = self.w_down.forward(&gated)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_config() {
        let config = MLPConfig::new(64);
        assert_eq!(config.d_model, 64);
        assert_eq!(config.d_ff, 256);
    }

    #[test]
    fn test_mlp_creation() {
        let config = MLPConfig::new(64);
        let mlp = MLP::new(&config).unwrap();
        assert_eq!(mlp.d_model(), 64);
        assert_eq!(mlp.d_ff(), 256);
    }

    #[test]
    fn test_mlp_forward() {
        let config = MLPConfig::new(4).with_hidden_dim(8);
        let mlp = MLP::new(&config).unwrap();

        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = mlp.forward(&input).unwrap();

        // Output should have same shape as input (batch, d_model)
        assert_eq!(output.shape(), &vec![1, 4]);
    }

    #[test]
    fn test_gated_mlp_creation() {
        let mlp = GatedMLP::new(64, 256, true, Precision::F32).unwrap();
        let input = BoundedTensor::zeros(vec![10, 64]);
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.shape(), &vec![10, 64]);
    }
}
