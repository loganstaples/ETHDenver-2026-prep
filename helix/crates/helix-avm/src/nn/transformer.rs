//! Transformer block implementation with error bound tracking.
//!
//! Implements a full transformer decoder block: Attention → Add & Norm → MLP → Add & Norm

use super::attention::{AttentionConfig, AttentionError, MultiHeadAttention};
use super::linear::{Linear, LinearError};
use super::mlp::{ActivationType, MLP, MLPConfig, MLPError};
use crate::models::serialization::{Checkpoint, CheckpointMetadata, ModelCheckpoint, SerializationError};
use crate::ops::normalization;
use helix_core::types::{BoundedTensor, Precision};
use thiserror::Error;

/// Errors from transformer operations.
#[derive(Error, Debug)]
pub enum TransformerError {
    #[error("Attention error: {0}")]
    Attention(#[from] AttentionError),

    #[error("MLP error: {0}")]
    MLP(#[from] MLPError),

    #[error("Linear error: {0}")]
    Linear(#[from] LinearError),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Normalization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Layer normalization (GPT-2, BERT style).
    LayerNorm,
    /// RMS normalization (LLaMA style).
    RMSNorm,
}

/// Position of normalization relative to sublayer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormPosition {
    /// Pre-norm: norm(x) → sublayer → residual (GPT-2, LLaMA).
    Pre,
    /// Post-norm: sublayer → residual → norm (original Transformer).
    Post,
}

/// Configuration for a transformer block.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feedforward hidden dimension.
    pub d_ff: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Activation function for MLP.
    pub activation: ActivationType,
    /// Normalization type.
    pub norm_type: NormType,
    /// Normalization position.
    pub norm_position: NormPosition,
    /// Epsilon for normalization.
    pub norm_eps: f64,
    /// Whether to use bias in linear layers.
    pub bias: bool,
    /// Precision for computation.
    pub precision: Precision,
}

impl TransformerConfig {
    /// Creates a new transformer configuration.
    pub fn new(d_model: usize, num_heads: usize) -> Result<Self, TransformerError> {
        if d_model % num_heads != 0 {
            return Err(TransformerError::DimensionMismatch {
                expected: d_model,
                actual: num_heads,
            });
        }

        Ok(Self {
            d_model,
            num_heads,
            d_ff: d_model * 4,
            dropout: 0.0,
            activation: ActivationType::GELU,
            norm_type: NormType::LayerNorm,
            norm_position: NormPosition::Pre,
            norm_eps: 1e-5,
            bias: true,
            precision: Precision::F32,
        })
    }

    /// Creates a GPT-2 style configuration.
    pub fn gpt2_style(d_model: usize, num_heads: usize) -> Result<Self, TransformerError> {
        let mut config = Self::new(d_model, num_heads)?;
        config.activation = ActivationType::GELU;
        config.norm_type = NormType::LayerNorm;
        config.norm_position = NormPosition::Pre;
        Ok(config)
    }

    /// Creates a LLaMA style configuration.
    pub fn llama_style(d_model: usize, num_heads: usize) -> Result<Self, TransformerError> {
        let mut config = Self::new(d_model, num_heads)?;
        config.activation = ActivationType::SiLU;
        config.norm_type = NormType::RMSNorm;
        config.norm_position = NormPosition::Pre;
        config.bias = false;
        Ok(config)
    }

    /// Sets the feedforward dimension.
    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    /// Sets the activation function.
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Sets the normalization type.
    pub fn with_norm_type(mut self, norm_type: NormType) -> Self {
        self.norm_type = norm_type;
        self
    }
}

/// A single transformer block.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Self-attention layer.
    attention: MultiHeadAttention,
    /// MLP block.
    mlp: MLP,
    /// Configuration.
    config: TransformerConfig,
}

impl TransformerBlock {
    /// Creates a new transformer block from configuration.
    pub fn new(config: TransformerConfig) -> Result<Self, TransformerError> {
        let attention_config = AttentionConfig::new(config.d_model, config.num_heads)?;
        let attention = MultiHeadAttention::new(&attention_config)?;

        let mlp_config = MLPConfig::new(config.d_model)
            .with_hidden_dim(config.d_ff)
            .with_activation(config.activation)
            .with_bias(config.bias);
        let mlp = MLP::new(&mlp_config)?;

        Ok(Self {
            attention,
            mlp,
            config,
        })
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Applies normalization based on configuration.
    fn apply_norm(&self, input: &BoundedTensor) -> BoundedTensor {
        match self.config.norm_type {
            NormType::LayerNorm => {
                normalization::layer_norm(input, self.config.norm_eps, self.config.precision)
            }
            NormType::RMSNorm => {
                normalization::rms_norm(input, self.config.norm_eps, self.config.precision)
            }
        }
    }

    /// Forward pass for the transformer block.
    ///
    /// For pre-norm (GPT-2/LLaMA style):
    ///   x = x + attention(norm(x))
    ///   x = x + mlp(norm(x))
    ///
    /// For post-norm (original Transformer):
    ///   x = norm(x + attention(x))
    ///   x = norm(x + mlp(x))
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, TransformerError> {
        let x = match self.config.norm_position {
            NormPosition::Pre => {
                // Pre-norm attention
                let normed = self.apply_norm(input);
                let attn_out = self.attention.forward_self(&normed)?;
                let x = input.add(&attn_out); // Residual

                // Pre-norm MLP
                let normed = self.apply_norm(&x);
                let mlp_out = self.mlp.forward(&normed)?;
                x.add(&mlp_out) // Residual
            }
            NormPosition::Post => {
                // Post-norm attention
                let attn_out = self.attention.forward_self(input)?;
                let x = input.add(&attn_out);
                let x = self.apply_norm(&x);

                // Post-norm MLP
                let mlp_out = self.mlp.forward(&x)?;
                let x = x.add(&mlp_out);
                self.apply_norm(&x)
            }
        };

        Ok(x)
    }
}

/// A stack of transformer blocks.
#[derive(Debug, Clone)]
pub struct TransformerStack {
    /// Sequence of transformer blocks.
    blocks: Vec<TransformerBlock>,
    /// Configuration.
    config: TransformerConfig,
}

impl TransformerStack {
    /// Creates a new transformer stack.
    pub fn new(num_layers: usize, config: TransformerConfig) -> Result<Self, TransformerError> {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(config.clone())?);
        }

        Ok(Self { blocks, config })
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Forward pass through all transformer blocks.
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, TransformerError> {
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

/// Complete transformer model (embeddings + stack + output).
#[derive(Debug, Clone)]
pub struct TransformerModel {
    /// Transformer block stack.
    stack: TransformerStack,
    /// Final layer norm.
    final_norm_type: NormType,
    /// Normalization epsilon.
    norm_eps: f64,
    /// Precision.
    precision: Precision,
}

impl TransformerModel {
    /// Creates a new transformer model.
    pub fn new(num_layers: usize, config: TransformerConfig) -> Result<Self, TransformerError> {
        let final_norm_type = config.norm_type;
        let norm_eps = config.norm_eps;
        let precision = config.precision;
        let stack = TransformerStack::new(num_layers, config)?;

        Ok(Self {
            stack,
            final_norm_type,
            norm_eps,
            precision,
        })
    }

    /// Forward pass: hidden_states → transformer_stack → final_norm
    pub fn forward(&self, hidden_states: &BoundedTensor) -> Result<BoundedTensor, TransformerError> {
        let x = self.stack.forward(hidden_states)?;

        // Final normalization
        let output = match self.final_norm_type {
            NormType::LayerNorm => normalization::layer_norm(&x, self.norm_eps, self.precision),
            NormType::RMSNorm => normalization::rms_norm(&x, self.norm_eps, self.precision),
        };

        Ok(output)
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.stack.num_layers()
    }
}

// ---------------------------------------------------------------------------
// ProvableBlock — decompose transformer into circuit-compatible MLP steps
// ---------------------------------------------------------------------------

/// Trait for model components that can extract their parameters for checkpointing
/// and decompose into 2-layer MLP steps for circuit proving.
pub trait ProvableBlock {
    /// Collects all named parameters (weight tensors) in deterministic order.
    fn collect_parameters(&self) -> Vec<(String, BoundedTensor)>;

    /// Returns the total parameter count.
    fn param_count(&self) -> usize {
        self.collect_parameters()
            .iter()
            .map(|(_, t)| t.len())
            .sum()
    }

    /// Decomposes this block into pairs of Linear layers suitable for
    /// `build_training_witness`. Each pair represents one provable MLP step.
    ///
    /// For a transformer block with attention (Q,K,V,O projections) and MLP (fc1, fc2),
    /// the MLP sub-block is directly provable. Attention is decomposed into
    /// its linear projection pairs.
    fn decompose_to_mlp_pairs(&self) -> Vec<(String, Linear, Linear)>;
}

impl ProvableBlock for TransformerBlock {
    fn collect_parameters(&self) -> Vec<(String, BoundedTensor)> {
        let mut params = Vec::new();

        // Attention parameters (deterministic order: Q, K, V, O)
        params.push(("attn.w_q.weight".to_string(), self.attention.w_q().weights().clone()));
        if let Some(b) = self.attention.w_q().bias() {
            params.push(("attn.w_q.bias".to_string(), b.clone()));
        }
        params.push(("attn.w_k.weight".to_string(), self.attention.w_k().weights().clone()));
        if let Some(b) = self.attention.w_k().bias() {
            params.push(("attn.w_k.bias".to_string(), b.clone()));
        }
        params.push(("attn.w_v.weight".to_string(), self.attention.w_v().weights().clone()));
        if let Some(b) = self.attention.w_v().bias() {
            params.push(("attn.w_v.bias".to_string(), b.clone()));
        }
        params.push(("attn.w_o.weight".to_string(), self.attention.w_o().weights().clone()));
        if let Some(b) = self.attention.w_o().bias() {
            params.push(("attn.w_o.bias".to_string(), b.clone()));
        }

        // MLP parameters (fc1, fc2)
        params.push(("mlp.fc1.weight".to_string(), self.mlp.fc1().weights().clone()));
        if let Some(b) = self.mlp.fc1().bias() {
            params.push(("mlp.fc1.bias".to_string(), b.clone()));
        }
        params.push(("mlp.fc2.weight".to_string(), self.mlp.fc2().weights().clone()));
        if let Some(b) = self.mlp.fc2().bias() {
            params.push(("mlp.fc2.bias".to_string(), b.clone()));
        }

        params
    }

    fn decompose_to_mlp_pairs(&self) -> Vec<(String, Linear, Linear)> {
        let mut pairs = Vec::new();

        // The MLP sub-block is a direct 2-layer MLP: fc1 → activation → fc2
        // This maps exactly to the circuit's 2-layer MLP structure.
        pairs.push((
            "mlp".to_string(),
            self.mlp.fc1().clone(),
            self.mlp.fc2().clone(),
        ));

        // Attention projections can be decomposed as:
        // Step 1: Q,K projection (input → QK space)
        // Step 2: V,O projection (attention output → model space)
        // These are paired as (W_q, W_o) and (W_v, W_k) for proving linearity
        pairs.push((
            "attn.qo".to_string(),
            self.attention.w_q().clone(),
            self.attention.w_o().clone(),
        ));
        pairs.push((
            "attn.vk".to_string(),
            self.attention.w_v().clone(),
            self.attention.w_k().clone(),
        ));

        pairs
    }
}

impl ProvableBlock for TransformerStack {
    fn collect_parameters(&self) -> Vec<(String, BoundedTensor)> {
        let mut params = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            for (name, tensor) in block.collect_parameters() {
                params.push((format!("layers.{i}.{name}"), tensor));
            }
        }
        params
    }

    fn decompose_to_mlp_pairs(&self) -> Vec<(String, Linear, Linear)> {
        let mut pairs = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            for (name, l1, l2) in block.decompose_to_mlp_pairs() {
                pairs.push((format!("layers.{i}.{name}"), l1, l2));
            }
        }
        pairs
    }
}

impl ModelCheckpoint for TransformerBlock {
    fn checkpoint(&self) -> Checkpoint {
        let mut cp = Checkpoint::new(
            CheckpointMetadata::new("transformer_block", "transformer")
                .with_param_count(self.param_count()),
        );
        for (name, tensor) in self.collect_parameters() {
            cp.add_tensor(&name, &tensor);
        }
        cp
    }

    fn load_checkpoint(&mut self, _checkpoint: &Checkpoint) -> Result<(), SerializationError> {
        // Loading requires reconstructing layers from checkpoint tensors.
        // For now, return an error indicating this is not yet supported
        // because TransformerBlock contains non-public fields (attention, mlp)
        // that need coordinated reconstruction.
        Err(SerializationError::InvalidFormat(
            "TransformerBlock load_checkpoint requires full reconstruction; \
             use TransformerBlock::new() with loaded weights instead"
                .to_string(),
        ))
    }

    fn model_name(&self) -> &str {
        "transformer_block"
    }

    fn model_type(&self) -> &str {
        "transformer"
    }
}

impl ModelCheckpoint for TransformerStack {
    fn checkpoint(&self) -> Checkpoint {
        let mut cp = Checkpoint::new(
            CheckpointMetadata::new("transformer_stack", "transformer")
                .with_param_count(self.param_count()),
        );
        for (name, tensor) in self.collect_parameters() {
            cp.add_tensor(&name, &tensor);
        }
        cp
    }

    fn load_checkpoint(&mut self, _checkpoint: &Checkpoint) -> Result<(), SerializationError> {
        Err(SerializationError::InvalidFormat(
            "TransformerStack load_checkpoint requires full reconstruction"
                .to_string(),
        ))
    }

    fn model_name(&self) -> &str {
        "transformer_stack"
    }

    fn model_type(&self) -> &str {
        "transformer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::new(64, 8).unwrap();
        assert_eq!(config.d_model, 64);
        assert_eq!(config.d_ff, 256);
    }

    #[test]
    fn test_transformer_config_invalid() {
        let result = TransformerConfig::new(65, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_block_creation() {
        let config = TransformerConfig::new(64, 8).unwrap();
        let block = TransformerBlock::new(config).unwrap();
        assert_eq!(block.d_model(), 64);
    }

    #[test]
    fn test_transformer_block_forward() {
        let config = TransformerConfig::new(16, 4).unwrap();
        let block = TransformerBlock::new(config).unwrap();

        let input = BoundedTensor::zeros(vec![8, 16]); // 8 tokens, 16 dim
        let output = block.forward(&input).unwrap();

        assert_eq!(output.shape(), &vec![8, 16]);
    }

    #[test]
    fn test_transformer_stack() {
        let config = TransformerConfig::new(16, 4).unwrap();
        let stack = TransformerStack::new(3, config).unwrap();

        assert_eq!(stack.num_layers(), 3);

        let input = BoundedTensor::zeros(vec![8, 16]);
        let output = stack.forward(&input).unwrap();

        assert_eq!(output.shape(), &vec![8, 16]);
    }

    #[test]
    fn test_gpt2_style_config() {
        let config = TransformerConfig::gpt2_style(768, 12).unwrap();
        assert_eq!(config.activation, ActivationType::GELU);
        assert_eq!(config.norm_position, NormPosition::Pre);
    }

    #[test]
    fn test_llama_style_config() {
        let config = TransformerConfig::llama_style(768, 12).unwrap();
        assert_eq!(config.activation, ActivationType::SiLU);
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert!(!config.bias);
    }

    #[test]
    fn test_provable_block_collect_parameters() {
        let config = TransformerConfig::new(16, 4).unwrap();
        let block = TransformerBlock::new(config).unwrap();

        let params = block.collect_parameters();
        // Should have: Q,K,V,O weights + biases (8) + fc1,fc2 weights + biases (4) = 12
        assert_eq!(params.len(), 12);

        // Check naming convention
        let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"attn.w_q.weight"));
        assert!(names.contains(&"attn.w_q.bias"));
        assert!(names.contains(&"mlp.fc1.weight"));
        assert!(names.contains(&"mlp.fc2.weight"));
    }

    #[test]
    fn test_provable_block_param_count() {
        let config = TransformerConfig::new(16, 4).unwrap();
        let block = TransformerBlock::new(config).unwrap();

        let count = block.param_count();
        // 4 attention projections: 4 * (16*16 + 16) = 4 * 272 = 1088
        // 2 MLP layers: (16*64 + 64) + (64*16 + 16) = 1088 + 1040 = 2128
        // Total = 1088 + 2128 = 3216
        assert!(count > 0);
        assert_eq!(count, 4 * (16 * 16 + 16) + (16 * 64 + 64) + (64 * 16 + 16));
    }

    #[test]
    fn test_provable_block_decompose_to_mlp_pairs() {
        let config = TransformerConfig::new(16, 4).unwrap();
        let block = TransformerBlock::new(config).unwrap();

        let pairs = block.decompose_to_mlp_pairs();
        // Should have 3 pairs: mlp, attn.qo, attn.vk
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "mlp");
        assert_eq!(pairs[1].0, "attn.qo");
        assert_eq!(pairs[2].0, "attn.vk");

        // MLP pair should have compatible dimensions
        let (_, fc1, fc2) = &pairs[0];
        assert_eq!(fc1.out_features(), fc2.in_features());
    }

    #[test]
    fn test_provable_stack_collect_parameters() {
        let config = TransformerConfig::new(16, 4).unwrap();
        let stack = TransformerStack::new(2, config).unwrap();

        let params = stack.collect_parameters();
        // 2 blocks * 12 params each = 24
        assert_eq!(params.len(), 24);

        // Check hierarchical naming
        let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"layers.0.attn.w_q.weight"));
        assert!(names.contains(&"layers.1.mlp.fc2.weight"));
    }

    #[test]
    fn test_transformer_block_checkpoint() {
        let config = TransformerConfig::new(16, 4).unwrap();
        let block = TransformerBlock::new(config).unwrap();

        let checkpoint = block.checkpoint();
        assert_eq!(checkpoint.metadata.model_type, "transformer");
        assert!(checkpoint.total_params() > 0);
        assert_eq!(checkpoint.tensor_names().len(), 12);
    }
}
