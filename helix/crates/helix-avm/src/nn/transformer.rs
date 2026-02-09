//! Transformer block implementation with error bound tracking.
//!
//! Implements a full transformer decoder block: Attention → Add & Norm → MLP → Add & Norm

use super::attention::{AttentionConfig, AttentionError, AttentionGradients, MultiHeadAttention};
use super::linear::{Linear, LinearError};
use super::mlp::{ActivationType, MLP, MLPConfig, MLPError, MLPGradients};
use crate::models::serialization::{Checkpoint, CheckpointMetadata, ModelCheckpoint, SerializationError};
use crate::ops::{self, normalization, MatMulError};
use helix_core::types::{BoundedTensor, BoundedValue, Precision};
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

    #[error("Matrix multiplication error: {0}")]
    MatMul(#[from] MatMulError),

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

    /// Backward pass for transformer block.
    ///
    /// For pre-norm, the forward is:
    ///   attn_input = norm(input)
    ///   attn_out = attention(attn_input)
    ///   residual1 = input + attn_out
    ///   mlp_input = norm(residual1)
    ///   mlp_out = mlp(mlp_input)
    ///   output = residual1 + mlp_out
    ///
    /// Backward reverses this chain. For post-norm, a similar chain applies.
    ///
    /// Returns TransformerGradients containing gradient for input plus attention and MLP gradients.
    pub fn backward(
        &self,
        grad_output: &BoundedTensor,
        input: &BoundedTensor,
    ) -> Result<TransformerGradients, TransformerError> {
        let precision = self.config.precision;

        match self.config.norm_position {
            NormPosition::Pre => {
                // Recompute forward intermediates
                let attn_input_normed = self.apply_norm(input);
                let attn_out = self.attention.forward_self(&attn_input_normed)?;
                let residual1 = input.add(&attn_out);
                let mlp_input_normed = self.apply_norm(&residual1);

                // Recompute MLP intermediates for backward
                let hidden_pre_activation = self.mlp.fc1().forward(&mlp_input_normed)?;
                let hidden_post_activation = match self.config.activation {
                    ActivationType::ReLU => {
                        crate::ops::activation::relu(&hidden_pre_activation)
                    }
                    ActivationType::GELU => {
                        crate::ops::activation::gelu(&hidden_pre_activation, precision)
                    }
                    ActivationType::SiLU => {
                        let sig = crate::ops::activation::sigmoid(
                            &hidden_pre_activation,
                            precision,
                        );
                        hidden_pre_activation.hadamard(&sig)
                    }
                };

                // Step 1: output = residual1 + mlp_out
                // dL/d_residual1 = grad_output (from residual connection)
                // dL/d_mlp_out = grad_output
                let grad_mlp_out = grad_output.clone();

                // Step 2: Backward through MLP
                let mlp_grads = self.mlp.backward(
                    &grad_mlp_out,
                    &mlp_input_normed,
                    &hidden_pre_activation,
                    &hidden_post_activation,
                )?;

                // dL/d_mlp_input_normed = mlp_grads.grad_input
                // Through LayerNorm backward (simplified: pass through)
                let grad_residual1_from_mlp = &mlp_grads.grad_input;

                // dL/d_residual1 = grad_output + grad from MLP path
                let grad_residual1 = grad_output.add(grad_residual1_from_mlp);

                // Step 3: residual1 = input + attn_out
                // dL/d_input_from_residual = grad_residual1
                // dL/d_attn_out = grad_residual1
                let grad_attn_out = grad_residual1.clone();

                // Step 4: Backward through attention
                // Recompute attention weights for backward
                let q_proj = self.attention.w_q().forward(&attn_input_normed)?;
                let k_proj = self.attention.w_k().forward(&attn_input_normed)?;
                let _v_proj = self.attention.w_v().forward(&attn_input_normed)?;
                let d_k = self.attention.head_dim() as f64;
                let scale = 1.0 / d_k.sqrt();

                let k_t = k_proj.transpose();
                let scores = ops::matmul(&q_proj, &k_t, precision)?;
                let scaled_scores = scores.scale(BoundedValue::exact(scale));
                let attention_weights =
                    softmax_rows_for_backward(&scaled_scores);

                let attn_grads = self.attention.backward(
                    &grad_attn_out,
                    &attn_input_normed,
                    &attn_input_normed,
                    &attn_input_normed,
                    &attention_weights,
                )?;

                // Through LayerNorm backward for attention input (simplified pass-through)
                let grad_input_from_attn = &attn_grads.grad_query;

                // dL/d_input = grad_residual1 + grad from attention path through norm
                let grad_input = grad_residual1.add(grad_input_from_attn);

                Ok(TransformerGradients {
                    grad_input,
                    attention_grads: attn_grads,
                    mlp_grads,
                })
            }
            NormPosition::Post => {
                // Post-norm backward
                // Recompute intermediates
                let attn_out = self.attention.forward_self(input)?;
                let pre_norm1 = input.add(&attn_out);
                let residual1 = self.apply_norm(&pre_norm1);

                let hidden_pre_activation = self.mlp.fc1().forward(&residual1)?;
                let hidden_post_activation = match self.config.activation {
                    ActivationType::ReLU => {
                        crate::ops::activation::relu(&hidden_pre_activation)
                    }
                    ActivationType::GELU => {
                        crate::ops::activation::gelu(&hidden_pre_activation, precision)
                    }
                    ActivationType::SiLU => {
                        let sig = crate::ops::activation::sigmoid(
                            &hidden_pre_activation,
                            precision,
                        );
                        hidden_pre_activation.hadamard(&sig)
                    }
                };

                // Backward through final norm (simplified pass-through)
                let grad_pre_norm2 = grad_output.clone();

                // Backward through MLP
                let mlp_grads = self.mlp.backward(
                    &grad_pre_norm2,
                    &residual1,
                    &hidden_pre_activation,
                    &hidden_post_activation,
                )?;

                let grad_residual1 = grad_pre_norm2.add(&mlp_grads.grad_input);

                // Backward through first norm (simplified pass-through)
                let grad_pre_norm1 = grad_residual1;

                // Backward through attention
                // Recompute attention weights
                let q_proj = self.attention.w_q().forward(input)?;
                let k_proj = self.attention.w_k().forward(input)?;
                let _v_proj = self.attention.w_v().forward(input)?;
                let d_k = self.attention.head_dim() as f64;
                let scale = 1.0 / d_k.sqrt();
                let k_t = k_proj.transpose();
                let scores = ops::matmul(&q_proj, &k_t, precision)?;
                let scaled_scores = scores.scale(BoundedValue::exact(scale));
                let attention_weights =
                    softmax_rows_for_backward(&scaled_scores);

                let attn_grads = self.attention.backward(
                    &grad_pre_norm1,
                    input,
                    input,
                    input,
                    &attention_weights,
                )?;

                let grad_input = grad_pre_norm1.add(&attn_grads.grad_query);

                Ok(TransformerGradients {
                    grad_input,
                    attention_grads: attn_grads,
                    mlp_grads,
                })
            }
        }
    }

    /// Returns a reference to the attention layer.
    pub fn attention(&self) -> &MultiHeadAttention {
        &self.attention
    }

    /// Returns a reference to the MLP block.
    pub fn mlp_block(&self) -> &MLP {
        &self.mlp
    }

    /// Returns a reference to the transformer config.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

/// Gradients produced by the transformer block backward pass.
#[derive(Debug, Clone)]
pub struct TransformerGradients {
    /// Gradient with respect to the block input.
    pub grad_input: BoundedTensor,
    /// Gradients from the attention sub-layer.
    pub attention_grads: AttentionGradients,
    /// Gradients from the MLP sub-layer.
    pub mlp_grads: MLPGradients,
}

/// Row-wise softmax helper for transformer backward (recomputes attention weights).
fn softmax_rows_for_backward(input: &BoundedTensor) -> BoundedTensor {
    if !input.is_matrix() {
        return input.clone();
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.data();
    let mut result = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        let row_start = i * cols;
        let row_max = (0..cols)
            .map(|j| data[row_start + j].value())
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_vals: Vec<f64> = (0..cols)
            .map(|j| (data[row_start + j].value() - row_max).exp())
            .collect();
        let sum: f64 = exp_vals.iter().sum();

        for exp_val in &exp_vals {
            result.push(BoundedValue::exact(exp_val / sum));
        }
    }

    BoundedTensor::new(result, input.shape().clone())
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
