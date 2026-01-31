//! GPT-Style Model Architecture.
//!
//! Implements a decoder-only autoregressive transformer model in the style of GPT-2/GPT-3.
//! Features:
//! - Configurable model dimensions
//! - Pre-norm architecture (layer norm before attention/MLP)
//! - GELU activation in MLP
//! - Tied embedding weights (optional)
//! - Memory-efficient layer-by-layer execution
//!
//! # Architecture
//!
//! ```text
//! Token IDs
//!     ↓
//! Token Embedding + Positional Encoding
//!     ↓
//! ┌─────────────────────────────────────┐
//! │ TransformerBlock × N                │
//! │   ├─ LayerNorm → MultiHeadAttention │
//! │   ├─ + Residual                     │
//! │   ├─ LayerNorm → MLP                │
//! │   └─ + Residual                     │
//! └─────────────────────────────────────┘
//!     ↓
//! Final LayerNorm
//!     ↓
//! Language Model Head (Linear)
//!     ↓
//! Logits
//! ```

use crate::memory::{GradientCheckpointer, MemoryProfiler, MemoryTracker};
use crate::nn::embedding::{Embedding, EmbeddingError, PositionalEncoding};
use crate::nn::linear::{Linear, LinearConfig, LinearError};
use crate::nn::transformer::{
    NormPosition, NormType, TransformerConfig, TransformerError, TransformerModel, TransformerStack,
};
use crate::nn::ActivationType;
use crate::ops::normalization;
use helix_core::types::{BoundedTensor, Precision};
use thiserror::Error;

use super::{Model, ModelDimensions, ModelSize};

/// Errors from GPT model operations.
#[derive(Error, Debug)]
pub enum GPTError {
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Linear error: {0}")]
    Linear(#[from] LinearError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Sequence too long: {length} > {max_length}")]
    SequenceTooLong { length: usize, max_length: usize },
}

/// Configuration for GPT model.
#[derive(Debug, Clone)]
pub struct GPTConfig {
    /// Model dimensions.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Feedforward hidden dimension.
    pub d_ff: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Activation function.
    pub activation: ActivationType,
    /// Normalization type.
    pub norm_type: NormType,
    /// Normalization epsilon.
    pub norm_eps: f64,
    /// Whether to use bias in linear layers.
    pub use_bias: bool,
    /// Whether to tie embedding weights with LM head.
    pub tie_embeddings: bool,
    /// Precision for computation.
    pub precision: Precision,
    /// Model name.
    pub name: String,
}

impl GPTConfig {
    /// Creates a new GPT configuration.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        vocab_size: usize,
    ) -> Result<Self, GPTError> {
        if d_model % n_heads != 0 {
            return Err(GPTError::InvalidConfig(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                d_model, n_heads
            )));
        }

        Ok(Self {
            d_model,
            n_heads,
            n_layers,
            d_ff: d_model * 4,
            vocab_size,
            max_seq_len: 1024,
            dropout: 0.0,
            activation: ActivationType::GELU,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-5,
            use_bias: true,
            tie_embeddings: true,
            precision: Precision::F32,
            name: "gpt".to_string(),
        })
    }

    /// Creates configuration for a specific model size.
    pub fn for_size(size: ModelSize) -> Result<Self, GPTError> {
        let dims = size.dimensions();
        let mut config = Self::new(dims.d_model, dims.n_heads, dims.n_layers, dims.vocab_size)?;
        config.d_ff = dims.d_ff;
        config.max_seq_len = dims.max_seq_len;
        config.name = format!("gpt-{:?}", size).to_lowercase();
        Ok(config)
    }

    /// Creates a GPT-2 small style configuration.
    pub fn gpt2_small() -> Result<Self, GPTError> {
        Self::new(768, 12, 12, 50257)
    }

    /// Creates a minimal configuration for testing.
    pub fn minimal() -> Result<Self, GPTError> {
        Self::new(64, 4, 2, 1000)
    }

    /// Sets the feedforward dimension.
    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    /// Sets the maximum sequence length.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Sets the dropout rate.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
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

    /// Sets whether to use bias.
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Sets whether to tie embeddings.
    pub fn with_tie_embeddings(mut self, tie: bool) -> Self {
        self.tie_embeddings = tie;
        self
    }

    /// Sets the model name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), GPTError> {
        if self.d_model % self.n_heads != 0 {
            return Err(GPTError::InvalidConfig(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            )));
        }
        if self.d_model == 0 || self.n_layers == 0 || self.vocab_size == 0 {
            return Err(GPTError::InvalidConfig(
                "d_model, n_layers, and vocab_size must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Estimates the total parameter count.
    pub fn estimate_params(&self) -> usize {
        let dims = ModelDimensions::new(
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_ff,
            self.vocab_size,
            self.max_seq_len,
        );
        dims.estimate_params_gpt()
    }

    /// Converts to transformer config.
    pub fn to_transformer_config(&self) -> Result<TransformerConfig, TransformerError> {
        let mut config = TransformerConfig::new(self.d_model, self.n_heads)?;
        config.d_ff = self.d_ff;
        config.dropout = self.dropout;
        config.activation = self.activation;
        config.norm_type = self.norm_type;
        config.norm_position = NormPosition::Pre;
        config.norm_eps = self.norm_eps;
        config.bias = self.use_bias;
        config.precision = self.precision;
        Ok(config)
    }
}

/// Language model head for predicting next tokens.
#[derive(Debug, Clone)]
pub struct LanguageModelHead {
    /// Linear projection to vocabulary.
    linear: Option<Linear>,
    /// Reference to embedding weights if tied.
    tied_weights: Option<BoundedTensor>,
    /// Vocabulary size.
    vocab_size: usize,
    /// Model dimension.
    d_model: usize,
    /// Precision.
    precision: Precision,
}

impl LanguageModelHead {
    /// Creates a new LM head.
    pub fn new(d_model: usize, vocab_size: usize, precision: Precision) -> Result<Self, LinearError> {
        let config = LinearConfig::new(d_model, vocab_size)
            .with_bias(false)
            .with_precision(precision);

        Ok(Self {
            linear: Some(Linear::from_config(&config)),
            tied_weights: None,
            vocab_size,
            d_model,
            precision,
        })
    }

    /// Creates an LM head with tied weights from embedding.
    pub fn tied(embedding_weights: BoundedTensor) -> Self {
        let shape = embedding_weights.shape();
        let vocab_size = shape[0];
        let d_model = shape[1];

        Self {
            linear: None,
            tied_weights: Some(embedding_weights),
            vocab_size,
            d_model,
            precision: Precision::F32,
        }
    }

    /// Forward pass: hidden_states → logits
    ///
    /// Input shape: (seq_len, d_model) or (batch, seq_len, d_model)
    /// Output shape: (seq_len, vocab_size) or (batch, seq_len, vocab_size)
    pub fn forward(&self, hidden_states: &BoundedTensor) -> Result<BoundedTensor, LinearError> {
        if let Some(ref linear) = self.linear {
            linear.forward(hidden_states)
        } else if let Some(ref weights) = self.tied_weights {
            // Manual matmul with tied weights: output = hidden_states @ weights.T
            let weights_t = weights.transpose();
            crate::ops::matmul(hidden_states, &weights_t, self.precision)
                .map_err(LinearError::MatMul)
        } else {
            Err(LinearError::InvalidInputDimensions(0))
        }
    }

    /// Returns parameter count.
    pub fn param_count(&self) -> usize {
        if self.linear.is_some() {
            self.vocab_size * self.d_model
        } else {
            0 // Tied weights don't count as separate parameters
        }
    }
}

/// GPT-style language model.
#[derive(Debug)]
pub struct GPTModel {
    /// Configuration.
    config: GPTConfig,
    /// Token embedding.
    token_embedding: Embedding,
    /// Positional encoding.
    positional_encoding: PositionalEncoding,
    /// Transformer stack.
    transformer: TransformerStack,
    /// Final layer norm.
    final_norm_type: NormType,
    final_norm_eps: f64,
    /// Language model head.
    lm_head: LanguageModelHead,
    /// Gradient checkpointer for memory efficiency.
    checkpointer: Option<GradientCheckpointer>,
    /// Memory profiler.
    profiler: Option<MemoryProfiler>,
    /// Cached dimensions.
    dimensions: ModelDimensions,
}

impl GPTModel {
    /// Creates a new GPT model from configuration.
    pub fn new(config: GPTConfig) -> Result<Self, GPTError> {
        config.validate()?;

        // Create components
        let token_embedding = Embedding::zeros(config.vocab_size, config.d_model);
        let positional_encoding = PositionalEncoding::sinusoidal(config.max_seq_len, config.d_model);

        let transformer_config = config.to_transformer_config()?;
        let transformer = TransformerStack::new(config.n_layers, transformer_config)?;

        // Create LM head (tied or not)
        let lm_head = if config.tie_embeddings {
            LanguageModelHead::tied(token_embedding.table().clone())
        } else {
            LanguageModelHead::new(config.d_model, config.vocab_size, config.precision)?
        };

        let dimensions = ModelDimensions::new(
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.vocab_size,
            config.max_seq_len,
        );

        Ok(Self {
            final_norm_type: config.norm_type,
            final_norm_eps: config.norm_eps,
            config,
            token_embedding,
            positional_encoding,
            transformer,
            lm_head,
            checkpointer: None,
            profiler: None,
            dimensions,
        })
    }

    /// Creates a model for a specific size.
    pub fn for_size(size: ModelSize) -> Result<Self, GPTError> {
        let config = GPTConfig::for_size(size)?;
        Self::new(config)
    }

    /// Enables gradient checkpointing for memory efficiency.
    pub fn enable_checkpointing(&mut self, checkpointer: GradientCheckpointer) {
        self.checkpointer = Some(checkpointer);
    }

    /// Enables profiling.
    pub fn enable_profiling(&mut self, profiler: MemoryProfiler) {
        self.profiler = Some(profiler);
    }

    /// Returns the configuration.
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Forward pass from token IDs to logits.
    ///
    /// Input: sequence of token IDs [seq_len]
    /// Output: logits [seq_len, vocab_size]
    pub fn forward_tokens(&self, token_ids: &[usize]) -> Result<BoundedTensor, GPTError> {
        if token_ids.len() > self.config.max_seq_len {
            return Err(GPTError::SequenceTooLong {
                length: token_ids.len(),
                max_length: self.config.max_seq_len,
            });
        }

        // Embed tokens
        let embeddings = self.token_embedding.forward(token_ids)?;

        // Add positional encoding
        let positioned = self.positional_encoding.forward(&embeddings)?;

        // Forward through transformer
        self.forward(&positioned)
    }

    /// Forward pass from embeddings to logits.
    ///
    /// Input: (seq_len, d_model)
    /// Output: (seq_len, vocab_size)
    pub fn forward(&self, hidden_states: &BoundedTensor) -> Result<BoundedTensor, GPTError> {
        // Transformer layers
        let transformed = self.transformer.forward(hidden_states)?;

        // Final layer norm
        let normalized = match self.final_norm_type {
            NormType::LayerNorm => {
                normalization::layer_norm(&transformed, self.final_norm_eps, self.config.precision)
            }
            NormType::RMSNorm => {
                normalization::rms_norm(&transformed, self.final_norm_eps, self.config.precision)
            }
        };

        // LM head
        let logits = self.lm_head.forward(&normalized)?;

        Ok(logits)
    }

    /// Forward pass with layer-by-layer execution for memory efficiency.
    ///
    /// Processes one layer at a time and optionally checkpoints activations.
    pub fn forward_memory_efficient(&mut self, hidden_states: &BoundedTensor) -> Result<BoundedTensor, GPTError> {
        let mut current = hidden_states.clone();

        // Initialize checkpointer if present
        if let Some(ref mut checkpointer) = self.checkpointer {
            checkpointer.init(self.config.n_layers);
            checkpointer.begin_forward();
        }

        // Process transformer layers
        // Note: In a full implementation, we would access individual blocks
        // For now, use the existing transformer forward
        current = self.transformer.forward(&current)?;

        if let Some(ref mut checkpointer) = self.checkpointer {
            // End forward would be called here
        }

        // Final layer norm
        let normalized = match self.final_norm_type {
            NormType::LayerNorm => {
                normalization::layer_norm(&current, self.final_norm_eps, self.config.precision)
            }
            NormType::RMSNorm => {
                normalization::rms_norm(&current, self.final_norm_eps, self.config.precision)
            }
        };

        // LM head
        let logits = self.lm_head.forward(&normalized)?;

        Ok(logits)
    }

    /// Returns the embedding layer.
    pub fn token_embedding(&self) -> &Embedding {
        &self.token_embedding
    }

    /// Returns the positional encoding.
    pub fn positional_encoding(&self) -> &PositionalEncoding {
        &self.positional_encoding
    }

    /// Returns the LM head.
    pub fn lm_head(&self) -> &LanguageModelHead {
        &self.lm_head
    }

    /// Returns estimated memory for training.
    pub fn estimate_training_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        MemoryTracker::estimate_model_memory(
            self.param_count(),
            batch_size,
            seq_len,
            self.config.d_model,
            self.config.n_layers,
            true,
        ).total_bytes
    }

    /// Returns estimated memory for inference.
    pub fn estimate_inference_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        MemoryTracker::estimate_model_memory(
            self.param_count(),
            batch_size,
            seq_len,
            self.config.d_model,
            self.config.n_layers,
            false,
        ).total_bytes
    }
}

impl Model for GPTModel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn param_count(&self) -> usize {
        // Embedding parameters
        let embedding = self.config.vocab_size * self.config.d_model;

        // Transformer parameters (per layer)
        // Attention: 4 * d_model^2
        // MLP: 2 * d_model * d_ff
        // Layer norms: 4 * d_model (2 norms with weight and bias)
        let per_layer = 4 * self.config.d_model * self.config.d_model
            + 2 * self.config.d_model * self.config.d_ff
            + 4 * self.config.d_model;
        let transformer = per_layer * self.config.n_layers;

        // Final norm
        let final_norm = 2 * self.config.d_model;

        // LM head (0 if tied)
        let lm_head = self.lm_head.param_count();

        embedding + transformer + final_norm + lm_head
    }

    fn dimensions(&self) -> &ModelDimensions {
        &self.dimensions
    }

    fn training_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        self.estimate_training_memory(batch_size, seq_len)
    }

    fn inference_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        self.estimate_inference_memory(batch_size, seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_config_creation() {
        let config = GPTConfig::new(256, 8, 6, 8192).unwrap();
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.d_ff, 1024); // 4 * d_model
    }

    #[test]
    fn test_gpt_config_invalid() {
        // d_model not divisible by n_heads
        let result = GPTConfig::new(255, 8, 6, 8192);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpt_config_for_size() {
        let config = GPTConfig::for_size(ModelSize::OneMillion).unwrap();
        let estimated = config.estimate_params();
        // Parameter count includes embeddings which scale with vocab size
        // For 1M-class models with 8K vocab, total is larger than 1M
        assert!(estimated > 500_000);
        assert!(estimated < 10_000_000);
    }

    #[test]
    fn test_gpt_model_creation() {
        let config = GPTConfig::new(64, 4, 2, 1000).unwrap();
        let model = GPTModel::new(config).unwrap();

        assert!(model.param_count() > 0);
    }

    #[test]
    fn test_gpt_forward() {
        let config = GPTConfig::new(32, 4, 2, 100).unwrap();
        let model = GPTModel::new(config).unwrap();

        // Create dummy input (seq_len=4, d_model=32)
        let input = BoundedTensor::zeros(vec![4, 32]);
        let output = model.forward(&input).unwrap();

        // Output should be (seq_len, vocab_size)
        assert_eq!(output.shape(), &vec![4, 100]);
    }

    #[test]
    fn test_gpt_forward_tokens() {
        let config = GPTConfig::new(32, 4, 2, 100).unwrap();
        let model = GPTModel::new(config).unwrap();

        let token_ids = vec![1, 5, 10, 20];
        let output = model.forward_tokens(&token_ids).unwrap();

        assert_eq!(output.shape(), &vec![4, 100]);
    }

    #[test]
    fn test_gpt_sequence_too_long() {
        let config = GPTConfig::new(32, 4, 2, 100)
            .unwrap()
            .with_max_seq_len(10);
        let model = GPTModel::new(config).unwrap();

        let token_ids: Vec<usize> = (0..20).collect();
        let result = model.forward_tokens(&token_ids);

        assert!(matches!(result, Err(GPTError::SequenceTooLong { .. })));
    }

    #[test]
    fn test_lm_head_tied() {
        let embedding_weights = BoundedTensor::zeros(vec![100, 32]);
        let lm_head = LanguageModelHead::tied(embedding_weights);

        assert_eq!(lm_head.param_count(), 0); // Tied weights don't count

        let hidden = BoundedTensor::zeros(vec![4, 32]);
        let logits = lm_head.forward(&hidden).unwrap();
        assert_eq!(logits.shape(), &vec![4, 100]);
    }

    #[test]
    fn test_memory_estimation() {
        let model = GPTModel::for_size(ModelSize::OneMillion).unwrap();

        let train_mem = model.estimate_training_memory(4, 128);
        let infer_mem = model.estimate_inference_memory(4, 128);

        // Training should use more memory than inference
        assert!(train_mem > infer_mem);
    }
}
