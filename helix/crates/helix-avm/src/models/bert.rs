//! BERT-Style Model Architecture.
//!
//! Implements an encoder-only bidirectional transformer model in the style of BERT.
//! Features:
//! - Bidirectional attention (no causal masking)
//! - Token + Segment + Position embeddings
//! - Post-norm architecture (original BERT)
//! - Pooler for classification tasks
//! - Masked language modeling head
//!
//! # Architecture
//!
//! ```text
//! Token IDs + Segment IDs
//!         ↓
//! Token Embedding + Segment Embedding + Positional Encoding
//!         ↓
//! ┌─────────────────────────────────────┐
//! │ TransformerBlock × N                │
//! │   ├─ MultiHeadAttention → LayerNorm │
//! │   ├─ + Residual                     │
//! │   ├─ MLP → LayerNorm                │
//! │   └─ + Residual                     │
//! └─────────────────────────────────────┘
//!         ↓
//!    ┌────┴────┐
//!    ↓         ↓
//! Pooler    Sequence Output
//!    ↓
//! [CLS] representation
//! ```

use crate::memory::{MemoryProfiler, MemoryTracker};
use crate::nn::embedding::{Embedding, EmbeddingError, PositionalEncoding};
use crate::nn::linear::{Linear, LinearConfig, LinearError};
use crate::nn::transformer::{
    NormPosition, NormType, TransformerConfig, TransformerError, TransformerStack,
};
use crate::nn::ActivationType;
use crate::ops::{self, activation};
use helix_core::types::{BoundedTensor, BoundedValue, Precision};
use thiserror::Error;

use super::{Model, ModelDimensions, ModelSize};

/// Errors from BERT model operations.
#[derive(Error, Debug)]
pub enum BERTError {
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Linear error: {0}")]
    Linear(#[from] LinearError),

    #[error("MatMul error: {0}")]
    MatMul(#[from] ops::MatMulError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Sequence too long: {length} > {max_length}")]
    SequenceTooLong { length: usize, max_length: usize },

    #[error("Segment ID out of range: {id} >= {max_segments}")]
    InvalidSegmentId { id: usize, max_segments: usize },
}

/// Configuration for BERT model.
#[derive(Debug, Clone)]
pub struct BERTConfig {
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
    /// Number of segment types (typically 2 for sentence A/B).
    pub num_segments: usize,
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
    /// Precision for computation.
    pub precision: Precision,
    /// Model name.
    pub name: String,
}

impl BERTConfig {
    /// Creates a new BERT configuration.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        vocab_size: usize,
    ) -> Result<Self, BERTError> {
        if d_model % n_heads != 0 {
            return Err(BERTError::InvalidConfig(format!(
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
            max_seq_len: 512,
            num_segments: 2,
            dropout: 0.0,
            activation: ActivationType::GELU,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-12, // BERT uses smaller epsilon
            use_bias: true,
            precision: Precision::F32,
            name: "bert".to_string(),
        })
    }

    /// Creates configuration for a specific model size.
    pub fn for_size(size: ModelSize) -> Result<Self, BERTError> {
        let dims = size.dimensions();
        let mut config = Self::new(dims.d_model, dims.n_heads, dims.n_layers, dims.vocab_size)?;
        config.d_ff = dims.d_ff;
        config.max_seq_len = dims.max_seq_len;
        config.name = format!("bert-{:?}", size).to_lowercase();
        Ok(config)
    }

    /// Creates a BERT-base style configuration.
    pub fn bert_base() -> Result<Self, BERTError> {
        Self::new(768, 12, 12, 30522)
    }

    /// Creates a BERT-small style configuration.
    pub fn bert_small() -> Result<Self, BERTError> {
        Self::new(512, 8, 4, 30522)
    }

    /// Creates a minimal configuration for testing.
    pub fn minimal() -> Result<Self, BERTError> {
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

    /// Sets the number of segments.
    pub fn with_num_segments(mut self, num_segments: usize) -> Self {
        self.num_segments = num_segments;
        self
    }

    /// Sets the dropout rate.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Sets the model name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), BERTError> {
        if self.d_model % self.n_heads != 0 {
            return Err(BERTError::InvalidConfig(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            )));
        }
        if self.d_model == 0 || self.n_layers == 0 || self.vocab_size == 0 {
            return Err(BERTError::InvalidConfig(
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
        dims.estimate_params_bert()
    }

    /// Converts to transformer config.
    pub fn to_transformer_config(&self) -> Result<TransformerConfig, TransformerError> {
        let mut config = TransformerConfig::new(self.d_model, self.n_heads)?;
        config.d_ff = self.d_ff;
        config.dropout = self.dropout;
        config.activation = self.activation;
        config.norm_type = self.norm_type;
        config.norm_position = NormPosition::Post; // BERT uses post-norm
        config.norm_eps = self.norm_eps;
        config.bias = self.use_bias;
        config.precision = self.precision;
        Ok(config)
    }
}

/// Pooler layer for extracting sentence-level representation.
#[derive(Debug, Clone)]
pub struct BERTPooler {
    /// Linear transformation.
    dense: Linear,
    /// Precision.
    precision: Precision,
}

impl BERTPooler {
    /// Creates a new pooler.
    pub fn new(d_model: usize, precision: Precision) -> Result<Self, LinearError> {
        let config = LinearConfig::new(d_model, d_model)
            .with_bias(true)
            .with_precision(precision);

        Ok(Self {
            dense: Linear::from_config(&config),
            precision,
        })
    }

    /// Forward pass: extracts [CLS] token representation.
    ///
    /// Input: sequence output (seq_len, d_model) or (batch, seq_len, d_model)
    /// Output: pooled output (d_model) or (batch, d_model)
    pub fn forward(&self, sequence_output: &BoundedTensor) -> Result<BoundedTensor, BERTError> {
        let shape = sequence_output.shape();

        // Extract first token ([CLS])
        let first_token = if shape.len() == 2 {
            // (seq_len, d_model) -> (d_model)
            let d_model = shape[1];
            let data: Vec<BoundedValue<f64>> = sequence_output.data()[..d_model].to_vec();
            BoundedTensor::new(data, vec![d_model])
        } else if shape.len() == 3 {
            // (batch, seq_len, d_model) -> (batch, d_model)
            let batch_size = shape[0];
            let seq_len = shape[1];
            let d_model = shape[2];

            let mut data = Vec::with_capacity(batch_size * d_model);
            for b in 0..batch_size {
                let start = b * seq_len * d_model;
                data.extend_from_slice(&sequence_output.data()[start..start + d_model]);
            }
            BoundedTensor::new(data, vec![batch_size, d_model])
        } else {
            return Err(BERTError::InvalidConfig(
                "Invalid sequence output dimensions".to_string(),
            ));
        };

        // Dense transformation
        let pooled = self.dense.forward(&first_token)?;

        // Tanh activation
        let activated = activation::tanh(&pooled, self.precision);

        Ok(activated)
    }

    /// Returns parameter count.
    pub fn param_count(&self) -> usize {
        let shape = self.dense.weights().shape();
        shape[0] * shape[1] + shape[0] // weights + bias
    }
}

/// Masked Language Model head for BERT.
#[derive(Debug, Clone)]
pub struct MLMHead {
    /// Dense transformation.
    dense: Linear,
    /// Output projection (often tied with embeddings).
    output: Option<Linear>,
    /// Tied embedding weights.
    tied_weights: Option<BoundedTensor>,
    /// Precision.
    precision: Precision,
}

impl MLMHead {
    /// Creates a new MLM head.
    pub fn new(d_model: usize, vocab_size: usize, precision: Precision) -> Result<Self, LinearError> {
        let dense_config = LinearConfig::new(d_model, d_model)
            .with_bias(true)
            .with_precision(precision);

        let output_config = LinearConfig::new(d_model, vocab_size)
            .with_bias(true)
            .with_precision(precision);

        Ok(Self {
            dense: Linear::from_config(&dense_config),
            output: Some(Linear::from_config(&output_config)),
            tied_weights: None,
            precision,
        })
    }

    /// Creates an MLM head with tied output weights.
    pub fn tied(d_model: usize, embedding_weights: BoundedTensor, precision: Precision) -> Result<Self, LinearError> {
        let dense_config = LinearConfig::new(d_model, d_model)
            .with_bias(true)
            .with_precision(precision);

        Ok(Self {
            dense: Linear::from_config(&dense_config),
            output: None,
            tied_weights: Some(embedding_weights),
            precision,
        })
    }

    /// Forward pass: hidden states → vocabulary logits.
    pub fn forward(&self, hidden_states: &BoundedTensor) -> Result<BoundedTensor, BERTError> {
        // Dense + GELU
        let x = self.dense.forward(hidden_states)?;
        let x = activation::gelu(&x, self.precision);

        // Output projection
        if let Some(ref output) = self.output {
            Ok(output.forward(&x)?)
        } else if let Some(ref weights) = self.tied_weights {
            let weights_t = weights.transpose();
            Ok(ops::matmul(&x, &weights_t, self.precision)?)
        } else {
            Err(BERTError::InvalidConfig("No output weights".to_string()))
        }
    }

    /// Returns parameter count.
    pub fn param_count(&self) -> usize {
        let dense_shape = self.dense.weights().shape();
        let dense_params = dense_shape[0] * dense_shape[1] + dense_shape[0];

        let output_params = if let Some(ref output) = self.output {
            let shape = output.weights().shape();
            shape[0] * shape[1] + shape[0]
        } else {
            0
        };

        dense_params + output_params
    }
}

/// BERT-style encoder model.
#[derive(Debug)]
pub struct BERTModel {
    /// Configuration.
    config: BERTConfig,
    /// Token embedding.
    token_embedding: Embedding,
    /// Segment embedding.
    segment_embedding: Embedding,
    /// Positional encoding.
    positional_encoding: PositionalEncoding,
    /// Transformer encoder.
    encoder: TransformerStack,
    /// Pooler for [CLS] representation.
    pooler: BERTPooler,
    /// MLM head (optional).
    mlm_head: Option<MLMHead>,
    /// Memory profiler.
    profiler: Option<MemoryProfiler>,
    /// Cached dimensions.
    dimensions: ModelDimensions,
}

impl BERTModel {
    /// Creates a new BERT model.
    pub fn new(config: BERTConfig) -> Result<Self, BERTError> {
        config.validate()?;

        // Create embeddings
        let token_embedding = Embedding::zeros(config.vocab_size, config.d_model);
        let segment_embedding = Embedding::zeros(config.num_segments, config.d_model);
        let positional_encoding = PositionalEncoding::sinusoidal(config.max_seq_len, config.d_model);

        // Create encoder
        let transformer_config = config.to_transformer_config()?;
        let encoder = TransformerStack::new(config.n_layers, transformer_config)?;

        // Create pooler
        let pooler = BERTPooler::new(config.d_model, config.precision)?;

        let dimensions = ModelDimensions::new(
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.vocab_size,
            config.max_seq_len,
        );

        Ok(Self {
            config,
            token_embedding,
            segment_embedding,
            positional_encoding,
            encoder,
            pooler,
            mlm_head: None,
            profiler: None,
            dimensions,
        })
    }

    /// Creates a model for a specific size.
    pub fn for_size(size: ModelSize) -> Result<Self, BERTError> {
        let config = BERTConfig::for_size(size)?;
        Self::new(config)
    }

    /// Adds MLM head for masked language modeling.
    pub fn with_mlm_head(mut self) -> Result<Self, BERTError> {
        let mlm_head = MLMHead::tied(
            self.config.d_model,
            self.token_embedding.table().clone(),
            self.config.precision,
        )?;
        self.mlm_head = Some(mlm_head);
        Ok(self)
    }

    /// Returns the configuration.
    pub fn config(&self) -> &BERTConfig {
        &self.config
    }

    /// Creates input embeddings from token and segment IDs.
    pub fn embed(
        &self,
        token_ids: &[usize],
        segment_ids: &[usize],
    ) -> Result<BoundedTensor, BERTError> {
        if token_ids.len() != segment_ids.len() {
            return Err(BERTError::InvalidConfig(
                "token_ids and segment_ids must have same length".to_string(),
            ));
        }

        if token_ids.len() > self.config.max_seq_len {
            return Err(BERTError::SequenceTooLong {
                length: token_ids.len(),
                max_length: self.config.max_seq_len,
            });
        }

        // Validate segment IDs
        for &seg_id in segment_ids {
            if seg_id >= self.config.num_segments {
                return Err(BERTError::InvalidSegmentId {
                    id: seg_id,
                    max_segments: self.config.num_segments,
                });
            }
        }

        // Get embeddings
        let token_embeds = self.token_embedding.forward(token_ids)?;
        let segment_embeds = self.segment_embedding.forward(segment_ids)?;

        // Add embeddings
        let combined = token_embeds.add(&segment_embeds);

        // Add positional encoding
        let positioned = self.positional_encoding.forward(&combined)?;

        Ok(positioned)
    }

    /// Forward pass from embeddings.
    ///
    /// Input: (seq_len, d_model)
    /// Returns: (sequence_output, pooled_output)
    pub fn forward(
        &self,
        hidden_states: &BoundedTensor,
    ) -> Result<(BoundedTensor, BoundedTensor), BERTError> {
        // Encoder
        let sequence_output = self.encoder.forward(hidden_states)?;

        // Pooler
        let pooled_output = self.pooler.forward(&sequence_output)?;

        Ok((sequence_output, pooled_output))
    }

    /// Forward pass from token and segment IDs.
    pub fn forward_tokens(
        &self,
        token_ids: &[usize],
        segment_ids: &[usize],
    ) -> Result<(BoundedTensor, BoundedTensor), BERTError> {
        let embeddings = self.embed(token_ids, segment_ids)?;
        self.forward(&embeddings)
    }

    /// Forward pass for masked language modeling.
    pub fn forward_mlm(
        &self,
        hidden_states: &BoundedTensor,
    ) -> Result<BoundedTensor, BERTError> {
        let (sequence_output, _) = self.forward(hidden_states)?;

        if let Some(ref mlm_head) = self.mlm_head {
            mlm_head.forward(&sequence_output)
        } else {
            Err(BERTError::InvalidConfig("MLM head not initialized".to_string()))
        }
    }

    /// Returns the pooler.
    pub fn pooler(&self) -> &BERTPooler {
        &self.pooler
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

impl Model for BERTModel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn param_count(&self) -> usize {
        // Token embedding
        let token_embed = self.config.vocab_size * self.config.d_model;

        // Segment embedding
        let segment_embed = self.config.num_segments * self.config.d_model;

        // Transformer parameters
        let per_layer = 4 * self.config.d_model * self.config.d_model
            + 2 * self.config.d_model * self.config.d_ff
            + 4 * self.config.d_model;
        let encoder = per_layer * self.config.n_layers;

        // Pooler
        let pooler = self.pooler.param_count();

        // MLM head
        let mlm = self.mlm_head.as_ref().map(|h| h.param_count()).unwrap_or(0);

        token_embed + segment_embed + encoder + pooler + mlm
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
    fn test_bert_config_creation() {
        let config = BERTConfig::new(256, 8, 6, 8192).unwrap();
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.num_segments, 2);
    }

    #[test]
    fn test_bert_config_invalid() {
        let result = BERTConfig::new(255, 8, 6, 8192);
        assert!(result.is_err());
    }

    #[test]
    fn test_bert_model_creation() {
        let config = BERTConfig::new(64, 4, 2, 1000).unwrap();
        let model = BERTModel::new(config).unwrap();

        assert!(model.param_count() > 0);
    }

    #[test]
    fn test_bert_embedding() {
        let config = BERTConfig::new(32, 4, 2, 100).unwrap();
        let model = BERTModel::new(config).unwrap();

        let token_ids = vec![1, 5, 10, 20];
        let segment_ids = vec![0, 0, 1, 1];

        let embeddings = model.embed(&token_ids, &segment_ids).unwrap();
        assert_eq!(embeddings.shape(), &vec![4, 32]);
    }

    #[test]
    fn test_bert_forward() {
        let config = BERTConfig::new(32, 4, 2, 100).unwrap();
        let model = BERTModel::new(config).unwrap();

        let input = BoundedTensor::zeros(vec![4, 32]);
        let (seq_output, pooled_output) = model.forward(&input).unwrap();

        assert_eq!(seq_output.shape(), &vec![4, 32]);
        assert_eq!(pooled_output.shape(), &vec![32]);
    }

    #[test]
    fn test_bert_forward_tokens() {
        let config = BERTConfig::new(32, 4, 2, 100).unwrap();
        let model = BERTModel::new(config).unwrap();

        let token_ids = vec![1, 5, 10, 20];
        let segment_ids = vec![0, 0, 1, 1];

        let (seq_output, pooled_output) = model.forward_tokens(&token_ids, &segment_ids).unwrap();

        assert_eq!(seq_output.shape(), &vec![4, 32]);
        assert_eq!(pooled_output.shape(), &vec![32]);
    }

    #[test]
    fn test_bert_with_mlm() {
        let config = BERTConfig::new(32, 4, 2, 100).unwrap();
        let model = BERTModel::new(config).unwrap().with_mlm_head().unwrap();

        let input = BoundedTensor::zeros(vec![4, 32]);
        let logits = model.forward_mlm(&input).unwrap();

        assert_eq!(logits.shape(), &vec![4, 100]);
    }

    #[test]
    fn test_pooler() {
        let pooler = BERTPooler::new(32, Precision::F32).unwrap();

        let seq_output = BoundedTensor::zeros(vec![8, 32]);
        let pooled = pooler.forward(&seq_output).unwrap();

        assert_eq!(pooled.shape(), &vec![32]);
    }

    #[test]
    fn test_invalid_segment_id() {
        let config = BERTConfig::new(32, 4, 2, 100).unwrap();
        let model = BERTModel::new(config).unwrap();

        let token_ids = vec![1, 5];
        let segment_ids = vec![0, 5]; // Invalid segment ID

        let result = model.embed(&token_ids, &segment_ids);
        assert!(matches!(result, Err(BERTError::InvalidSegmentId { .. })));
    }
}
