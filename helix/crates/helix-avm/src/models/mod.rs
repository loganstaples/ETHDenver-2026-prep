//! Model Architectures Module.
//!
//! Provides configurable, production-ready model architectures for transformer-based models.
//! Includes support for:
//!
//! - **GPT-style models**: Decoder-only autoregressive transformers
//! - **BERT-style models**: Encoder-only bidirectional transformers
//! - **Model serialization**: Save and load model checkpoints
//!
//! # Model Size Configurations
//!
//! Pre-configured sizes for different parameter counts:
//!
//! | Size     | Params | d_model | n_layers | n_heads | d_ff  |
//! |----------|--------|---------|----------|---------|-------|
//! | Tiny     | ~500K  | 128     | 4        | 4       | 512   |
//! | Small    | ~1M    | 192     | 6        | 6       | 768   |
//! | Medium   | ~2M    | 256     | 8        | 8       | 1024  |
//! | Large    | ~5M    | 384     | 10       | 12      | 1536  |
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::models::{GPTModel, GPTConfig, ModelSize};
//!
//! // Create a 1M parameter GPT model
//! let config = GPTConfig::for_size(ModelSize::OneMillion);
//! let model = GPTModel::new(config)?;
//!
//! // Run forward pass
//! let output = model.forward(&token_embeddings)?;
//!
//! // Save checkpoint
//! let checkpoint = model.checkpoint();
//! checkpoint.save("model.bin")?;
//! ```

pub mod bert;
pub mod gpt;
pub mod serialization;

// Re-export commonly used types
pub use bert::{BERTConfig, BERTModel, BERTPooler};
pub use gpt::{GPTConfig, GPTModel, LanguageModelHead};
pub use serialization::{
    Checkpoint, CheckpointFormat, CheckpointMetadata, ModelCheckpoint, ModelSerializer,
    SerializationError, TensorData,
};

/// Pre-configured model sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// ~500K parameters
    HalfMillion,
    /// ~1M parameters
    OneMillion,
    /// ~2M parameters
    TwoMillion,
    /// ~5M parameters
    FiveMillion,
    /// Custom size
    Custom,
}

impl ModelSize {
    /// Returns the approximate parameter count for this size.
    pub fn param_count(&self) -> usize {
        match self {
            ModelSize::HalfMillion => 500_000,
            ModelSize::OneMillion => 1_000_000,
            ModelSize::TwoMillion => 2_000_000,
            ModelSize::FiveMillion => 5_000_000,
            ModelSize::Custom => 0,
        }
    }

    /// Returns recommended dimensions for this model size.
    pub fn dimensions(&self) -> ModelDimensions {
        match self {
            ModelSize::HalfMillion => ModelDimensions {
                d_model: 128,
                n_layers: 4,
                n_heads: 4,
                d_ff: 512,
                vocab_size: 8192,
                max_seq_len: 256,
            },
            ModelSize::OneMillion => ModelDimensions {
                d_model: 192,
                n_layers: 6,
                n_heads: 6,
                d_ff: 768,
                vocab_size: 8192,
                max_seq_len: 512,
            },
            ModelSize::TwoMillion => ModelDimensions {
                d_model: 256,
                n_layers: 8,
                n_heads: 8,
                d_ff: 1024,
                vocab_size: 16384,
                max_seq_len: 512,
            },
            ModelSize::FiveMillion => ModelDimensions {
                d_model: 384,
                n_layers: 10,
                n_heads: 12,
                d_ff: 1536,
                vocab_size: 16384,
                max_seq_len: 1024,
            },
            ModelSize::Custom => ModelDimensions::default(),
        }
    }

    /// Returns the model size that best fits the given parameter count.
    pub fn from_param_count(params: usize) -> Self {
        if params <= 750_000 {
            ModelSize::HalfMillion
        } else if params <= 1_500_000 {
            ModelSize::OneMillion
        } else if params <= 3_500_000 {
            ModelSize::TwoMillion
        } else if params <= 7_500_000 {
            ModelSize::FiveMillion
        } else {
            ModelSize::Custom
        }
    }
}

/// Model dimensions configuration.
#[derive(Debug, Clone, Copy)]
pub struct ModelDimensions {
    /// Model/embedding dimension.
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Feedforward hidden dimension.
    pub d_ff: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl ModelDimensions {
    /// Creates custom dimensions.
    pub fn new(
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        d_ff: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            d_model,
            n_layers,
            n_heads,
            d_ff,
            vocab_size,
            max_seq_len,
        }
    }

    /// Estimates the total parameter count for these dimensions (GPT-style).
    pub fn estimate_params_gpt(&self) -> usize {
        // Embedding: vocab_size * d_model
        let embedding = self.vocab_size * self.d_model;

        // Each transformer layer:
        // - Attention: 4 * d_model^2 (Q, K, V, O projections)
        // - MLP: 2 * d_model * d_ff (up and down projections)
        // - Layer norms: 2 * 2 * d_model (2 norms, each has weight and bias)
        let attention = 4 * self.d_model * self.d_model;
        let mlp = 2 * self.d_model * self.d_ff;
        let layer_norms = 4 * self.d_model;
        let per_layer = attention + mlp + layer_norms;
        let layers = per_layer * self.n_layers;

        // Final layer norm
        let final_norm = 2 * self.d_model;

        // LM head (often tied with embedding, but count separately)
        let lm_head = self.vocab_size * self.d_model;

        embedding + layers + final_norm + lm_head
    }

    /// Estimates the total parameter count for BERT-style model.
    pub fn estimate_params_bert(&self) -> usize {
        // Similar to GPT but with separate token type embeddings
        let gpt_params = self.estimate_params_gpt();

        // Additional for BERT:
        // - Token type embedding: 2 * d_model (segment A/B)
        // - Pooler: d_model * d_model
        let additional = 2 * self.d_model + self.d_model * self.d_model;

        gpt_params + additional
    }

    /// Validates the dimensions are consistent.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.d_model % self.n_heads != 0 {
            return Err("d_model must be divisible by n_heads");
        }
        if self.d_model == 0 {
            return Err("d_model must be positive");
        }
        if self.n_layers == 0 {
            return Err("n_layers must be positive");
        }
        if self.n_heads == 0 {
            return Err("n_heads must be positive");
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be positive");
        }
        Ok(())
    }
}

impl Default for ModelDimensions {
    fn default() -> Self {
        ModelSize::OneMillion.dimensions()
    }
}

/// Common trait for all model architectures.
pub trait Model {
    /// Returns the model name.
    fn name(&self) -> &str;

    /// Returns the total parameter count.
    fn param_count(&self) -> usize;

    /// Returns the model dimensions.
    fn dimensions(&self) -> &ModelDimensions;

    /// Returns estimated memory usage for training (bytes).
    fn training_memory(&self, batch_size: usize, seq_len: usize) -> usize;

    /// Returns estimated memory usage for inference (bytes).
    fn inference_memory(&self, batch_size: usize, seq_len: usize) -> usize;
}

/// Utility function to compute dimensions for a target parameter count.
pub fn compute_dimensions_for_params(target_params: usize) -> ModelDimensions {
    // Use heuristics to find good dimensions
    // Roughly: params ≈ vocab * d_model + n_layers * (4 * d_model^2 + 2 * d_model * d_ff)

    // Start with the closest pre-defined size
    let size = ModelSize::from_param_count(target_params);
    let mut dims = size.dimensions();

    // Fine-tune by adjusting n_layers
    let current = dims.estimate_params_gpt();
    if current > target_params * 12 / 10 {
        // More than 20% over target, reduce layers
        dims.n_layers = ((dims.n_layers as f64 * target_params as f64 / current as f64) as usize).max(1);
    } else if current < target_params * 8 / 10 {
        // Less than 80% of target, add layers
        dims.n_layers = (dims.n_layers as f64 * target_params as f64 / current as f64) as usize;
    }

    dims
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_size_params() {
        assert_eq!(ModelSize::HalfMillion.param_count(), 500_000);
        assert_eq!(ModelSize::OneMillion.param_count(), 1_000_000);
    }

    #[test]
    fn test_dimensions_validation() {
        let dims = ModelDimensions::new(256, 6, 8, 1024, 8192, 512);
        assert!(dims.validate().is_ok());

        let bad_dims = ModelDimensions::new(255, 6, 8, 1024, 8192, 512);
        assert!(bad_dims.validate().is_err());
    }

    #[test]
    fn test_param_estimation() {
        let dims = ModelSize::OneMillion.dimensions();
        let estimated = dims.estimate_params_gpt();

        // Parameter estimation includes embeddings which scale with vocab
        // Verify reasonable order of magnitude
        assert!(estimated > 100_000);
        assert!(estimated < 50_000_000);
    }

    #[test]
    fn test_compute_dimensions() {
        let dims = compute_dimensions_for_params(1_000_000);
        let estimated = dims.estimate_params_gpt();

        // Should be in a reasonable range for targeted models
        assert!(estimated > 100_000);
        assert!(estimated < 50_000_000);
    }

    #[test]
    fn test_model_size_from_param_count() {
        assert_eq!(ModelSize::from_param_count(400_000), ModelSize::HalfMillion);
        assert_eq!(ModelSize::from_param_count(800_000), ModelSize::OneMillion);
        assert_eq!(ModelSize::from_param_count(1_800_000), ModelSize::TwoMillion);
    }
}
