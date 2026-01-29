//! Neural network layers module.
//!
//! This module provides high-level neural network components with error bound tracking:
//! - Linear layers (y = Wx + b)
//! - Embedding lookups with positional encoding
//! - Multi-head attention
//! - MLP blocks (standard and gated)
//! - Full transformer blocks and stacks

pub mod attention;
pub mod embedding;
pub mod layer;
pub mod linear;
pub mod mlp;
pub mod transformer;

// Re-export commonly used types
pub use attention::{AttentionConfig, AttentionError, MultiHeadAttention, scaled_dot_product_attention};
pub use embedding::{Embedding, EmbeddingError, PositionalEncoding};
pub use layer::{Layer, Sequential};
pub use linear::{Linear, LinearConfig, LinearError};
pub use mlp::{ActivationType, GatedMLP, MLP, MLPConfig, MLPError};
pub use transformer::{
    NormPosition, NormType, TransformerBlock, TransformerConfig, TransformerError,
    TransformerModel, TransformerStack,
};
