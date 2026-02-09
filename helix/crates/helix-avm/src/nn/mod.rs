//! Neural network layers module.
//!
//! This module provides high-level neural network components with error bound tracking:
//! - Linear layers (y = Wx + b)
//! - Embedding lookups with positional encoding
//! - Multi-head attention
//! - MLP blocks (standard and gated)
//! - Full transformer blocks and stacks
//! - Layer normalization (LayerNorm and RMSNorm)
//! - Quantized layers with INT8/INT4 support
//! - Large model support with memory-efficient execution
//! - Validation and reference testing infrastructure

pub mod attention;
pub mod embedding;
pub mod large_model;
pub mod layer;
pub mod layer_norm;
pub mod linear;
pub mod mlp;
pub mod quantized;
pub mod reference;
pub mod transformer;
pub mod validation;

// Re-export commonly used types
pub use attention::{
    AttentionConfig, AttentionError, AttentionGradients, MultiHeadAttention,
    scaled_dot_product_attention,
    EfficientAttentionConfig, EfficientMultiHeadAttention, chunked_scaled_dot_product_attention,
    estimate_attention_memory,
};
pub use embedding::{Embedding, EmbeddingError, PositionalEncoding};
pub use layer::{Layer, Sequential};
pub use linear::{Linear, LinearConfig, LinearError};
pub use mlp::{ActivationType, GatedMLP, MLP, MLPConfig, MLPError, MLPGradients};
pub use transformer::{
    NormPosition, NormType, ProvableBlock, TransformerBlock, TransformerConfig, TransformerError,
    TransformerGradients, TransformerModel, TransformerStack,
};

// Re-export quantized layer types
pub use quantized::{
    AccuracyMetrics, GradientStats, QATConfig, QuantizedActivation, QuantizedActivationConfig,
    QuantizedActivationType, QuantizedGELU, QuantizedLayer, QuantizedLayerError,
    QuantizedLinear, QuantizedLinearConfig, QuantizedLinearGradients, QuantizedReLU,
    QuantizedSequential, QuantizedSiLU, Trainable, TrainingState,
};

// Re-export large model support
pub use large_model::{
    LargeModelConfig, LargeModelError, LargeModelExecutor,
    chunked_attention, compute_optimal_batch_size, estimate_forward_memory,
};

// Re-export layer normalization types
pub use layer_norm::{
    LayerNorm, LayerNormConfig, LayerNormError, RMSNorm,
    fused_layer_norm, fused_rms_norm,
};

// Re-export validation types
pub use validation::{
    LayerValidator, OutputValidator, StabilityChecker, StabilityReport,
    ValidationConfig, ValidationError, ValidationResult,
    validate_attention_forward, validate_linear_forward,
};

// Re-export reference testing types
pub use reference::{
    PyTorchReference, ReferenceTestVector, ReferenceTestSuite,
    generate_linear_reference, generate_attention_reference,
    generate_softmax_reference, generate_layer_norm_reference,
};
