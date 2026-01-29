//! Error Bound Algebra for HELIX.
//!
//! Provides rigorous error propagation analysis for all neural network
//! operations, enabling provable bounds on computation accuracy.

pub mod attention;
pub mod clipping;
pub mod residual;
pub mod rms_norm;
pub mod softmax;
pub mod stochastic;

// Re-export key types
pub use attention::{AttentionBounds, MultiHeadAttentionError};
pub use clipping::{ClippingBounds, GradientClipError};
pub use residual::{ResidualBounds, SkipConnectionError};
pub use rms_norm::{LayerNormBounds, RMSNormBounds};
pub use softmax::{SoftmaxBounds, SoftmaxError};
pub use stochastic::{RoundingBounds, StochasticRoundingError};
