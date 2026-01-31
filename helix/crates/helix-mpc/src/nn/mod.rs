//! Secure neural network layers that operate on secret-shared weights.
//!
//! These layers mirror the structure of the helix-avm neural network layers
//! but perform computation on additive secret shares, ensuring that no single
//! party ever sees the full model weights.
//!
//! Each layer takes secret-shared inputs and weights and produces
//! secret-shared outputs using the MPC protocols.

pub mod linear;
pub mod attention;
pub mod transformer;
pub mod embedding;

pub use linear::SecureLinear;
pub use attention::SecureAttention;
pub use transformer::SecureTransformerBlock;
pub use embedding::SecureEmbedding;
