//! MPC protocols for secure computation on secret-shared values.
//!
//! These protocols enable parties to compute functions on their shares
//! without revealing the underlying secrets.
//!
//! Linear operations (add, subtract, scale by public constant) can be done
//! locally by each party on their shares. Non-linear operations (multiply,
//! activations) require inter-party communication.
//!
//! # Available Protocols
//!
//! - `arithmetic`: Basic operations (add, multiply via Beaver triples)
//! - `matmul`: Secure matrix multiplication
//! - `activation`: Activation functions (ReLU, GELU, etc.)
//! - `comparison`: Secure comparison and sign computation
//! - `normalization`: Layer normalization and batch normalization
//! - `reshare`: Periodic share refreshing

pub mod activation;
pub mod arithmetic;
pub mod comparison;
pub mod matmul;
pub mod normalization;
pub mod reshare;

pub use activation::SecureActivation;
pub use arithmetic::SecureArithmetic;
pub use comparison::{
    BitDecomposition, ComparisonConfig, GarbledComparison, SecureComparison, SecureReLUWithGradient,
};
pub use matmul::SecureMatmul;
pub use normalization::SecureNormalization;
pub use reshare::Resharing;
