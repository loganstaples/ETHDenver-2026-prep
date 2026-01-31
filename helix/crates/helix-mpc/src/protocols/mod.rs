//! MPC protocols for secure computation on secret-shared values.
//!
//! These protocols enable parties to compute functions on their shares
//! without revealing the underlying secrets.
//!
//! Linear operations (add, subtract, scale by public constant) can be done
//! locally by each party on their shares. Non-linear operations (multiply,
//! activations) require inter-party communication.

pub mod arithmetic;
pub mod matmul;
pub mod activation;
pub mod normalization;
pub mod reshare;

pub use arithmetic::SecureArithmetic;
pub use matmul::SecureMatmul;
pub use activation::SecureActivation;
pub use normalization::SecureNormalization;
pub use reshare::Resharing;
