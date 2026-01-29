//! Operations module - ML operations with error tracking.

pub mod activation;
pub mod basic;
pub mod conv;
pub mod matmul;
pub mod normalization;
pub mod reduction;
pub mod softmax;

pub use activation::{gelu, leaky_relu, relu, sigmoid, tanh};
pub use basic::{add, mul, neg, scale, sub, BasicOpError};
pub use matmul::{dot, matmul, MatMulError};
pub use normalization::{layer_norm, rms_norm};
pub use reduction::{max, mean, min, sum};
pub use softmax::{log_softmax, softmax, SoftmaxError};
