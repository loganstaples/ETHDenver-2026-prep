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
pub use conv::{
    // Core convolution
    conv1d, conv2d, conv2d_with_config, conv2d_transpose, conv2d_transpose_with_config,
    // Pooling
    max_pool2d, avg_pool2d, global_avg_pool2d, global_max_pool2d, adaptive_avg_pool2d,
    // Specialized convolutions
    depthwise_conv2d, depthwise_separable_conv2d, group_conv2d,
    // Backward passes
    conv1d_backward_input, conv1d_backward_weight,
    conv2d_backward_input, conv2d_backward_weight,
    max_pool2d_backward, avg_pool2d_backward,
    // Optimized implementations
    conv2d_im2col, conv2d_tiled, conv2d_auto,
    im2col, col2im, reshape_kernel_for_im2col,
    // Config types
    Conv2dConfig, Pool2dConfig, MaxPool2dResult, ConvError,
    OptimizedConv2dConfig, TILE_SIZE, IM2COL_THRESHOLD,
};
pub use matmul::{dot, matmul, MatMulError};
pub use normalization::{layer_norm, rms_norm};
pub use reduction::{max, mean, min, sum};
pub use softmax::{log_softmax, softmax, SoftmaxError};
