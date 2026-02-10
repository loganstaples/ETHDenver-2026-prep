//! Convolution operations with error bound tracking.
//!
//! This module implements convolutional operations for neural networks with rigorous
//! error bound propagation. All operations track how errors accumulate through the
//! computation, enabling the approximate ZK proof system to verify bounded correctness.
//!
//! # Supported Operations
//!
//! - `conv1d` - 1D convolution (for sequence models)
//! - `conv2d` - 2D convolution with stride, padding, dilation (for CNNs)
//! - `conv2d_transpose` - Transposed/deconvolution (for backward pass and upsampling)
//! - `max_pool2d` - Max pooling with index tracking
//! - `avg_pool2d` - Average pooling
//! - `depthwise_conv2d` - Depthwise convolution (for MobileNet)
//! - `depthwise_separable_conv2d` - Full depthwise separable conv
//! - `group_conv2d` - Group convolution (for ResNeXt)
//!
//! # Error Propagation
//!
//! Convolution is fundamentally a series of multiply-accumulate operations:
//!
//! ```text
//! out[n,c,h,w] = sum_{c_in,kh,kw} input[n,c_in,h+kh,w+kw] * kernel[c,c_in,kh,kw]
//! ```
//!
//! For each multiply: error = |a| * err_b + |b| * err_a + err_a * err_b + |a*b| * ε
//! For accumulation: errors sum linearly
//!
//! The total error for one output element is bounded by the sum of all individual
//! multiply-accumulate errors in the receptive field.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};
use thiserror::Error;

/// Errors from convolution operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ConvError {
    #[error("Invalid kernel dimensions: expected {expected}D, got {got}D")]
    InvalidKernel { expected: usize, got: usize },

    #[error("Invalid input dimensions: expected {expected}D, got {got}D")]
    InvalidDimensions { expected: usize, got: usize },

    #[error("Channel mismatch: input has {input} channels, kernel expects {kernel}")]
    ChannelMismatch { input: usize, kernel: usize },

    #[error("Invalid stride: must be > 0, got ({0}, {1})")]
    InvalidStride(usize, usize),

    #[error("Invalid dilation: must be > 0, got ({0}, {1})")]
    InvalidDilation(usize, usize),

    #[error("Invalid groups: {groups} must divide both input channels ({in_channels}) and output channels ({out_channels})")]
    InvalidGroups {
        groups: usize,
        in_channels: usize,
        out_channels: usize,
    },

    #[error("Invalid pool size: must be > 0")]
    InvalidPoolSize,

    #[error("Output size would be zero or negative")]
    InvalidOutputSize,
}

/// Configuration for 2D convolution.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

impl Conv2dConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

/// Configuration for pooling operations.
#[derive(Debug, Clone, Copy)]
pub struct Pool2dConfig {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Pool2dConfig {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size, // Default stride = kernel size (non-overlapping)
            padding: (0, 0),
        }
    }

    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }
}

// ============================================================================
// MEMORY LAYOUT OPTIMIZATION
// ============================================================================

/// Block size for cache-friendly tiled convolution.
/// Tuned for typical L1 cache sizes (32-64KB).
pub const TILE_SIZE: usize = 8;

/// Threshold above which to use im2col-based convolution.
/// For smaller tensors, direct convolution is faster due to lower overhead.
pub const IM2COL_THRESHOLD: usize = 64;

/// Configuration for memory-optimized convolution.
#[derive(Debug, Clone, Copy)]
pub struct OptimizedConv2dConfig {
    /// Base conv2d configuration
    pub conv_config: Conv2dConfig,
    /// Whether to use im2col transformation
    pub use_im2col: bool,
    /// Tile size for blocked convolution (0 = auto)
    pub tile_size: usize,
    /// Whether to precompute kernel layout
    pub precompute_kernel: bool,
}

impl Default for OptimizedConv2dConfig {
    fn default() -> Self {
        Self {
            conv_config: Conv2dConfig::default(),
            use_im2col: true,
            tile_size: 0, // Auto
            precompute_kernel: true,
        }
    }
}

impl OptimizedConv2dConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn conv_config(mut self, config: Conv2dConfig) -> Self {
        self.conv_config = config;
        self
    }

    pub fn use_im2col(mut self, use_im2col: bool) -> Self {
        self.use_im2col = use_im2col;
        self
    }

    pub fn tile_size(mut self, tile_size: usize) -> Self {
        self.tile_size = tile_size;
        self
    }
}

/// Im2col (image to column) transformation for cache-friendly convolution.
///
/// Converts input patches into columns of a matrix, enabling convolution
/// to be computed as a single matrix multiplication. This improves cache
/// utilization significantly for larger tensors.
///
/// # Memory layout
/// Output shape: (batch, out_h * out_w, in_channels * kernel_h * kernel_w)
///
/// # Performance
/// - Better cache utilization for large kernels
/// - Enables BLAS-accelerated matrix multiplication
/// - Higher memory overhead (O(out_h * out_w * kernel_size))
pub fn im2col(
    input: &BoundedTensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> BoundedTensor {
    let has_batch = input.ndim() == 4;

    let (batch_size, in_channels, in_h, in_w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let out_h = conv_output_size(in_h, kernel_size.0, stride.0, padding.0, dilation.0);
    let out_w = conv_output_size(in_w, kernel_size.1, stride.1, padding.1, dilation.1);

    let col_height = out_h * out_w;
    let col_width = in_channels * kernel_size.0 * kernel_size.1;

    let mut col_data = Vec::with_capacity(batch_size * col_height * col_width);

    for n in 0..batch_size {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for ic in 0..in_channels {
                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh * dilation.0;
                            let iw = ow * stride.1 + kw * dilation.1;

                            let val = if ih < padding.0
                                || iw < padding.1
                                || ih >= in_h + padding.0
                                || iw >= in_w + padding.1
                            {
                                BoundedValue::exact(0.0)
                            } else {
                                let real_h = ih - padding.0;
                                let real_w = iw - padding.1;
                                if has_batch {
                                    *input.get(&[n, ic, real_h, real_w])
                                        .unwrap_or(&BoundedValue::exact(0.0))
                                } else {
                                    *input.get(&[ic, real_h, real_w])
                                        .unwrap_or(&BoundedValue::exact(0.0))
                                }
                            };

                            col_data.push(val);
                        }
                    }
                }
            }
        }
    }

    BoundedTensor::new(col_data, vec![batch_size, col_height, col_width])
}

/// Col2im (column to image) transformation for backward pass.
///
/// Inverse of im2col: scatters column data back to image format.
/// Used in conv2d_transpose and gradient computation.
pub fn col2im(
    col: &BoundedTensor,
    output_shape: (usize, usize, usize, usize), // (batch, channels, height, width)
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> BoundedTensor {
    let (batch_size, channels, out_h, out_w) = output_shape;
    let mut result_data = vec![BoundedValue::exact(0.0); batch_size * channels * out_h * out_w];

    let col_h = col.shape()[1]; // out_h * out_w of original conv
    let col_w = col.shape()[2]; // channels * kH * kW

    let conv_out_h = (out_h + 2 * padding.0 - (kernel_size.0 - 1) * dilation.0 - 1) / stride.0 + 1;
    let conv_out_w = (out_w + 2 * padding.1 - (kernel_size.1 - 1) * dilation.1 - 1) / stride.1 + 1;

    for n in 0..batch_size {
        let mut col_idx = 0;
        for oh in 0..conv_out_h {
            for ow in 0..conv_out_w {
                for c in 0..channels {
                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let h = oh * stride.0 + kh * dilation.0;
                            let w = ow * stride.1 + kw * dilation.1;

                            if h >= padding.0 && h < out_h + padding.0
                                && w >= padding.1 && w < out_w + padding.1 {
                                let real_h = h - padding.0;
                                let real_w = w - padding.1;

                                let out_idx = n * channels * out_h * out_w
                                    + c * out_h * out_w
                                    + real_h * out_w
                                    + real_w;

                                let col_val = col.data()[n * col_h * col_w + col_idx];
                                let current = result_data[out_idx];
                                let new_val = current.value() + col_val.value();
                                let new_err = current.absolute_error() + col_val.absolute_error();
                                result_data[out_idx] = BoundedValue::new(new_val, ErrorMargin::absolute(new_err));
                            }

                            col_idx += 1;
                        }
                    }
                }
                if col_idx >= col_w {
                    col_idx = 0;
                }
            }
        }
    }

    BoundedTensor::new(result_data, vec![batch_size, channels, out_h, out_w])
}

/// Reshapes kernel for im2col-based convolution.
///
/// Transforms kernel from (out_ch, in_ch, kH, kW) to (out_ch, in_ch * kH * kW)
/// for matrix multiplication with im2col output.
pub fn reshape_kernel_for_im2col(kernel: &BoundedTensor) -> BoundedTensor {
    let (out_channels, in_channels, kernel_h, kernel_w) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    let kernel_cols = in_channels * kernel_h * kernel_w;
    kernel.reshape(vec![out_channels, kernel_cols])
}

/// Optimized 2D convolution using im2col transformation.
///
/// For large tensors, this is significantly faster than direct convolution
/// due to better cache utilization and potential for BLAS acceleration.
///
/// # When to use
/// - Input spatial size > 32x32
/// - Kernel size > 1x1
/// - batch_size * out_channels * out_h * out_w > IM2COL_THRESHOLD^2
pub fn conv2d_im2col(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    config: Conv2dConfig,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions { expected: 4, got: input.ndim() });
    }

    let (batch_size, in_channels, in_h, in_w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let (out_channels, _, kernel_h, kernel_w) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    let Conv2dConfig { stride, padding, dilation, groups: _ } = config;

    let out_h = conv_output_size(in_h, kernel_h, stride.0, padding.0, dilation.0);
    let out_w = conv_output_size(in_w, kernel_w, stride.1, padding.1, dilation.1);

    // Step 1: im2col transformation
    let col = im2col(input, (kernel_h, kernel_w), stride, padding, dilation);

    // Step 2: Reshape kernel for matrix multiplication
    let kernel_mat = reshape_kernel_for_im2col(kernel);

    // Step 3: Matrix multiplication (col @ kernel^T)
    // col: (batch, out_h*out_w, in_ch*kH*kW)
    // kernel_mat: (out_ch, in_ch*kH*kW)
    // result: (batch, out_h*out_w, out_ch)
    let precision_error = precision.max_relative_error();
    let k = in_channels * kernel_h * kernel_w;

    let mut result_data = Vec::with_capacity(batch_size * out_h * out_w * out_channels);

    for n in 0..batch_size {
        for spatial in 0..(out_h * out_w) {
            for oc in 0..out_channels {
                let mut sum = 0.0;
                let mut accumulated_error = 0.0;

                for i in 0..k {
                    let col_idx = n * (out_h * out_w) * k + spatial * k + i;
                    let kernel_idx = oc * k + i;

                    let col_val = col.data()[col_idx];
                    let kernel_val = kernel_mat.data()[kernel_idx];

                    let prod = col_val.value() * kernel_val.value();
                    let prod_error = propagate_mac_error(
                        col_val.value(),
                        col_val.absolute_error(),
                        kernel_val.value(),
                        kernel_val.absolute_error(),
                        precision_error,
                    );

                    sum += prod;
                    accumulated_error += prod_error;
                }

                let final_error = accumulated_error + sum.abs() * precision_error;
                result_data.push(BoundedValue::new(sum, ErrorMargin::absolute(final_error)));
            }
        }
    }

    // Reshape from (batch, out_h*out_w, out_ch) to (batch, out_ch, out_h, out_w)
    let mut final_data = vec![BoundedValue::exact(0.0); result_data.len()];
    for n in 0..batch_size {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for oc in 0..out_channels {
                    let src_idx = n * out_h * out_w * out_channels
                        + (oh * out_w + ow) * out_channels
                        + oc;
                    let dst_idx = n * out_channels * out_h * out_w
                        + oc * out_h * out_w
                        + oh * out_w
                        + ow;
                    final_data[dst_idx] = result_data[src_idx];
                }
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, out_channels, out_h, out_w]
    } else {
        vec![out_channels, out_h, out_w]
    };

    Ok(BoundedTensor::new(final_data, shape))
}

/// Blocked/tiled convolution for improved cache utilization.
///
/// Processes output in tiles to keep working data in L1/L2 cache.
/// Best for very large feature maps where im2col memory overhead is prohibitive.
pub fn conv2d_tiled(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    config: Conv2dConfig,
    tile_size: usize,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    // For simplicity, fall back to regular conv2d for now
    // A full implementation would process output in tile_size x tile_size blocks
    let tile_size = if tile_size == 0 { TILE_SIZE } else { tile_size };

    // Validate inputs same as conv2d
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions { expected: 4, got: input.ndim() });
    }

    let (batch_size, in_channels, in_h, in_w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let (out_channels, kernel_in_channels, kernel_h, kernel_w) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    let Conv2dConfig { stride, padding, dilation, groups } = config;

    if in_channels / groups != kernel_in_channels {
        return Err(ConvError::ChannelMismatch {
            input: in_channels / groups,
            kernel: kernel_in_channels,
        });
    }

    let out_h = conv_output_size(in_h, kernel_h, stride.0, padding.0, dilation.0);
    let out_w = conv_output_size(in_w, kernel_w, stride.1, padding.1, dilation.1);

    let precision_error = precision.max_relative_error();
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let mut result_data = vec![BoundedValue::exact(0.0); batch_size * out_channels * out_h * out_w];

    // Process in tiles for better cache utilization
    for n in 0..batch_size {
        for oc_tile in (0..out_channels).step_by(tile_size) {
            let oc_end = (oc_tile + tile_size).min(out_channels);

            for oh_tile in (0..out_h).step_by(tile_size) {
                let oh_end = (oh_tile + tile_size).min(out_h);

                for ow_tile in (0..out_w).step_by(tile_size) {
                    let ow_end = (ow_tile + tile_size).min(out_w);

                    // Process tile
                    for oc in oc_tile..oc_end {
                        let group = oc / out_channels_per_group;
                        let ic_start = group * in_channels_per_group;

                        for oh in oh_tile..oh_end {
                            for ow in ow_tile..ow_end {
                                let mut sum = 0.0;
                                let mut accumulated_error = 0.0;

                                for ic_offset in 0..in_channels_per_group {
                                    let ic = ic_start + ic_offset;

                                    for kh in 0..kernel_h {
                                        for kw in 0..kernel_w {
                                            let ih = oh * stride.0 + kh * dilation.0;
                                            let iw = ow * stride.1 + kw * dilation.1;

                                            let input_val = if ih < padding.0
                                                || iw < padding.1
                                                || ih >= in_h + padding.0
                                                || iw >= in_w + padding.1
                                            {
                                                BoundedValue::exact(0.0)
                                            } else {
                                                let real_h = ih - padding.0;
                                                let real_w = iw - padding.1;
                                                if has_batch {
                                                    *input.get(&[n, ic, real_h, real_w])
                                                        .unwrap_or(&BoundedValue::exact(0.0))
                                                } else {
                                                    *input.get(&[ic, real_h, real_w])
                                                        .unwrap_or(&BoundedValue::exact(0.0))
                                                }
                                            };

                                            let kernel_val = *kernel.get(&[oc, ic_offset, kh, kw]).unwrap();

                                            let prod = input_val.value() * kernel_val.value();
                                            let prod_error = propagate_mac_error(
                                                input_val.value(),
                                                input_val.absolute_error(),
                                                kernel_val.value(),
                                                kernel_val.absolute_error(),
                                                precision_error,
                                            );

                                            sum += prod;
                                            accumulated_error += prod_error;
                                        }
                                    }
                                }

                                let final_error = accumulated_error + sum.abs() * precision_error;
                                let out_idx = if has_batch {
                                    n * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow
                                } else {
                                    oc * out_h * out_w + oh * out_w + ow
                                };
                                result_data[out_idx] = BoundedValue::new(sum, ErrorMargin::absolute(final_error));
                            }
                        }
                    }
                }
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, out_channels, out_h, out_w]
    } else {
        vec![out_channels, out_h, out_w]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

/// Automatically selects the best convolution algorithm based on tensor sizes.
///
/// # Selection criteria
/// - Small tensors: Direct convolution (lower overhead)
/// - Medium tensors: Tiled convolution (cache-friendly)
/// - Large tensors: Im2col (best throughput, if memory allows)
pub fn conv2d_auto(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    config: Conv2dConfig,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;

    let (_, _in_channels, in_h, in_w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let spatial_size = in_h * in_w;
    let kernel_size = kernel.shape()[2] * kernel.shape()[3];

    // Heuristic for algorithm selection
    if spatial_size < IM2COL_THRESHOLD * IM2COL_THRESHOLD / 16 {
        // Small tensors: direct convolution
        conv2d_with_config(input, kernel, config, precision)
    } else if kernel_size > 9 && spatial_size > IM2COL_THRESHOLD * IM2COL_THRESHOLD {
        // Large kernels and spatial: im2col for best throughput
        conv2d_im2col(input, kernel, config, precision)
    } else {
        // Medium size: tiled for cache efficiency
        conv2d_tiled(input, kernel, config, TILE_SIZE, precision)
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculates output size for a convolution dimension.
#[inline]
fn conv_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_kernel = (kernel_size - 1) * dilation + 1;
    (input_size + 2 * padding - effective_kernel) / stride + 1
}

/// Gets a value from tensor with padding (returns 0 if out of bounds).
#[inline]
#[allow(dead_code)]
fn get_padded_value(
    tensor: &BoundedTensor,
    indices: &[usize],
    padding: (usize, usize),
    spatial_dims: (usize, usize), // (H, W) dimensions in the tensor
) -> BoundedValue<f64> {
    // indices are [N, C, H, W] or [C, H, W]
    let h_idx = indices.len() - 2;
    let w_idx = indices.len() - 1;

    let h = indices[h_idx];
    let w = indices[w_idx];

    // Check if within padded region
    if h < padding.0 || w < padding.1 {
        return BoundedValue::exact(0.0);
    }

    let real_h = h - padding.0;
    let real_w = w - padding.1;

    if real_h >= spatial_dims.0 || real_w >= spatial_dims.1 {
        return BoundedValue::exact(0.0);
    }

    // Build actual indices
    let mut actual_indices = indices.to_vec();
    actual_indices[h_idx] = real_h;
    actual_indices[w_idx] = real_w;

    *tensor.get(&actual_indices).unwrap_or(&BoundedValue::exact(0.0))
}

/// Propagates error through a multiply-accumulate operation.
#[inline]
fn propagate_mac_error(
    a_val: f64,
    a_err: f64,
    b_val: f64,
    b_err: f64,
    precision_error: f64,
) -> f64 {
    let prod = a_val * b_val;
    a_val.abs() * b_err + b_val.abs() * a_err + a_err * b_err + prod.abs() * precision_error
}

// ============================================================================
// 1D CONVOLUTION
// ============================================================================

/// 1D convolution with error tracking.
///
/// # Input format
/// - `input`: (batch, in_channels, length) or (in_channels, length)
/// - `kernel`: (out_channels, in_channels, kernel_size)
///
/// # Output format
/// - (batch, out_channels, out_length) or (out_channels, out_length)
///
/// # Error propagation
/// Each output element accumulates errors from `in_channels * kernel_size` MAC operations.
pub fn conv1d(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    stride: usize,
    padding: usize,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    // Validate dimensions
    let has_batch = input.ndim() == 3;
    if input.ndim() != 2 && input.ndim() != 3 {
        return Err(ConvError::InvalidDimensions {
            expected: 3,
            got: input.ndim(),
        });
    }
    if kernel.ndim() != 3 {
        return Err(ConvError::InvalidKernel {
            expected: 3,
            got: kernel.ndim(),
        });
    }

    let (batch_size, in_channels, in_length) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2])
    } else {
        (1, input.shape()[0], input.shape()[1])
    };

    let (out_channels, kernel_in_channels, kernel_size) =
        (kernel.shape()[0], kernel.shape()[1], kernel.shape()[2]);

    if in_channels != kernel_in_channels {
        return Err(ConvError::ChannelMismatch {
            input: in_channels,
            kernel: kernel_in_channels,
        });
    }

    if stride == 0 {
        return Err(ConvError::InvalidStride(stride, 1));
    }

    // Calculate output size
    let out_length = conv_output_size(in_length, kernel_size, stride, padding, 1);
    if out_length == 0 {
        return Err(ConvError::InvalidOutputSize);
    }

    let precision_error = precision.max_relative_error();
    let result_size = batch_size * out_channels * out_length;
    let mut result_data = Vec::with_capacity(result_size);

    for n in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..out_length {
                let mut sum = 0.0;
                let mut accumulated_error = 0.0;

                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        let in_pos = ol * stride + k;

                        // Get input value with padding
                        let input_val = if in_pos < padding || in_pos >= in_length + padding {
                            BoundedValue::exact(0.0)
                        } else {
                            let real_pos = in_pos - padding;
                            if has_batch {
                                *input.get(&[n, ic, real_pos]).unwrap_or(&BoundedValue::exact(0.0))
                            } else {
                                *input.get(&[ic, real_pos]).unwrap_or(&BoundedValue::exact(0.0))
                            }
                        };

                        let kernel_val = *kernel.get(&[oc, ic, k]).unwrap();

                        // Compute product with error propagation
                        let prod = input_val.value() * kernel_val.value();
                        let prod_error = propagate_mac_error(
                            input_val.value(),
                            input_val.absolute_error(),
                            kernel_val.value(),
                            kernel_val.absolute_error(),
                            precision_error,
                        );

                        sum += prod;
                        accumulated_error += prod_error;
                    }
                }

                // Add final rounding error
                let final_error = accumulated_error + sum.abs() * precision_error;
                result_data.push(BoundedValue::new(sum, ErrorMargin::absolute(final_error)));
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, out_channels, out_length]
    } else {
        vec![out_channels, out_length]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

// ============================================================================
// 2D CONVOLUTION
// ============================================================================

/// 2D convolution with error tracking.
///
/// # Input format
/// - `input`: (batch, in_channels, height, width) or (in_channels, height, width)
/// - `kernel`: (out_channels, in_channels/groups, kernel_h, kernel_w)
///
/// # Output format
/// - (batch, out_channels, out_h, out_w) or (out_channels, out_h, out_w)
///
/// # Features
/// - Supports arbitrary stride, padding, dilation
/// - Supports group convolution (groups parameter)
/// - Full error bound propagation through all MAC operations
pub fn conv2d(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    conv2d_with_config(input, kernel, Conv2dConfig::default().stride(stride).padding(padding), precision)
}

/// 2D convolution with full configuration.
pub fn conv2d_with_config(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    config: Conv2dConfig,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    // Validate dimensions
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }
    if kernel.ndim() != 4 {
        return Err(ConvError::InvalidKernel {
            expected: 4,
            got: kernel.ndim(),
        });
    }

    let (batch_size, in_channels, in_h, in_w) = if has_batch {
        (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        )
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let (out_channels, kernel_in_channels, kernel_h, kernel_w) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    let Conv2dConfig {
        stride,
        padding,
        dilation,
        groups,
    } = config;

    // Validate parameters
    if stride.0 == 0 || stride.1 == 0 {
        return Err(ConvError::InvalidStride(stride.0, stride.1));
    }
    if dilation.0 == 0 || dilation.1 == 0 {
        return Err(ConvError::InvalidDilation(dilation.0, dilation.1));
    }
    if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(ConvError::InvalidGroups {
            groups,
            in_channels,
            out_channels,
        });
    }

    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    if kernel_in_channels != in_channels_per_group {
        return Err(ConvError::ChannelMismatch {
            input: in_channels_per_group,
            kernel: kernel_in_channels,
        });
    }

    // Calculate output size
    let out_h = conv_output_size(in_h, kernel_h, stride.0, padding.0, dilation.0);
    let out_w = conv_output_size(in_w, kernel_w, stride.1, padding.1, dilation.1);

    if out_h == 0 || out_w == 0 {
        return Err(ConvError::InvalidOutputSize);
    }

    let precision_error = precision.max_relative_error();
    let result_size = batch_size * out_channels * out_h * out_w;
    let mut result_data = Vec::with_capacity(result_size);

    for n in 0..batch_size {
        for oc in 0..out_channels {
            let group = oc / out_channels_per_group;
            let ic_start = group * in_channels_per_group;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;
                    let mut accumulated_error = 0.0;

                    for ic_offset in 0..in_channels_per_group {
                        let ic = ic_start + ic_offset;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                // Calculate input position with dilation
                                let ih = oh * stride.0 + kh * dilation.0;
                                let iw = ow * stride.1 + kw * dilation.1;

                                // Get input value with padding
                                let input_val = if ih < padding.0
                                    || iw < padding.1
                                    || ih >= in_h + padding.0
                                    || iw >= in_w + padding.1
                                {
                                    BoundedValue::exact(0.0)
                                } else {
                                    let real_h = ih - padding.0;
                                    let real_w = iw - padding.1;
                                    if has_batch {
                                        *input
                                            .get(&[n, ic, real_h, real_w])
                                            .unwrap_or(&BoundedValue::exact(0.0))
                                    } else {
                                        *input
                                            .get(&[ic, real_h, real_w])
                                            .unwrap_or(&BoundedValue::exact(0.0))
                                    }
                                };

                                let kernel_val = *kernel.get(&[oc, ic_offset, kh, kw]).unwrap();

                                // Compute product with error propagation
                                let prod = input_val.value() * kernel_val.value();
                                let prod_error = propagate_mac_error(
                                    input_val.value(),
                                    input_val.absolute_error(),
                                    kernel_val.value(),
                                    kernel_val.absolute_error(),
                                    precision_error,
                                );

                                sum += prod;
                                accumulated_error += prod_error;
                            }
                        }
                    }

                    // Add final rounding error
                    let final_error = accumulated_error + sum.abs() * precision_error;
                    result_data.push(BoundedValue::new(sum, ErrorMargin::absolute(final_error)));
                }
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, out_channels, out_h, out_w]
    } else {
        vec![out_channels, out_h, out_w]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

// ============================================================================
// TRANSPOSED 2D CONVOLUTION
// ============================================================================

/// Transposed 2D convolution (deconvolution) with error tracking.
///
/// Used for:
/// 1. Backward pass of conv2d (computing input gradients)
/// 2. Upsampling in decoder networks
///
/// # Input format
/// - `input`: (batch, in_channels, height, width) or (in_channels, height, width)
/// - `kernel`: (in_channels, out_channels/groups, kernel_h, kernel_w)
///
/// # Output format
/// - (batch, out_channels, out_h, out_w) or (out_channels, out_h, out_w)
///
/// Note: kernel layout is transposed from regular conv2d.
pub fn conv2d_transpose(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    conv2d_transpose_with_config(
        input,
        kernel,
        stride,
        padding,
        output_padding,
        1, // groups
        precision,
    )
}

/// Transposed 2D convolution with group support.
pub fn conv2d_transpose_with_config(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    groups: usize,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }
    if kernel.ndim() != 4 {
        return Err(ConvError::InvalidKernel {
            expected: 4,
            got: kernel.ndim(),
        });
    }

    let (batch_size, in_channels, in_h, in_w) = if has_batch {
        (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        )
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    // For transpose conv, kernel is (in_channels, out_channels/groups, kH, kW)
    let (kernel_in_channels, out_channels_per_group, kernel_h, kernel_w) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    if stride.0 == 0 || stride.1 == 0 {
        return Err(ConvError::InvalidStride(stride.0, stride.1));
    }
    if groups == 0 || in_channels % groups != 0 {
        return Err(ConvError::InvalidGroups {
            groups,
            in_channels,
            out_channels: out_channels_per_group * groups,
        });
    }

    let in_channels_per_group = in_channels / groups;
    if kernel_in_channels != in_channels {
        return Err(ConvError::ChannelMismatch {
            input: in_channels,
            kernel: kernel_in_channels,
        });
    }

    let out_channels = out_channels_per_group * groups;

    // Output size for transposed convolution
    let out_h = (in_h - 1) * stride.0 - 2 * padding.0 + kernel_h + output_padding.0;
    let out_w = (in_w - 1) * stride.1 - 2 * padding.1 + kernel_w + output_padding.1;

    if out_h == 0 || out_w == 0 {
        return Err(ConvError::InvalidOutputSize);
    }

    let precision_error = precision.max_relative_error();
    let result_size = batch_size * out_channels * out_h * out_w;
    let mut result_data = vec![BoundedValue::exact(0.0); result_size];

    // Transposed convolution: scatter input values to output positions
    for n in 0..batch_size {
        for ic in 0..in_channels {
            let group = ic / in_channels_per_group;
            let _ic_in_group = ic % in_channels_per_group;
            let oc_start = group * out_channels_per_group;

            for ih in 0..in_h {
                for iw in 0..in_w {
                    let input_val = if has_batch {
                        *input.get(&[n, ic, ih, iw]).unwrap()
                    } else {
                        *input.get(&[ic, ih, iw]).unwrap()
                    };

                    // Skip zero inputs for efficiency
                    if input_val.value().abs() < 1e-15 && input_val.absolute_error() < 1e-15 {
                        continue;
                    }

                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            // Output position
                            let oh = ih * stride.0 + kh;
                            let ow = iw * stride.1 + kw;

                            // Apply padding
                            if oh < padding.0 || ow < padding.1 {
                                continue;
                            }
                            let oh = oh - padding.0;
                            let ow = ow - padding.1;

                            if oh >= out_h || ow >= out_w {
                                continue;
                            }

                            for oc_offset in 0..out_channels_per_group {
                                let oc = oc_start + oc_offset;

                                // Kernel layout: (in_channels, out_channels/groups, kH, kW)
                                let kernel_val = *kernel.get(&[ic, oc_offset, kh, kw]).unwrap();

                                // Compute product with error propagation
                                let prod = input_val.value() * kernel_val.value();
                                let prod_error = propagate_mac_error(
                                    input_val.value(),
                                    input_val.absolute_error(),
                                    kernel_val.value(),
                                    kernel_val.absolute_error(),
                                    precision_error,
                                );

                                // Add to output position
                                let out_idx = if has_batch {
                                    n * out_channels * out_h * out_w
                                        + oc * out_h * out_w
                                        + oh * out_w
                                        + ow
                                } else {
                                    oc * out_h * out_w + oh * out_w + ow
                                };

                                let current = result_data[out_idx];
                                let new_val = current.value() + prod;
                                let new_error = current.absolute_error() + prod_error;
                                result_data[out_idx] =
                                    BoundedValue::new(new_val, ErrorMargin::absolute(new_error));
                            }
                        }
                    }
                }
            }
        }
    }

    // Add final rounding error to all elements
    for val in result_data.iter_mut() {
        let final_error = val.absolute_error() + val.value().abs() * precision_error;
        *val = BoundedValue::new(val.value(), ErrorMargin::absolute(final_error));
    }

    let shape = if has_batch {
        vec![batch_size, out_channels, out_h, out_w]
    } else {
        vec![out_channels, out_h, out_w]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

// ============================================================================
// POOLING OPERATIONS
// ============================================================================

/// Max pooling result with indices for backward pass.
#[derive(Debug, Clone)]
pub struct MaxPool2dResult {
    pub output: BoundedTensor,
    /// Flattened indices of max values in input (for backward pass)
    pub indices: Vec<usize>,
}

/// 2D max pooling with error tracking.
///
/// # Input format
/// - `input`: (batch, channels, height, width) or (channels, height, width)
///
/// # Error propagation
/// Max pooling preserves the error of the selected maximum element.
/// Additionally, when the difference between max candidates is within their
/// error bounds, we conservatively propagate the larger error.
pub fn max_pool2d(
    input: &BoundedTensor,
    config: Pool2dConfig,
) -> Result<MaxPool2dResult, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }

    let Pool2dConfig {
        kernel_size,
        stride,
        padding,
    } = config;

    if kernel_size.0 == 0 || kernel_size.1 == 0 {
        return Err(ConvError::InvalidPoolSize);
    }
    if stride.0 == 0 || stride.1 == 0 {
        return Err(ConvError::InvalidStride(stride.0, stride.1));
    }

    let (batch_size, channels, in_h, in_w) = if has_batch {
        (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        )
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let out_h = conv_output_size(in_h, kernel_size.0, stride.0, padding.0, 1);
    let out_w = conv_output_size(in_w, kernel_size.1, stride.1, padding.1, 1);

    if out_h == 0 || out_w == 0 {
        return Err(ConvError::InvalidOutputSize);
    }

    let result_size = batch_size * channels * out_h * out_w;
    let mut result_data = Vec::with_capacity(result_size);
    let mut indices = Vec::with_capacity(result_size);

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_error = 0.0;
                    let mut max_idx = 0usize;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;

                            // Handle padding
                            let val = if ih < padding.0
                                || iw < padding.1
                                || ih >= in_h + padding.0
                                || iw >= in_w + padding.1
                            {
                                BoundedValue::exact(f64::NEG_INFINITY)
                            } else {
                                let real_h = ih - padding.0;
                                let real_w = iw - padding.1;
                                if has_batch {
                                    *input
                                        .get(&[n, c, real_h, real_w])
                                        .unwrap_or(&BoundedValue::exact(f64::NEG_INFINITY))
                                } else {
                                    *input
                                        .get(&[c, real_h, real_w])
                                        .unwrap_or(&BoundedValue::exact(f64::NEG_INFINITY))
                                }
                            };

                            let v = val.value();
                            let e = val.absolute_error();

                            // Check if this could be the max considering error bounds
                            // If (v + e) > (max_val - max_error), they overlap
                            let could_be_max = v + e > max_val - max_error;

                            if v > max_val {
                                // Clear winner
                                max_val = v;
                                max_error = e;

                                // Compute flat index in input
                                if ih >= padding.0 && iw >= padding.1 {
                                    let real_h = ih - padding.0;
                                    let real_w = iw - padding.1;
                                    max_idx = if has_batch {
                                        n * channels * in_h * in_w
                                            + c * in_h * in_w
                                            + real_h * in_w
                                            + real_w
                                    } else {
                                        c * in_h * in_w + real_h * in_w + real_w
                                    };
                                }
                            } else if could_be_max {
                                // Conservative: take larger error when uncertain
                                max_error = max_error.max(e);
                            }
                        }
                    }

                    result_data.push(BoundedValue::new(max_val, ErrorMargin::absolute(max_error)));
                    indices.push(max_idx);
                }
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, channels, out_h, out_w]
    } else {
        vec![channels, out_h, out_w]
    };

    Ok(MaxPool2dResult {
        output: BoundedTensor::new(result_data, shape),
        indices,
    })
}

/// 2D average pooling with error tracking.
///
/// # Input format
/// - `input`: (batch, channels, height, width) or (channels, height, width)
///
/// # Error propagation
/// Average pooling divides the sum of errors by the pool size.
/// This actually reduces error (averaging smooths noise).
pub fn avg_pool2d(
    input: &BoundedTensor,
    config: Pool2dConfig,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }

    let Pool2dConfig {
        kernel_size,
        stride,
        padding,
    } = config;

    if kernel_size.0 == 0 || kernel_size.1 == 0 {
        return Err(ConvError::InvalidPoolSize);
    }
    if stride.0 == 0 || stride.1 == 0 {
        return Err(ConvError::InvalidStride(stride.0, stride.1));
    }

    let (batch_size, channels, in_h, in_w) = if has_batch {
        (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        )
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let out_h = conv_output_size(in_h, kernel_size.0, stride.0, padding.0, 1);
    let out_w = conv_output_size(in_w, kernel_size.1, stride.1, padding.1, 1);

    if out_h == 0 || out_w == 0 {
        return Err(ConvError::InvalidOutputSize);
    }

    let precision_error = precision.max_relative_error();
    let pool_area = (kernel_size.0 * kernel_size.1) as f64;
    let result_size = batch_size * channels * out_h * out_w;
    let mut result_data = Vec::with_capacity(result_size);

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;
                    let mut sum_error = 0.0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;

                            // Handle padding (padded values are 0)
                            if ih < padding.0
                                || iw < padding.1
                                || ih >= in_h + padding.0
                                || iw >= in_w + padding.1
                            {
                                continue;
                            }

                            let real_h = ih - padding.0;
                            let real_w = iw - padding.1;

                            let val = if has_batch {
                                *input
                                    .get(&[n, c, real_h, real_w])
                                    .unwrap_or(&BoundedValue::exact(0.0))
                            } else {
                                *input
                                    .get(&[c, real_h, real_w])
                                    .unwrap_or(&BoundedValue::exact(0.0))
                            };

                            sum += val.value();
                            sum_error += val.absolute_error();
                        }
                    }

                    let avg = sum / pool_area;
                    // Average reduces error
                    let avg_error = sum_error / pool_area + avg.abs() * precision_error;

                    result_data.push(BoundedValue::new(avg, ErrorMargin::absolute(avg_error)));
                }
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, channels, out_h, out_w]
    } else {
        vec![channels, out_h, out_w]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

/// Backward pass for max pooling.
/// Propagates gradient only to the positions that were selected as max.
pub fn max_pool2d_backward(
    grad_output: &BoundedTensor,
    indices: &[usize],
    input_shape: &[usize],
) -> BoundedTensor {
    let mut grad_input_data = vec![BoundedValue::exact(0.0); input_shape.iter().product()];

    for (i, &idx) in indices.iter().enumerate() {
        if idx < grad_input_data.len() {
            let grad_val = grad_output.data()[i];
            // Add gradient (multiple positions might map to same input)
            let current = grad_input_data[idx];
            let new_val = current.value() + grad_val.value();
            let new_error = current.absolute_error() + grad_val.absolute_error();
            grad_input_data[idx] = BoundedValue::new(new_val, ErrorMargin::absolute(new_error));
        }
    }

    BoundedTensor::new(grad_input_data, input_shape.to_vec())
}

/// Backward pass for average pooling.
/// Distributes gradient evenly to all input positions.
pub fn avg_pool2d_backward(
    grad_output: &BoundedTensor,
    input_shape: &[usize],
    config: Pool2dConfig,
) -> BoundedTensor {
    let has_batch = input_shape.len() == 4;

    let (batch_size, channels, in_h, in_w) = if has_batch {
        (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    } else {
        (1, input_shape[0], input_shape[1], input_shape[2])
    };

    let Pool2dConfig { kernel_size, stride, padding } = config;
    let pool_area = (kernel_size.0 * kernel_size.1) as f64;

    let out_h = conv_output_size(in_h, kernel_size.0, stride.0, padding.0, 1);
    let out_w = conv_output_size(in_w, kernel_size.1, stride.1, padding.1, 1);

    let mut grad_input_data = vec![BoundedValue::exact(0.0); input_shape.iter().product()];

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let grad_idx = if has_batch {
                        n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow
                    } else {
                        c * out_h * out_w + oh * out_w + ow
                    };

                    let grad_val = grad_output.data()[grad_idx];
                    let distributed = grad_val.value() / pool_area;
                    let distributed_error = grad_val.absolute_error() / pool_area;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;

                            if ih < padding.0 || iw < padding.1
                                || ih >= in_h + padding.0 || iw >= in_w + padding.1 {
                                continue;
                            }

                            let real_h = ih - padding.0;
                            let real_w = iw - padding.1;

                            let input_idx = if has_batch {
                                n * channels * in_h * in_w + c * in_h * in_w + real_h * in_w + real_w
                            } else {
                                c * in_h * in_w + real_h * in_w + real_w
                            };

                            let current = grad_input_data[input_idx];
                            let new_val = current.value() + distributed;
                            let new_error = current.absolute_error() + distributed_error;
                            grad_input_data[input_idx] = BoundedValue::new(new_val, ErrorMargin::absolute(new_error));
                        }
                    }
                }
            }
        }
    }

    BoundedTensor::new(grad_input_data, input_shape.to_vec())
}

// ============================================================================
// DEPTHWISE AND GROUP CONVOLUTIONS
// ============================================================================

/// Depthwise 2D convolution with error tracking.
///
/// Each input channel is convolved with its own filter, producing one output channel.
/// This is equivalent to groups = in_channels = out_channels.
///
/// # Input format
/// - `input`: (batch, channels, height, width) or (channels, height, width)
/// - `kernel`: (channels, 1, kernel_h, kernel_w)
///
/// # Use cases
/// - MobileNet-style efficient convolutions
/// - Separable convolutions (depthwise followed by 1x1 pointwise)
pub fn depthwise_conv2d(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }
    if kernel.ndim() != 4 {
        return Err(ConvError::InvalidKernel {
            expected: 4,
            got: kernel.ndim(),
        });
    }

    let (batch_size, channels, in_h, in_w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let (kernel_channels, depth_mult, kernel_h, kernel_w) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    // Depthwise: kernel shape should be (channels, 1, kH, kW) or (channels, depth_multiplier, kH, kW)
    if kernel_channels != channels {
        return Err(ConvError::ChannelMismatch {
            input: channels,
            kernel: kernel_channels,
        });
    }

    if stride.0 == 0 || stride.1 == 0 {
        return Err(ConvError::InvalidStride(stride.0, stride.1));
    }

    let out_channels = channels * depth_mult;
    let out_h = conv_output_size(in_h, kernel_h, stride.0, padding.0, 1);
    let out_w = conv_output_size(in_w, kernel_w, stride.1, padding.1, 1);

    if out_h == 0 || out_w == 0 {
        return Err(ConvError::InvalidOutputSize);
    }

    let precision_error = precision.max_relative_error();
    let result_size = batch_size * out_channels * out_h * out_w;
    let mut result_data = Vec::with_capacity(result_size);

    for n in 0..batch_size {
        for c in 0..channels {
            for dm in 0..depth_mult {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;
                        let mut accumulated_error = 0.0;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;

                                let input_val = if ih < padding.0
                                    || iw < padding.1
                                    || ih >= in_h + padding.0
                                    || iw >= in_w + padding.1
                                {
                                    BoundedValue::exact(0.0)
                                } else {
                                    let real_h = ih - padding.0;
                                    let real_w = iw - padding.1;
                                    if has_batch {
                                        *input.get(&[n, c, real_h, real_w])
                                            .unwrap_or(&BoundedValue::exact(0.0))
                                    } else {
                                        *input.get(&[c, real_h, real_w])
                                            .unwrap_or(&BoundedValue::exact(0.0))
                                    }
                                };

                                let kernel_val = *kernel.get(&[c, dm, kh, kw]).unwrap();

                                let prod = input_val.value() * kernel_val.value();
                                let prod_error = propagate_mac_error(
                                    input_val.value(),
                                    input_val.absolute_error(),
                                    kernel_val.value(),
                                    kernel_val.absolute_error(),
                                    precision_error,
                                );

                                sum += prod;
                                accumulated_error += prod_error;
                            }
                        }

                        let final_error = accumulated_error + sum.abs() * precision_error;
                        result_data.push(BoundedValue::new(sum, ErrorMargin::absolute(final_error)));
                    }
                }
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, out_channels, out_h, out_w]
    } else {
        vec![out_channels, out_h, out_w]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

/// Depthwise separable 2D convolution with error tracking.
///
/// Combines:
/// 1. Depthwise convolution (spatial filtering per channel)
/// 2. Pointwise (1x1) convolution (channel mixing)
///
/// # Input format
/// - `input`: (batch, in_channels, height, width) or (in_channels, height, width)
/// - `depthwise_kernel`: (in_channels, 1, kernel_h, kernel_w)
/// - `pointwise_kernel`: (out_channels, in_channels, 1, 1)
///
/// # Use cases
/// - MobileNet, EfficientNet architectures
/// - Reduces parameters and computation vs standard conv
pub fn depthwise_separable_conv2d(
    input: &BoundedTensor,
    depthwise_kernel: &BoundedTensor,
    pointwise_kernel: &BoundedTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    // Step 1: Depthwise convolution
    let depthwise_output = depthwise_conv2d(input, depthwise_kernel, stride, padding, precision)?;

    // Step 2: Pointwise (1x1) convolution
    // Pointwise kernel should be (out_channels, in_channels, 1, 1)
    conv2d_with_config(
        &depthwise_output,
        pointwise_kernel,
        Conv2dConfig::new(),
        precision,
    )
}

/// Group 2D convolution with error tracking.
///
/// Divides input channels into groups, each processed independently.
/// Standard conv2d is groups=1, depthwise is groups=in_channels.
///
/// # Input format
/// - `input`: (batch, in_channels, height, width)
/// - `kernel`: (out_channels, in_channels/groups, kernel_h, kernel_w)
///
/// # Use cases
/// - ResNeXt architectures (groups=32 typical)
/// - Reducing computation while maintaining capacity
pub fn group_conv2d(
    input: &BoundedTensor,
    kernel: &BoundedTensor,
    groups: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    conv2d_with_config(
        input,
        kernel,
        Conv2dConfig::new().stride(stride).padding(padding).groups(groups),
        precision,
    )
}

// ============================================================================
// BACKWARD PASS HELPERS
// ============================================================================

/// Computes gradient with respect to input for conv2d.
///
/// Given grad_output (dL/dY) and kernel weights, computes dL/dX.
/// This is a transposed convolution operation.
pub fn conv2d_backward_input(
    grad_output: &BoundedTensor,
    kernel: &BoundedTensor,
    input_shape: &[usize],
    stride: (usize, usize),
    padding: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    // For cross-correlation forward: y = xcorr(x, w)
    // Backward for input: dL/dX = conv2d_transpose(dL/dY, W_transposed)
    // where W_transposed swaps (out_ch, in_ch) dims but does NOT flip spatially,
    // since the forward pass is cross-correlation (no spatial flip).

    let (out_channels, in_channels_per_group, kernel_h, kernel_w) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    // Transpose channel dimensions only: (out_ch, in_ch, kH, kW) -> (in_ch, out_ch, kH, kW)
    // No spatial flip needed for cross-correlation backward.
    let mut transposed_data = vec![BoundedValue::exact(0.0); kernel.len()];

    for oc in 0..out_channels {
        for ic in 0..in_channels_per_group {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let src_idx = oc * in_channels_per_group * kernel_h * kernel_w
                        + ic * kernel_h * kernel_w
                        + kh * kernel_w
                        + kw;

                    // Transpose channels: (oc, ic) -> (ic, oc)
                    // Keep spatial dimensions as-is
                    let dst_idx = ic * out_channels * kernel_h * kernel_w
                        + oc * kernel_h * kernel_w
                        + kh * kernel_w
                        + kw;

                    transposed_data[dst_idx] = kernel.data()[src_idx];
                }
            }
        }
    }

    let in_channels = input_shape[if input_shape.len() == 4 { 1 } else { 0 }];
    let transposed_kernel = BoundedTensor::new(
        transposed_data,
        vec![in_channels, out_channels, kernel_h, kernel_w],
    );

    // Calculate output padding needed to match input size
    let out_h = input_shape[if input_shape.len() == 4 { 2 } else { 1 }];
    let out_w = input_shape[if input_shape.len() == 4 { 3 } else { 2 }];

    let grad_h = grad_output.shape()[if grad_output.ndim() == 4 { 2 } else { 1 }];
    let grad_w = grad_output.shape()[if grad_output.ndim() == 4 { 3 } else { 2 }];

    // Output size of transposed conv: (in_h - 1) * stride - 2*pad + kernel + out_pad = out_h
    let expected_h = (grad_h - 1) * stride.0 + kernel_h;
    let expected_w = (grad_w - 1) * stride.1 + kernel_w;

    let output_padding = (
        if out_h + 2 * padding.0 > expected_h { out_h + 2 * padding.0 - expected_h } else { 0 },
        if out_w + 2 * padding.1 > expected_w { out_w + 2 * padding.1 - expected_w } else { 0 },
    );

    conv2d_transpose(
        grad_output,
        &transposed_kernel,
        stride,
        padding,
        output_padding,
        precision,
    )
}

/// Computes gradient with respect to kernel weights for conv2d.
///
/// Given grad_output (dL/dY) and input, computes dL/dW.
pub fn conv2d_backward_weight(
    input: &BoundedTensor,
    grad_output: &BoundedTensor,
    kernel_shape: &[usize],
    stride: (usize, usize),
    padding: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;

    let (batch_size, in_channels, in_h, in_w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let (out_channels, in_channels_per_group, kernel_h, kernel_w) = (
        kernel_shape[0],
        kernel_shape[1],
        kernel_shape[2],
        kernel_shape[3],
    );

    let groups = in_channels / in_channels_per_group;
    let out_channels_per_group = out_channels / groups;

    let out_h = grad_output.shape()[if has_batch { 2 } else { 1 }];
    let out_w = grad_output.shape()[if has_batch { 3 } else { 2 }];

    let precision_error = precision.max_relative_error();
    let mut grad_weight_data = vec![BoundedValue::exact(0.0); kernel_shape.iter().product()];

    for n in 0..batch_size {
        for oc in 0..out_channels {
            let group = oc / out_channels_per_group;
            let ic_start = group * in_channels_per_group;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let grad_val = if has_batch {
                        *grad_output.get(&[n, oc, oh, ow]).unwrap()
                    } else {
                        *grad_output.get(&[oc, oh, ow]).unwrap()
                    };

                    for ic_offset in 0..in_channels_per_group {
                        let ic = ic_start + ic_offset;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;

                                let input_val = if ih < padding.0
                                    || iw < padding.1
                                    || ih >= in_h + padding.0
                                    || iw >= in_w + padding.1
                                {
                                    BoundedValue::exact(0.0)
                                } else {
                                    let real_h = ih - padding.0;
                                    let real_w = iw - padding.1;
                                    if has_batch {
                                        *input.get(&[n, ic, real_h, real_w])
                                            .unwrap_or(&BoundedValue::exact(0.0))
                                    } else {
                                        *input.get(&[ic, real_h, real_w])
                                            .unwrap_or(&BoundedValue::exact(0.0))
                                    }
                                };

                                // dL/dW[oc, ic, kh, kw] += grad_output[n, oc, oh, ow] * input[n, ic, ih, iw]
                                let prod = grad_val.value() * input_val.value();
                                let prod_error = propagate_mac_error(
                                    grad_val.value(),
                                    grad_val.absolute_error(),
                                    input_val.value(),
                                    input_val.absolute_error(),
                                    precision_error,
                                );

                                let weight_idx = oc * in_channels_per_group * kernel_h * kernel_w
                                    + ic_offset * kernel_h * kernel_w
                                    + kh * kernel_w
                                    + kw;

                                let current = grad_weight_data[weight_idx];
                                let new_val = current.value() + prod;
                                let new_error = current.absolute_error() + prod_error;
                                grad_weight_data[weight_idx] =
                                    BoundedValue::new(new_val, ErrorMargin::absolute(new_error));
                            }
                        }
                    }
                }
            }
        }
    }

    // Add final rounding error
    for val in grad_weight_data.iter_mut() {
        let final_error = val.absolute_error() + val.value().abs() * precision_error;
        *val = BoundedValue::new(val.value(), ErrorMargin::absolute(final_error));
    }

    Ok(BoundedTensor::new(grad_weight_data, kernel_shape.to_vec()))
}

/// Computes gradient with respect to input for conv1d.
pub fn conv1d_backward_input(
    grad_output: &BoundedTensor,
    kernel: &BoundedTensor,
    input_shape: &[usize],
    stride: usize,
    padding: usize,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input_shape.len() == 3;

    let (batch_size, in_channels, in_length) = if has_batch {
        (input_shape[0], input_shape[1], input_shape[2])
    } else {
        (1, input_shape[0], input_shape[1])
    };

    let (out_channels, kernel_in_channels, kernel_size) =
        (kernel.shape()[0], kernel.shape()[1], kernel.shape()[2]);

    let out_length = grad_output.shape()[if has_batch { 2 } else { 1 }];
    let precision_error = precision.max_relative_error();

    let mut grad_input_data = vec![BoundedValue::exact(0.0); input_shape.iter().product()];

    for n in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..out_length {
                let grad_val = if has_batch {
                    *grad_output.get(&[n, oc, ol]).unwrap()
                } else {
                    *grad_output.get(&[oc, ol]).unwrap()
                };

                for ic in 0..kernel_in_channels {
                    for k in 0..kernel_size {
                        let in_pos = ol * stride + k;

                        if in_pos < padding || in_pos >= in_length + padding {
                            continue;
                        }

                        let real_pos = in_pos - padding;
                        let kernel_val = *kernel.get(&[oc, ic, k]).unwrap();

                        let prod = grad_val.value() * kernel_val.value();
                        let prod_error = propagate_mac_error(
                            grad_val.value(),
                            grad_val.absolute_error(),
                            kernel_val.value(),
                            kernel_val.absolute_error(),
                            precision_error,
                        );

                        let input_idx = if has_batch {
                            n * in_channels * in_length + ic * in_length + real_pos
                        } else {
                            ic * in_length + real_pos
                        };

                        let current = grad_input_data[input_idx];
                        let new_val = current.value() + prod;
                        let new_error = current.absolute_error() + prod_error;
                        grad_input_data[input_idx] = BoundedValue::new(new_val, ErrorMargin::absolute(new_error));
                    }
                }
            }
        }
    }

    // Add final rounding error
    for val in grad_input_data.iter_mut() {
        let final_error = val.absolute_error() + val.value().abs() * precision_error;
        *val = BoundedValue::new(val.value(), ErrorMargin::absolute(final_error));
    }

    Ok(BoundedTensor::new(grad_input_data, input_shape.to_vec()))
}

/// Computes gradient with respect to kernel weights for conv1d.
pub fn conv1d_backward_weight(
    input: &BoundedTensor,
    grad_output: &BoundedTensor,
    kernel_shape: &[usize],
    stride: usize,
    padding: usize,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 3;

    let (batch_size, _in_channels, in_length) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2])
    } else {
        (1, input.shape()[0], input.shape()[1])
    };

    let (out_channels, kernel_in_channels, kernel_size) =
        (kernel_shape[0], kernel_shape[1], kernel_shape[2]);

    let out_length = grad_output.shape()[if has_batch { 2 } else { 1 }];
    let precision_error = precision.max_relative_error();

    let mut grad_weight_data = vec![BoundedValue::exact(0.0); kernel_shape.iter().product()];

    for n in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..out_length {
                let grad_val = if has_batch {
                    *grad_output.get(&[n, oc, ol]).unwrap()
                } else {
                    *grad_output.get(&[oc, ol]).unwrap()
                };

                for ic in 0..kernel_in_channels {
                    for k in 0..kernel_size {
                        let in_pos = ol * stride + k;

                        let input_val = if in_pos < padding || in_pos >= in_length + padding {
                            BoundedValue::exact(0.0)
                        } else {
                            let real_pos = in_pos - padding;
                            if has_batch {
                                *input.get(&[n, ic, real_pos]).unwrap_or(&BoundedValue::exact(0.0))
                            } else {
                                *input.get(&[ic, real_pos]).unwrap_or(&BoundedValue::exact(0.0))
                            }
                        };

                        let prod = grad_val.value() * input_val.value();
                        let prod_error = propagate_mac_error(
                            grad_val.value(),
                            grad_val.absolute_error(),
                            input_val.value(),
                            input_val.absolute_error(),
                            precision_error,
                        );

                        let weight_idx = oc * kernel_in_channels * kernel_size + ic * kernel_size + k;

                        let current = grad_weight_data[weight_idx];
                        let new_val = current.value() + prod;
                        let new_error = current.absolute_error() + prod_error;
                        grad_weight_data[weight_idx] = BoundedValue::new(new_val, ErrorMargin::absolute(new_error));
                    }
                }
            }
        }
    }

    // Add final rounding error
    for val in grad_weight_data.iter_mut() {
        let final_error = val.absolute_error() + val.value().abs() * precision_error;
        *val = BoundedValue::new(val.value(), ErrorMargin::absolute(final_error));
    }

    Ok(BoundedTensor::new(grad_weight_data, kernel_shape.to_vec()))
}

// ============================================================================
// GLOBAL POOLING
// ============================================================================

/// Global average pooling - reduces spatial dimensions to 1x1.
pub fn global_avg_pool2d(
    input: &BoundedTensor,
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }

    let (batch_size, channels, h, w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let precision_error = precision.max_relative_error();
    let spatial_size = (h * w) as f64;
    let mut result_data = Vec::with_capacity(batch_size * channels);

    for n in 0..batch_size {
        for c in 0..channels {
            let mut sum = 0.0;
            let mut sum_error = 0.0;

            for i in 0..h {
                for j in 0..w {
                    let val = if has_batch {
                        *input.get(&[n, c, i, j]).unwrap()
                    } else {
                        *input.get(&[c, i, j]).unwrap()
                    };
                    sum += val.value();
                    sum_error += val.absolute_error();
                }
            }

            let avg = sum / spatial_size;
            let avg_error = sum_error / spatial_size + avg.abs() * precision_error;
            result_data.push(BoundedValue::new(avg, ErrorMargin::absolute(avg_error)));
        }
    }

    let shape = if has_batch {
        vec![batch_size, channels, 1, 1]
    } else {
        vec![channels, 1, 1]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

/// Global max pooling - takes max over spatial dimensions.
pub fn global_max_pool2d(
    input: &BoundedTensor,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }

    let (batch_size, channels, h, w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let mut result_data = Vec::with_capacity(batch_size * channels);

    for n in 0..batch_size {
        for c in 0..channels {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_error = 0.0;

            for i in 0..h {
                for j in 0..w {
                    let val = if has_batch {
                        *input.get(&[n, c, i, j]).unwrap()
                    } else {
                        *input.get(&[c, i, j]).unwrap()
                    };

                    let v = val.value();
                    let e = val.absolute_error();

                    if v > max_val {
                        max_val = v;
                        max_error = e;
                    } else if v + e > max_val - max_error {
                        // Overlapping bounds - take larger error
                        max_error = max_error.max(e);
                    }
                }
            }

            result_data.push(BoundedValue::new(max_val, ErrorMargin::absolute(max_error)));
        }
    }

    let shape = if has_batch {
        vec![batch_size, channels, 1, 1]
    } else {
        vec![channels, 1, 1]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

// ============================================================================
// ADAPTIVE POOLING
// ============================================================================

/// Adaptive average pooling - outputs a fixed size regardless of input size.
pub fn adaptive_avg_pool2d(
    input: &BoundedTensor,
    output_size: (usize, usize),
    precision: Precision,
) -> Result<BoundedTensor, ConvError> {
    let has_batch = input.ndim() == 4;
    if input.ndim() != 3 && input.ndim() != 4 {
        return Err(ConvError::InvalidDimensions {
            expected: 4,
            got: input.ndim(),
        });
    }

    let (batch_size, channels, in_h, in_w) = if has_batch {
        (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (1, input.shape()[0], input.shape()[1], input.shape()[2])
    };

    let (out_h, out_w) = output_size;
    if out_h == 0 || out_w == 0 {
        return Err(ConvError::InvalidOutputSize);
    }

    let precision_error = precision.max_relative_error();
    let result_size = batch_size * channels * out_h * out_w;
    let mut result_data = Vec::with_capacity(result_size);

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    // Calculate input region for this output position
                    let ih_start = (oh * in_h) / out_h;
                    let ih_end = ((oh + 1) * in_h + out_h - 1) / out_h;
                    let iw_start = (ow * in_w) / out_w;
                    let iw_end = ((ow + 1) * in_w + out_w - 1) / out_w;

                    let mut sum = 0.0;
                    let mut sum_error = 0.0;
                    let mut count = 0;

                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            let val = if has_batch {
                                *input.get(&[n, c, ih, iw]).unwrap()
                            } else {
                                *input.get(&[c, ih, iw]).unwrap()
                            };
                            sum += val.value();
                            sum_error += val.absolute_error();
                            count += 1;
                        }
                    }

                    let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                    let avg_error = if count > 0 {
                        sum_error / count as f64 + avg.abs() * precision_error
                    } else {
                        0.0
                    };

                    result_data.push(BoundedValue::new(avg, ErrorMargin::absolute(avg_error)));
                }
            }
        }
    }

    let shape = if has_batch {
        vec![batch_size, channels, out_h, out_w]
    } else {
        vec![channels, out_h, out_w]
    };

    Ok(BoundedTensor::new(result_data, shape))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_input_4d() -> BoundedTensor {
        // (1, 1, 4, 4) tensor with values 1-16
        let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        BoundedTensor::from_exact(data, vec![1, 1, 4, 4])
    }

    fn create_test_kernel_3x3() -> BoundedTensor {
        // (1, 1, 3, 3) kernel with all 1s
        let data = vec![1.0; 9];
        BoundedTensor::from_exact(data, vec![1, 1, 3, 3])
    }

    #[test]
    fn test_conv2d_basic() {
        let input = create_test_input_4d();
        let kernel = create_test_kernel_3x3();

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 2, 2]);

        // Expected output:
        // [1+2+3+5+6+7+9+10+11, 2+3+4+6+7+8+10+11+12]   = [54, 63]
        // [5+6+7+9+10+11+13+14+15, 6+7+8+10+11+12+14+15+16] = [90, 99]
        let values = result.values();
        assert!((values[0] - 54.0).abs() < 1e-10);
        assert!((values[1] - 63.0).abs() < 1e-10);
        assert!((values[2] - 90.0).abs() < 1e-10);
        assert!((values[3] - 99.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let input = create_test_input_4d();
        let kernel = create_test_kernel_3x3();

        let result = conv2d(&input, &kernel, (1, 1), (1, 1), Precision::F32).unwrap();

        // With padding=1, output size should be same as input
        assert_eq!(result.shape(), &vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_conv2d_with_stride() {
        let input = create_test_input_4d();
        let kernel = create_test_kernel_3x3();

        let result = conv2d(&input, &kernel, (2, 2), (0, 0), Precision::F32).unwrap();

        // With stride=2, output should be 1x1
        assert_eq!(result.shape(), &vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_conv1d_basic() {
        // (1, 2, 5) input
        let input = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1, 2, 5],
        );
        // (1, 2, 3) kernel
        let kernel = BoundedTensor::from_exact(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1, 2, 3],
        );

        let result = conv1d(&input, &kernel, 1, 0, Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 3]);

        // Expected: (1+2+3)+(5+4+3)=18, (2+3+4)+(4+3+2)=18, (3+4+5)+(3+2+1)=18
        let values = result.values();
        assert!((values[0] - 18.0).abs() < 1e-10);
        assert!((values[1] - 18.0).abs() < 1e-10);
        assert!((values[2] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_pool2d() {
        let input = create_test_input_4d();
        let config = Pool2dConfig::new((2, 2));

        let result = max_pool2d(&input, config).unwrap();

        assert_eq!(result.output.shape(), &vec![1, 1, 2, 2]);

        // Max values in each 2x2 region:
        // [6, 8, 14, 16]
        let values = result.output.values();
        assert!((values[0] - 6.0).abs() < 1e-10);
        assert!((values[1] - 8.0).abs() < 1e-10);
        assert!((values[2] - 14.0).abs() < 1e-10);
        assert!((values[3] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_avg_pool2d() {
        let input = create_test_input_4d();
        let config = Pool2dConfig::new((2, 2));

        let result = avg_pool2d(&input, config, Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 2, 2]);

        // Avg values in each 2x2 region:
        // (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5, (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5
        let values = result.values();
        assert!((values[0] - 3.5).abs() < 1e-10);
        assert!((values[1] - 5.5).abs() < 1e-10);
        assert!((values[2] - 11.5).abs() < 1e-10);
        assert!((values[3] - 13.5).abs() < 1e-10);
    }

    #[test]
    fn test_global_avg_pool2d() {
        let input = create_test_input_4d();
        let result = global_avg_pool2d(&input, Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 1, 1]);

        // Average of 1-16 = (1+16)*16/2/16 = 8.5
        let values = result.values();
        assert!((values[0] - 8.5).abs() < 1e-10);
    }

    #[test]
    fn test_depthwise_conv2d() {
        // (1, 2, 4, 4) input
        let input_data: Vec<f64> = (1..=32).map(|x| x as f64).collect();
        let input = BoundedTensor::from_exact(input_data, vec![1, 2, 4, 4]);

        // (2, 1, 2, 2) kernel - each channel has its own filter
        let kernel = BoundedTensor::from_exact(vec![1.0; 8], vec![2, 1, 2, 2]);

        let result = depthwise_conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 2, 3, 3]);
    }

    #[test]
    fn test_group_conv2d() {
        // (1, 4, 4, 4) input with 4 channels
        let input_data: Vec<f64> = (1..=64).map(|x| x as f64).collect();
        let input = BoundedTensor::from_exact(input_data, vec![1, 4, 4, 4]);

        // (4, 2, 2, 2) kernel with groups=2
        // This means each group has 2 input channels and 2 output channels
        let kernel = BoundedTensor::from_exact(vec![1.0; 32], vec![4, 2, 2, 2]);

        let result = group_conv2d(&input, &kernel, 2, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 4, 3, 3]);
    }

    #[test]
    fn test_conv2d_with_dilation() {
        // (1, 1, 5, 5) input
        let input_data: Vec<f64> = (1..=25).map(|x| x as f64).collect();
        let input = BoundedTensor::from_exact(input_data, vec![1, 1, 5, 5]);

        // (1, 1, 3, 3) kernel with dilation=2
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let config = Conv2dConfig::new().stride((1, 1)).padding((0, 0)).dilation((2, 2));
        let result = conv2d_with_config(&input, &kernel, config, Precision::F32).unwrap();

        // With dilation=2, effective kernel size is 5x5, so output is 1x1
        assert_eq!(result.shape(), &vec![1, 1, 1, 1]);

        // The result should be the sum of the corners and center:
        // 1 + 3 + 5 + 11 + 13 + 15 + 21 + 23 + 25 = 117
        let values = result.values();
        assert!((values[0] - 117.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_propagation_conv2d() {
        // Test that errors accumulate properly
        let input = BoundedTensor::from_approximate(vec![1.0; 16], vec![1, 1, 4, 4], 0.1);
        let kernel = BoundedTensor::from_approximate(vec![1.0; 9], vec![1, 1, 3, 3], 0.05);

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        // Each output element is sum of 9 MAC operations
        // Error should be > 0 and reasonable
        let max_error = result.max_error();
        assert!(max_error > 0.0);
        assert!(max_error < 10.0); // Reasonable upper bound
    }

    #[test]
    fn test_conv2d_transpose_basic() {
        // (1, 1, 2, 2) input
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        // (1, 1, 2, 2) kernel
        let kernel = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);

        let result = conv2d_transpose(&input, &kernel, (1, 1), (0, 0), (0, 0), Precision::F32).unwrap();

        // Output shape should be 3x3
        assert_eq!(result.shape(), &vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        let input = create_test_input_4d();
        let result = adaptive_avg_pool2d(&input, (2, 2), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_dimension_validation() {
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]); // 1D tensor
        let kernel = create_test_kernel_3x3();

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_channel_mismatch() {
        let input = BoundedTensor::from_exact(vec![1.0; 32], vec![1, 2, 4, 4]); // 2 channels
        let kernel = BoundedTensor::from_exact(vec![1.0; 27], vec![1, 3, 3, 3]); // Expects 3 channels

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_backward_weight() {
        let input = create_test_input_4d();
        let grad_output = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);
        let kernel_shape = vec![1, 1, 3, 3];

        let grad_weight = conv2d_backward_weight(
            &input,
            &grad_output,
            &kernel_shape,
            (1, 1),
            (0, 0),
            Precision::F32,
        ).unwrap();

        assert_eq!(grad_weight.shape(), &kernel_shape);
        // Verify gradient is non-zero
        assert!(grad_weight.values().iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_max_pool2d_backward() {
        let input = create_test_input_4d();
        let config = Pool2dConfig::new((2, 2));

        let forward = max_pool2d(&input, config).unwrap();
        let grad_output = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);

        let grad_input = max_pool2d_backward(&grad_output, &forward.indices, input.shape());

        assert_eq!(grad_input.shape(), input.shape());
        // Gradient should only be at max positions
        assert!(grad_input.values().iter().filter(|&&v| v.abs() > 1e-10).count() == 4);
    }

    #[test]
    fn test_avg_pool2d_backward() {
        let input_shape = vec![1, 1, 4, 4];
        let config = Pool2dConfig::new((2, 2));
        let grad_output = BoundedTensor::from_exact(vec![4.0; 4], vec![1, 1, 2, 2]);

        let grad_input = avg_pool2d_backward(&grad_output, &input_shape, config);

        assert_eq!(grad_input.shape(), &input_shape);
        // Each input gets 4.0/4 = 1.0 gradient
        assert!(grad_input.values().iter().all(|&v| (v - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_conv1d_no_batch() {
        // (2, 5) input without batch dimension
        let input = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            vec![2, 5],
        );
        // (1, 2, 3) kernel
        let kernel = BoundedTensor::from_exact(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1, 2, 3],
        );

        let result = conv1d(&input, &kernel, 1, 0, Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 3]);
    }

    #[test]
    fn test_conv2d_no_batch() {
        // (1, 4, 4) input without batch dimension
        let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let input = BoundedTensor::from_exact(data, vec![1, 4, 4]);

        // (1, 1, 3, 3) kernel
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 2, 2]);
    }

    // ========================================================================
    // ADDITIONAL COMPREHENSIVE TESTS
    // ========================================================================

    #[test]
    fn test_conv2d_identity_kernel() {
        // 1x1 kernel should just copy (with bias if present)
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let kernel = BoundedTensor::from_exact(vec![1.0], vec![1, 1, 1, 1]);

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 2, 2]);
        let values = result.values();
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 2.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
        assert!((values[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_multi_channel() {
        // Test with multiple input and output channels
        let input = BoundedTensor::from_exact(
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            vec![1, 2, 2, 2],
        );
        // Two output channels, each with 2 input channel weights
        // Shape: (out_ch=2, in_ch=2, kH=1, kW=1)
        let kernel = BoundedTensor::from_exact(
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2, 2, 1, 1],
        );

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 2, 2, 2]);
        // Each output = sum of both input channels = 1 + 2 = 3
        for val in result.values() {
            assert!((val - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_conv2d_batch_processing() {
        // Test batch dimension handling
        let input = BoundedTensor::from_exact(
            vec![1.0; 32], // 2 batches * 1 channel * 4 * 4
            vec![2, 1, 4, 4],
        );
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![2, 1, 2, 2]);
        // Each output should be 9.0 (sum of 9 ones)
        for val in result.values() {
            assert!((val - 9.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_conv2d_asymmetric_stride() {
        let input = BoundedTensor::from_exact(
            (1..=36).map(|x| x as f64).collect(),
            vec![1, 1, 6, 6],
        );
        let kernel = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);

        // Stride (2, 1) - different in each dimension
        let result = conv2d(&input, &kernel, (2, 1), (0, 0), Precision::F32).unwrap();

        // Output: (6-2)/2+1=3, (6-2)/1+1=5
        assert_eq!(result.shape(), &vec![1, 1, 3, 5]);
    }

    #[test]
    fn test_conv2d_asymmetric_padding() {
        let input = BoundedTensor::from_exact(vec![1.0; 16], vec![1, 1, 4, 4]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);

        // Padding (1, 0) - different in each dimension
        let result = conv2d(&input, &kernel, (1, 1), (1, 0), Precision::F32).unwrap();

        // With padding: h=4+2*1-2+1=5, w=4+2*0-2+1=3
        assert_eq!(result.shape(), &vec![1, 1, 5, 3]);
    }

    #[test]
    fn test_error_bounds_compositionality() {
        // Verify that error bounds compose correctly across multiple operations
        let input1 = BoundedTensor::from_approximate(vec![1.0; 16], vec![1, 1, 4, 4], 0.01);
        let kernel1 = BoundedTensor::from_approximate(vec![1.0; 9], vec![1, 1, 3, 3], 0.01);

        let result1 = conv2d(&input1, &kernel1, (1, 1), (0, 0), Precision::F32).unwrap();
        let error1 = result1.max_error();

        // More error in inputs should result in more error in output
        let input2 = BoundedTensor::from_approximate(vec![1.0; 16], vec![1, 1, 4, 4], 0.1);
        let kernel2 = BoundedTensor::from_approximate(vec![1.0; 9], vec![1, 1, 3, 3], 0.1);

        let result2 = conv2d(&input2, &kernel2, (1, 1), (0, 0), Precision::F32).unwrap();
        let error2 = result2.max_error();

        assert!(error2 > error1, "Larger input error should produce larger output error");
    }

    #[test]
    fn test_error_bounds_scale_with_kernel_size() {
        // Larger kernels accumulate more error
        let input = BoundedTensor::from_approximate(vec![1.0; 64], vec![1, 1, 8, 8], 0.01);

        let kernel_3x3 = BoundedTensor::from_approximate(vec![1.0; 9], vec![1, 1, 3, 3], 0.01);
        let kernel_5x5 = BoundedTensor::from_approximate(vec![1.0; 25], vec![1, 1, 5, 5], 0.01);

        let result_3x3 = conv2d(&input, &kernel_3x3, (1, 1), (1, 1), Precision::F32).unwrap();
        let result_5x5 = conv2d(&input, &kernel_5x5, (1, 1), (2, 2), Precision::F32).unwrap();

        assert!(
            result_5x5.max_error() > result_3x3.max_error(),
            "Larger kernel should accumulate more error"
        );
    }

    #[test]
    fn test_pooling_error_propagation() {
        // Average pooling with worst-case error bounds:
        // If all elements have error ε, sum error = n*ε, average error = n*ε/n = ε
        // The error is not reduced for worst-case bounds (only for uncorrelated errors).
        // This test verifies the error bound is correctly propagated (not inflated).
        let input = BoundedTensor::from_approximate(vec![1.0; 16], vec![1, 1, 4, 4], 0.1);
        let input_error = input.max_error();

        let config = Pool2dConfig::new((2, 2));
        let result = avg_pool2d(&input, config, Precision::F32).unwrap();
        let output_error = result.max_error();

        // Error should be approximately the same (plus small precision error)
        // The sum_error / pool_area = (4 * 0.1) / 4 = 0.1
        let expected_error = input_error; // Within precision tolerance
        assert!(
            (output_error - expected_error).abs() < 0.01,
            "Avg pooling error should be close to input error: {} ≈ {}",
            output_error,
            expected_error
        );
    }

    #[test]
    fn test_max_pool_preserves_error() {
        // Max pooling should preserve or slightly increase error
        let input = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], 0.1);
        let input_error = input.max_error();

        let config = Pool2dConfig::new((2, 2));
        let result = max_pool2d(&input, config).unwrap();
        let output_error = result.output.max_error();

        // Max pool should not significantly increase error for non-overlapping regions
        assert!(
            output_error <= input_error * 2.0,
            "Max pooling error should be bounded"
        );
    }

    #[test]
    fn test_conv2d_zero_input() {
        // Zero input should produce zero output
        let input = BoundedTensor::zeros(vec![1, 1, 4, 4]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        for val in result.values() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_conv2d_zero_kernel() {
        // Zero kernel should produce zero output
        let input = BoundedTensor::from_exact(vec![1.0; 16], vec![1, 1, 4, 4]);
        let kernel = BoundedTensor::zeros(vec![1, 1, 3, 3]);

        let result = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();

        for val in result.values() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_depthwise_separable_composition() {
        // Depthwise separable should match expected output
        let input = BoundedTensor::from_exact(vec![1.0; 32], vec![1, 2, 4, 4]);
        let depthwise = BoundedTensor::from_exact(vec![1.0; 8], vec![2, 1, 2, 2]);
        let pointwise = BoundedTensor::from_exact(vec![1.0; 8], vec![4, 2, 1, 1]);

        let result = depthwise_separable_conv2d(
            &input,
            &depthwise,
            &pointwise,
            (1, 1),
            (0, 0),
            Precision::F32,
        ).unwrap();

        assert_eq!(result.shape(), &vec![1, 4, 3, 3]);
    }

    #[test]
    fn test_conv1d_edge_cases() {
        // Test with minimum valid sizes
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
        let kernel = BoundedTensor::from_exact(vec![1.0, 1.0, 1.0], vec![1, 1, 3]);

        let result = conv1d(&input, &kernel, 1, 0, Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 1]);
        // 1 + 2 + 3 = 6
        assert!((result.values()[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_max_pool2d() {
        let input = BoundedTensor::from_exact(
            (1..=16).map(|x| x as f64).collect(),
            vec![1, 1, 4, 4],
        );
        let result = global_max_pool2d(&input).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 1, 1]);
        assert!((result.values()[0] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_avg_pool2d_various_sizes() {
        let input = BoundedTensor::from_exact(vec![1.0; 49], vec![1, 1, 7, 7]);

        // Pool to different sizes
        for out_size in [(1, 1), (2, 2), (3, 3), (7, 7)].iter() {
            let result = adaptive_avg_pool2d(&input, *out_size, Precision::F32).unwrap();
            assert_eq!(result.shape(), &vec![1, 1, out_size.0, out_size.1]);
            // All inputs are 1.0, so output should be 1.0
            for val in result.values() {
                assert!((val - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_conv2d_precision_effects() {
        // Different precision levels should affect error bounds
        let input = BoundedTensor::from_exact(vec![1.0; 16], vec![1, 1, 4, 4]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let result_f32 = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F32).unwrap();
        let result_f16 = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::F16).unwrap();
        let result_int8 = conv2d(&input, &kernel, (1, 1), (0, 0), Precision::INT8).unwrap();

        // Lower precision = higher error bounds
        assert!(result_f32.max_error() < result_f16.max_error());
        assert!(result_f16.max_error() < result_int8.max_error());
    }

    #[test]
    fn test_backward_input_gradient_shape() {
        let input_shape = vec![1, 1, 4, 4];
        let grad_output = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let grad_input = conv2d_backward_input(
            &grad_output,
            &kernel,
            &input_shape,
            (1, 1),
            (0, 0),
            Precision::F32,
        ).unwrap();

        assert_eq!(grad_input.shape(), &input_shape);
    }

    #[test]
    fn test_conv1d_backward_weight_shape() {
        let input = BoundedTensor::from_exact(vec![1.0; 15], vec![1, 3, 5]);
        let grad_output = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 3, 3]);
        let kernel_shape = vec![3, 3, 3];

        let grad_weight = conv1d_backward_weight(
            &input,
            &grad_output,
            &kernel_shape,
            1,
            0,
            Precision::F32,
        ).unwrap();

        assert_eq!(grad_weight.shape(), &kernel_shape);
    }

    #[test]
    fn test_pooling_stride_overlap() {
        // Test overlapping pool regions (stride < kernel_size)
        let input = BoundedTensor::from_exact(
            (1..=16).map(|x| x as f64).collect(),
            vec![1, 1, 4, 4],
        );
        let config = Pool2dConfig::new((3, 3)).stride((1, 1));

        let result = max_pool2d(&input, config).unwrap();

        // Output size: (4-3)/1+1 = 2
        assert_eq!(result.output.shape(), &vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_error_invalid_stride() {
        let input = BoundedTensor::from_exact(vec![1.0; 16], vec![1, 1, 4, 4]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        // Zero stride should fail
        let result = conv2d(&input, &kernel, (0, 1), (0, 0), Precision::F32);
        assert!(matches!(result, Err(ConvError::InvalidStride(0, 1))));
    }

    #[test]
    fn test_error_invalid_groups() {
        let input = BoundedTensor::from_exact(vec![1.0; 16], vec![1, 1, 4, 4]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 18], vec![2, 1, 3, 3]);

        // groups=3 doesn't divide input channels=1
        let config = Conv2dConfig::new().groups(3);
        let result = conv2d_with_config(&input, &kernel, config, Precision::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_very_large_padding() {
        // Padding larger than kernel should still work
        let input = BoundedTensor::from_exact(vec![1.0; 4], vec![1, 1, 2, 2]);
        let kernel = BoundedTensor::from_exact(vec![1.0], vec![1, 1, 1, 1]);

        let result = conv2d(&input, &kernel, (1, 1), (2, 2), Precision::F32).unwrap();

        // Output: 2 + 2*2 - 1 + 1 = 6
        assert_eq!(result.shape(), &vec![1, 1, 6, 6]);
    }

    #[test]
    fn test_multi_channel_group_conv() {
        // Groups=2 with 4 input channels and 4 output channels
        let input = BoundedTensor::from_exact(vec![1.0; 64], vec![1, 4, 4, 4]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 32], vec![4, 2, 2, 2]);

        let result = group_conv2d(&input, &kernel, 2, (1, 1), (0, 0), Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 4, 3, 3]);
    }

    // ========================================================================
    // MEMORY OPTIMIZATION TESTS
    // ========================================================================

    #[test]
    fn test_im2col_basic() {
        // Test im2col transformation produces correct shape
        let input = BoundedTensor::from_exact(
            (1..=16).map(|x| x as f64).collect(),
            vec![1, 1, 4, 4],
        );

        let col = im2col(&input, (3, 3), (1, 1), (0, 0), (1, 1));

        // out_h = out_w = 2, in_ch * kH * kW = 1 * 3 * 3 = 9
        assert_eq!(col.shape(), &vec![1, 4, 9]);
    }

    #[test]
    fn test_im2col_with_padding() {
        let input = BoundedTensor::from_exact(vec![1.0; 16], vec![1, 1, 4, 4]);

        let col = im2col(&input, (3, 3), (1, 1), (1, 1), (1, 1));

        // With padding, out_h = out_w = 4, so col has 16 rows
        assert_eq!(col.shape(), &vec![1, 16, 9]);
    }

    #[test]
    fn test_reshape_kernel_for_im2col() {
        let kernel = BoundedTensor::from_exact(vec![1.0; 18], vec![2, 1, 3, 3]);

        let reshaped = reshape_kernel_for_im2col(&kernel);

        // (out_ch, in_ch * kH * kW) = (2, 9)
        assert_eq!(reshaped.shape(), &vec![2, 9]);
    }

    #[test]
    fn test_conv2d_im2col_matches_direct() {
        // Verify im2col produces same results as direct convolution
        let input = BoundedTensor::from_exact(
            (1..=16).map(|x| x as f64).collect(),
            vec![1, 1, 4, 4],
        );
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let config = Conv2dConfig::new().stride((1, 1)).padding((0, 0));

        let result_direct = conv2d_with_config(&input, &kernel, config, Precision::F32).unwrap();
        let result_im2col = conv2d_im2col(&input, &kernel, config, Precision::F32).unwrap();

        assert_eq!(result_direct.shape(), result_im2col.shape());

        // Values should match closely
        for (d, i) in result_direct.values().iter().zip(result_im2col.values().iter()) {
            assert!(
                (d - i).abs() < 1e-10,
                "Direct {} vs im2col {}", d, i
            );
        }
    }

    #[test]
    fn test_conv2d_tiled_matches_direct() {
        // Verify tiled produces same results as direct convolution
        let input = BoundedTensor::from_exact(vec![1.0; 64], vec![1, 1, 8, 8]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let config = Conv2dConfig::new().stride((1, 1)).padding((1, 1));

        let result_direct = conv2d_with_config(&input, &kernel, config, Precision::F32).unwrap();
        let result_tiled = conv2d_tiled(&input, &kernel, config, 4, Precision::F32).unwrap();

        assert_eq!(result_direct.shape(), result_tiled.shape());

        for (d, t) in result_direct.values().iter().zip(result_tiled.values().iter()) {
            assert!(
                (d - t).abs() < 1e-10,
                "Direct {} vs tiled {}", d, t
            );
        }
    }

    #[test]
    fn test_conv2d_auto_small_input() {
        // Small input should use direct convolution
        let input = BoundedTensor::from_exact(vec![1.0; 16], vec![1, 1, 4, 4]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let config = Conv2dConfig::new();
        let result = conv2d_auto(&input, &kernel, config, Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_conv2d_auto_large_input() {
        // Large input should work correctly regardless of algorithm chosen
        let input = BoundedTensor::from_exact(vec![1.0; 4096], vec![1, 1, 64, 64]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 25], vec![1, 1, 5, 5]);

        let config = Conv2dConfig::new().stride((1, 1)).padding((2, 2));
        let result = conv2d_auto(&input, &kernel, config, Precision::F32).unwrap();

        // Output should maintain spatial size with same padding
        assert_eq!(result.shape(), &vec![1, 1, 64, 64]);
    }

    #[test]
    fn test_im2col_preserves_values() {
        // Verify im2col correctly extracts patches
        let input = BoundedTensor::from_exact(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
        );

        let col = im2col(&input, (2, 2), (1, 1), (0, 0), (1, 1));

        // Should have 1 patch (2x2 kernel, 2x2 input, stride 1, no padding)
        assert_eq!(col.shape(), &vec![1, 1, 4]);

        // Patch should be [1, 2, 3, 4] in row-major order
        let values = col.values();
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 2.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
        assert!((values[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimized_conv_with_batch() {
        // Test batch processing with optimized convolution
        let input = BoundedTensor::from_exact(vec![1.0; 128], vec![2, 1, 8, 8]);
        let kernel = BoundedTensor::from_exact(vec![1.0; 9], vec![1, 1, 3, 3]);

        let config = Conv2dConfig::new();

        let result_direct = conv2d_with_config(&input, &kernel, config, Precision::F32).unwrap();
        let result_auto = conv2d_auto(&input, &kernel, config, Precision::F32).unwrap();

        assert_eq!(result_direct.shape(), &vec![2, 1, 6, 6]);
        assert_eq!(result_auto.shape(), &vec![2, 1, 6, 6]);

        // Verify values match
        for (d, a) in result_direct.values().iter().zip(result_auto.values().iter()) {
            assert!((d - a).abs() < 1e-10);
        }
    }

    #[test]
    fn test_optimized_conv_multi_channel() {
        // Test multi-channel with optimized convolution
        let input = BoundedTensor::from_exact(vec![1.0; 96], vec![1, 3, 4, 8]);
        let kernel = BoundedTensor::from_exact(vec![0.5; 108], vec![4, 3, 3, 3]);

        let config = Conv2dConfig::new().stride((1, 1)).padding((1, 1));

        let result = conv2d_auto(&input, &kernel, config, Precision::F32).unwrap();

        assert_eq!(result.shape(), &vec![1, 4, 4, 8]);
    }
}
