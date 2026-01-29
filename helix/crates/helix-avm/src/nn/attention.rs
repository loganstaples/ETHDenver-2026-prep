//! Multi-head attention implementation with error bound tracking.
//!
//! Implements scaled dot-product attention and multi-head attention.

use super::linear::{Linear, LinearError};
use crate::ops::{self, softmax, MatMulError, SoftmaxError};
use helix_core::types::{BoundedTensor, BoundedValue, Precision};
use thiserror::Error;

/// Errors from attention operations.
#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Linear layer error: {0}")]
    Linear(#[from] LinearError),

    #[error("Matrix multiplication error: {0}")]
    MatMul(#[from] MatMulError),

    #[error("Softmax error: {0}")]
    Softmax(#[from] SoftmaxError),

    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("Head dimension must divide model dimension evenly: {model_dim} / {num_heads}")]
    HeadDimensionMismatch { model_dim: usize, num_heads: usize },
}

/// Configuration for multi-head attention.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Model dimension (embedding size).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dropout rate (not used in approximate computation).
    pub dropout: f64,
    /// Whether to use bias in projections.
    pub bias: bool,
    /// Precision for computation.
    pub precision: Precision,
}

impl AttentionConfig {
    /// Creates a new attention configuration.
    pub fn new(d_model: usize, num_heads: usize) -> Result<Self, AttentionError> {
        if d_model % num_heads != 0 {
            return Err(AttentionError::HeadDimensionMismatch {
                model_dim: d_model,
                num_heads,
            });
        }

        Ok(Self {
            d_model,
            num_heads,
            dropout: 0.0,
            bias: true,
            precision: Precision::F32,
        })
    }

    /// Dimension per head.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
}

/// Scaled dot-product attention.
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
pub fn scaled_dot_product_attention(
    query: &BoundedTensor,
    key: &BoundedTensor,
    value: &BoundedTensor,
    precision: Precision,
) -> Result<BoundedTensor, AttentionError> {
    // Q, K, V shapes: (seq_len, d_k) or (batch, seq_len, d_k)
    let d_k = if query.is_matrix() {
        query.shape()[1]
    } else if query.ndim() == 3 {
        query.shape()[2]
    } else {
        return Err(AttentionError::DimensionMismatch {
            expected: "2D or 3D tensor".to_string(),
            actual: format!("{}D tensor", query.ndim()),
        });
    };

    let scale = 1.0 / (d_k as f64).sqrt();
    let scale_val = BoundedValue::exact(scale);

    // For 2D case: Q @ K^T
    if query.is_matrix() {
        let key_t = key.transpose();
        let scores = ops::matmul(query, &key_t, precision)?;

        // Scale scores
        let scaled_scores = scores.scale(scale_val);

        // Softmax over last dimension (each row)
        let attention_weights = softmax_rows(&scaled_scores, precision)?;

        // Weighted sum: attention @ V
        let output = ops::matmul(&attention_weights, value, precision)?;
        Ok(output)
    } else {
        // 3D case: handle batch dimension
        // For now, process one batch at a time
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];

        let mut results = Vec::new();

        for b in 0..batch_size {
            let q_slice = extract_batch_slice(query, b)?;
            let k_slice = extract_batch_slice(key, b)?;
            let v_slice = extract_batch_slice(value, b)?;

            let output = scaled_dot_product_attention(&q_slice, &k_slice, &v_slice, precision)?;
            results.push(output);
        }

        // Combine batch results
        combine_batch_slices(&results, batch_size, seq_len, d_k)
    }
}

/// Applies softmax to each row of a 2D tensor.
fn softmax_rows(input: &BoundedTensor, precision: Precision) -> Result<BoundedTensor, AttentionError> {
    if !input.is_matrix() {
        return Err(AttentionError::DimensionMismatch {
            expected: "2D tensor".to_string(),
            actual: format!("{}D tensor", input.ndim()),
        });
    }

    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut result = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        // Extract row
        let mut row_data = Vec::with_capacity(cols);
        for j in 0..cols {
            row_data.push(*input.get(&[i, j]).unwrap());
        }
        let row = BoundedTensor::new(row_data, vec![cols]);

        // Apply softmax
        let softmax_row = softmax::softmax(&row, precision)?;

        // Add to result
        for j in 0..cols {
            result.push(*softmax_row.get(&[j]).unwrap());
        }
    }

    Ok(BoundedTensor::new(result, vec![rows, cols]))
}

/// Extracts a 2D slice from a 3D tensor at a given batch index.
fn extract_batch_slice(tensor: &BoundedTensor, batch_idx: usize) -> Result<BoundedTensor, AttentionError> {
    if tensor.ndim() != 3 {
        return Err(AttentionError::DimensionMismatch {
            expected: "3D tensor".to_string(),
            actual: format!("{}D tensor", tensor.ndim()),
        });
    }

    let seq_len = tensor.shape()[1];
    let dim = tensor.shape()[2];
    let mut data = Vec::with_capacity(seq_len * dim);

    for i in 0..seq_len {
        for j in 0..dim {
            // Calculate flat index for 3D tensor
            let flat_idx = batch_idx * seq_len * dim + i * dim + j;
            data.push(tensor.data()[flat_idx]);
        }
    }

    Ok(BoundedTensor::new(data, vec![seq_len, dim]))
}

/// Combines 2D batch slices into a 3D tensor.
fn combine_batch_slices(
    slices: &[BoundedTensor],
    batch_size: usize,
    seq_len: usize,
    dim: usize,
) -> Result<BoundedTensor, AttentionError> {
    let mut data = Vec::with_capacity(batch_size * seq_len * dim);

    for slice in slices {
        data.extend(slice.data().iter().copied());
    }

    Ok(BoundedTensor::new(data, vec![batch_size, seq_len, dim]))
}

/// Multi-head attention layer.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Model dimension.
    d_model: usize,
    /// Q projection.
    w_q: Linear,
    /// K projection.
    w_k: Linear,
    /// V projection.
    w_v: Linear,
    /// Output projection.
    w_o: Linear,
    /// Precision.
    precision: Precision,
}

impl MultiHeadAttention {
    /// Creates a new multi-head attention layer.
    pub fn new(config: &AttentionConfig) -> Result<Self, AttentionError> {
        let d_model = config.d_model;
        let head_dim = config.head_dim();

        // Initialize projections with zeros (would be random in real implementation)
        let w_q = Linear::from_raw(
            vec![0.0; d_model * d_model],
            vec![d_model, d_model],
            if config.bias { Some(vec![0.0; d_model]) } else { None },
            config.precision,
        )?;

        let w_k = Linear::from_raw(
            vec![0.0; d_model * d_model],
            vec![d_model, d_model],
            if config.bias { Some(vec![0.0; d_model]) } else { None },
            config.precision,
        )?;

        let w_v = Linear::from_raw(
            vec![0.0; d_model * d_model],
            vec![d_model, d_model],
            if config.bias { Some(vec![0.0; d_model]) } else { None },
            config.precision,
        )?;

        let w_o = Linear::from_raw(
            vec![0.0; d_model * d_model],
            vec![d_model, d_model],
            if config.bias { Some(vec![0.0; d_model]) } else { None },
            config.precision,
        )?;

        Ok(Self {
            num_heads: config.num_heads,
            head_dim,
            d_model,
            w_q,
            w_k,
            w_v,
            w_o,
            precision: config.precision,
        })
    }

    /// Creates multi-head attention with specified projection weights.
    pub fn with_weights(
        w_q: Linear,
        w_k: Linear,
        w_v: Linear,
        w_o: Linear,
        num_heads: usize,
        precision: Precision,
    ) -> Result<Self, AttentionError> {
        let d_model = w_q.in_features();
        if d_model % num_heads != 0 {
            return Err(AttentionError::HeadDimensionMismatch {
                model_dim: d_model,
                num_heads,
            });
        }

        Ok(Self {
            num_heads,
            head_dim: d_model / num_heads,
            d_model,
            w_q,
            w_k,
            w_v,
            w_o,
            precision,
        })
    }

    /// Forward pass for multi-head attention.
    ///
    /// Input shapes: (seq_len, d_model) for query, key, value
    /// Output shape: (seq_len, d_model)
    pub fn forward(
        &self,
        query: &BoundedTensor,
        key: &BoundedTensor,
        value: &BoundedTensor,
    ) -> Result<BoundedTensor, AttentionError> {
        // Project Q, K, V
        let q = self.w_q.forward(query)?;
        let k = self.w_k.forward(key)?;
        let v = self.w_v.forward(value)?;

        // For simplicity, use single-head attention scaled by head_dim
        // In a full implementation, we'd split into heads and concatenate
        let attention_output = scaled_dot_product_attention(&q, &k, &v, self.precision)?;

        // Output projection
        let output = self.w_o.forward(&attention_output)?;

        Ok(output)
    }

    /// Self-attention forward pass (Q = K = V = input).
    pub fn forward_self(&self, input: &BoundedTensor) -> Result<BoundedTensor, AttentionError> {
        self.forward(input, input, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(64, 8).unwrap();
        assert_eq!(config.head_dim(), 8);
    }

    #[test]
    fn test_attention_config_invalid() {
        let result = AttentionConfig::new(65, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        // Simple 2x2 identity-like attention
        let q = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let output = scaled_dot_product_attention(&q, &k, &v, Precision::F32).unwrap();
        assert_eq!(output.shape(), &vec![2, 2]);
    }

    #[test]
    fn test_multi_head_attention_creation() {
        let config = AttentionConfig::new(64, 8).unwrap();
        let mha = MultiHeadAttention::new(&config).unwrap();
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.head_dim, 8);
    }

    #[test]
    fn test_softmax_rows() {
        let input = BoundedTensor::from_exact(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]);
        let output = softmax_rows(&input, Precision::F32).unwrap();

        // Each row should sum to 1
        let values = output.values();
        let sum_row0 = values[0] + values[1];
        let sum_row1 = values[2] + values[3];
        assert!((sum_row0 - 1.0).abs() < 1e-10);
        assert!((sum_row1 - 1.0).abs() < 1e-10);
    }
}
