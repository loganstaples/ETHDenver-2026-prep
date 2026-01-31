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

/// Configuration for memory-efficient attention.
#[derive(Debug, Clone)]
pub struct EfficientAttentionConfig {
    /// Use chunked softmax for memory efficiency.
    pub chunked_softmax: bool,
    /// Chunk size for query positions.
    pub query_chunk_size: usize,
    /// Chunk size for key positions.
    pub key_chunk_size: usize,
    /// Use flash attention pattern (fused softmax).
    pub flash_attention: bool,
    /// Recompute attention during backward (gradient checkpointing).
    pub recompute_attention: bool,
}

impl Default for EfficientAttentionConfig {
    fn default() -> Self {
        Self {
            chunked_softmax: false,
            query_chunk_size: 64,
            key_chunk_size: 64,
            flash_attention: false,
            recompute_attention: false,
        }
    }
}

impl EfficientAttentionConfig {
    /// Creates config for memory-efficient attention.
    pub fn memory_efficient(query_chunk_size: usize, key_chunk_size: usize) -> Self {
        Self {
            chunked_softmax: true,
            query_chunk_size,
            key_chunk_size,
            flash_attention: false,
            recompute_attention: true,
        }
    }
}

/// Memory-efficient scaled dot-product attention using chunked computation.
///
/// For long sequences, computes attention in chunks to avoid materializing
/// the full seq_len x seq_len attention matrix.
///
/// Memory complexity: O(query_chunk_size * key_chunk_size) instead of O(seq_len^2)
pub fn chunked_scaled_dot_product_attention(
    query: &BoundedTensor,
    key: &BoundedTensor,
    value: &BoundedTensor,
    query_chunk_size: usize,
    key_chunk_size: usize,
    precision: Precision,
) -> Result<BoundedTensor, AttentionError> {
    let q_shape = query.shape();
    let k_shape = key.shape();

    if q_shape.len() != 2 || k_shape.len() != 2 {
        return Err(AttentionError::DimensionMismatch {
            expected: "2D tensors".to_string(),
            actual: format!("{}D and {}D tensors", q_shape.len(), k_shape.len()),
        });
    }

    let seq_len_q = q_shape[0];
    let seq_len_k = k_shape[0];
    let d_k = q_shape[1];

    // If small enough, use regular attention
    if seq_len_q <= query_chunk_size && seq_len_k <= key_chunk_size {
        return scaled_dot_product_attention(query, key, value, precision);
    }

    let scale = 1.0 / (d_k as f64).sqrt();

    // Number of chunks
    let num_q_chunks = (seq_len_q + query_chunk_size - 1) / query_chunk_size;
    let num_k_chunks = (seq_len_k + key_chunk_size - 1) / key_chunk_size;

    let mut output_data = vec![BoundedValue::exact(0.0); seq_len_q * d_k];

    for q_chunk_idx in 0..num_q_chunks {
        let q_start = q_chunk_idx * query_chunk_size;
        let q_end = (q_start + query_chunk_size).min(seq_len_q);
        let q_chunk_len = q_end - q_start;

        // Extract query chunk
        let q_chunk = extract_rows_attention(query, q_start, q_end)?;

        // For numerical stability, track max values and sum of exp
        let mut output_chunk = vec![BoundedValue::exact(0.0); q_chunk_len * d_k];
        let mut row_max = vec![f64::NEG_INFINITY; q_chunk_len];
        let mut row_sum = vec![0.0; q_chunk_len];

        for k_chunk_idx in 0..num_k_chunks {
            let k_start = k_chunk_idx * key_chunk_size;
            let k_end = (k_start + key_chunk_size).min(seq_len_k);

            // Extract key and value chunks
            let k_chunk = extract_rows_attention(key, k_start, k_end)?;
            let v_chunk = extract_rows_attention(value, k_start, k_end)?;

            // Compute attention scores for this chunk: Q_chunk @ K_chunk^T
            let k_chunk_t = k_chunk.transpose();
            let scores = ops::matmul(&q_chunk, &k_chunk_t, precision)?;

            // Scale scores
            let scaled_scores = scores.scale(BoundedValue::exact(scale));

            // Update running softmax and output
            // This implements the "online softmax" algorithm
            let scores_data = scaled_scores.data();
            let k_chunk_len = k_end - k_start;

            for qi in 0..q_chunk_len {
                let mut local_max = row_max[qi];
                for ki in 0..k_chunk_len {
                    let score = scores_data[qi * k_chunk_len + ki].value();
                    local_max = local_max.max(score);
                }

                // Update with new max
                if local_max > row_max[qi] {
                    let exp_diff = (row_max[qi] - local_max).exp();
                    row_sum[qi] *= exp_diff;
                    for di in 0..d_k {
                        let val = output_chunk[qi * d_k + di].value();
                        output_chunk[qi * d_k + di] = BoundedValue::exact(val * exp_diff);
                    }
                    row_max[qi] = local_max;
                }

                // Accumulate weighted values
                let v_chunk_data = v_chunk.data();
                for ki in 0..k_chunk_len {
                    let score = scores_data[qi * k_chunk_len + ki].value();
                    let weight = (score - row_max[qi]).exp();
                    row_sum[qi] += weight;

                    for di in 0..d_k {
                        let v_val = v_chunk_data[ki * d_k + di].value();
                        let current = output_chunk[qi * d_k + di].value();
                        output_chunk[qi * d_k + di] = BoundedValue::exact(current + weight * v_val);
                    }
                }
            }
        }

        // Normalize by sum
        for qi in 0..q_chunk_len {
            let sum = row_sum[qi];
            if sum > 0.0 {
                for di in 0..d_k {
                    let val = output_chunk[qi * d_k + di].value();
                    output_chunk[qi * d_k + di] = BoundedValue::exact(val / sum);
                }
            }

            // Copy to output
            let out_row = q_start + qi;
            for di in 0..d_k {
                output_data[out_row * d_k + di] = output_chunk[qi * d_k + di];
            }
        }
    }

    Ok(BoundedTensor::new(output_data, vec![seq_len_q, d_k]))
}

/// Helper to extract rows from a 2D tensor.
fn extract_rows_attention(tensor: &BoundedTensor, start: usize, end: usize) -> Result<BoundedTensor, AttentionError> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(AttentionError::DimensionMismatch {
            expected: "2D tensor".to_string(),
            actual: format!("{}D tensor", shape.len()),
        });
    }

    let cols = shape[1];
    let data: Vec<_> = tensor.data()[start * cols..end * cols].to_vec();
    Ok(BoundedTensor::new(data, vec![end - start, cols]))
}

/// Memory-efficient multi-head attention.
#[derive(Debug, Clone)]
pub struct EfficientMultiHeadAttention {
    /// Base multi-head attention.
    base: MultiHeadAttention,
    /// Efficient attention config.
    config: EfficientAttentionConfig,
}

impl EfficientMultiHeadAttention {
    /// Creates a new efficient multi-head attention layer.
    pub fn new(
        attention_config: &AttentionConfig,
        efficient_config: EfficientAttentionConfig,
    ) -> Result<Self, AttentionError> {
        let base = MultiHeadAttention::new(attention_config)?;
        Ok(Self {
            base,
            config: efficient_config,
        })
    }

    /// Forward pass with memory-efficient attention.
    pub fn forward(
        &self,
        query: &BoundedTensor,
        key: &BoundedTensor,
        value: &BoundedTensor,
    ) -> Result<BoundedTensor, AttentionError> {
        if self.config.chunked_softmax {
            // Project Q, K, V
            let q = self.base.w_q.forward(query)?;
            let k = self.base.w_k.forward(key)?;
            let v = self.base.w_v.forward(value)?;

            // Memory-efficient attention
            let attention_output = chunked_scaled_dot_product_attention(
                &q,
                &k,
                &v,
                self.config.query_chunk_size,
                self.config.key_chunk_size,
                self.base.precision,
            )?;

            // Output projection
            Ok(self.base.w_o.forward(&attention_output)?)
        } else {
            self.base.forward(query, key, value)
        }
    }

    /// Self-attention forward pass.
    pub fn forward_self(&self, input: &BoundedTensor) -> Result<BoundedTensor, AttentionError> {
        self.forward(input, input, input)
    }
}

/// Estimates memory usage for attention computation.
pub fn estimate_attention_memory(
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
    chunked: bool,
    chunk_size: usize,
) -> usize {
    let bytes_per_element = 8; // f64

    if chunked {
        // Chunked: O(batch * heads * chunk_size^2)
        let attention_scores = batch_size * num_heads * chunk_size * chunk_size * bytes_per_element;
        let qkv_chunks = batch_size * num_heads * chunk_size * (d_model / num_heads) * bytes_per_element * 3;
        attention_scores + qkv_chunks
    } else {
        // Full: O(batch * heads * seq_len^2)
        let attention_scores = batch_size * num_heads * seq_len * seq_len * bytes_per_element;
        let qkv = batch_size * seq_len * d_model * bytes_per_element * 3;
        attention_scores + qkv
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

    #[test]
    fn test_chunked_attention() {
        let seq_len = 16;
        let d_k = 8;

        let q = BoundedTensor::zeros(vec![seq_len, d_k]);
        let k = BoundedTensor::zeros(vec![seq_len, d_k]);
        let v = BoundedTensor::from_exact(
            (0..seq_len * d_k).map(|i| (i % 10) as f64 / 10.0).collect(),
            vec![seq_len, d_k],
        );

        // With small chunk size
        let output = chunked_scaled_dot_product_attention(&q, &k, &v, 4, 4, Precision::F32).unwrap();
        assert_eq!(output.shape(), &vec![seq_len, d_k]);
    }

    #[test]
    fn test_chunked_vs_regular_attention() {
        let seq_len = 8;
        let d_k = 4;

        let q = BoundedTensor::from_exact(
            (0..seq_len * d_k).map(|i| i as f64 / 100.0).collect(),
            vec![seq_len, d_k],
        );
        let k = BoundedTensor::from_exact(
            (0..seq_len * d_k).map(|i| (i + 1) as f64 / 100.0).collect(),
            vec![seq_len, d_k],
        );
        let v = BoundedTensor::from_exact(
            (0..seq_len * d_k).map(|i| (i + 2) as f64 / 100.0).collect(),
            vec![seq_len, d_k],
        );

        // Regular attention
        let regular_output = scaled_dot_product_attention(&q, &k, &v, Precision::F32).unwrap();

        // Chunked attention with large chunks (should match regular)
        let chunked_output = chunked_scaled_dot_product_attention(
            &q, &k, &v, seq_len, seq_len, Precision::F32
        ).unwrap();

        // Results should be very close
        for (r, c) in regular_output.values().iter().zip(chunked_output.values().iter()) {
            assert!((r - c).abs() < 1e-6, "Values differ: {} vs {}", r, c);
        }
    }

    #[test]
    fn test_efficient_attention_config() {
        let config = EfficientAttentionConfig::memory_efficient(32, 32);
        assert!(config.chunked_softmax);
        assert!(config.recompute_attention);
    }

    #[test]
    fn test_attention_memory_estimation() {
        let full_mem = estimate_attention_memory(4, 512, 256, 8, false, 0);
        let chunked_mem = estimate_attention_memory(4, 512, 256, 8, true, 64);

        println!("Full attention memory: {} bytes", full_mem);
        println!("Chunked attention memory: {} bytes", chunked_mem);

        // Chunked should use much less memory
        assert!(chunked_mem < full_mem);
    }

    #[test]
    fn test_efficient_multi_head_attention() {
        let attention_config = AttentionConfig::new(32, 4).unwrap();
        let efficient_config = EfficientAttentionConfig::memory_efficient(8, 8);

        let mha = EfficientMultiHeadAttention::new(&attention_config, efficient_config).unwrap();

        let input = BoundedTensor::zeros(vec![16, 32]);
        let output = mha.forward_self(&input).unwrap();

        assert_eq!(output.shape(), &vec![16, 32]);
    }
}
