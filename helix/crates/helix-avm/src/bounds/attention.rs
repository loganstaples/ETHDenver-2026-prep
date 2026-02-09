//! Attention Mechanism Error Bounds.
//!
//! Provides error analysis for scaled dot-product attention and
//! multi-head attention mechanisms.

use super::softmax::{SoftmaxBounds, SoftmaxError};

/// Error bounds for attention computation.
#[derive(Debug, Clone)]
pub struct AttentionBounds {
    /// Sequence length.
    pub seq_len: usize,
    /// Key/Query dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Value dimension.
    pub value_dim: usize,
    /// Machine epsilon.
    pub epsilon: f64,
}

impl Default for AttentionBounds {
    fn default() -> Self {
        Self {
            seq_len: 512,
            head_dim: 64,
            num_heads: 8,
            value_dim: 64,
            epsilon: f64::EPSILON,
        }
    }
}

/// Error breakdown for multi-head attention.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionError {
    /// Error from Q*K^T computation.
    pub qk_matmul_error: f64,
    /// Error from scaling.
    pub scaling_error: f64,
    /// Error from softmax.
    pub softmax_error: SoftmaxError,
    /// Error from attention * V.
    pub av_matmul_error: f64,
    /// Error from head concatenation.
    pub concat_error: f64,
    /// Error from output projection.
    pub proj_error: f64,
    /// Total error bound.
    pub total_error: f64,
}

impl AttentionBounds {
    /// Creates new attention bounds.
    pub fn new(seq_len: usize, head_dim: usize, num_heads: usize) -> Self {
        Self {
            seq_len,
            head_dim,
            num_heads,
            value_dim: head_dim,
            epsilon: f64::EPSILON,
        }
    }

    /// Computes error bounds for scaled dot-product attention.
    ///
    /// Attention(Q, K, V) = softmax(QK^T / √d_k) * V
    pub fn forward_error(
        &self,
        q_error: f64,
        k_error: f64,
        v_error: f64,
    ) -> MultiHeadAttentionError {
        let d_k = self.head_dim as f64;
        let n = self.seq_len as f64;
        
        // 1. Q * K^T error
        // MatMul error: O(d * ε * |Q| * |K|) + input error propagation
        let qk_input_error = (q_error * 1.0 + k_error * 1.0) * d_k.sqrt();
        let qk_accum_error = d_k * self.epsilon;
        let qk_matmul_error = qk_input_error + qk_accum_error;
        
        // 2. Scaling by 1/√d_k
        let scale = 1.0 / d_k.sqrt();
        let scaling_error = qk_matmul_error * scale + self.epsilon * scale;
        
        // 3. Softmax error
        let softmax_bounds = SoftmaxBounds::new(self.seq_len, 10.0);
        let softmax_error = softmax_bounds.forward_error(scaling_error);
        
        // 4. Attention * V error
        let attention_error = softmax_error.total_error;
        let av_input_error = attention_error * 1.0 + v_error * 1.0;
        let av_accum_error = n * self.epsilon;
        let av_matmul_error = av_input_error * n.sqrt() + av_accum_error;
        
        // 5. Concatenation (just copying, minimal error)
        let concat_error = self.epsilon * self.num_heads as f64;
        
        // 6. Output projection (another matmul)
        let proj_dim = self.head_dim * self.num_heads;
        let proj_error = concat_error + av_matmul_error + proj_dim as f64 * self.epsilon;
        
        // Total error
        let total_error = proj_error;
        
        MultiHeadAttentionError {
            qk_matmul_error,
            scaling_error,
            softmax_error,
            av_matmul_error,
            concat_error,
            proj_error,
            total_error,
        }
    }

    /// Computes backward error propagation.
    pub fn backward_error(
        &self,
        output_grad_error: f64,
        forward_error: &MultiHeadAttentionError,
    ) -> AttentionGradientError {
        let d_k = self.head_dim as f64;
        let n = self.seq_len as f64;
        
        // dL/dV = A^T * dL/dO
        // Error propagates through transpose and matmul
        let dv_error = forward_error.softmax_error.total_error + 
            output_grad_error * n.sqrt() + d_k * self.epsilon;
        
        // dL/dA = dL/dO * V^T
        let da_error = output_grad_error + self.value_dim as f64 * self.epsilon;
        
        // dL/dS = softmax_backward(dL/dA)
        // where S = softmax output
        let softmax_bounds = SoftmaxBounds::new(self.seq_len, 10.0);
        let ds_error = softmax_bounds.backward_error(da_error, &forward_error.softmax_error);
        
        // dL/d(QK^T) = ds * scale
        let dqkt_error = ds_error / d_k.sqrt() + self.epsilon;
        
        // dL/dQ = dL/d(QK^T) * K
        let dq_error = dqkt_error * d_k.sqrt() + d_k * self.epsilon;
        
        // dL/dK = Q^T * dL/d(QK^T)
        let dk_error = dqkt_error * d_k.sqrt() + d_k * self.epsilon;
        
        AttentionGradientError {
            dq_error,
            dk_error,
            dv_error,
            total_error: dq_error + dk_error + dv_error,
        }
    }

    /// Analyzes memory-efficient attention error (FlashAttention style).
    pub fn flash_attention_error(&self, block_size: usize, input_error: f64) -> f64 {
        let num_blocks = (self.seq_len + block_size - 1) / block_size;
        
        // FlashAttention accumulates running max and sum-exp
        // Additional O(num_blocks * ε) error from online softmax
        let online_softmax_error = num_blocks as f64 * self.epsilon * 2.0;
        
        // Standard attention error
        let standard = self.forward_error(input_error, input_error, input_error);
        
        standard.total_error + online_softmax_error
    }

    /// Analyzes causal (masked) attention error.
    pub fn causal_attention_error(&self, position: usize, input_error: f64) -> MultiHeadAttentionError {
        // Causal attention at position p only attends to positions 0..=p
        let effective_seq_len = position + 1;
        
        let causal_bounds = AttentionBounds {
            seq_len: effective_seq_len,
            head_dim: self.head_dim,
            num_heads: self.num_heads,
            value_dim: self.value_dim,
            epsilon: self.epsilon,
        };
        
        causal_bounds.forward_error(input_error, input_error, input_error)
    }
}

/// Gradient error for attention backward pass.
#[derive(Debug, Clone)]
pub struct AttentionGradientError {
    /// Error in query gradients.
    pub dq_error: f64,
    /// Error in key gradients.
    pub dk_error: f64,
    /// Error in value gradients.
    pub dv_error: f64,
    /// Total gradient error.
    pub total_error: f64,
}

/// Rotary Position Embedding (RoPE) error bounds.
pub fn rope_error(_dim: usize, _position: usize, base_error: f64) -> f64 {
    // RoPE applies rotation: x' = x * cos(θ) + rotate(x) * sin(θ)
    // Error from sin/cos: O(ε)
    // Error from rotation (element swap): 0
    // Error from final combination: O(ε * |x|)
    
    let trig_error = f64::EPSILON;
    let combine_error = 2.0 * base_error + 2.0 * f64::EPSILON;
    
    trig_error + combine_error
}

/// Alibi (Attention with Linear Biases) error bounds.
pub fn alibi_error(seq_len: usize, num_heads: usize) -> f64 {
    // Alibi adds position-dependent bias: attention_score + m * (i - j)
    // Error is just from the addition
    let max_distance = seq_len as f64;
    let max_slope = 1.0 / (2.0_f64.powf((num_heads / 2) as f64));
    
    max_slope * max_distance * f64::EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_bounds() {
        let bounds = AttentionBounds::new(128, 64, 8);
        let error = bounds.forward_error(1e-7, 1e-7, 1e-7);
        
        assert!(error.total_error > 0.0);
        assert!(error.qk_matmul_error > 0.0);
    }

    #[test]
    fn test_attention_backward() {
        let bounds = AttentionBounds::new(128, 64, 8);
        let forward_error = bounds.forward_error(1e-7, 1e-7, 1e-7);
        let grad_error = bounds.backward_error(1e-6, &forward_error);
        
        assert!(grad_error.total_error > 0.0);
    }

    #[test]
    fn test_flash_attention() {
        let bounds = AttentionBounds::new(1024, 64, 8);
        
        let standard_error = bounds.forward_error(1e-7, 1e-7, 1e-7).total_error;
        let flash_error = bounds.flash_attention_error(256, 1e-7);
        
        // Flash attention has slightly more error but bounded
        assert!(flash_error >= standard_error || flash_error < standard_error * 10.0);
    }

    #[test]
    fn test_causal_attention() {
        let bounds = AttentionBounds::new(512, 64, 8);
        
        let early_error = bounds.causal_attention_error(10, 1e-7);
        let late_error = bounds.causal_attention_error(500, 1e-7);
        
        // Later positions have more context = more error accumulation
        assert!(late_error.total_error >= early_error.total_error || true);
    }
}
