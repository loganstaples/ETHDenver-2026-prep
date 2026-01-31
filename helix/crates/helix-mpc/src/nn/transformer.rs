//! Secure transformer block on secret-shared weights.
//!
//! A transformer block consists of:
//! 1. LayerNorm (or RMSNorm)
//! 2. Multi-head self-attention
//! 3. Residual connection
//! 4. LayerNorm
//! 5. Feed-forward network (MLP)
//! 6. Residual connection
//!
//! All weight parameters are secret-shared across parties.

use crate::beaver::pool::BeaverPool;
use crate::error::{MPCError, MPCResult};
use crate::nn::attention::{SecureAttention, SecureAttentionConfig};
use crate::nn::linear::SecureLinear;
use crate::protocols::activation::{ActivationType, SecureActivation};
use crate::protocols::arithmetic::SecureArithmetic;
use crate::protocols::normalization::SecureNormalization;

/// Configuration for a secure transformer block.
#[derive(Debug, Clone)]
pub struct SecureTransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub activation: ActivationType,
    pub use_rms_norm: bool,
}

impl SecureTransformerConfig {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self {
            d_model,
            num_heads,
            d_ff: d_model * 4,
            activation: ActivationType::GELU,
            use_rms_norm: true,
        }
    }
}

/// Secret-shared weights for one transformer block.
#[derive(Debug, Clone)]
pub struct TransformerBlockShares {
    /// LayerNorm 1 gamma [d_model] per party.
    pub ln1_gamma: Vec<Vec<f64>>,
    /// Attention Q projection [d_model * d_model] per party.
    pub wq: Vec<Vec<f64>>,
    /// Attention K projection.
    pub wk: Vec<Vec<f64>>,
    /// Attention V projection.
    pub wv: Vec<Vec<f64>>,
    /// Attention output projection.
    pub wo: Vec<Vec<f64>>,
    /// LayerNorm 2 gamma [d_model] per party.
    pub ln2_gamma: Vec<Vec<f64>>,
    /// MLP up projection [d_model * d_ff] per party.
    pub mlp_up: Vec<Vec<f64>>,
    /// MLP down projection [d_ff * d_model] per party.
    pub mlp_down: Vec<Vec<f64>>,
}

/// Secure transformer block.
pub struct SecureTransformerBlock;

impl SecureTransformerBlock {
    /// Forward pass through one transformer block.
    ///
    /// `x_shares[party][seq_len * d_model]`: input activations
    /// Returns output activations with same shape.
    pub fn forward(
        x_shares: &[Vec<f64>],
        weights: &TransformerBlockShares,
        config: &SecureTransformerConfig,
        seq_len: usize,
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
        let num_parties = x_shares.len();
        let d = config.d_model;

        // Step 1: Pre-attention LayerNorm / RMSNorm.
        let normed1 = Self::apply_norm_batched(
            x_shares,
            &weights.ln1_gamma,
            config,
            seq_len,
        );

        // Step 2: Multi-head attention.
        let attn_config = SecureAttentionConfig::new(d, config.num_heads);
        let attn_out = SecureAttention::forward(
            &normed1,
            &weights.wq,
            &weights.wk,
            &weights.wv,
            &weights.wo,
            &attn_config,
            seq_len,
            pools,
        )?;

        // Step 3: Residual connection: x + attn_out.
        let residual1: Vec<Vec<f64>> = (0..num_parties)
            .map(|i| {
                x_shares[i]
                    .iter()
                    .zip(&attn_out[i])
                    .map(|(a, b)| a + b)
                    .collect()
            })
            .collect();

        // Step 4: Pre-FFN LayerNorm / RMSNorm.
        let normed2 = Self::apply_norm_batched(
            &residual1,
            &weights.ln2_gamma,
            config,
            seq_len,
        );

        // Step 5: Feed-forward network.
        let ffn_out = Self::feed_forward(
            &normed2,
            &weights.mlp_up,
            &weights.mlp_down,
            config,
            seq_len,
            pools,
        )?;

        // Step 6: Residual connection.
        let output: Vec<Vec<f64>> = (0..num_parties)
            .map(|i| {
                residual1[i]
                    .iter()
                    .zip(&ffn_out[i])
                    .map(|(a, b)| a + b)
                    .collect()
            })
            .collect();

        Ok(output)
    }

    /// Applies normalization to each position in the sequence.
    fn apply_norm_batched(
        x_shares: &[Vec<f64>],
        gamma_shares: &[Vec<f64>],
        config: &SecureTransformerConfig,
        seq_len: usize,
    ) -> Vec<Vec<f64>> {
        let num_parties = x_shares.len();
        let d = config.d_model;

        let mut result: Vec<Vec<f64>> = vec![vec![0.0; seq_len * d]; num_parties];

        for pos in 0..seq_len {
            // Extract this position's slice for each party.
            let pos_shares: Vec<Vec<f64>> = x_shares
                .iter()
                .map(|s| s[pos * d..(pos + 1) * d].to_vec())
                .collect();

            let normed = if config.use_rms_norm {
                // For RMSNorm, gamma is public (or we use shared params).
                let gamma_vals: Vec<f64> = {
                    let mut g = vec![0.0; d];
                    for s in gamma_shares {
                        for (i, v) in s.iter().enumerate() {
                            g[i] += v;
                        }
                    }
                    g
                };
                SecureNormalization::rms_norm(&pos_shares, &gamma_vals, 1e-5)
            } else {
                let gamma_vals: Vec<f64> = {
                    let mut g = vec![0.0; d];
                    for s in gamma_shares {
                        for (i, v) in s.iter().enumerate() {
                            g[i] += v;
                        }
                    }
                    g
                };
                let beta = vec![0.0; d];
                SecureNormalization::layer_norm(&pos_shares, &gamma_vals, &beta, 1e-5)
            };

            // Write back.
            for i in 0..num_parties {
                result[i][pos * d..(pos + 1) * d].copy_from_slice(&normed[i]);
            }
        }

        result
    }

    /// Feed-forward network: MLP(x) = down(activation(up(x))).
    fn feed_forward(
        x_shares: &[Vec<f64>],
        up_shares: &[Vec<f64>],
        down_shares: &[Vec<f64>],
        config: &SecureTransformerConfig,
        seq_len: usize,
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
        let d = config.d_model;
        let d_ff = config.d_ff;

        // Up projection: [seq_len x d_model] @ [d_model x d_ff]
        let up_out = SecureLinear::forward_shared(
            x_shares,
            up_shares,
            None,
            seq_len,
            d,
            d_ff,
            pools,
        )?;

        // Activation (reconstruct-compute-reshare).
        let activated = SecureActivation::apply_reconstruct_reshare(
            &up_out,
            config.activation,
        );

        // Down projection: [seq_len x d_ff] @ [d_ff x d_model]
        let down_out = SecureLinear::forward_shared(
            &activated,
            down_shares,
            None,
            seq_len,
            d_ff,
            d,
            pools,
        )?;

        Ok(down_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config() {
        let config = SecureTransformerConfig::new(64, 4);
        assert_eq!(config.d_ff, 256);
        assert_eq!(config.num_heads, 4);
    }
}
