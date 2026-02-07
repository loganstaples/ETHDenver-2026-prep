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

use rand::Rng;

use crate::beaver::pool::BeaverPool;
use crate::error::MPCResult;
use crate::field::Fr;
use crate::nn::attention::{SecureAttention, SecureAttentionConfig};
use crate::nn::linear::SecureLinear;
use crate::protocols::activation::{ActivationType, SecureActivation};
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
    pub ln1_gamma: Vec<Vec<Fr>>,
    /// Attention Q projection [d_model * d_model] per party.
    pub wq: Vec<Vec<Fr>>,
    /// Attention K projection.
    pub wk: Vec<Vec<Fr>>,
    /// Attention V projection.
    pub wv: Vec<Vec<Fr>>,
    /// Attention output projection.
    pub wo: Vec<Vec<Fr>>,
    /// LayerNorm 2 gamma [d_model] per party.
    pub ln2_gamma: Vec<Vec<Fr>>,
    /// MLP up projection [d_model * d_ff] per party.
    pub mlp_up: Vec<Vec<Fr>>,
    /// MLP down projection [d_ff * d_model] per party.
    pub mlp_down: Vec<Vec<Fr>>,
}

/// Secure transformer block.
pub struct SecureTransformerBlock;

impl SecureTransformerBlock {
    /// Forward pass through one transformer block.
    ///
    /// `x_shares[party][seq_len * d_model]`: input activations
    /// Returns output activations with same shape.
    pub fn forward(
        x_shares: &[Vec<Fr>],
        weights: &TransformerBlockShares,
        config: &SecureTransformerConfig,
        seq_len: usize,
        pools: &mut [BeaverPool],
        rng: &mut impl Rng,
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = x_shares.len();
        let d = config.d_model;

        // Step 1: Pre-attention LayerNorm / RMSNorm.
        let normed1 = Self::apply_norm_batched(
            x_shares,
            &weights.ln1_gamma,
            config,
            seq_len,
            rng,
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
            rng,
        )?;

        // Step 3: Residual connection: x + attn_out.
        let residual1: Vec<Vec<Fr>> = (0..num_parties)
            .map(|i| {
                x_shares[i]
                    .iter()
                    .zip(&attn_out[i])
                    .map(|(a, b)| Fr::add(a, b))
                    .collect()
            })
            .collect();

        // Step 4: Pre-FFN LayerNorm / RMSNorm.
        let normed2 = Self::apply_norm_batched(
            &residual1,
            &weights.ln2_gamma,
            config,
            seq_len,
            rng,
        );

        // Step 5: Feed-forward network.
        let ffn_out = Self::feed_forward(
            &normed2,
            &weights.mlp_up,
            &weights.mlp_down,
            config,
            seq_len,
            pools,
            rng,
        )?;

        // Step 6: Residual connection.
        let output: Vec<Vec<Fr>> = (0..num_parties)
            .map(|i| {
                residual1[i]
                    .iter()
                    .zip(&ffn_out[i])
                    .map(|(a, b)| Fr::add(a, b))
                    .collect()
            })
            .collect();

        Ok(output)
    }

    /// Applies normalization to each position in the sequence.
    fn apply_norm_batched(
        x_shares: &[Vec<Fr>],
        gamma_shares: &[Vec<Fr>],
        config: &SecureTransformerConfig,
        seq_len: usize,
        rng: &mut impl Rng,
    ) -> Vec<Vec<Fr>> {
        let num_parties = x_shares.len();
        let d = config.d_model;

        let mut result: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; seq_len * d]; num_parties];

        for pos in 0..seq_len {
            // Extract this position's slice for each party.
            let pos_shares: Vec<Vec<Fr>> = x_shares
                .iter()
                .map(|s| s[pos * d..(pos + 1) * d].to_vec())
                .collect();

            let normed = if config.use_rms_norm {
                // For RMSNorm, gamma is reconstructed to public.
                let gamma_vals: Vec<f64> = {
                    let mut g = vec![Fr::ZERO; d];
                    for s in gamma_shares {
                        for (i, v) in s.iter().enumerate() {
                            g[i] = Fr::add(&g[i], v);
                        }
                    }
                    g.iter().map(|v| v.to_f64()).collect()
                };
                SecureNormalization::rms_norm(&pos_shares, &gamma_vals, 1e-5, rng)
            } else {
                let gamma_vals: Vec<f64> = {
                    let mut g = vec![Fr::ZERO; d];
                    for s in gamma_shares {
                        for (i, v) in s.iter().enumerate() {
                            g[i] = Fr::add(&g[i], v);
                        }
                    }
                    g.iter().map(|v| v.to_f64()).collect()
                };
                let beta = vec![0.0; d];
                SecureNormalization::layer_norm(&pos_shares, &gamma_vals, &beta, 1e-5, rng)
            };

            // Write back.
            for i in 0..num_parties {
                for (j, v) in normed[i].iter().enumerate() {
                    result[i][pos * d + j] = v.clone();
                }
            }
        }

        result
    }

    /// Feed-forward network: MLP(x) = down(activation(up(x))).
    fn feed_forward(
        x_shares: &[Vec<Fr>],
        up_shares: &[Vec<Fr>],
        down_shares: &[Vec<Fr>],
        config: &SecureTransformerConfig,
        seq_len: usize,
        pools: &mut [BeaverPool],
        rng: &mut impl Rng,
    ) -> MPCResult<Vec<Vec<Fr>>> {
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
            rng,
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
