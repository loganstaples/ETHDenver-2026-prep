//! Secure multi-head attention on secret-shared weights.
//!
//! Computes Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! The Q, K, V, O projection weights are secret-shared. The attention
//! computation involves:
//! 1. Linear projections (secure matmul with shared weights)
//! 2. Scaled dot-product attention (matmul + softmax + matmul)
//! 3. Output projection (secure matmul with shared weights)
//!
//! The softmax step requires reconstruction (reveals attention patterns
//! but not weights), consistent with the MPC-ML security model.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::beaver::pool::BeaverPool;
use crate::error::MPCResult;
use crate::field::Fr;
use crate::nn::linear::SecureLinear;
use crate::protocols::matmul::SecureMatmul;
use crate::protocols::normalization::SecureNormalization;

/// Configuration for secure multi-head attention.
#[derive(Debug, Clone)]
pub struct SecureAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl SecureAttentionConfig {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0);
        Self {
            d_model,
            num_heads,
            head_dim: d_model / num_heads,
        }
    }
}

/// Secure multi-head attention layer.
pub struct SecureAttention;

impl SecureAttention {
    /// Forward pass for multi-head attention.
    ///
    /// `x_shares`: party shares of input [seq_len * d_model]
    /// `wq_shares`, `wk_shares`, `wv_shares`, `wo_shares`: party shares of
    ///   projection weights, each [d_model * d_model]
    /// `config`: attention configuration
    ///
    /// Returns party shares of output [seq_len * d_model].
    pub fn forward(
        x_shares: &[Vec<Fr>],
        wq_shares: &[Vec<Fr>],
        wk_shares: &[Vec<Fr>],
        wv_shares: &[Vec<Fr>],
        wo_shares: &[Vec<Fr>],
        config: &SecureAttentionConfig,
        seq_len: usize,
        pools: &mut [BeaverPool],
        rng: &mut impl Rng,
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = x_shares.len();
        let d = config.d_model;
        let h = config.num_heads;
        let dk = config.head_dim;

        // Step 1: Project to Q, K, V using shared weights.
        // Q = x @ Wq^T, K = x @ Wk^T, V = x @ Wv^T
        let q_shares = SecureLinear::forward_shared(
            x_shares, wq_shares, None, seq_len, d, d, pools,
        )?;
        let k_shares = SecureLinear::forward_shared(
            x_shares, wk_shares, None, seq_len, d, d, pools,
        )?;
        let v_shares = SecureLinear::forward_shared(
            x_shares, wv_shares, None, seq_len, d, d, pools,
        )?;

        // Step 2: Split into heads and compute attention per head.
        // Generate a seed from the caller's rng for reproducible per-head RNGs.
        let base_seed: u64 = rng.gen();

        // Extract all head slices upfront.
        let head_data: Vec<(Vec<Vec<Fr>>, Vec<Vec<Fr>>, Vec<Vec<Fr>>)> = (0..h)
            .map(|head| {
                let q_head = extract_head_shares(&q_shares, seq_len, d, head, dk);
                let k_head = extract_head_shares(&k_shares, seq_len, d, head, dk);
                let v_head = extract_head_shares(&v_shares, seq_len, d, head, dk);
                (q_head, k_head, v_head)
            })
            .collect();

        // Split pools across heads so each gets its own mutable pool slice.
        let mut sub_pools: Vec<Vec<BeaverPool>> = pools
            .iter_mut()
            .map(|pool| pool.split(h))
            .collect();

        // Transpose: sub_pools[party][head] -> per_head_pools[head][party]
        let per_head_pools: Vec<Vec<BeaverPool>> = (0..h)
            .map(|head| {
                sub_pools
                    .iter_mut()
                    .map(|party_subs| {
                        // Take from the end to avoid index shifting
                        std::mem::replace(
                            &mut party_subs[head],
                            BeaverPool::new(0, num_parties, 1),
                        )
                    })
                    .collect()
            })
            .collect();

        // Process heads in parallel using rayon when feature is enabled.
        #[cfg(feature = "parallel")]
        let head_results: Vec<MPCResult<Vec<Vec<Fr>>>> = {
            use rayon::prelude::*;
            per_head_pools
                .into_par_iter()
                .enumerate()
                .map(|(head_idx, mut head_pools)| {
                    let mut head_rng = ChaCha20Rng::seed_from_u64(base_seed.wrapping_add(head_idx as u64));
                    let (ref q_head, ref k_head, ref v_head) = head_data[head_idx];
                    Self::scaled_dot_product_attention(
                        q_head,
                        k_head,
                        v_head,
                        seq_len,
                        dk,
                        &mut head_pools,
                        &mut head_rng,
                    )
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let head_results: Vec<MPCResult<Vec<Vec<Fr>>>> = {
            per_head_pools
                .into_iter()
                .enumerate()
                .map(|(head_idx, mut head_pools)| {
                    let mut head_rng = ChaCha20Rng::seed_from_u64(base_seed.wrapping_add(head_idx as u64));
                    let (ref q_head, ref k_head, ref v_head) = head_data[head_idx];
                    Self::scaled_dot_product_attention(
                        q_head,
                        k_head,
                        v_head,
                        seq_len,
                        dk,
                        &mut head_pools,
                        &mut head_rng,
                    )
                })
                .collect()
        };

        // Collect results, propagating any errors.
        let head_outputs: Vec<Vec<Vec<Fr>>> = head_results
            .into_iter()
            .collect::<MPCResult<Vec<_>>>()?;

        // Step 3: Concatenate heads.
        let concat_shares = concat_head_shares(&head_outputs, seq_len, h, dk);

        // Step 4: Output projection: out = concat @ Wo^T.
        let output = SecureLinear::forward_shared(
            &concat_shares, wo_shares, None, seq_len, d, d, pools,
        )?;

        Ok(output)
    }

    /// Scaled dot-product attention for a single head.
    ///
    /// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    ///
    /// Q, K, V are each [seq_len x head_dim].
    fn scaled_dot_product_attention(
        q_shares: &[Vec<Fr>],
        k_shares: &[Vec<Fr>],
        v_shares: &[Vec<Fr>],
        seq_len: usize,
        head_dim: usize,
        pools: &mut [BeaverPool],
        rng: &mut impl Rng,
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = q_shares.len();
        let scale = Fr::from_f64(1.0 / (head_dim as f64).sqrt());

        // Step A: Compute [Q @ K^T] — results in [seq_len x seq_len]
        // K^T is [head_dim x seq_len]
        let kt_shares: Vec<Vec<Fr>> = k_shares
            .iter()
            .map(|k| transpose_flat(k, seq_len, head_dim))
            .collect();

        let scores_shares = SecureMatmul::simulate_matmul_from_pool(
            q_shares,
            &kt_shares,
            pools,
            seq_len,
            head_dim,
            seq_len,
        )?;

        // Step B: Scale by 1/sqrt(d_k) — local operation
        // Use mpc_scale for exact linear fixed-point arithmetic on shares
        let scaled: Vec<Vec<Fr>> = scores_shares
            .iter()
            .map(|s| s.iter().map(|v| v.mpc_scale(&scale)).collect())
            .collect();

        // Step C: Softmax — reconstruct, compute, reshare
        // Apply softmax to each row (each query position).
        let attn_rows: Vec<Vec<Vec<Fr>>> = (0..num_parties)
            .map(|i| {
                (0..seq_len)
                    .map(|row| {
                        scaled[i][row * seq_len..(row + 1) * seq_len].to_vec()
                    })
                    .collect()
            })
            .collect();

        let softmax_attn = SecureNormalization::batch_softmax(&attn_rows, rng);

        // Flatten back to [seq_len x seq_len].
        let attn_shares: Vec<Vec<Fr>> = softmax_attn
            .iter()
            .map(|party_rows| {
                let mut flat = Vec::with_capacity(seq_len * seq_len);
                for row in party_rows {
                    flat.extend_from_slice(row);
                }
                flat
            })
            .collect();

        // Step D: [attn_weights] @ [V] — [seq_len x seq_len] @ [seq_len x head_dim]
        let output = SecureMatmul::simulate_matmul_from_pool(
            &attn_shares,
            v_shares,
            pools,
            seq_len,
            seq_len,
            head_dim,
        )?;

        Ok(output)
    }
}

/// Extracts one attention head's data from the concatenated representation.
fn extract_head_shares(
    shares: &[Vec<Fr>],
    seq_len: usize,
    d_model: usize,
    head_idx: usize,
    head_dim: usize,
) -> Vec<Vec<Fr>> {
    shares
        .iter()
        .map(|s| {
            let mut head_data = Vec::with_capacity(seq_len * head_dim);
            for row in 0..seq_len {
                let start = row * d_model + head_idx * head_dim;
                for i in 0..head_dim {
                    head_data.push(s[start + i].clone());
                }
            }
            head_data
        })
        .collect()
}

/// Concatenates attention head outputs back into full d_model representation.
fn concat_head_shares(
    heads: &[Vec<Vec<Fr>>],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<Vec<Fr>> {
    let d_model = num_heads * head_dim;
    let num_parties = heads[0].len();

    (0..num_parties)
        .map(|party| {
            let mut concat = vec![Fr::ZERO; seq_len * d_model];
            for head in 0..num_heads {
                for row in 0..seq_len {
                    for d in 0..head_dim {
                        concat[row * d_model + head * head_dim + d] =
                            heads[head][party][row * head_dim + d].clone();
                    }
                }
            }
            concat
        })
        .collect()
}

/// Transpose a flattened [rows x cols] matrix.
fn transpose_flat(data: &[Fr], rows: usize, cols: usize) -> Vec<Fr> {
    let mut result = vec![Fr::ZERO; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j].clone();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_head() {
        // 2 positions, d_model=4, 2 heads, head_dim=2
        // data = [a0 a1 b0 b1 | c0 c1 d0 d1]
        // head 0 should be [a0 a1 | c0 c1]
        // head 1 should be [b0 b1 | d0 d1]
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                .iter()
                .map(|&v| Fr::from_f64(v))
                .collect::<Vec<Fr>>()
        ];
        let head0 = extract_head_shares(&data, 2, 4, 0, 2);
        let expected0: Vec<Fr> = vec![1.0, 2.0, 5.0, 6.0].iter().map(|&v| Fr::from_f64(v)).collect();
        for (a, b) in head0[0].iter().zip(expected0.iter()) {
            assert!(a.ct_eq(b).to_bool());
        }

        let head1 = extract_head_shares(&data, 2, 4, 1, 2);
        let expected1: Vec<Fr> = vec![3.0, 4.0, 7.0, 8.0].iter().map(|&v| Fr::from_f64(v)).collect();
        for (a, b) in head1[0].iter().zip(expected1.iter()) {
            assert!(a.ct_eq(b).to_bool());
        }
    }

    #[test]
    fn test_concat_heads() {
        let head0 = vec![
            vec![1.0, 2.0, 5.0, 6.0].iter().map(|&v| Fr::from_f64(v)).collect::<Vec<Fr>>()
        ]; // party 0's head 0
        let head1 = vec![
            vec![3.0, 4.0, 7.0, 8.0].iter().map(|&v| Fr::from_f64(v)).collect::<Vec<Fr>>()
        ]; // party 0's head 1
        let heads = vec![head0, head1];

        let concat = concat_head_shares(&heads, 2, 2, 2);
        let expected: Vec<Fr> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .map(|&v| Fr::from_f64(v))
            .collect();
        for (a, b) in concat[0].iter().zip(expected.iter()) {
            assert!(a.ct_eq(b).to_bool());
        }
    }

    #[test]
    fn test_attention_config() {
        let config = SecureAttentionConfig::new(64, 4);
        assert_eq!(config.head_dim, 16);
    }
}
