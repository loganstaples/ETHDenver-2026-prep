//! Secure embedding layer on secret-shared weights.
//!
//! The embedding table is secret-shared across parties. For a given input
//! token index, each party looks up their share of the embedding vector.
//! Since all parties see the same token indices (input data is public),
//! this is a local operation with no communication.
//!
//! Positional embeddings can be added as a public constant (learned
//! positional embeddings would also need to be shared, but sinusoidal
//! ones are public by construction).

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;

/// Secure embedding layer.
pub struct SecureEmbedding;

impl SecureEmbedding {
    /// Looks up embeddings for given token indices.
    ///
    /// `token_ids`: sequence of token indices (public, known to all parties).
    /// `embedding_shares[party][vocab_size * d_model]`: party's share of
    ///   the embedding table, stored row-major (Fr field elements).
    /// `vocab_size`: vocabulary size.
    /// `d_model`: embedding dimension.
    ///
    /// Returns `result[party][seq_len * d_model]`: party's share of the
    /// embedded sequence.
    pub fn forward(
        token_ids: &[usize],
        embedding_shares: &[Vec<Fr>],
        vocab_size: usize,
        d_model: usize,
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let _num_parties = embedding_shares.len();
        let seq_len = token_ids.len();

        // Validate token indices.
        for &tid in token_ids {
            if tid >= vocab_size {
                return Err(MPCError::ProtocolError(format!(
                    "Token ID {} exceeds vocab size {}",
                    tid, vocab_size,
                )));
            }
        }

        // Each party looks up their share of the embedding for each token.
        // This is entirely local — no communication needed.
        let result: Vec<Vec<Fr>> = embedding_shares
            .iter()
            .map(|share| {
                let mut output = Vec::with_capacity(seq_len * d_model);
                for &tid in token_ids {
                    let start = tid * d_model;
                    let end = start + d_model;
                    output.extend_from_slice(&share[start..end]);
                }
                output
            })
            .collect();

        Ok(result)
    }

    /// Adds sinusoidal positional encoding to embedded shares.
    ///
    /// Since positional encodings are public (deterministic, not learned),
    /// only party 0 adds them.
    pub fn add_positional_encoding(
        x_shares: &mut [Vec<Fr>],
        seq_len: usize,
        d_model: usize,
        max_seq_len: usize,
    ) {
        // Generate sinusoidal encodings (computed in f64, converted to Fr).
        let pe = sinusoidal_encoding_fr(seq_len, d_model, max_seq_len);

        // Only party 0 adds the positional encoding.
        for pos in 0..seq_len {
            for d in 0..d_model {
                let idx = pos * d_model + d;
                x_shares[0][idx] = Fr::add(&x_shares[0][idx], &pe[idx]);
            }
        }
    }

    /// Backward pass: compute gradient for embeddings.
    ///
    /// The gradient for an embedding lookup scatters the output gradient
    /// back to the corresponding embedding rows.
    pub fn backward(
        dy_shares: &[Vec<Fr>],
        token_ids: &[usize],
        vocab_size: usize,
        d_model: usize,
    ) -> Vec<Vec<Fr>> {
        dy_shares
            .iter()
            .map(|dy| {
                let mut grad = vec![Fr::ZERO; vocab_size * d_model];
                for (pos, &tid) in token_ids.iter().enumerate() {
                    let start = tid * d_model;
                    for d in 0..d_model {
                        grad[start + d] = Fr::add(&grad[start + d], &dy[pos * d_model + d]);
                    }
                }
                grad
            })
            .collect()
    }
}

/// Generates sinusoidal positional encodings as Fr field elements.
///
/// The computation is done in f64 (since sin/cos are transcendental functions)
/// then converted to Fr. This is appropriate because positional encodings
/// are public constants.
fn sinusoidal_encoding_fr(seq_len: usize, d_model: usize, _max_seq_len: usize) -> Vec<Fr> {
    let mut pe = Vec::with_capacity(seq_len * d_model);

    for pos in 0..seq_len {
        for d in 0..d_model {
            let val = if d % 2 == 0 {
                let angle = pos as f64 / (10000.0_f64).powf(d as f64 / d_model as f64);
                angle.sin()
            } else {
                let angle = pos as f64 / (10000.0_f64).powf((d - 1) as f64 / d_model as f64);
                angle.cos()
            };
            pe.push(Fr::from_f64(val));
        }
    }

    pe
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_into_shares(data: &[Fr], n: usize, seed: u64) -> Vec<Vec<Fr>> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let dim = data.len();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; n];
        for d in 0..dim {
            let mut sum = Fr::ZERO;
            for i in 0..n - 1 {
                let r = Fr::random(&mut rng);
                shares[i][d] = r.clone();
                sum = Fr::add(&sum, &r);
            }
            shares[n - 1][d] = Fr::sub(&data[d], &sum);
        }
        shares
    }

    fn reconstruct(shares: &[Vec<Fr>]) -> Vec<Fr> {
        let dim = shares[0].len();
        let mut result = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                result[i] = Fr::add(&result[i], v);
            }
        }
        result
    }

    #[test]
    fn test_embedding_lookup() {
        let vocab_size = 5;
        let d_model = 3;

        // Embedding table: each row is the embedding for that token.
        let mut embedding = Vec::with_capacity(vocab_size * d_model);
        for i in 0..vocab_size {
            for d in 0..d_model {
                embedding.push(Fr::from_f64((i * 10 + d) as f64));
            }
        }

        let shares = split_into_shares(&embedding, 3, 42);
        let tokens = vec![0, 2, 4]; // Look up tokens 0, 2, 4

        let result_shares =
            SecureEmbedding::forward(&tokens, &shares, vocab_size, d_model).unwrap();

        let result = reconstruct(&result_shares);

        // Token 0: [0, 1, 2]
        assert!((result[0].to_f64() - 0.0).abs() < 1e-6);
        assert!((result[1].to_f64() - 1.0).abs() < 1e-6);
        assert!((result[2].to_f64() - 2.0).abs() < 1e-6);

        // Token 2: [20, 21, 22]
        assert!((result[3].to_f64() - 20.0).abs() < 1e-6);
        assert!((result[4].to_f64() - 21.0).abs() < 1e-6);
        assert!((result[5].to_f64() - 22.0).abs() < 1e-6);

        // Token 4: [40, 41, 42]
        assert!((result[6].to_f64() - 40.0).abs() < 1e-6);
        assert!((result[7].to_f64() - 41.0).abs() < 1e-6);
        assert!((result[8].to_f64() - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_backward() {
        let vocab_size = 4;
        let d_model = 2;
        let tokens = vec![1, 3, 1]; // Token 1 appears twice

        // 3 positions x 2 dims
        let dy = vec![vec![
            Fr::from_f64(1.0), Fr::from_f64(2.0),
            Fr::from_f64(3.0), Fr::from_f64(4.0),
            Fr::from_f64(5.0), Fr::from_f64(6.0),
        ]];

        let grad = SecureEmbedding::backward(&dy, &tokens, vocab_size, d_model);

        // Token 1 grad = [1,2] + [5,6] = [6, 8]
        assert!((grad[0][1 * d_model].to_f64() - 6.0).abs() < 1e-6);
        assert!((grad[0][1 * d_model + 1].to_f64() - 8.0).abs() < 1e-6);

        // Token 3 grad = [3, 4]
        assert!((grad[0][3 * d_model].to_f64() - 3.0).abs() < 1e-6);
        assert!((grad[0][3 * d_model + 1].to_f64() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_out_of_range_token() {
        let shares = vec![vec![Fr::ZERO; 6]]; // vocab=2, d=3
        let result = SecureEmbedding::forward(&[5], &shares, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_positional_encoding() {
        let pe = sinusoidal_encoding_fr(4, 4, 100);
        assert_eq!(pe.len(), 16); // 4 positions x 4 dims

        // Position 0: sin(0)=0, cos(0)=1
        assert!((pe[0].to_f64() - 0.0).abs() < 1e-6);
        assert!((pe[1].to_f64() - 1.0).abs() < 1e-6);
    }
}
