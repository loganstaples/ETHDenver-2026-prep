//! Secure linear layer: y = x @ W^T + b on secret-shared weights.
//!
//! The weight matrix W and bias b are secret-shared across parties.
//! The input x may be either public or shared.
//!
//! Two modes:
//! - **Shared input, shared weights**: Full Beaver-triple matmul.
//!   Used when the previous layer produced shared output.
//! - **Public input, shared weights**: Local computation only.
//!   Used for the first layer when input data is known to all parties.

use crate::beaver::pool::BeaverPool;
use crate::beaver::triple::MatrixBeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::protocols::matmul::SecureMatmul;
use crate::protocols::arithmetic::SecureArithmetic;

/// Configuration for a secure linear layer.
#[derive(Debug, Clone)]
pub struct SecureLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
}

/// Secure linear layer operating on shares.
pub struct SecureLinear;

impl SecureLinear {
    /// Forward pass with shared input and shared weights.
    ///
    /// Computes [y] = [x] @ [W]^T + [b]
    ///
    /// `x_shares[party][element]`: party's share of input [batch * in_features]
    /// `w_shares[party][element]`: party's share of W [out_features * in_features]
    /// `b_shares[party][element]`: party's share of bias [out_features] (optional)
    /// `pools`: Beaver triple pools for each party
    ///
    /// Returns `result_shares[party][element]` for output [batch * out_features].
    pub fn forward_shared(
        x_shares: &[Vec<f64>],
        w_shares: &[Vec<f64>],
        b_shares: Option<&[Vec<f64>]>,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
        let num_parties = x_shares.len();

        // We need W^T, so transpose each party's share.
        let wt_shares: Vec<Vec<f64>> = w_shares
            .iter()
            .map(|w| transpose_flat(w, out_features, in_features))
            .collect();

        // [y] = [x] @ [W^T]  using Beaver matmul.
        // x is [batch x in_features], W^T is [in_features x out_features]
        let mut y_shares = SecureMatmul::simulate_matmul_from_pool(
            x_shares,
            &wt_shares,
            pools,
            batch_size,
            in_features,
            out_features,
        )?;

        // Add bias if present: [y] += [b] (broadcast across batch).
        if let Some(bias) = b_shares {
            for i in 0..num_parties {
                for row in 0..batch_size {
                    for col in 0..out_features {
                        y_shares[i][row * out_features + col] += bias[i][col];
                    }
                }
            }
        }

        Ok(y_shares)
    }

    /// Forward pass with public input and shared weights.
    ///
    /// Computes [y] = x_pub @ [W]^T + [b]
    ///
    /// Since x is public, each party can compute x @ W_i^T locally.
    /// No communication or Beaver triples needed.
    pub fn forward_public_input(
        x_public: &[f64],
        w_shares: &[Vec<f64>],
        b_shares: Option<&[Vec<f64>]>,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<Vec<f64>> {
        let num_parties = w_shares.len();

        // Each party computes x @ W_i^T locally.
        let mut y_shares: Vec<Vec<f64>> = w_shares
            .iter()
            .map(|w| {
                let wt = transpose_flat(w, out_features, in_features);
                SecureMatmul::matmul_by_public(
                    x_public,
                    &wt,
                    batch_size,
                    in_features,
                    out_features,
                )
            })
            .collect();

        // The above is wrong - x is public and w is shared, so we need
        // public_matmul_by_shared which gives x @ [W^T]_i for each party.
        // That's exactly what matmul_by_public does with x as first arg
        // but it multiplies a_share @ b_public. We need to swap.
        y_shares = w_shares
            .iter()
            .map(|w| {
                let wt = transpose_flat(w, out_features, in_features);
                SecureMatmul::public_matmul_by_shared(
                    x_public,
                    &wt,
                    batch_size,
                    in_features,
                    out_features,
                )
            })
            .collect();

        // Add bias.
        if let Some(bias) = b_shares {
            for i in 0..num_parties {
                for row in 0..batch_size {
                    for col in 0..out_features {
                        y_shares[i][row * out_features + col] += bias[i][col];
                    }
                }
            }
        }

        y_shares
    }

    /// Backward pass: computes gradient shares.
    ///
    /// Given [dy] (gradient of loss w.r.t. output), computes:
    /// - [dx] = [dy] @ [W]    (gradient w.r.t. input)
    /// - [dW] = [dy]^T @ [x]  (gradient w.r.t. weights)
    /// - [db] = sum([dy])      (gradient w.r.t. bias, local operation)
    pub fn backward_shared(
        dy_shares: &[Vec<f64>],
        x_shares: &[Vec<f64>],
        w_shares: &[Vec<f64>],
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        pools: &mut [BeaverPool],
    ) -> MPCResult<LinearGradients> {
        let num_parties = dy_shares.len();

        // [dx] = [dy] @ [W]
        // dy is [batch x out], W is [out x in], result is [batch x in]
        let dx_shares = SecureMatmul::simulate_matmul_from_pool(
            dy_shares,
            w_shares,
            pools,
            batch_size,
            out_features,
            in_features,
        )?;

        // [dW] = [dy]^T @ [x]
        // dy^T is [out x batch], x is [batch x in], result is [out x in]
        let dyt_shares: Vec<Vec<f64>> = dy_shares
            .iter()
            .map(|dy| transpose_flat(dy, batch_size, out_features))
            .collect();

        let dw_shares = SecureMatmul::simulate_matmul_from_pool(
            &dyt_shares,
            x_shares,
            pools,
            out_features,
            batch_size,
            in_features,
        )?;

        // [db] = sum_rows([dy]) — local operation
        let db_shares: Vec<Vec<f64>> = dy_shares
            .iter()
            .map(|dy| {
                let mut bias_grad = vec![0.0; out_features];
                for row in 0..batch_size {
                    for col in 0..out_features {
                        bias_grad[col] += dy[row * out_features + col];
                    }
                }
                bias_grad
            })
            .collect();

        Ok(LinearGradients {
            dx_shares,
            dw_shares,
            db_shares,
        })
    }
}

/// Gradients produced by the backward pass.
pub struct LinearGradients {
    /// Gradient w.r.t. input: [party][batch * in_features].
    pub dx_shares: Vec<Vec<f64>>,
    /// Gradient w.r.t. weights: [party][out_features * in_features].
    pub dw_shares: Vec<Vec<f64>>,
    /// Gradient w.r.t. bias: [party][out_features].
    pub db_shares: Vec<Vec<f64>>,
}

/// Transpose a flattened [rows x cols] matrix to [cols x rows].
fn transpose_flat(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;

    fn split_into_shares(data: &[f64], n: usize, seed: u64) -> Vec<Vec<f64>> {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha20Rng;

        let dim = data.len();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut shares: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];
        for d in 0..dim {
            let mut sum = 0.0;
            for i in 0..n - 1 {
                let r: f64 = rng.gen_range(-100.0..100.0);
                shares[i][d] = r;
                sum += r;
            }
            shares[n - 1][d] = data[d] - sum;
        }
        shares
    }

    fn reconstruct(shares: &[Vec<f64>]) -> Vec<f64> {
        let dim = shares[0].len();
        let mut result = vec![0.0; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                result[i] += v;
            }
        }
        result
    }

    fn plain_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn test_public_input_linear() {
        let in_f = 3;
        let out_f = 2;
        let batch = 2;

        // x = [[1,2,3],[4,5,6]]  (2x3)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // W = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  (2x3, out x in)
        let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        // b = [0.01, 0.02]
        let b = vec![0.01, 0.02];

        // Expected: x @ W^T + b
        let wt = transpose_flat(&w, out_f, in_f);
        let expected = {
            let mut y = plain_matmul(&x, &wt, batch, in_f, out_f);
            for row in 0..batch {
                for col in 0..out_f {
                    y[row * out_f + col] += b[col];
                }
            }
            y
        };

        let w_shares = split_into_shares(&w, 3, 42);
        let b_shares = split_into_shares(&b, 3, 99);

        let result_shares = SecureLinear::forward_public_input(
            &x,
            &w_shares,
            Some(&b_shares),
            batch,
            in_f,
            out_f,
        );

        let result = reconstruct(&result_shares);

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-6,
                "Public input linear: {} vs {}",
                r,
                e,
            );
        }
    }

    #[test]
    fn test_shared_input_linear() {
        let mut dealer = TrustedDealer::with_seed(42);
        let in_f = 2;
        let out_f = 2;
        let batch = 1;

        let x = vec![1.0, 2.0];
        let w = vec![0.5, 0.3, 0.2, 0.4];
        let b = vec![0.1, 0.2];

        let wt = transpose_flat(&w, out_f, in_f);
        let expected = {
            let mut y = plain_matmul(&x, &wt, batch, in_f, out_f);
            y[0] += b[0];
            y[1] += b[1];
            y
        };

        let x_shares = split_into_shares(&x, 3, 10);
        let w_shares = split_into_shares(&w, 3, 20);
        let b_shares = split_into_shares(&b, 3, 30);

        // Create Beaver pool with matrix triples.
        let triples = dealer.generate_matrix_triples(5, batch, in_f, out_f, 3);
        let mut pools: Vec<BeaverPool> = (0..3)
            .map(|i| {
                let mut p = BeaverPool::new(i, 3, 64);
                p.fill_matrix(batch, in_f, out_f, triples[i].clone());
                p
            })
            .collect();

        let result_shares = SecureLinear::forward_shared(
            &x_shares,
            &w_shares,
            Some(&b_shares),
            batch,
            in_f,
            out_f,
            &mut pools,
        )
        .unwrap();

        let result = reconstruct(&result_shares);

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-3,
                "Shared input linear: {} vs {}",
                r,
                e,
            );
        }
    }

    #[test]
    fn test_transpose_flat() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let transposed = transpose_flat(&data, 2, 3);
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // 3x2
    }
}
