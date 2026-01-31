//! Secure matrix multiplication protocol.
//!
//! Computes [C] = [A] @ [B] where A and B are secret-shared matrices.
//!
//! Uses matrix Beaver triples: pre-generated (U, V, W) where W = U @ V.
//! The protocol is:
//! 1. Each party computes [D] = [A] - [U], [E] = [B] - [V]
//! 2. Parties reconstruct D and E (these are random, reveal nothing)
//! 3. Each party computes [C] = [W] + D @ [V] + [U] @ E + D @ E
//!    (only party 0 adds D @ E to avoid double-counting)

use crate::beaver::pool::BeaverPool;
use crate::beaver::triple::MatrixBeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::sharing::tensor::TensorShare;

/// Secure matrix multiplication.
pub struct SecureMatmul;

impl SecureMatmul {
    /// Performs one party's local computation for matrix Beaver multiplication.
    ///
    /// `a_share`: this party's share of A [m x k], flattened row-major
    /// `b_share`: this party's share of B [k x n], flattened row-major
    /// `triple`: this party's matrix Beaver triple share
    /// `opened_d`: reconstructed D = A - U [m x k]
    /// `opened_e`: reconstructed E = B - V [k x n]
    /// `party_index`: 0-based index of this party
    ///
    /// Returns this party's share of C = A @ B [m x n].
    pub fn beaver_matmul_local(
        triple: &MatrixBeaverTriple,
        opened_d: &[f64],
        opened_e: &[f64],
        party_index: usize,
    ) -> Vec<f64> {
        let (m, k, n) = (triple.m, triple.k, triple.n);

        // [C] = [W] + D @ [V] + [U] @ E
        let mut result = triple.c.clone();

        // D @ [V]: public D times this party's V share
        let dv = matmul_plain(opened_d, &triple.b, m, k, n);
        for i in 0..m * n {
            result[i] += dv[i];
        }

        // [U] @ E: this party's U share times public E
        let ue = matmul_plain(&triple.a, opened_e, m, k, n);
        for i in 0..m * n {
            result[i] += ue[i];
        }

        // D @ E: only party 0 adds this
        if party_index == 0 {
            let de = matmul_plain(opened_d, opened_e, m, k, n);
            for i in 0..m * n {
                result[i] += de[i];
            }
        }

        result
    }

    /// Computes the masked values D = A - U, E = B - V for this party.
    pub fn beaver_mask(
        a_share: &[f64],
        b_share: &[f64],
        triple: &MatrixBeaverTriple,
    ) -> (Vec<f64>, Vec<f64>) {
        let d: Vec<f64> = a_share.iter().zip(&triple.a).map(|(a, u)| a - u).collect();
        let e: Vec<f64> = b_share.iter().zip(&triple.b).map(|(b, v)| b - v).collect();
        (d, e)
    }

    /// Simulates the full secure matrix multiplication for all parties.
    ///
    /// Takes all parties' shares and triples, returns all parties' result shares.
    pub fn simulate_matmul(
        a_shares: &[Vec<f64>],
        b_shares: &[Vec<f64>],
        triples: &[MatrixBeaverTriple],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Vec<f64>> {
        let num_parties = a_shares.len();
        assert_eq!(b_shares.len(), num_parties);
        assert_eq!(triples.len(), num_parties);

        // Step 1: Each party computes D_i, E_i.
        let masks: Vec<(Vec<f64>, Vec<f64>)> = a_shares
            .iter()
            .zip(b_shares)
            .zip(triples)
            .map(|((a, b), t)| Self::beaver_mask(a, b, t))
            .collect();

        // Step 2: Reconstruct D and E (sum all parties' shares).
        let mut opened_d = vec![0.0; m * k];
        let mut opened_e = vec![0.0; k * n];

        for (d_share, e_share) in &masks {
            for (i, v) in d_share.iter().enumerate() {
                opened_d[i] += v;
            }
            for (i, v) in e_share.iter().enumerate() {
                opened_e[i] += v;
            }
        }

        // Step 3: Each party computes their result share.
        (0..num_parties)
            .map(|i| Self::beaver_matmul_local(&triples[i], &opened_d, &opened_e, i))
            .collect()
    }

    /// Simulates secure matmul using triples from pools.
    pub fn simulate_matmul_from_pool(
        a_shares: &[Vec<f64>],
        b_shares: &[Vec<f64>],
        pools: &mut [BeaverPool],
        m: usize,
        k: usize,
        n: usize,
    ) -> MPCResult<Vec<Vec<f64>>> {
        let num_parties = a_shares.len();
        if pools.len() != num_parties {
            return Err(MPCError::InsufficientParties {
                required: num_parties,
                available: pools.len(),
            });
        }

        let triples: Vec<MatrixBeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_matrix(m, k, n))
            .collect::<MPCResult<Vec<_>>>()?;

        Ok(Self::simulate_matmul(a_shares, b_shares, &triples, m, k, n))
    }

    /// Secure matrix-vector multiplication: [A] @ [v].
    /// Treats v as a [k x 1] matrix.
    pub fn simulate_matvec(
        a_shares: &[Vec<f64>],
        v_shares: &[Vec<f64>],
        triples: &[MatrixBeaverTriple],
        m: usize,
        k: usize,
    ) -> Vec<Vec<f64>> {
        Self::simulate_matmul(a_shares, v_shares, triples, m, k, 1)
    }

    /// Multiplies a shared matrix by a public matrix.
    /// This is a local operation (no communication needed).
    ///
    /// Computes [C] = [A] @ B_pub where each party computes A_i @ B_pub.
    pub fn matmul_by_public(
        a_share: &[f64],
        b_public: &[f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f64> {
        matmul_plain(a_share, b_public, m, k, n)
    }

    /// Multiplies a public matrix by a shared matrix.
    /// This is a local operation (no communication needed).
    pub fn public_matmul_by_shared(
        a_public: &[f64],
        b_share: &[f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f64> {
        matmul_plain(a_public, b_share, m, k, n)
    }

    /// Secure outer product: [u] outer [v] = [u * v^T].
    pub fn simulate_outer_product(
        u_shares: &[Vec<f64>], // each [m]
        v_shares: &[Vec<f64>], // each [n]
        triples: &[MatrixBeaverTriple],
        m: usize,
        n: usize,
    ) -> Vec<Vec<f64>> {
        // Outer product is matmul of [m x 1] @ [1 x n]
        Self::simulate_matmul(u_shares, v_shares, triples, m, 1, n)
    }
}

/// Plain matrix multiplication: C = A @ B.
/// A is [m x k], B is [k x n], result C is [m x n].
/// All in row-major flattened form.
fn matmul_plain(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;

    #[test]
    fn test_plain_matmul() {
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul_plain(&a, &b, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-10);
        assert!((c[1] - 22.0).abs() < 1e-10);
        assert!((c[2] - 43.0).abs() < 1e-10);
        assert!((c[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_secure_matmul() {
        let mut dealer = TrustedDealer::with_seed(42);
        let (m, k, n) = (2, 3, 2);

        // A = [[1,2,3],[4,5,6]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B = [[1,0],[0,1],[1,1]]
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        // Expected C = A@B = [[4,5],[10,11]]
        let expected = matmul_plain(&a, &b, m, k, n);

        // Split A and B into 3 shares.
        let a_shares: Vec<Vec<f64>> = split_into_shares(&a, 3, 42);
        let b_shares: Vec<Vec<f64>> = split_into_shares(&b, 3, 99);

        let triples = dealer.generate_matrix_triple(m, k, n, 3);

        let result_shares =
            SecureMatmul::simulate_matmul(&a_shares, &b_shares, &triples, m, k, n);

        // Reconstruct result.
        let result = reconstruct_shares(&result_shares);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Matmul element {} wrong: {} vs {}",
                i,
                r,
                e,
            );
        }
    }

    #[test]
    fn test_secure_matvec() {
        let mut dealer = TrustedDealer::with_seed(42);
        let (m, k) = (3, 2);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3x2]
        let v = vec![1.0, 1.0]; // [2x1]
        let expected = matmul_plain(&a, &v, m, k, 1); // [3, 7, 11]

        let a_shares = split_into_shares(&a, 3, 42);
        let v_shares = split_into_shares(&v, 3, 99);
        let triples = dealer.generate_matrix_triple(m, k, 1, 3);

        let result_shares = SecureMatmul::simulate_matvec(&a_shares, &v_shares, &triples, m, k);
        let result = reconstruct_shares(&result_shares);

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[test]
    fn test_matmul_by_public() {
        // Party 0's share of A is [1, 2, 3, 4]
        let a_share = vec![1.0, 2.0, 3.0, 4.0];
        let b_public = vec![1.0, 0.0, 0.0, 1.0];
        let result = SecureMatmul::matmul_by_public(&a_share, &b_public, 2, 2, 2);
        // [1*1+2*0, 1*0+2*1; 3*1+4*0, 3*0+4*1] = [1, 2, 3, 4]
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_non_square_matmul() {
        let mut dealer = TrustedDealer::with_seed(42);
        let (m, k, n) = (2, 4, 3);

        let a: Vec<f64> = (0..m * k).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..k * n).map(|i| i as f64 * 0.2).collect();
        let expected = matmul_plain(&a, &b, m, k, n);

        let a_shares = split_into_shares(&a, 3, 42);
        let b_shares = split_into_shares(&b, 3, 99);
        let triples = dealer.generate_matrix_triple(m, k, n, 3);

        let result = reconstruct_shares(
            &SecureMatmul::simulate_matmul(&a_shares, &b_shares, &triples, m, k, n),
        );

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-3, "{} vs {}", r, e);
        }
    }

    // Test helpers

    fn split_into_shares(data: &[f64], n: usize, seed: u64) -> Vec<Vec<f64>> {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let dim = data.len();
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

    fn reconstruct_shares(shares: &[Vec<f64>]) -> Vec<f64> {
        let dim = shares[0].len();
        let mut result = vec![0.0; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                result[i] += v;
            }
        }
        result
    }
}
