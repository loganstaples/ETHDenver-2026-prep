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
use crate::field::Fr;

/// Secure matrix multiplication.
pub struct SecureMatmul;

impl SecureMatmul {
    /// Performs one party's local computation for matrix Beaver multiplication.
    ///
    /// `triple`: this party's matrix Beaver triple share
    /// `opened_d`: reconstructed D = A - U [m x k]
    /// `opened_e`: reconstructed E = B - V [k x n]
    /// `party_index`: 0-based index of this party
    ///
    /// Returns this party's share of C = A @ B [m x n].
    pub fn beaver_matmul_local(
        triple: &MatrixBeaverTriple,
        opened_d: &[Fr],
        opened_e: &[Fr],
        party_index: usize,
    ) -> Vec<Fr> {
        let (m, k, n) = (triple.m, triple.k, triple.n);

        // [C] = [W] + D @ [V] + [U] @ E
        let mut result = triple.c.clone();

        // D @ [V]: public D times this party's V share
        let dv = matmul_plain(opened_d, &triple.b, m, k, n);
        for i in 0..m * n {
            result[i] = Fr::add(&result[i], &dv[i]);
        }

        // [U] @ E: this party's U share times public E
        let ue = matmul_plain(&triple.a, opened_e, m, k, n);
        for i in 0..m * n {
            result[i] = Fr::add(&result[i], &ue[i]);
        }

        // D @ E: only party 0 adds this
        if party_index == 0 {
            let de = matmul_plain(opened_d, opened_e, m, k, n);
            for i in 0..m * n {
                result[i] = Fr::add(&result[i], &de[i]);
            }
        }

        result
    }

    /// Computes the masked values D = A - U, E = B - V for this party.
    pub fn beaver_mask(
        a_share: &[Fr],
        b_share: &[Fr],
        triple: &MatrixBeaverTriple,
    ) -> (Vec<Fr>, Vec<Fr>) {
        let d: Vec<Fr> = a_share
            .iter()
            .zip(&triple.a)
            .map(|(a, u)| Fr::sub(a, u))
            .collect();
        let e: Vec<Fr> = b_share
            .iter()
            .zip(&triple.b)
            .map(|(b, v)| Fr::sub(b, v))
            .collect();
        (d, e)
    }

    /// Simulates the full secure matrix multiplication for all parties.
    ///
    /// Takes all parties' shares and triples, returns all parties' result shares.
    pub fn simulate_matmul(
        a_shares: &[Vec<Fr>],
        b_shares: &[Vec<Fr>],
        triples: &[MatrixBeaverTriple],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Vec<Fr>> {
        let num_parties = a_shares.len();
        assert_eq!(b_shares.len(), num_parties);
        assert_eq!(triples.len(), num_parties);

        // Step 1: Each party computes D_i, E_i.
        let masks: Vec<(Vec<Fr>, Vec<Fr>)> = a_shares
            .iter()
            .zip(b_shares)
            .zip(triples)
            .map(|((a, b), t)| Self::beaver_mask(a, b, t))
            .collect();

        // Step 2: Reconstruct D and E (sum all parties' shares).
        let mut opened_d = vec![Fr::ZERO; m * k];
        let mut opened_e = vec![Fr::ZERO; k * n];

        for (d_share, e_share) in &masks {
            for (i, v) in d_share.iter().enumerate() {
                opened_d[i] = Fr::add(&opened_d[i], v);
            }
            for (i, v) in e_share.iter().enumerate() {
                opened_e[i] = Fr::add(&opened_e[i], v);
            }
        }

        // Step 3: Each party computes their result share.
        (0..num_parties)
            .map(|i| Self::beaver_matmul_local(&triples[i], &opened_d, &opened_e, i))
            .collect()
    }

    /// Simulates secure matmul using triples from pools.
    pub fn simulate_matmul_from_pool(
        a_shares: &[Vec<Fr>],
        b_shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
        m: usize,
        k: usize,
        n: usize,
    ) -> MPCResult<Vec<Vec<Fr>>> {
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
        a_shares: &[Vec<Fr>],
        v_shares: &[Vec<Fr>],
        triples: &[MatrixBeaverTriple],
        m: usize,
        k: usize,
    ) -> Vec<Vec<Fr>> {
        Self::simulate_matmul(a_shares, v_shares, triples, m, k, 1)
    }

    /// Multiplies a shared matrix by a public matrix.
    /// This is a local operation (no communication needed).
    ///
    /// Computes [C] = [A] @ B_pub where each party computes A_i @ B_pub.
    pub fn matmul_by_public(
        a_share: &[Fr],
        b_public: &[Fr],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Fr> {
        matmul_plain(a_share, b_public, m, k, n)
    }

    /// Multiplies a public matrix by a shared matrix.
    /// This is a local operation (no communication needed).
    pub fn public_matmul_by_shared(
        a_public: &[Fr],
        b_share: &[Fr],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Fr> {
        matmul_plain(a_public, b_share, m, k, n)
    }

    /// Secure outer product: [u] outer [v] = [u * v^T].
    pub fn simulate_outer_product(
        u_shares: &[Vec<Fr>], // each [m]
        v_shares: &[Vec<Fr>], // each [n]
        triples: &[MatrixBeaverTriple],
        m: usize,
        n: usize,
    ) -> Vec<Vec<Fr>> {
        // Outer product is matmul of [m x 1] @ [1 x n]
        Self::simulate_matmul(u_shares, v_shares, triples, m, 1, n)
    }

    /// Serializes a matrix of Fr elements to bytes for batched network transport.
    ///
    /// Format: [4-byte count LE][32-byte Fr elements...]
    /// Sending the entire D/E matrix in one message instead of per-element
    /// reduces round trips from O(m*k + k*n) to O(1).
    pub fn serialize_matrix_batch(matrix: &[Fr]) -> Vec<u8> {
        crate::protocols::arithmetic::SecureArithmetic::serialize_share_batch(matrix)
    }

    /// Deserializes a matrix batch from bytes.
    pub fn deserialize_matrix_batch(data: &[u8]) -> MPCResult<Vec<Fr>> {
        crate::protocols::arithmetic::SecureArithmetic::deserialize_share_batch(data)
    }

    /// Batched matrix Beaver mask: computes D = A - U and E = B - V for the
    /// entire matrix at once, returning vectors ready for single-message
    /// network transmission.
    ///
    /// This replaces per-element communication with a single broadcast of the
    /// D [m*k] and E [k*n] vectors.
    pub fn batched_beaver_mask(
        a_share: &[Fr],
        b_share: &[Fr],
        triple: &MatrixBeaverTriple,
    ) -> (Vec<Fr>, Vec<Fr>) {
        Self::beaver_mask(a_share, b_share, triple)
    }
}

/// Plain matrix multiplication: C = A @ B.
/// A is [m x k], B is [k x n], result C is [m x n].
/// All in row-major flattened form.
/// Uses fixed_mul for proper fixed-point arithmetic.
fn matmul_plain(a: &[Fr], b: &[Fr], m: usize, k: usize, n: usize) -> Vec<Fr> {
    let mut c = vec![Fr::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Fr::ZERO;
            for l in 0..k {
                sum = Fr::add(&sum, &a[i * k + l].fixed_mul(&b[l * n + j]));
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
    use crate::field::ops::sum;

    #[test]
    fn test_plain_matmul() {
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        let a = vec![
            Fr::from_f64(1.0),
            Fr::from_f64(2.0),
            Fr::from_f64(3.0),
            Fr::from_f64(4.0),
        ];
        let b = vec![
            Fr::from_f64(5.0),
            Fr::from_f64(6.0),
            Fr::from_f64(7.0),
            Fr::from_f64(8.0),
        ];
        let c = matmul_plain(&a, &b, 2, 2, 2);

        let c0 = c[0].to_f64();
        let c1 = c[1].to_f64();
        let c2 = c[2].to_f64();
        let c3 = c[3].to_f64();

        assert!((c0 - 19.0).abs() < 0.01, "got {}", c0);
        assert!((c1 - 22.0).abs() < 0.01, "got {}", c1);
        assert!((c2 - 43.0).abs() < 0.01, "got {}", c2);
        assert!((c3 - 50.0).abs() < 0.01, "got {}", c3);
    }

    #[test]
    fn test_secure_matmul() {
        let mut dealer = TrustedDealer::with_seed(42);
        let (m, k, n) = (2, 3, 2);

        // A = [[1,2,3],[4,5,6]]
        let a: Vec<Fr> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(Fr::from_f64)
            .collect();
        // B = [[1,0],[0,1],[1,1]]
        let b: Vec<Fr> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            .into_iter()
            .map(Fr::from_f64)
            .collect();
        // Expected C = A@B = [[4,5],[10,11]]
        let expected = matmul_plain(&a, &b, m, k, n);

        // Split A and B into 3 shares.
        let a_shares: Vec<Vec<Fr>> = split_into_shares(&a, 3, 42);
        let b_shares: Vec<Vec<Fr>> = split_into_shares(&b, 3, 99);

        let triples = dealer.generate_matrix_triple(m, k, n, 3);

        let result_shares =
            SecureMatmul::simulate_matmul(&a_shares, &b_shares, &triples, m, k, n);

        // Reconstruct result.
        let result = reconstruct_shares(&result_shares);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            let r_f64 = r.to_f64();
            let e_f64 = e.to_f64();
            assert!(
                (r_f64 - e_f64).abs() < 0.01,
                "Matmul element {} wrong: {} vs {}",
                i,
                r_f64,
                e_f64,
            );
        }
    }

    #[test]
    fn test_secure_matvec() {
        let mut dealer = TrustedDealer::with_seed(42);
        let (m, k) = (3, 2);

        let a: Vec<Fr> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(Fr::from_f64)
            .collect(); // [3x2]
        let v: Vec<Fr> = vec![1.0, 1.0].into_iter().map(Fr::from_f64).collect(); // [2x1]
        let expected = matmul_plain(&a, &v, m, k, 1); // [3, 7, 11]

        let a_shares = split_into_shares(&a, 3, 42);
        let v_shares = split_into_shares(&v, 3, 99);
        let triples = dealer.generate_matrix_triple(m, k, 1, 3);

        let result_shares = SecureMatmul::simulate_matvec(&a_shares, &v_shares, &triples, m, k);
        let result = reconstruct_shares(&result_shares);

        for (r, e) in result.iter().zip(expected.iter()) {
            let r_f64 = r.to_f64();
            let e_f64 = e.to_f64();
            assert!((r_f64 - e_f64).abs() < 0.01);
        }
    }

    #[test]
    fn test_matmul_by_public() {
        // Party 0's share of A is [1, 2, 3, 4]
        let a_share: Vec<Fr> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(Fr::from_f64)
            .collect();
        let b_public: Vec<Fr> = vec![1.0, 0.0, 0.0, 1.0]
            .into_iter()
            .map(Fr::from_f64)
            .collect();
        let result = SecureMatmul::matmul_by_public(&a_share, &b_public, 2, 2, 2);
        // [1*1+2*0, 1*0+2*1; 3*1+4*0, 3*0+4*1] = [1, 2, 3, 4]
        let expected: Vec<Fr> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(Fr::from_f64)
            .collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(r.ct_eq(e).to_bool());
        }
    }

    #[test]
    fn test_non_square_matmul() {
        let mut dealer = TrustedDealer::with_seed(42);
        let (m, k, n) = (2, 4, 3);

        let a: Vec<Fr> = (0..m * k)
            .map(|i| Fr::from_f64(i as f64 * 0.1))
            .collect();
        let b: Vec<Fr> = (0..k * n)
            .map(|i| Fr::from_f64(i as f64 * 0.2))
            .collect();
        let expected = matmul_plain(&a, &b, m, k, n);

        let a_shares = split_into_shares(&a, 3, 42);
        let b_shares = split_into_shares(&b, 3, 99);
        let triples = dealer.generate_matrix_triple(m, k, n, 3);

        let result = reconstruct_shares(
            &SecureMatmul::simulate_matmul(&a_shares, &b_shares, &triples, m, k, n),
        );

        for (r, e) in result.iter().zip(expected.iter()) {
            let r_f64 = r.to_f64();
            let e_f64 = e.to_f64();
            assert!((r_f64 - e_f64).abs() < 0.01, "{} vs {}", r_f64, e_f64);
        }
    }

    // Test helpers

    fn split_into_shares(data: &[Fr], n: usize, seed: u64) -> Vec<Vec<Fr>> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let dim = data.len();
        let mut shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; n];

        for d in 0..dim {
            let mut s = Fr::ZERO;
            for i in 0..n - 1 {
                let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
                shares[i][d] = r.clone();
                s = Fr::add(&s, &r);
            }
            shares[n - 1][d] = Fr::sub(&data[d], &s);
        }

        shares
    }

    fn reconstruct_shares(shares: &[Vec<Fr>]) -> Vec<Fr> {
        let dim = shares[0].len();
        let mut result = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                result[i] = Fr::add(&result[i], v);
            }
        }
        result
    }
}
