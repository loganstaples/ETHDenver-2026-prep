//! Trusted dealer for Beaver triple generation.
//!
//! In the preprocessing phase, a trusted dealer generates random triples
//! (a, b, c = a*b) and distributes additive shares to each party.
//!
//! This is the simplest approach and is suitable for the demo. In production,
//! the dealer role would be replaced by a distributed protocol (see `distributed.rs`).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::error::MPCResult;
use crate::field::Fr;
use super::triple::{BeaverTriple, MatrixBeaverTriple, VectorBeaverTriple};

/// A trusted dealer that generates Beaver triples and distributes shares.
#[derive(Debug)]
pub struct TrustedDealer {
    rng: ChaCha20Rng,
}

impl TrustedDealer {
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Generates a random field element in fixed-point format.
    /// This ensures Beaver triples are compatible with fixed-point MPC values.
    fn random_value(&mut self) -> Fr {
        // Generate a random value in a bounded range and convert to fixed-point
        // This keeps the random values in a similar magnitude to typical ML values
        let val: f64 = self.rng.gen_range(-1000.0..1000.0);
        Fr::from_f64(val)
    }

    /// Generates shares of a single scalar Beaver triple for n parties.
    ///
    /// Returns a Vec of length n, where `result[i]` is party i's share.
    /// The shares are additive: sum of all a-shares = a, sum of b-shares = b,
    /// sum of c-shares = c, and c = a * b.
    pub fn generate_scalar_triple(
        &mut self,
        num_parties: usize,
    ) -> Vec<BeaverTriple> {
        let a = self.random_value();
        let b = self.random_value();
        // Use fixed_mul for proper fixed-point arithmetic
        let c = a.fixed_mul(&b);

        self.additive_share_triple(&a, &b, &c, num_parties)
    }

    /// Generates a batch of scalar Beaver triples.
    pub fn generate_scalar_triples(
        &mut self,
        count: usize,
        num_parties: usize,
    ) -> Vec<Vec<BeaverTriple>> {
        // result[party_idx][triple_idx]
        let mut per_party: Vec<Vec<BeaverTriple>> = (0..num_parties)
            .map(|_| Vec::with_capacity(count))
            .collect();

        for _ in 0..count {
            let shares = self.generate_scalar_triple(num_parties);
            for (i, share) in shares.into_iter().enumerate() {
                per_party[i].push(share);
            }
        }

        per_party
    }

    /// Generates shares of a vector Beaver triple.
    pub fn generate_vector_triple(
        &mut self,
        dim: usize,
        num_parties: usize,
    ) -> Vec<VectorBeaverTriple> {
        let a: Vec<Fr> = (0..dim).map(|_| self.random_value()).collect();
        let b: Vec<Fr> = (0..dim).map(|_| self.random_value()).collect();
        // Use fixed_mul for proper fixed-point arithmetic
        let c: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| x.fixed_mul(y)).collect();

        self.additive_share_vector_triple(&a, &b, &c, dim, num_parties)
    }

    /// Generates shares of a matrix Beaver triple for matmul A[m,k] @ B[k,n] = C[m,n].
    pub fn generate_matrix_triple(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        num_parties: usize,
    ) -> Vec<MatrixBeaverTriple> {
        // Generate random matrices A[m,k] and B[k,n].
        let a: Vec<Fr> = (0..m * k).map(|_| self.random_value()).collect();
        let b: Vec<Fr> = (0..k * n).map(|_| self.random_value()).collect();

        // Compute C = A @ B using fixed_mul for proper fixed-point arithmetic.
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

        self.additive_share_matrix_triple(&a, &b, &c, m, k, n, num_parties)
    }

    /// Generates a batch of matrix triples for the same dimensions.
    pub fn generate_matrix_triples(
        &mut self,
        count: usize,
        m: usize,
        k: usize,
        n: usize,
        num_parties: usize,
    ) -> Vec<Vec<MatrixBeaverTriple>> {
        let mut per_party: Vec<Vec<MatrixBeaverTriple>> = (0..num_parties)
            .map(|_| Vec::with_capacity(count))
            .collect();

        for _ in 0..count {
            let shares = self.generate_matrix_triple(m, k, n, num_parties);
            for (i, share) in shares.into_iter().enumerate() {
                per_party[i].push(share);
            }
        }

        per_party
    }

    /// Creates additive shares of a scalar triple.
    fn additive_share_triple(
        &mut self,
        a: &Fr,
        b: &Fr,
        c: &Fr,
        n: usize,
    ) -> Vec<BeaverTriple> {
        let mut shares = Vec::with_capacity(n);
        let mut a_sum = Fr::ZERO;
        let mut b_sum = Fr::ZERO;
        let mut c_sum = Fr::ZERO;

        for _ in 0..n - 1 {
            let ai = self.random_value();
            let bi = self.random_value();
            let ci = self.random_value();
            a_sum = Fr::add(&a_sum, &ai);
            b_sum = Fr::add(&b_sum, &bi);
            c_sum = Fr::add(&c_sum, &ci);
            shares.push(BeaverTriple::new(ai, bi, ci));
        }

        // Last share ensures sums are correct.
        shares.push(BeaverTriple::new(
            Fr::sub(a, &a_sum),
            Fr::sub(b, &b_sum),
            Fr::sub(c, &c_sum),
        ));
        shares
    }

    /// Creates additive shares of a vector triple.
    fn additive_share_vector_triple(
        &mut self,
        a: &[Fr],
        b: &[Fr],
        c: &[Fr],
        dim: usize,
        n: usize,
    ) -> Vec<VectorBeaverTriple> {
        let mut shares: Vec<VectorBeaverTriple> = (0..n)
            .map(|_| VectorBeaverTriple::new(
                vec![Fr::ZERO; dim],
                vec![Fr::ZERO; dim],
                vec![Fr::ZERO; dim],
            ))
            .collect();

        for d in 0..dim {
            let scalar_shares = self.additive_share_triple(&a[d], &b[d], &c[d], n);
            for (i, s) in scalar_shares.into_iter().enumerate() {
                shares[i].a[d] = s.a;
                shares[i].b[d] = s.b;
                shares[i].c[d] = s.c;
            }
        }

        shares
    }

    /// Creates additive shares of a matrix triple.
    fn additive_share_matrix_triple(
        &mut self,
        a: &[Fr],
        b: &[Fr],
        c: &[Fr],
        m: usize,
        k: usize,
        n: usize,
        num_parties: usize,
    ) -> Vec<MatrixBeaverTriple> {
        let mut shares: Vec<MatrixBeaverTriple> = (0..num_parties)
            .map(|_| {
                MatrixBeaverTriple::new(
                    vec![Fr::ZERO; m * k],
                    vec![Fr::ZERO; k * n],
                    vec![Fr::ZERO; m * n],
                    m,
                    k,
                    n,
                )
            })
            .collect();

        // Share each element of A.
        for idx in 0..m * k {
            let s = self.additive_share_scalar(&a[idx], num_parties);
            for (i, val) in s.into_iter().enumerate() {
                shares[i].a[idx] = val;
            }
        }

        // Share each element of B.
        for idx in 0..k * n {
            let s = self.additive_share_scalar(&b[idx], num_parties);
            for (i, val) in s.into_iter().enumerate() {
                shares[i].b[idx] = val;
            }
        }

        // Share each element of C.
        for idx in 0..m * n {
            let s = self.additive_share_scalar(&c[idx], num_parties);
            for (i, val) in s.into_iter().enumerate() {
                shares[i].c[idx] = val;
            }
        }

        shares
    }

    /// Simple additive share of a single scalar into n parts.
    fn additive_share_scalar(&mut self, value: &Fr, n: usize) -> Vec<Fr> {
        let mut shares = Vec::with_capacity(n);
        let mut sum = Fr::ZERO;
        for _ in 0..n - 1 {
            let r = self.random_value();
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        shares.push(Fr::sub(value, &sum));
        shares
    }
}

impl Default for TrustedDealer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ops::sum;

    #[test]
    fn test_scalar_triple_correctness() {
        let mut dealer = TrustedDealer::with_seed(42);
        let shares = dealer.generate_scalar_triple(3);

        // Sum of a-shares, b-shares, c-shares should satisfy c = a*b.
        let a = sum(&shares.iter().map(|s| s.a.clone()).collect::<Vec<_>>());
        let b = sum(&shares.iter().map(|s| s.b.clone()).collect::<Vec<_>>());
        let c = sum(&shares.iter().map(|s| s.c.clone()).collect::<Vec<_>>());

        // Use fixed_mul for consistent fixed-point arithmetic
        let expected_c = a.fixed_mul(&b);
        assert!(
            c.ct_eq(&expected_c).to_bool(),
            "Triple incorrect: c != a*b",
        );
    }

    #[test]
    fn test_scalar_triple_batch() {
        let mut dealer = TrustedDealer::with_seed(42);
        let per_party = dealer.generate_scalar_triples(100, 3);

        assert_eq!(per_party.len(), 3);
        assert_eq!(per_party[0].len(), 100);

        // Verify each triple.
        for idx in 0..100 {
            let a = sum(&per_party.iter().map(|p| p[idx].a.clone()).collect::<Vec<_>>());
            let b = sum(&per_party.iter().map(|p| p[idx].b.clone()).collect::<Vec<_>>());
            let c = sum(&per_party.iter().map(|p| p[idx].c.clone()).collect::<Vec<_>>());

            // Use fixed_mul for consistent fixed-point arithmetic
            let expected_c = a.fixed_mul(&b);
            assert!(
                c.ct_eq(&expected_c).to_bool(),
                "Triple {} incorrect",
                idx,
            );
        }
    }

    #[test]
    fn test_vector_triple() {
        let mut dealer = TrustedDealer::with_seed(42);
        let dim = 5;
        let shares = dealer.generate_vector_triple(dim, 3);

        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].dim, dim);

        for d in 0..dim {
            let a = sum(&shares.iter().map(|s| s.a[d].clone()).collect::<Vec<_>>());
            let b = sum(&shares.iter().map(|s| s.b[d].clone()).collect::<Vec<_>>());
            let c = sum(&shares.iter().map(|s| s.c[d].clone()).collect::<Vec<_>>());

            // Use fixed_mul for consistent fixed-point arithmetic
            let expected_c = a.fixed_mul(&b);
            assert!(
                c.ct_eq(&expected_c).to_bool(),
                "Vector triple[{}] incorrect",
                d,
            );
        }
    }

    #[test]
    fn test_matrix_triple() {
        let mut dealer = TrustedDealer::with_seed(42);
        let (m, k, n) = (2, 3, 2);
        let shares = dealer.generate_matrix_triple(m, k, n, 3);

        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].m, m);
        assert_eq!(shares[0].k, k);
        assert_eq!(shares[0].n, n);

        // Reconstruct A, B, C.
        let mut a = vec![Fr::ZERO; m * k];
        let mut b = vec![Fr::ZERO; k * n];
        let mut c = vec![Fr::ZERO; m * n];

        for s in &shares {
            for i in 0..m * k {
                a[i] = Fr::add(&a[i], &s.a[i]);
            }
            for i in 0..k * n {
                b[i] = Fr::add(&b[i], &s.b[i]);
            }
            for i in 0..m * n {
                c[i] = Fr::add(&c[i], &s.c[i]);
            }
        }

        // Verify C = A @ B using fixed_mul.
        for i in 0..m {
            for j in 0..n {
                let mut expected = Fr::ZERO;
                for l in 0..k {
                    expected = Fr::add(&expected, &a[i * k + l].fixed_mul(&b[l * n + j]));
                }
                assert!(
                    c[i * n + j].ct_eq(&expected).to_bool(),
                    "Matrix triple [{},{}] incorrect",
                    i,
                    j,
                );
            }
        }
    }

    #[test]
    fn test_two_party_triple() {
        let mut dealer = TrustedDealer::with_seed(42);
        let shares = dealer.generate_scalar_triple(2);

        let a = Fr::add(&shares[0].a, &shares[1].a);
        let b = Fr::add(&shares[0].b, &shares[1].b);
        let c = Fr::add(&shares[0].c, &shares[1].c);

        // Use fixed_mul for consistent fixed-point arithmetic
        let expected_c = a.fixed_mul(&b);
        assert!(c.ct_eq(&expected_c).to_bool());
    }
}
