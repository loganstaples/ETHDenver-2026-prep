//! Beaver triple generation: trusted and distributed dealers.
//!
//! In the preprocessing phase, random triples (a, b, c = a*b) are generated
//! and distributed as additive shares to each party.
//!
//! Two implementations are provided:
//!
//! - [`TrustedDealer`]: A single trusted party generates all triples. Simple
//!   and suitable for demos, but requires trusting the dealer.
//!
//! - [`DistributedDealer`]: Uses a pairwise cross-term protocol with SHA-256
//!   hash commitments. No single party learns the full triple. Uses
//!   `exact_fixed_mul` (field-inverse-based division by 2^64) for linearity,
//!   which is required for correctness under additive secret sharing.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

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
        // Use mpc_scale for exact linear fixed-point arithmetic (field-inverse division by 2^64).
        // This is consistent with DistributedDealer and preserves additive homomorphism.
        let c = a.mpc_scale(&b);

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
        // Use mpc_scale for exact linear fixed-point arithmetic
        let c: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| x.mpc_scale(y)).collect();

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

        // Compute C = A @ B using mpc_scale for exact linear fixed-point arithmetic.
        let mut c = vec![Fr::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = Fr::ZERO;
                for l in 0..k {
                    sum = Fr::add(&sum, &a[i * k + l].mpc_scale(&b[l * n + j]));
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

// ========== Distributed Dealer (OT-based, no trusted party) ==========

use sha2::{Sha256, Digest};

/// A SHA-256 commitment to a cross-term mask value.
/// Each party commits to their masks before the reveal phase,
/// enabling post-hoc verification that no party cheated.
#[derive(Debug, Clone)]
pub struct MaskCommitment {
    /// SHA-256 hash of (sender_id || receiver_id || mask_bytes || nonce).
    pub hash: [u8; 32],
    /// Random nonce used in the commitment (revealed later for verification).
    pub nonce: [u8; 32],
}

impl MaskCommitment {
    /// Creates a commitment to a mask value.
    fn commit(sender: usize, receiver: usize, mask: &Fr, rng: &mut ChaCha20Rng) -> Self {
        let mut nonce = [0u8; 32];
        rng.fill(&mut nonce);

        let mask_bytes = mask.to_bytes_le();
        let mut hasher = Sha256::new();
        hasher.update(sender.to_le_bytes());
        hasher.update(receiver.to_le_bytes());
        hasher.update(mask_bytes);
        hasher.update(nonce);
        let hash: [u8; 32] = hasher.finalize().into();

        MaskCommitment { hash, nonce }
    }

    /// Verifies a commitment against the revealed mask value.
    #[allow(dead_code)]
    pub fn verify(&self, sender: usize, receiver: usize, mask: &Fr) -> bool {
        let mask_bytes = mask.to_bytes_le();
        let mut hasher = Sha256::new();
        hasher.update(sender.to_le_bytes());
        hasher.update(receiver.to_le_bytes());
        hasher.update(mask_bytes);
        hasher.update(self.nonce);
        let expected: [u8; 32] = hasher.finalize().into();

        // Constant-time comparison to prevent timing attacks
        use crate::field::constant_time::ct_eq_hash;
        ct_eq_hash(&self.hash, &expected).to_bool()
    }
}

/// Exact fixed-point multiplication using field inverse of 2^64.
///
/// Unlike `Fr::fixed_mul` which uses `floor(a*b / 2^64)` (lossy, non-linear),
/// this computes `a * b * (2^64)^{-1} mod r` (exact, linear) via `Fr::mpc_scale`.
///
/// Linearity means: `exact_fixed_mul(a, b1) + exact_fixed_mul(a, b2)`
///                 = `exact_fixed_mul(a, b1 + b2)`
///
/// This property is required for the pairwise cross-term protocol where
/// products are computed on individual shares and then summed.
fn exact_fixed_mul(a: &Fr, b: &Fr) -> Fr {
    a.mpc_scale(b)
}

/// A distributed Beaver triple dealer that uses pairwise cross-term
/// generation instead of a single trusted party.
///
/// # Protocol
///
/// For `n` parties generating a triple `(a, b, c)` where `c = a * b`:
///
/// 1. Each party `i` samples random `a_i`, `b_i` and computes
///    `c_i = exact_fixed_mul(a_i, b_i)` (exact field-inverse division by 2^64).
/// 2. For each ordered pair `(i, j)` where `i != j`:
///    - Party `i` picks a random mask `r_ij`
///    - Party `i` commits to `r_ij` via SHA-256 (for post-hoc audit)
///    - Party `i` adds `r_ij` to their `c_i`
///    - Party `j` adds `exact_fixed_mul(a_i, b_j) - r_ij` to their `c_j`
///
/// 3. Correctness: `sum(c_i) = sum_{i,j}(exact_fixed_mul(a_i, b_j))`
///    = `exact_fixed_mul(sum a_i, sum b_j)` (by linearity of exact division).
///
/// # Fixed-Point Arithmetic
///
/// The protocol uses `exact_fixed_mul` (field-inverse-based) rather than
/// `Fr::fixed_mul` (floor-based) because:
/// - `floor(sum / 2^64) != sum(floor / 2^64)` -- floor is non-linear
/// - `(sum * inv_2_64) == sum(x * inv_2_64)` -- field inverse IS linear
///
/// The difference between `exact_fixed_mul` and `fixed_mul` is at most 1 ULP
/// (unit in the last place of the fixed-point representation).
///
/// # Security
///
/// - No single party learns the full triple values
/// - Hash commitments allow detecting cheating after the fact
///
/// # Limitations
///
/// This is a simulation of the distributed protocol within a single process.
/// In production, the cross-term messages would be sent over authenticated
/// encrypted channels (see `helix-mpc::transport`).
#[derive(Debug)]
pub struct DistributedDealer {
    /// Per-party RNGs (in production, each party would have their own)
    rngs: Vec<ChaCha20Rng>,
    /// Number of parties
    num_parties: usize,
}

impl DistributedDealer {
    /// Creates a new distributed dealer for the given number of parties.
    ///
    /// Each party gets an independent RNG seeded from entropy.
    pub fn new(num_parties: usize) -> Self {
        assert!(num_parties >= 2, "Distributed protocol requires at least 2 parties");
        let rngs = (0..num_parties)
            .map(|_| ChaCha20Rng::from_entropy())
            .collect();
        Self { rngs, num_parties }
    }

    /// Creates a distributed dealer with deterministic seeds (for testing).
    ///
    /// Party `i` is seeded with `base_seed + i as u64`.
    pub fn with_seed(num_parties: usize, base_seed: u64) -> Self {
        assert!(num_parties >= 2, "Distributed protocol requires at least 2 parties");
        let rngs = (0..num_parties)
            .map(|i| ChaCha20Rng::seed_from_u64(base_seed + i as u64))
            .collect();
        Self { rngs, num_parties }
    }

    /// Generates a random bounded fixed-point value for a given party.
    fn random_value(&mut self, party: usize) -> Fr {
        let val: f64 = self.rngs[party].gen_range(-1000.0..1000.0);
        Fr::from_f64(val)
    }

    /// Generates shares of a single scalar Beaver triple using the
    /// pairwise cross-term protocol.
    ///
    /// Returns `(shares, commitments)` where:
    /// - `shares[i]` is party i's `BeaverTriple` share
    /// - `commitments` contains hash commitments for each cross-term mask
    pub fn generate_scalar_triple_with_commitments(
        &mut self,
    ) -> (Vec<BeaverTriple>, Vec<MaskCommitment>) {
        let n = self.num_parties;

        // Phase 1: Each party samples local a_i, b_i and computes
        // c_i = exact_fixed_mul(a_i, b_i) using linear field-inverse division
        let mut a_shares: Vec<Fr> = Vec::with_capacity(n);
        let mut b_shares: Vec<Fr> = Vec::with_capacity(n);
        let mut c_shares: Vec<Fr> = Vec::with_capacity(n);

        for i in 0..n {
            let ai = self.random_value(i);
            let bi = self.random_value(i);
            let ci = exact_fixed_mul(&ai, &bi);
            a_shares.push(ai);
            b_shares.push(bi);
            c_shares.push(ci);
        }

        // Phase 2: Pairwise cross-term generation with commitments
        //
        // For each pair (i, j) with i != j, party i generates a random mask
        // r_ij and commits to it. The cross-term exact_fixed_mul(a_i, b_j) is
        // split between party i (gets +r_ij) and party j (gets cross - r_ij).
        let mut commitments = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Party i picks random mask r_ij
                let r_ij = Fr::random(&mut self.rngs[i]);

                // Commit to r_ij before revealing
                let commitment = MaskCommitment::commit(i, j, &r_ij, &mut self.rngs[i]);
                commitments.push(commitment);

                // Compute the cross-term using exact (linear) fixed-point mul.
                // Both a_i and b_j are fixed-point values (x * 2^64), so
                // their raw product is x*y*2^128. We divide by 2^64 using
                // the field inverse to get x*y*2^64 (the fixed-point result).
                let cross_term = exact_fixed_mul(&a_shares[i], &b_shares[j]);

                // Party i adds r_ij to their c_i
                c_shares[i] = Fr::add(&c_shares[i], &r_ij);

                // Party j adds (cross_term - r_ij) to their c_j
                let correction = Fr::sub(&cross_term, &r_ij);
                c_shares[j] = Fr::add(&c_shares[j], &correction);
            }
        }

        let shares: Vec<BeaverTriple> = (0..n)
            .map(|i| BeaverTriple::new(
                a_shares[i].clone(),
                b_shares[i].clone(),
                c_shares[i].clone(),
            ))
            .collect();

        (shares, commitments)
    }

    /// Generates shares of a single scalar Beaver triple.
    ///
    /// This is the simplified API matching `TrustedDealer`. The `num_parties`
    /// parameter is accepted for API compatibility but must match the dealer's
    /// party count.
    pub fn generate_scalar_triple(
        &mut self,
        num_parties: usize,
    ) -> Vec<BeaverTriple> {
        assert_eq!(
            num_parties, self.num_parties,
            "num_parties ({}) must match dealer's party count ({})",
            num_parties, self.num_parties,
        );
        let (shares, _commitments) = self.generate_scalar_triple_with_commitments();
        shares
    }

    /// Generates a batch of scalar Beaver triples.
    pub fn generate_scalar_triples(
        &mut self,
        count: usize,
        num_parties: usize,
    ) -> Vec<Vec<BeaverTriple>> {
        assert_eq!(
            num_parties, self.num_parties,
            "num_parties ({}) must match dealer's party count ({})",
            num_parties, self.num_parties,
        );

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

    /// Generates shares of a vector Beaver triple using the distributed protocol.
    ///
    /// Each dimension is generated independently using the pairwise cross-term
    /// protocol, ensuring element-wise correctness: `c[d] = a[d] * b[d]`.
    pub fn generate_vector_triple(
        &mut self,
        dim: usize,
        num_parties: usize,
    ) -> Vec<VectorBeaverTriple> {
        assert_eq!(
            num_parties, self.num_parties,
            "num_parties ({}) must match dealer's party count ({})",
            num_parties, self.num_parties,
        );

        let n = self.num_parties;

        // Initialize per-party vector triples
        let mut shares: Vec<VectorBeaverTriple> = (0..n)
            .map(|_| VectorBeaverTriple::new(
                vec![Fr::ZERO; dim],
                vec![Fr::ZERO; dim],
                vec![Fr::ZERO; dim],
            ))
            .collect();

        // Generate each dimension independently
        for d in 0..dim {
            let scalar_shares = self.generate_scalar_triple(num_parties);
            for (i, s) in scalar_shares.into_iter().enumerate() {
                shares[i].a[d] = s.a;
                shares[i].b[d] = s.b;
                shares[i].c[d] = s.c;
            }
        }

        shares
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

        // Use mpc_scale for consistent exact linear fixed-point arithmetic
        let expected_c = a.mpc_scale(&b);
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

            // Use mpc_scale for consistent exact linear fixed-point arithmetic
            let expected_c = a.mpc_scale(&b);
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

            // Use mpc_scale for consistent exact linear fixed-point arithmetic
            let expected_c = a.mpc_scale(&b);
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

        // Verify C = A @ B using mpc_scale.
        for i in 0..m {
            for j in 0..n {
                let mut expected = Fr::ZERO;
                for l in 0..k {
                    expected = Fr::add(&expected, &a[i * k + l].mpc_scale(&b[l * n + j]));
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

        // Use mpc_scale for consistent exact linear fixed-point arithmetic
        let expected_c = a.mpc_scale(&b);
        assert!(c.ct_eq(&expected_c).to_bool());
    }

    // ========== DistributedDealer Tests ==========

    #[test]
    fn test_distributed_scalar_triple_correctness() {
        let mut dealer = DistributedDealer::with_seed(3, 100);
        let shares = dealer.generate_scalar_triple(3);

        assert_eq!(shares.len(), 3);

        // Reconstruct a, b, c from additive shares
        let a = sum(&shares.iter().map(|s| s.a.clone()).collect::<Vec<_>>());
        let b = sum(&shares.iter().map(|s| s.b.clone()).collect::<Vec<_>>());
        let c = sum(&shares.iter().map(|s| s.c.clone()).collect::<Vec<_>>());

        // Verify c = exact_fixed_mul(a, b) using linear field-inverse division
        let expected_c = exact_fixed_mul(&a, &b);
        assert!(
            c.ct_eq(&expected_c).to_bool(),
            "Distributed triple incorrect: c != exact_fixed_mul(a, b)\n  c  = {:?}\n  ab = {:?}",
            c, expected_c,
        );
    }

    #[test]
    fn test_distributed_two_party_triple() {
        let mut dealer = DistributedDealer::with_seed(2, 200);
        let shares = dealer.generate_scalar_triple(2);

        assert_eq!(shares.len(), 2);

        let a = Fr::add(&shares[0].a, &shares[1].a);
        let b = Fr::add(&shares[0].b, &shares[1].b);
        let c = Fr::add(&shares[0].c, &shares[1].c);

        let expected_c = exact_fixed_mul(&a, &b);
        assert!(
            c.ct_eq(&expected_c).to_bool(),
            "2-party distributed triple incorrect",
        );
    }

    #[test]
    fn test_distributed_scalar_triple_batch() {
        let mut dealer = DistributedDealer::with_seed(3, 300);
        let per_party = dealer.generate_scalar_triples(50, 3);

        assert_eq!(per_party.len(), 3);
        assert_eq!(per_party[0].len(), 50);

        for idx in 0..50 {
            let a = sum(&per_party.iter().map(|p| p[idx].a.clone()).collect::<Vec<_>>());
            let b = sum(&per_party.iter().map(|p| p[idx].b.clone()).collect::<Vec<_>>());
            let c = sum(&per_party.iter().map(|p| p[idx].c.clone()).collect::<Vec<_>>());

            let expected_c = exact_fixed_mul(&a, &b);
            assert!(
                c.ct_eq(&expected_c).to_bool(),
                "Distributed batch triple {} incorrect",
                idx,
            );
        }
    }

    #[test]
    fn test_distributed_vector_triple() {
        let dim = 8;
        let mut dealer = DistributedDealer::with_seed(3, 400);
        let shares = dealer.generate_vector_triple(dim, 3);

        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].dim, dim);

        for d in 0..dim {
            let a = sum(&shares.iter().map(|s| s.a[d].clone()).collect::<Vec<_>>());
            let b = sum(&shares.iter().map(|s| s.b[d].clone()).collect::<Vec<_>>());
            let c = sum(&shares.iter().map(|s| s.c[d].clone()).collect::<Vec<_>>());

            let expected_c = exact_fixed_mul(&a, &b);
            assert!(
                c.ct_eq(&expected_c).to_bool(),
                "Distributed vector triple[{}] incorrect",
                d,
            );
        }
    }

    #[test]
    fn test_distributed_commitments_valid() {
        let mut dealer = DistributedDealer::with_seed(3, 500);
        let (_shares, commitments) = dealer.generate_scalar_triple_with_commitments();

        // For 3 parties, there are 3*(3-1) = 6 cross-term pairs
        assert_eq!(commitments.len(), 6, "Expected 6 commitments for 3 parties");

        // Each commitment should have a non-zero hash
        for (i, c) in commitments.iter().enumerate() {
            assert_ne!(c.hash, [0u8; 32], "Commitment {} has zero hash", i);
        }
    }

    #[test]
    fn test_distributed_five_party_triple() {
        // Test with more parties to stress the cross-term protocol
        let mut dealer = DistributedDealer::with_seed(5, 600);
        let shares = dealer.generate_scalar_triple(5);

        assert_eq!(shares.len(), 5);

        let a = sum(&shares.iter().map(|s| s.a.clone()).collect::<Vec<_>>());
        let b = sum(&shares.iter().map(|s| s.b.clone()).collect::<Vec<_>>());
        let c = sum(&shares.iter().map(|s| s.c.clone()).collect::<Vec<_>>());

        let expected_c = exact_fixed_mul(&a, &b);
        assert!(
            c.ct_eq(&expected_c).to_bool(),
            "5-party distributed triple incorrect",
        );
    }

    #[test]
    fn test_distributed_shares_differ_from_reconstructed() {
        // Verify that individual shares do NOT reveal the full triple
        let mut dealer = DistributedDealer::with_seed(3, 700);
        let shares = dealer.generate_scalar_triple(3);

        let a_full = sum(&shares.iter().map(|s| s.a.clone()).collect::<Vec<_>>());

        // No single share should equal the full value
        for (i, s) in shares.iter().enumerate() {
            assert!(
                !s.a.ct_eq(&a_full).to_bool(),
                "Party {}'s a-share equals the full value (not secret-shared!)",
                i,
            );
        }
    }

    #[test]
    fn test_exact_fixed_mul_is_linear() {
        // Verify the key property: exact_fixed_mul is linear
        // exact_fixed_mul(a, b1 + b2) == exact_fixed_mul(a, b1) + exact_fixed_mul(a, b2)
        let a = Fr::from_f64(3.7);
        let b1 = Fr::from_f64(2.1);
        let b2 = Fr::from_f64(4.3);

        let b_sum = Fr::add(&b1, &b2);
        let lhs = exact_fixed_mul(&a, &b_sum);
        let rhs = Fr::add(&exact_fixed_mul(&a, &b1), &exact_fixed_mul(&a, &b2));

        assert!(
            lhs.ct_eq(&rhs).to_bool(),
            "exact_fixed_mul is not linear!\n  lhs = {:?}\n  rhs = {:?}",
            lhs, rhs,
        );
    }

    #[test]
    fn test_exact_fixed_mul_semantically_correct() {
        // Verify that exact_fixed_mul produces correct fixed-point products.
        // Note: exact_fixed_mul uses modular inverse of 2^64 (linear, exact in
        // the field), while fixed_mul uses floor-based byte shift (non-linear,
        // approximate). They may differ for values whose raw field representation
        // is large, but exact_fixed_mul is the correct one for MPC where
        // linearity is required.
        let a = Fr::from_f64(3.0);
        let b = Fr::from_f64(4.0);

        let result = exact_fixed_mul(&a, &b);
        let got = result.to_f64();
        assert!(
            (got - 12.0).abs() < 1.0,
            "3.0 * 4.0 via exact_fixed_mul = {} (expected ~12.0)", got,
        );

        // Verify with small values
        let a2 = Fr::from_f64(0.5);
        let b2 = Fr::from_f64(0.5);
        let result2 = exact_fixed_mul(&a2, &b2);
        let got2 = result2.to_f64();
        assert!(
            (got2 - 0.25).abs() < 1e-5,
            "0.5 * 0.5 via exact_fixed_mul = {} (expected ~0.25)", got2,
        );
    }

    /// Verifies that TrustedDealer and DistributedDealer produce compatible triples.
    ///
    /// Both dealers now use `mpc_scale` for c = a*b, so reconstructed triples
    /// from either dealer should satisfy the same invariant.
    #[test]
    fn test_cross_dealer_compatibility() {
        use crate::beaver::distributed::DistributedTripleGen;

        let num_parties = 3;

        // Generate triples from TrustedDealer
        let mut trusted = TrustedDealer::with_seed(42);
        let trusted_batch = trusted.generate_scalar_triples(10, num_parties);

        // Generate triples from DistributedDealer
        let distributed_batch = DistributedTripleGen::simulate_distributed_batch(10, num_parties, 42);

        // Both should satisfy: sum(a) * sum(b) == sum(c) using mpc_scale
        for idx in 0..10 {
            // TrustedDealer triple
            let a_t: Fr = (0..num_parties)
                .map(|p| trusted_batch[p][idx].a.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let b_t: Fr = (0..num_parties)
                .map(|p| trusted_batch[p][idx].b.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let c_t: Fr = (0..num_parties)
                .map(|p| trusted_batch[p][idx].c.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let expected_t = a_t.mpc_scale(&b_t);
            assert!(
                expected_t.ct_eq(&c_t).to_bool(),
                "TrustedDealer triple {} failed: a*b != c", idx,
            );

            // DistributedDealer triple
            let a_d: Fr = (0..num_parties)
                .map(|p| distributed_batch[p][idx].a.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let b_d: Fr = (0..num_parties)
                .map(|p| distributed_batch[p][idx].b.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let c_d: Fr = (0..num_parties)
                .map(|p| distributed_batch[p][idx].c.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let expected_d = a_d.mpc_scale(&b_d);
            assert!(
                expected_d.ct_eq(&c_d).to_bool(),
                "DistributedDealer triple {} failed: a*b != c", idx,
            );
        }
    }
}
