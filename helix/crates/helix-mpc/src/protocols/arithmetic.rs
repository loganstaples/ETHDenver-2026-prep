//! Secure arithmetic protocols on secret-shared values.
//!
//! # Operations
//!
//! **Local (no communication):**
//! - Addition of two shared values: [x] + [y] → each party computes x_i + y_i
//! - Subtraction: [x] - [y] → each party computes x_i - y_i
//! - Scaling by public constant c: c * [x] → each party computes c * x_i
//! - Addition of public constant c: [x] + c → party 0 computes x_0 + c, others unchanged
//!
//! **Interactive (requires communication + Beaver triples):**
//! - Multiplication: [x] * [y] → Beaver triple protocol
//! - Division by shared value: [x] / [y] → via Newton's method on shares
//! - Element-wise vector operations

use crate::beaver::pool::BeaverPool;
use crate::beaver::triple::BeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::sharing::tensor::TensorShare;

/// Secure arithmetic operations on shares.
///
/// Each method takes the party's local share(s) and returns the party's
/// share of the result. Communication happens through the `open` callback
/// which reconstructs public values needed by the protocol.
pub struct SecureArithmetic;

impl SecureArithmetic {
    // ========== LOCAL OPERATIONS (no communication) ==========

    /// Adds two scalar shares locally: [x+y]_i = [x]_i + [y]_i.
    pub fn add_shares(x_share: f64, y_share: f64) -> f64 {
        x_share + y_share
    }

    /// Subtracts two scalar shares locally: [x-y]_i = [x]_i - [y]_i.
    pub fn sub_shares(x_share: f64, y_share: f64) -> f64 {
        x_share - y_share
    }

    /// Multiplies a share by a public constant: [c*x]_i = c * [x]_i.
    pub fn scale_share(x_share: f64, public_constant: f64) -> f64 {
        x_share * public_constant
    }

    /// Adds a public constant to a share.
    /// Only party 0 adds the constant; others leave unchanged.
    pub fn add_public(x_share: f64, public_constant: f64, party_index: usize) -> f64 {
        if party_index == 0 {
            x_share + public_constant
        } else {
            x_share
        }
    }

    /// Adds two tensor shares element-wise (local operation).
    pub fn add_tensor_shares(x: &TensorShare, y: &TensorShare) -> MPCResult<TensorShare> {
        x.add(y)
    }

    /// Subtracts two tensor shares element-wise (local operation).
    pub fn sub_tensor_shares(x: &TensorShare, y: &TensorShare) -> MPCResult<TensorShare> {
        x.sub(y)
    }

    /// Scales a tensor share by a public constant (local operation).
    pub fn scale_tensor_share(x: &TensorShare, scalar: f64) -> TensorShare {
        x.scale(scalar)
    }

    // ========== INTERACTIVE OPERATIONS (require communication) ==========

    /// Secure scalar multiplication using a Beaver triple.
    ///
    /// Protocol:
    /// 1. Consume triple ([a], [b], [c]) from pool
    /// 2. Compute [d] = [x] - [a], [e] = [y] - [b] locally
    /// 3. Open d and e (public values, reveal nothing about x, y)
    /// 4. Compute [x*y] = [c] + d*[b] + e*[a] + d*e (only party 0 adds d*e)
    ///
    /// The `opened_d` and `opened_e` parameters are the reconstructed values
    /// of d and e. In a real system, parties would send their d/e shares to
    /// each other and sum them.
    pub fn multiply_shares(
        x_share: f64,
        y_share: f64,
        triple: &BeaverTriple,
        opened_d: f64,
        opened_e: f64,
        party_index: usize,
    ) -> f64 {
        // [xy] = [c] + d*[b] + e*[a] + d*e (d*e only added by party 0)
        let mut result = triple.c + opened_d * triple.b + opened_e * triple.a;
        if party_index == 0 {
            result += opened_d * opened_e;
        }
        result
    }

    /// Computes the d and e values needed for Beaver multiplication.
    /// These are the party's shares of (x - a) and (y - b).
    /// All parties send these to each other and sum to get opened_d, opened_e.
    pub fn beaver_mask(
        x_share: f64,
        y_share: f64,
        triple: &BeaverTriple,
    ) -> (f64, f64) {
        let d_share = x_share - triple.a;
        let e_share = y_share - triple.b;
        (d_share, e_share)
    }

    /// Simulates the full Beaver multiplication protocol for all parties.
    ///
    /// Takes all parties' shares and triples, returns all parties' result shares.
    /// This is used for testing; in production each party runs independently.
    pub fn simulate_multiply(
        x_shares: &[f64],
        y_shares: &[f64],
        triples: &[BeaverTriple],
    ) -> Vec<f64> {
        let n = x_shares.len();
        assert_eq!(y_shares.len(), n);
        assert_eq!(triples.len(), n);

        // Step 1: Each party computes d_i, e_i.
        let d_shares: Vec<f64> = x_shares
            .iter()
            .zip(triples)
            .map(|(x, t)| x - t.a)
            .collect();

        let e_shares: Vec<f64> = y_shares
            .iter()
            .zip(triples)
            .map(|(y, t)| y - t.b)
            .collect();

        // Step 2: Reconstruct d and e (all parties see these).
        let d: f64 = d_shares.iter().sum();
        let e: f64 = e_shares.iter().sum();

        // Step 3: Each party computes their result share.
        (0..n)
            .map(|i| Self::multiply_shares(x_shares[i], y_shares[i], &triples[i], d, e, i))
            .collect()
    }

    /// Simulates element-wise multiplication of two shared vectors.
    pub fn simulate_vector_multiply(
        x_shares: &[Vec<f64>],
        y_shares: &[Vec<f64>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
        let n = x_shares.len(); // number of parties
        let dim = x_shares[0].len();

        let mut result: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];

        for elem in 0..dim {
            // Get triples for this element.
            let triples: Vec<BeaverTriple> = pools
                .iter_mut()
                .map(|p| p.take_scalar())
                .collect::<MPCResult<Vec<_>>>()?;

            let x_elem: Vec<f64> = x_shares.iter().map(|s| s[elem]).collect();
            let y_elem: Vec<f64> = y_shares.iter().map(|s| s[elem]).collect();

            let prod = Self::simulate_multiply(&x_elem, &y_elem, &triples);

            for (i, val) in prod.into_iter().enumerate() {
                result[i][elem] = val;
            }
        }

        Ok(result)
    }

    /// Secure reciprocal approximation: computes [1/x] from [x].
    ///
    /// Uses Newton's method: r_{n+1} = r_n * (2 - x * r_n)
    /// Starting from a public initial guess.
    pub fn simulate_reciprocal(
        x_shares: &[f64],
        initial_guess: f64,
        iterations: usize,
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        let n = x_shares.len();

        // Initialize r = initial_guess (public).
        let mut r_shares: Vec<f64> = (0..n)
            .map(|i| if i == 0 { initial_guess } else { 0.0 })
            .collect();

        for _ in 0..iterations {
            // Compute [x * r] using Beaver multiplication.
            let triples: Vec<BeaverTriple> = pools
                .iter_mut()
                .map(|p| p.take_scalar())
                .collect::<MPCResult<Vec<_>>>()?;

            let xr = Self::simulate_multiply(x_shares, &r_shares, &triples);

            // Compute [2 - x*r]: subtract from public 2.
            let two_minus_xr: Vec<f64> = xr
                .iter()
                .enumerate()
                .map(|(i, v)| Self::add_public(-v, 2.0, i))
                .collect();

            // Compute [r * (2 - x*r)] using another Beaver triple.
            let triples2: Vec<BeaverTriple> = pools
                .iter_mut()
                .map(|p| p.take_scalar())
                .collect::<MPCResult<Vec<_>>>()?;

            r_shares = Self::simulate_multiply(&r_shares, &two_minus_xr, &triples2);
        }

        Ok(r_shares)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;

    #[test]
    fn test_local_add() {
        // Shares of x=10: [3, 7], shares of y=5: [2, 3]
        let result_0 = SecureArithmetic::add_shares(3.0, 2.0);
        let result_1 = SecureArithmetic::add_shares(7.0, 3.0);
        assert!((result_0 + result_1 - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_local_scale() {
        let result_0 = SecureArithmetic::scale_share(3.0, 2.5);
        let result_1 = SecureArithmetic::scale_share(7.0, 2.5);
        assert!((result_0 + result_1 - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_public() {
        let r0 = SecureArithmetic::add_public(3.0, 5.0, 0);
        let r1 = SecureArithmetic::add_public(7.0, 5.0, 1);
        assert!((r0 + r1 - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_beaver_multiplication() {
        let mut dealer = TrustedDealer::with_seed(42);

        let x = 7.0;
        let y = 3.0;
        let expected = x * y;

        // Share x and y among 3 parties.
        let x_shares = vec![2.5, -1.3, 5.8]; // sum = 7.0
        let y_shares = vec![1.1, 0.4, 1.5]; // sum = 3.0

        let triples = dealer.generate_scalar_triple(3);

        let result_shares = SecureArithmetic::simulate_multiply(&x_shares, &y_shares, &triples);
        let result: f64 = result_shares.iter().sum();

        assert!(
            (result - expected).abs() < 1e-6,
            "Beaver multiply failed: {} vs {}",
            result,
            expected,
        );
    }

    #[test]
    fn test_beaver_multiplication_many() {
        let mut dealer = TrustedDealer::with_seed(42);

        for (x, y) in &[(1.0, 1.0), (0.0, 5.0), (-3.0, 4.0), (100.0, 0.01)] {
            let x_shares = vec![x / 2.0, x / 3.0, x - x / 2.0 - x / 3.0];
            let y_shares = vec![y / 2.0, y / 3.0, y - y / 2.0 - y / 3.0];

            let triples = dealer.generate_scalar_triple(3);
            let result: f64 = SecureArithmetic::simulate_multiply(&x_shares, &y_shares, &triples)
                .iter()
                .sum();

            assert!(
                (result - x * y).abs() < 1e-4,
                "Multiply {}*{}: expected {}, got {}",
                x,
                y,
                x * y,
                result,
            );
        }
    }

    #[test]
    fn test_beaver_mask() {
        let triple = BeaverTriple::new(5.0, 3.0, 15.0);
        let (d, e) = SecureArithmetic::beaver_mask(7.0, 4.0, &triple);
        assert!((d - 2.0).abs() < 1e-10); // 7 - 5
        assert!((e - 1.0).abs() < 1e-10); // 4 - 3
    }

    #[test]
    fn test_reciprocal() {
        let mut dealer = TrustedDealer::with_seed(42);
        let per_party = dealer.generate_scalar_triples(100, 3);

        let mut pools: Vec<BeaverPool> = (0..3)
            .map(|i| {
                let mut p = BeaverPool::new(i, 3, 64);
                p.fill_scalar(per_party[i].clone());
                p
            })
            .collect();

        let x = 4.0;
        let x_shares = vec![1.5, 0.3, x - 1.5 - 0.3];

        let result_shares =
            SecureArithmetic::simulate_reciprocal(&x_shares, 0.3, 5, &mut pools).unwrap();

        let result: f64 = result_shares.iter().sum();
        assert!(
            (result - 0.25).abs() < 0.01,
            "Reciprocal of {} failed: got {}",
            x,
            result,
        );
    }
}
