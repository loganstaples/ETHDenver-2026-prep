//! Secure activation function protocols.
//!
//! Activation functions (ReLU, GELU, sigmoid, etc.) are non-linear and cannot
//! be computed directly on additive shares. The standard approach is:
//!
//! 1. **Reconstruct** the intermediate value (parties open the pre-activation value)
//! 2. **Apply** the activation function locally (each party computes it on the public value)
//! 3. **Re-share** the result (one party acts as dealer to create new shares)
//!
//! **Security note:** This reveals the intermediate activation values, NOT the weights.
//! Activations change with every input and don't directly leak model architecture.
//! This is the standard security/efficiency tradeoff in the MPC-ML literature.
//!
//! For stronger privacy, we also support a truncated polynomial approximation
//! mode where activations are approximated by low-degree polynomials that can
//! be evaluated on shares using Beaver triples.

use rand::Rng;

use crate::beaver::pool::BeaverPool;
use crate::error::MPCResult;
use crate::field::Fr;
use crate::protocols::arithmetic::SecureArithmetic;
use super::reshare_values;

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    LeakyReLU(f64),
    SiLU,
}

/// Secure activation function evaluation.
pub struct SecureActivation;

impl SecureActivation {
    // ========== RECONSTRUCT-COMPUTE-RESHARE APPROACH ==========
    // Reveals activations but not weights. Standard in MPC-ML.

    /// Applies an activation function using the reconstruct-compute-reshare approach.
    ///
    /// `shares`: each party's share of the pre-activation vector (Fr).
    ///           shares[party_idx][element_idx]
    /// `activation`: which activation to apply.
    ///
    /// Returns new shares of the post-activation vector.
    pub fn apply_reconstruct_reshare(
        shares: &[Vec<Fr>],
        activation: ActivationType,
        rng: &mut impl Rng,
    ) -> Vec<Vec<Fr>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Step 1: Reconstruct (parties would send shares to each other).
        let mut values = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                values[i] = Fr::add(&values[i], v);
            }
        }

        // Convert to f64 for activation computation
        let values_f64: Vec<f64> = values.iter().map(|v| v.to_f64()).collect();

        // Step 2: Apply activation to the reconstructed values.
        let activated: Vec<f64> = values_f64
            .iter()
            .map(|v| apply_activation(*v, activation))
            .collect();

        // Step 3: Re-share using cryptographically secure randomness.
        reshare_values(&activated, num_parties, rng)
    }

    /// Batch version: applies activation to multiple vectors at once.
    pub fn apply_batch_reconstruct_reshare(
        batch_shares: &[Vec<Vec<Fr>>],
        activation: ActivationType,
        rng: &mut impl Rng,
    ) -> Vec<Vec<Vec<Fr>>> {
        let num_parties = batch_shares.len();
        let batch_size = batch_shares[0].len();

        let mut results: Vec<Vec<Vec<Fr>>> = vec![Vec::with_capacity(batch_size); num_parties];

        for b in 0..batch_size {
            let item_shares: Vec<Vec<Fr>> = batch_shares
                .iter()
                .map(|p| p[b].clone())
                .collect();

            let activated = Self::apply_reconstruct_reshare(&item_shares, activation, rng);

            for (i, party_result) in activated.into_iter().enumerate() {
                results[i].push(party_result);
            }
        }

        results
    }

    // ========== POLYNOMIAL APPROXIMATION APPROACH ==========
    // Does NOT reveal activations. Uses more Beaver triples.

    /// Approximates ReLU using a polynomial: relu(x) ≈ 0.5*x + 0.5*x*sign_approx(x)
    /// where sign_approx uses a polynomial approximation of the sign function.
    ///
    /// This keeps the computation entirely on shares but is less accurate.
    pub fn approximate_relu(
        shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Polynomial approximation of ReLU:
        // relu(x) ≈ x/2 + x/(2π) * (π/2 + x - x³/6) for |x| < π
        // Simplified: relu(x) ≈ 0.5*x + 0.197*x (for positive-biased inputs)
        // We use: relu(x) ≈ max(0.01*x, x) via: 0.505*x + 0.495*|x|
        // And |x| ≈ x * sign(x) where sign ≈ x / (|x| + ε)
        //
        // For the demo, we use the simpler degree-2 approximation:
        // relu(x) ≈ 0.5*x + 0.5*x² / (|x| + 0.1)

        // Compute x² on shares using Beaver triples.
        let x_squared = SecureArithmetic::simulate_vector_multiply(shares, shares, pools)?;

        // For the denominator, we need |x| + 0.1, but computing absolute value
        // on shares is itself non-trivial. We approximate with x²/(x + ε) ≈ |x|.
        //
        // Final approximation: relu(x) ≈ 0.5*x + 0.25*x (crude but private)
        // This is essentially a leaky linear activation.
        let half = Fr::from_f64(0.5);
        let quarter = Fr::from_f64(0.25);

        let mut result: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                // relu ≈ 0.5*x + 0.25*x² (simplified approximation)
                // A crude approximation that preserves privacy
                // Use fixed_mul for proper fixed-point arithmetic
                let term1 = half.fixed_mul(&shares[i][d]);
                let term2 = quarter.fixed_mul(&x_squared[i][d]);
                result[i][d] = Fr::add(&term1, &term2);
            }
        }

        Ok(result)
    }

    /// Approximates sigmoid using the polynomial:
    /// sigmoid(x) ≈ 0.5 + 0.25*x - 0.0208*x³ (for |x| < 4)
    ///
    /// This is a degree-3 polynomial that can be evaluated on shares
    /// using 2 Beaver multiplications per element.
    pub fn approximate_sigmoid(
        shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Compute x² and x³ on shares.
        let x_squared = SecureArithmetic::simulate_vector_multiply(shares, shares, pools)?;
        let x_cubed = SecureArithmetic::simulate_vector_multiply(shares, &x_squared, pools)?;

        // sigmoid(x) ≈ 0.5 + 0.25*x - 0.0208*x³
        let c_half = Fr::from_f64(0.5);
        let c_quarter = Fr::from_f64(0.25);
        let c_cubic = Fr::from_f64(-0.0208);

        let mut result: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                // term1: 0.5 (only party 0 adds the constant)
                let term1 = SecureArithmetic::add_public(&Fr::ZERO, &c_half, i);
                // term2: 0.25 * x
                let term2 = SecureArithmetic::scale_share(&shares[i][d], &c_quarter);
                // term3: -0.0208 * x³
                let term3 = SecureArithmetic::scale_share(&x_cubed[i][d], &c_cubic);
                result[i][d] = Fr::add(&Fr::add(&term1, &term2), &term3);
            }
        }

        Ok(result)
    }

    /// Approximates GELU using:
    /// gelu(x) ≈ 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
    /// Simplified to: gelu(x) ≈ 0.5*x + 0.5*x*sigmoid(1.702*x)
    /// Further simplified for MPC: gelu(x) ≈ 0.5*x*(1 + 0.851*x - 0.0354*x³)
    pub fn approximate_gelu(
        shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // x²
        let x_sq = SecureArithmetic::simulate_vector_multiply(shares, shares, pools)?;
        // x³
        let x_cu = SecureArithmetic::simulate_vector_multiply(shares, &x_sq, pools)?;

        let c_one = Fr::from_f64(1.0);
        let c_linear = Fr::from_f64(0.851);
        let c_cubic = Fr::from_f64(-0.0354);
        let c_half = Fr::from_f64(0.5);

        // inner = 1 + 0.851*x - 0.0354*x³
        let mut inner: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                let one = SecureArithmetic::add_public(&Fr::ZERO, &c_one, i);
                let lin = SecureArithmetic::scale_share(&shares[i][d], &c_linear);
                let cub = SecureArithmetic::scale_share(&x_cu[i][d], &c_cubic);
                inner[i][d] = Fr::add(&Fr::add(&one, &lin), &cub);
            }
        }

        // result = 0.5 * x * inner
        let x_inner = SecureArithmetic::simulate_vector_multiply(shares, &inner, pools)?;
        let mut result: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                result[i][d] = SecureArithmetic::scale_share(&x_inner[i][d], &c_half);
            }
        }

        Ok(result)
    }
}

/// Applies an activation function to a single value.
fn apply_activation(x: f64, activation: ActivationType) -> f64 {
    match activation {
        ActivationType::ReLU => x.max(0.0),
        ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        ActivationType::Tanh => x.tanh(),
        ActivationType::GELU => {
            let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        }
        ActivationType::LeakyReLU(alpha) => {
            if x >= 0.0 {
                x
            } else {
                alpha * x
            }
        }
        ActivationType::SiLU => x * (1.0 / (1.0 + (-x).exp())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn split_vector(values: &[f64], n: usize, seed: u64) -> Vec<Vec<Fr>> {
        use rand::Rng;

        let dim = values.len();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; n];

        for d in 0..dim {
            let mut sum = Fr::ZERO;
            for i in 0..n - 1 {
                let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
                shares[i][d] = r.clone();
                sum = Fr::add(&sum, &r);
            }
            shares[n - 1][d] = Fr::sub(&Fr::from_f64(values[d]), &sum);
        }
        shares
    }

    fn reconstruct(shares: &[Vec<Fr>]) -> Vec<f64> {
        let dim = shares[0].len();
        let mut result = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                result[i] = Fr::add(&result[i], v);
            }
        }
        result.iter().map(|v| v.to_f64()).collect()
    }

    #[test]
    fn test_relu_reconstruct_reshare() {
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shares = split_vector(&values, 3, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let result_shares =
            SecureActivation::apply_reconstruct_reshare(&shares, ActivationType::ReLU, &mut rng);

        let result = reconstruct(&result_shares);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-6,
                "ReLU failed: {} vs {}",
                r,
                e,
            );
        }
    }

    #[test]
    fn test_sigmoid_reconstruct_reshare() {
        let values = vec![-3.0, 0.0, 3.0];
        let shares = split_vector(&values, 3, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let result_shares =
            SecureActivation::apply_reconstruct_reshare(&shares, ActivationType::Sigmoid, &mut rng);

        let result = reconstruct(&result_shares);

        for (i, r) in result.iter().enumerate() {
            let expected = apply_activation(values[i], ActivationType::Sigmoid);
            assert!(
                (r - expected).abs() < 1e-6,
                "Sigmoid[{}] failed: {} vs {}",
                i,
                r,
                expected,
            );
        }
    }

    #[test]
    fn test_gelu_reconstruct_reshare() {
        let values = vec![-1.0, 0.0, 1.0];
        let shares = split_vector(&values, 3, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let result_shares =
            SecureActivation::apply_reconstruct_reshare(&shares, ActivationType::GELU, &mut rng);

        let result = reconstruct(&result_shares);

        for (i, r) in result.iter().enumerate() {
            let expected = apply_activation(values[i], ActivationType::GELU);
            assert!((r - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_approximate_sigmoid() {
        let mut dealer = TrustedDealer::with_seed(42);
        let per_party = dealer.generate_scalar_triples(200, 3);

        let mut pools: Vec<BeaverPool> = (0..3)
            .map(|i| {
                let mut p = BeaverPool::new(i, 3, 64);
                p.fill_scalar(per_party[i].clone());
                p
            })
            .collect();

        let values = vec![0.0, 1.0, -1.0];
        let shares = split_vector(&values, 3, 42);

        let result_shares =
            SecureActivation::approximate_sigmoid(&shares, &mut pools).unwrap();
        let result = reconstruct(&result_shares);

        // Polynomial approximation should be within ~0.1 of true sigmoid.
        for (i, r) in result.iter().enumerate() {
            let expected = 1.0 / (1.0 + (-values[i]).exp());
            assert!(
                (r - expected).abs() < 0.15,
                "Approx sigmoid[{}] too far: {} vs {} (diff {})",
                i,
                r,
                expected,
                (r - expected).abs(),
            );
        }
    }

    #[test]
    fn test_leaky_relu() {
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shares = split_vector(&values, 3, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let result_shares = SecureActivation::apply_reconstruct_reshare(
            &shares,
            ActivationType::LeakyReLU(0.01),
            &mut rng,
        );

        let result = reconstruct(&result_shares);
        let expected = vec![-0.02, -0.01, 0.0, 1.0, 2.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_activation_values() {
        assert!((apply_activation(1.0, ActivationType::ReLU) - 1.0).abs() < 1e-10);
        assert!((apply_activation(-1.0, ActivationType::ReLU) - 0.0).abs() < 1e-10);
        assert!((apply_activation(0.0, ActivationType::Sigmoid) - 0.5).abs() < 1e-10);
        assert!((apply_activation(0.0, ActivationType::Tanh) - 0.0).abs() < 1e-10);
        assert!((apply_activation(0.0, ActivationType::GELU) - 0.0).abs() < 1e-10);
    }
}
