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

use crate::beaver::pool::BeaverPool;
use crate::error::{MPCError, MPCResult};
use crate::protocols::arithmetic::SecureArithmetic;

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
    /// `shares`: each party's share of the pre-activation vector.
    ///           shares[party_idx][element_idx]
    /// `activation`: which activation to apply.
    ///
    /// Returns new shares of the post-activation vector.
    pub fn apply_reconstruct_reshare(
        shares: &[Vec<f64>],
        activation: ActivationType,
    ) -> Vec<Vec<f64>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Step 1: Reconstruct (parties would send shares to each other).
        let mut values = vec![0.0; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                values[i] += v;
            }
        }

        // Step 2: Apply activation to the reconstructed values.
        let activated: Vec<f64> = values
            .iter()
            .map(|v| apply_activation(*v, activation))
            .collect();

        // Step 3: Re-share (party 0 acts as dealer, gives full value as its share,
        // others get zero). In a real system, proper random sharing would be used.
        reshare_values(&activated, num_parties)
    }

    /// Batch version: applies activation to multiple vectors at once.
    pub fn apply_batch_reconstruct_reshare(
        batch_shares: &[Vec<Vec<f64>>],
        activation: ActivationType,
    ) -> Vec<Vec<Vec<f64>>> {
        let num_parties = batch_shares.len();
        let batch_size = batch_shares[0].len();

        let mut results: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(batch_size); num_parties];

        for b in 0..batch_size {
            let item_shares: Vec<Vec<f64>> = batch_shares
                .iter()
                .map(|p| p[b].clone())
                .collect();

            let activated = Self::apply_reconstruct_reshare(&item_shares, activation);

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
        shares: &[Vec<f64>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
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
        let mut result: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                // relu ≈ 0.5*x + 0.5*x²/(x² + 0.1)
                // Simplified to: 0.5*x + 0.5*x for positive regime
                // This is a placeholder; production would use garbled circuits.
                result[i][d] = 0.5 * shares[i][d] + 0.5 * x_squared[i][d].signum() * x_squared[i][d].abs().sqrt() * 0.5;
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
        shares: &[Vec<f64>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Compute x² and x³ on shares.
        let x_squared = SecureArithmetic::simulate_vector_multiply(shares, shares, pools)?;
        let x_cubed = SecureArithmetic::simulate_vector_multiply(shares, &x_squared, pools)?;

        // sigmoid(x) ≈ 0.5 + 0.25*x - 0.0208*x³
        let mut result: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                let term1 = SecureArithmetic::add_public(0.0, 0.5, i);
                let term2 = SecureArithmetic::scale_share(shares[i][d], 0.25);
                let term3 = SecureArithmetic::scale_share(x_cubed[i][d], -0.0208);
                result[i][d] = term1 + term2 + term3;
            }
        }

        Ok(result)
    }

    /// Approximates GELU using:
    /// gelu(x) ≈ 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
    /// Simplified to: gelu(x) ≈ 0.5*x + 0.5*x*sigmoid(1.702*x)
    /// Further simplified for MPC: gelu(x) ≈ 0.5*x*(1 + 0.851*x - 0.0354*x³)
    pub fn approximate_gelu(
        shares: &[Vec<f64>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // x²
        let x_sq = SecureArithmetic::simulate_vector_multiply(shares, shares, pools)?;
        // x³
        let x_cu = SecureArithmetic::simulate_vector_multiply(shares, &x_sq, pools)?;

        // inner = 1 + 0.851*x - 0.0354*x³
        let mut inner: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                let one = SecureArithmetic::add_public(0.0, 1.0, i);
                let lin = SecureArithmetic::scale_share(shares[i][d], 0.851);
                let cub = SecureArithmetic::scale_share(x_cu[i][d], -0.0354);
                inner[i][d] = one + lin + cub;
            }
        }

        // result = 0.5 * x * inner
        let x_inner = SecureArithmetic::simulate_vector_multiply(shares, &inner, pools)?;
        let mut result: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                result[i][d] = SecureArithmetic::scale_share(x_inner[i][d], 0.5);
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

/// Creates additive shares of values using a deterministic split.
/// Party 0 gets the value minus random offsets; others get random offsets.
fn reshare_values(values: &[f64], num_parties: usize) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha20Rng;

    let dim = values.len();
    let mut rng = ChaCha20Rng::seed_from_u64(0xAC71A710);
    let mut shares: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_parties];

    for d in 0..dim {
        let mut sum = 0.0;
        for i in 0..num_parties - 1 {
            let r: f64 = rng.gen_range(-100.0..100.0);
            shares[i][d] = r;
            sum += r;
        }
        shares[num_parties - 1][d] = values[d] - sum;
    }

    shares
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;

    fn split_vector(values: &[f64], n: usize, seed: u64) -> Vec<Vec<f64>> {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha20Rng;

        let dim = values.len();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut shares: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];

        for d in 0..dim {
            let mut sum = 0.0;
            for i in 0..n - 1 {
                let r: f64 = rng.gen_range(-100.0..100.0);
                shares[i][d] = r;
                sum += r;
            }
            shares[n - 1][d] = values[d] - sum;
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

    #[test]
    fn test_relu_reconstruct_reshare() {
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shares = split_vector(&values, 3, 42);

        let result_shares =
            SecureActivation::apply_reconstruct_reshare(&shares, ActivationType::ReLU);

        let result = reconstruct(&result_shares);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-10,
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

        let result_shares =
            SecureActivation::apply_reconstruct_reshare(&shares, ActivationType::Sigmoid);

        let result = reconstruct(&result_shares);

        for (i, r) in result.iter().enumerate() {
            let expected = apply_activation(values[i], ActivationType::Sigmoid);
            assert!(
                (r - expected).abs() < 1e-10,
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

        let result_shares =
            SecureActivation::apply_reconstruct_reshare(&shares, ActivationType::GELU);

        let result = reconstruct(&result_shares);

        for (i, r) in result.iter().enumerate() {
            let expected = apply_activation(values[i], ActivationType::GELU);
            assert!((r - expected).abs() < 1e-10);
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

        let result_shares = SecureActivation::apply_reconstruct_reshare(
            &shares,
            ActivationType::LeakyReLU(0.01),
        );

        let result = reconstruct(&result_shares);
        let expected = vec![-0.02, -0.01, 0.0, 1.0, 2.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
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
