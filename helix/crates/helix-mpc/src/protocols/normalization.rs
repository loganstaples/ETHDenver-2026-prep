//! Secure normalization protocols (LayerNorm, RMSNorm).
//!
//! Normalization layers are critical for transformer training stability.
//! Like activations, they are non-linear and require special handling in MPC.
//!
//! # Approach
//!
//! We use the reconstruct-normalize-reshare approach:
//! 1. Reconstruct the pre-normalization vector (reveals intermediate values, not weights)
//! 2. Compute mean, variance, and normalized values
//! 3. Re-share the normalized values
//!
//! The public scale (gamma) and shift (beta) parameters of LayerNorm
//! can be applied as local operations on shares.

use crate::beaver::pool::BeaverPool;
use crate::error::MPCResult;
use crate::field::Fr;

/// Secure normalization operations.
pub struct SecureNormalization;

impl SecureNormalization {
    /// Secure Layer Normalization using reconstruct-normalize-reshare.
    ///
    /// LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    ///
    /// `shares`: each party's share of the input vector [dim].
    /// `gamma`: public scale parameter [dim].
    /// `beta`: public shift parameter [dim].
    /// `eps`: small constant for numerical stability.
    ///
    /// Returns new shares of the normalized vector.
    pub fn layer_norm(
        shares: &[Vec<Fr>],
        gamma: &[f64],
        beta: &[f64],
        eps: f64,
    ) -> Vec<Vec<Fr>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Step 1: Reconstruct the input.
        let mut values = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                values[i] = Fr::add(&values[i], v);
            }
        }

        // Convert to f64 for normalization computation
        let values_f64: Vec<f64> = values.iter().map(|v| v.to_f64()).collect();

        // Step 2: Compute LayerNorm.
        let mean: f64 = values_f64.iter().sum::<f64>() / dim as f64;
        let variance: f64 =
            values_f64.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv_std = 1.0 / (variance + eps).sqrt();

        let normalized: Vec<f64> = values_f64
            .iter()
            .enumerate()
            .map(|(i, v)| gamma[i] * (v - mean) * inv_std + beta[i])
            .collect();

        // Step 3: Re-share.
        reshare_values(&normalized, num_parties)
    }

    /// Secure RMS Normalization.
    ///
    /// RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
    ///
    /// RMSNorm is simpler than LayerNorm (no mean subtraction) and is used
    /// in modern architectures like LLaMA.
    pub fn rms_norm(
        shares: &[Vec<Fr>],
        gamma: &[f64],
        eps: f64,
    ) -> Vec<Vec<Fr>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Reconstruct.
        let mut values = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                values[i] = Fr::add(&values[i], v);
            }
        }

        // Convert to f64
        let values_f64: Vec<f64> = values.iter().map(|v| v.to_f64()).collect();

        // RMS.
        let rms = (values_f64.iter().map(|v| v * v).sum::<f64>() / dim as f64 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        let normalized: Vec<f64> = values_f64
            .iter()
            .enumerate()
            .map(|(i, v)| gamma[i] * v * inv_rms)
            .collect();

        reshare_values(&normalized, num_parties)
    }

    /// Secure softmax using reconstruct-compute-reshare.
    ///
    /// softmax(x)_i = exp(x_i) / sum(exp(x_j))
    ///
    /// Uses the numerically stable variant: subtract max before exp.
    pub fn softmax(shares: &[Vec<Fr>]) -> Vec<Vec<Fr>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Reconstruct.
        let mut values = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                values[i] = Fr::add(&values[i], v);
            }
        }

        // Convert to f64
        let values_f64: Vec<f64> = values.iter().map(|v| v.to_f64()).collect();

        // Numerically stable softmax.
        let max_val = values_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = values_f64.iter().map(|v| (v - max_val).exp()).collect();
        let sum_exp: f64 = exp_values.iter().sum();
        let softmax_values: Vec<f64> = exp_values.iter().map(|e| e / sum_exp).collect();

        reshare_values(&softmax_values, num_parties)
    }

    /// Batch softmax: applies softmax independently to each row of a matrix.
    ///
    /// `shares[party][row][col]` → applies softmax across cols for each row.
    pub fn batch_softmax(shares: &[Vec<Vec<Fr>>]) -> Vec<Vec<Vec<Fr>>> {
        let num_parties = shares.len();
        let num_rows = shares[0].len();

        let mut result: Vec<Vec<Vec<Fr>>> = vec![Vec::with_capacity(num_rows); num_parties];

        for row in 0..num_rows {
            let row_shares: Vec<Vec<Fr>> = shares
                .iter()
                .map(|p| p[row].clone())
                .collect();

            let softmax_row = Self::softmax(&row_shares);

            for (i, party_result) in softmax_row.into_iter().enumerate() {
                result[i].push(party_result);
            }
        }

        result
    }

    /// Applies LayerNorm where gamma and beta are also secret-shared.
    /// This is needed when gamma/beta are part of the model weights.
    ///
    /// The protocol:
    /// 1. Reconstruct input x (reveals activations)
    /// 2. Compute normalized = (x - mean) / sqrt(var + eps) (public values)
    /// 3. Multiply [gamma] * normalized using Beaver triples
    /// 4. Add [beta]
    pub fn layer_norm_shared_params(
        shares: &[Vec<Fr>],
        gamma_shares: &[Vec<Fr>],
        beta_shares: &[Vec<Fr>],
        eps: f64,
        _pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = shares.len();
        let dim = shares[0].len();

        // Reconstruct input.
        let mut values = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.iter().enumerate() {
                values[i] = Fr::add(&values[i], v);
            }
        }

        // Convert to f64
        let values_f64: Vec<f64> = values.iter().map(|v| v.to_f64()).collect();

        // Compute normalization constants (public).
        let mean: f64 = values_f64.iter().sum::<f64>() / dim as f64;
        let variance: f64 =
            values_f64.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // normalized = (x - mean) * inv_std (public values)
        let normalized: Vec<f64> = values_f64.iter().map(|v| (v - mean) * inv_std).collect();

        // [result] = [gamma] * normalized + [beta]
        // gamma * normalized is a local operation (multiply share by public value)
        // Use fixed_mul for proper fixed-point arithmetic
        let mut result: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];
        for i in 0..num_parties {
            for d in 0..dim {
                let gamma_scaled = gamma_shares[i][d].fixed_mul(&Fr::from_f64(normalized[d]));
                result[i][d] = Fr::add(&gamma_scaled, &beta_shares[i][d]);
            }
        }

        Ok(result)
    }
}

/// Creates additive shares of values.
fn reshare_values(values: &[f64], num_parties: usize) -> Vec<Vec<Fr>> {
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha20Rng;

    let dim = values.len();
    let mut rng = ChaCha20Rng::seed_from_u64(0xA0EDA112E);
    let mut shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];

    for d in 0..dim {
        let target = Fr::from_f64(values[d]);
        let mut sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
            shares[i][d] = r.clone();
            sum = Fr::add(&sum, &r);
        }
        shares[num_parties - 1][d] = Fr::sub(&target, &sum);
    }

    shares
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ops::sum;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn split_vector(values: &[f64], n: usize, seed: u64) -> Vec<Vec<Fr>> {
        let dim = values.len();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; n];

        for d in 0..dim {
            let target = Fr::from_f64(values[d]);
            let mut s = Fr::ZERO;
            for i in 0..n - 1 {
                let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
                shares[i][d] = r.clone();
                s = Fr::add(&s, &r);
            }
            shares[n - 1][d] = Fr::sub(&target, &s);
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
    fn test_layer_norm() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let shares = split_vector(&values, 3, 42);

        let result_shares = SecureNormalization::layer_norm(&shares, &gamma, &beta, 1e-5);
        let result = reconstruct(&result_shares);

        // LayerNorm should produce zero-mean, unit-variance output.
        let mean: f64 = result.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-10, "Mean not zero: {}", mean);

        let var: f64 = result.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        assert!((var - 1.0).abs() < 0.01, "Variance not 1: {}", var);
    }

    #[test]
    fn test_rms_norm() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let shares = split_vector(&values, 3, 42);

        let result_shares = SecureNormalization::rms_norm(&shares, &gamma, 1e-5);
        let result = reconstruct(&result_shares);

        // RMSNorm: x / rms * gamma
        let rms = (values.iter().map(|v| v * v).sum::<f64>() / 4.0).sqrt();
        for (i, r) in result.iter().enumerate() {
            let expected = values[i] / rms;
            assert!(
                (r - expected).abs() < 1e-6,
                "RMSNorm[{}]: {} vs {}",
                i,
                r,
                expected,
            );
        }
    }

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let shares = split_vector(&values, 3, 42);

        let result_shares = SecureNormalization::softmax(&shares);
        let result = reconstruct(&result_shares);

        // Softmax should sum to 1.
        let total: f64 = result.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Softmax doesn't sum to 1: {}",
            total,
        );

        // Values should be in [0, 1] and ordered.
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_layer_norm_with_gamma_beta() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4]; // Scale by 2
        let beta = vec![1.0; 4]; // Shift by 1
        let shares = split_vector(&values, 3, 42);

        let result_shares = SecureNormalization::layer_norm(&shares, &gamma, &beta, 1e-5);
        let result = reconstruct(&result_shares);

        // Mean of result should be beta (since gamma * normalized has mean 0).
        let mean: f64 = result.iter().sum::<f64>() / 4.0;
        assert!(
            (mean - 1.0).abs() < 1e-10,
            "Mean with beta not correct: {}",
            mean,
        );
    }

    #[test]
    fn test_batch_softmax() {
        let row1 = vec![1.0, 2.0, 3.0];
        let row2 = vec![0.0, 0.0, 0.0];

        let shares: Vec<Vec<Vec<Fr>>> = {
            let s1 = split_vector(&row1, 3, 42);
            let s2 = split_vector(&row2, 3, 99);
            (0..3)
                .map(|i| vec![s1[i].clone(), s2[i].clone()])
                .collect()
        };

        let result_shares = SecureNormalization::batch_softmax(&shares);

        // Row 2 (uniform input) should give uniform softmax.
        let row2_result: Vec<f64> = {
            let dim = result_shares[0][1].len();
            let mut r = vec![Fr::ZERO; dim];
            for p in &result_shares {
                for (i, v) in p[1].iter().enumerate() {
                    r[i] = Fr::add(&r[i], v);
                }
            }
            r.iter().map(|v| v.to_f64()).collect()
        };

        for v in &row2_result {
            assert!(
                (v - 1.0 / 3.0).abs() < 1e-10,
                "Uniform softmax wrong: {}",
                v,
            );
        }
    }
}
