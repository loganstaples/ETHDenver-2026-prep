//! Shamir's Secret Sharing: k-of-n threshold scheme.
//!
//! Based on polynomial interpolation over a finite field. To share a secret s:
//! 1. Choose a random polynomial p(x) of degree k-1, where p(0) = s
//! 2. Evaluate p(i) for each party i to produce their share
//! 3. Any k shares can reconstruct s via Lagrange interpolation
//! 4. Fewer than k shares reveal no information about s
//!
//! We work in a prime field F_p where p is a Mersenne prime (2^31 - 1).
//! Values are mapped to/from this field for sharing, then converted back to f64.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;
use super::{ScalarShare, SecretSharingScheme, ShareId, VectorShare};

/// The prime field modulus: 2^31 - 1 (Mersenne prime).
const FIELD_PRIME: i64 = 2_147_483_647;

/// Shamir's secret sharing scheme.
#[derive(Debug, Clone)]
pub struct ShamirSharing {
    /// Reconstruction threshold: how many shares are needed.
    threshold: usize,
    /// RNG for generating random polynomial coefficients.
    rng: ChaCha20Rng,
    /// Scale factor for mapping f64 values to the field.
    /// Values are multiplied by this before sharing and divided after reconstruction.
    scale: f64,
}

impl ShamirSharing {
    /// Creates a new Shamir sharing instance.
    ///
    /// `threshold` is the minimum number of shares needed for reconstruction (k).
    pub fn new(threshold: usize) -> Self {
        assert!(threshold >= 1, "threshold must be at least 1");
        Self {
            threshold,
            rng: ChaCha20Rng::from_entropy(),
            scale: 1e6, // 6 decimal places of precision
        }
    }

    /// Creates a new instance with a fixed seed.
    pub fn with_seed(threshold: usize, seed: u64) -> Self {
        Self {
            threshold,
            rng: ChaCha20Rng::seed_from_u64(seed),
            scale: 1e6,
        }
    }

    /// Sets the scale factor for f64 ↔ field conversion.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Returns the threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Maps an f64 value into the prime field.
    fn to_field(&self, value: f64) -> i64 {
        let scaled = (value * self.scale).round() as i64;
        ((scaled % FIELD_PRIME) + FIELD_PRIME) % FIELD_PRIME
    }

    /// Maps a field element back to f64.
    fn from_field(&self, field_val: i64) -> f64 {
        // Handle the "negative" half of the field: values > p/2 represent negatives.
        let half = FIELD_PRIME / 2;
        let signed = if field_val > half {
            field_val - FIELD_PRIME
        } else {
            field_val
        };
        signed as f64 / self.scale
    }

    /// Modular arithmetic helpers.
    fn mod_add(a: i64, b: i64) -> i64 {
        ((a + b) % FIELD_PRIME + FIELD_PRIME) % FIELD_PRIME
    }

    fn mod_mul(a: i64, b: i64) -> i64 {
        ((a as i128 * b as i128) % FIELD_PRIME as i128) as i64
    }

    fn mod_sub(a: i64, b: i64) -> i64 {
        ((a - b) % FIELD_PRIME + FIELD_PRIME) % FIELD_PRIME
    }

    /// Modular inverse using extended Euclidean algorithm.
    fn mod_inv(a: i64) -> i64 {
        let mut old_r = a;
        let mut r = FIELD_PRIME;
        let mut old_s: i64 = 1;
        let mut s: i64 = 0;

        while r != 0 {
            let q = old_r / r;
            let temp_r = r;
            r = old_r - q * r;
            old_r = temp_r;
            let temp_s = s;
            s = old_s - q * s;
            old_s = temp_s;
        }

        ((old_s % FIELD_PRIME as i64) + FIELD_PRIME as i64) % FIELD_PRIME as i64
    }

    /// Generates a random polynomial of degree k-1 with p(0) = secret.
    fn random_polynomial(&mut self, secret: i64, degree: usize) -> Vec<i64> {
        let mut coeffs = Vec::with_capacity(degree + 1);
        coeffs.push(secret); // p(0) = secret

        for _ in 1..=degree {
            let coeff = self.rng.gen_range(0..FIELD_PRIME);
            coeffs.push(coeff);
        }

        coeffs
    }

    /// Evaluates polynomial at point x.
    fn eval_polynomial(coeffs: &[i64], x: i64) -> i64 {
        let mut result = 0i64;
        let mut x_power = 1i64;

        for &coeff in coeffs {
            result = Self::mod_add(result, Self::mod_mul(coeff, x_power));
            x_power = Self::mod_mul(x_power, x);
        }

        result
    }

    /// Lagrange interpolation to reconstruct f(0) from points.
    fn lagrange_interpolate(points: &[(i64, i64)]) -> i64 {
        let k = points.len();
        let mut result = 0i64;

        for i in 0..k {
            let (xi, yi) = points[i];

            // Compute Lagrange basis polynomial L_i(0) = ∏_{j≠i} (0 - x_j) / (x_i - x_j)
            let mut numerator = 1i64;
            let mut denominator = 1i64;

            for j in 0..k {
                if i == j {
                    continue;
                }
                let (xj, _) = points[j];
                // numerator *= (0 - x_j) = -x_j
                numerator = Self::mod_mul(numerator, Self::mod_sub(0, xj));
                // denominator *= (x_i - x_j)
                denominator = Self::mod_mul(denominator, Self::mod_sub(xi, xj));
            }

            let basis = Self::mod_mul(numerator, Self::mod_inv(denominator));
            result = Self::mod_add(result, Self::mod_mul(yi, basis));
        }

        result
    }

    /// Shares a single field element.
    fn share_field_element(
        &mut self,
        secret: i64,
        num_parties: usize,
    ) -> Vec<(i64, i64)> {
        let degree = self.threshold - 1;
        let poly = self.random_polynomial(secret, degree);

        (1..=num_parties)
            .map(|i| {
                let x = i as i64;
                let y = Self::eval_polynomial(&poly, x);
                (x, y)
            })
            .collect()
    }

    /// Re-shares a value held by one party into new Shamir shares.
    /// The old share value becomes the new secret.
    pub fn reshare_scalar(
        &mut self,
        share_value: f64,
        num_parties: usize,
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<ScalarShare>> {
        if parties.len() != num_parties {
            return Err(MPCError::ShareCountMismatch {
                expected: num_parties,
                got: parties.len(),
            });
        }

        let field_val = self.to_field(share_value);
        let points = self.share_field_element(field_val, num_parties);

        let shares = points
            .into_iter()
            .enumerate()
            .map(|(i, (_, y))| {
                let val = self.from_field(y);
                ScalarShare::new(
                    ShareId::new(parties[i].clone(), secret_id, i + 1),
                    val,
                )
            })
            .collect();

        Ok(shares)
    }

    /// Degree reduction after multiplication of two shared values.
    /// When two degree-(k-1) sharings are multiplied, the result is degree 2(k-1).
    /// This protocol reduces it back to degree k-1 using a re-sharing step.
    pub fn degree_reduce(
        &mut self,
        product_shares: &[ScalarShare],
        parties: &[PartyId],
    ) -> MPCResult<Vec<ScalarShare>> {
        // Reconstruct the product (requires 2k-1 shares).
        let required = 2 * self.threshold - 1;
        if product_shares.len() < required {
            return Err(MPCError::InsufficientShares {
                required,
                available: product_shares.len(),
            });
        }

        // Reconstruct and re-share with degree k-1.
        let value = self.reconstruct_scalar(product_shares)?;
        self.share_scalar(value, "degree-reduced", parties)
    }
}

impl SecretSharingScheme for ShamirSharing {
    fn share_scalar(
        &self,
        secret: f64,
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<ScalarShare>> {
        let n = parties.len();
        if n < self.threshold {
            return Err(MPCError::InvalidThreshold {
                threshold: self.threshold,
                num_parties: n,
            });
        }

        let mut rng_clone = self.rng.clone();
        let mut sharing = ShamirSharing {
            threshold: self.threshold,
            rng: rng_clone,
            scale: self.scale,
        };

        let field_secret = sharing.to_field(secret);
        let points = sharing.share_field_element(field_secret, n);

        let shares = points
            .into_iter()
            .enumerate()
            .map(|(i, (x, y))| {
                // Store the field evaluation point x in the share index.
                let val = sharing.from_field(y);
                ScalarShare::new(
                    ShareId::new(parties[i].clone(), secret_id, x as usize),
                    val,
                )
            })
            .collect();

        Ok(shares)
    }

    fn reconstruct_scalar(&self, shares: &[ScalarShare]) -> MPCResult<f64> {
        if shares.len() < self.threshold {
            return Err(MPCError::InsufficientShares {
                required: self.threshold,
                available: shares.len(),
            });
        }

        // Use only the first `threshold` shares.
        let points: Vec<(i64, i64)> = shares[..self.threshold]
            .iter()
            .map(|s| {
                let x = s.id.index as i64;
                let y = self.to_field(s.value);
                (x, y)
            })
            .collect();

        let secret_field = Self::lagrange_interpolate(&points);
        Ok(self.from_field(secret_field))
    }

    fn share_vector(
        &self,
        secret: &[f64],
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<VectorShare>> {
        let n = parties.len();
        if n < self.threshold {
            return Err(MPCError::InvalidThreshold {
                threshold: self.threshold,
                num_parties: n,
            });
        }

        let dim = secret.len();
        let mut sharing = ShamirSharing {
            threshold: self.threshold,
            rng: self.rng.clone(),
            scale: self.scale,
        };

        let mut result: Vec<VectorShare> = parties
            .iter()
            .enumerate()
            .map(|(i, p)| {
                VectorShare::new(
                    ShareId::new(p.clone(), secret_id, i + 1),
                    vec![0.0; dim],
                )
            })
            .collect();

        for elem_idx in 0..dim {
            let field_val = sharing.to_field(secret[elem_idx]);
            let points = sharing.share_field_element(field_val, n);

            for (party_idx, (_, y)) in points.into_iter().enumerate() {
                result[party_idx].values[elem_idx] = sharing.from_field(y);
            }
        }

        Ok(result)
    }

    fn reconstruct_vector(&self, shares: &[VectorShare]) -> MPCResult<Vec<f64>> {
        if shares.len() < self.threshold {
            return Err(MPCError::InsufficientShares {
                required: self.threshold,
                available: shares.len(),
            });
        }

        let dim = shares[0].values.len();
        for s in shares {
            if s.values.len() != dim {
                return Err(MPCError::ShapeMismatch {
                    expected: vec![dim],
                    got: vec![s.values.len()],
                });
            }
        }

        let mut result = vec![0.0; dim];
        let use_shares = &shares[..self.threshold];

        for elem_idx in 0..dim {
            let points: Vec<(i64, i64)> = use_shares
                .iter()
                .map(|s| {
                    let x = s.id.index as i64;
                    let y = self.to_field(s.values[elem_idx]);
                    (x, y)
                })
                .collect();

            let field_val = Self::lagrange_interpolate(&points);
            result[elem_idx] = self.from_field(field_val);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_field_roundtrip() {
        let sharing = ShamirSharing::with_seed(2, 42);

        for val in &[0.0, 1.0, -1.0, 3.14159, -100.5, 999.999] {
            let field = sharing.to_field(*val);
            let back = sharing.from_field(field);
            assert!(
                (back - val).abs() < 1e-5,
                "Field roundtrip failed for {}: got {}",
                val,
                back,
            );
        }
    }

    #[test]
    fn test_mod_inv() {
        for a in &[1i64, 2, 3, 7, 100, 999999] {
            let inv = ShamirSharing::mod_inv(*a);
            let product = ShamirSharing::mod_mul(*a, inv);
            assert_eq!(product, 1, "mod_inv failed for {}: inv={}, product={}", a, inv, product);
        }
    }

    #[test]
    fn test_polynomial_evaluation() {
        // p(x) = 5 + 3x + 2x^2, evaluate at x=2: 5 + 6 + 8 = 19
        let coeffs = vec![5, 3, 2];
        let result = ShamirSharing::eval_polynomial(&coeffs, 2);
        assert_eq!(result, 19);
    }

    #[test]
    fn test_2_of_3_sharing() {
        let sharing = ShamirSharing::with_seed(2, 42);
        let parties = test_parties(3);
        let secret = 42.0;

        let shares = sharing.share_scalar(secret, "test", &parties).unwrap();
        assert_eq!(shares.len(), 3);

        // Any 2 shares should reconstruct the secret.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let subset = vec![shares[i].clone(), shares[j].clone()];
                let recon = sharing.reconstruct_scalar(&subset).unwrap();
                assert!(
                    (recon - secret).abs() < 0.01,
                    "2-of-3 reconstruction failed with shares {},{}: got {}",
                    i,
                    j,
                    recon,
                );
            }
        }
    }

    #[test]
    fn test_3_of_5_sharing() {
        let sharing = ShamirSharing::with_seed(3, 42);
        let parties = test_parties(5);
        let secret = -17.5;

        let shares = sharing.share_scalar(secret, "test", &parties).unwrap();
        assert_eq!(shares.len(), 5);

        // Any 3 shares should work.
        let subset = vec![shares[0].clone(), shares[2].clone(), shares[4].clone()];
        let recon = sharing.reconstruct_scalar(&subset).unwrap();
        assert!(
            (recon - secret).abs() < 0.01,
            "3-of-5 reconstruction failed: got {}",
            recon,
        );

        // 2 shares should fail.
        let too_few = vec![shares[0].clone(), shares[1].clone()];
        assert!(sharing.reconstruct_scalar(&too_few).is_err());
    }

    #[test]
    fn test_vector_sharing() {
        let sharing = ShamirSharing::with_seed(2, 42);
        let parties = test_parties(3);
        let secret = vec![1.0, -2.5, 3.14, 0.0, 100.0];

        let shares = sharing.share_vector(&secret, "vec", &parties).unwrap();
        assert_eq!(shares.len(), 3);

        let recon = sharing.reconstruct_vector(&shares[..2]).unwrap();
        for (a, b) in recon.iter().zip(secret.iter()) {
            assert!(
                (a - b).abs() < 0.01,
                "Vector recon failed: {} vs {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_lagrange_basic() {
        // Secret = 42, polynomial p(x) = 42 + 7x (degree 1, k=2)
        // p(1) = 49, p(2) = 56, p(3) = 63
        let points = vec![(1, 49), (2, 56)];
        let result = ShamirSharing::lagrange_interpolate(&points);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_reshare() {
        let mut sharing = ShamirSharing::with_seed(2, 42);
        let parties = test_parties(3);
        let original = 7.77;

        let new_shares = sharing
            .reshare_scalar(original, 3, "reshared", &parties)
            .unwrap();

        let recon = sharing.reconstruct_scalar(&new_shares[..2]).unwrap();
        assert!(
            (recon - original).abs() < 0.01,
            "Reshare failed: {} vs {}",
            recon,
            original,
        );
    }
}
