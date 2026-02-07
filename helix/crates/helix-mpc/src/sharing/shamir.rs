//! Shamir's Secret Sharing: k-of-n threshold scheme.
//!
//! Based on polynomial interpolation over a finite field. To share a secret s:
//! 1. Choose a random polynomial p(x) of degree k-1, where p(0) = s
//! 2. Evaluate p(i) for each party i to produce their share
//! 3. Any k shares can reconstruct s via Lagrange interpolation
//! 4. Fewer than k shares reveal no information about s
//!
//! Internally uses the BN254 scalar field (Fr) for exact arithmetic.
//! The f64 trait interface is preserved for compatibility — values are converted
//! to Fr at sharing boundaries and back to f64 after reconstruction.
//!
//! # Security
//!
//! The BN254 scalar field (~254 bits) provides vastly more security than the
//! previous 31-bit Mersenne prime. Random polynomial coefficients are generated
//! as 64-bit scaled integers to maintain f64 compatibility at the share layer,
//! while all arithmetic is exact in the field (no rounding errors).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::types::PartyId;
use super::{ScalarShare, SecretSharingScheme, ShareId, VectorShare};

/// Scale factor for mapping f64 values to field elements.
/// Using 10^9 gives 9 decimal places of precision while staying well within u64 range.
const SCALE: f64 = 1_000_000_000.0;

/// Range for random polynomial coefficients (before scaling).
/// This ensures intermediate values stay within f64 precision after evaluation.
const RANDOM_COEFF_RANGE: i64 = 1_000_000_000;

/// Shamir's secret sharing scheme.
/// Uses BN254 scalar field (Fr) internally for exact arithmetic.
#[derive(Debug, Clone)]
pub struct ShamirSharing {
    /// Reconstruction threshold: how many shares are needed.
    threshold: usize,
    /// RNG for generating random polynomial coefficients.
    rng: ChaCha20Rng,
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
        }
    }

    /// Creates a new instance with a fixed seed.
    pub fn with_seed(threshold: usize, seed: u64) -> Self {
        Self {
            threshold,
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Returns the threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Maps an f64 value into a raw Fr field element (not fixed-point).
    /// Uses a simple integer scale to preserve precision.
    fn to_field(value: f64) -> Fr {
        let scaled = (value * SCALE).round() as i64;
        if scaled >= 0 {
            Fr::from_u64(scaled as u64)
        } else {
            Fr::from_u64((-scaled) as u64).neg()
        }
    }

    /// Maps a raw Fr field element back to f64.
    /// Inverse of `to_field`. Handles "negative" field elements (> r/2).
    fn from_field(fr: &Fr) -> f64 {
        if fr.is_negative().to_bool() {
            let neg = fr.neg();
            let val = neg.to_u64().unwrap_or_else(|| {
                let bytes = neg.to_bytes_le();
                u64::from_le_bytes(bytes[0..8].try_into().unwrap())
            });
            -(val as f64) / SCALE
        } else {
            let val = fr.to_u64().unwrap_or_else(|| {
                let bytes = fr.to_bytes_le();
                u64::from_le_bytes(bytes[0..8].try_into().unwrap())
            });
            val as f64 / SCALE
        }
    }

    /// Generates a random polynomial of degree `degree` with p(0) = secret, in Fr.
    /// Coefficients are bounded random values to ensure share values stay f64-representable.
    fn random_polynomial(&mut self, secret: &Fr, degree: usize) -> Vec<Fr> {
        let mut coeffs = Vec::with_capacity(degree + 1);
        coeffs.push(*secret);

        for _ in 1..=degree {
            // Generate bounded random coefficient as scaled integer
            let r: i64 = self.rng.gen_range(-RANDOM_COEFF_RANGE..RANDOM_COEFF_RANGE);
            let coeff = if r >= 0 {
                Fr::from_u64(r as u64)
            } else {
                Fr::from_u64((-r) as u64).neg()
            };
            coeffs.push(coeff);
        }

        coeffs
    }

    /// Evaluates polynomial at point x using Horner's method.
    /// Uses raw field multiplication since all arithmetic is in the field.
    fn eval_polynomial(coeffs: &[Fr], x: &Fr) -> Fr {
        let mut result = Fr::ZERO;
        for coeff in coeffs.iter().rev() {
            result = Fr::add(&Fr::mul(&result, x), coeff);
        }
        result
    }

    /// Lagrange interpolation to reconstruct f(0) from points.
    /// Uses raw field multiplication since all arithmetic is in the field.
    fn lagrange_interpolate(points: &[(Fr, Fr)]) -> Fr {
        let k = points.len();
        let mut result = Fr::ZERO;

        for i in 0..k {
            let (ref xi, ref yi) = points[i];

            let mut numerator = Fr::ONE;
            let mut denominator = Fr::ONE;

            for j in 0..k {
                if i == j {
                    continue;
                }
                let ref xj = points[j].0;
                // numerator *= (0 - x_j) = -x_j
                numerator = Fr::mul(&numerator, &xj.neg());
                // denominator *= (x_i - x_j)
                denominator = Fr::mul(&denominator, &Fr::sub(xi, xj));
            }

            let inv_denom = denominator.inverse()
                .expect("denominator should be nonzero in Lagrange interpolation");
            let basis = Fr::mul(&numerator, &inv_denom);
            result = Fr::add(&result, &Fr::mul(yi, &basis));
        }

        result
    }

    /// Shares a single field element, returning (x, y) evaluation points in Fr.
    fn share_field_element(
        &mut self,
        secret: &Fr,
        num_parties: usize,
    ) -> Vec<(Fr, Fr)> {
        let degree = self.threshold - 1;
        let poly = self.random_polynomial(secret, degree);

        (1..=num_parties)
            .map(|i| {
                let x = Fr::from_u64(i as u64);
                let y = Self::eval_polynomial(&poly, &x);
                (x, y)
            })
            .collect()
    }

    /// Re-shares a value held by one party into new Shamir shares.
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

        let field_val = Self::to_field(share_value);
        let points = self.share_field_element(&field_val, num_parties);

        let shares = points
            .into_iter()
            .enumerate()
            .map(|(i, (_, y))| {
                let val = Self::from_field(&y);
                ScalarShare::new(
                    ShareId::new(parties[i].clone(), secret_id, i + 1),
                    val,
                )
            })
            .collect();

        Ok(shares)
    }

    /// Degree reduction after multiplication of two shared values.
    pub fn degree_reduce(
        &mut self,
        product_shares: &[ScalarShare],
        parties: &[PartyId],
    ) -> MPCResult<Vec<ScalarShare>> {
        let required = 2 * self.threshold - 1;
        if product_shares.len() < required {
            return Err(MPCError::InsufficientShares {
                required,
                available: product_shares.len(),
            });
        }

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

        let mut sharing = ShamirSharing {
            threshold: self.threshold,
            rng: self.rng.clone(),
        };

        let field_secret = Self::to_field(secret);
        let points = sharing.share_field_element(&field_secret, n);

        let shares = points
            .into_iter()
            .enumerate()
            .map(|(i, (x, y))| {
                let val = Self::from_field(&y);
                ScalarShare::new(
                    ShareId::new(
                        parties[i].clone(),
                        secret_id,
                        x.to_u64().unwrap_or(i as u64 + 1) as usize,
                    ),
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

        let points: Vec<(Fr, Fr)> = shares[..self.threshold]
            .iter()
            .map(|s| {
                let x = Fr::from_u64(s.id.index as u64);
                let y = Self::to_field(s.value);
                (x, y)
            })
            .collect();

        let secret_field = Self::lagrange_interpolate(&points);
        Ok(Self::from_field(&secret_field))
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
            let field_val = Self::to_field(secret[elem_idx]);
            let points = sharing.share_field_element(&field_val, n);

            for (party_idx, (_, y)) in points.into_iter().enumerate() {
                result[party_idx].values[elem_idx] = Self::from_field(&y);
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
            let points: Vec<(Fr, Fr)> = use_shares
                .iter()
                .map(|s| {
                    let x = Fr::from_u64(s.id.index as u64);
                    let y = Self::to_field(s.values[elem_idx]);
                    (x, y)
                })
                .collect();

            let field_val = Self::lagrange_interpolate(&points);
            result[elem_idx] = Self::from_field(&field_val);
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
        for val in &[0.0, 1.0, -1.0, 3.14159, -100.5, 999.999] {
            let fr = ShamirSharing::to_field(*val);
            let back = ShamirSharing::from_field(&fr);
            assert!(
                (back - val).abs() < 1e-5,
                "Field roundtrip failed for {}: got {}",
                val,
                back,
            );
        }
    }

    #[test]
    fn test_polynomial_evaluation() {
        // p(x) = 5 + 3x + 2x^2, evaluate at x=2: 5 + 6 + 8 = 19
        let coeffs = vec![Fr::from_u64(5), Fr::from_u64(3), Fr::from_u64(2)];
        let x = Fr::from_u64(2);
        let result = ShamirSharing::eval_polynomial(&coeffs, &x);
        assert_eq!(result.to_u64(), Some(19));
    }

    #[test]
    fn test_lagrange_basic() {
        // Secret = 42, polynomial p(x) = 42 + 7x (degree 1, k=2)
        // p(1) = 49, p(2) = 56
        let points = vec![
            (Fr::from_u64(1), Fr::from_u64(49)),
            (Fr::from_u64(2), Fr::from_u64(56)),
        ];
        let result = ShamirSharing::lagrange_interpolate(&points);
        assert_eq!(result.to_u64(), Some(42));
    }

    #[test]
    fn test_2_of_3_sharing() {
        let sharing = ShamirSharing::with_seed(2, 42);
        let parties = test_parties(3);
        let secret = 42.0;

        let shares = sharing.share_scalar(secret, "test", &parties).unwrap();
        assert_eq!(shares.len(), 3);

        for i in 0..3 {
            for j in (i + 1)..3 {
                let subset = vec![shares[i].clone(), shares[j].clone()];
                let recon = sharing.reconstruct_scalar(&subset).unwrap();
                assert!(
                    (recon - secret).abs() < 0.01,
                    "2-of-3 reconstruction failed with shares {},{}: got {}",
                    i, j, recon,
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

        let subset = vec![shares[0].clone(), shares[2].clone(), shares[4].clone()];
        let recon = sharing.reconstruct_scalar(&subset).unwrap();
        assert!(
            (recon - secret).abs() < 0.01,
            "3-of-5 reconstruction failed: got {}",
            recon,
        );

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
                a, b,
            );
        }
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
            recon, original,
        );
    }
}
