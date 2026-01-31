//! Additive secret sharing: s = s_1 + s_2 + ... + s_n.
//!
//! The simplest secret sharing scheme. To share a secret s among n parties:
//! 1. Generate n-1 random values r_1, ..., r_{n-1}
//! 2. Compute s_n = s - (r_1 + ... + r_{n-1})
//! 3. Give share s_i to party i
//!
//! Reconstruction: sum all shares.
//!
//! Properties:
//! - Information-theoretic security: any n-1 shares reveal nothing about s
//! - Linear homomorphism: shares of (a + b) = shares of a + shares of b
//! - Requires ALL shares for reconstruction (n-of-n threshold)

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;
use super::{ScalarShare, SecretSharingScheme, ShareId, VectorShare};

/// Additive secret sharing scheme.
#[derive(Debug, Clone)]
pub struct AdditiveSharing {
    /// RNG for generating random shares. Uses a cryptographic RNG.
    rng: ChaCha20Rng,
    /// Range for random share values. Shares are drawn uniformly from
    /// [-share_range, share_range]. Must be large enough to mask the secret
    /// but small enough to avoid f64 precision issues.
    share_range: f64,
}

impl AdditiveSharing {
    /// Creates a new additive sharing instance with a random seed.
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            share_range: 1e9,
        }
    }

    /// Creates a new instance with a fixed seed (for reproducibility in tests).
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            share_range: 1e9,
        }
    }

    /// Sets the range for random share values.
    pub fn with_range(mut self, range: f64) -> Self {
        self.share_range = range;
        self
    }

    /// Generates a random value in [-share_range, share_range].
    fn random_value(&mut self) -> f64 {
        self.rng.gen_range(-self.share_range..self.share_range)
    }

    /// Splits a scalar into additive shares.
    fn split_scalar(&mut self, secret: f64, n: usize) -> Vec<f64> {
        assert!(n >= 2, "Need at least 2 parties for secret sharing");

        let mut shares = Vec::with_capacity(n);
        let mut sum = 0.0;

        // Generate n-1 random shares.
        for _ in 0..n - 1 {
            let r = self.random_value();
            shares.push(r);
            sum += r;
        }

        // Last share ensures the sum equals the secret.
        shares.push(secret - sum);
        shares
    }

    /// Reconstructs a scalar from additive shares (just sum them).
    fn reconstruct_from_values(shares: &[f64]) -> f64 {
        shares.iter().sum()
    }

    /// Re-shares an existing share into new sub-shares that sum to the original.
    /// Used to refresh shares periodically (prevents gradient accumulation attacks).
    pub fn reshare_scalar(
        &mut self,
        share_value: f64,
        n: usize,
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<ScalarShare>> {
        if parties.len() != n {
            return Err(MPCError::ShareCountMismatch {
                expected: n,
                got: parties.len(),
            });
        }

        let new_shares = self.split_scalar(share_value, n);
        let result = new_shares
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                ScalarShare::new(
                    ShareId::new(parties[i].clone(), secret_id, i),
                    val,
                )
            })
            .collect();

        Ok(result)
    }

    /// Re-shares a vector share. Each element of the vector is re-shared independently.
    pub fn reshare_vector(
        &mut self,
        share: &VectorShare,
        n: usize,
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<VectorShare>> {
        if parties.len() != n {
            return Err(MPCError::ShareCountMismatch {
                expected: n,
                got: parties.len(),
            });
        }

        let dim = share.values.len();
        let mut result: Vec<VectorShare> = parties
            .iter()
            .enumerate()
            .map(|(i, p)| {
                VectorShare::new(
                    ShareId::new(p.clone(), secret_id, i),
                    vec![0.0; dim],
                )
            })
            .collect();

        for elem_idx in 0..dim {
            let sub_shares = self.split_scalar(share.values[elem_idx], n);
            for (party_idx, val) in sub_shares.into_iter().enumerate() {
                result[party_idx].values[elem_idx] = val;
            }
        }

        Ok(result)
    }
}

impl Default for AdditiveSharing {
    fn default() -> Self {
        Self::new()
    }
}

impl SecretSharingScheme for AdditiveSharing {
    fn share_scalar(
        &self,
        secret: f64,
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<ScalarShare>> {
        let n = parties.len();
        if n < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: n,
            });
        }

        let mut rng = self.rng.clone();
        let mut sharing = AdditiveSharing {
            rng,
            share_range: self.share_range,
        };

        let values = sharing.split_scalar(secret, n);
        let shares = values
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                ScalarShare::new(
                    ShareId::new(parties[i].clone(), secret_id, i),
                    val,
                )
            })
            .collect();

        Ok(shares)
    }

    fn reconstruct_scalar(&self, shares: &[ScalarShare]) -> MPCResult<f64> {
        if shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 2,
                available: 0,
            });
        }

        let values: Vec<f64> = shares.iter().map(|s| s.value).collect();
        Ok(Self::reconstruct_from_values(&values))
    }

    fn share_vector(
        &self,
        secret: &[f64],
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<VectorShare>> {
        let n = parties.len();
        if n < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: n,
            });
        }

        let dim = secret.len();
        let mut sharing = AdditiveSharing {
            rng: self.rng.clone(),
            share_range: self.share_range,
        };

        // Initialize empty vector shares for each party.
        let mut result: Vec<VectorShare> = parties
            .iter()
            .enumerate()
            .map(|(i, p)| {
                VectorShare::new(
                    ShareId::new(p.clone(), secret_id, i),
                    vec![0.0; dim],
                )
            })
            .collect();

        // Share each element independently.
        for elem_idx in 0..dim {
            let values = sharing.split_scalar(secret[elem_idx], n);
            for (party_idx, val) in values.into_iter().enumerate() {
                result[party_idx].values[elem_idx] = val;
            }
        }

        Ok(result)
    }

    fn reconstruct_vector(&self, shares: &[VectorShare]) -> MPCResult<Vec<f64>> {
        if shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 2,
                available: 0,
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
        for s in shares {
            for (i, val) in s.values.iter().enumerate() {
                result[i] += val;
            }
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
    fn test_share_and_reconstruct_scalar() {
        let sharing = AdditiveSharing::with_seed(42);
        let parties = test_parties(3);
        let secret = 3.14159;

        let shares = sharing.share_scalar(secret, "test", &parties).unwrap();
        assert_eq!(shares.len(), 3);

        // No single share should equal the secret.
        for s in &shares {
            assert!((s.value - secret).abs() > 1.0);
        }

        let reconstructed = sharing.reconstruct_scalar(&shares).unwrap();
        assert!(
            (reconstructed - secret).abs() < 1e-6,
            "Reconstruction error: {} vs {}",
            reconstructed,
            secret,
        );
    }

    #[test]
    fn test_share_and_reconstruct_vector() {
        let sharing = AdditiveSharing::with_seed(42);
        let parties = test_parties(3);
        let secret = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let shares = sharing.share_vector(&secret, "vec", &parties).unwrap();
        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].len(), 5);

        let reconstructed = sharing.reconstruct_vector(&shares).unwrap();
        for (a, b) in reconstructed.iter().zip(secret.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Vector reconstruction error: {} vs {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_additive_homomorphism() {
        let sharing = AdditiveSharing::with_seed(42);
        let parties = test_parties(3);

        let a = 10.0;
        let b = 20.0;

        let shares_a = sharing.share_scalar(a, "a", &parties).unwrap();
        // Use a different seed instance for b to get different randomness.
        let sharing_b = AdditiveSharing::with_seed(99);
        let shares_b = sharing_b.share_scalar(b, "b", &parties).unwrap();

        // Add shares element-wise: share_c[i] = share_a[i] + share_b[i]
        let shares_c: Vec<ScalarShare> = shares_a
            .iter()
            .zip(shares_b.iter())
            .enumerate()
            .map(|(i, (sa, sb))| {
                ScalarShare::new(
                    ShareId::new(parties[i].clone(), "c", i),
                    sa.value + sb.value,
                )
            })
            .collect();

        let result = sharing.reconstruct_scalar(&shares_c).unwrap();
        assert!(
            (result - 30.0).abs() < 1e-10,
            "Homomorphic addition failed: {} vs 30.0",
            result,
        );
    }

    #[test]
    fn test_reshare() {
        let mut sharing = AdditiveSharing::with_seed(42);
        let parties = test_parties(3);
        let secret = 42.0;

        let shares = sharing
            .share_scalar(secret, "orig", &parties)
            .unwrap();

        // Re-share the first party's share.
        let new_shares = sharing
            .reshare_scalar(shares[0].value, 3, "reshared", &parties)
            .unwrap();

        // The new shares should sum to the original share value.
        let recon: f64 = new_shares.iter().map(|s| s.value).sum();
        assert!(
            (recon - shares[0].value).abs() < 1e-10,
            "Reshare error: {} vs {}",
            recon,
            shares[0].value,
        );
    }

    #[test]
    fn test_two_party() {
        let sharing = AdditiveSharing::with_seed(42);
        let parties = test_parties(2);
        let secret = 100.0;

        let shares = sharing.share_scalar(secret, "test", &parties).unwrap();
        assert_eq!(shares.len(), 2);

        let recon = sharing.reconstruct_scalar(&shares).unwrap();
        assert!((recon - secret).abs() < 1e-10);
    }

    #[test]
    fn test_many_parties() {
        let sharing = AdditiveSharing::with_seed(42);
        let parties = test_parties(10);
        let secret = -7.5;

        let shares = sharing.share_scalar(secret, "test", &parties).unwrap();
        assert_eq!(shares.len(), 10);

        let recon = sharing.reconstruct_scalar(&shares).unwrap();
        assert!((recon - secret).abs() < 1e-8);
    }

    #[test]
    fn test_insufficient_parties() {
        let sharing = AdditiveSharing::with_seed(42);
        let parties = test_parties(1);

        let result = sharing.share_scalar(1.0, "test", &parties);
        assert!(result.is_err());
    }
}
