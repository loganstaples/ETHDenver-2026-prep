//! Share verification: detecting cheating parties.
//!
//! During MPC, parties might try to:
//! 1. Use different shares than they were given
//! 2. Send incorrect d/e values during Beaver multiplication
//! 3. Submit fabricated outputs
//!
//! This module provides checks to detect such misbehavior.

use sha2::{Sha256, Digest};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;
use super::commitment::ShareCommitment;

/// Verifier for share consistency and honest behavior.
pub struct ShareVerifier;

impl ShareVerifier {
    /// Verifies that a party's revealed value matches their commitment.
    pub fn verify_opening(
        commitment: &ShareCommitment,
        revealed_value: &[f64],
        blinding: &[u8; 32],
    ) -> MPCResult<()> {
        if commitment.verify_vector(revealed_value, blinding) {
            Ok(())
        } else {
            Err(MPCError::ShareCommitmentMismatch {
                party: commitment.party.clone(),
            })
        }
    }

    /// Verifies Beaver multiplication consistency.
    ///
    /// After parties open d = x - a and e = y - b, we can check that
    /// each party's d_i and e_i are consistent with their committed
    /// shares of x, a, y, b.
    pub fn verify_beaver_consistency(
        d_share: f64,
        e_share: f64,
        x_commitment: &ShareCommitment,
        y_commitment: &ShareCommitment,
        a_commitment: &ShareCommitment,
        b_commitment: &ShareCommitment,
        x_value: f64,
        y_value: f64,
        a_value: f64,
        b_value: f64,
        x_blinding: &[u8; 32],
        y_blinding: &[u8; 32],
        a_blinding: &[u8; 32],
        b_blinding: &[u8; 32],
    ) -> MPCResult<()> {
        // Verify all commitments open correctly.
        if !x_commitment.verify_scalar(x_value, x_blinding) {
            return Err(MPCError::ShareCommitmentMismatch {
                party: x_commitment.party.clone(),
            });
        }
        if !y_commitment.verify_scalar(y_value, y_blinding) {
            return Err(MPCError::ShareCommitmentMismatch {
                party: y_commitment.party.clone(),
            });
        }
        if !a_commitment.verify_scalar(a_value, a_blinding) {
            return Err(MPCError::ShareCommitmentMismatch {
                party: a_commitment.party.clone(),
            });
        }
        if !b_commitment.verify_scalar(b_value, b_blinding) {
            return Err(MPCError::ShareCommitmentMismatch {
                party: b_commitment.party.clone(),
            });
        }

        // Check d = x - a and e = y - b.
        let expected_d = x_value - a_value;
        let expected_e = y_value - b_value;

        if (d_share - expected_d).abs() > 1e-10 {
            return Err(MPCError::MaliciousBehavior {
                party: x_commitment.party.clone(),
                description: format!(
                    "d_share {} != x - a = {} - {} = {}",
                    d_share, x_value, a_value, expected_d,
                ),
            });
        }

        if (e_share - expected_e).abs() > 1e-10 {
            return Err(MPCError::MaliciousBehavior {
                party: y_commitment.party.clone(),
                description: format!(
                    "e_share {} != y - b = {} - {} = {}",
                    e_share, y_value, b_value, expected_e,
                ),
            });
        }

        Ok(())
    }

    /// Verifies that shares of a secret sum correctly.
    ///
    /// Used for spot-checking: the dealer commits to the secret,
    /// and we can verify sum(shares) == secret.
    pub fn verify_additive_shares(
        shares: &[f64],
        expected_sum: f64,
        tolerance: f64,
    ) -> bool {
        let sum: f64 = shares.iter().sum();
        (sum - expected_sum).abs() <= tolerance
    }

    /// Verifies gradient share norms are within bounds.
    ///
    /// A party submitting a gradient with an unusually large norm
    /// may be trying to poison the model.
    pub fn verify_gradient_norm(
        gradient_share: &[f64],
        max_norm: f64,
    ) -> bool {
        let norm: f64 = gradient_share.iter().map(|v| v * v).sum::<f64>().sqrt();
        norm <= max_norm
    }

    /// Computes a hash-based fingerprint of a share for quick comparison.
    pub fn share_fingerprint(data: &[f64]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in data {
            hasher.update(v.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Verifies that a party's share hasn't been tampered with since the
    /// last known good fingerprint.
    pub fn verify_fingerprint(
        data: &[f64],
        expected_fingerprint: &[u8; 32],
    ) -> bool {
        let current = Self::share_fingerprint(data);
        current == *expected_fingerprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::commitment::BlindingGenerator;

    #[test]
    fn test_verify_opening() {
        let party = PartyId::from_index(0);
        let mut gen = BlindingGenerator::with_seed(42);
        let blinding = gen.generate();
        let values = vec![1.0, 2.0, 3.0];

        let commitment = ShareCommitment::commit_vector(&party, &values, &blinding, "test");

        assert!(ShareVerifier::verify_opening(&commitment, &values, &blinding).is_ok());
        assert!(ShareVerifier::verify_opening(&commitment, &[1.0, 2.0, 4.0], &blinding).is_err());
    }

    #[test]
    fn test_additive_shares_verification() {
        let shares = vec![10.0, 15.0, 17.0]; // sum = 42
        assert!(ShareVerifier::verify_additive_shares(&shares, 42.0, 1e-10));
        assert!(!ShareVerifier::verify_additive_shares(&shares, 43.0, 0.5));
    }

    #[test]
    fn test_gradient_norm() {
        let small_grad = vec![0.1, 0.2, 0.3];
        let large_grad = vec![100.0, 200.0, 300.0];

        assert!(ShareVerifier::verify_gradient_norm(&small_grad, 1.0));
        assert!(!ShareVerifier::verify_gradient_norm(&large_grad, 1.0));
    }

    #[test]
    fn test_fingerprint() {
        let data = vec![1.0, 2.0, 3.0];
        let fp = ShareVerifier::share_fingerprint(&data);
        assert!(ShareVerifier::verify_fingerprint(&data, &fp));
        assert!(!ShareVerifier::verify_fingerprint(&[1.0, 2.0, 4.0], &fp));
    }
}
