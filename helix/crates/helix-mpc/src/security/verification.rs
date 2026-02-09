//! Share verification: detecting cheating parties.
//!
//! During MPC, parties might try to:
//! 1. Use different shares than they were given
//! 2. Send incorrect d/e values during Beaver multiplication
//! 3. Submit fabricated outputs
//! 4. Provide inconsistent shares across operations
//! 5. Perform replay attacks with old shares
//!
//! This module provides comprehensive checks to detect such misbehavior.
//!
//! # Verification Types
//!
//! - **Opening verification**: Checks revealed values match commitments
//! - **Beaver consistency**: Validates multiplication protocol execution
//! - **Share consistency**: Ensures shares sum correctly
//! - **Gradient bounds**: Prevents gradient poisoning attacks
//! - **Cross-round consistency**: Detects replay attacks
//! - **Statistical verification**: Probabilistic checks for efficiency

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use super::commitment::ShareCommitment;
use crate::error::{MPCError, MPCResult};
use crate::field::ct_eq_hash;
use crate::types::PartyId;

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
    /// last known good fingerprint (constant-time comparison).
    pub fn verify_fingerprint(
        data: &[f64],
        expected_fingerprint: &[u8; 32],
    ) -> bool {
        let current = Self::share_fingerprint(data);
        ct_eq_hash(&current, expected_fingerprint).to_bool()
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

// ============================================================================
// Advanced Consistency Checking
// ============================================================================

/// Tracks share state for consistency verification across operations.
#[derive(Debug)]
pub struct ShareConsistencyTracker {
    /// Known fingerprints for each party's shares
    party_fingerprints: HashMap<String, HashMap<String, [u8; 32]>>,
    /// Sequence numbers for replay detection
    sequence_numbers: HashMap<String, u64>,
    /// Historical share values for trend detection
    share_history: HashMap<String, Vec<f64>>,
    /// Maximum allowed share deviation
    max_deviation: f64,
}

impl ShareConsistencyTracker {
    pub fn new(max_deviation: f64) -> Self {
        Self {
            party_fingerprints: HashMap::new(),
            sequence_numbers: HashMap::new(),
            share_history: HashMap::new(),
            max_deviation,
        }
    }

    /// Registers a share for a party.
    pub fn register_share(
        &mut self,
        party_id: &str,
        share_id: &str,
        share_data: &[f64],
    ) {
        let fingerprint = ShareVerifier::share_fingerprint(share_data);

        let party_fps = self
            .party_fingerprints
            .entry(party_id.to_string())
            .or_insert_with(HashMap::new);

        party_fps.insert(share_id.to_string(), fingerprint);

        // Track history for the first element (for trend analysis)
        if !share_data.is_empty() {
            let key = format!("{}:{}", party_id, share_id);
            self.share_history
                .entry(key)
                .or_insert_with(Vec::new)
                .push(share_data[0]);
        }
    }

    /// Verifies that a share hasn't changed since registration.
    pub fn verify_share_unchanged(
        &self,
        party_id: &str,
        share_id: &str,
        share_data: &[f64],
    ) -> bool {
        if let Some(party_fps) = self.party_fingerprints.get(party_id) {
            if let Some(expected_fp) = party_fps.get(share_id) {
                return ShareVerifier::verify_fingerprint(share_data, expected_fp);
            }
        }
        // No previous registration - this is the first time
        true
    }

    /// Checks if the sequence number is valid (not a replay).
    pub fn check_sequence(&mut self, party_id: &str, seq: u64) -> bool {
        let current = self.sequence_numbers.entry(party_id.to_string()).or_insert(0);

        if seq > *current {
            *current = seq;
            true
        } else {
            // Replay detected!
            false
        }
    }

    /// Detects anomalous share changes (potential cheating).
    pub fn detect_anomaly(&self, party_id: &str, share_id: &str, new_value: f64) -> Option<String> {
        let key = format!("{}:{}", party_id, share_id);

        if let Some(history) = self.share_history.get(&key) {
            if history.len() < 2 {
                return None;
            }

            // Compute expected change based on history
            let last = history[history.len() - 1];
            let second_last = history[history.len() - 2];
            let expected_delta = last - second_last;

            let actual_delta = new_value - last;

            // Check if change is anomalously large
            if (actual_delta - expected_delta).abs() > self.max_deviation {
                return Some(format!(
                    "Anomalous change detected: expected delta ~{:.4}, got {:.4}",
                    expected_delta, actual_delta
                ));
            }
        }

        None
    }
}

/// Cross-party share verification.
pub struct CrossPartyVerifier {
    /// Number of parties
    num_parties: usize,
    /// Statistical security parameter
    security_bits: usize,
}

impl CrossPartyVerifier {
    pub fn new(num_parties: usize, security_bits: usize) -> Self {
        Self {
            num_parties,
            security_bits,
        }
    }

    /// Verifies that all parties' shares of a value sum correctly.
    pub fn verify_share_sum(
        &self,
        shares: &[f64],
        expected_sum: f64,
    ) -> MPCResult<()> {
        if shares.len() != self.num_parties {
            return Err(MPCError::ShareCountMismatch {
                expected: self.num_parties,
                got: shares.len(),
            });
        }

        let actual_sum: f64 = shares.iter().sum();
        if (actual_sum - expected_sum).abs() > 1e-6 {
            return Err(MPCError::ReconstructionFailed {
                reason: format!(
                    "Share sum {} != expected {}",
                    actual_sum, expected_sum
                ),
            });
        }

        Ok(())
    }

    /// Performs statistical verification using random spot checks.
    ///
    /// Instead of checking all values, we sample random positions
    /// and verify those. This is faster but provides probabilistic guarantees.
    pub fn statistical_verify(
        &self,
        all_shares: &[Vec<f64>],
        expected_sums: &[f64],
        seed: u64,
    ) -> MPCResult<()> {
        let num_values = expected_sums.len();
        if num_values == 0 {
            return Ok(());
        }

        // Number of samples based on security parameter
        let num_samples = (self.security_bits * 2).min(num_values);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Random sampling
        let mut checked = 0;
        while checked < num_samples {
            let idx = rng.gen_range(0..num_values);

            // Sum shares for this index
            let sum: f64 = all_shares
                .iter()
                .map(|party_shares| party_shares.get(idx).copied().unwrap_or(0.0))
                .sum();

            if (sum - expected_sums[idx]).abs() > 1e-6 {
                return Err(MPCError::MaliciousBehavior {
                    party: PartyId::new("unknown"),
                    description: format!(
                        "Statistical check failed at index {}: sum {} != expected {}",
                        idx, sum, expected_sums[idx]
                    ),
                });
            }

            checked += 1;
        }

        Ok(())
    }

    /// Verifies linear combinations of shares.
    ///
    /// This is useful for batch verification: instead of checking n equations,
    /// we take a random linear combination and check one equation.
    pub fn verify_random_linear_combination(
        &self,
        all_shares: &[Vec<f64>],
        expected_sums: &[f64],
        seed: u64,
    ) -> MPCResult<()> {
        let num_values = expected_sums.len();
        if num_values == 0 {
            return Ok(());
        }

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random coefficients
        let coeffs: Vec<f64> = (0..num_values).map(|_| rng.gen_range(1.0..1000.0)).collect();

        // Compute random linear combination of shares
        let combined_sum: f64 = (0..num_values)
            .map(|idx| {
                let share_sum: f64 = all_shares
                    .iter()
                    .map(|party| party.get(idx).copied().unwrap_or(0.0))
                    .sum();
                coeffs[idx] * share_sum
            })
            .sum();

        // Expected combined value
        let expected_combined: f64 = coeffs
            .iter()
            .zip(expected_sums.iter())
            .map(|(c, e)| c * e)
            .sum();

        if (combined_sum - expected_combined).abs() > 1e-3 {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new("unknown"),
                description: "Random linear combination check failed".into(),
            });
        }

        Ok(())
    }
}

/// Beaver triple verification.
pub struct BeaverTripleVerifier {
    /// Tolerance for floating-point comparison
    tolerance: f64,
}

impl BeaverTripleVerifier {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Verifies that a set of Beaver triple shares is valid.
    ///
    /// For shares (a_i, b_i, c_i) from each party:
    /// - sum(a_i) = a
    /// - sum(b_i) = b
    /// - sum(c_i) = c
    /// - c = a * b
    pub fn verify_triple(
        &self,
        a_shares: &[f64],
        b_shares: &[f64],
        c_shares: &[f64],
    ) -> MPCResult<()> {
        let a: f64 = a_shares.iter().sum();
        let b: f64 = b_shares.iter().sum();
        let c: f64 = c_shares.iter().sum();

        let expected_c = a * b;
        if (c - expected_c).abs() > self.tolerance {
            return Err(MPCError::BeaverVerificationFailed);
        }

        Ok(())
    }

    /// Verifies Beaver multiplication result.
    ///
    /// Given the protocol output shares [xy]_i and the opened d, e values,
    /// verifies that sum([xy]_i) = x * y.
    pub fn verify_multiplication(
        &self,
        x_shares: &[f64],
        y_shares: &[f64],
        result_shares: &[f64],
    ) -> MPCResult<()> {
        let x: f64 = x_shares.iter().sum();
        let y: f64 = y_shares.iter().sum();
        let result: f64 = result_shares.iter().sum();

        let expected = x * y;
        if (result - expected).abs() > self.tolerance {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new("unknown"),
                description: format!(
                    "Multiplication result {} != expected {} = {} * {}",
                    result, expected, x, y
                ),
            });
        }

        Ok(())
    }

    /// Batch verifies multiple multiplications.
    pub fn batch_verify_multiplications(
        &self,
        operations: &[(Vec<f64>, Vec<f64>, Vec<f64>)],
        seed: u64,
    ) -> MPCResult<()> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random coefficients
        let coeffs: Vec<f64> = (0..operations.len())
            .map(|_| rng.gen_range(1.0..1000.0))
            .collect();

        // Compute combined check
        let mut combined_expected = 0.0;
        let mut combined_actual = 0.0;

        for (i, (x_shares, y_shares, result_shares)) in operations.iter().enumerate() {
            let x: f64 = x_shares.iter().sum();
            let y: f64 = y_shares.iter().sum();
            let result: f64 = result_shares.iter().sum();

            combined_expected += coeffs[i] * x * y;
            combined_actual += coeffs[i] * result;
        }

        if (combined_actual - combined_expected).abs() > self.tolerance * operations.len() as f64 {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new("unknown"),
                description: "Batch multiplication verification failed".into(),
            });
        }

        Ok(())
    }
}

/// Output verification for final results.
pub struct OutputVerifier {
    /// Commitments to expected outputs
    output_commitments: HashMap<String, [u8; 32]>,
}

impl OutputVerifier {
    pub fn new() -> Self {
        Self {
            output_commitments: HashMap::new(),
        }
    }

    /// Registers an expected output commitment.
    pub fn expect_output(&mut self, id: &str, values: &[f64]) {
        let commitment = ShareVerifier::share_fingerprint(values);
        self.output_commitments.insert(id.to_string(), commitment);
    }

    /// Verifies a revealed output matches the commitment.
    pub fn verify_output(&self, id: &str, values: &[f64]) -> bool {
        if let Some(expected) = self.output_commitments.get(id) {
            ShareVerifier::verify_fingerprint(values, expected)
        } else {
            false
        }
    }

    /// Verifies output is within expected bounds.
    pub fn verify_output_bounds(
        &self,
        values: &[f64],
        min: f64,
        max: f64,
    ) -> bool {
        values.iter().all(|v| *v >= min && *v <= max)
    }

    /// Verifies gradient output is reasonable (not poisoned).
    pub fn verify_gradient_output(
        &self,
        gradients: &[f64],
        max_norm: f64,
        max_element: f64,
    ) -> MPCResult<()> {
        // Check individual elements
        for (_i, g) in gradients.iter().enumerate() {
            if g.abs() > max_element {
                return Err(MPCError::ErrorBoundExceeded {
                    computed: g.abs(),
                    maximum: max_element,
                });
            }
        }

        // Check L2 norm
        let norm: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm > max_norm {
            return Err(MPCError::ErrorBoundExceeded {
                computed: norm,
                maximum: max_norm,
            });
        }

        Ok(())
    }
}

impl Default for OutputVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests_advanced {
    use super::*;

    #[test]
    fn test_share_consistency_tracker() {
        let mut tracker = ShareConsistencyTracker::new(10.0);

        let party = "party-0";
        let share_id = "weights-0";
        let data = vec![1.0, 2.0, 3.0];

        tracker.register_share(party, share_id, &data);
        assert!(tracker.verify_share_unchanged(party, share_id, &data));
        assert!(!tracker.verify_share_unchanged(party, share_id, &[1.0, 2.0, 4.0]));
    }

    #[test]
    fn test_sequence_replay_detection() {
        let mut tracker = ShareConsistencyTracker::new(10.0);

        assert!(tracker.check_sequence("party-0", 1));
        assert!(tracker.check_sequence("party-0", 2));
        assert!(!tracker.check_sequence("party-0", 1)); // Replay!
        assert!(tracker.check_sequence("party-0", 5)); // Jump OK
    }

    #[test]
    fn test_cross_party_verification() {
        let verifier = CrossPartyVerifier::new(3, 40);

        let shares = vec![vec![1.0], vec![2.0], vec![3.0]]; // sum = 6
        let expected = vec![6.0];

        assert!(verifier.statistical_verify(&shares, &expected, 42).is_ok());
        assert!(verifier.verify_random_linear_combination(&shares, &expected, 42).is_ok());
    }

    #[test]
    fn test_beaver_verification() {
        let verifier = BeaverTripleVerifier::new(1e-6);

        // Valid triple: a=3, b=4, c=12
        let a_shares = vec![1.0, 1.0, 1.0]; // sum = 3
        let b_shares = vec![1.5, 1.5, 1.0]; // sum = 4
        let c_shares = vec![4.0, 4.0, 4.0]; // sum = 12

        assert!(verifier.verify_triple(&a_shares, &b_shares, &c_shares).is_ok());

        // Invalid triple
        let bad_c = vec![4.0, 4.0, 5.0]; // sum = 13 != 3*4
        assert!(verifier.verify_triple(&a_shares, &b_shares, &bad_c).is_err());
    }

    #[test]
    fn test_output_verifier() {
        let mut verifier = OutputVerifier::new();

        let output = vec![1.0, 2.0, 3.0];
        verifier.expect_output("result", &output);

        assert!(verifier.verify_output("result", &output));
        assert!(!verifier.verify_output("result", &[1.0, 2.0, 4.0]));
        assert!(verifier.verify_output_bounds(&output, 0.0, 5.0));
    }

    #[test]
    fn test_gradient_verification() {
        let verifier = OutputVerifier::new();

        let good_gradient = vec![0.1, -0.2, 0.15];
        assert!(verifier.verify_gradient_output(&good_gradient, 1.0, 0.5).is_ok());

        let poisoned = vec![0.1, 100.0, 0.15];
        assert!(verifier.verify_gradient_output(&poisoned, 1.0, 0.5).is_err());
    }
}
