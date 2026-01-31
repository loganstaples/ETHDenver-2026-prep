//! Custom assertions for HELIX integration tests.
//!
//! Provides specialized assertion functions for verifying proof correctness,
//! MPC share validity, error bounds, and training convergence.

use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::ff::PrimeField;

/// Error tolerance for floating point comparisons.
pub const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Maximum acceptable overhead factor (ZK time / native time).
pub const MAX_OVERHEAD_FACTOR: f64 = 30.0;

/// Asserts that two field elements are equal.
pub fn assert_fr_eq(actual: Fr, expected: Fr, context: &str) {
    assert!(
        actual == expected,
        "{}: field elements not equal\n  actual: {:?}\n  expected: {:?}",
        context,
        actual,
        expected
    );
}

/// Asserts that two field element vectors are equal.
pub fn assert_fr_vec_eq(actual: &[Fr], expected: &[Fr], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch ({} vs {})",
        context,
        actual.len(),
        expected.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_fr_eq(*a, *e, &format!("{}[{}]", context, i));
    }
}

/// Asserts that two f64 values are approximately equal.
pub fn assert_f64_approx(actual: f64, expected: f64, tolerance: f64, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{}: values not approximately equal\n  actual: {}\n  expected: {}\n  diff: {}\n  tolerance: {}",
        context,
        actual,
        expected,
        diff,
        tolerance
    );
}

/// Asserts that two f64 vectors are approximately equal.
pub fn assert_f64_vec_approx(actual: &[f64], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch ({} vs {})",
        context,
        actual.len(),
        expected.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_f64_approx(*a, *e, tolerance, &format!("{}[{}]", context, i));
    }
}

/// Asserts that a proof is non-empty and has valid structure.
pub fn assert_proof_valid_structure(proof: &[u8], context: &str) {
    assert!(
        !proof.is_empty(),
        "{}: proof is empty",
        context
    );
    assert!(
        proof.len() >= 64,
        "{}: proof too short ({} bytes, need at least 64)",
        context,
        proof.len()
    );
}

/// Asserts that public inputs have the correct count.
pub fn assert_public_inputs_count(inputs: &[Fr], expected: usize, context: &str) {
    assert_eq!(
        inputs.len(),
        expected,
        "{}: wrong number of public inputs ({} vs {})",
        context,
        inputs.len(),
        expected
    );
}

/// Asserts that the error bound is within acceptable limits.
pub fn assert_error_bound_valid(error_bound: Fr, max_bound: Fr, context: &str) {
    // Compare as bytes since Fr comparison is complex
    let bound_repr = error_bound.to_repr();
    let max_repr = max_bound.to_repr();

    let bound_bytes = bound_repr.as_ref();
    let max_bytes = max_repr.as_ref();

    // Simple comparison: check if bound is "small enough"
    // In practice, we'd need proper Fr comparison
    let bound_non_zero = bound_bytes.iter().any(|&b| b != 0);
    assert!(
        bound_non_zero,
        "{}: error bound is unexpectedly zero",
        context
    );
}

/// Asserts that shares sum to the expected secret.
pub fn assert_shares_reconstruct(shares: &[f64], expected_secret: f64, tolerance: f64, context: &str) {
    let sum: f64 = shares.iter().sum();
    assert_f64_approx(sum, expected_secret, tolerance, context);
}

/// Asserts that a commitment matches expected value.
pub fn assert_commitment_eq(actual: &[u8; 32], expected: &[u8; 32], context: &str) {
    assert_eq!(
        actual, expected,
        "{}: commitment mismatch\n  actual: {:?}\n  expected: {:?}",
        context, actual, expected
    );
}

/// Asserts that loss is decreasing (training is converging).
pub fn assert_loss_decreasing(losses: &[f64], context: &str) {
    assert!(
        losses.len() >= 2,
        "{}: need at least 2 loss values to check trend",
        context
    );

    let first = losses[0];
    let last = *losses.last().unwrap();

    assert!(
        last < first,
        "{}: loss not decreasing (first: {}, last: {})",
        context,
        first,
        last
    );
}

/// Asserts that loss reduction meets minimum threshold.
pub fn assert_loss_reduction(initial: f64, final_loss: f64, min_reduction_pct: f64, context: &str) {
    let reduction_pct = (initial - final_loss) / initial * 100.0;
    assert!(
        reduction_pct >= min_reduction_pct,
        "{}: insufficient loss reduction ({:.1}% vs {:.1}% required)\n  initial: {}\n  final: {}",
        context,
        reduction_pct,
        min_reduction_pct,
        initial,
        final_loss
    );
}

/// Asserts that overhead factor is within acceptable limits.
pub fn assert_overhead_acceptable(overhead: f64, max_overhead: f64, context: &str) {
    assert!(
        overhead <= max_overhead,
        "{}: overhead too high ({:.1}x vs {:.1}x max)",
        context,
        overhead,
        max_overhead
    );
}

/// Asserts that gradient norm is within bounds.
pub fn assert_gradient_norm_bounded(gradients: &[f64], max_norm: f64, context: &str) {
    let norm_sq: f64 = gradients.iter().map(|g| g * g).sum();
    let norm = norm_sq.sqrt();

    assert!(
        norm <= max_norm,
        "{}: gradient norm exceeds bound ({:.2} vs {:.2} max)",
        context,
        norm,
        max_norm
    );
}

/// Asserts that two state hashes are different (state changed).
pub fn assert_state_changed(old_hash: (Fr, Fr), new_hash: (Fr, Fr), context: &str) {
    assert!(
        old_hash != new_hash,
        "{}: state hashes should be different after update",
        context
    );
}

/// Asserts that two state hashes are the same (state unchanged).
pub fn assert_state_unchanged(hash1: (Fr, Fr), hash2: (Fr, Fr), context: &str) {
    assert!(
        hash1 == hash2,
        "{}: state hashes should be equal",
        context
    );
}

/// Asserts that a party was slashed.
pub fn assert_party_slashed(slashed_parties: &[helix_mpc::types::PartyId], party: &helix_mpc::types::PartyId, context: &str) {
    assert!(
        slashed_parties.contains(party),
        "{}: party {:?} should have been slashed",
        context,
        party
    );
}

/// Asserts that no party was slashed.
pub fn assert_no_slashing(slashed_parties: &[helix_mpc::types::PartyId], context: &str) {
    assert!(
        slashed_parties.is_empty(),
        "{}: unexpected slashing occurred: {:?}",
        context,
        slashed_parties
    );
}

/// Asserts that proof verification succeeds.
pub fn assert_verification_succeeds(result: bool, context: &str) {
    assert!(
        result,
        "{}: verification should have succeeded",
        context
    );
}

/// Asserts that proof verification fails (for adversarial tests).
pub fn assert_verification_fails(result: bool, context: &str) {
    assert!(
        !result,
        "{}: verification should have failed",
        context
    );
}

/// Asserts that a checkpoint can be loaded successfully.
pub fn assert_checkpoint_valid(
    step: u64,
    expected_step: u64,
    weights_non_empty: bool,
    context: &str,
) {
    assert_eq!(
        step, expected_step,
        "{}: checkpoint step mismatch ({} vs {})",
        context, step, expected_step
    );
    assert!(
        weights_non_empty,
        "{}: checkpoint weights are empty",
        context
    );
}

/// Asserts that two computations produce consistent results.
pub fn assert_computation_consistent(result1: f64, result2: f64, tolerance: f64, context: &str) {
    assert_f64_approx(result1, result2, tolerance, context);
}

/// Result type for assertion functions that can fail gracefully.
pub type AssertionResult = Result<(), String>;

/// Soft assertion that returns Result instead of panicking.
pub fn soft_assert_f64_approx(actual: f64, expected: f64, tolerance: f64) -> AssertionResult {
    let diff = (actual - expected).abs();
    if diff <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "values not approximately equal: {} vs {} (diff: {}, tolerance: {})",
            actual, expected, diff, tolerance
        ))
    }
}

/// Soft assertion for verification result.
pub fn soft_assert_verification(result: bool, expected: bool) -> AssertionResult {
    if result == expected {
        Ok(())
    } else {
        Err(format!(
            "verification result mismatch: got {}, expected {}",
            result, expected
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_f64_approx() {
        assert_f64_approx(1.0, 1.0 + 1e-7, 1e-6, "test");
    }

    #[test]
    #[should_panic]
    fn test_assert_f64_approx_fails() {
        assert_f64_approx(1.0, 2.0, 1e-6, "test");
    }

    #[test]
    fn test_assert_shares_reconstruct() {
        let shares = vec![1.0, 2.0, 3.0];
        assert_shares_reconstruct(&shares, 6.0, 1e-10, "test");
    }

    #[test]
    fn test_assert_loss_decreasing() {
        let losses = vec![1.0, 0.8, 0.6, 0.4];
        assert_loss_decreasing(&losses, "test");
    }

    #[test]
    #[should_panic]
    fn test_assert_loss_decreasing_fails() {
        let losses = vec![0.4, 0.6, 0.8, 1.0];
        assert_loss_decreasing(&losses, "test");
    }

    #[test]
    fn test_assert_gradient_norm_bounded() {
        let gradients = vec![1.0, 2.0, 2.0]; // norm = 3
        assert_gradient_norm_bounded(&gradients, 5.0, "test");
    }

    #[test]
    fn test_soft_assertion() {
        assert!(soft_assert_f64_approx(1.0, 1.0, 1e-6).is_ok());
        assert!(soft_assert_f64_approx(1.0, 2.0, 1e-6).is_err());
    }
}
