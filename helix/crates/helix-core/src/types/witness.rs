//! Training step witness generation with error checksum.
//!
//! This module wires the error checksum through the ZK witness pipeline,
//! computing `SHA256(accumulated_error || step_number || model_id || error_budget)`
//! and exporting it as a `u64` Fr element for circuit public input PI[7].
//!
//! # Overview
//!
//! The [`TrainingStepWitnessData`] struct encapsulates all data for a single
//! training step's ZK proof. Its [`Provable`] implementation:
//!
//! 1. Computes the error checksum via [`ErrorCommitment`]
//! 2. Exports 8 public inputs matching the contract's `submitProof()` format
//! 3. Generates a witness containing all intermediate values
//!
//! # Circuit Public Inputs
//!
//! ```text
//! PI[0]: old_weight_hash_lo  (lower 64 bits of 128-bit LE half)
//! PI[1]: old_weight_hash_hi  (lower 64 bits of upper 128-bit LE half)
//! PI[2]: new_weight_hash_lo  (lower 64 bits of 128-bit LE half)
//! PI[3]: new_weight_hash_hi  (lower 64 bits of upper 128-bit LE half)
//! PI[4]: loss                (f64 scaled to u64 via 1e12)
//! PI[5]: error_bound         (accumulated error, f64 scaled to u64 via 1e12)
//! PI[6]: step_number         (u64)
//! PI[7]: error_checksum      (first 8 bytes of SHA256 as LE u64)
//! ```
//!
//! # Determinism
//!
//! The error checksum computation is fully deterministic:
//! - f64 values are scaled to u64 via `(value.abs() * 1e12).min(u64::MAX) as u64`
//! - All values are serialized as little-endian bytes
//! - SHA256 is a deterministic hash function
//! - The result is the first 8 bytes read as `u64::from_le_bytes`
//!
//! This matches the Solidity `_computeErrorChecksum()` function exactly,
//! and will match the circuit constraint added by A1.
//!
//! # Example
//!
//! ```
//! use helix_core::types::witness::TrainingStepWitnessData;
//! use helix_core::Provable;
//!
//! let step = TrainingStepWitnessData::new(
//!     [0xAA; 32],   // old_weight_hash
//!     [0xBB; 32],   // new_weight_hash
//!     0.5,          // loss
//!     0.001,        // accumulated_error
//!     42,           // step_number
//!     [1u8; 32],    // model_id
//!     0.01,         // error_budget
//! );
//!
//! let witness = step.generate_witness();
//! let pi = step.public_inputs();
//! assert_eq!(pi.len(), 8);
//!
//! // PI[7] is the error checksum
//! let checksum = pi[7];
//! assert_ne!(checksum, 0);
//! ```

use serde::{Deserialize, Serialize};

use crate::traits::provable::{Provable, SimpleWitness};
use super::error_commitment::ErrorCommitment;

/// Scale factor for converting f64 to u64 field elements (10^12).
const ERROR_SCALE: f64 = 1e12;

/// Data needed to generate a training step witness with error checksum.
///
/// This type encapsulates all information for a single training step's ZK proof.
/// When [`Provable::generate_witness`] is called, it computes the real error
/// checksum via SHA256 and includes it in the witness. The [`Provable::public_inputs`]
/// method returns 8 u64 values with the error checksum at PI[7].
///
/// The error checksum is computed as:
/// ```text
/// SHA256(accumulated_error_scaled_LE64 || step_number_LE64 || model_id_32 || error_budget_scaled_LE64)
/// ```
/// where scaled values use `(value.abs() * 1e12).min(u64::MAX) as u64`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingStepWitnessData {
    /// Hash of weights before this training step (32 bytes).
    pub old_weight_hash: [u8; 32],
    /// Hash of weights after this training step (32 bytes).
    pub new_weight_hash: [u8; 32],
    /// Loss value for this step.
    pub loss: f64,
    /// Accumulated error bound up to and including this step.
    pub accumulated_error: f64,
    /// Training step number.
    pub step_number: u64,
    /// Model identifier (32 bytes).
    pub model_id: [u8; 32],
    /// Error budget limit for the training run.
    pub error_budget: f64,
}

impl TrainingStepWitnessData {
    /// Creates new witness data for a training step.
    pub fn new(
        old_weight_hash: [u8; 32],
        new_weight_hash: [u8; 32],
        loss: f64,
        accumulated_error: f64,
        step_number: u64,
        model_id: [u8; 32],
        error_budget: f64,
    ) -> Self {
        Self {
            old_weight_hash,
            new_weight_hash,
            loss,
            accumulated_error,
            step_number,
            model_id,
            error_budget,
        }
    }

    /// Creates witness data from an existing [`ErrorCommitment`] plus step-specific values.
    pub fn from_commitment(
        commitment: &ErrorCommitment,
        old_weight_hash: [u8; 32],
        new_weight_hash: [u8; 32],
        loss: f64,
    ) -> Self {
        Self {
            old_weight_hash,
            new_weight_hash,
            loss,
            accumulated_error: commitment.accumulated_error,
            step_number: commitment.step_number,
            model_id: commitment.model_id,
            error_budget: commitment.budget_limit,
        }
    }

    /// Returns the [`ErrorCommitment`] for this step.
    ///
    /// The commitment encapsulates the SHA256 checksum computation and provides
    /// the canonical error checksum value used as circuit PI[7].
    pub fn error_commitment(&self) -> ErrorCommitment {
        ErrorCommitment::new(
            self.accumulated_error,
            self.step_number,
            self.model_id,
            self.error_budget,
        )
    }

    /// Computes the error checksum as a `u64` Fr element for circuit PI[7].
    ///
    /// This is the canonical checksum computation:
    /// ```text
    /// SHA256(accumulated_error_scaled_LE64 || step_number_LE64 || model_id_32 || error_budget_scaled_LE64)
    /// ```
    /// The first 8 bytes of the hash are read as `u64::from_le_bytes`.
    ///
    /// This value is deterministic and matches:
    /// - The Solidity `_computeErrorChecksum()` function
    /// - The circuit constraint that A1 will add
    /// - The `ErrorCommitment::to_contract_u64()` output
    pub fn error_checksum_fr(&self) -> u64 {
        self.error_commitment().to_contract_u64()
    }

    /// Returns the full 32-byte SHA256 error checksum.
    ///
    /// Useful when the circuit needs the full hash (e.g., for split lo/hi representation).
    pub fn error_checksum_full(&self) -> [u8; 32] {
        self.error_commitment().compute_checksum()
    }

    /// Returns whether this step's accumulated error is within the budget.
    pub fn within_budget(&self) -> bool {
        self.accumulated_error <= self.error_budget
    }

    /// Splits a 32-byte hash into (lo, hi) u64 values for circuit public inputs.
    ///
    /// Takes the first 16 bytes as a LE u128 then truncates to u64 (lo),
    /// and the last 16 bytes as a LE u128 then truncates to u64 (hi).
    /// This matches the contract's `_hashPair(lo, hi)` representation.
    fn hash_to_pi_pair(hash: &[u8; 32]) -> (u64, u64) {
        let lo = u128::from_le_bytes(hash[0..16].try_into().unwrap());
        let hi = u128::from_le_bytes(hash[16..32].try_into().unwrap());
        (lo as u64, hi as u64)
    }

    /// Scales an f64 value to u64 for deterministic circuit representation.
    ///
    /// Uses the same scaling as `ErrorCommitment::scale_to_u64`:
    /// `(value.abs() * 1e12).min(u64::MAX) as u64`
    fn scale_to_u64(value: f64) -> u64 {
        let scaled = (value.abs() * ERROR_SCALE).min(u64::MAX as f64);
        scaled as u64
    }
}

impl Provable for TrainingStepWitnessData {
    type Witness = SimpleWitness;

    /// Generates the witness for a training step ZK proof.
    ///
    /// The witness contains all intermediate values needed by the circuit:
    /// - Weight hashes (old and new, split into u64 chunks)
    /// - Loss and error values (scaled)
    /// - Step metadata
    /// - The computed error checksum
    /// - Model ID and budget (for checksum verification in-circuit)
    fn generate_witness(&self) -> Self::Witness {
        let mut witness = SimpleWitness::empty();

        // Weight hashes as u64 field elements (4 x u64 each = 32 bytes)
        for chunk in self.old_weight_hash.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            witness.push(u64::from_le_bytes(bytes));
        }
        for chunk in self.new_weight_hash.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            witness.push(u64::from_le_bytes(bytes));
        }

        // Loss (scaled to u64)
        witness.push(Self::scale_to_u64(self.loss));

        // Accumulated error (scaled to u64)
        witness.push(Self::scale_to_u64(self.accumulated_error));

        // Step number
        witness.push(self.step_number);

        // Error checksum (the key PI[7] value, computed via SHA256)
        witness.push(self.error_checksum_fr());

        // Model ID as field elements (4 x u64)
        for chunk in self.model_id.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            witness.push(u64::from_le_bytes(bytes));
        }

        // Error budget (scaled to u64)
        witness.push(Self::scale_to_u64(self.error_budget));

        witness
    }

    /// Returns the 8 circuit public inputs for this training step.
    ///
    /// ```text
    /// [old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, loss, error_bound, step_number, error_checksum]
    /// ```
    ///
    /// PI[7] is the error checksum: first 8 bytes of
    /// `SHA256(accumulated_error_scaled || step_number || model_id || error_budget_scaled)`
    /// interpreted as a little-endian u64.
    fn public_inputs(&self) -> Vec<u64> {
        let (old_lo, old_hi) = Self::hash_to_pi_pair(&self.old_weight_hash);
        let (new_lo, new_hi) = Self::hash_to_pi_pair(&self.new_weight_hash);
        let loss_scaled = Self::scale_to_u64(self.loss);

        let commitment = self.error_commitment();
        commitment
            .to_public_inputs(old_lo, old_hi, new_lo, new_hi, loss_scaled)
            .to_vec()
    }

    /// Returns the circuit identifier for ML training step proofs.
    fn circuit_id(&self) -> &'static str {
        "ml_training_step_v2"
    }
}

/// Validates that a set of public inputs contains a correct error checksum.
///
/// Given the 8 public inputs and the error commitment parameters, verifies
/// that PI[7] matches the expected SHA256-based checksum.
///
/// # Arguments
/// * `public_inputs` - The 8 u64 public inputs from a proof
/// * `accumulated_error` - Expected accumulated error
/// * `step_number` - Expected step number
/// * `model_id` - Expected model ID
/// * `error_budget` - Expected error budget
///
/// # Returns
/// `true` if PI[7] matches the recomputed checksum, `false` otherwise.
pub fn verify_pi_error_checksum(
    public_inputs: &[u64],
    accumulated_error: f64,
    step_number: u64,
    model_id: [u8; 32],
    error_budget: f64,
) -> bool {
    if public_inputs.len() < 8 {
        return false;
    }
    let commitment = ErrorCommitment::new(accumulated_error, step_number, model_id, error_budget);
    commitment.verify_contract_checksum(public_inputs[7])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::provable::Witness;

    fn make_test_step() -> TrainingStepWitnessData {
        TrainingStepWitnessData::new(
            [0xAA; 32],
            [0xBB; 32],
            0.5,
            0.001,
            42,
            [1u8; 32],
            0.01,
        )
    }

    #[test]
    fn test_witness_generation() {
        let step = make_test_step();
        let witness = step.generate_witness();
        let elements = witness.to_field_elements();

        // Should contain: 4 (old hash) + 4 (new hash) + 1 (loss) + 1 (error) +
        //                 1 (step) + 1 (checksum) + 4 (model_id) + 1 (budget) = 17
        assert_eq!(elements.len(), 17);
    }

    #[test]
    fn test_public_inputs_length() {
        let step = make_test_step();
        let pi = step.public_inputs();
        assert_eq!(pi.len(), 8);
    }

    #[test]
    fn test_public_inputs_pi7_is_error_checksum() {
        let step = make_test_step();
        let pi = step.public_inputs();

        // PI[7] should match the error checksum
        let expected_checksum = step.error_checksum_fr();
        assert_eq!(pi[7], expected_checksum);
    }

    #[test]
    fn test_error_checksum_deterministic() {
        let step1 = make_test_step();
        let step2 = make_test_step();

        assert_eq!(step1.error_checksum_fr(), step2.error_checksum_fr());
    }

    #[test]
    fn test_error_checksum_changes_with_error() {
        let step1 = make_test_step();
        let mut step2 = make_test_step();
        step2.accumulated_error = 0.002;

        assert_ne!(step1.error_checksum_fr(), step2.error_checksum_fr());
    }

    #[test]
    fn test_error_checksum_changes_with_step_number() {
        let step1 = make_test_step();
        let mut step2 = make_test_step();
        step2.step_number = 43;

        assert_ne!(step1.error_checksum_fr(), step2.error_checksum_fr());
    }

    #[test]
    fn test_error_checksum_changes_with_model_id() {
        let step1 = make_test_step();
        let mut step2 = make_test_step();
        step2.model_id = [2u8; 32];

        assert_ne!(step1.error_checksum_fr(), step2.error_checksum_fr());
    }

    #[test]
    fn test_error_checksum_changes_with_budget() {
        let step1 = make_test_step();
        let mut step2 = make_test_step();
        step2.error_budget = 0.02;

        assert_ne!(step1.error_checksum_fr(), step2.error_checksum_fr());
    }

    #[test]
    fn test_error_checksum_matches_commitment() {
        let step = make_test_step();
        let commitment = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);

        assert_eq!(step.error_checksum_fr(), commitment.to_contract_u64());
    }

    #[test]
    fn test_error_checksum_matches_sha256() {
        use sha2::{Digest, Sha256};

        let step = make_test_step();

        // Manually compute the expected checksum
        let error_scaled = ((0.001_f64).abs() * 1e12).min(u64::MAX as f64) as u64;
        let budget_scaled = ((0.01_f64).abs() * 1e12).min(u64::MAX as f64) as u64;

        let mut hasher = Sha256::new();
        hasher.update(error_scaled.to_le_bytes());
        hasher.update(42u64.to_le_bytes());
        hasher.update([1u8; 32]);
        hasher.update(budget_scaled.to_le_bytes());
        let hash = hasher.finalize();
        let expected = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        assert_eq!(step.error_checksum_fr(), expected);
    }

    #[test]
    fn test_public_inputs_structure() {
        let step = make_test_step();
        let pi = step.public_inputs();

        // PI[5] should be the scaled accumulated error
        let expected_error_scaled = ((0.001_f64).abs() * 1e12) as u64;
        assert_eq!(pi[5], expected_error_scaled);

        // PI[6] should be the step number
        assert_eq!(pi[6], 42);

        // PI[7] should be the error checksum
        assert_eq!(pi[7], step.error_checksum_fr());
    }

    #[test]
    fn test_circuit_id() {
        let step = make_test_step();
        assert_eq!(step.circuit_id(), "ml_training_step_v2");
    }

    #[test]
    fn test_from_commitment() {
        let commitment = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);
        let step = TrainingStepWitnessData::from_commitment(
            &commitment,
            [0xAA; 32],
            [0xBB; 32],
            0.5,
        );

        assert_eq!(step.accumulated_error, 0.001);
        assert_eq!(step.step_number, 42);
        assert_eq!(step.model_id, [1u8; 32]);
        assert_eq!(step.error_budget, 0.01);
        assert_eq!(step.error_checksum_fr(), commitment.to_contract_u64());
    }

    #[test]
    fn test_within_budget() {
        let step = make_test_step();
        assert!(step.within_budget()); // 0.001 <= 0.01

        let over_budget = TrainingStepWitnessData::new(
            [0xAA; 32],
            [0xBB; 32],
            0.5,
            0.02, // exceeds budget of 0.01
            42,
            [1u8; 32],
            0.01,
        );
        assert!(!over_budget.within_budget());
    }

    #[test]
    fn test_verify_pi_error_checksum() {
        let step = make_test_step();
        let pi = step.public_inputs();

        assert!(verify_pi_error_checksum(&pi, 0.001, 42, [1u8; 32], 0.01));
        assert!(!verify_pi_error_checksum(&pi, 0.002, 42, [1u8; 32], 0.01));
        assert!(!verify_pi_error_checksum(&pi, 0.001, 43, [1u8; 32], 0.01));
    }

    #[test]
    fn test_verify_pi_error_checksum_short_input() {
        assert!(!verify_pi_error_checksum(&[0; 7], 0.001, 42, [1u8; 32], 0.01));
    }

    #[test]
    fn test_witness_includes_checksum_in_elements() {
        let step = make_test_step();
        let witness = step.generate_witness();
        let elements = witness.to_field_elements();

        // The error checksum should be at index 11 (after 4+4+1+1+1)
        let checksum_idx = 11;
        assert_eq!(elements[checksum_idx], step.error_checksum_fr());
    }

    #[test]
    fn test_cross_language_alignment() {
        // Use the same test vector as error_commitment cross-language tests
        let model_id = ErrorCommitment::model_id_from_uint256(1);
        let step = TrainingStepWitnessData::new(
            [0xAA; 32],
            [0xBB; 32],
            0.5,
            0.001,
            42,
            model_id,
            0.01,
        );

        let pi = step.public_inputs();

        // Verify against manually computed Solidity-compatible checksum
        use sha2::{Digest, Sha256};
        let mut preimage = Vec::with_capacity(56);
        preimage.extend_from_slice(&1_000_000_000u64.to_le_bytes()); // 0.001 * 1e12
        preimage.extend_from_slice(&42u64.to_le_bytes());
        preimage.extend_from_slice(&model_id);
        preimage.extend_from_slice(&10_000_000_000u64.to_le_bytes()); // 0.01 * 1e12

        let mut hasher = Sha256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize();
        let expected = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        assert_eq!(pi[7], expected);
    }

    #[test]
    fn test_error_checksum_full() {
        let step = make_test_step();
        let full = step.error_checksum_full();
        let compact = step.error_checksum_fr();

        // Compact should be first 8 bytes of full
        let from_full = u64::from_le_bytes(full[0..8].try_into().unwrap());
        assert_eq!(compact, from_full);
    }

    #[test]
    fn test_witness_determinism() {
        let step = make_test_step();
        let w1 = step.generate_witness();
        let w2 = step.generate_witness();
        assert_eq!(w1.to_field_elements(), w2.to_field_elements());
    }
}
