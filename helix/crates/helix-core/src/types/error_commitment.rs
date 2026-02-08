//! Error commitment types for ZK circuit integration.
//!
//! This module provides cryptographic commitments to accumulated error bounds,
//! enabling on-chain verification that training stayed within error budget.
//!
//! # Overview
//!
//! The error checksum is computed as:
//! ```text
//! checksum = SHA256(accumulated_error || step_number || model_id || budget_limit)
//! ```
//!
//! This commitment is included as a public input in ZK proofs, allowing:
//! - On-chain verification that error bounds weren't tampered with
//! - Proof that training stayed within allocated error budget
//! - Linkage between error tracking and specific training steps
//!
//! # Example
//!
//! ```
//! use helix_core::types::error_commitment::{ErrorCommitment, ErrorCommitmentBuilder};
//!
//! let commitment = ErrorCommitmentBuilder::new()
//!     .accumulated_error(0.00123)
//!     .step_number(42)
//!     .model_id([1u8; 32])
//!     .budget_limit(0.01)
//!     .build();
//!
//! let checksum = commitment.compute_checksum();
//! assert!(commitment.verify_within_budget());
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Scale factor for converting f64 error values to u64 (10^12 for 12 decimal places).
const ERROR_SCALE: f64 = 1e12;

/// An error commitment that can be included in ZK proof public inputs.
///
/// The commitment binds together:
/// - The accumulated error bound
/// - The training step number
/// - The model identifier
/// - The error budget limit
///
/// This ensures that error tracking is cryptographically linked to the
/// specific training context and cannot be modified without invalidating
/// the ZK proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorCommitment {
    /// Accumulated error bound (absolute value).
    pub accumulated_error: f64,
    /// Training step number.
    pub step_number: u64,
    /// Model identifier (32 bytes, typically a hash).
    pub model_id: [u8; 32],
    /// Maximum allowed error budget.
    pub budget_limit: f64,
    /// Precomputed checksum (lazily computed if None).
    checksum: Option<[u8; 32]>,
}

impl ErrorCommitment {
    /// Creates a new error commitment.
    pub fn new(
        accumulated_error: f64,
        step_number: u64,
        model_id: [u8; 32],
        budget_limit: f64,
    ) -> Self {
        Self {
            accumulated_error,
            step_number,
            model_id,
            budget_limit,
            checksum: None,
        }
    }

    /// Computes the cryptographic checksum for this commitment.
    ///
    /// The checksum is computed as:
    /// ```text
    /// SHA256(accumulated_error_scaled || step_number || model_id || budget_limit_scaled)
    /// ```
    ///
    /// Where scaled values are u64 representations with 12 decimal places of precision.
    pub fn compute_checksum(&self) -> [u8; 32] {
        if let Some(cached) = self.checksum {
            return cached;
        }

        let mut hasher = Sha256::new();

        // Add accumulated error (scaled to u64)
        let error_scaled = Self::scale_to_u64(self.accumulated_error);
        hasher.update(error_scaled.to_le_bytes());

        // Add step number
        hasher.update(self.step_number.to_le_bytes());

        // Add model ID
        hasher.update(self.model_id);

        // Add budget limit (scaled to u64)
        let budget_scaled = Self::scale_to_u64(self.budget_limit);
        hasher.update(budget_scaled.to_le_bytes());

        let result = hasher.finalize();
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&result);
        checksum
    }

    /// Returns the checksum as two u128 values (lo, hi) for circuit use.
    ///
    /// This split format matches how other commitments (weight hashes) are
    /// represented in the circuit public inputs.
    pub fn checksum_split(&self) -> (u128, u128) {
        let checksum = self.compute_checksum();

        let lo = u128::from_le_bytes(checksum[0..16].try_into().unwrap());
        let hi = u128::from_le_bytes(checksum[16..32].try_into().unwrap());

        (lo, hi)
    }

    /// Returns the checksum as a single u64 for compact representation.
    ///
    /// Uses the first 8 bytes of the SHA256 hash. This provides 64 bits
    /// of collision resistance, which is sufficient for error bound
    /// verification in the context of a larger ZK proof.
    pub fn checksum_compact(&self) -> u64 {
        let checksum = self.compute_checksum();
        u64::from_le_bytes(checksum[0..8].try_into().unwrap())
    }

    /// Verifies that the accumulated error is within the budget limit.
    pub fn verify_within_budget(&self) -> bool {
        self.accumulated_error <= self.budget_limit
    }

    /// Returns the fraction of budget consumed (0.0 to 1.0+).
    pub fn budget_utilization(&self) -> f64 {
        if self.budget_limit <= 0.0 {
            return f64::INFINITY;
        }
        self.accumulated_error / self.budget_limit
    }

    /// Returns the remaining error budget.
    pub fn remaining_budget(&self) -> f64 {
        (self.budget_limit - self.accumulated_error).max(0.0)
    }

    /// Creates an updated commitment for the next step.
    ///
    /// # Arguments
    /// * `additional_error` - Error accumulated in this step
    pub fn next_step(&self, additional_error: f64) -> Self {
        Self::new(
            self.accumulated_error + additional_error,
            self.step_number + 1,
            self.model_id,
            self.budget_limit,
        )
    }

    /// Scales a f64 value to u64 for deterministic hashing.
    fn scale_to_u64(value: f64) -> u64 {
        let scaled = (value.abs() * ERROR_SCALE).min(u64::MAX as f64);
        scaled as u64
    }

    /// Returns the accumulated error as a scaled u64.
    pub fn accumulated_error_scaled(&self) -> u64 {
        Self::scale_to_u64(self.accumulated_error)
    }

    /// Returns the budget limit as a scaled u64.
    pub fn budget_limit_scaled(&self) -> u64 {
        Self::scale_to_u64(self.budget_limit)
    }
}

/// Builder for creating ErrorCommitment instances.
#[derive(Debug, Default)]
pub struct ErrorCommitmentBuilder {
    accumulated_error: Option<f64>,
    step_number: Option<u64>,
    model_id: Option<[u8; 32]>,
    budget_limit: Option<f64>,
}

impl ErrorCommitmentBuilder {
    /// Creates a new builder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the accumulated error.
    pub fn accumulated_error(mut self, error: f64) -> Self {
        self.accumulated_error = Some(error);
        self
    }

    /// Sets the step number.
    pub fn step_number(mut self, step: u64) -> Self {
        self.step_number = Some(step);
        self
    }

    /// Sets the model ID from bytes.
    pub fn model_id(mut self, id: [u8; 32]) -> Self {
        self.model_id = Some(id);
        self
    }

    /// Sets the model ID from a u64 (padded to 32 bytes).
    pub fn model_id_from_u64(mut self, id: u64) -> Self {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&id.to_le_bytes());
        self.model_id = Some(bytes);
        self
    }

    /// Sets the budget limit.
    pub fn budget_limit(mut self, limit: f64) -> Self {
        self.budget_limit = Some(limit);
        self
    }

    /// Builds the ErrorCommitment.
    ///
    /// # Panics
    /// Panics if any required field is not set.
    pub fn build(self) -> ErrorCommitment {
        ErrorCommitment::new(
            self.accumulated_error.expect("accumulated_error is required"),
            self.step_number.expect("step_number is required"),
            self.model_id.expect("model_id is required"),
            self.budget_limit.expect("budget_limit is required"),
        )
    }

    /// Builds the ErrorCommitment, returning None if any field is missing.
    pub fn try_build(self) -> Option<ErrorCommitment> {
        Some(ErrorCommitment::new(
            self.accumulated_error?,
            self.step_number?,
            self.model_id?,
            self.budget_limit?,
        ))
    }
}

/// Tracks error accumulation across multiple training steps with commitment generation.
///
/// This tracker maintains a running commitment that can be efficiently updated
/// as new errors are accumulated during training. The checksum is cached and
/// only recomputed when the state changes (i.e., after `record_step()`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCommitmentTracker {
    /// Current accumulated error.
    accumulated_error: f64,
    /// Current step number.
    current_step: u64,
    /// Model identifier.
    model_id: [u8; 32],
    /// Error budget limit.
    budget_limit: f64,
    /// History of error values per step (optional, for debugging).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    error_history: Vec<f64>,
    /// Whether to track history.
    track_history: bool,
    /// Cached checksum (invalidated on state changes).
    #[serde(skip)]
    cached_checksum: Option<[u8; 32]>,
}

impl ErrorCommitmentTracker {
    /// Creates a new tracker.
    pub fn new(model_id: [u8; 32], budget_limit: f64) -> Self {
        Self {
            accumulated_error: 0.0,
            current_step: 0,
            model_id,
            budget_limit,
            error_history: Vec::new(),
            track_history: false,
            cached_checksum: None,
        }
    }

    /// Creates a new tracker with history tracking enabled.
    pub fn with_history(model_id: [u8; 32], budget_limit: f64) -> Self {
        Self {
            accumulated_error: 0.0,
            current_step: 0,
            model_id,
            budget_limit,
            error_history: Vec::new(),
            track_history: true,
            cached_checksum: None,
        }
    }

    /// Records error for the current step and advances to the next step.
    pub fn record_step(&mut self, step_error: f64) {
        self.accumulated_error += step_error;
        if self.track_history {
            self.error_history.push(step_error);
        }
        self.current_step += 1;
        self.cached_checksum = None; // Invalidate cache
    }

    /// Returns the current commitment.
    pub fn commitment(&self) -> ErrorCommitment {
        ErrorCommitment::new(
            self.accumulated_error,
            self.current_step,
            self.model_id,
            self.budget_limit,
        )
    }

    /// Returns the current checksum, using a cached value when available.
    pub fn checksum(&mut self) -> [u8; 32] {
        if let Some(cached) = self.cached_checksum {
            return cached;
        }
        let checksum = self.commitment().compute_checksum();
        self.cached_checksum = Some(checksum);
        checksum
    }

    /// Returns the compact checksum (64-bit).
    pub fn checksum_compact(&mut self) -> u64 {
        let checksum = self.checksum();
        u64::from_le_bytes(checksum[0..8].try_into().unwrap())
    }

    /// Checks if still within budget.
    pub fn within_budget(&self) -> bool {
        self.accumulated_error <= self.budget_limit
    }

    /// Returns budget utilization (0.0 to 1.0+).
    pub fn utilization(&self) -> f64 {
        if self.budget_limit <= 0.0 {
            return f64::INFINITY;
        }
        self.accumulated_error / self.budget_limit
    }

    /// Returns the current step number.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Returns the accumulated error.
    pub fn accumulated_error(&self) -> f64 {
        self.accumulated_error
    }

    /// Returns the error history if tracking is enabled.
    pub fn error_history(&self) -> &[f64] {
        &self.error_history
    }

    /// Resets the tracker for a new training run.
    pub fn reset(&mut self) {
        self.accumulated_error = 0.0;
        self.current_step = 0;
        self.error_history.clear();
        self.cached_checksum = None;
    }
}

/// Represents the public inputs related to error commitment in a ZK proof.
///
/// This struct matches the format expected by the circuit and smart contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorCommitmentPublicInputs {
    /// Total accumulated error (scaled to u64).
    pub total_error: u64,
    /// Error checksum (compact 64-bit form).
    pub error_checksum: u64,
}

impl ErrorCommitmentPublicInputs {
    /// Creates public inputs from an ErrorCommitment.
    pub fn from_commitment(commitment: &ErrorCommitment) -> Self {
        Self {
            total_error: commitment.accumulated_error_scaled(),
            error_checksum: commitment.checksum_compact(),
        }
    }

    /// Verifies that the checksum matches the given parameters.
    pub fn verify(
        &self,
        accumulated_error: f64,
        step_number: u64,
        model_id: [u8; 32],
        budget_limit: f64,
    ) -> bool {
        let commitment = ErrorCommitment::new(accumulated_error, step_number, model_id, budget_limit);
        let expected = Self::from_commitment(&commitment);
        self.total_error == expected.total_error && self.error_checksum == expected.error_checksum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_commitment_basic() {
        let commitment = ErrorCommitment::new(
            0.001,
            42,
            [1u8; 32],
            0.01,
        );

        assert!(commitment.verify_within_budget());
        assert!((commitment.budget_utilization() - 0.1).abs() < 1e-10);
        assert!((commitment.remaining_budget() - 0.009).abs() < 1e-10);
    }

    #[test]
    fn test_error_commitment_checksum_deterministic() {
        let c1 = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);
        let c2 = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);

        assert_eq!(c1.compute_checksum(), c2.compute_checksum());
    }

    #[test]
    fn test_error_commitment_checksum_changes_with_inputs() {
        let base = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);

        // Different error
        let different_error = ErrorCommitment::new(0.002, 42, [1u8; 32], 0.01);
        assert_ne!(base.compute_checksum(), different_error.compute_checksum());

        // Different step
        let different_step = ErrorCommitment::new(0.001, 43, [1u8; 32], 0.01);
        assert_ne!(base.compute_checksum(), different_step.compute_checksum());

        // Different model ID
        let different_model = ErrorCommitment::new(0.001, 42, [2u8; 32], 0.01);
        assert_ne!(base.compute_checksum(), different_model.compute_checksum());

        // Different budget
        let different_budget = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.02);
        assert_ne!(base.compute_checksum(), different_budget.compute_checksum());
    }

    #[test]
    fn test_error_commitment_next_step() {
        let c1 = ErrorCommitment::new(0.001, 0, [1u8; 32], 0.01);
        let c2 = c1.next_step(0.0005);

        assert_eq!(c2.step_number, 1);
        assert!((c2.accumulated_error - 0.0015).abs() < 1e-15);
        assert_eq!(c2.model_id, c1.model_id);
        assert_eq!(c2.budget_limit, c1.budget_limit);
    }

    #[test]
    fn test_error_commitment_builder() {
        let commitment = ErrorCommitmentBuilder::new()
            .accumulated_error(0.005)
            .step_number(100)
            .model_id([42u8; 32])
            .budget_limit(0.01)
            .build();

        assert!((commitment.accumulated_error - 0.005).abs() < 1e-15);
        assert_eq!(commitment.step_number, 100);
        assert_eq!(commitment.model_id, [42u8; 32]);
        assert!((commitment.budget_limit - 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_error_commitment_tracker() {
        let model_id = [1u8; 32];
        let mut tracker = ErrorCommitmentTracker::with_history(model_id, 0.01);

        // Record some steps
        tracker.record_step(0.001);
        tracker.record_step(0.002);
        tracker.record_step(0.001);

        assert_eq!(tracker.current_step(), 3);
        assert!((tracker.accumulated_error() - 0.004).abs() < 1e-15);
        assert!(tracker.within_budget());
        assert!((tracker.utilization() - 0.4).abs() < 1e-10);

        // Check history
        assert_eq!(tracker.error_history().len(), 3);
        assert!((tracker.error_history()[0] - 0.001).abs() < 1e-15);
        assert!((tracker.error_history()[1] - 0.002).abs() < 1e-15);

        // Checksum should be deterministic and cached
        let checksum1 = tracker.checksum();
        let checksum2 = tracker.checksum(); // Should return cached value
        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn test_error_commitment_tracker_checksum_caching() {
        let model_id = [1u8; 32];
        let mut tracker = ErrorCommitmentTracker::new(model_id, 0.01);

        tracker.record_step(0.001);

        // First call computes and caches
        let checksum1 = tracker.checksum();
        // Second call returns cached value (same result)
        let checksum2 = tracker.checksum();
        assert_eq!(checksum1, checksum2);

        // After record_step, cache is invalidated and recomputed
        tracker.record_step(0.002);
        let checksum3 = tracker.checksum();
        assert_ne!(checksum1, checksum3, "Checksum should change after record_step");

        // Cached value should persist until next mutation
        let checksum4 = tracker.checksum();
        assert_eq!(checksum3, checksum4);
    }

    #[test]
    fn test_error_commitment_tracker_budget_exceeded() {
        let mut tracker = ErrorCommitmentTracker::new([1u8; 32], 0.005);

        tracker.record_step(0.003);
        assert!(tracker.within_budget());

        tracker.record_step(0.003);
        assert!(!tracker.within_budget());
        assert!(tracker.utilization() > 1.0);
    }

    #[test]
    fn test_error_commitment_public_inputs() {
        let commitment = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);
        let public_inputs = ErrorCommitmentPublicInputs::from_commitment(&commitment);

        // Verify reconstruction works
        assert!(public_inputs.verify(0.001, 42, [1u8; 32], 0.01));

        // Wrong parameters should fail
        assert!(!public_inputs.verify(0.002, 42, [1u8; 32], 0.01)); // wrong error
        assert!(!public_inputs.verify(0.001, 43, [1u8; 32], 0.01)); // wrong step
    }

    #[test]
    fn test_checksum_split() {
        let commitment = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);
        let (lo, hi) = commitment.checksum_split();
        let checksum = commitment.compute_checksum();

        // Verify the split reconstructs correctly
        let lo_bytes = lo.to_le_bytes();
        let hi_bytes = hi.to_le_bytes();
        assert_eq!(&checksum[0..16], &lo_bytes);
        assert_eq!(&checksum[16..32], &hi_bytes);
    }

    #[test]
    fn test_scaling_precision() {
        // Test that scaling preserves precision for typical error values
        let commitment = ErrorCommitment::new(0.000000001234, 0, [0u8; 32], 1.0);
        let scaled = commitment.accumulated_error_scaled();

        // With 1e12 scale, 0.000000001234 * 1e12 = 1234
        assert_eq!(scaled, 1234);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = ErrorCommitmentTracker::with_history([1u8; 32], 0.01);
        tracker.record_step(0.001);
        tracker.record_step(0.002);

        tracker.reset();

        assert_eq!(tracker.current_step(), 0);
        assert!((tracker.accumulated_error() - 0.0).abs() < 1e-15);
        assert!(tracker.error_history().is_empty());
    }

    /// Verifies that Rust checksum_compact() produces the exact same result
    /// as Solidity _computeErrorChecksum() for known inputs.
    ///
    /// The Solidity function builds a 56-byte preimage:
    ///   [errorBound as LE u64 (8)] [stepNumber as LE u64 (8)] [modelId (32)] [errorBudget as LE u64 (8)]
    /// Then computes SHA256, and extracts first 8 bytes as LE u64.
    ///
    /// In Solidity, modelId is a uint256 stored as 32-byte big-endian.
    /// In Rust, model_id is [u8; 32] written as-is.
    /// When Solidity passes modelId=1, the 32 bytes are [0,0,...,0,1] (big-endian).
    /// So the Rust model_id should be [0,0,...,0,1] to match Solidity modelId=1.
    #[test]
    fn test_cross_language_checksum_alignment() {
        use sha2::{Digest, Sha256};

        // Known test values matching Solidity test fixtures:
        // errorBound = 1_000_000_000 (0.001 * 1e12)
        // stepNumber = 42
        // modelId = 1 (as uint256, big-endian 32 bytes: [0..0, 1])
        // errorBudget = 10_000_000_000 (0.01 * 1e12)
        let error_bound: u64 = 1_000_000_000;
        let step_number: u64 = 42;
        let mut model_id = [0u8; 32];
        model_id[31] = 1; // Matches Solidity uint256(1) as big-endian bytes32
        let error_budget: u64 = 10_000_000_000;

        // Build the 56-byte preimage manually (matching Solidity layout)
        let mut preimage = Vec::with_capacity(56);
        preimage.extend_from_slice(&error_bound.to_le_bytes());  // 8 bytes
        preimage.extend_from_slice(&step_number.to_le_bytes());  // 8 bytes
        preimage.extend_from_slice(&model_id);                   // 32 bytes
        preimage.extend_from_slice(&error_budget.to_le_bytes()); // 8 bytes
        assert_eq!(preimage.len(), 56);

        // Compute expected checksum
        let mut hasher = Sha256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize();
        let expected_compact = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        // Now compute via ErrorCommitment
        let commitment = ErrorCommitment::new(0.001, 42, model_id, 0.01);

        // Verify scaling matches
        assert_eq!(commitment.accumulated_error_scaled(), error_bound);
        assert_eq!(commitment.budget_limit_scaled(), error_budget);

        // Verify checksum_compact matches the manually computed value
        let actual_compact = commitment.checksum_compact();
        assert_eq!(
            actual_compact, expected_compact,
            "Rust checksum_compact() must match Solidity _computeErrorChecksum().\n\
             Expected: {expected_compact} (0x{expected_compact:016x})\n\
             Actual:   {actual_compact} (0x{actual_compact:016x})"
        );

        // Verify the full checksum also matches
        let full_checksum = commitment.compute_checksum();
        assert_eq!(&full_checksum[..], &hash[..]);
    }

    /// Tests alignment with a second set of known values to guard against
    /// coincidental single-case matches.
    #[test]
    fn test_cross_language_checksum_alignment_second_vector() {
        use sha2::{Digest, Sha256};

        // Second test vector:
        // error = 0.005 → scaled = 5_000_000_000
        // step = 100
        // modelId = 0x4242...42 (all 0x42 bytes)
        // budget = 0.05 → scaled = 50_000_000_000
        let error_bound: u64 = 5_000_000_000;
        let step_number: u64 = 100;
        let model_id = [0x42u8; 32];
        let error_budget: u64 = 50_000_000_000;

        let mut preimage = Vec::with_capacity(56);
        preimage.extend_from_slice(&error_bound.to_le_bytes());
        preimage.extend_from_slice(&step_number.to_le_bytes());
        preimage.extend_from_slice(&model_id);
        preimage.extend_from_slice(&error_budget.to_le_bytes());

        let mut hasher = Sha256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize();
        let expected_compact = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        let commitment = ErrorCommitment::new(0.005, 100, model_id, 0.05);
        assert_eq!(commitment.accumulated_error_scaled(), error_bound);
        assert_eq!(commitment.budget_limit_scaled(), error_budget);
        assert_eq!(commitment.checksum_compact(), expected_compact);
    }

    /// Verify the preimage byte layout matches the Solidity assembly exactly.
    #[test]
    fn test_preimage_byte_layout() {
        // Construct a commitment and verify the internal preimage structure
        let error_val = 0.001;
        let step = 42u64;
        let model_id = [0xABu8; 32];
        let budget = 0.01;

        let commitment = ErrorCommitment::new(error_val, step, model_id, budget);

        // Manually construct what the hash input should be
        let error_scaled = (error_val.abs() * 1e12) as u64;
        let budget_scaled = (budget.abs() * 1e12) as u64;

        let mut expected_preimage = Vec::new();
        expected_preimage.extend_from_slice(&error_scaled.to_le_bytes());
        expected_preimage.extend_from_slice(&step.to_le_bytes());
        expected_preimage.extend_from_slice(&model_id);
        expected_preimage.extend_from_slice(&budget_scaled.to_le_bytes());

        // Hash it and compare
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&expected_preimage);
        let expected_hash: [u8; 32] = hasher.finalize().into();

        assert_eq!(commitment.compute_checksum(), expected_hash);
    }
}
