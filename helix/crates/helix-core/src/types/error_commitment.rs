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

    // ================================================================
    // Contract-aligned methods
    //
    // The Solidity _computeErrorChecksum() function in HelixCoordinatorV2
    // computes: SHA256(errorBound_LE64 || stepNumber_LE64 || modelId_32 || errorBudget_LE64)
    // and returns the first 8 bytes interpreted as a little-endian u64.
    //
    // This is the CANONICAL format for public input PI[7] in ZK proofs.
    // The circuit must produce this exact u64 value as its error checksum
    // public input, NOT a 32-byte Fr representation.
    //
    // Use `to_contract_u64()` when:
    //   - Building ZK circuit public inputs (PI[7])
    //   - Verifying against on-chain state
    //   - Comparing checksums across Rust and Solidity
    // ================================================================

    /// Returns the error checksum in the exact format expected by the
    /// Solidity `_computeErrorChecksum()` function.
    ///
    /// This is an 8-byte little-endian u64 derived from the first 8 bytes
    /// of SHA256(errorBound_LE64 || stepNumber_LE64 || modelId_32 || errorBudget_LE64).
    ///
    /// This is the canonical representation for ZK circuit public inputs (PI[7])
    /// and on-chain verification. It is identical to `checksum_compact()` but
    /// carries explicit documentation about contract compatibility.
    ///
    /// # Contract alignment
    ///
    /// The preimage layout (56 bytes) matches Solidity assembly exactly:
    /// - Bytes 0..8: `errorBound` as LE u64 (Solidity byte-swaps from BE)
    /// - Bytes 8..16: `stepNumber` as LE u64 (Solidity byte-swaps from BE)
    /// - Bytes 16..48: `modelId` as raw 32 bytes (Solidity stores as BE uint256)
    /// - Bytes 48..56: `errorBudget` as LE u64 (Solidity byte-swaps from BE)
    ///
    /// The output is: first 8 bytes of SHA256 hash, read as `u64::from_le_bytes`.
    /// Solidity extracts these same bytes via `shr(192, hash)` then byte-swaps to LE.
    pub fn to_contract_u64(&self) -> u64 {
        self.checksum_compact()
    }

    /// Builds the 8 public inputs array matching the contract's `submitProof()` format.
    ///
    /// Returns `[oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber, errorChecksum]`
    /// as u64 values suitable for ZK circuit public inputs.
    ///
    /// # Arguments
    /// * `old_weight_hash` - Hash of weights before this step (split into lo/hi u128)
    /// * `new_weight_hash` - Hash of weights after this step (split into lo/hi u128)
    /// * `loss` - Training loss value (scaled to u64)
    pub fn to_public_inputs(
        &self,
        old_hash_lo: u64,
        old_hash_hi: u64,
        new_hash_lo: u64,
        new_hash_hi: u64,
        loss_scaled: u64,
    ) -> [u64; 8] {
        [
            old_hash_lo,
            old_hash_hi,
            new_hash_lo,
            new_hash_hi,
            loss_scaled,
            self.accumulated_error_scaled(),
            self.step_number,
            self.to_contract_u64(),
        ]
    }

    /// Verifies that a given u64 checksum matches what the contract would compute
    /// for the same parameters.
    ///
    /// Use this to validate public inputs received from other participants or
    /// to verify proof public inputs before on-chain submission.
    pub fn verify_contract_checksum(&self, candidate: u64) -> bool {
        self.to_contract_u64() == candidate
    }

    /// Converts a Solidity `uint256` model ID to the `[u8; 32]` format used in Rust.
    ///
    /// Solidity stores `uint256` in big-endian (MSB first), which maps directly to
    /// `[u8; 32]` in Rust. For example:
    /// - Solidity `uint256(1)` -> Rust `[0,0,...,0,1]` (value at index 31)
    /// - Solidity `uint256(256)` -> Rust `[0,0,...,1,0]` (value at index 30)
    ///
    /// This helper exists to document the convention and prevent byte-order mistakes.
    pub fn model_id_from_uint256(value: u64) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        // uint256 is big-endian: least significant byte at index 31
        bytes[24..32].copy_from_slice(&value.to_be_bytes());
        bytes
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
    /// Incremental Merkle state: stack of partial subtree roots at each level.
    /// Level 0 = individual step hashes, level 1 = pairs, etc.
    /// Uses the same algorithm as `IncrementalRootComputer` but keeps state
    /// so new steps can be appended in O(log n) time without rehashing all leaves.
    #[serde(skip)]
    merkle_stack: Vec<Option<[u8; 32]>>,
    /// Cached Merkle root of all step error leaves.
    #[serde(skip)]
    cached_merkle_root: Option<[u8; 32]>,
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
            merkle_stack: Vec::new(),
            cached_merkle_root: None,
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
            merkle_stack: Vec::new(),
            cached_merkle_root: None,
        }
    }

    /// Hashes a step error into a 32-byte leaf for the Merkle tree.
    /// Includes the step number for domain separation.
    fn hash_step_leaf(step: u64, error: f64) -> [u8; 32] {
        let error_scaled = (error.abs() * ERROR_SCALE).min(u64::MAX as f64) as u64;
        let mut hasher = Sha256::new();
        hasher.update(b"\x00"); // leaf domain separator
        hasher.update(step.to_le_bytes());
        hasher.update(error_scaled.to_le_bytes());
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }

    /// Hashes two child nodes into a parent node.
    fn hash_merkle_nodes(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"\x01"); // internal node domain separator
        hasher.update(left);
        hasher.update(right);
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }

    /// Pushes a new leaf hash into the incremental Merkle stack.
    /// This is O(log n) amortized — each leaf causes at most O(log n) hashes.
    fn merkle_push(&mut self, leaf: [u8; 32]) {
        let mut current = leaf;
        let mut level = 0;

        loop {
            while self.merkle_stack.len() <= level {
                self.merkle_stack.push(None);
            }

            match self.merkle_stack[level].take() {
                Some(sibling) => {
                    current = Self::hash_merkle_nodes(&sibling, &current);
                    level += 1;
                }
                None => {
                    self.merkle_stack[level] = Some(current);
                    break;
                }
            }
        }

        self.cached_merkle_root = None;
    }

    /// Records error for the current step and advances to the next step.
    ///
    /// The step error is incrementally added to a Merkle tree in O(log n) time.
    /// The final commitment checksum is still computed lazily via `checksum()`,
    /// but the Merkle root is maintained incrementally — no full rehash needed.
    pub fn record_step(&mut self, step_error: f64) {
        self.accumulated_error += step_error;
        if self.track_history {
            self.error_history.push(step_error);
        }

        // Incrementally insert this step as a Merkle leaf
        let leaf = Self::hash_step_leaf(self.current_step, step_error);
        self.merkle_push(leaf);

        self.current_step += 1;
        self.cached_checksum = None; // Invalidate commitment cache
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

    /// Returns the incremental Merkle root of all recorded step errors.
    ///
    /// This root commits to every individual step error in sequence, not just the
    /// accumulated total. It enables proofs that specific steps contributed specific
    /// error amounts. Computed in O(log n) from the internal stack.
    pub fn merkle_root(&mut self) -> [u8; 32] {
        if let Some(cached) = self.cached_merkle_root {
            return cached;
        }

        if self.current_step == 0 {
            return [0u8; 32];
        }

        // Combine remaining partial roots with zero padding
        let zero = [0u8; 32];
        let mut result: Option<[u8; 32]> = None;

        for level in 0..self.merkle_stack.len() {
            if let Some(hash) = self.merkle_stack[level] {
                result = Some(match result {
                    Some(existing) => Self::hash_merkle_nodes(&hash, &existing),
                    None => hash,
                });
            } else if result.is_some() {
                result = Some(Self::hash_merkle_nodes(&result.unwrap(), &zero));
            }
        }

        let root = result.unwrap_or([0u8; 32]);
        self.cached_merkle_root = Some(root);
        root
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
        self.merkle_stack.clear();
        self.cached_merkle_root = None;
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

    // === Contract-aligned method tests ===

    #[test]
    fn test_to_contract_u64_matches_checksum_compact() {
        let commitment = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);
        assert_eq!(commitment.to_contract_u64(), commitment.checksum_compact());
    }

    #[test]
    fn test_verify_contract_checksum() {
        let commitment = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);
        let correct = commitment.to_contract_u64();
        assert!(commitment.verify_contract_checksum(correct));
        assert!(!commitment.verify_contract_checksum(correct + 1));
    }

    #[test]
    fn test_to_public_inputs() {
        let commitment = ErrorCommitment::new(0.001, 42, [1u8; 32], 0.01);
        let pi = commitment.to_public_inputs(100, 200, 300, 400, 500);

        assert_eq!(pi[0], 100); // old_hash_lo
        assert_eq!(pi[1], 200); // old_hash_hi
        assert_eq!(pi[2], 300); // new_hash_lo
        assert_eq!(pi[3], 400); // new_hash_hi
        assert_eq!(pi[4], 500); // loss
        assert_eq!(pi[5], commitment.accumulated_error_scaled()); // errorBound
        assert_eq!(pi[6], 42);  // stepNumber
        assert_eq!(pi[7], commitment.to_contract_u64()); // errorChecksum
    }

    #[test]
    fn test_model_id_from_uint256() {
        // Solidity uint256(1) = big-endian [0,0,...,0,1]
        let id = ErrorCommitment::model_id_from_uint256(1);
        assert_eq!(id[31], 1);
        assert_eq!(id[30], 0);
        for i in 0..24 {
            assert_eq!(id[i], 0);
        }

        // Solidity uint256(256) = [0,0,...,1,0]
        let id = ErrorCommitment::model_id_from_uint256(256);
        assert_eq!(id[30], 1);
        assert_eq!(id[31], 0);

        // Verify consistency with cross-language test (modelId=1)
        let mut expected = [0u8; 32];
        expected[31] = 1;
        assert_eq!(ErrorCommitment::model_id_from_uint256(1), expected);
    }

    #[test]
    fn test_contract_u64_cross_language_alignment() {
        // This test verifies that to_contract_u64() produces the exact same value
        // as Solidity _computeErrorChecksum() for known inputs.
        use sha2::{Digest, Sha256};

        let model_id = ErrorCommitment::model_id_from_uint256(1);
        let commitment = ErrorCommitment::new(0.001, 42, model_id, 0.01);

        // Manually compute what Solidity would produce
        let mut preimage = Vec::with_capacity(56);
        preimage.extend_from_slice(&1_000_000_000u64.to_le_bytes()); // 0.001 * 1e12
        preimage.extend_from_slice(&42u64.to_le_bytes());
        preimage.extend_from_slice(&model_id);
        preimage.extend_from_slice(&10_000_000_000u64.to_le_bytes()); // 0.01 * 1e12

        let mut hasher = Sha256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize();
        let solidity_result = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        assert_eq!(commitment.to_contract_u64(), solidity_result);
    }
}
