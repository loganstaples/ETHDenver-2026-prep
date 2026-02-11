//! Merkle commitment to model weights using SHA-256 hash.
//!
//! Uses native SHA-256 hashing for model weight commitments.
//! The Poseidon circuit gadget (from halo2_gadgets) has been removed because
//! it is incompatible with the PSE fork of halo2. For in-circuit hashing,
//! see `compute_state_hash_v2` in `ml/training_step_v2.rs`.

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, ErrorFront, Instance},
};
use halo2curves::ff::PrimeField;
use sha2::{Sha256, Digest};
use std::marker::PhantomData;

/// Configuration for the ModelCommit chip.
#[derive(Clone, Debug)]
pub struct ModelCommitConfig<F: PrimeField> {
    pub advice: Column<Advice>,
    pub instance: Column<Instance>,
    _marker: PhantomData<F>,
}

/// Chip for proving membership in a Merkle tree of model weights.
/// Uses SHA-256 for native commitment computation.
///
/// Note: In-circuit Poseidon hashing has been removed due to halo2_gadgets
/// incompatibility with the PSE fork. For production use, integrate PSE's
/// standalone poseidon crate or use the SHA-256 state hash from training_step_v2.
pub struct ModelCommitChip<F: PrimeField> {
    _config: ModelCommitConfig<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> ModelCommitChip<F> {
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        instance: Column<Instance>,
    ) -> ModelCommitConfig<F> {
        let advice = meta.advice_column();
        meta.enable_equality(advice);
        meta.enable_equality(instance);

        ModelCommitConfig {
            advice,
            instance,
            _marker: PhantomData,
        }
    }

    pub fn new(config: ModelCommitConfig<F>) -> Self {
        Self {
            _config: config,
            _marker: PhantomData,
        }
    }

    /// Verifies that a leaf is in the Merkle tree at the given path.
    ///
    /// # Panics
    ///
    /// Always panics. In-circuit Merkle verification is not implemented.
    /// Native SHA-256 Merkle verification is done outside the circuit via
    /// `compute_merkle_root`. In-circuit verification requires integrating
    /// PSE's standalone Poseidon crate for in-circuit hashing.
    pub fn verify_path(
        &self,
        _layouter: impl Layouter<F>,
        _leaf: Value<F>,
        _path_elements: Vec<Value<F>>,
        _path_indices: Vec<Value<bool>>,
    ) -> Result<(), ErrorFront> {
        unimplemented!(
            "In-circuit Merkle path verification is not implemented. \
             Use native `compute_merkle_root` for out-of-circuit verification, \
             or integrate PSE's standalone Poseidon crate for in-circuit hashing."
        )
    }
}

/// Computes a SHA-256 commitment of model weights (native, not in-circuit).
///
/// This is the production-ready native hash used for weight commitments.
/// Returns (lo, hi) as two 128-bit halves of the 256-bit hash.
pub fn compute_model_commitment(weights: &[u8]) -> ([u8; 16], [u8; 16]) {
    let hash = Sha256::digest(weights);
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    lo.copy_from_slice(&hash[..16]);
    hi.copy_from_slice(&hash[16..]);
    (lo, hi)
}

/// Computes a SHA-256 Merkle root from leaf hashes.
pub fn compute_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current_level: Vec<[u8; 32]> = leaves.to_vec();
    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        for pair in current_level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            if pair.len() > 1 {
                hasher.update(&pair[1]);
            } else {
                hasher.update(&pair[0]); // duplicate last for odd count
            }
            let hash: [u8; 32] = hasher.finalize().into();
            next_level.push(hash);
        }
        current_level = next_level;
    }
    current_level[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_commitment() {
        let weights = vec![1u8, 2, 3, 4, 5];
        let (lo, hi) = compute_model_commitment(&weights);
        assert_ne!(lo, [0u8; 16]);
        assert_ne!(hi, [0u8; 16]);
    }

    #[test]
    fn test_merkle_root_single_leaf() {
        let leaf = [42u8; 32];
        let root = compute_merkle_root(&[leaf]);
        assert_eq!(root, leaf);
    }

    #[test]
    fn test_merkle_root_two_leaves() {
        let leaf1 = [1u8; 32];
        let leaf2 = [2u8; 32];
        let root = compute_merkle_root(&[leaf1, leaf2]);
        assert_ne!(root, leaf1);
        assert_ne!(root, leaf2);
    }
}
