//! Commitment to gradient update vector.
//!
//! Uses SHA-256 hashing for gradient commitments (same as model_commit).
//! The Poseidon circuit gadget has been removed due to halo2_gadgets
//! incompatibility with the PSE fork of halo2.

use crate::commitment::model_commit::{ModelCommitChip, ModelCommitConfig};

// Re-use ModelCommitChip logic since SHA-256 hashing is the same for both.
// Gradient commitment is just the hash of the gradient vector components.
// For large vectors, we merkleize them to avoid huge public inputs.

pub type GradientCommitChip<F> = ModelCommitChip<F>;

pub type GradientCommitConfig<F> = ModelCommitConfig<F>;
