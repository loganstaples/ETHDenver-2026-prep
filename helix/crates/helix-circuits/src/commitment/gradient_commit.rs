//! Commitment to gradient update vector.

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use halo2curves::ff::PrimeField;
use crate::commitment::model_commit::{ModelCommitChip, ModelCommitConfig};

// Re-use ModelCommitChip logic since generic hashing is similar.
// Gradient commitment might just be the hash of the vector components.
// For large vectors, we typically merkleize them too to avoid huge public inputs.
// So GradientCommitChip is effectively a Merkle Tree chip too.

pub type GradientCommitChip<F, S, const WIDTH: usize, const RATE: usize, const L: usize> = 
    ModelCommitChip<F, S, WIDTH, RATE, L>;

pub type GradientCommitConfig<F, const WIDTH: usize, const RATE: usize> = 
    ModelCommitConfig<F, WIDTH, RATE>;
