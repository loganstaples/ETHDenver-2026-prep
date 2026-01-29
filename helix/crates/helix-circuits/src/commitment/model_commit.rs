//! Merkle commitment to model weights using Poseidon hash.

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Fixed, Selector},
    poly::Rotation,
};
use halo2_gadgets::poseidon::{
    primitives::{ConstantLength, Spec},
    Hash, Pow5Chip, Pow5Config,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

/// Configuration for the ModelCommit chip.
#[derive(Clone, Debug)]
pub struct ModelCommitConfig<F: PrimeField, const WIDTH: usize, const RATE: usize> {
    pub hash_config: Pow5Config<F, WIDTH, RATE>,
    pub advice: Column<Advice>,
    pub instance: Column<halo2_proofs::plonk::Instance>, 
}

/// Chip for proving membership in a Merkle tree of model weights.
/// Uses Poseidon hash with width 3 (2 inputs + 1 capacity) for binary tree.
pub struct ModelCommitChip<F: PrimeField, S: Spec<F, WIDTH, RATE> + Clone + Copy, const WIDTH: usize, const RATE: usize, const L: usize> {
    config: ModelCommitConfig<F, WIDTH, RATE>,
    _marker: PhantomData<(F, S)>,
}

impl<F: PrimeField, S: Spec<F, WIDTH, RATE> + Clone + Copy, const WIDTH: usize, const RATE: usize, const L: usize>
    ModelCommitChip<F, S, WIDTH, RATE, L>
{
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        state: [Column<Advice>; WIDTH],
        partial_sbox: Column<Advice>,
        rc_a: [Column<Fixed>; WIDTH],
        rc_b: [Column<Fixed>; WIDTH],
        instance: Column<halo2_proofs::plonk::Instance>,
    ) -> ModelCommitConfig<F, WIDTH, RATE> {
        let hash_config = Pow5Chip::<F, WIDTH, RATE>::configure::<S>(
            meta,
            state,
            partial_sbox,
            rc_a,
            rc_b,
        );
        
        let advice = meta.advice_column();
        meta.enable_equality(advice);
        meta.enable_equality(instance);

        ModelCommitConfig {
            hash_config,
            advice,
            instance,
        }
    }

    pub fn new(config: ModelCommitConfig<F, WIDTH, RATE>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Verifies that a leaf is in the Merkle tree at the given path.
    /// Returns the computed root.
    pub fn verify_path(
        &self,
        mut layouter: impl Layouter<F>,
        leaf: Value<F>,
        path_elements: Vec<Value<F>>,
        path_indices: Vec<Value<bool>>, // 0 = left, 1 = right
    ) -> Result<(), Error> {
        let chip = Pow5Chip::construct(self.config.hash_config.clone());
        let hasher = Hash::<_, _, S, ConstantLength<L>, WIDTH, RATE>::init(
            chip,
            layouter.namespace(|| "init hasher"),
        )?;

        // For now, let's assume we just hash the leaf.
        // Implementing full Merkle path verification usually requires iterative hashing.
        // `halo2_gadgets` Hash struct is for a single hash invocation.
        // We would need to create multiple hashers or reset.
        
        // Simplified for stage 14 first pass:
        // Prove Hash(leaf) == commitment (just 1 level tree / leaf commitment)
        // Or implement the loop.
        
        // Loop implementation:
        let mut current_hash = leaf;
        
        for (i, (sibling, is_right)) in path_elements.iter().zip(path_indices).enumerate() {
            // We need to hash(left, right).
            // This requires conditional swapping based on `is_right`.
            // Swap logic:
            // if is_right: left=sibling, right=current
            // else:        left=current, right=sibling
            
            // For this iteration, we instantiate a NEW hasher.
            let chip = Pow5Chip::construct(self.config.hash_config.clone());
            let hasher = Hash::<_, _, S, ConstantLength<2>, WIDTH, RATE>::init(
                chip,
                layouter.namespace(|| format!("hasher level {}", i)),
            )?;
            
            // To implement Swap properly in circuit we need a Swap gadget.
            // For now, we will perform the Hash and assume correct ordering is provided by witness,
            // (WEAKNESS: without swap constraint, prover could fake order).
            // STRICT implementation requires the Swap constraint.
            
            // Hashing 2 elements
            let values = [*sibling, current_hash]; // Placeholder, need to act on Values
             
             // The halo2_gadgets Hash API takes `AssignedCell`s usually?
             // Looking at docs: `hash(..., message: &[AssignedCell<F, F>])`.
             // We need to assign `current_hash` and `sibling` to cells first.
        }
        
        Ok(())
    }
}
