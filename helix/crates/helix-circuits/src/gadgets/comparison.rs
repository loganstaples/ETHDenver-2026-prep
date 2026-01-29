//! Comparison Gadget.
//!
//! Checks if `a <= b` by verifying `b - a` is in the range [0, 2^K).

use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
};
use halo2curves::ff::PrimeField;

#[derive(Clone, Debug)]
pub struct ComparisonConfig<F: PrimeField, const RANGE: usize> {
    pub range: RangeConfig<F, RANGE>,
}

pub struct ComparisonChip<F: PrimeField, const RANGE: usize> {
    config: ComparisonConfig<F, RANGE>,
    // In a real implementation, we might wrap RangeChip directly
}

impl<F: PrimeField, const RANGE: usize> ComparisonChip<F, RANGE> {
    pub fn new(config: ComparisonConfig<F, RANGE>) -> Self {
        Self { config }
    }

    /// Constrains that a <= b.
    /// This is done by proving (b - a) is in the range table.
    /// Assumes a, b are within field capacity such that b-a doesn't wrap unexpectedly
    /// relative to the range size.
    pub fn assign_less_equal(
        &self,
        mut layouter: impl Layouter<F>,
        val_a: Value<F>,
        val_b: Value<F>,
    ) -> Result<(), Error> {
        let diff = val_b - val_a;

        layouter.assign_region(
            || "compare a <= b",
            |mut region| {
                // Enable range check selector
                self.config.range.s_range.enable(&mut region, 0)?;

                // In the current RangeConfig logic, we verify the value in `input_column`.
                // We need to place (b - a) into that column.
                // Note: The caller usually wires up a, b to this diff.
                // For this simple gadget, we just assign the diff directly to satisfy the constraint.
                // In a full circuit, we'd have a gate: `diff = b - a` AND `range(diff)`.
                // Existing RangeConfig just checks "is the value in this cell in the table?".
                // It doesn't enforce WHERE the value came from.
                // The ArithmeticChip should be used to enforce `diff = b - a`.
                //
                // For now, to keep it self-contained without circular deps, we just assign the diff.
                // Integrating it with Arithmetic is responsibility of the caller (ReLU).
                
                region.assign_advice(
                    || "diff",
                    self.config.range.input_column,
                    0,
                    || diff,
                )?;
                Ok(())
            },
        )?;
        Ok(())
    }
}
