//! Bounded Addition Gadget.
//!
//! Verifies:
//! 1. value_c = value_a + value_b
//! 2. error_c = error_a + error_b
//! 3. error_c is within [0, MAX_ERROR)

use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::ErrorFront,
};
use halo2curves::ff::PrimeField;

#[derive(Clone, Debug)]
pub struct BoundedAddConfig<F: PrimeField, const RANGE: usize> {
    pub arithmetic: ArithmeticConfig,
    pub range: RangeConfig<F, RANGE>,
}

pub struct BoundedAddChip<F: PrimeField, const RANGE: usize> {
    config: BoundedAddConfig<F, RANGE>,
    pub arithmetic_chip: ArithmeticChip<F>,
    pub range_chip: RangeChip<F, RANGE>,
}

impl<F: PrimeField, const RANGE: usize> BoundedAddChip<F, RANGE> {
    pub fn new(config: BoundedAddConfig<F, RANGE>) -> Self {
        let arithmetic_chip = ArithmeticChip::new(config.arithmetic.clone());
        let range_chip = RangeChip::new(config.range.clone());
        Self {
            config,
            arithmetic_chip,
            range_chip,
        }
    }

    /// Assigns a bounded addition operation.
    ///
    /// All constraints are in a single region to ensure copy constraints
    /// bind shared values across all gates. Previously, 3 separate regions
    /// assigned err_c independently, allowing a malicious prover to use
    /// different values for err_c in the error addition vs range check.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        val_a: Value<F>,
        err_a: Value<F>,
        val_b: Value<F>,
        err_b: Value<F>,
        val_c: Value<F>,
        err_c: Value<F>,
    ) -> Result<(), ErrorFront> {
        layouter.assign_region(
            || "bounded_add_all",
            |mut region| {
                // Row 0: val_a + val_b = val_c
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "val_a", self.config.arithmetic.a, 0, || val_a)?;
                region.assign_advice(|| "val_b", self.config.arithmetic.b, 0, || val_b)?;
                region.assign_advice(|| "val_c", self.config.arithmetic.c, 0, || val_c)?;

                // Row 1: err_a + err_b = err_c
                self.config.arithmetic.s_add.enable(&mut region, 1)?;
                region.assign_advice(|| "err_a", self.config.arithmetic.a, 1, || err_a)?;
                region.assign_advice(|| "err_b", self.config.arithmetic.b, 1, || err_b)?;
                let err_c_1 = region.assign_advice(|| "err_c", self.config.arithmetic.c, 1, || err_c)?;

                // Row 2: range check err_c
                self.config.range.s_range.enable(&mut region, 2)?;
                let err_c_2 = region.assign_advice(
                    || "err_c_range",
                    self.config.range.input_column,
                    2,
                    || err_c,
                )?;

                // Copy constraint: err_c at row 1 (arithmetic output) must equal
                // err_c at row 2 (range check input). This prevents a malicious
                // prover from using different values.
                region.constrain_equal(err_c_1.cell(), err_c_2.cell())?;

                Ok(())
            },
        )?;

        Ok(())
    }
}
