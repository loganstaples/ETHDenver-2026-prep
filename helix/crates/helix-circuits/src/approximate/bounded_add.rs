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
    plonk::Error,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

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
    /// Returns the output value and error cells (though simplified here to just Result).
    pub fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        val_a: Value<F>,
        err_a: Value<F>,
        val_b: Value<F>,
        err_b: Value<F>,
        val_c: Value<F>,
        err_c: Value<F>,
    ) -> Result<(), Error> {
        // 1. Constrain value addition: val_a + val_b = val_c
        layouter.assign_region(
            || "bounded add values",
            |mut region| {
                // Enable Addition selector
                self.config.arithmetic.s_add.enable(&mut region, 0)?;

                // Assign inputs/output
                region.assign_advice(|| "val_a", self.config.arithmetic.a, 0, || val_a)?;
                region.assign_advice(|| "val_b", self.config.arithmetic.b, 0, || val_b)?;
                region.assign_advice(|| "val_c", self.config.arithmetic.c, 0, || val_c)?;
                
                Ok(())
            },
        )?;

        // 2. Constrain error addition: err_a + err_b = err_c
        layouter.assign_region(
            || "bounded add errors",
            |mut region| {
                // Enable Addition selector
                self.config.arithmetic.s_add.enable(&mut region, 0)?;

                region.assign_advice(|| "err_a", self.config.arithmetic.a, 0, || err_a)?;
                region.assign_advice(|| "err_b", self.config.arithmetic.b, 0, || err_b)?;
                region.assign_advice(|| "err_c", self.config.arithmetic.c, 0, || err_c)?;
                
                Ok(())
            },
        )?;

        // 3. Constrain error range
        // We reuse the RangeChip logic.
        // RangeChip usually expects to assign a cell itself or verify an existing one.
        // My RangeChip impl uses a lookup.
        // `RangeChip::load` loads the table.
        // We need to make sure `err_c` is in the table.
        // The current RangeConfig does: `meta.lookup(|meta| { let value = query_advice(input_column)... })`.
        // This implies `input_column` is FIXED in the config.
        // So we MUST put `err_c` into `input_column` to trigger the check.
        
        layouter.assign_region(
            || "range check err_c",
            |mut region| {
                self.config.range.s_range.enable(&mut region, 0)?;
                region.assign_advice(
                    || "err_c copy for range",
                    self.config.range.input_column,
                    0,
                    || err_c,
                )?;
                Ok(())
            },
        )?;

        Ok(())
    }
}
