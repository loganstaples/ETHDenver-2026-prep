//! Bounded Multiplication Gadget.
//!
//! Verifies:
//! 1. value_c = value_a * value_b
//! 2. error_c = |a|*err_b + |b|*err_a + err_a*err_b
//!    (Assuming positive inputs for this stage: err_c = a*err_b + b*err_a + err_a*err_b)
//! 3. error_c is within [0, MAX_ERROR)

use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Error, ErrorFront},
};
use halo2curves::ff::PrimeField;

#[derive(Clone, Debug)]
pub struct BoundedMulConfig<F: PrimeField, const RANGE: usize> {
    pub arithmetic: ArithmeticConfig,
    pub range: RangeConfig<F, RANGE>,
}

pub struct BoundedMulChip<F: PrimeField, const RANGE: usize> {
    config: BoundedMulConfig<F, RANGE>,
    pub arithmetic_chip: ArithmeticChip<F>,
    pub range_chip: RangeChip<F, RANGE>,
}

impl<F: PrimeField, const RANGE: usize> BoundedMulChip<F, RANGE> {
    pub fn new(config: BoundedMulConfig<F, RANGE>) -> Self {
        let arithmetic_chip = ArithmeticChip::new(config.arithmetic.clone());
        let range_chip = RangeChip::new(config.range.clone());
        Self {
            config,
            arithmetic_chip,
            range_chip,
        }
    }

    /// Assigns a bounded multiplication operation.
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
        // 1. Constrain value multiplication: val_a * val_b = val_c
        layouter.assign_region(
            || "bounded mul values",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "val_a", self.config.arithmetic.a, 0, || val_a)?;
                region.assign_advice(|| "val_b", self.config.arithmetic.b, 0, || val_b)?;
                region.assign_advice(|| "val_c", self.config.arithmetic.c, 0, || val_c)?;
                Ok(())
            },
        )?;

        // 2. Constrain error propagation
        // err_c = (val_a * err_b) + (val_b * err_a) + (err_a * err_b)
        // We need intermediate values.
        let term1 = val_a * err_b;
        let term2 = val_b * err_a;
        let term3 = err_a * err_b;
        
        // We can do this in steps using the arithmetic chip (Add/Mul gates).
        // Step 2a: t1 = val_a * err_b
        layouter.assign_region(
            || "error term 1",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "val_a", self.config.arithmetic.a, 0, || val_a)?;
                region.assign_advice(|| "err_b", self.config.arithmetic.b, 0, || err_b)?;
                region.assign_advice(|| "term1", self.config.arithmetic.c, 0, || term1)?;
                Ok(())
            },
        )?;

        // Step 2b: t2 = val_b * err_a
        layouter.assign_region(
            || "error term 2",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "val_b", self.config.arithmetic.a, 0, || val_b)?;
                region.assign_advice(|| "err_a", self.config.arithmetic.b, 0, || err_a)?;
                region.assign_advice(|| "term2", self.config.arithmetic.c, 0, || term2)?;
                Ok(())
            },
        )?;

        // Step 2c: t3 = err_a * err_b
        layouter.assign_region(
            || "error term 3",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "err_a", self.config.arithmetic.a, 0, || err_a)?;
                region.assign_advice(|| "err_b", self.config.arithmetic.b, 0, || err_b)?;
                region.assign_advice(|| "term3", self.config.arithmetic.c, 0, || term3)?;
                Ok(())
            },
        )?;

        // Step 2d: Accumulate sums.
        // sum1 = t1 + t2
        let sum1 = term1 + term2;
        layouter.assign_region(
            || "sum1",
            |mut region| {
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "term1", self.config.arithmetic.a, 0, || term1)?;
                region.assign_advice(|| "term2", self.config.arithmetic.b, 0, || term2)?;
                region.assign_advice(|| "sum1", self.config.arithmetic.c, 0, || sum1)?;
                Ok(())
            },
        )?;

        // err_c = sum1 + t3
        // Note: We should verify this `err_c` matches user input `err_c`.
        // The arithmetic gate ensures `input_a + input_b = output`.
        // So we assign `sum1`, `t3`, and `err_c`.
        layouter.assign_region(
            || "calc err_c",
            |mut region| {
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "sum1", self.config.arithmetic.a, 0, || sum1)?;
                region.assign_advice(|| "term3", self.config.arithmetic.b, 0, || term3)?;
                region.assign_advice(|| "err_c", self.config.arithmetic.c, 0, || err_c)?;
                Ok(())
            },
        )?;

        // 3. Range check err_c
        layouter.assign_region(
            || "range check err_c",
            |mut region| {
                self.config.range.s_range.enable(&mut region, 0)?;
                region.assign_advice(
                    || "err_c copy",
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
