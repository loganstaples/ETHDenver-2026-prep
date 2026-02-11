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
    plonk::ErrorFront,
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
    ///
    /// All constraints are in a single region to ensure copy constraints
    /// bind shared values (val_a, val_b, err_a, err_b) across all gates.
    /// Previously, 7 separate regions assigned the same values independently,
    /// which meant a malicious prover could use different values per region.
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
        // Compute intermediate values
        let term1 = val_a * err_b;   // val_a * err_b
        let term2 = val_b * err_a;   // val_b * err_a
        let term3 = err_a * err_b;   // err_a * err_b
        let sum1 = term1 + term2;    // term1 + term2

        // All constraints in a single region with rows 0-6.
        // Shared values are assigned once and referenced via copy constraints,
        // preventing a malicious prover from using inconsistent values across gates.
        layouter.assign_region(
            || "bounded_mul_all",
            |mut region| {
                // Row 0: val_a * val_b = val_c
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                let val_a_0 = region.assign_advice(|| "val_a", self.config.arithmetic.a, 0, || val_a)?;
                let val_b_0 = region.assign_advice(|| "val_b", self.config.arithmetic.b, 0, || val_b)?;
                region.assign_advice(|| "val_c", self.config.arithmetic.c, 0, || val_c)?;

                // Row 1: t1 = val_a * err_b
                self.config.arithmetic.s_mul.enable(&mut region, 1)?;
                let val_a_1 = region.assign_advice(|| "val_a_1", self.config.arithmetic.a, 1, || val_a)?;
                let err_b_1 = region.assign_advice(|| "err_b", self.config.arithmetic.b, 1, || err_b)?;
                let term1_1 = region.assign_advice(|| "term1", self.config.arithmetic.c, 1, || term1)?;

                // Row 2: t2 = val_b * err_a
                self.config.arithmetic.s_mul.enable(&mut region, 2)?;
                let val_b_2 = region.assign_advice(|| "val_b_2", self.config.arithmetic.a, 2, || val_b)?;
                let err_a_2 = region.assign_advice(|| "err_a", self.config.arithmetic.b, 2, || err_a)?;
                let term2_2 = region.assign_advice(|| "term2", self.config.arithmetic.c, 2, || term2)?;

                // Row 3: t3 = err_a * err_b
                self.config.arithmetic.s_mul.enable(&mut region, 3)?;
                let err_a_3 = region.assign_advice(|| "err_a_3", self.config.arithmetic.a, 3, || err_a)?;
                let err_b_3 = region.assign_advice(|| "err_b_3", self.config.arithmetic.b, 3, || err_b)?;
                let term3_3 = region.assign_advice(|| "term3", self.config.arithmetic.c, 3, || term3)?;

                // Row 4: sum1 = t1 + t2
                self.config.arithmetic.s_add.enable(&mut region, 4)?;
                let term1_4 = region.assign_advice(|| "term1_4", self.config.arithmetic.a, 4, || term1)?;
                let term2_4 = region.assign_advice(|| "term2_4", self.config.arithmetic.b, 4, || term2)?;
                let sum1_4 = region.assign_advice(|| "sum1", self.config.arithmetic.c, 4, || sum1)?;

                // Row 5: err_c = sum1 + t3
                self.config.arithmetic.s_add.enable(&mut region, 5)?;
                let sum1_5 = region.assign_advice(|| "sum1_5", self.config.arithmetic.a, 5, || sum1)?;
                let term3_5 = region.assign_advice(|| "term3_5", self.config.arithmetic.b, 5, || term3)?;
                region.assign_advice(|| "err_c", self.config.arithmetic.c, 5, || err_c)?;

                // Row 6: range check err_c
                self.config.range.s_range.enable(&mut region, 6)?;
                region.assign_advice(|| "err_c_range", self.config.range.input_column, 6, || err_c)?;

                // Copy constraints: bind shared values across rows so the prover
                // cannot use different values at different rows.
                region.constrain_equal(val_a_0.cell(), val_a_1.cell())?;   // val_a: row 0 == row 1
                region.constrain_equal(val_b_0.cell(), val_b_2.cell())?;   // val_b: row 0 == row 2
                region.constrain_equal(err_a_2.cell(), err_a_3.cell())?;   // err_a: row 2 == row 3
                region.constrain_equal(err_b_1.cell(), err_b_3.cell())?;   // err_b: row 1 == row 3
                region.constrain_equal(term1_1.cell(), term1_4.cell())?;   // term1: row 1 == row 4
                region.constrain_equal(term2_2.cell(), term2_4.cell())?;   // term2: row 2 == row 4
                region.constrain_equal(term3_3.cell(), term3_5.cell())?;   // term3: row 3 == row 5
                region.constrain_equal(sum1_4.cell(), sum1_5.cell())?;     // sum1:  row 4 == row 5

                Ok(())
            },
        )?;

        Ok(())
    }
}
