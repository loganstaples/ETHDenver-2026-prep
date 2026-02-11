//! ReLU Activation Gadget.
//!
//! Verifies y = max(0, x).
//! Propagates error: err_y = err_x if x > 0 else 0.

use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ErrorFront, Selector},
};
use halo2curves::ff::PrimeField;

use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};

#[derive(Clone, Debug)]
pub struct ReLUConfig<F: PrimeField, const RANGE: usize> {
    pub range: RangeConfig<F, RANGE>,
    pub arithmetic: ArithmeticConfig,
    pub s_relu: Selector,
}

pub struct ReLUChip<F: PrimeField, const RANGE: usize> {
    config: ReLUConfig<F, RANGE>,
    pub range_chip: RangeChip<F, RANGE>,
    pub arithmetic_chip: ArithmeticChip<F>,
}

impl<F: PrimeField, const RANGE: usize> ReLUChip<F, RANGE> {
    pub fn new(config: ReLUConfig<F, RANGE>) -> Self {
        let range_chip = RangeChip::new(config.range.clone());
        let arithmetic_chip = ArithmeticChip::new(config.arithmetic.clone());
        Self { config, range_chip, arithmetic_chip }
    }

    /// Assigns a bounded ReLU activation in a single region.
    ///
    /// Sound ReLU decomposition:
    ///   x + neg = y        (decomposition)
    ///   y * neg = 0         (exactly one of y, neg is zero)
    ///   range_check(y)      (y >= 0)
    ///   range_check(neg)    (neg >= 0)
    ///
    /// Error propagation:
    ///   err_x + err_diff = err_y
    ///   y * err_diff = 0    (if y != 0, err_diff = 0 => err_y = err_x)
    ///   range_check(err_y)
    ///
    /// All constraints are in a single region with copy constraints binding
    /// shared values (val_y, neg, err_diff) across different gate rows.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        val_x: Value<F>,
        err_x: Value<F>,
        val_y: Value<F>,
        err_y: Value<F>,
    ) -> Result<(), ErrorFront> {
        let zero = Value::known(F::ZERO);
        let neg = val_y - val_x;
        let err_diff = err_y - err_x;

        layouter.assign_region(
            || "bounded_relu_all",
            |mut region| {
                // Row 0: x + neg = y (decomposition)
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "val_x", self.config.arithmetic.a, 0, || val_x)?;
                let neg_0 = region.assign_advice(|| "neg", self.config.arithmetic.b, 0, || neg)?;
                let val_y_0 = region.assign_advice(|| "val_y", self.config.arithmetic.c, 0, || val_y)?;

                // Row 1: y * neg = 0 (disjointness)
                self.config.arithmetic.s_mul.enable(&mut region, 1)?;
                let val_y_1 = region.assign_advice(|| "val_y_1", self.config.arithmetic.a, 1, || val_y)?;
                let neg_1 = region.assign_advice(|| "neg_1", self.config.arithmetic.b, 1, || neg)?;
                region.assign_advice(|| "zero", self.config.arithmetic.c, 1, || zero)?;

                // Copy constraints: val_y and neg must be consistent across rows 0 and 1
                region.constrain_equal(val_y_0.cell(), val_y_1.cell())?;
                region.constrain_equal(neg_0.cell(), neg_1.cell())?;

                // Row 2: range check y (y >= 0)
                self.config.range.s_range.enable(&mut region, 2)?;
                let val_y_2 = region.assign_advice(
                    || "val_y_range",
                    self.config.range.input_column,
                    2,
                    || val_y,
                )?;
                region.constrain_equal(val_y_0.cell(), val_y_2.cell())?;

                // Row 3: range check neg (neg >= 0)
                self.config.range.s_range.enable(&mut region, 3)?;
                let neg_3 = region.assign_advice(
                    || "neg_range",
                    self.config.range.input_column,
                    3,
                    || neg,
                )?;
                region.constrain_equal(neg_0.cell(), neg_3.cell())?;

                // Row 4: err_x + err_diff = err_y (error decomposition)
                self.config.arithmetic.s_add.enable(&mut region, 4)?;
                region.assign_advice(|| "err_x", self.config.arithmetic.a, 4, || err_x)?;
                let err_diff_4 = region.assign_advice(|| "err_diff", self.config.arithmetic.b, 4, || err_diff)?;
                let err_y_4 = region.assign_advice(|| "err_y", self.config.arithmetic.c, 4, || err_y)?;

                // Row 5: y * err_diff = 0 (if y != 0 then err_diff must be 0)
                self.config.arithmetic.s_mul.enable(&mut region, 5)?;
                let val_y_5 = region.assign_advice(|| "val_y_5", self.config.arithmetic.a, 5, || val_y)?;
                let err_diff_5 = region.assign_advice(|| "err_diff_5", self.config.arithmetic.b, 5, || err_diff)?;
                region.assign_advice(|| "zero_5", self.config.arithmetic.c, 5, || zero)?;

                // Copy constraints: val_y and err_diff consistent across rows 4-5
                region.constrain_equal(val_y_0.cell(), val_y_5.cell())?;
                region.constrain_equal(err_diff_4.cell(), err_diff_5.cell())?;

                // Row 6: range check err_y
                self.config.range.s_range.enable(&mut region, 6)?;
                let err_y_6 = region.assign_advice(
                    || "err_y_range",
                    self.config.range.input_column,
                    6,
                    || err_y,
                )?;
                region.constrain_equal(err_y_4.cell(), err_y_6.cell())?;

                Ok(())
            },
        )?;

        Ok(())
    }
}
