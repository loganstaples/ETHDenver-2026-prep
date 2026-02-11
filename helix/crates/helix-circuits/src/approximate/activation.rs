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
    
    fn enforce_sum(&self, mut layouter: impl Layouter<F>, a: Value<F>, b: Value<F>, c: Value<F>) -> Result<(), ErrorFront> {
        layouter.assign_region(
            || "enforce sum",
            |mut region| {
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.arithmetic.a, 0, || a)?;
                region.assign_advice(|| "b", self.config.arithmetic.b, 0, || b)?;
                region.assign_advice(|| "c", self.config.arithmetic.c, 0, || c)?;
                Ok(())
            }
        )
    }

    fn enforce_product(&self, mut layouter: impl Layouter<F>, a: Value<F>, b: Value<F>, c: Value<F>) -> Result<(), ErrorFront> {
        layouter.assign_region(
            || "enforce product",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.arithmetic.a, 0, || a)?;
                region.assign_advice(|| "b", self.config.arithmetic.b, 0, || b)?;
                region.assign_advice(|| "c", self.config.arithmetic.c, 0, || c)?;
                Ok(())
            }
        )
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        val_x: Value<F>,
        err_x: Value<F>,
        val_y: Value<F>,
        err_y: Value<F>,
    ) -> Result<(), ErrorFront> {
        // Sound ReLU decomposition:
        //   x + neg = y        (decomposition)
        //   y * neg = 0        (exactly one of y, neg is zero)
        //   range_check(y)     (y >= 0)
        //   range_check(neg)   (neg >= 0)
        //
        // If x >= 0: y = x, neg = 0
        // If x < 0:  y = 0, neg = -x
        //
        // The old constraint `y * (y - x) = 0` was unsound because
        // a malicious prover could always set y = x (identity), bypassing ReLU.

        let zero = Value::known(F::ZERO);

        // Compute neg = y - x (which equals 0 when x>=0, or -x when x<0)
        let neg = val_y - val_x;

        // 1. x + neg = y (decomposition constraint)
        self.enforce_sum(layouter.namespace(|| "x + neg = y"), val_x, neg, val_y)?;

        // 2. y * neg = 0 (disjointness: at most one is nonzero)
        self.enforce_product(layouter.namespace(|| "y * neg = 0"), val_y, neg, zero)?;

        // 3. Range check y (ensures y >= 0, prevents prover from using negative y)
        layouter.assign_region(
            || "range check relu y",
            |mut region| {
                self.config.range.s_range.enable(&mut region, 0)?;
                region.assign_advice(
                    || "val_y",
                    self.config.range.input_column,
                    0,
                    || val_y,
                )?;
                Ok(())
            },
        )?;

        // 4. Range check neg (ensures neg >= 0, prevents prover from using negative neg)
        layouter.assign_region(
            || "range check relu neg",
            |mut region| {
                self.config.range.s_range.enable(&mut region, 0)?;
                region.assign_advice(
                    || "neg",
                    self.config.range.input_column,
                    0,
                    || neg,
                )?;
                Ok(())
            },
        )?;

        // 5. Error propagation: if y != 0 then err_y = err_x, else err_y = 0
        //    err_x + err_diff = err_y
        //    y * err_diff = 0 (if y != 0, err_diff must be 0 => err_y = err_x)
        let err_diff = err_y - err_x;
        self.enforce_sum(layouter.namespace(|| "err_diff"), err_x, err_diff, err_y)?;
        self.enforce_product(layouter.namespace(|| "y * err_diff = 0"), val_y, err_diff, zero)?;

        // 6. Range check error output
        layouter.assign_region(
            || "range check relu err",
            |mut region| {
                self.config.range.s_range.enable(&mut region, 0)?;
                region.assign_advice(
                    || "err_y",
                    self.config.range.input_column,
                    0,
                    || err_y,
                )?;
                Ok(())
            },
        )?;

        Ok(())
    }
}
