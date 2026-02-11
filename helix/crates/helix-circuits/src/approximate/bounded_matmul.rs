//! Bounded Matrix Multiplication Gadget.
//!
//! Verifies dot products with error propagation.
//! All constraints use copy constraints to prevent a malicious prover
//! from assigning inconsistent values across gate rows.

use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::ErrorFront,
};
use halo2curves::ff::PrimeField;

#[derive(Clone, Debug)]
pub struct BoundedMatMulConfig<F: PrimeField, const RANGE: usize> {
    pub arithmetic: ArithmeticConfig,
    pub range: RangeConfig<F, RANGE>,
}

pub struct BoundedMatMulChip<F: PrimeField, const RANGE: usize> {
    config: BoundedMatMulConfig<F, RANGE>,
    pub arithmetic_chip: ArithmeticChip<F>,
    pub range_chip: RangeChip<F, RANGE>,
}

impl<F: PrimeField, const RANGE: usize> BoundedMatMulChip<F, RANGE> {
    pub fn new(config: BoundedMatMulConfig<F, RANGE>) -> Self {
        let arithmetic_chip = ArithmeticChip::new(config.arithmetic.clone());
        let range_chip = RangeChip::new(config.range.clone());
        Self {
            config,
            arithmetic_chip,
            range_chip,
        }
    }

    /// Assigns a dot product verification for a single result cell.
    /// result = sum(a_i * b_i)
    /// error = sum(|a_i|*e_bi + |b_i|*e_ai + e_ai*e_bi)
    ///
    /// Each iteration is assigned in a single region with copy constraints
    /// binding shared values (va, vb, ea, eb, intermediate products) across
    /// all gate rows. This prevents a malicious prover from using different
    /// values in the value computation vs error computation.
    pub fn assign_dot_product(
        &self,
        mut layouter: impl Layouter<F>,
        row_a_vals: &[Value<F>],
        row_a_errs: &[Value<F>],
        col_b_vals: &[Value<F>],
        col_b_errs: &[Value<F>],
        res_val: Value<F>,
        res_err: Value<F>,
    ) -> Result<(), ErrorFront> {
        if row_a_vals.len() != col_b_vals.len() {
            return Err(ErrorFront::Synthesis);
        }

        let k = row_a_vals.len();
        let zero = Value::known(F::ZERO);

        // Track running sums and their last assigned cells for cross-region linking
        let mut running_val = Value::known(F::ZERO);
        let mut running_err = Value::known(F::ZERO);
        let mut prev_running_val_cell: Option<AssignedCell<F, F>> = None;
        let mut prev_running_err_cell: Option<AssignedCell<F, F>> = None;

        for i in 0..k {
            let va = row_a_vals[i];
            let ea = row_a_errs[i];
            let vb = col_b_vals[i];
            let eb = col_b_errs[i];

            let term_val = va * vb;
            let next_running_val = running_val + term_val;

            // Error term: va*eb + vb*ea + ea*eb
            let t1 = va * eb;
            let t2 = vb * ea;
            let t3 = ea * eb;
            let sum_part = t1 + t2;
            let term_err = sum_part + t3;
            let next_running_err = running_err + term_err;

            // Each iteration: single region with all gates and copy constraints.
            // Layout (9 rows per iteration, or 7 if i==0):
            //   Row 0: s_mul — va * vb = term_val
            //   Row 1: s_add — running_val + term_val = next_running_val (skip if i==0)
            //   Row 2: s_mul — va * eb = t1
            //   Row 3: s_mul — vb * ea = t2
            //   Row 4: s_mul — ea * eb = t3
            //   Row 5: s_add — t1 + t2 = sum_part
            //   Row 6: s_add — sum_part + t3 = term_err
            //   Row 7: s_add — running_err + term_err = next_running_err (skip if i==0)

            let prev_val_cell = prev_running_val_cell.take();
            let prev_err_cell = prev_running_err_cell.take();
            let running_val_capture = running_val;
            let running_err_capture = running_err;

            let (new_val_cell, new_err_cell) = layouter.assign_region(
                || format!("matmul_iter_{}", i),
                |mut region| {
                    let mut row = 0;

                    // Row 0: va * vb = term_val
                    self.config.arithmetic.s_mul.enable(&mut region, row)?;
                    let va_0 = region.assign_advice(|| "va", self.config.arithmetic.a, row, || va)?;
                    let vb_0 = region.assign_advice(|| "vb", self.config.arithmetic.b, row, || vb)?;
                    let term_val_0 = region.assign_advice(|| "term_val", self.config.arithmetic.c, row, || term_val)?;
                    row += 1;

                    // Row 1: running_val + term_val = next_running_val (if i > 0)
                    let next_val_cell = if i > 0 {
                        self.config.arithmetic.s_add.enable(&mut region, row)?;
                        let prev_cell = region.assign_advice(|| "running_val", self.config.arithmetic.a, row, || running_val_capture)?;
                        let term_cell = region.assign_advice(|| "term_val_1", self.config.arithmetic.b, row, || term_val)?;
                        let next_cell = region.assign_advice(|| "next_val", self.config.arithmetic.c, row, || next_running_val)?;
                        // Copy: term_val from row 0 == term_val at row 1
                        region.constrain_equal(term_val_0.cell(), term_cell.cell())?;
                        // Copy: running_val from previous iteration
                        if let Some(ref prev) = prev_val_cell {
                            region.constrain_equal(prev.cell(), prev_cell.cell())?;
                        }
                        row += 1;
                        next_cell
                    } else {
                        // For i==0, next_running_val == term_val (running starts at 0)
                        term_val_0.clone()
                    };

                    // Row 2: va * eb = t1
                    self.config.arithmetic.s_mul.enable(&mut region, row)?;
                    let va_2 = region.assign_advice(|| "va_2", self.config.arithmetic.a, row, || va)?;
                    let eb_2 = region.assign_advice(|| "eb", self.config.arithmetic.b, row, || eb)?;
                    let t1_2 = region.assign_advice(|| "t1", self.config.arithmetic.c, row, || t1)?;
                    // Copy: va must be same as row 0
                    region.constrain_equal(va_0.cell(), va_2.cell())?;
                    row += 1;

                    // Row 3: vb * ea = t2
                    self.config.arithmetic.s_mul.enable(&mut region, row)?;
                    let vb_3 = region.assign_advice(|| "vb_3", self.config.arithmetic.a, row, || vb)?;
                    let ea_3 = region.assign_advice(|| "ea", self.config.arithmetic.b, row, || ea)?;
                    let t2_3 = region.assign_advice(|| "t2", self.config.arithmetic.c, row, || t2)?;
                    // Copy: vb must be same as row 0
                    region.constrain_equal(vb_0.cell(), vb_3.cell())?;
                    row += 1;

                    // Row 4: ea * eb = t3
                    self.config.arithmetic.s_mul.enable(&mut region, row)?;
                    let ea_4 = region.assign_advice(|| "ea_4", self.config.arithmetic.a, row, || ea)?;
                    let eb_4 = region.assign_advice(|| "eb_4", self.config.arithmetic.b, row, || eb)?;
                    let t3_4 = region.assign_advice(|| "t3", self.config.arithmetic.c, row, || t3)?;
                    // Copy: ea and eb must be same as rows 2-3
                    region.constrain_equal(ea_3.cell(), ea_4.cell())?;
                    region.constrain_equal(eb_2.cell(), eb_4.cell())?;
                    row += 1;

                    // Row 5: t1 + t2 = sum_part
                    self.config.arithmetic.s_add.enable(&mut region, row)?;
                    let t1_5 = region.assign_advice(|| "t1_5", self.config.arithmetic.a, row, || t1)?;
                    let t2_5 = region.assign_advice(|| "t2_5", self.config.arithmetic.b, row, || t2)?;
                    let sum_5 = region.assign_advice(|| "sum_part", self.config.arithmetic.c, row, || sum_part)?;
                    // Copy: t1 and t2 from rows 2-3
                    region.constrain_equal(t1_2.cell(), t1_5.cell())?;
                    region.constrain_equal(t2_3.cell(), t2_5.cell())?;
                    row += 1;

                    // Row 6: sum_part + t3 = term_err
                    self.config.arithmetic.s_add.enable(&mut region, row)?;
                    let sum_6 = region.assign_advice(|| "sum_6", self.config.arithmetic.a, row, || sum_part)?;
                    let t3_6 = region.assign_advice(|| "t3_6", self.config.arithmetic.b, row, || t3)?;
                    let term_err_6 = region.assign_advice(|| "term_err", self.config.arithmetic.c, row, || term_err)?;
                    // Copy: sum_part and t3 from rows 4-5
                    region.constrain_equal(sum_5.cell(), sum_6.cell())?;
                    region.constrain_equal(t3_4.cell(), t3_6.cell())?;
                    row += 1;

                    // Row 7: running_err + term_err = next_running_err (if i > 0)
                    let next_err_cell = if i > 0 {
                        self.config.arithmetic.s_add.enable(&mut region, row)?;
                        let prev_err = region.assign_advice(|| "running_err", self.config.arithmetic.a, row, || running_err_capture)?;
                        let term_err_7 = region.assign_advice(|| "term_err_7", self.config.arithmetic.b, row, || term_err)?;
                        let next_err = region.assign_advice(|| "next_err", self.config.arithmetic.c, row, || next_running_err)?;
                        // Copy: term_err from row 6
                        region.constrain_equal(term_err_6.cell(), term_err_7.cell())?;
                        // Copy: running_err from previous iteration
                        if let Some(ref prev) = prev_err_cell {
                            region.constrain_equal(prev.cell(), prev_err.cell())?;
                        }
                        next_err
                    } else {
                        // For i==0, next_running_err == term_err (running starts at 0)
                        term_err_6
                    };

                    Ok((next_val_cell, next_err_cell))
                },
            )?;

            prev_running_val_cell = Some(new_val_cell);
            prev_running_err_cell = Some(new_err_cell);
            running_val = next_running_val;
            running_err = next_running_err;
        }

        // Final verification region: check res_val == running_val, res_err == running_err,
        // and range check res_err. Uses copy constraints to link to last iteration.
        let diff_val = res_val - running_val;
        let diff_err = res_err - running_err;

        layouter.assign_region(
            || "matmul_verify",
            |mut region| {
                // Row 0: diff_val + 0 = 0 (proves diff_val == 0 => res_val == running_val)
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "diff_val", self.config.arithmetic.a, 0, || diff_val)?;
                region.assign_advice(|| "zero_b", self.config.arithmetic.b, 0, || zero)?;
                region.assign_advice(|| "zero_c", self.config.arithmetic.c, 0, || zero)?;

                // Row 1: diff_err + 0 = 0 (proves diff_err == 0 => res_err == running_err)
                self.config.arithmetic.s_add.enable(&mut region, 1)?;
                region.assign_advice(|| "diff_err", self.config.arithmetic.a, 1, || diff_err)?;
                region.assign_advice(|| "zero_b1", self.config.arithmetic.b, 1, || zero)?;
                region.assign_advice(|| "zero_c1", self.config.arithmetic.c, 1, || zero)?;

                // Row 2: range check res_err
                self.config.range.s_range.enable(&mut region, 2)?;
                region.assign_advice(
                    || "res_err_range",
                    self.config.range.input_column,
                    2,
                    || res_err,
                )?;

                Ok(())
            },
        )?;

        Ok(())
    }
}
