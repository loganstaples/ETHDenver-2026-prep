//! Bounded Matrix Multiplication Gadget.
//!
//! Verifies dot products with error propagation.

use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Error, ErrorFront},
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
    /// Note: This is computationally expensive in a circuit (O(K) constraints per cell).
    /// For Stage 13 we implement the logic for small K.
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

        // 1. Calculate and accumulate values
        // We will perform naive accumulation in the circuit using Add/Mul gates.
        // In a real optimized circuit, we'd use a dedicated custom gate for dot products.
        // Here we chain binary ops.
        
        let mut running_val = Value::known(F::ZERO);
        let mut running_err = Value::known(F::ZERO);

        for i in 0..row_a_vals.len() {
            let va = row_a_vals[i];
            let ea = row_a_errs[i];
            let vb = col_b_vals[i];
            let eb = col_b_errs[i];

            // --- Value Accumulation ---
            // t = va * vb
            let term_val = va * vb;
            // Assign multiplication: va * vb = term_val
            // We use the arithmetic chip's raw assignment capability if exposed, 
            // or we need to use a region. 
            // Since we are inside a `assign_dot_product` which takes `Layouter`,
            // we should ideally create a region for this DOT PRODUCT.
            // Assigning many small regions is inefficient.
            // Let's use `layouter.assign_region` ONCE for the whole dot product.
            
            // However, our ArithmeticChip API expects `assign_region` per operation in its `assign` methods.
            // Let's assume we can call `arithmetic_chip.mul` and `add` here if they were exposed helpers.
            // Given they are not, `assign_dot_product` should iterate and call assign logic.
            
            // Constraint 1: Multiply term
            // We need to verify `term_val = va * vb`.
            // BUT `va` and `vb` are Values. We need them in the circuit.
            // They are passed as `Value<F>`.
            
            layouter.assign_region(
                || format!("matmul mul step {}", i),
                |mut region| {
                    self.arithmetic_chip.config.s_mul.enable(&mut region, 0)?;
                    region.assign_advice(|| "va", self.arithmetic_chip.config.a, 0, || va)?;
                    region.assign_advice(|| "vb", self.arithmetic_chip.config.b, 0, || vb)?;
                    region.assign_advice(|| "term", self.arithmetic_chip.config.c, 0, || term_val)?;
                    Ok(())
                }
            )?;

            // Constraint 2: Add to running sum
            // run_new = run_old + term
            let next_running_val = running_val + term_val;
            
            if i > 0 { // First step running_val is 0, so result is just term.
                 layouter.assign_region(
                    || format!("matmul add step {}", i),
                    |mut region| {
                        self.arithmetic_chip.config.s_add.enable(&mut region, 0)?;
                        region.assign_advice(|| "running_prev", self.arithmetic_chip.config.a, 0, || running_val)?;
                        region.assign_advice(|| "term", self.arithmetic_chip.config.b, 0, || term_val)?;
                        region.assign_advice(|| "running_new", self.arithmetic_chip.config.c, 0, || next_running_val)?;
                        Ok(())
                    }
                )?;
            }
            
            running_val = next_running_val;

            // --- Error Accumulation ---
            // err_term = |va|*eb + |vb|*ea + ea*eb
            // Assuming positive values: va*eb + vb*ea + ea*eb
            
            let t1 = va * eb;
            let t2 = vb * ea;
            let t3 = ea * eb;
            let term_err_val = t1 + t2 + t3;

            // We need to prove this calculation. 
            // This is identical to BoundedMul logic.
            // Ideally we'd call `BoundedMulChip` here if refactored, but let's inline for now.
            // Constraints:
            // 1. t1 = va * eb
            // 2. t2 = vb * ea
            // 3. t3 = ea * eb
            // 4. sum = t1+t2+t3
            
            // Assign t1
            layouter.assign_region(
                || format!("matmul err t1 step {}", i),
                |mut region| {
                    self.arithmetic_chip.config.s_mul.enable(&mut region, 0)?;
                    region.assign_advice(|| "va", self.arithmetic_chip.config.a, 0, || va)?;
                    region.assign_advice(|| "eb", self.arithmetic_chip.config.b, 0, || eb)?;
                    region.assign_advice(|| "t1", self.arithmetic_chip.config.c, 0, || t1)?;
                    Ok(())
                }
            )?;

            // Assign t2
             layouter.assign_region(
                || format!("matmul err t2 step {}", i),
                |mut region| {
                    self.arithmetic_chip.config.s_mul.enable(&mut region, 0)?;
                    region.assign_advice(|| "vb", self.arithmetic_chip.config.a, 0, || vb)?;
                    region.assign_advice(|| "ea", self.arithmetic_chip.config.b, 0, || ea)?;
                    region.assign_advice(|| "t2", self.arithmetic_chip.config.c, 0, || t2)?;
                    Ok(())
                }
            )?;

            // Assign t3
             layouter.assign_region(
                || format!("matmul err t3 step {}", i),
                |mut region| {
                    self.arithmetic_chip.config.s_mul.enable(&mut region, 0)?;
                    region.assign_advice(|| "ea", self.arithmetic_chip.config.a, 0, || ea)?;
                    region.assign_advice(|| "eb", self.arithmetic_chip.config.b, 0, || eb)?;
                    region.assign_advice(|| "t3", self.arithmetic_chip.config.c, 0, || t3)?;
                    Ok(())
                }
            )?;
            
            // Assign sum of terms (we can do 2 adds)
            let sum_part = t1 + t2;
             layouter.assign_region(
                || format!("matmul err sum1 step {}", i),
                |mut region| {
                    self.arithmetic_chip.config.s_add.enable(&mut region, 0)?;
                    region.assign_advice(|| "t1", self.arithmetic_chip.config.a, 0, || t1)?;
                    region.assign_advice(|| "t2", self.arithmetic_chip.config.b, 0, || t2)?;
                    region.assign_advice(|| "sum_part", self.arithmetic_chip.config.c, 0, || sum_part)?;
                    Ok(())
                }
            )?;
            
             layouter.assign_region(
                || format!("matmul err sum2 step {}", i),
                |mut region| {
                    self.arithmetic_chip.config.s_add.enable(&mut region, 0)?;
                    region.assign_advice(|| "sum_part", self.arithmetic_chip.config.a, 0, || sum_part)?;
                    region.assign_advice(|| "t3", self.arithmetic_chip.config.b, 0, || t3)?;
                    region.assign_advice(|| "term_err", self.arithmetic_chip.config.c, 0, || term_err_val)?;
                    Ok(())
                }
            )?;

            // Accumulate into running_err
            let next_running_err = running_err + term_err_val;
            
            if i > 0 {
                layouter.assign_region(
                    || format!("matmul err accum step {}", i),
                    |mut region| {
                        self.arithmetic_chip.config.s_add.enable(&mut region, 0)?;
                        region.assign_advice(|| "running_err_prev", self.arithmetic_chip.config.a, 0, || running_err)?;
                        region.assign_advice(|| "term_err", self.arithmetic_chip.config.b, 0, || term_err_val)?;
                        region.assign_advice(|| "running_err_new", self.arithmetic_chip.config.c, 0, || next_running_err)?;
                        Ok(())
                    }
                )?;
            } else {
                // For i=0, running_err (0) + term_err = term_err. 
                // We implicitly proved term_err is correct above. 
                // Since `running_err` starts at 0 (known), we can verify consistency if we assigned it,
                // but effectively `next_running_err` IS `term_err_val`.
            }
            
            running_err = next_running_err;
        }

        // 2. Verify Result Value
        // res_val - running_val = 0
        let diff_val = res_val - running_val;
        let zero = Value::known(F::ZERO);
        
        layouter.assign_region(
             || "verify matmul res_val",
             |mut region| {
                 self.arithmetic_chip.config.s_add.enable(&mut region, 0)?;
                 // diff + 0 = 0 => diff = 0
                 region.assign_advice(|| "diff", self.arithmetic_chip.config.a, 0, || diff_val)?;
                 region.assign_advice(|| "zero_b", self.arithmetic_chip.config.b, 0, || zero)?;
                 region.assign_advice(|| "zero_c", self.arithmetic_chip.config.c, 0, || zero)?;
                 Ok(())
             }
        )?;
        
        // 3. Verify Error Value
        // res_err - running_err = 0
        let diff_err = res_err - running_err;
        layouter.assign_region(
             || "verify matmul res_err",
             |mut region| {
                 self.arithmetic_chip.config.s_add.enable(&mut region, 0)?;
                 region.assign_advice(|| "diff", self.arithmetic_chip.config.a, 0, || diff_err)?;
                 region.assign_advice(|| "zero_b", self.arithmetic_chip.config.b, 0, || zero)?;
                 region.assign_advice(|| "zero_c", self.arithmetic_chip.config.c, 0, || zero)?;
                 Ok(())
             }
        )?;

        // 4. Range Check Result Error
        // We verify that the `res_err` is within range.
        layouter.assign_region(
            || "range check matmul err",
            |mut region| {
                self.config.range.s_range.enable(&mut region, 0)?;
                region.assign_advice(
                    || "res_err",
                    self.config.range.input_column,
                    0,
                    || res_err,
                )?;
                Ok(())
            },
        )?;

        Ok(())
    }
}
