//! Bounded Linear Layer Circuit.
//!
//! Proves that a linear layer computation y = Wx + b is correct within error bounds.
//!
//! The circuit verifies:
//! 1. The matrix-vector multiplication Wx is computed correctly
//! 2. The bias addition is correct
//! 3. Error propagation follows the bounded arithmetic rules
//! 4. Final output error is within acceptable range

use crate::approximate::bounded_matmul::{BoundedMatMulChip, BoundedMatMulConfig};
use crate::approximate::bounded_add::{BoundedAddChip, BoundedAddConfig};
use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, Advice, ConstraintSystem, Error, Selector, Instance},
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

/// Configuration for the bounded linear layer circuit.
#[derive(Clone, Debug)]
pub struct BoundedLinearConfig<F: PrimeField, const RANGE: usize> {
    /// Arithmetic operations config.
    pub arithmetic: ArithmeticConfig,
    /// Range check config.
    pub range: RangeConfig<F, RANGE>,
    /// MatMul config (uses arithmetic + range).
    pub matmul: BoundedMatMulConfig<F, RANGE>,
    /// Selector for weight commitment verification.
    pub s_weight_commit: Selector,
    /// Advice columns for layer inputs/outputs.
    pub input: Column<Advice>,
    pub output: Column<Advice>,
    pub weight: Column<Advice>,
    pub bias: Column<Advice>,
    /// Public instance column for commitments.
    pub instance: Column<Instance>,
}

/// A chip that proves bounded linear layer execution.
pub struct BoundedLinearChip<F: PrimeField, const RANGE: usize> {
    config: BoundedLinearConfig<F, RANGE>,
    matmul_chip: BoundedMatMulChip<F, RANGE>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> BoundedLinearChip<F, RANGE> {
    /// Creates a new bounded linear layer chip.
    pub fn new(config: BoundedLinearConfig<F, RANGE>) -> Self {
        let matmul_chip = BoundedMatMulChip::new(config.matmul.clone());
        Self {
            config,
            matmul_chip,
            _marker: PhantomData,
        }
    }

    /// Configures the bounded linear layer circuit.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
    ) -> BoundedLinearConfig<F, RANGE> {
        // Create advice columns
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let input = meta.advice_column();
        let output = meta.advice_column();
        let weight = meta.advice_column();
        let bias = meta.advice_column();
        
        // Create instance column for public inputs (commitments)
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        
        // Enable equality for copy constraints
        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);
        meta.enable_equality(input);
        meta.enable_equality(output);
        meta.enable_equality(weight);
        meta.enable_equality(bias);
        
        // Create lookup table column for range checks
        let table = meta.lookup_table_column();
        
        // Configure arithmetic chip
        let arithmetic = ArithmeticChip::<F>::configure(meta, a, b, c);
        
        // Configure range chip
        let range = RangeConfig::configure(meta, table, c);
        
        // Create matmul config
        let matmul = BoundedMatMulConfig {
            arithmetic: arithmetic.clone(),
            range: range.clone(),
        };
        
        // Weight commitment selector
        let s_weight_commit = meta.selector();
        
        BoundedLinearConfig {
            arithmetic,
            range,
            matmul,
            s_weight_commit,
            input,
            output,
            weight,
            bias,
            instance,
        }
    }

    /// Assigns a linear layer computation: output = input @ weight.T + bias
    ///
    /// Parameters:
    /// - input_vals: Input vector values [batch, in_features]
    /// - input_errs: Input vector errors [batch, in_features]
    /// - weight_vals: Weight matrix values [out_features, in_features]
    /// - weight_errs: Weight matrix errors [out_features, in_features]
    /// - bias_vals: Bias vector values [out_features]
    /// - bias_errs: Bias vector errors [out_features]
    /// - output_vals: Expected output values [batch, out_features]
    /// - output_errs: Expected output errors [batch, out_features]
    pub fn assign_linear_layer(
        &self,
        mut layouter: impl Layouter<F>,
        // For simplicity, we work with a single input vector (batch=1)
        input_vals: &[Value<F>],
        input_errs: &[Value<F>],
        weight_vals: &[Vec<Value<F>>], // [out_features][in_features]
        weight_errs: &[Vec<Value<F>>],
        bias_vals: &[Value<F>],
        bias_errs: &[Value<F>],
        output_vals: &[Value<F>],
        output_errs: &[Value<F>],
    ) -> Result<(), Error> {
        let in_features = input_vals.len();
        let out_features = weight_vals.len();
        
        // Verify dimensions
        if weight_vals.iter().any(|w| w.len() != in_features) {
            return Err(Error::Synthesis);
        }
        if bias_vals.len() != out_features || output_vals.len() != out_features {
            return Err(Error::Synthesis);
        }
        
        // For each output neuron, we compute: o_j = sum_i(w_ji * x_i) + b_j
        for j in 0..out_features {
            // Step 1: Compute the dot product using bounded matmul
            // This proves: matmul_result = sum_i(w_ji * x_i) with bounded error
            
            let row_w_vals = &weight_vals[j];
            let row_w_errs = &weight_errs[j];
            
            // Compute expected matmul result (before bias)
            let matmul_val = compute_dot_product(row_w_vals, input_vals);
            let matmul_err = compute_dot_product_error(
                row_w_vals, row_w_errs,
                input_vals, input_errs
            );
            
            // Assign the dot product verification
            self.matmul_chip.assign_dot_product(
                layouter.namespace(|| format!("dot product for output {}", j)),
                row_w_vals,
                row_w_errs,
                input_vals,
                input_errs,
                matmul_val,
                matmul_err,
            )?;
            
            // Step 2: Add bias using bounded addition
            // This proves: output = matmul_result + bias with bounded error
            
            let bias_val = bias_vals[j];
            let bias_err = bias_errs[j];
            let final_output_val = output_vals[j];
            let final_output_err = output_errs[j];
            
            // Constrain: matmul_val + bias_val = output_val
            layouter.assign_region(
                || format!("bias addition for output {}", j),
                |mut region| {
                    self.config.arithmetic.s_add.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "matmul_result",
                        self.config.arithmetic.a,
                        0,
                        || matmul_val,
                    )?;
                    region.assign_advice(
                        || "bias",
                        self.config.arithmetic.b,
                        0,
                        || bias_val,
                    )?;
                    region.assign_advice(
                        || "output",
                        self.config.arithmetic.c,
                        0,
                        || final_output_val,
                    )?;
                    Ok(())
                },
            )?;
            
            // Constrain: matmul_err + bias_err = output_err
            layouter.assign_region(
                || format!("error addition for output {}", j),
                |mut region| {
                    self.config.arithmetic.s_add.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "matmul_error",
                        self.config.arithmetic.a,
                        0,
                        || matmul_err,
                    )?;
                    region.assign_advice(
                        || "bias_error",
                        self.config.arithmetic.b,
                        0,
                        || bias_err,
                    )?;
                    region.assign_advice(
                        || "output_error",
                        self.config.arithmetic.c,
                        0,
                        || final_output_err,
                    )?;
                    Ok(())
                },
            )?;
            
            // Step 3: Range check the output error
            layouter.assign_region(
                || format!("range check output error {}", j),
                |mut region| {
                    self.config.range.s_range.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "output_error_range",
                        self.config.range.input_column,
                        0,
                        || final_output_err,
                    )?;
                    Ok(())
                },
            )?;
        }
        
        Ok(())
    }
}

/// Compute the dot product of two vectors (Value version).
fn compute_dot_product<F: PrimeField>(a: &[Value<F>], b: &[Value<F>]) -> Value<F> {
    a.iter()
        .zip(b.iter())
        .fold(Value::known(F::ZERO), |acc, (ai, bi)| {
            acc + (*ai * *bi)
        })
}

/// Compute the error bound for a dot product.
/// Error = sum_i(|a_i| * err_b_i + |b_i| * err_a_i + err_a_i * err_b_i)
fn compute_dot_product_error<F: PrimeField>(
    a_vals: &[Value<F>],
    a_errs: &[Value<F>],
    b_vals: &[Value<F>],
    b_errs: &[Value<F>],
) -> Value<F> {
    a_vals.iter()
        .zip(a_errs.iter())
        .zip(b_vals.iter().zip(b_errs.iter()))
        .fold(Value::known(F::ZERO), |acc, ((av, ae), (bv, be))| {
            // term = |a| * err_b + |b| * err_a + err_a * err_b
            // In field arithmetic, we use actual values (assuming positive in demo)
            let term = (*av * *be) + (*bv * *ae) + (*ae * *be);
            acc + term
        })
}

/// A complete circuit for verifying a bounded linear layer.
#[derive(Clone, Default)]
pub struct BoundedLinearCircuit<F: PrimeField, const RANGE: usize> {
    /// Input values.
    pub input_vals: Vec<F>,
    /// Input errors.
    pub input_errs: Vec<F>,
    /// Weight matrix values (flattened, row-major).
    pub weight_vals: Vec<Vec<F>>,
    /// Weight matrix errors.
    pub weight_errs: Vec<Vec<F>>,
    /// Bias values.
    pub bias_vals: Vec<F>,
    /// Bias errors.
    pub bias_errs: Vec<F>,
    /// Output values.
    pub output_vals: Vec<F>,
    /// Output errors.
    pub output_errs: Vec<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> BoundedLinearCircuit<F, RANGE> {
    /// Creates a new circuit with the given parameters.
    pub fn new(
        input_vals: Vec<F>,
        input_errs: Vec<F>,
        weight_vals: Vec<Vec<F>>,
        weight_errs: Vec<Vec<F>>,
        bias_vals: Vec<F>,
        bias_errs: Vec<F>,
        output_vals: Vec<F>,
        output_errs: Vec<F>,
    ) -> Self {
        Self {
            input_vals,
            input_errs,
            weight_vals,
            weight_errs,
            bias_vals,
            bias_errs,
            output_vals,
            output_errs,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField, const RANGE: usize> Circuit<F> for BoundedLinearCircuit<F, RANGE> {
    type Config = BoundedLinearConfig<F, RANGE>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        BoundedLinearChip::<F, RANGE>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // Load range table
        let range_chip = RangeChip::<F, RANGE>::new(config.range.clone());
        range_chip.load(&mut layouter)?;
        
        // Create the linear layer chip
        let chip = BoundedLinearChip::<F, RANGE>::new(config);
        
        // Convert to Value types
        let input_vals: Vec<_> = self.input_vals.iter().map(|v| Value::known(*v)).collect();
        let input_errs: Vec<_> = self.input_errs.iter().map(|v| Value::known(*v)).collect();
        let weight_vals: Vec<Vec<_>> = self.weight_vals.iter()
            .map(|row| row.iter().map(|v| Value::known(*v)).collect())
            .collect();
        let weight_errs: Vec<Vec<_>> = self.weight_errs.iter()
            .map(|row| row.iter().map(|v| Value::known(*v)).collect())
            .collect();
        let bias_vals: Vec<_> = self.bias_vals.iter().map(|v| Value::known(*v)).collect();
        let bias_errs: Vec<_> = self.bias_errs.iter().map(|v| Value::known(*v)).collect();
        let output_vals: Vec<_> = self.output_vals.iter().map(|v| Value::known(*v)).collect();
        let output_errs: Vec<_> = self.output_errs.iter().map(|v| Value::known(*v)).collect();
        
        // Assign the linear layer
        chip.assign_linear_layer(
            layouter.namespace(|| "linear layer"),
            &input_vals,
            &input_errs,
            &weight_vals,
            &weight_errs,
            &bias_vals,
            &bias_errs,
            &output_vals,
            &output_errs,
        )?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_bounded_linear_layer_simple() {
        // Simple test: 1 input, 1 output
        // y = w * x + b = 2 * 3 + 1 = 7
        
        let input_vals = vec![Fr::from(3)];
        let input_errs = vec![Fr::from(1)];
        let weight_vals = vec![vec![Fr::from(2)]];
        let weight_errs = vec![vec![Fr::from(1)]];
        let bias_vals = vec![Fr::from(1)];
        let bias_errs = vec![Fr::from(1)];
        
        // output = 2*3 + 1 = 7
        let output_vals = vec![Fr::from(7)];
        
        // Error = |w|*err_x + |x|*err_w + err_w*err_x + err_b
        // = 2*1 + 3*1 + 1*1 + 1 = 7
        let output_errs = vec![Fr::from(7)];
        
        let circuit = BoundedLinearCircuit::<Fr, 100>::new(
            input_vals,
            input_errs,
            weight_vals,
            weight_errs,
            bias_vals,
            bias_errs,
            output_vals,
            output_errs,
        );
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_bounded_linear_layer_2x2() {
        // Test: 2 inputs, 2 outputs
        // Input: [1, 2] with errors [0, 0]
        // Weights: [[1, 1], [2, 2]] with errors [[0, 0], [0, 0]]
        // Bias: [0, 0] with errors [0, 0]
        // Output: [1*1 + 1*2 + 0, 2*1 + 2*2 + 0] = [3, 6]
        
        let input_vals = vec![Fr::from(1), Fr::from(2)];
        let input_errs = vec![Fr::from(0), Fr::from(0)];
        let weight_vals = vec![
            vec![Fr::from(1), Fr::from(1)],
            vec![Fr::from(2), Fr::from(2)],
        ];
        let weight_errs = vec![
            vec![Fr::from(0), Fr::from(0)],
            vec![Fr::from(0), Fr::from(0)],
        ];
        let bias_vals = vec![Fr::from(0), Fr::from(0)];
        let bias_errs = vec![Fr::from(0), Fr::from(0)];
        let output_vals = vec![Fr::from(3), Fr::from(6)];
        let output_errs = vec![Fr::from(0), Fr::from(0)];
        
        let circuit = BoundedLinearCircuit::<Fr, 100>::new(
            input_vals,
            input_errs,
            weight_vals,
            weight_errs,
            bias_vals,
            bias_errs,
            output_vals,
            output_errs,
        );
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_bounded_linear_layer_invalid_output() {
        // Same as simple test but with wrong output value
        let input_vals = vec![Fr::from(3)];
        let input_errs = vec![Fr::from(1)];
        let weight_vals = vec![vec![Fr::from(2)]];
        let weight_errs = vec![vec![Fr::from(1)]];
        let bias_vals = vec![Fr::from(1)];
        let bias_errs = vec![Fr::from(1)];
        
        // Wrong output: should be 7, but we claim 8
        let output_vals = vec![Fr::from(8)];
        let output_errs = vec![Fr::from(7)];
        
        let circuit = BoundedLinearCircuit::<Fr, 100>::new(
            input_vals,
            input_errs,
            weight_vals,
            weight_errs,
            bias_vals,
            bias_errs,
            output_vals,
            output_errs,
        );
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err());
    }
}
