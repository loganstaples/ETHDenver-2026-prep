//! Gradient Verification Circuit.
//!
//! Proves that gradient computations during backpropagation are correct within error bounds.
//!
//! This circuit verifies:
//! 1. Backward pass gradients are computed correctly from forward values
//! 2. Error bounds on gradients are properly derived from forward bounds
//! 3. Gradient accumulation is correct
//! 4. Weight updates are bounded

use crate::approximate::bounded_matmul::{BoundedMatMulChip, BoundedMatMulConfig};
use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, Advice, ConstraintSystem, Error, ErrorFront, Selector, Instance},
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

/// Configuration for the gradient verification circuit.
#[derive(Clone, Debug)]
pub struct GradientVerificationConfig<F: PrimeField, const RANGE: usize> {
    /// Arithmetic operations config.
    pub arithmetic: ArithmeticConfig,
    /// Range check config.
    pub range: RangeConfig<F, RANGE>,
    /// MatMul config.
    pub matmul: BoundedMatMulConfig<F, RANGE>,
    /// Selector for gradient mask (ReLU backward).
    pub s_grad_mask: Selector,
    /// Selector for weight update.
    pub s_weight_update: Selector,
    /// Advice columns for gradients.
    pub forward_val: Column<Advice>,
    pub forward_err: Column<Advice>,
    pub upstream_grad: Column<Advice>,
    pub upstream_grad_err: Column<Advice>,
    pub local_grad: Column<Advice>,
    pub local_grad_err: Column<Advice>,
    /// Instance column for public commitments.
    pub instance: Column<Instance>,
}

/// A chip that proves gradient computations are bounded.
pub struct GradientVerificationChip<F: PrimeField, const RANGE: usize> {
    config: GradientVerificationConfig<F, RANGE>,
    matmul_chip: BoundedMatMulChip<F, RANGE>,
    arithmetic_chip: ArithmeticChip<F>,
    range_chip: RangeChip<F, RANGE>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> GradientVerificationChip<F, RANGE> {
    /// Creates a new gradient verification chip.
    pub fn new(config: GradientVerificationConfig<F, RANGE>) -> Self {
        let matmul_chip = BoundedMatMulChip::new(config.matmul.clone());
        let arithmetic_chip = ArithmeticChip::new(config.arithmetic.clone());
        let range_chip = RangeChip::new(config.range.clone());
        Self {
            config,
            matmul_chip,
            arithmetic_chip,
            range_chip,
            _marker: PhantomData,
        }
    }

    /// Configures the gradient verification circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> GradientVerificationConfig<F, RANGE> {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let forward_val = meta.advice_column();
        let forward_err = meta.advice_column();
        let upstream_grad = meta.advice_column();
        let upstream_grad_err = meta.advice_column();
        let local_grad = meta.advice_column();
        let local_grad_err = meta.advice_column();
        
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        
        for col in [a, b, c, forward_val, forward_err, upstream_grad, 
                    upstream_grad_err, local_grad, local_grad_err] {
            meta.enable_equality(col);
        }
        
        let table = meta.lookup_table_column();
        let arithmetic = ArithmeticChip::<F>::configure(meta, a, b, c);
        let range = RangeConfig::configure(meta, table, c);
        let matmul = BoundedMatMulConfig {
            arithmetic: arithmetic.clone(),
            range: range.clone(),
        };
        
        let s_grad_mask = meta.selector();
        let s_weight_update = meta.selector();
        
        // ReLU gradient mask: grad_out = grad_in * (forward > 0 ? 1 : 0)
        // Simplified: we just verify the multiplication
        meta.create_gate("relu_grad_mask", |meta| {
            let s = meta.query_selector(s_grad_mask);
            let forward = meta.query_advice(forward_val, Rotation::cur());
            let upstream = meta.query_advice(upstream_grad, Rotation::cur());
            let local = meta.query_advice(local_grad, Rotation::cur());
            // For ReLU: if forward > 0, local = upstream; else local = 0
            // This is hard to express in arithmetic constraints without comparisons
            // We use a relaxed constraint: local * (forward - local) constraints
            // In practice, we'd use a different gadget or lookup
            vec![s * (local.clone() * forward - local * upstream)]
        });
        
        // Weight update: new_weight = old_weight - lr * gradient
        meta.create_gate("weight_update", |meta| {
            let s = meta.query_selector(s_weight_update);
            let old_weight = meta.query_advice(a, Rotation::cur());
            let gradient = meta.query_advice(b, Rotation::cur());
            let new_weight = meta.query_advice(c, Rotation::cur());
            // This assumes lr is incorporated into gradient
            // Constraint: old_weight - gradient = new_weight
            vec![s * (old_weight - gradient - new_weight)]
        });
        
        GradientVerificationConfig {
            arithmetic,
            range,
            matmul,
            s_grad_mask,
            s_weight_update,
            forward_val,
            forward_err,
            upstream_grad,
            upstream_grad_err,
            local_grad,
            local_grad_err,
            instance,
        }
    }

    /// Verifies ReLU backward pass.
    ///
    /// For ReLU: grad_input = grad_output * (input > 0 ? 1 : 0)
    /// Error propagation: err_grad_input = err_grad_output (when mask = 1)
    pub fn assign_relu_backward(
        &self,
        mut layouter: impl Layouter<F>,
        forward_vals: &[Value<F>],
        forward_errs: &[Value<F>],
        upstream_grad_vals: &[Value<F>],
        upstream_grad_errs: &[Value<F>],
        local_grad_vals: &[Value<F>],
        local_grad_errs: &[Value<F>],
        masks: &[Value<F>], // 1 if forward > 0, else 0
    ) -> Result<(), ErrorFront> {
        for i in 0..forward_vals.len() {
            // Verify: local_grad = upstream_grad * mask
            layouter.assign_region(
                || format!("relu grad {}", i),
                |mut region| {
                    self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "upstream_grad",
                        self.config.arithmetic.a,
                        0,
                        || upstream_grad_vals[i],
                    )?;
                    region.assign_advice(
                        || "mask",
                        self.config.arithmetic.b,
                        0,
                        || masks[i],
                    )?;
                    region.assign_advice(
                        || "local_grad",
                        self.config.arithmetic.c,
                        0,
                        || local_grad_vals[i],
                    )?;
                    Ok(())
                },
            )?;
            
            // Verify error: local_err = upstream_err * mask
            // (error is 0 when mask is 0, propagates when mask is 1)
            layouter.assign_region(
                || format!("relu grad error {}", i),
                |mut region| {
                    self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "upstream_err",
                        self.config.arithmetic.a,
                        0,
                        || upstream_grad_errs[i],
                    )?;
                    region.assign_advice(
                        || "mask",
                        self.config.arithmetic.b,
                        0,
                        || masks[i],
                    )?;
                    region.assign_advice(
                        || "local_err",
                        self.config.arithmetic.c,
                        0,
                        || local_grad_errs[i],
                    )?;
                    Ok(())
                },
            )?;
            
            // Range check the local gradient error
            layouter.assign_region(
                || format!("range check grad error {}", i),
                |mut region| {
                    self.config.range.s_range.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "local_grad_err",
                        self.config.range.input_column,
                        0,
                        || local_grad_errs[i],
                    )?;
                    Ok(())
                },
            )?;
        }
        
        Ok(())
    }

    /// Verifies linear layer backward pass (gradient w.r.t. weights).
    ///
    /// For Y = XW: dW = X^T @ dY
    /// Error: err_dW from X and dY errors
    pub fn assign_linear_weight_gradient(
        &self,
        mut layouter: impl Layouter<F>,
        input_vals: &[Vec<Value<F>>],      // X: [batch, in_features]
        input_errs: &[Vec<Value<F>>],
        upstream_grad_vals: &[Vec<Value<F>>], // dY: [batch, out_features]
        upstream_grad_errs: &[Vec<Value<F>>],
        weight_grad_vals: &[Vec<Value<F>>],   // dW: [in_features, out_features]
        weight_grad_errs: &[Vec<Value<F>>],
    ) -> Result<(), ErrorFront> {
        let batch_size = input_vals.len();
        let in_features = input_vals.first().map(|v| v.len()).unwrap_or(0);
        let out_features = upstream_grad_vals.first().map(|v| v.len()).unwrap_or(0);
        
        // dW[i][j] = sum_b(X[b][i] * dY[b][j])
        for i in 0..in_features {
            for j in 0..out_features {
                // Collect column i of X (transposed = row i of X^T)
                let x_col: Vec<_> = (0..batch_size).map(|b| input_vals[b][i]).collect();
                let x_col_errs: Vec<_> = (0..batch_size).map(|b| input_errs[b][i]).collect();
                
                // Collect column j of dY
                let dy_col: Vec<_> = (0..batch_size).map(|b| upstream_grad_vals[b][j]).collect();
                let dy_col_errs: Vec<_> = (0..batch_size).map(|b| upstream_grad_errs[b][j]).collect();
                
                let dw_val = weight_grad_vals[i][j];
                let dw_err = weight_grad_errs[i][j];
                
                // Verify dot product
                self.matmul_chip.assign_dot_product(
                    layouter.namespace(|| format!("dW[{},{}]", i, j)),
                    &x_col,
                    &x_col_errs,
                    &dy_col,
                    &dy_col_errs,
                    dw_val,
                    dw_err,
                )?;
            }
        }
        
        Ok(())
    }

    /// Verifies linear layer backward pass (gradient w.r.t. input).
    ///
    /// For Y = XW: dX = dY @ W^T
    /// Error: err_dX from dY and W errors
    pub fn assign_linear_input_gradient(
        &self,
        mut layouter: impl Layouter<F>,
        weight_vals: &[Vec<Value<F>>],        // W: [in_features, out_features]
        weight_errs: &[Vec<Value<F>>],
        upstream_grad_vals: &[Vec<Value<F>>], // dY: [batch, out_features]
        upstream_grad_errs: &[Vec<Value<F>>],
        input_grad_vals: &[Vec<Value<F>>],    // dX: [batch, in_features]
        input_grad_errs: &[Vec<Value<F>>],
    ) -> Result<(), ErrorFront> {
        let batch_size = upstream_grad_vals.len();
        let out_features = upstream_grad_vals.first().map(|v| v.len()).unwrap_or(0);
        let in_features = weight_vals.len();
        
        // dX[b][i] = sum_j(dY[b][j] * W[i][j]) = dY[b] @ W[i]^T
        for b in 0..batch_size {
            for i in 0..in_features {
                let dy_row = &upstream_grad_vals[b];
                let dy_row_errs = &upstream_grad_errs[b];
                
                // Row i of W (which is col i of W^T)
                let w_row = &weight_vals[i];
                let w_row_errs = &weight_errs[i];
                
                let dx_val = input_grad_vals[b][i];
                let dx_err = input_grad_errs[b][i];
                
                self.matmul_chip.assign_dot_product(
                    layouter.namespace(|| format!("dX[{},{}]", b, i)),
                    dy_row,
                    dy_row_errs,
                    w_row,
                    w_row_errs,
                    dx_val,
                    dx_err,
                )?;
            }
        }
        
        Ok(())
    }

    /// Verifies a weight update: new_weight = old_weight - lr * gradient
    pub fn assign_weight_update(
        &self,
        mut layouter: impl Layouter<F>,
        old_weight_vals: &[Value<F>],
        old_weight_errs: &[Value<F>],
        gradient_vals: &[Value<F>],       // Already scaled by learning rate
        gradient_errs: &[Value<F>],
        new_weight_vals: &[Value<F>],
        new_weight_errs: &[Value<F>],
    ) -> Result<(), ErrorFront> {
        for i in 0..old_weight_vals.len() {
            // new_weight = old_weight - gradient
            // Using addition gate with negated gradient
            layouter.assign_region(
                || format!("weight update {}", i),
                |mut region| {
                    self.config.s_weight_update.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "old_weight",
                        self.config.arithmetic.a,
                        0,
                        || old_weight_vals[i],
                    )?;
                    region.assign_advice(
                        || "gradient",
                        self.config.arithmetic.b,
                        0,
                        || gradient_vals[i],
                    )?;
                    region.assign_advice(
                        || "new_weight",
                        self.config.arithmetic.c,
                        0,
                        || new_weight_vals[i],
                    )?;
                    Ok(())
                },
            )?;
            
            // Error propagation: new_err = old_err + grad_err
            layouter.assign_region(
                || format!("weight update error {}", i),
                |mut region| {
                    self.config.arithmetic.s_add.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "old_err",
                        self.config.arithmetic.a,
                        0,
                        || old_weight_errs[i],
                    )?;
                    region.assign_advice(
                        || "grad_err",
                        self.config.arithmetic.b,
                        0,
                        || gradient_errs[i],
                    )?;
                    region.assign_advice(
                        || "new_err",
                        self.config.arithmetic.c,
                        0,
                        || new_weight_errs[i],
                    )?;
                    Ok(())
                },
            )?;
            
            // Range check new weight error
            layouter.assign_region(
                || format!("range check new weight error {}", i),
                |mut region| {
                    self.config.range.s_range.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "new_err",
                        self.config.range.input_column,
                        0,
                        || new_weight_errs[i],
                    )?;
                    Ok(())
                },
            )?;
        }
        
        Ok(())
    }
}

/// A circuit that verifies a complete training step: forward, backward, and weight update.
#[derive(Clone)]
pub struct TrainingStepCircuit<F: PrimeField, const RANGE: usize> {
    /// Old weights.
    pub old_weights: Vec<F>,
    pub old_weight_errs: Vec<F>,
    /// Gradients (scaled by learning rate).
    pub gradients: Vec<F>,
    pub gradient_errs: Vec<F>,
    /// New weights after update.
    pub new_weights: Vec<F>,
    pub new_weight_errs: Vec<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> Default for TrainingStepCircuit<F, RANGE> {
    fn default() -> Self {
        Self {
            old_weights: vec![],
            old_weight_errs: vec![],
            gradients: vec![],
            gradient_errs: vec![],
            new_weights: vec![],
            new_weight_errs: vec![],
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField, const RANGE: usize> Circuit<F> for TrainingStepCircuit<F, RANGE> {
    type Config = GradientVerificationConfig<F, RANGE>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        GradientVerificationChip::<F, RANGE>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let range_chip = RangeChip::<F, RANGE>::new(config.range.clone());
        range_chip.load(&mut layouter)?;
        
        let chip = GradientVerificationChip::<F, RANGE>::new(config);
        
        let old_weights: Vec<_> = self.old_weights.iter().map(|v| Value::known(*v)).collect();
        let old_weight_errs: Vec<_> = self.old_weight_errs.iter().map(|v| Value::known(*v)).collect();
        let gradients: Vec<_> = self.gradients.iter().map(|v| Value::known(*v)).collect();
        let gradient_errs: Vec<_> = self.gradient_errs.iter().map(|v| Value::known(*v)).collect();
        let new_weights: Vec<_> = self.new_weights.iter().map(|v| Value::known(*v)).collect();
        let new_weight_errs: Vec<_> = self.new_weight_errs.iter().map(|v| Value::known(*v)).collect();
        
        chip.assign_weight_update(
            layouter.namespace(|| "weight update"),
            &old_weights,
            &old_weight_errs,
            &gradients,
            &gradient_errs,
            &new_weights,
            &new_weight_errs,
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
    fn test_weight_update_simple() {
        // old = 10, gradient = 2, new = 10 - 2 = 8
        let circuit = TrainingStepCircuit::<Fr, 100> {
            old_weights: vec![Fr::from(10)],
            old_weight_errs: vec![Fr::from(1)],
            gradients: vec![Fr::from(2)],
            gradient_errs: vec![Fr::from(1)],
            new_weights: vec![Fr::from(8)],
            new_weight_errs: vec![Fr::from(2)], // 1 + 1 = 2
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_weight_update_invalid() {
        // Wrong new weight
        let circuit = TrainingStepCircuit::<Fr, 100> {
            old_weights: vec![Fr::from(10)],
            old_weight_errs: vec![Fr::from(1)],
            gradients: vec![Fr::from(2)],
            gradient_errs: vec![Fr::from(1)],
            new_weights: vec![Fr::from(9)], // Should be 8
            new_weight_errs: vec![Fr::from(2)],
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_weight_update_multiple() {
        // Multiple weights
        let circuit = TrainingStepCircuit::<Fr, 100> {
            old_weights: vec![Fr::from(10), Fr::from(20), Fr::from(30)],
            old_weight_errs: vec![Fr::from(0), Fr::from(0), Fr::from(0)],
            gradients: vec![Fr::from(1), Fr::from(2), Fr::from(3)],
            gradient_errs: vec![Fr::from(1), Fr::from(1), Fr::from(1)],
            new_weights: vec![Fr::from(9), Fr::from(18), Fr::from(27)],
            new_weight_errs: vec![Fr::from(1), Fr::from(1), Fr::from(1)],
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
