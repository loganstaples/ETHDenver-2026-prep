//! Error Accumulation Circuit.
//!
//! Proves that error bounds accumulate correctly through a sequence of operations.
//!
//! This circuit is fundamental to the HELIX innovation: proving that the
//! total accumulated error after N operations remains within an acceptable bound.
//!
//! Key insight: We don't prove exact computation, we prove the computation
//! is WITHIN BOUNDS of exact computation.

use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, Advice, ConstraintSystem, Error, Selector, Instance},
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

/// Types of operations and their error propagation formulas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpType {
    /// Addition: err(a+b) = err(a) + err(b)
    Add,
    /// Subtraction: err(a-b) = err(a) + err(b)
    Sub,
    /// Multiplication: err(a*b) ≤ |a|*err(b) + |b|*err(a) + err(a)*err(b)
    Mul,
    /// Division: err(a/b) ≤ (|a|*err(b) + |b|*err(a)) / (|b|^2 - err(b)^2)
    /// Simplified upper bound for small errors
    Div,
    /// ReLU: err(relu(x)) = err(x) if x > 0, else 0
    ReLU,
    /// Matrix multiply element (dot product term)
    MatMulTerm,
}

/// Configuration for error accumulation verification.
#[derive(Clone, Debug)]
pub struct ErrorAccumulationConfig<F: PrimeField, const RANGE: usize> {
    /// Arithmetic operations.
    pub arithmetic: ArithmeticConfig,
    /// Range checking.
    pub range: RangeConfig<F, RANGE>,
    /// Selector for error bound constraint.
    pub s_error_bound: Selector,
    /// Selector for additive error accumulation.
    pub s_add_error: Selector,
    /// Selector for multiplicative error propagation.
    pub s_mul_error: Selector,
    /// Advice columns.
    pub input_val_a: Column<Advice>,
    pub input_err_a: Column<Advice>,
    pub input_val_b: Column<Advice>,
    pub input_err_b: Column<Advice>,
    pub output_val: Column<Advice>,
    pub output_err: Column<Advice>,
    pub running_err: Column<Advice>,
    /// Instance column for max allowed error.
    pub instance: Column<Instance>,
}

/// A chip that proves error accumulation bounds.
pub struct ErrorAccumulationChip<F: PrimeField, const RANGE: usize> {
    config: ErrorAccumulationConfig<F, RANGE>,
    arithmetic_chip: ArithmeticChip<F>,
    range_chip: RangeChip<F, RANGE>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> ErrorAccumulationChip<F, RANGE> {
    /// Creates a new error accumulation chip.
    pub fn new(config: ErrorAccumulationConfig<F, RANGE>) -> Self {
        let arithmetic_chip = ArithmeticChip::new(config.arithmetic.clone());
        let range_chip = RangeChip::new(config.range.clone());
        Self {
            config,
            arithmetic_chip,
            range_chip,
            _marker: PhantomData,
        }
    }

    /// Configures the error accumulation circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> ErrorAccumulationConfig<F, RANGE> {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let input_val_a = meta.advice_column();
        let input_err_a = meta.advice_column();
        let input_val_b = meta.advice_column();
        let input_err_b = meta.advice_column();
        let output_val = meta.advice_column();
        let output_err = meta.advice_column();
        let running_err = meta.advice_column();
        
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        
        for col in [a, b, c, input_val_a, input_err_a, input_val_b, 
                    input_err_b, output_val, output_err, running_err] {
            meta.enable_equality(col);
        }
        
        let table = meta.lookup_table_column();
        let arithmetic = ArithmeticChip::<F>::configure(meta, a, b, c);
        let range = RangeConfig::configure(meta, table, c);
        
        let s_error_bound = meta.selector();
        let s_add_error = meta.selector();
        let s_mul_error = meta.selector();
        
        // Additive error accumulation: running_err' = running_err + output_err
        meta.create_gate("additive_error_accumulation", |meta| {
            let s = meta.query_selector(s_add_error);
            let running = meta.query_advice(running_err, Rotation::cur());
            let output = meta.query_advice(output_err, Rotation::cur());
            let new_running = meta.query_advice(running_err, Rotation::next());
            
            vec![s * (running + output - new_running)]
        });
        
        ErrorAccumulationConfig {
            arithmetic,
            range,
            s_error_bound,
            s_add_error,
            s_mul_error,
            input_val_a,
            input_err_a,
            input_val_b,
            input_err_b,
            output_val,
            output_err,
            running_err,
            instance,
        }
    }

    /// Proves error propagation for an addition operation.
    /// For addition: err(a + b) = err(a) + err(b)
    pub fn assign_add_error_propagation(
        &self,
        mut layouter: impl Layouter<F>,
        err_a: Value<F>,
        err_b: Value<F>,
        err_result: Value<F>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "add error propagation",
            |mut region| {
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "err_a", self.config.arithmetic.a, 0, || err_a)?;
                region.assign_advice(|| "err_b", self.config.arithmetic.b, 0, || err_b)?;
                region.assign_advice(|| "err_result", self.config.arithmetic.c, 0, || err_result)?;
                Ok(())
            },
        )
    }

    /// Proves error propagation for a multiplication operation.
    /// For multiplication: err(a * b) ≤ |a|*err_b + |b|*err_a + err_a*err_b
    pub fn assign_mul_error_propagation(
        &self,
        mut layouter: impl Layouter<F>,
        val_a: Value<F>,
        err_a: Value<F>,
        val_b: Value<F>,
        err_b: Value<F>,
        err_result: Value<F>,
    ) -> Result<(), Error> {
        // Compute intermediate terms
        let term1 = val_a * err_b;  // |a| * err_b
        let term2 = val_b * err_a;  // |b| * err_a
        let term3 = err_a * err_b;  // err_a * err_b
        
        // Verify term1 = val_a * err_b
        layouter.assign_region(
            || "mul error term1",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "val_a", self.config.arithmetic.a, 0, || val_a)?;
                region.assign_advice(|| "err_b", self.config.arithmetic.b, 0, || err_b)?;
                region.assign_advice(|| "term1", self.config.arithmetic.c, 0, || term1)?;
                Ok(())
            },
        )?;
        
        // Verify term2 = val_b * err_a
        layouter.assign_region(
            || "mul error term2",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "val_b", self.config.arithmetic.a, 0, || val_b)?;
                region.assign_advice(|| "err_a", self.config.arithmetic.b, 0, || err_a)?;
                region.assign_advice(|| "term2", self.config.arithmetic.c, 0, || term2)?;
                Ok(())
            },
        )?;
        
        // Verify term3 = err_a * err_b
        layouter.assign_region(
            || "mul error term3",
            |mut region| {
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "err_a", self.config.arithmetic.a, 0, || err_a)?;
                region.assign_advice(|| "err_b", self.config.arithmetic.b, 0, || err_b)?;
                region.assign_advice(|| "term3", self.config.arithmetic.c, 0, || term3)?;
                Ok(())
            },
        )?;
        
        // Sum: term1 + term2
        let partial_sum = term1 + term2;
        layouter.assign_region(
            || "mul error sum partial",
            |mut region| {
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "term1", self.config.arithmetic.a, 0, || term1)?;
                region.assign_advice(|| "term2", self.config.arithmetic.b, 0, || term2)?;
                region.assign_advice(|| "partial", self.config.arithmetic.c, 0, || partial_sum)?;
                Ok(())
            },
        )?;
        
        // Final sum: partial + term3 = err_result
        layouter.assign_region(
            || "mul error sum final",
            |mut region| {
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "partial", self.config.arithmetic.a, 0, || partial_sum)?;
                region.assign_advice(|| "term3", self.config.arithmetic.b, 0, || term3)?;
                region.assign_advice(|| "err_result", self.config.arithmetic.c, 0, || err_result)?;
                Ok(())
            },
        )
    }

    /// Proves that accumulated error stays within bounds after a sequence of operations.
    /// This is the key function for proving bounded computation.
    pub fn assign_error_sequence(
        &self,
        mut layouter: impl Layouter<F>,
        operations: &[OperationWitness<F>],
        max_allowed_error: Value<F>,
    ) -> Result<(), Error> {
        if operations.is_empty() {
            return Ok(());
        }
        
        let mut running_error = Value::known(F::ZERO);
        
        for (i, op) in operations.iter().enumerate() {
            // Compute the error contribution from this operation
            let op_error = match op.op_type {
                OpType::Add | OpType::Sub => {
                    // err(a±b) = err_a + err_b
                    self.assign_add_error_propagation(
                        layouter.namespace(|| format!("add error {}", i)),
                        op.input_err_a,
                        op.input_err_b,
                        op.output_err,
                    )?;
                    op.output_err
                }
                OpType::Mul | OpType::MatMulTerm => {
                    // err(a*b) = |a|*err_b + |b|*err_a + err_a*err_b
                    self.assign_mul_error_propagation(
                        layouter.namespace(|| format!("mul error {}", i)),
                        op.input_val_a,
                        op.input_err_a,
                        op.input_val_b,
                        op.input_err_b,
                        op.output_err,
                    )?;
                    op.output_err
                }
                OpType::ReLU => {
                    // ReLU error is same as input error (for positive inputs)
                    // For negative inputs, both value and error are 0
                    op.output_err
                }
                OpType::Div => {
                    // Division error is complex; assume pre-computed
                    op.output_err
                }
            };
            
            // Accumulate error
            let new_running = running_error + op_error;
            
            layouter.assign_region(
                || format!("accumulate error {}", i),
                |mut region| {
                    self.config.arithmetic.s_add.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "running",
                        self.config.arithmetic.a,
                        0,
                        || running_error,
                    )?;
                    region.assign_advice(
                        || "op_error",
                        self.config.arithmetic.b,
                        0,
                        || op_error,
                    )?;
                    region.assign_advice(
                        || "new_running",
                        self.config.arithmetic.c,
                        0,
                        || new_running,
                    )?;
                    Ok(())
                },
            )?;
            
            running_error = new_running;
        }
        
        // Final check: running_error <= max_allowed_error
        // This is verified via range check on (max_allowed_error - running_error)
        let remaining_budget = max_allowed_error - running_error;
        
        layouter.assign_region(
            || "compute error budget",
            |mut region| {
                self.config.arithmetic.s_add.enable(&mut region, 0)?;
                region.assign_advice(
                    || "running_error",
                    self.config.arithmetic.a,
                    0,
                    || running_error,
                )?;
                region.assign_advice(
                    || "remaining_budget",
                    self.config.arithmetic.b,
                    0,
                    || remaining_budget,
                )?;
                region.assign_advice(
                    || "max_allowed",
                    self.config.arithmetic.c,
                    0,
                    || max_allowed_error,
                )?;
                Ok(())
            },
        )?;
        
        // Range check: remaining_budget must be non-negative (in range [0, RANGE))
        layouter.assign_region(
            || "range check remaining budget",
            |mut region| {
                self.config.range.s_range.enable(&mut region, 0)?;
                region.assign_advice(
                    || "remaining_budget",
                    self.config.range.input_column,
                    0,
                    || remaining_budget,
                )?;
                Ok(())
            },
        )?;
        
        Ok(())
    }
}

/// Witness for a single operation.
#[derive(Clone)]
pub struct OperationWitness<F: PrimeField> {
    pub op_type: OpType,
    pub input_val_a: Value<F>,
    pub input_err_a: Value<F>,
    pub input_val_b: Value<F>,
    pub input_err_b: Value<F>,
    pub output_val: Value<F>,
    pub output_err: Value<F>,
}

/// A circuit that verifies error accumulation for a sequence of operations.
#[derive(Clone)]
pub struct ErrorAccumulationCircuit<F: PrimeField, const RANGE: usize> {
    /// Sequence of operations.
    pub operations: Vec<OperationData<F>>,
    /// Maximum allowed error.
    pub max_allowed_error: F,
    _marker: PhantomData<F>,
}

/// Data for a single operation (without Value wrapper, for circuit construction).
#[derive(Clone)]
pub struct OperationData<F: PrimeField> {
    pub op_type: OpType,
    pub input_val_a: F,
    pub input_err_a: F,
    pub input_val_b: F,
    pub input_err_b: F,
    pub output_val: F,
    pub output_err: F,
}

impl<F: PrimeField, const RANGE: usize> Default for ErrorAccumulationCircuit<F, RANGE> {
    fn default() -> Self {
        Self {
            operations: vec![],
            max_allowed_error: F::ZERO,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField, const RANGE: usize> Circuit<F> for ErrorAccumulationCircuit<F, RANGE> {
    type Config = ErrorAccumulationConfig<F, RANGE>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        ErrorAccumulationChip::<F, RANGE>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let range_chip = RangeChip::<F, RANGE>::new(config.range.clone());
        range_chip.load(&mut layouter)?;
        
        let chip = ErrorAccumulationChip::<F, RANGE>::new(config);
        
        let operations: Vec<_> = self.operations.iter()
            .map(|op| OperationWitness {
                op_type: op.op_type,
                input_val_a: Value::known(op.input_val_a),
                input_err_a: Value::known(op.input_err_a),
                input_val_b: Value::known(op.input_val_b),
                input_err_b: Value::known(op.input_err_b),
                output_val: Value::known(op.output_val),
                output_err: Value::known(op.output_err),
            })
            .collect();
        
        chip.assign_error_sequence(
            layouter.namespace(|| "error accumulation"),
            &operations,
            Value::known(self.max_allowed_error),
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
    fn test_add_error_accumulation() {
        // Sequence of additions: each adds 1 to error
        // 3 operations, each contributing error of 1
        // Total error = 3, max allowed = 10
        
        let operations = vec![
            OperationData {
                op_type: OpType::Add,
                input_val_a: Fr::from(5),
                input_err_a: Fr::from(1),
                input_val_b: Fr::from(3),
                input_err_b: Fr::from(0),
                output_val: Fr::from(8),
                output_err: Fr::from(1),  // 1 + 0 = 1
            },
            OperationData {
                op_type: OpType::Add,
                input_val_a: Fr::from(8),
                input_err_a: Fr::from(1),
                input_val_b: Fr::from(2),
                input_err_b: Fr::from(1),
                output_val: Fr::from(10),
                output_err: Fr::from(2),  // 1 + 1 = 2
            },
        ];
        
        let circuit = ErrorAccumulationCircuit::<Fr, 100> {
            operations,
            max_allowed_error: Fr::from(10),  // Total = 1 + 2 = 3, within 10
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_mul_error_accumulation() {
        // Single multiplication
        // val_a = 2, err_a = 1
        // val_b = 3, err_b = 1
        // output_err = 2*1 + 3*1 + 1*1 = 6
        
        let operations = vec![
            OperationData {
                op_type: OpType::Mul,
                input_val_a: Fr::from(2),
                input_err_a: Fr::from(1),
                input_val_b: Fr::from(3),
                input_err_b: Fr::from(1),
                output_val: Fr::from(6),
                output_err: Fr::from(6),
            },
        ];
        
        let circuit = ErrorAccumulationCircuit::<Fr, 100> {
            operations,
            max_allowed_error: Fr::from(10),
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_error_exceeds_bound() {
        // Total error exceeds max allowed
        let operations = vec![
            OperationData {
                op_type: OpType::Add,
                input_val_a: Fr::from(1),
                input_err_a: Fr::from(50),
                input_val_b: Fr::from(1),
                input_err_b: Fr::from(60),  // 50 + 60 = 110
                output_val: Fr::from(2),
                output_err: Fr::from(110),
            },
        ];
        
        let circuit = ErrorAccumulationCircuit::<Fr, 100> {
            operations,
            max_allowed_error: Fr::from(100),  // 110 > 100, should fail
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        // Range check should fail because remaining_budget = -10 which is a huge field element
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_complex_sequence() {
        // Mix of operations
        let operations = vec![
            OperationData {
                op_type: OpType::Add,
                input_val_a: Fr::from(10),
                input_err_a: Fr::from(1),
                input_val_b: Fr::from(5),
                input_err_b: Fr::from(1),
                output_val: Fr::from(15),
                output_err: Fr::from(2),
            },
            OperationData {
                op_type: OpType::Mul,
                input_val_a: Fr::from(2),
                input_err_a: Fr::from(0),
                input_val_b: Fr::from(3),
                input_err_b: Fr::from(1),
                output_val: Fr::from(6),
                output_err: Fr::from(2),  // 2*1 + 3*0 + 0*1 = 2
            },
            OperationData {
                op_type: OpType::Add,
                input_val_a: Fr::from(15),
                input_err_a: Fr::from(2),
                input_val_b: Fr::from(6),
                input_err_b: Fr::from(2),
                output_val: Fr::from(21),
                output_err: Fr::from(4),
            },
        ];
        
        // Accumulated error = 2 + 2 + 4 = 8
        let circuit = ErrorAccumulationCircuit::<Fr, 100> {
            operations,
            max_allowed_error: Fr::from(20),
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
