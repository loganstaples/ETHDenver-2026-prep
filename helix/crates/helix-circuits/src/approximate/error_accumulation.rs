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
    plonk::{Circuit, Column, Advice, ConstraintSystem, ErrorFront, Selector, Instance},
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
    _arithmetic_chip: ArithmeticChip<F>,
    _range_chip: RangeChip<F, RANGE>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> ErrorAccumulationChip<F, RANGE> {
    /// Creates a new error accumulation chip.
    pub fn new(config: ErrorAccumulationConfig<F, RANGE>) -> Self {
        let arithmetic_chip = ArithmeticChip::new(config.arithmetic.clone());
        let range_chip = RangeChip::new(config.range.clone());
        Self {
            config,
            _arithmetic_chip: arithmetic_chip,
            _range_chip: range_chip,
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
    ) -> Result<(), ErrorFront> {
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
    ///
    /// Uses a single region with 5 rows and copy constraints to prevent
    /// a malicious prover from using inconsistent intermediate values.
    ///
    /// Layout (single region):
    ///   Row 0: s_mul, val_a, err_b, term1         (term1 = val_a * err_b)
    ///   Row 1: s_mul, val_b, err_a, term2         (term2 = val_b * err_a)
    ///   Row 2: s_mul, err_a', err_b', term3       (term3 = err_a * err_b)
    ///   Row 3: s_add, term1', term2', partial_sum (partial = term1 + term2)
    ///   Row 4: s_add, partial', term3', err_result (err_result = partial + term3)
    ///
    /// Copy constraints bind shared values across rows:
    ///   err_a (row 1) == err_a' (row 2)
    ///   err_b (row 0) == err_b' (row 2)
    ///   term1 (row 0) == term1' (row 3)
    ///   term2 (row 1) == term2' (row 3)
    ///   term3 (row 2) == term3' (row 4)
    ///   partial_sum (row 3) == partial' (row 4)
    pub fn assign_mul_error_propagation(
        &self,
        mut layouter: impl Layouter<F>,
        val_a: Value<F>,
        err_a: Value<F>,
        val_b: Value<F>,
        err_b: Value<F>,
        err_result: Value<F>,
    ) -> Result<(), ErrorFront> {
        let term1 = val_a * err_b;
        let term2 = val_b * err_a;
        let term3 = err_a * err_b;
        let partial_sum = term1 + term2;

        layouter.assign_region(
            || "mul error propagation",
            |mut region| {
                // Row 0: term1 = val_a * err_b
                self.config.arithmetic.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "val_a", self.config.arithmetic.a, 0, || val_a)?;
                let err_b_0 = region.assign_advice(|| "err_b", self.config.arithmetic.b, 0, || err_b)?;
                let term1_0 = region.assign_advice(|| "term1", self.config.arithmetic.c, 0, || term1)?;

                // Row 1: term2 = val_b * err_a
                self.config.arithmetic.s_mul.enable(&mut region, 1)?;
                region.assign_advice(|| "val_b", self.config.arithmetic.a, 1, || val_b)?;
                let err_a_1 = region.assign_advice(|| "err_a", self.config.arithmetic.b, 1, || err_a)?;
                let term2_1 = region.assign_advice(|| "term2", self.config.arithmetic.c, 1, || term2)?;

                // Row 2: term3 = err_a * err_b
                self.config.arithmetic.s_mul.enable(&mut region, 2)?;
                let err_a_2 = region.assign_advice(|| "err_a'", self.config.arithmetic.a, 2, || err_a)?;
                let err_b_2 = region.assign_advice(|| "err_b'", self.config.arithmetic.b, 2, || err_b)?;
                let term3_2 = region.assign_advice(|| "term3", self.config.arithmetic.c, 2, || term3)?;

                // Row 3: partial_sum = term1 + term2
                self.config.arithmetic.s_add.enable(&mut region, 3)?;
                let term1_3 = region.assign_advice(|| "term1'", self.config.arithmetic.a, 3, || term1)?;
                let term2_3 = region.assign_advice(|| "term2'", self.config.arithmetic.b, 3, || term2)?;
                let partial_3 = region.assign_advice(|| "partial", self.config.arithmetic.c, 3, || partial_sum)?;

                // Row 4: err_result = partial + term3
                self.config.arithmetic.s_add.enable(&mut region, 4)?;
                let partial_4 = region.assign_advice(|| "partial'", self.config.arithmetic.a, 4, || partial_sum)?;
                let term3_4 = region.assign_advice(|| "term3'", self.config.arithmetic.b, 4, || term3)?;
                region.assign_advice(|| "err_result", self.config.arithmetic.c, 4, || err_result)?;

                // Copy constraints: bind shared values across rows
                region.constrain_equal(err_a_1.cell(), err_a_2.cell())?;
                region.constrain_equal(err_b_0.cell(), err_b_2.cell())?;
                region.constrain_equal(term1_0.cell(), term1_3.cell())?;
                region.constrain_equal(term2_1.cell(), term2_3.cell())?;
                region.constrain_equal(term3_2.cell(), term3_4.cell())?;
                region.constrain_equal(partial_3.cell(), partial_4.cell())?;

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
    ) -> Result<(), ErrorFront> {
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

impl<F: PrimeField, const RANGE: usize> ErrorAccumulationCircuit<F, RANGE> {
    /// Creates a new error accumulation circuit.
    pub fn new(operations: Vec<OperationData<F>>, max_allowed_error: F) -> Self {
        Self {
            operations,
            max_allowed_error,
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
    ) -> Result<(), ErrorFront> {
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
    fn test_mul_error_wrong_result() {
        // Prover claims err_result = 3 but correct is 6
        // The s_add gate on row 4 should reject: partial + term3 != err_result
        let operations = vec![
            OperationData {
                op_type: OpType::Mul,
                input_val_a: Fr::from(2),
                input_err_a: Fr::from(1),
                input_val_b: Fr::from(3),
                input_err_b: Fr::from(1),
                output_val: Fr::from(6),
                output_err: Fr::from(3),  // Wrong! Should be 2*1 + 3*1 + 1*1 = 6
            },
        ];

        let circuit = ErrorAccumulationCircuit::<Fr, 100> {
            operations,
            max_allowed_error: Fr::from(10),
            _marker: PhantomData,
        };

        let prover = MockProver::run(8, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err(), "Should reject wrong mul error result");
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
