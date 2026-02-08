//! Basic arithmetic gadget.

use halo2_proofs::{
    arithmetic::Field,
    plonk::{Advice, Column, ConstraintSystem, Selector},
    poly::Rotation,
};
use std::marker::PhantomData;

/// Configuration for the arithmetic gadget.
#[derive(Clone, Debug)]
pub struct ArithmeticConfig {
    /// Column for left operand.
    pub a: Column<Advice>,
    /// Column for right operand.
    pub b: Column<Advice>,
    /// Column for output.
    pub c: Column<Advice>,
    /// Selector for multiplication (a * b = c).
    pub s_mul: Selector,
    /// Selector for addition (a + b = c).
    pub s_add: Selector,
}

/// A chip providing arithmetic operations.
pub struct ArithmeticChip<F: Field> {
    pub config: ArithmeticConfig,
    _marker: PhantomData<F>,
}

impl<F: Field> ArithmeticChip<F> {
    /// Configures the arithmetic chip.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
    ) -> ArithmeticConfig {
        let s_mul = meta.selector();
        let s_add = meta.selector();

        meta.create_gate("mul", |meta| {
            let lhs = meta.query_advice(a, Rotation::cur());
            let rhs = meta.query_advice(b, Rotation::cur());
            let out = meta.query_advice(c, Rotation::cur());
            let s_mul = meta.query_selector(s_mul);

            vec![s_mul * (lhs * rhs - out)]
        });

        meta.create_gate("add", |meta| {
            let lhs = meta.query_advice(a, Rotation::cur());
            let rhs = meta.query_advice(b, Rotation::cur());
            let out = meta.query_advice(c, Rotation::cur());
            let s_add = meta.query_selector(s_add);

            vec![s_add * (lhs + rhs - out)]
        });

        ArithmeticConfig {
            a,
            b,
            c,
            s_mul,
            s_add,
        }
    }

    /// Creates a new arithmetic chip.
    pub fn new(config: ArithmeticConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }
}
