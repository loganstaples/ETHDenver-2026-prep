//! Range check gadget using a lookup table.
//!
//! Constrains a value to be within [0, RANGE).

use halo2_proofs::{
    circuit::{Layouter, Region, Value},
    plonk::{ConstraintSystem, Error, ErrorFront, Expression, TableColumn, Selector},
    poly::Rotation,
};
use std::marker::PhantomData;
use halo2_proofs::arithmetic::Field;
use halo2curves::ff::PrimeField; // Need PrimeField for From<u64>

/// Configuration for the Range Gadget.
#[derive(Clone, Debug)]
pub struct RangeConfig<F: PrimeField, const RANGE: usize> {
    /// The lookup table column containing values [0..RANGE).
    pub table_column: TableColumn,
    /// The advice column being range-constrained.
    pub input_column: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    /// Selector to enable range check.
    pub s_range: halo2_proofs::plonk::Selector,
    /// Phantom data for the field.
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> RangeConfig<F, RANGE> {
    /// Configures the range gadget.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        range_column: TableColumn,
        input_column: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    ) -> Self {
        let s_range = meta.complex_selector();
        
        meta.lookup("range_check", |meta| {
            let s = meta.query_selector(s_range);
            let value = meta.query_advice(input_column, Rotation::cur());
            // If selector is enabled, check value. If disabled, check 0 (which is in table).
            vec![(s * value, range_column)]
        });

        Self {
            table_column: range_column,
            input_column,
            s_range,
            _marker: PhantomData,
        }
    }
}

/// A chip that provides range checks.
pub struct RangeChip<F: PrimeField, const RANGE: usize> {
    config: RangeConfig<F, RANGE>,
}

impl<F: PrimeField, const RANGE: usize> RangeChip<F, RANGE> {
    /// Creates a new range chip.
    pub fn new(config: RangeConfig<F, RANGE>) -> Self {
        Self { config }
    }

    /// Loads the lookup table with values [0, RANGE).
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "range table",
            |mut table| {
                for i in 0..RANGE {
                    table.assign_cell(
                        || format!("range table row {}", i),
                        self.config.table_column,
                        i,
                        || Value::known(F::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }
}
