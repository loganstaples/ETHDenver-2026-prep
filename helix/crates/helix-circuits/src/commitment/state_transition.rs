//! Proves valid state transition: M_new = M_old - lr * G

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error, ErrorFront},
};
use halo2curves::ff::PrimeField;
use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::approximate::bounded_add::{BoundedAddChip, BoundedAddConfig};
use crate::approximate::bounded_mul::{BoundedMulChip, BoundedMulConfig};

#[derive(Clone, Debug)]
pub struct StateTransitionConfig<F: PrimeField, const RANGE: usize> {
    pub add: BoundedAddConfig<F, RANGE>,
    pub mul: BoundedMulConfig<F, RANGE>,
}

pub struct StateTransitionChip<F: PrimeField, const RANGE: usize> {
    pub config: StateTransitionConfig<F, RANGE>,
}

impl<F: PrimeField, const RANGE: usize> StateTransitionChip<F, RANGE> {
    pub fn new(config: StateTransitionConfig<F, RANGE>) -> Self {
        Self { config }
    }

    pub fn assign_transition(
        &self,
        mut layouter: impl Layouter<F>,
        old_w: Value<F>,
        old_err: Value<F>,
        grad: Value<F>,
        grad_err: Value<F>,
        lr: Value<F>,
        new_w: Value<F>,
        new_err: Value<F>,
    ) -> Result<(), ErrorFront> {
        let add_chip = BoundedAddChip::new(self.config.add.clone());
        let mul_chip = BoundedMulChip::new(self.config.mul.clone());

        // Step 1: calc update = lr * grad
        let update_val = lr * grad;
        let update_err = lr * grad_err; 
        
        let zero = Value::known(F::ZERO);
        mul_chip.assign(
            layouter.namespace(|| "calc update"), 
            lr, zero, 
            grad, grad_err, 
            update_val, update_err
        )?;

        // Step 2: neg_update = update * (-1)
        // We use a constant -1
        let minus_one = Value::known(-F::ONE);
        let neg_update_val = update_val * minus_one;
        let neg_update_err = update_err; // Error magnitude stays same

        mul_chip.assign(
            layouter.namespace(|| "negate update"),
            update_val, update_err,
            minus_one, zero,
            neg_update_val, neg_update_err
        )?;

        // Step 3: new_w = old_w + neg_update
        // We verify: old_w + neg_update = new_w
        add_chip.assign(
            layouter.namespace(|| "apply update"),
            old_w, old_err,
            neg_update_val, neg_update_err,
            new_w, new_err
        )?;

        Ok(())
    }
}
