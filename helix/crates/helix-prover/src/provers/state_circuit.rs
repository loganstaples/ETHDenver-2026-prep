use helix_circuits::halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::commitment::state_transition::{
    StateTransitionChip, StateTransitionConfig,
};
use helix_circuits::gadgets::{
    arithmetic::ArithmeticChip,
    range::{RangeChip, RangeConfig},
};

#[derive(Clone)]
pub struct StateCircuitConfig {
    pub st_config: StateTransitionConfig<Fr, 100>, // Range 100 for demo
}

#[derive(Clone, Default)]
pub struct StateTransitionCircuit {
    pub old_w: Option<Fr>,
    pub old_err: Option<Fr>,
    pub grad: Option<Fr>,
    pub grad_err: Option<Fr>,
    pub lr: Option<Fr>,
    pub new_w: Option<Fr>,
    pub new_err: Option<Fr>,
}

impl Circuit<Fr> for StateTransitionCircuit {
    type Config = StateCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let table = meta.lookup_table_column();
        
        let arith_config = ArithmeticChip::configure(meta, a, b, c);
        let range_config = RangeConfig::configure(meta, table, c);
        
        let st_config = StateTransitionConfig {
            add: helix_circuits::approximate::bounded_add::BoundedAddConfig {
                arithmetic: arith_config.clone(),
                range: range_config.clone(),
            },
            mul: helix_circuits::approximate::bounded_mul::BoundedMulConfig {
                arithmetic: arith_config,
                range: range_config,
            },
        };

        StateCircuitConfig { st_config }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let chip = StateTransitionChip::new(config.st_config);
        
        // Load table (hacky access via internal chips, ideally exposed)
        // We know StateTransitionChip uses BoundedAdd which uses RangeChip.
        // Let's reload range table here.
        // We need a RangeChip instance.
        let range_chip = RangeChip::<Fr, 100>::new(chip.config.add.range.clone());
        range_chip.load(&mut layouter)?;

        let old_w = Value::known(self.old_w.unwrap_or(Fr::zero()));
        let old_err = Value::known(self.old_err.unwrap_or(Fr::zero()));
        let grad = Value::known(self.grad.unwrap_or(Fr::zero()));
        let grad_err = Value::known(self.grad_err.unwrap_or(Fr::zero()));
        let lr = Value::known(self.lr.unwrap_or(Fr::zero()));
        let new_w = Value::known(self.new_w.unwrap_or(Fr::zero()));
        let new_err = Value::known(self.new_err.unwrap_or(Fr::zero()));

        chip.assign_transition(
            layouter,
            old_w, old_err,
            grad, grad_err,
            lr,
            new_w, new_err,
        )?;

        Ok(())
    }
}
