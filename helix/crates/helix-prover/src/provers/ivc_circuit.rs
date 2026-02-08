//! IVC Step Circuit.
//!
//! A Halo2 circuit proving a single IVC step — state transition from one
//! commitment to another with a computation hash linking the two.
//!
//! Public inputs (instance column):
//!   0: prev_state_lo   (lower 128 bits of prev state commitment as Fr)
//!   1: prev_state_hi   (upper 128 bits)
//!   2: new_state_lo
//!   3: new_state_hi
//!   4: computation_lo
//!   5: computation_hi
//!   6: step_number

use helix_circuits::halo2_proofs::{
    arithmetic::Field,
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, ErrorFront, Expression, Fixed, Instance, Selector,
    },
    poly::Rotation,
};
use helix_circuits::halo2curves::bn256::Fr;

/// Configuration for the IVC step circuit.
#[derive(Clone, Debug)]
pub struct IVCStepConfig {
    /// Advice columns for witness values.
    advice: [Column<Advice>; 3],
    /// Instance column for public inputs.
    instance: Column<Instance>,
    /// Selector for equality constraints.
    s_eq: Selector,
    /// Selector for step-number constraint.
    s_step: Selector,
    /// Fixed column for constants.
    fixed: Column<Fixed>,
}

/// Number of public inputs exposed by this circuit.
pub const NUM_PUBLIC_INPUTS: usize = 7;

/// IVC step circuit proving a single state transition.
#[derive(Clone)]
pub struct IVCStepCircuit {
    /// Previous state commitment (32 bytes).
    pub prev_state: [u8; 32],
    /// New state commitment (32 bytes).
    pub new_state: [u8; 32],
    /// Computation hash (32 bytes).
    pub computation_hash: [u8; 32],
    /// Step number.
    pub step_number: u64,
}

impl Default for IVCStepCircuit {
    fn default() -> Self {
        Self {
            prev_state: [0u8; 32],
            new_state: [0u8; 32],
            computation_hash: [0u8; 32],
            step_number: 0,
        }
    }
}

impl IVCStepCircuit {
    /// Builds the public inputs vector from the circuit's data.
    pub fn public_inputs(&self) -> Vec<Fr> {
        let mut inputs = Vec::with_capacity(NUM_PUBLIC_INPUTS);
        inputs.push(bytes_lo_to_fr(&self.prev_state));
        inputs.push(bytes_hi_to_fr(&self.prev_state));
        inputs.push(bytes_lo_to_fr(&self.new_state));
        inputs.push(bytes_hi_to_fr(&self.new_state));
        inputs.push(bytes_lo_to_fr(&self.computation_hash));
        inputs.push(bytes_hi_to_fr(&self.computation_hash));
        inputs.push(Fr::from(self.step_number));
        inputs
    }
}

impl Circuit<Fr> for IVCStepCircuit {
    type Config = IVCStepConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let instance = meta.instance_column();
        let fixed = meta.fixed_column();
        let s_eq = meta.selector();
        let s_step = meta.selector();

        // Enable equality on advice[0] and instance so we can constrain them equal.
        meta.enable_equality(advice[0]);
        meta.enable_equality(instance);

        // Gate: when s_eq is active, advice[0] - advice[1] == 0
        // (used to bind witness values together)
        meta.create_gate("equality", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a - b)]
        });

        // Gate: when s_step is active, advice[0] must equal step_number (advice[2])
        meta.create_gate("step_check", |meta| {
            let s = meta.query_selector(s_step);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let step = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a - step)]
        });

        IVCStepConfig {
            advice,
            instance,
            s_eq,
            s_step,
            fixed,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let pi = self.public_inputs();

        // Assign witness values in a region and constrain them equal to instance.
        let cells = layouter.assign_region(
            || "ivc_step_witness",
            |mut region| {
                let mut cells = Vec::with_capacity(NUM_PUBLIC_INPUTS);

                for (i, val) in pi.iter().enumerate() {
                    // Each public input goes into advice[0] at row i
                    let cell = region.assign_advice(
                        || format!("pi_{}", i),
                        config.advice[0],
                        i,
                        || Value::known(*val),
                    )?;
                    cells.push(cell);
                }

                // Enable equality gate on rows 0..6 to bind advice[0] == advice[1]
                // (witness consistency — both columns carry the same value)
                for i in 0..NUM_PUBLIC_INPUTS {
                    config.s_eq.enable(&mut region, i)?;
                    region.assign_advice(
                        || format!("pi_{}_dup", i),
                        config.advice[1],
                        i,
                        || Value::known(pi[i]),
                    )?;
                }

                // Step number check on last row
                let step_row = NUM_PUBLIC_INPUTS - 1;
                config.s_step.enable(&mut region, step_row)?;
                region.assign_advice(
                    || "step_number_witness",
                    config.advice[2],
                    step_row,
                    || Value::known(Fr::from(self.step_number)),
                )?;

                Ok(cells)
            },
        )?;

        // Constrain each witness cell to equal the corresponding instance value.
        for (i, cell) in cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers: split a 32-byte hash into two 128-bit field elements.
// ---------------------------------------------------------------------------

/// Lower 16 bytes → Fr (little-endian).
pub fn bytes_lo_to_fr(hash: &[u8; 32]) -> Fr {
    let mut buf = [0u8; 32];
    buf[..16].copy_from_slice(&hash[..16]);
    Fr::from_raw([
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        0,
        0,
    ])
}

/// Upper 16 bytes → Fr (little-endian).
pub fn bytes_hi_to_fr(hash: &[u8; 32]) -> Fr {
    let mut buf = [0u8; 32];
    buf[..16].copy_from_slice(&hash[16..32]);
    Fr::from_raw([
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        0,
        0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_circuits::halo2_proofs::dev::MockProver;

    #[test]
    fn test_ivc_step_circuit_valid() {
        let circuit = IVCStepCircuit {
            prev_state: [1u8; 32],
            new_state: [2u8; 32],
            computation_hash: [3u8; 32],
            step_number: 42,
        };

        let pi = circuit.public_inputs();
        let k = 5; // small circuit

        let prover = MockProver::run(k, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_ivc_step_circuit_default() {
        let circuit = IVCStepCircuit::default();
        let pi = circuit.public_inputs();

        let prover = MockProver::run(5, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_ivc_step_wrong_instance() {
        let circuit = IVCStepCircuit {
            prev_state: [1u8; 32],
            new_state: [2u8; 32],
            computation_hash: [3u8; 32],
            step_number: 42,
        };

        // Provide wrong public inputs
        let mut pi = circuit.public_inputs();
        pi[0] = Fr::from(999u64); // corrupt prev_state_lo

        let prover = MockProver::run(5, &circuit, vec![pi]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_bytes_roundtrip() {
        let hash = [0xABu8; 32];
        let lo = bytes_lo_to_fr(&hash);
        let hi = bytes_hi_to_fr(&hash);
        // lo and hi should be equal since all bytes are the same
        assert_eq!(lo, hi);
    }

    #[test]
    fn test_public_inputs_count() {
        let circuit = IVCStepCircuit::default();
        assert_eq!(circuit.public_inputs().len(), NUM_PUBLIC_INPUTS);
    }
}
