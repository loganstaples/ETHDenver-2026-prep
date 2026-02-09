//! Batch Proving Circuit.
//!
//! Wraps N instances of an inner circuit into a single proof, sharing the same SRS
//! and circuit configuration. Each sub-instance gets its own set of public inputs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    BatchCircuit<C>                               │
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐     │
//! │  │ Instance 0   │  │ Instance 1   │  ...  │ Instance N-1 │     │
//! │  │ (rows 0..R)  │  │ (rows R..2R) │       │              │     │
//! │  └──────────────┘  └──────────────┘       └──────────────┘     │
//! │                                                                  │
//! │  Public Inputs: [PI_0..PI_7, PI_8..PI_15, ..., PI_{N*8-1}]    │
//! │  Shared Config:  Same gates, columns, lookups                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let witnesses: Vec<MLTrainingStepV2Witness> = compute_witnesses(...);
//! let batch = MLBatchCircuit::new(witnesses, 256)?;
//! let pi = batch.public_inputs();
//! let prover = MockProver::run(k, &batch, vec![pi]).unwrap();
//! prover.assert_satisfied();
//! ```

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, ErrorFront},
};
use halo2curves::bn256::Fr;

use super::training_step_v2::{
    MLTrainingStepV2Circuit, MLTrainingStepV2Config, MLTrainingStepV2Witness,
    NUM_PUBLIC_INPUTS, load_relu_table, load_exp_table,
};

/// Maximum number of instances in a batch.
pub const MAX_BATCH_SIZE: usize = 32;

/// A batch circuit that proves N training step instances in a single proof.
///
/// All instances share the same circuit configuration (gates, columns, lookup tables),
/// but each has its own witness and public inputs. The public inputs are concatenated
/// in order: [step0_PI0, ..., step0_PI7, step1_PI0, ..., step1_PI7, ...].
#[derive(Clone)]
pub struct MLBatchCircuit {
    /// Individual circuit instances.
    circuits: Vec<MLTrainingStepV2Circuit>,
    /// Number of instances (for padding).
    batch_size: usize,
}

impl Default for MLBatchCircuit {
    fn default() -> Self {
        Self {
            circuits: vec![MLTrainingStepV2Circuit::default()],
            batch_size: 1,
        }
    }
}

impl MLBatchCircuit {
    /// Creates a new batch circuit from multiple witnesses.
    ///
    /// All witnesses share the same ReLU range. Returns an error if
    /// the batch exceeds `MAX_BATCH_SIZE`.
    pub fn new(
        witnesses: Vec<MLTrainingStepV2Witness>,
        relu_range: usize,
    ) -> Result<Self, String> {
        if witnesses.is_empty() {
            return Err("Batch must contain at least one witness".to_string());
        }
        if witnesses.len() > MAX_BATCH_SIZE {
            return Err(format!(
                "Batch size {} exceeds maximum {}",
                witnesses.len(),
                MAX_BATCH_SIZE
            ));
        }

        let circuits: Vec<MLTrainingStepV2Circuit> = witnesses
            .into_iter()
            .map(|w| MLTrainingStepV2Circuit::from_witness(w, relu_range))
            .collect::<Result<Vec<_>, _>>()?;

        let batch_size = circuits.len();
        Ok(Self { circuits, batch_size })
    }

    /// Creates a batch circuit from pre-built circuits.
    pub fn from_circuits(circuits: Vec<MLTrainingStepV2Circuit>) -> Result<Self, String> {
        if circuits.is_empty() {
            return Err("Batch must contain at least one circuit".to_string());
        }
        if circuits.len() > MAX_BATCH_SIZE {
            return Err(format!(
                "Batch size {} exceeds maximum {}",
                circuits.len(),
                MAX_BATCH_SIZE
            ));
        }
        let batch_size = circuits.len();
        Ok(Self { circuits, batch_size })
    }

    /// Returns the number of instances in this batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the total number of public inputs (batch_size * NUM_PUBLIC_INPUTS).
    pub fn num_public_inputs(&self) -> usize {
        self.batch_size * NUM_PUBLIC_INPUTS
    }

    /// Computes the concatenated public inputs for all instances.
    pub fn public_inputs(&self) -> Vec<Fr> {
        let mut pi = Vec::with_capacity(self.num_public_inputs());
        for circuit in &self.circuits {
            pi.extend_from_slice(&circuit.witness.public_inputs());
        }
        pi
    }
}

impl Circuit<Fr> for MLBatchCircuit {
    type Config = MLTrainingStepV2Config;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        // Reuse the same circuit configuration — gates, columns, and lookup tables
        // are shared across all batch instances.
        MLTrainingStepV2Circuit::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        // Load lookup tables once (shared across all instances)
        let relu_range = self.circuits.first()
            .map(|c| c.relu_range)
            .unwrap_or(256);
        let exp_range = self.circuits.first()
            .map(|c| c.exp_range)
            .unwrap_or(128);
        let exp_scale = self.circuits.first()
            .map(|c| c.exp_scale)
            .unwrap_or(64);

        load_relu_table(&config, &mut layouter, relu_range)?;
        load_exp_table(&config, &mut layouter, exp_range, exp_scale)?;

        // Synthesize each instance with its own PI offset, delegating to
        // the same verified logic used by the single-instance circuit.
        for (idx, circuit) in self.circuits.iter().enumerate() {
            let pi_offset = idx * NUM_PUBLIC_INPUTS;
            circuit.synthesize_instance(&config, &mut layouter, pi_offset)?;
        }

        Ok(())
    }
}

/// Batch proving result containing aggregated information.
#[derive(Debug, Clone)]
pub struct BatchProofResult {
    /// Serialized proof bytes.
    pub proof: Vec<u8>,
    /// Concatenated public inputs for all instances.
    pub public_inputs: Vec<Fr>,
    /// Number of instances proven.
    pub batch_size: usize,
    /// Total error bound across all instances.
    pub total_error_bound: Fr,
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    fn make_minimal_witness(step: u64) -> MLTrainingStepV2Witness {
        use crate::ml::training_step_v2::{compute_witness_v2, compute_state_hash_v2};

        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        // Tiny weights (must be small to fit in ReLU lookup range)
        // Stored as flat row-major: w1[j*d_in + i] for neuron j, input i
        let w1 = vec![Fr::from(1u64); d_hid * d_in];
        let b1 = vec![Fr::from(1u64); d_hid];
        let w2 = vec![Fr::from(1u64); d_out * d_hid];
        let b2 = vec![Fr::from(1u64); d_out];
        let input = vec![Fr::from(1u64); d_in];
        let target = vec![Fr::from(2u64); d_out];
        let lr = Fr::from(1u64);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        // For test purposes, new_hash can be dummy (weight update changes it)
        let new_hash = (Fr::from(step + 100), Fr::from(step + 200));

        compute_witness_v2(
            d_in, d_hid, d_out,
            &input, &target,
            &w1, &b1, &w2, &b2,
            lr, old_hash, new_hash, step,
            Fr::from(1u64), // base_error
        )
    }

    #[test]
    fn test_batch_circuit_creation() {
        let w1 = make_minimal_witness(0);
        let w2 = make_minimal_witness(1);

        let batch = MLBatchCircuit::new(vec![w1, w2], 256).unwrap();
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.num_public_inputs(), 16); // 2 * 8
    }

    #[test]
    fn test_batch_circuit_single_instance() {
        let w = make_minimal_witness(0);
        let batch = MLBatchCircuit::new(vec![w], 256).unwrap();
        let pi = batch.public_inputs();

        assert_eq!(pi.len(), NUM_PUBLIC_INPUTS);

        // k=14 is sufficient for a single training step with lookup tables
        let prover = MockProver::run(14, &batch, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_batch_circuit_two_instances() {
        let w1 = make_minimal_witness(0);
        let w2 = make_minimal_witness(1);
        let batch = MLBatchCircuit::new(vec![w1, w2], 256).unwrap();
        let pi = batch.public_inputs();

        assert_eq!(pi.len(), 2 * NUM_PUBLIC_INPUTS);

        // k=15 gives more rows for two instances
        let prover = MockProver::run(15, &batch, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_batch_rejects_empty() {
        let result = MLBatchCircuit::new(vec![], 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_rejects_oversized() {
        let witnesses: Vec<MLTrainingStepV2Witness> = (0..33)
            .map(|i| make_minimal_witness(i))
            .collect();
        let result = MLBatchCircuit::new(witnesses, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_from_circuits() {
        let w = make_minimal_witness(0);
        let circuit = MLTrainingStepV2Circuit::from_witness(w, 256).unwrap();
        let batch = MLBatchCircuit::from_circuits(vec![circuit]).unwrap();
        assert_eq!(batch.batch_size(), 1);
    }

    #[test]
    fn test_batch_proof_result() {
        let result = BatchProofResult {
            proof: vec![0xDE, 0xAD],
            public_inputs: vec![Fr::from(1u64), Fr::from(2u64)],
            batch_size: 1,
            total_error_bound: Fr::from(5u64),
        };
        assert_eq!(result.batch_size, 1);
    }
}
