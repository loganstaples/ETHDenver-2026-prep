//! Witness generation from execution traces.

pub mod serializer;
pub mod collector;

use crate::vm::ExecutionTrace;
use helix_core::types::BoundedTensor;
use serde::{Deserialize, Serialize};

/// A witness for ZK proof generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AVMWitness {
    /// Public inputs (visible to verifier).
    pub public_inputs: Vec<u8>,
    /// Private inputs (hidden from verifier).
    pub private_inputs: Vec<u8>,
    /// Intermediate values for each execution step.
    pub intermediate_values: Vec<Vec<u8>>,
    /// Maximum error at any point during execution.
    pub max_error: f64,
    /// Total number of operations.
    pub num_operations: usize,
}

impl AVMWitness {
    /// Creates an empty witness.
    pub fn new() -> Self {
        Self {
            public_inputs: Vec::new(),
            private_inputs: Vec::new(),
            intermediate_values: Vec::new(),
            max_error: 0.0,
            num_operations: 0,
        }
    }

    /// Returns the total size of the witness in bytes.
    pub fn size_bytes(&self) -> usize {
        self.public_inputs.len()
            + self.private_inputs.len()
            + self.intermediate_values.iter().map(|v| v.len()).sum::<usize>()
    }
}

impl Default for AVMWitness {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifies that a witness is consistent with the claimed error bounds.
pub fn verify_error_bounds(witness: &AVMWitness, max_allowed_error: f64) -> bool {
    witness.max_error <= max_allowed_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use collector::WitnessCollector;

    #[test]
    fn test_witness_collection() {
        let trace = ExecutionTrace::new();
        let inputs = vec![BoundedTensor::zeros(vec![2, 2])];
        let outputs = vec![BoundedTensor::zeros(vec![2, 2])];

        let collector = WitnessCollector::new();
        let witnesses = collector.collect(&trace, &inputs, &outputs);

        // Should return 1 witness (empty trace handled gracefully)
        assert_eq!(witnesses.len(), 1);
        assert_eq!(witnesses[0].num_operations, 0);
    }
}
