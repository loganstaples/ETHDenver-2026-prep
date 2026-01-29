//! Witness generation from execution traces.

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

/// Builder for constructing witnesses from execution traces.
pub struct WitnessBuilder {
    /// Whether to include intermediate values.
    include_intermediates: bool,
    /// Maximum number of steps to include.
    max_steps: Option<usize>,
}

impl WitnessBuilder {
    /// Creates a new witness builder.
    pub fn new() -> Self {
        Self {
            include_intermediates: true,
            max_steps: None,
        }
    }

    /// Sets whether to include intermediate values.
    pub fn with_intermediates(mut self, include: bool) -> Self {
        self.include_intermediates = include;
        self
    }

    /// Sets the maximum number of steps.
    pub fn with_max_steps(mut self, max: usize) -> Self {
        self.max_steps = Some(max);
        self
    }

    /// Builds a witness from an execution trace.
    pub fn build(
        &self,
        trace: &ExecutionTrace,
        initial_inputs: &[BoundedTensor],
        final_outputs: &[BoundedTensor],
    ) -> AVMWitness {
        // Serialize public inputs
        let public_inputs = self.serialize_tensors(initial_inputs);

        // Serialize private inputs (intermediate computations)
        let private_inputs = self.serialize_tensors(final_outputs);

        // Serialize intermediate values if requested
        let intermediate_values = if self.include_intermediates {
            let steps = match self.max_steps {
                Some(max) => &trace.steps()[..max.min(trace.len())],
                None => trace.steps(),
            };
            steps
                .iter()
                .filter_map(|step| step.output.as_ref())
                .map(|t| self.serialize_tensor(t))
                .collect()
        } else {
            Vec::new()
        };

        AVMWitness {
            public_inputs,
            private_inputs,
            intermediate_values,
            max_error: trace.max_error(),
            num_operations: trace.total_ops(),
        }
    }

    /// Serializes a tensor to bytes.
    fn serialize_tensor(&self, tensor: &BoundedTensor) -> Vec<u8> {
        // Simple serialization: shape + values + errors
        let mut bytes = Vec::new();

        // Write number of dimensions
        bytes.extend_from_slice(&(tensor.shape().len() as u32).to_le_bytes());

        // Write shape
        for dim in tensor.shape() {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }

        // Write values and errors
        for v in tensor.data() {
            bytes.extend_from_slice(&v.value().to_le_bytes());
            bytes.extend_from_slice(&v.absolute_error().to_le_bytes());
        }

        bytes
    }

    /// Serializes multiple tensors.
    fn serialize_tensors(&self, tensors: &[BoundedTensor]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for t in tensors {
            let tensor_bytes = self.serialize_tensor(t);
            bytes.extend_from_slice(&(tensor_bytes.len() as u64).to_le_bytes());
            bytes.extend(tensor_bytes);
        }
        bytes
    }
}

impl Default for WitnessBuilder {
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

    #[test]
    fn test_witness_builder() {
        let trace = ExecutionTrace::new();
        let inputs = vec![BoundedTensor::zeros(vec![2, 2])];
        let outputs = vec![BoundedTensor::zeros(vec![2, 2])];

        let builder = WitnessBuilder::new();
        let witness = builder.build(&trace, &inputs, &outputs);

        assert_eq!(witness.num_operations, 0);
        assert!(witness.size_bytes() > 0);
    }
}
