//! Witness generation from execution traces.

pub mod serializer;
pub mod collector;

pub use collector::{
    StreamingConfig, StreamingSummary, StreamingWitnessCollector,
    WitnessChunkHeader, WitnessCollector,
};

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
    use crate::vm::trace::{ExecutionTrace, TraceStep};
    use crate::vm::instruction::Instruction;
    use helix_core::types::BoundedTensor;
    use helix_core::types::error_margin::ErrorMargin;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Creates a dummy `TraceStep` with the given cumulative error and an
    /// output tensor of shape `[2]` filled with `value`.
    fn make_step(pc: usize, cumulative_error: f64, value: f64) -> TraceStep {
        let output = BoundedTensor::zeros(vec![2]);
        // Overwrite with `value` for distinctness.
        let mut out = output;
        for bv in out.data_mut() {
            *bv = helix_core::types::bounded_value::BoundedValue::exact(value);
        }

        TraceStep {
            pc,
            instruction: Instruction::nop(),
            inputs: vec![],
            output: Some(out),
            step_error: ErrorMargin::ZERO,
            cumulative_error,
        }
    }

    // -----------------------------------------------------------------------
    // Existing test
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // StreamingWitnessCollector tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_correct_chunk_count() {
        // 7 steps, chunk size 3 => 3 chunks (3 + 3 + 1).
        let config = StreamingConfig {
            chunk_size: 3,
            include_intermediates: true,
            compute_boundary_hash: true,
        };
        let mut collector = StreamingWitnessCollector::new(config);

        for i in 0..7 {
            collector.push_step(&make_step(i, i as f64 * 0.1, i as f64));
        }

        let summary = collector.finish();

        assert_eq!(summary.total_chunks, 3);
        assert_eq!(summary.chunks.len(), 3);
        assert_eq!(summary.witnesses.len(), 3);

        // Chunk sizes: 3, 3, 1.
        assert_eq!(summary.chunks[0].num_operations, 3);
        assert_eq!(summary.chunks[1].num_operations, 3);
        assert_eq!(summary.chunks[2].num_operations, 1);
    }

    #[test]
    fn test_streaming_total_operations() {
        let n = 10;
        let config = StreamingConfig {
            chunk_size: 4,
            include_intermediates: false,
            compute_boundary_hash: false,
        };
        let mut collector = StreamingWitnessCollector::new(config);

        for i in 0..n {
            collector.push_step(&make_step(i, 0.0, 0.0));
        }

        let summary = collector.finish();

        // Sum of operations across all chunks must equal input length.
        let ops_sum: usize = summary.chunks.iter().map(|c| c.num_operations).sum();
        assert_eq!(ops_sum, n);
        assert_eq!(summary.total_operations, n);
    }

    #[test]
    fn test_streaming_boundary_hash_chaining() {
        // output_hash of chunk N must equal input_hash of chunk N+1.
        let config = StreamingConfig {
            chunk_size: 2,
            include_intermediates: true,
            compute_boundary_hash: true,
        };
        let mut collector = StreamingWitnessCollector::new(config);

        for i in 0..6 {
            collector.push_step(&make_step(i, 0.01 * i as f64, i as f64));
        }

        let summary = collector.finish();

        assert_eq!(summary.total_chunks, 3);

        // First chunk input hash should be all zeros (no prior chunk).
        assert_eq!(summary.chunks[0].input_hash, [0u8; 32]);

        // Chain verification.
        for window in summary.chunks.windows(2) {
            assert_eq!(
                window[0].output_hash, window[1].input_hash,
                "output_hash of chunk {} must equal input_hash of chunk {}",
                window[0].chunk_index, window[1].chunk_index
            );
        }
    }

    #[test]
    fn test_streaming_empty_trace() {
        let config = StreamingConfig::default();
        let collector = StreamingWitnessCollector::new(config);
        let summary = collector.finish();

        // Empty trace should still produce one chunk for consistency.
        assert_eq!(summary.total_chunks, 1);
        assert_eq!(summary.total_operations, 0);
        assert_eq!(summary.chunks[0].num_operations, 0);
        assert_eq!(summary.witnesses.len(), 1);
    }

    #[test]
    fn test_streaming_max_error_tracking() {
        let config = StreamingConfig {
            chunk_size: 3,
            include_intermediates: false,
            compute_boundary_hash: false,
        };
        let mut collector = StreamingWitnessCollector::new(config);

        // Errors: 0.1, 0.5, 0.3 | 0.8, 0.2
        let errors = [0.1, 0.5, 0.3, 0.8, 0.2];
        for (i, &err) in errors.iter().enumerate() {
            collector.push_step(&make_step(i, err, 0.0));
        }

        let summary = collector.finish();

        assert_eq!(summary.total_chunks, 2);
        // Chunk 0: max(0.1, 0.5, 0.3) = 0.5
        assert!((summary.chunks[0].max_error - 0.5).abs() < 1e-12);
        // Chunk 1: max(0.8, 0.2) = 0.8
        assert!((summary.chunks[1].max_error - 0.8).abs() < 1e-12);
        // Overall: 0.8
        assert!((summary.overall_max_error - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_streaming_current_chunk_size() {
        let config = StreamingConfig {
            chunk_size: 5,
            include_intermediates: false,
            compute_boundary_hash: false,
        };
        let mut collector = StreamingWitnessCollector::new(config);

        assert_eq!(collector.current_chunk_size(), 0);

        collector.push_step(&make_step(0, 0.0, 0.0));
        assert_eq!(collector.current_chunk_size(), 1);

        collector.push_step(&make_step(1, 0.0, 0.0));
        collector.push_step(&make_step(2, 0.0, 0.0));
        assert_eq!(collector.current_chunk_size(), 3);

        // Push 2 more to reach chunk_size=5, should auto-flush.
        collector.push_step(&make_step(3, 0.0, 0.0));
        collector.push_step(&make_step(4, 0.0, 0.0));
        assert_eq!(collector.current_chunk_size(), 0);
    }

    #[test]
    fn test_streaming_exact_chunk_boundary() {
        // 6 steps, chunk size 3 => exactly 2 full chunks, no partial.
        let config = StreamingConfig {
            chunk_size: 3,
            include_intermediates: false,
            compute_boundary_hash: true,
        };
        let mut collector = StreamingWitnessCollector::new(config);

        for i in 0..6 {
            collector.push_step(&make_step(i, 0.0, i as f64));
        }

        let summary = collector.finish();

        assert_eq!(summary.total_chunks, 2);
        assert_eq!(summary.total_operations, 6);
        assert_eq!(summary.chunks[0].num_operations, 3);
        assert_eq!(summary.chunks[1].num_operations, 3);
    }

    // -----------------------------------------------------------------------
    // Serialization round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_tensor_roundtrip() {
        let tensor = BoundedTensor::zeros(vec![3, 4]);
        let bytes = serializer::serialize_tensor(&tensor);
        let (recovered, consumed) = serializer::deserialize_tensor(&bytes).unwrap();

        assert_eq!(consumed, bytes.len());
        assert_eq!(recovered.shape(), tensor.shape());
        assert_eq!(recovered.data().len(), tensor.data().len());
        for (a, b) in recovered.data().iter().zip(tensor.data().iter()) {
            assert!((a.value() - b.value()).abs() < 1e-15);
        }
    }

    #[test]
    fn test_deserialize_tensors_roundtrip() {
        let tensors = vec![
            BoundedTensor::zeros(vec![2, 3]),
            BoundedTensor::zeros(vec![5]),
        ];
        let bytes = serializer::serialize_tensors(&tensors);
        let recovered = serializer::deserialize_tensors(&bytes).unwrap();

        assert_eq!(recovered.len(), tensors.len());
        for (r, t) in recovered.iter().zip(tensors.iter()) {
            assert_eq!(r.shape(), t.shape());
        }
    }

    #[test]
    fn test_compute_state_hash_deterministic() {
        let data = b"some boundary state bytes";
        let h1 = serializer::compute_state_hash(data);
        let h2 = serializer::compute_state_hash(data);
        assert_eq!(h1, h2);

        // Different data should (almost certainly) produce a different hash.
        let h3 = serializer::compute_state_hash(b"different data");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_deserialize_tensor_malformed() {
        // Empty slice.
        assert!(serializer::deserialize_tensor(&[]).is_none());

        // Rank=2 but not enough bytes for shape dims.
        assert!(serializer::deserialize_tensor(&[2, 0, 0]).is_none());
    }
}
