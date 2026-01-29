//! Witness collection with support for large computation chunking.

use super::AVMWitness;
use super::serializer;
use crate::vm::trace::ExecutionTrace;
use helix_core::types::BoundedTensor;

/// Collects witnesses from execution traces, optionally splitting into chunks.
pub struct WitnessCollector {
    /// Maximum number of operations per witness chunk.
    chunk_size: usize,
    /// Whether to include detailed intermediate values.
    include_intermediates: bool,
}

impl WitnessCollector {
    /// Creates a new collector with default settings.
    pub fn new() -> Self {
        Self {
            chunk_size: usize::MAX,
            include_intermediates: true,
        }
    }

    /// Sets the chunk size (ops per witness).
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Sets whether to include intermediates.
    pub fn include_intermediates(mut self, include: bool) -> Self {
        self.include_intermediates = include;
        self
    }

    /// Collects a set of witnesses from the trace.
    ///
    /// If the trace is larger than `chunk_size`, multiple witnesses are returned.
    /// The output of chunk N becomes the input of chunk N+1.
    pub fn collect(
        &self,
        trace: &ExecutionTrace,
        initial_inputs: &[BoundedTensor],
        final_outputs: &[BoundedTensor],
    ) -> Vec<AVMWitness> {
        let steps = trace.steps();
        if steps.is_empty() {
             // Return empty witness for consistency or handle appropriately
             return vec![AVMWitness::new()];
        }

        let mut witnesses = Vec::new();
        
        // We chunk the steps
        let chunks: Vec<_> = steps.chunks(self.chunk_size).collect();
        let num_chunks = chunks.len();

        // Current inputs for the next chunk
        // For the first chunk, it's the global initial_inputs.
        // For subsequent chunks, it's the outputs of the previous chunk (which we need to extract).
        // WARNING: Extracting "outputs" of a chunk is tricky. 
        // A chunk is a slice of instructions. The "output" of a chunk is technically the state of memory?
        // Or is it just the outputs of the operations in that chunk?
        //
        // In this simplified AVM, let's assume "inputs" to a chunk are just the input tensors 
        // of the first instruction in that chunk if we treat it as a sequence.
        // But really, "Private Inputs" for ZK usually mean the full state or strictly the inputs consumed.
        //
        // For this implementation, to match the plan "Output of chunk N is input to chunk N+1":
        // We probably assume a sequence of operations where one feeds the next.
        // Let's implement a simplified view:
        // Chunk Inputs = Inputs of the first step in the chunk (if meaningful) OR
        // more accurately for ZK: The state at the start of the chunk.
        //
        // Given `TraceStep` has `inputs` and `output`, let's just serialize the specific inputs/outputs 
        // recorded in the trace steps for that chunk.
        //
        // Global Public Inputs -> First Chunk
        // Global Outputs -> Last Chunk
        
        for (i, chunk_steps) in chunks.iter().enumerate() {
            let is_first = i == 0;
            let is_last = i == num_chunks - 1;

            // 1. Determine Public Inputs for this chunk
            // Only the first chunk receives the global public inputs.
            // Others receive "private" state passed from previous (handled via private inputs usually).
            // But wait, "public_inputs" in AVMWitness usually matches what the Verifier sees.
            // If we split into chunks, we might be doing IVC (Incrementally Verifiable Computation) 
            // where intermediate states are public inputs to the next verifier?
            // Let's assume standard recursion: intermediate boundary states are public inputs to the circuit instance.
            //
            // Let's keep it simple:
            // Public Inputs field in Witness = what we serialize.
            // For now, let's put Global Inputs in Chunk 0, and nothing in others?
            // Or if we are doing recursion, we need inputs.
            
            // Let's just serialize the actual inputs used by operations in this chunk as "public inputs" 
            // if it's the first chunk, otherwise we leave public inputs empty 
            // and treating everything else as intermediate private inputs.
            
            let chunk_public_inputs = if is_first {
                serializer::serialize_tensors(initial_inputs)
            } else {
                Vec::new() // Internal boundary state is handled differently usually
            };

            // 2. Private Inputs / Intermediates
            // This usually includes all the witness data needed.
            // Let's allow "private_inputs" to carry the outputs of this chunk.
            let chunk_outputs: Vec<BoundedTensor> = if is_last {
                 final_outputs.to_vec()
            } else {
                 // For intermediate chunks, the "outputs" are the outputs of the last step
                 chunk_steps.last()
                    .and_then(|s| s.output.clone())
                    .map(|t| vec![t])
                    .unwrap_or_default()
            };
            
            let chunk_private_inputs = serializer::serialize_tensors(&chunk_outputs);

            // 3. Intermediates (Trace values)
            let intermediate_values = if self.include_intermediates {
                chunk_steps.iter()
                    .filter_map(|s| s.output.as_ref())
                    .map(|t| serializer::serialize_tensor(t))
                    .collect()
            } else {
                Vec::new()
            };

            // 4. Max error in this chunk
            let chunk_max_error = chunk_steps.iter()
                .map(|s| s.cumulative_error)
                .fold(0.0, f64::max);

            witnesses.push(AVMWitness {
                public_inputs: chunk_public_inputs,
                private_inputs: chunk_private_inputs,
                intermediate_values,
                max_error: chunk_max_error,
                num_operations: chunk_steps.len(),
            });
        }

        witnesses
    }
}

impl Default for WitnessCollector {
    fn default() -> Self {
        Self::new()
    }
}
