//! Witness collection with support for large computation chunking and streaming.
//!
//! This module provides two collectors:
//! - [`WitnessCollector`]: Batch collector that processes an entire `ExecutionTrace` at once.
//! - [`StreamingWitnessCollector`]: Incremental collector that processes steps one at a time,
//!   flushing chunks as they fill. Suitable for large traces where holding all steps in memory
//!   is undesirable, and for IVC pipelines that need boundary state hashes.

use serde::{Deserialize, Serialize};

use super::AVMWitness;
use super::serializer;
use crate::vm::trace::{ExecutionTrace, TraceStep};
use helix_core::types::BoundedTensor;

// ---------------------------------------------------------------------------
// WitnessCollector (batch, existing)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// StreamingWitnessCollector (incremental)
// ---------------------------------------------------------------------------

/// Configuration for streaming witness collection.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum operations per chunk (default: 10000).
    pub chunk_size: usize,
    /// Whether to include all intermediate values in each chunk witness.
    pub include_intermediates: bool,
    /// Whether to compute boundary hashes for IVC chaining.
    pub compute_boundary_hash: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10_000,
            include_intermediates: true,
            compute_boundary_hash: true,
        }
    }
}

/// Metadata for a single witness chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessChunkHeader {
    /// Zero-based chunk index.
    pub chunk_index: usize,
    /// Number of operations in this chunk.
    pub num_operations: usize,
    /// Maximum error in this chunk.
    pub max_error: f64,
    /// Hash of this chunk's input state (boundary from previous chunk or zeros for the first).
    pub input_hash: [u8; 32],
    /// Hash of this chunk's output state.
    pub output_hash: [u8; 32],
}

/// Summary of a streaming witness collection session.
#[derive(Debug, Clone)]
pub struct StreamingSummary {
    /// Headers for every completed chunk.
    pub chunks: Vec<WitnessChunkHeader>,
    /// Total operations across all chunks.
    pub total_operations: usize,
    /// Number of chunks produced.
    pub total_chunks: usize,
    /// Overall maximum error across all chunks.
    pub overall_max_error: f64,
    /// The collected witnesses, one per chunk.
    pub witnesses: Vec<AVMWitness>,
}

/// Streaming witness collector that processes trace steps incrementally.
///
/// Steps are pushed one at a time via [`push_step`]. When the current chunk
/// reaches `config.chunk_size`, it is automatically flushed. Call [`finish`]
/// to flush any remaining partial chunk and obtain the [`StreamingSummary`].
pub struct StreamingWitnessCollector {
    config: StreamingConfig,
    /// Steps accumulated for the current (unflushed) chunk.
    current_chunk_ops: Vec<TraceStep>,
    /// Index of the next chunk to be flushed.
    current_chunk_index: usize,
    /// Already-flushed chunks: (header, witness).
    completed_chunks: Vec<(WitnessChunkHeader, AVMWitness)>,
    /// Output hash of the most recently flushed chunk (used as the next chunk's input hash).
    last_output_hash: [u8; 32],
}

impl StreamingWitnessCollector {
    /// Creates a new streaming collector with the given configuration.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            current_chunk_ops: Vec::new(),
            current_chunk_index: 0,
            completed_chunks: Vec::new(),
            last_output_hash: [0u8; 32],
        }
    }

    /// Pushes a single execution step into the collector.
    ///
    /// If the current chunk is full after this push, it is automatically flushed.
    pub fn push_step(&mut self, step: &TraceStep) {
        self.current_chunk_ops.push(step.clone());

        if self.current_chunk_ops.len() >= self.config.chunk_size {
            self.flush_chunk();
        }
    }

    /// Returns the number of operations in the current (unflushed) chunk.
    pub fn current_chunk_size(&self) -> usize {
        self.current_chunk_ops.len()
    }

    /// Flushes the current chunk, producing a header and witness.
    ///
    /// This is called automatically when the chunk reaches `chunk_size`, but
    /// may also be called manually to force a flush at an arbitrary boundary.
    /// If there are no pending operations this is a no-op.
    pub fn flush_chunk(&mut self) {
        if self.current_chunk_ops.is_empty() {
            return;
        }

        let chunk_index = self.current_chunk_index;
        let num_operations = self.current_chunk_ops.len();

        // --- max error ---
        let max_error = self
            .current_chunk_ops
            .iter()
            .map(|s| s.cumulative_error)
            .fold(0.0, f64::max);

        // --- intermediate values ---
        let intermediate_values = if self.config.include_intermediates {
            self.current_chunk_ops
                .iter()
                .filter_map(|s| s.output.as_ref())
                .map(|t| serializer::serialize_tensor(t))
                .collect()
        } else {
            Vec::new()
        };

        // --- boundary state bytes for hashing ---
        // The boundary state of a chunk is the serialized outputs of its last step.
        // This gives IVC verifiers a compact fingerprint of the computation's state
        // at this boundary.
        let boundary_bytes: Vec<u8> = self
            .current_chunk_ops
            .last()
            .and_then(|s| s.output.as_ref())
            .map(|t| serializer::serialize_tensor(t))
            .unwrap_or_default();

        let input_hash = self.last_output_hash;
        let output_hash = if self.config.compute_boundary_hash {
            serializer::compute_state_hash(&boundary_bytes)
        } else {
            [0u8; 32]
        };

        // --- build witness ---
        // Public inputs: the input hash (boundary from previous chunk).
        // Private inputs: the boundary bytes of this chunk's output.
        let witness = AVMWitness {
            public_inputs: input_hash.to_vec(),
            private_inputs: boundary_bytes,
            intermediate_values,
            max_error,
            num_operations,
        };

        let header = WitnessChunkHeader {
            chunk_index,
            num_operations,
            max_error,
            input_hash,
            output_hash,
        };

        self.last_output_hash = output_hash;
        self.completed_chunks.push((header, witness));
        self.current_chunk_index += 1;
        self.current_chunk_ops.clear();
    }

    /// Finishes the streaming session.
    ///
    /// Any remaining operations are flushed as a final (possibly partial) chunk.
    /// If no steps were ever pushed, a single empty chunk is produced for
    /// consistency with [`WitnessCollector`].
    pub fn finish(mut self) -> StreamingSummary {
        // Flush leftover ops (or produce the empty-chunk if nothing was pushed).
        if !self.current_chunk_ops.is_empty() || self.completed_chunks.is_empty() {
            self.flush_empty_or_remaining();
        }

        let total_operations: usize = self
            .completed_chunks
            .iter()
            .map(|(h, _)| h.num_operations)
            .sum();
        let overall_max_error = self
            .completed_chunks
            .iter()
            .map(|(h, _)| h.max_error)
            .fold(0.0, f64::max);
        let total_chunks = self.completed_chunks.len();

        let (chunks, witnesses): (Vec<_>, Vec<_>) =
            self.completed_chunks.into_iter().unzip();

        StreamingSummary {
            chunks,
            total_operations,
            total_chunks,
            overall_max_error,
            witnesses,
        }
    }

    /// Helper: flush remaining ops, or produce a single empty chunk if nothing was pushed.
    fn flush_empty_or_remaining(&mut self) {
        if self.current_chunk_ops.is_empty() && self.completed_chunks.is_empty() {
            // Produce one empty chunk (mirrors WitnessCollector behaviour on empty trace).
            let header = WitnessChunkHeader {
                chunk_index: 0,
                num_operations: 0,
                max_error: 0.0,
                input_hash: [0u8; 32],
                output_hash: if self.config.compute_boundary_hash {
                    serializer::compute_state_hash(&[])
                } else {
                    [0u8; 32]
                },
            };
            let witness = AVMWitness::new();
            self.completed_chunks.push((header, witness));
        } else if !self.current_chunk_ops.is_empty() {
            self.flush_chunk();
        }
    }
}
