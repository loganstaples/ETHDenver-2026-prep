//! Memory Management Module for Large Model Support.
//!
//! Provides infrastructure for training large models (500K-2M+ parameters) with bounded
//! memory usage. Key features:
//!
//! - **Chunked Loading**: Load and process model weights in chunks to bound memory
//! - **Gradient Checkpointing**: Trade compute for memory by recomputing activations
//! - **Memory Budget**: Track and enforce memory limits during training
//! - **Profiling**: Layer-by-layer memory and compute profiling
//!
//! # Memory Optimization Strategies
//!
//! 1. **Layer-by-Layer Execution**: Process one transformer layer at a time,
//!    freeing activations before processing the next layer.
//!
//! 2. **Gradient Checkpointing**: Only store activations at checkpoint boundaries,
//!    recompute intermediate activations during backward pass.
//!
//! 3. **Chunked Attention**: Split attention computation into chunks to avoid
//!    materializing the full attention matrix.
//!
//! 4. **Mixed Precision**: Use INT8/INT4 weights with FP32 accumulation.
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::memory::{MemoryBudget, MemoryTracker, GradientCheckpointer};
//!
//! // Create a memory budget (4GB)
//! let budget = MemoryBudget::new(4 * 1024 * 1024 * 1024);
//!
//! // Track memory usage during training
//! let mut tracker = MemoryTracker::new(budget);
//!
//! // Allocate weights
//! tracker.allocate("weights", weight_size)?;
//!
//! // Use gradient checkpointing for activations
//! let checkpointer = GradientCheckpointer::new(CheckpointStrategy::SqrtN);
//! ```

pub mod budget;
pub mod chunked_loading;
pub mod gradient_checkpoint;
pub mod profiler;

// Re-export commonly used types
pub use budget::{AllocationError, MemoryBudget, MemoryRegion, MemoryTracker};
pub use chunked_loading::{
    ChunkConfig, ChunkIterator, ChunkedTensor, ChunkedWeightLoader, WeightChunk,
};
pub use gradient_checkpoint::{
    ActivationCache, CheckpointBoundary, CheckpointStrategy, GradientCheckpointer,
    RecomputeSchedule,
};
pub use profiler::{
    LayerProfile, MemoryEvent, MemoryEventType, MemoryProfiler, ModelProfile, ProfilingConfig,
};
