//! Library exports for helix-avm.
//!
//! The Approximate Virtual Machine (AVM) provides infrastructure for neural network
//! execution with bounded error tracking. Key features:
//!
//! - **Bounded Computation**: All operations track error margins
//! - **Quantization**: INT8/INT4 support with error accounting
//! - **Memory Efficiency**: Chunked loading, gradient checkpointing, memory budgeting
//! - **Large Model Support**: Scales to 500K-2M+ parameters with bounded memory
//! - **Model Architectures**: Production-ready GPT and BERT implementations

pub mod arithmetic;
pub mod bounds;
pub mod circuit_bridge;
pub mod gradient;
pub mod memory;
pub mod models;
pub mod nn;
pub mod ops;
pub mod quantization;
pub mod vm;
pub mod witness;

// Re-export core VM types
pub use vm::{
    AVMExecutor, AVMMemory, ExecutionError, ExecutionResult, ExecutionTrace,
    ExecutorConfig, GasMeter, Instruction, MemAddr, Opcode, Operand, Program,
    RegIdx, TraceStep,
};

// Re-export witness types
pub use witness::collector::WitnessCollector;
pub use witness::AVMWitness;

// Re-export memory management types
pub use memory::{
    AllocationError, MemoryBudget, MemoryRegion, MemoryTracker,
    ChunkConfig, ChunkedTensor, ChunkedWeightLoader, WeightChunk,
    CheckpointStrategy, GradientCheckpointer, ActivationCache,
    MemoryProfiler, ModelProfile, LayerProfile, ProfilingConfig,
};

// Re-export model types
pub use models::{
    GPTConfig, GPTModel, BERTConfig, BERTModel,
    ModelSize, ModelDimensions, Model,
    Checkpoint, CheckpointMetadata, ModelSerializer, SerializationError,
};

// Re-export large model support
pub use nn::large_model::{
    LargeModelConfig, LargeModelExecutor, LargeModelError,
    chunked_attention, estimate_forward_memory, compute_optimal_batch_size,
};

