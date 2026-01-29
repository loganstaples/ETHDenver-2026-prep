//! Library exports for helix-avm.

pub mod arithmetic;
pub mod gradient;
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
