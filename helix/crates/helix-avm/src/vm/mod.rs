//! VM module - core virtual machine components.

pub mod executor;
pub mod gas;
pub mod instruction;
pub mod memory;
pub mod opcode;
pub mod trace;

pub use executor::{AVMExecutor, ExecutionError, ExecutionResult, ExecutorConfig};
pub use gas::GasMeter;
pub use instruction::{Instruction, Operand, Program, RegIdx, MemAddr};
pub use memory::AVMMemory;
pub use opcode::Opcode;
pub use trace::{ExecutionTrace, TraceStep};
