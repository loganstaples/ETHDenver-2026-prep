//! Main execution loop for the AVM.

use super::gas::GasMeter;
use super::instruction::{Instruction, Operand, Program};
use super::memory::AVMMemory;
use super::opcode::Opcode;
use super::trace::ExecutionTrace;
use crate::ops;
use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};
use thiserror::Error;

/// Errors that can occur during AVM execution.
#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Invalid register index: {0}")]
    InvalidRegister(u8),

    #[error("Register {0} is empty")]
    EmptyRegister(u8),

    #[error("Invalid memory address: {0}")]
    InvalidAddress(u32),

    #[error("Invalid operand for {opcode}: expected {expected}")]
    InvalidOperand { opcode: String, expected: String },

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Out of gas: used {used}, limit {limit}")]
    OutOfGas { used: u64, limit: u64 },

    #[error("Error bound exceeded: {error} > {max}")]
    ErrorBoundExceeded { error: f64, max: f64 },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Assertion failed at PC {pc}")]
    AssertionFailed { pc: usize },

    #[error("Stack underflow")]
    StackUnderflow,

    #[error("Operation error: {0}")]
    OperationError(String),

    #[error("Unsupported opcode: {0}")]
    UnsupportedOpcode(String),
}

// Error conversions
impl From<crate::ops::BasicOpError> for ExecutionError {
    fn from(e: crate::ops::BasicOpError) -> Self {
        ExecutionError::OperationError(e.to_string())
    }
}

impl From<crate::ops::MatMulError> for ExecutionError {
    fn from(e: crate::ops::MatMulError) -> Self {
        ExecutionError::OperationError(e.to_string())
    }
}

impl From<crate::ops::SoftmaxError> for ExecutionError {
    fn from(e: crate::ops::SoftmaxError) -> Self {
        ExecutionError::OperationError(e.to_string())
    }
}


/// Result type for execution.
pub type ExecutionResult<T> = Result<T, ExecutionError>;

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Whether to collect execution trace.
    pub collect_trace: bool,
    /// Maximum allowed error before halting.
    pub max_error: f64,
    /// Gas limit (0 = unlimited).
    pub gas_limit: u64,
    /// Default precision for operations.
    pub precision: Precision,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            collect_trace: true,
            max_error: 0.01,
            gas_limit: 0,
            precision: Precision::F32,
        }
    }
}

/// The main AVM executor.
pub struct AVMExecutor {
    /// Memory state.
    memory: AVMMemory,
    /// Execution trace (if enabled).
    trace: ExecutionTrace,
    /// Gas meter.
    gas: GasMeter,
    /// Configuration.
    config: ExecutorConfig,
    /// Current precision.
    precision: Precision,
    /// Program counter.
    pc: usize,
    /// Whether execution has halted.
    halted: bool,
}

impl AVMExecutor {
    /// Creates a new executor with the given configuration.
    pub fn new(config: ExecutorConfig) -> Self {
        let gas = if config.gas_limit > 0 {
            GasMeter::new(config.gas_limit)
        } else {
            GasMeter::unlimited()
        };

        Self {
            memory: AVMMemory::new(),
            trace: ExecutionTrace::new(),
            gas,
            precision: config.precision,
            config,
            pc: 0,
            halted: false,
        }
    }

    /// Creates an executor with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ExecutorConfig::default())
    }

    /// Returns a reference to memory.
    pub fn memory(&self) -> &AVMMemory {
        &self.memory
    }

    /// Returns a mutable reference to memory.
    pub fn memory_mut(&mut self) -> &mut AVMMemory {
        &mut self.memory
    }

    /// Returns the execution trace.
    pub fn trace(&self) -> &ExecutionTrace {
        &self.trace
    }

    /// Returns the gas meter.
    pub fn gas(&self) -> &GasMeter {
        &self.gas
    }

    /// Returns the current program counter.
    pub fn pc(&self) -> usize {
        self.pc
    }

    /// Returns true if execution has halted.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Executes a program.
    pub fn execute(&mut self, program: &Program) -> ExecutionResult<()> {
        self.pc = 0;
        self.halted = false;

        while !self.halted && self.pc < program.len() {
            let instr = &program.instructions[self.pc];
            self.step(instr.clone())?;
            self.pc += 1;
        }

        Ok(())
    }

    /// Executes a single instruction.
    pub fn step(&mut self, instr: Instruction) -> ExecutionResult<Option<BoundedTensor>> {
        // Collect inputs for trace
        let inputs = self.collect_inputs(&instr)?;

        // Check gas
        let element_count = inputs.first().map(|t| t.len()).unwrap_or(1);
        if !self.gas.charge(instr.opcode, element_count) {
            return Err(ExecutionError::OutOfGas {
                used: self.gas.gas_used(),
                limit: self.config.gas_limit,
            });
        }

        // Execute the instruction
        let (output, step_error) = self.dispatch(&instr, &inputs)?;

        // Record trace
        if self.config.collect_trace {
            self.trace.record(
                self.pc,
                instr.clone(),
                inputs,
                output.clone(),
                step_error,
            );
        }

        // Check error bounds
        if self.trace.current_cumulative_error() > self.config.max_error {
            return Err(ExecutionError::ErrorBoundExceeded {
                error: self.trace.current_cumulative_error(),
                max: self.config.max_error,
            });
        }

        // Store output
        if let (Some(output_tensor), Some(Operand::Reg(r))) = (&output, &instr.dst) {
            self.memory.set_reg(*r, output_tensor.clone());
        }

        Ok(output)
    }

    /// Collects input tensors for an instruction.
    fn collect_inputs(&self, instr: &Instruction) -> ExecutionResult<Vec<BoundedTensor>> {
        let mut inputs = Vec::new();
        for src in &instr.srcs {
            if let Operand::Reg(r) = src {
                let tensor = self.memory.get_reg(*r).ok_or(ExecutionError::EmptyRegister(*r))?;
                inputs.push(tensor.clone());
            }
        }
        Ok(inputs)
    }

    /// Dispatches an instruction to the appropriate handler.
    fn dispatch(
        &mut self,
        instr: &Instruction,
        inputs: &[BoundedTensor],
    ) -> ExecutionResult<(Option<BoundedTensor>, ErrorMargin)> {
        match instr.opcode {
            Opcode::Nop => Ok((None, ErrorMargin::ZERO)),
            Opcode::Halt => {
                self.halted = true;
                Ok((None, ErrorMargin::ZERO))
            }

            // Memory operations
            Opcode::Load => {
                let addr = instr.srcs.first()
                    .and_then(|o| o.as_addr())
                    .ok_or_else(|| ExecutionError::InvalidOperand {
                        opcode: "LOAD".into(),
                        expected: "address".into(),
                    })?;
                let tensor = self.memory.load(addr)
                    .ok_or(ExecutionError::InvalidAddress(addr))?
                    .clone();
                Ok((Some(tensor), ErrorMargin::ZERO))
            }

            Opcode::Store => {
                if let (Some(Operand::Addr(addr)), Some(tensor)) = (&instr.dst, inputs.first()) {
                    self.memory.store(*addr, tensor.clone());
                }
                Ok((None, ErrorMargin::ZERO))
            }

            Opcode::Copy => {
                let tensor = inputs.first().cloned();
                Ok((tensor, ErrorMargin::ZERO))
            }

            // Basic arithmetic
            Opcode::Add => {
                let (a, b) = (inputs.first(), inputs.get(1));
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let result = ops::basic::add(a, b)?;
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    _ => Err(ExecutionError::InvalidOperand {
                        opcode: "ADD".into(),
                        expected: "two tensors".into(),
                    }),
                }
            }

            Opcode::Sub => {
                let (a, b) = (inputs.first(), inputs.get(1));
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let result = ops::basic::sub(a, b)?;
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    _ => Err(ExecutionError::InvalidOperand {
                        opcode: "SUB".into(),
                        expected: "two tensors".into(),
                    }),
                }
            }

            Opcode::Mul => {
                let (a, b) = (inputs.first(), inputs.get(1));
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let result = ops::basic::mul(a, b)?;
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    _ => Err(ExecutionError::InvalidOperand {
                        opcode: "MUL".into(),
                        expected: "two tensors".into(),
                    }),
                }
            }

            Opcode::Neg => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::basic::neg(a);
                        Ok((Some(result), ErrorMargin::ZERO))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "NEG".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Matrix operations
            Opcode::MatMul => {
                let (a, b) = (inputs.first(), inputs.get(1));
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let result = ops::matmul::matmul(a, b, self.precision)?;
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    _ => Err(ExecutionError::InvalidOperand {
                        opcode: "MATMUL".into(),
                        expected: "two matrices".into(),
                    }),
                }
            }

            Opcode::Transpose => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = a.transpose();
                        Ok((Some(result), ErrorMargin::ZERO))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "TRANSPOSE".into(),
                        expected: "one matrix".into(),
                    }),
                }
            }

            // Activations
            Opcode::ReLU => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::activation::relu(a);
                        Ok((Some(result), ErrorMargin::ZERO))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "RELU".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            Opcode::Sigmoid => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::activation::sigmoid(a, self.precision);
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "SIGMOID".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            Opcode::Softmax => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::softmax::softmax(a, self.precision)?;
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "SOFTMAX".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Set precision
            Opcode::SetPrecision => {
                if let Some(Operand::Precision(p)) = instr.srcs.first() {
                    self.precision = *p;
                }
                Ok((None, ErrorMargin::ZERO))
            }

            // Element-wise division with near-zero clamping
            Opcode::Div => {
                let (a, b) = (inputs.first(), inputs.get(1));
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let result = ops::basic::div(a, b)?;
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    _ => Err(ExecutionError::InvalidOperand {
                        opcode: "DIV".into(),
                        expected: "two tensors".into(),
                    }),
                }
            }

            // Scalar multiply
            Opcode::Scale => {
                let a = inputs.first();
                let scalar = instr.srcs.get(1).and_then(|o| o.as_immediate());
                match (a, scalar) {
                    (Some(a), Some(s)) => {
                        let result = ops::basic::scale(a, s);
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    (Some(a), None) => {
                        // If second operand is a tensor, do element-wise mul
                        if let Some(b) = inputs.get(1) {
                            let result = ops::basic::mul(a, b)?;
                            let error = ErrorMargin::absolute(result.max_error());
                            Ok((Some(result), error))
                        } else {
                            Err(ExecutionError::InvalidOperand {
                                opcode: "SCALE".into(),
                                expected: "tensor and scalar".into(),
                            })
                        }
                    }
                    _ => Err(ExecutionError::InvalidOperand {
                        opcode: "SCALE".into(),
                        expected: "tensor and scalar".into(),
                    }),
                }
            }

            // Dot product
            Opcode::Dot => {
                let (a, b) = (inputs.first(), inputs.get(1));
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let result = ops::matmul::dot(a, b, self.precision)?;
                        let error = ErrorMargin::absolute(result.absolute_error());
                        let tensor = BoundedTensor::new(vec![result], vec![1]);
                        Ok((Some(tensor), error))
                    }
                    _ => Err(ExecutionError::InvalidOperand {
                        opcode: "DOT".into(),
                        expected: "two tensors".into(),
                    }),
                }
            }

            // Tanh activation
            Opcode::Tanh => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::activation::tanh(a, self.precision);
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "TANH".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // GELU activation
            Opcode::GeLU => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::activation::gelu(a, self.precision);
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "GELU".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Leaky ReLU activation
            Opcode::LeakyReLU => {
                let a = inputs.first();
                let alpha = instr.srcs.get(1).and_then(|o| o.as_immediate()).unwrap_or(0.01);
                match a {
                    Some(a) => {
                        let result = ops::activation::leaky_relu(a, alpha);
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "LRELU".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Layer normalization
            Opcode::LayerNorm => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::normalization::layer_norm(a, 1e-5, self.precision);
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "LAYERNORM".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Batch normalization (treated as layer norm in inference mode)
            Opcode::BatchNorm => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::normalization::layer_norm(a, 1e-5, self.precision);
                        let error = ErrorMargin::absolute(result.max_error());
                        Ok((Some(result), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "BATCHNORM".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Sum reduction
            Opcode::Sum => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::reduction::sum(a);
                        let error = ErrorMargin::absolute(result.absolute_error());
                        let tensor = BoundedTensor::new(vec![result], vec![1]);
                        Ok((Some(tensor), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "SUM".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Mean reduction
            Opcode::Mean => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::reduction::mean(a);
                        let error = ErrorMargin::absolute(result.absolute_error());
                        let tensor = BoundedTensor::new(vec![result], vec![1]);
                        Ok((Some(tensor), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "MEAN".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Max reduction
            Opcode::Max => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::reduction::max(a);
                        let error = ErrorMargin::absolute(result.absolute_error());
                        let tensor = BoundedTensor::new(vec![result], vec![1]);
                        Ok((Some(tensor), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "MAX".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Min reduction
            Opcode::Min => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let result = ops::reduction::min(a);
                        let error = ErrorMargin::absolute(result.absolute_error());
                        let tensor = BoundedTensor::new(vec![result], vec![1]);
                        Ok((Some(tensor), error))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "MIN".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Error bound check
            Opcode::BoundCheck => {
                let a = inputs.first();
                let max_err = instr.srcs.get(1).and_then(|o| o.as_immediate()).unwrap_or(self.config.max_error);
                match a {
                    Some(a) => {
                        let current_error = a.max_error();
                        if current_error > max_err {
                            Err(ExecutionError::ErrorBoundExceeded {
                                error: current_error,
                                max: max_err,
                            })
                        } else {
                            Ok((None, ErrorMargin::ZERO))
                        }
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "BOUNDCHECK".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Get current error bound
            Opcode::GetError => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let error = a.max_error();
                        let result = BoundedTensor::new(
                            vec![BoundedValue::exact(error)],
                            vec![1],
                        );
                        Ok((Some(result), ErrorMargin::ZERO))
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "GETERROR".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // Assert condition (non-zero)
            Opcode::Assert => {
                let a = inputs.first();
                match a {
                    Some(a) => {
                        let is_nonzero = a.data().iter().any(|v| v.value().abs() > f64::EPSILON);
                        if is_nonzero {
                            Ok((None, ErrorMargin::ZERO))
                        } else {
                            Err(ExecutionError::AssertionFailed { pc: self.pc })
                        }
                    }
                    None => Err(ExecutionError::InvalidOperand {
                        opcode: "ASSERT".into(),
                        expected: "one tensor".into(),
                    }),
                }
            }

            // LoadImmediate not handled here (would need scalar to tensor)
            Opcode::LoadImmediate => {
                let val = instr.srcs.first().and_then(|o| o.as_immediate()).unwrap_or(0.0);
                let tensor = BoundedTensor::new(vec![BoundedValue::exact(val)], vec![1]);
                Ok((Some(tensor), ErrorMargin::ZERO))
            }
        }
    }

    /// Resets the executor state.
    pub fn reset(&mut self) {
        self.memory.clear();
        self.trace.clear();
        self.gas.reset();
        self.pc = 0;
        self.halted = false;
        self.precision = self.config.precision;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let exec = AVMExecutor::with_defaults();
        assert!(!exec.is_halted());
        assert_eq!(exec.pc(), 0);
    }

    #[test]
    fn test_halt_instruction() {
        let mut exec = AVMExecutor::with_defaults();
        let mut prog = Program::new();
        prog.push(Instruction::halt());

        exec.execute(&prog).unwrap();
        assert!(exec.is_halted());
    }
}
