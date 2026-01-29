//! Opcodes for the Approximate Virtual Machine.

use serde::{Deserialize, Serialize};

/// Opcodes supported by the AVM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Opcode {
    /// No operation
    Nop = 0x00,

    // === Memory Operations ===
    /// Load tensor from memory to register: LOAD reg, addr
    Load = 0x10,
    /// Store tensor from register to memory: STORE addr, reg
    Store = 0x11,
    /// Load immediate scalar value: LOADI reg, value
    LoadImmediate = 0x12,
    /// Copy register: COPY dst, src
    Copy = 0x13,

    // === Basic Arithmetic ===
    /// Element-wise add: ADD dst, src1, src2
    Add = 0x20,
    /// Element-wise subtract: SUB dst, src1, src2
    Sub = 0x21,
    /// Element-wise multiply: MUL dst, src1, src2
    Mul = 0x22,
    /// Element-wise divide: DIV dst, src1, src2
    Div = 0x23,
    /// Negate: NEG dst, src
    Neg = 0x24,
    /// Scalar multiply: SCALE dst, src, scalar
    Scale = 0x25,

    // === Matrix Operations ===
    /// Matrix multiplication: MATMUL dst, A, B
    MatMul = 0x30,
    /// Transpose: TRANSPOSE dst, src
    Transpose = 0x31,
    /// Dot product: DOT dst, src1, src2
    Dot = 0x32,

    // === Activation Functions ===
    /// ReLU activation: RELU dst, src
    ReLU = 0x40,
    /// Sigmoid activation: SIGMOID dst, src
    Sigmoid = 0x41,
    /// Tanh activation: TANH dst, src
    Tanh = 0x42,
    /// GELU activation: GELU dst, src
    GeLU = 0x43,
    /// Leaky ReLU: LRELU dst, src, alpha
    LeakyReLU = 0x44,

    // === Normalization ===
    /// Softmax: SOFTMAX dst, src
    Softmax = 0x50,
    /// Layer normalization: LAYERNORM dst, src
    LayerNorm = 0x51,
    /// Batch normalization: BATCHNORM dst, src
    BatchNorm = 0x52,

    // === Reduction Operations ===
    /// Sum reduction: SUM dst, src, axis
    Sum = 0x60,
    /// Mean reduction: MEAN dst, src, axis
    Mean = 0x61,
    /// Max reduction: MAX dst, src, axis
    Max = 0x62,
    /// Min reduction: MIN dst, src, axis
    Min = 0x63,

    // === Error Bound Operations ===
    /// Check error bounds: BOUNDCHECK src, max_error
    BoundCheck = 0x70,
    /// Get current error: GETERROR dst, src
    GetError = 0x71,
    /// Set precision: SETPRECISION precision_level
    SetPrecision = 0x72,

    // === Control Flow ===
    /// Halt execution: HALT
    Halt = 0xF0,
    /// Assert condition: ASSERT cond
    Assert = 0xF1,
}

impl Opcode {
    /// Returns the number of input operands for this opcode.
    pub fn num_inputs(&self) -> usize {
        match self {
            Opcode::Nop | Opcode::Halt => 0,
            Opcode::Load | Opcode::LoadImmediate | Opcode::Neg
            | Opcode::ReLU | Opcode::Sigmoid | Opcode::Tanh | Opcode::GeLU
            | Opcode::Softmax | Opcode::LayerNorm | Opcode::BatchNorm
            | Opcode::Transpose | Opcode::GetError | Opcode::Assert => 1,
            Opcode::Store | Opcode::Copy | Opcode::Add | Opcode::Sub
            | Opcode::Mul | Opcode::Div | Opcode::Scale | Opcode::MatMul
            | Opcode::Dot | Opcode::LeakyReLU | Opcode::Sum | Opcode::Mean
            | Opcode::Max | Opcode::Min | Opcode::BoundCheck => 2,
            Opcode::SetPrecision => 0,
        }
    }

    /// Returns true if this opcode produces an output.
    pub fn has_output(&self) -> bool {
        !matches!(
            self,
            Opcode::Nop | Opcode::Store | Opcode::Halt
            | Opcode::BoundCheck | Opcode::Assert | Opcode::SetPrecision
        )
    }

    /// Returns the mnemonic string for this opcode.
    pub fn mnemonic(&self) -> &'static str {
        match self {
            Opcode::Nop => "NOP",
            Opcode::Load => "LOAD",
            Opcode::Store => "STORE",
            Opcode::LoadImmediate => "LOADI",
            Opcode::Copy => "COPY",
            Opcode::Add => "ADD",
            Opcode::Sub => "SUB",
            Opcode::Mul => "MUL",
            Opcode::Div => "DIV",
            Opcode::Neg => "NEG",
            Opcode::Scale => "SCALE",
            Opcode::MatMul => "MATMUL",
            Opcode::Transpose => "TRANSPOSE",
            Opcode::Dot => "DOT",
            Opcode::ReLU => "RELU",
            Opcode::Sigmoid => "SIGMOID",
            Opcode::Tanh => "TANH",
            Opcode::GeLU => "GELU",
            Opcode::LeakyReLU => "LRELU",
            Opcode::Softmax => "SOFTMAX",
            Opcode::LayerNorm => "LAYERNORM",
            Opcode::BatchNorm => "BATCHNORM",
            Opcode::Sum => "SUM",
            Opcode::Mean => "MEAN",
            Opcode::Max => "MAX",
            Opcode::Min => "MIN",
            Opcode::BoundCheck => "BOUNDCHECK",
            Opcode::GetError => "GETERROR",
            Opcode::SetPrecision => "SETPRECISION",
            Opcode::Halt => "HALT",
            Opcode::Assert => "ASSERT",
        }
    }
}

impl std::fmt::Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.mnemonic())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_properties() {
        assert_eq!(Opcode::Add.num_inputs(), 2);
        assert!(Opcode::Add.has_output());
        assert!(!Opcode::Halt.has_output());
    }
}
