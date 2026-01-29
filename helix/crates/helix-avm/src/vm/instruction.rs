//! Instructions for the AVM.

use super::opcode::Opcode;
use helix_core::Precision;
use serde::{Deserialize, Serialize};

/// Register index type.
pub type RegIdx = u8;

/// Memory address type.
pub type MemAddr = u32;

/// An operand for an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Operand {
    /// Register reference
    Reg(RegIdx),
    /// Immediate scalar value
    Immediate(f64),
    /// Memory address
    Addr(MemAddr),
    /// Axis for reduction operations
    Axis(i32),
    /// Precision level
    Precision(Precision),
}

impl Operand {
    /// Returns the register index if this is a register operand.
    pub fn as_reg(&self) -> Option<RegIdx> {
        match self {
            Operand::Reg(r) => Some(*r),
            _ => None,
        }
    }

    /// Returns the immediate value if this is an immediate operand.
    pub fn as_immediate(&self) -> Option<f64> {
        match self {
            Operand::Immediate(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the memory address if this is an address operand.
    pub fn as_addr(&self) -> Option<MemAddr> {
        match self {
            Operand::Addr(a) => Some(*a),
            _ => None,
        }
    }
}

/// A single AVM instruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    /// The operation to perform.
    pub opcode: Opcode,
    /// Destination operand (if any).
    pub dst: Option<Operand>,
    /// Source operands.
    pub srcs: Vec<Operand>,
}

impl Instruction {
    /// Creates a new instruction.
    pub fn new(opcode: Opcode, dst: Option<Operand>, srcs: Vec<Operand>) -> Self {
        Self { opcode, dst, srcs }
    }

    /// Creates a no-op instruction.
    pub fn nop() -> Self {
        Self::new(Opcode::Nop, None, vec![])
    }

    /// Creates a halt instruction.
    pub fn halt() -> Self {
        Self::new(Opcode::Halt, None, vec![])
    }

    /// Creates a load instruction.
    pub fn load(dst: RegIdx, addr: MemAddr) -> Self {
        Self::new(
            Opcode::Load,
            Some(Operand::Reg(dst)),
            vec![Operand::Addr(addr)],
        )
    }

    /// Creates a store instruction.
    pub fn store(addr: MemAddr, src: RegIdx) -> Self {
        Self::new(
            Opcode::Store,
            Some(Operand::Addr(addr)),
            vec![Operand::Reg(src)],
        )
    }

    /// Creates an add instruction.
    pub fn add(dst: RegIdx, src1: RegIdx, src2: RegIdx) -> Self {
        Self::new(
            Opcode::Add,
            Some(Operand::Reg(dst)),
            vec![Operand::Reg(src1), Operand::Reg(src2)],
        )
    }

    /// Creates a matmul instruction.
    pub fn matmul(dst: RegIdx, a: RegIdx, b: RegIdx) -> Self {
        Self::new(
            Opcode::MatMul,
            Some(Operand::Reg(dst)),
            vec![Operand::Reg(a), Operand::Reg(b)],
        )
    }

    /// Creates a ReLU instruction.
    pub fn relu(dst: RegIdx, src: RegIdx) -> Self {
        Self::new(
            Opcode::ReLU,
            Some(Operand::Reg(dst)),
            vec![Operand::Reg(src)],
        )
    }

    /// Creates a softmax instruction.
    pub fn softmax(dst: RegIdx, src: RegIdx) -> Self {
        Self::new(
            Opcode::Softmax,
            Some(Operand::Reg(dst)),
            vec![Operand::Reg(src)],
        )
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.opcode)?;
        if let Some(dst) = &self.dst {
            write!(f, " {:?}", dst)?;
        }
        for src in &self.srcs {
            write!(f, ", {:?}", src)?;
        }
        Ok(())
    }
}

/// A program is a sequence of instructions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Program {
    /// The instructions in execution order.
    pub instructions: Vec<Instruction>,
}

impl Program {
    /// Creates an empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an instruction to the program.
    pub fn push(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    /// Returns the number of instructions.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Returns true if the program is empty.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_creation() {
        let add = Instruction::add(0, 1, 2);
        assert_eq!(add.opcode, Opcode::Add);
        assert_eq!(add.dst, Some(Operand::Reg(0)));
        assert_eq!(add.srcs.len(), 2);
    }

    #[test]
    fn test_program() {
        let mut prog = Program::new();
        prog.push(Instruction::load(0, 0));
        prog.push(Instruction::relu(1, 0));
        prog.push(Instruction::halt());
        assert_eq!(prog.len(), 3);
    }
}
