//! Memory management for the AVM.

use helix_core::types::BoundedTensor;
use std::collections::HashMap;

use super::instruction::MemAddr;

/// Number of general-purpose registers.
pub const NUM_REGISTERS: usize = 32;

/// AVM memory model with registers and addressable memory.
#[derive(Debug, Clone)]
pub struct AVMMemory {
    /// General-purpose tensor registers.
    registers: [Option<BoundedTensor>; NUM_REGISTERS],
    /// Addressable memory (heap).
    memory: HashMap<MemAddr, BoundedTensor>,
    /// Stack pointer for the operand stack.
    stack: Vec<BoundedTensor>,
}

impl AVMMemory {
    /// Creates a new empty memory.
    pub fn new() -> Self {
        Self {
            registers: Default::default(),
            memory: HashMap::new(),
            stack: Vec::new(),
        }
    }

    // === Register Operations ===

    /// Gets a register value.
    pub fn get_reg(&self, idx: u8) -> Option<&BoundedTensor> {
        self.registers.get(idx as usize).and_then(|r| r.as_ref())
    }

    /// Sets a register value.
    pub fn set_reg(&mut self, idx: u8, value: BoundedTensor) {
        if (idx as usize) < NUM_REGISTERS {
            self.registers[idx as usize] = Some(value);
        }
    }

    /// Clears a register.
    pub fn clear_reg(&mut self, idx: u8) {
        if (idx as usize) < NUM_REGISTERS {
            self.registers[idx as usize] = None;
        }
    }

    // === Memory Operations ===

    /// Loads a tensor from memory.
    pub fn load(&self, addr: MemAddr) -> Option<&BoundedTensor> {
        self.memory.get(&addr)
    }

    /// Stores a tensor to memory.
    pub fn store(&mut self, addr: MemAddr, value: BoundedTensor) {
        self.memory.insert(addr, value);
    }

    /// Removes a tensor from memory.
    pub fn free(&mut self, addr: MemAddr) -> Option<BoundedTensor> {
        self.memory.remove(&addr)
    }

    /// Returns the number of allocated memory entries.
    pub fn memory_size(&self) -> usize {
        self.memory.len()
    }

    // === Stack Operations ===

    /// Pushes a tensor onto the stack.
    pub fn push(&mut self, value: BoundedTensor) {
        self.stack.push(value);
    }

    /// Pops a tensor from the stack.
    pub fn pop(&mut self) -> Option<BoundedTensor> {
        self.stack.pop()
    }

    /// Peeks at the top of the stack.
    pub fn peek(&self) -> Option<&BoundedTensor> {
        self.stack.last()
    }

    /// Returns the current stack depth.
    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    /// Clears all memory.
    pub fn clear(&mut self) {
        for reg in &mut self.registers {
            *reg = None;
        }
        self.memory.clear();
        self.stack.clear();
    }
}

impl Default for AVMMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_core::types::BoundedTensor;

    #[test]
    fn test_registers() {
        let mut mem = AVMMemory::new();
        let tensor = BoundedTensor::zeros(vec![2, 2]);
        
        mem.set_reg(0, tensor.clone());
        assert!(mem.get_reg(0).is_some());
        assert!(mem.get_reg(1).is_none());
        
        mem.clear_reg(0);
        assert!(mem.get_reg(0).is_none());
    }

    #[test]
    fn test_memory() {
        let mut mem = AVMMemory::new();
        let tensor = BoundedTensor::zeros(vec![3, 3]);
        
        mem.store(100, tensor);
        assert!(mem.load(100).is_some());
        assert!(mem.load(200).is_none());
    }

    #[test]
    fn test_stack() {
        let mut mem = AVMMemory::new();
        
        mem.push(BoundedTensor::zeros(vec![1]));
        mem.push(BoundedTensor::zeros(vec![2]));
        
        assert_eq!(mem.stack_depth(), 2);
        let t = mem.pop().unwrap();
        assert_eq!(t.shape(), &vec![2]);
    }
}
