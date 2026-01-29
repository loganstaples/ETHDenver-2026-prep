//! Gas metering for AVM execution.

use super::opcode::Opcode;
use serde::{Deserialize, Serialize};

/// Gas costs for different operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasCosts {
    /// Base cost for any instruction.
    pub base: u64,
    /// Cost per element for element-wise operations.
    pub per_element: u64,
    /// Cost multiplier for matrix operations (per element of result).
    pub matmul_per_element: u64,
    /// Cost for memory load.
    pub memory_load: u64,
    /// Cost for memory store.
    pub memory_store: u64,
}

impl Default for GasCosts {
    fn default() -> Self {
        Self {
            base: 1,
            per_element: 1,
            matmul_per_element: 10,
            memory_load: 5,
            memory_store: 5,
        }
    }
}

/// Gas meter for tracking execution costs.
#[derive(Debug, Clone, Default)]
pub struct GasMeter {
    /// Total gas used.
    gas_used: u64,
    /// Gas limit (0 = unlimited).
    gas_limit: u64,
    /// Cost configuration.
    costs: GasCosts,
}

impl GasMeter {
    /// Creates a new gas meter with the given limit.
    pub fn new(gas_limit: u64) -> Self {
        Self {
            gas_used: 0,
            gas_limit,
            costs: GasCosts::default(),
        }
    }

    /// Creates an unlimited gas meter.
    pub fn unlimited() -> Self {
        Self::new(0)
    }

    /// Returns the current gas used.
    pub fn gas_used(&self) -> u64 {
        self.gas_used
    }

    /// Returns the remaining gas.
    pub fn gas_remaining(&self) -> Option<u64> {
        if self.gas_limit == 0 {
            None
        } else {
            Some(self.gas_limit.saturating_sub(self.gas_used))
        }
    }

    /// Charges gas for an operation.
    /// Returns false if out of gas.
    pub fn charge(&mut self, opcode: Opcode, element_count: usize) -> bool {
        let cost = self.compute_cost(opcode, element_count);
        
        if self.gas_limit > 0 && self.gas_used + cost > self.gas_limit {
            return false;
        }
        
        self.gas_used += cost;
        true
    }

    /// Computes the gas cost for an operation.
    pub fn compute_cost(&self, opcode: Opcode, element_count: usize) -> u64 {
        let elements = element_count as u64;
        
        match opcode {
            Opcode::Nop | Opcode::Halt => 0,
            Opcode::Load | Opcode::LoadImmediate => self.costs.memory_load,
            Opcode::Store => self.costs.memory_store,
            Opcode::MatMul => self.costs.base + elements * self.costs.matmul_per_element,
            Opcode::Softmax | Opcode::LayerNorm | Opcode::BatchNorm => {
                self.costs.base + elements * self.costs.per_element * 3
            }
            _ => self.costs.base + elements * self.costs.per_element,
        }
    }

    /// Resets the gas meter.
    pub fn reset(&mut self) {
        self.gas_used = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gas_charging() {
        let mut meter = GasMeter::new(100);
        assert!(meter.charge(Opcode::Add, 10));
        assert!(meter.gas_used() > 0);
    }

    #[test]
    fn test_out_of_gas() {
        let mut meter = GasMeter::new(5);
        assert!(!meter.charge(Opcode::MatMul, 1000));
    }
}
