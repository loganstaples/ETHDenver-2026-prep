//! Execution trace for ZK witness generation.

use super::instruction::Instruction;
use super::opcode::Opcode;
use helix_core::types::{BoundedTensor, ErrorMargin};
use serde::{Deserialize, Serialize};

/// A single step in the execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// Program counter (instruction index).
    pub pc: usize,
    /// The instruction executed.
    pub instruction: Instruction,
    /// Input tensor values (cloned for witness).
    pub inputs: Vec<BoundedTensor>,
    /// Output tensor value (if any).
    pub output: Option<BoundedTensor>,
    /// Error accumulated in this step.
    pub step_error: ErrorMargin,
    /// Cumulative error up to this point.
    pub cumulative_error: f64,
}

/// Complete execution trace for a program.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// All execution steps in order.
    steps: Vec<TraceStep>,
    /// Maximum error seen during execution.
    max_error: f64,
    /// Total number of operations executed.
    total_ops: usize,
}

impl ExecutionTrace {
    /// Creates an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a single execution step.
    pub fn record(
        &mut self,
        pc: usize,
        instruction: Instruction,
        inputs: Vec<BoundedTensor>,
        output: Option<BoundedTensor>,
        step_error: ErrorMargin,
    ) {
        let step_error_abs = step_error.to_absolute(1.0);
        let cumulative_error = self.current_cumulative_error() + step_error_abs;

        if cumulative_error > self.max_error {
            self.max_error = cumulative_error;
        }

        self.steps.push(TraceStep {
            pc,
            instruction,
            inputs,
            output,
            step_error,
            cumulative_error,
        });
        self.total_ops += 1;
    }

    /// Returns the current cumulative error.
    pub fn current_cumulative_error(&self) -> f64 {
        self.steps.last().map(|s| s.cumulative_error).unwrap_or(0.0)
    }

    /// Returns the maximum error seen.
    pub fn max_error(&self) -> f64 {
        self.max_error
    }

    /// Returns the total number of operations.
    pub fn total_ops(&self) -> usize {
        self.total_ops
    }

    /// Returns all steps.
    pub fn steps(&self) -> &[TraceStep] {
        &self.steps
    }

    /// Returns the number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Clears the trace.
    pub fn clear(&mut self) {
        self.steps.clear();
        self.max_error = 0.0;
        self.total_ops = 0;
    }

    /// Returns operation counts by opcode.
    pub fn op_counts(&self) -> std::collections::HashMap<Opcode, usize> {
        let mut counts = std::collections::HashMap::new();
        for step in &self.steps {
            *counts.entry(step.instruction.opcode).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_recording() {
        let mut trace = ExecutionTrace::new();
        
        trace.record(
            0,
            Instruction::halt(),
            vec![],
            None,
            ErrorMargin::ZERO,
        );
        
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.total_ops(), 1);
    }
}
