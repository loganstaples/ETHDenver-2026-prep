//! Gradient accumulation with error tracking.

use super::autodiff::NodeIndex;
use helix_core::types::BoundedTensor;
use std::collections::HashMap;

/// Accumulates gradients across multiple backward passes (e.g., for batch training).
#[derive(Debug, Default)]
pub struct GradientAccumulator {
    /// Aggregated gradients: NodeIndex -> Summed Gradient
    accumulated_grads: HashMap<NodeIndex, BoundedTensor>,
    /// Number of steps accumulated.
    step_count: usize,
}

impl GradientAccumulator {
    /// Creates a new empty accumulator.
    pub fn new() -> Self {
        Self {
            accumulated_grads: HashMap::new(),
            step_count: 0,
        }
    }

    /// Adds a set of gradients to the accumulator.
    pub fn accumulate(&mut self, grads: HashMap<NodeIndex, BoundedTensor>) {
        self.step_count += 1;
        for (idx, grad) in grads {
            if let Some(existing) = self.accumulated_grads.get_mut(&idx) {
                *existing = existing.add(&grad);
            } else {
                self.accumulated_grads.insert(idx, grad);
            }
        }
    }

    /// Returns the average gradient for each parameter.
    ///
    /// The error bounds are also averaged (sum of errors / N).
    pub fn average(&self) -> HashMap<NodeIndex, BoundedTensor> {
        if self.step_count == 0 {
            return HashMap::new();
        }

        let scale = helix_core::types::BoundedValue::exact(1.0 / self.step_count as f64);
        
        self.accumulated_grads
            .iter()
            .map(|(idx, sum)| (*idx, sum.scale(scale)))
            .collect()
    }

    /// Clears the accumulator.
    pub fn clear(&mut self) {
        self.accumulated_grads.clear();
        self.step_count = 0;
    }
    
    /// Returns the number of accumulated steps.
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}
