//! Error budget allocation algorithms for mixed-precision training.
//!
//! This module provides sophisticated algorithms for allocating error budgets
//! across different parts of a neural network to maximize training efficiency
//! while maintaining accuracy guarantees.

use super::precision::Precision;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for error budget allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAllocationConfig {
    /// Total error budget for training.
    pub total_budget: f64,
    /// Allocation strategy to use.
    pub strategy: AllocationStrategy,
    /// Minimum allocation per component (fraction of total).
    pub min_allocation: f64,
    /// Maximum allocation per component (fraction of total).
    pub max_allocation: f64,
    /// Reserve for unexpected errors (fraction of total).
    pub reserve_fraction: f64,
    /// Update frequency (steps between reallocation).
    pub update_interval: usize,
    /// Smoothing factor for historical data (0-1).
    pub smoothing_factor: f64,
}

impl Default for BudgetAllocationConfig {
    fn default() -> Self {
        Self {
            total_budget: 0.01,
            strategy: AllocationStrategy::Weighted,
            min_allocation: 0.01,
            max_allocation: 0.3,
            reserve_fraction: 0.1,
            update_interval: 100,
            smoothing_factor: 0.9,
        }
    }
}

/// Strategy for allocating error budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Equal allocation to all components.
    Equal,
    /// Weighted by component sensitivity.
    Weighted,
    /// Proportional to compute cost.
    ProportionalToCost,
    /// Adaptive based on historical consumption.
    Adaptive,
    /// Hierarchical (layer-wise priorities).
    Hierarchical,
    /// Optimal (minimize total loss impact).
    Optimal,
}

/// Component that receives budget allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetComponent {
    /// Component name.
    pub name: String,
    /// Component type.
    pub component_type: ComponentType,
    /// Sensitivity to error (0-1, higher = more sensitive).
    pub sensitivity: f64,
    /// Relative compute cost.
    pub compute_cost: f64,
    /// Historical error consumption.
    pub historical_consumption: Vec<f64>,
    /// Allocated budget.
    pub allocated_budget: f64,
    /// Consumed budget so far.
    pub consumed: f64,
    /// Precision assigned to this component.
    pub precision: Precision,
}

/// Type of component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentType {
    /// Embedding layer.
    Embedding,
    /// Attention mechanism.
    Attention,
    /// Feed-forward layer.
    FeedForward,
    /// Normalization layer.
    Normalization,
    /// Output projection.
    Output,
    /// Loss computation.
    Loss,
    /// Gradient computation.
    Gradient,
    /// Weight update.
    WeightUpdate,
    /// Custom component.
    Custom,
}

impl ComponentType {
    /// Returns the default sensitivity for this component type.
    pub fn default_sensitivity(&self) -> f64 {
        match self {
            ComponentType::Loss => 1.0,
            ComponentType::WeightUpdate => 0.95,
            ComponentType::Gradient => 0.9,
            ComponentType::Normalization => 0.85,
            ComponentType::Output => 0.8,
            ComponentType::Attention => 0.7,
            ComponentType::FeedForward => 0.6,
            ComponentType::Embedding => 0.5,
            ComponentType::Custom => 0.5,
        }
    }

    /// Returns the default compute cost multiplier.
    pub fn default_compute_cost(&self) -> f64 {
        match self {
            ComponentType::Attention => 3.0,
            ComponentType::FeedForward => 2.0,
            ComponentType::Embedding => 1.0,
            ComponentType::Normalization => 0.5,
            ComponentType::Output => 1.0,
            ComponentType::Loss => 0.1,
            ComponentType::Gradient => 2.0,
            ComponentType::WeightUpdate => 0.5,
            ComponentType::Custom => 1.0,
        }
    }
}

impl BudgetComponent {
    /// Creates a new budget component.
    pub fn new(name: &str, component_type: ComponentType) -> Self {
        Self {
            name: name.to_string(),
            component_type,
            sensitivity: component_type.default_sensitivity(),
            compute_cost: component_type.default_compute_cost(),
            historical_consumption: Vec::new(),
            allocated_budget: 0.0,
            consumed: 0.0,
            precision: Precision::F32,
        }
    }

    /// Sets sensitivity.
    pub fn with_sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity.clamp(0.0, 1.0);
        self
    }

    /// Sets compute cost.
    pub fn with_compute_cost(mut self, cost: f64) -> Self {
        self.compute_cost = cost;
        self
    }

    /// Records error consumption.
    pub fn consume(&mut self, amount: f64) {
        self.consumed += amount;
        self.historical_consumption.push(amount);

        // Keep only recent history
        const MAX_HISTORY: usize = 1000;
        if self.historical_consumption.len() > MAX_HISTORY {
            self.historical_consumption.remove(0);
        }
    }

    /// Returns remaining budget.
    pub fn remaining(&self) -> f64 {
        (self.allocated_budget - self.consumed).max(0.0)
    }

    /// Returns consumption rate (average per step).
    pub fn consumption_rate(&self) -> f64 {
        if self.historical_consumption.is_empty() {
            return 0.0;
        }
        self.historical_consumption.iter().sum::<f64>()
            / self.historical_consumption.len() as f64
    }

    /// Returns projected total consumption.
    pub fn projected_consumption(&self, remaining_steps: usize) -> f64 {
        self.consumed + self.consumption_rate() * remaining_steps as f64
    }

    /// Checks if component is over budget.
    pub fn is_over_budget(&self) -> bool {
        self.consumed > self.allocated_budget
    }

    /// Resets consumption (e.g., at epoch boundary).
    pub fn reset_consumption(&mut self) {
        self.consumed = 0.0;
    }
}

/// Allocation result for a single component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentAllocation {
    /// Component name.
    pub name: String,
    /// Allocated budget.
    pub budget: f64,
    /// Recommended precision.
    pub precision: Precision,
    /// Allocation confidence.
    pub confidence: f64,
    /// Reason for allocation.
    pub reason: String,
}

/// Result of budget allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationResult {
    /// Allocations by component.
    pub allocations: Vec<ComponentAllocation>,
    /// Total allocated budget.
    pub total_allocated: f64,
    /// Reserve remaining.
    pub reserve: f64,
    /// Strategy used.
    pub strategy: AllocationStrategy,
    /// Timestamp.
    pub timestamp_ms: u64,
}

impl AllocationResult {
    /// Gets allocation for a specific component.
    pub fn get(&self, name: &str) -> Option<&ComponentAllocation> {
        self.allocations.iter().find(|a| a.name == name)
    }

    /// Converts to a map for easy lookup.
    pub fn to_map(&self) -> HashMap<String, ComponentAllocation> {
        self.allocations
            .iter()
            .map(|a| (a.name.clone(), a.clone()))
            .collect()
    }
}

/// Error budget allocator.
#[derive(Debug, Clone)]
pub struct BudgetAllocator {
    config: BudgetAllocationConfig,
    components: Vec<BudgetComponent>,
    current_allocation: Option<AllocationResult>,
    last_update_step: usize,
    total_consumed: f64,
}

impl BudgetAllocator {
    /// Creates a new budget allocator.
    pub fn new(config: BudgetAllocationConfig) -> Self {
        Self {
            config,
            components: Vec::new(),
            current_allocation: None,
            last_update_step: 0,
            total_consumed: 0.0,
        }
    }

    /// Adds a component to track.
    pub fn add_component(&mut self, component: BudgetComponent) {
        self.components.push(component);
    }

    /// Removes a component.
    pub fn remove_component(&mut self, name: &str) {
        self.components.retain(|c| c.name != name);
    }

    /// Gets a component by name.
    pub fn get_component(&self, name: &str) -> Option<&BudgetComponent> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Gets a mutable component by name.
    pub fn get_component_mut(&mut self, name: &str) -> Option<&mut BudgetComponent> {
        self.components.iter_mut().find(|c| c.name == name)
    }

    /// Allocates budget to all components.
    pub fn allocate(&mut self) -> AllocationResult {
        let usable_budget =
            self.config.total_budget * (1.0 - self.config.reserve_fraction) - self.total_consumed;

        let allocations = match self.config.strategy {
            AllocationStrategy::Equal => self.allocate_equal(usable_budget),
            AllocationStrategy::Weighted => self.allocate_weighted(usable_budget),
            AllocationStrategy::ProportionalToCost => self.allocate_proportional_to_cost(usable_budget),
            AllocationStrategy::Adaptive => self.allocate_adaptive(usable_budget),
            AllocationStrategy::Hierarchical => self.allocate_hierarchical(usable_budget),
            AllocationStrategy::Optimal => self.allocate_optimal(usable_budget),
        };

        let total_allocated: f64 = allocations.iter().map(|a| a.budget).sum();

        let result = AllocationResult {
            allocations,
            total_allocated,
            reserve: self.config.total_budget * self.config.reserve_fraction,
            strategy: self.config.strategy,
            timestamp_ms: current_timestamp_ms(),
        };

        // Apply allocations to components
        for alloc in &result.allocations {
            if let Some(component) = self.get_component_mut(&alloc.name) {
                component.allocated_budget = alloc.budget;
                component.precision = alloc.precision;
            }
        }

        self.current_allocation = Some(result.clone());
        result
    }

    /// Equal allocation strategy.
    fn allocate_equal(&self, budget: f64) -> Vec<ComponentAllocation> {
        let n = self.components.len();
        if n == 0 {
            return Vec::new();
        }

        let per_component = budget / n as f64;
        let clamped = per_component.clamp(
            self.config.min_allocation * self.config.total_budget,
            self.config.max_allocation * self.config.total_budget,
        );

        self.components
            .iter()
            .map(|c| ComponentAllocation {
                name: c.name.clone(),
                budget: clamped,
                precision: self.precision_for_budget(clamped, &c),
                confidence: 0.8,
                reason: "Equal allocation".to_string(),
            })
            .collect()
    }

    /// Weighted allocation by sensitivity.
    fn allocate_weighted(&self, budget: f64) -> Vec<ComponentAllocation> {
        let total_sensitivity: f64 = self.components.iter().map(|c| c.sensitivity).sum();

        if total_sensitivity == 0.0 {
            return self.allocate_equal(budget);
        }

        self.components
            .iter()
            .map(|c| {
                let weight = c.sensitivity / total_sensitivity;
                let raw_budget = budget * weight;
                let clamped = raw_budget.clamp(
                    self.config.min_allocation * self.config.total_budget,
                    self.config.max_allocation * self.config.total_budget,
                );

                ComponentAllocation {
                    name: c.name.clone(),
                    budget: clamped,
                    precision: self.precision_for_budget(clamped, &c),
                    confidence: 0.85,
                    reason: format!("Weighted by sensitivity {:.2}", c.sensitivity),
                }
            })
            .collect()
    }

    /// Proportional to compute cost allocation.
    fn allocate_proportional_to_cost(&self, budget: f64) -> Vec<ComponentAllocation> {
        let total_cost: f64 = self.components.iter().map(|c| c.compute_cost).sum();

        if total_cost == 0.0 {
            return self.allocate_equal(budget);
        }

        self.components
            .iter()
            .map(|c| {
                let weight = c.compute_cost / total_cost;
                let raw_budget = budget * weight;
                let clamped = raw_budget.clamp(
                    self.config.min_allocation * self.config.total_budget,
                    self.config.max_allocation * self.config.total_budget,
                );

                ComponentAllocation {
                    name: c.name.clone(),
                    budget: clamped,
                    precision: self.precision_for_budget(clamped, &c),
                    confidence: 0.8,
                    reason: format!("Proportional to cost {:.2}", c.compute_cost),
                }
            })
            .collect()
    }

    /// Adaptive allocation based on historical consumption.
    fn allocate_adaptive(&self, budget: f64) -> Vec<ComponentAllocation> {
        // Combine sensitivity with historical consumption
        let total_weight: f64 = self
            .components
            .iter()
            .map(|c| {
                let consumption_weight = if c.historical_consumption.is_empty() {
                    1.0
                } else {
                    // Higher consumption = needs more budget
                    (c.consumption_rate() * 100.0 + 1.0).ln()
                };
                c.sensitivity * consumption_weight
            })
            .sum();

        if total_weight == 0.0 {
            return self.allocate_weighted(budget);
        }

        self.components
            .iter()
            .map(|c| {
                let consumption_weight = if c.historical_consumption.is_empty() {
                    1.0
                } else {
                    (c.consumption_rate() * 100.0 + 1.0).ln()
                };
                let weight = (c.sensitivity * consumption_weight) / total_weight;

                // Smooth with historical allocation
                let current_alloc = c.allocated_budget;
                let new_alloc = budget * weight;
                let smoothed = self.config.smoothing_factor * current_alloc
                    + (1.0 - self.config.smoothing_factor) * new_alloc;

                let clamped = smoothed.clamp(
                    self.config.min_allocation * self.config.total_budget,
                    self.config.max_allocation * self.config.total_budget,
                );

                ComponentAllocation {
                    name: c.name.clone(),
                    budget: clamped,
                    precision: self.precision_for_budget(clamped, &c),
                    confidence: 0.9,
                    reason: format!(
                        "Adaptive: sensitivity={:.2}, consumption_rate={:.4}",
                        c.sensitivity,
                        c.consumption_rate()
                    ),
                }
            })
            .collect()
    }

    /// Hierarchical allocation by component type priority.
    fn allocate_hierarchical(&self, budget: f64) -> Vec<ComponentAllocation> {
        // Define hierarchy: loss > weight_update > gradient > output > attention > ff > embedding
        let priority_order = [
            ComponentType::Loss,
            ComponentType::WeightUpdate,
            ComponentType::Gradient,
            ComponentType::Output,
            ComponentType::Normalization,
            ComponentType::Attention,
            ComponentType::FeedForward,
            ComponentType::Embedding,
            ComponentType::Custom,
        ];

        let mut remaining = budget;
        let mut allocations = Vec::new();

        for priority_type in priority_order {
            let components_of_type: Vec<_> = self
                .components
                .iter()
                .filter(|c| c.component_type == priority_type)
                .collect();

            if components_of_type.is_empty() {
                continue;
            }

            // Allocate based on remaining budget and priority
            let priority_fraction = match priority_type {
                ComponentType::Loss | ComponentType::WeightUpdate => 0.3,
                ComponentType::Gradient | ComponentType::Output => 0.2,
                ComponentType::Normalization | ComponentType::Attention => 0.15,
                _ => 0.1,
            };

            let type_budget = remaining * priority_fraction;
            let per_component = type_budget / components_of_type.len() as f64;

            for c in components_of_type {
                let clamped = per_component.clamp(
                    self.config.min_allocation * self.config.total_budget,
                    self.config.max_allocation * self.config.total_budget,
                );

                allocations.push(ComponentAllocation {
                    name: c.name.clone(),
                    budget: clamped,
                    precision: self.precision_for_budget(clamped, c),
                    confidence: 0.85,
                    reason: format!("Hierarchical: {:?} priority", priority_type),
                });

                remaining -= clamped;
            }
        }

        allocations
    }

    /// Optimal allocation (minimizes loss impact).
    fn allocate_optimal(&self, budget: f64) -> Vec<ComponentAllocation> {
        // Use gradient descent to find optimal allocation
        // This is a simplified version; full optimization would use convex solver

        let n = self.components.len();
        if n == 0 {
            return Vec::new();
        }

        // Initial allocation
        let mut allocations: Vec<f64> = vec![budget / n as f64; n];

        // Gradient descent iterations
        const ITERATIONS: usize = 100;
        const LEARNING_RATE: f64 = 0.01;

        for _ in 0..ITERATIONS {
            // Compute gradient for each allocation
            let gradients: Vec<f64> = self
                .components
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    // Loss impact ~ sensitivity / allocation
                    // Gradient ~ -sensitivity / allocation^2
                    if allocations[i] > 1e-10 {
                        -c.sensitivity / (allocations[i] * allocations[i])
                    } else {
                        0.0
                    }
                })
                .collect();

            // Update allocations
            for i in 0..n {
                allocations[i] -= LEARNING_RATE * gradients[i];
                allocations[i] = allocations[i].max(self.config.min_allocation * self.config.total_budget);
            }

            // Project onto budget constraint
            let total: f64 = allocations.iter().sum();
            if total > budget {
                for a in &mut allocations {
                    *a *= budget / total;
                }
            }
        }

        // Create allocation results
        self.components
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let clamped = allocations[i].clamp(
                    self.config.min_allocation * self.config.total_budget,
                    self.config.max_allocation * self.config.total_budget,
                );

                ComponentAllocation {
                    name: c.name.clone(),
                    budget: clamped,
                    precision: self.precision_for_budget(clamped, c),
                    confidence: 0.9,
                    reason: "Optimal allocation (loss minimization)".to_string(),
                }
            })
            .collect()
    }

    /// Determines precision based on allocated budget.
    fn precision_for_budget(&self, budget: f64, component: &BudgetComponent) -> Precision {
        // Higher budget allows lower precision (more error tolerance)
        // Lower budget requires higher precision (less error tolerance)

        let budget_ratio = budget / self.config.total_budget;

        if budget_ratio < 0.05 || component.sensitivity > 0.9 {
            Precision::F32
        } else if budget_ratio < 0.1 || component.sensitivity > 0.7 {
            Precision::BF16
        } else if budget_ratio < 0.2 || component.sensitivity > 0.5 {
            Precision::F16
        } else {
            Precision::INT8
        }
    }

    /// Records consumption for a component.
    pub fn consume(&mut self, component_name: &str, amount: f64) {
        if let Some(component) = self.get_component_mut(component_name) {
            component.consume(amount);
            self.total_consumed += amount;
        }
    }

    /// Checks if reallocation is needed.
    pub fn should_reallocate(&self, current_step: usize) -> bool {
        current_step - self.last_update_step >= self.config.update_interval
    }

    /// Updates allocation if needed.
    pub fn maybe_reallocate(&mut self, current_step: usize) -> Option<AllocationResult> {
        if self.should_reallocate(current_step) {
            self.last_update_step = current_step;
            Some(self.allocate())
        } else {
            None
        }
    }

    /// Returns total consumed budget.
    pub fn total_consumed(&self) -> f64 {
        self.total_consumed
    }

    /// Returns remaining budget.
    pub fn remaining_budget(&self) -> f64 {
        self.config.total_budget - self.total_consumed
    }

    /// Returns current allocation.
    pub fn current_allocation(&self) -> Option<&AllocationResult> {
        self.current_allocation.as_ref()
    }

    /// Generates a summary report.
    pub fn summary(&self) -> BudgetSummary {
        let component_summaries: Vec<_> = self
            .components
            .iter()
            .map(|c| ComponentSummary {
                name: c.name.clone(),
                allocated: c.allocated_budget,
                consumed: c.consumed,
                remaining: c.remaining(),
                utilization: if c.allocated_budget > 0.0 {
                    c.consumed / c.allocated_budget
                } else {
                    0.0
                },
                is_over_budget: c.is_over_budget(),
            })
            .collect();

        BudgetSummary {
            total_budget: self.config.total_budget,
            total_consumed: self.total_consumed,
            remaining: self.remaining_budget(),
            utilization: if self.config.total_budget > 0.0 {
                self.total_consumed / self.config.total_budget
            } else {
                0.0
            },
            component_summaries,
            strategy: self.config.strategy,
        }
    }
}

/// Summary of budget state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSummary {
    pub total_budget: f64,
    pub total_consumed: f64,
    pub remaining: f64,
    pub utilization: f64,
    pub component_summaries: Vec<ComponentSummary>,
    pub strategy: AllocationStrategy,
}

/// Summary for a single component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSummary {
    pub name: String,
    pub allocated: f64,
    pub consumed: f64,
    pub remaining: f64,
    pub utilization: f64,
    pub is_over_budget: bool,
}

impl std::fmt::Display for BudgetSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Error Budget Summary")?;
        writeln!(f, "Strategy: {:?}", self.strategy)?;
        writeln!(
            f,
            "Total: {:.6} | Consumed: {:.6} | Remaining: {:.6} | Utilization: {:.1}%",
            self.total_budget,
            self.total_consumed,
            self.remaining,
            self.utilization * 100.0
        )?;
        writeln!(f, "\nComponent Details:")?;
        for c in &self.component_summaries {
            let status = if c.is_over_budget { "[OVER]" } else { "[OK]" };
            writeln!(
                f,
                "  {}: allocated={:.6}, consumed={:.6}, util={:.1}% {}",
                c.name,
                c.allocated,
                c.consumed,
                c.utilization * 100.0,
                status
            )?;
        }
        Ok(())
    }
}

/// Helper to get current timestamp.
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_allocation() {
        let config = BudgetAllocationConfig {
            total_budget: 0.1,
            strategy: AllocationStrategy::Equal,
            ..Default::default()
        };

        let mut allocator = BudgetAllocator::new(config);
        allocator.add_component(BudgetComponent::new("layer1", ComponentType::Attention));
        allocator.add_component(BudgetComponent::new("layer2", ComponentType::FeedForward));

        let result = allocator.allocate();
        assert_eq!(result.allocations.len(), 2);

        // Should be roughly equal (may differ due to clamping)
        let diff = (result.allocations[0].budget - result.allocations[1].budget).abs();
        assert!(diff < 0.01);
    }

    #[test]
    fn test_weighted_allocation() {
        let config = BudgetAllocationConfig {
            total_budget: 0.1,
            strategy: AllocationStrategy::Weighted,
            min_allocation: 0.0,
            max_allocation: 1.0,
            ..Default::default()
        };

        let mut allocator = BudgetAllocator::new(config);
        allocator.add_component(
            BudgetComponent::new("high_sens", ComponentType::Loss).with_sensitivity(0.9),
        );
        allocator.add_component(
            BudgetComponent::new("low_sens", ComponentType::Embedding).with_sensitivity(0.1),
        );

        let result = allocator.allocate();

        // High sensitivity should get more budget
        let high = result.get("high_sens").unwrap();
        let low = result.get("low_sens").unwrap();
        assert!(high.budget > low.budget);
    }

    #[test]
    fn test_consumption_tracking() {
        let config = BudgetAllocationConfig::default();
        let mut allocator = BudgetAllocator::new(config);
        allocator.add_component(BudgetComponent::new("layer1", ComponentType::Attention));

        allocator.allocate();

        allocator.consume("layer1", 0.001);
        allocator.consume("layer1", 0.002);

        let component = allocator.get_component("layer1").unwrap();
        assert!((component.consumed - 0.003).abs() < 1e-10);
        assert_eq!(component.historical_consumption.len(), 2);
    }

    #[test]
    fn test_adaptive_allocation() {
        let config = BudgetAllocationConfig {
            total_budget: 0.1,
            strategy: AllocationStrategy::Adaptive,
            update_interval: 10,
            smoothing_factor: 0.5,
            ..Default::default()
        };

        let mut allocator = BudgetAllocator::new(config);
        allocator.add_component(BudgetComponent::new("layer1", ComponentType::Attention));
        allocator.add_component(BudgetComponent::new("layer2", ComponentType::FeedForward));

        // Initial allocation
        allocator.allocate();

        // Simulate higher consumption for layer1
        for _ in 0..10 {
            allocator.consume("layer1", 0.002);
            allocator.consume("layer2", 0.0005);
        }

        // Reallocate
        let result = allocator.allocate();

        // Layer1 should have higher allocation due to higher consumption
        let l1 = result.get("layer1").unwrap();
        let l2 = result.get("layer2").unwrap();
        assert!(l1.budget >= l2.budget); // Should get at least equal
    }

    #[test]
    fn test_hierarchical_allocation() {
        let config = BudgetAllocationConfig {
            total_budget: 0.1,
            strategy: AllocationStrategy::Hierarchical,
            ..Default::default()
        };

        let mut allocator = BudgetAllocator::new(config);
        allocator.add_component(BudgetComponent::new("loss", ComponentType::Loss));
        allocator.add_component(BudgetComponent::new("attention", ComponentType::Attention));
        allocator.add_component(BudgetComponent::new("embedding", ComponentType::Embedding));

        let result = allocator.allocate();

        // Loss should have highest priority (most budget)
        let loss = result.get("loss").unwrap();
        let attention = result.get("attention").unwrap();
        let embedding = result.get("embedding").unwrap();

        assert!(loss.budget >= attention.budget);
        assert!(attention.budget >= embedding.budget);
    }

    #[test]
    fn test_summary() {
        let config = BudgetAllocationConfig::default();
        let mut allocator = BudgetAllocator::new(config);
        allocator.add_component(BudgetComponent::new("layer1", ComponentType::Attention));
        allocator.allocate();

        allocator.consume("layer1", 0.005);

        let summary = allocator.summary();
        assert_eq!(summary.component_summaries.len(), 1);
        assert!((summary.total_consumed - 0.005).abs() < 1e-10);
    }
}
