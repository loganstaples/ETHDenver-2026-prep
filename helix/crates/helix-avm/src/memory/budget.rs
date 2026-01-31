//! Memory Budget Tracking and Enforcement.
//!
//! Provides fine-grained memory tracking for neural network training, ensuring
//! that memory usage stays within configurable limits.

use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during memory allocation.
#[derive(Error, Debug, Clone)]
pub enum AllocationError {
    #[error("Out of memory: requested {requested} bytes, but only {available} available (budget: {budget})")]
    OutOfMemory {
        requested: usize,
        available: usize,
        budget: usize,
    },

    #[error("Region '{name}' already exists")]
    RegionExists { name: String },

    #[error("Region '{name}' not found")]
    RegionNotFound { name: String },

    #[error("Cannot shrink region '{name}' below current usage: {current} > {target}")]
    CannotShrink {
        name: String,
        current: usize,
        target: usize,
    },
}

/// A memory region representing a category of memory usage.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Name of the region.
    pub name: String,
    /// Maximum bytes allowed for this region.
    pub limit: usize,
    /// Current bytes allocated in this region.
    pub current: usize,
    /// Peak bytes ever allocated in this region.
    pub peak: usize,
    /// Number of allocations in this region.
    pub allocation_count: usize,
    /// Whether this region is pinned (cannot be freed automatically).
    pub pinned: bool,
}

impl MemoryRegion {
    /// Creates a new memory region.
    pub fn new(name: impl Into<String>, limit: usize) -> Self {
        Self {
            name: name.into(),
            limit,
            current: 0,
            peak: 0,
            allocation_count: 0,
            pinned: false,
        }
    }

    /// Returns the available bytes in this region.
    pub fn available(&self) -> usize {
        self.limit.saturating_sub(self.current)
    }

    /// Returns the utilization ratio (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.limit == 0 {
            0.0
        } else {
            self.current as f64 / self.limit as f64
        }
    }

    /// Returns the peak utilization ratio.
    pub fn peak_utilization(&self) -> f64 {
        if self.limit == 0 {
            0.0
        } else {
            self.peak as f64 / self.limit as f64
        }
    }

    /// Pins this region (prevents automatic freeing).
    pub fn pin(&mut self) {
        self.pinned = true;
    }

    /// Unpins this region.
    pub fn unpin(&mut self) {
        self.pinned = false;
    }
}

/// Memory budget configuration.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Total memory budget in bytes.
    pub total_bytes: usize,
    /// Memory reserved for weights.
    pub weight_budget: usize,
    /// Memory reserved for activations.
    pub activation_budget: usize,
    /// Memory reserved for gradients.
    pub gradient_budget: usize,
    /// Memory reserved for optimizer state.
    pub optimizer_budget: usize,
    /// Memory reserved for workspace (temporary allocations).
    pub workspace_budget: usize,
    /// Allow exceeding budget with warnings.
    pub soft_limit: bool,
    /// Target memory utilization (0.0 to 1.0).
    pub target_utilization: f64,
}

impl MemoryBudget {
    /// Creates a new memory budget with the given total bytes.
    pub fn new(total_bytes: usize) -> Self {
        // Default allocation: 40% weights, 30% activations, 20% gradients, 10% workspace
        Self {
            total_bytes,
            weight_budget: (total_bytes as f64 * 0.4) as usize,
            activation_budget: (total_bytes as f64 * 0.3) as usize,
            gradient_budget: (total_bytes as f64 * 0.2) as usize,
            optimizer_budget: 0,
            workspace_budget: (total_bytes as f64 * 0.1) as usize,
            soft_limit: false,
            target_utilization: 0.9,
        }
    }

    /// Creates a budget optimized for inference (no gradients).
    pub fn for_inference(total_bytes: usize) -> Self {
        Self {
            total_bytes,
            weight_budget: (total_bytes as f64 * 0.7) as usize,
            activation_budget: (total_bytes as f64 * 0.25) as usize,
            gradient_budget: 0,
            optimizer_budget: 0,
            workspace_budget: (total_bytes as f64 * 0.05) as usize,
            soft_limit: false,
            target_utilization: 0.95,
        }
    }

    /// Creates a budget optimized for training.
    pub fn for_training(total_bytes: usize) -> Self {
        // Training needs: weights + gradients + optimizer state + activations
        Self {
            total_bytes,
            weight_budget: (total_bytes as f64 * 0.25) as usize,
            activation_budget: (total_bytes as f64 * 0.25) as usize,
            gradient_budget: (total_bytes as f64 * 0.25) as usize,
            optimizer_budget: (total_bytes as f64 * 0.2) as usize,
            workspace_budget: (total_bytes as f64 * 0.05) as usize,
            soft_limit: false,
            target_utilization: 0.85,
        }
    }

    /// Creates a budget for a specific model size.
    pub fn for_model_params(num_params: usize, bytes_per_param: f64, training: bool) -> Self {
        let weight_bytes = (num_params as f64 * bytes_per_param) as usize;

        if training {
            // Training: weights + gradients + optimizer (Adam needs 2x for momentum)
            let gradient_bytes = weight_bytes;
            let optimizer_bytes = weight_bytes * 2;
            let activation_bytes = weight_bytes; // Rough estimate
            let workspace_bytes = weight_bytes / 10;
            let total = weight_bytes + gradient_bytes + optimizer_bytes + activation_bytes + workspace_bytes;

            Self {
                total_bytes: total,
                weight_budget: weight_bytes,
                activation_budget: activation_bytes,
                gradient_budget: gradient_bytes,
                optimizer_budget: optimizer_bytes,
                workspace_budget: workspace_bytes,
                soft_limit: false,
                target_utilization: 0.85,
            }
        } else {
            // Inference: just weights and activations
            let activation_bytes = weight_bytes / 2;
            let workspace_bytes = weight_bytes / 20;
            let total = weight_bytes + activation_bytes + workspace_bytes;

            Self {
                total_bytes: total,
                weight_budget: weight_bytes,
                activation_budget: activation_bytes,
                gradient_budget: 0,
                optimizer_budget: 0,
                workspace_budget: workspace_bytes,
                soft_limit: false,
                target_utilization: 0.9,
            }
        }
    }

    /// Sets the budget with explicit allocations.
    pub fn with_allocations(
        mut self,
        weights: usize,
        activations: usize,
        gradients: usize,
        optimizer: usize,
        workspace: usize,
    ) -> Self {
        self.weight_budget = weights;
        self.activation_budget = activations;
        self.gradient_budget = gradients;
        self.optimizer_budget = optimizer;
        self.workspace_budget = workspace;
        self
    }

    /// Enables soft limits (warns but doesn't fail on overflow).
    pub fn with_soft_limit(mut self) -> Self {
        self.soft_limit = true;
        self
    }

    /// Returns the sum of all region budgets.
    pub fn total_allocated(&self) -> usize {
        self.weight_budget
            + self.activation_budget
            + self.gradient_budget
            + self.optimizer_budget
            + self.workspace_budget
    }

    /// Validates that region budgets don't exceed total.
    pub fn validate(&self) -> bool {
        self.total_allocated() <= self.total_bytes
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        // Default to 8GB
        Self::new(8 * 1024 * 1024 * 1024)
    }
}

/// Tracks memory usage across multiple regions.
#[derive(Debug)]
pub struct MemoryTracker {
    /// Memory budget configuration.
    budget: MemoryBudget,
    /// Named memory regions.
    regions: HashMap<String, MemoryRegion>,
    /// Total current memory usage.
    total_current: usize,
    /// Peak total memory usage.
    total_peak: usize,
    /// Number of allocation failures.
    allocation_failures: usize,
    /// Warning threshold (fraction of budget).
    warning_threshold: f64,
    /// Warnings issued.
    warnings: Vec<String>,
}

impl MemoryTracker {
    /// Creates a new memory tracker with the given budget.
    pub fn new(budget: MemoryBudget) -> Self {
        let mut tracker = Self {
            budget: budget.clone(),
            regions: HashMap::new(),
            total_current: 0,
            total_peak: 0,
            allocation_failures: 0,
            warning_threshold: 0.8,
            warnings: Vec::new(),
        };

        // Initialize default regions
        tracker
            .regions
            .insert("weights".to_string(), MemoryRegion::new("weights", budget.weight_budget));
        tracker.regions.insert(
            "activations".to_string(),
            MemoryRegion::new("activations", budget.activation_budget),
        );
        tracker.regions.insert(
            "gradients".to_string(),
            MemoryRegion::new("gradients", budget.gradient_budget),
        );
        tracker.regions.insert(
            "optimizer".to_string(),
            MemoryRegion::new("optimizer", budget.optimizer_budget),
        );
        tracker.regions.insert(
            "workspace".to_string(),
            MemoryRegion::new("workspace", budget.workspace_budget),
        );

        tracker
    }

    /// Returns the memory budget.
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Returns current total memory usage.
    pub fn current_usage(&self) -> usize {
        self.total_current
    }

    /// Returns peak total memory usage.
    pub fn peak_usage(&self) -> usize {
        self.total_peak
    }

    /// Returns available memory.
    pub fn available(&self) -> usize {
        self.budget.total_bytes.saturating_sub(self.total_current)
    }

    /// Returns current utilization ratio.
    pub fn utilization(&self) -> f64 {
        if self.budget.total_bytes == 0 {
            0.0
        } else {
            self.total_current as f64 / self.budget.total_bytes as f64
        }
    }

    /// Creates a custom memory region.
    pub fn create_region(&mut self, name: &str, limit: usize) -> Result<(), AllocationError> {
        if self.regions.contains_key(name) {
            return Err(AllocationError::RegionExists {
                name: name.to_string(),
            });
        }

        self.regions
            .insert(name.to_string(), MemoryRegion::new(name, limit));
        Ok(())
    }

    /// Allocates memory in a region.
    pub fn allocate(&mut self, region: &str, bytes: usize) -> Result<(), AllocationError> {
        let region_obj = self.regions.get_mut(region).ok_or(AllocationError::RegionNotFound {
            name: region.to_string(),
        })?;

        // Check region limit
        if region_obj.current + bytes > region_obj.limit {
            if !self.budget.soft_limit {
                self.allocation_failures += 1;
                return Err(AllocationError::OutOfMemory {
                    requested: bytes,
                    available: region_obj.available(),
                    budget: region_obj.limit,
                });
            } else {
                self.warnings.push(format!(
                    "Region '{}' exceeded budget: {} + {} > {}",
                    region, region_obj.current, bytes, region_obj.limit
                ));
            }
        }

        // Check total limit
        if self.total_current + bytes > self.budget.total_bytes {
            if !self.budget.soft_limit {
                self.allocation_failures += 1;
                return Err(AllocationError::OutOfMemory {
                    requested: bytes,
                    available: self.available(),
                    budget: self.budget.total_bytes,
                });
            } else {
                self.warnings.push(format!(
                    "Total memory exceeded budget: {} + {} > {}",
                    self.total_current, bytes, self.budget.total_bytes
                ));
            }
        }

        // Perform allocation
        region_obj.current += bytes;
        region_obj.allocation_count += 1;
        if region_obj.current > region_obj.peak {
            region_obj.peak = region_obj.current;
        }

        self.total_current += bytes;
        if self.total_current > self.total_peak {
            self.total_peak = self.total_current;
        }

        // Check warning threshold
        let utilization = self.utilization();
        if utilization > self.warning_threshold {
            self.warnings.push(format!(
                "Memory utilization at {:.1}% ({} / {} bytes)",
                utilization * 100.0,
                self.total_current,
                self.budget.total_bytes
            ));
        }

        Ok(())
    }

    /// Frees memory in a region.
    pub fn free(&mut self, region: &str, bytes: usize) -> Result<(), AllocationError> {
        let region_obj = self.regions.get_mut(region).ok_or(AllocationError::RegionNotFound {
            name: region.to_string(),
        })?;

        region_obj.current = region_obj.current.saturating_sub(bytes);
        self.total_current = self.total_current.saturating_sub(bytes);

        Ok(())
    }

    /// Frees all memory in a region.
    pub fn free_region(&mut self, region: &str) -> Result<usize, AllocationError> {
        let region_obj = self.regions.get_mut(region).ok_or(AllocationError::RegionNotFound {
            name: region.to_string(),
        })?;

        if region_obj.pinned {
            return Ok(0);
        }

        let freed = region_obj.current;
        region_obj.current = 0;
        self.total_current = self.total_current.saturating_sub(freed);

        Ok(freed)
    }

    /// Returns a reference to a region.
    pub fn region(&self, name: &str) -> Option<&MemoryRegion> {
        self.regions.get(name)
    }

    /// Returns a mutable reference to a region.
    pub fn region_mut(&mut self, name: &str) -> Option<&mut MemoryRegion> {
        self.regions.get_mut(name)
    }

    /// Returns all regions.
    pub fn regions(&self) -> &HashMap<String, MemoryRegion> {
        &self.regions
    }

    /// Returns the number of allocation failures.
    pub fn allocation_failures(&self) -> usize {
        self.allocation_failures
    }

    /// Returns accumulated warnings.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Clears warnings.
    pub fn clear_warnings(&mut self) {
        self.warnings.clear();
    }

    /// Resets all tracking (but keeps regions).
    pub fn reset(&mut self) {
        for region in self.regions.values_mut() {
            region.current = 0;
            region.peak = 0;
            region.allocation_count = 0;
        }
        self.total_current = 0;
        self.total_peak = 0;
        self.allocation_failures = 0;
        self.warnings.clear();
    }

    /// Returns a summary of memory usage.
    pub fn summary(&self) -> MemorySummary {
        let region_summaries: Vec<RegionSummary> = self
            .regions
            .values()
            .map(|r| RegionSummary {
                name: r.name.clone(),
                current: r.current,
                peak: r.peak,
                limit: r.limit,
                utilization: r.utilization(),
                allocation_count: r.allocation_count,
            })
            .collect();

        MemorySummary {
            total_current: self.total_current,
            total_peak: self.total_peak,
            total_budget: self.budget.total_bytes,
            utilization: self.utilization(),
            regions: region_summaries,
            allocation_failures: self.allocation_failures,
            warning_count: self.warnings.len(),
        }
    }

    /// Checks if an allocation would fit.
    pub fn can_allocate(&self, region: &str, bytes: usize) -> bool {
        if let Some(region_obj) = self.regions.get(region) {
            region_obj.current + bytes <= region_obj.limit
                && self.total_current + bytes <= self.budget.total_bytes
        } else {
            false
        }
    }

    /// Estimates memory needed for a model.
    pub fn estimate_model_memory(
        num_params: usize,
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
        num_layers: usize,
        training: bool,
    ) -> ModelMemoryEstimate {
        // Weight memory (assuming f32 = 4 bytes)
        let weight_bytes = num_params * 4;

        // Activation memory per layer (rough estimate)
        // Each layer stores: input, attention scores, attention output, FFN intermediate
        let activation_per_layer = batch_size * seq_len * d_model * 4 * 4; // 4 tensors
        let activation_bytes = activation_per_layer * num_layers;

        // Gradient memory (same as weights for full precision)
        let gradient_bytes = if training { weight_bytes } else { 0 };

        // Optimizer state (Adam: 2x weights for momentum and variance)
        let optimizer_bytes = if training { weight_bytes * 2 } else { 0 };

        // Workspace (temporary buffers, attention matrices, etc.)
        let workspace_bytes = batch_size * seq_len * seq_len * 4; // Attention matrix

        let total = weight_bytes + activation_bytes + gradient_bytes + optimizer_bytes + workspace_bytes;

        ModelMemoryEstimate {
            weight_bytes,
            activation_bytes,
            gradient_bytes,
            optimizer_bytes,
            workspace_bytes,
            total_bytes: total,
            recommended_budget: (total as f64 * 1.2) as usize, // 20% headroom
        }
    }
}

/// Summary of memory usage.
#[derive(Debug, Clone)]
pub struct MemorySummary {
    /// Current total memory usage.
    pub total_current: usize,
    /// Peak total memory usage.
    pub total_peak: usize,
    /// Total memory budget.
    pub total_budget: usize,
    /// Current utilization.
    pub utilization: f64,
    /// Per-region summaries.
    pub regions: Vec<RegionSummary>,
    /// Number of allocation failures.
    pub allocation_failures: usize,
    /// Number of warnings issued.
    pub warning_count: usize,
}

/// Summary of a single region.
#[derive(Debug, Clone)]
pub struct RegionSummary {
    /// Region name.
    pub name: String,
    /// Current usage.
    pub current: usize,
    /// Peak usage.
    pub peak: usize,
    /// Region limit.
    pub limit: usize,
    /// Utilization.
    pub utilization: f64,
    /// Number of allocations.
    pub allocation_count: usize,
}

/// Estimated memory requirements for a model.
#[derive(Debug, Clone)]
pub struct ModelMemoryEstimate {
    /// Bytes for weights.
    pub weight_bytes: usize,
    /// Bytes for activations.
    pub activation_bytes: usize,
    /// Bytes for gradients.
    pub gradient_bytes: usize,
    /// Bytes for optimizer state.
    pub optimizer_bytes: usize,
    /// Bytes for workspace.
    pub workspace_bytes: usize,
    /// Total bytes.
    pub total_bytes: usize,
    /// Recommended budget (with headroom).
    pub recommended_budget: usize,
}

impl ModelMemoryEstimate {
    /// Formats the estimate as a human-readable string.
    pub fn format(&self) -> String {
        format!(
            "Memory Estimate:\n  Weights: {:.2} MB\n  Activations: {:.2} MB\n  Gradients: {:.2} MB\n  Optimizer: {:.2} MB\n  Workspace: {:.2} MB\n  Total: {:.2} MB\n  Recommended Budget: {:.2} MB",
            self.weight_bytes as f64 / 1024.0 / 1024.0,
            self.activation_bytes as f64 / 1024.0 / 1024.0,
            self.gradient_bytes as f64 / 1024.0 / 1024.0,
            self.optimizer_bytes as f64 / 1024.0 / 1024.0,
            self.workspace_bytes as f64 / 1024.0 / 1024.0,
            self.total_bytes as f64 / 1024.0 / 1024.0,
            self.recommended_budget as f64 / 1024.0 / 1024.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_creation() {
        let budget = MemoryBudget::new(1024 * 1024 * 1024); // 1GB
        assert!(budget.validate());
        assert_eq!(budget.total_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_tracker_allocation() {
        let budget = MemoryBudget::new(1024 * 1024); // 1MB
        let mut tracker = MemoryTracker::new(budget);

        // Allocate some weights
        tracker.allocate("weights", 100_000).unwrap();
        assert_eq!(tracker.current_usage(), 100_000);

        // Free them
        tracker.free("weights", 50_000).unwrap();
        assert_eq!(tracker.current_usage(), 50_000);
    }

    #[test]
    fn test_memory_tracker_overflow() {
        let budget = MemoryBudget::new(1024); // 1KB
        let mut tracker = MemoryTracker::new(budget);

        // Try to allocate more than budget
        let result = tracker.allocate("weights", 10_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_tracker_soft_limit() {
        let budget = MemoryBudget::new(1024).with_soft_limit();
        let mut tracker = MemoryTracker::new(budget);

        // Should succeed with warning
        tracker.allocate("weights", 10_000).unwrap();
        assert!(!tracker.warnings().is_empty());
    }

    #[test]
    fn test_memory_region() {
        let mut region = MemoryRegion::new("test", 1000);
        assert_eq!(region.available(), 1000);
        assert_eq!(region.utilization(), 0.0);

        region.current = 500;
        assert_eq!(region.available(), 500);
        assert_eq!(region.utilization(), 0.5);
    }

    #[test]
    fn test_model_memory_estimate() {
        let estimate = MemoryTracker::estimate_model_memory(
            1_000_000, // 1M params
            4,         // batch size
            128,       // seq len
            256,       // d_model
            6,         // num layers
            true,      // training
        );

        // Weights should be ~4MB
        assert!(estimate.weight_bytes >= 4_000_000);

        // Total should include all components
        assert!(estimate.total_bytes > estimate.weight_bytes);
    }

    #[test]
    fn test_memory_budget_for_model() {
        let budget = MemoryBudget::for_model_params(500_000, 4.0, true);
        assert!(budget.validate());
        assert!(budget.weight_budget >= 2_000_000);
    }
}
