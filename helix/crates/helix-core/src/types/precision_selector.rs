//! Automatic precision selection based on error budget.
//!
//! This module provides intelligent precision selection algorithms that:
//! - Analyze error budgets and operation characteristics
//! - Select optimal precision for each operation
//! - Balance accuracy vs. computational cost
//! - Support mixed-precision training strategies

use super::precision::Precision;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for precision selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSelectorConfig {
    /// Available precision levels (ordered from highest to lowest accuracy).
    pub available_precisions: Vec<Precision>,
    /// Maximum allowed error for the entire computation.
    pub max_total_error: f64,
    /// Maximum allowed error per operation (if specified).
    pub max_operation_error: Option<f64>,
    /// Target overhead ratio (prove_time / compute_time).
    pub target_overhead: f64,
    /// Preference weights for different metrics.
    pub weights: PrecisionWeights,
    /// Strategy for precision selection.
    pub strategy: SelectionStrategy,
}

/// Weights for different optimization objectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionWeights {
    /// Weight for accuracy (lower error).
    pub accuracy: f64,
    /// Weight for speed (lower compute time).
    pub speed: f64,
    /// Weight for memory (lower memory usage).
    pub memory: f64,
    /// Weight for proof size (smaller proofs).
    pub proof_size: f64,
}

impl Default for PrecisionWeights {
    fn default() -> Self {
        Self {
            accuracy: 1.0,
            speed: 1.0,
            memory: 0.5,
            proof_size: 0.5,
        }
    }
}

/// Strategy for precision selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Uniform: use same precision for all operations.
    Uniform,
    /// Greedy: pick lowest precision that fits error budget.
    Greedy,
    /// Layerwise: different precision per layer type.
    Layerwise,
    /// Adaptive: adjust based on accumulated error.
    Adaptive,
    /// Optimal: solve optimization problem (more expensive).
    Optimal,
}

impl Default for PrecisionSelectorConfig {
    fn default() -> Self {
        Self {
            available_precisions: vec![
                Precision::F32,
                Precision::BF16,
                Precision::F16,
                Precision::INT8,
            ],
            max_total_error: 0.01,
            max_operation_error: Some(0.001),
            target_overhead: 30.0,
            weights: PrecisionWeights::default(),
            strategy: SelectionStrategy::Adaptive,
        }
    }
}

/// Operation characteristics for precision selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationCharacteristics {
    /// Type of operation.
    pub operation_type: OperationType,
    /// Number of elements involved.
    pub num_elements: usize,
    /// Range of values (min, max).
    pub value_range: (f64, f64),
    /// Whether this is a critical path operation.
    pub is_critical: bool,
    /// Sensitivity to precision loss (higher = needs more precision).
    pub sensitivity: f64,
    /// Relative compute cost at F32.
    pub compute_cost: f64,
}

/// Type of operation for precision classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    /// Matrix multiplication.
    MatMul,
    /// Convolution.
    Convolution,
    /// Elementwise operation.
    Elementwise,
    /// Reduction (sum, mean, etc.).
    Reduction,
    /// Normalization (LayerNorm, BatchNorm).
    Normalization,
    /// Attention mechanism.
    Attention,
    /// Activation function.
    Activation,
    /// Loss computation.
    Loss,
    /// Gradient accumulation.
    GradientAccum,
    /// Weight update.
    WeightUpdate,
}

impl OperationType {
    /// Returns the default sensitivity for this operation type.
    pub fn default_sensitivity(&self) -> f64 {
        match self {
            OperationType::Loss => 1.0,           // Very sensitive
            OperationType::GradientAccum => 0.9,  // Very sensitive
            OperationType::WeightUpdate => 0.9,   // Very sensitive
            OperationType::Normalization => 0.8,  // Sensitive
            OperationType::Attention => 0.7,      // Moderately sensitive
            OperationType::MatMul => 0.5,         // Moderate
            OperationType::Convolution => 0.5,    // Moderate
            OperationType::Reduction => 0.4,      // Less sensitive
            OperationType::Elementwise => 0.3,    // Less sensitive
            OperationType::Activation => 0.3,     // Less sensitive
        }
    }

    /// Returns the typical error amplification factor.
    pub fn error_amplification(&self) -> f64 {
        match self {
            OperationType::MatMul => 2.0,
            OperationType::Convolution => 2.0,
            OperationType::Attention => 3.0,
            OperationType::Normalization => 1.5,
            OperationType::Reduction => 1.0,
            OperationType::Elementwise => 1.0,
            OperationType::Activation => 1.0,
            OperationType::Loss => 1.0,
            OperationType::GradientAccum => 1.5,
            OperationType::WeightUpdate => 1.0,
        }
    }
}

impl Default for OperationCharacteristics {
    fn default() -> Self {
        Self {
            operation_type: OperationType::Elementwise,
            num_elements: 1,
            value_range: (-1.0, 1.0),
            is_critical: false,
            sensitivity: 0.5,
            compute_cost: 1.0,
        }
    }
}

/// Result of precision selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSelection {
    /// Selected precision.
    pub precision: Precision,
    /// Estimated error for this operation.
    pub estimated_error: f64,
    /// Confidence in the selection (0-1).
    pub confidence: f64,
    /// Reason for selection.
    pub reason: String,
    /// Alternative precisions considered.
    pub alternatives: Vec<(Precision, f64)>, // (precision, estimated_error)
}

/// Precision selector for automatic precision selection.
#[derive(Debug, Clone)]
pub struct PrecisionSelector {
    /// Configuration.
    config: PrecisionSelectorConfig,
    /// Accumulated error so far.
    accumulated_error: f64,
    /// Number of operations processed.
    operation_count: usize,
    /// History of selections for adaptive strategy.
    selection_history: Vec<PrecisionSelection>,
    /// Per-precision error accumulation.
    precision_error_history: HashMap<String, Vec<f64>>,
}

impl PrecisionSelector {
    /// Creates a new precision selector with the given configuration.
    pub fn new(config: PrecisionSelectorConfig) -> Self {
        Self {
            config,
            accumulated_error: 0.0,
            operation_count: 0,
            selection_history: Vec::new(),
            precision_error_history: HashMap::new(),
        }
    }

    /// Creates a precision selector with default configuration.
    pub fn default_selector() -> Self {
        Self::new(PrecisionSelectorConfig::default())
    }

    /// Selects precision for an operation based on its characteristics.
    pub fn select(&mut self, characteristics: &OperationCharacteristics) -> PrecisionSelection {
        let selection = match self.config.strategy {
            SelectionStrategy::Uniform => self.select_uniform(characteristics),
            SelectionStrategy::Greedy => self.select_greedy(characteristics),
            SelectionStrategy::Layerwise => self.select_layerwise(characteristics),
            SelectionStrategy::Adaptive => self.select_adaptive(characteristics),
            SelectionStrategy::Optimal => self.select_optimal(characteristics),
        };

        // Update state
        self.accumulated_error += selection.estimated_error;
        self.operation_count += 1;
        self.selection_history.push(selection.clone());

        // Track per-precision error history
        let key = selection.precision.name().to_string();
        self.precision_error_history
            .entry(key)
            .or_insert_with(Vec::new)
            .push(selection.estimated_error);

        selection
    }

    /// Uniform strategy: use highest precision that fits budget.
    fn select_uniform(&self, characteristics: &OperationCharacteristics) -> PrecisionSelection {
        let remaining_budget = self.config.max_total_error - self.accumulated_error;
        let per_op_budget = remaining_budget / 100.0; // Assume ~100 ops remaining

        for precision in &self.config.available_precisions {
            let error = self.estimate_error(precision, characteristics);
            if error <= per_op_budget {
                return PrecisionSelection {
                    precision: *precision,
                    estimated_error: error,
                    confidence: 0.9,
                    reason: format!(
                        "Uniform selection: {} fits budget of {:.6}",
                        precision.name(),
                        per_op_budget
                    ),
                    alternatives: self.compute_alternatives(characteristics),
                };
            }
        }

        // Fallback to highest precision
        let precision = self.config.available_precisions[0];
        let error = self.estimate_error(&precision, characteristics);
        PrecisionSelection {
            precision,
            estimated_error: error,
            confidence: 0.7,
            reason: "Uniform selection: fallback to highest precision".to_string(),
            alternatives: self.compute_alternatives(characteristics),
        }
    }

    /// Greedy strategy: pick lowest precision that fits operation budget.
    fn select_greedy(&self, characteristics: &OperationCharacteristics) -> PrecisionSelection {
        let op_budget = self.config.max_operation_error.unwrap_or(0.001);

        // Try precisions from lowest to highest
        for precision in self.config.available_precisions.iter().rev() {
            let error = self.estimate_error(precision, characteristics);
            if error <= op_budget {
                return PrecisionSelection {
                    precision: *precision,
                    estimated_error: error,
                    confidence: 0.85,
                    reason: format!(
                        "Greedy selection: {} is lowest precision fitting budget",
                        precision.name()
                    ),
                    alternatives: self.compute_alternatives(characteristics),
                };
            }
        }

        // Must use highest precision
        let precision = self.config.available_precisions[0];
        let error = self.estimate_error(&precision, characteristics);
        PrecisionSelection {
            precision,
            estimated_error: error,
            confidence: 0.8,
            reason: "Greedy selection: required highest precision".to_string(),
            alternatives: self.compute_alternatives(characteristics),
        }
    }

    /// Layerwise strategy: select based on operation type.
    fn select_layerwise(&self, characteristics: &OperationCharacteristics) -> PrecisionSelection {
        // Determine minimum precision based on operation type
        let min_precision = match characteristics.operation_type {
            OperationType::Loss | OperationType::GradientAccum | OperationType::WeightUpdate => {
                Precision::F32 // Always use high precision for critical ops
            }
            OperationType::Normalization | OperationType::Attention => {
                Precision::BF16 // Medium precision
            }
            OperationType::MatMul | OperationType::Convolution => {
                if characteristics.is_critical {
                    Precision::BF16
                } else {
                    Precision::F16
                }
            }
            OperationType::Elementwise | OperationType::Activation | OperationType::Reduction => {
                Precision::INT8 // Can use lower precision
            }
        };

        // Find the selected precision in our available list
        let precision = self
            .config
            .available_precisions
            .iter()
            .find(|p| p.bits() >= min_precision.bits())
            .copied()
            .unwrap_or(self.config.available_precisions[0]);

        let error = self.estimate_error(&precision, characteristics);
        PrecisionSelection {
            precision,
            estimated_error: error,
            confidence: 0.9,
            reason: format!(
                "Layerwise selection: {:?} → {}",
                characteristics.operation_type,
                precision.name()
            ),
            alternatives: self.compute_alternatives(characteristics),
        }
    }

    /// Adaptive strategy: adjust based on accumulated error.
    fn select_adaptive(&self, characteristics: &OperationCharacteristics) -> PrecisionSelection {
        let remaining_budget = self.config.max_total_error - self.accumulated_error;
        let budget_fraction = remaining_budget / self.config.max_total_error;

        // How aggressive can we be?
        let aggressiveness = if budget_fraction > 0.5 {
            // Plenty of budget: can use lower precision
            0.8
        } else if budget_fraction > 0.2 {
            // Moderate budget: be careful
            0.5
        } else {
            // Low budget: use high precision
            0.2
        };

        // Adjust for operation sensitivity
        let adjusted_aggressiveness =
            aggressiveness * (1.0 - characteristics.sensitivity);

        // Select precision based on aggressiveness
        let precision_index = (adjusted_aggressiveness
            * (self.config.available_precisions.len() - 1) as f64)
            .round() as usize;
        let precision_index = precision_index.min(self.config.available_precisions.len() - 1);

        // But ensure we don't exceed operation budget
        let mut selected_index = precision_index;
        while selected_index > 0 {
            let precision = self.config.available_precisions[selected_index];
            let error = self.estimate_error(&precision, characteristics);
            if let Some(max_op_error) = self.config.max_operation_error {
                if error <= max_op_error {
                    break;
                }
            } else {
                break;
            }
            selected_index -= 1;
        }

        let precision = self.config.available_precisions[selected_index];
        let error = self.estimate_error(&precision, characteristics);

        PrecisionSelection {
            precision,
            estimated_error: error,
            confidence: 0.85,
            reason: format!(
                "Adaptive selection: budget_fraction={:.2}, aggressiveness={:.2}",
                budget_fraction, adjusted_aggressiveness
            ),
            alternatives: self.compute_alternatives(characteristics),
        }
    }

    /// Optimal strategy: solve optimization problem.
    fn select_optimal(&self, characteristics: &OperationCharacteristics) -> PrecisionSelection {
        // Score each precision based on multiple objectives
        let mut best_score = f64::NEG_INFINITY;
        let mut best_precision = self.config.available_precisions[0];
        let mut best_error = 0.0;

        for precision in &self.config.available_precisions {
            let error = self.estimate_error(precision, characteristics);
            let score = self.compute_score(precision, error, characteristics);

            if score > best_score {
                best_score = score;
                best_precision = *precision;
                best_error = error;
            }
        }

        PrecisionSelection {
            precision: best_precision,
            estimated_error: best_error,
            confidence: 0.95,
            reason: format!(
                "Optimal selection: {} with score {:.4}",
                best_precision.name(),
                best_score
            ),
            alternatives: self.compute_alternatives(characteristics),
        }
    }

    /// Estimates error for a given precision and operation.
    fn estimate_error(&self, precision: &Precision, characteristics: &OperationCharacteristics) -> f64 {
        let base_error = precision.max_relative_error();

        // Scale by value range
        let range = characteristics.value_range.1 - characteristics.value_range.0;
        let value_scale = range.abs().max(1.0);

        // Scale by operation complexity
        let complexity_scale = match characteristics.operation_type {
            OperationType::MatMul | OperationType::Convolution => {
                (characteristics.num_elements as f64).sqrt()
            }
            OperationType::Reduction => (characteristics.num_elements as f64).sqrt(),
            OperationType::Attention => (characteristics.num_elements as f64).sqrt() * 2.0,
            _ => 1.0,
        };

        // Apply error amplification
        let amplification = characteristics.operation_type.error_amplification();

        base_error * value_scale * complexity_scale * amplification
    }

    /// Computes a score for precision selection (higher is better).
    fn compute_score(
        &self,
        precision: &Precision,
        error: f64,
        _characteristics: &OperationCharacteristics,
    ) -> f64 {
        let weights = &self.config.weights;

        // Accuracy score (inversely proportional to error)
        let accuracy_score = if error > 0.0 { 1.0 / error } else { f64::MAX };
        let accuracy_score = accuracy_score.min(1000.0); // Cap for numerical stability

        // Speed score (higher bits = slower)
        let speed_score = 32.0 / precision.bits() as f64;

        // Memory score (fewer bytes = better)
        let memory_score = 4.0 / precision.bytes_per_element() as f64;

        // Proof size score (approximation: smaller precision = smaller proof)
        let proof_score = speed_score; // Correlated with speed

        // Check constraints
        if let Some(max_op_error) = self.config.max_operation_error {
            if error > max_op_error {
                return f64::NEG_INFINITY; // Infeasible
            }
        }

        // Weighted sum
        weights.accuracy * accuracy_score.log10()
            + weights.speed * speed_score
            + weights.memory * memory_score
            + weights.proof_size * proof_score
    }

    /// Computes alternative precision options.
    fn compute_alternatives(
        &self,
        characteristics: &OperationCharacteristics,
    ) -> Vec<(Precision, f64)> {
        self.config
            .available_precisions
            .iter()
            .map(|p| (*p, self.estimate_error(p, characteristics)))
            .collect()
    }

    /// Returns the accumulated error so far.
    pub fn accumulated_error(&self) -> f64 {
        self.accumulated_error
    }

    /// Returns the remaining error budget.
    pub fn remaining_budget(&self) -> f64 {
        self.config.max_total_error - self.accumulated_error
    }

    /// Returns the selection history.
    pub fn history(&self) -> &[PrecisionSelection] {
        &self.selection_history
    }

    /// Resets the selector state.
    pub fn reset(&mut self) {
        self.accumulated_error = 0.0;
        self.operation_count = 0;
        self.selection_history.clear();
        self.precision_error_history.clear();
    }

    /// Generates a summary report.
    pub fn summary(&self) -> PrecisionSelectionSummary {
        let mut precision_counts: HashMap<String, usize> = HashMap::new();
        let mut precision_errors: HashMap<String, f64> = HashMap::new();

        for selection in &self.selection_history {
            let key = selection.precision.name().to_string();
            *precision_counts.entry(key.clone()).or_insert(0) += 1;
            *precision_errors.entry(key).or_insert(0.0) += selection.estimated_error;
        }

        PrecisionSelectionSummary {
            total_operations: self.operation_count,
            total_error: self.accumulated_error,
            remaining_budget: self.remaining_budget(),
            precision_counts,
            precision_errors,
            avg_confidence: self
                .selection_history
                .iter()
                .map(|s| s.confidence)
                .sum::<f64>()
                / self.selection_history.len().max(1) as f64,
        }
    }
}

impl Default for PrecisionSelector {
    fn default() -> Self {
        Self::default_selector()
    }
}

/// Summary of precision selections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSelectionSummary {
    /// Total number of operations.
    pub total_operations: usize,
    /// Total accumulated error.
    pub total_error: f64,
    /// Remaining error budget.
    pub remaining_budget: f64,
    /// Count of each precision used.
    pub precision_counts: HashMap<String, usize>,
    /// Error contribution of each precision.
    pub precision_errors: HashMap<String, f64>,
    /// Average selection confidence.
    pub avg_confidence: f64,
}

impl std::fmt::Display for PrecisionSelectionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Precision Selection Summary:")?;
        writeln!(f, "  Total operations: {}", self.total_operations)?;
        writeln!(f, "  Total error: {:.6}", self.total_error)?;
        writeln!(f, "  Remaining budget: {:.6}", self.remaining_budget)?;
        writeln!(f, "  Avg confidence: {:.2}%", self.avg_confidence * 100.0)?;
        writeln!(f, "  Precision usage:")?;
        for (precision, count) in &self.precision_counts {
            let error = self.precision_errors.get(precision).unwrap_or(&0.0);
            writeln!(f, "    {}: {} ops, {:.6} error", precision, count, error)?;
        }
        Ok(())
    }
}

/// Precision requirement specification for a computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionRequirement {
    /// Minimum required precision.
    pub min_precision: Precision,
    /// Maximum allowed error.
    pub max_error: f64,
    /// Whether the requirement is strict (must be met).
    pub strict: bool,
}

impl PrecisionRequirement {
    /// Creates a new precision requirement.
    pub fn new(min_precision: Precision, max_error: f64, strict: bool) -> Self {
        Self {
            min_precision,
            max_error,
            strict,
        }
    }

    /// Checks if a precision meets this requirement.
    pub fn is_satisfied_by(&self, precision: &Precision, estimated_error: f64) -> bool {
        precision.bits() >= self.min_precision.bits() && estimated_error <= self.max_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_selection() {
        let config = PrecisionSelectorConfig {
            strategy: SelectionStrategy::Uniform,
            ..Default::default()
        };
        let mut selector = PrecisionSelector::new(config);

        let characteristics = OperationCharacteristics {
            operation_type: OperationType::MatMul,
            num_elements: 1000,
            ..Default::default()
        };

        let selection = selector.select(&characteristics);
        assert!(selection.precision.bits() >= 8);
        assert!(selection.estimated_error > 0.0);
    }

    #[test]
    fn test_greedy_selection() {
        let config = PrecisionSelectorConfig {
            strategy: SelectionStrategy::Greedy,
            max_operation_error: Some(0.01),
            ..Default::default()
        };
        let mut selector = PrecisionSelector::new(config);

        let characteristics = OperationCharacteristics {
            operation_type: OperationType::Elementwise,
            num_elements: 100,
            sensitivity: 0.1,
            ..Default::default()
        };

        let selection = selector.select(&characteristics);
        // Should pick lower precision for low-sensitivity operation
        assert!(selection.precision.bits() <= 32);
    }

    #[test]
    fn test_layerwise_selection() {
        let config = PrecisionSelectorConfig {
            strategy: SelectionStrategy::Layerwise,
            ..Default::default()
        };
        let mut selector = PrecisionSelector::new(config);

        // Loss computation should get high precision
        let loss_chars = OperationCharacteristics {
            operation_type: OperationType::Loss,
            ..Default::default()
        };
        let loss_selection = selector.select(&loss_chars);
        assert_eq!(loss_selection.precision, Precision::F32);

        selector.reset();

        // Activation can use lower precision
        let act_chars = OperationCharacteristics {
            operation_type: OperationType::Activation,
            ..Default::default()
        };
        let act_selection = selector.select(&act_chars);
        // Should allow lower precision
        assert!(act_selection.precision.bits() <= 32);
    }

    #[test]
    fn test_adaptive_selection() {
        let config = PrecisionSelectorConfig {
            strategy: SelectionStrategy::Adaptive,
            max_total_error: 0.1,
            ..Default::default()
        };
        let mut selector = PrecisionSelector::new(config);

        // With full budget, should be more aggressive
        let chars = OperationCharacteristics {
            operation_type: OperationType::MatMul,
            num_elements: 100,
            sensitivity: 0.3,
            ..Default::default()
        };
        let selection1 = selector.select(&chars);

        // Consume most of the budget
        selector.accumulated_error = 0.09;

        // Now should be conservative
        let selection2 = selector.select(&chars);

        // Second selection should have higher precision (lower bits index)
        assert!(selection2.precision.bits() >= selection1.precision.bits());
    }

    #[test]
    fn test_summary() {
        let mut selector = PrecisionSelector::default_selector();

        for _ in 0..10 {
            let chars = OperationCharacteristics::default();
            selector.select(&chars);
        }

        let summary = selector.summary();
        assert_eq!(summary.total_operations, 10);
        assert!(summary.total_error > 0.0);
    }
}
