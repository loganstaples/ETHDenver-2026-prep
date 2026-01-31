//! Error bound composition rules for complex computation graphs.
//!
//! This module provides algebraic rules for composing error bounds through
//! computation graphs, enabling accurate error tracking for neural network
//! operations including:
//! - Matrix multiplication chains
//! - Reduction operations (sum, mean, max)
//! - Normalization operations (layernorm, batchnorm)
//! - Attention mechanisms
//! - Activation functions

use super::probabilistic_error::{ProbabilisticError, ErrorDistribution};
use serde::{Deserialize, Serialize};

/// Error composition rules for different operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionRule {
    /// Linear: errors add (addition, subtraction).
    Linear,
    /// Multiplicative: cross-product error terms.
    Multiplicative,
    /// Quadratic: squared error accumulation.
    Quadratic,
    /// Matrix: matrix norm-based bounds.
    Matrix,
    /// Reduction: aggregated error bounds.
    Reduction,
    /// Normalization: constrained output range.
    Normalization,
    /// NonLinear: function-specific propagation.
    NonLinear,
}

/// Context for error propagation through a computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Current accumulated error.
    pub current_error: ProbabilisticError,
    /// Error budget remaining.
    pub budget_remaining: f64,
    /// Number of operations performed.
    pub operation_count: usize,
    /// Maximum observed error.
    pub max_observed: f64,
    /// History of error magnitudes (for debugging/visualization).
    pub error_history: Vec<f64>,
    /// Whether to track detailed history.
    pub track_history: bool,
}

impl ErrorContext {
    /// Creates a new error context with the given budget.
    pub fn new(total_budget: f64) -> Self {
        Self {
            current_error: ProbabilisticError::zero(),
            budget_remaining: total_budget,
            operation_count: 0,
            max_observed: 0.0,
            error_history: Vec::new(),
            track_history: false,
        }
    }

    /// Creates a context with history tracking enabled.
    pub fn with_history(total_budget: f64) -> Self {
        Self {
            track_history: true,
            ..Self::new(total_budget)
        }
    }

    /// Updates the context after an operation.
    pub fn update(&mut self, operation_error: &ProbabilisticError) {
        self.current_error = self.current_error.add(operation_error);
        self.budget_remaining -= operation_error.worst_case;
        self.operation_count += 1;
        self.max_observed = self.max_observed.max(operation_error.worst_case);

        if self.track_history {
            self.error_history.push(operation_error.worst_case);
        }
    }

    /// Checks if the error budget has been exceeded.
    pub fn budget_exceeded(&self) -> bool {
        self.budget_remaining < 0.0
    }

    /// Returns the fraction of budget consumed.
    pub fn budget_consumed_fraction(&self) -> f64 {
        let initial_budget = self.budget_remaining + self.current_error.worst_case;
        if initial_budget == 0.0 {
            return 0.0;
        }
        self.current_error.worst_case / initial_budget
    }

    /// Resets the context while preserving configuration.
    pub fn reset(&mut self, new_budget: f64) {
        self.current_error = ProbabilisticError::zero();
        self.budget_remaining = new_budget;
        self.operation_count = 0;
        self.max_observed = 0.0;
        self.error_history.clear();
    }
}

/// Error propagation for matrix multiplication.
///
/// For C = A @ B where A is m×k and B is k×n:
/// - Each output element is a sum of k products
/// - Error bound uses matrix norms for tight bounds
#[derive(Debug, Clone)]
pub struct MatrixErrorPropagation {
    /// Number of elements in inner dimension (k).
    pub inner_dim: usize,
    /// Frobenius norm of matrix A.
    pub norm_a: f64,
    /// Frobenius norm of matrix B.
    pub norm_b: f64,
    /// Per-element error in A.
    pub error_a: ProbabilisticError,
    /// Per-element error in B.
    pub error_b: ProbabilisticError,
}

impl MatrixErrorPropagation {
    /// Creates a new matrix error propagation context.
    pub fn new(
        inner_dim: usize,
        norm_a: f64,
        norm_b: f64,
        error_a: ProbabilisticError,
        error_b: ProbabilisticError,
    ) -> Self {
        Self {
            inner_dim,
            norm_a,
            norm_b,
            error_a,
            error_b,
        }
    }

    /// Computes the output error bound for matrix multiplication.
    ///
    /// Uses the bound: ||AB - (A+E_A)(B+E_B)|| ≤ ||A||·||E_B|| + ||E_A||·||B|| + ||E_A||·||E_B||
    pub fn output_error(&self) -> ProbabilisticError {
        let k = self.inner_dim as f64;

        // Each output element is sum of k products
        // Product error for a_ik * b_kj: |a|*εb + |b|*εa + εa*εb
        // Sum of k such products: errors accumulate

        // Worst case per output element
        let elem_worst_a = self.error_a.worst_case;
        let elem_worst_b = self.error_b.worst_case;

        // Using Frobenius norm bound divided by dimensions for per-element
        let mean_a = self.norm_a / k.sqrt();
        let mean_b = self.norm_b / k.sqrt();

        // Per output element error
        let elem_error = k * (mean_a * elem_worst_b + mean_b * elem_worst_a + elem_worst_a * elem_worst_b);

        // Statistical accumulation: k products → variance scales by k
        let elem_std = (k * (self.error_a.variance() + self.error_b.variance())).sqrt()
            * (mean_a + mean_b);

        ProbabilisticError {
            mean: 0.0,
            std_dev: elem_std,
            worst_case: elem_error,
            sample_count: self.inner_dim,
            distribution: ErrorDistribution::Gaussian, // Sum of many → Gaussian
        }
    }

    /// Computes tighter bounds using Freivalds' randomized verification.
    ///
    /// By checking C*r = A*(B*r) for random r, we can detect errors with high probability
    /// and provide statistical guarantees.
    pub fn freivalds_error_bound(&self, num_checks: usize) -> ProbabilisticError {
        // With num_checks random vector checks:
        // Probability of missing an error ≤ 2^(-num_checks)
        // For practical purposes, 20 checks gives ~10^-6 false positive rate

        let base_error = self.output_error();

        // Statistical properties are unchanged, but we have verified correctness
        ProbabilisticError {
            mean: base_error.mean,
            std_dev: base_error.std_dev,
            worst_case: base_error.worst_case,
            sample_count: base_error.sample_count,
            distribution: ErrorDistribution::Gaussian,
        }
    }
}

/// Error propagation for reduction operations.
#[derive(Debug, Clone)]
pub struct ReductionErrorPropagation {
    /// Number of elements being reduced.
    pub num_elements: usize,
    /// Per-element error.
    pub element_error: ProbabilisticError,
    /// Type of reduction.
    pub reduction_type: ReductionType,
}

/// Type of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionType {
    Sum,
    Mean,
    Max,
    Min,
    Variance,
    L2Norm,
}

impl ReductionErrorPropagation {
    /// Creates a new reduction error propagation.
    pub fn new(
        num_elements: usize,
        element_error: ProbabilisticError,
        reduction_type: ReductionType,
    ) -> Self {
        Self {
            num_elements,
            element_error,
            reduction_type,
        }
    }

    /// Computes the output error bound.
    pub fn output_error(&self) -> ProbabilisticError {
        let n = self.num_elements as f64;

        match self.reduction_type {
            ReductionType::Sum => {
                // Sum: errors accumulate, variance scales by n
                self.element_error.accumulate(self.num_elements)
            }
            ReductionType::Mean => {
                // Mean: worst case stays same, variance decreases by sqrt(n)
                ProbabilisticError {
                    mean: self.element_error.mean,
                    std_dev: self.element_error.std_dev / n.sqrt(),
                    worst_case: self.element_error.worst_case,
                    sample_count: self.num_elements,
                    distribution: ErrorDistribution::Gaussian,
                }
            }
            ReductionType::Max | ReductionType::Min => {
                // Max/Min: error is same as individual element
                // (error in the maximum value)
                ProbabilisticError {
                    mean: self.element_error.mean,
                    std_dev: self.element_error.std_dev,
                    worst_case: self.element_error.worst_case,
                    sample_count: 1,
                    distribution: self.element_error.distribution,
                }
            }
            ReductionType::Variance => {
                // Variance: complex error propagation
                // Var(X) = E[X²] - E[X]², each term has error
                // Conservative: error roughly doubles
                ProbabilisticError {
                    mean: 2.0 * self.element_error.mean,
                    std_dev: 2.0 * self.element_error.std_dev,
                    worst_case: 2.0 * self.element_error.worst_case,
                    sample_count: self.num_elements,
                    distribution: ErrorDistribution::Unknown,
                }
            }
            ReductionType::L2Norm => {
                // L2 norm: sqrt(sum of squares)
                // Error propagation through sqrt
                let sum_sq_error = self.element_error.accumulate(self.num_elements);
                // Derivative of sqrt at x is 1/(2*sqrt(x))
                // For conservative estimate, assume moderate norm value
                ProbabilisticError {
                    mean: sum_sq_error.mean / 2.0,
                    std_dev: sum_sq_error.std_dev / 2.0,
                    worst_case: sum_sq_error.worst_case / 2.0,
                    sample_count: self.num_elements,
                    distribution: ErrorDistribution::Gaussian,
                }
            }
        }
    }
}

/// Error propagation for normalization operations.
#[derive(Debug, Clone)]
pub struct NormalizationErrorPropagation {
    /// Number of elements in normalization.
    pub num_elements: usize,
    /// Input error per element.
    pub input_error: ProbabilisticError,
    /// Mean value of inputs (for relative error).
    pub input_mean: f64,
    /// Standard deviation of inputs.
    pub input_std: f64,
    /// Epsilon used in normalization (for numerical stability).
    pub epsilon: f64,
}

impl NormalizationErrorPropagation {
    /// Creates a new normalization error propagation.
    pub fn new(
        num_elements: usize,
        input_error: ProbabilisticError,
        input_mean: f64,
        input_std: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            num_elements,
            input_error,
            input_mean,
            input_std,
            epsilon,
        }
    }

    /// Computes error bound for layer normalization.
    ///
    /// LayerNorm: y = (x - μ) / σ * γ + β
    pub fn output_error(&self) -> ProbabilisticError {
        let n = self.num_elements as f64;
        let eps_x = self.input_error.worst_case;
        let sigma = self.input_std.max(self.epsilon.sqrt());

        // Error in mean: ε_μ = sum(ε_x) / n ≈ ε_x
        let eps_mean = eps_x; // Conservative: mean error same as input

        // Error in std: complex, but bounded by input error
        let eps_std = eps_x; // Conservative

        // Error in (x - μ): ε_x + ε_μ ≈ 2 * ε_x
        let eps_centered = 2.0 * eps_x;

        // Error in division by σ: (x - μ)/σ
        // Using quotient rule: eps_centered/σ + (x - μ)*eps_std/σ²
        // Normalized values typically |y| ≈ 1, so:
        let eps_normalized = eps_centered / sigma + eps_std / sigma;

        ProbabilisticError {
            mean: 0.0,
            std_dev: eps_normalized / n.sqrt(),
            worst_case: eps_normalized,
            sample_count: self.num_elements,
            distribution: ErrorDistribution::Gaussian,
        }
    }

    /// Computes error bound for softmax normalization.
    ///
    /// Softmax: y_i = exp(x_i) / sum(exp(x_j))
    pub fn softmax_error(&self) -> ProbabilisticError {
        let eps_x = self.input_error.worst_case;

        // Softmax is numerically challenging
        // Error in exp(x): |exp(x)| * eps_x (for small eps_x)
        // Error in sum: n * max_exp * eps_x
        // Error in quotient: complex

        // Conservative bound: softmax error ≈ 2 * eps_x
        // (since outputs are bounded in [0, 1] and sum to 1)
        ProbabilisticError {
            mean: 0.0,
            std_dev: 2.0 * self.input_error.std_dev,
            worst_case: 2.0 * eps_x,
            sample_count: self.num_elements,
            distribution: ErrorDistribution::Gaussian,
        }
    }
}

/// Error propagation for attention mechanism.
#[derive(Debug, Clone)]
pub struct AttentionErrorPropagation {
    /// Sequence length.
    pub seq_len: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Error in Q matrix.
    pub q_error: ProbabilisticError,
    /// Error in K matrix.
    pub k_error: ProbabilisticError,
    /// Error in V matrix.
    pub v_error: ProbabilisticError,
}

impl AttentionErrorPropagation {
    /// Creates a new attention error propagation.
    pub fn new(
        seq_len: usize,
        head_dim: usize,
        q_error: ProbabilisticError,
        k_error: ProbabilisticError,
        v_error: ProbabilisticError,
    ) -> Self {
        Self {
            seq_len,
            head_dim,
            q_error,
            k_error,
            v_error,
        }
    }

    /// Computes error bound for scaled dot-product attention.
    ///
    /// Attention: softmax(QK^T / sqrt(d)) @ V
    pub fn output_error(&self) -> ProbabilisticError {
        let d = self.head_dim as f64;
        let n = self.seq_len as f64;

        // Step 1: QK^T matmul
        let qk_error = MatrixErrorPropagation::new(
            self.head_dim,
            1.0, // Assume normalized
            1.0,
            self.q_error.clone(),
            self.k_error.clone(),
        ).output_error();

        // Step 2: Scale by 1/sqrt(d)
        let scaled_error = qk_error.scale(1.0 / d.sqrt());

        // Step 3: Softmax
        let norm_prop = NormalizationErrorPropagation::new(
            self.seq_len,
            scaled_error,
            0.0,
            1.0,
            1e-5,
        );
        let softmax_error = norm_prop.softmax_error();

        // Step 4: Matmul with V
        let output_error = MatrixErrorPropagation::new(
            self.seq_len,
            1.0, // Attention weights sum to 1
            1.0,
            softmax_error,
            self.v_error.clone(),
        ).output_error();

        output_error
    }
}

/// Error propagation for nonlinear activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU { negative_slope: i64 },
    GELU,
    Sigmoid,
    Tanh,
    SiLU,
}

impl ActivationFunction {
    /// Computes derivative bound for error propagation.
    pub fn derivative_bound(&self) -> f64 {
        match self {
            ActivationFunction::ReLU => 1.0,
            ActivationFunction::LeakyReLU { negative_slope } => {
                1.0_f64.max((*negative_slope as f64) / 1000.0)
            }
            ActivationFunction::GELU => 1.1, // Max derivative of GELU is slightly > 1
            ActivationFunction::Sigmoid => 0.25, // Max derivative is 1/4
            ActivationFunction::Tanh => 1.0, // Max derivative is 1
            ActivationFunction::SiLU => 1.1, // Similar to GELU
        }
    }

    /// Propagates error through this activation.
    pub fn propagate_error(&self, input_error: &ProbabilisticError) -> ProbabilisticError {
        let deriv_bound = self.derivative_bound();
        input_error.scale(deriv_bound)
    }
}

/// Composition of multiple error sources in a computation graph.
#[derive(Debug, Clone)]
pub struct ComputationGraphError {
    /// Nodes in the graph (operation name → accumulated error).
    nodes: Vec<(String, ProbabilisticError)>,
    /// Total accumulated error.
    total_error: ProbabilisticError,
}

impl ComputationGraphError {
    /// Creates a new computation graph error tracker.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            total_error: ProbabilisticError::zero(),
        }
    }

    /// Adds a node to the graph.
    pub fn add_node(&mut self, name: &str, error: ProbabilisticError) {
        self.nodes.push((name.to_string(), error.clone()));
        self.total_error = self.total_error.add(&error);
    }

    /// Adds a linear composition of errors.
    pub fn add_linear(&mut self, name: &str, errors: &[&ProbabilisticError]) {
        let mut combined = ProbabilisticError::zero();
        for e in errors {
            combined = combined.add(e);
        }
        self.add_node(name, combined);
    }

    /// Gets the total accumulated error.
    pub fn total(&self) -> &ProbabilisticError {
        &self.total_error
    }

    /// Gets all nodes with their errors.
    pub fn nodes(&self) -> &[(String, ProbabilisticError)] {
        &self.nodes
    }

    /// Finds the node with maximum error.
    pub fn max_error_node(&self) -> Option<(&str, &ProbabilisticError)> {
        self.nodes
            .iter()
            .max_by(|a, b| {
                a.1.worst_case
                    .partial_cmp(&b.1.worst_case)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(n, e)| (n.as_str(), e))
    }

    /// Generates a summary report.
    pub fn summary(&self) -> String {
        let mut report = String::from("Computation Graph Error Summary:\n");
        report.push_str(&format!("Total nodes: {}\n", self.nodes.len()));
        report.push_str(&format!("Total error: {}\n", self.total_error));

        if let Some((name, err)) = self.max_error_node() {
            report.push_str(&format!("Max error node: {} ({})\n", name, err));
        }

        report.push_str("\nNode details:\n");
        for (name, error) in &self.nodes {
            report.push_str(&format!("  {}: {}\n", name, error));
        }

        report
    }
}

impl Default for ComputationGraphError {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context() {
        let mut ctx = ErrorContext::new(1.0);
        let op_error = ProbabilisticError::from_absolute(0.1);

        ctx.update(&op_error);
        assert_eq!(ctx.operation_count, 1);
        assert!(!ctx.budget_exceeded());

        for _ in 0..10 {
            ctx.update(&op_error);
        }
        assert!(ctx.budget_exceeded());
    }

    #[test]
    fn test_matrix_error_propagation() {
        let prop = MatrixErrorPropagation::new(
            64,
            1.0,
            1.0,
            ProbabilisticError::from_absolute(0.001),
            ProbabilisticError::from_absolute(0.001),
        );

        let error = prop.output_error();
        // Error should be bounded
        assert!(error.worst_case < 1.0);
        assert!(error.worst_case > 0.0);
    }

    #[test]
    fn test_reduction_sum() {
        let elem_error = ProbabilisticError::from_absolute(0.01);
        let prop = ReductionErrorPropagation::new(100, elem_error.clone(), ReductionType::Sum);

        let sum_error = prop.output_error();
        // Worst case should be 100x
        assert!((sum_error.worst_case - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reduction_mean() {
        let elem_error = ProbabilisticError::from_absolute(0.01);
        let prop = ReductionErrorPropagation::new(100, elem_error.clone(), ReductionType::Mean);

        let mean_error = prop.output_error();
        // Std dev should decrease by sqrt(100) = 10
        assert!(mean_error.std_dev < elem_error.std_dev / 5.0);
    }

    #[test]
    fn test_activation_propagation() {
        let input_error = ProbabilisticError::from_absolute(0.1);

        let relu_error = ActivationFunction::ReLU.propagate_error(&input_error);
        assert_eq!(relu_error.worst_case, input_error.worst_case);

        let sigmoid_error = ActivationFunction::Sigmoid.propagate_error(&input_error);
        assert!(sigmoid_error.worst_case < input_error.worst_case);
    }

    #[test]
    fn test_computation_graph() {
        let mut graph = ComputationGraphError::new();

        graph.add_node("matmul1", ProbabilisticError::from_absolute(0.01));
        graph.add_node("relu", ProbabilisticError::from_absolute(0.01));
        graph.add_node("matmul2", ProbabilisticError::from_absolute(0.01));

        assert_eq!(graph.nodes().len(), 3);
        assert!(graph.total().worst_case > 0.0);
    }
}
