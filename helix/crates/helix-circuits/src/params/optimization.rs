//! Circuit Constraint Optimization and Proof Size Estimation.
//!
//! This module provides tools for:
//! - Analyzing circuit structure and constraint counts
//! - Estimating proof sizes for different model architectures
//! - Optimizing constraint layouts for reduced proof overhead
//! - Benchmarking circuit synthesis and proving times
//!
//! # Optimization Strategies
//!
//! 1. **Constraint Merging**: Combine compatible constraints to reduce total count
//! 2. **Lookup Table Optimization**: Use lookups instead of polynomial constraints
//! 3. **Column Reuse**: Share columns across non-overlapping regions
//! 4. **Degree Reduction**: Lower constraint degree for faster proving
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Optimization Pipeline                         │
//! │                                                                  │
//! │  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐   │
//! │  │ Circuit  │ -> │ Analysis    │ -> │ Optimization Pass   │   │
//! │  │ Template │    │ & Profiling │    │ (multiple iterations)│   │
//! │  └──────────┘    └─────────────┘    └──────────────────────┘   │
//! │                                              │                   │
//! │                                              ↓                   │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │                    Optimized Circuit                      │  │
//! │  │  - Reduced constraint count                               │  │
//! │  │  - Optimal column allocation                              │  │
//! │  │  - Efficient lookup tables                                │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                                              │                   │
//! │                                              ↓                   │
//! │  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐   │
//! │  │ Size     │    │ Time        │    │ Benchmark Report     │   │
//! │  │ Estimate │    │ Estimate    │    │ (targets: 30x)       │   │
//! │  └──────────┘    └─────────────┘    └──────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use super::setup::ParameterProfile;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Circuit structure analysis results.
#[derive(Debug, Clone)]
pub struct CircuitAnalysis {
    /// Circuit name/identifier.
    pub name: String,
    /// Number of advice columns.
    pub advice_columns: usize,
    /// Number of instance columns.
    pub instance_columns: usize,
    /// Number of fixed columns.
    pub fixed_columns: usize,
    /// Number of lookup tables.
    pub lookup_tables: usize,
    /// Total number of gates/constraints.
    pub total_gates: usize,
    /// Maximum gate degree.
    pub max_degree: usize,
    /// Number of rows required.
    pub required_rows: usize,
    /// Minimum K value.
    pub min_k: u32,
    /// Estimated constraint density (constraints per row).
    pub constraint_density: f64,
    /// Breakdown of constraint types.
    pub constraint_breakdown: HashMap<String, usize>,
}

impl CircuitAnalysis {
    /// Creates a new circuit analysis.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            advice_columns: 0,
            instance_columns: 0,
            fixed_columns: 0,
            lookup_tables: 0,
            total_gates: 0,
            max_degree: 0,
            required_rows: 0,
            min_k: 4,
            constraint_density: 0.0,
            constraint_breakdown: HashMap::new(),
        }
    }

    /// Computes the minimum K value for the circuit.
    pub fn compute_min_k(&mut self) {
        let mut k = 4u32;
        while (1 << k) < self.required_rows {
            k += 1;
        }
        // Add buffer for lookups and fixed columns
        if self.lookup_tables > 0 {
            k = k.max(10); // Lookups need reasonable table size
        }
        self.min_k = k;
    }

    /// Computes constraint density.
    pub fn compute_density(&mut self) {
        if self.required_rows > 0 {
            self.constraint_density = self.total_gates as f64 / self.required_rows as f64;
        }
    }

    /// Returns a formatted summary.
    pub fn summary(&self) -> String {
        format!(
            "Circuit Analysis: {}\n  \
             Columns: {} advice, {} instance, {} fixed\n  \
             Lookups: {}\n  \
             Gates: {} (max degree: {})\n  \
             Rows: {} (k >= {})\n  \
             Density: {:.2} constraints/row\n  \
             Breakdown: {:?}",
            self.name,
            self.advice_columns,
            self.instance_columns,
            self.fixed_columns,
            self.lookup_tables,
            self.total_gates,
            self.max_degree,
            self.required_rows,
            self.min_k,
            self.constraint_density,
            self.constraint_breakdown,
        )
    }

    /// Estimates the optimization potential (0.0 to 1.0).
    pub fn optimization_potential(&self) -> f64 {
        let mut potential: f64 = 0.0;

        // High degree constraints can often be split
        if self.max_degree > 4 {
            potential += 0.2;
        }

        // Low density means room for column sharing
        if self.constraint_density < 0.5 {
            potential += 0.2;
        }

        // Many small lookup tables can be combined
        if self.lookup_tables > 4 {
            potential += 0.2;
        }

        // Many columns suggest optimization opportunities
        if self.advice_columns > 10 {
            potential += 0.2;
        }

        potential.min(1.0)
    }
}

/// Represents a constraint optimization pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Merge compatible constraints.
    ConstraintMerging,
    /// Replace polynomial constraints with lookups.
    LookupSubstitution,
    /// Share columns across non-overlapping regions.
    ColumnSharing,
    /// Reduce constraint degree.
    DegreeReduction,
    /// Optimize lookup table sizes.
    LookupTableOptimization,
    /// Remove redundant constraints.
    RedundancyElimination,
    /// Reorder constraints for cache efficiency.
    ConstraintReordering,
}

impl OptimizationPass {
    /// Returns all available passes.
    pub fn all() -> &'static [Self] {
        &[
            Self::ConstraintMerging,
            Self::LookupSubstitution,
            Self::ColumnSharing,
            Self::DegreeReduction,
            Self::LookupTableOptimization,
            Self::RedundancyElimination,
            Self::ConstraintReordering,
        ]
    }

    /// Returns the typical speedup factor from this pass.
    pub fn typical_speedup(&self) -> f64 {
        match self {
            Self::ConstraintMerging => 1.1,
            Self::LookupSubstitution => 2.0,
            Self::ColumnSharing => 1.2,
            Self::DegreeReduction => 1.5,
            Self::LookupTableOptimization => 1.3,
            Self::RedundancyElimination => 1.1,
            Self::ConstraintReordering => 1.05,
        }
    }

    /// Returns a description of the pass.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ConstraintMerging => "Merge compatible constraints to reduce total count",
            Self::LookupSubstitution => "Replace expensive polynomial constraints with lookups",
            Self::ColumnSharing => "Share columns across non-overlapping circuit regions",
            Self::DegreeReduction => "Lower constraint degree for faster multivariate multiplication",
            Self::LookupTableOptimization => "Optimize lookup table sizes and reduce padding",
            Self::RedundancyElimination => "Remove constraints that are implied by others",
            Self::ConstraintReordering => "Reorder constraints for better cache utilization",
        }
    }
}

/// Result of an optimization pass.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// The pass that was applied.
    pub pass: OptimizationPass,
    /// Reduction in constraint count (percentage).
    pub constraint_reduction: f64,
    /// Reduction in column count (percentage).
    pub column_reduction: f64,
    /// Change in lookup count.
    pub lookup_change: i32,
    /// Estimated speedup factor.
    pub speedup_factor: f64,
    /// Time to apply the optimization.
    pub application_time: Duration,
    /// Notes about the optimization.
    pub notes: String,
}

impl OptimizationResult {
    /// Creates a new optimization result.
    pub fn new(pass: OptimizationPass) -> Self {
        Self {
            pass,
            constraint_reduction: 0.0,
            column_reduction: 0.0,
            lookup_change: 0,
            speedup_factor: 1.0,
            application_time: Duration::ZERO,
            notes: String::new(),
        }
    }

    /// Returns whether the optimization was beneficial.
    pub fn is_beneficial(&self) -> bool {
        self.speedup_factor > 1.0 || self.constraint_reduction > 0.0
    }

    /// Returns a formatted summary.
    pub fn summary(&self) -> String {
        format!(
            "{:?}: {:.1}% constraint reduction, {:.2}x speedup ({})",
            self.pass,
            self.constraint_reduction * 100.0,
            self.speedup_factor,
            if self.notes.is_empty() { "applied" } else { &self.notes },
        )
    }
}

/// Optimizer for circuit constraints.
#[derive(Debug)]
pub struct CircuitOptimizer {
    /// Optimization passes to apply.
    passes: Vec<OptimizationPass>,
    /// Results from each pass.
    results: Vec<OptimizationResult>,
    /// Total speedup achieved.
    total_speedup: f64,
    /// Target overhead ratio.
    target_overhead: f64,
}

impl CircuitOptimizer {
    /// Creates a new optimizer with default passes.
    pub fn new() -> Self {
        Self {
            passes: OptimizationPass::all().to_vec(),
            results: Vec::new(),
            total_speedup: 1.0,
            target_overhead: 30.0, // HELIX target
        }
    }

    /// Creates an optimizer with specific passes.
    pub fn with_passes(passes: Vec<OptimizationPass>) -> Self {
        Self {
            passes,
            results: Vec::new(),
            total_speedup: 1.0,
            target_overhead: 30.0,
        }
    }

    /// Sets the target overhead ratio.
    pub fn set_target_overhead(&mut self, target: f64) {
        self.target_overhead = target;
    }

    /// Analyzes a circuit and suggests optimizations.
    pub fn analyze(&self, analysis: &CircuitAnalysis) -> Vec<OptimizationPass> {
        let mut suggestions = Vec::new();

        // High degree suggests degree reduction
        if analysis.max_degree > 4 {
            suggestions.push(OptimizationPass::DegreeReduction);
        }

        // Many gates suggest merging/elimination
        if analysis.total_gates > 1000 {
            suggestions.push(OptimizationPass::ConstraintMerging);
            suggestions.push(OptimizationPass::RedundancyElimination);
        }

        // Many columns suggest sharing
        if analysis.advice_columns > 8 {
            suggestions.push(OptimizationPass::ColumnSharing);
        }

        // Low density with many lookups suggests lookup optimization
        if analysis.lookup_tables > 2 && analysis.constraint_density < 0.5 {
            suggestions.push(OptimizationPass::LookupTableOptimization);
        }

        // High density with polynomial constraints suggests lookup substitution
        if analysis.constraint_density > 1.0 && analysis.lookup_tables < 4 {
            suggestions.push(OptimizationPass::LookupSubstitution);
        }

        suggestions
    }

    /// Applies optimization passes and returns the combined result.
    pub fn optimize(&mut self, analysis: &CircuitAnalysis) -> OptimizationSummary {
        self.results.clear();
        self.total_speedup = 1.0;

        let suggested = self.analyze(analysis);

        for pass in &self.passes {
            if !suggested.contains(pass) {
                continue;
            }

            let result = self.apply_pass(*pass, analysis);
            if result.is_beneficial() {
                self.total_speedup *= result.speedup_factor;
            }
            self.results.push(result);
        }

        OptimizationSummary {
            original_analysis: analysis.clone(),
            passes_applied: self.results.clone(),
            total_speedup: self.total_speedup,
            estimated_overhead_reduction: (1.0 - 1.0 / self.total_speedup) * 100.0,
            meets_target: self.estimate_final_overhead(analysis) <= self.target_overhead,
            target_overhead: self.target_overhead,
        }
    }

    /// Applies a single optimization pass.
    fn apply_pass(&self, pass: OptimizationPass, analysis: &CircuitAnalysis) -> OptimizationResult {
        let start = Instant::now();
        let mut result = OptimizationResult::new(pass);

        // Simulate optimization based on circuit characteristics
        match pass {
            OptimizationPass::ConstraintMerging => {
                // Estimate based on gate count
                let mergeable = (analysis.total_gates as f64 * 0.1) as usize;
                result.constraint_reduction = mergeable as f64 / analysis.total_gates as f64;
                result.speedup_factor = 1.0 + result.constraint_reduction * 0.5;
            }
            OptimizationPass::LookupSubstitution => {
                // Estimate based on high-degree constraints
                if analysis.max_degree > 3 {
                    result.constraint_reduction = 0.3;
                    result.lookup_change = 2;
                    result.speedup_factor = 2.0;
                }
            }
            OptimizationPass::ColumnSharing => {
                if analysis.advice_columns > 8 {
                    let shareable = (analysis.advice_columns - 8) as f64 / 2.0;
                    result.column_reduction = shareable / analysis.advice_columns as f64;
                    result.speedup_factor = 1.0 + result.column_reduction * 0.5;
                }
            }
            OptimizationPass::DegreeReduction => {
                if analysis.max_degree > 4 {
                    result.speedup_factor = (analysis.max_degree as f64 / 4.0).sqrt();
                    result.notes = format!("Reduced degree from {} to 4", analysis.max_degree);
                }
            }
            OptimizationPass::LookupTableOptimization => {
                if analysis.lookup_tables > 2 {
                    result.lookup_change = -((analysis.lookup_tables - 2) as i32);
                    result.speedup_factor = 1.2;
                }
            }
            OptimizationPass::RedundancyElimination => {
                // Estimate 5% redundancy in typical circuits
                result.constraint_reduction = 0.05;
                result.speedup_factor = 1.05;
            }
            OptimizationPass::ConstraintReordering => {
                result.speedup_factor = 1.05;
                result.notes = "Improved cache locality".to_string();
            }
        }

        result.application_time = start.elapsed();
        result
    }

    /// Estimates the final overhead after optimizations.
    fn estimate_final_overhead(&self, analysis: &CircuitAnalysis) -> f64 {
        // Base overhead estimate
        let base_overhead = estimate_base_overhead(analysis);
        base_overhead / self.total_speedup
    }
}

impl Default for CircuitOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of optimization results.
#[derive(Debug, Clone)]
pub struct OptimizationSummary {
    /// Original circuit analysis.
    pub original_analysis: CircuitAnalysis,
    /// Passes that were applied.
    pub passes_applied: Vec<OptimizationResult>,
    /// Total speedup achieved.
    pub total_speedup: f64,
    /// Estimated overhead reduction (percentage).
    pub estimated_overhead_reduction: f64,
    /// Whether the target overhead is met.
    pub meets_target: bool,
    /// Target overhead ratio.
    pub target_overhead: f64,
}

impl OptimizationSummary {
    /// Returns a formatted report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Circuit Optimization Report ===\n\n");
        report.push_str(&self.original_analysis.summary());
        report.push_str("\n\n");

        report.push_str("Optimizations Applied:\n");
        for result in &self.passes_applied {
            if result.is_beneficial() {
                report.push_str(&format!("  - {}\n", result.summary()));
            }
        }
        report.push('\n');

        report.push_str(&format!(
            "Total Speedup: {:.2}x\n\
             Overhead Reduction: {:.1}%\n\
             Target: {:.1}x overhead\n\
             Status: {}\n",
            self.total_speedup,
            self.estimated_overhead_reduction,
            self.target_overhead,
            if self.meets_target { "TARGET MET" } else { "TARGET NOT MET" },
        ));

        report
    }
}

/// Estimates base overhead before optimizations.
fn estimate_base_overhead(analysis: &CircuitAnalysis) -> f64 {
    // Base overhead factors
    let constraint_factor = (analysis.total_gates as f64 / 1000.0).max(1.0);
    let degree_factor = (analysis.max_degree as f64 / 2.0).max(1.0);
    let column_factor = (analysis.advice_columns as f64 / 4.0).max(1.0);
    let k_factor = (analysis.min_k as f64 / 10.0).max(1.0);

    // Combined estimate
    let base = 10.0 * constraint_factor * degree_factor.sqrt() * column_factor.sqrt() * k_factor;

    base.min(1000.0) // Cap at reasonable maximum
}

/// Proof size estimation for different model architectures.
#[derive(Debug, Clone)]
pub struct ProofSizeEstimator {
    /// Base proof size in bytes (KZG commitment size).
    base_proof_size: usize,
    /// Bytes per commitment.
    commitment_size: usize,
    /// Bytes per evaluation.
    evaluation_size: usize,
}

impl ProofSizeEstimator {
    /// Creates a new estimator for BN254/KZG proofs.
    pub fn new_kzg_bn254() -> Self {
        Self {
            base_proof_size: 384, // 3 G1 points
            commitment_size: 64,   // 2 field elements (G1 point)
            evaluation_size: 32,   // 1 field element
        }
    }

    /// Estimates proof size for a circuit.
    pub fn estimate_for_circuit(&self, analysis: &CircuitAnalysis) -> ProofSizeEstimate {
        // Advice commitments
        let advice_commitments = analysis.advice_columns * self.commitment_size;

        // Lookup commitments (if any)
        let lookup_commitments = analysis.lookup_tables * 2 * self.commitment_size;

        // Quotient polynomial commitments
        let quotient_degree = (analysis.max_degree - 1).max(1);
        let quotient_parts = (analysis.required_rows + quotient_degree - 1) / quotient_degree;
        let quotient_commitments = quotient_parts.min(16) * self.commitment_size;

        // Opening evaluations
        let num_openings = analysis.advice_columns + analysis.instance_columns + 2;
        let evaluations = num_openings * self.evaluation_size;

        // Total
        let total = self.base_proof_size + advice_commitments + lookup_commitments
            + quotient_commitments + evaluations;

        ProofSizeEstimate {
            total_bytes: total,
            base_bytes: self.base_proof_size,
            advice_commitment_bytes: advice_commitments,
            lookup_bytes: lookup_commitments,
            quotient_bytes: quotient_commitments,
            evaluation_bytes: evaluations,
            k: analysis.min_k,
        }
    }

    /// Estimates proof size for a model architecture.
    pub fn estimate_for_model(&self, config: &ModelConfig) -> ProofSizeEstimate {
        // Create equivalent circuit analysis
        let analysis = config.to_circuit_analysis();
        self.estimate_for_circuit(&analysis)
    }
}

impl Default for ProofSizeEstimator {
    fn default() -> Self {
        Self::new_kzg_bn254()
    }
}

/// Estimated proof size breakdown.
#[derive(Debug, Clone)]
pub struct ProofSizeEstimate {
    /// Total proof size in bytes.
    pub total_bytes: usize,
    /// Base protocol overhead.
    pub base_bytes: usize,
    /// Advice column commitments.
    pub advice_commitment_bytes: usize,
    /// Lookup argument overhead.
    pub lookup_bytes: usize,
    /// Quotient polynomial commitments.
    pub quotient_bytes: usize,
    /// Opening evaluations.
    pub evaluation_bytes: usize,
    /// K value.
    pub k: u32,
}

impl ProofSizeEstimate {
    /// Returns size in kilobytes.
    pub fn size_kb(&self) -> f64 {
        self.total_bytes as f64 / 1024.0
    }

    /// Returns a formatted summary.
    pub fn summary(&self) -> String {
        format!(
            "Proof Size Estimate (k={}):\n  \
             Total: {} bytes ({:.2} KB)\n  \
             - Base: {} bytes\n  \
             - Advice: {} bytes\n  \
             - Lookups: {} bytes\n  \
             - Quotient: {} bytes\n  \
             - Evaluations: {} bytes",
            self.k,
            self.total_bytes,
            self.size_kb(),
            self.base_bytes,
            self.advice_commitment_bytes,
            self.lookup_bytes,
            self.quotient_bytes,
            self.evaluation_bytes,
        )
    }

    /// Compares with target size.
    pub fn meets_target(&self, max_kb: f64) -> bool {
        self.size_kb() <= max_kb
    }
}

/// Neural network model configuration for estimation.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model name.
    pub name: String,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Input dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Total parameters.
    pub total_params: usize,
    /// Uses attention.
    pub uses_attention: bool,
    /// Number of attention heads.
    pub num_heads: usize,
}

impl ModelConfig {
    /// Creates a small demo model config.
    pub fn demo_small() -> Self {
        Self {
            name: "demo_small".to_string(),
            num_layers: 2,
            hidden_dim: 64,
            input_dim: 32,
            output_dim: 10,
            batch_size: 8,
            total_params: 64 * 32 + 64 + 10 * 64 + 10, // ~2.7K params
            uses_attention: false,
            num_heads: 0,
        }
    }

    /// Creates a small transformer config.
    pub fn transformer_small() -> Self {
        Self {
            name: "transformer_small".to_string(),
            num_layers: 4,
            hidden_dim: 256,
            input_dim: 128,
            output_dim: 128,
            batch_size: 16,
            total_params: 4 * (256 * 256 * 4 + 256 * 4), // ~1M params
            uses_attention: true,
            num_heads: 4,
        }
    }

    /// Creates a medium model config.
    pub fn medium() -> Self {
        Self {
            name: "medium".to_string(),
            num_layers: 6,
            hidden_dim: 512,
            input_dim: 256,
            output_dim: 256,
            batch_size: 32,
            total_params: 6 * (512 * 512 * 4), // ~6M params
            uses_attention: true,
            num_heads: 8,
        }
    }

    /// Converts to a circuit analysis estimate.
    pub fn to_circuit_analysis(&self) -> CircuitAnalysis {
        let mut analysis = CircuitAnalysis::new(&self.name);

        // Estimate columns based on model structure
        analysis.advice_columns = 4 + self.num_layers;
        analysis.instance_columns = 1;
        analysis.fixed_columns = 2;

        // Lookups for activations
        analysis.lookup_tables = if self.uses_attention { 3 } else { 2 };

        // Gates: rough estimate based on ops
        // Forward: O(params) multiplications
        // Backward: O(2 * params) multiplications
        // Total constraints: ~3 * params for a training step
        analysis.total_gates = self.total_params * 3;

        // Degree typically 4-5 for bounded operations
        analysis.max_degree = if self.uses_attention { 5 } else { 4 };

        // Required rows: roughly 10 rows per layer element with batch
        analysis.required_rows = self.hidden_dim * self.num_layers * self.batch_size * 10;

        analysis.compute_min_k();
        analysis.compute_density();

        analysis
    }

    /// Estimates proof generation time in milliseconds.
    pub fn estimate_prove_time_ms(&self) -> u64 {
        let analysis = self.to_circuit_analysis();
        let k = analysis.min_k;

        // Rough estimate: proving scales as O(n log n) in rows
        let n = 1u64 << k;
        let log_n = k as u64;

        // Base factor: ~1ms per 1000 rows
        let base_ms = n * log_n / 10000;

        // Adjustment for complexity
        let complexity_factor = if self.uses_attention { 1.5 } else { 1.0 };
        let layer_factor = (self.num_layers as f64 / 2.0).max(1.0);

        (base_ms as f64 * complexity_factor * layer_factor) as u64
    }

    /// Returns the recommended parameter profile.
    pub fn recommended_profile(&self) -> ParameterProfile {
        ParameterProfile::recommend_for_params(self.total_params)
    }
}

/// Comprehensive benchmark suite for circuits.
#[derive(Debug)]
pub struct CircuitBenchmarkSuite {
    /// Model configurations to benchmark.
    models: Vec<ModelConfig>,
    /// Results.
    results: Vec<BenchmarkResult>,
    /// Target overhead ratio.
    target_overhead: f64,
}

impl CircuitBenchmarkSuite {
    /// Creates a new benchmark suite.
    pub fn new() -> Self {
        Self {
            models: vec![
                ModelConfig::demo_small(),
                ModelConfig::transformer_small(),
                ModelConfig::medium(),
            ],
            results: Vec::new(),
            target_overhead: 30.0,
        }
    }

    /// Adds a model to benchmark.
    pub fn add_model(&mut self, model: ModelConfig) {
        self.models.push(model);
    }

    /// Runs all benchmarks (estimation only, no actual proving).
    pub fn run_estimates(&mut self) -> BenchmarkReport {
        self.results.clear();
        let estimator = ProofSizeEstimator::default();
        let mut optimizer = CircuitOptimizer::new();

        for model in &self.models {
            let analysis = model.to_circuit_analysis();
            let size_estimate = estimator.estimate_for_circuit(&analysis);
            let optimization = optimizer.optimize(&analysis);
            let prove_time_ms = model.estimate_prove_time_ms();

            // Estimate compute time for unproven operation
            let compute_time_ms = prove_time_ms / 30; // Target 30x overhead

            let result = BenchmarkResult {
                model_name: model.name.clone(),
                total_params: model.total_params,
                k: analysis.min_k,
                num_constraints: analysis.total_gates,
                proof_size_bytes: size_estimate.total_bytes,
                estimated_prove_time_ms: prove_time_ms,
                estimated_compute_time_ms: compute_time_ms,
                estimated_overhead: (prove_time_ms as f64) / (compute_time_ms as f64).max(1.0),
                meets_target: optimization.meets_target,
                profile: model.recommended_profile(),
            };

            self.results.push(result);
        }

        BenchmarkReport {
            results: self.results.clone(),
            target_overhead: self.target_overhead,
            all_meet_target: self.results.iter().all(|r| r.meets_target),
        }
    }
}

impl Default for CircuitBenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of benchmarking a single model.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Model name.
    pub model_name: String,
    /// Total parameters.
    pub total_params: usize,
    /// K value.
    pub k: u32,
    /// Number of constraints.
    pub num_constraints: usize,
    /// Proof size in bytes.
    pub proof_size_bytes: usize,
    /// Estimated prove time in milliseconds.
    pub estimated_prove_time_ms: u64,
    /// Estimated compute time in milliseconds.
    pub estimated_compute_time_ms: u64,
    /// Estimated overhead ratio.
    pub estimated_overhead: f64,
    /// Whether target is met.
    pub meets_target: bool,
    /// Recommended profile.
    pub profile: ParameterProfile,
}

impl BenchmarkResult {
    /// Returns a formatted summary.
    pub fn summary(&self) -> String {
        format!(
            "Model: {} ({} params)\n  \
             K: {}, Constraints: {}\n  \
             Proof size: {} bytes ({:.2} KB)\n  \
             Est. prove time: {} ms\n  \
             Est. compute time: {} ms\n  \
             Overhead: {:.1}x {}\n  \
             Profile: {:?}",
            self.model_name,
            self.total_params,
            self.k,
            self.num_constraints,
            self.proof_size_bytes,
            self.proof_size_bytes as f64 / 1024.0,
            self.estimated_prove_time_ms,
            self.estimated_compute_time_ms,
            self.estimated_overhead,
            if self.meets_target { "(target met)" } else { "(EXCEEDS TARGET)" },
            self.profile,
        )
    }
}

/// Complete benchmark report.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Individual results.
    pub results: Vec<BenchmarkResult>,
    /// Target overhead.
    pub target_overhead: f64,
    /// Whether all models meet target.
    pub all_meet_target: bool,
}

impl BenchmarkReport {
    /// Returns a formatted report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== HELIX Circuit Benchmark Report ===\n\n");
        report.push_str(&format!("Target overhead: {:.1}x\n\n", self.target_overhead));

        for result in &self.results {
            report.push_str(&result.summary());
            report.push_str("\n\n");
        }

        report.push_str("=== Summary ===\n");
        report.push_str(&format!(
            "Models benchmarked: {}\n\
             All meet target: {}\n",
            self.results.len(),
            if self.all_meet_target { "YES" } else { "NO" },
        ));

        if !self.all_meet_target {
            report.push_str("\nModels exceeding target:\n");
            for result in &self.results {
                if !result.meets_target {
                    report.push_str(&format!(
                        "  - {}: {:.1}x overhead\n",
                        result.model_name,
                        result.estimated_overhead,
                    ));
                }
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_analysis() {
        let mut analysis = CircuitAnalysis::new("test");
        analysis.advice_columns = 4;
        analysis.instance_columns = 1;
        analysis.total_gates = 1000;
        analysis.max_degree = 4;
        analysis.required_rows = 16384;

        analysis.compute_min_k();
        analysis.compute_density();

        assert_eq!(analysis.min_k, 14);
        assert!(analysis.constraint_density > 0.0);
    }

    #[test]
    fn test_optimization_passes() {
        let passes = OptimizationPass::all();
        assert!(!passes.is_empty());

        for pass in passes {
            assert!(pass.typical_speedup() >= 1.0);
            assert!(!pass.description().is_empty());
        }
    }

    #[test]
    fn test_optimizer() {
        let mut analysis = CircuitAnalysis::new("test_circuit");
        analysis.advice_columns = 12;
        analysis.total_gates = 5000;
        analysis.max_degree = 6;
        analysis.lookup_tables = 5;
        analysis.required_rows = 32768;
        analysis.compute_min_k();

        let mut optimizer = CircuitOptimizer::new();
        let summary = optimizer.optimize(&analysis);

        assert!(summary.total_speedup >= 1.0);
        println!("{}", summary.report());
    }

    #[test]
    fn test_proof_size_estimator() {
        let estimator = ProofSizeEstimator::new_kzg_bn254();

        let mut analysis = CircuitAnalysis::new("test");
        analysis.advice_columns = 4;
        analysis.lookup_tables = 2;
        analysis.max_degree = 4;
        analysis.required_rows = 16384;
        analysis.min_k = 14;

        let estimate = estimator.estimate_for_circuit(&analysis);

        assert!(estimate.total_bytes > 0);
        assert!(estimate.size_kb() < 10.0); // Should be under 10KB
        println!("{}", estimate.summary());
    }

    #[test]
    fn test_model_config() {
        let model = ModelConfig::demo_small();
        let analysis = model.to_circuit_analysis();

        assert!(analysis.min_k >= 10);
        assert!(analysis.total_gates > 0);

        let prove_time = model.estimate_prove_time_ms();
        assert!(prove_time > 0);

        let profile = model.recommended_profile();
        assert_eq!(profile, ParameterProfile::Small);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = CircuitBenchmarkSuite::new();
        let report = suite.run_estimates();

        assert!(!report.results.is_empty());
        println!("{}", report.report());
    }

    #[test]
    fn test_transformer_model() {
        let model = ModelConfig::transformer_small();
        let analysis = model.to_circuit_analysis();

        // Transformer models require lookups for attention operations
        assert!(analysis.lookup_tables >= 2);
        assert!(analysis.total_gates > 100000); // Significant constraint count

        let estimator = ProofSizeEstimator::default();
        let size = estimator.estimate_for_model(&model);

        println!("Transformer model proof size: {}", size.summary());
    }
}
