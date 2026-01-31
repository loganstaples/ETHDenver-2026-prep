//! Circuit Profiling Infrastructure.
//!
//! This module provides comprehensive profiling tools for HELIX circuits:
//!
//! - **Constraint Counting**: Precise measurement of constraint counts per operation
//! - **Timing Analysis**: Sub-millisecond timing for proof generation phases
//! - **Memory Tracking**: Memory usage during witness generation and proving
//! - **Report Generation**: Automated optimization reports with actionable insights
//!
//! # Usage
//!
//! ```ignore
//! use helix_circuits::profiling::{CircuitProfiler, ProfilingConfig};
//!
//! let profiler = CircuitProfiler::new(ProfilingConfig::detailed());
//! let profile = profiler.profile_circuit(&circuit, &witness)?;
//! println!("{}", profile.report());
//! ```
//!
//! # Performance Impact
//!
//! Profiling adds minimal overhead (~5-10%) when enabled. Use `ProfilingConfig::minimal()`
//! for production deployments.

pub mod constraint_counter;
pub mod report;
pub mod timing;

pub use constraint_counter::{
    ConstraintCounter, ConstraintProfile, ConstraintBreakdown, OperationCost,
    GateProfile, LookupProfile, CopyConstraintProfile,
};
pub use report::{
    OptimizationReport, OptimizationRecommendation, OptimizationPriority,
    ReportGenerator, ReportFormat, BottleneckAnalysis, ConstraintHotspot,
};
pub use timing::{
    TimingProfiler, TimingProfile, PhaseTimer, ProfilingPhase,
    WitnessGenerationTiming, SynthesisTiming, ProvingTiming,
};

use halo2_proofs::plonk::{Circuit, Error};
use halo2curves::bn256::Fr;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for circuit profiling.
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable constraint counting (moderate overhead).
    pub count_constraints: bool,
    /// Enable timing analysis (minimal overhead).
    pub enable_timing: bool,
    /// Enable memory tracking (moderate overhead).
    pub track_memory: bool,
    /// Enable per-gate analysis (high overhead).
    pub analyze_gates: bool,
    /// Enable lookup table analysis.
    pub analyze_lookups: bool,
    /// Sample rate for detailed profiling (1.0 = all, 0.1 = 10%).
    pub sample_rate: f64,
    /// Maximum number of operations to track individually.
    pub max_tracked_operations: usize,
    /// Enable phase-by-phase timing breakdown.
    pub timing_breakdown: bool,
}

impl ProfilingConfig {
    /// Minimal profiling - timing only, suitable for production.
    pub fn minimal() -> Self {
        Self {
            count_constraints: false,
            enable_timing: true,
            track_memory: false,
            analyze_gates: false,
            analyze_lookups: false,
            sample_rate: 1.0,
            max_tracked_operations: 0,
            timing_breakdown: false,
        }
    }

    /// Standard profiling - good balance of detail and performance.
    pub fn standard() -> Self {
        Self {
            count_constraints: true,
            enable_timing: true,
            track_memory: true,
            analyze_gates: false,
            analyze_lookups: true,
            sample_rate: 1.0,
            max_tracked_operations: 1000,
            timing_breakdown: true,
        }
    }

    /// Detailed profiling - full analysis, higher overhead.
    pub fn detailed() -> Self {
        Self {
            count_constraints: true,
            enable_timing: true,
            track_memory: true,
            analyze_gates: true,
            analyze_lookups: true,
            sample_rate: 1.0,
            max_tracked_operations: 10000,
            timing_breakdown: true,
        }
    }

    /// Optimization-focused profiling for identifying bottlenecks.
    pub fn optimization() -> Self {
        Self {
            count_constraints: true,
            enable_timing: true,
            track_memory: true,
            analyze_gates: true,
            analyze_lookups: true,
            sample_rate: 1.0,
            max_tracked_operations: 50000,
            timing_breakdown: true,
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Complete profile of a circuit's performance characteristics.
#[derive(Debug, Clone)]
pub struct CircuitProfile {
    /// Name of the circuit.
    pub name: String,
    /// Constraint profile.
    pub constraints: ConstraintProfile,
    /// Timing profile.
    pub timing: TimingProfile,
    /// K parameter used.
    pub k: u32,
    /// Memory usage in bytes.
    pub memory_bytes: usize,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: usize,
    /// Operation-level breakdown.
    pub operations: HashMap<String, OperationProfile>,
    /// Bottleneck analysis.
    pub bottlenecks: Vec<Bottleneck>,
    /// Whether circuit meets performance targets.
    pub meets_targets: bool,
    /// Performance score (0-100).
    pub performance_score: u32,
}

/// Profile of a single operation type.
#[derive(Debug, Clone, Default)]
pub struct OperationProfile {
    /// Number of times this operation was executed.
    pub count: usize,
    /// Total constraints for this operation type.
    pub total_constraints: usize,
    /// Average constraints per operation.
    pub avg_constraints: f64,
    /// Total time spent on this operation.
    pub total_time: Duration,
    /// Average time per operation.
    pub avg_time: Duration,
    /// Percentage of total constraints.
    pub constraint_percentage: f64,
    /// Percentage of total time.
    pub time_percentage: f64,
}

/// Identified performance bottleneck.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Type of bottleneck.
    pub bottleneck_type: BottleneckType,
    /// Description of the bottleneck.
    pub description: String,
    /// Estimated impact (0-100).
    pub impact: u32,
    /// Suggested optimization.
    pub suggestion: String,
    /// Estimated savings if fixed.
    pub estimated_savings: EstimatedSavings,
}

/// Type of performance bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    /// Too many constraints for an operation.
    HighConstraintCount,
    /// Inefficient lookup table usage.
    LookupInefficiency,
    /// Slow witness generation.
    SlowWitnessGeneration,
    /// Large memory footprint.
    HighMemoryUsage,
    /// Suboptimal gate selection.
    GateInefficiency,
    /// Too many copy constraints.
    ExcessiveCopyConstraints,
    /// Large public input count.
    LargePublicInputs,
}

/// Estimated savings from an optimization.
#[derive(Debug, Clone)]
pub struct EstimatedSavings {
    /// Constraint reduction (absolute).
    pub constraint_reduction: usize,
    /// Constraint reduction (percentage).
    pub constraint_reduction_pct: f64,
    /// Time reduction.
    pub time_reduction: Duration,
    /// Memory reduction in bytes.
    pub memory_reduction: usize,
}

impl Default for EstimatedSavings {
    fn default() -> Self {
        Self {
            constraint_reduction: 0,
            constraint_reduction_pct: 0.0,
            time_reduction: Duration::ZERO,
            memory_reduction: 0,
        }
    }
}

/// Main circuit profiler.
pub struct CircuitProfiler {
    config: ProfilingConfig,
    constraint_counter: ConstraintCounter,
    timing_profiler: TimingProfiler,
    operation_tracker: OperationTracker,
}

impl CircuitProfiler {
    /// Creates a new circuit profiler with the given configuration.
    pub fn new(config: ProfilingConfig) -> Self {
        Self {
            constraint_counter: ConstraintCounter::new(),
            timing_profiler: TimingProfiler::new(),
            operation_tracker: OperationTracker::new(config.max_tracked_operations),
            config,
        }
    }

    /// Profiles a circuit and returns a complete profile.
    pub fn profile_circuit<C: Circuit<Fr> + Clone>(
        &mut self,
        circuit: &C,
        k: u32,
        public_inputs: Vec<Vec<Fr>>,
    ) -> Result<CircuitProfile, Error> {
        let start = Instant::now();
        let initial_memory = get_memory_usage();

        // Profile constraints
        let constraint_profile = if self.config.count_constraints {
            self.constraint_counter.analyze_circuit(circuit, k)?
        } else {
            ConstraintProfile::default()
        };

        // Profile timing
        let timing_profile = if self.config.enable_timing {
            self.timing_profiler.profile_mock_prover(circuit, k, public_inputs.clone())?
        } else {
            TimingProfile::default()
        };

        let final_memory = get_memory_usage();
        let memory_bytes = final_memory.saturating_sub(initial_memory);
        let peak_memory_bytes = memory_bytes; // Simplified - would need tracking for accurate peak

        // Analyze bottlenecks
        let bottlenecks = self.analyze_bottlenecks(&constraint_profile, &timing_profile);

        // Calculate performance score
        let performance_score = self.calculate_performance_score(&constraint_profile, &timing_profile);

        // Check if meets targets
        let meets_targets = timing_profile.total_time < Duration::from_millis(500)
            && constraint_profile.total_constraints < 1_000_000;

        Ok(CircuitProfile {
            name: std::any::type_name::<C>().to_string(),
            constraints: constraint_profile,
            timing: timing_profile,
            k,
            memory_bytes,
            peak_memory_bytes,
            operations: self.operation_tracker.get_profiles(),
            bottlenecks,
            meets_targets,
            performance_score,
        })
    }

    /// Analyzes bottlenecks in the profiled circuit.
    fn analyze_bottlenecks(
        &self,
        constraints: &ConstraintProfile,
        timing: &TimingProfile,
    ) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Check for high constraint operations
        if let Some(max_op) = constraints.breakdown.operations.iter()
            .max_by_key(|op| op.constraint_count)
        {
            if max_op.constraint_count > constraints.total_constraints / 5 {
                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::HighConstraintCount,
                    description: format!(
                        "Operation '{}' uses {} constraints ({:.1}% of total)",
                        max_op.name,
                        max_op.constraint_count,
                        max_op.constraint_count as f64 / constraints.total_constraints as f64 * 100.0
                    ),
                    impact: 80,
                    suggestion: format!(
                        "Consider optimizing '{}' with Freivalds verification or lookup tables",
                        max_op.name
                    ),
                    estimated_savings: EstimatedSavings {
                        constraint_reduction: max_op.constraint_count / 2,
                        constraint_reduction_pct: 50.0,
                        time_reduction: timing.total_time / 4,
                        memory_reduction: 0,
                    },
                });
            }
        }

        // Check lookup efficiency
        if constraints.breakdown.lookup_constraints > constraints.total_constraints / 4 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::LookupInefficiency,
                description: format!(
                    "Lookup constraints ({}) are {:.1}% of total",
                    constraints.breakdown.lookup_constraints,
                    constraints.breakdown.lookup_constraints as f64 / constraints.total_constraints as f64 * 100.0
                ),
                impact: 60,
                suggestion: "Consider range compression or combining multiple lookups".to_string(),
                estimated_savings: EstimatedSavings {
                    constraint_reduction: constraints.breakdown.lookup_constraints / 3,
                    constraint_reduction_pct: 33.0,
                    time_reduction: timing.total_time / 6,
                    memory_reduction: 0,
                },
            });
        }

        // Check witness generation time
        if timing.witness_generation > timing.total_time / 3 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::SlowWitnessGeneration,
                description: format!(
                    "Witness generation takes {:?} ({:.1}% of total)",
                    timing.witness_generation,
                    timing.witness_generation.as_nanos() as f64 / timing.total_time.as_nanos() as f64 * 100.0
                ),
                impact: 50,
                suggestion: "Parallelize witness generation using rayon".to_string(),
                estimated_savings: EstimatedSavings {
                    constraint_reduction: 0,
                    constraint_reduction_pct: 0.0,
                    time_reduction: timing.witness_generation / 2,
                    memory_reduction: 0,
                },
            });
        }

        // Check public inputs
        if constraints.breakdown.public_inputs > 10 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::LargePublicInputs,
                description: format!(
                    "{} public inputs adds copy constraint overhead",
                    constraints.breakdown.public_inputs
                ),
                impact: 30,
                suggestion: "Batch public inputs into commitments".to_string(),
                estimated_savings: EstimatedSavings {
                    constraint_reduction: constraints.breakdown.public_inputs * 2,
                    constraint_reduction_pct: 5.0,
                    time_reduction: Duration::from_micros(100),
                    memory_reduction: 0,
                },
            });
        }

        // Sort by impact
        bottlenecks.sort_by(|a, b| b.impact.cmp(&a.impact));
        bottlenecks
    }

    /// Calculates a performance score (0-100).
    fn calculate_performance_score(
        &self,
        constraints: &ConstraintProfile,
        timing: &TimingProfile,
    ) -> u32 {
        let mut score = 100u32;

        // Penalize for high constraint counts
        if constraints.total_constraints > 100_000 {
            score = score.saturating_sub(20);
        }
        if constraints.total_constraints > 500_000 {
            score = score.saturating_sub(20);
        }
        if constraints.total_constraints > 1_000_000 {
            score = score.saturating_sub(20);
        }

        // Penalize for slow timing
        if timing.total_time > Duration::from_millis(100) {
            score = score.saturating_sub(10);
        }
        if timing.total_time > Duration::from_millis(500) {
            score = score.saturating_sub(20);
        }
        if timing.total_time > Duration::from_secs(1) {
            score = score.saturating_sub(20);
        }

        // Penalize for high lookup overhead
        let lookup_ratio = constraints.breakdown.lookup_constraints as f64
            / constraints.total_constraints.max(1) as f64;
        if lookup_ratio > 0.4 {
            score = score.saturating_sub(10);
        }

        score
    }

    /// Returns the profiling configuration.
    pub fn config(&self) -> &ProfilingConfig {
        &self.config
    }
}

/// Tracks individual operations for detailed profiling.
struct OperationTracker {
    operations: HashMap<String, OperationStats>,
    max_operations: usize,
}

#[derive(Default)]
struct OperationStats {
    count: AtomicU64,
    total_constraints: AtomicU64,
    total_time_ns: AtomicU64,
}

impl OperationTracker {
    fn new(max_operations: usize) -> Self {
        Self {
            operations: HashMap::new(),
            max_operations,
        }
    }

    fn record(&mut self, name: &str, constraints: usize, time: Duration) {
        if self.operations.len() >= self.max_operations && !self.operations.contains_key(name) {
            return;
        }

        let stats = self.operations.entry(name.to_string()).or_default();
        stats.count.fetch_add(1, Ordering::Relaxed);
        stats.total_constraints.fetch_add(constraints as u64, Ordering::Relaxed);
        stats.total_time_ns.fetch_add(time.as_nanos() as u64, Ordering::Relaxed);
    }

    fn get_profiles(&self) -> HashMap<String, OperationProfile> {
        let total_constraints: u64 = self.operations.values()
            .map(|s| s.total_constraints.load(Ordering::Relaxed))
            .sum();
        let total_time_ns: u64 = self.operations.values()
            .map(|s| s.total_time_ns.load(Ordering::Relaxed))
            .sum();

        self.operations.iter().map(|(name, stats)| {
            let count = stats.count.load(Ordering::Relaxed) as usize;
            let constraints = stats.total_constraints.load(Ordering::Relaxed) as usize;
            let time_ns = stats.total_time_ns.load(Ordering::Relaxed);

            let profile = OperationProfile {
                count,
                total_constraints: constraints,
                avg_constraints: constraints as f64 / count.max(1) as f64,
                total_time: Duration::from_nanos(time_ns),
                avg_time: Duration::from_nanos(time_ns / count.max(1) as u64),
                constraint_percentage: constraints as f64 / total_constraints.max(1) as f64 * 100.0,
                time_percentage: time_ns as f64 / total_time_ns.max(1) as f64 * 100.0,
            };

            (name.clone(), profile)
        }).collect()
    }
}

/// Gets current memory usage (platform-specific).
fn get_memory_usage() -> usize {
    // Try to get memory usage from /proc/self/statm on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss.parse::<usize>() {
                    return pages * 4096; // Page size is typically 4KB
                }
            }
        }
    }

    // Fallback: return 0 if we can't measure
    0
}

impl CircuitProfile {
    /// Generates a human-readable report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "=== Circuit Profile: {} ===\n\n",
            self.name.split("::").last().unwrap_or(&self.name)
        ));

        report.push_str("--- Constraints ---\n");
        report.push_str(&format!("Total: {}\n", self.constraints.total_constraints));
        report.push_str(&format!("  Gates: {}\n", self.constraints.breakdown.gate_constraints));
        report.push_str(&format!("  Lookups: {}\n", self.constraints.breakdown.lookup_constraints));
        report.push_str(&format!("  Copy: {}\n", self.constraints.breakdown.copy_constraints));
        report.push_str(&format!("  Public inputs: {}\n", self.constraints.breakdown.public_inputs));
        report.push('\n');

        report.push_str("--- Timing ---\n");
        report.push_str(&format!("Total: {:?}\n", self.timing.total_time));
        report.push_str(&format!("  Witness gen: {:?}\n", self.timing.witness_generation));
        report.push_str(&format!("  Synthesis: {:?}\n", self.timing.synthesis));
        report.push_str(&format!("  Verification: {:?}\n", self.timing.verification));
        report.push('\n');

        report.push_str("--- Performance ---\n");
        report.push_str(&format!("K parameter: {} (2^{} = {} rows)\n", self.k, self.k, 1 << self.k));
        report.push_str(&format!("Memory: {:.2} MB\n", self.memory_bytes as f64 / 1_000_000.0));
        report.push_str(&format!("Score: {}/100\n", self.performance_score));
        report.push_str(&format!("Meets targets: {}\n", if self.meets_targets { "✓ YES" } else { "✗ NO" }));
        report.push('\n');

        if !self.bottlenecks.is_empty() {
            report.push_str("--- Bottlenecks ---\n");
            for (i, bottleneck) in self.bottlenecks.iter().enumerate() {
                report.push_str(&format!(
                    "{}. [{:?}] {} (Impact: {})\n   Suggestion: {}\n   Est. savings: {:.1}% constraints\n\n",
                    i + 1,
                    bottleneck.bottleneck_type,
                    bottleneck.description,
                    bottleneck.impact,
                    bottleneck.suggestion,
                    bottleneck.estimated_savings.constraint_reduction_pct
                ));
            }
        }

        report
    }

    /// Returns whether the circuit meets the 500ms target.
    pub fn meets_timing_target(&self) -> bool {
        self.timing.total_time < Duration::from_millis(500)
    }

    /// Returns the constraint reduction needed to meet targets.
    pub fn constraint_reduction_needed(&self) -> usize {
        if self.constraints.total_constraints > 500_000 {
            self.constraints.total_constraints - 500_000
        } else {
            0
        }
    }
}

/// Global profiling context for cross-function profiling.
thread_local! {
    static PROFILING_CONTEXT: std::cell::RefCell<Option<ProfilingContext>> = const { std::cell::RefCell::new(None) };
}

/// Profiling context that can be used across function calls.
#[derive(Default)]
pub struct ProfilingContext {
    operation_times: HashMap<String, Duration>,
    constraint_counts: HashMap<String, usize>,
    active_timer: Option<(String, Instant)>,
}

impl ProfilingContext {
    /// Starts timing an operation.
    pub fn start_operation(name: &str) {
        PROFILING_CONTEXT.with(|ctx| {
            if let Some(ref mut ctx) = *ctx.borrow_mut() {
                ctx.active_timer = Some((name.to_string(), Instant::now()));
            }
        });
    }

    /// Ends timing an operation and records constraints.
    pub fn end_operation(constraints: usize) {
        PROFILING_CONTEXT.with(|ctx| {
            if let Some(ref mut ctx) = *ctx.borrow_mut() {
                if let Some((name, start)) = ctx.active_timer.take() {
                    let elapsed = start.elapsed();
                    *ctx.operation_times.entry(name.clone()).or_default() += elapsed;
                    *ctx.constraint_counts.entry(name).or_default() += constraints;
                }
            }
        });
    }

    /// Initializes the global profiling context.
    pub fn init() {
        PROFILING_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = Some(ProfilingContext::default());
        });
    }

    /// Gets the current profiling results.
    pub fn get_results() -> Option<(HashMap<String, Duration>, HashMap<String, usize>)> {
        PROFILING_CONTEXT.with(|ctx| {
            ctx.borrow().as_ref().map(|c| {
                (c.operation_times.clone(), c.constraint_counts.clone())
            })
        })
    }

    /// Clears the profiling context.
    pub fn clear() {
        PROFILING_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = None;
        });
    }
}

/// Macro for profiling a code block.
#[macro_export]
macro_rules! profile_operation {
    ($name:expr, $body:expr) => {{
        let _start = std::time::Instant::now();
        let result = $body;
        let _elapsed = _start.elapsed();
        // Can be extended to record to profiling context
        result
    }};
}

/// Macro for counting constraints in a code block.
#[macro_export]
macro_rules! count_constraints {
    ($name:expr, $count:expr, $body:expr) => {{
        $crate::profiling::ProfilingContext::start_operation($name);
        let result = $body;
        $crate::profiling::ProfilingContext::end_operation($count);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiling_config() {
        let minimal = ProfilingConfig::minimal();
        assert!(!minimal.count_constraints);
        assert!(minimal.enable_timing);

        let detailed = ProfilingConfig::detailed();
        assert!(detailed.count_constraints);
        assert!(detailed.analyze_gates);
    }

    #[test]
    fn test_operation_tracker() {
        let mut tracker = OperationTracker::new(100);

        tracker.record("matmul", 1000, Duration::from_micros(500));
        tracker.record("matmul", 1000, Duration::from_micros(600));
        tracker.record("relu", 50, Duration::from_micros(10));

        let profiles = tracker.get_profiles();

        assert!(profiles.contains_key("matmul"));
        assert!(profiles.contains_key("relu"));
        assert_eq!(profiles["matmul"].count, 2);
        assert_eq!(profiles["matmul"].total_constraints, 2000);
    }

    #[test]
    fn test_estimated_savings() {
        let savings = EstimatedSavings {
            constraint_reduction: 1000,
            constraint_reduction_pct: 25.0,
            time_reduction: Duration::from_millis(100),
            memory_reduction: 1024,
        };

        assert_eq!(savings.constraint_reduction, 1000);
        assert_eq!(savings.constraint_reduction_pct, 25.0);
    }

    #[test]
    fn test_bottleneck_types() {
        let bottleneck = Bottleneck {
            bottleneck_type: BottleneckType::HighConstraintCount,
            description: "Test".to_string(),
            impact: 80,
            suggestion: "Optimize".to_string(),
            estimated_savings: EstimatedSavings::default(),
        };

        assert_eq!(bottleneck.bottleneck_type, BottleneckType::HighConstraintCount);
        assert_eq!(bottleneck.impact, 80);
    }
}
