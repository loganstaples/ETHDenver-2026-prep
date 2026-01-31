//! helix-core: Shared foundation types, traits, and utilities for HELIX.
//!
//! This crate provides the common building blocks used across all HELIX components:
//! - Bounded values with tracked error margins
//! - Precision levels for approximate computation
//! - Traits for approximate operations and ZK witnesses
//! - Configuration and error types
//! - Data loading, sharding, and model serialization
//! - Advanced error algebra and precision management
//! - Comprehensive benchmarking with standardized models

pub mod archive;
pub mod benchmark;
pub mod config;
pub mod constants;
pub mod data;
pub mod demo;
pub mod error;
pub mod integration;
pub mod traits;
pub mod types;

// Re-export commonly used items at crate root
pub use config::{HelixConfig, ProverConfig, TrainingConfig, VMConfig};
pub use error::{ArithmeticError, BoundsError, CircuitError, HelixError, HelixResult};
pub use traits::{ApproximateOp, BinarySerializable, Provable, Witness};

// Core types
pub use types::{BoundedTensor, BoundedValue, ErrorMargin, Precision, Shape};

// Probabilistic error types
pub use types::{
    ConfidenceInterval, ConfidenceLevel, ErrorDistribution, ProbabilisticError,
};

// Error composition types
pub use types::{
    ActivationFunction, AttentionErrorPropagation, CompositionRule,
    ComputationGraphError, ErrorContext, MatrixErrorPropagation,
    NormalizationErrorPropagation, ReductionErrorPropagation, ReductionType,
};

// Precision selection types
pub use types::{
    OperationCharacteristics, OperationType, PrecisionRequirement,
    PrecisionSelection, PrecisionSelectionSummary, PrecisionSelector,
    PrecisionSelectorConfig, PrecisionWeights, SelectionStrategy,
};

// Tensor fusion types
pub use types::{
    ElementwiseOp, FusedErrorAnalysis, FusedOperation, FusedOperationConfig,
    FusionError, FusionOptimizer, FusionOpportunity, FusionPattern,
    OperationInfo, OpType,
};

// Precision scheduler types
pub use types::{
    LayerPrecisionSchedule, PrecisionScheduler, PrecisionSchedulerConfig,
    PrecisionSchedulerSummary, SchedulerDecision, TrainingPhase, TrainingStats,
};

// Error visualization types
pub use types::{
    Alert, AlertLevel, AnnotationType, ChartAnnotation, ChartData, ChartSeries,
    ChartType, DashboardSummary, ErrorDashboardState, ErrorSnapshot,
    ErrorTimeSeries, GaugeData, GaugeZone, ProofStats, SeriesStatistics, SseMessage,
    create_error_trend_chart, create_precision_distribution_chart,
};

// Monte Carlo error types
pub use types::{
    BootstrapResult, ExceedanceProbability, MonteCarloConfig, MonteCarloEstimator,
    MonteCarloResult, MultistageMonteCarlo, VarianceReductionAnalysis,
};

// Error checkpoint types
pub use types::{
    CheckpointDiff, CheckpointManager, ErrorBudgetState, ErrorCheckpoint,
    LayerErrorState, PrecisionCheckpointState, RecoveryOptions, RecoveryResult,
    TrainingCheckpointStats, recover_from_checkpoint,
};

// Error budget types
pub use types::{
    AllocationResult, AllocationStrategy, BudgetAllocator, BudgetAllocationConfig,
    BudgetComponent, BudgetSummary, ComponentAllocation, ComponentSummary,
    ComponentType,
};

// Re-export benchmark types
pub use benchmark::{
    BenchmarkConfig, BenchmarkResult, BenchmarkRunner, CircuitBenchmarkResult,
    GasCosts, MemorySnapshot, OverheadAnalysis, Statistics,
    run_quick_suite, run_standard_suite,
    // Standard model benchmarks
    Architecture, BenchmarkComparison, ErrorStats, ModelBenchmarkResult,
    ModelConfig, ModelSize, StandardBenchmarkSuite, compare_results,
};
