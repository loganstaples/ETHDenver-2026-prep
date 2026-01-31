//! Types module - re-exports all type definitions.

pub mod bounded_value;
pub mod error_margin;
pub mod precision;
pub mod tensor;

// Advanced error algebra modules
pub mod probabilistic_error;
pub mod error_composition;
pub mod precision_selector;
pub mod tensor_fusion;
pub mod precision_scheduler;
pub mod error_visualization;
pub mod monte_carlo_error;
pub mod error_checkpoint;
pub mod error_budget;

// Re-export core types
pub use bounded_value::BoundedValue;
pub use error_margin::ErrorMargin;
pub use precision::Precision;
pub use tensor::{BoundedTensor, Shape};

// Re-export probabilistic error types
pub use probabilistic_error::{
    ConfidenceInterval, ConfidenceLevel, ErrorDistribution, ProbabilisticError,
};

// Re-export error composition types
pub use error_composition::{
    ActivationFunction, AttentionErrorPropagation, CompositionRule,
    ComputationGraphError, ErrorContext, MatrixErrorPropagation,
    NormalizationErrorPropagation, ReductionErrorPropagation, ReductionType,
};

// Re-export precision selection types
pub use precision_selector::{
    OperationCharacteristics, OperationType, PrecisionRequirement,
    PrecisionSelection, PrecisionSelectionSummary, PrecisionSelector,
    PrecisionSelectorConfig, PrecisionWeights, SelectionStrategy,
};

// Re-export tensor fusion types
pub use tensor_fusion::{
    ElementwiseOp, FusedErrorAnalysis, FusedOperation, FusedOperationConfig,
    FusionError, FusionOptimizer, FusionOpportunity, FusionPattern,
    OperationInfo, OpType, ActivationType as FusionActivationType,
};

// Re-export precision scheduler types
pub use precision_scheduler::{
    LayerPrecisionSchedule, PrecisionScheduler, PrecisionSchedulerConfig,
    PrecisionSchedulerSummary, SchedulerDecision, TrainingPhase, TrainingStats,
};

// Re-export error visualization types
pub use error_visualization::{
    Alert, AlertLevel, AnnotationType, ChartAnnotation, ChartData, ChartSeries,
    ChartType, DashboardSummary, ErrorDashboardState, ErrorSnapshot,
    ErrorTimeSeries, GaugeData, GaugeZone, ProofStats, SeriesStatistics, SseMessage,
};

// Re-export Monte Carlo error types
pub use monte_carlo_error::{
    BootstrapResult, ExceedanceProbability, MonteCarloConfig, MonteCarloEstimator,
    MonteCarloResult, MultistageMonteCarlo, VarianceReductionAnalysis,
    ActivationType as MonteCarloActivationType,
};

// Re-export error checkpoint types
pub use error_checkpoint::{
    CheckpointDiff, CheckpointManager, ErrorBudgetState, ErrorCheckpoint,
    LayerErrorState, PrecisionCheckpointState, RecoveryOptions, RecoveryResult,
    TrainingCheckpointStats,
};

// Re-export error budget types
pub use error_budget::{
    AllocationResult, AllocationStrategy, BudgetAllocator, BudgetAllocationConfig,
    BudgetComponent, BudgetSummary, ComponentAllocation, ComponentSummary,
    ComponentType,
};

// Convenience type alias for error creation
pub use error_visualization::{
    create_error_trend_chart, create_precision_distribution_chart,
};

pub use error_checkpoint::recover_from_checkpoint;
