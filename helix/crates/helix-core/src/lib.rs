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
//! - Merkle tree commitments for training data verification
//! - Batch membership proofs for verifiable training
//! - Data provenance tracking and attestations
//! - Multi-source data fetching (IPFS, Filecoin, S3)

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
pub mod validation;

#[cfg(test)]
mod fuzz_tests;

// Re-export commonly used items at crate root
pub use config::{HelixConfig, ProverConfig, TrainingConfig, VMConfig};
pub use error::{
    ArithmeticError, BoundsError, CircuitError, DataError, ErrorContext, ErrorSeverity,
    HelixError, HelixResult, LogContext, NetworkError, OverflowError, ResultExt,
    SerializationError, ValidationError,
};
pub use traits::{ApproximateOp, BinarySerializable, Provable, Witness};

// Validation utilities
pub use validation::{
    validate_config, validate_vm_config, validate_prover_config, validate_training_config,
    validate_finite, validate_range, validate_positive, validate_non_negative,
    validate_vector, validate_non_empty, validate_shape, validate_bounded_value,
    sanitize_f64, sanitize_vector, clamp_to_range,
    validate_serialization_roundtrip, validate_tensor_roundtrip,
    InputSanitizer, RecoveryStrategy, RecoveryContext, RecoveryAction,
    ValidationCollector,
};

// Core types
pub use types::{
    BoundedTensor, BoundedValue, BoundedValueResult, ErrorMargin, IntoBounded, Precision, Shape,
    TensorBuilder, MAX_ERROR_BOUND, MIN_POSITIVE_VALUE, DIVISION_THRESHOLD,
    MAX_TENSOR_ELEMENTS, MAX_TENSOR_DIMS,
};

// Probabilistic error types
pub use types::{
    ConfidenceInterval, ConfidenceLevel, ErrorDistribution, ProbabilisticError,
};

// Error composition types
pub use types::{
    ActivationFunction, AttentionErrorPropagation, CompositionRule,
    ComputationGraphError, ErrorContext as CompositionErrorContext, MatrixErrorPropagation,
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

// Merkle tree and data verification types
pub use data::{
    // Merkle tree
    Hash, MerkleError, MerkleHasher, MerkleProof, MerkleTree, MerkleTreeBuilder,
    MerkleTreeConfig, MultiProof, ProofDirection, ProofStep, Sha256Hasher, TreePosition,
    HASH_SIZE,
    // Streaming and memory-efficient builders
    StreamingMerkleBuilder, StreamingConfig, StreamingStats,
    ChunkedMerkleBuilder, IncrementalRootComputer,
    ParallelMerkleBuilder, ParallelConfig,
    SparseMerkleTree, ProofBatchVerifier,
    // Commitments
    BatchCommitment, CommitmentError, CommitmentManager, DatasetCommitment, SampleCommitment,
    // Membership proofs
    AggregatedBatchProof, BatchMembershipProof, MembershipProofError, MembershipProofGenerator,
    MembershipVerifier, SampleMembershipProof, VerificationStats,
    // Provenance
    Attestation, AttestationType, CustodyRecord, Custodian, CustodianType, DataOrigin,
    DataTransformation, ProvenanceBuilder, ProvenanceChainSummary, ProvenanceError,
    ProvenanceId, ProvenanceRecord, ProvenanceRegistry, TransformationType,
    // Data sources
    DataCache, DataChunk, DataSource, DataSourceError, DataSourceResult, DataStream,
    FallbackBehavior, FetchOptions, MultiSourceFetcher, PoolConfig, ResourceMetadata,
    UploadOptions, WritableDataSource, IpfsDataSource, IpfsSourceConfig,
    FilecoinClient, FilecoinConfig, FilecoinDataSource, S3Config, S3DataSource,
    // Sharding types
    WorkerId, WorkerInfo, WorkerStatus, ShardStats, ShardRegistry, LocalityAwareAssigner,
    ShardStreamer, ShardStreamConfig, DataShard, DataSharder, ShardId, ShardingConfig,
    ShardingStrategy, ShardAssignment,
};
