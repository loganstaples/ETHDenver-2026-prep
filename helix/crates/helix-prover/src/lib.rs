//! HELIX Prover - Reliable Proof Generation Pipeline.
//!
//! This crate provides the complete proof generation infrastructure for HELIX,
//! including:
//!
//! - **pipeline**: Core proving pipeline with comprehensive error handling
//!   - Automatic retry on failure with exponential backoff
//!   - Progress callbacks for long-running proofs
//!   - Self-verification before returning proofs
//!   - Timeout handling with graceful cancellation
//!   - Deterministic proof generation (same inputs → same proof)
//!
//! - **provers**: Specialized provers for different circuit types
//!   - ML Training Prover V2 with witness validation
//!   - Batch proving with progress tracking
//!
//! - **cache**: Multi-layer caching infrastructure
//!   - Witness-based proof cache for deduplication
//!   - Key cache for proving/verification keys
//!   - Incremental proof cache for session-based proving
//!
//! - **health**: Prover health check system
//!   - System resource monitoring
//!   - Pipeline initialization status
//!   - Readiness and liveness probes
//!
//! - **chunking**: Splits large ML computations into provable chunks
//! - **parallel**: Parallel proof generation with thread pools
//! - **aggregation**: Combines multiple proofs using Merkle trees
//! - **ivc**: Incrementally Verifiable Computation for chained proofs
//! - **keys**: Key generation and management
//! - **serialization**: Proof serialization for storage and transmission
//! - **gkr**: Orion-style ZK-GKR prover for neural network circuits
//! - **metal**: Metal GPU acceleration (macOS)
//! - **cuda**: CUDA GPU acceleration (NVIDIA)
//! - **gpu**: Unified GPU infrastructure (memory pools, async ops, multi-GPU)
//! - **backends**: Unified prover backend abstraction
//! - **benchmarks**: GPU vs CPU performance benchmarking

pub mod aggregation;
pub mod backends;
pub mod benchmarks;
pub mod cache;
pub mod chunking;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod gkr;
pub mod gpu;
pub mod health;
pub mod ivc;
pub mod keys;
pub mod metal;
pub mod output;
pub mod parallel;
pub mod pipeline;
pub mod provers;
pub mod serialization;

// Re-export from helix-circuits
pub use helix_circuits::halo2_proofs;
pub use helix_circuits::halo2curves;

// Re-export commonly used types
pub use aggregation::{AggregatedProof, AggregationId, CommitmentTree, ProofAggregator};

// Re-export GKR prover types
pub use gkr::{
    CircuitLayer, DenseMultilinear, GKRConfig, GKRError, GKRLayerProof, GKRProof, GKRProver,
    GKRResult, GKRVerifier, Gate, GateType, LayeredCircuit, MultilinearPolynomial,
    SparseMultilinear, SumcheckProof, SumcheckRound, Wire, ZKConfig, ZKMask, ZeroKnowledgeLayer,
};

// Re-export backend types
pub use backends::{
    BackendCapabilities, BackendConfig, BackendError, BackendId, BackendResult, BackendSelector,
    BackendType, CircuitDescription, ProofData, ProverBackend, UnifiedProver, UnifiedProverConfig,
    VerifierBackend, WitnessData,
};

// Re-export Metal types
pub use metal::{get_device_info, is_metal_available, MetalConfig, MetalStats};

// Re-export GPU infrastructure types
pub use gpu::{
    AsyncOp, AsyncOpQueue, DeviceSelector, GpuBackendType, GpuConfig, GpuMemoryPool,
    LoadBalanceStrategy, MultiGpuManager, OpHandle, OpStatus, PoolConfig, PoolStats, PooledBuffer,
    SyncBarrier, WorkDistributor,
};

// Re-export GPU prover types
pub use provers::gpu_prover::{GpuBackend, GpuError, GpuProver, GpuStatsSnapshot, ProfilingInfo};

// Re-export benchmark types
pub use benchmarks::{
    global_profiler, Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite,
    Backend as ProfilerBackend, GpuProfiler, MsmBenchmark, NttBenchmark, OpGuard, OpStats, OpType,
    ProfileReport, ProfiledOp, ProofGenBenchmark,
};

// Re-export chunking types
pub use chunking::{ChunkId, ChunkingConfig, ComputationChunk, ComputationChunker};

// Re-export IVC types
pub use ivc::{IVCConfig, IVCProver, IVCState, IVCStep};

// Re-export key management types
pub use keys::{CircuitKeys, FileKeyStore, InMemoryKeyStore, KeyId, KeyMetadata};

// Re-export parallel proving types
pub use parallel::{BatchProofResult, ChunkProof, ParallelConfig, ParallelProver, ProofStatus};

// Re-export pipeline types
pub use pipeline::{
    CancellationToken, ExtractedVkData, PipelineConfig, PipelineError, PipelineResult,
    ProofPhase, ProofProgress, ProofResult, ProgressCallback, ProvingStats, ProverPipeline,
    RetryConfig, no_progress_callback,
};

// Re-export serialization types
pub use serialization::{ProofFormat, ProofSerializer, SerializedProof};

// Re-export step prover types
pub use provers::step_prover::{MLTrainingStepProof, TrainingStepData, TrainingStepProver};

// Re-export training prover V1 types
pub use provers::training_prover::{MLTrainingProver, TrainingProofResult};

// Re-export training prover V2 types (enhanced with reliability features)
pub use provers::training_prover_v2::{
    BatchProofResult as BatchProofResultV2, BatchTrainingProverV2, MLTrainingProverV2,
    TrainingProofResultV2, TrainingProverError, TrainingProverResult, TrainingWeights,
    V2ProverConfig, WitnessValidationError, WitnessValidationResult, validate_witness,
};

// Re-export cache types
pub use cache::{
    // Key cache
    KeyCache, KeyCacheConfig, KeyCacheStats,
    // Proof cache
    CachedProof, ProofCache, ProofCacheConfig, ProofCacheStats, ProvingSession, SessionState,
    SharedProofCache, shared_cache, shared_cache_with_config,
    // Witness cache
    EvictionStrategy, SharedWitnessCache, WitnessCache, WitnessCacheConfig, WitnessCacheStats,
    WitnessCachedProof, WitnessHash, WitnessHashBuilder, shared_witness_cache,
    shared_witness_cache_with_config,
};

// Re-export health check types
pub use health::{
    ComponentHealth, HealthCheckConfig, HealthChecker, HealthIssue, HealthReport, HealthStatus,
    IssueSeverity, PerformanceBaseline, ReadinessResult, SystemInfo,
    // Global functions
    full_health_check, global_health_checker, is_prover_ready, liveness_probe, quick_health_check,
    readiness_probe,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        // Core types
        ChunkId, ComputationChunker, IVCProver, MLTrainingProver, ParallelProver, ProofAggregator,
        ProofSerializer, ProverPipeline,
        TrainingStepData, TrainingStepProver,
        // V2 prover types (production-ready)
        BatchTrainingProverV2, MLTrainingProverV2, TrainingProofResultV2, TrainingWeights,
        V2ProverConfig, validate_witness,
        // Pipeline types
        CancellationToken, PipelineConfig, PipelineError, ProofPhase, ProofProgress, RetryConfig,
        no_progress_callback,
        // GKR prover types
        GKRConfig, GKRProof, GKRProver, GKRVerifier, LayeredCircuit,
        // Backend types
        BackendType, ProofData, UnifiedProver,
        // GPU types
        GpuBackendType, GpuConfig, GpuProver,
        // Cache types
        SharedWitnessCache, WitnessCache, WitnessHash, WitnessHashBuilder, shared_witness_cache,
        // Health check types
        HealthChecker, HealthReport, HealthStatus, full_health_check, is_prover_ready,
        quick_health_check, readiness_probe,
        // Benchmark and profiling types
        BenchmarkResult, BenchmarkSuite, GpuProfiler, global_profiler,
    };
}
