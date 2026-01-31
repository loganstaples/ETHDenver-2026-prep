//! HELIX Prover - Proof Generation Pipeline.
//!
//! This crate provides the complete proof generation infrastructure for HELIX,
//! including:
//!
//! - **chunking**: Splits large ML computations into provable chunks
//! - **parallel**: Parallel proof generation with thread pools
//! - **aggregation**: Combines multiple proofs using Merkle trees
//! - **ivc**: Incrementally Verifiable Computation for chained proofs
//! - **keys**: Key generation and management
//! - **serialization**: Proof serialization for storage and transmission
//! - **pipeline**: Core proving pipeline (Halo2)
//! - **provers**: Specialized provers for different circuit types
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
    GKRProver, GKRVerifier, GKRProof, GKRConfig, GKRLayerProof,
    LayeredCircuit, CircuitLayer, Gate, GateType, Wire,
    SumcheckProof, SumcheckRound,
    DenseMultilinear, SparseMultilinear, MultilinearPolynomial,
    ZeroKnowledgeLayer, ZKConfig, ZKMask,
    GKRError, GKRResult,
};

// Re-export backend types
pub use backends::{
    ProverBackend, VerifierBackend, BackendCapabilities,
    BackendId, BackendConfig, BackendResult, BackendError,
    ProofData, CircuitDescription, WitnessData,
    BackendType, UnifiedProver, UnifiedProverConfig, BackendSelector,
};

// Re-export Metal types
pub use metal::{MetalConfig, MetalStats, is_metal_available, get_device_info};

// Re-export GPU infrastructure types
pub use gpu::{
    GpuBackendType, GpuConfig,
    GpuMemoryPool, PooledBuffer, PoolConfig, PoolStats,
    AsyncOpQueue, AsyncOp, OpHandle, OpStatus, SyncBarrier,
    MultiGpuManager, DeviceSelector, WorkDistributor, LoadBalanceStrategy,
};

// Re-export GPU prover types
pub use provers::gpu_prover::{GpuBackend, GpuProver, GpuError, ProfilingInfo, GpuStatsSnapshot};

// Re-export benchmark types
pub use benchmarks::{
    Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite,
    MsmBenchmark, NttBenchmark, ProofGenBenchmark,
    // Profiler types
    GpuProfiler, ProfiledOp, ProfileReport, OpGuard, OpStats,
    OpType, Backend as ProfilerBackend, global_profiler,
};
pub use chunking::{ChunkId, ChunkingConfig, ComputationChunk, ComputationChunker};
pub use ivc::{IVCConfig, IVCProver, IVCState, IVCStep};
pub use keys::{FileKeyStore, InMemoryKeyStore, KeyId, KeyMetadata};
pub use parallel::{BatchProofResult, ChunkProof, ParallelConfig, ParallelProver, ProofStatus};
pub use pipeline::{ExtractedVkData, ProverPipeline};
pub use provers::step_prover::{MLTrainingStepProof, TrainingStepData, TrainingStepProver};
pub use provers::training_prover::{MLTrainingProver, TrainingProofResult};
pub use provers::training_prover_v2::{
    MLTrainingProverV2, TrainingProofResultV2, BatchTrainingProverV2,
    BatchProofResult as BatchProofResultV2, TrainingWeights, V2ProverConfig,
};
pub use serialization::{ProofFormat, ProofSerializer, SerializedProof};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        ChunkId, ComputationChunker, IVCProver, MLTrainingProver,
        ParallelProver, ProofAggregator, ProofSerializer, ProverPipeline,
        TrainingStepData, TrainingStepProver,
        // V2 prover types
        MLTrainingProverV2, BatchTrainingProverV2, TrainingWeights,
        // GKR prover types
        GKRProver, GKRVerifier, GKRProof, GKRConfig, LayeredCircuit,
        // Backend types
        UnifiedProver, BackendType, ProofData,
        // GPU types
        GpuProver, GpuBackendType, GpuConfig,
        // Benchmark and profiling types
        BenchmarkSuite, BenchmarkResult, GpuProfiler, global_profiler,
    };
}
