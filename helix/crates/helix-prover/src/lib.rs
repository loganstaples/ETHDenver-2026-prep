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
pub use helix_circuits::verifier::VkData;

// Re-export commonly used types
pub use aggregation::{AggregatedProof, AggregationId, CommitmentTree, KZGAggregatedProof, KZGBatchAggregator, ProofAggregator, SmartAggregationResult, RLCAggregationProver, AggregatedTrainingProof};

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
pub use ivc::{AccumulatorSnapshot, DeciderProof, IVCConfig, IVCProver, IVCState, IVCStep};

// Re-export key management types
pub use keys::{CircuitKeys, FileKeyStore, InMemoryKeyStore, KeyId, KeyMetadata, HELIX_SRS_SEED};

// Re-export parallel proving types
pub use parallel::{BatchError, BatchProofResult, ChunkProof, ParallelConfig, ParallelProver, ProofStatus};

// Re-export pipeline types
pub use pipeline::{
    CancellationToken, ExtractedVkData, PipelineConfig, PipelineError, PipelineResult,
    ProofPhase, ProofProgress, ProofResult, ProgressCallback, ProvingStats, ProverPipeline,
    RetryConfig, no_progress_callback,
};

// Re-export serialization types
pub use serialization::{ProofFormat, ProofSerializer, SerializedProof};

// Re-export batch prover types
pub use provers::batch_prover::{
    AggregatedBatchProof, BatchProveError, BatchProver, BatchConfig, BatchResult, BatchStatus,
    EpochProver, EpochProofResult, IVCBatchResult, StreamingBatchResult,
};

// Re-export step prover types
pub use provers::step_prover::{MLTrainingStepProof, TrainingStepData, TrainingStepProver};

// Re-export training prover V1 types
pub use provers::training_prover::{MLTrainingProver, TrainingProofResult};

// Re-export training prover V2 types (enhanced with reliability features)
pub use provers::training_prover_v2::{
    BatchProofResult as BatchProofResultV2, BatchTrainingProverV2, EvmProofBundle,
    MLTrainingProverV2, TrainingProofResultV2, TrainingProverError, TrainingProverResult,
    TrainingWeights, V2ProverConfig, WitnessValidationError, WitnessValidationResult,
    validate_witness,
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
    IssueSeverity, PerformanceBaseline, ProofHealthCheck, ReadinessResult, StartupResult, SystemInfo,
    // Global functions
    full_health_check, global_health_checker, initialize_prover_system, is_prover_ready,
    liveness_probe, quick_health_check, readiness_probe,
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
        HealthChecker, HealthReport, HealthStatus, StartupResult, full_health_check,
        initialize_prover_system, is_prover_ready, quick_health_check, readiness_probe,
        // Benchmark and profiling types
        BenchmarkResult, BenchmarkSuite, GpuProfiler, global_profiler,
    };
}

// =============================================================================
// Test Serialization for Heavy Proof Tests
// =============================================================================

/// Shared lock to serialize heavy proof-generation tests.
///
/// Halo2 proof generation (k=12-14) uses significant memory (~100MB+ per SRS)
/// and internally parallelizes via rayon. When cargo test runs many such tests
/// concurrently (default = num_cpus threads), they contend for memory and CPU,
/// causing OOM or extreme slowdown. This lock ensures at most one heavy proof
/// test runs at a time.
#[cfg(test)]
pub(crate) static PROOF_TEST_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

// =============================================================================
// Full Pipeline Integration Tests
// =============================================================================

#[cfg(test)]
mod pipeline_integration_tests {
    use crate::chunking::{ChunkId, ComputationChunk, ComputationType};
    use crate::parallel::{ParallelConfig, ParallelProver, ChunkProof};
    use crate::aggregation::ProofAggregator;
    use crate::ivc::{IVCProver, IVCConfig, IVCStep};
    use crate::serialization::{ProofSerializer, SerializationConfig, CompressionMode};

    /// Full pipeline test: chunks → parallel prove → aggregate → IVC fold → compress.
    ///
    /// This test exercises the complete proving pipeline from chunk creation
    /// through to a compressed, serialized proof artifact.
    #[test]
    fn test_full_pipeline_chunks_to_compressed() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Step 1: Create computation chunks representing training steps.
        let chunks: Vec<ComputationChunk> = (0..4).map(|i| {
            let mut input = [0u8; 32];
            input[0] = i as u8;
            let mut output = input;
            output[0] = (i + 1) as u8;

            ComputationChunk {
                id: ChunkId(i),
                parent_id: if i > 0 { Some(ChunkId(i - 1)) } else { None },
                input_commitment: input,
                output_commitment: output,
                computation_type: ComputationType::Forward,
                layer_range: (0, 2),
                error_bound: 0.01,
            }
        }).collect();

        // Step 2: Prove chunks in parallel.
        let prover = ParallelProver::with_config(ParallelConfig {
            num_threads: 2,
            ..Default::default()
        });
        prover.submit_batch(chunks.clone());
        prover.start();
        let chunk_proofs = prover.wait_all().expect("parallel prove should succeed");
        prover.stop();

        assert_eq!(chunk_proofs.len(), 4, "All 4 chunks must produce proofs");
        for proof in &chunk_proofs {
            assert!(!proof.proof.is_empty(), "Each proof must be non-empty");
            assert!(proof.proof.len() > 64, "Each proof should be a real Halo2 KZG proof");
        }

        // Step 3: Aggregate the chunk proofs.
        let mut aggregator = ProofAggregator::new();
        aggregator.add_proofs(chunk_proofs.clone());
        let aggregated = aggregator.aggregate_single()
            .expect("Aggregation should produce a result");

        assert!(!aggregated.proof.is_empty(), "Aggregated proof must be non-empty");
        assert_eq!(aggregated.chunk_ids.len(), 4, "Aggregation must include all 4 proofs");
        assert_ne!(aggregated.root_commitment, [0u8; 32], "Root commitment must be non-trivial");

        // Verify the aggregated proof
        assert!(aggregator.verify(&aggregated), "Aggregated proof must verify");

        // Step 4: IVC fold — chain the chunk results through IVC.
        let initial = [0u8; 32];
        let config = IVCConfig {
            steps_per_fold: 10,
            store_intermediates: true,
            ..Default::default()
        };
        let mut ivc_prover = IVCProver::with_config(initial, config);

        for (i, chunk) in chunks.iter().enumerate() {
            let step = IVCStep {
                step: (i + 1) as u64,
                input_state: chunk.input_commitment,
                output_state: chunk.output_commitment,
                computation_hash: {
                    use sha2::{Digest, Sha256};
                    let mut h = Sha256::new();
                    h.update(chunk.id.0.to_le_bytes());
                    h.finalize().into()
                },
                step_error: chunk.error_bound,
                proof: chunk_proofs.iter()
                    .find(|p| p.chunk_id == chunk.id)
                    .map(|p| p.proof.clone())
                    .unwrap_or_default(),
            };
            ivc_prover.add_step(step).unwrap();
        }

        assert_eq!(ivc_prover.state().step, 4);
        assert_eq!(ivc_prover.accumulator().num_steps, 4);

        // Finalize the IVC chain
        let final_proof = ivc_prover.finalize().unwrap();
        assert!(!final_proof.is_empty());
        assert!(final_proof.starts_with(b"HELIX_IVC_FINAL:"));

        // Verify the IVC chain
        assert!(ivc_prover.verify(), "IVC folded proof must verify");

        // Step 5: Serialize with compression (use IVC state serialization).
        let serializer = ProofSerializer::with_config(SerializationConfig {
            compression: CompressionMode::Gzip,
            compression_level: 6,
            ..Default::default()
        });

        let ivc_state = ivc_prover.state().clone();
        let serialized = serializer.serialize_ivc(&ivc_state).unwrap();
        assert!(!serialized.data.is_empty());

        // Verify deserialization roundtrip
        let deserialized = serializer.deserialize_ivc(&serialized).unwrap();
        assert_eq!(deserialized.step, ivc_state.step);
        assert_eq!(deserialized.state_commitment, ivc_state.state_commitment);

        // Verify checksum is non-zero
        assert_ne!(serialized.checksum, [0u8; 32], "Checksum should be non-trivial");
    }

    /// Test that chunk proofs from parallel proving can be verified individually.
    #[test]
    fn test_pipeline_proof_verification() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::pipeline::ProverPipeline;
        use crate::provers::ivc_circuit::IVCStepCircuit;
        use helix_circuits::halo2curves::bn256::Fr;

        let chunk = ComputationChunk {
            id: ChunkId(7),
            parent_id: None,
            input_commitment: [0u8; 32],
            output_commitment: [1u8; 32],
            computation_type: ComputationType::Forward,
            layer_range: (0, 2),
            error_bound: 0.01,
        };

        // Prove via parallel prover (short timeout to prevent hangs if worker panics)
        let prover = ParallelProver::with_config(ParallelConfig {
            num_threads: 1,
            proof_timeout_secs: 60,
            max_retries_per_proof: 1,
            ..Default::default()
        });
        prover.submit(chunk.clone(), 100);
        prover.start();
        let proof = prover.wait_for(chunk.id).expect("Proof should complete");
        prover.stop();

        // Reconstruct circuit and verify
        let circuit = IVCStepCircuit {
            prev_state: chunk.input_commitment,
            new_state: chunk.output_commitment,
            computation_hash: {
                use sha2::{Digest, Sha256};
                let mut h = Sha256::new();
                h.update(chunk.id.0.to_le_bytes());
                h.update(chunk.layer_range.0.to_le_bytes());
                h.update(chunk.layer_range.1.to_le_bytes());
                h.finalize().into()
            },
            step_number: chunk.id.0,
        };

        let pi: Vec<Fr> = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        let mut verifier = ProverPipeline::<IVCStepCircuit>::new(5);
        verifier.setup(&IVCStepCircuit::default()).unwrap();
        assert!(verifier.verify(&proof.proof, &pi_refs).unwrap(),
            "Proof from parallel prover must verify");
    }

    /// Test aggregation with extracted chunk proofs (v3 format).
    #[test]
    fn test_pipeline_aggregate_extract_roundtrip() {
        let chunk_proofs: Vec<ChunkProof> = (0..3).map(|i| {
            ChunkProof {
                chunk_id: ChunkId(i),
                proof: vec![0xAA; 128 + i as usize * 10],
                public_inputs: vec![[i as u8; 32]],
                error_bound: 0.01,
                generation_time_ms: 100,
            }
        }).collect();

        let mut aggregator = ProofAggregator::new();
        aggregator.add_proofs(chunk_proofs.clone());
        let aggregated = aggregator.aggregate_single().unwrap();

        // Extract and verify
        let extracted = ProofAggregator::extract_chunk_proofs(&aggregated)
            .expect("Should extract chunk proofs from v3 format");
        assert_eq!(extracted.len(), 3, "Should extract all 3 chunk proofs");

        for (i, ep) in extracted.iter().enumerate() {
            assert_eq!(ep.chunk_id, ChunkId(i as u64));
            assert_eq!(ep.proof.len(), 128 + i * 10);
            assert_eq!(ep.public_inputs, vec![[i as u8; 32]]);
        }
    }

    /// Test compression modes with real proof data.
    #[test]
    fn test_pipeline_all_compression_modes() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Generate a real proof to compress (short timeout to prevent hangs)
        let prover = ParallelProver::with_config(ParallelConfig {
            num_threads: 1,
            proof_timeout_secs: 60,
            max_retries_per_proof: 1,
            ..Default::default()
        });
        let chunk = ComputationChunk {
            id: ChunkId(0),
            parent_id: None,
            input_commitment: [0u8; 32],
            output_commitment: [1u8; 32],
            computation_type: ComputationType::Forward,
            layer_range: (0, 2),
            error_bound: 0.01,
        };
        prover.submit(chunk.clone(), 100);
        prover.start();
        let proof = prover.wait_for(chunk.id).unwrap();
        prover.stop();

        // Serialize the chunk proof using each compression mode
        for mode in &[CompressionMode::None, CompressionMode::Gzip, CompressionMode::Lz4, CompressionMode::Zstd] {
            let serializer = ProofSerializer::with_config(SerializationConfig {
                compression: *mode,
                compression_level: 6,
                ..Default::default()
            });

            let serialized = serializer.serialize_chunk(&proof).unwrap();
            let deserialized = serializer.deserialize_chunk(&serialized).unwrap();
            assert_eq!(deserialized.proof, proof.proof,
                "Roundtrip failed for {:?}", mode);
            assert_eq!(deserialized.chunk_id, proof.chunk_id);
        }
    }
}
