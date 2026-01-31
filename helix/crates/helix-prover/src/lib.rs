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
//! - **pipeline**: Core proving pipeline
//! - **provers**: Specialized provers for different circuit types

pub mod aggregation;
pub mod cache;
pub mod chunking;
pub mod ivc;
pub mod keys;
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
pub use chunking::{ChunkId, ChunkingConfig, ComputationChunk, ComputationChunker};
pub use ivc::{IVCConfig, IVCProver, IVCState, IVCStep};
pub use keys::{FileKeyStore, InMemoryKeyStore, KeyId, KeyMetadata};
pub use parallel::{BatchProofResult, ChunkProof, ParallelConfig, ParallelProver, ProofStatus};
pub use pipeline::{ExtractedVkData, ProverPipeline};
pub use provers::step_prover::{MLTrainingStepProof, TrainingStepData, TrainingStepProver};
pub use provers::training_prover::{MLTrainingProver, TrainingProofResult};
pub use serialization::{ProofFormat, ProofSerializer, SerializedProof};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        ChunkId, ComputationChunker, IVCProver, MLTrainingProver,
        ParallelProver, ProofAggregator, ProofSerializer, ProverPipeline,
        TrainingStepData, TrainingStepProver,
    };
}
