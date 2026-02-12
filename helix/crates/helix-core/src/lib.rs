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
//!
//! # Module Organization
//!
//! Most types live under their respective modules. Only the most essential types
//! are re-exported at the crate root for ergonomics. Access other types via
//! `helix_core::types::*`, `helix_core::data::*`, etc.

pub mod archive;
pub mod benchmark;
pub mod config;
pub mod constants;
pub mod data;
pub mod demo;
pub mod error;
pub mod traits;
pub mod types;
pub mod validation;

#[cfg(test)]
mod fuzz_tests;

// ============================================================================
// Core re-exports: essential types used across the workspace
// ============================================================================

// Config and error types
pub use config::{ChainConfig, HelixConfig, ProverConfig, TlsConfig, TrainingConfig, VMConfig};
pub use error::{
    ArithmeticError, BoundsError, CircuitError, DataError, ErrorContext, ErrorSeverity,
    HelixError, HelixResult, LogContext, NetworkError, OverflowError, ResultExt,
    SerializationError, ValidationError,
};

// Traits
pub use traits::{ApproximateOp, BinarySerializable, Provable, Witness};

// Core bounded arithmetic types
pub use types::{
    BoundedTensor, BoundedValue, BoundedValueResult, ErrorMargin, GradTensor, IntoBounded,
    Precision, Shape, TensorBuilder, MAX_ERROR_BOUND, MIN_POSITIVE_VALUE, DIVISION_THRESHOLD,
    MAX_TENSOR_ELEMENTS, MAX_TENSOR_DIMS,
};

// Probabilistic error types (used alongside BoundedValue)
pub use types::{
    ConfidenceInterval, ConfidenceLevel, ErrorDistribution, ProbabilisticError,
};

// Data verification essentials
pub use data::{Hash, MerkleTree, Sha256Hasher, HASH_SIZE};

// Data loading
pub use data::{DataLoader, InMemoryDataLoader, ShardedDataLoader};

// Coordination types for distributed training
pub use types::{
    AttestationChain, DatasetSource, GradientCommitment, ModelArchitecture, NodeId,
    RoundDescriptor, SessionId, TrainingAttestation, TrainingParams, TrainingRoundConfig,
    TrainingStepReceipt,
};

// Training session orchestration
pub use types::{
    SessionConfig, SessionPhase, SessionSummary, TrainingSession,
};

// Proof chain for state hash continuity
pub use types::{
    ChainVerification, ProofChain, ProofEntry,
};

// CSV/binary data loading
pub use data::{CsvDataLoader, CsvLoaderConfig, Normalization};

// Model checkpoints
pub use data::{CheckpointErrorState, CheckpointLayer, CheckpointTensor, ModelCheckpoint};

// Witness pipeline (error checksum wired through for circuit PI[7])
pub use types::{TrainingStepWitnessData, verify_pi_error_checksum};

// Error commitment types
pub use types::{ErrorCommitment, ErrorCommitmentBuilder, ErrorCommitmentPublicInputs, ErrorCommitmentTracker};
