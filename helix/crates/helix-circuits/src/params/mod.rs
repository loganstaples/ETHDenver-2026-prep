//! Circuit Parameters and Key Management.
//!
//! This module provides the infrastructure for HELIX's ZK proof system:
//! - Trusted setup and SRS generation
//! - Proving and verification key management
//! - Circuit-specific parameter profiles
//! - Key caching and serialization
//! - Circuit optimization and proof size estimation
//!
//! # Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    HELIX Parameter System                        │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │                    setup.rs                               │  │
//! │  │  - Powers of Tau ceremony                                 │  │
//! │  │  - SRS generation and caching                             │  │
//! │  │  - Parameter profiles (Small/Medium/Large)                │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                              ↓                                   │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │                    keys.rs                                │  │
//! │  │  - Proving key generation and management                  │  │
//! │  │  - Verification key extraction and serialization          │  │
//! │  │  - Key bundles and caching                                │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                              ↓                                   │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │                 optimization.rs                           │  │
//! │  │  - Circuit constraint optimization                        │  │
//! │  │  - Proof size estimation                                  │  │
//! │  │  - Benchmark suite for model architectures                │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod keys;
pub mod optimization;
pub mod setup;

// Re-export key types from setup
pub use setup::{
    // Core SRS types
    HelixSRS,
    SRSMetadata,
    SRSCache,

    // Parameter profiles
    ParameterProfile,

    // Ceremony types
    PowersOfTauCeremony,
    FinalizedCeremony,
    CeremonyContribution,

    // Benchmark
    SRSGenerationBenchmark,

    // Errors
    SetupError,

    // Constants
    MAX_K,
    MIN_K,
    SRS_FORMAT_VERSION,
    SRS_MAGIC,

    // Global cache functions
    global_cache,
    init_global_cache,
};

// Re-export key types from keys
pub use keys::{
    // Proving key types
    HelixProvingKey,
    ProvingKeyMetadata,

    // Verification key types
    HelixVerificationKey,
    VerificationKeyMetadata,

    // Bundle and cache
    KeyBundle,
    KeyCache,

    // Benchmark
    KeygenBenchmark,

    // Circuit config
    CircuitConfig,

    // Commitment scheme
    CommitmentScheme,

    // Errors
    KeyError,

    // Constants
    KEY_FORMAT_VERSION,
    PK_MAGIC,
    VK_MAGIC,
};

// Re-export optimization types
pub use optimization::{
    // Analysis
    CircuitAnalysis,

    // Optimization
    CircuitOptimizer,
    OptimizationPass,
    OptimizationResult,
    OptimizationSummary,

    // Proof size estimation
    ProofSizeEstimator,
    ProofSizeEstimate,

    // Model configuration
    ModelConfig,

    // Benchmarking
    CircuitBenchmarkSuite,
    BenchmarkResult as OptBenchmarkResult,
    BenchmarkReport,
};
