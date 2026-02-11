//! Circuit Parameters and Key Management.
//!
//! This module provides the infrastructure for HELIX's ZK proof system:
//! - Trusted setup and SRS generation
//! - Proving and verification key management
//! - Circuit-specific parameter profiles
//! - Key caching and serialization
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
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod keys;
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
    RealKeyBundle,
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

