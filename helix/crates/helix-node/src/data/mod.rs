//! Verified Data Loading Module for HELIX.
//!
//! Provides cryptographically verified data loading with:
//! - Merkle proof verification for each batch
//! - Dataset commitment verification at job initialization
//! - Data availability checking before training
//! - Integration with helix-core verification types

pub mod verified_loader;
pub mod availability;
pub mod commitment_check;

pub use verified_loader::{
    VerifiedDataLoader, VerifiedDataLoaderConfig, VerifiedBatch,
    BatchVerificationResult, DataVerificationError, VerificationStats as LoaderVerificationStats,
};

pub use availability::{
    DataAvailabilityChecker, AvailabilityConfig, AvailabilityStatus,
    AvailabilityCheckResult, DataSourceHealth, RedundancyLevel,
};

pub use commitment_check::{
    CommitmentVerifier, CommitmentVerifierConfig, CommitmentCheckResult,
    OnChainCommitment, CommitmentCache, CommitmentStatus,
};
