//! Security module: share commitments, verification, and audit trails.
//!
//! This module provides cryptographic primitives for MPC security:
//! - `commitment`: Hash-based and Pedersen commitments for share/gradient verification
//! - `verification`: Share consistency and correctness checking
//! - `audit`: Logging and audit trails for debugging
//! - `mac`: SPDZ-style MACs for computation integrity verification
//! - `byzantine`: Byzantine fault detection and handling

pub mod audit;
pub mod byzantine;
pub mod commitment;
pub mod mac;
pub mod verification;

pub use audit::AuditLog;
pub use byzantine::{
    ByzantineChecker, ByzantineDetector, DetectedFault, FaultType, PartyState, PartyStatus,
    SlashingAccusation,
};
pub use commitment::{
    AggregatedGradientCommitment, BlindingGenerator, GradientCommitment, MerkleProof,
    ModelCommitments, PedersenCommitment, PedersenGenerators, ShareCommitment,
};
pub use mac::{
    AuthenticatedMessage, AuthenticatedShare, MACKey, MACVerifier, MessageAuthenticator,
    SessionAuthState,
};
pub use verification::{
    BeaverTripleVerifier, CrossPartyVerifier, OutputVerifier, ShareConsistencyTracker,
    ShareVerifier,
};
