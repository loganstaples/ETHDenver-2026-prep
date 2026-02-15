//! helix-mpc: Multi-Party Computation engine for HELIX trustless AI training.
//!
//! This crate provides the cryptographic MPC protocols that enable model weight
//! privacy during distributed training. Workers compute on secret-shared weights
//! without ever seeing the full model.
//!
//! # Architecture
//!
//! - `sharing`: Additive and Shamir secret sharing for scalars, tensors, and model weights
//! - `beaver`: Beaver triple generation and pool management for secure multiplication
//! - `protocols`: Core MPC protocols (secure arithmetic, matrix ops, activations, aggregation)
//! - `nn`: Secure neural network layers that operate on secret-shared weights
//! - `session`: MPC session management, key rotation, and channel multiplexing
//! - `security`: Share commitments, MAC batching, verification, and audit trails
//! - `integration`: Bridge to helix-node training coordinator
//! - `field`: Finite field arithmetic for proper MPC (replaces f64)
//! - `verification`: Formal verification stubs and protocol invariants
//! - `profiling`: Performance profiling and bottleneck identification
//! - `proofs`: MPC-specific ZK proofs (share validity, aggregation, MAC verification)

#![allow(unexpected_cfgs)]

pub mod beaver;
pub mod error;
pub mod field;
pub mod integration;
pub mod mac_verification;
pub mod mnist;
pub mod mpc_trainer;
pub mod nn;
pub mod poseidon;
pub mod profiling;
pub mod proofs;
pub mod protocols;
pub mod security;
pub mod session;
pub mod share_distribution;
pub mod sharing;
pub mod types;
pub mod verification;

pub use error::MPCError;
pub use field::{Fr, FieldElement, CtChoice, SecureBuffer, SecureVec};
pub use types::{MPCConfig, PartyId, ShareId};
pub use share_distribution::{
    ShareDistributor, ShareReceiver, CheckpointCommitment, WeightReconstructor,
    EncryptedShare, WeightShare, DistributionResult, VectorCommitment, CommitmentShare,
    encrypt_share_for_owner, generate_x25519_keypair,
};
