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
pub mod blame_report;
pub mod checkpoint_attestation;
pub mod cheater_recovery;
pub mod e2e_integration;
pub mod error;
pub mod error_containment;
pub mod field;
pub mod graceful_shutdown;
pub mod integration;
pub mod mac_verification;
pub mod mnist;
pub mod mpc_trainer;
pub mod network_distribution;
pub mod nn;
pub mod poseidon;
pub mod profiling;
pub mod proofs;
pub mod protocols;
pub mod recovery;
pub mod recovery_integration;
pub mod message_validation;
pub mod resilience;
pub mod resilient_training;
pub mod security;
pub mod session;
pub mod share_distribution;
pub mod share_redistribution;
pub mod sharing;
pub mod types;
pub mod verification;
pub mod zk_integration;

pub use error::MPCError;
pub use field::{Fr, FieldElement, CtChoice, SecureBuffer, SecureVec};
pub use types::{MPCConfig, PartyId, ShareId};
pub use share_distribution::{
    ShareDistributor, ShareReceiver, CheckpointCommitment, WeightReconstructor,
    EncryptedShare, WeightShare, DistributionResult, VectorCommitment, CommitmentShare,
    encrypt_share_for_owner, generate_x25519_keypair,
};
pub use network_distribution::{
    distribute_shares, reconstruct_shares,
    worker_receive_distribution, worker_send_final_share,
    NetworkDistributionResult, ReconstructionResult, WorkerShareState,
    ProtocolMessage, send_message, recv_message,
};
