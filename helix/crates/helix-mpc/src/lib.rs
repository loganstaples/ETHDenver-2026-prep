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
//! - `protocols`: Core MPC protocols (secure arithmetic, matrix ops, activations)
//! - `nn`: Secure neural network layers that operate on secret-shared weights
//! - `session`: MPC session management and inter-party communication
//! - `security`: Share commitments, verification, and audit trails
//! - `integration`: Bridge to helix-node training coordinator
//! - `field`: Finite field arithmetic for proper MPC (replaces f64)

pub mod beaver;
pub mod error;
pub mod field;
pub mod integration;
pub mod nn;
pub mod protocols;
pub mod security;
pub mod session;
pub mod sharing;
pub mod types;

pub use error::MPCError;
pub use field::{FieldConfig, FieldElement, FieldShare, FieldSharing, FieldBeaverTriple, FieldSecureMultiply};
pub use types::{MPCConfig, PartyId, ShareId};
