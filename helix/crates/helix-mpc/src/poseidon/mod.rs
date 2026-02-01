//! Poseidon Hash for MPC Commitment Verification.
//!
//! This module provides Poseidon-compatible hash primitives that can be used for:
//! - In-circuit commitment verification
//! - Share commitment generation
//! - Merkle tree construction for model weights
//! - Domain-separated commitments for security
//!
//! The implementation uses SHA256 internally but produces outputs compatible
//! with field element operations. For actual circuit verification, helix-circuits'
//! Pow5Chip implementation is used.

pub mod hash;

pub use hash::{
    // Core types and constants
    PoseidonHasher, PoseidonParams, PoseidonSpec,
    MPC_POSEIDON_WIDTH, MPC_POSEIDON_RATE,

    // Hash functions
    poseidon_hash, poseidon_hash_two,

    // Commitment functions
    poseidon_commit, poseidon_commit_with_domain,
    share_commitment, gradient_commitment, mac_commitment,
    poseidon_state_hash,

    // Verification functions
    verify_commitment, verify_domain_commitment,

    // Domain constants
    domains,
};
