//! Integration Test Modules
//!
//! This module contains comprehensive integration tests for the HELIX system.
//! Each submodule focuses on a specific aspect of the system's behavior.

// Note: These modules are compiled as separate test binaries via Cargo.toml [[test]] entries.
// This mod.rs exists for documentation and potential future use as a library.

/// End-to-end training pipeline tests
pub mod end_to_end;

/// MPC + ZK proof integration tests
pub mod mpc_zkp_integration;

/// Adversarial node simulation tests
pub mod adversarial;

/// Network partition and recovery tests
pub mod network_partition;

/// Native vs EVM verification consistency tests
pub mod verification_consistency;

/// Training convergence with known-good models
pub mod training_convergence;

/// Checkpoint and resume functionality tests
pub mod checkpoint_resume;

/// Cross-platform compatibility tests
pub mod cross_platform;
