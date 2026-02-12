//! Common test utilities and harness for HELIX integration tests.
//!
//! This module provides:
//! - Test fixtures for models, proofs, and MPC shares
//! - Mock network simulation
//! - Performance measurement utilities
//! - Deterministic randomness for reproducible tests

pub mod fixtures;
pub mod harness;
pub mod mocks;
pub mod metrics;
pub mod assertions;

#[cfg(feature = "on-chain")]
pub mod anvil;

#[cfg(feature = "integration")]
pub mod e2e;

pub use fixtures::*;
pub use harness::*;
pub use mocks::*;
pub use metrics::*;
pub use assertions::*;
