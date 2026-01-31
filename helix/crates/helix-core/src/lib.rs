//! helix-core: Shared foundation types, traits, and utilities for HELIX.
//!
//! This crate provides the common building blocks used across all HELIX components:
//! - Bounded values with tracked error margins
//! - Precision levels for approximate computation
//! - Traits for approximate operations and ZK witnesses
//! - Configuration and error types
//! - Data loading, sharding, and model serialization

pub mod archive;
pub mod benchmark;
pub mod config;
pub mod constants;
pub mod data;
pub mod demo;
pub mod error;
pub mod integration;
pub mod traits;
pub mod types;

// Re-export commonly used items at crate root
pub use config::{HelixConfig, ProverConfig, TrainingConfig, VMConfig};
pub use error::{ArithmeticError, BoundsError, CircuitError, HelixError, HelixResult};
pub use traits::{ApproximateOp, BinarySerializable, Provable, Witness};
pub use types::{BoundedTensor, BoundedValue, ErrorMargin, Precision, Shape};

// Re-export benchmark types
pub use benchmark::{
    BenchmarkConfig, BenchmarkResult, BenchmarkRunner, CircuitBenchmarkResult,
    GasCosts, MemorySnapshot, OverheadAnalysis, Statistics,
    run_quick_suite, run_standard_suite,
};


