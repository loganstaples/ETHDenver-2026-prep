//! Integration layer between helix-mpc and the training system.
//!
//! Provides the bridge between the MPC engine and helix-node's
//! training coordinator, enabling secure distributed training.

pub mod training;
pub mod pipeline;

pub use training::SecureTrainingCoordinator;
pub use pipeline::SecurePipeline;
