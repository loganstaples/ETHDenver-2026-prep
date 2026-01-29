//! Node Roles for HELIX Network.
//!
//! Defines the different roles a node can take in the HELIX network:
//! - Compute Node: Performs local training and generates proofs
//! - Aggregator Node: Coordinates training rounds and aggregates gradients
//! - Verifier Node: Verifies proofs and maintains consensus

pub mod aggregator;
pub mod compute;
pub mod verifier;

// Re-export commonly used types
pub use aggregator::{AggregatedResult, AggregatorConfig, AggregatorNode, AggregatorState, AggregatorStats};
pub use compute::{ComputeConfig, ComputeNode, ComputeState, ComputeStats, TrainingResult};
