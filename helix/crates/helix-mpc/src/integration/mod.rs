//! Integration layer between helix-mpc and the training system.
//!
//! Provides the bridge between the MPC engine and helix-node's
//! training coordinator, enabling secure distributed training.
//!
//! # Modules
//!
//! - `training`: Secure training coordinator for MPC operations
//! - `pipeline`: High-level secure training pipeline
//! - `witness_format`: MPC witness format for ZK circuits
//! - `zk_pipeline`: MPC to ZK proof generation pipeline
//! - `zk_training`: Integrated training flow with ZK proofs

pub mod pipeline;
pub mod training;
pub mod witness_format;
pub mod zk_pipeline;
pub mod zk_training;

pub use pipeline::SecurePipeline;
pub use training::SecureTrainingCoordinator;
pub use witness_format::{MPCTrainingWitness, ReconstructedWitness, ShareWitness, WitnessBuilder};
pub use zk_pipeline::{
    BatchProofGenerator, BatchProofResult, ProofHook, ProofHookCollector, ProofOperation,
    ProofResult, ShareProofCommitment, ZKPipelineConfig, ZKProofPipeline,
};
pub use zk_training::{ZKTrainingCoordinator, ZKTrainingConfig, ZKTrainingStep};
