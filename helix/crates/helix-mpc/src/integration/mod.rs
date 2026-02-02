//! Integration layer between helix-mpc and the training system.
//!
//! Provides the bridge between the MPC engine and helix-node's
//! training coordinator, enabling secure distributed training with
//! real Halo2 ZK proofs.
//!
//! # Modules
//!
//! - `training`: Secure training coordinator for MPC operations
//! - `pipeline`: High-level secure training pipeline
//! - `witness_format`: MPC witness format for ZK circuits
//! - `witness`: MPC to circuit witness conversion
//! - `circuit_bridge`: Bridge to helix-prover for real Halo2 proofs
//! - `zk_pipeline`: MPC to ZK proof generation pipeline
//! - `zk_training`: Integrated training flow with ZK proofs
//!
//! # MPC-ZK Integration Architecture
//!
//! The key innovation is enabling workers to compute on secret shares AND
//! generate ZK proofs of correct computation. The flow is:
//!
//! 1. Workers use `ProvedArithmetic` for MPC operations (captures witness)
//! 2. Workers exchange shares and reconstruct intermediate values
//! 3. `CircuitBridge` converts witness to Halo2 format
//! 4. Halo2 proof is generated and verified
//! 5. Proof is submitted on-chain for verification

pub mod circuit_bridge;
pub mod pipeline;
pub mod training;
pub mod witness;
pub mod witness_format;
pub mod zk_pipeline;
pub mod zk_training;
pub mod zk_training_integration;

// Core training types
pub use pipeline::SecurePipeline;
pub use training::SecureTrainingCoordinator;

// Witness format types
pub use witness_format::{MPCTrainingWitness, ReconstructedWitness, ShareWitness, WitnessBuilder, WitnessAggregator};
pub use witness::{CircuitWitness, ShareCommitmentData, generate_freivalds_challenges};

// Circuit bridge for real Halo2 proofs
pub use circuit_bridge::{
    CircuitBridge, CircuitBridgeConfig, Halo2ProofResult,
    SharedCircuitBridge, create_shared_bridge, compute_compatible_state_hash,
    share_encoding, witness_capture_to_reconstructed,
    ProofGenerationStats, TrackedCircuitBridge,
};

// ZK pipeline types
pub use zk_pipeline::{
    BatchProofGenerator, BatchProofResult, ProofHook, ProofHookCollector, ProofOperation,
    ProofResult, ShareProofCommitment, ZKPipelineConfig, ZKProofPipeline,
};
pub use zk_training::{ZKTrainingCoordinator, ZKTrainingConfig, ZKTrainingStep};

// Integrated MPC-ZK training (CRITICAL PATH)
pub use zk_training_integration::{
    IntegratedTrainingConfig, IntegratedTrainingCoordinator, IntegratedTrainingStep,
    ReconstructedModel, TrainingReport, WorkerState,
};
