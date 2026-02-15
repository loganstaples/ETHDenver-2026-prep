//! HELIX RPC Client Module
//!
//! Provides JSON-RPC client for communicating with helix-node instances.
//! Supports all training, proof, staking, and network operations.
//!
//! # Architecture
//!
//! - `HelixRpcClient` - Real JSON-RPC client for production nodes
//! - `MockRpcClient` - Simulated client for demos and testing
//! - `UnifiedRpcClient` - Wrapper that can use either real or mock
//! - `RealTimeProofTracker` - Real-time proof generation tracking
//! - `RealTimeTrainingTracker` - Real-time training progress tracking

#[cfg(feature = "chain")]
pub mod chain;

#[cfg(feature = "chain")]
pub mod chain_v3;

#[cfg(feature = "chain")]
pub mod chain_v4;

pub mod client;

#[cfg(feature = "chain")]
pub use chain::{ChainClient, ChainCircuitBreaker, ChainCircuitState, ChainModelState, ChainRoundState, ChainStakeInfo, ForgeDeployResult, TrainingProofInputs};

#[cfg(feature = "chain")]
pub use chain_v3::{
    ChainClientV3, ChainRoundExtState, ChainV3StakeInfo,
    ChainRewardPoolInfo, ChainParticipantStats, ForgeDeployResultV3,
};

#[cfg(feature = "chain")]
pub use chain_v4::{
    ChainClientV4, V4DeployResult, V4JobSummary, V4WorkerInfo,
    V4Checkpoint, V4MACFailureReport,
    build_checkpoint_message, build_mac_failure_message, build_completion_message,
    sign_checkpoint, sign_mac_failure, sign_completion,
};

pub use client::{
    // Core client types
    HelixRpcClient, HelixRpcConfig, RpcError,
    // Circuit breaker
    RpcCircuitBreaker, RpcCircuitBreakerState,
    // Status types
    TrainingStatus, TrainingPhase, ProofStatus, ProofPhase,
    NetworkStatus, WorkerInfo, WorkerStatus,
    ModelInfo, RoundInfo, StakingInfo, SlashingEvent,
    ProofSubmission, TrainingProgress, TrainingResultData,
    NodeCapabilities, HealthStatus, ComponentHealth,
    GenerateProofAck,
    // Snapshot type
    DemoSnapshot,
    // Mock client for demos
    MockRpcClient,
    // Unified client
    UnifiedRpcClient,
    // Real-time tracking
    RealTimeProofTracker, RealTimeTrainingTracker,
};
