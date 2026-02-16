pub mod benchmark;
#[cfg(feature = "chain")]
pub mod checkpoint_submitter;
pub mod client;
pub mod commands;
pub mod config;
pub mod dashboard;
pub mod demo;
pub mod error;
pub mod full_orchestration;
pub mod health;
pub mod help;
pub mod model;
pub mod mpc_orchestration;
pub mod orchestration;
pub mod orchestrator;
pub mod progress;
pub mod rpc;
pub mod session;
pub mod share_distributor;
pub mod visualization;
pub mod wallet;
pub mod worker_entry;
pub mod zk_proof_layer;

// Re-export key types for external consumers
pub use client::HelixClient;
pub use dashboard::{DashboardConfig, DashboardState, TrainingJobRequest, TrainingSessionState};
pub use error::HelixError;
pub use model::{ModelArchitecture, ModelHandle, SdkModelConfig, TrainingParams};
pub use orchestration::{
    DeploymentResult, OrchestratorConfig, ProcessHealthReport, TrainingOrchestrator, TrainingResult,
};
pub use orchestrator::{LiveTrainingOrchestrator, NetworkOrchestrator};
pub use rpc::client::{
    ModelWeightsResponse, RoundSummary, TrainingHistory, TrainingReport,
};
pub use session::{SessionResult, TrainingEvent, TrainingProgress, TrainingSession};
pub use full_orchestration::{
    FullOrchestrationConfig, FullOrchestrationResult, FullOrchestrator, CheaterInfo,
    ProgressCallback, ProgressEvent, ZkMode,
};
pub use worker_entry::{WorkerConfig, WorkerResult, launch_worker};
pub use zk_proof_layer::{ZkCheckpointProofResult, ZkProofConfig, ZkProofLayer, ZkProofStats};
