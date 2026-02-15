pub mod benchmark;
pub mod client;
pub mod commands;
pub mod config;
pub mod dashboard;
pub mod demo;
pub mod error;
pub mod health;
pub mod help;
pub mod model;
pub mod orchestration;
pub mod orchestrator;
pub mod progress;
pub mod rpc;
pub mod session;
pub mod share_distributor;
pub mod visualization;
pub mod wallet;

// Re-export key types for external consumers
pub use client::HelixClient;
pub use dashboard::{DashboardConfig, DashboardState};
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
