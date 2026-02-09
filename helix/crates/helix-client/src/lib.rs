pub mod benchmark;
pub mod client;
pub mod commands;
pub mod config;
pub mod dashboard;
pub mod demo;
pub mod help;
pub mod orchestration;
pub mod progress;
pub mod rpc;
pub mod visualization;
pub mod wallet;

// Re-export key types for external consumers
pub use client::HelixClient;
pub use dashboard::{DashboardConfig, DashboardState};
pub use orchestration::{
    DeploymentResult, OrchestratorConfig, ProcessHealthReport, TrainingOrchestrator, TrainingResult,
};
