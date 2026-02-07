pub mod client;
pub mod commands;
pub mod config;
pub mod dashboard;
pub mod demo;
pub mod help;
pub mod progress;
pub mod rpc;
pub mod visualization;
pub mod wallet;

// Re-export key types for external consumers
pub use dashboard::{DashboardConfig, DashboardState};
