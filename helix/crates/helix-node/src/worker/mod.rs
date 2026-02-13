//! Worker Auto-Discovery & Participation Daemon.
//!
//! Provides autonomous worker behavior for HELIX nodes:
//! - Listens for training round announcements via gossip protocol
//! - Evaluates whether to join based on capabilities, stake, and capacity
//! - Executes training steps with ZK proof generation
//! - Reports results back to the aggregator
//! - Tracks participation history, earnings, and reputation

pub mod daemon;
pub mod evaluator;
pub mod tracker;

pub use daemon::{WorkerDaemon, WorkerDaemonEvent, ActiveRoundState};
pub use evaluator::RoundEvaluator;
pub use tracker::{ParticipationTracker, RoundRecord, EarningsSummary};
