//! MPC session management and inter-party communication.
//!
//! An `MPCSession` coordinates the lifecycle of a multi-party computation:
//! - Participant registration and role assignment
//! - Preprocessing (Beaver triple generation)
//! - Input sharing (dealer distributes model weight shares)
//! - Computation phases (forward/backward passes)
//! - Output reconstruction
//! - Periodic re-sharing

pub mod manager;
pub mod channel;

pub use manager::MPCSession;
pub use channel::{MPCChannel, LocalChannel, Message};
