//! MPC session management and inter-party communication.
//!
//! An `MPCSession` coordinates the lifecycle of a multi-party computation:
//! - Participant registration and role assignment
//! - Preprocessing (Beaver triple generation)
//! - Input sharing (dealer distributes model weight shares)
//! - Computation phases (forward/backward passes)
//! - Output reconstruction
//! - Periodic re-sharing
//!
//! # Communication Channels
//!
//! Two channel implementations are provided:
//! - `LocalChannel`: In-memory channel for local testing/demos (single process)
//! - `NetworkChannel`: TCP/TLS channel for real distributed MPC (multiple processes)
//!
//! # Session Establishment
//!
//! The `establishment` module provides a secure protocol for:
//! - Key exchange using Diffie-Hellman
//! - Mutual authentication with signatures
//! - Session key derivation

pub mod channel;
pub mod establishment;
pub mod manager;
pub mod network;

pub use channel::{LocalChannel, MPCChannel, Message, MessageType};
pub use establishment::{
    AuthenticationMessage, EstablishedSession, KeyExchangeMessage, SessionConfig,
    SessionEstablishment, SessionPhase, simulate_session_establishment,
};
pub use manager::MPCSession;
pub use network::{AsyncMPCChannel, ConnectionState, NetworkChannel, NetworkConfig, TlsConfig};
