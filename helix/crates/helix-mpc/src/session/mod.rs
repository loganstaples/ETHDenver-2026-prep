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
pub mod checkpoint;
pub mod establishment;
pub mod health;
pub mod key_rotation;
pub mod manager;
pub mod multiplexer;
pub mod network;
pub mod node_transport;
pub mod party_selection;
pub mod registry;
pub mod secure_channel;
pub mod transport;

#[cfg(test)]
mod integration_tests;

pub use channel::{LocalChannel, MPCChannel, Message, MessageType};
pub use establishment::{
    AuthenticationMessage, EstablishedSession, KeyExchangeMessage, SessionConfig,
    SessionEstablishment, SessionPhase, simulate_session_establishment,
};
pub use manager::{ConnectedSession, MPCSession, TransportSession};
pub use network::{AsyncMPCChannel, ConnectionState, NetworkChannel, NetworkConfig, TlsConfig};
pub use key_rotation::{
    KeyRotationConfig, KeyRotationManager, KeyRotationStats, PFSManager,
    RotationMessage, RotationState, VersionedKey,
};
pub use multiplexer::{
    BatchedMessage, MessageAggregator, MultiplexedChannel, MultiplexerConfig,
    MultiplexerStats, StreamId, StreamMessage, StreamStats,
};
pub use party_selection::{
    HeartbeatMessage, PartyMetricsSummary, PartySelector, RoundRobinSelector,
    SelectionConfig,
};
pub use transport::{HandshakeMessage, LocalTransport, MPCTransport};
#[cfg(feature = "network-mpc")]
pub use transport::TcpTransport;
pub use checkpoint::{SessionCheckpoint, SessionPersistence};
pub use health::{HealthConfig, HealthEvent, HealthStatus, PartyHealthMonitor};
pub use node_transport::{
    MPCMessageRouter, NodeConnectionBridge, NodeTransport, PeerMapping, TaggedMessage,
};
pub use registry::{
    PartyCapabilities, PartyRegistry, PartyStatus, RegisteredParty, RegistryConfig,
    SelectionCriteria,
};
