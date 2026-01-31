//! Network Module for HELIX P2P Communication.
//!
//! Provides peer-to-peer networking infrastructure including:
//! - Message types for all network communication
//! - Peer discovery and routing
//! - Gossip protocol for message propagation
//! - State synchronization between nodes
//! - TCP/TLS transport layer
//! - Wire format serialization
//! - mDNS-based local discovery

pub mod discovery;
pub mod gossip;
pub mod mdns_discovery;
pub mod messages;
pub mod runner;
pub mod sync;
pub mod transport;
pub mod wire;

// Re-export commonly used types
pub use discovery::{ConnectedPeer, DiscoveryConfig, PeerDiscovery};
pub use gossip::{GossipConfig, GossipProtocol, GossipStats};
pub use mdns_discovery::{CombinedDiscovery, MdnsConfig, MdnsDiscovery, MdnsEvent, MdnsError};
pub use messages::{
    DiscoveryMessage, GradientMessage, HeartbeatMessage, MessagePayload, NetworkMessage,
    NodeCapabilities, PeerId, PeerInfo, SyncMessage, TrainingMessage, TrainingParams,
};
pub use runner::{NetworkEvent, NetworkRunner, NetworkRunnerBuilder, NetworkRunnerConfig};
pub use sync::{Checkpoint, NetworkState, StateSync, SyncConfig, SyncStatus};
pub use transport::{
    ConnectionPool, ConnectionState, PeerConnection, TcpTransport, Transport, TransportConfig,
    TransportError,
};
pub use wire::{
    BinaryGradient, FrameHeader, FrameReader, FrameWriter, MessageFlags, MessageTypeId,
    WireCodec, WireError, HEADER_SIZE, MAGIC, MAX_MESSAGE_SIZE, PROTOCOL_VERSION,
};
