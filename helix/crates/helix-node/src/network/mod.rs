//! Network Module for HELIX P2P Communication.
//!
//! Provides peer-to-peer networking infrastructure including:
//! - Message types for all network communication
//! - Peer discovery and routing
//! - Gossip protocol for message propagation
//! - State synchronization between nodes

pub mod discovery;
pub mod gossip;
pub mod messages;
pub mod sync;

// Re-export commonly used types
pub use discovery::{ConnectedPeer, DiscoveryConfig, PeerDiscovery};
pub use gossip::{GossipConfig, GossipProtocol, GossipStats};
pub use messages::{
    DiscoveryMessage, GradientMessage, HeartbeatMessage, MessagePayload, NetworkMessage,
    NodeCapabilities, PeerId, PeerInfo, SyncMessage, TrainingMessage, TrainingParams,
};
pub use sync::{Checkpoint, NetworkState, StateSync, SyncConfig, SyncStatus};
