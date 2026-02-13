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
//! - Network security and hardening

pub mod discovery;
pub mod eclipse;
pub mod gossip;
pub mod mdns_discovery;
pub mod messages;
pub mod partition_detect;
pub mod rate_limit;
pub mod reputation;
pub mod runner;
pub mod sybil;
pub mod sync;
pub mod transport;
pub mod wire;

// Re-export commonly used types
pub use discovery::{ConnectedPeer, DiscoveryConfig, PeerDiscovery};
pub use gossip::{GossipConfig, GossipProtocol, GossipStats};
pub use mdns_discovery::{CombinedDiscovery, MdnsConfig, MdnsDiscovery, MdnsEvent, MdnsError};
pub use messages::{
    ConsensusMessage, DiscoveryMessage, GradientMessage, HeartbeatMessage, MessageDedup,
    MessagePayload, MessageRejectReason, MpcDataMessage, NetworkMessage, NodeCapabilities,
    PeerId, PeerInfo, PeerKeyRegistry, SyncMessage, TrainingMessage, TrainingParams,
};
pub use runner::{NetworkEvent, NetworkRunner, NetworkRunnerBuilder, NetworkRunnerConfig, ReconnectState};
pub use sync::{Checkpoint, NetworkState, StateSync, SyncConfig, SyncStatus};
pub use transport::{
    ConnectionPool, ConnectionState, PeerConnection, TcpTransport, Transport, TransportConfig,
    TransportError,
};
pub use wire::{
    BinaryGradient, FrameHeader, FrameReader, FrameWriter, MessageFlags, MessageTypeId,
    WireCodec, WireError, HEADER_SIZE, MAGIC, MAX_MESSAGE_SIZE, PROTOCOL_VERSION,
};

// Network security re-exports
pub use eclipse::{
    DiversityStats, EclipsePreventionConfig, EclipseResistantPeerManager, PeerNetworkInfo,
};
pub use partition_detect::{
    PartitionAction, PartitionDetectionConfig, PartitionDetector, PartitionDetectorStats,
    PartitionEvent, PartitionStatus, PeerConnectivity, PeerConnectivityInfo,
};
pub use rate_limit::{
    BlacklistEntry, PeerRateLimit, RateLimitConfig, RateLimitStats, RateLimiter,
};
pub use reputation::{
    BehaviorEvent, PeerReputation, ReputationConfig, ReputationDimension, ReputationManager,
    ReputationStats,
};
pub use sybil::{
    PeerStake, SelectionResult, SybilResistanceConfig, SybilResistantSelector,
};
