//! Peer Discovery for HELIX Network.
//!
//! Implements peer discovery mechanisms including bootstrap nodes,
//! DHT-based discovery, and peer exchange.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use super::messages::{
    DiscoveryMessage, MessagePayload, NetworkMessage, NodeCapabilities, PeerId, PeerInfo,
};

/// Configuration for peer discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Bootstrap node addresses.
    pub bootstrap_nodes: Vec<String>,
    /// Maximum number of peers to maintain.
    pub max_peers: usize,
    /// Peer discovery interval (seconds).
    pub discovery_interval_secs: u64,
    /// Peer timeout (seconds before considered stale).
    pub peer_timeout_secs: u64,
    /// Minimum reputation to keep peer.
    pub min_reputation: i32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            bootstrap_nodes: Vec::new(),
            max_peers: 50,
            discovery_interval_secs: 30,
            peer_timeout_secs: 120,
            min_reputation: -10,
        }
    }
}

/// Manages peer discovery and routing table.
pub struct PeerDiscovery {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: DiscoveryConfig,
    /// Known peers.
    peers: Arc<RwLock<HashMap<PeerId, PeerInfo>>>,
    /// Connected peers.
    connected: Arc<RwLock<HashMap<PeerId, ConnectedPeer>>>,
    /// Pending connection attempts.
    pending: Arc<RwLock<HashMap<PeerId, u64>>>,
}

/// Information about a connected peer.
#[derive(Debug, Clone)]
pub struct ConnectedPeer {
    /// Peer info.
    pub info: PeerInfo,
    /// Connection established time.
    pub connected_at: u64,
    /// Last message received.
    pub last_message: u64,
    /// Messages sent.
    pub messages_sent: u64,
    /// Messages received.
    pub messages_received: u64,
}

impl PeerDiscovery {
    /// Creates a new peer discovery manager.
    pub fn new(local_id: PeerId, config: DiscoveryConfig) -> Self {
        Self {
            local_id,
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(HashMap::new())),
            pending: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns our local peer ID.
    pub fn local_id(&self) -> &PeerId {
        &self.local_id
    }

    /// Adds a peer to the known peers list.
    pub async fn add_peer(&self, info: PeerInfo) {
        let mut peers = self.peers.write().await;
        if peers.len() < self.config.max_peers {
            peers.insert(info.id.clone(), info);
        }
    }

    /// Removes a peer from the known peers list.
    pub async fn remove_peer(&self, id: &PeerId) {
        let mut peers = self.peers.write().await;
        peers.remove(id);
        
        let mut connected = self.connected.write().await;
        connected.remove(id);
    }

    /// Gets info about a specific peer.
    pub async fn get_peer(&self, id: &PeerId) -> Option<PeerInfo> {
        let peers = self.peers.read().await;
        peers.get(id).cloned()
    }

    /// Gets all known peers.
    pub async fn get_all_peers(&self) -> Vec<PeerInfo> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }

    /// Gets all connected peers.
    pub async fn get_connected_peers(&self) -> Vec<ConnectedPeer> {
        let connected = self.connected.read().await;
        connected.values().cloned().collect()
    }

    /// Marks a peer as connected.
    pub async fn mark_connected(&self, info: PeerInfo) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let connected_peer = ConnectedPeer {
            info: info.clone(),
            connected_at: now,
            last_message: now,
            messages_sent: 0,
            messages_received: 0,
        };

        let mut connected = self.connected.write().await;
        connected.insert(info.id.clone(), connected_peer);

        // Also add to known peers
        let mut peers = self.peers.write().await;
        peers.insert(info.id.clone(), info);
    }

    /// Updates peer's last seen time.
    pub async fn update_last_seen(&self, id: &PeerId) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(id) {
            peer.last_seen = now;
        }

        let mut connected = self.connected.write().await;
        if let Some(conn) = connected.get_mut(id) {
            conn.last_message = now;
            conn.messages_received += 1;
        }
    }

    /// Updates peer reputation.
    pub async fn update_reputation(&self, id: &PeerId, delta: i32) {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(id) {
            peer.reputation = peer.reputation.saturating_add(delta);
            
            // Remove peer if reputation too low
            if peer.reputation < self.config.min_reputation {
                peers.remove(id);
            }
        }
    }

    /// Cleans up stale peers.
    pub async fn cleanup_stale_peers(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let timeout = self.config.peer_timeout_secs;

        let mut peers = self.peers.write().await;
        peers.retain(|_, peer| now - peer.last_seen < timeout);

        let mut connected = self.connected.write().await;
        connected.retain(|_, conn| now - conn.last_message < timeout);
    }

    /// Creates a join request message.
    pub fn create_join_request(&self, capabilities: NodeCapabilities, listen_addr: String) -> NetworkMessage {
        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Discovery(DiscoveryMessage::JoinRequest {
                capabilities,
                listen_addr,
            }),
        )
    }

    /// Creates a peer announcement message.
    pub fn create_announcement(&self, capabilities: NodeCapabilities, address: String) -> NetworkMessage {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let peer_info = PeerInfo {
            id: self.local_id.clone(),
            address,
            capabilities,
            last_seen: now,
            reputation: 0,
        };

        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Discovery(DiscoveryMessage::Announce { peer: peer_info }),
        )
    }

    /// Creates a get peers request.
    pub fn create_get_peers_request(&self) -> NetworkMessage {
        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Discovery(DiscoveryMessage::GetPeers),
        )
    }

    /// Handles an incoming discovery message.
    pub async fn handle_discovery_message(
        &self,
        sender: PeerId,
        msg: DiscoveryMessage,
    ) -> Option<NetworkMessage> {
        self.update_last_seen(&sender).await;

        match msg {
            DiscoveryMessage::JoinRequest { capabilities, listen_addr } => {
                let peer_info = PeerInfo {
                    id: sender.clone(),
                    address: listen_addr,
                    capabilities,
                    last_seen: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    reputation: 0,
                };
                
                self.add_peer(peer_info).await;
                
                // Send response with known peers
                let peers = self.get_all_peers().await;
                Some(NetworkMessage::new(
                    self.local_id.clone(),
                    MessagePayload::Discovery(DiscoveryMessage::JoinResponse {
                        accepted: true,
                        peers,
                    }),
                ))
            }
            DiscoveryMessage::JoinResponse { accepted, peers } => {
                if accepted {
                    for peer in peers {
                        self.add_peer(peer).await;
                    }
                }
                None
            }
            DiscoveryMessage::Announce { peer } => {
                self.add_peer(peer).await;
                None
            }
            DiscoveryMessage::GetPeers => {
                let peers = self.get_all_peers().await;
                Some(NetworkMessage::new(
                    self.local_id.clone(),
                    MessagePayload::Discovery(DiscoveryMessage::Peers { peers }),
                ))
            }
            DiscoveryMessage::Peers { peers } => {
                for peer in peers {
                    self.add_peer(peer).await;
                }
                None
            }
            DiscoveryMessage::Leave => {
                self.remove_peer(&sender).await;
                None
            }
        }
    }

    /// Gets peers by capability.
    pub async fn get_peers_with_capability<F>(&self, filter: F) -> Vec<PeerInfo>
    where
        F: Fn(&NodeCapabilities) -> bool,
    {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|p| filter(&p.capabilities))
            .cloned()
            .collect()
    }

    /// Gets n random peers.
    pub async fn get_random_peers(&self, n: usize) -> Vec<PeerInfo> {
        use rand::seq::SliceRandom;
        
        let peers = self.peers.read().await;
        let mut all_peers: Vec<_> = peers.values().cloned().collect();
        all_peers.shuffle(&mut rand::thread_rng());
        all_peers.truncate(n);
        all_peers
    }

    /// Number of known peers.
    pub async fn peer_count(&self) -> usize {
        self.peers.read().await.len()
    }

    /// Number of connected peers.
    pub async fn connected_count(&self) -> usize {
        self.connected.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_discovery_add() {
        let local_id = PeerId::random();
        let discovery = PeerDiscovery::new(local_id, DiscoveryConfig::default());

        let peer = PeerInfo {
            id: PeerId::random(),
            address: "127.0.0.1:8000".to_string(),
            capabilities: NodeCapabilities::default(),
            last_seen: 0,
            reputation: 0,
        };

        discovery.add_peer(peer.clone()).await;
        
        assert_eq!(discovery.peer_count().await, 1);
    }

    #[tokio::test]
    async fn test_peer_discovery_connected() {
        let local_id = PeerId::random();
        let discovery = PeerDiscovery::new(local_id, DiscoveryConfig::default());

        let peer = PeerInfo {
            id: PeerId::random(),
            address: "127.0.0.1:8000".to_string(),
            capabilities: NodeCapabilities::default(),
            last_seen: 0,
            reputation: 0,
        };

        discovery.mark_connected(peer).await;
        
        assert_eq!(discovery.connected_count().await, 1);
    }
}
