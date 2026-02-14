//! Peer Registry for HELIX Network.
//!
//! Provides a centralized, thread-safe registry of connected peers with:
//! - Role-based queries (find all aggregators, all workers, etc.)
//! - Heartbeat tracking and health monitoring
//! - Automatic health degradation when heartbeats are missed
//! - Connection statistics (messages sent/received, uptime)
//!
//! The registry is designed to be shared across the connection manager,
//! message router, and role-specific logic via `Arc<PeerRegistry>`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::config::NodeRole;
use super::messages::{NodeCapabilities, PeerId};

/// Health status of a connected peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerHealth {
    /// Peer is responsive and sending heartbeats on time.
    Healthy,
    /// Peer has missed some heartbeats but is not yet considered dead.
    Degraded {
        /// Number of consecutive missed heartbeats.
        missed_heartbeats: u32,
    },
    /// Peer is considered disconnected (too many missed heartbeats).
    Disconnected,
}

impl PeerHealth {
    /// Returns true if the peer is healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, PeerHealth::Healthy)
    }

    /// Returns true if the peer is disconnected.
    pub fn is_disconnected(&self) -> bool {
        matches!(self, PeerHealth::Disconnected)
    }
}

/// Information about a registered (connected) peer.
pub struct RegisteredPeer {
    /// Peer's unique ID.
    pub peer_id: PeerId,
    /// Peer's role in the network.
    pub role: NodeRole,
    /// Peer's capabilities.
    pub capabilities: NodeCapabilities,
    /// Peer's listen address (where other peers can connect to it).
    pub listen_addr: SocketAddr,
    /// Peer's ed25519 public key, if provided during handshake.
    pub public_key: Option<Vec<u8>>,
    /// When this peer connected.
    pub connected_at: Instant,
    /// Last time we received a heartbeat (or any message).
    pub last_heartbeat: Instant,
    /// Current health status.
    pub health: PeerHealth,
    /// Last heartbeat sequence number received.
    pub heartbeat_seq: u64,
    /// Total messages sent to this peer.
    pub messages_sent: AtomicU64,
    /// Total messages received from this peer.
    pub messages_received: AtomicU64,
}

/// Snapshot of a registered peer (no atomics, cheaply cloneable).
#[derive(Debug, Clone)]
pub struct PeerSnapshot {
    pub peer_id: PeerId,
    pub role: NodeRole,
    pub capabilities: NodeCapabilities,
    pub listen_addr: SocketAddr,
    pub public_key: Option<Vec<u8>>,
    pub connected_at: Instant,
    pub last_heartbeat: Instant,
    pub health: PeerHealth,
    pub heartbeat_seq: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
}

impl RegisteredPeer {
    /// Creates a snapshot of this peer.
    pub fn snapshot(&self) -> PeerSnapshot {
        PeerSnapshot {
            peer_id: self.peer_id.clone(),
            role: self.role,
            capabilities: self.capabilities.clone(),
            listen_addr: self.listen_addr,
            public_key: self.public_key.clone(),
            connected_at: self.connected_at,
            last_heartbeat: self.last_heartbeat,
            health: self.health,
            heartbeat_seq: self.heartbeat_seq,
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
        }
    }

    /// Returns how long this peer has been connected.
    pub fn uptime(&self) -> Duration {
        self.connected_at.elapsed()
    }

    /// Returns how long since the last heartbeat.
    pub fn since_last_heartbeat(&self) -> Duration {
        self.last_heartbeat.elapsed()
    }
}

/// Thread-safe registry of connected peers.
pub struct PeerRegistry {
    peers: RwLock<HashMap<PeerId, RegisteredPeer>>,
    /// Maximum peers to accept (0 = unlimited).
    max_peers: usize,
}

impl PeerRegistry {
    /// Creates a new peer registry.
    pub fn new(max_peers: usize) -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            max_peers,
        }
    }

    /// Registers a new peer. Returns false if at capacity.
    pub fn register(
        &self,
        peer_id: PeerId,
        role: NodeRole,
        capabilities: NodeCapabilities,
        listen_addr: SocketAddr,
        public_key: Option<Vec<u8>>,
    ) -> bool {
        let mut peers = self.peers.write();

        // Allow re-registration (update existing)
        if peers.contains_key(&peer_id) {
            if let Some(existing) = peers.get_mut(&peer_id) {
                existing.role = role;
                existing.capabilities = capabilities;
                existing.listen_addr = listen_addr;
                existing.public_key = public_key;
                existing.last_heartbeat = Instant::now();
                existing.health = PeerHealth::Healthy;
            }
            return true;
        }

        // Check capacity
        if self.max_peers > 0 && peers.len() >= self.max_peers {
            return false;
        }

        let now = Instant::now();
        peers.insert(peer_id.clone(), RegisteredPeer {
            peer_id,
            role,
            capabilities,
            listen_addr,
            public_key,
            connected_at: now,
            last_heartbeat: now,
            health: PeerHealth::Healthy,
            heartbeat_seq: 0,
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
        });

        true
    }

    /// Removes a peer from the registry.
    pub fn deregister(&self, peer_id: &PeerId) -> bool {
        self.peers.write().remove(peer_id).is_some()
    }

    /// Returns a snapshot of a specific peer.
    pub fn get(&self, peer_id: &PeerId) -> Option<PeerSnapshot> {
        self.peers.read().get(peer_id).map(|p| p.snapshot())
    }

    /// Returns true if the peer is registered.
    pub fn contains(&self, peer_id: &PeerId) -> bool {
        self.peers.read().contains_key(peer_id)
    }

    /// Returns the listen address for a peer.
    pub fn get_addr(&self, peer_id: &PeerId) -> Option<SocketAddr> {
        self.peers.read().get(peer_id).map(|p| p.listen_addr)
    }

    /// Returns the number of registered peers.
    pub fn count(&self) -> usize {
        self.peers.read().len()
    }

    /// Returns the IDs of all registered peers.
    pub fn peer_ids(&self) -> Vec<PeerId> {
        self.peers.read().keys().cloned().collect()
    }

    /// Returns snapshots of all registered peers.
    pub fn all_peers(&self) -> Vec<PeerSnapshot> {
        self.peers.read().values().map(|p| p.snapshot()).collect()
    }

    /// Returns snapshots of peers with a specific role.
    pub fn peers_by_role(&self, role: NodeRole) -> Vec<PeerSnapshot> {
        self.peers.read()
            .values()
            .filter(|p| p.role == role)
            .map(|p| p.snapshot())
            .collect()
    }

    /// Returns snapshots of healthy peers only.
    pub fn healthy_peers(&self) -> Vec<PeerSnapshot> {
        self.peers.read()
            .values()
            .filter(|p| p.health.is_healthy())
            .map(|p| p.snapshot())
            .collect()
    }

    /// Returns snapshots of peers that match a capability filter.
    pub fn peers_with_capability<F: Fn(&NodeCapabilities) -> bool>(
        &self,
        filter: F,
    ) -> Vec<PeerSnapshot> {
        self.peers.read()
            .values()
            .filter(|p| filter(&p.capabilities))
            .map(|p| p.snapshot())
            .collect()
    }

    /// Records a heartbeat from a peer. Updates last_heartbeat and seq.
    pub fn record_heartbeat(&self, peer_id: &PeerId, seq: u64) {
        let mut peers = self.peers.write();
        if let Some(peer) = peers.get_mut(peer_id) {
            peer.last_heartbeat = Instant::now();
            peer.heartbeat_seq = seq;
            peer.health = PeerHealth::Healthy;
        }
    }

    /// Records that a message was received from a peer.
    pub fn record_message_received(&self, peer_id: &PeerId) {
        let peers = self.peers.read();
        if let Some(peer) = peers.get(peer_id) {
            peer.messages_received.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records that a message was sent to a peer.
    pub fn record_message_sent(&self, peer_id: &PeerId) {
        let peers = self.peers.read();
        if let Some(peer) = peers.get(peer_id) {
            peer.messages_sent.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Checks health of all peers based on heartbeat freshness.
    /// Returns the list of peers that transitioned to Disconnected.
    pub fn check_health(
        &self,
        heartbeat_timeout: Duration,
        max_missed: u32,
    ) -> Vec<PeerId> {
        let mut disconnected = Vec::new();
        let mut peers = self.peers.write();

        for peer in peers.values_mut() {
            let elapsed = peer.last_heartbeat.elapsed();

            if elapsed < heartbeat_timeout {
                peer.health = PeerHealth::Healthy;
            } else {
                let missed = (elapsed.as_secs() / heartbeat_timeout.as_secs().max(1)) as u32;
                if missed > max_missed {
                    if !peer.health.is_disconnected() {
                        disconnected.push(peer.peer_id.clone());
                    }
                    peer.health = PeerHealth::Disconnected;
                } else {
                    peer.health = PeerHealth::Degraded {
                        missed_heartbeats: missed,
                    };
                }
            }
        }

        disconnected
    }

    /// Removes all disconnected peers and returns their info.
    pub fn remove_disconnected(&self) -> Vec<PeerSnapshot> {
        let mut peers = self.peers.write();
        let disconnected: Vec<PeerId> = peers
            .values()
            .filter(|p| p.health.is_disconnected())
            .map(|p| p.peer_id.clone())
            .collect();

        disconnected
            .into_iter()
            .filter_map(|id| peers.remove(&id).map(|p| p.snapshot()))
            .collect()
    }

    /// Returns summary statistics.
    pub fn stats(&self) -> RegistryStats {
        let peers = self.peers.read();
        let mut healthy = 0;
        let mut degraded = 0;
        let mut disconnected = 0;
        let mut by_role: HashMap<NodeRole, usize> = HashMap::new();

        for peer in peers.values() {
            match peer.health {
                PeerHealth::Healthy => healthy += 1,
                PeerHealth::Degraded { .. } => degraded += 1,
                PeerHealth::Disconnected => disconnected += 1,
            }
            *by_role.entry(peer.role).or_insert(0) += 1;
        }

        RegistryStats {
            total: peers.len(),
            healthy,
            degraded,
            disconnected,
            by_role,
        }
    }
}

/// Summary statistics from the peer registry.
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total: usize,
    pub healthy: usize,
    pub degraded: usize,
    pub disconnected: usize,
    pub by_role: HashMap<NodeRole, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_caps(train: bool, aggregate: bool) -> NodeCapabilities {
        NodeCapabilities {
            can_train: train,
            can_aggregate: aggregate,
            can_prove: false,
            gpu_memory_mb: 0,
            cpu_cores: 4,
            storage_gb: 50,
        }
    }

    #[test]
    fn test_register_and_get() {
        let registry = PeerRegistry::new(10);
        let peer_id = PeerId::random();
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        assert!(registry.register(
            peer_id.clone(),
            NodeRole::Compute,
            make_caps(true, false),
            addr,
            None,
        ));

        assert_eq!(registry.count(), 1);
        assert!(registry.contains(&peer_id));

        let snap = registry.get(&peer_id).unwrap();
        assert_eq!(snap.role, NodeRole::Compute);
        assert_eq!(snap.listen_addr, addr);
        assert!(snap.capabilities.can_train);
        assert!(snap.health.is_healthy());
    }

    #[test]
    fn test_register_at_capacity() {
        let registry = PeerRegistry::new(2);
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        assert!(registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None));
        assert!(registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None));
        // Third should fail
        assert!(!registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None));
        assert_eq!(registry.count(), 2);
    }

    #[test]
    fn test_register_unlimited() {
        let registry = PeerRegistry::new(0);
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        for _ in 0..100 {
            assert!(registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None));
        }
        assert_eq!(registry.count(), 100);
    }

    #[test]
    fn test_reregister_updates() {
        let registry = PeerRegistry::new(10);
        let peer_id = PeerId::random();
        let addr1: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:9001".parse().unwrap();

        assert!(registry.register(peer_id.clone(), NodeRole::Compute, make_caps(true, false), addr1, None));
        assert!(registry.register(peer_id.clone(), NodeRole::Aggregator, make_caps(false, true), addr2, None));

        assert_eq!(registry.count(), 1);
        let snap = registry.get(&peer_id).unwrap();
        assert_eq!(snap.role, NodeRole::Aggregator);
        assert_eq!(snap.listen_addr, addr2);
    }

    #[test]
    fn test_deregister() {
        let registry = PeerRegistry::new(10);
        let peer_id = PeerId::random();
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(peer_id.clone(), NodeRole::Compute, make_caps(true, false), addr, None);
        assert!(registry.deregister(&peer_id));
        assert!(!registry.contains(&peer_id));
        assert_eq!(registry.count(), 0);

        // Deregister non-existent
        assert!(!registry.deregister(&PeerId::random()));
    }

    #[test]
    fn test_peers_by_role() {
        let registry = PeerRegistry::new(10);
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None);
        registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None);
        registry.register(PeerId::random(), NodeRole::Aggregator, make_caps(false, true), addr, None);

        let workers = registry.peers_by_role(NodeRole::Compute);
        assert_eq!(workers.len(), 2);

        let aggregators = registry.peers_by_role(NodeRole::Aggregator);
        assert_eq!(aggregators.len(), 1);

        let verifiers = registry.peers_by_role(NodeRole::Verifier);
        assert_eq!(verifiers.len(), 0);
    }

    #[test]
    fn test_peers_with_capability() {
        let registry = PeerRegistry::new(10);
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None);
        registry.register(PeerId::random(), NodeRole::Aggregator, make_caps(false, true), addr, None);

        let trainers = registry.peers_with_capability(|c| c.can_train);
        assert_eq!(trainers.len(), 1);

        let aggregators = registry.peers_with_capability(|c| c.can_aggregate);
        assert_eq!(aggregators.len(), 1);
    }

    #[test]
    fn test_heartbeat_tracking() {
        let registry = PeerRegistry::new(10);
        let peer_id = PeerId::random();
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(peer_id.clone(), NodeRole::Compute, make_caps(true, false), addr, None);

        registry.record_heartbeat(&peer_id, 1);
        let snap = registry.get(&peer_id).unwrap();
        assert_eq!(snap.heartbeat_seq, 1);
        assert!(snap.health.is_healthy());

        registry.record_heartbeat(&peer_id, 5);
        let snap = registry.get(&peer_id).unwrap();
        assert_eq!(snap.heartbeat_seq, 5);
    }

    #[test]
    fn test_message_counters() {
        let registry = PeerRegistry::new(10);
        let peer_id = PeerId::random();
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(peer_id.clone(), NodeRole::Compute, make_caps(true, false), addr, None);

        registry.record_message_sent(&peer_id);
        registry.record_message_sent(&peer_id);
        registry.record_message_received(&peer_id);

        let snap = registry.get(&peer_id).unwrap();
        assert_eq!(snap.messages_sent, 2);
        assert_eq!(snap.messages_received, 1);
    }

    #[test]
    fn test_health_check() {
        let registry = PeerRegistry::new(10);
        let peer_id = PeerId::random();
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(peer_id.clone(), NodeRole::Compute, make_caps(true, false), addr, None);

        // Immediately after registration, should be healthy
        let disconnected = registry.check_health(Duration::from_secs(5), 3);
        assert!(disconnected.is_empty());

        let snap = registry.get(&peer_id).unwrap();
        assert!(snap.health.is_healthy());
    }

    #[test]
    fn test_stats() {
        let registry = PeerRegistry::new(10);
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None);
        registry.register(PeerId::random(), NodeRole::Compute, make_caps(true, false), addr, None);
        registry.register(PeerId::random(), NodeRole::Aggregator, make_caps(false, true), addr, None);

        let stats = registry.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.healthy, 3);
        assert_eq!(stats.degraded, 0);
        assert_eq!(stats.disconnected, 0);
        assert_eq!(*stats.by_role.get(&NodeRole::Compute).unwrap(), 2);
        assert_eq!(*stats.by_role.get(&NodeRole::Aggregator).unwrap(), 1);
    }

    #[test]
    fn test_remove_disconnected() {
        let registry = PeerRegistry::new(10);
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        registry.register(peer1.clone(), NodeRole::Compute, make_caps(true, false), addr, None);
        registry.register(peer2.clone(), NodeRole::Compute, make_caps(true, false), addr, None);

        // Manually set peer1 to Disconnected
        {
            let mut peers = registry.peers.write();
            if let Some(p) = peers.get_mut(&peer1) {
                p.health = PeerHealth::Disconnected;
            }
        }

        let removed = registry.remove_disconnected();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].peer_id, peer1);
        assert_eq!(registry.count(), 1);
        assert!(registry.contains(&peer2));
    }

    #[test]
    fn test_get_addr() {
        let registry = PeerRegistry::new(10);
        let peer_id = PeerId::random();
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        registry.register(peer_id.clone(), NodeRole::Compute, make_caps(true, false), addr, None);

        assert_eq!(registry.get_addr(&peer_id), Some(addr));
        assert_eq!(registry.get_addr(&PeerId::random()), None);
    }

    #[test]
    fn test_peer_ids() {
        let registry = PeerRegistry::new(10);
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        let id1 = PeerId::random();
        let id2 = PeerId::random();

        registry.register(id1.clone(), NodeRole::Compute, make_caps(true, false), addr, None);
        registry.register(id2.clone(), NodeRole::Compute, make_caps(true, false), addr, None);

        let ids = registry.peer_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }
}
