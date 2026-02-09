//! Network Runner - Wires GossipProtocol to actual network transport.
//!
//! This module bridges the gossip protocol logic with the TCP/TLS transport layer,
//! providing a complete network stack for HELIX P2P communication.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::mpsc;

use super::discovery::{DiscoveryConfig, PeerDiscovery};
use super::eclipse::{
    ConnectionType, DiscoverySource, EclipsePreventionConfig, EclipseResistantPeerManager,
    PeerNetworkInfo,
};
use super::gossip::{GossipConfig, GossipProtocol};
use super::messages::{
    DiscoveryMessage, GradientMessage, HeartbeatMessage, MessagePayload, NetworkMessage,
    NodeCapabilities, PeerId, PeerInfo, SyncMessage, TrainingMessage,
};
use super::partition_detect::{PartitionAction, PartitionDetectionConfig, PartitionDetector};
use super::rate_limit::{MessageType, RateLimitConfig, RateLimitResult, RateLimiter};
use super::sync::{StateSync, SyncConfig};
use super::transport::{ConnectionPool, TcpTransport, Transport, TransportConfig, TransportError};

/// Network event types.
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// New peer discovered.
    PeerDiscovered(PeerInfo),
    /// Peer disconnected.
    PeerDisconnected(PeerId),
    /// Training message received.
    TrainingMessage { from: PeerId, message: TrainingMessage },
    /// Gradient message received.
    GradientMessage { from: PeerId, message: GradientMessage },
    /// Sync message received.
    SyncMessage { from: PeerId, message: SyncMessage },
    /// Heartbeat received.
    Heartbeat { from: PeerId, message: HeartbeatMessage },
    /// Network error.
    Error { peer: Option<PeerId>, error: String },
}

/// Configuration for the network runner.
#[derive(Debug, Clone)]
pub struct NetworkRunnerConfig {
    /// Transport configuration.
    pub transport: TransportConfig,
    /// Gossip configuration.
    pub gossip: GossipConfig,
    /// Discovery configuration.
    pub discovery: DiscoveryConfig,
    /// Sync configuration.
    pub sync: SyncConfig,
    /// Rate limiting configuration.
    pub rate_limit: RateLimitConfig,
    /// Eclipse prevention configuration.
    pub eclipse: EclipsePreventionConfig,
    /// Partition detection configuration.
    pub partition_detect: PartitionDetectionConfig,
    /// Gossip send interval (ms).
    pub gossip_send_interval_ms: u64,
    /// Cache cleanup interval (seconds).
    pub cache_cleanup_interval_secs: u64,
    /// Maximum outbound messages per tick.
    pub max_outbound_per_tick: usize,
}

impl Default for NetworkRunnerConfig {
    fn default() -> Self {
        Self {
            transport: TransportConfig::default(),
            gossip: GossipConfig::default(),
            discovery: DiscoveryConfig::default(),
            sync: SyncConfig::default(),
            rate_limit: RateLimitConfig::default(),
            eclipse: EclipsePreventionConfig::default(),
            partition_detect: PartitionDetectionConfig::default(),
            gossip_send_interval_ms: 100,
            cache_cleanup_interval_secs: 60,
            max_outbound_per_tick: 50,
        }
    }
}

/// The network runner manages the complete network stack.
pub struct NetworkRunner {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: NetworkRunnerConfig,
    /// Transport layer.
    transport: Arc<TcpTransport>,
    /// Connection pool.
    pool: Arc<ConnectionPool>,
    /// Gossip protocol.
    gossip: Arc<GossipProtocol>,
    /// Peer discovery.
    discovery: Arc<PeerDiscovery>,
    /// State sync.
    sync: Arc<StateSync>,
    /// Rate limiter for per-peer and global rate limiting.
    rate_limiter: Arc<parking_lot::Mutex<RateLimiter>>,
    /// Eclipse-resistant peer manager for diversity enforcement.
    eclipse_manager: Arc<parking_lot::Mutex<EclipseResistantPeerManager>>,
    /// Network partition detector.
    partition_detector: Arc<PartitionDetector>,
    /// Event channel sender.
    event_tx: mpsc::Sender<NetworkEvent>,
    /// Event channel receiver (for external consumption).
    event_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<NetworkEvent>>>,
    /// Running state.
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Our capabilities.
    capabilities: NodeCapabilities,
    /// Our listen address.
    listen_addr: String,
    /// Whether training should be paused due to network issues
    /// (partition detection sets this to true when PauseTraining/Halt is recommended).
    training_paused: Arc<std::sync::atomic::AtomicBool>,
}

impl NetworkRunner {
    /// Creates a new network runner.
    pub fn new(
        local_id: PeerId,
        config: NetworkRunnerConfig,
        capabilities: NodeCapabilities,
    ) -> Result<Self, TransportError> {
        let transport = Arc::new(TcpTransport::new(local_id.clone(), config.transport.clone())?);
        let pool = Arc::new(ConnectionPool::new(transport.clone()));
        let gossip = Arc::new(GossipProtocol::new(local_id.clone(), config.gossip.clone()));
        let discovery = Arc::new(PeerDiscovery::new(local_id.clone(), config.discovery.clone()));
        let sync = Arc::new(StateSync::new(local_id.clone(), config.sync.clone()));

        let rate_limiter = Arc::new(parking_lot::Mutex::new(
            RateLimiter::new(config.rate_limit.clone()),
        ));
        let eclipse_manager = Arc::new(parking_lot::Mutex::new(
            EclipseResistantPeerManager::new(config.eclipse.clone()),
        ));
        let mut partition_detector = PartitionDetector::new(config.partition_detect.clone());
        partition_detector.set_our_addr(config.transport.listen_addr);
        let partition_detector = Arc::new(partition_detector);

        let (event_tx, event_rx) = mpsc::channel(10000);
        let listen_addr = config.transport.listen_addr.to_string();

        Ok(Self {
            local_id,
            config,
            transport,
            pool,
            gossip,
            discovery,
            sync,
            rate_limiter,
            eclipse_manager,
            partition_detector,
            event_tx,
            event_rx: Arc::new(tokio::sync::Mutex::new(event_rx)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            capabilities,
            listen_addr,
            training_paused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Starts the network runner.
    pub async fn start(&self) -> Result<(), TransportError> {
        // Start transport listener
        self.transport.start().await?;

        self.running.store(true, std::sync::atomic::Ordering::SeqCst);

        // Spawn message receive loop
        let running = self.running.clone();
        let transport = self.transport.clone();
        let gossip = self.gossip.clone();
        let discovery = self.discovery.clone();
        let sync = self.sync.clone();
        let event_tx = self.event_tx.clone();
        let pool = self.pool.clone();
        let rate_limiter = self.rate_limiter.clone();
        let eclipse_manager = self.eclipse_manager.clone();

        tokio::spawn(async move {
            Self::receive_loop(running, transport, gossip, discovery, sync, event_tx, pool, rate_limiter, eclipse_manager).await;
        });

        // Spawn gossip send loop
        let running = self.running.clone();
        let gossip = self.gossip.clone();
        let pool = self.pool.clone();
        let discovery = self.discovery.clone();
        let interval_ms = self.config.gossip_send_interval_ms;
        let max_per_tick = self.config.max_outbound_per_tick;

        tokio::spawn(async move {
            Self::gossip_send_loop(running, gossip, pool, discovery, interval_ms, max_per_tick).await;
        });

        // Spawn cache cleanup loop
        let running = self.running.clone();
        let gossip = self.gossip.clone();
        let discovery = self.discovery.clone();
        let cleanup_interval = self.config.cache_cleanup_interval_secs;
        let partition_detector = self.partition_detector.clone();
        let rate_limiter_cleanup = self.rate_limiter.clone();
        let event_tx_cleanup = self.event_tx.clone();
        let training_paused = self.training_paused.clone();

        tokio::spawn(async move {
            Self::cleanup_loop(
                running, gossip, discovery, cleanup_interval,
                partition_detector, rate_limiter_cleanup, event_tx_cleanup,
                training_paused,
            ).await;
        });

        Ok(())
    }

    /// Stops the network runner.
    pub async fn stop(&self) -> Result<(), TransportError> {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        self.transport.stop().await
    }

    /// Broadcasts a message via gossip.
    pub async fn broadcast(&self, payload: MessagePayload) {
        let message = NetworkMessage::new(self.local_id.clone(), payload);
        let peers = self.get_connected_peer_ids().await;
        self.gossip.broadcast(message, &peers).await;
    }

    /// Sends a direct message to a peer (not gossiped).
    pub async fn send_direct(&self, peer_id: &PeerId, payload: MessagePayload) -> Result<(), TransportError> {
        let message = NetworkMessage::new(self.local_id.clone(), payload);
        self.pool.send(peer_id, message).await
    }

    /// Connects to a peer.
    pub async fn connect_peer(&self, peer_info: PeerInfo) -> Result<(), TransportError> {
        let addr: SocketAddr = peer_info.address.parse()
            .map_err(|_| TransportError::PeerNotFound(format!("Invalid address: {}", peer_info.address)))?;

        // Check eclipse diversity before accepting outbound connection
        {
            let peer_net_info = PeerNetworkInfo::new(
                peer_info.id.clone(),
                Some(addr.ip()),
                ConnectionType::Outbound,
                DiscoverySource::Manual,
            );
            let mut eclipse = self.eclipse_manager.lock();
            if let Err(e) = eclipse.try_add_peer(peer_net_info) {
                log::warn!(
                    "Eclipse diversity check rejected outbound peer {}: {}",
                    peer_info.id, e
                );
                // Still allow the connection — eclipse is advisory for outbound
            }
        }

        // Register peer for partition monitoring
        self.partition_detector.register_peer(addr);

        self.pool.register_peer(peer_info.id.clone(), addr);
        self.discovery.mark_connected(peer_info.clone()).await;

        // Send join request
        let join_msg = self.discovery.create_join_request(
            self.capabilities.clone(),
            self.listen_addr.clone(),
        );
        self.pool.send(&peer_info.id, join_msg).await?;

        let _ = self.event_tx.send(NetworkEvent::PeerDiscovered(peer_info)).await;
        Ok(())
    }

    /// Disconnects from a peer.
    pub async fn disconnect_peer(&self, peer_id: &PeerId) {
        self.pool.unregister_peer(peer_id);
        self.discovery.remove_peer(peer_id).await;
        self.eclipse_manager.lock().remove_peer(peer_id);
        let _ = self.event_tx.send(NetworkEvent::PeerDisconnected(peer_id.clone())).await;
    }

    /// Gets the next network event.
    pub async fn next_event(&self) -> Option<NetworkEvent> {
        let mut rx = self.event_rx.lock().await;
        rx.recv().await
    }

    /// Gets the event receiver for external use.
    pub fn event_receiver(&self) -> Arc<tokio::sync::Mutex<mpsc::Receiver<NetworkEvent>>> {
        self.event_rx.clone()
    }

    /// Returns our peer ID.
    pub fn local_id(&self) -> &PeerId {
        &self.local_id
    }

    /// Returns the gossip protocol stats.
    pub async fn gossip_stats(&self) -> super::gossip::GossipStats {
        self.gossip.stats().await
    }

    /// Returns the number of connected peers.
    pub async fn peer_count(&self) -> usize {
        self.discovery.connected_count().await
    }

    /// Returns all connected peers.
    pub async fn connected_peers(&self) -> Vec<PeerInfo> {
        self.discovery.get_all_peers().await
    }

    /// Returns the state sync manager.
    pub fn sync(&self) -> &Arc<StateSync> {
        &self.sync
    }

    /// Returns the discovery manager.
    pub fn discovery(&self) -> &Arc<PeerDiscovery> {
        &self.discovery
    }

    /// Returns the rate limiter.
    pub fn rate_limiter(&self) -> &Arc<parking_lot::Mutex<RateLimiter>> {
        &self.rate_limiter
    }

    /// Returns the eclipse-resistant peer manager.
    pub fn eclipse_manager(&self) -> &Arc<parking_lot::Mutex<EclipseResistantPeerManager>> {
        &self.eclipse_manager
    }

    /// Returns the partition detector.
    pub fn partition_detector(&self) -> &Arc<PartitionDetector> {
        &self.partition_detector
    }

    /// Returns whether training is paused due to network partition.
    pub fn is_training_paused(&self) -> bool {
        self.training_paused.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn get_connected_peer_ids(&self) -> Vec<PeerId> {
        self.discovery.get_all_peers().await
            .into_iter()
            .map(|p| p.id)
            .collect()
    }

    /// Maps a message payload to a rate limit message type.
    fn payload_to_message_type(payload: &MessagePayload) -> MessageType {
        match payload {
            MessagePayload::Discovery(_) => MessageType::Discovery,
            MessagePayload::Training(_) => MessageType::Training,
            MessagePayload::Gradient(_) => MessageType::Gradient,
            MessagePayload::Sync(_) => MessageType::Sync,
            MessagePayload::Heartbeat(_) => MessageType::Heartbeat,
        }
    }

    async fn receive_loop(
        running: Arc<std::sync::atomic::AtomicBool>,
        transport: Arc<TcpTransport>,
        gossip: Arc<GossipProtocol>,
        discovery: Arc<PeerDiscovery>,
        sync: Arc<StateSync>,
        event_tx: mpsc::Sender<NetworkEvent>,
        pool: Arc<ConnectionPool>,
        rate_limiter: Arc<parking_lot::Mutex<RateLimiter>>,
        eclipse_manager: Arc<parking_lot::Mutex<EclipseResistantPeerManager>>,
    ) {
        while running.load(std::sync::atomic::Ordering::SeqCst) {
            match transport.recv().await {
                Ok((from, message)) => {
                    // === Rate limiting (Task B3.4) ===
                    let msg_type = Self::payload_to_message_type(&message.payload);
                    {
                        let mut limiter = rate_limiter.lock();
                        let result = limiter.check_rate_limit(&message.sender, msg_type);
                        match result {
                            RateLimitResult::Allowed => {}
                            RateLimitResult::Blacklisted => {
                                log::debug!(
                                    "Dropping message from blacklisted peer {}",
                                    message.sender
                                );
                                continue;
                            }
                            RateLimitResult::GlobalLimitExceeded => {
                                log::warn!("Global rate limit exceeded, dropping message from {}", message.sender);
                                continue;
                            }
                            RateLimitResult::PeerLimitExceeded { violations, .. } => {
                                log::debug!(
                                    "Rate limited peer {} (violations: {})",
                                    message.sender, violations
                                );
                                continue;
                            }
                            RateLimitResult::AutoBlacklisted => {
                                log::warn!(
                                    "Auto-blacklisted peer {} due to repeated violations",
                                    message.sender
                                );
                                continue;
                            }
                            RateLimitResult::ConnectionFlood => {
                                log::warn!("Connection flood from {}", message.sender);
                                continue;
                            }
                        }
                    }

                    // Update peer last seen
                    discovery.update_last_seen(&message.sender).await;

                    // Get available peers for gossip forwarding
                    let peers: Vec<PeerId> = discovery.get_all_peers().await
                        .into_iter()
                        .map(|p| p.id)
                        .collect();

                    // Handle via gossip (for dedup and forwarding)
                    let (should_process, _forward_to) = gossip.handle_message(
                        message.clone(),
                        &from,
                        &peers,
                    ).await;

                    if !should_process {
                        continue;
                    }

                    // Process message by type
                    match message.payload {
                        MessagePayload::Discovery(disc_msg) => {
                            if let Some(response) = discovery.handle_discovery_message(
                                message.sender.clone(),
                                disc_msg.clone(),
                            ).await {
                                // Send response directly
                                let _ = pool.send(&message.sender, response).await;
                            }

                            // === Inbound peer registration (Task B3.6) ===
                            // Register inbound peers on JoinRequest so we can route responses
                            if let DiscoveryMessage::JoinRequest { capabilities, listen_addr } = disc_msg {
                                // === Eclipse diversity check for inbound peers (Task B3.5) ===
                                if let Ok(addr) = listen_addr.parse::<SocketAddr>() {
                                    let peer_net_info = PeerNetworkInfo::new(
                                        message.sender.clone(),
                                        Some(addr.ip()),
                                        ConnectionType::Inbound,
                                        DiscoverySource::PeerExchange,
                                    );
                                    let eclipse_result = eclipse_manager.lock().try_add_peer(peer_net_info);
                                    if let Err(e) = eclipse_result {
                                        log::warn!(
                                            "Eclipse diversity check rejected inbound peer {}: {}",
                                            message.sender, e
                                        );
                                        continue; // Drop the connection — don't register
                                    }

                                    pool.register_peer(message.sender.clone(), addr);
                                }

                                let peer_info = PeerInfo {
                                    id: message.sender.clone(),
                                    address: listen_addr.clone(),
                                    capabilities,
                                    last_seen: message.timestamp,
                                    reputation: 0,
                                };

                                // Mark as connected in discovery
                                discovery.mark_connected(peer_info.clone()).await;

                                let _ = event_tx.send(NetworkEvent::PeerDiscovered(peer_info)).await;
                            }
                        }
                        MessagePayload::Training(train_msg) => {
                            let _ = event_tx.send(NetworkEvent::TrainingMessage {
                                from: message.sender,
                                message: train_msg,
                            }).await;
                        }
                        MessagePayload::Gradient(grad_msg) => {
                            let _ = event_tx.send(NetworkEvent::GradientMessage {
                                from: message.sender,
                                message: grad_msg,
                            }).await;
                        }
                        MessagePayload::Sync(sync_msg) => {
                            // Handle sync message and possibly respond
                            if let Some(response) = sync.handle_sync_message(
                                message.sender.clone(),
                                sync_msg.clone(),
                            ).await {
                                let _ = pool.send(&message.sender, response).await;
                            }

                            let _ = event_tx.send(NetworkEvent::SyncMessage {
                                from: message.sender,
                                message: sync_msg,
                            }).await;
                        }
                        MessagePayload::Heartbeat(hb_msg) => {
                            let _ = event_tx.send(NetworkEvent::Heartbeat {
                                from: message.sender,
                                message: hb_msg,
                            }).await;
                        }
                    }
                }
                Err(TransportError::ConnectionClosed) => {
                    // Connection closed, this is normal during shutdown
                    if !running.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                }
                Err(e) => {
                    let _ = event_tx.send(NetworkEvent::Error {
                        peer: None,
                        error: e.to_string(),
                    }).await;
                }
            }
        }
    }

    async fn gossip_send_loop(
        running: Arc<std::sync::atomic::AtomicBool>,
        gossip: Arc<GossipProtocol>,
        pool: Arc<ConnectionPool>,
        discovery: Arc<PeerDiscovery>,
        interval_ms: u64,
        max_per_tick: usize,
    ) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));

        while running.load(std::sync::atomic::Ordering::SeqCst) {
            interval.tick().await;

            // Take pending outbound messages
            let outbound = gossip.take_outbound(max_per_tick).await;

            for (peer_id, message) in outbound {
                // Get peer address from discovery
                if let Some(peer) = discovery.get_peer(&peer_id).await {
                    if let Ok(addr) = peer.address.parse::<SocketAddr>() {
                        pool.register_peer(peer_id.clone(), addr);
                    }
                }

                // Send message
                if let Err(e) = pool.send(&peer_id, message).await {
                    log::debug!("Failed to send gossip message to {:?}: {}", peer_id, e);
                }
            }
        }
    }

    async fn cleanup_loop(
        running: Arc<std::sync::atomic::AtomicBool>,
        gossip: Arc<GossipProtocol>,
        discovery: Arc<PeerDiscovery>,
        cleanup_interval_secs: u64,
        partition_detector: Arc<PartitionDetector>,
        rate_limiter: Arc<parking_lot::Mutex<RateLimiter>>,
        event_tx: mpsc::Sender<NetworkEvent>,
        training_paused: Arc<std::sync::atomic::AtomicBool>,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(cleanup_interval_secs));

        while running.load(std::sync::atomic::Ordering::SeqCst) {
            interval.tick().await;

            // Cleanup gossip cache
            gossip.cleanup_cache().await;

            // Cleanup stale peers
            discovery.cleanup_stale_peers().await;

            // Cleanup expired rate limiter entries
            rate_limiter.lock().cleanup();

            // === Partition detection (Task B3.5) ===
            match partition_detector.detect_partition().await {
                Ok(status) => {
                    if status.is_partitioned {
                        log::warn!(
                            "Network partition detected: {} connected, {} unreachable, action={:?}",
                            status.connected_peers,
                            status.unreachable_peers,
                            status.recommended_action,
                        );
                        match status.recommended_action {
                            PartitionAction::PauseTraining | PartitionAction::Halt => {
                                // Actually pause training — orchestrator checks this flag
                                training_paused.store(true, std::sync::atomic::Ordering::SeqCst);
                                log::warn!(
                                    "Training PAUSED due to network partition (action={:?})",
                                    status.recommended_action,
                                );
                                let _ = event_tx.send(NetworkEvent::Error {
                                    peer: None,
                                    error: format!(
                                        "Network partition: {} unreachable peers, training paused (action={:?})",
                                        status.unreachable_peers,
                                        status.recommended_action,
                                    ),
                                }).await;
                            }
                            _ => {}
                        }
                    } else {
                        // Network healthy — resume training if previously paused
                        if training_paused.load(std::sync::atomic::Ordering::SeqCst) {
                            training_paused.store(false, std::sync::atomic::Ordering::SeqCst);
                            log::info!("Training RESUMED: network partition resolved");
                        }
                    }
                }
                Err(e) => {
                    log::debug!("Partition detection skipped: {}", e);
                }
            }
        }
    }
}

/// Builder for NetworkRunner.
pub struct NetworkRunnerBuilder {
    local_id: Option<PeerId>,
    config: NetworkRunnerConfig,
    capabilities: NodeCapabilities,
}

impl NetworkRunnerBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            local_id: None,
            config: NetworkRunnerConfig::default(),
            capabilities: NodeCapabilities::default(),
        }
    }

    /// Sets the local peer ID.
    pub fn local_id(mut self, id: PeerId) -> Self {
        self.local_id = Some(id);
        self
    }

    /// Sets the listen address.
    pub fn listen_addr(mut self, addr: SocketAddr) -> Self {
        self.config.transport.listen_addr = addr;
        self
    }

    /// Enables TLS.
    pub fn with_tls(mut self, cert_path: String, key_path: String) -> Self {
        self.config.transport.use_tls = true;
        self.config.transport.cert_path = Some(cert_path);
        self.config.transport.key_path = Some(key_path);
        self
    }

    /// Sets bootstrap nodes.
    pub fn bootstrap_nodes(mut self, nodes: Vec<String>) -> Self {
        self.config.discovery.bootstrap_nodes = nodes;
        self
    }

    /// Sets node capabilities.
    pub fn capabilities(mut self, caps: NodeCapabilities) -> Self {
        self.capabilities = caps;
        self
    }

    /// Sets gossip fanout.
    pub fn gossip_fanout(mut self, fanout: usize) -> Self {
        self.config.gossip.fanout = fanout;
        self
    }

    /// Builds the network runner.
    pub fn build(self) -> Result<NetworkRunner, TransportError> {
        let local_id = self.local_id.unwrap_or_else(PeerId::random);
        NetworkRunner::new(local_id, self.config, self.capabilities)
    }
}

impl Default for NetworkRunnerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_runner_builder() {
        let runner = NetworkRunnerBuilder::new()
            .local_id(PeerId::random())
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .gossip_fanout(4)
            .build();

        assert!(runner.is_ok());
    }

    #[tokio::test]
    async fn test_network_runner_creation() {
        let local_id = PeerId::random();
        let config = NetworkRunnerConfig::default();
        let caps = NodeCapabilities::default();

        let runner = NetworkRunner::new(local_id, config, caps);
        assert!(runner.is_ok());
    }
}
