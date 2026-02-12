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
    ConsensusMessage, DiscoveryMessage, GradientMessage, HeartbeatMessage, MessagePayload,
    NetworkMessage, NodeCapabilities, PeerId, PeerInfo, PeerKeyRegistry, SyncMessage,
    TrainingMessage,
};
use super::partition_detect::{PartitionAction, PartitionDetectionConfig, PartitionDetector};
use super::rate_limit::{BlacklistReason, MessageType, RateLimitConfig, RateLimitResult, RateLimiter};
use super::reputation::{BehaviorEvent, BehaviorEventType, ReputationConfig, ReputationManager};
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
    /// BFT consensus message received (2-phase commit for gradient aggregation).
    ConsensusMessage { from: PeerId, message: ConsensusMessage },
    /// MPC protocol data received (Beaver triples, secret shares, etc.)
    MpcDataMessage { from: PeerId, message: super::messages::MpcDataMessage },
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
    /// Reputation scoring configuration.
    pub reputation: ReputationConfig,
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
            reputation: ReputationConfig::default(),
            gossip_send_interval_ms: 100,
            cache_cleanup_interval_secs: 60,
            max_outbound_per_tick: 50,
        }
    }
}

/// Tracks per-peer invalid signature counts for auto-blacklisting.
struct SignatureStats {
    /// Invalid signature count per peer.
    invalid_counts: HashMap<PeerId, u32>,
    /// Total invalid signatures across all peers.
    total_invalid: u64,
    /// Threshold: peers exceeding this many invalid signatures are auto-blacklisted.
    auto_blacklist_threshold: u32,
}

impl SignatureStats {
    fn new(auto_blacklist_threshold: u32) -> Self {
        Self {
            invalid_counts: HashMap::new(),
            total_invalid: 0,
            auto_blacklist_threshold,
        }
    }

    /// Records an invalid signature from a peer. Returns true if the peer
    /// should be auto-blacklisted (exceeded threshold).
    fn record_invalid(&mut self, peer_id: &PeerId) -> bool {
        self.total_invalid += 1;
        let count = self.invalid_counts.entry(peer_id.clone()).or_insert(0);
        *count += 1;
        *count > self.auto_blacklist_threshold
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
    /// Peer key registry for signature verification.
    peer_keys: Arc<parking_lot::Mutex<PeerKeyRegistry>>,
    /// Signature validation statistics and auto-blacklist tracking.
    sig_stats: Arc<parking_lot::Mutex<SignatureStats>>,
    /// Peer reputation manager for scoring and behavioral tracking.
    reputation_manager: Arc<parking_lot::Mutex<ReputationManager>>,
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
        let pool = Arc::new(ConnectionPool::with_max_size(
            transport.clone(),
            config.transport.max_pool_size,
        ));
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

        let reputation_manager = Arc::new(parking_lot::Mutex::new(
            ReputationManager::new(config.reputation.clone()),
        ));

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
            peer_keys: Arc::new(parking_lot::Mutex::new(PeerKeyRegistry::new())),
            sig_stats: Arc::new(parking_lot::Mutex::new(SignatureStats::new(10))),
            reputation_manager,
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
        let peer_keys = self.peer_keys.clone();
        let sig_stats = self.sig_stats.clone();
        let reputation_manager = self.reputation_manager.clone();

        tokio::spawn(async move {
            Self::receive_loop(running, transport, gossip, discovery, sync, event_tx, pool, rate_limiter, eclipse_manager, peer_keys, sig_stats, reputation_manager).await;
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
        let reputation_cleanup = self.reputation_manager.clone();

        tokio::spawn(async move {
            Self::cleanup_loop(
                running, gossip, discovery, cleanup_interval,
                partition_detector, rate_limiter_cleanup, event_tx_cleanup,
                training_paused, reputation_cleanup,
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
        self.reputation_manager.lock().record_disconnect(peer_id, true);
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

    /// Returns the peer key registry (for registering peer public keys).
    pub fn peer_keys(&self) -> &Arc<parking_lot::Mutex<PeerKeyRegistry>> {
        &self.peer_keys
    }

    /// Returns the signature validation statistics.
    pub fn signature_stats(&self) -> (u64, HashMap<PeerId, u32>) {
        let stats = self.sig_stats.lock();
        (stats.total_invalid, stats.invalid_counts.clone())
    }

    /// Returns the reputation manager.
    pub fn reputation_manager(&self) -> &Arc<parking_lot::Mutex<ReputationManager>> {
        &self.reputation_manager
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
            // Consensus messages use the Training rate limit bucket
            // since they are part of the training coordination flow.
            MessagePayload::Consensus(_) => MessageType::Training,
            // MPC messages use the Training rate limit bucket.
            MessagePayload::MpcData(_) => MessageType::Training,
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
        peer_keys: Arc<parking_lot::Mutex<PeerKeyRegistry>>,
        sig_stats: Arc<parking_lot::Mutex<SignatureStats>>,
        reputation_manager: Arc<parking_lot::Mutex<ReputationManager>>,
    ) {
        while running.load(std::sync::atomic::Ordering::SeqCst) {
            match transport.recv().await {
                Ok((from, message)) => {
                    // === Signature verification ===
                    // Verify message signature against the sender's registered
                    // public key. When the `crypto-sign` feature is disabled,
                    // this is a no-op that always passes (see PeerKeyRegistry
                    // docs for the security trade-off).
                    {
                        let keys = peer_keys.lock();
                        if !keys.verify_message(&message) {
                            let mut stats = sig_stats.lock();
                            let should_blacklist = stats.record_invalid(&message.sender);
                            log::warn!(
                                "Invalid signature from peer {} (total invalid: {})",
                                message.sender, stats.total_invalid,
                            );

                            // Record reputation event for signature failure
                            reputation_manager.lock().record_event(
                                &message.sender,
                                BehaviorEvent::new(BehaviorEventType::ProtocolViolation { severity: 3 }),
                            );

                            if should_blacklist {
                                log::warn!(
                                    "Auto-blacklisting peer {} after {} invalid signatures",
                                    message.sender, stats.auto_blacklist_threshold,
                                );
                                // Record severe reputation event for auto-blacklist
                                reputation_manager.lock().record_event(
                                    &message.sender,
                                    BehaviorEvent::new(BehaviorEventType::ProtocolViolation { severity: 5 }),
                                );
                                // Blacklist via rate limiter to reuse existing infra
                                rate_limiter.lock().blacklist_peer(
                                    &message.sender,
                                    BlacklistReason::MaliciousBehavior {
                                        details: "Repeated invalid signatures".to_string(),
                                    },
                                );
                            }
                            continue;
                        }
                    }

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
                                reputation_manager.lock().record_event(
                                    &message.sender,
                                    BehaviorEvent::new(BehaviorEventType::ProtocolViolation { severity: 5 }),
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

                    // Record positive reputation event for valid messages that passed all checks
                    reputation_manager.lock().record_heartbeat(&message.sender);

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

                                let rep_score = reputation_manager.lock().score(&message.sender) as i32;
                                let peer_info = PeerInfo {
                                    id: message.sender.clone(),
                                    address: listen_addr.clone(),
                                    capabilities,
                                    last_seen: message.timestamp,
                                    reputation: rep_score,
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
                        MessagePayload::Consensus(consensus_msg) => {
                            let _ = event_tx.send(NetworkEvent::ConsensusMessage {
                                from: message.sender,
                                message: consensus_msg,
                            }).await;
                        }
                        MessagePayload::MpcData(mpc_msg) => {
                            let _ = event_tx.send(NetworkEvent::MpcDataMessage {
                                from: message.sender,
                                message: mpc_msg,
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
        reputation_manager: Arc<parking_lot::Mutex<ReputationManager>>,
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

            // Apply reputation decay and check ban expiries
            reputation_manager.lock().apply_decay();

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

    #[test]
    fn test_payload_to_message_type_mapping() {
        // Discovery
        let disc = MessagePayload::Discovery(DiscoveryMessage::GetPeers);
        assert!(matches!(NetworkRunner::payload_to_message_type(&disc), MessageType::Discovery));

        // Training
        let train = MessagePayload::Training(TrainingMessage::ParticipateRequest { round_id: 1 });
        assert!(matches!(NetworkRunner::payload_to_message_type(&train), MessageType::Training));

        // Gradient
        let grad = MessagePayload::Gradient(GradientMessage::ShareGradient {
            round_id: 1,
            gradient_commitment: [0u8; 32],
            commitment_nonce: [0u8; 16],
            error_bound: 0.1,
            proof: vec![],
        });
        assert!(matches!(NetworkRunner::payload_to_message_type(&grad), MessageType::Gradient));

        // Consensus maps to Training bucket
        let consensus = MessagePayload::Consensus(ConsensusMessage::Propose {
            round_id: 1,
            aggregated_commitment: [0u8; 32],
            proposer_binding: [0u8; 32],
            num_gradients: 3,
            error_bound: 0.1,
            nonce: [0u8; 16],
        });
        assert!(matches!(NetworkRunner::payload_to_message_type(&consensus), MessageType::Training));
    }

    #[test]
    fn test_signature_stats_tracking() {
        let mut stats = SignatureStats::new(3);
        let peer = PeerId::from_string("bad-peer");

        assert!(!stats.record_invalid(&peer)); // 1st - below threshold
        assert!(!stats.record_invalid(&peer)); // 2nd - below threshold
        assert!(!stats.record_invalid(&peer)); // 3rd - at threshold
        assert!(stats.record_invalid(&peer));  // 4th - exceeds threshold

        assert_eq!(stats.total_invalid, 4);
        assert_eq!(*stats.invalid_counts.get(&peer).unwrap(), 4);
    }

    #[tokio::test]
    async fn test_reputation_manager_accessible() {
        let runner = NetworkRunnerBuilder::new()
            .local_id(PeerId::random())
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap();

        // Reputation manager should be initialized with default config
        let rep_mgr = runner.reputation_manager();
        let mgr = rep_mgr.lock();
        assert_eq!(mgr.peer_count(), 0);

        // Score for unknown peer should be the initial score (50.0)
        let peer = PeerId::from_string("test-peer");
        assert_eq!(mgr.score(&peer), 50.0);
    }

    #[test]
    fn test_builder_with_tls_config() {
        // Verify builder accepts TLS config and capabilities without panicking.
        // Note: build() may fail due to non-existent cert paths, so we only
        // test that the builder pattern itself works.
        let builder = NetworkRunnerBuilder::new()
            .local_id(PeerId::random())
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .with_tls("/path/to/cert.pem".into(), "/path/to/key.pem".into())
            .gossip_fanout(6)
            .capabilities(NodeCapabilities {
                can_train: true,
                can_aggregate: true,
                can_prove: false,
                gpu_memory_mb: 4096,
                cpu_cores: 8,
                storage_gb: 200,
            });

        // Builder should be configured (we verify by building without TLS)
        let runner_no_tls = NetworkRunnerBuilder::new()
            .local_id(PeerId::random())
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .gossip_fanout(6)
            .capabilities(NodeCapabilities {
                can_train: true,
                can_aggregate: true,
                can_prove: false,
                gpu_memory_mb: 4096,
                cpu_cores: 8,
                storage_gb: 200,
            })
            .build();
        assert!(runner_no_tls.is_ok());
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = NetworkRunnerConfig::default();
        assert_eq!(config.gossip_send_interval_ms, 100);
        assert_eq!(config.cache_cleanup_interval_secs, 60);
        assert_eq!(config.max_outbound_per_tick, 50);
        // Reputation config should be initialized with defaults
        assert_eq!(config.reputation.initial_score, 50.0);
        assert_eq!(config.reputation.ban_threshold, 10.0);
    }
}
