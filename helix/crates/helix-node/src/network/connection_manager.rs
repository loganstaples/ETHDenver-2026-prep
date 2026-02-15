//! Connection Manager for HELIX P2P Network.
//!
//! High-level orchestrator that manages the full lifecycle of peer connections:
//! - TCP listener for accepting inbound connections
//! - Outbound connection establishment with handshake
//! - Periodic heartbeat sending and health monitoring
//! - Automatic reconnection with exponential backoff
//! - Message routing via length-prefixed bincode framing
//! - Integration with [`PeerRegistry`] for peer tracking
//!
//! The connection manager sits on top of raw TCP (or TLS) connections and
//! the handshake protocol, providing a clean API for sending and receiving
//! [`NetworkMessage`]s between authenticated peers.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

use crate::config::NodeRole;
use super::handshake::{
    self, HandshakeError, HandshakeResult, HANDSHAKE_TIMEOUT,
};
use super::messages::{
    deserialize_message, serialize_message, HeartbeatMessage, MessagePayload,
    NetworkMessage, NodeCapabilities, PeerId,
};
use super::peer_registry::{PeerRegistry, PeerSnapshot};

/// Maximum message size (16 MB, matching wire.rs).
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Events emitted by the connection manager.
#[derive(Debug, Clone)]
pub enum P2PEvent {
    /// A new peer completed the handshake and is connected.
    PeerConnected {
        peer_id: PeerId,
        role: NodeRole,
        addr: SocketAddr,
    },
    /// A peer disconnected (gracefully or due to error).
    PeerDisconnected {
        peer_id: PeerId,
        reason: String,
    },
    /// An application-level message was received from a peer.
    MessageReceived {
        from: PeerId,
        message: NetworkMessage,
    },
    /// A handshake with an incoming or outgoing peer failed.
    HandshakeFailed {
        addr: SocketAddr,
        reason: String,
    },
    /// A peer's health changed to Degraded or Disconnected.
    PeerHealthChanged {
        peer_id: PeerId,
        healthy: bool,
    },
}

/// Configuration for the connection manager.
#[derive(Debug, Clone)]
pub struct ConnectionManagerConfig {
    /// This node's peer ID.
    pub local_id: PeerId,
    /// This node's role.
    pub role: NodeRole,
    /// This node's capabilities.
    pub capabilities: NodeCapabilities,
    /// Address to listen on for incoming connections (use port 0 for OS-assigned).
    pub listen_addr: SocketAddr,
    /// Optional ed25519 public key bytes for identity verification.
    pub public_key: Option<Vec<u8>>,
    /// Optional x25519 public key bytes for MPC key exchange.
    pub x25519_pubkey: Option<Vec<u8>>,
    /// Optional Ethereum address bytes for on-chain identity.
    pub eth_address: Option<Vec<u8>>,
    /// Interval between heartbeat sends.
    pub heartbeat_interval: Duration,
    /// How long to wait before considering a peer's heartbeat late.
    pub heartbeat_timeout: Duration,
    /// Maximum missed heartbeats before marking peer as disconnected.
    pub max_missed_heartbeats: u32,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
    /// Maximum number of peers (0 = unlimited).
    pub max_peers: usize,
    /// Handshake timeout.
    pub handshake_timeout: Duration,
    /// Reconnection base interval for exponential backoff.
    pub reconnect_base_interval: Duration,
    /// Maximum reconnection interval.
    pub reconnect_max_interval: Duration,
    /// Maximum reconnection attempts before giving up.
    pub reconnect_max_attempts: u32,
}

impl Default for ConnectionManagerConfig {
    fn default() -> Self {
        Self {
            local_id: PeerId::random(),
            role: NodeRole::Compute,
            capabilities: NodeCapabilities::default(),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            public_key: None,
            x25519_pubkey: None,
            eth_address: None,
            heartbeat_interval: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(15),
            max_missed_heartbeats: 3,
            connect_timeout: Duration::from_secs(10),
            max_peers: 256,
            handshake_timeout: HANDSHAKE_TIMEOUT,
            reconnect_base_interval: Duration::from_secs(2),
            reconnect_max_interval: Duration::from_secs(60),
            reconnect_max_attempts: 10,
        }
    }
}

/// Connection manager errors.
#[derive(Debug, thiserror::Error)]
pub enum ConnectionManagerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Handshake failed: {0}")]
    Handshake(#[from] HandshakeError),
    #[error("Connection closed")]
    ConnectionClosed,
    #[error("Peer not found: {0}")]
    PeerNotFound(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Connection timeout")]
    Timeout,
    #[error("Already connected to peer: {0}")]
    AlreadyConnected(String),
    #[error("Peer registry full")]
    RegistryFull,
    #[error("Not running")]
    NotRunning,
}

/// Tracks an active connection's send channel.
struct ActiveConnection {
    send_tx: mpsc::Sender<NetworkMessage>,
    remote_addr: SocketAddr,
}

/// Reconnection state for a disconnected peer.
#[derive(Debug, Clone)]
struct ReconnectEntry {
    addr: SocketAddr,
    role: NodeRole,
    capabilities: NodeCapabilities,
    attempts: u32,
    next_retry: tokio::time::Instant,
}

/// The main P2P connection manager.
pub struct ConnectionManager {
    config: ConnectionManagerConfig,
    registry: Arc<PeerRegistry>,
    connections: Arc<RwLock<HashMap<PeerId, ActiveConnection>>>,
    event_tx: mpsc::Sender<P2PEvent>,
    event_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<P2PEvent>>>,
    inbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
    inbound_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<(PeerId, NetworkMessage)>>>,
    actual_listen_addr: Arc<RwLock<Option<SocketAddr>>>,
    running: Arc<AtomicBool>,
    heartbeat_seq: Arc<AtomicU64>,
    reconnect_queue: Arc<RwLock<HashMap<PeerId, ReconnectEntry>>>,
}

impl ConnectionManager {
    /// Creates a new connection manager.
    pub fn new(config: ConnectionManagerConfig) -> Self {
        let registry = Arc::new(PeerRegistry::new(config.max_peers));
        let (event_tx, event_rx) = mpsc::channel(10_000);
        let (inbound_tx, inbound_rx) = mpsc::channel(10_000);

        Self {
            config,
            registry,
            connections: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx: Arc::new(tokio::sync::Mutex::new(event_rx)),
            inbound_tx,
            inbound_rx: Arc::new(tokio::sync::Mutex::new(inbound_rx)),
            actual_listen_addr: Arc::new(RwLock::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            heartbeat_seq: Arc::new(AtomicU64::new(0)),
            reconnect_queue: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Starts the connection manager (binds listener, starts background tasks).
    pub async fn start(&self) -> Result<(), ConnectionManagerError> {
        let listener = TcpListener::bind(self.config.listen_addr).await?;
        let bound_addr = listener.local_addr()?;
        *self.actual_listen_addr.write() = Some(bound_addr);
        self.running.store(true, Ordering::SeqCst);

        tracing::info!(
            "Connection manager started: peer_id={}, listen={}, role={:?}",
            self.config.local_id, bound_addr, self.config.role,
        );

        // Spawn accept loop
        self.spawn_accept_loop(listener);

        // Spawn inbound message router
        self.spawn_message_router();

        // Spawn heartbeat sender
        self.spawn_heartbeat_sender();

        // Spawn health checker
        self.spawn_health_checker();

        // Spawn reconnection loop
        self.spawn_reconnection_loop();

        Ok(())
    }

    /// Stops the connection manager.
    pub async fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

        // Drop all connections (senders will close, causing read/write loops to exit)
        self.connections.write().clear();

        tracing::info!("Connection manager stopped: peer_id={}", self.config.local_id);
    }

    /// Returns the actual listen address (resolved after start if port 0 was used).
    pub fn listen_addr(&self) -> Option<SocketAddr> {
        *self.actual_listen_addr.read()
    }

    /// Returns the peer registry.
    pub fn registry(&self) -> &Arc<PeerRegistry> {
        &self.registry
    }

    /// Returns our local peer ID.
    pub fn local_id(&self) -> &PeerId {
        &self.config.local_id
    }

    /// Returns the number of active connections.
    pub fn connection_count(&self) -> usize {
        self.connections.read().len()
    }

    /// Returns the next P2P event (peer connected/disconnected, message received, etc.).
    pub async fn next_event(&self) -> Option<P2PEvent> {
        let mut rx = self.event_rx.lock().await;
        rx.recv().await
    }

    /// Returns the next inbound message (from any peer).
    pub async fn next_message(&self) -> Option<(PeerId, NetworkMessage)> {
        let mut rx = self.inbound_rx.lock().await;
        rx.recv().await
    }

    /// Connects to a remote peer at the given address.
    /// Performs TCP connect + handshake and returns the peer's ID on success.
    pub async fn connect_to(&self, addr: SocketAddr) -> Result<PeerId, ConnectionManagerError> {
        if !self.running.load(Ordering::SeqCst) {
            return Err(ConnectionManagerError::NotRunning);
        }

        let listen_addr_str = self.listen_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| self.config.listen_addr.to_string());

        // TCP connect with timeout
        let mut stream = tokio::time::timeout(
            self.config.connect_timeout,
            TcpStream::connect(addr),
        )
        .await
        .map_err(|_| ConnectionManagerError::Timeout)?
        .map_err(ConnectionManagerError::Io)?;

        let _ = stream.set_nodelay(true);

        // Perform handshake
        let result = handshake::perform_handshake_outbound(
            &mut stream,
            &self.config.local_id,
            self.config.role,
            &self.config.capabilities,
            &listen_addr_str,
            self.config.public_key.as_deref(),
            self.config.handshake_timeout,
            self.config.x25519_pubkey.as_deref(),
            self.config.eth_address.as_deref(),
        )
        .await?;

        let peer_id = result.peer_id.clone();

        // Check if already connected
        if self.connections.read().contains_key(&peer_id) {
            return Err(ConnectionManagerError::AlreadyConnected(peer_id.to_string()));
        }

        // Parse listen addr for registry
        let peer_listen_addr: SocketAddr = result.listen_addr.parse().unwrap_or(addr);

        // Register in peer registry with full MPC identity info
        if !self.registry.register_full(
            peer_id.clone(),
            result.role,
            result.capabilities.clone(),
            peer_listen_addr,
            result.public_key.clone(),
            result.x25519_pubkey.clone(),
            result.eth_address.clone(),
        ) {
            return Err(ConnectionManagerError::RegistryFull);
        }

        // Set up message channels and spawn read/write loops
        self.setup_connection(peer_id.clone(), stream, addr);

        // Emit event
        let _ = self.event_tx.send(P2PEvent::PeerConnected {
            peer_id: peer_id.clone(),
            role: result.role,
            addr,
        }).await;

        // Remove from reconnect queue if present
        self.reconnect_queue.write().remove(&peer_id);

        Ok(peer_id)
    }

    /// Sends a message to a specific peer.
    pub async fn send(
        &self,
        peer_id: &PeerId,
        message: NetworkMessage,
    ) -> Result<(), ConnectionManagerError> {
        let send_tx = {
            let conns = self.connections.read();
            let conn = conns.get(peer_id)
                .ok_or_else(|| ConnectionManagerError::PeerNotFound(peer_id.to_string()))?;
            conn.send_tx.clone()
        };

        send_tx.send(message).await
            .map_err(|_| ConnectionManagerError::ConnectionClosed)?;

        self.registry.record_message_sent(peer_id);
        Ok(())
    }

    /// Sends a payload to a specific peer (wraps it in a NetworkMessage).
    pub async fn send_payload(
        &self,
        peer_id: &PeerId,
        payload: MessagePayload,
    ) -> Result<(), ConnectionManagerError> {
        let message = NetworkMessage::new(self.config.local_id.clone(), payload);
        self.send(peer_id, message).await
    }

    /// Broadcasts a message to all connected peers.
    /// Returns a list of (peer_id, result) pairs.
    pub async fn broadcast(
        &self,
        message: NetworkMessage,
    ) -> Vec<(PeerId, Result<(), ConnectionManagerError>)> {
        let peers: Vec<(PeerId, mpsc::Sender<NetworkMessage>)> = {
            self.connections.read()
                .iter()
                .map(|(id, conn)| (id.clone(), conn.send_tx.clone()))
                .collect()
        };

        let mut results = Vec::with_capacity(peers.len());
        for (peer_id, tx) in peers {
            let result = tx.send(message.clone()).await
                .map_err(|_| ConnectionManagerError::ConnectionClosed);
            if result.is_ok() {
                self.registry.record_message_sent(&peer_id);
            }
            results.push((peer_id, result));
        }
        results
    }

    /// Broadcasts a payload to all connected peers.
    pub async fn broadcast_payload(
        &self,
        payload: MessagePayload,
    ) -> Vec<(PeerId, Result<(), ConnectionManagerError>)> {
        let message = NetworkMessage::new(self.config.local_id.clone(), payload);
        self.broadcast(message).await
    }

    /// Disconnects from a specific peer.
    pub async fn disconnect(&self, peer_id: &PeerId) {
        self.connections.write().remove(peer_id);
        self.registry.deregister(peer_id);

        let _ = self.event_tx.send(P2PEvent::PeerDisconnected {
            peer_id: peer_id.clone(),
            reason: "Disconnected by local node".to_string(),
        }).await;
    }

    /// Returns snapshots of all connected peers.
    pub fn connected_peers(&self) -> Vec<PeerSnapshot> {
        self.registry.all_peers()
    }

    /// Returns snapshots of peers with a specific role.
    pub fn peers_by_role(&self, role: NodeRole) -> Vec<PeerSnapshot> {
        self.registry.peers_by_role(role)
    }

    // ── Internal: accept loop ────────────────────────────────────────────

    fn spawn_accept_loop(&self, listener: TcpListener) {
        let running = self.running.clone();
        let config = self.config.clone();
        let registry = self.registry.clone();
        let connections = self.connections.clone();
        let event_tx = self.event_tx.clone();
        let inbound_tx = self.inbound_tx.clone();

        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                let accept_result = tokio::select! {
                    result = listener.accept() => result,
                    _ = tokio::time::sleep(Duration::from_millis(100)) => continue,
                };

                match accept_result {
                    Ok((stream, addr)) => {
                        let _ = stream.set_nodelay(true);

                        let config = config.clone();
                        let registry = registry.clone();
                        let connections = connections.clone();
                        let event_tx = event_tx.clone();
                        let inbound_tx = inbound_tx.clone();
                        let running = running.clone();

                        tokio::spawn(async move {
                            Self::handle_inbound(
                                stream, addr, config, registry, connections,
                                event_tx, inbound_tx, running,
                            ).await;
                        });
                    }
                    Err(e) => {
                        if running.load(Ordering::SeqCst) {
                            tracing::error!("Accept error: {}", e);
                        }
                    }
                }
            }
        });
    }

    async fn handle_inbound(
        mut stream: TcpStream,
        addr: SocketAddr,
        config: ConnectionManagerConfig,
        registry: Arc<PeerRegistry>,
        connections: Arc<RwLock<HashMap<PeerId, ActiveConnection>>>,
        event_tx: mpsc::Sender<P2PEvent>,
        inbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
        running: Arc<AtomicBool>,
    ) {
        let listen_addr_str = config.listen_addr.to_string();
        let current_peers = registry.count();

        // Perform inbound handshake
        let result = handshake::perform_handshake_inbound(
            &mut stream,
            &config.local_id,
            config.role,
            &config.capabilities,
            &listen_addr_str,
            config.public_key.as_deref(),
            config.handshake_timeout,
            config.max_peers,
            current_peers,
            config.x25519_pubkey.as_deref(),
            config.eth_address.as_deref(),
        )
        .await;

        match result {
            Ok(hs_result) => {
                let peer_id = hs_result.peer_id.clone();

                // Check if already connected
                if connections.read().contains_key(&peer_id) {
                    tracing::debug!("Duplicate connection from {}, ignoring", peer_id);
                    return;
                }

                let peer_listen_addr: SocketAddr = hs_result.listen_addr.parse().unwrap_or(addr);

                // Register in peer registry with full MPC identity info
                if !registry.register_full(
                    peer_id.clone(),
                    hs_result.role,
                    hs_result.capabilities.clone(),
                    peer_listen_addr,
                    hs_result.public_key.clone(),
                    hs_result.x25519_pubkey.clone(),
                    hs_result.eth_address.clone(),
                ) {
                    tracing::warn!("Registry full, dropping inbound from {}", peer_id);
                    return;
                }

                // Set up message channels and spawn read/write loops
                let (send_tx, send_rx) = mpsc::channel::<NetworkMessage>(1_000);
                let (reader, writer) = stream.into_split();

                // Store connection
                connections.write().insert(peer_id.clone(), ActiveConnection {
                    send_tx,
                    remote_addr: addr,
                });

                // Spawn read loop
                {
                    let peer_id = peer_id.clone();
                    let inbound_tx = inbound_tx.clone();
                    let connections = connections.clone();
                    let registry = registry.clone();
                    let event_tx = event_tx.clone();
                    let running = running.clone();

                    tokio::spawn(async move {
                        message_read_loop(reader, &peer_id, &inbound_tx, &registry).await;
                        // Connection closed
                        if running.load(Ordering::SeqCst) {
                            connections.write().remove(&peer_id);
                            registry.deregister(&peer_id);
                            let _ = event_tx.send(P2PEvent::PeerDisconnected {
                                peer_id: peer_id.clone(),
                                reason: "Connection closed by remote".to_string(),
                            }).await;
                        }
                    });
                }

                // Spawn write loop
                tokio::spawn(async move {
                    message_write_loop(writer, send_rx).await;
                });

                let _ = event_tx.send(P2PEvent::PeerConnected {
                    peer_id,
                    role: hs_result.role,
                    addr,
                }).await;
            }
            Err(e) => {
                tracing::debug!("Handshake failed from {}: {}", addr, e);
                let _ = event_tx.send(P2PEvent::HandshakeFailed {
                    addr,
                    reason: e.to_string(),
                }).await;
            }
        }
    }

    // ── Internal: setup connection for outbound ──────────────────────────

    fn setup_connection(&self, peer_id: PeerId, stream: TcpStream, addr: SocketAddr) {
        let (send_tx, send_rx) = mpsc::channel::<NetworkMessage>(1_000);
        let (reader, writer) = stream.into_split();

        // Store connection
        self.connections.write().insert(peer_id.clone(), ActiveConnection {
            send_tx,
            remote_addr: addr,
        });

        // Spawn read loop
        {
            let peer_id = peer_id.clone();
            let inbound_tx = self.inbound_tx.clone();
            let connections = self.connections.clone();
            let registry = self.registry.clone();
            let event_tx = self.event_tx.clone();
            let running = self.running.clone();
            let reconnect_queue = self.reconnect_queue.clone();
            let reconnect_base = self.config.reconnect_base_interval;

            tokio::spawn(async move {
                message_read_loop(reader, &peer_id, &inbound_tx, &registry).await;
                // Connection closed
                if running.load(Ordering::SeqCst) {
                    // Get peer info before removing
                    let peer_snap = registry.get(&peer_id);

                    connections.write().remove(&peer_id);
                    registry.deregister(&peer_id);

                    // Schedule reconnection if we knew this peer
                    if let Some(snap) = peer_snap {
                        reconnect_queue.write().insert(peer_id.clone(), ReconnectEntry {
                            addr: snap.listen_addr,
                            role: snap.role,
                            capabilities: snap.capabilities,
                            attempts: 0,
                            next_retry: tokio::time::Instant::now() + reconnect_base,
                        });
                    }

                    let _ = event_tx.send(P2PEvent::PeerDisconnected {
                        peer_id: peer_id.clone(),
                        reason: "Connection closed by remote".to_string(),
                    }).await;
                }
            });
        }

        // Spawn write loop
        tokio::spawn(async move {
            message_write_loop(writer, send_rx).await;
        });
    }

    // ── Internal: message router ─────────────────────────────────────────

    fn spawn_message_router(&self) {
        // The message router takes inbound messages from read loops and
        // emits them as P2PEvents. This allows consumers to receive messages
        // via next_event() or next_message() depending on preference.
        let running = self.running.clone();
        let inbound_rx = self.inbound_rx.clone();
        let event_tx = self.event_tx.clone();
        let registry = self.registry.clone();

        tokio::spawn(async move {
            let mut rx = inbound_rx.lock().await;
            while running.load(Ordering::SeqCst) {
                match rx.recv().await {
                    Some((from, message)) => {
                        // Update heartbeat on any message (liveness signal)
                        registry.record_heartbeat(&from, 0);

                        // Handle heartbeat messages specially
                        if let MessagePayload::Heartbeat(ref hb) = message.payload {
                            registry.record_heartbeat(&from, hb.seq);

                            // If it's a ping, send pong back
                            if !hb.is_pong {
                                // Pong is handled by the caller if needed
                            }
                        }

                        let _ = event_tx.send(P2PEvent::MessageReceived {
                            from,
                            message,
                        }).await;
                    }
                    None => break,
                }
            }
        });
    }

    // ── Internal: heartbeat sender ───────────────────────────────────────

    fn spawn_heartbeat_sender(&self) {
        let running = self.running.clone();
        let connections = self.connections.clone();
        let local_id = self.config.local_id.clone();
        let heartbeat_seq = self.heartbeat_seq.clone();
        let interval = self.config.heartbeat_interval;

        tokio::spawn(async move {
            // Wait for the first interval before sending (don't send immediately on start)
            loop {
                tokio::time::sleep(interval).await;

                if !running.load(Ordering::SeqCst) {
                    break;
                }

                let seq = heartbeat_seq.fetch_add(1, Ordering::Relaxed);
                let hb_msg = NetworkMessage::new(
                    local_id.clone(),
                    MessagePayload::Heartbeat(HeartbeatMessage {
                        seq,
                        is_pong: false,
                        load: 0, // Could be filled with actual load metrics
                    }),
                );

                let peers: Vec<(PeerId, mpsc::Sender<NetworkMessage>)> = {
                    connections.read()
                        .iter()
                        .map(|(id, conn)| (id.clone(), conn.send_tx.clone()))
                        .collect()
                };

                for (_peer_id, tx) in peers {
                    let _ = tx.send(hb_msg.clone()).await;
                }
            }
        });
    }

    // ── Internal: health checker ─────────────────────────────────────────

    fn spawn_health_checker(&self) {
        let running = self.running.clone();
        let registry = self.registry.clone();
        let connections = self.connections.clone();
        let event_tx = self.event_tx.clone();
        let heartbeat_timeout = self.config.heartbeat_timeout;
        let max_missed = self.config.max_missed_heartbeats;
        let reconnect_queue = self.reconnect_queue.clone();
        let reconnect_base = self.config.reconnect_base_interval;

        tokio::spawn(async move {
            // Wait for the first interval before checking (give peers time to connect)
            loop {
                tokio::time::sleep(heartbeat_timeout).await;

                if !running.load(Ordering::SeqCst) {
                    break;
                }

                // Check health of all peers
                let newly_disconnected = registry.check_health(heartbeat_timeout, max_missed);

                for peer_id in newly_disconnected {
                    tracing::warn!("Peer {} marked as disconnected (missed heartbeats)", peer_id);

                    // Get peer info before removing
                    let peer_snap = registry.get(&peer_id);

                    // Remove connection and registry entry
                    connections.write().remove(&peer_id);
                    registry.deregister(&peer_id);

                    // Schedule reconnection
                    if let Some(snap) = peer_snap {
                        reconnect_queue.write().insert(peer_id.clone(), ReconnectEntry {
                            addr: snap.listen_addr,
                            role: snap.role,
                            capabilities: snap.capabilities,
                            attempts: 0,
                            next_retry: tokio::time::Instant::now() + reconnect_base,
                        });
                    }

                    let _ = event_tx.send(P2PEvent::PeerDisconnected {
                        peer_id: peer_id.clone(),
                        reason: "Heartbeat timeout".to_string(),
                    }).await;

                    let _ = event_tx.send(P2PEvent::PeerHealthChanged {
                        peer_id,
                        healthy: false,
                    }).await;
                }
            }
        });
    }

    // ── Internal: reconnection loop ──────────────────────────────────────

    fn spawn_reconnection_loop(&self) {
        let running = self.running.clone();
        let reconnect_queue = self.reconnect_queue.clone();
        let config = self.config.clone();
        let registry = self.registry.clone();
        let connections = self.connections.clone();
        let event_tx = self.event_tx.clone();
        let inbound_tx = self.inbound_tx.clone();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(1));

            while running.load(Ordering::SeqCst) {
                ticker.tick().await;

                if !running.load(Ordering::SeqCst) {
                    break;
                }

                let now = tokio::time::Instant::now();

                // Collect entries ready for retry
                let ready: Vec<(PeerId, ReconnectEntry)> = {
                    reconnect_queue.read()
                        .iter()
                        .filter(|(_, entry)| now >= entry.next_retry)
                        .map(|(id, entry)| (id.clone(), entry.clone()))
                        .collect()
                };

                for (peer_id, entry) in ready {
                    if entry.attempts >= config.reconnect_max_attempts {
                        tracing::warn!(
                            "Giving up reconnection to {} after {} attempts",
                            peer_id, entry.attempts,
                        );
                        reconnect_queue.write().remove(&peer_id);
                        continue;
                    }

                    tracing::debug!(
                        "Attempting reconnection to {} (attempt {}/{})",
                        peer_id, entry.attempts + 1, config.reconnect_max_attempts,
                    );

                    let listen_addr_str = config.listen_addr.to_string();

                    // Try to connect and handshake
                    match tokio::time::timeout(
                        config.connect_timeout,
                        TcpStream::connect(entry.addr),
                    ).await {
                        Ok(Ok(mut stream)) => {
                            let _ = stream.set_nodelay(true);

                            match handshake::perform_handshake_outbound(
                                &mut stream,
                                &config.local_id,
                                config.role,
                                &config.capabilities,
                                &listen_addr_str,
                                config.public_key.as_deref(),
                                config.handshake_timeout,
                                config.x25519_pubkey.as_deref(),
                                config.eth_address.as_deref(),
                            ).await {
                                Ok(result) => {
                                    let actual_peer_id = result.peer_id.clone();
                                    let peer_listen_addr: SocketAddr =
                                        result.listen_addr.parse().unwrap_or(entry.addr);

                                    // Register in peer registry with full MPC identity info
                                    registry.register_full(
                                        actual_peer_id.clone(),
                                        result.role,
                                        result.capabilities.clone(),
                                        peer_listen_addr,
                                        result.public_key.clone(),
                                        result.x25519_pubkey.clone(),
                                        result.eth_address.clone(),
                                    );

                                    // Set up connection
                                    let (send_tx, send_rx) = mpsc::channel::<NetworkMessage>(1_000);
                                    let (reader, writer) = stream.into_split();
                                    let addr = entry.addr;

                                    connections.write().insert(actual_peer_id.clone(), ActiveConnection {
                                        send_tx,
                                        remote_addr: addr,
                                    });

                                    // Spawn read loop
                                    {
                                        let pid = actual_peer_id.clone();
                                        let inbound_tx = inbound_tx.clone();
                                        let connections = connections.clone();
                                        let registry = registry.clone();
                                        let event_tx = event_tx.clone();
                                        let running = running.clone();
                                        let reconnect_queue = reconnect_queue.clone();
                                        let reconnect_base = config.reconnect_base_interval;

                                        tokio::spawn(async move {
                                            message_read_loop(reader, &pid, &inbound_tx, &registry).await;
                                            if running.load(Ordering::SeqCst) {
                                                let snap = registry.get(&pid);
                                                connections.write().remove(&pid);
                                                registry.deregister(&pid);
                                                if let Some(s) = snap {
                                                    reconnect_queue.write().insert(pid.clone(), ReconnectEntry {
                                                        addr: s.listen_addr,
                                                        role: s.role,
                                                        capabilities: s.capabilities,
                                                        attempts: 0,
                                                        next_retry: tokio::time::Instant::now() + reconnect_base,
                                                    });
                                                }
                                                let _ = event_tx.send(P2PEvent::PeerDisconnected {
                                                    peer_id: pid,
                                                    reason: "Connection closed by remote".to_string(),
                                                }).await;
                                            }
                                        });
                                    }

                                    // Spawn write loop
                                    tokio::spawn(async move {
                                        message_write_loop(writer, send_rx).await;
                                    });

                                    reconnect_queue.write().remove(&peer_id);

                                    let _ = event_tx.send(P2PEvent::PeerConnected {
                                        peer_id: actual_peer_id,
                                        role: result.role,
                                        addr: entry.addr,
                                    }).await;

                                    tracing::info!("Reconnected to {} at {}", peer_id, entry.addr);
                                }
                                Err(e) => {
                                    Self::bump_reconnect_entry(
                                        &reconnect_queue, &peer_id, &entry, &config,
                                    );
                                    tracing::debug!("Reconnect handshake failed for {}: {}", peer_id, e);
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            Self::bump_reconnect_entry(
                                &reconnect_queue, &peer_id, &entry, &config,
                            );
                            tracing::debug!("Reconnect TCP failed for {}: {}", peer_id, e);
                        }
                        Err(_) => {
                            Self::bump_reconnect_entry(
                                &reconnect_queue, &peer_id, &entry, &config,
                            );
                            tracing::debug!("Reconnect timed out for {}", peer_id);
                        }
                    }
                }
            }
        });
    }

    fn bump_reconnect_entry(
        queue: &Arc<RwLock<HashMap<PeerId, ReconnectEntry>>>,
        peer_id: &PeerId,
        entry: &ReconnectEntry,
        config: &ConnectionManagerConfig,
    ) {
        let next_attempts = entry.attempts + 1;
        let delay = config.reconnect_base_interval * 2u32.pow(next_attempts);
        let capped_delay = delay.min(config.reconnect_max_interval);

        queue.write().insert(peer_id.clone(), ReconnectEntry {
            addr: entry.addr,
            role: entry.role,
            capabilities: entry.capabilities.clone(),
            attempts: next_attempts,
            next_retry: tokio::time::Instant::now() + capped_delay,
        });
    }
}

// ── Free functions for read/write loops ──────────────────────────────────

/// Read loop: reads length-prefixed bincode NetworkMessages from a reader.
async fn message_read_loop<R: AsyncRead + Unpin>(
    mut reader: R,
    peer_id: &PeerId,
    inbound_tx: &mpsc::Sender<(PeerId, NetworkMessage)>,
    registry: &Arc<PeerRegistry>,
) {
    let mut len_buf = [0u8; 4];

    loop {
        // Read 4-byte length prefix
        match reader.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(_) => break, // Connection closed or error
        }

        let msg_len = u32::from_be_bytes(len_buf) as usize;
        if msg_len > MAX_MESSAGE_SIZE {
            tracing::error!("Message too large from {}: {} bytes", peer_id, msg_len);
            break;
        }

        // Read message body
        let mut body = vec![0u8; msg_len];
        match reader.read_exact(&mut body).await {
            Ok(_) => {}
            Err(_) => break,
        }

        // Deserialize
        match deserialize_message(&body) {
            Ok(message) => {
                registry.record_message_received(peer_id);
                if inbound_tx.send((peer_id.clone(), message)).await.is_err() {
                    break; // Receiver dropped
                }
            }
            Err(e) => {
                tracing::error!("Deserialization error from {}: {}", peer_id, e);
            }
        }
    }
}

/// Write loop: reads NetworkMessages from a channel and writes them length-prefixed.
async fn message_write_loop<W: AsyncWrite + Unpin>(
    mut writer: W,
    mut rx: mpsc::Receiver<NetworkMessage>,
) {
    while let Some(message) = rx.recv().await {
        match serialize_message(&message) {
            Ok(bytes) => {
                let len = bytes.len() as u32;
                if writer.write_all(&len.to_be_bytes()).await.is_err() {
                    break;
                }
                if writer.write_all(&bytes).await.is_err() {
                    break;
                }
                if writer.flush().await.is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::error!("Serialization error: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ConnectionManagerConfig::default();
        assert_eq!(config.heartbeat_interval, Duration::from_secs(5));
        assert_eq!(config.heartbeat_timeout, Duration::from_secs(15));
        assert_eq!(config.max_missed_heartbeats, 3);
        assert_eq!(config.max_peers, 256);
    }

    #[tokio::test]
    async fn test_connection_manager_start_stop() {
        let config = ConnectionManagerConfig {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            ..Default::default()
        };
        let mgr = ConnectionManager::new(config);
        mgr.start().await.unwrap();

        assert!(mgr.listen_addr().is_some());
        assert!(mgr.running.load(Ordering::SeqCst));
        assert_eq!(mgr.connection_count(), 0);

        mgr.stop().await;
        assert!(!mgr.running.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_two_node_connection() {
        // Start node A (aggregator)
        let config_a = ConnectionManagerConfig {
            local_id: PeerId::from_string("node-a"),
            role: NodeRole::Aggregator,
            capabilities: NodeCapabilities {
                can_aggregate: true,
                ..Default::default()
            },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(1),
            heartbeat_timeout: Duration::from_secs(5),
            ..Default::default()
        };
        let node_a = ConnectionManager::new(config_a);
        node_a.start().await.unwrap();
        let addr_a = node_a.listen_addr().unwrap();

        // Start node B (worker)
        let config_b = ConnectionManagerConfig {
            local_id: PeerId::from_string("node-b"),
            role: NodeRole::Compute,
            capabilities: NodeCapabilities {
                can_train: true,
                ..Default::default()
            },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(1),
            heartbeat_timeout: Duration::from_secs(5),
            ..Default::default()
        };
        let node_b = ConnectionManager::new(config_b);
        node_b.start().await.unwrap();

        // Node B connects to Node A
        let peer_id = node_b.connect_to(addr_a).await.unwrap();
        assert_eq!(peer_id, PeerId::from_string("node-a"));

        // Wait for Node A to process the inbound handshake
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Both should see each other
        assert_eq!(node_b.connection_count(), 1);
        assert_eq!(node_a.connection_count(), 1);

        // Node B should see Node A as aggregator
        let a_peers = node_b.registry().peers_by_role(NodeRole::Aggregator);
        assert_eq!(a_peers.len(), 1);
        assert_eq!(a_peers[0].peer_id, PeerId::from_string("node-a"));

        // Node A should see Node B as compute
        let b_peers = node_a.registry().peers_by_role(NodeRole::Compute);
        assert_eq!(b_peers.len(), 1);
        assert_eq!(b_peers[0].peer_id, PeerId::from_string("node-b"));

        // Clean up
        node_a.stop().await;
        node_b.stop().await;
    }

    #[tokio::test]
    async fn test_message_exchange() {
        // Start two nodes
        let config_a = ConnectionManagerConfig {
            local_id: PeerId::from_string("sender"),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60), // Long interval to avoid noise
            ..Default::default()
        };
        let node_a = ConnectionManager::new(config_a);
        node_a.start().await.unwrap();
        let addr_a = node_a.listen_addr().unwrap();

        let config_b = ConnectionManagerConfig {
            local_id: PeerId::from_string("receiver"),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let node_b = ConnectionManager::new(config_b);
        node_b.start().await.unwrap();

        // Connect B -> A
        node_b.connect_to(addr_a).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Send a message from A to B with a unique seq
        let msg = NetworkMessage::new(
            PeerId::from_string("sender"),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 42,
                is_pong: false,
                load: 50,
            }),
        );

        node_a.send(&PeerId::from_string("receiver"), msg).await.unwrap();

        // B should receive the message (skip non-matching events)
        let mut found = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(200), node_b.next_event()).await {
                Ok(Some(P2PEvent::MessageReceived { from, message })) => {
                    if from == PeerId::from_string("sender") {
                        if let MessagePayload::Heartbeat(hb) = message.payload {
                            if hb.seq == 42 {
                                assert_eq!(hb.load, 50);
                                assert!(!hb.is_pong);
                                found = true;
                                break;
                            }
                        }
                    }
                }
                Ok(Some(_)) => continue, // Skip other events
                _ => continue,
            }
        }
        assert!(found, "Node B should have received the test heartbeat with seq=42");

        node_a.stop().await;
        node_b.stop().await;
    }

    #[tokio::test]
    async fn test_three_node_discovery() {
        // Start aggregator
        let config_agg = ConnectionManagerConfig {
            local_id: PeerId::from_string("aggregator"),
            role: NodeRole::Aggregator,
            capabilities: NodeCapabilities { can_aggregate: true, ..Default::default() },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let agg = ConnectionManager::new(config_agg);
        agg.start().await.unwrap();
        let agg_addr = agg.listen_addr().unwrap();

        // Start worker 1
        let config_w1 = ConnectionManagerConfig {
            local_id: PeerId::from_string("worker-1"),
            role: NodeRole::Compute,
            capabilities: NodeCapabilities { can_train: true, ..Default::default() },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let worker1 = ConnectionManager::new(config_w1);
        worker1.start().await.unwrap();

        // Start worker 2
        let config_w2 = ConnectionManagerConfig {
            local_id: PeerId::from_string("worker-2"),
            role: NodeRole::Compute,
            capabilities: NodeCapabilities { can_train: true, ..Default::default() },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let worker2 = ConnectionManager::new(config_w2);
        worker2.start().await.unwrap();

        // Both workers connect to aggregator
        worker1.connect_to(agg_addr).await.unwrap();
        worker2.connect_to(agg_addr).await.unwrap();

        // Wait for connections to establish
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Aggregator should see both workers
        assert_eq!(agg.connection_count(), 2);
        let workers = agg.registry().peers_by_role(NodeRole::Compute);
        assert_eq!(workers.len(), 2);

        // Each worker should see the aggregator
        assert_eq!(worker1.connection_count(), 1);
        assert_eq!(worker2.connection_count(), 1);

        let agg_peers_from_w1 = worker1.registry().peers_by_role(NodeRole::Aggregator);
        assert_eq!(agg_peers_from_w1.len(), 1);
        assert_eq!(agg_peers_from_w1[0].peer_id, PeerId::from_string("aggregator"));

        // Aggregator broadcasts a message to all workers
        let broadcast_msg = NetworkMessage::new(
            PeerId::from_string("aggregator"),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 99,
                is_pong: false,
                load: 10,
            }),
        );
        let results = agg.broadcast(broadcast_msg).await;
        assert_eq!(results.len(), 2);
        for (_, result) in &results {
            assert!(result.is_ok());
        }

        // Cleanup
        agg.stop().await;
        worker1.stop().await;
        worker2.stop().await;
    }

    #[tokio::test]
    async fn test_disconnect() {
        let config_a = ConnectionManagerConfig {
            local_id: PeerId::from_string("node-a"),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let node_a = ConnectionManager::new(config_a);
        node_a.start().await.unwrap();
        let addr_a = node_a.listen_addr().unwrap();

        let config_b = ConnectionManagerConfig {
            local_id: PeerId::from_string("node-b"),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let node_b = ConnectionManager::new(config_b);
        node_b.start().await.unwrap();

        node_b.connect_to(addr_a).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert_eq!(node_b.connection_count(), 1);
        assert_eq!(node_a.connection_count(), 1);

        // Disconnect from B's side
        node_b.disconnect(&PeerId::from_string("node-a")).await;
        assert_eq!(node_b.connection_count(), 0);
        assert!(!node_b.registry().contains(&PeerId::from_string("node-a")));

        // A should detect the disconnect shortly
        tokio::time::sleep(Duration::from_millis(200)).await;

        node_a.stop().await;
        node_b.stop().await;
    }

    #[tokio::test]
    async fn test_connect_to_nonexistent() {
        let config = ConnectionManagerConfig {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            connect_timeout: Duration::from_millis(500),
            ..Default::default()
        };
        let mgr = ConnectionManager::new(config);
        mgr.start().await.unwrap();

        // Try to connect to a port that nothing is listening on
        let result = mgr.connect_to("127.0.0.1:59999".parse().unwrap()).await;
        assert!(result.is_err());

        mgr.stop().await;
    }

    #[tokio::test]
    async fn test_send_to_unknown_peer() {
        let config = ConnectionManagerConfig {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            ..Default::default()
        };
        let mgr = ConnectionManager::new(config);
        mgr.start().await.unwrap();

        let msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 0,
                is_pong: false,
                load: 0,
            }),
        );

        let result = mgr.send(&PeerId::from_string("nonexistent"), msg).await;
        assert!(matches!(result, Err(ConnectionManagerError::PeerNotFound(_))));

        mgr.stop().await;
    }
}
