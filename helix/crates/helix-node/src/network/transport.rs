//! TCP/TLS Network Transport Layer.
//!
//! Provides the underlying network transport for HELIX P2P communication.
//! Supports both plaintext TCP (for local demos) and TLS (for production).

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::{Buf, BufMut, BytesMut};
use futures::stream::StreamExt;
use parking_lot::RwLock;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_rustls::{TlsAcceptor, TlsConnector, client::TlsStream as ClientTlsStream, server::TlsStream as ServerTlsStream};

use super::messages::{deserialize_message, serialize_message, NetworkMessage, PeerId};

/// Maximum message size (16 MB).
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Connection read timeout.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Connection write timeout.
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// Transport configuration.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Listen address.
    pub listen_addr: SocketAddr,
    /// Enable TLS.
    pub use_tls: bool,
    /// TLS certificate path (PEM).
    pub cert_path: Option<String>,
    /// TLS private key path (PEM).
    pub key_path: Option<String>,
    /// Maximum connections per peer.
    pub max_connections_per_peer: usize,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Enable TCP nodelay.
    pub tcp_nodelay: bool,
    /// Keepalive interval.
    pub keepalive_secs: Option<u64>,
    /// Maximum total peers in the connection pool (0 = unlimited).
    pub max_pool_size: usize,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:9000".parse().unwrap(),
            use_tls: false,
            cert_path: None,
            key_path: None,
            max_connections_per_peer: 2,
            connect_timeout: Duration::from_secs(10),
            tcp_nodelay: true,
            keepalive_secs: Some(30),
            max_pool_size: 256,
        }
    }
}

/// Transport error types.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("TLS error: {0}")]
    Tls(String),
    #[error("Connection closed")]
    ConnectionClosed,
    #[error("Message too large: {size} > {max}")]
    MessageTooLarge { size: usize, max: usize },
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Connection timeout")]
    Timeout,
    #[error("Peer not found: {0}")]
    PeerNotFound(String),
    #[error("Connection pool exhausted")]
    PoolExhausted,
    #[error("Invalid certificate")]
    InvalidCertificate,
}

/// Abstract transport trait for sending/receiving messages.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Sends a message to a peer.
    async fn send(&self, peer: &PeerId, addr: &SocketAddr, message: NetworkMessage) -> Result<(), TransportError>;

    /// Receives a message (returns sender and message).
    async fn recv(&self) -> Result<(PeerId, NetworkMessage), TransportError>;

    /// Starts the transport listener.
    async fn start(&self) -> Result<(), TransportError>;

    /// Stops the transport.
    async fn stop(&self) -> Result<(), TransportError>;

    /// Returns the local address.
    fn local_addr(&self) -> SocketAddr;
}

/// Connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is being established.
    Connecting,
    /// Connection is active.
    Connected,
    /// Connection is being gracefully closed.
    Closing,
    /// Connection is closed.
    Closed,
}

/// A peer connection.
pub struct PeerConnection {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Remote address.
    pub remote_addr: SocketAddr,
    /// Connection state.
    pub state: ConnectionState,
    /// Messages sent.
    pub messages_sent: u64,
    /// Messages received.
    pub messages_received: u64,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Bytes received.
    pub bytes_received: u64,
    /// Last activity timestamp.
    pub last_activity: u64,
    /// Send channel.
    sender: mpsc::Sender<NetworkMessage>,
}

impl PeerConnection {
    /// Sends a message on this connection.
    pub async fn send(&self, message: NetworkMessage) -> Result<(), TransportError> {
        self.sender.send(message).await
            .map_err(|_| TransportError::ConnectionClosed)
    }
}

/// TCP/TLS transport implementation.
pub struct TcpTransport {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: TransportConfig,
    /// Active connections by peer ID.
    connections: Arc<RwLock<HashMap<PeerId, Vec<Arc<PeerConnection>>>>>,
    /// Inbound message channel.
    inbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
    /// Actual bound address (updated after start_listener when port 0 is used).
    actual_addr: Arc<RwLock<Option<SocketAddr>>>,
    /// Inbound message receiver.
    inbound_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<(PeerId, NetworkMessage)>>>,
    /// Running state.
    running: Arc<std::sync::atomic::AtomicBool>,
    /// TLS connector (for outbound connections).
    tls_connector: Option<TlsConnector>,
    /// TLS acceptor (for inbound connections).
    tls_acceptor: Option<TlsAcceptor>,
}

impl TcpTransport {
    /// Creates a new TCP transport.
    pub fn new(local_id: PeerId, config: TransportConfig) -> Result<Self, TransportError> {
        let (inbound_tx, inbound_rx) = mpsc::channel(10000);

        let (tls_connector, tls_acceptor) = if config.use_tls {
            let (connector, acceptor) = Self::setup_tls(&config)?;
            (Some(connector), Some(acceptor))
        } else {
            (None, None)
        };

        Ok(Self {
            local_id,
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            inbound_tx,
            actual_addr: Arc::new(RwLock::new(None)),
            inbound_rx: Arc::new(tokio::sync::Mutex::new(inbound_rx)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            tls_connector,
            tls_acceptor,
        })
    }

    /// Creates a new TCP transport with a self-signed certificate for demos.
    pub fn new_with_self_signed(local_id: PeerId, mut config: TransportConfig) -> Result<Self, TransportError> {
        let (inbound_tx, inbound_rx) = mpsc::channel(10000);

        // Generate self-signed certificate
        let (connector, acceptor) = if config.use_tls {
            config.use_tls = true;
            Self::setup_self_signed_tls()?
        } else {
            return Self::new(local_id, config);
        };

        Ok(Self {
            local_id,
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            inbound_tx,
            actual_addr: Arc::new(RwLock::new(None)),
            inbound_rx: Arc::new(tokio::sync::Mutex::new(inbound_rx)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            tls_connector: Some(connector),
            tls_acceptor: Some(acceptor),
        })
    }

    fn setup_tls(config: &TransportConfig) -> Result<(TlsConnector, TlsAcceptor), TransportError> {
        use std::fs::File;
        use std::io::BufReader;

        // Load certificates
        let cert_path = config.cert_path.as_ref()
            .ok_or_else(|| TransportError::Tls("Certificate path required".into()))?;
        let key_path = config.key_path.as_ref()
            .ok_or_else(|| TransportError::Tls("Key path required".into()))?;

        let cert_file = File::open(cert_path)
            .map_err(|e| TransportError::Tls(format!("Failed to open cert: {}", e)))?;
        let mut cert_reader = BufReader::new(cert_file);
        let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut cert_reader)
            .filter_map(|r| r.ok())
            .collect();

        let key_file = File::open(key_path)
            .map_err(|e| TransportError::Tls(format!("Failed to open key: {}", e)))?;
        let mut key_reader = BufReader::new(key_file);
        let key = rustls_pemfile::private_key(&mut key_reader)
            .map_err(|e| TransportError::Tls(format!("Failed to read key: {}", e)))?
            .ok_or_else(|| TransportError::Tls("No private key found".into()))?;

        // Create server config
        let server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs.clone(), key.clone_key())
            .map_err(|e| TransportError::Tls(format!("Server config error: {}", e)))?;

        let acceptor = TlsAcceptor::from(Arc::new(server_config));

        // Create client config with root certs
        let mut root_store = rustls::RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        // Also add our own cert for peer-to-peer
        for cert in &certs {
            let _ = root_store.add(cert.clone());
        }

        let client_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(certs, key)
            .map_err(|e| TransportError::Tls(format!("Client config error: {}", e)))?;

        let connector = TlsConnector::from(Arc::new(client_config));

        Ok((connector, acceptor))
    }

    fn setup_self_signed_tls() -> Result<(TlsConnector, TlsAcceptor), TransportError> {
        use rcgen::{CertifiedKey, generate_simple_self_signed};

        // Generate self-signed certificate
        let subject_alt_names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
        let CertifiedKey { cert, key_pair } = generate_simple_self_signed(subject_alt_names)
            .map_err(|e| TransportError::Tls(format!("Failed to generate cert: {}", e)))?;

        let cert_der = CertificateDer::from(cert.der().to_vec());
        let key_der = PrivateKeyDer::try_from(key_pair.serialize_der())
            .map_err(|e| TransportError::Tls(format!("Failed to serialize key: {}", e)))?;

        // Create server config
        let server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der.clone()], key_der.clone_key())
            .map_err(|e| TransportError::Tls(format!("Server config error: {}", e)))?;

        let acceptor = TlsAcceptor::from(Arc::new(server_config));

        // Create client config (accepts self-signed)
        let mut root_store = rustls::RootCertStore::empty();
        let _ = root_store.add(cert_der.clone());

        let client_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let connector = TlsConnector::from(Arc::new(client_config));

        Ok((connector, acceptor))
    }

    /// Connects to a peer.
    pub async fn connect(&self, peer_id: PeerId, addr: SocketAddr) -> Result<Arc<PeerConnection>, TransportError> {
        // Check if we already have a connection
        {
            let conns = self.connections.read();
            if let Some(peer_conns) = conns.get(&peer_id) {
                if let Some(conn) = peer_conns.first() {
                    if conn.state == ConnectionState::Connected {
                        return Ok(conn.clone());
                    }
                }
            }
        }

        // Establish new connection
        let stream = tokio::time::timeout(
            self.config.connect_timeout,
            TcpStream::connect(addr)
        ).await
            .map_err(|_| TransportError::Timeout)?
            .map_err(TransportError::Io)?;

        if self.config.tcp_nodelay {
            stream.set_nodelay(true)?;
        }

        // Create channels for this connection
        let (send_tx, mut send_rx) = mpsc::channel::<NetworkMessage>(1000);
        let inbound_tx = self.inbound_tx.clone();
        let remote_peer_id = peer_id.clone();

        // Handle TLS or plain TCP
        if let Some(ref connector) = self.tls_connector {
            let server_name = ServerName::try_from("localhost")
                .map_err(|_| TransportError::Tls("Invalid server name".into()))?;
            let tls_stream = connector.connect(server_name, stream).await
                .map_err(|e| TransportError::Tls(e.to_string()))?;

            let (read_half, write_half) = tokio::io::split(tls_stream);

            // Spawn read task
            let peer_id_clone = remote_peer_id.clone();
            tokio::spawn(async move {
                Self::read_loop_tls(read_half, peer_id_clone, inbound_tx).await;
            });

            // Spawn write task
            tokio::spawn(async move {
                Self::write_loop_tls(write_half, &mut send_rx).await;
            });
        } else {
            let (read_half, write_half) = stream.into_split();

            // Spawn read task
            let peer_id_clone = remote_peer_id.clone();
            tokio::spawn(async move {
                Self::read_loop(read_half, peer_id_clone, inbound_tx).await;
            });

            // Spawn write task
            tokio::spawn(async move {
                Self::write_loop(write_half, &mut send_rx).await;
            });
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let connection = Arc::new(PeerConnection {
            peer_id: peer_id.clone(),
            remote_addr: addr,
            state: ConnectionState::Connected,
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            last_activity: now,
            sender: send_tx,
        });

        // Store connection
        {
            let mut conns = self.connections.write();
            conns.entry(peer_id).or_default().push(connection.clone());
        }

        Ok(connection)
    }

    async fn read_loop(
        mut reader: tokio::net::tcp::OwnedReadHalf,
        peer_id: PeerId,
        inbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
    ) {
        let mut buf = BytesMut::with_capacity(8192);

        loop {
            // Read length prefix (4 bytes)
            if buf.len() < 4 {
                buf.reserve(4 - buf.len());
                match reader.read_buf(&mut buf).await {
                    Ok(0) => break, // Connection closed
                    Ok(_) => {}
                    Err(_) => break,
                }
                continue;
            }

            let msg_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;

            if msg_len > MAX_MESSAGE_SIZE {
                tracing::error!("Message too large from {:?}: {}", peer_id, msg_len);
                break;
            }

            // Read message body
            while buf.len() < 4 + msg_len {
                buf.reserve(4 + msg_len - buf.len());
                match reader.read_buf(&mut buf).await {
                    Ok(0) => return, // Connection closed
                    Ok(_) => {}
                    Err(_) => return,
                }
            }

            // Parse message
            buf.advance(4); // Skip length prefix
            let msg_bytes: Vec<u8> = buf.split_to(msg_len).to_vec();

            match deserialize_message(&msg_bytes) {
                Ok(message) => {
                    if inbound_tx.send((peer_id.clone(), message)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to deserialize message: {}", e);
                }
            }
        }
    }

    async fn read_loop_tls<S: tokio::io::AsyncRead + Unpin>(
        mut reader: ReadHalf<S>,
        peer_id: PeerId,
        inbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
    ) {
        let mut buf = BytesMut::with_capacity(8192);

        loop {
            // Read length prefix (4 bytes)
            while buf.len() < 4 {
                buf.reserve(4 - buf.len());
                match reader.read_buf(&mut buf).await {
                    Ok(0) => return, // Connection closed
                    Ok(_) => {}
                    Err(_) => return,
                }
            }

            let msg_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;

            if msg_len > MAX_MESSAGE_SIZE {
                tracing::error!("Message too large from {:?}: {}", peer_id, msg_len);
                return;
            }

            // Read message body
            while buf.len() < 4 + msg_len {
                buf.reserve(4 + msg_len - buf.len());
                match reader.read_buf(&mut buf).await {
                    Ok(0) => return,
                    Ok(_) => {}
                    Err(_) => return,
                }
            }

            // Parse message
            buf.advance(4);
            let msg_bytes: Vec<u8> = buf.split_to(msg_len).to_vec();

            match deserialize_message(&msg_bytes) {
                Ok(message) => {
                    if inbound_tx.send((peer_id.clone(), message)).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to deserialize message: {}", e);
                }
            }
        }
    }

    async fn write_loop(
        mut writer: tokio::net::tcp::OwnedWriteHalf,
        recv: &mut mpsc::Receiver<NetworkMessage>,
    ) {
        while let Some(message) = recv.recv().await {
            match serialize_message(&message) {
                Ok(bytes) => {
                    let len = bytes.len() as u32;
                    let mut frame = BytesMut::with_capacity(4 + bytes.len());
                    frame.put_u32(len);
                    frame.extend_from_slice(&bytes);

                    if writer.write_all(&frame).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize message: {}", e);
                }
            }
        }
    }

    async fn write_loop_tls<S: tokio::io::AsyncWrite + Unpin>(
        mut writer: WriteHalf<S>,
        recv: &mut mpsc::Receiver<NetworkMessage>,
    ) {
        while let Some(message) = recv.recv().await {
            match serialize_message(&message) {
                Ok(bytes) => {
                    let len = bytes.len() as u32;
                    let mut frame = BytesMut::with_capacity(4 + bytes.len());
                    frame.put_u32(len);
                    frame.extend_from_slice(&bytes);

                    if writer.write_all(&frame).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize message: {}", e);
                }
            }
        }
    }

    /// Starts the listener.
    async fn start_listener(&self) -> Result<(), TransportError> {
        let listener = TcpListener::bind(self.config.listen_addr).await?;
        // Save the actual bound address (important when port 0 is used)
        *self.actual_addr.write() = Some(listener.local_addr()?);
        let running = self.running.clone();
        let inbound_tx = self.inbound_tx.clone();
        let tls_acceptor = self.tls_acceptor.clone();
        let tcp_nodelay = self.config.tcp_nodelay;
        let connections = self.connections.clone();

        running.store(true, std::sync::atomic::Ordering::SeqCst);

        tokio::spawn(async move {
            while running.load(std::sync::atomic::Ordering::SeqCst) {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        if tcp_nodelay {
                            let _ = stream.set_nodelay(true);
                        }

                        let inbound_tx = inbound_tx.clone();
                        let tls_acceptor = tls_acceptor.clone();
                        let connections = connections.clone();

                        tokio::spawn(async move {
                            Self::handle_connection(stream, addr, inbound_tx, tls_acceptor, connections).await;
                        });
                    }
                    Err(e) => {
                        tracing::error!("Accept error: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    async fn handle_connection(
        stream: TcpStream,
        addr: SocketAddr,
        inbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
        tls_acceptor: Option<TlsAcceptor>,
        _connections: Arc<RwLock<HashMap<PeerId, Vec<Arc<PeerConnection>>>>>,
    ) {
        // For inbound connections, we don't know the peer ID yet
        // We'll get it from the first message
        let placeholder_id = PeerId::from_string(format!("unknown-{}", addr));

        if let Some(acceptor) = tls_acceptor {
            match acceptor.accept(stream).await {
                Ok(tls_stream) => {
                    let (read_half, _write_half) = tokio::io::split(tls_stream);
                    Self::read_loop_tls(read_half, placeholder_id, inbound_tx).await;
                }
                Err(e) => {
                    tracing::error!("TLS accept error: {}", e);
                }
            }
        } else {
            let (read_half, _write_half) = stream.into_split();
            Self::read_loop(read_half, placeholder_id, inbound_tx).await;
        }
    }

    /// Returns the number of active connections.
    pub fn connection_count(&self) -> usize {
        self.connections.read()
            .values()
            .map(|v| v.len())
            .sum()
    }

    /// Gets connection stats for a peer.
    pub fn get_peer_stats(&self, peer_id: &PeerId) -> Option<(u64, u64, u64, u64)> {
        let conns = self.connections.read();
        conns.get(peer_id).and_then(|v| {
            v.first().map(|c| (c.messages_sent, c.messages_received, c.bytes_sent, c.bytes_received))
        })
    }

    /// Disconnects from a peer.
    pub fn disconnect(&self, peer_id: &PeerId) {
        let mut conns = self.connections.write();
        conns.remove(peer_id);
    }
}

#[async_trait]
impl Transport for TcpTransport {
    async fn send(&self, peer: &PeerId, addr: &SocketAddr, message: NetworkMessage) -> Result<(), TransportError> {
        // Get or create connection
        let connection = self.connect(peer.clone(), *addr).await?;
        connection.send(message).await
    }

    async fn recv(&self) -> Result<(PeerId, NetworkMessage), TransportError> {
        let mut rx = self.inbound_rx.lock().await;
        rx.recv().await.ok_or(TransportError::ConnectionClosed)
    }

    async fn start(&self) -> Result<(), TransportError> {
        self.start_listener().await
    }

    async fn stop(&self) -> Result<(), TransportError> {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn local_addr(&self) -> SocketAddr {
        self.actual_addr
            .read()
            .unwrap_or(self.config.listen_addr)
    }
}

/// Connection pool for managing multiple connections to peers.
pub struct ConnectionPool {
    /// Transport layer.
    transport: Arc<TcpTransport>,
    /// Peer addresses.
    peer_addrs: Arc<RwLock<HashMap<PeerId, SocketAddr>>>,
    /// Maximum pool size (0 = unlimited).
    max_pool_size: usize,
}

impl ConnectionPool {
    /// Creates a new connection pool.
    pub fn new(transport: Arc<TcpTransport>) -> Self {
        Self {
            transport,
            peer_addrs: Arc::new(RwLock::new(HashMap::new())),
            max_pool_size: 256,
        }
    }

    /// Creates a new connection pool with a maximum size.
    pub fn with_max_size(transport: Arc<TcpTransport>, max_pool_size: usize) -> Self {
        Self {
            transport,
            peer_addrs: Arc::new(RwLock::new(HashMap::new())),
            max_pool_size,
        }
    }

    /// Registers a peer address. Returns false if the pool is at capacity.
    pub fn register_peer(&self, peer_id: PeerId, addr: SocketAddr) -> bool {
        let mut addrs = self.peer_addrs.write();
        // Allow re-registration of existing peers (address update)
        if addrs.contains_key(&peer_id) {
            addrs.insert(peer_id, addr);
            return true;
        }
        if self.max_pool_size > 0 && addrs.len() >= self.max_pool_size {
            tracing::warn!(
                "Connection pool at capacity ({}/{}), rejecting peer {}",
                addrs.len(),
                self.max_pool_size,
                peer_id,
            );
            return false;
        }
        addrs.insert(peer_id, addr);
        true
    }

    /// Unregisters a peer.
    pub fn unregister_peer(&self, peer_id: &PeerId) {
        self.peer_addrs.write().remove(peer_id);
        self.transport.disconnect(peer_id);
    }

    /// Sends a message to a peer.
    pub async fn send(&self, peer_id: &PeerId, message: NetworkMessage) -> Result<(), TransportError> {
        let addr = {
            self.peer_addrs.read().get(peer_id).copied()
        };

        match addr {
            Some(addr) => self.transport.send(peer_id, &addr, message).await,
            None => Err(TransportError::PeerNotFound(peer_id.0.clone())),
        }
    }

    /// Broadcasts a message to all peers.
    pub async fn broadcast(&self, message: NetworkMessage) -> Vec<(PeerId, Result<(), TransportError>)> {
        let peers: Vec<(PeerId, SocketAddr)> = {
            self.peer_addrs.read().iter()
                .map(|(id, addr)| (id.clone(), *addr))
                .collect()
        };

        let mut results = Vec::new();
        for (peer_id, addr) in peers {
            let result = self.transport.send(&peer_id, &addr, message.clone()).await;
            results.push((peer_id, result));
        }
        results
    }

    /// Returns the number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.peer_addrs.read().len()
    }

    /// Returns all registered peer IDs.
    pub fn peers(&self) -> Vec<PeerId> {
        self.peer_addrs.read().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::messages::{HeartbeatMessage, MessagePayload};

    #[tokio::test]
    async fn test_transport_config_default() {
        let config = TransportConfig::default();
        assert!(!config.use_tls);
        assert!(config.tcp_nodelay);
    }

    #[tokio::test]
    async fn test_transport_creation() {
        let local_id = PeerId::random();
        let config = TransportConfig::default();
        let transport = TcpTransport::new(local_id, config);
        assert!(transport.is_ok());
    }

    #[tokio::test]
    async fn test_connection_pool() {
        let local_id = PeerId::random();
        let config = TransportConfig::default();
        let transport = Arc::new(TcpTransport::new(local_id, config).unwrap());
        let pool = ConnectionPool::new(transport);

        let peer_id = PeerId::random();
        pool.register_peer(peer_id.clone(), "127.0.0.1:9001".parse().unwrap());

        assert_eq!(pool.peer_count(), 1);
        assert!(pool.peers().contains(&peer_id));

        pool.unregister_peer(&peer_id);
        assert_eq!(pool.peer_count(), 0);
    }

    #[tokio::test]
    async fn test_connection_pool_max_size_enforced() {
        let local_id = PeerId::random();
        let config = TransportConfig::default();
        let transport = Arc::new(TcpTransport::new(local_id, config).unwrap());
        let pool = ConnectionPool::with_max_size(transport, 3);

        // Fill to capacity
        for i in 0..3 {
            let peer = PeerId::from_string(format!("peer-{i}"));
            let addr: SocketAddr = format!("127.0.0.1:{}", 9001 + i).parse().unwrap();
            assert!(pool.register_peer(peer, addr));
        }
        assert_eq!(pool.peer_count(), 3);

        // 4th peer should be rejected
        let extra = PeerId::from_string("peer-extra".to_string());
        assert!(!pool.register_peer(extra, "127.0.0.1:9999".parse().unwrap()));
        assert_eq!(pool.peer_count(), 3);
    }

    #[tokio::test]
    async fn test_connection_pool_reregister_existing_peer() {
        let local_id = PeerId::random();
        let config = TransportConfig::default();
        let transport = Arc::new(TcpTransport::new(local_id, config).unwrap());
        let pool = ConnectionPool::with_max_size(transport, 2);

        let peer = PeerId::from_string("peer-1".to_string());
        assert!(pool.register_peer(peer.clone(), "127.0.0.1:9001".parse().unwrap()));
        let peer2 = PeerId::from_string("peer-2".to_string());
        assert!(pool.register_peer(peer2, "127.0.0.1:9002".parse().unwrap()));
        assert_eq!(pool.peer_count(), 2);

        // Re-registering an existing peer with new address should succeed
        assert!(pool.register_peer(peer, "127.0.0.1:9003".parse().unwrap()));
        assert_eq!(pool.peer_count(), 2);
    }

    #[tokio::test]
    async fn test_connection_pool_unlimited_when_zero() {
        let local_id = PeerId::random();
        let config = TransportConfig::default();
        let transport = Arc::new(TcpTransport::new(local_id, config).unwrap());
        let pool = ConnectionPool::with_max_size(transport, 0);

        // Should accept many peers when limit is 0 (unlimited)
        for i in 0..50 {
            let peer = PeerId::from_string(format!("peer-{i}"));
            let addr: SocketAddr = format!("127.0.0.1:{}", 9001 + i).parse().unwrap();
            assert!(pool.register_peer(peer, addr));
        }
        assert_eq!(pool.peer_count(), 50);
    }

    #[test]
    fn test_transport_config_max_pool_size_default() {
        let config = TransportConfig::default();
        assert_eq!(config.max_pool_size, 256);
    }
}
