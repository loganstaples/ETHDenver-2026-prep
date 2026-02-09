//! Transport abstraction for MPC communication.
//!
//! Provides a unified trait for sending/receiving raw bytes between MPC parties,
//! with implementations for local (in-memory) and TCP transports.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as TokioMutex;

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Handshake message exchanged when peers first connect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMessage {
    /// The sender's party ID.
    pub party_id: PartyId,
    /// Protocol version for compatibility checking.
    pub protocol_version: u32,
    /// Number of parties expected in the session.
    pub num_parties: usize,
    /// Random seed contribution for synchronized randomness.
    pub seed_contribution: [u8; 32],
}

/// The current protocol version.
pub const PROTOCOL_VERSION: u32 = 1;

/// Transport layer for MPC message passing.
///
/// This trait abstracts over the actual communication mechanism (in-memory, TCP,
/// etc.) and provides a simple async send/recv interface operating on raw bytes.
#[async_trait]
pub trait MPCTransport: Send + Sync {
    /// Sends a message to a specific party.
    async fn send(&self, party: &PartyId, msg: &[u8]) -> MPCResult<()>;

    /// Receives a message from a specific party.
    ///
    /// Blocks until a message is available or the channel closes.
    async fn recv(&self, party: &PartyId) -> MPCResult<Vec<u8>>;

    /// Broadcasts a message to all other parties.
    async fn broadcast(&self, msg: &[u8]) -> MPCResult<()>;

    /// Returns the local party's ID.
    fn party_id(&self) -> &PartyId;

    /// Returns all peer party IDs (excluding self).
    fn peers(&self) -> Vec<PartyId>;
}

// ---------------------------------------------------------------------------
// LocalTransport — in-memory transport for single-process testing
// ---------------------------------------------------------------------------

/// In-memory transport that routes messages through shared async channels.
///
/// All parties must be in the same process. Uses `tokio::sync::mpsc` channels
/// for async message delivery.
pub struct LocalTransport {
    party: PartyId,
    peers_list: Vec<PartyId>,
    /// Outgoing channels: party_id → sender
    senders: HashMap<String, tokio::sync::mpsc::Sender<Vec<u8>>>,
    /// Incoming channels: party_id → receiver (mutex for &self recv)
    receivers: HashMap<String, TokioMutex<tokio::sync::mpsc::Receiver<Vec<u8>>>>,
}

impl LocalTransport {
    /// Creates a set of interconnected local transports for the given parties.
    ///
    /// Returns one transport per party, all wired together with bidirectional
    /// channels.
    pub fn create_mesh(parties: &[PartyId]) -> Vec<Self> {
        let n = parties.len();

        // For each ordered pair (i, j), create a channel from i→j.
        let mut tx_map: HashMap<(usize, usize), tokio::sync::mpsc::Sender<Vec<u8>>> =
            HashMap::new();
        let mut rx_map: HashMap<(usize, usize), tokio::sync::mpsc::Receiver<Vec<u8>>> =
            HashMap::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let (tx, rx) = tokio::sync::mpsc::channel(4096);
                    tx_map.insert((i, j), tx);
                    rx_map.insert((i, j), rx);
                }
            }
        }

        let mut transports = Vec::with_capacity(n);

        for i in 0..n {
            let mut senders = HashMap::new();
            let mut receivers = HashMap::new();
            let mut peers_list = Vec::new();

            for j in 0..n {
                if i != j {
                    // i sends to j: use tx from (i,j)
                    let tx = tx_map.remove(&(i, j)).unwrap();
                    // i receives from j: use rx from (j,i)
                    let rx = rx_map.remove(&(j, i)).unwrap();
                    senders.insert(parties[j].0.clone(), tx);
                    receivers.insert(parties[j].0.clone(), TokioMutex::new(rx));
                    peers_list.push(parties[j].clone());
                }
            }

            transports.push(LocalTransport {
                party: parties[i].clone(),
                peers_list,
                senders,
                receivers,
            });
        }

        transports
    }
}

#[async_trait]
impl MPCTransport for LocalTransport {
    async fn send(&self, party: &PartyId, msg: &[u8]) -> MPCResult<()> {
        let tx = self.senders.get(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!("no channel to party {}", party))
        })?;
        tx.send(msg.to_vec()).await.map_err(|e| {
            MPCError::CommunicationError(format!("send to {} failed: {}", party, e))
        })
    }

    async fn recv(&self, party: &PartyId) -> MPCResult<Vec<u8>> {
        let rx_mutex = self.receivers.get(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!("no channel from party {}", party))
        })?;
        let mut rx = rx_mutex.lock().await;
        rx.recv().await.ok_or_else(|| {
            MPCError::CommunicationError(format!("channel from {} closed", party))
        })
    }

    async fn broadcast(&self, msg: &[u8]) -> MPCResult<()> {
        for peer in &self.peers_list {
            self.send(peer, msg).await?;
        }
        Ok(())
    }

    fn party_id(&self) -> &PartyId {
        &self.party
    }

    fn peers(&self) -> Vec<PartyId> {
        self.peers_list.clone()
    }
}

// ---------------------------------------------------------------------------
// TcpTransport — TCP transport for distributed MPC
// ---------------------------------------------------------------------------

#[cfg(feature = "network-mpc")]
use std::net::SocketAddr;
#[cfg(feature = "network-mpc")]
use std::time::Duration;

/// TCP-based transport for real distributed MPC.
///
/// Each party binds a TCP listener and connects to all peers. Messages are
/// framed as `[4-byte big-endian length][payload]`. Connections are established
/// using a deterministic strategy: the party with the lexicographically smaller
/// ID initiates the connection to avoid duplicates.
#[cfg(feature = "network-mpc")]
pub struct TcpTransport {
    party: PartyId,
    peers_list: Vec<PartyId>,
    /// Outgoing: party_id → sender half feeding the writer task
    senders: HashMap<String, tokio::sync::mpsc::Sender<Vec<u8>>>,
    /// Incoming: party_id → receiver half from the reader task
    receivers: HashMap<String, TokioMutex<tokio::sync::mpsc::Receiver<Vec<u8>>>>,
    /// The address we actually bound to.
    pub local_addr: SocketAddr,
}

#[cfg(feature = "network-mpc")]
impl TcpTransport {
    /// Creates a new TCP transport by binding and connecting to all peers.
    ///
    /// `bind_addr` — local address to listen on (use port 0 for OS-assigned).
    /// `party` — this party's ID.
    /// `peer_addrs` — map from peer party ID to their listen address.
    ///
    /// Connection strategy: the party with the smaller ID connects to the
    /// party with the larger ID. This prevents duplicate connections.
    pub async fn bind(
        bind_addr: SocketAddr,
        party: PartyId,
        peer_addrs: &HashMap<PartyId, SocketAddr>,
    ) -> MPCResult<Self> {
        use tokio::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind(bind_addr)
            .await
            .map_err(|e| MPCError::CommunicationError(format!("bind failed: {}", e)))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| MPCError::CommunicationError(e.to_string()))?;

        tracing::info!("Party {} listening on {}", party, local_addr);

        let mut peers_list: Vec<PartyId> = peer_addrs.keys().cloned().collect();
        peers_list.sort();

        // Create per-peer channel pairs
        let mut out_tx_map: HashMap<String, tokio::sync::mpsc::Sender<Vec<u8>>> = HashMap::new();
        let mut out_rx_map: HashMap<String, tokio::sync::mpsc::Receiver<Vec<u8>>> = HashMap::new();
        let mut in_tx_map: HashMap<String, tokio::sync::mpsc::Sender<Vec<u8>>> = HashMap::new();
        let mut in_rx_map: HashMap<String, tokio::sync::mpsc::Receiver<Vec<u8>>> = HashMap::new();

        for peer in &peers_list {
            let (otx, orx) = tokio::sync::mpsc::channel(4096);
            out_tx_map.insert(peer.0.clone(), otx);
            out_rx_map.insert(peer.0.clone(), orx);

            let (itx, irx) = tokio::sync::mpsc::channel(4096);
            in_tx_map.insert(peer.0.clone(), itx);
            in_rx_map.insert(peer.0.clone(), irx);
        }

        let party_str = party.0.clone();

        // Peers with smaller IDs will connect to us; we accept them
        let accept_peers: Vec<PartyId> = peers_list
            .iter()
            .filter(|p| p.0 < party_str)
            .cloned()
            .collect();

        // We connect to peers with larger IDs
        let connect_peers: Vec<(PartyId, SocketAddr)> = peers_list
            .iter()
            .filter(|p| p.0 > party_str)
            .map(|p| (p.clone(), peer_addrs[p]))
            .collect();

        // Spawn accept task
        let accept_handle = if !accept_peers.is_empty() {
            let accept_count = accept_peers.len();
            let in_tx = in_tx_map.clone();
            let accept_party = party.clone();
            // Move out the receivers for accepted peers
            let mut accept_out_rx: HashMap<String, tokio::sync::mpsc::Receiver<Vec<u8>>> =
                HashMap::new();
            for p in &accept_peers {
                if let Some(rx) = out_rx_map.remove(&p.0) {
                    accept_out_rx.insert(p.0.clone(), rx);
                }
            }

            Some(tokio::spawn(async move {
                let mut accepted = 0usize;
                let mut accept_out_rx = accept_out_rx;

                while accepted < accept_count {
                    match listener.accept().await {
                        Ok((mut stream, _)) => {
                            // Read peer's handshake
                            let peer_hs = match read_handshake(&mut stream).await {
                                Ok(hs) => hs,
                                Err(e) => {
                                    tracing::warn!("handshake read failed: {}", e);
                                    continue;
                                }
                            };

                            let peer_id = peer_hs.party_id.0.clone();

                            // Send our handshake
                            let our_hs = HandshakeMessage {
                                party_id: accept_party.clone(),
                                protocol_version: PROTOCOL_VERSION,
                                num_parties: accept_count + 1,
                                seed_contribution: [0u8; 32],
                            };
                            if write_handshake(&mut stream, &our_hs).await.is_err() {
                                continue;
                            }

                            // Split stream and spawn reader/writer
                            let (reader, writer) = tokio::io::split(stream);

                            if let Some(itx) = in_tx.get(&peer_id) {
                                tokio::spawn(reader_loop(reader, itx.clone()));
                            }
                            if let Some(orx) = accept_out_rx.remove(&peer_id) {
                                tokio::spawn(writer_loop(writer, orx));
                            }

                            accepted += 1;
                        }
                        Err(e) => {
                            tracing::error!("accept failed: {}", e);
                        }
                    }
                }
            }))
        } else {
            None
        };

        // Spawn connect tasks
        let mut connect_handles = Vec::new();
        for (peer, addr) in connect_peers {
            let our_hs = HandshakeMessage {
                party_id: party.clone(),
                protocol_version: PROTOCOL_VERSION,
                num_parties: peers_list.len() + 1,
                seed_contribution: [0u8; 32],
            };

            let itx = in_tx_map.get(&peer.0).cloned();
            let orx = out_rx_map.remove(&peer.0);

            let handle = tokio::spawn(async move {
                // Retry connection
                let mut attempts = 0u32;
                let mut stream = loop {
                    match TcpStream::connect(addr).await {
                        Ok(s) => break s,
                        Err(e) => {
                            attempts += 1;
                            if attempts > 50 {
                                tracing::error!("gave up connecting to {} at {}: {}", peer, addr, e);
                                return;
                            }
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                    }
                };

                // Send our handshake
                if write_handshake(&mut stream, &our_hs).await.is_err() {
                    return;
                }
                // Read peer's handshake
                if read_handshake(&mut stream).await.is_err() {
                    return;
                }

                let (reader, writer) = tokio::io::split(stream);
                if let Some(itx) = itx {
                    tokio::spawn(reader_loop(reader, itx));
                }
                if let Some(orx) = orx {
                    tokio::spawn(writer_loop(writer, orx));
                }
            });
            connect_handles.push(handle);
        }

        // Wait for all connections
        if let Some(h) = accept_handle {
            let _ = tokio::time::timeout(Duration::from_secs(10), h).await;
        }
        for h in connect_handles {
            let _ = tokio::time::timeout(Duration::from_secs(10), h).await;
        }

        // Build receivers map
        let mut receivers = HashMap::new();
        for (pid, rx) in in_rx_map {
            receivers.insert(pid, TokioMutex::new(rx));
        }

        Ok(Self {
            party,
            peers_list,
            senders: out_tx_map,
            receivers,
            local_addr,
        })
    }
}

/// Reads a length-prefixed handshake from a stream.
#[cfg(feature = "network-mpc")]
async fn read_handshake<S: tokio::io::AsyncRead + Unpin>(
    stream: &mut S,
) -> MPCResult<HandshakeMessage> {
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| MPCError::CommunicationError(format!("handshake read len: {}", e)))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 1024 * 1024 {
        return Err(MPCError::CommunicationError("handshake too large".into()));
    }
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .await
        .map_err(|e| MPCError::CommunicationError(format!("handshake read body: {}", e)))?;
    bincode::deserialize(&buf)
        .map_err(|e| MPCError::CommunicationError(format!("handshake decode: {}", e)))
}

/// Writes a length-prefixed handshake to a stream.
#[cfg(feature = "network-mpc")]
async fn write_handshake<S: tokio::io::AsyncWrite + Unpin>(
    stream: &mut S,
    hs: &HandshakeMessage,
) -> MPCResult<()> {
    use tokio::io::AsyncWriteExt;
    let data = bincode::serialize(hs)
        .map_err(|e| MPCError::CommunicationError(format!("handshake encode: {}", e)))?;
    let len = (data.len() as u32).to_be_bytes();
    stream
        .write_all(&len)
        .await
        .map_err(|e| MPCError::CommunicationError(format!("handshake write: {}", e)))?;
    stream
        .write_all(&data)
        .await
        .map_err(|e| MPCError::CommunicationError(format!("handshake write: {}", e)))?;
    stream
        .flush()
        .await
        .map_err(|e| MPCError::CommunicationError(format!("handshake flush: {}", e)))?;
    Ok(())
}

/// Maximum allowed message size (64 MB). Prevents memory exhaustion from
/// malicious or corrupted length prefixes.
#[cfg(feature = "network-mpc")]
pub const MAX_MSG_SIZE: usize = 64 * 1024 * 1024;

/// Background task: reads length-prefixed messages and forwards to channel.
/// Enforces MAX_MSG_SIZE and logs errors before dropping the connection.
#[cfg(feature = "network-mpc")]
async fn reader_loop(
    mut reader: tokio::io::ReadHalf<tokio::net::TcpStream>,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
) {
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    loop {
        if let Err(e) = reader.read_exact(&mut len_buf).await {
            tracing::debug!("reader_loop: peer disconnected (read len): {}", e);
            break;
        }
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_MSG_SIZE {
            tracing::error!(
                "reader_loop: message too large ({} bytes, max {}), dropping connection",
                len,
                MAX_MSG_SIZE
            );
            break;
        }
        let mut buf = vec![0u8; len];
        if let Err(e) = reader.read_exact(&mut buf).await {
            tracing::debug!("reader_loop: peer disconnected (read body): {}", e);
            break;
        }
        if tx.send(buf).await.is_err() {
            tracing::debug!("reader_loop: channel closed, stopping");
            break;
        }
    }
}

/// Background task: reads from channel and writes length-prefixed messages.
/// Enforces MAX_MSG_SIZE on outgoing messages and logs errors.
#[cfg(feature = "network-mpc")]
async fn writer_loop(
    mut writer: tokio::io::WriteHalf<tokio::net::TcpStream>,
    mut rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
) {
    use tokio::io::AsyncWriteExt;
    while let Some(data) = rx.recv().await {
        if data.len() > MAX_MSG_SIZE {
            tracing::error!(
                "writer_loop: refusing to send message of {} bytes (max {})",
                data.len(),
                MAX_MSG_SIZE
            );
            continue;
        }
        let len = (data.len() as u32).to_be_bytes();
        if let Err(e) = writer.write_all(&len).await {
            tracing::debug!("writer_loop: write len failed: {}", e);
            break;
        }
        if let Err(e) = writer.write_all(&data).await {
            tracing::debug!("writer_loop: write body failed: {}", e);
            break;
        }
        if let Err(e) = writer.flush().await {
            tracing::debug!("writer_loop: flush failed: {}", e);
            break;
        }
    }
}

#[cfg(feature = "network-mpc")]
impl TcpTransport {
    /// Receives a message with a timeout.
    pub async fn recv_timeout(
        &self,
        party: &PartyId,
        timeout: Duration,
    ) -> MPCResult<Vec<u8>> {
        let rx_mutex = self.receivers.get(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!("no channel from party {}", party))
        })?;
        let mut rx = rx_mutex.lock().await;
        tokio::time::timeout(timeout, rx.recv())
            .await
            .map_err(|_| MPCError::Timeout {
                party: party.clone(),
                phase: "recv".into(),
            })?
            .ok_or_else(|| {
                MPCError::CommunicationError(format!("channel from {} closed", party))
            })
    }

    /// Returns the total number of parties (self + peers).
    pub fn num_parties(&self) -> usize {
        self.peers_list.len() + 1
    }
}

#[cfg(feature = "network-mpc")]
#[async_trait]
impl MPCTransport for TcpTransport {
    async fn send(&self, party: &PartyId, msg: &[u8]) -> MPCResult<()> {
        if msg.len() > MAX_MSG_SIZE {
            return Err(MPCError::CommunicationError(format!(
                "message too large: {} bytes (max {})",
                msg.len(),
                MAX_MSG_SIZE
            )));
        }
        let tx = self.senders.get(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!("no channel to party {}", party))
        })?;
        tx.send(msg.to_vec()).await.map_err(|e| {
            MPCError::CommunicationError(format!("send to {} failed: {}", party, e))
        })
    }

    async fn recv(&self, party: &PartyId) -> MPCResult<Vec<u8>> {
        let rx_mutex = self.receivers.get(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!("no channel from party {}", party))
        })?;
        let mut rx = rx_mutex.lock().await;
        rx.recv().await.ok_or_else(|| {
            MPCError::CommunicationError(format!("channel from {} closed", party))
        })
    }

    async fn broadcast(&self, msg: &[u8]) -> MPCResult<()> {
        for peer in &self.peers_list {
            self.send(peer, msg).await?;
        }
        Ok(())
    }

    fn party_id(&self) -> &PartyId {
        &self.party
    }

    fn peers(&self) -> Vec<PartyId> {
        self.peers_list.clone()
    }
}

// ---------------------------------------------------------------------------
// AuthenticatedTransport — HMAC + sequence numbers for replay protection
// ---------------------------------------------------------------------------

use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// A message wrapper that includes HMAC authentication and a sequence number
/// for replay protection.
///
/// The HMAC is computed as `SHA256(session_key || sequence || payload)`, which
/// binds the authentication tag to both the content and the ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedMessage {
    /// Monotonically increasing sequence number per sender.
    pub sequence: u64,
    /// The raw payload bytes.
    pub payload: Vec<u8>,
    /// HMAC tag: SHA256(session_key || sequence_le_bytes || payload).
    pub hmac: [u8; 32],
}

impl AuthenticatedMessage {
    /// Computes the HMAC for the given key, sequence, and payload.
    fn compute_hmac(session_key: &[u8; 32], sequence: u64, payload: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(session_key);
        hasher.update(sequence.to_le_bytes());
        hasher.update(payload);
        let result = hasher.finalize();
        let mut tag = [0u8; 32];
        tag.copy_from_slice(&result);
        tag
    }

    /// Creates a new authenticated message with the computed HMAC.
    pub fn new(session_key: &[u8; 32], sequence: u64, payload: Vec<u8>) -> Self {
        let hmac = Self::compute_hmac(session_key, sequence, &payload);
        Self {
            sequence,
            payload,
            hmac,
        }
    }

    /// Verifies the HMAC in constant time and returns the payload on success.
    pub fn verify_and_extract(
        &self,
        session_key: &[u8; 32],
    ) -> Result<&[u8], &'static str> {
        let expected = Self::compute_hmac(session_key, self.sequence, &self.payload);
        // Constant-time comparison to prevent timing side-channels.
        if constant_time_eq(&self.hmac, &expected) {
            Ok(&self.payload)
        } else {
            Err("HMAC verification failed")
        }
    }
}

/// Constant-time byte array comparison. Returns true iff `a == b`.
///
/// Uses XOR accumulation so the comparison time is independent of where
/// the first difference occurs, preventing timing side-channel attacks.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff: u8 = 0;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// A transport wrapper that authenticates every message with HMAC-SHA256
/// and tracks per-peer sequence numbers for replay protection.
///
/// Wraps any [`MPCTransport`] implementation transparently. On `send()`,
/// the payload is wrapped in an [`AuthenticatedMessage`] with an incrementing
/// sequence number and HMAC tag. On `recv()`, the HMAC is verified and the
/// sequence number is checked against the expected minimum for that peer.
///
/// # Sequence Number Policy
///
/// The receiver accepts any message where `sequence >= expected_seq` to
/// tolerate out-of-order delivery (common in async channels under load).
/// After accepting, it updates `expected_seq = sequence + 1`.
pub struct AuthenticatedTransport<T: MPCTransport> {
    inner: T,
    session_key: [u8; 32],
    /// Per-peer send sequence counters. The key is `(our_party_id, peer_id)`.
    /// In practice we use a single atomic since sends are serialized per-peer
    /// by the caller, but an atomic is simpler and correct for concurrent use.
    send_seq: AtomicU64,
    /// Per-peer expected receive sequence number.
    /// Protected by a std::sync::Mutex because the critical section is tiny
    /// (just a HashMap lookup + u64 compare-and-update).
    recv_seq: Mutex<HashMap<String, u64>>,
}

impl<T: MPCTransport> AuthenticatedTransport<T> {
    /// Creates a new authenticated transport with the given session key.
    ///
    /// The session key must be a shared secret known to all parties (e.g.,
    /// derived from a key exchange during session establishment).
    pub fn new(inner: T, session_key: [u8; 32]) -> Self {
        Self {
            inner,
            session_key,
            send_seq: AtomicU64::new(0),
            recv_seq: Mutex::new(HashMap::new()),
        }
    }

    /// Creates an authenticated transport by deriving a session key from
    /// the handshake seed contributions of all parties.
    ///
    /// The key is computed as `SHA256("helix-mpc-auth" || seed_0 || seed_1 || ...)`
    /// where seeds are sorted by party ID for determinism.
    pub fn from_seeds(inner: T, mut seed_contributions: Vec<(String, [u8; 32])>) -> Self {
        // Sort by party ID for deterministic key derivation regardless of
        // the order in which handshakes completed.
        seed_contributions.sort_by(|a, b| a.0.cmp(&b.0));

        let mut hasher = Sha256::new();
        hasher.update(b"helix-mpc-auth");
        for (party_id, seed) in &seed_contributions {
            hasher.update(party_id.as_bytes());
            hasher.update(seed);
        }
        let result = hasher.finalize();
        let mut session_key = [0u8; 32];
        session_key.copy_from_slice(&result);

        Self::new(inner, session_key)
    }

    /// Returns a reference to the session key (for testing/debugging).
    pub fn session_key(&self) -> &[u8; 32] {
        &self.session_key
    }
}

#[async_trait]
impl<T: MPCTransport + Send + Sync> MPCTransport for AuthenticatedTransport<T> {
    async fn send(&self, party: &PartyId, msg: &[u8]) -> MPCResult<()> {
        let seq = self.send_seq.fetch_add(1, Ordering::SeqCst);
        let auth_msg = AuthenticatedMessage::new(&self.session_key, seq, msg.to_vec());
        let envelope = bincode::serialize(&auth_msg).map_err(|e| {
            MPCError::CommunicationError(format!(
                "failed to serialize authenticated message: {}",
                e
            ))
        })?;
        self.inner.send(party, &envelope).await
    }

    async fn recv(&self, party: &PartyId) -> MPCResult<Vec<u8>> {
        let envelope = self.inner.recv(party).await?;
        let auth_msg: AuthenticatedMessage = bincode::deserialize(&envelope).map_err(|e| {
            MPCError::CommunicationError(format!(
                "failed to deserialize authenticated message: {}",
                e
            ))
        })?;

        // Verify HMAC before checking sequence to avoid leaking sequence info
        // on tampered messages.
        auth_msg.verify_and_extract(&self.session_key).map_err(|_| {
            MPCError::CommunicationError(format!(
                "HMAC verification failed for message from {}",
                party
            ))
        })?;

        // Check and update the expected sequence number for this peer.
        {
            let mut seq_map = self.recv_seq.lock().map_err(|_| {
                MPCError::CommunicationError("recv_seq mutex poisoned".into())
            })?;
            let expected = seq_map.entry(party.0.clone()).or_insert(0);
            if auth_msg.sequence < *expected {
                return Err(MPCError::CommunicationError(format!(
                    "sequence number regression from {}: got {}, expected >= {}",
                    party, auth_msg.sequence, *expected
                )));
            }
            *expected = auth_msg.sequence + 1;
        }

        Ok(auth_msg.payload)
    }

    async fn broadcast(&self, msg: &[u8]) -> MPCResult<()> {
        for peer in self.inner.peers() {
            self.send(&peer, msg).await?;
        }
        Ok(())
    }

    fn party_id(&self) -> &PartyId {
        self.inner.party_id()
    }

    fn peers(&self) -> Vec<PartyId> {
        self.inner.peers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[tokio::test]
    async fn test_local_transport_send_recv() {
        let parties = test_parties(3);
        let transports = LocalTransport::create_mesh(&parties);

        // Party 0 sends to party 1
        transports[0]
            .send(&parties[1], b"hello from 0")
            .await
            .unwrap();

        let msg = transports[1].recv(&parties[0]).await.unwrap();
        assert_eq!(msg, b"hello from 0");
    }

    #[tokio::test]
    async fn test_local_transport_broadcast() {
        let parties = test_parties(3);
        let transports = LocalTransport::create_mesh(&parties);

        // Party 0 broadcasts
        transports[0].broadcast(b"broadcast msg").await.unwrap();

        // Parties 1 and 2 should receive it
        let msg1 = transports[1].recv(&parties[0]).await.unwrap();
        let msg2 = transports[2].recv(&parties[0]).await.unwrap();
        assert_eq!(msg1, b"broadcast msg");
        assert_eq!(msg2, b"broadcast msg");
    }

    #[tokio::test]
    async fn test_local_transport_bidirectional() {
        let parties = test_parties(2);
        let transports = LocalTransport::create_mesh(&parties);

        // Simultaneous send
        transports[0]
            .send(&parties[1], b"from 0")
            .await
            .unwrap();
        transports[1]
            .send(&parties[0], b"from 1")
            .await
            .unwrap();

        let msg0 = transports[0].recv(&parties[1]).await.unwrap();
        let msg1 = transports[1].recv(&parties[0]).await.unwrap();
        assert_eq!(msg0, b"from 1");
        assert_eq!(msg1, b"from 0");
    }

    #[tokio::test]
    async fn test_local_transport_peers() {
        let parties = test_parties(3);
        let transports = LocalTransport::create_mesh(&parties);

        assert_eq!(transports[0].party_id(), &parties[0]);
        assert_eq!(transports[0].peers().len(), 2);
        assert_eq!(transports[1].peers().len(), 2);
    }

    // -----------------------------------------------------------------------
    // AuthenticatedTransport tests
    // -----------------------------------------------------------------------

    /// Helper: wraps each LocalTransport in an AuthenticatedTransport with a
    /// shared session key.
    fn wrap_authenticated(
        transports: Vec<LocalTransport>,
        session_key: [u8; 32],
    ) -> Vec<AuthenticatedTransport<LocalTransport>> {
        transports
            .into_iter()
            .map(|t| AuthenticatedTransport::new(t, session_key))
            .collect()
    }

    #[tokio::test]
    async fn test_authenticated_roundtrip() {
        let parties = test_parties(3);
        let raw = LocalTransport::create_mesh(&parties);
        let key = [0xABu8; 32];
        let transports = wrap_authenticated(raw, key);

        // Party 0 sends to party 1
        transports[0]
            .send(&parties[1], b"authenticated hello")
            .await
            .unwrap();

        let msg = transports[1].recv(&parties[0]).await.unwrap();
        assert_eq!(msg, b"authenticated hello");

        // Party 1 sends to party 2
        transports[1]
            .send(&parties[2], b"second message")
            .await
            .unwrap();

        let msg2 = transports[2].recv(&parties[1]).await.unwrap();
        assert_eq!(msg2, b"second message");

        // Party 2 sends back to party 0 (full triangle)
        transports[2]
            .send(&parties[0], b"closing the loop")
            .await
            .unwrap();

        let msg3 = transports[0].recv(&parties[2]).await.unwrap();
        assert_eq!(msg3, b"closing the loop");
    }

    #[tokio::test]
    async fn test_hmac_tamper_detection() {
        let parties = test_parties(2);
        let raw = LocalTransport::create_mesh(&parties);

        // We need to manually intervene in the channel to tamper with a
        // message, so we use raw transports for sending and wrap only the
        // receiver side with authentication.

        // Create a shared key
        let key = [0x42u8; 32];

        // Send an authenticated message through the raw channel
        let auth_msg = AuthenticatedMessage::new(&key, 0, b"secret data".to_vec());
        let mut envelope = bincode::serialize(&auth_msg).unwrap();

        // Tamper with a byte in the middle of the serialized envelope.
        // The envelope layout (bincode): sequence(8) + payload_len(8) + payload(N) + hmac(32).
        // We flip a byte in the payload region to corrupt it.
        let tamper_offset = 20; // safely inside the payload region
        if tamper_offset < envelope.len() {
            envelope[tamper_offset] ^= 0xFF;
        }

        // Send the tampered envelope through the raw transport
        raw[0].send(&parties[1], &envelope).await.unwrap();

        // Wrap party 1 in authenticated transport for receiving
        let auth_recv = AuthenticatedTransport::new(raw.into_iter().nth(1).unwrap(), key);

        // Receiving should fail because the HMAC won't match
        let result = auth_recv.recv(&parties[0]).await;
        assert!(
            result.is_err(),
            "tampered message should fail HMAC verification"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("HMAC verification failed")
                || err_msg.contains("failed to deserialize"),
            "error should mention HMAC failure or deserialization, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_sequence_number_tracking() {
        let parties = test_parties(2);
        let raw = LocalTransport::create_mesh(&parties);
        let key = [0x99u8; 32];
        let transports = wrap_authenticated(raw, key);

        // Send 5 messages from party 0 to party 1
        for i in 0u64..5 {
            let payload = format!("message {}", i);
            transports[0]
                .send(&parties[1], payload.as_bytes())
                .await
                .unwrap();
        }

        // Receive them all — sequence numbers should be 0..5
        for i in 0u64..5 {
            let msg = transports[1].recv(&parties[0]).await.unwrap();
            let expected = format!("message {}", i);
            assert_eq!(msg, expected.as_bytes());
        }

        // Verify the send counter has advanced to 5
        assert_eq!(transports[0].send_seq.load(Ordering::SeqCst), 5);

        // Verify the recv counter for party-0 on party-1's side is 5
        let recv_map = transports[1].recv_seq.lock().unwrap();
        assert_eq!(*recv_map.get("party-0").unwrap(), 5);
    }

    #[tokio::test]
    async fn test_sequence_regression_rejected() {
        // Manually craft two messages with regressing sequence numbers
        // and send them through the raw transport to verify the receiver
        // rejects the second one.
        let parties = test_parties(2);
        let raw = LocalTransport::create_mesh(&parties);
        let key = [0x77u8; 32];

        // Send message with seq=5 first (out of order but acceptable as first)
        let msg_high = AuthenticatedMessage::new(&key, 5, b"high seq".to_vec());
        let envelope_high = bincode::serialize(&msg_high).unwrap();
        raw[0].send(&parties[1], &envelope_high).await.unwrap();

        // Send message with seq=3 (regression — should be rejected)
        let msg_low = AuthenticatedMessage::new(&key, 3, b"low seq".to_vec());
        let envelope_low = bincode::serialize(&msg_low).unwrap();
        raw[0].send(&parties[1], &envelope_low).await.unwrap();

        let auth_recv = AuthenticatedTransport::new(raw.into_iter().nth(1).unwrap(), key);

        // First recv succeeds (seq=5, expected was 0)
        let result1 = auth_recv.recv(&parties[0]).await;
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), b"high seq");

        // Second recv fails (seq=3, but expected >= 6)
        let result2 = auth_recv.recv(&parties[0]).await;
        assert!(result2.is_err());
        let err_msg = format!("{}", result2.unwrap_err());
        assert!(
            err_msg.contains("sequence number regression"),
            "expected sequence regression error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_constant_time_eq_correctness() {
        let a = [0xABu8; 32];
        let b = [0xABu8; 32];
        let mut c = [0xABu8; 32];
        c[31] = 0x00;

        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
        assert!(!constant_time_eq(&[0u8; 32], &[1u8; 32]));
        assert!(constant_time_eq(&[0u8; 32], &[0u8; 32]));
    }

    #[test]
    fn test_authenticated_message_hmac_deterministic() {
        let key = [0x55u8; 32];
        let msg1 = AuthenticatedMessage::new(&key, 42, b"payload".to_vec());
        let msg2 = AuthenticatedMessage::new(&key, 42, b"payload".to_vec());
        assert_eq!(msg1.hmac, msg2.hmac, "same inputs should produce same HMAC");

        // Different sequence should produce different HMAC
        let msg3 = AuthenticatedMessage::new(&key, 43, b"payload".to_vec());
        assert_ne!(msg1.hmac, msg3.hmac, "different seq should produce different HMAC");

        // Different key should produce different HMAC
        let key2 = [0x66u8; 32];
        let msg4 = AuthenticatedMessage::new(&key2, 42, b"payload".to_vec());
        assert_ne!(msg1.hmac, msg4.hmac, "different key should produce different HMAC");
    }

    #[test]
    fn test_from_seeds_deterministic() {
        let seeds = vec![
            ("party-0".to_string(), [0x11u8; 32]),
            ("party-1".to_string(), [0x22u8; 32]),
            ("party-2".to_string(), [0x33u8; 32]),
        ];

        // Different ordering should produce the same key (sorted internally)
        let seeds_reversed = vec![
            ("party-2".to_string(), [0x33u8; 32]),
            ("party-0".to_string(), [0x11u8; 32]),
            ("party-1".to_string(), [0x22u8; 32]),
        ];

        let parties = test_parties(2);
        let raw1 = LocalTransport::create_mesh(&parties);
        let raw2 = LocalTransport::create_mesh(&parties);

        let t1 = AuthenticatedTransport::from_seeds(
            raw1.into_iter().next().unwrap(),
            seeds,
        );
        let t2 = AuthenticatedTransport::from_seeds(
            raw2.into_iter().next().unwrap(),
            seeds_reversed,
        );

        assert_eq!(
            t1.session_key(),
            t2.session_key(),
            "from_seeds should be order-independent"
        );
    }
}
