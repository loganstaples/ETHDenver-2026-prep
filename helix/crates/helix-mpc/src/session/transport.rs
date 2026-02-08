//! Transport abstraction for MPC communication.
//!
//! Provides a unified trait for sending/receiving raw bytes between MPC parties,
//! with implementations for local (in-memory) and TCP transports.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

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
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
}
