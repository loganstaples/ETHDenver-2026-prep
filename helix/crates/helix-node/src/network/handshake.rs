//! P2P Handshake Protocol for HELIX Network.
//!
//! Implements a cryptographic handshake for peer identity verification.
//! When two nodes connect, the initiator sends a `HandshakeInit` containing
//! its peer ID, role, capabilities, listen address, and a random nonce.
//! The responder validates the message and replies with a `HandshakeAck`
//! echoing the nonce to prove liveness. If validation fails, a
//! `HandshakeReject` is sent with a human-readable reason.
//!
//! The handshake uses the same length-prefix + bincode framing as the
//! transport layer for consistency.

use std::time::Duration;

use rand::RngCore;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::config::NodeRole;
use super::messages::{NodeCapabilities, PeerId};

/// Current handshake protocol version.
pub const HANDSHAKE_VERSION: u8 = 1;

/// Maximum handshake message size (64 KB — handshakes are small).
const MAX_HANDSHAKE_SIZE: usize = 64 * 1024;

/// Default handshake timeout.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Handshake protocol messages exchanged before entering the message loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandshakeMessage {
    /// Sent by the initiator (outbound connector).
    Init {
        /// Initiator's peer ID.
        peer_id: PeerId,
        /// Protocol version for compatibility checking.
        protocol_version: u8,
        /// Node role (Aggregator, Compute, Verifier).
        role: NodeRole,
        /// Node capabilities (what this peer can do).
        capabilities: NodeCapabilities,
        /// Listen address where this peer accepts connections.
        listen_addr: String,
        /// Optional ed25519 public key (32 bytes) for identity verification.
        public_key: Option<Vec<u8>>,
        /// Random nonce for liveness proof (responder must echo it).
        nonce: [u8; 32],
        /// Optional x25519 public key (32 bytes) for MPC key exchange.
        x25519_pubkey: Option<Vec<u8>>,
        /// Optional Ethereum address (20 bytes) for on-chain identity.
        eth_address: Option<Vec<u8>>,
    },
    /// Sent by the responder (inbound acceptor) to confirm the handshake.
    Ack {
        /// Responder's peer ID.
        peer_id: PeerId,
        /// Protocol version.
        protocol_version: u8,
        /// Node role.
        role: NodeRole,
        /// Node capabilities.
        capabilities: NodeCapabilities,
        /// Listen address.
        listen_addr: String,
        /// Optional ed25519 public key.
        public_key: Option<Vec<u8>>,
        /// Echo of the initiator's nonce (proves the responder actually read our Init).
        nonce_echo: [u8; 32],
        /// Optional x25519 public key (32 bytes) for MPC key exchange.
        x25519_pubkey: Option<Vec<u8>>,
        /// Optional Ethereum address (20 bytes) for on-chain identity.
        eth_address: Option<Vec<u8>>,
    },
    /// Sent by the responder to reject the handshake.
    Reject {
        /// Human-readable rejection reason.
        reason: String,
    },
}

/// Result of a successful handshake.
#[derive(Debug, Clone)]
pub struct HandshakeResult {
    /// Remote peer's ID.
    pub peer_id: PeerId,
    /// Remote peer's role.
    pub role: NodeRole,
    /// Remote peer's capabilities.
    pub capabilities: NodeCapabilities,
    /// Remote peer's listen address (for other peers to connect to).
    pub listen_addr: String,
    /// Remote peer's ed25519 public key, if provided.
    pub public_key: Option<Vec<u8>>,
    /// Remote peer's x25519 public key for MPC key exchange, if provided.
    pub x25519_pubkey: Option<Vec<u8>>,
    /// Remote peer's Ethereum address for on-chain identity, if provided.
    pub eth_address: Option<Vec<u8>>,
}

/// Handshake errors.
#[derive(Debug, thiserror::Error)]
pub enum HandshakeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Handshake rejected: {0}")]
    Rejected(String),
    #[error("Protocol version mismatch: local={local}, remote={remote}")]
    VersionMismatch { local: u8, remote: u8 },
    #[error("Nonce mismatch: responder did not echo our challenge")]
    NonceMismatch,
    #[error("Unexpected message: expected {expected}, got {got}")]
    UnexpectedMessage { expected: String, got: String },
    #[error("Message too large: {size} > {max}")]
    MessageTooLarge { size: usize, max: usize },
    #[error("Handshake timed out after {0:?}")]
    Timeout(Duration),
    #[error("Connection closed during handshake")]
    ConnectionClosed,
}

/// Performs the outbound (initiator) side of the handshake.
///
/// Sends `HandshakeInit` with a random nonce, waits for `HandshakeAck`
/// with the echoed nonce, and returns the remote peer's information.
pub async fn perform_handshake_outbound<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    local_id: &PeerId,
    role: NodeRole,
    capabilities: &NodeCapabilities,
    listen_addr: &str,
    public_key: Option<&[u8]>,
    timeout: Duration,
    x25519_pubkey: Option<&[u8]>,
    eth_address: Option<&[u8]>,
) -> Result<HandshakeResult, HandshakeError> {
    // Generate random nonce
    let mut nonce = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut nonce);

    let init = HandshakeMessage::Init {
        peer_id: local_id.clone(),
        protocol_version: HANDSHAKE_VERSION,
        role,
        capabilities: capabilities.clone(),
        listen_addr: listen_addr.to_string(),
        public_key: public_key.map(|k| k.to_vec()),
        nonce,
        x25519_pubkey: x25519_pubkey.map(|k| k.to_vec()),
        eth_address: eth_address.map(|a| a.to_vec()),
    };

    // Send Init
    write_handshake_message(stream, &init).await?;

    // Wait for Ack/Reject with timeout
    let response = tokio::time::timeout(timeout, read_handshake_message(stream))
        .await
        .map_err(|_| HandshakeError::Timeout(timeout))??;

    match response {
        HandshakeMessage::Ack {
            peer_id,
            protocol_version,
            role: remote_role,
            capabilities: remote_caps,
            listen_addr: remote_addr,
            public_key: remote_pk,
            nonce_echo,
            x25519_pubkey: remote_x25519,
            eth_address: remote_eth,
        } => {
            // Verify protocol version
            if protocol_version != HANDSHAKE_VERSION {
                return Err(HandshakeError::VersionMismatch {
                    local: HANDSHAKE_VERSION,
                    remote: protocol_version,
                });
            }

            // Verify nonce echo
            if nonce_echo != nonce {
                return Err(HandshakeError::NonceMismatch);
            }

            Ok(HandshakeResult {
                peer_id,
                role: remote_role,
                capabilities: remote_caps,
                listen_addr: remote_addr,
                public_key: remote_pk,
                x25519_pubkey: remote_x25519,
                eth_address: remote_eth,
            })
        }
        HandshakeMessage::Reject { reason } => {
            Err(HandshakeError::Rejected(reason))
        }
        HandshakeMessage::Init { .. } => {
            Err(HandshakeError::UnexpectedMessage {
                expected: "Ack or Reject".to_string(),
                got: "Init".to_string(),
            })
        }
    }
}

/// Performs the inbound (responder) side of the handshake.
///
/// Reads `HandshakeInit` from the initiator, validates it, and sends
/// `HandshakeAck` with the echoed nonce. Returns the remote peer's info.
pub async fn perform_handshake_inbound<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    local_id: &PeerId,
    role: NodeRole,
    capabilities: &NodeCapabilities,
    listen_addr: &str,
    public_key: Option<&[u8]>,
    timeout: Duration,
    max_peers: usize,
    current_peers: usize,
    x25519_pubkey: Option<&[u8]>,
    eth_address: Option<&[u8]>,
) -> Result<HandshakeResult, HandshakeError> {
    // Read Init with timeout
    let init = tokio::time::timeout(timeout, read_handshake_message(stream))
        .await
        .map_err(|_| HandshakeError::Timeout(timeout))??;

    match init {
        HandshakeMessage::Init {
            peer_id,
            protocol_version,
            role: remote_role,
            capabilities: remote_caps,
            listen_addr: remote_addr,
            public_key: remote_pk,
            nonce,
            x25519_pubkey: remote_x25519,
            eth_address: remote_eth,
        } => {
            // Check protocol version
            if protocol_version != HANDSHAKE_VERSION {
                let reject = HandshakeMessage::Reject {
                    reason: format!(
                        "Protocol version mismatch: we support v{}, you sent v{}",
                        HANDSHAKE_VERSION, protocol_version
                    ),
                };
                let _ = write_handshake_message(stream, &reject).await;
                return Err(HandshakeError::VersionMismatch {
                    local: HANDSHAKE_VERSION,
                    remote: protocol_version,
                });
            }

            // Check capacity
            if max_peers > 0 && current_peers >= max_peers {
                let reject = HandshakeMessage::Reject {
                    reason: format!(
                        "Peer limit reached ({}/{})",
                        current_peers, max_peers,
                    ),
                };
                let _ = write_handshake_message(stream, &reject).await;
                return Err(HandshakeError::Rejected("Peer limit reached".to_string()));
            }

            // Send Ack with echoed nonce
            let ack = HandshakeMessage::Ack {
                peer_id: local_id.clone(),
                protocol_version: HANDSHAKE_VERSION,
                role,
                capabilities: capabilities.clone(),
                listen_addr: listen_addr.to_string(),
                public_key: public_key.map(|k| k.to_vec()),
                nonce_echo: nonce,
                x25519_pubkey: x25519_pubkey.map(|k| k.to_vec()),
                eth_address: eth_address.map(|a| a.to_vec()),
            };
            write_handshake_message(stream, &ack).await?;

            Ok(HandshakeResult {
                peer_id,
                role: remote_role,
                capabilities: remote_caps,
                listen_addr: remote_addr,
                public_key: remote_pk,
                x25519_pubkey: remote_x25519,
                eth_address: remote_eth,
            })
        }
        HandshakeMessage::Ack { .. } => {
            Err(HandshakeError::UnexpectedMessage {
                expected: "Init".to_string(),
                got: "Ack".to_string(),
            })
        }
        HandshakeMessage::Reject { reason } => {
            Err(HandshakeError::Rejected(reason))
        }
    }
}

/// Writes a handshake message using 4-byte BE length prefix + bincode.
async fn write_handshake_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &HandshakeMessage,
) -> Result<(), HandshakeError> {
    let bytes = bincode::serialize(msg)
        .map_err(|e| HandshakeError::Serialization(e.to_string()))?;

    if bytes.len() > MAX_HANDSHAKE_SIZE {
        return Err(HandshakeError::MessageTooLarge {
            size: bytes.len(),
            max: MAX_HANDSHAKE_SIZE,
        });
    }

    let len = bytes.len() as u32;
    writer.write_all(&len.to_be_bytes()).await?;
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

/// Reads a handshake message using 4-byte BE length prefix + bincode.
async fn read_handshake_message<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<HandshakeMessage, HandshakeError> {
    // Read 4-byte length prefix
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(HandshakeError::ConnectionClosed);
        }
        Err(e) => return Err(HandshakeError::Io(e)),
    }

    let msg_len = u32::from_be_bytes(len_buf) as usize;
    if msg_len > MAX_HANDSHAKE_SIZE {
        return Err(HandshakeError::MessageTooLarge {
            size: msg_len,
            max: MAX_HANDSHAKE_SIZE,
        });
    }

    // Read message body
    let mut body = vec![0u8; msg_len];
    match reader.read_exact(&mut body).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(HandshakeError::ConnectionClosed);
        }
        Err(e) => return Err(HandshakeError::Io(e)),
    }

    bincode::deserialize(&body)
        .map_err(|e| HandshakeError::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    fn test_caps() -> NodeCapabilities {
        NodeCapabilities {
            can_train: true,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 4096,
            cpu_cores: 8,
            storage_gb: 100,
        }
    }

    #[tokio::test]
    async fn test_handshake_success() {
        let (mut client_stream, mut server_stream) = duplex(64 * 1024);

        let client_id = PeerId::random();
        let server_id = PeerId::random();
        let client_caps = test_caps();
        let server_caps = NodeCapabilities {
            can_aggregate: true,
            ..test_caps()
        };

        let client = tokio::spawn({
            let client_id = client_id.clone();
            let client_caps = client_caps.clone();
            async move {
                perform_handshake_outbound(
                    &mut client_stream,
                    &client_id,
                    NodeRole::Compute,
                    &client_caps,
                    "127.0.0.1:9001",
                    None,
                    HANDSHAKE_TIMEOUT,
                    None,
                    None,
                ).await
            }
        });

        let server = tokio::spawn({
            let server_id = server_id.clone();
            let server_caps = server_caps.clone();
            async move {
                perform_handshake_inbound(
                    &mut server_stream,
                    &server_id,
                    NodeRole::Aggregator,
                    &server_caps,
                    "127.0.0.1:9000",
                    None,
                    HANDSHAKE_TIMEOUT,
                    50,
                    0,
                    None,
                    None,
                ).await
            }
        });

        let (client_result, server_result) = tokio::join!(client, server);
        let client_result = client_result.unwrap().unwrap();
        let server_result = server_result.unwrap().unwrap();

        // Client should see server's info
        assert_eq!(client_result.peer_id, server_id);
        assert_eq!(client_result.role, NodeRole::Aggregator);
        assert!(client_result.capabilities.can_aggregate);

        // Server should see client's info
        assert_eq!(server_result.peer_id, client_id);
        assert_eq!(server_result.role, NodeRole::Compute);
        assert!(server_result.capabilities.can_train);
    }

    #[tokio::test]
    async fn test_handshake_with_public_keys() {
        let (mut client_stream, mut server_stream) = duplex(64 * 1024);

        let client_pk = vec![1u8; 32];
        let server_pk = vec![2u8; 32];

        let client = tokio::spawn({
            let pk = client_pk.clone();
            async move {
                perform_handshake_outbound(
                    &mut client_stream,
                    &PeerId::random(),
                    NodeRole::Compute,
                    &test_caps(),
                    "127.0.0.1:9001",
                    Some(&pk),
                    HANDSHAKE_TIMEOUT,
                    None,
                    None,
                ).await
            }
        });

        let server = tokio::spawn({
            let pk = server_pk.clone();
            async move {
                perform_handshake_inbound(
                    &mut server_stream,
                    &PeerId::random(),
                    NodeRole::Aggregator,
                    &test_caps(),
                    "127.0.0.1:9000",
                    Some(&pk),
                    HANDSHAKE_TIMEOUT,
                    50,
                    0,
                    None,
                    None,
                ).await
            }
        });

        let (client_result, server_result) = tokio::join!(client, server);
        let client_result = client_result.unwrap().unwrap();
        let server_result = server_result.unwrap().unwrap();

        assert_eq!(client_result.public_key.unwrap(), server_pk);
        assert_eq!(server_result.public_key.unwrap(), client_pk);
    }

    #[tokio::test]
    async fn test_handshake_reject_at_capacity() {
        let (mut client_stream, mut server_stream) = duplex(64 * 1024);

        let client = tokio::spawn(async move {
            perform_handshake_outbound(
                &mut client_stream,
                &PeerId::random(),
                NodeRole::Compute,
                &test_caps(),
                "127.0.0.1:9001",
                None,
                HANDSHAKE_TIMEOUT,
                None,
                None,
            ).await
        });

        let server = tokio::spawn(async move {
            // max_peers=5, current_peers=5 → at capacity
            perform_handshake_inbound(
                &mut server_stream,
                &PeerId::random(),
                NodeRole::Aggregator,
                &test_caps(),
                "127.0.0.1:9000",
                None,
                HANDSHAKE_TIMEOUT,
                5,
                5,
                None,
                None,
            ).await
        });

        let (client_result, server_result) = tokio::join!(client, server);
        assert!(matches!(
            client_result.unwrap(),
            Err(HandshakeError::Rejected(_))
        ));
        assert!(matches!(
            server_result.unwrap(),
            Err(HandshakeError::Rejected(_))
        ));
    }

    #[tokio::test]
    async fn test_handshake_timeout() {
        let (mut client_stream, _server_stream) = duplex(64 * 1024);

        // Server never responds, so client should time out
        let result = perform_handshake_outbound(
            &mut client_stream,
            &PeerId::random(),
            NodeRole::Compute,
            &test_caps(),
            "127.0.0.1:9001",
            None,
            Duration::from_millis(100),
            None,
            None,
        ).await;

        assert!(matches!(result, Err(HandshakeError::Timeout(_))));
    }

    #[tokio::test]
    async fn test_handshake_connection_closed() {
        let (mut client_stream, server_stream) = duplex(64 * 1024);

        // Drop server side immediately
        drop(server_stream);

        let result = perform_handshake_outbound(
            &mut client_stream,
            &PeerId::random(),
            NodeRole::Compute,
            &test_caps(),
            "127.0.0.1:9001",
            None,
            HANDSHAKE_TIMEOUT,
            None,
            None,
        ).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handshake_message_roundtrip() {
        let (mut writer, mut reader) = duplex(64 * 1024);

        let msg = HandshakeMessage::Init {
            peer_id: PeerId::from_string("test-peer"),
            protocol_version: HANDSHAKE_VERSION,
            role: NodeRole::Compute,
            capabilities: test_caps(),
            listen_addr: "127.0.0.1:9000".to_string(),
            public_key: Some(vec![42u8; 32]),
            nonce: [0xAB; 32],
            x25519_pubkey: Some(vec![0xCC; 32]),
            eth_address: Some(vec![0xDD; 20]),
        };

        write_handshake_message(&mut writer, &msg).await.unwrap();
        drop(writer); // Close write end so reader sees EOF after message

        let received = read_handshake_message(&mut reader).await.unwrap();

        match received {
            HandshakeMessage::Init { peer_id, nonce, public_key, x25519_pubkey, eth_address, .. } => {
                assert_eq!(peer_id, PeerId::from_string("test-peer"));
                assert_eq!(nonce, [0xAB; 32]);
                assert_eq!(public_key.unwrap(), vec![42u8; 32]);
                assert_eq!(x25519_pubkey.unwrap(), vec![0xCC; 32]);
                assert_eq!(eth_address.unwrap(), vec![0xDD; 20]);
            }
            _ => panic!("Expected Init message"),
        }
    }
}
