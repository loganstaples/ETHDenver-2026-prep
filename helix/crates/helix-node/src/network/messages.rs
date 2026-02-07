//! Message Types for P2P Communication.
//!
//! Defines all message types exchanged between HELIX network nodes.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Unique identifier for a peer.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeerId(pub String);

impl PeerId {
    /// Creates a new random peer ID.
    pub fn random() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Creates a peer ID from a string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Network message envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    /// Message ID.
    pub id: String,
    /// Sender peer ID.
    pub sender: PeerId,
    /// Message timestamp.
    pub timestamp: u64,
    /// Message payload.
    pub payload: MessagePayload,
    /// Hop count (for gossip limiting).
    pub hops: u8,
    /// Signature (optional).
    pub signature: Option<Vec<u8>>,
}

impl NetworkMessage {
    /// Creates a new message.
    pub fn new(sender: PeerId, payload: MessagePayload) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            sender,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            payload,
            hops: 0,
            signature: None,
        }
    }

    /// Increments the hop count.
    pub fn increment_hops(&mut self) {
        self.hops = self.hops.saturating_add(1);
    }

    /// Computes the signing hash for this message (SHA-256 of canonical fields).
    fn signing_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.sender.0.as_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        // Serialize payload deterministically with bincode
        if let Ok(payload_bytes) = bincode::serialize(&self.payload) {
            hasher.update(&payload_bytes);
        }
        hasher.update([self.hops]);
        hasher.finalize().into()
    }

    /// Signs this message with an ed25519 signing key.
    #[cfg(feature = "crypto-sign")]
    pub fn sign(&mut self, key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;
        let hash = self.signing_hash();
        let sig = key.sign(&hash);
        self.signature = Some(sig.to_bytes().to_vec());
    }

    /// Verifies this message's signature against a verifying key.
    #[cfg(feature = "crypto-sign")]
    pub fn verify_signature(&self, vk: &ed25519_dalek::VerifyingKey) -> bool {
        use ed25519_dalek::{Signature, Verifier};
        let sig_bytes = match &self.signature {
            Some(s) => s,
            None => return false,
        };
        let sig = match Signature::from_slice(sig_bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let hash = self.signing_hash();
        vk.verify(&hash, &sig).is_ok()
    }

    /// Stub sign when crypto-sign is disabled (no-op).
    #[cfg(not(feature = "crypto-sign"))]
    pub fn sign_noop(&mut self) {
        // No-op: signing not available without crypto-sign feature
    }

    /// Stub verify when crypto-sign is disabled (always true).
    #[cfg(not(feature = "crypto-sign"))]
    pub fn verify_signature_noop(&self) -> bool {
        true
    }
}

/// Message payload types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Peer discovery messages.
    Discovery(DiscoveryMessage),
    /// Training coordination messages.
    Training(TrainingMessage),
    /// Gradient exchange messages.
    Gradient(GradientMessage),
    /// State synchronization messages.
    Sync(SyncMessage),
    /// Heartbeat/ping messages.
    Heartbeat(HeartbeatMessage),
}

/// Peer discovery messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryMessage {
    /// Request to join the network.
    JoinRequest {
        /// Node capabilities.
        capabilities: NodeCapabilities,
        /// Listen address.
        listen_addr: String,
    },
    /// Response to join request.
    JoinResponse {
        /// Whether join was accepted.
        accepted: bool,
        /// List of known peers.
        peers: Vec<PeerInfo>,
    },
    /// Announce peer presence.
    Announce {
        /// Peer info.
        peer: PeerInfo,
    },
    /// Request for known peers.
    GetPeers,
    /// Response with known peers.
    Peers {
        /// List of peers.
        peers: Vec<PeerInfo>,
    },
    /// Peer is leaving.
    Leave,
}

/// Information about a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Peer ID.
    pub id: PeerId,
    /// Network address.
    pub address: String,
    /// Node capabilities.
    pub capabilities: NodeCapabilities,
    /// Last seen timestamp.
    pub last_seen: u64,
    /// Reputation score.
    pub reputation: i32,
}

/// Node capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Can perform training computations.
    pub can_train: bool,
    /// Can aggregate gradients.
    pub can_aggregate: bool,
    /// Can generate proofs.
    pub can_prove: bool,
    /// Available GPU memory (MB).
    pub gpu_memory_mb: u32,
    /// Available CPU cores.
    pub cpu_cores: u32,
    /// Available disk storage (GB).
    pub storage_gb: u32,
}

/// Training coordination messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingMessage {
    /// Start a new training round.
    RoundStart {
        /// Round ID.
        round_id: u64,
        /// Model hash.
        model_hash: [u8; 32],
        /// Training parameters.
        params: TrainingParams,
    },
    /// Report training completion.
    RoundComplete {
        /// Round ID.
        round_id: u64,
        /// Result hash.
        result_hash: [u8; 32],
    },
    /// Request to participate in training.
    ParticipateRequest {
        /// Round ID.
        round_id: u64,
    },
    /// Response to participation request.
    ParticipateResponse {
        /// Round ID.
        round_id: u64,
        /// Whether accepted.
        accepted: bool,
        /// Assigned data shard.
        shard_id: Option<u32>,
    },
}

/// Training parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: u32,
    /// Number of local epochs.
    pub local_epochs: u32,
    /// Maximum error bound.
    pub max_error_bound: f64,
}

/// Gradient exchange messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientMessage {
    /// Share computed gradients.
    ShareGradient {
        /// Round ID.
        round_id: u64,
        /// Gradient commitment.
        gradient_commitment: [u8; 32],
        /// Error bound.
        error_bound: f64,
        /// Proof of computation.
        proof: Vec<u8>,
    },
    /// Request gradients from a peer.
    RequestGradient {
        /// Round ID.
        round_id: u64,
    },
    /// Full gradient data.
    GradientData {
        /// Round ID.
        round_id: u64,
        /// Serialized gradients.
        data: Vec<u8>,
    },
    /// Aggregated gradient result.
    AggregatedGradient {
        /// Round ID.
        round_id: u64,
        /// Aggregated commitment.
        commitment: [u8; 32],
        /// Aggregation proof.
        proof: Vec<u8>,
    },
}

/// State synchronization messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMessage {
    /// Request current state.
    GetState,
    /// State response.
    State {
        /// Latest round.
        latest_round: u64,
        /// Model state hash.
        model_hash: [u8; 32],
        /// Accumulated error.
        accumulated_error: f64,
    },
    /// Request specific block/checkpoint.
    GetCheckpoint {
        /// Round number.
        round: u64,
    },
    /// Checkpoint data.
    Checkpoint {
        /// Round number.
        round: u64,
        /// Checkpoint data.
        data: Vec<u8>,
    },
}

/// Heartbeat/ping messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    /// Sequence number.
    pub seq: u64,
    /// Is this a pong (response)?
    pub is_pong: bool,
    /// Current load (0-100).
    pub load: u8,
}

/// Serializes a message to bytes using bincode.
pub fn serialize_message(msg: &NetworkMessage) -> Result<Vec<u8>, Box<bincode::ErrorKind>> {
    bincode::serialize(msg)
}

/// Deserializes a message from bytes using bincode.
pub fn deserialize_message(data: &[u8]) -> Result<NetworkMessage, Box<bincode::ErrorKind>> {
    bincode::deserialize(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        );

        let bytes = serialize_message(&msg).unwrap();
        let decoded = deserialize_message(&bytes).unwrap();

        assert_eq!(msg.id, decoded.id);
    }

    #[test]
    fn test_peer_id() {
        let id = PeerId::random();
        assert!(!id.0.is_empty());
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_sign_verify_roundtrip() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let mut msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        );

        msg.sign(&signing_key);
        assert!(msg.signature.is_some());
        assert!(msg.verify_signature(&verifying_key));
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_tampered_payload_rejected() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let mut msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        );

        msg.sign(&signing_key);

        // Tamper with payload
        msg.payload = MessagePayload::Heartbeat(HeartbeatMessage {
            seq: 999,
            is_pong: true,
            load: 0,
        });

        assert!(!msg.verify_signature(&verifying_key));
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_wrong_key_rejected() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let wrong_key = SigningKey::generate(&mut OsRng);
        let wrong_vk = wrong_key.verifying_key();

        let mut msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        );

        msg.sign(&signing_key);
        assert!(!msg.verify_signature(&wrong_vk));
    }

    #[cfg(not(feature = "crypto-sign"))]
    #[test]
    fn test_noop_signing() {
        let mut msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        );

        msg.sign_noop();
        assert!(msg.verify_signature_noop());
    }
}
