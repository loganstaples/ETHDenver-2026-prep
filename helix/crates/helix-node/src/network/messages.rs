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
    ///
    /// # Security trade-off
    ///
    /// When the `crypto-sign` feature is disabled, **all messages are accepted
    /// without signature verification**. This means any peer can impersonate
    /// any other peer by forging the `sender` field. This mode is only suitable
    /// for development/testing or trusted private networks. In production, the
    /// `crypto-sign` feature MUST be enabled to enforce message authentication.
    #[cfg(not(feature = "crypto-sign"))]
    pub fn verify_signature_noop(&self) -> bool {
        true
    }
}

/// Registry mapping peer IDs to their public verification keys.
///
/// When `crypto-sign` is enabled, every incoming message is verified against
/// the sender's registered public key. When disabled, verification is a no-op
/// (see `verify_signature_noop` above for the security implications).
pub struct PeerKeyRegistry {
    #[cfg(feature = "crypto-sign")]
    keys: std::collections::HashMap<PeerId, ed25519_dalek::VerifyingKey>,
    #[cfg(not(feature = "crypto-sign"))]
    _phantom: (),
}

impl PeerKeyRegistry {
    /// Creates a new empty peer key registry.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "crypto-sign")]
            keys: std::collections::HashMap::new(),
            #[cfg(not(feature = "crypto-sign"))]
            _phantom: (),
        }
    }

    /// Registers a peer's public verification key.
    #[cfg(feature = "crypto-sign")]
    pub fn register(&mut self, peer_id: PeerId, key: ed25519_dalek::VerifyingKey) {
        self.keys.insert(peer_id, key);
    }

    /// Registers a peer's public verification key (no-op without crypto-sign).
    #[cfg(not(feature = "crypto-sign"))]
    pub fn register(&mut self, _peer_id: PeerId, _key_bytes: &[u8]) {
        // No-op: signing not available without crypto-sign feature
    }

    /// Verifies a message's signature against the sender's registered key.
    ///
    /// Returns `true` if the signature is valid, or if `crypto-sign` is
    /// disabled (all messages pass in that case — see security note above).
    /// Returns `false` if the sender has no registered key or the signature
    /// is missing/invalid.
    #[cfg(feature = "crypto-sign")]
    pub fn verify_message(&self, message: &NetworkMessage) -> bool {
        match self.keys.get(&message.sender) {
            Some(vk) => message.verify_signature(vk),
            // If we don't have the key yet (e.g. first JoinRequest), allow it
            // only if it has no signature (unsigned discovery messages are OK)
            None => message.signature.is_none(),
        }
    }

    /// Verifies a message's signature (no-op without crypto-sign: always true).
    ///
    /// # Security trade-off
    ///
    /// Without the `crypto-sign` feature, this always returns `true`. This
    /// means that **no message authentication is performed**, and any peer can
    /// forge messages from any other peer. Enable `crypto-sign` for production.
    #[cfg(not(feature = "crypto-sign"))]
    pub fn verify_message(&self, _message: &NetworkMessage) -> bool {
        true
    }

    /// Returns whether any keys are registered (always false without crypto-sign).
    #[cfg(feature = "crypto-sign")]
    pub fn has_key(&self, peer_id: &PeerId) -> bool {
        self.keys.contains_key(peer_id)
    }

    #[cfg(not(feature = "crypto-sign"))]
    pub fn has_key(&self, _peer_id: &PeerId) -> bool {
        false
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
    /// BFT consensus messages (2-phase commit for gradient aggregation).
    Consensus(ConsensusMessage),
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
    /// Model input dimension.
    pub d_in: usize,
    /// Model hidden dimension.
    pub d_hid: usize,
    /// Model output dimension.
    pub d_out: usize,
    /// Random seed for deterministic model init.
    pub model_seed: u64,
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

/// BFT consensus messages for 2-phase commit gradient aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Phase 1: Leader proposes aggregated gradient commitment.
    Propose {
        /// Round ID this proposal is for.
        round_id: u64,
        /// SHA-256 of sorted accepted gradient commitments.
        aggregated_commitment: [u8; 32],
        /// Cryptographic binding commitment from the proposer (H(proposal || proposer_id || nonce)).
        proposer_binding: [u8; 32],
        /// Number of individual gradients included.
        num_gradients: usize,
        /// Combined error bound for this aggregation.
        error_bound: f64,
        /// Nonce used in proposer_binding.
        nonce: [u8; 16],
    },
    /// Phase 2: Participant votes on the proposal.
    Vote {
        /// Round ID this vote is for.
        round_id: u64,
        /// Whether this participant accepts the proposal.
        accept: bool,
        /// Voter's independently computed commitment (should match proposal).
        voter_commitment: [u8; 32],
        /// Reason for rejection (if accept is false).
        reason: Option<String>,
    },
    /// Decision: Quorum reached, commit the aggregated gradient.
    Commit {
        /// Round ID.
        round_id: u64,
        /// Final agreed-upon commitment.
        final_commitment: [u8; 32],
        /// Number of votes in favor (>= 2f+1).
        votes_for: usize,
        /// Total participants in this consensus round.
        total_participants: usize,
    },
    /// Decision: Consensus failed, abort this round.
    Abort {
        /// Round ID.
        round_id: u64,
        /// Reason for abort.
        reason: String,
    },
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

    #[test]
    fn test_peer_key_registry_without_crypto() {
        let registry = PeerKeyRegistry::new();
        let msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        );

        // Without crypto-sign, all messages pass
        assert!(registry.verify_message(&msg));
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_peer_key_registry_verify() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let peer_id = PeerId::random();
        let mut registry = PeerKeyRegistry::new();
        registry.register(peer_id.clone(), verifying_key);

        let mut msg = NetworkMessage::new(
            peer_id.clone(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        );

        msg.sign(&signing_key);
        assert!(registry.verify_message(&msg));

        // Tamper
        msg.payload = MessagePayload::Heartbeat(HeartbeatMessage {
            seq: 999,
            is_pong: true,
            load: 0,
        });
        assert!(!registry.verify_message(&msg));
    }
}
