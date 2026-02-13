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

/// Maximum age (in seconds) for a message to be accepted.
/// Messages older than this are considered stale and rejected.
pub const MESSAGE_MAX_AGE_SECS: u64 = 300; // 5 minutes

/// Maximum clock skew tolerance (in seconds).
/// Messages from slightly in the future (within this margin) are still accepted.
pub const MESSAGE_CLOCK_SKEW_SECS: u64 = 30;

/// Network message envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    /// Message ID.
    pub id: String,
    /// Sender peer ID.
    pub sender: PeerId,
    /// Message timestamp (Unix seconds).
    pub timestamp: u64,
    /// Cryptographic nonce for replay protection (16 random bytes).
    pub nonce: [u8; 16],
    /// Message payload.
    pub payload: MessagePayload,
    /// Hop count (for gossip limiting).
    pub hops: u8,
    /// Signature (optional).
    pub signature: Option<Vec<u8>>,
}

impl NetworkMessage {
    /// Creates a new message with a random nonce.
    pub fn new(sender: PeerId, payload: MessagePayload) -> Self {
        let mut nonce = [0u8; 16];
        // Use OsRng for cryptographic nonce to prevent prediction
        use rand::RngCore;
        rand::rngs::OsRng.fill_bytes(&mut nonce);

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            sender,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            nonce,
            payload,
            hops: 0,
            signature: None,
        }
    }

    /// Increments the hop count.
    pub fn increment_hops(&mut self) {
        self.hops = self.hops.saturating_add(1);
    }

    /// Returns the message age in seconds relative to the given current time.
    /// Returns `None` if the message timestamp is in the future beyond clock skew.
    pub fn age_secs(&self, now_secs: u64) -> Option<u64> {
        if self.timestamp > now_secs + MESSAGE_CLOCK_SKEW_SECS {
            None // too far in the future
        } else if self.timestamp > now_secs {
            Some(0) // within clock skew tolerance
        } else {
            Some(now_secs - self.timestamp)
        }
    }

    /// Checks if this message is stale (too old to accept).
    pub fn is_stale(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        match self.age_secs(now) {
            Some(age) => age > MESSAGE_MAX_AGE_SECS,
            None => true, // future message beyond clock skew
        }
    }

    /// Computes the signing hash for this message (SHA-256 of canonical fields).
    /// Includes all fields except the signature itself to prevent malleability.
    fn signing_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.sender.0.as_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.nonce);
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

    /// Registers a peer's key from raw bytes (32-byte ed25519 public key).
    /// Returns true if registration succeeded.
    #[cfg(feature = "crypto-sign")]
    pub fn register_from_bytes(&mut self, peer_id: PeerId, key_bytes: &[u8]) -> bool {
        if key_bytes.len() != 32 {
            return false;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(key_bytes);
        match ed25519_dalek::VerifyingKey::from_bytes(&arr) {
            Ok(vk) => {
                self.keys.insert(peer_id, vk);
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(not(feature = "crypto-sign"))]
    pub fn register_from_bytes(&mut self, _peer_id: PeerId, _key_bytes: &[u8]) -> bool {
        true // Always succeed when crypto-sign is disabled
    }

    /// Returns the number of registered keys.
    #[cfg(feature = "crypto-sign")]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    #[cfg(not(feature = "crypto-sign"))]
    pub fn key_count(&self) -> usize {
        0
    }
}

/// Message deduplication and replay detection cache.
///
/// Tracks recently seen message IDs + nonces to reject replayed messages.
/// Also enforces timestamp freshness (staleness check).
pub struct MessageDedup {
    /// Set of recently seen message IDs.
    seen_ids: std::collections::HashSet<String>,
    /// Maximum entries before eviction.
    max_entries: usize,
    /// Whether to enforce timestamp freshness.
    enforce_freshness: bool,
}

impl MessageDedup {
    /// Creates a new dedup cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            seen_ids: std::collections::HashSet::with_capacity(max_entries),
            max_entries,
            enforce_freshness: true,
        }
    }

    /// Creates a dedup cache that doesn't enforce freshness (for testing).
    pub fn permissive(max_entries: usize) -> Self {
        Self {
            seen_ids: std::collections::HashSet::with_capacity(max_entries),
            max_entries,
            enforce_freshness: false,
        }
    }

    /// Checks if a message should be accepted (not a replay, not stale).
    ///
    /// Returns `Ok(())` if the message is fresh and not a duplicate.
    /// Returns `Err(reason)` if the message should be rejected.
    pub fn check(&mut self, msg: &NetworkMessage) -> Result<(), MessageRejectReason> {
        // Check staleness
        if self.enforce_freshness && msg.is_stale() {
            return Err(MessageRejectReason::Stale {
                timestamp: msg.timestamp,
            });
        }

        // Check for future messages
        if self.enforce_freshness {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            if msg.timestamp > now + MESSAGE_CLOCK_SKEW_SECS {
                return Err(MessageRejectReason::FutureTimestamp {
                    timestamp: msg.timestamp,
                    now,
                });
            }
        }

        // Build dedup key from message ID + nonce (nonce prevents ID reuse across restarts)
        let dedup_key = format!("{}:{}", msg.id, hex::encode(msg.nonce));

        // Check duplicate
        if self.seen_ids.contains(&dedup_key) {
            return Err(MessageRejectReason::Duplicate {
                message_id: msg.id.clone(),
            });
        }

        // Evict if at capacity (simple clear strategy)
        if self.seen_ids.len() >= self.max_entries {
            self.seen_ids.clear();
        }

        self.seen_ids.insert(dedup_key);
        Ok(())
    }

    /// Number of tracked messages.
    pub fn len(&self) -> usize {
        self.seen_ids.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seen_ids.is_empty()
    }
}

impl Default for MessageDedup {
    fn default() -> Self {
        Self::new(10_000)
    }
}

/// Reason a message was rejected by the dedup/freshness check.
#[derive(Debug, Clone)]
pub enum MessageRejectReason {
    /// Message timestamp is too old.
    Stale { timestamp: u64 },
    /// Message timestamp is too far in the future.
    FutureTimestamp { timestamp: u64, now: u64 },
    /// Message ID + nonce was already seen.
    Duplicate { message_id: String },
}

impl std::fmt::Display for MessageRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stale { timestamp } => write!(f, "stale message (timestamp={})", timestamp),
            Self::FutureTimestamp { timestamp, now } => {
                write!(f, "future message (timestamp={}, now={})", timestamp, now)
            }
            Self::Duplicate { message_id } => write!(f, "duplicate message (id={})", message_id),
        }
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
    /// MPC protocol messages (Beaver triples, secret sharing, etc.)
    MpcData(MpcDataMessage),
    /// Worker registration messages.
    Registration(RegistrationMessage),
    /// Round management messages (configuration, proof submission, completion).
    RoundManagement(RoundManagementMessage),
}

/// MPC data message carrying raw bytes for the MPC layer.
///
/// Used to tunnel MPC protocol messages (secret shares, Beaver triples,
/// garbled circuits, etc.) through the node's existing TCP/TLS connections.
/// The `session_id` enables multiplexing multiple concurrent MPC sessions
/// over the same peer connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpcDataMessage {
    /// Session ID for multiplexing multiple MPC sessions.
    pub session_id: String,
    /// Raw MPC payload bytes.
    pub data: Vec<u8>,
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
        /// ed25519 public key bytes (32 bytes) for identity verification.
        /// None when crypto-sign is disabled.
        #[serde(default)]
        public_key: Option<Vec<u8>>,
    },
    /// Response to join request.
    JoinResponse {
        /// Whether join was accepted.
        accepted: bool,
        /// List of known peers.
        peers: Vec<PeerInfo>,
        /// Reason for rejection (if not accepted).
        #[serde(default)]
        reject_reason: Option<String>,
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
    /// Aggregator distributes model weights to workers at round start.
    ///
    /// The `checkpoint_data` contains a serialized `ModelCheckpoint` (via `to_bytes()`).
    /// Workers deserialize this to reconstruct the `MlpModel` for training.
    ModelWeights {
        /// Round ID this model is for.
        round_id: u64,
        /// Serialized ModelCheckpoint bytes.
        checkpoint_data: Vec<u8>,
        /// SHA-256 of the model weights (for integrity verification).
        weight_hash: [u8; 32],
    },
    /// Aggregator distributes updated weights after aggregation.
    ///
    /// Sent after a round completes to give workers the new model state
    /// for the next round of training.
    UpdatedWeights {
        /// Round ID this update is from (the completed round).
        round_id: u64,
        /// Serialized ModelCheckpoint bytes with aggregated weights.
        checkpoint_data: Vec<u8>,
        /// New weight hash after aggregation.
        weight_hash: [u8; 32],
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
    /// Number of layers in the model.
    pub num_layers: u32,
    /// Activation function type (0=ReLU, 1=Sigmoid, 2=Tanh, 3=GeLU, 4=LeakyReLU).
    pub activation_type: u8,
}

/// Gradient exchange messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientMessage {
    /// Share computed gradients.
    ShareGradient {
        /// Round ID.
        round_id: u64,
        /// Gradient commitment (hiding: SHA-256(gradient || nonce)).
        gradient_commitment: [u8; 32],
        /// Random nonce used in the hiding commitment.
        /// Revealed in Phase 2 so peers can verify commitment correctness.
        commitment_nonce: [u8; 16],
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
    /// Worker sends updated model weights after local training.
    ///
    /// The `checkpoint_data` contains a serialized `ModelCheckpoint` of the worker's
    /// model state after training. The aggregator uses these to compute averaged weights.
    WeightUpdate {
        /// Round ID.
        round_id: u64,
        /// Serialized ModelCheckpoint bytes of the worker's updated model.
        checkpoint_data: Vec<u8>,
        /// SHA-256 of the updated weights.
        weight_hash: [u8; 32],
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

/// Worker registration messages for explicit aggregator enrollment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegistrationMessage {
    /// Worker requests to register with the aggregator.
    WorkerRegister {
        /// Worker's capabilities.
        capabilities: NodeCapabilities,
        /// Worker's listen address for direct communication.
        listen_addr: String,
        /// Optional public key bytes for identity verification.
        #[serde(default)]
        public_key: Option<Vec<u8>>,
        /// Maximum concurrent training steps this worker can handle.
        max_concurrent_steps: u32,
        /// Available memory in MB for model weights.
        available_memory_mb: u64,
    },
    /// Worker requests to deregister from the aggregator.
    WorkerDeregister {
        /// Reason for leaving (graceful shutdown, error, etc.).
        reason: String,
    },
    /// Aggregator confirms worker registration.
    WorkerRegistered {
        /// Whether registration was accepted.
        accepted: bool,
        /// Assigned worker ID within the aggregator's registry.
        worker_slot: Option<u32>,
        /// Rejection reason if not accepted.
        reject_reason: Option<String>,
        /// Current round ID if a round is in progress (so worker can catch up).
        current_round_id: Option<u64>,
    },
    /// Aggregator acknowledges worker deregistration.
    WorkerDeregistered {
        /// Confirmation that deregistration was processed.
        acknowledged: bool,
    },
}

/// Round management messages for configuration, proof collection, and completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoundManagementMessage {
    /// Aggregator configures a new training round and sends to all registered workers.
    RoundConfigure {
        /// Unique round identifier.
        round_id: u64,
        /// On-chain model ID.
        model_id: u64,
        /// Model architecture dimensions.
        model_dims: ModelDims,
        /// Reference to the training dataset (IPFS hash, URL, etc.).
        dataset_ref: String,
        /// Number of training steps each worker should perform.
        steps_per_worker: u32,
        /// Learning rate for this round.
        learning_rate: f64,
        /// Maximum error budget for this round.
        error_budget: f64,
        /// Minimum number of workers required to proceed.
        min_workers: u32,
        /// Round deadline (Unix timestamp in seconds).
        deadline: u64,
        /// SHA-256 hash of the current model weights.
        current_model_hash: [u8; 32],
    },
    /// Worker acknowledges round configuration and signals readiness.
    WorkerReady {
        /// Round ID being acknowledged.
        round_id: u64,
        /// Whether the worker is ready to proceed.
        ready: bool,
        /// Reason if not ready.
        reason: Option<String>,
    },
    /// Aggregator signals workers to begin training.
    BeginTraining {
        /// Round ID.
        round_id: u64,
        /// Final list of participating worker peer IDs.
        participants: Vec<String>,
    },
    /// Worker submits proof of completed training step(s).
    ProofSubmission {
        /// Round ID.
        round_id: u64,
        /// ZK proof bytes.
        proof: Vec<u8>,
        /// Public inputs for the proof (8 elements as U256 bytes).
        public_inputs: Vec<[u8; 32]>,
        /// Error bound achieved.
        error_bound: f64,
        /// Number of training steps completed.
        steps_completed: u32,
        /// SHA-256 hash of updated model weights after training.
        new_model_hash: [u8; 32],
        /// Serialized model checkpoint (for FedAvg aggregation).
        checkpoint_data: Vec<u8>,
    },
    /// Aggregator acknowledges proof receipt.
    ProofAcknowledged {
        /// Round ID.
        round_id: u64,
        /// Whether the proof passed local validation.
        valid: bool,
        /// Rejection reason if invalid.
        reason: Option<String>,
    },
    /// Aggregator broadcasts round completion with final commitment.
    RoundCompleted {
        /// Round ID.
        round_id: u64,
        /// Final aggregated model commitment hash.
        final_commitment: [u8; 32],
        /// On-chain transaction hash (if submitted).
        tx_hash: Option<[u8; 32]>,
        /// New model hash after aggregation.
        new_model_hash: [u8; 32],
        /// Total error bound for the round.
        total_error_bound: f64,
        /// Number of workers that contributed.
        num_contributors: u32,
    },
    /// Aggregator signals round failure.
    RoundFailed {
        /// Round ID.
        round_id: u64,
        /// Failure reason.
        reason: String,
    },
}

/// Model architecture dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDims {
    /// Input dimension.
    pub d_in: usize,
    /// Hidden dimension.
    pub d_hid: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Number of layers.
    pub num_layers: u32,
    /// Number of attention heads (0 for MLP-only).
    pub num_heads: u32,
    /// Activation function type (0=ReLU, 1=Sigmoid, 2=Tanh, 3=GeLU, 4=LeakyReLU).
    pub activation_type: u8,
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

    // ---- Nonce and replay protection tests ----

    #[test]
    fn test_message_has_nonce() {
        let msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );
        // Nonce should be non-zero (random)
        assert_ne!(msg.nonce, [0u8; 16]);
    }

    #[test]
    fn test_two_messages_have_different_nonces() {
        let sender = PeerId::random();
        let payload = MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 });
        let msg1 = NetworkMessage::new(sender.clone(), payload.clone());
        let msg2 = NetworkMessage::new(sender, payload);
        assert_ne!(msg1.nonce, msg2.nonce, "each message must have a unique nonce");
    }

    #[test]
    fn test_nonce_included_in_signing_hash() {
        let sender = PeerId::random();
        let payload = MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 });

        let mut msg1 = NetworkMessage::new(sender.clone(), payload.clone());
        let mut msg2 = msg1.clone();
        msg2.nonce = [0xFF; 16]; // different nonce

        let hash1 = msg1.signing_hash();
        let hash2 = msg2.signing_hash();
        assert_ne!(hash1, hash2, "different nonces must produce different signing hashes");
    }

    #[test]
    fn test_fresh_message_not_stale() {
        let msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );
        assert!(!msg.is_stale());
    }

    #[test]
    fn test_old_message_is_stale() {
        let mut msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );
        // Set timestamp to 10 minutes ago
        msg.timestamp = msg.timestamp.saturating_sub(MESSAGE_MAX_AGE_SECS + 60);
        assert!(msg.is_stale());
    }

    #[test]
    fn test_future_message_is_stale() {
        let mut msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );
        // Set timestamp 5 minutes in the future (beyond clock skew tolerance)
        msg.timestamp += MESSAGE_CLOCK_SKEW_SECS + 300;
        assert!(msg.is_stale());
    }

    #[test]
    fn test_dedup_rejects_replayed_message() {
        let mut dedup = MessageDedup::permissive(100);
        let msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );

        // First time: accepted
        assert!(dedup.check(&msg).is_ok());
        // Second time: rejected as duplicate
        let result = dedup.check(&msg);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MessageRejectReason::Duplicate { .. }));
    }

    #[test]
    fn test_dedup_accepts_different_messages() {
        let mut dedup = MessageDedup::permissive(100);
        let sender = PeerId::random();

        let msg1 = NetworkMessage::new(
            sender.clone(),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );
        let msg2 = NetworkMessage::new(
            sender,
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 2, is_pong: false, load: 50 }),
        );

        assert!(dedup.check(&msg1).is_ok());
        assert!(dedup.check(&msg2).is_ok());
    }

    #[test]
    fn test_dedup_rejects_stale_message() {
        let mut dedup = MessageDedup::new(100); // with freshness enforcement
        let mut msg = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );
        msg.timestamp = msg.timestamp.saturating_sub(MESSAGE_MAX_AGE_SECS + 60);

        let result = dedup.check(&msg);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MessageRejectReason::Stale { .. }));
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_replayed_signed_message_rejected() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let peer_id = PeerId::random();
        let mut registry = PeerKeyRegistry::new();
        registry.register(peer_id.clone(), verifying_key);

        let mut msg = NetworkMessage::new(
            peer_id,
            MessagePayload::Heartbeat(HeartbeatMessage { seq: 1, is_pong: false, load: 50 }),
        );
        msg.sign(&signing_key);

        // Signature is valid
        assert!(registry.verify_message(&msg));

        // But dedup rejects the replay
        let mut dedup = MessageDedup::permissive(100);
        assert!(dedup.check(&msg).is_ok());
        assert!(dedup.check(&msg).is_err(), "replayed message must be rejected");
    }
}
