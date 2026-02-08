//! Coordination types for distributed training.
//!
//! Shared types that all HELIX crates import for distributed coordination:
//! node identity, session management, round descriptors, and training receipts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Unique node identity wrapping an ed25519 public key (32 bytes).
///
/// Every participant in the HELIX network is identified by their ed25519 public
/// key. This provides a cryptographic identity that can be used for message
/// signing and verification.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    /// Creates a NodeId from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the raw bytes of this node ID.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Returns a short hex representation (first 8 hex chars).
    pub fn short_hex(&self) -> String {
        hex_encode(&self.0[..4])
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({}..)", self.short_hex())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex_encode(&self.0))
    }
}

/// Unique identifier for a training session.
///
/// A session groups multiple training rounds for a single model training run.
/// The session ID is derived from the model ID, start time, and initiator.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub [u8; 32]);

impl SessionId {
    /// Creates a SessionId from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Derives a session ID from its components.
    pub fn derive(model_id: &[u8; 32], start_time: u64, initiator: &NodeId) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(model_id);
        hasher.update(start_time.to_le_bytes());
        hasher.update(initiator.as_bytes());
        let result = hasher.finalize();
        let mut id = [0u8; 32];
        id.copy_from_slice(&result);
        Self(id)
    }

    /// Returns the raw bytes of this session ID.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Returns a short hex representation (first 8 hex chars).
    pub fn short_hex(&self) -> String {
        hex_encode(&self.0[..4])
    }
}

impl fmt::Debug for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SessionId({}..)", self.short_hex())
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex_encode(&self.0))
    }
}

/// Training hyperparameters shared across participants in a round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Learning rate for this round.
    pub learning_rate: f64,
    /// Batch size each participant must use.
    pub batch_size: usize,
    /// Maximum error bound allowed per step.
    pub max_error_per_step: f64,
    /// Total error budget for the round.
    pub error_budget: f64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            max_error_per_step: 0.001,
            error_budget: 0.01,
        }
    }
}

/// Describes a single training round in a session.
///
/// A round is a coordinated training step where multiple participants
/// compute gradients on their local data shards and submit ZK proofs
/// of correct computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundDescriptor {
    /// Model being trained (32-byte identifier, typically a hash).
    pub model_id: [u8; 32],
    /// Round number within the session (0-indexed).
    pub round_number: u64,
    /// Participants expected to contribute to this round.
    pub participants: Vec<NodeId>,
    /// Unix timestamp deadline for proof submission.
    pub deadline: u64,
    /// Training parameters for this round.
    pub params: TrainingParams,
    /// Session this round belongs to.
    pub session_id: SessionId,
    /// Current model weight commitment (hash of weights before this round).
    pub weight_commitment: [u8; 32],
}

impl RoundDescriptor {
    /// Returns a deterministic round identifier derived from session and round number.
    pub fn round_id(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.session_id.as_bytes());
        hasher.update(self.round_number.to_le_bytes());
        let result = hasher.finalize();
        let mut id = [0u8; 32];
        id.copy_from_slice(&result);
        id
    }

    /// Checks whether a given node is a participant in this round.
    pub fn is_participant(&self, node_id: &NodeId) -> bool {
        self.participants.contains(node_id)
    }

    /// Returns the number of participants.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }
}

/// A cryptographic commitment to a gradient update.
///
/// This binds the gradient computation to the specific round, participant,
/// and model state, enabling on-chain verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientCommitment {
    /// Hash of the gradient tensor.
    pub gradient_hash: [u8; 32],
    /// Node that computed this gradient.
    pub node_id: NodeId,
    /// Round this gradient belongs to.
    pub round_number: u64,
    /// Weight hash before applying this gradient.
    pub old_weight_hash: [u8; 32],
    /// Weight hash after applying this gradient.
    pub new_weight_hash: [u8; 32],
    /// Error bound for this gradient computation.
    pub error_bound: f64,
}

impl GradientCommitment {
    /// Computes a binding commitment hash over all fields.
    pub fn commitment_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.gradient_hash);
        hasher.update(self.node_id.as_bytes());
        hasher.update(self.round_number.to_le_bytes());
        hasher.update(self.old_weight_hash);
        hasher.update(self.new_weight_hash);
        hasher.update(self.error_bound.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// A receipt proving that a training step was correctly executed.
///
/// Issued after a ZK proof is verified (either locally or on-chain).
/// Can be used to claim rewards or as evidence of participation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingStepReceipt {
    /// The round this receipt covers.
    pub round_number: u64,
    /// The node that performed the training step.
    pub node_id: NodeId,
    /// Session this receipt belongs to.
    pub session_id: SessionId,
    /// Hash of the ZK proof that was verified.
    pub proof_hash: [u8; 32],
    /// Gradient commitment for this step.
    pub gradient_commitment: GradientCommitment,
    /// Error commitment checksum (compact, matches on-chain).
    pub error_checksum: u64,
    /// Unix timestamp when the proof was verified.
    pub verified_at: u64,
    /// On-chain transaction hash, if submitted (None for local-only verification).
    pub tx_hash: Option<[u8; 32]>,
}

impl TrainingStepReceipt {
    /// Returns a deterministic receipt ID.
    pub fn receipt_id(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.session_id.as_bytes());
        hasher.update(self.round_number.to_le_bytes());
        hasher.update(self.node_id.as_bytes());
        hasher.update(self.proof_hash);
        let result = hasher.finalize();
        let mut id = [0u8; 32];
        id.copy_from_slice(&result);
        id
    }
}

/// Encodes bytes as lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node_id(seed: u8) -> NodeId {
        NodeId::from_bytes([seed; 32])
    }

    #[test]
    fn test_node_id_display() {
        let node = test_node_id(0xab);
        assert_eq!(node.short_hex(), "abababab");
        assert_eq!(format!("{:?}", node), "NodeId(abababab..)");
        assert_eq!(node.to_string().len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_node_id_equality() {
        let a = test_node_id(1);
        let b = test_node_id(1);
        let c = test_node_id(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_session_id_derive_deterministic() {
        let model_id = [1u8; 32];
        let initiator = test_node_id(42);
        let s1 = SessionId::derive(&model_id, 1000, &initiator);
        let s2 = SessionId::derive(&model_id, 1000, &initiator);
        assert_eq!(s1, s2);

        // Different inputs produce different session IDs
        let s3 = SessionId::derive(&model_id, 1001, &initiator);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_round_descriptor() {
        let node_a = test_node_id(1);
        let node_b = test_node_id(2);
        let session = SessionId::from_bytes([10u8; 32]);

        let round = RoundDescriptor {
            model_id: [0u8; 32],
            round_number: 5,
            participants: vec![node_a, node_b],
            deadline: 1700000000,
            params: TrainingParams::default(),
            session_id: session,
            weight_commitment: [0u8; 32],
        };

        assert_eq!(round.participant_count(), 2);
        assert!(round.is_participant(&node_a));
        assert!(!round.is_participant(&test_node_id(99)));

        // Round ID is deterministic
        let id1 = round.round_id();
        let id2 = round.round_id();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_gradient_commitment_hash_deterministic() {
        let gc = GradientCommitment {
            gradient_hash: [1u8; 32],
            node_id: test_node_id(1),
            round_number: 0,
            old_weight_hash: [2u8; 32],
            new_weight_hash: [3u8; 32],
            error_bound: 0.001,
        };

        let h1 = gc.commitment_hash();
        let h2 = gc.commitment_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_gradient_commitment_hash_changes_with_inputs() {
        let base = GradientCommitment {
            gradient_hash: [1u8; 32],
            node_id: test_node_id(1),
            round_number: 0,
            old_weight_hash: [2u8; 32],
            new_weight_hash: [3u8; 32],
            error_bound: 0.001,
        };
        let different = GradientCommitment {
            round_number: 1,
            ..base.clone()
        };
        assert_ne!(base.commitment_hash(), different.commitment_hash());
    }

    #[test]
    fn test_training_step_receipt() {
        let node = test_node_id(1);
        let session = SessionId::from_bytes([10u8; 32]);

        let receipt = TrainingStepReceipt {
            round_number: 5,
            node_id: node,
            session_id: session,
            proof_hash: [0xaa; 32],
            gradient_commitment: GradientCommitment {
                gradient_hash: [1u8; 32],
                node_id: node,
                round_number: 5,
                old_weight_hash: [2u8; 32],
                new_weight_hash: [3u8; 32],
                error_bound: 0.001,
            },
            error_checksum: 12345,
            verified_at: 1700000000,
            tx_hash: None,
        };

        let id1 = receipt.receipt_id();
        let id2 = receipt.receipt_id();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_training_params_default() {
        let params = TrainingParams::default();
        assert!((params.learning_rate - 0.001).abs() < 1e-15);
        assert_eq!(params.batch_size, 32);
    }
}
