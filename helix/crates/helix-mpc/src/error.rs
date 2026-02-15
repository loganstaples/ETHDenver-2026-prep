//! Error types for MPC operations.

use crate::types::PartyId;
use thiserror::Error;

/// Errors that can occur during MPC operations.
#[derive(Debug, Error)]
pub enum MPCError {
    #[error("insufficient shares: need {required}, have {available}")]
    InsufficientShares { required: usize, available: usize },

    #[error("share shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("share count mismatch: expected {expected}, got {got}")]
    ShareCountMismatch { expected: usize, got: usize },

    #[error("unknown party: {0}")]
    UnknownParty(PartyId),

    #[error("duplicate party: {0}")]
    DuplicateParty(PartyId),

    #[error("party {0} not authorized for this operation")]
    Unauthorized(PartyId),

    #[error("beaver triple pool exhausted: requested {requested}, available {available}")]
    BeaverPoolExhausted { requested: usize, available: usize },

    #[error("beaver triple verification failed")]
    BeaverVerificationFailed,

    #[error("commitment verification failed for party {party}")]
    CommitmentVerificationFailed { party: PartyId },

    #[error("share commitment mismatch: party {party} share does not match commitment")]
    ShareCommitmentMismatch { party: PartyId },

    #[error("reconstruction failed: {reason}")]
    ReconstructionFailed { reason: String },

    #[error("invalid threshold: k={threshold} must satisfy 1 <= k <= n={num_parties}")]
    InvalidThreshold {
        threshold: usize,
        num_parties: usize,
    },

    #[error("field arithmetic error: {0}")]
    FieldError(String),

    #[error("session error: {0}")]
    SessionError(String),

    #[error("protocol error: {0}")]
    ProtocolError(String),

    #[error("communication error: {0}")]
    CommunicationError(String),

    #[error("timeout waiting for party {party} in phase {phase}")]
    Timeout { party: PartyId, phase: String },

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("share index {index} out of range for {num_parties} parties")]
    IndexOutOfRange { index: usize, num_parties: usize },

    #[error("operation requires {required} parties but only {available} are active")]
    InsufficientParties { required: usize, available: usize },

    #[error("resharing failed: {0}")]
    ResharingFailed(String),

    #[error("malicious behavior detected from party {party}: {description}")]
    MaliciousBehavior {
        party: PartyId,
        description: String,
    },

    #[error("dimension mismatch in secure matmul: [{a_rows}x{a_cols}] @ [{b_rows}x{b_cols}]")]
    MatmulDimensionMismatch {
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
    },

    #[error("error bound exceeded: computed {computed}, maximum allowed {maximum}")]
    ErrorBoundExceeded { computed: f64, maximum: f64 },

    #[error("invalid round: expected {expected}, got {got}")]
    InvalidRound { expected: u64, got: u64 },

    #[error("key rotation error: {0}")]
    KeyRotationError(String),

    #[error("channel multiplexing error: {0}")]
    MultiplexError(String),

    #[error("message too large: {size} bytes exceeds maximum {max_size} bytes")]
    MessageTooLarge { size: usize, max_size: usize },

    #[error("replay attack detected: duplicate sequence {sequence} from party {party}")]
    ReplayAttack { sequence: u64, party: PartyId },

    #[error("MAC verification failed at step {step}: cheater identified as party {cheater:?}")]
    MACCheckFailed { step: u64, cheater: Option<usize> },
}

/// Result type for MPC operations.
pub type MPCResult<T> = Result<T, MPCError>;
