//! Error types for the HELIX project.

use thiserror::Error;

/// Top-level error type for HELIX operations.
#[derive(Error, Debug)]
pub enum HelixError {
    /// Arithmetic operation error.
    #[error("Arithmetic error: {0}")]
    Arithmetic(#[from] ArithmeticError),

    /// Error bound violation.
    #[error("Bounds error: {0}")]
    Bounds(#[from] BoundsError),

    /// Numeric overflow error.
    #[error("Overflow error: {0}")]
    Overflow(#[from] OverflowError),

    /// ZK circuit error.
    #[error("Circuit error: {0}")]
    Circuit(#[from] CircuitError),

    /// Network communication error.
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),

    /// Data verification error.
    #[error("Data error: {0}")]
    Data(#[from] DataError),

    /// Configuration error.
    #[error("Config error: {0}")]
    Config(String),

    /// I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors during arithmetic operations.
#[derive(Error, Debug)]
pub enum ArithmeticError {
    /// Division by zero.
    #[error("Division by zero")]
    DivisionByZero,

    /// Invalid operand (NaN, Inf, etc.).
    #[error("Invalid operand: {0}")]
    InvalidOperand(String),

    /// Dimension mismatch for matrix operations.
    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Operation not supported for given types.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Errors related to error bounds.
#[derive(Error, Debug)]
pub enum BoundsError {
    /// Computed error exceeds maximum allowed.
    #[error("Error bound exceeded: computed {computed}, max allowed {max_allowed}")]
    ExceedsMaximum { computed: f64, max_allowed: f64 },

    /// Negative error margin (invalid).
    #[error("Negative error margin: {0}")]
    NegativeMargin(f64),

    /// Error accumulation exceeded threshold.
    #[error("Error accumulation exceeded: total {total}, threshold {threshold}")]
    AccumulationExceeded { total: f64, threshold: f64 },

    /// Inconsistent bounds (lower > upper).
    #[error("Inconsistent bounds: lower {lower} > upper {upper}")]
    InconsistentBounds { lower: f64, upper: f64 },
}

/// Numeric overflow errors.
#[derive(Error, Debug)]
pub enum OverflowError {
    /// Integer overflow.
    #[error("Integer overflow in {operation}")]
    IntegerOverflow { operation: String },

    /// Floating-point overflow (infinity).
    #[error("Floating-point overflow in {operation}")]
    FloatOverflow { operation: String },

    /// Underflow (value too small to represent).
    #[error("Underflow in {operation}")]
    Underflow { operation: String },
}

/// Errors in ZK circuit operations.
#[derive(Error, Debug)]
pub enum CircuitError {
    /// Invalid witness.
    #[error("Invalid witness: {0}")]
    InvalidWitness(String),

    /// Constraint violation.
    #[error("Constraint violation at index {index}: {message}")]
    ConstraintViolation { index: usize, message: String },

    /// Proof generation failed.
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    /// Proof verification failed.
    #[error("Proof verification failed: {0}")]
    VerificationFailed(String),

    /// Circuit synthesis error.
    #[error("Circuit synthesis error: {0}")]
    SynthesisError(String),
}

/// Network communication errors.
#[derive(Error, Debug)]
pub enum NetworkError {
    /// Connection failed.
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Timeout.
    #[error("Operation timed out after {seconds} seconds")]
    Timeout { seconds: u64 },

    /// Peer not found.
    #[error("Peer not found: {peer_id}")]
    PeerNotFound { peer_id: String },

    /// Invalid message.
    #[error("Invalid message: {0}")]
    InvalidMessage(String),

    /// Protocol error.
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Errors related to data verification and provenance.
#[derive(Error, Debug)]
pub enum DataError {
    /// Merkle tree error.
    #[error("Merkle tree error: {0}")]
    MerkleError(String),

    /// Commitment verification failed.
    #[error("Commitment verification failed: {0}")]
    CommitmentVerificationFailed(String),

    /// Membership proof invalid.
    #[error("Membership proof invalid: {0}")]
    MembershipProofInvalid(String),

    /// Data integrity violation.
    #[error("Data integrity error: expected hash {expected}, got {actual}")]
    IntegrityError { expected: String, actual: String },

    /// Provenance chain broken.
    #[error("Provenance chain broken at step {step}: {reason}")]
    ProvenanceChainBroken { step: usize, reason: String },

    /// Data source error.
    #[error("Data source error: {0}")]
    SourceError(String),

    /// Sample not found.
    #[error("Sample not found: index {0}")]
    SampleNotFound(usize),

    /// Batch not found.
    #[error("Batch not found: index {0}")]
    BatchNotFound(usize),

    /// Invalid data format.
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    /// Data too large.
    #[error("Data too large: {size} bytes (max: {max})")]
    DataTooLarge { size: usize, max: usize },
}

/// Result type alias for HELIX operations.
pub type HelixResult<T> = Result<T, HelixError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ArithmeticError::DivisionByZero;
        assert_eq!(format!("{}", err), "Division by zero");

        let bounds_err = BoundsError::ExceedsMaximum {
            computed: 0.5,
            max_allowed: 0.1,
        };
        assert!(format!("{}", bounds_err).contains("0.5"));
    }

    #[test]
    fn test_error_conversion() {
        let arith_err = ArithmeticError::DivisionByZero;
        let helix_err: HelixError = arith_err.into();
        assert!(matches!(helix_err, HelixError::Arithmetic(_)));
    }
}
