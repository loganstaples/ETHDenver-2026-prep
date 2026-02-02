//! Error types for the HELIX project.
//!
//! This module provides comprehensive error handling with:
//! - Detailed error context and structured fields
//! - Error chains for root cause analysis
//! - Recovery hints for non-fatal errors
//! - Structured logging context for debugging

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
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

    /// Validation error.
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] SerializationError),

    /// Configuration error.
    #[error("Config error: {0}")]
    Config(String),

    /// I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Error with additional context.
    #[error("{message}")]
    WithContext {
        message: String,
        context: ErrorContext,
        #[source]
        source: Option<Box<HelixError>>,
    },
}

impl HelixError {
    /// Wraps this error with additional context.
    pub fn with_context(self, ctx: ErrorContext) -> Self {
        HelixError::WithContext {
            message: self.to_string(),
            context: ctx,
            source: Some(Box::new(self)),
        }
    }

    /// Creates a config error with context.
    pub fn config(message: impl Into<String>) -> Self {
        HelixError::Config(message.into())
    }

    /// Returns the recovery hint if available.
    pub fn recovery_hint(&self) -> Option<&str> {
        match self {
            HelixError::WithContext { context, .. } => context.recovery_hint.as_deref(),
            HelixError::Bounds(e) => e.recovery_hint(),
            HelixError::Overflow(e) => e.recovery_hint(),
            HelixError::Validation(e) => e.recovery_hint(),
            _ => None,
        }
    }

    /// Returns true if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            HelixError::WithContext { context, .. } => context.recoverable,
            HelixError::Bounds(e) => e.is_recoverable(),
            HelixError::Overflow(e) => e.is_recoverable(),
            HelixError::Validation(e) => e.is_recoverable(),
            HelixError::Arithmetic(_) => false,
            HelixError::Circuit(_) => false,
            HelixError::Network(_) => true, // Network errors are often transient
            HelixError::Data(_) => false,
            HelixError::Config(_) => false,
            HelixError::Io(_) => true, // IO errors may be transient
            HelixError::Serialization(e) => e.is_recoverable(),
        }
    }

    /// Returns the severity level of this error.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            HelixError::WithContext { context, .. } => context.severity,
            HelixError::Circuit(_) => ErrorSeverity::Critical,
            HelixError::Data(_) => ErrorSeverity::Error,
            HelixError::Bounds(e) => e.severity(),
            HelixError::Overflow(_) => ErrorSeverity::Warning,
            HelixError::Validation(e) => e.severity(),
            HelixError::Network(_) => ErrorSeverity::Warning,
            HelixError::Arithmetic(_) => ErrorSeverity::Error,
            HelixError::Config(_) => ErrorSeverity::Error,
            HelixError::Io(_) => ErrorSeverity::Warning,
            HelixError::Serialization(_) => ErrorSeverity::Error,
        }
    }

    /// Converts to structured log context.
    pub fn to_log_context(&self) -> LogContext {
        LogContext {
            error_type: self.error_type_name(),
            message: self.to_string(),
            severity: self.severity(),
            recoverable: self.is_recoverable(),
            recovery_hint: self.recovery_hint().map(String::from),
            fields: self.structured_fields(),
        }
    }

    /// Returns the error type name for logging.
    fn error_type_name(&self) -> &'static str {
        match self {
            HelixError::Arithmetic(_) => "ArithmeticError",
            HelixError::Bounds(_) => "BoundsError",
            HelixError::Overflow(_) => "OverflowError",
            HelixError::Circuit(_) => "CircuitError",
            HelixError::Network(_) => "NetworkError",
            HelixError::Data(_) => "DataError",
            HelixError::Validation(_) => "ValidationError",
            HelixError::Serialization(_) => "SerializationError",
            HelixError::Config(_) => "ConfigError",
            HelixError::Io(_) => "IoError",
            HelixError::WithContext { .. } => "ContextualError",
        }
    }

    /// Returns structured fields for logging.
    fn structured_fields(&self) -> HashMap<String, String> {
        let mut fields = HashMap::new();
        match self {
            HelixError::Bounds(e) => e.add_fields(&mut fields),
            HelixError::Overflow(e) => e.add_fields(&mut fields),
            HelixError::Validation(e) => e.add_fields(&mut fields),
            HelixError::WithContext { context, .. } => {
                fields.extend(context.fields.clone());
            }
            _ => {}
        }
        fields
    }
}

/// Error severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational - not actually an error.
    Info,
    /// Warning - operation succeeded but with concerns.
    Warning,
    /// Error - operation failed but system can continue.
    Error,
    /// Critical - system integrity may be compromised.
    Critical,
}

impl Default for ErrorSeverity {
    fn default() -> Self {
        ErrorSeverity::Error
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Info => write!(f, "INFO"),
            ErrorSeverity::Warning => write!(f, "WARN"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Contextual information attached to errors.
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// Operation that was being performed.
    pub operation: Option<String>,
    /// Component where the error occurred.
    pub component: Option<String>,
    /// Additional structured fields.
    pub fields: HashMap<String, String>,
    /// Recovery hint for the user/system.
    pub recovery_hint: Option<String>,
    /// Whether this error is recoverable.
    pub recoverable: bool,
    /// Severity level.
    pub severity: ErrorSeverity,
}

impl ErrorContext {
    /// Creates a new error context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the operation name.
    pub fn operation(mut self, op: impl Into<String>) -> Self {
        self.operation = Some(op.into());
        self
    }

    /// Sets the component name.
    pub fn component(mut self, comp: impl Into<String>) -> Self {
        self.component = Some(comp.into());
        self
    }

    /// Adds a field to the context.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Sets the recovery hint.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.recovery_hint = Some(hint.into());
        self
    }

    /// Marks this error as recoverable.
    pub fn recoverable(mut self) -> Self {
        self.recoverable = true;
        self
    }

    /// Sets the severity level.
    pub fn severity(mut self, sev: ErrorSeverity) -> Self {
        self.severity = sev;
        self
    }
}

/// Structured log context for error reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogContext {
    /// Error type name.
    pub error_type: &'static str,
    /// Error message.
    pub message: String,
    /// Severity level.
    pub severity: ErrorSeverity,
    /// Whether the error is recoverable.
    pub recoverable: bool,
    /// Recovery hint.
    pub recovery_hint: Option<String>,
    /// Additional structured fields.
    pub fields: HashMap<String, String>,
}

impl fmt::Display for LogContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.error_type, self.message)?;
        if !self.fields.is_empty() {
            write!(f, " {{")?;
            for (i, (k, v)) in self.fields.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}={}", k, v)?;
            }
            write!(f, "}}")?;
        }
        if let Some(hint) = &self.recovery_hint {
            write!(f, " [hint: {}]", hint)?;
        }
        Ok(())
    }
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

    /// NaN detected in computation.
    #[error("NaN detected in {operation}")]
    NaNDetected { operation: String },

    /// Infinity detected in computation.
    #[error("Infinity detected in {operation}: {value}")]
    InfinityDetected { operation: String, value: f64 },

    /// Shape mismatch for tensor operations.
    #[error("Shape mismatch for {operation}: left {left:?}, right {right:?}")]
    ShapeMismatch {
        operation: String,
        left: Vec<usize>,
        right: Vec<usize>,
    },

    /// Invalid index for tensor access.
    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
}

impl ArithmeticError {
    /// Creates a NaN detected error.
    pub fn nan_detected(operation: impl Into<String>) -> Self {
        ArithmeticError::NaNDetected {
            operation: operation.into(),
        }
    }

    /// Creates an infinity detected error.
    pub fn infinity_detected(operation: impl Into<String>, value: f64) -> Self {
        ArithmeticError::InfinityDetected {
            operation: operation.into(),
            value,
        }
    }

    /// Creates a shape mismatch error.
    pub fn shape_mismatch(
        operation: impl Into<String>,
        left: Vec<usize>,
        right: Vec<usize>,
    ) -> Self {
        ArithmeticError::ShapeMismatch {
            operation: operation.into(),
            left,
            right,
        }
    }

    /// Creates an index out of bounds error.
    pub fn index_out_of_bounds(index: Vec<usize>, shape: Vec<usize>) -> Self {
        ArithmeticError::IndexOutOfBounds { index, shape }
    }
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

    /// Error bound would overflow.
    #[error("Error bound overflow in {operation}: value {value}")]
    ErrorBoundOverflow { operation: String, value: f64 },

    /// Error propagation resulted in invalid state.
    #[error("Invalid error propagation: {message}")]
    InvalidPropagation { message: String },
}

impl BoundsError {
    /// Returns the recovery hint for this error.
    pub fn recovery_hint(&self) -> Option<&str> {
        match self {
            BoundsError::ExceedsMaximum { .. } => {
                Some("Consider reducing precision or using smaller batch sizes")
            }
            BoundsError::AccumulationExceeded { .. } => {
                Some("Consider checkpointing to reset error accumulation")
            }
            BoundsError::ErrorBoundOverflow { .. } => {
                Some("Clamp error bounds to maximum representable value")
            }
            BoundsError::InvalidPropagation { .. } => {
                Some("Verify input values are within expected ranges")
            }
            _ => None,
        }
    }

    /// Returns whether this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            BoundsError::ExceedsMaximum { .. } => true,
            BoundsError::AccumulationExceeded { .. } => true,
            BoundsError::ErrorBoundOverflow { .. } => true,
            BoundsError::NegativeMargin(_) => false,
            BoundsError::InconsistentBounds { .. } => false,
            BoundsError::InvalidPropagation { .. } => false,
        }
    }

    /// Returns the severity of this error.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            BoundsError::ExceedsMaximum { .. } => ErrorSeverity::Warning,
            BoundsError::AccumulationExceeded { .. } => ErrorSeverity::Warning,
            BoundsError::ErrorBoundOverflow { .. } => ErrorSeverity::Warning,
            BoundsError::NegativeMargin(_) => ErrorSeverity::Error,
            BoundsError::InconsistentBounds { .. } => ErrorSeverity::Error,
            BoundsError::InvalidPropagation { .. } => ErrorSeverity::Error,
        }
    }

    /// Adds structured fields for logging.
    pub fn add_fields(&self, fields: &mut HashMap<String, String>) {
        match self {
            BoundsError::ExceedsMaximum {
                computed,
                max_allowed,
            } => {
                fields.insert("computed".to_string(), format!("{:.6e}", computed));
                fields.insert("max_allowed".to_string(), format!("{:.6e}", max_allowed));
            }
            BoundsError::AccumulationExceeded { total, threshold } => {
                fields.insert("total".to_string(), format!("{:.6e}", total));
                fields.insert("threshold".to_string(), format!("{:.6e}", threshold));
            }
            BoundsError::ErrorBoundOverflow { operation, value } => {
                fields.insert("operation".to_string(), operation.clone());
                fields.insert("value".to_string(), format!("{:.6e}", value));
            }
            _ => {}
        }
    }

    /// Creates an error bound overflow error.
    pub fn overflow(operation: impl Into<String>, value: f64) -> Self {
        BoundsError::ErrorBoundOverflow {
            operation: operation.into(),
            value,
        }
    }

    /// Creates an invalid propagation error.
    pub fn invalid_propagation(message: impl Into<String>) -> Self {
        BoundsError::InvalidPropagation {
            message: message.into(),
        }
    }
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

    /// Accumulation overflow.
    #[error("Accumulation overflow: {accumulated} + {adding} exceeds limit")]
    AccumulationOverflow { accumulated: f64, adding: f64 },

    /// Error margin overflow during composition.
    #[error("Error margin overflow: {current} * {multiplier} exceeds representable range")]
    ErrorMarginOverflow { current: f64, multiplier: f64 },
}

impl OverflowError {
    /// Returns the recovery hint for this error.
    pub fn recovery_hint(&self) -> Option<&str> {
        match self {
            OverflowError::IntegerOverflow { .. } => Some("Use checked arithmetic operations"),
            OverflowError::FloatOverflow { .. } => Some("Clamp values to prevent overflow"),
            OverflowError::Underflow { .. } => Some("Use subnormal numbers or higher precision"),
            OverflowError::AccumulationOverflow { .. } => {
                Some("Reset accumulation or use smaller increments")
            }
            OverflowError::ErrorMarginOverflow { .. } => Some("Clamp error margins to maximum"),
        }
    }

    /// Returns whether this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        true // All overflow errors can be recovered by clamping
    }

    /// Adds structured fields for logging.
    pub fn add_fields(&self, fields: &mut HashMap<String, String>) {
        match self {
            OverflowError::IntegerOverflow { operation } => {
                fields.insert("operation".to_string(), operation.clone());
            }
            OverflowError::FloatOverflow { operation } => {
                fields.insert("operation".to_string(), operation.clone());
            }
            OverflowError::Underflow { operation } => {
                fields.insert("operation".to_string(), operation.clone());
            }
            OverflowError::AccumulationOverflow { accumulated, adding } => {
                fields.insert("accumulated".to_string(), format!("{:.6e}", accumulated));
                fields.insert("adding".to_string(), format!("{:.6e}", adding));
            }
            OverflowError::ErrorMarginOverflow { current, multiplier } => {
                fields.insert("current".to_string(), format!("{:.6e}", current));
                fields.insert("multiplier".to_string(), format!("{:.6e}", multiplier));
            }
        }
    }

    /// Creates an integer overflow error.
    pub fn integer(operation: impl Into<String>) -> Self {
        OverflowError::IntegerOverflow {
            operation: operation.into(),
        }
    }

    /// Creates a float overflow error.
    pub fn float(operation: impl Into<String>) -> Self {
        OverflowError::FloatOverflow {
            operation: operation.into(),
        }
    }

    /// Creates an underflow error.
    pub fn underflow(operation: impl Into<String>) -> Self {
        OverflowError::Underflow {
            operation: operation.into(),
        }
    }

    /// Creates an accumulation overflow error.
    pub fn accumulation(accumulated: f64, adding: f64) -> Self {
        OverflowError::AccumulationOverflow { accumulated, adding }
    }

    /// Creates an error margin overflow error.
    pub fn error_margin(current: f64, multiplier: f64) -> Self {
        OverflowError::ErrorMarginOverflow { current, multiplier }
    }
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

/// Validation errors with field-level detail.
#[derive(Error, Debug)]
pub enum ValidationError {
    /// Required field is missing.
    #[error("Missing required field: {field}")]
    MissingField { field: String },

    /// Field value is out of range.
    #[error("Field {field} out of range: {value} not in [{min}, {max}]")]
    OutOfRange {
        field: String,
        value: f64,
        min: f64,
        max: f64,
    },

    /// Field value is invalid.
    #[error("Invalid value for {field}: {value} ({reason})")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    /// Type mismatch.
    #[error("Type mismatch for {field}: expected {expected}, got {actual}")]
    TypeMismatch {
        field: String,
        expected: String,
        actual: String,
    },

    /// Shape validation failed.
    #[error("Invalid shape for {field}: {shape:?} ({reason})")]
    InvalidShape {
        field: String,
        shape: Vec<usize>,
        reason: String,
    },

    /// Configuration is invalid.
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    /// Input contains invalid data.
    #[error("Invalid input data: {message}")]
    InvalidInput { message: String },

    /// Serialization round-trip failed.
    #[error("Serialization round-trip failed: {message}")]
    SerializationRoundTrip { message: String },

    /// Multiple validation errors.
    #[error("Multiple validation errors: {}", format_validation_errors(.errors))]
    Multiple { errors: Vec<ValidationError> },
}

fn format_validation_errors(errors: &[ValidationError]) -> String {
    errors
        .iter()
        .enumerate()
        .map(|(i, e)| format!("  {}. {}", i + 1, e))
        .collect::<Vec<_>>()
        .join("\n")
}

impl ValidationError {
    /// Creates a missing field error.
    pub fn missing_field(field: impl Into<String>) -> Self {
        ValidationError::MissingField {
            field: field.into(),
        }
    }

    /// Creates an out of range error.
    pub fn out_of_range(field: impl Into<String>, value: f64, min: f64, max: f64) -> Self {
        ValidationError::OutOfRange {
            field: field.into(),
            value,
            min,
            max,
        }
    }

    /// Creates an invalid value error.
    pub fn invalid_value(
        field: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        ValidationError::InvalidValue {
            field: field.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Creates a type mismatch error.
    pub fn type_mismatch(
        field: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        ValidationError::TypeMismatch {
            field: field.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Creates an invalid shape error.
    pub fn invalid_shape(
        field: impl Into<String>,
        shape: Vec<usize>,
        reason: impl Into<String>,
    ) -> Self {
        ValidationError::InvalidShape {
            field: field.into(),
            shape,
            reason: reason.into(),
        }
    }

    /// Creates an invalid config error.
    pub fn invalid_config(message: impl Into<String>) -> Self {
        ValidationError::InvalidConfig {
            message: message.into(),
        }
    }

    /// Creates an invalid input error.
    pub fn invalid_input(message: impl Into<String>) -> Self {
        ValidationError::InvalidInput {
            message: message.into(),
        }
    }

    /// Creates a serialization round-trip error.
    pub fn serialization_roundtrip(message: impl Into<String>) -> Self {
        ValidationError::SerializationRoundTrip {
            message: message.into(),
        }
    }

    /// Creates a multiple errors wrapper.
    pub fn multiple(errors: Vec<ValidationError>) -> Self {
        ValidationError::Multiple { errors }
    }

    /// Returns the recovery hint for this error.
    pub fn recovery_hint(&self) -> Option<&str> {
        match self {
            ValidationError::MissingField { .. } => Some("Provide the required field"),
            ValidationError::OutOfRange { .. } => Some("Adjust value to be within valid range"),
            ValidationError::InvalidValue { .. } => Some("Provide a valid value"),
            ValidationError::TypeMismatch { .. } => Some("Provide correct type"),
            ValidationError::InvalidShape { .. } => Some("Provide tensor with correct shape"),
            ValidationError::InvalidConfig { .. } => Some("Review configuration parameters"),
            ValidationError::InvalidInput { .. } => Some("Validate input data before processing"),
            ValidationError::SerializationRoundTrip { .. } => {
                Some("Check serialization format compatibility")
            }
            ValidationError::Multiple { .. } => Some("Fix all listed validation errors"),
        }
    }

    /// Returns whether this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        true // All validation errors are recoverable by fixing the input
    }

    /// Returns the severity of this error.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ValidationError::Multiple { .. } => ErrorSeverity::Error,
            _ => ErrorSeverity::Warning,
        }
    }

    /// Adds structured fields for logging.
    pub fn add_fields(&self, fields: &mut HashMap<String, String>) {
        match self {
            ValidationError::MissingField { field } => {
                fields.insert("field".to_string(), field.clone());
            }
            ValidationError::OutOfRange {
                field,
                value,
                min,
                max,
            } => {
                fields.insert("field".to_string(), field.clone());
                fields.insert("value".to_string(), format!("{}", value));
                fields.insert("min".to_string(), format!("{}", min));
                fields.insert("max".to_string(), format!("{}", max));
            }
            ValidationError::InvalidValue {
                field,
                value,
                reason,
            } => {
                fields.insert("field".to_string(), field.clone());
                fields.insert("value".to_string(), value.clone());
                fields.insert("reason".to_string(), reason.clone());
            }
            ValidationError::InvalidShape {
                field,
                shape,
                reason,
            } => {
                fields.insert("field".to_string(), field.clone());
                fields.insert("shape".to_string(), format!("{:?}", shape));
                fields.insert("reason".to_string(), reason.clone());
            }
            _ => {}
        }
    }
}

/// Serialization-specific errors.
#[derive(Error, Debug)]
pub enum SerializationError {
    /// JSON serialization failed.
    #[error("JSON serialization failed: {0}")]
    JsonError(String),

    /// Binary serialization failed.
    #[error("Binary serialization failed: {0}")]
    BinaryError(String),

    /// Deserialization failed.
    #[error("Deserialization failed: {0}")]
    DeserializationError(String),

    /// Version mismatch.
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    /// Data corruption detected.
    #[error("Data corruption detected: {0}")]
    Corruption(String),
}

impl SerializationError {
    /// Returns whether this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            SerializationError::JsonError(_) => false,
            SerializationError::BinaryError(_) => false,
            SerializationError::DeserializationError(_) => false,
            SerializationError::VersionMismatch { .. } => true, // May be able to upgrade
            SerializationError::Corruption(_) => false,
        }
    }

    /// Creates a JSON error.
    pub fn json(message: impl Into<String>) -> Self {
        SerializationError::JsonError(message.into())
    }

    /// Creates a binary error.
    pub fn binary(message: impl Into<String>) -> Self {
        SerializationError::BinaryError(message.into())
    }

    /// Creates a deserialization error.
    pub fn deserialization(message: impl Into<String>) -> Self {
        SerializationError::DeserializationError(message.into())
    }

    /// Creates a version mismatch error.
    pub fn version_mismatch(expected: u32, actual: u32) -> Self {
        SerializationError::VersionMismatch { expected, actual }
    }

    /// Creates a corruption error.
    pub fn corruption(message: impl Into<String>) -> Self {
        SerializationError::Corruption(message.into())
    }
}

impl From<serde_json::Error> for SerializationError {
    fn from(e: serde_json::Error) -> Self {
        SerializationError::JsonError(e.to_string())
    }
}

impl From<serde_json::Error> for HelixError {
    fn from(e: serde_json::Error) -> Self {
        HelixError::Serialization(SerializationError::from(e))
    }
}

/// Result type alias for HELIX operations.
pub type HelixResult<T> = Result<T, HelixError>;

/// Extension trait for adding context to Results.
pub trait ResultExt<T> {
    /// Adds error context to a Result.
    fn with_context(self, ctx: ErrorContext) -> HelixResult<T>;

    /// Adds operation context to a Result.
    fn in_operation(self, op: impl Into<String>) -> HelixResult<T>;

    /// Adds component context to a Result.
    fn in_component(self, comp: impl Into<String>) -> HelixResult<T>;
}

impl<T, E: Into<HelixError>> ResultExt<T> for Result<T, E> {
    fn with_context(self, ctx: ErrorContext) -> HelixResult<T> {
        self.map_err(|e| e.into().with_context(ctx))
    }

    fn in_operation(self, op: impl Into<String>) -> HelixResult<T> {
        self.with_context(ErrorContext::new().operation(op))
    }

    fn in_component(self, comp: impl Into<String>) -> HelixResult<T> {
        self.with_context(ErrorContext::new().component(comp))
    }
}

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

    #[test]
    fn test_error_context() {
        let err = ArithmeticError::DivisionByZero;
        let helix_err: HelixError = err.into();
        let ctx = ErrorContext::new()
            .operation("division")
            .component("bounded_value")
            .field("divisor", "0.0")
            .hint("Check divisor is non-zero")
            .recoverable();

        let with_ctx = helix_err.with_context(ctx);
        assert!(with_ctx.is_recoverable());
        assert!(with_ctx.recovery_hint().is_some());
    }

    #[test]
    fn test_log_context() {
        let err = BoundsError::ExceedsMaximum {
            computed: 0.5,
            max_allowed: 0.1,
        };
        let helix_err: HelixError = err.into();
        let log_ctx = helix_err.to_log_context();

        assert_eq!(log_ctx.error_type, "BoundsError");
        assert!(log_ctx.fields.contains_key("computed"));
        assert!(log_ctx.fields.contains_key("max_allowed"));
    }

    #[test]
    fn test_validation_error() {
        let err = ValidationError::out_of_range("learning_rate", 2.0, 0.0, 1.0);
        assert!(err.is_recoverable());
        assert!(err.recovery_hint().is_some());

        let err: HelixError = err.into();
        let log_ctx = err.to_log_context();
        assert!(log_ctx.fields.contains_key("field"));
    }

    #[test]
    fn test_multiple_validation_errors() {
        let errors = vec![
            ValidationError::missing_field("name"),
            ValidationError::out_of_range("rate", -0.5, 0.0, 1.0),
        ];
        let err = ValidationError::multiple(errors);
        let msg = format!("{}", err);
        assert!(msg.contains("name"));
        assert!(msg.contains("rate"));
    }

    #[test]
    fn test_error_severity() {
        let critical = HelixError::Circuit(CircuitError::VerificationFailed("test".into()));
        assert_eq!(critical.severity(), ErrorSeverity::Critical);

        let warning = HelixError::Bounds(BoundsError::ExceedsMaximum {
            computed: 0.5,
            max_allowed: 0.1,
        });
        assert_eq!(warning.severity(), ErrorSeverity::Warning);
    }

    #[test]
    fn test_result_ext() {
        fn failing_op() -> Result<i32, ArithmeticError> {
            Err(ArithmeticError::DivisionByZero)
        }

        let result: HelixResult<i32> = failing_op().in_operation("test_division");
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_infinity_errors() {
        let nan_err = ArithmeticError::nan_detected("matrix_multiply");
        assert!(format!("{}", nan_err).contains("NaN"));

        let inf_err = ArithmeticError::infinity_detected("exp", f64::INFINITY);
        assert!(format!("{}", inf_err).contains("Infinity"));
    }

    #[test]
    fn test_serialization_error() {
        let err = SerializationError::version_mismatch(2, 1);
        assert!(err.is_recoverable());

        let err = SerializationError::corruption("checksum mismatch");
        assert!(!err.is_recoverable());
    }
}
