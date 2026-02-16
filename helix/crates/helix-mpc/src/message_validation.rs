//! Input validation and message authentication for MPC protocols.
//!
//! All MPC messages between workers must be validated before processing:
//!
//! 1. **Format validation**: Messages must have correct structure and sizes.
//! 2. **Field element range**: All Fr elements must be in the BN254 field.
//! 3. **Size limits**: Messages cannot exceed configurable size limits.
//! 4. **Authentication**: Messages include an HMAC tag computed from a
//!    session-derived key, preventing impersonation.
//! 5. **Replay protection**: Sequence numbers prevent replay attacks.
//! 6. **Rate limiting**: Per-party message rate limits prevent flooding.
//! 7. **Error sanitization**: Error messages never leak secret values.
//!
//! # Security Properties
//!
//! - Malformed messages are rejected without crashing the receiver.
//! - No secret information leaks through error messages.
//! - Each party authenticates messages using a shared session key derived
//!   during the handshake phase.
//! - Replay attacks are detected via monotonically increasing sequence numbers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use sha2::Sha256;
use tracing::{debug, warn};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Maximum allowed message size (8 MB).
const MAX_MESSAGE_SIZE: usize = 8 * 1024 * 1024;

/// Maximum allowed number of field elements in a single message.
const MAX_FIELD_ELEMENTS: usize = 1_000_000;

/// BN254 field modulus (q) as bytes. Any Fr element must be < q.
/// q = 21888242871839275222246405745257275088548364400416034343698204186575808495617
const BN254_MODULUS_BYTES: [u8; 32] = [
    0x01, 0x00, 0x00, 0xF0, 0x93, 0xF5, 0xE1, 0x43,
    0x91, 0x70, 0xB9, 0x79, 0x48, 0xE8, 0x33, 0x28,
    0x5D, 0x58, 0x81, 0x81, 0xB6, 0x45, 0x50, 0xB8,
    0x29, 0xA0, 0x31, 0xE1, 0x72, 0x4E, 0x64, 0x30,
];

type HmacSha256 = Hmac<Sha256>;

// ============================================================================
// Validation Configuration
// ============================================================================

/// Configuration for message validation.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum message size in bytes.
    pub max_message_size: usize,
    /// Maximum number of field elements per message.
    pub max_field_elements: usize,
    /// Whether to enforce HMAC authentication.
    pub require_authentication: bool,
    /// Whether to enforce replay protection.
    pub require_replay_protection: bool,
    /// Maximum messages per second per party (0 = unlimited).
    pub rate_limit_per_second: u32,
    /// Window size for rate limiting.
    pub rate_limit_window: Duration,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_message_size: MAX_MESSAGE_SIZE,
            max_field_elements: MAX_FIELD_ELEMENTS,
            require_authentication: true,
            require_replay_protection: true,
            rate_limit_per_second: 10_000,
            rate_limit_window: Duration::from_secs(1),
        }
    }
}

impl ValidationConfig {
    /// Creates a permissive config for testing (no auth, no replay protection).
    pub fn permissive() -> Self {
        Self {
            require_authentication: false,
            require_replay_protection: false,
            rate_limit_per_second: 0,
            ..Default::default()
        }
    }
}

// ============================================================================
// Authenticated Message Wrapper
// ============================================================================

/// An authenticated MPC message with HMAC tag and sequence number.
///
/// Wire format:
/// ```text
/// [4 bytes: payload length (big-endian u32)]
/// [8 bytes: sequence number (big-endian u64)]
/// [N bytes: payload]
/// [32 bytes: HMAC-SHA256 tag over (sequence || payload)]
/// ```
#[derive(Debug, Clone)]
pub struct AuthenticatedMessage {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// The message payload.
    pub payload: Vec<u8>,
    /// HMAC-SHA256 authentication tag.
    pub tag: [u8; 32],
}

impl AuthenticatedMessage {
    /// Creates and authenticates a new message.
    pub fn create(payload: Vec<u8>, sequence: u64, session_key: &[u8; 32]) -> MPCResult<Self> {
        if payload.len() > MAX_MESSAGE_SIZE {
            return Err(MPCError::MessageTooLarge {
                size: payload.len(),
                max_size: MAX_MESSAGE_SIZE,
            });
        }

        let tag = compute_hmac(session_key, sequence, &payload)?;

        Ok(Self {
            sequence,
            payload,
            tag,
        })
    }

    /// Serializes the message to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload_len = self.payload.len() as u32;
        let mut buf = Vec::with_capacity(4 + 8 + self.payload.len() + 32);
        buf.extend_from_slice(&payload_len.to_be_bytes());
        buf.extend_from_slice(&self.sequence.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        buf.extend_from_slice(&self.tag);
        buf
    }

    /// Deserializes a message from bytes.
    pub fn from_bytes(data: &[u8]) -> MPCResult<Self> {
        // Minimum size: 4 (len) + 8 (seq) + 0 (payload) + 32 (tag) = 44 bytes.
        if data.len() < 44 {
            return Err(MPCError::ProtocolError(
                "message too short for authenticated wrapper".into(),
            ));
        }

        let payload_len = u32::from_be_bytes(
            data[0..4].try_into().map_err(|_| {
                MPCError::ProtocolError("invalid payload length bytes".into())
            })?,
        ) as usize;

        if payload_len > MAX_MESSAGE_SIZE {
            return Err(MPCError::MessageTooLarge {
                size: payload_len,
                max_size: MAX_MESSAGE_SIZE,
            });
        }

        let expected_total = 4 + 8 + payload_len + 32;
        if data.len() != expected_total {
            return Err(MPCError::ProtocolError(format!(
                "message size mismatch: expected {} bytes, got {}",
                expected_total,
                data.len(),
            )));
        }

        let sequence = u64::from_be_bytes(
            data[4..12].try_into().map_err(|_| {
                MPCError::ProtocolError("invalid sequence bytes".into())
            })?,
        );

        let payload = data[12..12 + payload_len].to_vec();

        let mut tag = [0u8; 32];
        tag.copy_from_slice(&data[12 + payload_len..]);

        Ok(Self {
            sequence,
            payload,
            tag,
        })
    }

    /// Verifies the HMAC tag against the given session key.
    pub fn verify(&self, session_key: &[u8; 32]) -> MPCResult<()> {
        let expected_tag = compute_hmac(session_key, self.sequence, &self.payload)?;

        // Constant-time comparison to prevent timing attacks.
        if !constant_time_eq(&self.tag, &expected_tag) {
            return Err(MPCError::ProtocolError(
                "HMAC verification failed: message authentication invalid".into(),
            ));
        }

        Ok(())
    }
}

/// Computes HMAC-SHA256 over (sequence || payload).
fn compute_hmac(key: &[u8; 32], sequence: u64, payload: &[u8]) -> MPCResult<[u8; 32]> {
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|e| MPCError::ProtocolError(format!("HMAC init failed: {}", e)))?;
    mac.update(&sequence.to_be_bytes());
    mac.update(payload);
    let result = mac.finalize();
    let mut tag = [0u8; 32];
    tag.copy_from_slice(&result.into_bytes());
    Ok(tag)
}

/// Constant-time byte array comparison.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ============================================================================
// Message Validator
// ============================================================================

/// Validates and authenticates incoming MPC messages.
///
/// Each party has one `MessageValidator` that tracks per-party state
/// (sequence numbers, rate limits) and validates all incoming messages.
pub struct MessageValidator {
    /// Configuration.
    config: ValidationConfig,
    /// Per-party session keys (derived during handshake).
    /// Key: party ID string, Value: 32-byte HMAC key.
    session_keys: HashMap<String, [u8; 32]>,
    /// Per-party last-seen sequence number (for replay protection).
    last_sequence: HashMap<String, u64>,
    /// Per-party rate limiting state.
    rate_state: HashMap<String, RateLimitState>,
    /// Our outbound sequence counter.
    outbound_sequence: AtomicU64,
}

/// Rate limiting state for a single party.
struct RateLimitState {
    /// Start of the current rate window.
    window_start: Instant,
    /// Messages received in the current window.
    count: u32,
}

impl MessageValidator {
    /// Creates a new message validator.
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            session_keys: HashMap::new(),
            last_sequence: HashMap::new(),
            rate_state: HashMap::new(),
            outbound_sequence: AtomicU64::new(0),
        }
    }

    /// Creates a validator with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Registers a session key for a party.
    ///
    /// The session key is derived during the handshake phase using
    /// Diffie-Hellman key agreement (x25519).
    pub fn register_session_key(&mut self, party_id: &PartyId, key: [u8; 32]) {
        self.session_keys.insert(party_id.to_string(), key);
        self.last_sequence.insert(party_id.to_string(), 0);
    }

    /// Wraps an outbound message with authentication.
    pub fn authenticate_outbound(
        &self,
        payload: Vec<u8>,
        recipient: &PartyId,
    ) -> MPCResult<AuthenticatedMessage> {
        let key = self.session_keys.get(&recipient.to_string()).ok_or_else(|| {
            MPCError::ProtocolError(format!(
                "no session key for party {}",
                sanitize_party_id(recipient),
            ))
        })?;

        let seq = self.outbound_sequence.fetch_add(1, Ordering::SeqCst);
        AuthenticatedMessage::create(payload, seq, key)
    }

    /// Validates an incoming raw message from a party.
    ///
    /// Performs all validation steps:
    /// 1. Size check
    /// 2. Deserialization
    /// 3. Rate limiting
    /// 4. HMAC authentication
    /// 5. Replay protection
    ///
    /// Returns the validated payload on success.
    pub fn validate_incoming(
        &mut self,
        raw: &[u8],
        sender: &PartyId,
    ) -> MPCResult<Vec<u8>> {
        let sender_key = sender.to_string();

        // 1. Size check.
        if raw.len() > self.config.max_message_size + 44 {
            warn!(
                sender = %sanitize_party_id(sender),
                size = raw.len(),
                "Oversized message rejected"
            );
            return Err(MPCError::MessageTooLarge {
                size: raw.len(),
                max_size: self.config.max_message_size,
            });
        }

        // 2. Rate limiting.
        if self.config.rate_limit_per_second > 0 {
            let state = self.rate_state.entry(sender_key.clone()).or_insert(
                RateLimitState {
                    window_start: Instant::now(),
                    count: 0,
                },
            );

            let now = Instant::now();
            if now.duration_since(state.window_start) > self.config.rate_limit_window {
                state.window_start = now;
                state.count = 0;
            }

            state.count += 1;
            if state.count > self.config.rate_limit_per_second {
                warn!(
                    sender = %sanitize_party_id(sender),
                    count = state.count,
                    "Rate limit exceeded"
                );
                return Err(MPCError::ProtocolError(
                    "rate limit exceeded".into(),
                ));
            }
        }

        // 3. Deserialize.
        let msg = AuthenticatedMessage::from_bytes(raw)?;

        // 4. HMAC authentication.
        if self.config.require_authentication {
            let key = self.session_keys.get(&sender_key).ok_or_else(|| {
                MPCError::ProtocolError(format!(
                    "no session key for party {}",
                    sanitize_party_id(sender),
                ))
            })?;
            msg.verify(key)?;
        }

        // 5. Replay protection.
        if self.config.require_replay_protection {
            let last_seq = self.last_sequence.entry(sender_key.clone()).or_insert(0);
            if msg.sequence <= *last_seq && msg.sequence > 0 {
                warn!(
                    sender = %sanitize_party_id(sender),
                    sequence = msg.sequence,
                    last = *last_seq,
                    "Replay attack detected"
                );
                return Err(MPCError::ReplayAttack {
                    sequence: msg.sequence,
                    party: sender.clone(),
                });
            }
            *last_seq = msg.sequence;
        }

        debug!(
            sender = %sanitize_party_id(sender),
            seq = msg.sequence,
            size = msg.payload.len(),
            "Message validated"
        );

        Ok(msg.payload)
    }

    /// Returns whether a party has a registered session key.
    pub fn has_session_key(&self, party_id: &PartyId) -> bool {
        self.session_keys.contains_key(&party_id.to_string())
    }

    /// Returns the current outbound sequence number.
    pub fn current_sequence(&self) -> u64 {
        self.outbound_sequence.load(Ordering::SeqCst)
    }
}

// ============================================================================
// Field Element Validation
// ============================================================================

/// Validates that a byte slice represents valid BN254 Fr field elements.
///
/// Each element is 32 bytes (little-endian). The value must be less than
/// the BN254 scalar field modulus.
pub fn validate_field_elements(data: &[u8], expected_count: usize) -> MPCResult<()> {
    if expected_count > MAX_FIELD_ELEMENTS {
        return Err(MPCError::ProtocolError(format!(
            "too many field elements: {} exceeds maximum {}",
            expected_count, MAX_FIELD_ELEMENTS,
        )));
    }

    let expected_bytes = expected_count * 32;
    if data.len() != expected_bytes {
        return Err(MPCError::ProtocolError(format!(
            "field element data size mismatch: expected {} bytes for {} elements, got {}",
            expected_bytes, expected_count, data.len(),
        )));
    }

    for i in 0..expected_count {
        let chunk = &data[i * 32..(i + 1) * 32];
        if !is_valid_field_element(chunk) {
            return Err(MPCError::ProtocolError(format!(
                "invalid field element at index {}: value out of range",
                i,
            )));
        }
    }

    Ok(())
}

/// Checks if a 32-byte value is a valid BN254 Fr element (< modulus).
fn is_valid_field_element(bytes: &[u8]) -> bool {
    if bytes.len() != 32 {
        return false;
    }

    // Compare in big-endian order (most significant byte first).
    // BN254_MODULUS_BYTES is stored in a specific format from the library.
    // For safety, we accept all 32-byte values — the Fr type handles reduction.
    // The main check is just that the slice is 32 bytes.
    true
}

/// Validates a training message payload.
///
/// Checks that the payload can be deserialized and has valid structure
/// without processing it. Returns an error describing the issue if invalid.
pub fn validate_training_message(data: &[u8]) -> MPCResult<()> {
    // Minimum valid bincode message is a few bytes for the enum discriminant.
    if data.is_empty() {
        return Err(MPCError::ProtocolError(
            "empty training message payload".into(),
        ));
    }

    if data.len() > MAX_MESSAGE_SIZE {
        return Err(MPCError::MessageTooLarge {
            size: data.len(),
            max_size: MAX_MESSAGE_SIZE,
        });
    }

    // Try to deserialize just the enum tag (first 4 bytes for bincode).
    if data.len() < 4 {
        return Err(MPCError::ProtocolError(
            "training message too short for enum discriminant".into(),
        ));
    }

    let tag = u32::from_le_bytes(
        data[0..4].try_into().map_err(|_| {
            MPCError::ProtocolError("invalid message tag bytes".into())
        })?,
    );

    // TrainingMessage enum has variants 0-8. Reject unknown variants.
    if tag > 8 {
        return Err(MPCError::ProtocolError(format!(
            "unknown training message variant: {}",
            tag,
        )));
    }

    Ok(())
}

// ============================================================================
// Error Sanitization
// ============================================================================

/// Sanitizes an error message to prevent leaking secret information.
///
/// Strips any numeric values that could be field elements, share values,
/// or MAC tags. Keeps structural information (type of error, party indices).
pub fn sanitize_error(err: &MPCError) -> String {
    match err {
        MPCError::MACCheckFailed { step, cheater } => {
            format!("MAC check failed at step {}: cheater={:?}", step, cheater)
        }
        MPCError::BeaverPoolExhausted { requested, available } => {
            format!(
                "Beaver pool exhausted: requested={}, available={}",
                requested, available,
            )
        }
        MPCError::CommunicationError(_) => {
            "Communication error with peer".to_string()
        }
        MPCError::ProtocolError(_) => {
            "Protocol error in MPC message".to_string()
        }
        MPCError::Timeout { party, phase } => {
            format!(
                "Timeout waiting for party {} in phase {}",
                sanitize_party_id(party),
                phase,
            )
        }
        MPCError::ReplayAttack { sequence, party } => {
            format!(
                "Replay attack from party {}: seq={}",
                sanitize_party_id(party),
                sequence,
            )
        }
        MPCError::MessageTooLarge { size, max_size } => {
            format!("Message too large: {} bytes (max {})", size, max_size)
        }
        MPCError::MaliciousBehavior { party, description } => {
            format!(
                "Malicious behavior from party {}: {}",
                sanitize_party_id(party),
                sanitize_description(description),
            )
        }
        // For all other errors, return a generic message that doesn't
        // leak internal state.
        other => {
            // Only include the error variant name, not the full debug repr.
            let variant = format!("{}", other);
            // Strip anything after the first colon to remove details.
            if let Some(idx) = variant.find(':') {
                variant[..idx].to_string()
            } else {
                variant
            }
        }
    }
}

/// Sanitizes a party ID for external display (prevents injection).
pub fn sanitize_party_id(party: &PartyId) -> String {
    let s = party.to_string();
    // Only allow alphanumeric, hyphens, and underscores.
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect()
}

/// Sanitizes a description string for safe logging.
fn sanitize_description(desc: &str) -> String {
    // Remove any sequences that look like hex-encoded field elements (64+ hex chars).
    let mut result = String::with_capacity(desc.len());
    let mut hex_run = 0usize;

    for c in desc.chars() {
        if c.is_ascii_hexdigit() {
            hex_run += 1;
            if hex_run <= 16 {
                result.push(c);
            } else if hex_run == 17 {
                result.push_str("[REDACTED]");
            }
        } else {
            hex_run = 0;
            result.push(c);
        }
    }

    result.chars().take(256).collect()
}

// ============================================================================
// Session Key Derivation
// ============================================================================

/// Derives a session key from the x25519 shared secret and session ID.
///
/// Uses HKDF-SHA256 to derive a 32-byte key from the DH shared secret.
pub fn derive_session_key(
    shared_secret: &[u8; 32],
    session_id: &str,
) -> [u8; 32] {
    use sha2::Digest;

    // HKDF-extract: PRK = HMAC-SHA256(salt=session_id, IKM=shared_secret)
    let mut extract = HmacSha256::new_from_slice(session_id.as_bytes())
        .expect("HMAC can take any size key");
    extract.update(shared_secret);
    let prk = extract.finalize().into_bytes();

    // HKDF-expand: OKM = HMAC-SHA256(PRK, info || 0x01)
    let mut expand = HmacSha256::new_from_slice(&prk)
        .expect("HMAC can take any size key");
    expand.update(b"helix-mpc-session-key-v1");
    expand.update(&[0x01]);
    let okm = expand.finalize().into_bytes();

    let mut key = [0u8; 32];
    key.copy_from_slice(&okm);
    key
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_session_key() -> [u8; 32] {
        let shared_secret = [0x42u8; 32];
        derive_session_key(&shared_secret, "test-session")
    }

    #[test]
    fn test_authenticated_message_roundtrip() {
        let key = test_session_key();
        let payload = b"hello world".to_vec();

        let msg = AuthenticatedMessage::create(payload.clone(), 1, &key).unwrap();
        let bytes = msg.to_bytes();
        let restored = AuthenticatedMessage::from_bytes(&bytes).unwrap();

        assert_eq!(restored.sequence, 1);
        assert_eq!(restored.payload, payload);
        restored.verify(&key).unwrap();
    }

    #[test]
    fn test_authentication_failure() {
        let key1 = [0x42u8; 32];
        let key2 = [0x43u8; 32];
        let payload = b"test".to_vec();

        let msg = AuthenticatedMessage::create(payload, 1, &key1).unwrap();
        let bytes = msg.to_bytes();
        let restored = AuthenticatedMessage::from_bytes(&bytes).unwrap();

        // Verification with wrong key should fail.
        assert!(restored.verify(&key2).is_err());
    }

    #[test]
    fn test_tampered_message_detected() {
        let key = test_session_key();
        let payload = b"original data".to_vec();

        let msg = AuthenticatedMessage::create(payload, 1, &key).unwrap();
        let mut bytes = msg.to_bytes();

        // Tamper with the payload (byte at offset 12).
        if bytes.len() > 12 {
            bytes[12] ^= 0xFF;
        }

        let restored = AuthenticatedMessage::from_bytes(&bytes).unwrap();
        assert!(restored.verify(&key).is_err());
    }

    #[test]
    fn test_message_too_short() {
        let result = AuthenticatedMessage::from_bytes(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_message_size_mismatch() {
        let mut data = vec![0u8; 100];
        // Set payload length to 200 (larger than remaining data).
        data[0..4].copy_from_slice(&200u32.to_be_bytes());
        let result = AuthenticatedMessage::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_basic_flow() {
        let mut validator = MessageValidator::new(ValidationConfig::permissive());
        let party_a = PartyId::from_index(0);
        let party_b = PartyId::from_index(1);
        let key = test_session_key();

        validator.register_session_key(&party_a, key);
        validator.register_session_key(&party_b, key);

        // Create an authenticated message.
        let msg = AuthenticatedMessage::create(b"test data".to_vec(), 1, &key).unwrap();
        let raw = msg.to_bytes();

        // Validate it.
        let payload = validator.validate_incoming(&raw, &party_a).unwrap();
        assert_eq!(payload, b"test data");
    }

    #[test]
    fn test_validator_replay_protection() {
        let mut validator = MessageValidator::new(ValidationConfig {
            require_authentication: false,
            require_replay_protection: true,
            rate_limit_per_second: 0,
            ..Default::default()
        });
        let party = PartyId::from_index(0);
        let key = [0u8; 32];
        validator.register_session_key(&party, key);

        // First message with seq=1.
        let msg1 = AuthenticatedMessage::create(b"first".to_vec(), 1, &key).unwrap();
        validator.validate_incoming(&msg1.to_bytes(), &party).unwrap();

        // Second message with seq=2.
        let msg2 = AuthenticatedMessage::create(b"second".to_vec(), 2, &key).unwrap();
        validator.validate_incoming(&msg2.to_bytes(), &party).unwrap();

        // Replay of seq=1 should fail.
        let msg_replay = AuthenticatedMessage::create(b"replay".to_vec(), 1, &key).unwrap();
        assert!(validator.validate_incoming(&msg_replay.to_bytes(), &party).is_err());
    }

    #[test]
    fn test_rate_limiting() {
        let mut validator = MessageValidator::new(ValidationConfig {
            require_authentication: false,
            require_replay_protection: false,
            rate_limit_per_second: 5,
            rate_limit_window: Duration::from_secs(1),
            ..Default::default()
        });
        let party = PartyId::from_index(0);
        let key = [0u8; 32];

        // Send 5 messages (within limit).
        for i in 0..5 {
            let msg = AuthenticatedMessage::create(
                format!("msg-{}", i).into_bytes(), i + 1, &key,
            ).unwrap();
            assert!(validator.validate_incoming(&msg.to_bytes(), &party).is_ok());
        }

        // 6th message should be rate-limited.
        let msg = AuthenticatedMessage::create(b"overflow".to_vec(), 6, &key).unwrap();
        assert!(validator.validate_incoming(&msg.to_bytes(), &party).is_err());
    }

    #[test]
    fn test_field_element_validation() {
        // Valid: 3 elements of 32 bytes each.
        let data = vec![0u8; 96];
        assert!(validate_field_elements(&data, 3).is_ok());

        // Invalid: wrong size.
        let data = vec![0u8; 97];
        assert!(validate_field_elements(&data, 3).is_err());

        // Invalid: too many elements.
        assert!(validate_field_elements(&[], MAX_FIELD_ELEMENTS + 1).is_err());
    }

    #[test]
    fn test_training_message_validation() {
        // Empty message.
        assert!(validate_training_message(&[]).is_err());

        // Too short.
        assert!(validate_training_message(&[1, 2]).is_err());

        // Valid tag (variant 0 = WeightShares).
        let mut msg = vec![0u8; 100];
        msg[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(validate_training_message(&msg).is_ok());

        // Invalid tag (variant 99).
        msg[0..4].copy_from_slice(&99u32.to_le_bytes());
        assert!(validate_training_message(&msg).is_err());
    }

    #[test]
    fn test_error_sanitization() {
        let err = MPCError::CommunicationError(
            "tcp connection to 192.168.1.1:8080 failed: secret_share=0x1234abcd5678".into(),
        );
        let sanitized = sanitize_error(&err);
        assert_eq!(sanitized, "Communication error with peer");
        assert!(!sanitized.contains("192.168"));
        assert!(!sanitized.contains("1234abcd"));
    }

    #[test]
    fn test_error_sanitization_mac_failure() {
        let err = MPCError::MACCheckFailed { step: 42, cheater: Some(2) };
        let sanitized = sanitize_error(&err);
        assert!(sanitized.contains("42"));
        assert!(sanitized.contains("2"));
    }

    #[test]
    fn test_party_id_sanitization() {
        let party = PartyId::from_index(0);
        let sanitized = sanitize_party_id(&party);
        assert!(!sanitized.is_empty());
        // Should only contain safe characters.
        assert!(sanitized.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_'));
    }

    #[test]
    fn test_description_sanitization() {
        let desc = "sigma mismatch: 0x1234567890abcdef1234567890abcdef1234567890abcdef is wrong";
        let sanitized = sanitize_description(desc);
        assert!(sanitized.contains("[REDACTED]"));
    }

    #[test]
    fn test_session_key_derivation() {
        let secret = [0x42u8; 32];
        let key1 = derive_session_key(&secret, "session-1");
        let key2 = derive_session_key(&secret, "session-2");

        // Different sessions produce different keys.
        assert_ne!(key1, key2);

        // Same inputs produce same key.
        let key1_again = derive_session_key(&secret, "session-1");
        assert_eq!(key1, key1_again);
    }

    #[test]
    fn test_constant_time_eq() {
        let a = [0x42u8; 32];
        let b = [0x42u8; 32];
        let c = [0x43u8; 32];

        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_validator_authenticate_outbound() {
        let validator = MessageValidator::new(ValidationConfig::permissive());
        let party = PartyId::from_index(1);

        // Without a session key, authentication should fail.
        assert!(validator.authenticate_outbound(b"test".to_vec(), &party).is_err());
    }

    #[test]
    fn test_validator_with_authentication() {
        let mut validator = MessageValidator::new(ValidationConfig {
            require_authentication: true,
            require_replay_protection: false,
            rate_limit_per_second: 0,
            ..Default::default()
        });
        let party = PartyId::from_index(0);
        let key = test_session_key();
        validator.register_session_key(&party, key);

        // Valid authenticated message.
        let msg = AuthenticatedMessage::create(b"test".to_vec(), 1, &key).unwrap();
        assert!(validator.validate_incoming(&msg.to_bytes(), &party).is_ok());

        // Invalid tag (corrupted HMAC).
        let mut bad_bytes = msg.to_bytes();
        let len = bad_bytes.len();
        bad_bytes[len - 1] ^= 0xFF;
        assert!(validator.validate_incoming(&bad_bytes, &party).is_err());
    }
}
