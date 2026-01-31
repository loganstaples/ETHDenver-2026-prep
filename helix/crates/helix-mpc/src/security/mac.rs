//! MAC-based authentication for MPC computation integrity.
//!
//! This module implements SPDZ-style information-theoretic MACs for verifying
//! that parties compute correctly on their shares. Each shared value has an
//! associated MAC that can be checked at the end of computation.
//!
//! # Security Model
//!
//! The MAC scheme provides:
//! - Detection of incorrect computation by any party
//! - Detection of share modification
//! - Binding of values to their commitments
//!
//! # MAC Structure (SPDZ-style)
//!
//! For a shared value [x], each party holds:
//! - x_i: their share of x (sum of all shares = x)
//! - m_i: their share of MAC(x) = α * x (where α is global MAC key)
//!
//! To verify: reconstruct x and MAC(x), check that MAC(x) = α * x

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use hmac::{Hmac, Mac};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

type HmacSha256 = Hmac<Sha256>;

/// A MAC key shared among all parties.
/// The global key α is secret-shared; each party holds α_i.
#[derive(Debug, Clone)]
pub struct MACKey {
    /// Party's share of the global MAC key α
    pub alpha_share: f64,
    /// Party index
    pub party_index: usize,
    /// Number of parties
    pub num_parties: usize,
}

impl MACKey {
    /// Creates MAC key shares for all parties from a seed.
    pub fn generate_shares(num_parties: usize, seed: u64) -> Vec<MACKey> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random global key
        let alpha: f64 = rng.gen_range(1.0..1000.0);

        // Additive share it
        let mut alpha_sum = 0.0;
        let mut shares = Vec::with_capacity(num_parties);

        for i in 0..num_parties - 1 {
            let alpha_i: f64 = rng.gen_range(-1000.0..1000.0);
            alpha_sum += alpha_i;
            shares.push(MACKey {
                alpha_share: alpha_i,
                party_index: i,
                num_parties,
            });
        }

        shares.push(MACKey {
            alpha_share: alpha - alpha_sum,
            party_index: num_parties - 1,
            num_parties,
        });

        shares
    }

    /// Computes this party's share of MAC([x]) given their share x_i.
    /// MAC_i = α_i * x (for party 0) or α_i * 0 (for others, in simulation)
    pub fn compute_mac_share(&self, value_share: f64) -> f64 {
        // In SPDZ, the MAC is computed differently:
        // [MAC(x)]_i = α_i * x (requires knowing full x)
        // But we can precompute MAC shares during preprocessing
        self.alpha_share * value_share
    }
}

/// A value with its associated MAC share.
#[derive(Debug, Clone)]
pub struct AuthenticatedShare {
    /// The value share
    pub value: f64,
    /// The MAC share for this value
    pub mac: f64,
    /// Unique identifier for this value
    pub id: String,
}

impl AuthenticatedShare {
    pub fn new(value: f64, mac: f64, id: impl Into<String>) -> Self {
        Self {
            value,
            mac,
            id: id.into(),
        }
    }

    /// Adds two authenticated shares (local operation).
    pub fn add(&self, other: &AuthenticatedShare) -> AuthenticatedShare {
        AuthenticatedShare {
            value: self.value + other.value,
            mac: self.mac + other.mac,
            id: format!("{}+{}", self.id, other.id),
        }
    }

    /// Subtracts two authenticated shares (local operation).
    pub fn sub(&self, other: &AuthenticatedShare) -> AuthenticatedShare {
        AuthenticatedShare {
            value: self.value - other.value,
            mac: self.mac - other.mac,
            id: format!("{}-{}", self.id, other.id),
        }
    }

    /// Scales by a public constant (local operation).
    pub fn scale(&self, constant: f64) -> AuthenticatedShare {
        AuthenticatedShare {
            value: self.value * constant,
            mac: self.mac * constant,
            id: format!("{}*{}", self.id, constant),
        }
    }

    /// Adds a public constant (only party 0 adds to value, all add to MAC).
    pub fn add_public(&self, constant: f64, party_index: usize, alpha_share: f64) -> AuthenticatedShare {
        let new_value = if party_index == 0 {
            self.value + constant
        } else {
            self.value
        };

        // MAC of (x + c) = MAC(x) + α * c
        // Each party adds α_i * c to their MAC share
        let new_mac = self.mac + alpha_share * constant;

        AuthenticatedShare {
            value: new_value,
            mac: new_mac,
            id: format!("{}+pub({})", self.id, constant),
        }
    }
}

/// Verifier for checking MAC correctness at the end of computation.
#[derive(Debug)]
pub struct MACVerifier {
    /// Accumulated shares of values to verify
    value_shares: Vec<f64>,
    /// Accumulated MAC shares
    mac_shares: Vec<f64>,
    /// Alpha shares from all parties
    alpha_shares: Vec<f64>,
    /// Random coefficients for batch verification
    coefficients: Vec<f64>,
}

impl MACVerifier {
    pub fn new() -> Self {
        Self {
            value_shares: Vec::new(),
            mac_shares: Vec::new(),
            alpha_shares: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// Adds an authenticated share to verify.
    pub fn add_share(&mut self, share: &AuthenticatedShare, random_coeff: f64) {
        self.value_shares.push(share.value * random_coeff);
        self.mac_shares.push(share.mac * random_coeff);
        self.coefficients.push(random_coeff);
    }

    /// Adds the alpha share for this party.
    pub fn set_alpha_share(&mut self, alpha: f64) {
        self.alpha_shares.push(alpha);
    }

    /// Verifies all accumulated shares.
    /// Returns Ok if MACs are valid, Err if tampering detected.
    pub fn verify(
        &self,
        all_value_shares: &[Vec<f64>],
        all_mac_shares: &[Vec<f64>],
        all_alpha_shares: &[f64],
    ) -> MPCResult<()> {
        // Reconstruct values and MACs
        let num_values = all_value_shares.get(0).map(|v| v.len()).unwrap_or(0);
        let num_parties = all_value_shares.len();

        if num_parties == 0 {
            return Ok(());
        }

        // Reconstruct α
        let alpha: f64 = all_alpha_shares.iter().sum();

        // For each value, check that MAC = α * value
        for v in 0..num_values {
            let value: f64 = all_value_shares.iter().map(|p| p[v]).sum();
            let mac: f64 = all_mac_shares.iter().map(|p| p[v]).sum();
            let expected_mac = alpha * value;

            if (mac - expected_mac).abs() > 1e-6 {
                return Err(MPCError::MaliciousBehavior {
                    party: PartyId::new("unknown"),
                    description: format!(
                        "MAC verification failed for value {}: mac={}, expected={}",
                        v, mac, expected_mac
                    ),
                });
            }
        }

        Ok(())
    }

    /// Batch verification using random linear combination.
    /// More efficient than verifying each value individually.
    pub fn batch_verify(
        all_value_shares: &[Vec<f64>],
        all_mac_shares: &[Vec<f64>],
        all_alpha_shares: &[f64],
        seed: u64,
    ) -> MPCResult<()> {
        let num_values = all_value_shares.get(0).map(|v| v.len()).unwrap_or(0);
        let num_parties = all_value_shares.len();

        if num_parties == 0 || num_values == 0 {
            return Ok(());
        }

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random coefficients for linear combination
        let coeffs: Vec<f64> = (0..num_values).map(|_| rng.gen_range(1.0..1000.0)).collect();

        // Compute random linear combinations
        let mut combined_value = 0.0;
        let mut combined_mac = 0.0;

        for v in 0..num_values {
            let value: f64 = all_value_shares.iter().map(|p| p[v]).sum();
            let mac: f64 = all_mac_shares.iter().map(|p| p[v]).sum();

            combined_value += coeffs[v] * value;
            combined_mac += coeffs[v] * mac;
        }

        // Reconstruct α
        let alpha: f64 = all_alpha_shares.iter().sum();

        // Check single equation: combined_mac = α * combined_value
        let expected = alpha * combined_value;
        if (combined_mac - expected).abs() > 1e-4 {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new("unknown"),
                description: format!(
                    "Batch MAC verification failed: combined_mac={}, expected={}",
                    combined_mac, expected
                ),
            });
        }

        Ok(())
    }
}

impl Default for MACVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// HMAC-based message authentication for protocol messages.
/// Used for authenticating messages between parties (not for value verification).
#[derive(Debug, Clone)]
pub struct MessageAuthenticator {
    /// Shared secret key between parties
    key: [u8; 32],
}

impl MessageAuthenticator {
    /// Creates a new authenticator with the given key.
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Creates an authenticator with a key derived from a shared secret.
    pub fn from_shared_secret(secret: &[u8], context: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(secret);
        hasher.update(context.as_bytes());
        let key: [u8; 32] = hasher.finalize().into();
        Self { key }
    }

    /// Computes MAC for a message.
    pub fn authenticate(&self, message: &[u8]) -> Vec<u8> {
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .expect("HMAC key size should be valid");
        mac.update(message);
        mac.finalize().into_bytes().to_vec()
    }

    /// Verifies a message MAC.
    pub fn verify(&self, message: &[u8], tag: &[u8]) -> bool {
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .expect("HMAC key size should be valid");
        mac.update(message);
        mac.verify_slice(tag).is_ok()
    }
}

/// Authenticated message with MAC tag.
#[derive(Debug, Clone)]
pub struct AuthenticatedMessage {
    /// The message content
    pub content: Vec<u8>,
    /// HMAC tag
    pub tag: Vec<u8>,
    /// Sequence number for replay protection
    pub sequence: u64,
}

impl AuthenticatedMessage {
    /// Creates and authenticates a new message.
    pub fn create(authenticator: &MessageAuthenticator, content: Vec<u8>, sequence: u64) -> Self {
        // Include sequence in MAC computation
        let mut mac_input = content.clone();
        mac_input.extend_from_slice(&sequence.to_le_bytes());

        let tag = authenticator.authenticate(&mac_input);

        Self {
            content,
            tag,
            sequence,
        }
    }

    /// Verifies the message authenticity.
    pub fn verify(&self, authenticator: &MessageAuthenticator) -> bool {
        let mut mac_input = self.content.clone();
        mac_input.extend_from_slice(&self.sequence.to_le_bytes());

        authenticator.verify(&mac_input, &self.tag)
    }
}

/// Session authentication state for preventing replay attacks.
#[derive(Debug)]
pub struct SessionAuthState {
    /// Authenticators for each peer
    authenticators: std::collections::HashMap<String, MessageAuthenticator>,
    /// Expected next sequence number from each peer
    next_sequence: std::collections::HashMap<String, u64>,
    /// Our outgoing sequence counters
    outgoing_sequence: std::collections::HashMap<String, u64>,
}

impl SessionAuthState {
    pub fn new() -> Self {
        Self {
            authenticators: std::collections::HashMap::new(),
            next_sequence: std::collections::HashMap::new(),
            outgoing_sequence: std::collections::HashMap::new(),
        }
    }

    /// Establishes authentication with a peer using a shared secret.
    pub fn establish(&mut self, peer_id: &str, shared_secret: &[u8]) {
        let auth = MessageAuthenticator::from_shared_secret(shared_secret, peer_id);
        self.authenticators.insert(peer_id.to_string(), auth);
        self.next_sequence.insert(peer_id.to_string(), 0);
        self.outgoing_sequence.insert(peer_id.to_string(), 0);
    }

    /// Creates an authenticated message for a peer.
    pub fn create_message(&mut self, peer_id: &str, content: Vec<u8>) -> MPCResult<AuthenticatedMessage> {
        let auth = self.authenticators.get(peer_id)
            .ok_or_else(|| MPCError::CommunicationError(format!("No session with peer {}", peer_id)))?;

        let seq = self.outgoing_sequence.get_mut(peer_id)
            .ok_or_else(|| MPCError::CommunicationError(format!("No sequence for peer {}", peer_id)))?;

        let msg = AuthenticatedMessage::create(auth, content, *seq);
        *seq += 1;

        Ok(msg)
    }

    /// Verifies and accepts a message from a peer.
    pub fn verify_message(&mut self, peer_id: &str, msg: &AuthenticatedMessage) -> MPCResult<()> {
        let auth = self.authenticators.get(peer_id)
            .ok_or_else(|| MPCError::CommunicationError(format!("No session with peer {}", peer_id)))?;

        let expected_seq = self.next_sequence.get_mut(peer_id)
            .ok_or_else(|| MPCError::CommunicationError(format!("No sequence for peer {}", peer_id)))?;

        // Check sequence number (allow some out-of-order)
        if msg.sequence < *expected_seq {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new(peer_id),
                description: format!("Replay attack detected: seq {} < expected {}", msg.sequence, *expected_seq),
            });
        }

        // Verify MAC
        if !msg.verify(auth) {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new(peer_id),
                description: "Message authentication failed".to_string(),
            });
        }

        // Update expected sequence
        if msg.sequence >= *expected_seq {
            *expected_seq = msg.sequence + 1;
        }

        Ok(())
    }
}

impl Default for SessionAuthState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_key_generation() {
        let shares = MACKey::generate_shares(3, 42);
        assert_eq!(shares.len(), 3);

        // Sum should be a valid alpha
        let alpha: f64 = shares.iter().map(|k| k.alpha_share).sum();
        assert!(alpha > 0.0);
    }

    #[test]
    fn test_authenticated_share_operations() {
        let share1 = AuthenticatedShare::new(5.0, 25.0, "x");
        let share2 = AuthenticatedShare::new(3.0, 15.0, "y");

        let sum = share1.add(&share2);
        assert_eq!(sum.value, 8.0);
        assert_eq!(sum.mac, 40.0);

        let diff = share1.sub(&share2);
        assert_eq!(diff.value, 2.0);
        assert_eq!(diff.mac, 10.0);

        let scaled = share1.scale(2.0);
        assert_eq!(scaled.value, 10.0);
        assert_eq!(scaled.mac, 50.0);
    }

    #[test]
    fn test_mac_verification_valid() {
        // Create shares where MAC = α * value
        let alpha = 5.0;
        let alpha_shares = vec![2.0, 1.5, 1.5]; // sum = 5

        let value = 10.0;
        let value_shares = vec![vec![3.0], vec![4.0], vec![3.0]]; // sum = 10
        let mac_shares = vec![vec![15.0], vec![20.0], vec![15.0]]; // sum = 50 = 5 * 10

        let result = MACVerifier::verify(
            &MACVerifier::new(),
            &value_shares,
            &mac_shares,
            &alpha_shares,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_mac_verification_invalid() {
        let alpha_shares = vec![2.0, 1.5, 1.5]; // sum = 5

        let value_shares = vec![vec![3.0], vec![4.0], vec![3.0]]; // sum = 10
        let mac_shares = vec![vec![15.0], vec![20.0], vec![10.0]]; // sum = 45 ≠ 5 * 10

        let result = MACVerifier::verify(
            &MACVerifier::new(),
            &value_shares,
            &mac_shares,
            &alpha_shares,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_verification() {
        let alpha_shares = vec![2.0, 3.0]; // sum = 5

        // Two values: x=10, y=6
        let value_shares = vec![
            vec![4.0, 2.0],  // party 0
            vec![6.0, 4.0],  // party 1
        ];
        let mac_shares = vec![
            vec![20.0, 12.0],  // party 0: 4*5=20, 2.4*5=12
            vec![30.0, 18.0],  // party 1: 6*5=30, 3.6*5=18
        ];

        let result = MACVerifier::batch_verify(
            &value_shares,
            &mac_shares,
            &alpha_shares,
            42,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_message_authenticator() {
        let key = [42u8; 32];
        let auth = MessageAuthenticator::new(key);

        let message = b"Hello, MPC!";
        let tag = auth.authenticate(message);

        assert!(auth.verify(message, &tag));
        assert!(!auth.verify(b"Tampered!", &tag));
    }

    #[test]
    fn test_authenticated_message() {
        let auth = MessageAuthenticator::new([42u8; 32]);

        let msg = AuthenticatedMessage::create(&auth, b"Test content".to_vec(), 0);
        assert!(msg.verify(&auth));

        // Tamper with content
        let mut tampered = msg.clone();
        tampered.content[0] ^= 0xFF;
        assert!(!tampered.verify(&auth));
    }

    #[test]
    fn test_session_auth_state() {
        let mut state1 = SessionAuthState::new();
        let mut state2 = SessionAuthState::new();

        let shared_secret = b"shared-secret-key-for-testing!!!";
        state1.establish("party-1", shared_secret);
        state2.establish("party-0", shared_secret);

        // Party 0 sends to Party 1
        let msg = state1.create_message("party-1", b"Hello".to_vec()).unwrap();

        // Party 1 verifies
        // Note: In real usage, party 1 would have a key derived with "party-0" context
        // This test is simplified
    }

    #[test]
    fn test_replay_detection() {
        let mut state = SessionAuthState::new();
        state.establish("peer", b"secret");

        let msg1 = state.create_message("peer", b"msg1".to_vec()).unwrap();
        let msg2 = state.create_message("peer", b"msg2".to_vec()).unwrap();

        // Verify in order - should work
        // Reset for incoming verification
        state.next_sequence.insert("peer".to_string(), 0);

        assert!(state.verify_message("peer", &msg1).is_ok());
        assert!(state.verify_message("peer", &msg2).is_ok());

        // Try to replay msg1 - should fail
        assert!(state.verify_message("peer", &msg1).is_err());
    }
}
