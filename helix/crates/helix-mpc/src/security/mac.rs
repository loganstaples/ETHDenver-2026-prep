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
//!
//! All arithmetic is performed in the BN254 scalar field (Fr) for
//! cryptographic soundness — no floating-point tolerance.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use hmac::{Hmac, Mac};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::types::PartyId;

type HmacSha256 = Hmac<Sha256>;

/// A MAC key shared among all parties.
/// The global key α is secret-shared; each party holds α_i.
#[derive(Debug, Clone)]
pub struct MACKey {
    /// Party's share of the global MAC key α (in Fr)
    pub alpha_share: Fr,
    /// Party index
    pub party_index: usize,
    /// Number of parties
    pub num_parties: usize,
}

impl MACKey {
    /// Creates MAC key shares for all parties from a seed.
    pub fn generate_shares(num_parties: usize, seed: u64) -> Vec<MACKey> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random global key α in Fr
        let alpha = Fr::random(&mut rng);

        // Additive share it
        let mut alpha_sum = Fr::ZERO;
        let mut shares = Vec::with_capacity(num_parties);

        for i in 0..num_parties - 1 {
            let alpha_i = Fr::random(&mut rng);
            alpha_sum = Fr::add(&alpha_sum, &alpha_i);
            shares.push(MACKey {
                alpha_share: alpha_i,
                party_index: i,
                num_parties,
            });
        }

        shares.push(MACKey {
            alpha_share: Fr::sub(&alpha, &alpha_sum),
            party_index: num_parties - 1,
            num_parties,
        });

        shares
    }

    /// Computes this party's share of MAC([x]) given their share x_i.
    /// MAC_i = α_i * x_i
    pub fn compute_mac_share(&self, value_share: &Fr) -> Fr {
        Fr::mul(&self.alpha_share, value_share)
    }
}

/// A value with its associated MAC share, using Fr field elements.
#[derive(Debug, Clone)]
pub struct AuthenticatedShare {
    /// The value share (in Fr)
    pub value: Fr,
    /// The MAC share for this value (in Fr)
    pub mac: Fr,
    /// Unique identifier for this value
    pub id: String,
}

impl AuthenticatedShare {
    pub fn new(value: Fr, mac: Fr, id: impl Into<String>) -> Self {
        Self {
            value,
            mac,
            id: id.into(),
        }
    }

    /// Adds two authenticated shares (local operation).
    pub fn add(&self, other: &AuthenticatedShare) -> AuthenticatedShare {
        AuthenticatedShare {
            value: Fr::add(&self.value, &other.value),
            mac: Fr::add(&self.mac, &other.mac),
            id: format!("{}+{}", self.id, other.id),
        }
    }

    /// Subtracts two authenticated shares (local operation).
    pub fn sub(&self, other: &AuthenticatedShare) -> AuthenticatedShare {
        AuthenticatedShare {
            value: Fr::sub(&self.value, &other.value),
            mac: Fr::sub(&self.mac, &other.mac),
            id: format!("{}-{}", self.id, other.id),
        }
    }

    /// Scales by a public constant (local operation).
    pub fn scale(&self, constant: &Fr) -> AuthenticatedShare {
        AuthenticatedShare {
            value: Fr::mul(&self.value, constant),
            mac: Fr::mul(&self.mac, constant),
            id: format!("{}*c", self.id),
        }
    }

    /// Adds a public constant (only party 0 adds to value, all add to MAC).
    pub fn add_public(&self, constant: &Fr, party_index: usize, alpha_share: &Fr) -> AuthenticatedShare {
        let new_value = if party_index == 0 {
            Fr::add(&self.value, constant)
        } else {
            self.value
        };

        // MAC of (x + c) = MAC(x) + α * c
        // Each party adds α_i * c to their MAC share
        let new_mac = Fr::add(&self.mac, &Fr::mul(alpha_share, constant));

        AuthenticatedShare {
            value: new_value,
            mac: new_mac,
            id: format!("{}+pub", self.id),
        }
    }
}

/// Verifier for checking MAC correctness at the end of computation.
/// Uses exact Fr arithmetic — no tolerance needed.
#[derive(Debug)]
pub struct MACVerifier {
    /// Accumulated shares of values to verify
    value_shares: Vec<Fr>,
    /// Accumulated MAC shares
    mac_shares: Vec<Fr>,
    /// Alpha shares from all parties
    alpha_shares: Vec<Fr>,
    /// Random coefficients for batch verification
    coefficients: Vec<Fr>,
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
    /// The random_coeff is a raw field element (not fixed-point), used as an abstract scaling factor.
    pub fn add_share(&mut self, share: &AuthenticatedShare, random_coeff: &Fr) {
        self.value_shares.push(Fr::mul(&share.value, random_coeff));
        self.mac_shares.push(Fr::mul(&share.mac, random_coeff));
        self.coefficients.push(*random_coeff);
    }

    /// Adds the alpha share for this party.
    pub fn set_alpha_share(&mut self, alpha: Fr) {
        self.alpha_shares.push(alpha);
    }

    /// Verifies all accumulated shares.
    /// Returns Ok if MACs are valid, Err if tampering detected.
    /// Uses exact Fr equality — no floating-point tolerance.
    pub fn verify(
        &self,
        all_value_shares: &[Vec<Fr>],
        all_mac_shares: &[Vec<Fr>],
        all_alpha_shares: &[Fr],
    ) -> MPCResult<()> {
        let num_values = all_value_shares.first().map(|v| v.len()).unwrap_or(0);
        let num_parties = all_value_shares.len();

        if num_parties == 0 {
            return Ok(());
        }

        // Reconstruct α
        let mut alpha = Fr::ZERO;
        for a in all_alpha_shares {
            alpha = Fr::add(&alpha, a);
        }

        // For each value, check that MAC = α * value (exact)
        for v in 0..num_values {
            let mut value = Fr::ZERO;
            let mut mac = Fr::ZERO;
            for p in 0..num_parties {
                value = Fr::add(&value, &all_value_shares[p][v]);
                mac = Fr::add(&mac, &all_mac_shares[p][v]);
            }
            let expected_mac = Fr::mul(&alpha, &value);

            if !mac.ct_eq(&expected_mac).to_bool() {
                return Err(MPCError::MaliciousBehavior {
                    party: PartyId::new("unknown"),
                    description: format!(
                        "MAC verification failed for value {}",
                        v,
                    ),
                });
            }
        }

        Ok(())
    }

    /// Batch verification using random linear combination.
    /// More efficient than verifying each value individually.
    pub fn batch_verify(
        all_value_shares: &[Vec<Fr>],
        all_mac_shares: &[Vec<Fr>],
        all_alpha_shares: &[Fr],
        seed: u64,
    ) -> MPCResult<()> {
        let num_values = all_value_shares.first().map(|v| v.len()).unwrap_or(0);
        let num_parties = all_value_shares.len();

        if num_parties == 0 || num_values == 0 {
            return Ok(());
        }

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random coefficients for linear combination
        let coeffs: Vec<Fr> = (0..num_values).map(|_| Fr::random(&mut rng)).collect();

        // Compute random linear combinations
        let mut combined_value = Fr::ZERO;
        let mut combined_mac = Fr::ZERO;

        for v in 0..num_values {
            let mut value = Fr::ZERO;
            let mut mac = Fr::ZERO;
            for p in 0..num_parties {
                value = Fr::add(&value, &all_value_shares[p][v]);
                mac = Fr::add(&mac, &all_mac_shares[p][v]);
            }

            // Use raw field multiplication for random coefficients (not fixed-point)
            // since coefficients are abstract scaling factors, not fixed-point values
            combined_value = Fr::add(&combined_value, &Fr::mul(&coeffs[v], &value));
            combined_mac = Fr::add(&combined_mac, &Fr::mul(&coeffs[v], &mac));
        }

        // Reconstruct α
        let mut alpha = Fr::ZERO;
        for a in all_alpha_shares {
            alpha = Fr::add(&alpha, a);
        }

        // Check single equation: combined_mac = α * combined_value (exact)
        let expected = Fr::mul(&alpha, &combined_value);
        if !combined_mac.ct_eq(&expected).to_bool() {
            return Err(MPCError::MaliciousBehavior {
                party: PartyId::new("unknown"),
                description: "Batch MAC verification failed".to_string(),
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

        // Sum should reconstruct alpha
        let mut alpha = Fr::ZERO;
        for k in &shares {
            alpha = Fr::add(&alpha, &k.alpha_share);
        }
        // alpha should be nonzero
        assert!(!alpha.ct_eq(&Fr::ZERO).to_bool());
    }

    #[test]
    fn test_authenticated_share_operations() {
        // Using raw field elements (Fr::from_u64) for consistent Fr::mul semantics
        let v5 = Fr::from_u64(5);
        let m25 = Fr::from_u64(25);  // alpha=5, mac=5*5=25
        let v3 = Fr::from_u64(3);
        let m15 = Fr::from_u64(15);  // alpha=5, mac=5*3=15

        let share1 = AuthenticatedShare::new(v5, m25, "x");
        let share2 = AuthenticatedShare::new(v3, m15, "y");

        // add: value=5+3=8, mac=25+15=40
        let sum = share1.add(&share2);
        assert_eq!(sum.value.to_u64(), Some(8));
        assert_eq!(sum.mac.to_u64(), Some(40));

        // sub: value=5-3=2, mac=25-15=10
        let diff = share1.sub(&share2);
        assert_eq!(diff.value.to_u64(), Some(2));
        assert_eq!(diff.mac.to_u64(), Some(10));

        // scale by 2: value=5*2=10, mac=25*2=50
        let scaled = share1.scale(&Fr::from_u64(2));
        assert_eq!(scaled.value.to_u64(), Some(10));
        assert_eq!(scaled.mac.to_u64(), Some(50));
    }

    #[test]
    fn test_mac_verification_valid() {
        // alpha = 5, value = 10, MAC = alpha * value = 50 (raw field mul)
        let alpha_shares = vec![Fr::from_u64(2), Fr::from_u64(1), Fr::from_u64(2)];

        let value_shares = vec![
            vec![Fr::from_u64(3)],
            vec![Fr::from_u64(4)],
            vec![Fr::from_u64(3)],
        ];
        // MAC shares must sum to alpha * value = 5 * 10 = 50 (raw field mul)
        let mac_shares = vec![
            vec![Fr::from_u64(15)],
            vec![Fr::from_u64(20)],
            vec![Fr::from_u64(15)],
        ];

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
        let alpha_shares = vec![Fr::from_u64(2), Fr::from_u64(1), Fr::from_u64(2)];

        let value_shares = vec![
            vec![Fr::from_u64(3)],
            vec![Fr::from_u64(4)],
            vec![Fr::from_u64(3)],
        ];
        // Bad MAC: sum = 45, not 50
        let mac_shares = vec![
            vec![Fr::from_u64(15)],
            vec![Fr::from_u64(20)],
            vec![Fr::from_u64(10)],
        ];

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
        let alpha_shares = vec![Fr::from_u64(2), Fr::from_u64(3)]; // alpha = 5

        // Two values: x=10, y=6
        let value_shares = vec![
            vec![Fr::from_u64(4), Fr::from_u64(2)],
            vec![Fr::from_u64(6), Fr::from_u64(4)],
        ];
        // MACs: alpha*x = 5*10 = 50, alpha*y = 5*6 = 30 (raw field mul)
        let mac_shares = vec![
            vec![Fr::from_u64(20), Fr::from_u64(12)],
            vec![Fr::from_u64(30), Fr::from_u64(18)],
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
        let mut _state2 = SessionAuthState::new();

        let shared_secret = b"shared-secret-key-for-testing!!!";
        state1.establish("party-1", shared_secret);
        _state2.establish("party-0", shared_secret);

        // Party 0 sends to Party 1
        let _msg = state1.create_message("party-1", b"Hello".to_vec()).unwrap();
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
