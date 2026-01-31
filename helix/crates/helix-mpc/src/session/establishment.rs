//! Secure session establishment protocol for MPC.
//!
//! This module implements the cryptographic protocol for establishing
//! a secure MPC session between parties:
//!
//! 1. **Key Exchange**: Diffie-Hellman key agreement between all pairs
//! 2. **Authentication**: Signature-based authentication with public keys
//! 3. **Session Key Derivation**: Derive session-specific encryption keys
//! 4. **Commitment Phase**: Exchange commitments to random values
//! 5. **Session ID Agreement**: All parties agree on session identifier
//!
//! # Security Properties
//!
//! - Forward secrecy (ephemeral keys)
//! - Mutual authentication
//! - Replay protection via session ID
//! - Man-in-the-middle protection

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// State of session establishment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    /// Initial state, no messages exchanged
    Initial,
    /// Sent our key exchange message
    KeyExchangeSent,
    /// Received key exchange from all parties
    KeyExchangeComplete,
    /// Sent authentication message
    AuthenticationSent,
    /// Verified all parties
    AuthenticationComplete,
    /// Session established successfully
    Established,
    /// Session establishment failed
    Failed,
}

/// Configuration for session establishment.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Our party ID
    pub party_id: PartyId,
    /// All party IDs in the session
    pub all_parties: Vec<PartyId>,
    /// Timeout for establishment protocol
    pub timeout: Duration,
    /// Our long-term signing key
    pub signing_key: [u8; 32],
    /// Long-term public keys of all parties
    pub party_public_keys: HashMap<String, [u8; 32]>,
}

/// Key exchange message sent between parties.
#[derive(Debug, Clone)]
pub struct KeyExchangeMessage {
    /// Sender party ID
    pub from: PartyId,
    /// Ephemeral DH public key
    pub ephemeral_public: [u8; 32],
    /// Commitment to random contribution
    pub random_commitment: [u8; 32],
    /// Timestamp for freshness
    pub timestamp: u64,
    /// Signature over the message
    pub signature: [u8; 64],
}

impl KeyExchangeMessage {
    /// Creates a new key exchange message.
    pub fn new(
        from: PartyId,
        ephemeral_public: [u8; 32],
        random_contribution: &[u8; 32],
        signing_key: &SigningKey,
    ) -> Self {
        // Compute commitment
        let mut hasher = Sha256::new();
        hasher.update(random_contribution);
        let random_commitment: [u8; 32] = hasher.finalize().into();

        // Get timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Compute signature
        let mut sign_data = Vec::new();
        sign_data.extend_from_slice(from.0.as_bytes());
        sign_data.extend_from_slice(&ephemeral_public);
        sign_data.extend_from_slice(&random_commitment);
        sign_data.extend_from_slice(&timestamp.to_le_bytes());

        let signature = signing_key.sign(&sign_data);

        Self {
            from,
            ephemeral_public,
            random_commitment,
            timestamp,
            signature: signature.to_bytes(),
        }
    }

    /// Verifies the message signature.
    pub fn verify(&self, verifying_key: &VerifyingKey) -> bool {
        let mut sign_data = Vec::new();
        sign_data.extend_from_slice(self.from.0.as_bytes());
        sign_data.extend_from_slice(&self.ephemeral_public);
        sign_data.extend_from_slice(&self.random_commitment);
        sign_data.extend_from_slice(&self.timestamp.to_le_bytes());

        let signature = Signature::from_bytes(&self.signature);

        verifying_key.verify(&sign_data, &signature).is_ok()
    }

    /// Checks if the message is fresh (not too old).
    pub fn is_fresh(&self, max_age_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now.saturating_sub(self.timestamp) <= max_age_secs
    }
}

/// Authentication message with revealed random value.
#[derive(Debug, Clone)]
pub struct AuthenticationMessage {
    /// Sender party ID
    pub from: PartyId,
    /// Revealed random contribution
    pub random_value: [u8; 32],
    /// Session ID we computed
    pub session_id: [u8; 32],
    /// Signature
    pub signature: [u8; 64],
}

impl AuthenticationMessage {
    pub fn new(
        from: PartyId,
        random_value: [u8; 32],
        session_id: [u8; 32],
        signing_key: &SigningKey,
    ) -> Self {
        let mut sign_data = Vec::new();
        sign_data.extend_from_slice(from.0.as_bytes());
        sign_data.extend_from_slice(&random_value);
        sign_data.extend_from_slice(&session_id);

        let signature = signing_key.sign(&sign_data);

        Self {
            from,
            random_value,
            session_id,
            signature: signature.to_bytes(),
        }
    }

    pub fn verify(&self, verifying_key: &VerifyingKey) -> bool {
        let mut sign_data = Vec::new();
        sign_data.extend_from_slice(self.from.0.as_bytes());
        sign_data.extend_from_slice(&self.random_value);
        sign_data.extend_from_slice(&self.session_id);

        let signature = Signature::from_bytes(&self.signature);

        verifying_key.verify(&sign_data, &signature).is_ok()
    }
}

/// Established session with derived keys.
#[derive(Debug)]
pub struct EstablishedSession {
    /// Unique session ID
    pub session_id: [u8; 32],
    /// Party IDs in the session
    pub parties: Vec<PartyId>,
    /// Pairwise shared keys (party_id -> key)
    pub pairwise_keys: HashMap<String, [u8; 32]>,
    /// Global session key (from all contributions)
    pub session_key: [u8; 32],
    /// Timestamp when established
    pub established_at: Instant,
    /// Session expiry time
    pub expires_at: Instant,
}

impl EstablishedSession {
    /// Derives an encryption key for a specific purpose.
    pub fn derive_key(&self, purpose: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.session_key);
        hasher.update(purpose.as_bytes());
        hasher.finalize().into()
    }

    /// Gets the pairwise key for communicating with another party.
    pub fn pairwise_key(&self, party_id: &str) -> Option<&[u8; 32]> {
        self.pairwise_keys.get(party_id)
    }

    /// Checks if the session has expired.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Returns session age.
    pub fn age(&self) -> Duration {
        self.established_at.elapsed()
    }
}

/// Session establishment state machine.
pub struct SessionEstablishment {
    /// Our party ID
    party_id: PartyId,
    /// All parties in the session
    all_parties: Vec<PartyId>,
    /// Current phase
    phase: SessionPhase,
    /// Our ephemeral DH secret
    ephemeral_secret: StaticSecret,
    /// Our random contribution
    random_contribution: [u8; 32],
    /// Our signing key
    signing_key: SigningKey,
    /// Party public keys for verification
    party_public_keys: HashMap<String, VerifyingKey>,
    /// Received key exchange messages
    received_key_exchanges: HashMap<String, KeyExchangeMessage>,
    /// Received authentication messages
    received_authentications: HashMap<String, AuthenticationMessage>,
    /// Computed pairwise shared secrets
    shared_secrets: HashMap<String, [u8; 32]>,
    /// Session start time
    started_at: Instant,
    /// Timeout
    timeout: Duration,
}

impl SessionEstablishment {
    /// Creates a new session establishment.
    pub fn new(config: SessionConfig) -> MPCResult<Self> {
        let mut rng = ChaCha20Rng::from_entropy();

        // Generate ephemeral DH key
        let mut ephemeral_bytes = [0u8; 32];
        rng.fill_bytes(&mut ephemeral_bytes);
        let ephemeral_secret = StaticSecret::from(ephemeral_bytes);

        // Generate random contribution
        let mut random_contribution = [0u8; 32];
        rng.fill_bytes(&mut random_contribution);

        // Parse signing key
        let signing_key = SigningKey::from_bytes(&config.signing_key);

        // Parse party public keys
        let mut party_public_keys = HashMap::new();
        for (party_id, pk_bytes) in &config.party_public_keys {
            let verifying_key = VerifyingKey::from_bytes(pk_bytes)
                .map_err(|e| MPCError::InvalidConfig(format!("Invalid public key for {}: {:?}", party_id, e)))?;
            party_public_keys.insert(party_id.clone(), verifying_key);
        }

        Ok(Self {
            party_id: config.party_id,
            all_parties: config.all_parties,
            phase: SessionPhase::Initial,
            ephemeral_secret,
            random_contribution,
            signing_key,
            party_public_keys,
            received_key_exchanges: HashMap::new(),
            received_authentications: HashMap::new(),
            shared_secrets: HashMap::new(),
            started_at: Instant::now(),
            timeout: config.timeout,
        })
    }

    /// Generates our key exchange message.
    pub fn generate_key_exchange(&mut self) -> KeyExchangeMessage {
        let ephemeral_public = X25519PublicKey::from(&self.ephemeral_secret).to_bytes();

        let msg = KeyExchangeMessage::new(
            self.party_id.clone(),
            ephemeral_public,
            &self.random_contribution,
            &self.signing_key,
        );

        self.phase = SessionPhase::KeyExchangeSent;
        msg
    }

    /// Processes a received key exchange message.
    pub fn receive_key_exchange(&mut self, msg: KeyExchangeMessage) -> MPCResult<()> {
        // Check we're in the right phase
        if self.phase != SessionPhase::KeyExchangeSent && self.phase != SessionPhase::Initial {
            return Err(MPCError::ProtocolError("Unexpected key exchange message".into()));
        }

        // Verify signature
        if let Some(verifying_key) = self.party_public_keys.get(&msg.from.0) {
            if !msg.verify(verifying_key) {
                return Err(MPCError::MaliciousBehavior {
                    party: msg.from.clone(),
                    description: "Invalid signature on key exchange".into(),
                });
            }
        } else {
            return Err(MPCError::UnknownParty(msg.from.clone()));
        }

        // Check freshness
        if !msg.is_fresh(300) {
            return Err(MPCError::MaliciousBehavior {
                party: msg.from.clone(),
                description: "Stale key exchange message".into(),
            });
        }

        // Compute shared secret
        let their_public = X25519PublicKey::from(msg.ephemeral_public);
        let shared_secret = self.ephemeral_secret.diffie_hellman(&their_public);
        self.shared_secrets.insert(msg.from.0.clone(), *shared_secret.as_bytes());

        // Store message
        self.received_key_exchanges.insert(msg.from.0.clone(), msg);

        // Check if we have all key exchanges
        let expected = self.all_parties.len() - 1; // All except ourselves
        if self.received_key_exchanges.len() >= expected {
            self.phase = SessionPhase::KeyExchangeComplete;
        }

        Ok(())
    }

    /// Generates our authentication message.
    pub fn generate_authentication(&mut self) -> MPCResult<AuthenticationMessage> {
        if self.phase != SessionPhase::KeyExchangeComplete {
            return Err(MPCError::ProtocolError(
                "Key exchange not complete".into(),
            ));
        }

        // Compute session ID from all commitments
        let session_id = self.compute_session_id();

        let msg = AuthenticationMessage::new(
            self.party_id.clone(),
            self.random_contribution,
            session_id,
            &self.signing_key,
        );

        self.phase = SessionPhase::AuthenticationSent;
        Ok(msg)
    }

    /// Processes a received authentication message.
    pub fn receive_authentication(&mut self, msg: AuthenticationMessage) -> MPCResult<()> {
        if self.phase != SessionPhase::AuthenticationSent && self.phase != SessionPhase::KeyExchangeComplete {
            return Err(MPCError::ProtocolError(
                "Unexpected authentication message".into(),
            ));
        }

        // Verify signature
        if let Some(verifying_key) = self.party_public_keys.get(&msg.from.0) {
            if !msg.verify(verifying_key) {
                return Err(MPCError::MaliciousBehavior {
                    party: msg.from.clone(),
                    description: "Invalid signature on authentication".into(),
                });
            }
        } else {
            return Err(MPCError::UnknownParty(msg.from.clone()));
        }

        // Verify the commitment opens correctly
        if let Some(ke_msg) = self.received_key_exchanges.get(&msg.from.0) {
            let mut hasher = Sha256::new();
            hasher.update(&msg.random_value);
            let expected_commitment: [u8; 32] = hasher.finalize().into();

            if expected_commitment != ke_msg.random_commitment {
                return Err(MPCError::MaliciousBehavior {
                    party: msg.from.clone(),
                    description: "Random value doesn't match commitment".into(),
                });
            }
        } else {
            return Err(MPCError::ProtocolError(
                format!("No key exchange from {}", msg.from),
            ));
        }

        // Verify session ID matches
        let our_session_id = self.compute_session_id();
        if msg.session_id != our_session_id {
            return Err(MPCError::MaliciousBehavior {
                party: msg.from.clone(),
                description: "Session ID mismatch".into(),
            });
        }

        // Store message
        self.received_authentications.insert(msg.from.0.clone(), msg);

        // Check if complete
        let expected = self.all_parties.len() - 1;
        if self.received_authentications.len() >= expected {
            self.phase = SessionPhase::AuthenticationComplete;
        }

        Ok(())
    }

    /// Computes the session ID from all commitments.
    fn compute_session_id(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Include all parties in sorted order for determinism
        let mut party_ids: Vec<&String> = self
            .received_key_exchanges
            .keys()
            .chain(std::iter::once(&self.party_id.0))
            .collect();
        party_ids.sort();

        for party_id in &party_ids {
            hasher.update(party_id.as_bytes());
            if *party_id == &self.party_id.0 {
                hasher.update(&self.random_contribution);
            } else if let Some(ke) = self.received_key_exchanges.get(*party_id) {
                hasher.update(&ke.random_commitment);
            }
        }

        hasher.finalize().into()
    }

    /// Finalizes the session establishment.
    pub fn finalize(&mut self, session_duration: Duration) -> MPCResult<EstablishedSession> {
        if self.phase != SessionPhase::AuthenticationComplete {
            return Err(MPCError::ProtocolError(
                "Authentication not complete".into(),
            ));
        }

        let session_id = self.compute_session_id();

        // Derive session key from all random values
        let mut hasher = Sha256::new();
        hasher.update(&session_id);
        hasher.update(&self.random_contribution);
        for auth in self.received_authentications.values() {
            hasher.update(&auth.random_value);
        }
        let session_key: [u8; 32] = hasher.finalize().into();

        // Derive pairwise keys
        let mut pairwise_keys = HashMap::new();
        for (party_id, shared_secret) in &self.shared_secrets {
            let mut hasher = Sha256::new();
            hasher.update(shared_secret);
            hasher.update(&session_id);
            let key: [u8; 32] = hasher.finalize().into();
            pairwise_keys.insert(party_id.clone(), key);
        }

        self.phase = SessionPhase::Established;

        Ok(EstablishedSession {
            session_id,
            parties: self.all_parties.clone(),
            pairwise_keys,
            session_key,
            established_at: Instant::now(),
            expires_at: Instant::now() + session_duration,
        })
    }

    /// Returns the current phase.
    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    /// Checks if the establishment has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.started_at.elapsed() > self.timeout
    }

    /// Returns how many key exchanges we've received.
    pub fn key_exchanges_received(&self) -> usize {
        self.received_key_exchanges.len()
    }

    /// Returns how many authentications we've received.
    pub fn authentications_received(&self) -> usize {
        self.received_authentications.len()
    }
}

/// Simulates a complete session establishment between parties.
/// Useful for testing and demos.
pub fn simulate_session_establishment(
    party_ids: &[PartyId],
    session_duration: Duration,
) -> MPCResult<Vec<EstablishedSession>> {
    let mut rng = ChaCha20Rng::from_entropy();

    // Generate key pairs for all parties
    let mut signing_keys: Vec<SigningKey> = Vec::new();
    let mut public_keys: HashMap<String, [u8; 32]> = HashMap::new();

    for party in party_ids {
        let mut key_bytes = [0u8; 32];
        rng.fill_bytes(&mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        public_keys.insert(party.0.clone(), verifying_key.to_bytes());
        signing_keys.push(signing_key);
    }

    // Create establishment instances
    let mut establishments: Vec<SessionEstablishment> = Vec::new();
    for (i, party) in party_ids.iter().enumerate() {
        let config = SessionConfig {
            party_id: party.clone(),
            all_parties: party_ids.to_vec(),
            timeout: Duration::from_secs(60),
            signing_key: signing_keys[i].to_bytes(),
            party_public_keys: public_keys.clone(),
        };
        establishments.push(SessionEstablishment::new(config)?);
    }

    // Phase 1: Generate and exchange key exchange messages
    let key_exchanges: Vec<KeyExchangeMessage> = establishments
        .iter_mut()
        .map(|e| e.generate_key_exchange())
        .collect();

    for (i, e) in establishments.iter_mut().enumerate() {
        for (j, ke) in key_exchanges.iter().enumerate() {
            if i != j {
                e.receive_key_exchange(ke.clone())?;
            }
        }
    }

    // Phase 2: Generate and exchange authentication messages
    let authentications: Vec<AuthenticationMessage> = establishments
        .iter_mut()
        .map(|e| e.generate_authentication())
        .collect::<MPCResult<Vec<_>>>()?;

    for (i, e) in establishments.iter_mut().enumerate() {
        for (j, auth) in authentications.iter().enumerate() {
            if i != j {
                e.receive_authentication(auth.clone())?;
            }
        }
    }

    // Phase 3: Finalize
    let sessions: Vec<EstablishedSession> = establishments
        .iter_mut()
        .map(|e| e.finalize(session_duration))
        .collect::<MPCResult<Vec<_>>>()?;

    // Verify all sessions have the same session ID
    let first_id = &sessions[0].session_id;
    for session in &sessions[1..] {
        if &session.session_id != first_id {
            return Err(MPCError::ProtocolError(
                "Session ID mismatch between parties".into(),
            ));
        }
    }

    Ok(sessions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_exchange_message() {
        let mut rng = ChaCha20Rng::from_entropy();

        let mut key_bytes = [0u8; 32];
        rng.fill_bytes(&mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        let mut ephemeral_bytes = [0u8; 32];
        rng.fill_bytes(&mut ephemeral_bytes);
        let ephemeral_secret = StaticSecret::from(ephemeral_bytes);
        let ephemeral_public = X25519PublicKey::from(&ephemeral_secret).to_bytes();

        let mut random = [0u8; 32];
        rng.fill_bytes(&mut random);

        let msg = KeyExchangeMessage::new(
            PartyId::from_index(0),
            ephemeral_public,
            &random,
            &signing_key,
        );

        assert!(msg.verify(&verifying_key));
        assert!(msg.is_fresh(300));
    }

    #[test]
    fn test_simulate_establishment() {
        let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();

        let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

        assert_eq!(sessions.len(), 3);

        // All sessions should have the same session ID
        let id = sessions[0].session_id;
        for s in &sessions[1..] {
            assert_eq!(s.session_id, id);
        }

        // All sessions should have pairwise keys to other parties
        for (i, s) in sessions.iter().enumerate() {
            for (j, party) in parties.iter().enumerate() {
                if i != j {
                    assert!(s.pairwise_key(&party.0).is_some());
                }
            }
        }
    }

    #[test]
    fn test_session_key_derivation() {
        let parties: Vec<PartyId> = (0..2).map(PartyId::from_index).collect();

        let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

        // Derived keys should be deterministic
        let key1 = sessions[0].derive_key("encryption");
        let key2 = sessions[1].derive_key("encryption");
        assert_eq!(key1, key2);

        // Different purposes should give different keys
        let auth_key = sessions[0].derive_key("authentication");
        assert_ne!(key1, auth_key);
    }

    #[test]
    fn test_session_expiry() {
        let parties: Vec<PartyId> = (0..2).map(PartyId::from_index).collect();

        // Very short session
        let sessions = simulate_session_establishment(&parties, Duration::from_millis(1)).unwrap();

        std::thread::sleep(Duration::from_millis(10));

        assert!(sessions[0].is_expired());
    }
}
