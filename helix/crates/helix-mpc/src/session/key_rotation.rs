//! Session key rotation and perfect forward secrecy.
//!
//! This module implements:
//! - Automatic key rotation at configurable intervals
//! - Perfect forward secrecy through ephemeral key deletion
//! - Key versioning for smooth transitions
//! - Secure key derivation chain
//!
//! # Security Properties
//!
//! 1. **Forward Secrecy**: Compromise of current keys doesn't reveal past communications
//! 2. **Key Freshness**: Regular rotation limits exposure window
//! 3. **Secure Erasure**: Old key material is securely deleted
//! 4. **Atomic Transition**: Key rotation is atomic to prevent state confusion
//!
//! # Key Rotation Protocol
//!
//! 1. All parties agree on rotation epoch
//! 2. Each party derives new ephemeral keys from current keys + fresh randomness
//! 3. New keys are committed before old keys are deleted
//! 4. Old keys are securely erased
//! 5. Rotation acknowledgment is broadcast

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Configuration for key rotation.
#[derive(Debug, Clone)]
pub struct KeyRotationConfig {
    /// Interval between automatic key rotations.
    pub rotation_interval: Duration,
    /// Minimum interval between rotations (rate limiting).
    pub min_rotation_interval: Duration,
    /// Number of old key versions to keep for in-flight messages.
    pub key_history_depth: usize,
    /// Whether to enable automatic rotation.
    pub auto_rotate: bool,
    /// Maximum age of a key before forced rotation.
    pub max_key_age: Duration,
    /// Require acknowledgment from all parties before completing rotation.
    pub require_ack: bool,
}

impl Default for KeyRotationConfig {
    fn default() -> Self {
        Self {
            rotation_interval: Duration::from_secs(3600),      // 1 hour
            min_rotation_interval: Duration::from_secs(60),    // 1 minute minimum
            key_history_depth: 3,                              // Keep 3 old versions
            auto_rotate: true,
            max_key_age: Duration::from_secs(86400),           // 24 hours max
            require_ack: true,
        }
    }
}

impl KeyRotationConfig {
    /// Creates a high-security configuration with frequent rotation.
    pub fn high_security() -> Self {
        Self {
            rotation_interval: Duration::from_secs(300),       // 5 minutes
            min_rotation_interval: Duration::from_secs(60),
            key_history_depth: 2,
            auto_rotate: true,
            max_key_age: Duration::from_secs(3600),            // 1 hour max
            require_ack: true,
        }
    }

    /// Creates a performance-optimized configuration with less frequent rotation.
    pub fn performance() -> Self {
        Self {
            rotation_interval: Duration::from_secs(7200),      // 2 hours
            min_rotation_interval: Duration::from_secs(300),
            key_history_depth: 2,
            auto_rotate: true,
            max_key_age: Duration::from_secs(86400 * 7),       // 1 week max
            require_ack: false,
        }
    }
}

/// A versioned key with metadata.
#[derive(Clone)]
pub struct VersionedKey {
    /// The key material.
    pub key: [u8; 32],
    /// Key version/epoch number.
    pub version: u64,
    /// When this key was created.
    created_at: Instant,
    /// Chain value for deriving next key.
    pub chain_key: [u8; 32],
}

impl Zeroize for VersionedKey {
    fn zeroize(&mut self) {
        self.key.zeroize();
        self.chain_key.zeroize();
    }
}

impl Drop for VersionedKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl VersionedKey {
    /// Creates a new versioned key.
    pub fn new(key: [u8; 32], version: u64, chain_key: [u8; 32]) -> Self {
        Self {
            key,
            version,
            created_at: Instant::now(),
            chain_key,
        }
    }

    /// Returns the age of this key.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Derives a purpose-specific key.
    pub fn derive_key(&self, purpose: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.key);
        hasher.update(&self.version.to_le_bytes());
        hasher.update(purpose.as_bytes());
        hasher.finalize().into()
    }

    /// Derives the next key in the chain (for forward secrecy).
    pub fn derive_next(&self) -> (VersionedKey, [u8; 32]) {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut fresh_randomness = [0u8; 32];
        rng.fill_bytes(&mut fresh_randomness);

        // Derive new key from chain key + fresh randomness
        let mut hasher = Sha256::new();
        hasher.update(&self.chain_key);
        hasher.update(&fresh_randomness);
        hasher.update(b"next_key");
        let new_key: [u8; 32] = hasher.finalize().into();

        // Derive new chain key
        let mut hasher = Sha256::new();
        hasher.update(&self.chain_key);
        hasher.update(&fresh_randomness);
        hasher.update(b"chain_key");
        let new_chain_key: [u8; 32] = hasher.finalize().into();

        let new_versioned = VersionedKey::new(new_key, self.version + 1, new_chain_key);

        (new_versioned, fresh_randomness)
    }
}

impl std::fmt::Debug for VersionedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VersionedKey")
            .field("version", &self.version)
            .field("created_at", &self.created_at)
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// State of a key rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationState {
    /// No rotation in progress.
    Idle,
    /// Rotation initiated, waiting for acknowledgments.
    Initiated,
    /// All parties acknowledged, committing new key.
    Committing,
    /// Rotation complete, old key scheduled for deletion.
    Complete,
    /// Rotation failed.
    Failed,
}

/// Tracks acknowledgments for a key rotation.
#[derive(Debug, Clone)]
pub struct RotationAcknowledgments {
    /// Version being rotated to.
    pub target_version: u64,
    /// Parties that have acknowledged.
    pub acknowledged: Vec<PartyId>,
    /// Total parties expected.
    pub expected_count: usize,
    /// When the rotation was initiated.
    pub initiated_at: Instant,
    /// Timeout for receiving all acknowledgments.
    pub timeout: Duration,
}

impl RotationAcknowledgments {
    pub fn new(target_version: u64, expected_count: usize, timeout: Duration) -> Self {
        Self {
            target_version,
            acknowledged: Vec::new(),
            expected_count,
            initiated_at: Instant::now(),
            timeout,
        }
    }

    pub fn acknowledge(&mut self, party: PartyId) -> bool {
        if !self.acknowledged.contains(&party) {
            self.acknowledged.push(party);
        }
        self.is_complete()
    }

    pub fn is_complete(&self) -> bool {
        self.acknowledged.len() >= self.expected_count
    }

    pub fn is_timed_out(&self) -> bool {
        self.initiated_at.elapsed() > self.timeout
    }
}

/// Key rotation message types.
#[derive(Debug, Clone)]
pub enum RotationMessage {
    /// Initiate rotation to new version.
    Initiate {
        from: PartyId,
        new_version: u64,
        /// Commitment to the new key (hash).
        key_commitment: [u8; 32],
        /// Fresh randomness contribution.
        randomness: [u8; 32],
    },
    /// Acknowledge rotation.
    Acknowledge {
        from: PartyId,
        version: u64,
        /// Party's randomness contribution.
        randomness: [u8; 32],
    },
    /// Commit the new key (after all acks received).
    Commit {
        from: PartyId,
        version: u64,
    },
    /// Rotation complete notification.
    Complete {
        from: PartyId,
        version: u64,
    },
    /// Abort rotation.
    Abort {
        from: PartyId,
        version: u64,
        reason: String,
    },
}

/// Manages key rotation for a session.
pub struct KeyRotationManager {
    /// Our party ID.
    party_id: PartyId,
    /// All parties in the session.
    parties: Vec<PartyId>,
    /// Configuration.
    config: KeyRotationConfig,
    /// Current key version.
    current_version: AtomicU64,
    /// Current and historical keys.
    keys: RwLock<HashMap<u64, VersionedKey>>,
    /// Pairwise keys (party_id -> version -> key).
    pairwise_keys: RwLock<HashMap<String, HashMap<u64, [u8; 32]>>>,
    /// Current rotation state.
    rotation_state: RwLock<RotationState>,
    /// Pending acknowledgments.
    pending_acks: RwLock<Option<RotationAcknowledgments>>,
    /// Collected randomness from parties for current rotation.
    collected_randomness: RwLock<HashMap<String, [u8; 32]>>,
    /// Last rotation time.
    last_rotation: RwLock<Instant>,
    /// Secure RNG.
    rng: RwLock<ChaCha20Rng>,
}

impl KeyRotationManager {
    /// Creates a new key rotation manager.
    pub fn new(
        party_id: PartyId,
        parties: Vec<PartyId>,
        initial_key: [u8; 32],
        initial_chain_key: [u8; 32],
        config: KeyRotationConfig,
    ) -> Self {
        let initial_versioned = VersionedKey::new(initial_key, 0, initial_chain_key);
        let mut keys = HashMap::new();
        keys.insert(0, initial_versioned);

        Self {
            party_id,
            parties,
            config,
            current_version: AtomicU64::new(0),
            keys: RwLock::new(keys),
            pairwise_keys: RwLock::new(HashMap::new()),
            rotation_state: RwLock::new(RotationState::Idle),
            pending_acks: RwLock::new(None),
            collected_randomness: RwLock::new(HashMap::new()),
            last_rotation: RwLock::new(Instant::now()),
            rng: RwLock::new(ChaCha20Rng::from_entropy()),
        }
    }

    /// Creates from an established session.
    pub fn from_session(
        party_id: PartyId,
        parties: Vec<PartyId>,
        session_key: [u8; 32],
        config: KeyRotationConfig,
    ) -> Self {
        // Derive initial chain key from session key
        let mut hasher = Sha256::new();
        hasher.update(&session_key);
        hasher.update(b"initial_chain");
        let chain_key: [u8; 32] = hasher.finalize().into();

        Self::new(party_id, parties, session_key, chain_key, config)
    }

    /// Returns the current key version.
    pub fn current_version(&self) -> u64 {
        self.current_version.load(Ordering::SeqCst)
    }

    /// Returns the current key.
    pub fn current_key(&self) -> Option<[u8; 32]> {
        let version = self.current_version();
        self.keys.read().get(&version).map(|k| k.key)
    }

    /// Returns a key by version.
    pub fn key_by_version(&self, version: u64) -> Option<[u8; 32]> {
        self.keys.read().get(&version).map(|k| k.key)
    }

    /// Derives a purpose-specific key from the current session key.
    pub fn derive_key(&self, purpose: &str) -> Option<[u8; 32]> {
        let version = self.current_version();
        self.keys.read().get(&version).map(|k| k.derive_key(purpose))
    }

    /// Derives a purpose-specific key from a specific version.
    pub fn derive_key_versioned(&self, purpose: &str, version: u64) -> Option<[u8; 32]> {
        self.keys.read().get(&version).map(|k| k.derive_key(purpose))
    }

    /// Checks if rotation is needed.
    pub fn needs_rotation(&self) -> bool {
        if !self.config.auto_rotate {
            return false;
        }

        let state = *self.rotation_state.read();
        if state != RotationState::Idle {
            return false;
        }

        let last_rotation = *self.last_rotation.read();
        if last_rotation.elapsed() < self.config.min_rotation_interval {
            return false;
        }

        // Check key age
        let version = self.current_version();
        if let Some(key) = self.keys.read().get(&version) {
            if key.age() > self.config.max_key_age {
                return true;
            }
            if key.age() > self.config.rotation_interval {
                return true;
            }
        }

        false
    }

    /// Initiates a key rotation.
    pub fn initiate_rotation(&self) -> MPCResult<RotationMessage> {
        let mut state = self.rotation_state.write();
        if *state != RotationState::Idle {
            return Err(MPCError::ProtocolError("Rotation already in progress".into()));
        }

        let current_version = self.current_version();
        let new_version = current_version + 1;

        // Generate our randomness contribution
        let mut randomness = [0u8; 32];
        self.rng.write().fill_bytes(&mut randomness);

        // Store our randomness
        self.collected_randomness.write().clear();
        self.collected_randomness.write().insert(self.party_id.0.clone(), randomness);

        // Create key commitment (hash of what we'll derive)
        let mut hasher = Sha256::new();
        hasher.update(&randomness);
        hasher.update(&new_version.to_le_bytes());
        let key_commitment: [u8; 32] = hasher.finalize().into();

        // Set up acknowledgment tracking
        *self.pending_acks.write() = Some(RotationAcknowledgments::new(
            new_version,
            self.parties.len(),
            Duration::from_secs(30),
        ));

        *state = RotationState::Initiated;

        Ok(RotationMessage::Initiate {
            from: self.party_id.clone(),
            new_version,
            key_commitment,
            randomness,
        })
    }

    /// Processes a rotation message.
    pub fn process_message(&self, msg: RotationMessage) -> MPCResult<Option<RotationMessage>> {
        match msg {
            RotationMessage::Initiate { from, new_version, key_commitment, randomness } => {
                self.handle_initiate(from, new_version, key_commitment, randomness)
            }
            RotationMessage::Acknowledge { from, version, randomness } => {
                self.handle_acknowledge(from, version, randomness)
            }
            RotationMessage::Commit { from, version } => {
                self.handle_commit(from, version)
            }
            RotationMessage::Complete { from, version } => {
                self.handle_complete(from, version)
            }
            RotationMessage::Abort { from, version, reason } => {
                self.handle_abort(from, version, reason)
            }
        }
    }

    fn handle_initiate(
        &self,
        from: PartyId,
        new_version: u64,
        _key_commitment: [u8; 32],
        randomness: [u8; 32],
    ) -> MPCResult<Option<RotationMessage>> {
        let current = self.current_version();
        if new_version != current + 1 {
            return Ok(Some(RotationMessage::Abort {
                from: self.party_id.clone(),
                version: new_version,
                reason: format!("Expected version {}, got {}", current + 1, new_version),
            }));
        }

        // Store initiator's randomness
        self.collected_randomness.write().insert(from.0, randomness);

        // Generate our randomness
        let mut our_randomness = [0u8; 32];
        self.rng.write().fill_bytes(&mut our_randomness);
        self.collected_randomness.write().insert(self.party_id.0.clone(), our_randomness);

        // Update state
        *self.rotation_state.write() = RotationState::Initiated;

        // Set up acknowledgment tracking
        *self.pending_acks.write() = Some(RotationAcknowledgments::new(
            new_version,
            self.parties.len(),
            Duration::from_secs(30),
        ));

        Ok(Some(RotationMessage::Acknowledge {
            from: self.party_id.clone(),
            version: new_version,
            randomness: our_randomness,
        }))
    }

    fn handle_acknowledge(
        &self,
        from: PartyId,
        version: u64,
        randomness: [u8; 32],
    ) -> MPCResult<Option<RotationMessage>> {
        // Store the randomness
        self.collected_randomness.write().insert(from.0.clone(), randomness);

        // Update acknowledgments
        let mut pending = self.pending_acks.write();
        if let Some(ref mut acks) = *pending {
            if acks.target_version != version {
                return Ok(None);
            }

            if acks.acknowledge(from) {
                // All parties acknowledged, proceed to commit
                drop(pending);
                *self.rotation_state.write() = RotationState::Committing;

                // Derive the new key from all randomness
                self.derive_and_commit_new_key(version)?;

                return Ok(Some(RotationMessage::Commit {
                    from: self.party_id.clone(),
                    version,
                }));
            }
        }

        Ok(None)
    }

    fn handle_commit(&self, _from: PartyId, version: u64) -> MPCResult<Option<RotationMessage>> {
        // Derive and commit if we haven't already
        let state = *self.rotation_state.read();
        if state == RotationState::Initiated {
            *self.rotation_state.write() = RotationState::Committing;
            self.derive_and_commit_new_key(version)?;
        }

        // Mark complete
        self.finalize_rotation(version)?;

        Ok(Some(RotationMessage::Complete {
            from: self.party_id.clone(),
            version,
        }))
    }

    fn handle_complete(&self, _from: PartyId, version: u64) -> MPCResult<Option<RotationMessage>> {
        // Ensure we're in the right state and version
        if self.current_version() != version {
            self.finalize_rotation(version)?;
        }
        Ok(None)
    }

    fn handle_abort(
        &self,
        _from: PartyId,
        _version: u64,
        _reason: String,
    ) -> MPCResult<Option<RotationMessage>> {
        *self.rotation_state.write() = RotationState::Failed;
        self.collected_randomness.write().clear();
        *self.pending_acks.write() = None;
        Ok(None)
    }

    /// Derives and commits the new key from collected randomness.
    fn derive_and_commit_new_key(&self, version: u64) -> MPCResult<()> {
        let randomness = self.collected_randomness.read();
        if randomness.len() != self.parties.len() {
            return Err(MPCError::ProtocolError(
                "Missing randomness contributions".into(),
            ));
        }

        // Combine all randomness deterministically
        let mut combined_randomness = [0u8; 32];
        let mut sorted_parties: Vec<_> = randomness.iter().collect();
        sorted_parties.sort_by(|a, b| a.0.cmp(b.0));

        let mut hasher = Sha256::new();
        for (party_id, party_randomness) in sorted_parties {
            hasher.update(party_id.as_bytes());
            hasher.update(party_randomness);
        }
        combined_randomness = hasher.finalize().into();

        // Get current key for chain derivation
        let current_version = version - 1;
        let keys = self.keys.read();
        let current_key = keys.get(&current_version)
            .ok_or_else(|| MPCError::ProtocolError("Current key not found".into()))?;

        // Derive new key
        let mut hasher = Sha256::new();
        hasher.update(&current_key.chain_key);
        hasher.update(&combined_randomness);
        hasher.update(b"new_session_key");
        let new_key: [u8; 32] = hasher.finalize().into();

        // Derive new chain key
        let mut hasher = Sha256::new();
        hasher.update(&current_key.chain_key);
        hasher.update(&combined_randomness);
        hasher.update(b"new_chain_key");
        let new_chain_key: [u8; 32] = hasher.finalize().into();

        drop(keys);

        // Store new key
        let new_versioned = VersionedKey::new(new_key, version, new_chain_key);
        self.keys.write().insert(version, new_versioned);

        Ok(())
    }

    /// Finalizes a rotation, updating version and cleaning old keys.
    fn finalize_rotation(&self, version: u64) -> MPCResult<()> {
        // Update current version
        self.current_version.store(version, Ordering::SeqCst);

        // Clean up old keys beyond history depth
        let mut keys = self.keys.write();
        let min_version = version.saturating_sub(self.config.key_history_depth as u64);
        keys.retain(|&v, key| {
            if v < min_version {
                // Securely erase old key
                let mut key_copy = key.clone();
                key_copy.zeroize();
                false
            } else {
                true
            }
        });

        // Reset state
        *self.rotation_state.write() = RotationState::Idle;
        *self.last_rotation.write() = Instant::now();
        self.collected_randomness.write().clear();
        *self.pending_acks.write() = None;

        Ok(())
    }

    /// Forces an immediate rotation (for security events).
    pub fn force_rotation(&self) -> MPCResult<RotationMessage> {
        // Reset state if stuck
        *self.rotation_state.write() = RotationState::Idle;
        self.initiate_rotation()
    }

    /// Gets the current rotation state.
    pub fn rotation_state(&self) -> RotationState {
        *self.rotation_state.read()
    }

    /// Gets statistics about key rotation.
    pub fn stats(&self) -> KeyRotationStats {
        let keys = self.keys.read();
        let current_version = self.current_version();

        KeyRotationStats {
            current_version,
            keys_in_memory: keys.len(),
            current_key_age: keys.get(&current_version).map(|k| k.age()),
            last_rotation: *self.last_rotation.read(),
            rotation_state: *self.rotation_state.read(),
        }
    }

    /// Adds a pairwise key for a party.
    pub fn add_pairwise_key(&self, party_id: &str, version: u64, key: [u8; 32]) {
        let mut pairwise = self.pairwise_keys.write();
        let party_keys = pairwise.entry(party_id.to_string()).or_default();
        party_keys.insert(version, key);

        // Clean old versions
        let min_version = version.saturating_sub(self.config.key_history_depth as u64);
        party_keys.retain(|&v, _| v >= min_version);
    }

    /// Gets a pairwise key for a party and version.
    pub fn pairwise_key(&self, party_id: &str, version: u64) -> Option<[u8; 32]> {
        self.pairwise_keys
            .read()
            .get(party_id)
            .and_then(|keys| keys.get(&version).copied())
    }
}

/// Statistics about key rotation.
#[derive(Debug, Clone)]
pub struct KeyRotationStats {
    pub current_version: u64,
    pub keys_in_memory: usize,
    pub current_key_age: Option<Duration>,
    pub last_rotation: Instant,
    pub rotation_state: RotationState,
}

/// Perfect Forward Secrecy manager.
///
/// Ensures ephemeral keys are properly deleted and that key material
/// cannot be recovered after use.
pub struct PFSManager {
    /// Ephemeral secrets pending deletion.
    pending_deletion: RwLock<Vec<EphemeralSecret>>,
    /// Secure random generator.
    rng: RwLock<ChaCha20Rng>,
}

/// An ephemeral secret that zeroizes on drop.
#[derive(ZeroizeOnDrop)]
struct EphemeralSecret {
    #[zeroize(skip)]
    id: u64,
    secret: [u8; 32],
}

impl PFSManager {
    pub fn new() -> Self {
        Self {
            pending_deletion: RwLock::new(Vec::new()),
            rng: RwLock::new(ChaCha20Rng::from_entropy()),
        }
    }

    /// Generates a new ephemeral DH keypair.
    pub fn generate_ephemeral(&self) -> (X25519PublicKey, u64) {
        let mut secret_bytes = [0u8; 32];
        self.rng.write().fill_bytes(&mut secret_bytes);

        let secret = StaticSecret::from(secret_bytes);
        let public = X25519PublicKey::from(&secret);

        // Generate ID and store for later deletion
        let id = self.rng.write().next_u64();
        let ephemeral = EphemeralSecret {
            id,
            secret: secret_bytes,
        };
        self.pending_deletion.write().push(ephemeral);

        (public, id)
    }

    /// Performs DH exchange and schedules secret for deletion.
    pub fn exchange_and_delete(
        &self,
        secret_id: u64,
        their_public: &X25519PublicKey,
    ) -> MPCResult<[u8; 32]> {
        let mut pending = self.pending_deletion.write();

        let idx = pending.iter().position(|s| s.id == secret_id)
            .ok_or_else(|| MPCError::ProtocolError("Ephemeral secret not found".into()))?;

        let ephemeral = pending.remove(idx);
        let secret = StaticSecret::from(ephemeral.secret);
        let shared_secret = secret.diffie_hellman(their_public);

        // ephemeral is dropped and zeroized here
        Ok(*shared_secret.as_bytes())
    }

    /// Securely deletes all pending ephemeral secrets.
    pub fn delete_all_pending(&self) {
        let mut pending = self.pending_deletion.write();
        // All secrets are zeroized when dropped
        pending.clear();
    }

    /// Returns count of pending secrets.
    pub fn pending_count(&self) -> usize {
        self.pending_deletion.read().len()
    }
}

impl Default for PFSManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_key_rotation_manager_creation() {
        let parties = test_parties(3);
        let key = [42u8; 32];
        let chain_key = [0u8; 32];
        let config = KeyRotationConfig::default();

        let manager = KeyRotationManager::new(
            parties[0].clone(),
            parties.clone(),
            key,
            chain_key,
            config,
        );

        assert_eq!(manager.current_version(), 0);
        assert_eq!(manager.current_key().unwrap(), key);
    }

    #[test]
    fn test_key_derivation() {
        let parties = test_parties(2);
        let key = [42u8; 32];
        let chain_key = [0u8; 32];
        let config = KeyRotationConfig::default();

        let manager = KeyRotationManager::new(
            parties[0].clone(),
            parties.clone(),
            key,
            chain_key,
            config,
        );

        let enc_key = manager.derive_key("encryption").unwrap();
        let auth_key = manager.derive_key("authentication").unwrap();

        // Different purposes should give different keys
        assert_ne!(enc_key, auth_key);

        // Same purpose should give same key
        assert_eq!(enc_key, manager.derive_key("encryption").unwrap());
    }

    #[test]
    fn test_versioned_key_derivation() {
        let key = [42u8; 32];
        let chain_key = [0u8; 32];
        let versioned = VersionedKey::new(key, 0, chain_key);

        let (next, _randomness) = versioned.derive_next();
        assert_eq!(next.version, 1);
        assert_ne!(next.key, key);
        assert_ne!(next.chain_key, chain_key);
    }

    #[test]
    fn test_rotation_initiation() {
        let parties = test_parties(3);
        let key = [42u8; 32];
        let chain_key = [0u8; 32];
        let mut config = KeyRotationConfig::default();
        config.min_rotation_interval = Duration::from_millis(0);

        let manager = KeyRotationManager::new(
            parties[0].clone(),
            parties.clone(),
            key,
            chain_key,
            config,
        );

        let msg = manager.initiate_rotation().unwrap();
        match msg {
            RotationMessage::Initiate { new_version, .. } => {
                assert_eq!(new_version, 1);
            }
            _ => panic!("Expected Initiate message"),
        }

        assert_eq!(manager.rotation_state(), RotationState::Initiated);
    }

    #[test]
    fn test_pfs_manager() {
        let pfs = PFSManager::new();

        // Generate ephemeral
        let (public1, id1) = pfs.generate_ephemeral();
        let (public2, id2) = pfs.generate_ephemeral();

        assert_eq!(pfs.pending_count(), 2);
        assert_ne!(id1, id2);

        // Exchange and delete
        let shared = pfs.exchange_and_delete(id1, &public2).unwrap();
        assert_eq!(pfs.pending_count(), 1);
        assert!(!shared.iter().all(|&b| b == 0));

        // Delete remaining
        pfs.delete_all_pending();
        assert_eq!(pfs.pending_count(), 0);
    }

    #[test]
    fn test_rotation_config_presets() {
        let default = KeyRotationConfig::default();
        let high_sec = KeyRotationConfig::high_security();
        let perf = KeyRotationConfig::performance();

        // High security should have shorter rotation interval
        assert!(high_sec.rotation_interval < default.rotation_interval);

        // Performance should have longer rotation interval
        assert!(perf.rotation_interval > default.rotation_interval);
    }

    #[test]
    fn test_needs_rotation() {
        let parties = test_parties(2);
        let key = [42u8; 32];
        let chain_key = [0u8; 32];
        let mut config = KeyRotationConfig::default();
        config.rotation_interval = Duration::from_millis(10);
        config.min_rotation_interval = Duration::from_millis(0);

        let manager = KeyRotationManager::new(
            parties[0].clone(),
            parties.clone(),
            key,
            chain_key,
            config,
        );

        // Initially shouldn't need rotation
        assert!(!manager.needs_rotation());

        // Wait for rotation interval
        std::thread::sleep(Duration::from_millis(20));
        assert!(manager.needs_rotation());
    }

    #[test]
    fn test_key_rotation_stats() {
        let parties = test_parties(2);
        let key = [42u8; 32];
        let chain_key = [0u8; 32];
        let config = KeyRotationConfig::default();

        let manager = KeyRotationManager::new(
            parties[0].clone(),
            parties.clone(),
            key,
            chain_key,
            config,
        );

        let stats = manager.stats();
        assert_eq!(stats.current_version, 0);
        assert_eq!(stats.keys_in_memory, 1);
        assert_eq!(stats.rotation_state, RotationState::Idle);
    }
}
