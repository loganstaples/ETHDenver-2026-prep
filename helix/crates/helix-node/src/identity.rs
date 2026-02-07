//! Node Identity with ed25519 Signing.
//!
//! Provides cryptographic identity for nodes, deriving PeerId from the
//! ed25519 public key and enabling message signing/verification.

#[cfg(feature = "crypto-sign")]
use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::network::messages::PeerId;

/// Node identity holding an ed25519 keypair.
#[cfg(feature = "crypto-sign")]
pub struct NodeIdentity {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    peer_id: PeerId,
}

#[cfg(feature = "crypto-sign")]
impl NodeIdentity {
    /// Generates a new random identity.
    pub fn generate() -> Self {
        use rand::rngs::OsRng;
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let peer_id = PeerId::from_string(hex::encode(verifying_key.as_bytes()));
        Self {
            signing_key,
            verifying_key,
            peer_id,
        }
    }

    /// Creates an identity from an existing signing key.
    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        let verifying_key = signing_key.verifying_key();
        let peer_id = PeerId::from_string(hex::encode(verifying_key.as_bytes()));
        Self {
            signing_key,
            verifying_key,
            peer_id,
        }
    }

    /// Returns the peer ID derived from the public key.
    pub fn peer_id(&self) -> &PeerId {
        &self.peer_id
    }

    /// Returns a reference to the signing key.
    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    /// Returns a reference to the verifying key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }
}

/// Stub identity for when crypto-sign is disabled.
#[cfg(not(feature = "crypto-sign"))]
pub struct NodeIdentity {
    peer_id: PeerId,
}

#[cfg(not(feature = "crypto-sign"))]
impl NodeIdentity {
    /// Generates a new random identity (UUID-based without crypto).
    pub fn generate() -> Self {
        Self {
            peer_id: PeerId::random(),
        }
    }

    /// Returns the peer ID.
    pub fn peer_id(&self) -> &PeerId {
        &self.peer_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_generate() {
        let id = NodeIdentity::generate();
        assert!(!id.peer_id().0.is_empty());
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_identity_deterministic_peer_id() {
        use ed25519_dalek::SigningKey;
        let key_bytes = [42u8; 32];
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let id1 = NodeIdentity::from_signing_key(signing_key);

        let signing_key2 = SigningKey::from_bytes(&key_bytes);
        let id2 = NodeIdentity::from_signing_key(signing_key2);

        assert_eq!(id1.peer_id().0, id2.peer_id().0);
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_identity_different_keys_different_ids() {
        let id1 = NodeIdentity::generate();
        let id2 = NodeIdentity::generate();
        assert_ne!(id1.peer_id().0, id2.peer_id().0);
    }
}
