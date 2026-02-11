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

/// Loads an existing identity from disk, or generates and persists a new one.
///
/// The identity key file is written atomically (temp file + rename) to
/// prevent corruption from partial writes. On subsequent runs the same
/// identity is loaded, keeping the node's PeerId stable.
#[cfg(feature = "crypto-sign")]
pub fn load_or_generate(data_dir: &std::path::Path) -> std::io::Result<NodeIdentity> {
    let key_path = data_dir.join("identity.key");
    if key_path.exists() {
        let bytes = std::fs::read(&key_path)?;
        if bytes.len() != 32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("identity.key has wrong size: expected 32, got {}", bytes.len()),
            ));
        }
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        Ok(NodeIdentity::from_signing_key(signing_key))
    } else {
        let identity = NodeIdentity::generate();
        std::fs::create_dir_all(data_dir)?;
        // Atomic write: temp file then rename
        let tmp_path = data_dir.join("identity.key.tmp");
        std::fs::write(&tmp_path, identity.signing_key().as_bytes())?;
        std::fs::rename(&tmp_path, &key_path)?;
        Ok(identity)
    }
}

/// Loads an existing identity from disk, or generates and persists a new one.
///
/// Without the `crypto-sign` feature, persists the PeerId string.
#[cfg(not(feature = "crypto-sign"))]
pub fn load_or_generate(data_dir: &std::path::Path) -> std::io::Result<NodeIdentity> {
    let key_path = data_dir.join("identity.key");
    if key_path.exists() {
        let peer_id_str = std::fs::read_to_string(&key_path)?;
        Ok(NodeIdentity {
            peer_id: PeerId::from_string(peer_id_str.trim()),
        })
    } else {
        let identity = NodeIdentity::generate();
        std::fs::create_dir_all(data_dir)?;
        let tmp_path = data_dir.join("identity.key.tmp");
        std::fs::write(&tmp_path, &identity.peer_id.0)?;
        std::fs::rename(&tmp_path, &key_path)?;
        Ok(identity)
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

    #[test]
    fn test_load_or_generate_creates_new() {
        let tmp = tempfile::tempdir().unwrap();
        let id = load_or_generate(tmp.path()).unwrap();
        assert!(!id.peer_id().0.is_empty());

        // Key file should exist
        assert!(tmp.path().join("identity.key").exists());
    }

    #[test]
    fn test_load_or_generate_reloads_same_id() {
        let tmp = tempfile::tempdir().unwrap();
        let id1 = load_or_generate(tmp.path()).unwrap();
        let id2 = load_or_generate(tmp.path()).unwrap();
        assert_eq!(id1.peer_id().0, id2.peer_id().0);
    }

    #[cfg(feature = "crypto-sign")]
    #[test]
    fn test_load_or_generate_rejects_bad_key_file() {
        let tmp = tempfile::tempdir().unwrap();
        // Write a file with wrong length
        std::fs::write(tmp.path().join("identity.key"), &[0u8; 16]).unwrap();
        let result = load_or_generate(tmp.path());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("wrong size"));
    }
}
