//! Encrypted Weight Storage.
//!
//! Wraps any storage backend with AES-256-GCM encryption using Argon2id
//! key derivation. This ensures model weights are encrypted at rest
//! (e.g., on IPFS) and only the model owner (or designated parties)
//! can decrypt the final trained weights.
//!
//! Matches the encryption scheme used in `helix-client`'s wallet encryption
//! for consistency across the HELIX ecosystem.

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, AeadCore, Nonce,
};
use argon2::{
    password_hash::SaltString,
    Argon2, PasswordHasher,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Encrypted data envelope with KDF parameters for decryption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedEnvelope {
    /// Version of the encryption scheme.
    pub version: u8,
    /// Argon2id salt (base64 string).
    pub salt: String,
    /// AES-256-GCM nonce (hex string).
    pub nonce: String,
    /// Ciphertext (raw bytes).
    pub ciphertext: Vec<u8>,
    /// Optional metadata tag (e.g., model hash for identification).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

/// Errors from encrypted storage operations.
#[derive(Debug)]
pub enum EncryptedStorageError {
    /// Key derivation failed.
    KeyDerivation(String),
    /// Encryption failed.
    Encryption(String),
    /// Decryption failed (wrong password or corrupted data).
    Decryption(String),
    /// Serialization error.
    Serialization(String),
}

impl std::fmt::Display for EncryptedStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyDerivation(e) => write!(f, "Key derivation error: {}", e),
            Self::Encryption(e) => write!(f, "Encryption error: {}", e),
            Self::Decryption(e) => write!(f, "Decryption error: {}", e),
            Self::Serialization(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for EncryptedStorageError {}

/// Encrypts data using AES-256-GCM with an Argon2id-derived key.
///
/// The password is used to derive a 256-bit encryption key via Argon2id.
/// A random salt and nonce are generated for each encryption operation,
/// ensuring the same plaintext produces different ciphertext each time.
pub fn encrypt_data(
    data: &[u8],
    password: &str,
    tag: Option<String>,
) -> Result<EncryptedEnvelope, EncryptedStorageError> {
    // Generate random salt for Argon2id
    let salt = SaltString::generate(&mut OsRng);

    // Derive 256-bit key from password using Argon2id
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| EncryptedStorageError::KeyDerivation(format!("Argon2id failed: {}", e)))?;

    let hash_bytes = password_hash
        .hash
        .ok_or_else(|| EncryptedStorageError::KeyDerivation("No hash output".into()))?;
    let key_bytes: [u8; 32] = hash_bytes.as_bytes()[..32]
        .try_into()
        .map_err(|_| EncryptedStorageError::KeyDerivation("Invalid key length".into()))?;

    // Encrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key_bytes)
        .map_err(|e| EncryptedStorageError::Encryption(format!("Cipher init failed: {}", e)))?;
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    let ciphertext = cipher
        .encrypt(&nonce, data)
        .map_err(|e| EncryptedStorageError::Encryption(format!("AES-GCM encrypt failed: {}", e)))?;

    Ok(EncryptedEnvelope {
        version: 1,
        salt: salt.to_string(),
        nonce: hex::encode(nonce),
        ciphertext,
        tag,
    })
}

/// Decrypts an encrypted envelope using the provided password.
///
/// Returns the original plaintext data, or an error if the password is wrong
/// or the data has been tampered with (AES-GCM provides authenticated encryption).
pub fn decrypt_data(
    envelope: &EncryptedEnvelope,
    password: &str,
) -> Result<Vec<u8>, EncryptedStorageError> {
    if envelope.version != 1 {
        return Err(EncryptedStorageError::Decryption(format!(
            "Unsupported encryption version: {}",
            envelope.version
        )));
    }

    // Reconstruct salt
    let salt = SaltString::from_b64(&envelope.salt)
        .map_err(|e| EncryptedStorageError::Decryption(format!("Invalid salt: {}", e)))?;

    // Derive key using Argon2id (same params as encryption)
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| EncryptedStorageError::KeyDerivation(format!("Argon2id failed: {}", e)))?;

    let hash_bytes = password_hash
        .hash
        .ok_or_else(|| EncryptedStorageError::KeyDerivation("No hash output".into()))?;
    let key_bytes: [u8; 32] = hash_bytes.as_bytes()[..32]
        .try_into()
        .map_err(|_| EncryptedStorageError::KeyDerivation("Invalid key length".into()))?;

    // Decrypt
    let cipher = Aes256Gcm::new_from_slice(&key_bytes)
        .map_err(|e| EncryptedStorageError::Decryption(format!("Cipher init failed: {}", e)))?;

    let nonce_bytes = hex::decode(&envelope.nonce)
        .map_err(|e| EncryptedStorageError::Decryption(format!("Invalid nonce: {}", e)))?;
    let nonce = Nonce::from_slice(&nonce_bytes);

    cipher
        .decrypt(nonce, envelope.ciphertext.as_ref())
        .map_err(|_| EncryptedStorageError::Decryption(
            "Decryption failed - wrong password or corrupted data".into(),
        ))
}

/// Serializes an encrypted envelope to bytes for storage.
///
/// Uses JSON encoding for robustness with string fields (salt, nonce).
pub fn serialize_envelope(envelope: &EncryptedEnvelope) -> Result<Vec<u8>, EncryptedStorageError> {
    serde_json::to_vec(envelope)
        .map_err(|e| EncryptedStorageError::Serialization(format!("Failed to serialize envelope: {}", e)))
}

/// Deserializes an encrypted envelope from bytes.
pub fn deserialize_envelope(data: &[u8]) -> Result<EncryptedEnvelope, EncryptedStorageError> {
    serde_json::from_slice(data)
        .map_err(|e| EncryptedStorageError::Serialization(format!("Failed to deserialize envelope: {}", e)))
}

/// Encrypted IPFS storage wrapper.
///
/// Encrypts checkpoint data before uploading to IPFS and decrypts after downloading.
/// The model owner controls the encryption password — without it, the data on IPFS
/// is indistinguishable from random bytes.
pub struct EncryptedIpfsStorage {
    /// Inner IPFS storage.
    inner: super::ipfs::IpfsStorage,
    /// Encryption password (derived from HELIX_MPC_STORAGE_KEY env var or config).
    password: String,
}

impl EncryptedIpfsStorage {
    /// Creates a new encrypted IPFS storage.
    pub fn new(api_url: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            inner: super::ipfs::IpfsStorage::new(api_url),
            password: password.into(),
        }
    }

    /// Encrypts and stores data on IPFS.
    ///
    /// Returns the CID of the encrypted data.
    pub async fn store_encrypted(
        &self,
        data: &[u8],
        tag: Option<String>,
    ) -> Result<String, String> {
        let envelope = encrypt_data(data, &self.password, tag)
            .map_err(|e| format!("Encryption failed: {}", e))?;
        let envelope_bytes = serialize_envelope(&envelope)
            .map_err(|e| format!("Serialization failed: {}", e))?;
        self.inner.store(&envelope_bytes).await
            .map_err(|e| format!("IPFS store failed: {}", e))
    }

    /// Downloads and decrypts data from IPFS.
    pub async fn load_encrypted(&self, cid: &str) -> Result<Vec<u8>, String> {
        let envelope_bytes = self.inner.fetch(cid).await
            .map_err(|e| format!("IPFS fetch failed: {}", e))?;
        let envelope = deserialize_envelope(&envelope_bytes)
            .map_err(|e| format!("Deserialization failed: {}", e))?;
        decrypt_data(&envelope, &self.password)
            .map_err(|e| format!("Decryption failed: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let data = b"model weights data here - could be megabytes";
        let password = "my-secure-password-123";

        let envelope = encrypt_data(data, password, Some("test-model".into())).unwrap();
        assert_eq!(envelope.version, 1);
        assert!(!envelope.ciphertext.is_empty());
        assert_ne!(envelope.ciphertext, data);
        assert_eq!(envelope.tag, Some("test-model".to_string()));

        let decrypted = decrypt_data(&envelope, password).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_wrong_password_fails() {
        let data = b"secret weights";
        let envelope = encrypt_data(data, "correct-password", None).unwrap();
        let result = decrypt_data(&envelope, "wrong-password");
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let data = b"secret weights";
        let mut envelope = encrypt_data(data, "password", None).unwrap();

        // Tamper with ciphertext
        if let Some(byte) = envelope.ciphertext.first_mut() {
            *byte ^= 0xFF;
        }

        let result = decrypt_data(&envelope, "password");
        assert!(result.is_err());
    }

    #[test]
    fn test_envelope_serialization_roundtrip() {
        let data = b"checkpoint data";
        let envelope = encrypt_data(data, "pass", None).unwrap();

        let bytes = serialize_envelope(&envelope).unwrap();
        let restored = deserialize_envelope(&bytes).unwrap();

        assert_eq!(restored.version, envelope.version);
        assert_eq!(restored.salt, envelope.salt);
        assert_eq!(restored.nonce, envelope.nonce);
        assert_eq!(restored.ciphertext, envelope.ciphertext);

        let decrypted = decrypt_data(&restored, "pass").unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_different_encryptions_produce_different_ciphertext() {
        let data = b"same data";
        let password = "same-password";

        let envelope1 = encrypt_data(data, password, None).unwrap();
        let envelope2 = encrypt_data(data, password, None).unwrap();

        // Different salt and nonce should produce different ciphertext
        assert_ne!(envelope1.salt, envelope2.salt);
        assert_ne!(envelope1.nonce, envelope2.nonce);
        assert_ne!(envelope1.ciphertext, envelope2.ciphertext);

        // Both should decrypt to the same data
        assert_eq!(decrypt_data(&envelope1, password).unwrap(), data);
        assert_eq!(decrypt_data(&envelope2, password).unwrap(), data);
    }

    #[test]
    fn test_large_data_encryption() {
        // 1MB of data
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let password = "large-data-password";

        let envelope = encrypt_data(&data, password, None).unwrap();
        let decrypted = decrypt_data(&envelope, password).unwrap();
        assert_eq!(decrypted, data);
    }
}
