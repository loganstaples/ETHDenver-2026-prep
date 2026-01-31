//! Platform-Specific Keychain Integration
//!
//! Provides secure key storage using native OS keychain services:
//! - macOS: Security.framework Keychain
//! - Linux: Secret Service D-Bus API (GNOME Keyring, KWallet)
//! - Windows: Data Protection API (DPAPI)
//!
//! Keys are never stored in plaintext and are protected by the OS-level
//! security mechanisms including user authentication and encryption.

use std::fmt;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::audit::{AuditEvent, AuditLogger, KeyOperation};

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "windows")]
pub mod windows;

/// Service name used for keychain storage
pub const SERVICE_NAME: &str = "com.helix.wallet";

/// Common types of items stored in keychain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeychainItemType {
    /// Private key
    PrivateKey,
    /// Mnemonic phrase
    Mnemonic,
    /// Encryption key for local storage
    EncryptionKey,
    /// API key or token
    ApiKey,
    /// Wallet password
    Password,
}

impl KeychainItemType {
    pub fn as_str(&self) -> &'static str {
        match self {
            KeychainItemType::PrivateKey => "private_key",
            KeychainItemType::Mnemonic => "mnemonic",
            KeychainItemType::EncryptionKey => "encryption_key",
            KeychainItemType::ApiKey => "api_key",
            KeychainItemType::Password => "password",
        }
    }
}

/// Metadata for a stored keychain item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeychainItemMetadata {
    /// Unique identifier
    pub id: String,
    /// Item type
    pub item_type: KeychainItemType,
    /// Associated wallet ID (if applicable)
    pub wallet_id: Option<String>,
    /// Creation timestamp
    pub created_at: i64,
    /// Last accessed timestamp
    pub last_accessed: i64,
    /// Additional description
    pub description: Option<String>,
}

/// Error types for keychain operations
#[derive(Debug, thiserror::Error)]
pub enum KeychainError {
    #[error("Keychain item not found: {0}")]
    NotFound(String),

    #[error("Access denied to keychain item: {0}")]
    AccessDenied(String),

    #[error("Keychain is locked")]
    KeychainLocked,

    #[error("User cancelled authentication")]
    UserCancelled,

    #[error("Invalid item data: {0}")]
    InvalidData(String),

    #[error("Duplicate item: {0}")]
    DuplicateItem(String),

    #[error("Keychain service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Platform not supported")]
    PlatformNotSupported,

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Trait for platform-specific keychain implementations
#[async_trait]
pub trait KeychainProvider: Send + Sync {
    /// Store a secret in the keychain
    async fn store(
        &self,
        key: &str,
        value: &[u8],
        item_type: KeychainItemType,
        metadata: Option<KeychainItemMetadata>,
    ) -> Result<(), KeychainError>;

    /// Retrieve a secret from the keychain
    async fn retrieve(&self, key: &str) -> Result<Vec<u8>, KeychainError>;

    /// Delete a secret from the keychain
    async fn delete(&self, key: &str) -> Result<(), KeychainError>;

    /// Check if a key exists
    async fn exists(&self, key: &str) -> Result<bool, KeychainError>;

    /// List all keys with a given prefix
    async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>, KeychainError>;

    /// Update a secret (atomic delete + store)
    async fn update(&self, key: &str, value: &[u8]) -> Result<(), KeychainError> {
        self.delete(key).await.ok(); // Ignore not found errors
        self.store(key, value, KeychainItemType::PrivateKey, None)
            .await
    }

    /// Get the name of this keychain provider
    fn provider_name(&self) -> &'static str;

    /// Check if the keychain is available and accessible
    async fn is_available(&self) -> bool;
}

/// Unified keychain manager that works across platforms
pub struct KeychainManager {
    /// The platform-specific provider
    provider: Box<dyn KeychainProvider>,
    /// Audit logger for security events
    audit_logger: AuditLogger,
}

impl KeychainManager {
    /// Create a new keychain manager with the appropriate platform provider
    pub fn new(audit_logger: AuditLogger) -> Result<Self> {
        let provider = Self::create_platform_provider()?;
        Ok(Self {
            provider,
            audit_logger,
        })
    }

    /// Create the appropriate provider for the current platform
    #[cfg(target_os = "macos")]
    fn create_platform_provider() -> Result<Box<dyn KeychainProvider>> {
        Ok(Box::new(macos::MacOSKeychain::new()?))
    }

    #[cfg(target_os = "linux")]
    fn create_platform_provider() -> Result<Box<dyn KeychainProvider>> {
        Ok(Box::new(linux::LinuxSecretService::new()?))
    }

    #[cfg(target_os = "windows")]
    fn create_platform_provider() -> Result<Box<dyn KeychainProvider>> {
        Ok(Box::new(windows::WindowsDpapi::new()?))
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    fn create_platform_provider() -> Result<Box<dyn KeychainProvider>> {
        Err(anyhow::anyhow!("Unsupported platform for keychain storage"))
    }

    /// Store a wallet private key
    pub async fn store_private_key(
        &self,
        wallet_id: &str,
        private_key: &[u8],
    ) -> Result<(), KeychainError> {
        let key = format!("{}.{}.private_key", SERVICE_NAME, wallet_id);

        self.provider
            .store(
                &key,
                private_key,
                KeychainItemType::PrivateKey,
                Some(KeychainItemMetadata {
                    id: wallet_id.to_string(),
                    item_type: KeychainItemType::PrivateKey,
                    wallet_id: Some(wallet_id.to_string()),
                    created_at: chrono::Utc::now().timestamp(),
                    last_accessed: chrono::Utc::now().timestamp(),
                    description: Some(format!("Private key for wallet {}", wallet_id)),
                }),
            )
            .await?;

        // Log the storage event
        let _ = self
            .audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyStoredInKeychain,
                Some(wallet_id.to_string()),
                Some(format!("Key stored in {}", self.provider.provider_name())),
            ))
            .await;

        Ok(())
    }

    /// Retrieve a wallet private key
    pub async fn retrieve_private_key(&self, wallet_id: &str) -> Result<Vec<u8>, KeychainError> {
        let key = format!("{}.{}.private_key", SERVICE_NAME, wallet_id);

        let result = self.provider.retrieve(&key).await?;

        // Log the retrieval event
        let _ = self
            .audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyRetrievedFromKeychain,
                Some(wallet_id.to_string()),
                Some(format!(
                    "Key retrieved from {}",
                    self.provider.provider_name()
                )),
            ))
            .await;

        Ok(result)
    }

    /// Delete a wallet private key
    pub async fn delete_private_key(&self, wallet_id: &str) -> Result<(), KeychainError> {
        let key = format!("{}.{}.private_key", SERVICE_NAME, wallet_id);

        self.provider.delete(&key).await?;

        // Log the deletion event
        let _ = self
            .audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyDeletedFromKeychain,
                Some(wallet_id.to_string()),
                Some(format!("Key deleted from {}", self.provider.provider_name())),
            ))
            .await;

        Ok(())
    }

    /// Store a mnemonic phrase
    pub async fn store_mnemonic(
        &self,
        wallet_id: &str,
        mnemonic: &str,
    ) -> Result<(), KeychainError> {
        let key = format!("{}.{}.mnemonic", SERVICE_NAME, wallet_id);

        self.provider
            .store(
                &key,
                mnemonic.as_bytes(),
                KeychainItemType::Mnemonic,
                Some(KeychainItemMetadata {
                    id: wallet_id.to_string(),
                    item_type: KeychainItemType::Mnemonic,
                    wallet_id: Some(wallet_id.to_string()),
                    created_at: chrono::Utc::now().timestamp(),
                    last_accessed: chrono::Utc::now().timestamp(),
                    description: Some(format!("Mnemonic for wallet {}", wallet_id)),
                }),
            )
            .await?;

        // Log the storage event
        let _ = self
            .audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyStoredInKeychain,
                Some(wallet_id.to_string()),
                Some("Mnemonic stored in keychain".to_string()),
            ))
            .await;

        Ok(())
    }

    /// Retrieve a mnemonic phrase
    pub async fn retrieve_mnemonic(&self, wallet_id: &str) -> Result<String, KeychainError> {
        let key = format!("{}.{}.mnemonic", SERVICE_NAME, wallet_id);

        let bytes = self.provider.retrieve(&key).await?;
        let mnemonic = String::from_utf8(bytes).map_err(|e| {
            KeychainError::InvalidData(format!("Invalid mnemonic encoding: {}", e))
        })?;

        // Log the retrieval event
        let _ = self
            .audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyRetrievedFromKeychain,
                Some(wallet_id.to_string()),
                Some("Mnemonic retrieved from keychain".to_string()),
            ))
            .await;

        Ok(mnemonic)
    }

    /// Store encryption key for local storage
    pub async fn store_encryption_key(
        &self,
        key_id: &str,
        encryption_key: &[u8],
    ) -> Result<(), KeychainError> {
        let key = format!("{}.encryption.{}", SERVICE_NAME, key_id);

        self.provider
            .store(&key, encryption_key, KeychainItemType::EncryptionKey, None)
            .await
    }

    /// Retrieve encryption key
    pub async fn retrieve_encryption_key(&self, key_id: &str) -> Result<Vec<u8>, KeychainError> {
        let key = format!("{}.encryption.{}", SERVICE_NAME, key_id);
        self.provider.retrieve(&key).await
    }

    /// List all stored wallet IDs
    pub async fn list_wallets(&self) -> Result<Vec<String>, KeychainError> {
        let prefix = format!("{}.wallet.", SERVICE_NAME);
        let keys = self.provider.list(Some(&prefix)).await?;

        let wallet_ids: Vec<String> = keys
            .iter()
            .filter_map(|k| {
                k.strip_prefix(&prefix)
                    .and_then(|s| s.split('.').next())
                    .map(|s| s.to_string())
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Ok(wallet_ids)
    }

    /// Check if keychain is available
    pub async fn is_available(&self) -> bool {
        self.provider.is_available().await
    }

    /// Get the provider name
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }
}

impl fmt::Debug for KeychainManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeychainManager")
            .field("provider", &self.provider.provider_name())
            .finish()
    }
}

/// In-memory keychain for testing and unsupported platforms
pub struct InMemoryKeychain {
    storage: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, Vec<u8>>>>,
}

impl InMemoryKeychain {
    pub fn new() -> Self {
        Self {
            storage: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for InMemoryKeychain {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl KeychainProvider for InMemoryKeychain {
    async fn store(
        &self,
        key: &str,
        value: &[u8],
        _item_type: KeychainItemType,
        _metadata: Option<KeychainItemMetadata>,
    ) -> Result<(), KeychainError> {
        let mut storage = self.storage.write().await;
        storage.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    async fn retrieve(&self, key: &str) -> Result<Vec<u8>, KeychainError> {
        let storage = self.storage.read().await;
        storage
            .get(key)
            .cloned()
            .ok_or_else(|| KeychainError::NotFound(key.to_string()))
    }

    async fn delete(&self, key: &str) -> Result<(), KeychainError> {
        let mut storage = self.storage.write().await;
        storage
            .remove(key)
            .ok_or_else(|| KeychainError::NotFound(key.to_string()))?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool, KeychainError> {
        let storage = self.storage.read().await;
        Ok(storage.contains_key(key))
    }

    async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>, KeychainError> {
        let storage = self.storage.read().await;
        let keys: Vec<String> = match prefix {
            Some(p) => storage
                .keys()
                .filter(|k| k.starts_with(p))
                .cloned()
                .collect(),
            None => storage.keys().cloned().collect(),
        };
        Ok(keys)
    }

    fn provider_name(&self) -> &'static str {
        "InMemory"
    }

    async fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_keychain() {
        let keychain = InMemoryKeychain::new();

        // Store
        keychain
            .store("test_key", b"secret_value", KeychainItemType::PrivateKey, None)
            .await
            .unwrap();

        // Retrieve
        let value = keychain.retrieve("test_key").await.unwrap();
        assert_eq!(value, b"secret_value");

        // Exists
        assert!(keychain.exists("test_key").await.unwrap());
        assert!(!keychain.exists("nonexistent").await.unwrap());

        // List
        let keys = keychain.list(Some("test_")).await.unwrap();
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&"test_key".to_string()));

        // Delete
        keychain.delete("test_key").await.unwrap();
        assert!(!keychain.exists("test_key").await.unwrap());
    }
}
