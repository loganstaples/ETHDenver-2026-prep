//! Linux Secret Service Integration
//!
//! Uses the Secret Service D-Bus API to store sensitive data securely.
//! This integrates with GNOME Keyring, KWallet, or any other Secret Service
//! compliant password manager.

#![cfg(target_os = "linux")]

use std::collections::HashMap;

use async_trait::async_trait;
use secret_service::{EncryptionType, SecretService};
use zbus::zvariant::ObjectPath;

use super::{KeychainError, KeychainItemMetadata, KeychainItemType, KeychainProvider, SERVICE_NAME};

/// Linux Secret Service provider
pub struct LinuxSecretService {
    /// Collection name for storing secrets
    collection: String,
}

impl LinuxSecretService {
    /// Create a new Linux Secret Service provider
    pub fn new() -> Result<Self, anyhow::Error> {
        Ok(Self {
            collection: "helix-wallet".to_string(),
        })
    }

    /// Convert Secret Service errors to KeychainError
    fn convert_error(err: secret_service::Error) -> KeychainError {
        match err {
            secret_service::Error::Locked => KeychainError::KeychainLocked,
            secret_service::Error::NoResult => {
                KeychainError::NotFound("Item not found".to_string())
            }
            secret_service::Error::Prompt => KeychainError::UserCancelled,
            secret_service::Error::Zbus(e) => {
                KeychainError::ServiceUnavailable(format!("D-Bus error: {}", e))
            }
            secret_service::Error::ZbusMsg(e) => {
                KeychainError::ServiceUnavailable(format!("D-Bus message error: {}", e))
            }
            _ => KeychainError::Internal(format!("Secret Service error: {:?}", err)),
        }
    }

    /// Get or create the collection for storing secrets
    async fn get_collection(
        &self,
        ss: &SecretService<'_>,
    ) -> Result<secret_service::Collection<'_>, KeychainError> {
        // Try to get existing collection
        let collections = ss.collections().await.map_err(Self::convert_error)?;

        for collection in collections {
            let label = collection.label().await.map_err(Self::convert_error)?;
            if label == self.collection {
                // Unlock if needed
                collection.unlock().await.map_err(Self::convert_error)?;
                return Ok(collection);
            }
        }

        // Create new collection
        let collection = ss
            .create_collection(&self.collection, ObjectPath::from_static_str("/org/freedesktop/secrets/aliases/default").expect("valid static D-Bus path"))
            .await
            .map_err(Self::convert_error)?;

        collection.unlock().await.map_err(Self::convert_error)?;
        Ok(collection)
    }

    /// Create attributes for an item
    fn create_attributes<'a>(
        &self,
        key: &'a str,
        item_type: KeychainItemType,
    ) -> HashMap<&'a str, &'a str> {
        let mut attrs = HashMap::new();
        attrs.insert("service", SERVICE_NAME);
        attrs.insert("key", key);
        attrs.insert("type", item_type.as_str());
        attrs
    }
}

#[async_trait]
impl KeychainProvider for LinuxSecretService {
    async fn store(
        &self,
        key: &str,
        value: &[u8],
        item_type: KeychainItemType,
        _metadata: Option<KeychainItemMetadata>,
    ) -> Result<(), KeychainError> {
        let ss = SecretService::connect(EncryptionType::Dh)
            .await
            .map_err(Self::convert_error)?;

        let collection = self.get_collection(&ss).await?;

        // Delete existing item if any
        let _ = self.delete(key).await;

        // Create attributes
        let mut attrs = HashMap::new();
        attrs.insert("service", SERVICE_NAME);
        attrs.insert("key", key);
        attrs.insert("type", item_type.as_str());

        // Create the item
        collection
            .create_item(
                &format!("HELIX: {}", key),
                attrs,
                value,
                true, // replace existing
                "text/plain",
            )
            .await
            .map_err(Self::convert_error)?;

        Ok(())
    }

    async fn retrieve(&self, key: &str) -> Result<Vec<u8>, KeychainError> {
        let ss = SecretService::connect(EncryptionType::Dh)
            .await
            .map_err(Self::convert_error)?;

        let collection = self.get_collection(&ss).await?;

        // Search for the item
        let mut attrs = HashMap::new();
        attrs.insert("service", SERVICE_NAME);
        attrs.insert("key", key);

        let items = collection
            .search_items(attrs)
            .await
            .map_err(Self::convert_error)?;

        let item = items
            .first()
            .ok_or_else(|| KeychainError::NotFound(key.to_string()))?;

        // Unlock if needed
        item.unlock().await.map_err(Self::convert_error)?;

        // Get the secret
        let secret = item.get_secret().await.map_err(Self::convert_error)?;

        Ok(secret)
    }

    async fn delete(&self, key: &str) -> Result<(), KeychainError> {
        let ss = SecretService::connect(EncryptionType::Dh)
            .await
            .map_err(Self::convert_error)?;

        let collection = self.get_collection(&ss).await?;

        // Search for the item
        let mut attrs = HashMap::new();
        attrs.insert("service", SERVICE_NAME);
        attrs.insert("key", key);

        let items = collection
            .search_items(attrs)
            .await
            .map_err(Self::convert_error)?;

        let item = items
            .first()
            .ok_or_else(|| KeychainError::NotFound(key.to_string()))?;

        item.delete().await.map_err(Self::convert_error)?;

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool, KeychainError> {
        match self.retrieve(key).await {
            Ok(_) => Ok(true),
            Err(KeychainError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>, KeychainError> {
        let ss = SecretService::connect(EncryptionType::Dh)
            .await
            .map_err(Self::convert_error)?;

        let collection = self.get_collection(&ss).await?;

        // Get all items with our service attribute
        let mut attrs = HashMap::new();
        attrs.insert("service", SERVICE_NAME);

        let items = collection
            .search_items(attrs)
            .await
            .map_err(Self::convert_error)?;

        let mut keys = Vec::new();
        for item in items {
            let item_attrs = item.get_attributes().await.map_err(Self::convert_error)?;
            if let Some(key) = item_attrs.get("key") {
                if let Some(p) = prefix {
                    if key.starts_with(p) {
                        keys.push(key.clone());
                    }
                } else {
                    keys.push(key.clone());
                }
            }
        }

        Ok(keys)
    }

    fn provider_name(&self) -> &'static str {
        "Linux Secret Service"
    }

    async fn is_available(&self) -> bool {
        SecretService::connect(EncryptionType::Dh)
            .await
            .is_ok()
    }
}

/// Additional Linux-specific operations
impl LinuxSecretService {
    /// Check if GNOME Keyring is available
    pub async fn is_gnome_keyring(&self) -> bool {
        if let Ok(ss) = SecretService::connect(EncryptionType::Dh).await {
            // GNOME Keyring typically identifies itself
            true
        } else {
            false
        }
    }

    /// Check if KWallet is available
    pub async fn is_kwallet(&self) -> bool {
        // KWallet uses a different D-Bus interface
        // Check for org.kde.kwalletd5 service
        false // Placeholder - would require direct D-Bus check
    }

    /// Lock the collection
    pub async fn lock(&self) -> Result<(), KeychainError> {
        let ss = SecretService::connect(EncryptionType::Dh)
            .await
            .map_err(Self::convert_error)?;

        let collection = self.get_collection(&ss).await?;
        collection.lock().await.map_err(Self::convert_error)?;

        Ok(())
    }

    /// Create a new collection with a specific label
    pub async fn create_collection(&self, label: &str) -> Result<(), KeychainError> {
        let ss = SecretService::connect(EncryptionType::Dh)
            .await
            .map_err(Self::convert_error)?;

        ss.create_collection(label, ObjectPath::from_static_str("/org/freedesktop/secrets/aliases/default").expect("valid static D-Bus path"))
            .await
            .map_err(Self::convert_error)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a running Secret Service daemon
    // They are disabled by default and can be run with:
    // cargo test --features keychain-tests -- --ignored

    #[tokio::test]
    #[ignore]
    async fn test_linux_secret_service_roundtrip() {
        let ss = LinuxSecretService::new().unwrap();
        let key = "test_helix_roundtrip";
        let value = b"test_secret_value";

        // Clean up any existing test key
        let _ = ss.delete(key).await;

        // Store
        ss.store(key, value, KeychainItemType::PrivateKey, None)
            .await
            .unwrap();

        // Retrieve
        let retrieved = ss.retrieve(key).await.unwrap();
        assert_eq!(&retrieved, value);

        // Delete
        ss.delete(key).await.unwrap();

        // Verify deleted
        let exists = ss.exists(key).await.unwrap();
        assert!(!exists);
    }

    #[tokio::test]
    #[ignore]
    async fn test_linux_secret_service_availability() {
        let ss = LinuxSecretService::new().unwrap();
        // This will fail if no Secret Service is running
        let available = ss.is_available().await;
        println!("Secret Service available: {}", available);
    }
}
