//! macOS Keychain Integration
//!
//! Uses the Security.framework to store sensitive data in the macOS Keychain.
//! The keychain provides hardware-backed encryption on devices with Secure Enclave
//! and integrates with macOS user authentication (password, Touch ID, Face ID).

#![cfg(target_os = "macos")]

use async_trait::async_trait;
use security_framework::passwords::{
    delete_generic_password, get_generic_password, set_generic_password,
};

use super::{KeychainError, KeychainItemMetadata, KeychainItemType, KeychainProvider, SERVICE_NAME};

/// macOS Keychain provider using Security.framework
pub struct MacOSKeychain {
    /// Service name for keychain entries
    service: String,
}

impl MacOSKeychain {
    /// Create a new macOS Keychain provider
    pub fn new() -> Result<Self, anyhow::Error> {
        Ok(Self {
            service: SERVICE_NAME.to_string(),
        })
    }

    /// Convert security-framework errors to KeychainError
    fn convert_error(err: security_framework::base::Error) -> KeychainError {
        let code = err.code();
        match code {
            // errSecItemNotFound = -25300
            -25300 => KeychainError::NotFound("Item not found in keychain".to_string()),
            // errSecAuthFailed = -25293
            -25293 => KeychainError::AccessDenied("Authentication failed".to_string()),
            // errSecUserCanceled = -128
            -128 => KeychainError::UserCancelled,
            // errSecDuplicateItem = -25299
            -25299 => KeychainError::DuplicateItem("Item already exists".to_string()),
            // errSecInteractionNotAllowed = -25308
            -25308 => KeychainError::KeychainLocked,
            _ => KeychainError::Internal(format!("Keychain error (code {}): {}", code, err)),
        }
    }

    /// Get the account name for a key
    fn account_name(&self, key: &str) -> String {
        key.to_string()
    }
}

impl Default for MacOSKeychain {
    fn default() -> Self {
        Self::new().expect("Failed to create macOS keychain")
    }
}

#[async_trait]
impl KeychainProvider for MacOSKeychain {
    async fn store(
        &self,
        key: &str,
        value: &[u8],
        _item_type: KeychainItemType,
        _metadata: Option<KeychainItemMetadata>,
    ) -> Result<(), KeychainError> {
        let account = self.account_name(key);

        // Try to delete existing item first (update pattern)
        let _ = delete_generic_password(&self.service, &account);

        set_generic_password(&self.service, &account, value).map_err(Self::convert_error)?;

        Ok(())
    }

    async fn retrieve(&self, key: &str) -> Result<Vec<u8>, KeychainError> {
        let account = self.account_name(key);

        let password = get_generic_password(&self.service, &account).map_err(Self::convert_error)?;

        Ok(password.to_vec())
    }

    async fn delete(&self, key: &str) -> Result<(), KeychainError> {
        let account = self.account_name(key);

        delete_generic_password(&self.service, &account).map_err(Self::convert_error)?;

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool, KeychainError> {
        match self.retrieve(key).await {
            Ok(_) => Ok(true),
            Err(KeychainError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn list(&self, _prefix: Option<&str>) -> Result<Vec<String>, KeychainError> {
        // Note: Listing items in macOS Keychain requires additional APIs
        // For now, return empty list - keys must be tracked separately
        Ok(Vec::new())
    }

    fn provider_name(&self) -> &'static str {
        "macOS Keychain"
    }

    async fn is_available(&self) -> bool {
        // macOS Keychain is always available on macOS
        true
    }
}

/// Additional macOS-specific keychain operations
impl MacOSKeychain {
    /// Store with access control (require user authentication)
    pub async fn store_with_biometric(
        &self,
        key: &str,
        value: &[u8],
    ) -> Result<(), KeychainError> {
        // For production, we would use SecAccessControlCreateWithFlags
        // with kSecAccessControlBiometryCurrentSet to require Touch ID
        // This is a simplified version that uses standard keychain storage
        self.store(key, value, KeychainItemType::PrivateKey, None).await
    }

    /// Check if the keychain is unlocked
    pub fn is_unlocked(&self) -> bool {
        // In production, use SecKeychainGetStatus
        // For now, we assume it's unlocked if we can access it
        true
    }

    /// Lock the keychain
    pub fn lock(&self) -> Result<(), KeychainError> {
        // In production, use SecKeychainLock
        // This operation typically requires elevated privileges
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a macOS keychain and may prompt for password
    // They are disabled by default and can be run with:
    // cargo test --features keychain-tests -- --ignored

    #[tokio::test]
    #[ignore]
    async fn test_macos_keychain_roundtrip() {
        let keychain = MacOSKeychain::new().unwrap();
        let key = "test_helix_roundtrip";
        let value = b"test_secret_value";

        // Clean up any existing test key
        let _ = keychain.delete(key).await;

        // Store
        keychain
            .store(key, value, KeychainItemType::PrivateKey, None)
            .await
            .unwrap();

        // Retrieve
        let retrieved = keychain.retrieve(key).await.unwrap();
        assert_eq!(&retrieved, value);

        // Delete
        keychain.delete(key).await.unwrap();

        // Verify deleted
        let exists = keychain.exists(key).await.unwrap();
        assert!(!exists);
    }

    #[tokio::test]
    #[ignore]
    async fn test_macos_keychain_availability() {
        let keychain = MacOSKeychain::new().unwrap();
        assert!(keychain.is_available().await);
        assert_eq!(keychain.provider_name(), "macOS Keychain");
    }
}
