//! Windows Data Protection API (DPAPI) Integration
//!
//! Uses Windows DPAPI to encrypt and store sensitive data.
//! Data is encrypted using the user's login credentials and can only
//! be decrypted by the same user on the same machine.

#![cfg(target_os = "windows")]

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

use async_trait::async_trait;
use windows::Win32::Security::Cryptography::{
    CryptProtectData, CryptUnprotectData, CRYPTPROTECT_LOCAL_MACHINE, CRYPTPROTECT_UI_FORBIDDEN,
    CRYPT_INTEGER_BLOB,
};
use windows::Win32::System::Memory::LocalFree;

use super::{KeychainError, KeychainItemMetadata, KeychainItemType, KeychainProvider, SERVICE_NAME};

/// Windows DPAPI provider
pub struct WindowsDpapi {
    /// Directory for storing encrypted files
    storage_dir: PathBuf,
    /// Additional entropy for encryption
    entropy: Vec<u8>,
}

impl WindowsDpapi {
    /// Create a new Windows DPAPI provider
    pub fn new() -> Result<Self, anyhow::Error> {
        let storage_dir = dirs::data_local_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find local app data directory"))?
            .join("helix")
            .join("keychain");

        fs::create_dir_all(&storage_dir)?;

        // Generate or load entropy
        let entropy_path = storage_dir.join(".entropy");
        let entropy = if entropy_path.exists() {
            fs::read(&entropy_path)?
        } else {
            let mut entropy = vec![0u8; 32];
            getrandom::getrandom(&mut entropy)
                .map_err(|e| anyhow::anyhow!("Failed to generate entropy: {}", e))?;
            fs::write(&entropy_path, &entropy)?;
            entropy
        };

        Ok(Self {
            storage_dir,
            entropy,
        })
    }

    /// Encrypt data using DPAPI
    fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>, KeychainError> {
        unsafe {
            let data_in = CRYPT_INTEGER_BLOB {
                cbData: data.len() as u32,
                pbData: data.as_ptr() as *mut u8,
            };

            let entropy_blob = CRYPT_INTEGER_BLOB {
                cbData: self.entropy.len() as u32,
                pbData: self.entropy.as_ptr() as *mut u8,
            };

            let mut data_out = CRYPT_INTEGER_BLOB {
                cbData: 0,
                pbData: std::ptr::null_mut(),
            };

            let result = CryptProtectData(
                &data_in,
                None,
                Some(&entropy_blob),
                None,
                None,
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut data_out,
            );

            if result.is_err() {
                return Err(KeychainError::Internal(format!(
                    "DPAPI encryption failed: {:?}",
                    result
                )));
            }

            let encrypted =
                std::slice::from_raw_parts(data_out.pbData, data_out.cbData as usize).to_vec();

            // Free the memory allocated by CryptProtectData
            LocalFree(data_out.pbData as *mut _);

            Ok(encrypted)
        }
    }

    /// Decrypt data using DPAPI
    fn decrypt(&self, encrypted: &[u8]) -> Result<Vec<u8>, KeychainError> {
        unsafe {
            let data_in = CRYPT_INTEGER_BLOB {
                cbData: encrypted.len() as u32,
                pbData: encrypted.as_ptr() as *mut u8,
            };

            let entropy_blob = CRYPT_INTEGER_BLOB {
                cbData: self.entropy.len() as u32,
                pbData: self.entropy.as_ptr() as *mut u8,
            };

            let mut data_out = CRYPT_INTEGER_BLOB {
                cbData: 0,
                pbData: std::ptr::null_mut(),
            };

            let result = CryptUnprotectData(
                &data_in,
                None,
                Some(&entropy_blob),
                None,
                None,
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut data_out,
            );

            if result.is_err() {
                return Err(KeychainError::Internal(format!(
                    "DPAPI decryption failed: {:?}",
                    result
                )));
            }

            let decrypted =
                std::slice::from_raw_parts(data_out.pbData, data_out.cbData as usize).to_vec();

            // Free the memory allocated by CryptUnprotectData
            LocalFree(data_out.pbData as *mut _);

            Ok(decrypted)
        }
    }

    /// Get the file path for a key
    fn key_path(&self, key: &str) -> PathBuf {
        // Hash the key to create a safe filename
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let hash = hex::encode(hasher.finalize());
        self.storage_dir.join(format!("{}.dpapi", &hash[..32]))
    }

    /// Get the metadata path for a key
    fn metadata_path(&self, key: &str) -> PathBuf {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let hash = hex::encode(hasher.finalize());
        self.storage_dir.join(format!("{}.meta", &hash[..32]))
    }

    /// Index file for mapping key names to hashes
    fn index_path(&self) -> PathBuf {
        self.storage_dir.join(".index.json")
    }

    /// Load or create the key index
    fn load_index(&self) -> Result<HashMap<String, String>, KeychainError> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(HashMap::new());
        }

        let content = fs::read_to_string(&path)
            .map_err(|e| KeychainError::Internal(format!("Failed to read index: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| KeychainError::Internal(format!("Failed to parse index: {}", e)))
    }

    /// Save the key index
    fn save_index(&self, index: &HashMap<String, String>) -> Result<(), KeychainError> {
        let path = self.index_path();
        let content = serde_json::to_string_pretty(index)
            .map_err(|e| KeychainError::Internal(format!("Failed to serialize index: {}", e)))?;

        fs::write(&path, content)
            .map_err(|e| KeychainError::Internal(format!("Failed to write index: {}", e)))
    }
}

#[async_trait]
impl KeychainProvider for WindowsDpapi {
    async fn store(
        &self,
        key: &str,
        value: &[u8],
        item_type: KeychainItemType,
        metadata: Option<KeychainItemMetadata>,
    ) -> Result<(), KeychainError> {
        // Encrypt the value
        let encrypted = self.encrypt(value)?;

        // Write to file
        let path = self.key_path(key);
        fs::write(&path, &encrypted)
            .map_err(|e| KeychainError::Internal(format!("Failed to write key file: {}", e)))?;

        // Save metadata if provided
        if let Some(meta) = metadata {
            let meta_path = self.metadata_path(key);
            let meta_json = serde_json::to_string(&meta)
                .map_err(|e| KeychainError::Internal(format!("Failed to serialize metadata: {}", e)))?;
            fs::write(&meta_path, meta_json)
                .map_err(|e| KeychainError::Internal(format!("Failed to write metadata: {}", e)))?;
        }

        // Update index
        let mut index = self.load_index()?;
        let hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(key.as_bytes());
            hex::encode(hasher.finalize())[..32].to_string()
        };
        index.insert(key.to_string(), hash);
        self.save_index(&index)?;

        Ok(())
    }

    async fn retrieve(&self, key: &str) -> Result<Vec<u8>, KeychainError> {
        let path = self.key_path(key);

        if !path.exists() {
            return Err(KeychainError::NotFound(key.to_string()));
        }

        let encrypted = fs::read(&path)
            .map_err(|e| KeychainError::Internal(format!("Failed to read key file: {}", e)))?;

        self.decrypt(&encrypted)
    }

    async fn delete(&self, key: &str) -> Result<(), KeychainError> {
        let path = self.key_path(key);
        let meta_path = self.metadata_path(key);

        if !path.exists() {
            return Err(KeychainError::NotFound(key.to_string()));
        }

        fs::remove_file(&path)
            .map_err(|e| KeychainError::Internal(format!("Failed to delete key file: {}", e)))?;

        // Remove metadata if exists
        let _ = fs::remove_file(&meta_path);

        // Update index
        let mut index = self.load_index()?;
        index.remove(key);
        self.save_index(&index)?;

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool, KeychainError> {
        Ok(self.key_path(key).exists())
    }

    async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>, KeychainError> {
        let index = self.load_index()?;

        let keys: Vec<String> = match prefix {
            Some(p) => index
                .keys()
                .filter(|k| k.starts_with(p))
                .cloned()
                .collect(),
            None => index.keys().cloned().collect(),
        };

        Ok(keys)
    }

    fn provider_name(&self) -> &'static str {
        "Windows DPAPI"
    }

    async fn is_available(&self) -> bool {
        // DPAPI is always available on Windows
        self.storage_dir.exists() || fs::create_dir_all(&self.storage_dir).is_ok()
    }
}

/// Additional Windows-specific operations
impl WindowsDpapi {
    /// Encrypt with machine scope (accessible by any user on this machine)
    pub fn encrypt_machine_scope(&self, data: &[u8]) -> Result<Vec<u8>, KeychainError> {
        unsafe {
            let data_in = CRYPT_INTEGER_BLOB {
                cbData: data.len() as u32,
                pbData: data.as_ptr() as *mut u8,
            };

            let mut data_out = CRYPT_INTEGER_BLOB {
                cbData: 0,
                pbData: std::ptr::null_mut(),
            };

            let result = CryptProtectData(
                &data_in,
                None,
                None,
                None,
                None,
                CRYPTPROTECT_LOCAL_MACHINE | CRYPTPROTECT_UI_FORBIDDEN,
                &mut data_out,
            );

            if result.is_err() {
                return Err(KeychainError::Internal(format!(
                    "DPAPI encryption failed: {:?}",
                    result
                )));
            }

            let encrypted =
                std::slice::from_raw_parts(data_out.pbData, data_out.cbData as usize).to_vec();

            LocalFree(data_out.pbData as *mut _);

            Ok(encrypted)
        }
    }

    /// Securely wipe the storage directory
    pub fn wipe(&self) -> Result<(), KeychainError> {
        if self.storage_dir.exists() {
            // Overwrite all files with random data before deletion
            for entry in fs::read_dir(&self.storage_dir)
                .map_err(|e| KeychainError::Internal(format!("Failed to read directory: {}", e)))?
            {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() {
                        // Get file size
                        if let Ok(metadata) = fs::metadata(&path) {
                            let size = metadata.len() as usize;
                            // Overwrite with random data
                            let mut random_data = vec![0u8; size];
                            let _ = getrandom::getrandom(&mut random_data);
                            let _ = fs::write(&path, &random_data);
                        }
                        // Delete the file
                        let _ = fs::remove_file(&path);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require Windows
    // They are disabled by default

    #[tokio::test]
    #[ignore]
    async fn test_windows_dpapi_roundtrip() {
        let dpapi = WindowsDpapi::new().unwrap();
        let key = "test_helix_roundtrip";
        let value = b"test_secret_value";

        // Clean up any existing test key
        let _ = dpapi.delete(key).await;

        // Store
        dpapi
            .store(key, value, KeychainItemType::PrivateKey, None)
            .await
            .unwrap();

        // Retrieve
        let retrieved = dpapi.retrieve(key).await.unwrap();
        assert_eq!(&retrieved, value);

        // Delete
        dpapi.delete(key).await.unwrap();

        // Verify deleted
        let exists = dpapi.exists(key).await.unwrap();
        assert!(!exists);
    }

    #[tokio::test]
    #[ignore]
    async fn test_windows_dpapi_availability() {
        let dpapi = WindowsDpapi::new().unwrap();
        assert!(dpapi.is_available().await);
        assert_eq!(dpapi.provider_name(), "Windows DPAPI");
    }
}
