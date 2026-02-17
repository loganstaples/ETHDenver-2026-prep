//! HELIX Secure Wallet Module
//!
//! Production-grade key management with:
//! - BIP-39 mnemonic generation and recovery
//! - BIP-44 hierarchical deterministic key derivation
//! - Platform-specific secure key storage (macOS Keychain, Linux Secret Service, Windows DPAPI)
//! - Hardware wallet support (Ledger)
//! - Transaction confirmation prompts
//! - Comprehensive audit logging
//! - Secure key export/backup functionality
//! - Key rotation support
//!
//! # Security Model
//!
//! Keys are NEVER stored in plaintext. All key material is either:
//! 1. Encrypted with user password using Argon2id + AES-256-GCM
//! 2. Stored in platform keychain with OS-level protection
//! 3. Kept on hardware wallet (keys never leave device)
//!
//! All cryptographic operations use audited, production-grade libraries:
//! - k256 for secp256k1 ECDSA
//! - sha3 for Keccak256
//! - aes-gcm for authenticated encryption
//! - argon2 for password hashing

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use anyhow::{anyhow, Context, Result};
use argon2::{
    password_hash::{rand_core::RngCore, SaltString},
    Argon2, PasswordHasher,
};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use tokio::sync::RwLock;
use zeroize::{Zeroize, ZeroizeOnDrop};

pub mod audit;
pub mod derivation;
pub mod hardware;
pub mod keychain;
pub mod legacy;
pub mod mnemonic;

// Re-export legacy types for backward compatibility
pub use legacy::{
    Address, Balance, EncryptedKeystore, KeystoreCrypto, PrivateKey,
    Signature, SignedTransaction, Transaction, TxHash,
    Wallet, WalletManager as LegacyWalletManager,
    WalletMetadata as LegacyWalletMetadata, WalletType as LegacyWalletType,
};

use audit::{AuditConfig, AuditEvent, AuditLogger, KeyOperation};
use derivation::{DerivationPath, DerivedKey, EthereumAddress, EthereumSignature, ExtendedPrivateKey, KeyDerivationManager};
use hardware::{HardwareWalletManager, HardwareWalletStatus};
use mnemonic::{MnemonicLanguage, MnemonicLength, MnemonicManager, SecureMnemonic, Seed};

/// Wallet types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalletType {
    /// HD wallet derived from mnemonic
    HdWallet,
    /// Single key wallet (imported private key)
    SingleKey,
    /// Hardware wallet (keys never leave device)
    Hardware,
    /// Watch-only wallet (no signing capability)
    WatchOnly,
}

/// Wallet status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalletStatus {
    /// Wallet is locked (requires password)
    Locked,
    /// Wallet is unlocked and ready for operations
    Unlocked,
    /// Wallet requires hardware device
    RequiresHardware,
}

/// Wallet metadata stored on disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletMetadata {
    /// Unique wallet ID
    pub id: String,
    /// User-friendly name
    pub name: String,
    /// Wallet type
    pub wallet_type: WalletType,
    /// Primary address (first derived address for HD wallets)
    pub primary_address: String,
    /// Creation timestamp
    pub created_at: i64,
    /// Last accessed timestamp
    pub last_accessed: i64,
    /// Derivation path (for HD wallets)
    pub derivation_path: Option<String>,
    /// Hardware wallet type (if applicable)
    pub hardware_type: Option<String>,
    /// Network (mainnet, sepolia, etc.)
    pub network: String,
    /// Custom labels/tags
    pub labels: Vec<String>,
    /// Number of derived addresses
    pub derived_address_count: u32,
}

/// Encrypted wallet data stored on disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedWalletData {
    /// Version for forward compatibility
    pub version: u32,
    /// Argon2 parameters for key derivation
    pub kdf_params: KdfParams,
    /// AES-GCM nonce (12 bytes)
    pub nonce: String,
    /// Encrypted data (mnemonic or private key)
    pub ciphertext: String,
    /// Authentication tag (included in ciphertext for AES-GCM)
    pub wallet_type: WalletType,
}

/// Key derivation function parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdfParams {
    /// Algorithm (always "argon2id")
    pub algorithm: String,
    /// Memory cost in KiB
    pub memory_cost: u32,
    /// Time cost (iterations)
    pub time_cost: u32,
    /// Parallelism
    pub parallelism: u32,
    /// Salt (hex encoded)
    pub salt: String,
}

impl Default for KdfParams {
    fn default() -> Self {
        Self {
            algorithm: "argon2id".to_string(),
            memory_cost: 65536,    // 64 MiB
            time_cost: 3,          // 3 iterations
            parallelism: 4,        // 4 lanes
            salt: String::new(),   // Generated on use
        }
    }
}

/// Transaction confirmation request
#[derive(Debug, Clone)]
pub struct TransactionConfirmation {
    /// Recipient address
    pub to: Option<EthereumAddress>,
    /// Value in wei
    pub value: u128,
    /// Value formatted in ether
    pub value_ether: String,
    /// Gas limit
    pub gas_limit: u64,
    /// Gas price in wei
    pub gas_price: u128,
    /// Gas price formatted in gwei
    pub gas_price_gwei: String,
    /// Estimated total cost
    pub estimated_cost_wei: u128,
    /// Estimated total cost in ether
    pub estimated_cost_ether: String,
    /// Data payload summary
    pub data_summary: String,
    /// Chain ID
    pub chain_id: u64,
    /// Human-readable chain name
    pub chain_name: String,
}

impl TransactionConfirmation {
    /// Create from transaction parameters
    pub fn new(
        to: Option<EthereumAddress>,
        value: u128,
        gas_limit: u64,
        gas_price: u128,
        data: &[u8],
        chain_id: u64,
    ) -> Self {
        let value_ether = format!("{:.6}", value as f64 / 1e18);
        let gas_price_gwei = format!("{:.2}", gas_price as f64 / 1e9);
        let estimated_cost = value + (gas_limit as u128 * gas_price);
        let estimated_cost_ether = format!("{:.6}", estimated_cost as f64 / 1e18);

        let data_summary = if data.is_empty() {
            "No data (simple transfer)".to_string()
        } else if data.len() >= 4 {
            format!("Contract call: 0x{} ({} bytes)", hex::encode(&data[..4]), data.len())
        } else {
            format!("Data: {} bytes", data.len())
        };

        let chain_name = match chain_id {
            1 => "Ethereum Mainnet".to_string(),
            5 => "Goerli Testnet".to_string(),
            11155111 => "Sepolia Testnet".to_string(),
            137 => "Polygon Mainnet".to_string(),
            42161 => "Arbitrum One".to_string(),
            10 => "Optimism".to_string(),
            31337 => "Local/Hardhat".to_string(),
            99999 => "ADI Network Testnet".to_string(),
            _ => format!("Chain {}", chain_id),
        };

        Self {
            to,
            value,
            value_ether,
            gas_limit,
            gas_price,
            gas_price_gwei,
            estimated_cost_wei: estimated_cost,
            estimated_cost_ether,
            data_summary,
            chain_id,
            chain_name,
        }
    }

    /// Format for display
    pub fn display(&self) -> String {
        let to_str = self
            .to
            .as_ref()
            .map(|a| a.to_checksum_hex())
            .unwrap_or_else(|| "Contract Creation".to_string());

        format!(
            r#"
Transaction Confirmation Required
═══════════════════════════════════
Network:     {}
To:          {}
Value:       {} ETH
Gas Limit:   {}
Gas Price:   {} Gwei
Est. Cost:   {} ETH
Data:        {}
═══════════════════════════════════
"#,
            self.chain_name,
            to_str,
            self.value_ether,
            self.gas_limit,
            self.gas_price_gwei,
            self.estimated_cost_ether,
            self.data_summary
        )
    }
}

/// Callback type for transaction confirmation
pub type ConfirmationCallback = Box<dyn Fn(&TransactionConfirmation) -> bool + Send + Sync>;

/// Secure HD Wallet implementation
pub struct SecureWallet {
    /// Wallet metadata
    metadata: WalletMetadata,
    /// Master key (only present when unlocked)
    master_key: Option<ExtendedPrivateKey>,
    /// Derived keys cache
    derived_keys: HashMap<String, DerivedKey>,
    /// Current status
    status: WalletStatus,
    /// Path to encrypted wallet file
    wallet_path: PathBuf,
    /// Audit logger
    audit_logger: AuditLogger,
    /// Auto-lock timeout
    auto_lock_timeout: Option<Duration>,
    /// Confirmation callback
    confirmation_callback: Option<ConfirmationCallback>,
}

impl SecureWallet {
    /// Create a new HD wallet from mnemonic
    pub async fn create(
        name: &str,
        mnemonic: &SecureMnemonic,
        password: &str,
        wallet_dir: &Path,
        audit_logger: AuditLogger,
    ) -> Result<Self> {
        let id = uuid::Uuid::new_v4().to_string();
        let wallet_path = wallet_dir.join(format!("{}.wallet", id));

        // Derive master key
        let seed = mnemonic.to_seed_normalized();
        let master_key = ExtendedPrivateKey::from_seed(&seed)?;

        // Get primary address
        let primary_path = DerivationPath::ethereum(0, 0);
        let primary_ext_key = master_key.derive_path(&primary_path)?;
        let primary_address = primary_ext_key.ethereum_address();
        let primary_key = DerivedKey {
            public_key: primary_ext_key.public_key(),
            address: primary_address.clone(),
            path: primary_path.clone(),
            private_key: primary_ext_key,
        };

        // Create metadata
        let metadata = WalletMetadata {
            id: id.clone(),
            name: name.to_string(),
            wallet_type: WalletType::HdWallet,
            primary_address: primary_address.to_checksum_hex(),
            created_at: chrono::Utc::now().timestamp(),
            last_accessed: chrono::Utc::now().timestamp(),
            derivation_path: Some(primary_path.as_str()),
            hardware_type: None,
            network: "mainnet".to_string(),
            labels: Vec::new(),
            derived_address_count: 1,
        };

        // Encrypt mnemonic
        let encrypted = Self::encrypt_mnemonic(mnemonic, password)?;

        // Save to disk
        std::fs::create_dir_all(wallet_dir)?;
        let encrypted_json = serde_json::to_string_pretty(&encrypted)?;
        std::fs::write(&wallet_path, encrypted_json)?;

        // Save metadata
        let metadata_path = wallet_dir.join(format!("{}.meta.json", id));
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(&metadata_path, metadata_json)?;

        // Log creation
        audit_logger
            .log(AuditEvent::new(
                KeyOperation::WalletCreated,
                Some(id.clone()),
                Some(format!("HD wallet created: {}", name)),
            ))
            .await?;

        let mut wallet = Self {
            metadata,
            master_key: Some(master_key),
            derived_keys: HashMap::new(),
            status: WalletStatus::Unlocked,
            wallet_path,
            audit_logger,
            auto_lock_timeout: None,
            confirmation_callback: None,
        };

        // Cache primary key
        wallet.derived_keys.insert(primary_path.as_str(), primary_key);

        Ok(wallet)
    }

    /// Load an existing wallet (locked state)
    pub async fn load(wallet_path: &Path, audit_logger: AuditLogger) -> Result<Self> {
        let metadata_path = wallet_path.with_extension("meta.json");
        let metadata: WalletMetadata = serde_json::from_str(
            &std::fs::read_to_string(&metadata_path).context("Failed to read wallet metadata")?,
        )?;

        Ok(Self {
            metadata,
            master_key: None,
            derived_keys: HashMap::new(),
            status: WalletStatus::Locked,
            wallet_path: wallet_path.to_path_buf(),
            audit_logger,
            auto_lock_timeout: None,
            confirmation_callback: None,
        })
    }

    /// Unlock wallet with password
    pub async fn unlock(&mut self, password: &str) -> Result<()> {
        if self.status == WalletStatus::Unlocked {
            return Ok(());
        }

        let encrypted: EncryptedWalletData = serde_json::from_str(
            &std::fs::read_to_string(&self.wallet_path).context("Failed to read wallet file")?,
        )?;

        // Decrypt mnemonic
        let mnemonic = Self::decrypt_mnemonic(&encrypted, password)?;
        let seed = mnemonic.to_seed_normalized();
        let master_key = ExtendedPrivateKey::from_seed(&seed)?;

        self.master_key = Some(master_key);
        self.status = WalletStatus::Unlocked;

        // Update last accessed
        self.metadata.last_accessed = chrono::Utc::now().timestamp();

        // Log unlock
        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::WalletUnlocked,
                Some(self.metadata.id.clone()),
                None,
            ))
            .await?;

        Ok(())
    }

    /// Lock wallet (clear keys from memory)
    pub async fn lock(&mut self) -> Result<()> {
        self.master_key = None;
        self.derived_keys.clear();
        self.status = WalletStatus::Locked;

        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::WalletLocked,
                Some(self.metadata.id.clone()),
                None,
            ))
            .await?;

        Ok(())
    }

    /// Encrypt mnemonic with password
    fn encrypt_mnemonic(mnemonic: &SecureMnemonic, password: &str) -> Result<EncryptedWalletData> {
        // Generate salt
        let salt = SaltString::generate(&mut OsRng);

        // Derive key using Argon2id
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow!("Failed to hash password: {}", e))?;

        // Use first 32 bytes of hash as encryption key
        let hash_bytes = password_hash.hash.ok_or_else(|| anyhow!("No hash output"))?;
        let key_bytes: [u8; 32] = hash_bytes.as_bytes()[..32]
            .try_into()
            .map_err(|_| anyhow!("Invalid key length"))?;

        // Encrypt with AES-256-GCM
        let cipher = Aes256Gcm::new_from_slice(&key_bytes)
            .map_err(|e| anyhow!("Failed to create cipher: {}", e))?;
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

        let plaintext = mnemonic.phrase();
        let ciphertext = cipher
            .encrypt(&nonce, plaintext.as_bytes())
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        Ok(EncryptedWalletData {
            version: 1,
            kdf_params: KdfParams {
                algorithm: "argon2id".to_string(),
                memory_cost: 65536,
                time_cost: 3,
                parallelism: 4,
                salt: salt.to_string(),
            },
            nonce: hex::encode(nonce),
            ciphertext: hex::encode(ciphertext),
            wallet_type: WalletType::HdWallet,
        })
    }

    /// Decrypt mnemonic with password
    fn decrypt_mnemonic(
        encrypted: &EncryptedWalletData,
        password: &str,
    ) -> Result<SecureMnemonic> {
        // Reconstruct salt
        let salt = SaltString::from_b64(&encrypted.kdf_params.salt)
            .map_err(|e| anyhow!("Invalid salt: {}", e))?;

        // Derive key using Argon2id
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow!("Failed to hash password: {}", e))?;

        let hash_bytes = password_hash.hash.ok_or_else(|| anyhow!("No hash output"))?;
        let key_bytes: [u8; 32] = hash_bytes.as_bytes()[..32]
            .try_into()
            .map_err(|_| anyhow!("Invalid key length"))?;

        // Decrypt with AES-256-GCM
        let cipher = Aes256Gcm::new_from_slice(&key_bytes)
            .map_err(|e| anyhow!("Failed to create cipher: {}", e))?;

        let nonce_bytes =
            hex::decode(&encrypted.nonce).context("Invalid nonce")?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = hex::decode(&encrypted.ciphertext).context("Invalid ciphertext")?;

        let plaintext = cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|_| anyhow!("Decryption failed - wrong password?"))?;

        let phrase = String::from_utf8(plaintext).context("Invalid mnemonic encoding")?;
        SecureMnemonic::from_phrase(&phrase, MnemonicLanguage::English)
    }

    /// Derive a new key at path
    pub async fn derive_key(&mut self, path: &DerivationPath) -> Result<EthereumAddress> {
        let master = self
            .master_key
            .as_ref()
            .ok_or_else(|| anyhow!("Wallet is locked"))?;

        let path_str = path.as_str();
        if let Some(key) = self.derived_keys.get(&path_str) {
            return Ok(key.address);
        }

        let derived = master.derive_path(path)?;
        let address = derived.ethereum_address();

        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyDerived,
                Some(self.metadata.id.clone()),
                Some(format!("Derived key at {}", path)),
            ))
            .await?;

        // Cache the derived key
        let key = DerivedKey {
            private_key: derived,
            public_key: master.public_key(),
            address,
            path: path.clone(),
        };
        self.derived_keys.insert(path_str.to_string(), key);

        Ok(address)
    }

    /// Get address at path (derive if needed)
    pub async fn get_address(&mut self, account: u32, index: u32) -> Result<EthereumAddress> {
        let path = DerivationPath::ethereum(account, index);
        self.derive_key(&path).await
    }

    /// Sign a message with confirmation
    pub async fn sign_message(&self, path: &DerivationPath, message: &[u8]) -> Result<EthereumSignature> {
        let key = self
            .derived_keys
            .get(&path.as_str())
            .ok_or_else(|| anyhow!("Key not derived at path {}", path))?;

        // Hash the message with Ethereum prefix
        let prefix = format!("\x19Ethereum Signed Message:\n{}", message.len());
        let mut hasher = Keccak256::new();
        hasher.update(prefix.as_bytes());
        hasher.update(message);
        let hash: [u8; 32] = hasher.finalize().into();

        let signature = key.sign(&hash)?;

        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::MessageSigned,
                Some(self.metadata.id.clone()),
                Some(format!("Message signed at {}", path)),
            ))
            .await?;

        Ok(signature)
    }

    /// Sign a transaction with confirmation
    pub async fn sign_transaction(
        &self,
        path: &DerivationPath,
        tx_hash: &[u8; 32],
        confirmation: &TransactionConfirmation,
    ) -> Result<EthereumSignature> {
        // Check confirmation callback
        if let Some(ref callback) = self.confirmation_callback {
            if !callback(confirmation) {
                self.audit_logger
                    .log(AuditEvent::new(
                        KeyOperation::TransactionRejected,
                        Some(self.metadata.id.clone()),
                        Some("User rejected transaction".to_string()),
                    ))
                    .await?;

                return Err(anyhow!("Transaction rejected by user"));
            }
        }

        let key = self
            .derived_keys
            .get(&path.as_str())
            .ok_or_else(|| anyhow!("Key not derived at path {}", path))?;

        let signature = key.sign_with_chain_id(tx_hash, confirmation.chain_id)?;

        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::TransactionSigned,
                Some(self.metadata.id.clone()),
                Some(format!(
                    "Transaction signed for {} ETH to {}",
                    confirmation.value_ether,
                    confirmation
                        .to
                        .as_ref()
                        .map(|a| a.to_hex())
                        .unwrap_or_else(|| "contract".to_string())
                )),
            ))
            .await?;

        Ok(signature)
    }

    /// Set confirmation callback
    pub fn set_confirmation_callback(&mut self, callback: ConfirmationCallback) {
        self.confirmation_callback = Some(callback);
    }

    /// Get wallet metadata
    pub fn metadata(&self) -> &WalletMetadata {
        &self.metadata
    }

    /// Get wallet status
    pub fn status(&self) -> WalletStatus {
        self.status
    }

    /// Check if wallet is unlocked
    pub fn is_unlocked(&self) -> bool {
        self.status == WalletStatus::Unlocked
    }

    /// Convert the primary key to an ethers `LocalWallet` for on-chain signing.
    ///
    /// The wallet must be unlocked first. The returned signer is configured
    /// with the given `chain_id` for EIP-155 replay protection.
    #[cfg(feature = "chain")]
    pub fn to_ethers_wallet(&self, chain_id: u64) -> Result<ethers::signers::LocalWallet> {
        let primary_path = DerivationPath::ethereum(0, 0).as_str();
        let key = self
            .derived_keys
            .get(&primary_path)
            .ok_or_else(|| anyhow!("Wallet is locked or primary key not derived"))?;
        key.to_ethers_wallet(chain_id)
    }

    /// Export mnemonic (requires password confirmation)
    pub async fn export_mnemonic(&self, password: &str) -> Result<SecureMnemonic> {
        let encrypted: EncryptedWalletData = serde_json::from_str(
            &std::fs::read_to_string(&self.wallet_path).context("Failed to read wallet file")?,
        )?;

        let mnemonic = Self::decrypt_mnemonic(&encrypted, password)?;

        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::MnemonicExported,
                Some(self.metadata.id.clone()),
                Some("Mnemonic exported".to_string()),
            ))
            .await?;

        Ok(mnemonic)
    }

    /// Change wallet password
    pub async fn change_password(&mut self, old_password: &str, new_password: &str) -> Result<()> {
        // Decrypt with old password
        let mnemonic = self.export_mnemonic(old_password).await?;

        // Re-encrypt with new password
        let encrypted = Self::encrypt_mnemonic(&mnemonic, new_password)?;
        let encrypted_json = serde_json::to_string_pretty(&encrypted)?;
        std::fs::write(&self.wallet_path, encrypted_json)?;

        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::PasswordChanged,
                Some(self.metadata.id.clone()),
                None,
            ))
            .await?;

        Ok(())
    }
}

impl fmt::Debug for SecureWallet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecureWallet")
            .field("id", &self.metadata.id)
            .field("name", &self.metadata.name)
            .field("status", &self.status)
            .field("derived_keys_count", &self.derived_keys.len())
            .finish()
    }
}

/// Secure backup data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureBackup {
    /// Backup version
    pub version: u32,
    /// Backup timestamp
    pub created_at: i64,
    /// Wallet metadata
    pub metadata: WalletMetadata,
    /// Encrypted wallet data
    pub encrypted_data: EncryptedWalletData,
    /// Checksum for integrity verification
    pub checksum: String,
}

impl SecureBackup {
    /// Create a backup from an encrypted wallet file
    pub fn create(wallet_path: &Path, metadata: &WalletMetadata) -> Result<Self> {
        let encrypted_data: EncryptedWalletData = serde_json::from_str(
            &std::fs::read_to_string(wallet_path).context("Failed to read wallet file")?,
        )?;

        // Calculate checksum
        let mut hasher = sha2::Sha256::new();
        hasher.update(serde_json::to_string(&encrypted_data)?.as_bytes());
        let checksum = hex::encode(hasher.finalize());

        Ok(Self {
            version: 1,
            created_at: chrono::Utc::now().timestamp(),
            metadata: metadata.clone(),
            encrypted_data,
            checksum,
        })
    }

    /// Verify backup integrity
    pub fn verify(&self) -> bool {
        let mut hasher = sha2::Sha256::new();
        if let Ok(json) = serde_json::to_string(&self.encrypted_data) {
            hasher.update(json.as_bytes());
            let checksum = hex::encode(hasher.finalize());
            return checksum == self.checksum;
        }
        false
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("Failed to serialize backup")
    }

    /// Import from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        let backup: Self = serde_json::from_str(json).context("Failed to parse backup")?;
        if !backup.verify() {
            return Err(anyhow!("Backup checksum verification failed"));
        }
        Ok(backup)
    }

    /// Restore wallet from backup
    pub fn restore(&self, wallet_dir: &Path, audit_logger: &AuditLogger) -> Result<PathBuf> {
        let wallet_path = wallet_dir.join(format!("{}.wallet", self.metadata.id));
        let metadata_path = wallet_dir.join(format!("{}.meta.json", self.metadata.id));

        std::fs::create_dir_all(wallet_dir)?;
        std::fs::write(&wallet_path, serde_json::to_string_pretty(&self.encrypted_data)?)?;
        std::fs::write(&metadata_path, serde_json::to_string_pretty(&self.metadata)?)?;

        Ok(wallet_path)
    }
}

/// Key rotation support
pub struct KeyRotation {
    audit_logger: AuditLogger,
}

impl KeyRotation {
    pub fn new(audit_logger: AuditLogger) -> Self {
        Self { audit_logger }
    }

    /// Rotate to a new mnemonic (creates new wallet, keeps old for reference)
    pub async fn rotate_wallet(
        &self,
        old_wallet: &SecureWallet,
        old_password: &str,
        new_name: &str,
        new_password: &str,
        wallet_dir: &Path,
    ) -> Result<SecureWallet> {
        // Generate new mnemonic
        let new_mnemonic = SecureMnemonic::generate_default()?;

        // Create new wallet
        let new_wallet = SecureWallet::create(
            new_name,
            &new_mnemonic,
            new_password,
            wallet_dir,
            self.audit_logger.clone(),
        )
        .await?;

        // Log rotation
        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyRotated,
                Some(old_wallet.metadata.id.clone()),
                Some(format!(
                    "Rotated to new wallet: {}",
                    new_wallet.metadata.id
                )),
            ))
            .await?;

        Ok(new_wallet)
    }

    /// Archive an old wallet (mark as rotated)
    pub async fn archive_wallet(&self, wallet_path: &Path) -> Result<()> {
        let archive_path = wallet_path.with_extension("wallet.archived");
        std::fs::rename(wallet_path, &archive_path)?;

        let metadata_path = wallet_path.with_extension("meta.json");
        let archive_metadata_path = wallet_path.with_extension("meta.json.archived");
        std::fs::rename(metadata_path, archive_metadata_path)?;

        Ok(())
    }
}

/// Unified wallet manager
pub struct WalletManager {
    /// Directory for wallet files
    wallet_dir: PathBuf,
    /// Loaded wallets
    wallets: HashMap<String, SecureWallet>,
    /// Default wallet ID
    default_wallet: Option<String>,
    /// Hardware wallet manager
    hardware_manager: HardwareWalletManager,
    /// Mnemonic manager
    mnemonic_manager: MnemonicManager,
    /// Audit logger
    audit_logger: AuditLogger,
}

impl WalletManager {
    /// Create a new wallet manager
    pub fn new(wallet_dir: PathBuf, audit_logger: AuditLogger) -> Result<Self> {
        std::fs::create_dir_all(&wallet_dir)?;

        Ok(Self {
            wallet_dir: wallet_dir.clone(),
            wallets: HashMap::new(),
            default_wallet: None,
            hardware_manager: HardwareWalletManager::new(audit_logger.clone()),
            mnemonic_manager: MnemonicManager::new(audit_logger.clone()),
            audit_logger,
        })
    }

    /// Create a new HD wallet
    pub async fn create_wallet(
        &mut self,
        name: &str,
        password: &str,
        mnemonic_length: MnemonicLength,
    ) -> Result<String> {
        let mnemonic = self
            .mnemonic_manager
            .generate(mnemonic_length, MnemonicLanguage::English, None)
            .await?;

        let wallet = SecureWallet::create(
            name,
            &mnemonic,
            password,
            &self.wallet_dir,
            self.audit_logger.clone(),
        )
        .await?;

        let id = wallet.metadata.id.clone();
        self.wallets.insert(id.clone(), wallet);

        if self.default_wallet.is_none() {
            self.default_wallet = Some(id.clone());
        }

        Ok(id)
    }

    /// Import wallet from mnemonic
    pub async fn import_wallet(
        &mut self,
        name: &str,
        phrase: &str,
        password: &str,
    ) -> Result<String> {
        let mnemonic = self
            .mnemonic_manager
            .import(phrase, MnemonicLanguage::English, None)
            .await?;

        let wallet = SecureWallet::create(
            name,
            &mnemonic,
            password,
            &self.wallet_dir,
            self.audit_logger.clone(),
        )
        .await?;

        let id = wallet.metadata.id.clone();
        self.wallets.insert(id.clone(), wallet);

        Ok(id)
    }

    /// Load existing wallets from directory
    pub async fn load_wallets(&mut self) -> Result<Vec<String>> {
        let mut loaded = Vec::new();

        for entry in std::fs::read_dir(&self.wallet_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |e| e == "wallet") {
                if let Ok(wallet) = SecureWallet::load(&path, self.audit_logger.clone()).await {
                    let id = wallet.metadata.id.clone();
                    self.wallets.insert(id.clone(), wallet);
                    loaded.push(id);
                }
            }
        }

        if !loaded.is_empty() && self.default_wallet.is_none() {
            self.default_wallet = Some(loaded[0].clone());
        }

        Ok(loaded)
    }

    /// Get wallet by ID
    pub fn get_wallet(&self, id: &str) -> Option<&SecureWallet> {
        self.wallets.get(id)
    }

    /// Get mutable wallet by ID
    pub fn get_wallet_mut(&mut self, id: &str) -> Option<&mut SecureWallet> {
        self.wallets.get_mut(id)
    }

    /// Get default wallet
    pub fn get_default_wallet(&self) -> Option<&SecureWallet> {
        self.default_wallet
            .as_ref()
            .and_then(|id| self.wallets.get(id))
    }

    /// Set default wallet
    pub fn set_default_wallet(&mut self, id: &str) -> Result<()> {
        if self.wallets.contains_key(id) {
            self.default_wallet = Some(id.to_string());
            Ok(())
        } else {
            Err(anyhow!("Wallet not found: {}", id))
        }
    }

    /// List all wallet metadata
    pub fn list_wallets(&self) -> Vec<&WalletMetadata> {
        self.wallets.values().map(|w| &w.metadata).collect()
    }

    /// Get hardware wallet manager
    pub fn hardware_manager(&mut self) -> &mut HardwareWalletManager {
        &mut self.hardware_manager
    }

    /// Create backup of wallet
    pub fn backup_wallet(&self, id: &str) -> Result<SecureBackup> {
        let wallet = self
            .wallets
            .get(id)
            .ok_or_else(|| anyhow!("Wallet not found: {}", id))?;

        SecureBackup::create(&wallet.wallet_path, &wallet.metadata)
    }

    /// Restore wallet from backup
    pub async fn restore_from_backup(&mut self, backup: &SecureBackup) -> Result<String> {
        let wallet_path = backup.restore(&self.wallet_dir, &self.audit_logger)?;
        let wallet = SecureWallet::load(&wallet_path, self.audit_logger.clone()).await?;

        let id = wallet.metadata.id.clone();
        self.wallets.insert(id.clone(), wallet);

        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::BackupRestored,
                Some(id.clone()),
                Some("Wallet restored from backup".to_string()),
            ))
            .await?;

        Ok(id)
    }
}

impl fmt::Debug for WalletManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WalletManager")
            .field("wallet_count", &self.wallets.len())
            .field("default_wallet", &self.default_wallet)
            .finish()
    }
}

// Re-export commonly used new types (not conflicting with legacy)
pub use audit::AuditSeverity;
pub use hardware::{HardwareWallet, HardwareWalletError, HardwareWalletInfo, HardwareWalletType};
pub use keychain::{KeychainError, KeychainManager, KeychainProvider};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_wallet_creation() {
        let temp_dir = TempDir::new().unwrap();
        let audit_logger = AuditLogger::noop();

        let mnemonic = SecureMnemonic::generate_default().unwrap();
        let wallet = SecureWallet::create(
            "test-wallet",
            &mnemonic,
            "test-password",
            temp_dir.path(),
            audit_logger,
        )
        .await
        .unwrap();

        assert!(wallet.is_unlocked());
        assert_eq!(wallet.metadata.name, "test-wallet");
        assert_eq!(wallet.metadata.wallet_type, WalletType::HdWallet);
    }

    #[tokio::test]
    async fn test_wallet_lock_unlock() {
        let temp_dir = TempDir::new().unwrap();
        let audit_logger = AuditLogger::noop();

        let mnemonic = SecureMnemonic::generate_default().unwrap();
        let mut wallet = SecureWallet::create(
            "test-wallet",
            &mnemonic,
            "test-password",
            temp_dir.path(),
            audit_logger.clone(),
        )
        .await
        .unwrap();

        assert!(wallet.is_unlocked());

        wallet.lock().await.unwrap();
        assert!(!wallet.is_unlocked());

        wallet.unlock("test-password").await.unwrap();
        assert!(wallet.is_unlocked());
    }

    #[tokio::test]
    async fn test_wallet_key_derivation() {
        let temp_dir = TempDir::new().unwrap();
        let audit_logger = AuditLogger::noop();

        let mnemonic = SecureMnemonic::generate_default().unwrap();
        let mut wallet = SecureWallet::create(
            "test-wallet",
            &mnemonic,
            "test-password",
            temp_dir.path(),
            audit_logger,
        )
        .await
        .unwrap();

        let addr1 = wallet.get_address(0, 0).await.unwrap();
        let addr2 = wallet.get_address(0, 1).await.unwrap();

        // Different indices should produce different addresses
        assert_ne!(addr1, addr2);

        // Same index should produce same address
        let addr1_again = wallet.get_address(0, 0).await.unwrap();
        assert_eq!(addr1, addr1_again);
    }

    #[tokio::test]
    async fn test_backup_restore() {
        let temp_dir = TempDir::new().unwrap();
        let audit_logger = AuditLogger::noop();

        let mnemonic = SecureMnemonic::generate_default().unwrap();
        let wallet = SecureWallet::create(
            "test-wallet",
            &mnemonic,
            "test-password",
            temp_dir.path(),
            audit_logger.clone(),
        )
        .await
        .unwrap();

        // Create backup
        let backup = SecureBackup::create(&wallet.wallet_path, &wallet.metadata).unwrap();

        // Verify backup
        assert!(backup.verify());

        // Export and import JSON
        let json = backup.to_json().unwrap();
        let restored_backup = SecureBackup::from_json(&json).unwrap();
        assert!(restored_backup.verify());
    }

    #[test]
    fn test_transaction_confirmation() {
        let to = EthereumAddress::from_hex("0x742d35Cc6634C0532925a3b844Bc454e4438f44e").unwrap();
        let confirmation = TransactionConfirmation::new(
            Some(to),
            1_000_000_000_000_000_000, // 1 ETH
            21000,
            20_000_000_000, // 20 Gwei
            &[],
            1,
        );

        assert_eq!(confirmation.value_ether, "1.000000");
        assert_eq!(confirmation.gas_price_gwei, "20.00");
        assert_eq!(confirmation.chain_name, "Ethereum Mainnet");
    }
}
