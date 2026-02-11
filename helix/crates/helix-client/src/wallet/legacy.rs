//! Legacy HELIX Wallet Module
//!
//! Provides basic key management and transaction signing for HELIX network
//! participants. For production use, prefer `wallet::SecureWallet` which
//! offers HD derivation, hardware wallet support, and audited encryption.
//!
//! **Deprecation notice:** The XOR-based keystore encryption in this module
//! is NOT cryptographically secure. Use `SecureWallet` with AES-256-GCM for
//! any real-world key storage.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use tokio::sync::RwLock;

/// Ethereum address type (20 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Address([u8; 20]);

impl Address {
    /// Create a zero address
    pub const fn zero() -> Self {
        Self([0u8; 20])
    }

    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 20]) -> Self {
        Self(bytes)
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8; 20] {
        &self.0
    }

    /// Convert to hex string with 0x prefix
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.0))
    }

    /// Parse from hex string (with or without 0x prefix)
    pub fn from_hex(s: &str) -> Result<Self> {
        let s = s.strip_prefix("0x").unwrap_or(s);
        let bytes = hex::decode(s).context("Invalid hex string")?;
        if bytes.len() != 20 {
            return Err(anyhow!("Address must be 20 bytes"));
        }
        let mut arr = [0u8; 20];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Serialize for Address {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Address {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Transaction hash type (32 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TxHash([u8; 32]);

impl TxHash {
    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string with 0x prefix
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.0))
    }
}

impl std::fmt::Display for TxHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Private key type (32 bytes)
#[derive(Clone)]
pub struct PrivateKey([u8; 32]);

impl PrivateKey {
    /// Generate a new random private key using OS-level CSPRNG.
    pub fn generate() -> Self {
        let mut key = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut key);
        Self(key)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Create from hex string
    pub fn from_hex(s: &str) -> Result<Self> {
        let s = s.strip_prefix("0x").unwrap_or(s);
        let bytes = hex::decode(s).context("Invalid hex string")?;
        if bytes.len() != 32 {
            return Err(anyhow!("Private key must be 32 bytes"));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }

    /// Get as bytes (use with caution - exposes secret)
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string (use with caution - exposes secret)
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.0))
    }

    /// Derive the public address from this private key
    pub fn to_address(&self) -> Address {
        // Simplified address derivation (in production, use secp256k1)
        let hash = Self::keccak256(&self.0);
        let mut addr = [0u8; 20];
        addr.copy_from_slice(&hash[12..32]);
        Address(addr)
    }

    /// Sign a message hash
    pub fn sign(&self, message_hash: &[u8; 32]) -> Signature {
        // Simplified signature (in production, use secp256k1)
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];

        // Mix private key with message hash
        for i in 0..32 {
            r[i] = self.0[i] ^ message_hash[i];
            s[i] = self.0[(i + 16) % 32] ^ message_hash[(i + 8) % 32];
        }

        // Hash to get final signature parts
        r = Self::keccak256(&r);
        s = Self::keccak256(&s);

        Signature { r, s, v: 27 }
    }

    /// Keccak256 hash using the `sha3` crate.
    fn keccak256(data: &[u8]) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
}

impl std::fmt::Debug for PrivateKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PrivateKey([REDACTED])")
    }
}

impl Drop for PrivateKey {
    fn drop(&mut self) {
        // Zero out the private key on drop for security
        self.0.fill(0);
    }
}

/// ECDSA signature
#[derive(Debug, Clone)]
pub struct Signature {
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub v: u8,
}

impl Signature {
    /// Encode signature as bytes (65 bytes: r ++ s ++ v)
    pub fn to_bytes(&self) -> [u8; 65] {
        let mut bytes = [0u8; 65];
        bytes[..32].copy_from_slice(&self.r);
        bytes[32..64].copy_from_slice(&self.s);
        bytes[64] = self.v;
        bytes
    }

    /// Encode as hex string
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.to_bytes()))
    }
}

/// Wallet types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalletType {
    /// In-memory wallet (keys not persisted)
    Memory,
    /// File-based encrypted wallet
    File,
    /// Hardware wallet (Ledger, Trezor)
    Hardware,
    /// Keystore file (Web3 format)
    Keystore,
}

/// Wallet metadata stored alongside keys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletMetadata {
    /// Wallet name
    pub name: String,
    /// Wallet type
    pub wallet_type: WalletType,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Address
    pub address: Address,
    /// Path to key file (if applicable)
    pub key_path: Option<PathBuf>,
    /// Network this wallet is configured for
    pub network: String,
    /// Custom labels/tags
    pub labels: Vec<String>,
}

/// Encrypted keystore format (compatible with Web3)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedKeystore {
    /// Version
    pub version: u32,
    /// Unique identifier
    pub id: String,
    /// Address
    pub address: String,
    /// Crypto parameters
    pub crypto: KeystoreCrypto,
}

/// Keystore crypto parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeystoreCrypto {
    /// Cipher algorithm
    pub cipher: String,
    /// Cipher text (encrypted private key)
    pub ciphertext: String,
    /// Cipher parameters
    pub cipherparams: CipherParams,
    /// Key derivation function
    pub kdf: String,
    /// KDF parameters
    pub kdfparams: KdfParams,
    /// MAC for verification
    pub mac: String,
}

/// Cipher parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CipherParams {
    /// Initialization vector
    pub iv: String,
}

/// Key derivation function parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdfParams {
    /// Derived key length
    pub dklen: u32,
    /// Number of iterations
    pub n: u32,
    /// Block size
    pub r: u32,
    /// Parallelization
    pub p: u32,
    /// Salt
    pub salt: String,
}

/// Transaction to be signed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Nonce
    pub nonce: u64,
    /// Gas price in wei
    pub gas_price: u128,
    /// Gas limit
    pub gas_limit: u64,
    /// Recipient address
    pub to: Option<Address>,
    /// Value in wei
    pub value: u128,
    /// Transaction data
    pub data: Vec<u8>,
    /// Chain ID
    pub chain_id: u64,
}

impl Transaction {
    /// Create a new transaction
    pub fn new() -> Self {
        Self {
            nonce: 0,
            gas_price: 20_000_000_000, // 20 gwei
            gas_limit: 21000,
            to: None,
            value: 0,
            data: Vec::new(),
            chain_id: 1,
        }
    }

    /// Set nonce
    pub fn with_nonce(mut self, nonce: u64) -> Self {
        self.nonce = nonce;
        self
    }

    /// Set gas price
    pub fn with_gas_price(mut self, gas_price: u128) -> Self {
        self.gas_price = gas_price;
        self
    }

    /// Set gas limit
    pub fn with_gas_limit(mut self, gas_limit: u64) -> Self {
        self.gas_limit = gas_limit;
        self
    }

    /// Set recipient
    pub fn with_to(mut self, to: Address) -> Self {
        self.to = Some(to);
        self
    }

    /// Set value
    pub fn with_value(mut self, value: u128) -> Self {
        self.value = value;
        self
    }

    /// Set data
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Set chain ID
    pub fn with_chain_id(mut self, chain_id: u64) -> Self {
        self.chain_id = chain_id;
        self
    }

    /// Calculate transaction hash for signing
    pub fn signing_hash(&self) -> [u8; 32] {
        // RLP encode transaction for signing (simplified)
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&self.nonce.to_be_bytes());
        encoded.extend_from_slice(&self.gas_price.to_be_bytes());
        encoded.extend_from_slice(&self.gas_limit.to_be_bytes());
        if let Some(to) = &self.to {
            encoded.extend_from_slice(to.as_bytes());
        }
        encoded.extend_from_slice(&self.value.to_be_bytes());
        encoded.extend_from_slice(&self.data);
        encoded.extend_from_slice(&self.chain_id.to_be_bytes());

        PrivateKey::keccak256(&encoded)
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self::new()
    }
}

/// Signed transaction ready for broadcast
#[derive(Debug, Clone)]
pub struct SignedTransaction {
    /// Original transaction
    pub transaction: Transaction,
    /// Signature
    pub signature: Signature,
    /// Transaction hash
    pub hash: TxHash,
}

impl SignedTransaction {
    /// Encode the signed transaction for broadcast
    pub fn encode(&self) -> Vec<u8> {
        // Simplified encoding (in production, use proper RLP)
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&self.transaction.nonce.to_be_bytes());
        encoded.extend_from_slice(&self.transaction.gas_price.to_be_bytes());
        encoded.extend_from_slice(&self.transaction.gas_limit.to_be_bytes());
        if let Some(to) = &self.transaction.to {
            encoded.extend_from_slice(to.as_bytes());
        }
        encoded.extend_from_slice(&self.transaction.value.to_be_bytes());
        encoded.extend_from_slice(&self.transaction.data);
        encoded.extend_from_slice(&self.signature.to_bytes());
        encoded
    }

    /// Get hex-encoded transaction
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.encode()))
    }
}

/// HELIX Wallet - manages keys and transaction signing
pub struct Wallet {
    /// Wallet metadata
    metadata: WalletMetadata,
    /// Private key (in memory only)
    private_key: PrivateKey,
    /// Current nonce
    nonce: Arc<RwLock<u64>>,
    /// Default chain ID
    chain_id: u64,
}

impl Wallet {
    /// Generate a new wallet with random keys
    pub fn generate(name: &str, network: &str) -> Self {
        let private_key = PrivateKey::generate();
        let address = private_key.to_address();

        Self {
            metadata: WalletMetadata {
                name: name.to_string(),
                wallet_type: WalletType::Memory,
                created_at: chrono::Utc::now(),
                address,
                key_path: None,
                network: network.to_string(),
                labels: Vec::new(),
            },
            private_key,
            nonce: Arc::new(RwLock::new(0)),
            chain_id: match network {
                "mainnet" => 1,
                "goerli" => 5,
                "sepolia" => 11155111,
                "local" | "localhost" | "hardhat" => 31337,
                "anvil" => 31337,
                _ => 31337,
            },
        }
    }

    /// Create wallet from private key
    pub fn from_private_key(name: &str, private_key: PrivateKey, network: &str) -> Self {
        let address = private_key.to_address();

        Self {
            metadata: WalletMetadata {
                name: name.to_string(),
                wallet_type: WalletType::Memory,
                created_at: chrono::Utc::now(),
                address,
                key_path: None,
                network: network.to_string(),
                labels: Vec::new(),
            },
            private_key,
            nonce: Arc::new(RwLock::new(0)),
            chain_id: match network {
                "mainnet" => 1,
                "goerli" => 5,
                "sepolia" => 11155111,
                "local" | "localhost" | "hardhat" => 31337,
                "anvil" => 31337,
                _ => 31337,
            },
        }
    }

    /// Load wallet from encrypted keystore file
    pub fn load_keystore(path: &Path, password: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read keystore file")?;
        let keystore: EncryptedKeystore = serde_json::from_str(&content)
            .context("Failed to parse keystore")?;

        // Decrypt private key (simplified - in production use proper decryption)
        let private_key = Self::decrypt_keystore(&keystore, password)?;
        let address = private_key.to_address();

        Ok(Self {
            metadata: WalletMetadata {
                name: keystore.id.clone(),
                wallet_type: WalletType::Keystore,
                created_at: chrono::Utc::now(),
                address,
                key_path: Some(path.to_path_buf()),
                network: "local".to_string(),
                labels: Vec::new(),
            },
            private_key,
            nonce: Arc::new(RwLock::new(0)),
            chain_id: 31337,
        })
    }

    /// Save wallet to encrypted keystore file
    pub fn save_keystore(&self, path: &Path, password: &str) -> Result<()> {
        let keystore = self.encrypt_keystore(password)?;
        let content = serde_json::to_string_pretty(&keystore)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Decrypt keystore (simplified)
    fn decrypt_keystore(keystore: &EncryptedKeystore, password: &str) -> Result<PrivateKey> {
        // In production, use proper scrypt/pbkdf2 + AES decryption
        // This is a simplified placeholder
        let ciphertext = hex::decode(&keystore.crypto.ciphertext)
            .context("Invalid ciphertext")?;
        let salt = hex::decode(&keystore.crypto.kdfparams.salt)
            .context("Invalid salt")?;

        // Derive key from password (simplified)
        let mut key = [0u8; 32];
        for (i, byte) in password.as_bytes().iter().enumerate() {
            key[i % 32] ^= *byte;
        }
        for (i, byte) in salt.iter().enumerate() {
            key[i % 32] ^= *byte;
        }

        // "Decrypt" (XOR for simplicity)
        let mut decrypted = [0u8; 32];
        for (i, byte) in ciphertext.iter().take(32).enumerate() {
            decrypted[i] = byte ^ key[i];
        }

        Ok(PrivateKey::from_bytes(decrypted))
    }

    /// Encrypt wallet for keystore (simplified)
    fn encrypt_keystore(&self, password: &str) -> Result<EncryptedKeystore> {
        // Generate random salt and IV using CSPRNG
        let mut salt = vec![0u8; 16];
        let mut iv = vec![0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        rand::rngs::OsRng.fill_bytes(&mut iv);

        // Derive key from password (simplified)
        let mut key = [0u8; 32];
        for (i, byte) in password.as_bytes().iter().enumerate() {
            key[i % 32] ^= *byte;
        }
        for (i, byte) in salt.iter().enumerate() {
            key[i % 32] ^= *byte;
        }

        // "Encrypt" (XOR for simplicity)
        let mut ciphertext = [0u8; 32];
        for i in 0..32 {
            ciphertext[i] = self.private_key.0[i] ^ key[i];
        }

        // Calculate MAC
        let mut mac_input = Vec::new();
        mac_input.extend_from_slice(&key[16..32]);
        mac_input.extend_from_slice(&ciphertext);
        let mac = PrivateKey::keccak256(&mac_input);

        Ok(EncryptedKeystore {
            version: 3,
            id: uuid::Uuid::new_v4().to_string(),
            address: self.metadata.address.to_hex()[2..].to_string(),
            crypto: KeystoreCrypto {
                cipher: "aes-128-ctr".to_string(),
                ciphertext: hex::encode(ciphertext),
                cipherparams: CipherParams {
                    iv: hex::encode(iv),
                },
                kdf: "scrypt".to_string(),
                kdfparams: KdfParams {
                    dklen: 32,
                    n: 262144,
                    r: 8,
                    p: 1,
                    salt: hex::encode(salt),
                },
                mac: hex::encode(mac),
            },
        })
    }

    /// Get wallet address
    pub fn address(&self) -> Address {
        self.metadata.address
    }

    /// Get wallet metadata
    pub fn metadata(&self) -> &WalletMetadata {
        &self.metadata
    }

    /// Sign a message
    pub fn sign_message(&self, message: &[u8]) -> Signature {
        // Hash the message with Ethereum prefix
        let prefix = format!("\x19Ethereum Signed Message:\n{}", message.len());
        let mut prefixed = Vec::new();
        prefixed.extend_from_slice(prefix.as_bytes());
        prefixed.extend_from_slice(message);

        let hash = PrivateKey::keccak256(&prefixed);
        self.private_key.sign(&hash)
    }

    /// Sign a transaction
    pub fn sign_transaction(&self, tx: Transaction) -> Result<SignedTransaction> {
        let signing_hash = tx.signing_hash();
        let signature = self.private_key.sign(&signing_hash);

        // Adjust v for EIP-155
        let v = signature.v + self.chain_id as u8 * 2 + 8;
        let adjusted_sig = Signature {
            r: signature.r,
            s: signature.s,
            v,
        };

        // Calculate transaction hash
        let mut tx_data = Vec::new();
        tx_data.extend_from_slice(&signing_hash);
        tx_data.extend_from_slice(&adjusted_sig.to_bytes());
        let tx_hash = TxHash::from_bytes(PrivateKey::keccak256(&tx_data));

        Ok(SignedTransaction {
            transaction: tx,
            signature: adjusted_sig,
            hash: tx_hash,
        })
    }

    /// Sign typed data (EIP-712)
    pub fn sign_typed_data(&self, domain_separator: &[u8; 32], struct_hash: &[u8; 32]) -> Signature {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(b"\x19\x01");
        encoded.extend_from_slice(domain_separator);
        encoded.extend_from_slice(struct_hash);

        let hash = PrivateKey::keccak256(&encoded);
        self.private_key.sign(&hash)
    }

    /// Get current nonce
    pub async fn get_nonce(&self) -> u64 {
        *self.nonce.read().await
    }

    /// Set nonce
    pub async fn set_nonce(&self, nonce: u64) {
        *self.nonce.write().await = nonce;
    }

    /// Increment nonce and return new value
    pub async fn increment_nonce(&self) -> u64 {
        let mut nonce = self.nonce.write().await;
        *nonce += 1;
        *nonce
    }

    /// Get chain ID
    pub fn chain_id(&self) -> u64 {
        self.chain_id
    }

    /// Set chain ID
    pub fn set_chain_id(&mut self, chain_id: u64) {
        self.chain_id = chain_id;
    }

    /// Export private key (use with extreme caution)
    pub fn export_private_key(&self) -> String {
        self.private_key.to_hex()
    }
}

impl std::fmt::Debug for Wallet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wallet")
            .field("metadata", &self.metadata)
            .field("private_key", &"[REDACTED]")
            .field("chain_id", &self.chain_id)
            .finish()
    }
}

/// Wallet manager for handling multiple wallets
pub struct WalletManager {
    /// Wallets by name
    wallets: Arc<RwLock<std::collections::HashMap<String, Wallet>>>,
    /// Default wallet name
    default_wallet: Arc<RwLock<Option<String>>>,
    /// Wallets directory
    wallets_dir: PathBuf,
}

impl WalletManager {
    /// Create a new wallet manager
    pub fn new(wallets_dir: PathBuf) -> Self {
        Self {
            wallets: Arc::new(RwLock::new(std::collections::HashMap::new())),
            default_wallet: Arc::new(RwLock::new(None)),
            wallets_dir,
        }
    }

    /// Initialize wallet manager and load existing wallets
    pub async fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.wallets_dir)?;

        // Load wallets from directory
        for entry in std::fs::read_dir(&self.wallets_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                // Try to load wallet metadata
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(metadata) = serde_json::from_str::<WalletMetadata>(&content) {
                        tracing::info!("Found wallet: {} ({})", metadata.name, metadata.address);
                    }
                }
            }
        }

        Ok(())
    }

    /// Create a new wallet
    pub async fn create_wallet(&self, name: &str, network: &str) -> Result<Address> {
        let wallet = Wallet::generate(name, network);
        let address = wallet.address();

        // Save wallet metadata
        let metadata_path = self.wallets_dir.join(format!("{}_metadata.json", name));
        let metadata_json = serde_json::to_string_pretty(&wallet.metadata())?;
        std::fs::write(&metadata_path, metadata_json)?;

        // Store in memory
        let mut wallets = self.wallets.write().await;
        wallets.insert(name.to_string(), wallet);

        // Set as default if first wallet
        let mut default = self.default_wallet.write().await;
        if default.is_none() {
            *default = Some(name.to_string());
        }

        Ok(address)
    }

    /// Import wallet from private key
    pub async fn import_private_key(&self, name: &str, private_key: &str, network: &str) -> Result<Address> {
        let pk = PrivateKey::from_hex(private_key)?;
        let wallet = Wallet::from_private_key(name, pk, network);
        let address = wallet.address();

        // Save wallet metadata
        let metadata_path = self.wallets_dir.join(format!("{}_metadata.json", name));
        let metadata_json = serde_json::to_string_pretty(&wallet.metadata())?;
        std::fs::write(&metadata_path, metadata_json)?;

        // Store in memory
        let mut wallets = self.wallets.write().await;
        wallets.insert(name.to_string(), wallet);

        Ok(address)
    }

    /// Get wallet by name
    pub async fn get_wallet(&self, name: &str) -> Option<Address> {
        let wallets = self.wallets.read().await;
        wallets.get(name).map(|w| w.address())
    }

    /// Get default wallet
    pub async fn get_default_wallet(&self) -> Option<Address> {
        let default = self.default_wallet.read().await;
        if let Some(name) = default.as_ref() {
            self.get_wallet(name).await
        } else {
            None
        }
    }

    /// Set default wallet
    pub async fn set_default_wallet(&self, name: &str) -> Result<()> {
        let wallets = self.wallets.read().await;
        if !wallets.contains_key(name) {
            return Err(anyhow!("Wallet '{}' not found", name));
        }

        let mut default = self.default_wallet.write().await;
        *default = Some(name.to_string());
        Ok(())
    }

    /// List all wallets
    pub async fn list_wallets(&self) -> Vec<WalletMetadata> {
        let wallets = self.wallets.read().await;
        wallets.values().map(|w| w.metadata().clone()).collect()
    }

    /// Sign a message with wallet
    pub async fn sign_message(&self, wallet_name: &str, message: &[u8]) -> Result<Signature> {
        let wallets = self.wallets.read().await;
        let wallet = wallets.get(wallet_name)
            .ok_or_else(|| anyhow!("Wallet '{}' not found", wallet_name))?;
        Ok(wallet.sign_message(message))
    }

    /// Sign a transaction with wallet
    pub async fn sign_transaction(&self, wallet_name: &str, tx: Transaction) -> Result<SignedTransaction> {
        let wallets = self.wallets.read().await;
        let wallet = wallets.get(wallet_name)
            .ok_or_else(|| anyhow!("Wallet '{}' not found", wallet_name))?;
        wallet.sign_transaction(tx)
    }

    /// Export wallet keystore
    pub async fn export_keystore(&self, wallet_name: &str, password: &str) -> Result<PathBuf> {
        let wallets = self.wallets.read().await;
        let wallet = wallets.get(wallet_name)
            .ok_or_else(|| anyhow!("Wallet '{}' not found", wallet_name))?;

        let path = self.wallets_dir.join(format!("{}.json", wallet_name));
        wallet.save_keystore(&path, password)?;
        Ok(path)
    }
}

/// Balance type for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    /// Balance in wei
    pub wei: u128,
    /// Balance in ether (as string for precision)
    pub ether: String,
}

impl Balance {
    /// Create from wei
    pub fn from_wei(wei: u128) -> Self {
        let ether = wei as f64 / 1e18;
        Self {
            wei,
            ether: format!("{:.6}", ether),
        }
    }

    /// Create from ether
    pub fn from_ether(ether: f64) -> Self {
        let wei = (ether * 1e18) as u128;
        Self {
            wei,
            ether: format!("{:.6}", ether),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_hex() {
        let addr = Address::from_bytes([0x12; 20]);
        let hex = addr.to_hex();
        assert!(hex.starts_with("0x"));

        let parsed = Address::from_hex(&hex).unwrap();
        assert_eq!(addr, parsed);
    }

    #[test]
    fn test_private_key_generation() {
        let pk1 = PrivateKey::generate();
        let pk2 = PrivateKey::generate();
        // Keys should be different
        assert_ne!(pk1.0, pk2.0);
    }

    #[test]
    fn test_wallet_generation() {
        let wallet = Wallet::generate("test", "local");
        let addr = wallet.address();
        assert_ne!(addr, Address::zero());
    }

    #[test]
    fn test_message_signing() {
        let wallet = Wallet::generate("test", "local");
        let message = b"Hello, HELIX!";
        let sig = wallet.sign_message(message);
        assert_ne!(sig.r, [0u8; 32]);
        assert_ne!(sig.s, [0u8; 32]);
    }

    #[test]
    fn test_transaction_signing() {
        let wallet = Wallet::generate("test", "local");
        let tx = Transaction::new()
            .with_to(Address::from_bytes([0xAB; 20]))
            .with_value(1_000_000_000_000_000_000); // 1 ETH

        let signed = wallet.sign_transaction(tx).unwrap();
        assert!(!signed.to_hex().is_empty());
    }
}
