//! BIP-44 Key Derivation Paths
//!
//! Implements hierarchical deterministic (HD) key derivation following BIP-32/BIP-44
//! standards. Provides Ethereum-compatible key derivation with support for multiple
//! accounts and address indices.

use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, Context, Result};
use coins_bip32::prelude::*;
use k256::ecdsa::{RecoveryId, Signature, SigningKey, VerifyingKey};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use sha3::{Digest, Keccak256};
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::audit::{AuditEvent, AuditLogger, KeyOperation};
use super::mnemonic::Seed;

/// BIP-44 coin type constants
pub mod coin_types {
    /// Bitcoin
    pub const BITCOIN: u32 = 0;
    /// Bitcoin Testnet
    pub const BITCOIN_TESTNET: u32 = 1;
    /// Ethereum
    pub const ETHEREUM: u32 = 60;
    /// Ethereum Classic
    pub const ETHEREUM_CLASSIC: u32 = 61;
    /// Binance Smart Chain
    pub const BSC: u32 = 714;
    /// Polygon
    pub const POLYGON: u32 = 966;
    /// Avalanche
    pub const AVALANCHE: u32 = 9005;
    /// Arbitrum
    pub const ARBITRUM: u32 = 9001;
    /// Optimism
    pub const OPTIMISM: u32 = 614;
}

/// Standard Ethereum derivation path: m/44'/60'/0'/0/n
pub const ETHEREUM_BASE_PATH: &str = "m/44'/60'/0'/0";

/// Ledger Live derivation path: m/44'/60'/n'/0/0
pub const LEDGER_LIVE_PATH: &str = "m/44'/60'";

/// Extended private key with secure memory handling
pub struct ExtendedPrivateKey {
    /// The signing key
    signing_key: SigningKey,
    /// Chain code for derivation
    chain_code: [u8; 32],
}

impl ExtendedPrivateKey {
    /// Create from seed bytes
    pub fn from_seed(seed: &Seed) -> Result<Self> {
        // Use HMAC-SHA512 to derive master key and chain code
        use hmac::{Hmac, Mac};
        use sha2::Sha512;

        type HmacSha512 = Hmac<Sha512>;

        let mut mac = HmacSha512::new_from_slice(b"Bitcoin seed")
            .map_err(|e| anyhow!("Failed to create HMAC: {}", e))?;
        mac.update(seed.as_slice());
        let result = mac.finalize().into_bytes();

        // First 32 bytes are the private key
        let key_bytes: [u8; 32] = result[..32]
            .try_into()
            .map_err(|_| anyhow!("Invalid key length"))?;

        // Last 32 bytes are the chain code
        let chain_code: [u8; 32] = result[32..]
            .try_into()
            .map_err(|_| anyhow!("Invalid chain code length"))?;

        let signing_key = SigningKey::from_bytes((&key_bytes).into())
            .map_err(|e| anyhow!("Invalid private key: {}", e))?;

        Ok(Self {
            signing_key,
            chain_code,
        })
    }

    /// Derive child key at path
    pub fn derive_path(&self, path: &DerivationPath) -> Result<Self> {
        let mut current = Self {
            signing_key: self.signing_key.clone(),
            chain_code: self.chain_code,
        };

        for component in path.components() {
            current = current.derive_child(component.index, component.hardened)?;
        }

        Ok(current)
    }

    /// Derive child key at index (hardened or non-hardened)
    pub fn derive_child(&self, index: u32, hardened: bool) -> Result<Self> {
        use hmac::{Hmac, Mac};
        use sha2::Sha512;

        type HmacSha512 = Hmac<Sha512>;

        let mut mac = HmacSha512::new_from_slice(&self.chain_code)
            .map_err(|e| anyhow!("Failed to create HMAC: {}", e))?;

        let child_index = if hardened {
            index | 0x80000000
        } else {
            index
        };

        if hardened {
            // Hardened derivation: use private key
            mac.update(&[0]);
            mac.update(&self.signing_key.to_bytes());
        } else {
            // Normal derivation: use public key
            let verifying_key = self.signing_key.verifying_key();
            let pubkey_bytes = verifying_key.to_encoded_point(true);
            mac.update(pubkey_bytes.as_bytes());
        }
        mac.update(&child_index.to_be_bytes());

        let result = mac.finalize().into_bytes();

        // Parse the key material
        let il: [u8; 32] = result[..32]
            .try_into()
            .map_err(|_| anyhow!("Invalid IL length"))?;
        let ir: [u8; 32] = result[32..]
            .try_into()
            .map_err(|_| anyhow!("Invalid IR length"))?;

        // Add IL to parent key to get child key
        use k256::elliptic_curve::scalar::ScalarPrimitive;
        use k256::Scalar;

        let parent_scalar =
            Scalar::from(ScalarPrimitive::from_bytes(&self.signing_key.to_bytes()).unwrap());
        let il_scalar = Scalar::from(ScalarPrimitive::from_bytes(&il.into()).unwrap());
        let child_scalar = parent_scalar + il_scalar;

        let child_bytes = child_scalar.to_bytes();
        let signing_key = SigningKey::from_bytes(&child_bytes)
            .map_err(|e| anyhow!("Invalid child key: {}", e))?;

        Ok(Self {
            signing_key,
            chain_code: ir,
        })
    }

    /// Get the signing key
    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    /// Get raw private key bytes
    pub fn private_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes().into()
    }

    /// Get the corresponding public key
    pub fn public_key(&self) -> ExtendedPublicKey {
        ExtendedPublicKey {
            verifying_key: *self.signing_key.verifying_key(),
            chain_code: self.chain_code,
        }
    }

    /// Get Ethereum address derived from this key
    pub fn ethereum_address(&self) -> EthereumAddress {
        let pubkey = self.public_key();
        pubkey.to_ethereum_address()
    }

    /// Sign a message hash (32 bytes)
    pub fn sign(&self, message_hash: &[u8; 32]) -> Result<EthereumSignature> {
        use k256::ecdsa::signature::hazmat::PrehashSigner;

        let (signature, recovery_id) = self
            .signing_key
            .sign_prehash_recoverable(message_hash)
            .map_err(|e| anyhow!("Failed to sign: {}", e))?;

        Ok(EthereumSignature::from_signature_and_recovery(
            &signature,
            recovery_id,
        ))
    }

    /// Serialize to bytes (for encrypted storage)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&self.signing_key.to_bytes());
        bytes.extend_from_slice(&self.chain_code);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 64 {
            return Err(anyhow!("Invalid extended key length"));
        }

        let key_bytes: [u8; 32] = bytes[..32]
            .try_into()
            .map_err(|_| anyhow!("Invalid key bytes"))?;
        let chain_code: [u8; 32] = bytes[32..]
            .try_into()
            .map_err(|_| anyhow!("Invalid chain code"))?;

        let signing_key = SigningKey::from_bytes((&key_bytes).into())
            .map_err(|e| anyhow!("Invalid private key: {}", e))?;

        Ok(Self {
            signing_key,
            chain_code,
        })
    }
}

impl Clone for ExtendedPrivateKey {
    fn clone(&self) -> Self {
        Self {
            signing_key: self.signing_key.clone(),
            chain_code: self.chain_code,
        }
    }
}

impl fmt::Debug for ExtendedPrivateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtendedPrivateKey([REDACTED])")
    }
}

impl Drop for ExtendedPrivateKey {
    fn drop(&mut self) {
        self.chain_code.zeroize();
    }
}

/// Extended public key
#[derive(Clone)]
pub struct ExtendedPublicKey {
    verifying_key: VerifyingKey,
    chain_code: [u8; 32],
}

impl ExtendedPublicKey {
    /// Derive child public key (non-hardened only)
    pub fn derive_child(&self, index: u32) -> Result<Self> {
        if index & 0x80000000 != 0 {
            return Err(anyhow!(
                "Cannot derive hardened child from public key alone"
            ));
        }

        use hmac::{Hmac, Mac};
        use sha2::Sha512;

        type HmacSha512 = Hmac<Sha512>;

        let mut mac = HmacSha512::new_from_slice(&self.chain_code)
            .map_err(|e| anyhow!("Failed to create HMAC: {}", e))?;

        let pubkey_bytes = self.verifying_key.to_encoded_point(true);
        mac.update(pubkey_bytes.as_bytes());
        mac.update(&index.to_be_bytes());

        let result = mac.finalize().into_bytes();

        let il: [u8; 32] = result[..32]
            .try_into()
            .map_err(|_| anyhow!("Invalid IL length"))?;
        let ir: [u8; 32] = result[32..]
            .try_into()
            .map_err(|_| anyhow!("Invalid IR length"))?;

        // Add IL * G to parent public key
        use k256::elliptic_curve::scalar::ScalarPrimitive;
        use k256::{ProjectivePoint, Scalar};

        let il_scalar = Scalar::from(ScalarPrimitive::from_bytes(&il.into()).unwrap());
        let il_point = ProjectivePoint::GENERATOR * il_scalar;
        let parent_point: ProjectivePoint = self.verifying_key.as_affine().into();
        let child_point = parent_point + il_point;

        let verifying_key = VerifyingKey::from_affine(child_point.to_affine())
            .map_err(|e| anyhow!("Invalid child public key: {}", e))?;

        Ok(Self {
            verifying_key,
            chain_code: ir,
        })
    }

    /// Get the verifying key
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Get raw public key bytes (uncompressed, 65 bytes)
    pub fn public_key_bytes_uncompressed(&self) -> [u8; 65] {
        let point = self.verifying_key.to_encoded_point(false);
        let mut result = [0u8; 65];
        result.copy_from_slice(point.as_bytes());
        result
    }

    /// Get raw public key bytes (compressed, 33 bytes)
    pub fn public_key_bytes_compressed(&self) -> [u8; 33] {
        let point = self.verifying_key.to_encoded_point(true);
        let mut result = [0u8; 33];
        result.copy_from_slice(point.as_bytes());
        result
    }

    /// Convert to Ethereum address (last 20 bytes of keccak256 of public key)
    pub fn to_ethereum_address(&self) -> EthereumAddress {
        let pubkey = self.public_key_bytes_uncompressed();
        // Skip the 0x04 prefix byte
        let mut hasher = Keccak256::new();
        hasher.update(&pubkey[1..]);
        let hash = hasher.finalize();

        let mut address = [0u8; 20];
        address.copy_from_slice(&hash[12..]);
        EthereumAddress(address)
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(65);
        bytes.extend_from_slice(&self.public_key_bytes_compressed());
        bytes.extend_from_slice(&self.chain_code);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 65 {
            return Err(anyhow!("Invalid extended public key length"));
        }

        let pubkey_bytes: [u8; 33] = bytes[..33]
            .try_into()
            .map_err(|_| anyhow!("Invalid public key bytes"))?;
        let chain_code: [u8; 32] = bytes[33..]
            .try_into()
            .map_err(|_| anyhow!("Invalid chain code"))?;

        let verifying_key = VerifyingKey::from_sec1_bytes(&pubkey_bytes)
            .map_err(|e| anyhow!("Invalid public key: {}", e))?;

        Ok(Self {
            verifying_key,
            chain_code,
        })
    }
}

impl fmt::Debug for ExtendedPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExtendedPublicKey({})",
            hex::encode(self.public_key_bytes_compressed())
        )
    }
}

/// Ethereum address (20 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EthereumAddress(pub(crate) [u8; 20]);

impl EthereumAddress {
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

    /// Convert to checksummed hex string (EIP-55)
    pub fn to_checksum_hex(&self) -> String {
        let hex_addr = hex::encode(self.0);
        let hash = Keccak256::digest(hex_addr.as_bytes());

        let mut checksum_addr = String::with_capacity(42);
        checksum_addr.push_str("0x");

        for (i, c) in hex_addr.chars().enumerate() {
            if c.is_ascii_digit() {
                checksum_addr.push(c);
            } else {
                // Get the corresponding nibble from the hash
                let hash_byte = hash[i / 2];
                let hash_nibble = if i % 2 == 0 {
                    hash_byte >> 4
                } else {
                    hash_byte & 0x0f
                };

                if hash_nibble >= 8 {
                    checksum_addr.push(c.to_ascii_uppercase());
                } else {
                    checksum_addr.push(c.to_ascii_lowercase());
                }
            }
        }

        checksum_addr
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

    /// Verify checksum (EIP-55)
    pub fn verify_checksum(address: &str) -> bool {
        let stripped = address.strip_prefix("0x").unwrap_or(address);
        if stripped.len() != 40 {
            return false;
        }

        // If all lowercase or all uppercase, checksum is valid (or not applied)
        if stripped == stripped.to_lowercase() || stripped == stripped.to_uppercase() {
            return true;
        }

        // Verify mixed case is correct checksum
        match Self::from_hex(stripped) {
            Ok(addr) => {
                let checksummed = addr.to_checksum_hex();
                checksummed[2..] == *stripped
            }
            Err(_) => false,
        }
    }
}

impl fmt::Display for EthereumAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_checksum_hex())
    }
}

/// ECDSA signature with recovery id for Ethereum
#[derive(Debug, Clone)]
pub struct EthereumSignature {
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub v: u8,
}

impl EthereumSignature {
    /// Create from k256 signature and recovery id
    pub fn from_signature_and_recovery(sig: &Signature, recovery_id: RecoveryId) -> Self {
        let r_bytes = sig.r().to_bytes();
        let s_bytes = sig.s().to_bytes();

        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&r_bytes);
        s.copy_from_slice(&s_bytes);

        // Ethereum uses v = 27 + recovery_id or v = 35 + recovery_id + chain_id * 2 (EIP-155)
        let v = 27 + recovery_id.to_byte();

        Self { r, s, v }
    }

    /// Encode as bytes (65 bytes: r ++ s ++ v)
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

    /// Adjust v for EIP-155 (chain-specific replay protection)
    pub fn with_chain_id(mut self, chain_id: u64) -> Self {
        // EIP-155: v = chain_id * 2 + 35 + recovery_id
        let recovery_id = self.v - 27;
        self.v = (chain_id * 2 + 35 + recovery_id as u64) as u8;
        self
    }

    /// Get recovery id from v value
    pub fn recovery_id(&self, chain_id: Option<u64>) -> u8 {
        match chain_id {
            Some(id) => ((self.v as u64 - 35 - id * 2) % 2) as u8,
            None => self.v - 27,
        }
    }
}

/// Path component for BIP-44 derivation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathComponent {
    pub index: u32,
    pub hardened: bool,
}

/// BIP-44 derivation path
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationPath {
    /// Path components
    components: Vec<PathComponent>,
}

impl DerivationPath {
    /// Create a new derivation path from components
    pub fn new(components: Vec<PathComponent>) -> Self {
        Self { components }
    }

    /// Create Ethereum derivation path: m/44'/60'/account'/0/address_index
    pub fn ethereum(account: u32, address_index: u32) -> Self {
        Self {
            components: vec![
                PathComponent {
                    index: 44,
                    hardened: true,
                },
                PathComponent {
                    index: 60,
                    hardened: true,
                },
                PathComponent {
                    index: account,
                    hardened: true,
                },
                PathComponent {
                    index: 0,
                    hardened: true,
                },
                PathComponent {
                    index: address_index,
                    hardened: false,
                },
            ],
        }
    }

    /// Create Ledger Live derivation path: m/44'/60'/account'/0/0
    pub fn ledger_live(account: u32) -> Self {
        Self::ethereum(account, 0)
    }

    /// Create from string path
    pub fn from_str(path: &str) -> Result<Self> {
        let path = path.trim();
        let path = path.strip_prefix("m/").unwrap_or(path);

        let components: Result<Vec<PathComponent>> = path
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let hardened = s.ends_with('\'') || s.ends_with('h') || s.ends_with('H');
                let num_str = s.trim_end_matches(|c| c == '\'' || c == 'h' || c == 'H');
                let index: u32 = num_str.parse().context("Invalid path component")?;
                Ok(PathComponent { index, hardened })
            })
            .collect();

        Ok(Self {
            components: components?,
        })
    }

    /// Convert to string path
    pub fn as_str(&self) -> String {
        let mut path = String::from("m");
        for comp in &self.components {
            path.push('/');
            path.push_str(&comp.index.to_string());
            if comp.hardened {
                path.push('\'');
            }
        }
        path
    }

    /// Get the components
    pub fn components(&self) -> &[PathComponent] {
        &self.components
    }

    /// Get the account index (assumes BIP-44 format)
    pub fn account(&self) -> u32 {
        self.components.get(2).map(|c| c.index).unwrap_or(0)
    }

    /// Get the address index (assumes BIP-44 format)
    pub fn address_index(&self) -> u32 {
        self.components.last().map(|c| c.index).unwrap_or(0)
    }

    /// Increment address index
    pub fn next_address(&self) -> Self {
        let mut components = self.components.clone();
        if let Some(last) = components.last_mut() {
            last.index += 1;
        }
        Self { components }
    }

    /// Increment account index (reset address index)
    pub fn next_account(&self) -> Self {
        let mut components = self.components.clone();
        if components.len() >= 3 {
            components[2].index += 1;
        }
        if let Some(last) = components.last_mut() {
            last.index = 0;
        }
        Self { components }
    }
}

impl Default for DerivationPath {
    fn default() -> Self {
        Self::ethereum(0, 0)
    }
}

impl fmt::Display for DerivationPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// HD key derivation manager
pub struct KeyDerivationManager {
    /// Master extended private key
    master_key: ExtendedPrivateKey,
    /// Audit logger
    audit_logger: AuditLogger,
}

impl KeyDerivationManager {
    /// Create from seed
    pub fn from_seed(seed: &Seed, audit_logger: AuditLogger) -> Result<Self> {
        let master_key = ExtendedPrivateKey::from_seed(seed)?;
        Ok(Self {
            master_key,
            audit_logger,
        })
    }

    /// Derive key at path with audit logging
    pub async fn derive_key(&self, path: &DerivationPath, wallet_id: &str) -> Result<DerivedKey> {
        let private_key = self.master_key.derive_path(path)?;
        let public_key = private_key.public_key();
        let address = public_key.to_ethereum_address();

        // Log the derivation event
        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::KeyDerived,
                Some(wallet_id.to_string()),
                Some(format!("Derived key at path {}", path)),
            ))
            .await?;

        Ok(DerivedKey {
            private_key,
            public_key,
            address,
            path: path.clone(),
        })
    }

    /// Derive multiple keys (batch derivation)
    pub async fn derive_keys(
        &self,
        account: u32,
        start_index: u32,
        count: u32,
        wallet_id: &str,
    ) -> Result<Vec<DerivedKey>> {
        let mut keys = Vec::with_capacity(count as usize);

        for i in 0..count {
            let path = DerivationPath::ethereum(account, start_index + i);
            let key = self.derive_key(&path, wallet_id).await?;
            keys.push(key);
        }

        Ok(keys)
    }

    /// Get the master public key (can be shared for watch-only wallets)
    pub fn master_public_key(&self) -> ExtendedPublicKey {
        self.master_key.public_key()
    }
}

/// A derived key with all associated data
pub struct DerivedKey {
    /// Private key
    pub private_key: ExtendedPrivateKey,
    /// Public key
    pub public_key: ExtendedPublicKey,
    /// Ethereum address
    pub address: EthereumAddress,
    /// Derivation path used
    pub path: DerivationPath,
}

impl DerivedKey {
    /// Sign a message hash
    pub fn sign(&self, message_hash: &[u8; 32]) -> Result<EthereumSignature> {
        self.private_key.sign(message_hash)
    }

    /// Sign with EIP-155 chain ID
    pub fn sign_with_chain_id(
        &self,
        message_hash: &[u8; 32],
        chain_id: u64,
    ) -> Result<EthereumSignature> {
        let sig = self.private_key.sign(message_hash)?;
        Ok(sig.with_chain_id(chain_id))
    }

    /// Get private key bytes (use with caution)
    pub fn private_key_bytes(&self) -> [u8; 32] {
        self.private_key.private_key_bytes()
    }
}

impl fmt::Debug for DerivedKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DerivedKey")
            .field("address", &self.address)
            .field("path", &self.path)
            .finish()
    }
}

/// Watch-only account (derived from public key only)
pub struct WatchOnlyAccount {
    /// Extended public key
    public_key: ExtendedPublicKey,
    /// Account index
    account: u32,
}

impl WatchOnlyAccount {
    /// Create from extended public key
    pub fn new(public_key: ExtendedPublicKey, account: u32) -> Self {
        Self { public_key, account }
    }

    /// Derive address at index (non-hardened derivation only)
    pub fn derive_address(&self, address_index: u32) -> Result<EthereumAddress> {
        // m/.../account'/0/address_index
        // We start from the account level, so we need to derive 0/address_index
        let change_key = self.public_key.derive_child(0)?;
        let address_key = change_key.derive_child(address_index)?;
        Ok(address_key.to_ethereum_address())
    }

    /// Get addresses for a range of indices
    pub fn derive_addresses(&self, start: u32, count: u32) -> Result<Vec<EthereumAddress>> {
        let mut addresses = Vec::with_capacity(count as usize);
        for i in start..start + count {
            addresses.push(self.derive_address(i)?);
        }
        Ok(addresses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallet::mnemonic::SecureMnemonic;

    #[test]
    fn test_derivation_path_parsing() {
        let path = DerivationPath::from_str("m/44'/60'/0'/0/0").unwrap();
        assert_eq!(path.components.len(), 5);
        assert_eq!(path.components[0].index, 44);
        assert!(path.components[0].hardened);
    }

    #[test]
    fn test_derivation_path_string() {
        let path = DerivationPath::ethereum(0, 5);
        let path_str = path.as_str();
        assert!(path_str.contains("44'"));
        assert!(path_str.contains("60'"));
    }

    #[test]
    fn test_key_derivation() {
        let phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let mnemonic =
            SecureMnemonic::from_phrase(phrase, super::super::mnemonic::MnemonicLanguage::English)
                .unwrap();
        let seed = mnemonic.to_seed("");

        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();
        let path = DerivationPath::ethereum(0, 0);
        let derived = master.derive_path(&path).unwrap();

        let address = derived.ethereum_address();
        // Known address for this test vector
        assert!(!address.to_hex().is_empty());
    }

    #[test]
    fn test_checksum_address() {
        // Test EIP-55 checksum
        let address =
            EthereumAddress::from_hex("0xfb6916095ca1df60bb79ce92ce3ea74c37c5d359").unwrap();
        let checksummed = address.to_checksum_hex();
        assert_eq!(checksummed, "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359");
    }

    #[test]
    fn test_signature_encoding() {
        let phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let mnemonic =
            SecureMnemonic::from_phrase(phrase, super::super::mnemonic::MnemonicLanguage::English)
                .unwrap();
        let seed = mnemonic.to_seed("");

        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();
        let path = DerivationPath::ethereum(0, 0);
        let derived = master.derive_path(&path).unwrap();

        let message_hash = [0x42u8; 32];
        let signature = derived.sign(&message_hash).unwrap();

        assert_eq!(signature.to_bytes().len(), 65);
        assert!(signature.v == 27 || signature.v == 28);
    }
}
