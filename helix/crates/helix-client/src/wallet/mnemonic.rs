//! BIP-39 Mnemonic Generation and Recovery
//!
//! Provides production-grade mnemonic phrase generation, validation, and seed derivation
//! following the BIP-39 standard. Supports 12, 15, 18, 21, and 24-word mnemonics.

use std::fmt;

use anyhow::{anyhow, Context, Result};
use coins_bip39::{English, Mnemonic, Wordlist};
use hmac::Hmac;
use pbkdf2::pbkdf2_hmac;
use rand::rngs::OsRng;
use sha2::Sha512;
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::audit::{AuditEvent, AuditLogger, KeyOperation};

/// Number of words in a mnemonic phrase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnemonicLength {
    /// 12 words (128 bits of entropy)
    Words12,
    /// 15 words (160 bits of entropy)
    Words15,
    /// 18 words (192 bits of entropy)
    Words18,
    /// 21 words (224 bits of entropy)
    Words21,
    /// 24 words (256 bits of entropy)
    Words24,
}

impl MnemonicLength {
    /// Get the number of words
    pub fn word_count(&self) -> usize {
        match self {
            MnemonicLength::Words12 => 12,
            MnemonicLength::Words15 => 15,
            MnemonicLength::Words18 => 18,
            MnemonicLength::Words21 => 21,
            MnemonicLength::Words24 => 24,
        }
    }

    /// Get the entropy bits
    pub fn entropy_bits(&self) -> usize {
        match self {
            MnemonicLength::Words12 => 128,
            MnemonicLength::Words15 => 160,
            MnemonicLength::Words18 => 192,
            MnemonicLength::Words21 => 224,
            MnemonicLength::Words24 => 256,
        }
    }

    /// Get the entropy bytes
    pub fn entropy_bytes(&self) -> usize {
        self.entropy_bits() / 8
    }

    /// Detect length from word count
    pub fn from_word_count(count: usize) -> Result<Self> {
        match count {
            12 => Ok(MnemonicLength::Words12),
            15 => Ok(MnemonicLength::Words15),
            18 => Ok(MnemonicLength::Words18),
            21 => Ok(MnemonicLength::Words21),
            24 => Ok(MnemonicLength::Words24),
            _ => Err(anyhow!(
                "Invalid mnemonic word count: {}. Must be 12, 15, 18, 21, or 24",
                count
            )),
        }
    }
}

impl Default for MnemonicLength {
    fn default() -> Self {
        MnemonicLength::Words24 // Maximum security by default
    }
}

/// Supported mnemonic languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MnemonicLanguage {
    #[default]
    English,
    // Note: coins-bip39 primarily supports English
    // Other languages would require a different implementation
}

/// Secure seed derived from mnemonic (512 bits)
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct Seed([u8; 64]);

impl Seed {
    /// Create seed from bytes
    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }

    /// Get seed as bytes
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }

    /// Get seed as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// Convert to hex string (use with caution)
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }
}

impl fmt::Debug for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Seed([REDACTED])")
    }
}

/// BIP-39 Mnemonic phrase with secure memory handling
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SecureMnemonic {
    /// The mnemonic phrase words
    words: Vec<String>,
    /// The language
    #[zeroize(skip)]
    language: MnemonicLanguage,
}

impl SecureMnemonic {
    /// Generate a new random mnemonic phrase
    pub fn generate(length: MnemonicLength, language: MnemonicLanguage) -> Result<Self> {
        let mut rng = OsRng;

        // Use coins-bip39 to generate mnemonic
        let mnemonic = Mnemonic::<English>::new_with_count(&mut rng, length.word_count())
            .map_err(|e| anyhow!("Failed to create mnemonic: {:?}", e))?;

        let phrase = mnemonic.to_phrase();
        let words: Vec<String> = phrase.split_whitespace().map(|s| s.to_string()).collect();

        Ok(Self { words, language })
    }

    /// Generate with default options (24 words, English)
    pub fn generate_default() -> Result<Self> {
        Self::generate(MnemonicLength::Words24, MnemonicLanguage::English)
    }

    /// Parse a mnemonic phrase from string
    pub fn from_phrase(phrase: &str, language: MnemonicLanguage) -> Result<Self> {
        // Normalize the phrase
        let normalized = phrase
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        // Validate by parsing with coins-bip39
        let _mnemonic = Mnemonic::<English>::new_from_phrase(&normalized)
            .map_err(|e| anyhow!("Invalid mnemonic phrase: {:?}", e))?;

        let words: Vec<String> = normalized.split_whitespace().map(|s| s.to_string()).collect();

        // Verify word count
        MnemonicLength::from_word_count(words.len())?;

        Ok(Self { words, language })
    }

    /// Parse mnemonic with automatic language detection (currently only English)
    pub fn from_phrase_auto_detect(phrase: &str) -> Result<Self> {
        Self::from_phrase(phrase, MnemonicLanguage::English)
    }

    /// Get the mnemonic phrase as a string
    pub fn phrase(&self) -> String {
        self.words.join(" ")
    }

    /// Get individual words
    pub fn words(&self) -> &[String] {
        &self.words
    }

    /// Get word count
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get the length type
    pub fn length(&self) -> Result<MnemonicLength> {
        MnemonicLength::from_word_count(self.words.len())
    }

    /// Get the language
    pub fn language(&self) -> MnemonicLanguage {
        self.language
    }

    /// Derive seed from mnemonic with optional passphrase
    ///
    /// The passphrase provides additional protection - even if someone obtains
    /// the mnemonic words, they cannot derive keys without the passphrase.
    pub fn to_seed(&self, passphrase: &str) -> Seed {
        let phrase = self.phrase();

        // BIP-39 specifies PBKDF2 with HMAC-SHA512, 2048 iterations
        let salt = format!("mnemonic{}", passphrase);
        let mut seed = [0u8; 64];

        pbkdf2_hmac::<Sha512>(phrase.as_bytes(), salt.as_bytes(), 2048, &mut seed);

        Seed(seed)
    }

    /// Derive seed with empty passphrase
    pub fn to_seed_normalized(&self) -> Seed {
        self.to_seed("")
    }

    /// Validate that the mnemonic checksum is correct
    pub fn validate_checksum(&self) -> Result<bool> {
        let phrase = self.phrase();
        match Mnemonic::<English>::new_from_phrase(&phrase) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get the entropy bytes (use with extreme caution)
    /// Note: This reconstructs entropy from the phrase
    pub fn to_entropy(&self) -> Result<Vec<u8>> {
        // Reconstruct entropy from phrase using coins-bip39
        // The library doesn't expose entropy directly, so we derive from phrase
        let phrase = self.phrase();
        let _mnemonic = Mnemonic::<English>::new_from_phrase(&phrase)
            .map_err(|e| anyhow!("Failed to parse mnemonic: {:?}", e))?;

        // Since coins-bip39 doesn't expose entropy directly, we'll compute it
        // This is a simplified approach - for full entropy recovery, we'd need
        // to reverse the checksum calculation
        let word_count = self.word_count();
        let entropy_bits = match word_count {
            12 => 128,
            15 => 160,
            18 => 192,
            21 => 224,
            24 => 256,
            _ => return Err(anyhow!("Invalid word count")),
        };

        // Convert words to indices and then to entropy
        let wordlist = English::get_all();
        let mut bits: Vec<bool> = Vec::new();

        for word in &self.words {
            let index = wordlist.iter().position(|w| *w == word.as_str())
                .ok_or_else(|| anyhow!("Word not in wordlist: {}", word))?;

            // Each word encodes 11 bits
            for i in (0..11).rev() {
                bits.push((index >> i) & 1 == 1);
            }
        }

        // Extract entropy bits (exclude checksum)
        let entropy_bytes = entropy_bits / 8;
        let mut entropy = vec![0u8; entropy_bytes];

        for (i, byte) in entropy.iter_mut().enumerate() {
            for j in 0..8 {
                if bits[i * 8 + j] {
                    *byte |= 1 << (7 - j);
                }
            }
        }

        Ok(entropy)
    }

    /// Create mnemonic from entropy bytes
    pub fn from_entropy(entropy: &[u8], language: MnemonicLanguage) -> Result<Self> {
        use sha2::{Sha256, Digest};

        // Validate entropy length
        let word_count = match entropy.len() {
            16 => 12,
            20 => 15,
            24 => 18,
            28 => 21,
            32 => 24,
            _ => return Err(anyhow!("Invalid entropy length: {}", entropy.len())),
        };

        // Calculate checksum
        let mut hasher = Sha256::new();
        hasher.update(entropy);
        let hash = hasher.finalize();
        let checksum_bits = entropy.len() / 4; // CS = ENT / 32 in bits

        // Convert entropy to bits
        let mut bits: Vec<bool> = Vec::new();
        for byte in entropy {
            for i in (0..8).rev() {
                bits.push((byte >> i) & 1 == 1);
            }
        }

        // Add checksum bits
        for i in 0..checksum_bits {
            bits.push((hash[i / 8] >> (7 - (i % 8))) & 1 == 1);
        }

        // Convert to words
        let wordlist = English::get_all();
        let mut words = Vec::new();

        for chunk in bits.chunks(11) {
            let mut index: usize = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    index |= 1 << (10 - i);
                }
            }
            words.push(wordlist[index].to_string());
        }

        Ok(Self { words, language })
    }
}

impl fmt::Debug for SecureMnemonic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecureMnemonic({} words, [REDACTED])", self.words.len())
    }
}

/// Mnemonic manager with audit logging
pub struct MnemonicManager {
    audit_logger: AuditLogger,
}

impl MnemonicManager {
    /// Create a new mnemonic manager
    pub fn new(audit_logger: AuditLogger) -> Self {
        Self { audit_logger }
    }

    /// Generate a new mnemonic with audit logging
    pub async fn generate(
        &self,
        length: MnemonicLength,
        language: MnemonicLanguage,
        wallet_id: Option<&str>,
    ) -> Result<SecureMnemonic> {
        let mnemonic = SecureMnemonic::generate(length, language)?;

        // Log the generation event
        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::MnemonicGenerated,
                wallet_id.map(|s| s.to_string()),
                Some(format!("{} words generated", length.word_count())),
            ))
            .await?;

        Ok(mnemonic)
    }

    /// Import a mnemonic with audit logging
    pub async fn import(
        &self,
        phrase: &str,
        language: MnemonicLanguage,
        wallet_id: Option<&str>,
    ) -> Result<SecureMnemonic> {
        let mnemonic = SecureMnemonic::from_phrase(phrase, language)?;

        // Log the import event
        self.audit_logger
            .log(AuditEvent::new(
                KeyOperation::MnemonicImported,
                wallet_id.map(|s| s.to_string()),
                Some(format!("{} words imported", mnemonic.word_count())),
            ))
            .await?;

        Ok(mnemonic)
    }

    /// Validate a mnemonic phrase
    pub fn validate(&self, phrase: &str, _language: MnemonicLanguage) -> Result<bool> {
        match SecureMnemonic::from_phrase(phrase, MnemonicLanguage::English) {
            Ok(m) => m.validate_checksum(),
            Err(_) => Ok(false),
        }
    }

    /// Get word suggestions for autocomplete
    pub fn get_word_suggestions(&self, prefix: &str, _language: MnemonicLanguage) -> Vec<&'static str> {
        let wordlist = English::get_all();
        wordlist
            .iter()
            .filter(|word| word.starts_with(prefix))
            .take(10)
            .copied()
            .collect()
    }

    /// Verify a specific word is in the wordlist
    pub fn verify_word(&self, word: &str, _language: MnemonicLanguage) -> bool {
        let wordlist = English::get_all();
        wordlist.contains(&word)
    }
}

/// Secure display of mnemonic words (shows only first letter of each word)
pub fn display_mnemonic_masked(mnemonic: &SecureMnemonic) -> String {
    mnemonic
        .words()
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let first = word.chars().next().unwrap_or('?');
            format!("{}. {}***", i + 1, first)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Format mnemonic for backup display (4 words per line)
pub fn format_mnemonic_for_backup(mnemonic: &SecureMnemonic) -> String {
    let words = mnemonic.words();
    let mut lines = Vec::new();

    for chunk in words.chunks(4) {
        let line: String = chunk
            .iter()
            .enumerate()
            .map(|(i, word)| {
                let num = lines.len() * 4 + i + 1;
                format!("{:2}. {:12}", num, word)
            })
            .collect::<Vec<_>>()
            .join(" ");
        lines.push(line);
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnemonic_generation() {
        let mnemonic = SecureMnemonic::generate_default().unwrap();
        assert_eq!(mnemonic.word_count(), 24);
        assert!(mnemonic.validate_checksum().unwrap());
    }

    #[test]
    fn test_mnemonic_lengths() {
        for length in [
            MnemonicLength::Words12,
            MnemonicLength::Words15,
            MnemonicLength::Words18,
            MnemonicLength::Words21,
            MnemonicLength::Words24,
        ] {
            let mnemonic =
                SecureMnemonic::generate(length, MnemonicLanguage::English).unwrap();
            assert_eq!(mnemonic.word_count(), length.word_count());
            assert!(mnemonic.validate_checksum().unwrap());
        }
    }

    #[test]
    fn test_mnemonic_recovery() {
        // Test vector from BIP-39
        let phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let mnemonic = SecureMnemonic::from_phrase(phrase, MnemonicLanguage::English).unwrap();
        assert_eq!(mnemonic.word_count(), 12);
        assert!(mnemonic.validate_checksum().unwrap());

        // Test seed derivation
        let seed = mnemonic.to_seed("TREZOR");
        let expected_seed = "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04";
        assert_eq!(seed.to_hex(), expected_seed);
    }

    #[test]
    fn test_invalid_mnemonic() {
        let invalid = "invalid mnemonic phrase that is not valid";
        let result = SecureMnemonic::from_phrase(invalid, MnemonicLanguage::English);
        assert!(result.is_err());
    }

    #[test]
    fn test_seed_derivation_consistency() {
        let mnemonic = SecureMnemonic::generate_default().unwrap();

        // Same passphrase should produce same seed
        let seed1 = mnemonic.to_seed("test");
        let seed2 = mnemonic.to_seed("test");
        assert_eq!(seed1.as_bytes(), seed2.as_bytes());

        // Different passphrase should produce different seed
        let seed3 = mnemonic.to_seed("different");
        assert_ne!(seed1.as_bytes(), seed3.as_bytes());
    }

    #[test]
    fn test_entropy_roundtrip() {
        let mnemonic1 = SecureMnemonic::generate_default().unwrap();
        let entropy = mnemonic1.to_entropy().unwrap();

        let mnemonic2 =
            SecureMnemonic::from_entropy(&entropy, MnemonicLanguage::English).unwrap();
        assert_eq!(mnemonic1.phrase(), mnemonic2.phrase());
    }
}
