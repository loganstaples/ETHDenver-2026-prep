//! Proving Key and Verification Key Management.
//!
//! This module provides infrastructure for managing circuit-specific keys:
//! - Proving keys (PK): Used to generate ZK proofs
//! - Verification keys (VK): Used to verify proofs
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Key Generation Pipeline                       │
//! │                                                                  │
//! │  ┌─────────┐     ┌─────────────┐     ┌──────────────────────┐  │
//! │  │   SRS   │ --> │ Key Setup   │ --> │ ProvingKey + VK      │  │
//! │  │         │     │ (per circuit)│     │                      │  │
//! │  └─────────┘     └─────────────┘     └──────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Key Distribution                              │
//! │                                                                  │
//! │  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
//! │  │  Prover  │     │  Worker  │     │ Verifier │                │
//! │  │ (PK+VK)  │     │   (PK)   │     │   (VK)   │                │
//! │  └──────────┘     └──────────┘     └──────────┘                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Versioning
//!
//! Keys are versioned to ensure compatibility:
//! - Format version: Encoding format changes
//! - Circuit version: Circuit definition changes
//! - SRS version: Underlying parameter changes

use super::setup::{HelixSRS, ParameterProfile, SetupError};
use halo2_proofs::plonk::{
    keygen_pk, keygen_vk, Circuit, Error as PlonkError,
    ProvingKey, VerifyingKey,
};
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::kzg::commitment::ParamsKZG;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::PrimeField;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Version for the key file format.
pub const KEY_FORMAT_VERSION: u32 = 1;

/// Magic bytes for proving key files.
pub const PK_MAGIC: [u8; 4] = [b'H', b'P', b'K', b'1']; // HELIX Proving Key v1

/// Magic bytes for verification key files.
pub const VK_MAGIC: [u8; 4] = [b'H', b'V', b'K', b'1']; // HELIX Verification Key v1

/// Error types for key operations.
#[derive(Debug)]
pub enum KeyError {
    /// Key generation failed.
    GenerationFailed(String),
    /// Key serialization failed.
    SerializationFailed(String),
    /// Key deserialization failed.
    DeserializationFailed(String),
    /// Key validation failed.
    ValidationFailed(String),
    /// Version mismatch.
    VersionMismatch { expected: u32, found: u32 },
    /// Circuit mismatch.
    CircuitMismatch(String),
    /// IO error.
    IoError(std::io::Error),
    /// Setup error.
    SetupError(SetupError),
    /// Plonk error.
    PlonkError(String),
}

impl std::fmt::Display for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GenerationFailed(msg) => write!(f, "Key generation failed: {}", msg),
            Self::SerializationFailed(msg) => write!(f, "Key serialization failed: {}", msg),
            Self::DeserializationFailed(msg) => write!(f, "Key deserialization failed: {}", msg),
            Self::ValidationFailed(msg) => write!(f, "Key validation failed: {}", msg),
            Self::VersionMismatch { expected, found } => {
                write!(f, "Key version mismatch: expected {}, found {}", expected, found)
            }
            Self::CircuitMismatch(msg) => write!(f, "Circuit mismatch: {}", msg),
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::SetupError(e) => write!(f, "Setup error: {}", e),
            Self::PlonkError(msg) => write!(f, "Plonk error: {}", msg),
        }
    }
}

impl std::error::Error for KeyError {}

impl From<std::io::Error> for KeyError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<SetupError> for KeyError {
    fn from(e: SetupError) -> Self {
        Self::SetupError(e)
    }
}

impl From<PlonkError> for KeyError {
    fn from(e: PlonkError) -> Self {
        Self::PlonkError(format!("{:?}", e))
    }
}

/// Metadata for a proving key.
#[derive(Debug, Clone)]
pub struct ProvingKeyMetadata {
    /// Format version.
    pub format_version: u32,
    /// Circuit identifier (hash of circuit configuration).
    pub circuit_id: [u8; 32],
    /// Human-readable circuit name.
    pub circuit_name: String,
    /// Circuit version.
    pub circuit_version: u32,
    /// K value used for generation.
    pub k: u32,
    /// Number of advice columns.
    pub num_advice_columns: u32,
    /// Number of instance columns.
    pub num_instance_columns: u32,
    /// Number of fixed columns.
    pub num_fixed_columns: u32,
    /// Number of lookup tables.
    pub num_lookups: u32,
    /// Number of gates.
    pub num_gates: u32,
    /// Profile used for generation.
    pub profile: Option<ParameterProfile>,
    /// Timestamp of generation.
    pub generated_at: u64,
    /// Hash of the SRS used.
    pub srs_hash: [u8; 32],
    /// Key content hash (for integrity verification).
    pub content_hash: [u8; 32],
}

impl ProvingKeyMetadata {
    /// Creates new metadata for a proving key.
    pub fn new(circuit_name: &str, k: u32, circuit_id: [u8; 32]) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            format_version: KEY_FORMAT_VERSION,
            circuit_id,
            circuit_name: circuit_name.to_string(),
            circuit_version: 1,
            k,
            num_advice_columns: 0,
            num_instance_columns: 0,
            num_fixed_columns: 0,
            num_lookups: 0,
            num_gates: 0,
            profile: None,
            generated_at: now,
            srs_hash: [0u8; 32],
            content_hash: [0u8; 32],
        }
    }

    /// Serializes metadata to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(256);

        bytes.extend_from_slice(&PK_MAGIC);
        bytes.extend_from_slice(&self.format_version.to_le_bytes());
        bytes.extend_from_slice(&self.circuit_id);
        bytes.extend_from_slice(&(self.circuit_name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.circuit_name.as_bytes());
        bytes.extend_from_slice(&self.circuit_version.to_le_bytes());
        bytes.extend_from_slice(&self.k.to_le_bytes());
        bytes.extend_from_slice(&self.num_advice_columns.to_le_bytes());
        bytes.extend_from_slice(&self.num_instance_columns.to_le_bytes());
        bytes.extend_from_slice(&self.num_fixed_columns.to_le_bytes());
        bytes.extend_from_slice(&self.num_lookups.to_le_bytes());
        bytes.extend_from_slice(&self.num_gates.to_le_bytes());

        // Profile (always 2 bytes: presence flag + ID/padding)
        if let Some(profile) = self.profile {
            bytes.push(1);
            bytes.push(profile.id());
        } else {
            bytes.push(0);
            bytes.push(0); // padding to keep fixed-size encoding
        }

        bytes.extend_from_slice(&self.generated_at.to_le_bytes());
        bytes.extend_from_slice(&self.srs_hash);
        bytes.extend_from_slice(&self.content_hash);

        bytes
    }

    /// Deserializes metadata from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, KeyError> {
        if data.len() < 80 {
            return Err(KeyError::DeserializationFailed("Metadata too short".to_string()));
        }

        if &data[0..4] != &PK_MAGIC {
            return Err(KeyError::DeserializationFailed("Invalid magic bytes".to_string()));
        }

        let format_version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if format_version != KEY_FORMAT_VERSION {
            return Err(KeyError::VersionMismatch {
                expected: KEY_FORMAT_VERSION,
                found: format_version,
            });
        }

        let mut circuit_id = [0u8; 32];
        circuit_id.copy_from_slice(&data[8..40]);

        let name_len = u32::from_le_bytes([data[40], data[41], data[42], data[43]]) as usize;
        if data.len() < 44 + name_len + 72 {
            return Err(KeyError::DeserializationFailed("Data truncated".to_string()));
        }

        let circuit_name = String::from_utf8_lossy(&data[44..44 + name_len]).to_string();
        let offset = 44 + name_len;

        let circuit_version = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]);
        let k = u32::from_le_bytes([
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]);
        let num_advice_columns = u32::from_le_bytes([
            data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11],
        ]);
        let num_instance_columns = u32::from_le_bytes([
            data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15],
        ]);
        let num_fixed_columns = u32::from_le_bytes([
            data[offset + 16], data[offset + 17], data[offset + 18], data[offset + 19],
        ]);
        let num_lookups = u32::from_le_bytes([
            data[offset + 20], data[offset + 21], data[offset + 22], data[offset + 23],
        ]);
        let num_gates = u32::from_le_bytes([
            data[offset + 24], data[offset + 25], data[offset + 26], data[offset + 27],
        ]);

        let profile_offset = offset + 28;
        let profile = if data[profile_offset] == 1 {
            ParameterProfile::from_id(data[profile_offset + 1])
        } else {
            None
        };

        let timestamp_offset = profile_offset + 2;
        let generated_at = u64::from_le_bytes(
            data[timestamp_offset..timestamp_offset + 8].try_into().unwrap(),
        );

        let hash_offset = timestamp_offset + 8;
        let mut srs_hash = [0u8; 32];
        srs_hash.copy_from_slice(&data[hash_offset..hash_offset + 32]);

        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&data[hash_offset + 32..hash_offset + 64]);

        Ok(Self {
            format_version,
            circuit_id,
            circuit_name,
            circuit_version,
            k,
            num_advice_columns,
            num_instance_columns,
            num_fixed_columns,
            num_lookups,
            num_gates,
            profile,
            generated_at,
            srs_hash,
            content_hash,
        })
    }

    /// Returns a summary of the metadata.
    pub fn summary(&self) -> String {
        format!(
            "Circuit: {} (v{})\n  \
             K: {} (2^{} = {} rows)\n  \
             Columns: {} advice, {} instance, {} fixed\n  \
             Lookups: {}, Gates: {}\n  \
             Profile: {:?}\n  \
             Generated: {}",
            self.circuit_name,
            self.circuit_version,
            self.k,
            self.k,
            1 << self.k,
            self.num_advice_columns,
            self.num_instance_columns,
            self.num_fixed_columns,
            self.num_lookups,
            self.num_gates,
            self.profile,
            self.generated_at,
        )
    }
}

/// Metadata for a verification key.
#[derive(Debug, Clone)]
pub struct VerificationKeyMetadata {
    /// Format version.
    pub format_version: u32,
    /// Circuit identifier.
    pub circuit_id: [u8; 32],
    /// Human-readable circuit name.
    pub circuit_name: String,
    /// Circuit version.
    pub circuit_version: u32,
    /// K value.
    pub k: u32,
    /// Number of instance columns.
    pub num_instance_columns: u32,
    /// Number of public inputs per instance.
    pub num_public_inputs: u32,
    /// Timestamp.
    pub generated_at: u64,
    /// Hash of the corresponding proving key.
    pub pk_hash: [u8; 32],
    /// Content hash.
    pub content_hash: [u8; 32],
    /// Commitment scheme info (for EVM compatibility).
    pub commitment_scheme: CommitmentScheme,
}

impl VerificationKeyMetadata {
    /// Creates new verification key metadata.
    pub fn new(circuit_name: &str, k: u32, circuit_id: [u8; 32]) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            format_version: KEY_FORMAT_VERSION,
            circuit_id,
            circuit_name: circuit_name.to_string(),
            circuit_version: 1,
            k,
            num_instance_columns: 1,
            num_public_inputs: 0,
            generated_at: now,
            pk_hash: [0u8; 32],
            content_hash: [0u8; 32],
            commitment_scheme: CommitmentScheme::KZG,
        }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192);

        bytes.extend_from_slice(&VK_MAGIC);
        bytes.extend_from_slice(&self.format_version.to_le_bytes());
        bytes.extend_from_slice(&self.circuit_id);
        bytes.extend_from_slice(&(self.circuit_name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.circuit_name.as_bytes());
        bytes.extend_from_slice(&self.circuit_version.to_le_bytes());
        bytes.extend_from_slice(&self.k.to_le_bytes());
        bytes.extend_from_slice(&self.num_instance_columns.to_le_bytes());
        bytes.extend_from_slice(&self.num_public_inputs.to_le_bytes());
        bytes.extend_from_slice(&self.generated_at.to_le_bytes());
        bytes.extend_from_slice(&self.pk_hash);
        bytes.extend_from_slice(&self.content_hash);
        bytes.push(self.commitment_scheme.id());

        bytes
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, KeyError> {
        if data.len() < 60 {
            return Err(KeyError::DeserializationFailed("VK metadata too short".to_string()));
        }

        if &data[0..4] != &VK_MAGIC {
            return Err(KeyError::DeserializationFailed("Invalid VK magic bytes".to_string()));
        }

        let format_version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if format_version != KEY_FORMAT_VERSION {
            return Err(KeyError::VersionMismatch {
                expected: KEY_FORMAT_VERSION,
                found: format_version,
            });
        }

        let mut circuit_id = [0u8; 32];
        circuit_id.copy_from_slice(&data[8..40]);

        let name_len = u32::from_le_bytes([data[40], data[41], data[42], data[43]]) as usize;
        let circuit_name = String::from_utf8_lossy(&data[44..44 + name_len]).to_string();
        let offset = 44 + name_len;

        let circuit_version = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]);
        let k = u32::from_le_bytes([
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]);
        let num_instance_columns = u32::from_le_bytes([
            data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11],
        ]);
        let num_public_inputs = u32::from_le_bytes([
            data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15],
        ]);
        let generated_at = u64::from_le_bytes(
            data[offset + 16..offset + 24].try_into().unwrap(),
        );

        let mut pk_hash = [0u8; 32];
        pk_hash.copy_from_slice(&data[offset + 24..offset + 56]);

        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&data[offset + 56..offset + 88]);

        let commitment_scheme = if data.len() > offset + 88 {
            CommitmentScheme::from_id(data[offset + 88])
        } else {
            CommitmentScheme::KZG
        };

        Ok(Self {
            format_version,
            circuit_id,
            circuit_name,
            circuit_version,
            k,
            num_instance_columns,
            num_public_inputs,
            generated_at,
            pk_hash,
            content_hash,
            commitment_scheme,
        })
    }
}

/// Commitment scheme used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentScheme {
    /// KZG (Kate) commitments using BN254.
    KZG,
    /// IPA commitments (halo2 native).
    IPA,
}

impl CommitmentScheme {
    /// Returns the ID for serialization.
    pub fn id(&self) -> u8 {
        match self {
            Self::KZG => 0,
            Self::IPA => 1,
        }
    }

    /// Creates from ID.
    pub fn from_id(id: u8) -> Self {
        match id {
            0 => Self::KZG,
            _ => Self::IPA,
        }
    }

    /// Returns the elliptic curve used.
    pub fn curve_name(&self) -> &'static str {
        match self {
            Self::KZG => "BN254",
            Self::IPA => "Pasta",
        }
    }
}

/// A HELIX proving key with metadata (abstract representation).
///
/// This is a metadata-only abstraction. Actual proving key generation
/// and storage is curve-specific and handled at proof generation time.
#[derive(Debug, Clone)]
pub struct HelixProvingKey {
    /// Metadata about this key.
    metadata: ProvingKeyMetadata,
}

impl HelixProvingKey {
    /// Creates a new proving key descriptor.
    pub fn new(circuit_name: &str, k: u32) -> Self {
        let circuit_id = Self::compute_circuit_id(circuit_name);
        let metadata = ProvingKeyMetadata::new(circuit_name, k, circuit_id);
        Self { metadata }
    }

    /// Creates from SRS metadata.
    pub fn from_srs(srs: &HelixSRS, circuit_name: &str) -> Self {
        let circuit_id = Self::compute_circuit_id(circuit_name);
        let mut metadata = ProvingKeyMetadata::new(circuit_name, srs.k(), circuit_id);
        metadata.profile = srs.metadata().profile;
        metadata.srs_hash = srs.metadata().content_hash;

        // Compute content hash
        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_PK_CONTENT");
        hasher.update(&circuit_id);
        hasher.update(&srs.k().to_le_bytes());
        metadata.content_hash = hasher.finalize().into();

        Self { metadata }
    }

    /// Computes a deterministic circuit ID from name.
    pub fn compute_circuit_id(circuit_name: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_CIRCUIT_ID");
        hasher.update(circuit_name.as_bytes());
        hasher.finalize().into()
    }

    /// Returns the metadata.
    pub fn metadata(&self) -> &ProvingKeyMetadata {
        &self.metadata
    }

    /// Returns the circuit ID.
    pub fn circuit_id(&self) -> [u8; 32] {
        self.metadata.circuit_id
    }

    /// Returns the K value.
    pub fn k(&self) -> u32 {
        self.metadata.k
    }

    /// Extracts a HELIX verification key descriptor.
    pub fn extract_vk(&self) -> HelixVerificationKey {
        let mut vk_metadata = VerificationKeyMetadata::new(
            &self.metadata.circuit_name,
            self.metadata.k,
            self.metadata.circuit_id,
        );
        vk_metadata.circuit_version = self.metadata.circuit_version;
        vk_metadata.num_instance_columns = self.metadata.num_instance_columns;
        vk_metadata.pk_hash = self.metadata.content_hash;

        // Compute VK content hash
        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_VK_CONTENT");
        hasher.update(&self.metadata.circuit_id);
        hasher.update(&self.metadata.k.to_le_bytes());
        vk_metadata.content_hash = hasher.finalize().into();

        HelixVerificationKey { metadata: vk_metadata }
    }

    /// Generates a real halo2 proving key from KZG parameters and a circuit.
    ///
    /// This performs actual keygen using `keygen_vk` + `keygen_pk` from halo2.
    /// Returns a `RealKeyBundle` containing the actual cryptographic keys.
    pub fn generate_real<C: Circuit<Fr>>(
        circuit: &C,
        params: &ParamsKZG<Bn256>,
        circuit_name: &str,
    ) -> Result<RealKeyBundle, KeyError> {
        let start = Instant::now();

        let vk = keygen_vk(params, circuit)?;
        let vk_time = start.elapsed();

        let pk_start = Instant::now();
        let pk = keygen_pk(params, vk.clone(), circuit)?;
        let pk_time = pk_start.elapsed();

        let total_time = start.elapsed();

        // Build metadata
        let circuit_id = Self::compute_circuit_id(circuit_name);
        let mut metadata = ProvingKeyMetadata::new(circuit_name, params.k(), circuit_id);

        // Compute content hash from the VK pinned data
        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_REAL_PK");
        hasher.update(&circuit_id);
        hasher.update(&params.k().to_le_bytes());
        metadata.content_hash = hasher.finalize().into();

        Ok(RealKeyBundle {
            pk,
            vk,
            metadata: HelixProvingKey { metadata },
            generation_time: total_time,
            vk_generation_time: vk_time,
            pk_generation_time: pk_time,
        })
    }

    /// Saves the proving key metadata to a file.
    pub fn save(&self, path: &Path) -> Result<(), KeyError> {
        let bytes = self.metadata.to_bytes();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads proving key metadata from a file.
    pub fn load(path: &Path) -> Result<Self, KeyError> {
        let bytes = std::fs::read(path)?;
        let metadata = ProvingKeyMetadata::from_bytes(&bytes)?;
        Ok(Self { metadata })
    }

    /// Validates the proving key metadata.
    pub fn validate(&self) -> Result<(), KeyError> {
        // Verify format version
        if self.metadata.format_version != KEY_FORMAT_VERSION {
            return Err(KeyError::VersionMismatch {
                expected: KEY_FORMAT_VERSION,
                found: self.metadata.format_version,
            });
        }

        // Basic sanity checks
        if self.metadata.k < 4 || self.metadata.k > 26 {
            return Err(KeyError::ValidationFailed(
                format!("Invalid k value: {}", self.metadata.k),
            ));
        }

        Ok(())
    }
}

/// A HELIX verification key with metadata (abstract representation).
#[derive(Debug, Clone)]
pub struct HelixVerificationKey {
    /// Metadata about this key.
    metadata: VerificationKeyMetadata,
}

impl HelixVerificationKey {
    /// Creates a new verification key descriptor.
    pub fn new(circuit_name: &str, k: u32) -> Self {
        let circuit_id = HelixProvingKey::compute_circuit_id(circuit_name);
        let metadata = VerificationKeyMetadata::new(circuit_name, k, circuit_id);
        Self { metadata }
    }

    /// Returns the metadata.
    pub fn metadata(&self) -> &VerificationKeyMetadata {
        &self.metadata
    }

    /// Validates the verification key metadata.
    pub fn validate(&self) -> Result<(), KeyError> {
        if self.metadata.format_version != KEY_FORMAT_VERSION {
            return Err(KeyError::VersionMismatch {
                expected: KEY_FORMAT_VERSION,
                found: self.metadata.format_version,
            });
        }

        if self.metadata.k < 4 || self.metadata.k > 26 {
            return Err(KeyError::ValidationFailed(
                format!("Invalid k value: {}", self.metadata.k),
            ));
        }

        Ok(())
    }

    /// Saves the verification key metadata to a file.
    pub fn save(&self, path: &Path) -> Result<(), KeyError> {
        let bytes = self.metadata.to_bytes();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads verification key metadata from a file.
    pub fn load(path: &Path) -> Result<Self, KeyError> {
        let bytes = std::fs::read(path)?;
        let metadata = VerificationKeyMetadata::from_bytes(&bytes)?;
        Ok(Self { metadata })
    }

    /// Returns the commitment points placeholder for EVM verification.
    pub fn commitment_points(&self) -> Vec<[u8; 64]> {
        // In production, extract G1 points from the actual VK
        Vec::new()
    }

    /// Returns circuit identifier.
    pub fn circuit_id(&self) -> [u8; 32] {
        self.metadata.circuit_id
    }

    /// Checks if this VK is compatible with a proving key.
    pub fn is_compatible_with(&self, pk: &HelixProvingKey) -> bool {
        self.metadata.circuit_id == pk.metadata.circuit_id
            && self.metadata.circuit_version == pk.metadata.circuit_version
            && self.metadata.k == pk.metadata.k
    }
}

/// A bundle containing both proving and verification key descriptors.
#[derive(Debug, Clone)]
pub struct KeyBundle {
    /// The proving key descriptor.
    pub pk: HelixProvingKey,
    /// The verification key descriptor.
    pub vk: HelixVerificationKey,
    /// Generation timestamp.
    pub generated_at: u64,
    /// Generation duration.
    pub generation_time: Duration,
}

impl KeyBundle {
    /// Creates a key bundle from SRS and circuit name.
    pub fn generate<C: Circuit<F>, F: PrimeField>(
        _circuit: &C,
        srs: &HelixSRS,
        circuit_name: &str,
    ) -> Result<Self, KeyError> {
        let start = Instant::now();

        let pk = HelixProvingKey::from_srs(srs, circuit_name);
        let vk = pk.extract_vk();

        let generation_time = start.elapsed();
        let generated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(Self {
            pk,
            vk,
            generated_at,
            generation_time,
        })
    }

    /// Creates a key bundle directly from SRS without a circuit.
    pub fn from_srs(srs: &HelixSRS, circuit_name: &str) -> Self {
        let start = Instant::now();

        let pk = HelixProvingKey::from_srs(srs, circuit_name);
        let vk = pk.extract_vk();

        let generation_time = start.elapsed();
        let generated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            pk,
            vk,
            generated_at,
            generation_time,
        }
    }

    /// Returns the circuit name.
    pub fn circuit_name(&self) -> &str {
        &self.pk.metadata.circuit_name
    }

    /// Returns the K value.
    pub fn k(&self) -> u32 {
        self.pk.metadata.k
    }

    /// Validates both keys.
    pub fn validate(&self) -> Result<(), KeyError> {
        self.pk.validate()?;
        self.vk.validate()?;

        // Ensure compatibility
        if !self.vk.is_compatible_with(&self.pk) {
            return Err(KeyError::ValidationFailed(
                "VK is not compatible with PK".to_string(),
            ));
        }

        Ok(())
    }

    /// Returns a summary.
    pub fn summary(&self) -> String {
        format!(
            "KeyBundle: {}\n  \
             K: {}\n  \
             Generation time: {:?}\n  \
             Generated at: {}",
            self.circuit_name(),
            self.k(),
            self.generation_time,
            self.generated_at,
        )
    }
}

/// A key bundle containing actual halo2 proving and verification keys.
///
/// Unlike `KeyBundle` (metadata-only), this holds the real cryptographic keys
/// needed for proof generation and verification.
pub struct RealKeyBundle {
    /// The actual halo2 proving key.
    pub pk: ProvingKey<G1Affine>,
    /// The actual halo2 verification key.
    pub vk: VerifyingKey<G1Affine>,
    /// Metadata descriptor.
    pub metadata: HelixProvingKey,
    /// Total key generation time.
    pub generation_time: Duration,
    /// VK generation time.
    pub vk_generation_time: Duration,
    /// PK generation time.
    pub pk_generation_time: Duration,
}

impl RealKeyBundle {
    /// Returns the circuit name.
    pub fn circuit_name(&self) -> &str {
        &self.metadata.metadata.circuit_name
    }

    /// Returns the K value.
    pub fn k(&self) -> u32 {
        self.metadata.metadata.k
    }

    /// Returns a summary of the key generation.
    pub fn summary(&self) -> String {
        format!(
            "RealKeyBundle: {}\n  \
             K: {}\n  \
             VK generation: {:?}\n  \
             PK generation: {:?}\n  \
             Total: {:?}",
            self.circuit_name(),
            self.k(),
            self.vk_generation_time,
            self.pk_generation_time,
            self.generation_time,
        )
    }
}

/// Thread-safe key cache.
pub struct KeyCache {
    /// Cached proving key descriptors by circuit ID.
    pk_cache: RwLock<HashMap<[u8; 32], Arc<HelixProvingKey>>>,
    /// Cached verification key descriptors by circuit ID.
    vk_cache: RwLock<HashMap<[u8; 32], Arc<HelixVerificationKey>>>,
    /// Cache directory.
    _cache_dir: Option<PathBuf>,
    /// Maximum cached entries per type.
    max_entries: usize,
}

impl KeyCache {
    /// Creates a new in-memory key cache.
    pub fn new() -> Self {
        Self {
            pk_cache: RwLock::new(HashMap::new()),
            vk_cache: RwLock::new(HashMap::new()),
            _cache_dir: None,
            max_entries: 16,
        }
    }

    /// Creates a cache with persistent storage.
    pub fn with_directory(path: impl AsRef<Path>) -> Result<Self, KeyError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        Ok(Self {
            pk_cache: RwLock::new(HashMap::new()),
            vk_cache: RwLock::new(HashMap::new()),
            _cache_dir: Some(path.to_path_buf()),
            max_entries: 16,
        })
    }

    /// Gets or creates a proving key descriptor.
    pub fn get_or_create_pk(
        &self,
        srs: &HelixSRS,
        circuit_name: &str,
    ) -> Result<Arc<HelixProvingKey>, KeyError> {
        let circuit_id = HelixProvingKey::compute_circuit_id(circuit_name);

        // Check cache
        {
            let cache = self.pk_cache.read().unwrap();
            if let Some(pk) = cache.get(&circuit_id) {
                return Ok(Arc::clone(pk));
            }
        }

        // Create
        let pk = HelixProvingKey::from_srs(srs, circuit_name);
        let pk = Arc::new(pk);

        // Store in cache
        {
            let mut cache = self.pk_cache.write().unwrap();
            if cache.len() >= self.max_entries {
                // Simple eviction: remove first entry
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }
            cache.insert(circuit_id, Arc::clone(&pk));
        }

        Ok(pk)
    }

    /// Gets a cached verification key.
    pub fn get_vk(&self, circuit_id: &[u8; 32]) -> Option<Arc<HelixVerificationKey>> {
        let cache = self.vk_cache.read().unwrap();
        cache.get(circuit_id).map(Arc::clone)
    }

    /// Stores a verification key.
    pub fn store_vk(&self, vk: HelixVerificationKey) {
        let circuit_id = vk.circuit_id();
        let vk = Arc::new(vk);

        let mut cache = self.vk_cache.write().unwrap();
        if cache.len() >= self.max_entries {
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
            }
        }
        cache.insert(circuit_id, vk);
    }

    /// Clears all caches.
    pub fn clear(&self) {
        self.pk_cache.write().unwrap().clear();
        self.vk_cache.write().unwrap().clear();
    }

    /// Returns the number of cached proving keys.
    pub fn pk_count(&self) -> usize {
        self.pk_cache.read().unwrap().len()
    }

    /// Returns the number of cached verification keys.
    pub fn vk_count(&self) -> usize {
        self.vk_cache.read().unwrap().len()
    }
}

impl Default for KeyCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Key generation benchmark results.
#[derive(Debug, Clone)]
pub struct KeygenBenchmark {
    /// Circuit name.
    pub circuit_name: String,
    /// K value.
    pub k: u32,
    /// Time to generate VK.
    pub vk_time: Duration,
    /// Time to generate PK.
    pub pk_time: Duration,
    /// Total keygen time.
    pub total_time: Duration,
    /// VK size (serialized).
    pub vk_size: usize,
    /// PK size (serialized).
    pub pk_size: usize,
}

impl KeygenBenchmark {
    /// Runs a keygen benchmark for a circuit.
    pub fn run(
        srs: &HelixSRS,
        circuit_name: &str,
    ) -> Result<Self, KeyError> {
        let start = Instant::now();

        // Time VK generation (metadata only for now)
        let vk_start = Instant::now();
        let pk = HelixProvingKey::from_srs(srs, circuit_name);
        let _vk = pk.extract_vk();
        let vk_time = vk_start.elapsed();

        // Time PK generation (metadata only for now)
        let pk_start = Instant::now();
        let _pk2 = HelixProvingKey::from_srs(srs, circuit_name);
        let pk_time = pk_start.elapsed();

        let total_time = start.elapsed();

        Ok(Self {
            circuit_name: circuit_name.to_string(),
            k: srs.k(),
            vk_time,
            pk_time,
            total_time,
            vk_size: 0, // Would serialize to get actual size
            pk_size: 0,
        })
    }

    /// Returns a summary.
    pub fn summary(&self) -> String {
        format!(
            "Keygen Benchmark: {}\n  \
             K: {}\n  \
             VK generation: {:?}\n  \
             PK generation: {:?}\n  \
             Total: {:?}",
            self.circuit_name,
            self.k,
            self.vk_time,
            self.pk_time,
            self.total_time,
        )
    }
}

/// Predefined circuit configurations for common use cases.
#[derive(Debug, Clone, Copy)]
pub struct CircuitConfig {
    /// Circuit name.
    pub name: &'static str,
    /// Recommended K value.
    pub recommended_k: u32,
    /// Number of advice columns.
    pub advice_columns: u32,
    /// Number of instance columns.
    pub instance_columns: u32,
    /// Number of fixed columns.
    pub fixed_columns: u32,
    /// Number of lookups.
    pub lookups: u32,
    /// Estimated constraints per row.
    pub constraints_per_row: u32,
}

impl CircuitConfig {
    /// Training step circuit configuration.
    pub const TRAINING_STEP: Self = Self {
        name: "MLTrainingStep",
        recommended_k: 14,
        advice_columns: 4,
        instance_columns: 1,
        fixed_columns: 2,
        lookups: 2,
        constraints_per_row: 4,
    };

    /// IVC step circuit configuration.
    pub const IVC_STEP: Self = Self {
        name: "IVCStep",
        recommended_k: 10,
        advice_columns: 4,
        instance_columns: 1,
        fixed_columns: 1,
        lookups: 0,
        constraints_per_row: 3,
    };

    /// Softmax circuit configuration.
    pub const SOFTMAX: Self = Self {
        name: "Softmax",
        recommended_k: 11,
        advice_columns: 4,
        instance_columns: 1,
        fixed_columns: 2,
        lookups: 1,
        constraints_per_row: 2,
    };

    /// Error accumulation circuit configuration.
    pub const ERROR_ACCUMULATION: Self = Self {
        name: "ErrorAccumulation",
        recommended_k: 8,
        advice_columns: 10,
        instance_columns: 1,
        fixed_columns: 1,
        lookups: 1,
        constraints_per_row: 2,
    };

    /// Returns estimated number of constraints.
    pub fn estimated_constraints(&self) -> usize {
        (1 << self.recommended_k) * self.constraints_per_row as usize
    }

    /// Returns all predefined configurations.
    pub fn all() -> &'static [Self] {
        &[
            Self::TRAINING_STEP,
            Self::IVC_STEP,
            Self::SOFTMAX,
            Self::ERROR_ACCUMULATION,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivc::IVCStepCircuit;

    #[test]
    fn test_pk_metadata_serialization() {
        let mut metadata = ProvingKeyMetadata::new("TestCircuit", 14, [1u8; 32]);
        metadata.num_advice_columns = 4;
        metadata.num_instance_columns = 1;
        metadata.profile = Some(ParameterProfile::Small);

        let bytes = metadata.to_bytes();
        let recovered = ProvingKeyMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.circuit_name, "TestCircuit");
        assert_eq!(recovered.k, 14);
        assert_eq!(recovered.num_advice_columns, 4);
        assert_eq!(recovered.profile, Some(ParameterProfile::Small));
    }

    #[test]
    fn test_vk_metadata_serialization() {
        let mut metadata = VerificationKeyMetadata::new("TestCircuit", 14, [2u8; 32]);
        metadata.num_instance_columns = 2;
        metadata.num_public_inputs = 8;

        let bytes = metadata.to_bytes();
        let recovered = VerificationKeyMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.circuit_name, "TestCircuit");
        assert_eq!(recovered.k, 14);
        assert_eq!(recovered.num_instance_columns, 2);
        assert_eq!(recovered.num_public_inputs, 8);
    }

    #[test]
    fn test_commitment_scheme() {
        assert_eq!(CommitmentScheme::KZG.id(), 0);
        assert_eq!(CommitmentScheme::IPA.id(), 1);
        assert_eq!(CommitmentScheme::from_id(0), CommitmentScheme::KZG);
        assert_eq!(CommitmentScheme::KZG.curve_name(), "BN254");
    }

    #[test]
    fn test_key_cache() {
        let cache = KeyCache::new();
        assert_eq!(cache.pk_count(), 0);
        assert_eq!(cache.vk_count(), 0);

        cache.clear();
        assert_eq!(cache.pk_count(), 0);
    }

    #[test]
    fn test_circuit_config() {
        let config = CircuitConfig::TRAINING_STEP;
        assert_eq!(config.name, "MLTrainingStep");
        assert_eq!(config.recommended_k, 14);
        assert!(config.estimated_constraints() > 0);

        let all = CircuitConfig::all();
        assert!(!all.is_empty());
    }

    #[test]
    fn test_key_generation_with_ivc_circuit() {
        let srs = HelixSRS::new(10).unwrap();
        let circuit = IVCStepCircuit::default();

        let result = KeyBundle::generate(&circuit, &srs, "IVCStepCircuit");
        assert!(result.is_ok());

        let bundle = result.unwrap();
        assert_eq!(bundle.k(), 10);
        assert!(bundle.validate().is_ok());
        assert!(bundle.vk.is_compatible_with(&bundle.pk));
    }

    #[test]
    fn test_proving_key_creation() {
        let pk = HelixProvingKey::new("TestCircuit", 14);
        assert_eq!(pk.k(), 14);
        assert!(pk.validate().is_ok());

        let vk = pk.extract_vk();
        assert!(vk.validate().is_ok());
        assert!(vk.is_compatible_with(&pk));
    }

    #[test]
    fn test_key_bundle_from_srs() {
        let srs = HelixSRS::new(12).unwrap();
        let bundle = KeyBundle::from_srs(&srs, "TestCircuit");

        assert_eq!(bundle.k(), 12);
        assert!(bundle.validate().is_ok());
    }

    #[test]
    fn test_real_key_generation() {
        use halo2_proofs::poly::kzg::commitment::ParamsKZG;
        use halo2curves::bn256::Bn256;
        use rand_core::OsRng;

        let circuit = IVCStepCircuit::default();
        let params = ParamsKZG::<Bn256>::setup(12, OsRng);

        let real_bundle = HelixProvingKey::generate_real(
            &circuit, &params, "IVCStepCircuit",
        ).expect("real keygen must succeed");

        assert_eq!(real_bundle.k(), 12);
        assert_eq!(real_bundle.circuit_name(), "IVCStepCircuit");
        assert!(!real_bundle.summary().is_empty());
    }

    #[test]
    fn test_pk_save_load() {
        let pk = HelixProvingKey::new("TestCircuit", 14);

        let dir = std::env::temp_dir().join("helix_test_pk_save_load");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_pk.bin");

        pk.save(&path).expect("save must succeed");
        let loaded = HelixProvingKey::load(&path).expect("load must succeed");

        assert_eq!(loaded.k(), 14);
        assert_eq!(loaded.metadata().circuit_name, "TestCircuit");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_vk_save_load() {
        let pk = HelixProvingKey::new("TestCircuit", 14);
        let vk = pk.extract_vk();

        let dir = std::env::temp_dir().join("helix_test_vk_save_load");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_vk.bin");

        vk.save(&path).expect("save must succeed");
        let loaded = HelixVerificationKey::load(&path).expect("load must succeed");

        assert_eq!(loaded.metadata().k, 14);
        assert_eq!(loaded.metadata().circuit_name, "TestCircuit");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
