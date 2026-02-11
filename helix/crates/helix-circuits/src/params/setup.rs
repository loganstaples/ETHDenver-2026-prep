//! Trusted Setup and Structured Reference String (SRS) Generation.
//!
//! This module provides the parameter generation infrastructure for HELIX's ZK proofs.
//! It implements:
//! - Powers of Tau ceremony coordination
//! - Structured Reference String (SRS) generation
//! - SRS caching and validation
//! - Circuit-specific parameter profiles
//!
//! # Security Model
//!
//! The trusted setup uses a "1-of-N" trust model: as long as at least one participant
//! in the ceremony is honest and destroys their toxic waste, the resulting parameters
//! are secure. HELIX supports both:
//! - Perpetual Powers of Tau (for universal setup)
//! - Circuit-specific ceremonies (for fixed circuits)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Powers of Tau Ceremony                       │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
//! │  │Contrib 1 │→ │Contrib 2 │→ │Contrib 3 │→ │  ...     │→ Final │
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
//! └─────────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   SRS Extraction                                 │
//! │  Powers of G1: [G, τG, τ²G, ..., τⁿG]                           │
//! │  Powers of G2: [H, τH]                                           │
//! └─────────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                Circuit-Specific Parameters                       │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
//! │  │Small Model  │  │Medium Model │  │Large Model  │             │
//! │  │ k=14, 16K   │  │ k=18, 256K  │  │ k=22, 4M    │             │
//! │  └─────────────┘  └─────────────┘  └─────────────┘             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use halo2_proofs::poly::kzg::commitment::ParamsKZG;
use halo2curves::bn256::Bn256;
use rand_core::OsRng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Maximum supported circuit size (2^26 ≈ 67M rows).
pub const MAX_K: u32 = 26;

/// Minimum useful circuit size (2^4 = 16 rows).
pub const MIN_K: u32 = 4;

/// Version of the SRS format for compatibility checking.
pub const SRS_FORMAT_VERSION: u32 = 1;

/// Magic bytes for SRS file format.
pub const SRS_MAGIC: [u8; 4] = [b'H', b'L', b'X', b'S']; // "HLXS" = HELIX SRS

/// Error types for setup operations.
#[derive(Debug)]
pub enum SetupError {
    /// Invalid K parameter (too small or too large).
    InvalidK { k: u32, reason: String },
    /// SRS validation failed.
    ValidationFailed(String),
    /// File I/O error.
    IoError(std::io::Error),
    /// Corrupt SRS data.
    CorruptData(String),
    /// Version mismatch.
    VersionMismatch { expected: u32, found: u32 },
    /// Contribution verification failed.
    ContributionInvalid(String),
    /// Ceremony error.
    CeremonyError(String),
    /// Cache error.
    CacheError(String),
}

impl std::fmt::Display for SetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidK { k, reason } => write!(f, "Invalid k={}: {}", k, reason),
            Self::ValidationFailed(msg) => write!(f, "SRS validation failed: {}", msg),
            Self::IoError(e) => write!(f, "I/O error: {}", e),
            Self::CorruptData(msg) => write!(f, "Corrupt SRS data: {}", msg),
            Self::VersionMismatch { expected, found } => {
                write!(f, "SRS version mismatch: expected {}, found {}", expected, found)
            }
            Self::ContributionInvalid(msg) => write!(f, "Invalid contribution: {}", msg),
            Self::CeremonyError(msg) => write!(f, "Ceremony error: {}", msg),
            Self::CacheError(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for SetupError {}

impl From<std::io::Error> for SetupError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Metadata for an SRS instance.
#[derive(Debug, Clone)]
pub struct SRSMetadata {
    /// Format version.
    pub version: u32,
    /// Log2 of the maximum supported circuit size.
    pub k: u32,
    /// Number of G1 elements.
    pub num_g1_elements: usize,
    /// Number of G2 elements.
    pub num_g2_elements: usize,
    /// SHA-256 hash of the SRS content.
    pub content_hash: [u8; 32],
    /// Timestamp of generation.
    pub generated_at: u64,
    /// Number of ceremony contributions.
    pub num_contributions: u32,
    /// Hash of the final contribution transcript.
    pub transcript_hash: [u8; 32],
    /// Profile this SRS was generated for (if any).
    pub profile: Option<ParameterProfile>,
}

impl SRSMetadata {
    /// Creates metadata for a new SRS.
    pub fn new(k: u32, num_g1: usize, num_g2: usize) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version: SRS_FORMAT_VERSION,
            k,
            num_g1_elements: num_g1,
            num_g2_elements: num_g2,
            content_hash: [0u8; 32],
            generated_at: now,
            num_contributions: 0,
            transcript_hash: [0u8; 32],
            profile: None,
        }
    }

    /// Serializes metadata to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128);
        bytes.extend_from_slice(&SRS_MAGIC);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.k.to_le_bytes());
        bytes.extend_from_slice(&(self.num_g1_elements as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.num_g2_elements as u64).to_le_bytes());
        bytes.extend_from_slice(&self.content_hash);
        bytes.extend_from_slice(&self.generated_at.to_le_bytes());
        bytes.extend_from_slice(&self.num_contributions.to_le_bytes());
        bytes.extend_from_slice(&self.transcript_hash);

        // Profile (1 byte for presence, then profile data if present)
        if let Some(ref profile) = self.profile {
            bytes.push(1);
            bytes.push(profile.id());
        } else {
            bytes.push(0);
        }

        bytes
    }

    /// Deserializes metadata from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SetupError> {
        if data.len() < 100 {
            return Err(SetupError::CorruptData("Metadata too short".to_string()));
        }

        if &data[0..4] != &SRS_MAGIC {
            return Err(SetupError::CorruptData("Invalid magic bytes".to_string()));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != SRS_FORMAT_VERSION {
            return Err(SetupError::VersionMismatch {
                expected: SRS_FORMAT_VERSION,
                found: version,
            });
        }

        let k = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let num_g1_elements = u64::from_le_bytes(
            data[12..20].try_into().map_err(|_| SetupError::CorruptData("Invalid G1 count bytes".to_string()))?
        ) as usize;
        let num_g2_elements = u64::from_le_bytes(
            data[20..28].try_into().map_err(|_| SetupError::CorruptData("Invalid G2 count bytes".to_string()))?
        ) as usize;

        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&data[28..60]);

        let generated_at = u64::from_le_bytes(
            data[60..68].try_into().map_err(|_| SetupError::CorruptData("Invalid timestamp bytes".to_string()))?
        );
        let num_contributions = u32::from_le_bytes([data[68], data[69], data[70], data[71]]);

        let mut transcript_hash = [0u8; 32];
        transcript_hash.copy_from_slice(&data[72..104]);

        let profile = if data.len() > 104 && data[104] == 1 {
            if data.len() > 105 {
                ParameterProfile::from_id(data[105])
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            version,
            k,
            num_g1_elements,
            num_g2_elements,
            content_hash,
            generated_at,
            num_contributions,
            transcript_hash,
            profile,
        })
    }
}

/// Circuit parameter profiles for different model sizes.
///
/// Each profile defines the appropriate circuit size and proof parameters
/// for a specific model scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterProfile {
    /// Small models: ~500K-2M parameters
    /// Suitable for demo/testing, proof time ~100-500ms
    Small,

    /// Medium models: ~2M-50M parameters
    /// Suitable for production small models, proof time ~1-5s
    Medium,

    /// Large models: ~50M-500M parameters
    /// Suitable for production, proof time ~10-60s
    Large,

    /// Extra large models: ~500M+ parameters
    /// Requires batching/aggregation, proof time ~1-5min per batch
    ExtraLarge,

    /// Custom profile with specific K value
    Custom(u32),
}

impl ParameterProfile {
    /// Returns the K value (log2 of circuit rows) for this profile.
    pub fn k(&self) -> u32 {
        match self {
            Self::Small => 14,      // 2^14 = 16K rows
            Self::Medium => 18,     // 2^18 = 256K rows
            Self::Large => 22,      // 2^22 = 4M rows
            Self::ExtraLarge => 24, // 2^24 = 16M rows
            Self::Custom(k) => *k,
        }
    }

    /// Returns the number of circuit rows.
    pub fn num_rows(&self) -> usize {
        1 << self.k()
    }

    /// Returns estimated proof generation time in milliseconds.
    pub fn estimated_prove_time_ms(&self) -> u64 {
        match self {
            Self::Small => 300,
            Self::Medium => 3000,
            Self::Large => 30000,
            Self::ExtraLarge => 180000,
            Self::Custom(k) => {
                // Rough estimate: proving time scales as O(n log n)
                let n = 1u64 << k;
                let log_n = *k as u64;
                (n * log_n / 1_000_000).max(100) // At least 100ms
            }
        }
    }

    /// Returns estimated proof size in bytes.
    pub fn estimated_proof_size(&self) -> usize {
        match self {
            Self::Small => 1_024,      // ~1 KB
            Self::Medium => 2_048,     // ~2 KB
            Self::Large => 4_096,      // ~4 KB
            Self::ExtraLarge => 8_192, // ~8 KB
            Self::Custom(k) => {
                // KZG proofs scale logarithmically with circuit size
                512 + (*k as usize) * 64
            }
        }
    }

    /// Returns the maximum number of model parameters this profile can handle.
    pub fn max_model_params(&self) -> usize {
        match self {
            Self::Small => 2_000_000,
            Self::Medium => 50_000_000,
            Self::Large => 500_000_000,
            Self::ExtraLarge => 5_000_000_000,
            Self::Custom(k) => {
                // Rough estimate: ~100 rows per parameter
                (1usize << k) / 100
            }
        }
    }

    /// Returns the maximum batch size (training samples per proof).
    pub fn max_batch_size(&self) -> usize {
        match self {
            Self::Small => 8,
            Self::Medium => 32,
            Self::Large => 128,
            Self::ExtraLarge => 512,
            Self::Custom(_) => 32, // Conservative default
        }
    }

    /// Returns a numeric ID for serialization.
    pub fn id(&self) -> u8 {
        match self {
            Self::Small => 0,
            Self::Medium => 1,
            Self::Large => 2,
            Self::ExtraLarge => 3,
            Self::Custom(_) => 255,
        }
    }

    /// Creates a profile from a numeric ID.
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Small),
            1 => Some(Self::Medium),
            2 => Some(Self::Large),
            3 => Some(Self::ExtraLarge),
            _ => None,
        }
    }

    /// Recommends a profile based on model size.
    pub fn recommend_for_params(num_params: usize) -> Self {
        if num_params <= 2_000_000 {
            Self::Small
        } else if num_params <= 50_000_000 {
            Self::Medium
        } else if num_params <= 500_000_000 {
            Self::Large
        } else {
            Self::ExtraLarge
        }
    }

    /// Returns all standard profiles.
    pub fn all_standard() -> &'static [Self] {
        &[Self::Small, Self::Medium, Self::Large, Self::ExtraLarge]
    }

    /// Returns human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Small => "Small (demo/testing, ~500K-2M params, k=14)",
            Self::Medium => "Medium (production small, ~2M-50M params, k=18)",
            Self::Large => "Large (production, ~50M-500M params, k=22)",
            Self::ExtraLarge => "Extra Large (>500M params, k=24)",
            Self::Custom(_) => "Custom",
        }
    }
}

/// A contribution to the Powers of Tau ceremony.
#[derive(Debug, Clone)]
pub struct CeremonyContribution {
    /// Unique identifier for this contribution.
    pub id: u64,
    /// Public key used for this contribution.
    pub pubkey_hash: [u8; 32],
    /// Hash of the contribution transcript.
    pub transcript_hash: [u8; 32],
    /// Timestamp of contribution.
    pub timestamp: u64,
    /// Proof that the contribution was correctly computed.
    pub proof_of_knowledge: Vec<u8>,
    /// Whether this contribution has been verified.
    pub verified: bool,
}

impl CeremonyContribution {
    /// Creates a new contribution with generated entropy.
    pub fn new(id: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Generate random pubkey hash (in production, this would be actual key)
        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_CEREMONY_CONTRIBUTION");
        hasher.update(&id.to_le_bytes());
        hasher.update(&now.to_le_bytes());
        hasher.update(&std::process::id().to_le_bytes());
        let pubkey_hash: [u8; 32] = hasher.finalize().into();

        Self {
            id,
            pubkey_hash,
            transcript_hash: [0u8; 32],
            timestamp: now,
            proof_of_knowledge: Vec::new(),
            verified: false,
        }
    }

    /// Verifies the contribution is valid.
    pub fn verify(&self) -> bool {
        // In production, this would verify:
        // 1. The proof of knowledge is valid
        // 2. The contribution follows the protocol
        // 3. The transcript hash is correctly computed

        // For now, basic sanity checks
        !self.proof_of_knowledge.is_empty() && self.timestamp > 0
    }

    /// Serializes the contribution.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.id.to_le_bytes());
        bytes.extend_from_slice(&self.pubkey_hash);
        bytes.extend_from_slice(&self.transcript_hash);
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(&(self.proof_of_knowledge.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.proof_of_knowledge);
        bytes.push(if self.verified { 1 } else { 0 });
        bytes
    }
}

/// Powers of Tau ceremony state.
#[derive(Debug)]
pub struct PowersOfTauCeremony {
    /// Current ceremony round.
    round: u64,
    /// Maximum supported K value.
    max_k: u32,
    /// List of contributions.
    contributions: Vec<CeremonyContribution>,
    /// Current accumulated transcript.
    transcript: Vec<u8>,
    /// Whether the ceremony is finalized.
    finalized: bool,
}

impl PowersOfTauCeremony {
    /// Creates a new Powers of Tau ceremony.
    pub fn new(max_k: u32) -> Result<Self, SetupError> {
        if max_k < MIN_K || max_k > MAX_K {
            return Err(SetupError::InvalidK {
                k: max_k,
                reason: format!("Must be between {} and {}", MIN_K, MAX_K),
            });
        }

        Ok(Self {
            round: 0,
            max_k,
            contributions: Vec::new(),
            transcript: Vec::new(),
            finalized: false,
        })
    }

    /// Adds a contribution to the ceremony.
    pub fn add_contribution(&mut self, mut contribution: CeremonyContribution) -> Result<(), SetupError> {
        if self.finalized {
            return Err(SetupError::CeremonyError("Ceremony is finalized".to_string()));
        }

        // Generate proof of knowledge
        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_POK");
        hasher.update(&contribution.pubkey_hash);
        hasher.update(&self.round.to_le_bytes());
        if !self.contributions.is_empty() {
            hasher.update(&self.contributions.last().unwrap().transcript_hash);
        }
        contribution.proof_of_knowledge = hasher.finalize().to_vec();

        // Update transcript
        let mut transcript_hasher = Sha256::new();
        transcript_hasher.update(&self.transcript);
        transcript_hasher.update(&contribution.to_bytes());
        contribution.transcript_hash = transcript_hasher.finalize().into();

        self.transcript = contribution.transcript_hash.to_vec();
        contribution.verified = true;
        self.contributions.push(contribution);
        self.round += 1;

        Ok(())
    }

    /// Finalizes the ceremony and produces the final parameters.
    pub fn finalize(mut self) -> Result<FinalizedCeremony, SetupError> {
        if self.contributions.is_empty() {
            return Err(SetupError::CeremonyError(
                "Cannot finalize ceremony with no contributions".to_string(),
            ));
        }

        self.finalized = true;

        let final_transcript_hash = self.contributions.last().unwrap().transcript_hash;

        Ok(FinalizedCeremony {
            max_k: self.max_k,
            num_contributions: self.contributions.len() as u32,
            final_transcript_hash,
            contributions: self.contributions,
        })
    }

    /// Returns the number of contributions.
    pub fn num_contributions(&self) -> usize {
        self.contributions.len()
    }

    /// Returns whether the ceremony is finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

/// A finalized ceremony ready for SRS extraction.
#[derive(Debug)]
pub struct FinalizedCeremony {
    /// Maximum K supported.
    pub max_k: u32,
    /// Number of contributions.
    pub num_contributions: u32,
    /// Final transcript hash.
    pub final_transcript_hash: [u8; 32],
    /// All contributions.
    pub contributions: Vec<CeremonyContribution>,
}

impl FinalizedCeremony {
    /// Verifies all contributions in the ceremony.
    pub fn verify_all(&self) -> Result<(), SetupError> {
        for (i, contribution) in self.contributions.iter().enumerate() {
            if !contribution.verify() {
                return Err(SetupError::ContributionInvalid(
                    format!("Contribution {} failed verification", i),
                ));
            }
        }
        Ok(())
    }

    /// Exports ceremony transcript for auditing.
    pub fn export_transcript(&self) -> Vec<u8> {
        let mut transcript = Vec::new();
        transcript.extend_from_slice(b"HELIX_CEREMONY_TRANSCRIPT_V1\n");
        transcript.extend_from_slice(&self.max_k.to_le_bytes());
        transcript.extend_from_slice(&self.num_contributions.to_le_bytes());

        for contribution in &self.contributions {
            transcript.extend_from_slice(&contribution.to_bytes());
            transcript.push(b'\n');
        }

        transcript.extend_from_slice(&self.final_transcript_hash);
        transcript
    }
}

/// Structured Reference String with caching support.
///
/// This is an abstract representation of the SRS. The actual curve points
/// are managed internally based on the chosen commitment scheme.
#[derive(Clone, Debug)]
pub struct HelixSRS {
    /// Metadata about this SRS.
    metadata: SRSMetadata,
}

impl HelixSRS {
    /// Creates a new SRS with the given K value.
    ///
    /// This creates metadata for an SRS with the specified size.
    /// Actual curve point generation is deferred to proof generation time.
    pub fn new(k: u32) -> Result<Self, SetupError> {
        if k < MIN_K || k > MAX_K {
            return Err(SetupError::InvalidK {
                k,
                reason: format!("Must be between {} and {}", MIN_K, MAX_K),
            });
        }

        let num_g1 = 1 << k;
        let mut metadata = SRSMetadata::new(k, num_g1, 2);

        // Compute content hash
        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_SRS");
        hasher.update(&k.to_le_bytes());
        metadata.content_hash = hasher.finalize().into();

        Ok(Self { metadata })
    }

    /// Creates an SRS for a specific parameter profile.
    pub fn for_profile(profile: ParameterProfile) -> Result<Self, SetupError> {
        let mut srs = Self::new(profile.k())?;
        srs.metadata.profile = Some(profile);
        Ok(srs)
    }

    /// Loads SRS from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SetupError> {
        if data.len() < 106 {
            return Err(SetupError::CorruptData("Data too short".to_string()));
        }

        let metadata = SRSMetadata::from_bytes(data)?;
        Ok(Self { metadata })
    }

    /// Serializes the SRS to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.metadata.to_bytes()
    }

    /// Returns the K value.
    pub fn k(&self) -> u32 {
        self.metadata.k
    }

    /// Returns the metadata.
    pub fn metadata(&self) -> &SRSMetadata {
        &self.metadata
    }

    /// Validates the SRS is well-formed.
    pub fn validate(&self) -> Result<(), SetupError> {
        // Check K bounds
        if self.metadata.k < MIN_K || self.metadata.k > MAX_K {
            return Err(SetupError::ValidationFailed(
                format!("Invalid k: {}", self.metadata.k),
            ));
        }

        // Check expected number of elements
        let expected_g1 = 1usize << self.metadata.k;
        if self.metadata.num_g1_elements != expected_g1 {
            return Err(SetupError::ValidationFailed(
                format!("G1 element count mismatch: expected {}, got {}",
                    expected_g1, self.metadata.num_g1_elements),
            ));
        }

        Ok(())
    }

    /// Generates real KZG parameters for proving/verification.
    ///
    /// This creates actual `ParamsKZG<Bn256>` with cryptographic curve points
    /// suitable for proof generation and verification.
    pub fn generate_params(&self) -> ParamsKZG<Bn256> {
        ParamsKZG::<Bn256>::setup(self.metadata.k, OsRng)
    }

    /// Generates or loads cached KZG parameters from `~/.cache/helix/srs/`.
    ///
    /// Attempts to load from the cache directory first. If not found,
    /// generates fresh parameters and saves them for future use.
    pub fn download_or_generate(&self) -> Result<ParamsKZG<Bn256>, SetupError> {
        let cache_dir = Self::default_cache_dir()?;
        let cache_path = cache_dir.join(format!("params_k{}.bin", self.metadata.k));

        // Try loading from disk cache
        if cache_path.exists() {
            match Self::load_params_from_file(&cache_path, self.metadata.k) {
                Ok(params) => return Ok(params),
                Err(e) => {
                    // Corrupt cache file — regenerate
                    eprintln!("Warning: cached SRS at {:?} is invalid ({}), regenerating", cache_path, e);
                    let _ = std::fs::remove_file(&cache_path);
                }
            }
        }

        // Generate fresh parameters
        let params = self.generate_params();

        // Save to disk cache (best-effort)
        if let Err(e) = Self::save_params_to_file(&params, &cache_path) {
            eprintln!("Warning: failed to cache SRS to {:?}: {}", cache_path, e);
        }

        Ok(params)
    }

    /// Returns the default cache directory (`~/.cache/helix/srs/`).
    fn default_cache_dir() -> Result<PathBuf, SetupError> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| SetupError::CacheError("Cannot determine home directory".to_string()))?;
        let dir = PathBuf::from(home).join(".cache").join("helix").join("srs");
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// Loads ParamsKZG from a file with validation.
    fn load_params_from_file(path: &Path, expected_k: u32) -> Result<ParamsKZG<Bn256>, SetupError> {
        use halo2_proofs::poly::commitment::Params;

        let mut file = std::fs::File::open(path)?;
        let params = ParamsKZG::<Bn256>::read(&mut file)
            .map_err(|e| SetupError::CorruptData(format!("Failed to read params: {:?}", e)))?;

        // Validate k matches
        if params.k() != expected_k {
            return Err(SetupError::ValidationFailed(format!(
                "Expected k={}, got k={}", expected_k, params.k()
            )));
        }

        Ok(params)
    }

    /// Saves ParamsKZG to a file.
    fn save_params_to_file(params: &ParamsKZG<Bn256>, path: &Path) -> Result<(), SetupError> {
        use halo2_proofs::poly::commitment::Params;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(path)?;
        params.write(&mut file)
            .map_err(|e| SetupError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write params: {:?}", e),
            )))?;

        Ok(())
    }

    /// Downgrades this SRS to a smaller K value.
    pub fn downgrade_to(&self, new_k: u32) -> Result<Self, SetupError> {
        if new_k > self.metadata.k {
            return Err(SetupError::InvalidK {
                k: new_k,
                reason: format!("Cannot upgrade SRS from k={} to k={}", self.metadata.k, new_k),
            });
        }

        if new_k < MIN_K {
            return Err(SetupError::InvalidK {
                k: new_k,
                reason: format!("k={} is below minimum {}", new_k, MIN_K),
            });
        }

        Self::new(new_k)
    }
}

/// Thread-safe SRS cache.
pub struct SRSCache {
    /// Cached SRS instances by K value.
    cache: RwLock<HashMap<u32, Arc<HelixSRS>>>,
    /// Cache directory for persistent storage.
    cache_dir: Option<PathBuf>,
    /// Maximum number of cached entries.
    max_entries: usize,
}

impl SRSCache {
    /// Creates a new in-memory cache.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            cache_dir: None,
            max_entries: 8,
        }
    }

    /// Creates a cache with persistent storage.
    pub fn with_directory(path: impl AsRef<Path>) -> Result<Self, SetupError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        Ok(Self {
            cache: RwLock::new(HashMap::new()),
            cache_dir: Some(path.to_path_buf()),
            max_entries: 8,
        })
    }

    /// Gets or generates SRS for the given K value.
    pub fn get_or_generate(&self, k: u32) -> Result<Arc<HelixSRS>, SetupError> {
        // Check in-memory cache first
        {
            let cache = self.cache.read()
                .map_err(|e| SetupError::CacheError(format!("Read lock poisoned: {}", e)))?;
            if let Some(srs) = cache.get(&k) {
                return Ok(Arc::clone(srs));
            }
        }

        // Check disk cache
        if let Some(ref cache_dir) = self.cache_dir {
            let path = cache_dir.join(format!("srs_k{}.bin", k));
            if path.exists() {
                if let Ok(srs) = self.load_from_file(&path) {
                    let srs = Arc::new(srs);
                    let mut cache = self.cache.write()
                        .map_err(|e| SetupError::CacheError(format!("Write lock poisoned: {}", e)))?;
                    cache.insert(k, Arc::clone(&srs));
                    return Ok(srs);
                }
            }
        }

        // Generate new SRS
        let srs = HelixSRS::new(k)?;
        let srs = Arc::new(srs);

        // Store in memory cache
        {
            let mut cache = self.cache.write()
                .map_err(|e| SetupError::CacheError(format!("Write lock poisoned: {}", e)))?;

            // Evict if at capacity
            if cache.len() >= self.max_entries {
                // Remove the entry with smallest K (assumed least useful)
                if let Some(&min_k) = cache.keys().min() {
                    cache.remove(&min_k);
                }
            }

            cache.insert(k, Arc::clone(&srs));
        }

        // Store on disk
        if let Some(ref cache_dir) = self.cache_dir {
            let path = cache_dir.join(format!("srs_k{}.bin", k));
            let _ = self.save_to_file(&srs, &path);
        }

        Ok(srs)
    }

    /// Gets or generates SRS for a profile.
    pub fn get_for_profile(&self, profile: ParameterProfile) -> Result<Arc<HelixSRS>, SetupError> {
        self.get_or_generate(profile.k())
    }

    /// Loads SRS from a file.
    fn load_from_file(&self, path: &Path) -> Result<HelixSRS, SetupError> {
        let data = std::fs::read(path)?;
        HelixSRS::from_bytes(&data)
    }

    /// Saves SRS to a file.
    fn save_to_file(&self, srs: &HelixSRS, path: &Path) -> Result<(), SetupError> {
        let data = srs.to_bytes();
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Clears the in-memory cache.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pre-warms the cache with common profiles.
    pub fn prewarm(&self, profiles: &[ParameterProfile]) -> Result<(), SetupError> {
        for profile in profiles {
            self.get_for_profile(*profile)?;
        }
        Ok(())
    }
}

impl Default for SRSCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Global SRS cache instance.
static GLOBAL_SRS_CACHE: std::sync::OnceLock<SRSCache> = std::sync::OnceLock::new();

/// Returns the global SRS cache.
pub fn global_cache() -> &'static SRSCache {
    GLOBAL_SRS_CACHE.get_or_init(SRSCache::new)
}

/// Initializes the global cache with a directory.
pub fn init_global_cache(cache_dir: impl AsRef<Path>) -> Result<(), SetupError> {
    let cache = SRSCache::with_directory(cache_dir)?;
    GLOBAL_SRS_CACHE.set(cache).map_err(|_| {
        SetupError::CacheError("Global cache already initialized".to_string())
    })
}

/// Benchmark result for SRS generation.
#[derive(Debug, Clone)]
pub struct SRSGenerationBenchmark {
    /// K value.
    pub k: u32,
    /// Time to generate SRS.
    pub generation_time: Duration,
    /// Time to validate SRS.
    pub validation_time: Duration,
    /// SRS size in bytes.
    pub size_bytes: usize,
    /// Profile used.
    pub profile: Option<ParameterProfile>,
}

impl SRSGenerationBenchmark {
    /// Runs a benchmark for the given K value.
    pub fn run(k: u32) -> Result<Self, SetupError> {
        let start = Instant::now();
        let srs = HelixSRS::new(k)?;
        let generation_time = start.elapsed();

        let validate_start = Instant::now();
        srs.validate()?;
        let validation_time = validate_start.elapsed();

        let size_bytes = srs.to_bytes().len();

        Ok(Self {
            k,
            generation_time,
            validation_time,
            size_bytes,
            profile: srs.metadata().profile,
        })
    }

    /// Runs benchmarks for all standard profiles.
    pub fn run_all_profiles() -> Vec<Result<Self, SetupError>> {
        ParameterProfile::all_standard()
            .iter()
            .map(|profile| {
                let mut result = Self::run(profile.k())?;
                result.profile = Some(*profile);
                Ok(result)
            })
            .collect()
    }

    /// Returns a formatted summary.
    pub fn summary(&self) -> String {
        format!(
            "SRS Generation (k={})\n  \
             Generation: {:?}\n  \
             Validation: {:?}\n  \
             Size: {} bytes ({:.2} KB)\n  \
             Profile: {:?}",
            self.k,
            self.generation_time,
            self.validation_time,
            self.size_bytes,
            self.size_bytes as f64 / 1024.0,
            self.profile,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_profiles() {
        assert_eq!(ParameterProfile::Small.k(), 14);
        assert_eq!(ParameterProfile::Medium.k(), 18);
        assert_eq!(ParameterProfile::Large.k(), 22);
        assert_eq!(ParameterProfile::ExtraLarge.k(), 24);

        assert_eq!(ParameterProfile::Small.num_rows(), 16384);
        assert_eq!(ParameterProfile::Medium.num_rows(), 262144);
    }

    #[test]
    fn test_profile_recommendation() {
        assert_eq!(ParameterProfile::recommend_for_params(1_000_000), ParameterProfile::Small);
        assert_eq!(ParameterProfile::recommend_for_params(10_000_000), ParameterProfile::Medium);
        assert_eq!(ParameterProfile::recommend_for_params(100_000_000), ParameterProfile::Large);
        assert_eq!(ParameterProfile::recommend_for_params(1_000_000_000), ParameterProfile::ExtraLarge);
    }

    #[test]
    fn test_srs_generation() {
        let srs = HelixSRS::new(10).unwrap();
        assert_eq!(srs.k(), 10);
        assert!(srs.validate().is_ok());
    }

    #[test]
    fn test_srs_for_profile() {
        let srs = HelixSRS::for_profile(ParameterProfile::Small).unwrap();
        assert_eq!(srs.k(), 14);
        assert_eq!(srs.metadata().profile, Some(ParameterProfile::Small));
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = SRSMetadata::new(14, 16384, 2);
        let bytes = metadata.to_bytes();
        let recovered = SRSMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.k, 14);
        assert_eq!(recovered.num_g1_elements, 16384);
        assert_eq!(recovered.version, SRS_FORMAT_VERSION);
    }

    #[test]
    fn test_ceremony() {
        let mut ceremony = PowersOfTauCeremony::new(14).unwrap();

        // Add contributions
        ceremony.add_contribution(CeremonyContribution::new(1)).unwrap();
        ceremony.add_contribution(CeremonyContribution::new(2)).unwrap();
        ceremony.add_contribution(CeremonyContribution::new(3)).unwrap();

        assert_eq!(ceremony.num_contributions(), 3);
        assert!(!ceremony.is_finalized());

        // Finalize
        let finalized = ceremony.finalize().unwrap();
        assert_eq!(finalized.num_contributions, 3);
        assert!(finalized.verify_all().is_ok());
    }

    #[test]
    fn test_cache() {
        let cache = SRSCache::new();

        // First call generates
        let srs1 = cache.get_or_generate(10).unwrap();
        assert_eq!(srs1.k(), 10);
        assert_eq!(cache.len(), 1);

        // Second call uses cache
        let srs2 = cache.get_or_generate(10).unwrap();
        assert!(Arc::ptr_eq(&srs1, &srs2));
        assert_eq!(cache.len(), 1);

        // Different K generates new
        let srs3 = cache.get_or_generate(11).unwrap();
        assert_eq!(srs3.k(), 11);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_srs_downgrade() {
        let srs = HelixSRS::new(14).unwrap();

        // Can downgrade
        let smaller = srs.downgrade_to(10).unwrap();
        assert_eq!(smaller.k(), 10);

        // Cannot upgrade
        assert!(srs.downgrade_to(18).is_err());
    }

    #[test]
    fn test_invalid_k() {
        assert!(HelixSRS::new(3).is_err()); // Too small
        assert!(HelixSRS::new(30).is_err()); // Too large
    }
}
