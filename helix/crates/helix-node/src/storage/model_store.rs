//! Model Persistence Backend.
//!
//! Provides multi-backend storage for trained model weights with manifest
//! tracking, integrity verification, and pluggable storage backends.
//!
//! Supported backends:
//! - **Local filesystem** (always available) — atomic writes with SHA-256 checksums
//! - **IPFS** (behind `ipfs-fetch` feature) — content-addressed storage via Kubo HTTP API
//! - **S3** (behind `s3-fetch` feature) — object storage with pre-signed URL retrieval
//!
//! The `ModelStore` facade manages backend selection, manifest persistence,
//! and integrity verification. Each stored model gets a `ModelManifestEntry`
//! recording its location, commitment hash, training metrics, and provenance.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ============================================================================
// Error Types
// ============================================================================

/// Errors from model store operations.
#[derive(Debug)]
pub enum ModelStoreError {
    /// Filesystem or network I/O error.
    Io(io::Error),
    /// Serialization or deserialization error.
    Serialization(String),
    /// Requested backend is not available (feature not enabled or misconfigured).
    BackendUnavailable(String),
    /// Model or round not found in any backend.
    NotFound {
        model_id: u64,
        round_id: u64,
    },
    /// Data integrity verification failed (commitment mismatch).
    IntegrityError {
        expected: String,
        actual: String,
    },
}

impl fmt::Display for ModelStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Model store IO error: {}", e),
            Self::Serialization(msg) => write!(f, "Model store serialization error: {}", msg),
            Self::BackendUnavailable(name) => {
                write!(f, "Storage backend '{}' is not available", name)
            }
            Self::NotFound { model_id, round_id } => {
                write!(
                    f,
                    "Model not found: model_id={}, round_id={}",
                    model_id, round_id
                )
            }
            Self::IntegrityError { expected, actual } => {
                write!(
                    f,
                    "Integrity check failed: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for ModelStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ModelStoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================================
// Manifest
// ============================================================================

/// Metadata about a single stored model checkpoint.
///
/// Each entry records where the model weights are stored, their integrity
/// commitment, and training metrics at the time of storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifestEntry {
    /// On-chain model identifier.
    pub model_id: u64,
    /// Training round that produced this checkpoint.
    pub round_id: u64,
    /// Backend that holds the data: `"local"`, `"ipfs"`, or `"s3"`.
    pub storage_backend: String,
    /// Backend-specific location: file path, IPFS CID, or S3 URL.
    pub storage_location: String,
    /// SHA-256 hash of the raw weight bytes (32 bytes).
    pub commitment: [u8; 32],
    /// Size of the raw weight data in bytes.
    pub size_bytes: usize,
    /// Number of workers that contributed gradients to this round.
    pub num_contributors: u32,
    /// Final loss value at the end of the round.
    pub loss: f64,
    /// Accumulated error bound from quantized/approximate computation.
    pub error_bound: f64,
    /// Total training steps completed through this round.
    pub steps_completed: u64,
    /// Unix timestamp (seconds) when this checkpoint was created.
    pub completed_at: u64,
    /// On-chain transaction hash if the round was finalized on-chain.
    pub tx_hash: Option<String>,
}

/// Manages the on-disk manifest of all stored model checkpoints.
///
/// The manifest is a JSON file that records every model checkpoint stored
/// through the `ModelStore`. It supports queries by model ID, round ID,
/// and latest-per-model lookups.
pub struct ModelManifest {
    /// All manifest entries, in insertion order.
    entries: Vec<ModelManifestEntry>,
    /// Path to the JSON manifest file on disk.
    manifest_path: PathBuf,
}

impl ModelManifest {
    /// Creates a new manifest backed by the given file path.
    ///
    /// If the file already exists, call `load()` after construction to
    /// populate entries from disk.
    pub fn new(manifest_path: impl Into<PathBuf>) -> Self {
        Self {
            entries: Vec::new(),
            manifest_path: manifest_path.into(),
        }
    }

    /// Adds an entry to the manifest. Does not persist to disk automatically;
    /// call `save()` afterwards.
    pub fn add_entry(&mut self, entry: ModelManifestEntry) {
        self.entries.push(entry);
    }

    /// Returns the most recent entry for the given model ID (by `round_id`),
    /// or `None` if no entries exist for that model.
    pub fn get_latest(&self, model_id: u64) -> Option<&ModelManifestEntry> {
        self.entries
            .iter()
            .filter(|e| e.model_id == model_id)
            .max_by_key(|e| e.round_id)
    }

    /// Returns the entry for a specific (model_id, round_id) pair,
    /// or `None` if not found.
    pub fn get_entry(&self, model_id: u64, round_id: u64) -> Option<&ModelManifestEntry> {
        self.entries
            .iter()
            .find(|e| e.model_id == model_id && e.round_id == round_id)
    }

    /// Returns all entries for the given model ID, sorted by round_id ascending.
    pub fn get_all(&self, model_id: u64) -> Vec<&ModelManifestEntry> {
        let mut matched: Vec<&ModelManifestEntry> = self
            .entries
            .iter()
            .filter(|e| e.model_id == model_id)
            .collect();
        matched.sort_by_key(|e| e.round_id);
        matched
    }

    /// Persists the manifest to disk as JSON.
    ///
    /// Uses atomic write (write to temp file, then rename) to prevent
    /// corruption from partial writes or crashes.
    pub fn save(&self) -> Result<(), ModelStoreError> {
        let json = serde_json::to_vec_pretty(&self.entries)
            .map_err(|e| ModelStoreError::Serialization(e.to_string()))?;

        // Ensure parent directory exists
        if let Some(parent) = self.manifest_path.parent() {
            fs::create_dir_all(parent)?;
        }

        atomic_write(&self.manifest_path, &json)
    }

    /// Loads manifest entries from disk, replacing any in-memory entries.
    ///
    /// If the manifest file does not exist, entries are cleared (fresh start).
    pub fn load(&mut self) -> Result<(), ModelStoreError> {
        match fs::read(&self.manifest_path) {
            Ok(data) => {
                self.entries = serde_json::from_slice(&data)
                    .map_err(|e| ModelStoreError::Serialization(e.to_string()))?;
                Ok(())
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                self.entries.clear();
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }
}

// ============================================================================
// Backend Trait
// ============================================================================

/// Trait for pluggable model storage backends.
///
/// All operations are synchronous. Backends that wrap async APIs (IPFS, S3)
/// should use `tokio::task::block_in_place` or equivalent to bridge.
pub trait ModelStoreBackend: Send + Sync {
    /// Stores model weight data and returns a backend-specific location string.
    ///
    /// For local storage, this is a file path.
    /// For IPFS, this is a CID.
    /// For S3, this is the object URL.
    fn store(&self, model_id: u64, round_id: u64, data: &[u8]) -> Result<String, ModelStoreError>;

    /// Loads model weight data from the given backend-specific location.
    fn load(&self, location: &str) -> Result<Vec<u8>, ModelStoreError>;

    /// Checks whether data exists at the given location.
    fn exists(&self, location: &str) -> bool;

    /// Returns the backend name: `"local"`, `"ipfs"`, or `"s3"`.
    fn backend_name(&self) -> &str;
}

// ============================================================================
// Local Backend
// ============================================================================

/// Local filesystem model storage backend.
///
/// Stores model weights in a structured directory layout:
/// ```text
/// {data_dir}/models/model_{id}/round_{round_id}.bin
/// {data_dir}/models/model_{id}/round_{round_id}.sha256
/// ```
///
/// Writes are atomic (write to temp file with unique name, then rename).
/// A SHA-256 checksum sidecar file is written alongside each data file.
pub struct LocalModelStore {
    /// Base directory for model storage.
    data_dir: PathBuf,
}

impl LocalModelStore {
    /// Creates a new local model store rooted at `data_dir`.
    ///
    /// Creates the `{data_dir}/models/` directory if it does not exist.
    pub fn new(data_dir: impl Into<PathBuf>) -> Result<Self, ModelStoreError> {
        let data_dir = data_dir.into();
        fs::create_dir_all(data_dir.join("models"))?;
        Ok(Self { data_dir })
    }

    /// Returns the file path for a model's round data.
    fn model_path(&self, model_id: u64, round_id: u64) -> PathBuf {
        self.data_dir
            .join("models")
            .join(format!("model_{}", model_id))
            .join(format!("round_{}.bin", round_id))
    }

    /// Returns the path for the SHA-256 checksum sidecar file.
    fn checksum_path(data_path: &Path) -> PathBuf {
        data_path.with_extension("sha256")
    }

    /// Verifies the integrity of a stored file against its checksum sidecar.
    pub fn verify_integrity(&self, data_path: &Path) -> Result<(), ModelStoreError> {
        let data = fs::read(data_path)?;
        let checksum_path = Self::checksum_path(data_path);

        match fs::read_to_string(&checksum_path) {
            Ok(expected) => {
                let actual = hex::encode(Sha256::digest(&data));
                if expected.trim() == actual {
                    Ok(())
                } else {
                    Err(ModelStoreError::IntegrityError {
                        expected: expected.trim().to_string(),
                        actual,
                    })
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // No checksum file — cannot verify, but do not fail
                log::warn!(
                    "No checksum sidecar for {}; skipping integrity check",
                    data_path.display()
                );
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }
}

impl ModelStoreBackend for LocalModelStore {
    fn store(&self, model_id: u64, round_id: u64, data: &[u8]) -> Result<String, ModelStoreError> {
        let path = self.model_path(model_id, round_id);

        // Ensure the model directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Atomic write: temp file with unique name, then rename
        atomic_write(&path, data)?;

        // Write SHA-256 checksum sidecar
        let checksum = hex::encode(Sha256::digest(data));
        let checksum_path = Self::checksum_path(&path);
        fs::write(&checksum_path, checksum.as_bytes())?;

        let location = path.to_string_lossy().to_string();
        log::info!(
            "Stored model_{}/round_{} locally ({} bytes): {}",
            model_id,
            round_id,
            data.len(),
            location
        );

        Ok(location)
    }

    fn load(&self, location: &str) -> Result<Vec<u8>, ModelStoreError> {
        let path = Path::new(location);
        match fs::read(path) {
            Ok(data) => {
                // Verify checksum if available
                self.verify_integrity(path)?;
                Ok(data)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                Err(ModelStoreError::Io(e))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn exists(&self, location: &str) -> bool {
        Path::new(location).exists()
    }

    fn backend_name(&self) -> &str {
        "local"
    }
}

// ============================================================================
// IPFS Backend
// ============================================================================

/// IPFS model storage backend.
///
/// Stores model weights on IPFS via the Kubo HTTP API and returns the CID
/// as the storage location. Falls back to a local cache directory for
/// faster repeated loads.
#[cfg(feature = "ipfs-fetch")]
pub struct IpfsModelStore {
    /// IPFS HTTP API endpoint (e.g. `http://localhost:5001`).
    api_url: String,
    /// HTTP client for IPFS API calls.
    client: reqwest::Client,
    /// Local cache directory for downloaded models.
    cache_dir: PathBuf,
    /// Request timeout.
    timeout: std::time::Duration,
}

#[cfg(feature = "ipfs-fetch")]
impl IpfsModelStore {
    /// Creates a new IPFS model store.
    ///
    /// `api_url` should point to a Kubo HTTP API (e.g. `http://localhost:5001`).
    /// `cache_dir` is used for local caching of downloaded model data.
    pub fn new(
        api_url: impl Into<String>,
        cache_dir: impl Into<PathBuf>,
    ) -> Result<Self, ModelStoreError> {
        let cache_dir = cache_dir.into();
        fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            api_url: api_url.into(),
            client: reqwest::Client::new(),
            cache_dir,
            timeout: std::time::Duration::from_secs(60),
        })
    }

    /// Returns the local cache path for a given CID.
    fn cache_path(&self, cid: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.bin", cid))
    }

    /// Stores data on IPFS via `POST /api/v0/add` and pins it.
    /// Returns the CID.
    fn ipfs_store(&self, data: &[u8]) -> Result<String, ModelStoreError> {
        // Use tokio::task::block_in_place to bridge async IPFS into sync context
        let rt = tokio::runtime::Handle::try_current().map_err(|_| {
            ModelStoreError::BackendUnavailable(
                "IPFS backend requires a running tokio runtime".into(),
            )
        })?;

        let api_url = self.api_url.clone();
        let client = self.client.clone();
        let timeout = self.timeout;
        let data = data.to_vec();

        tokio::task::block_in_place(|| {
            rt.block_on(async {
                let url = format!("{}/api/v0/add", api_url);

                let part = reqwest::multipart::Part::bytes(data).file_name("model.bin");
                let form = reqwest::multipart::Form::new().part("file", part);

                let response = client
                    .post(&url)
                    .multipart(form)
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|e| ModelStoreError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    return Err(ModelStoreError::BackendUnavailable(format!(
                        "IPFS add failed ({}): {}",
                        status, body
                    )));
                }

                #[derive(serde::Deserialize)]
                struct AddResponse {
                    #[serde(rename = "Hash")]
                    hash: String,
                }

                let add_resp: AddResponse = response
                    .json()
                    .await
                    .map_err(|e| ModelStoreError::Serialization(e.to_string()))?;

                // Pin the CID
                let pin_url = format!("{}/api/v0/pin/add?arg={}", api_url, add_resp.hash);
                let _ = client.post(&pin_url).timeout(timeout).send().await;

                Ok(add_resp.hash)
            })
        })
    }

    /// Fetches data from IPFS via `POST /api/v0/cat?arg=<cid>`.
    fn ipfs_fetch(&self, cid: &str) -> Result<Vec<u8>, ModelStoreError> {
        let rt = tokio::runtime::Handle::try_current().map_err(|_| {
            ModelStoreError::BackendUnavailable(
                "IPFS backend requires a running tokio runtime".into(),
            )
        })?;

        let api_url = self.api_url.clone();
        let client = self.client.clone();
        let timeout = self.timeout;
        let cid = cid.to_string();

        tokio::task::block_in_place(|| {
            rt.block_on(async {
                let url = format!("{}/api/v0/cat?arg={}", api_url, cid);

                let response = client
                    .post(&url)
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|e| ModelStoreError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    return Err(ModelStoreError::Io(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("IPFS cat failed ({}): {}", status, body),
                    )));
                }

                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| ModelStoreError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;

                Ok(bytes.to_vec())
            })
        })
    }
}

#[cfg(feature = "ipfs-fetch")]
impl ModelStoreBackend for IpfsModelStore {
    fn store(&self, model_id: u64, round_id: u64, data: &[u8]) -> Result<String, ModelStoreError> {
        let cid = self.ipfs_store(data)?;

        // Write to local cache for fast subsequent loads
        let cache_path = self.cache_path(&cid);
        if let Err(e) = fs::write(&cache_path, data) {
            log::warn!("Failed to write IPFS cache for CID {}: {}", cid, e);
        }

        log::info!(
            "Stored model_{}/round_{} on IPFS ({} bytes): {}",
            model_id,
            round_id,
            data.len(),
            cid
        );

        Ok(cid)
    }

    fn load(&self, location: &str) -> Result<Vec<u8>, ModelStoreError> {
        // Check local cache first
        let cache_path = self.cache_path(location);
        if cache_path.exists() {
            log::debug!("IPFS cache hit for CID {}", location);
            return fs::read(&cache_path).map_err(ModelStoreError::from);
        }

        // Fetch from IPFS
        let data = self.ipfs_fetch(location)?;

        // Populate cache
        if let Err(e) = fs::write(&cache_path, &data) {
            log::warn!("Failed to populate IPFS cache for CID {}: {}", location, e);
        }

        Ok(data)
    }

    fn exists(&self, location: &str) -> bool {
        // Check local cache; do not make a network call for existence check
        self.cache_path(location).exists()
    }

    fn backend_name(&self) -> &str {
        "ipfs"
    }
}

/// Stub IPFS backend returned when the `ipfs-fetch` feature is not enabled.
#[cfg(not(feature = "ipfs-fetch"))]
pub struct IpfsModelStore;

#[cfg(not(feature = "ipfs-fetch"))]
impl IpfsModelStore {
    /// Returns an error indicating the IPFS feature is not enabled.
    pub fn new(
        _api_url: impl Into<String>,
        _cache_dir: impl Into<PathBuf>,
    ) -> Result<Self, ModelStoreError> {
        Err(ModelStoreError::BackendUnavailable(
            "IPFS storage requires the 'ipfs-fetch' feature to be enabled".into(),
        ))
    }
}

#[cfg(not(feature = "ipfs-fetch"))]
impl ModelStoreBackend for IpfsModelStore {
    fn store(&self, _model_id: u64, _round_id: u64, _data: &[u8]) -> Result<String, ModelStoreError> {
        Err(ModelStoreError::BackendUnavailable(
            "IPFS storage requires the 'ipfs-fetch' feature".into(),
        ))
    }

    fn load(&self, _location: &str) -> Result<Vec<u8>, ModelStoreError> {
        Err(ModelStoreError::BackendUnavailable(
            "IPFS storage requires the 'ipfs-fetch' feature".into(),
        ))
    }

    fn exists(&self, _location: &str) -> bool {
        false
    }

    fn backend_name(&self) -> &str {
        "ipfs"
    }
}

// ============================================================================
// S3 Backend
// ============================================================================

/// S3-compatible object storage backend for model weights.
///
/// Stores models at `s3://{bucket}/helix-models/model_{id}/round_{round_id}.bin`.
/// Uses pre-signed URLs for retrieval when configured.
#[cfg(feature = "s3-fetch")]
pub struct S3ModelStore {
    /// S3 bucket name.
    bucket: String,
    /// AWS region (e.g. `us-east-1`).
    region: String,
    /// HTTP client for S3 API calls.
    client: reqwest::Client,
    /// Local cache directory for downloaded models.
    cache_dir: PathBuf,
    /// Request timeout.
    timeout: std::time::Duration,
}

#[cfg(feature = "s3-fetch")]
impl S3ModelStore {
    /// Creates a new S3 model store.
    pub fn new(
        bucket: impl Into<String>,
        region: impl Into<String>,
        cache_dir: impl Into<PathBuf>,
    ) -> Result<Self, ModelStoreError> {
        let cache_dir = cache_dir.into();
        fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            bucket: bucket.into(),
            region: region.into(),
            client: reqwest::Client::new(),
            cache_dir,
            timeout: std::time::Duration::from_secs(120),
        })
    }

    /// Returns the S3 object key for a model checkpoint.
    fn object_key(model_id: u64, round_id: u64) -> String {
        format!("helix-models/model_{}/round_{}.bin", model_id, round_id)
    }

    /// Returns the S3 URL for a stored object.
    fn object_url(&self, key: &str) -> String {
        format!(
            "https://{}.s3.{}.amazonaws.com/{}",
            self.bucket, self.region, key
        )
    }

    /// Returns the local cache path for an S3 object.
    fn cache_path(&self, key: &str) -> PathBuf {
        // Flatten the key to avoid nested directories in cache
        let flat_key = key.replace('/', "_");
        self.cache_dir.join(flat_key)
    }

    /// Uploads data to S3 via PUT request.
    fn s3_put(&self, key: &str, data: &[u8]) -> Result<String, ModelStoreError> {
        let rt = tokio::runtime::Handle::try_current().map_err(|_| {
            ModelStoreError::BackendUnavailable(
                "S3 backend requires a running tokio runtime".into(),
            )
        })?;

        let url = self.object_url(key);
        let client = self.client.clone();
        let timeout = self.timeout;
        let data = data.to_vec();

        tokio::task::block_in_place(|| {
            rt.block_on(async {
                let response = client
                    .put(&url)
                    .body(data)
                    .header("Content-Type", "application/octet-stream")
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|e| {
                        ModelStoreError::Io(io::Error::new(io::ErrorKind::Other, e.to_string()))
                    })?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    return Err(ModelStoreError::BackendUnavailable(format!(
                        "S3 PUT failed ({}): {}",
                        status, body
                    )));
                }

                Ok(url)
            })
        })
    }

    /// Downloads data from S3 via GET request.
    fn s3_get(&self, url: &str) -> Result<Vec<u8>, ModelStoreError> {
        let rt = tokio::runtime::Handle::try_current().map_err(|_| {
            ModelStoreError::BackendUnavailable(
                "S3 backend requires a running tokio runtime".into(),
            )
        })?;

        let client = self.client.clone();
        let timeout = self.timeout;
        let url = url.to_string();

        tokio::task::block_in_place(|| {
            rt.block_on(async {
                let response = client
                    .get(&url)
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|e| {
                        ModelStoreError::Io(io::Error::new(io::ErrorKind::Other, e.to_string()))
                    })?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    return Err(ModelStoreError::Io(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("S3 GET failed ({}): {}", status, body),
                    )));
                }

                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| {
                        ModelStoreError::Io(io::Error::new(io::ErrorKind::Other, e.to_string()))
                    })?;

                Ok(bytes.to_vec())
            })
        })
    }
}

#[cfg(feature = "s3-fetch")]
impl ModelStoreBackend for S3ModelStore {
    fn store(&self, model_id: u64, round_id: u64, data: &[u8]) -> Result<String, ModelStoreError> {
        let key = Self::object_key(model_id, round_id);
        let url = self.s3_put(&key, data)?;

        // Populate local cache
        let cache_path = self.cache_path(&key);
        if let Err(e) = fs::write(&cache_path, data) {
            log::warn!("Failed to write S3 cache for {}: {}", key, e);
        }

        log::info!(
            "Stored model_{}/round_{} on S3 ({} bytes): {}",
            model_id,
            round_id,
            data.len(),
            url
        );

        Ok(url)
    }

    fn load(&self, location: &str) -> Result<Vec<u8>, ModelStoreError> {
        // Try to extract the key from the URL for cache lookup
        let cache_key = if let Some(path) = location.split(".amazonaws.com/").nth(1) {
            path.to_string()
        } else {
            location.replace('/', "_")
        };

        let cache_path = self.cache_path(&cache_key);
        if cache_path.exists() {
            log::debug!("S3 cache hit for {}", location);
            return fs::read(&cache_path).map_err(ModelStoreError::from);
        }

        // Fetch from S3
        let data = self.s3_get(location)?;

        // Populate cache
        if let Err(e) = fs::write(&cache_path, &data) {
            log::warn!("Failed to populate S3 cache for {}: {}", location, e);
        }

        Ok(data)
    }

    fn exists(&self, location: &str) -> bool {
        // Check local cache only; avoid network call for existence check
        let cache_key = if let Some(path) = location.split(".amazonaws.com/").nth(1) {
            path.to_string()
        } else {
            location.replace('/', "_")
        };
        self.cache_path(&cache_key).exists()
    }

    fn backend_name(&self) -> &str {
        "s3"
    }
}

/// Stub S3 backend returned when the `s3-fetch` feature is not enabled.
#[cfg(not(feature = "s3-fetch"))]
pub struct S3ModelStore;

#[cfg(not(feature = "s3-fetch"))]
impl S3ModelStore {
    /// Returns an error indicating the S3 feature is not enabled.
    pub fn new(
        _bucket: impl Into<String>,
        _region: impl Into<String>,
        _cache_dir: impl Into<PathBuf>,
    ) -> Result<Self, ModelStoreError> {
        Err(ModelStoreError::BackendUnavailable(
            "S3 storage requires the 's3-fetch' feature to be enabled".into(),
        ))
    }
}

#[cfg(not(feature = "s3-fetch"))]
impl ModelStoreBackend for S3ModelStore {
    fn store(&self, _model_id: u64, _round_id: u64, _data: &[u8]) -> Result<String, ModelStoreError> {
        Err(ModelStoreError::BackendUnavailable(
            "S3 storage requires the 's3-fetch' feature".into(),
        ))
    }

    fn load(&self, _location: &str) -> Result<Vec<u8>, ModelStoreError> {
        Err(ModelStoreError::BackendUnavailable(
            "S3 storage requires the 's3-fetch' feature".into(),
        ))
    }

    fn exists(&self, _location: &str) -> bool {
        false
    }

    fn backend_name(&self) -> &str {
        "s3"
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the `ModelStore` facade.
#[derive(Debug, Clone)]
pub struct ModelStoreConfig {
    /// Base data directory for local storage and caches.
    pub data_dir: PathBuf,
    /// Whether to enable the IPFS backend (requires `ipfs-fetch` feature).
    pub enable_ipfs: bool,
    /// Whether to enable the S3 backend (requires `s3-fetch` feature).
    pub enable_s3: bool,
    /// S3 bucket name (required if `enable_s3` is true).
    pub s3_bucket: Option<String>,
    /// S3 region (required if `enable_s3` is true).
    pub s3_region: Option<String>,
    /// Maximum number of models to keep in local cache. 0 means unlimited.
    pub max_models_cached: usize,
}

impl Default for ModelStoreConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            enable_ipfs: false,
            enable_s3: false,
            s3_bucket: None,
            s3_region: None,
            max_models_cached: 100,
        }
    }
}

// ============================================================================
// ModelStore Facade
// ============================================================================

/// Main model persistence facade.
///
/// Manages multiple storage backends and a manifest of all stored models.
/// The primary backend (index 0, always local) is used for `store_model`.
/// The `load_model` method searches through all backends via the manifest.
pub struct ModelStore {
    /// Registered storage backends (index 0 is always local).
    backends: Vec<Box<dyn ModelStoreBackend>>,
    /// On-disk manifest tracking all stored models.
    manifest: ModelManifest,
    /// Index into `backends` for the primary store target.
    primary_backend: usize,
}

impl ModelStore {
    /// Creates a new `ModelStore` from the given configuration.
    ///
    /// Always initializes the local backend. Optionally initializes IPFS and S3
    /// backends based on config flags. The manifest is loaded from disk if it
    /// exists.
    pub fn new(config: ModelStoreConfig) -> Result<Self, ModelStoreError> {
        let mut backends: Vec<Box<dyn ModelStoreBackend>> = Vec::new();

        // Local backend is always available (index 0)
        let local = LocalModelStore::new(&config.data_dir)?;
        backends.push(Box::new(local));

        // IPFS backend (optional)
        if config.enable_ipfs {
            let cache_dir = config.data_dir.join("cache").join("ipfs");
            match IpfsModelStore::new("http://localhost:5001", cache_dir) {
                Ok(ipfs) => backends.push(Box::new(ipfs)),
                Err(e) => {
                    log::warn!("IPFS backend unavailable: {}", e);
                }
            }
        }

        // S3 backend (optional)
        if config.enable_s3 {
            let bucket = config.s3_bucket.as_deref().unwrap_or("helix-models");
            let region = config.s3_region.as_deref().unwrap_or("us-east-1");
            let cache_dir = config.data_dir.join("cache").join("s3");
            match S3ModelStore::new(bucket, region, cache_dir) {
                Ok(s3) => backends.push(Box::new(s3)),
                Err(e) => {
                    log::warn!("S3 backend unavailable: {}", e);
                }
            }
        }

        // Load manifest
        let manifest_path = config.data_dir.join("model_manifest.json");
        let mut manifest = ModelManifest::new(manifest_path);
        manifest.load()?;

        Ok(Self {
            backends,
            manifest,
            primary_backend: 0,
        })
    }

    /// Stores model weight data using the primary backend and records a manifest entry.
    ///
    /// Computes the SHA-256 commitment of the raw data, stores via the primary
    /// backend, creates a manifest entry with the provided metadata, and
    /// persists the manifest to disk.
    pub fn store_model(
        &mut self,
        model_id: u64,
        round_id: u64,
        data: &[u8],
        metadata: ModelStoreMetadata,
    ) -> Result<ModelManifestEntry, ModelStoreError> {
        let backend = &self.backends[self.primary_backend];
        let location = backend.store(model_id, round_id, data)?;

        let commitment: [u8; 32] = Sha256::digest(data).into();
        let completed_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = ModelManifestEntry {
            model_id,
            round_id,
            storage_backend: backend.backend_name().to_string(),
            storage_location: location,
            commitment,
            size_bytes: data.len(),
            num_contributors: metadata.num_contributors,
            loss: metadata.loss,
            error_bound: metadata.error_bound,
            steps_completed: metadata.steps_completed,
            completed_at,
            tx_hash: metadata.tx_hash,
        };

        self.manifest.add_entry(entry.clone());
        self.manifest.save()?;

        Ok(entry)
    }

    /// Loads model weight data for a specific (model_id, round_id).
    ///
    /// Looks up the manifest entry first, then loads from the appropriate backend.
    /// Verifies the SHA-256 commitment after loading.
    pub fn load_model(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<Vec<u8>, ModelStoreError> {
        let entry = self
            .manifest
            .get_entry(model_id, round_id)
            .ok_or(ModelStoreError::NotFound { model_id, round_id })?;

        // Find the backend that matches the entry
        let backend = self
            .backends
            .iter()
            .find(|b| b.backend_name() == entry.storage_backend)
            .ok_or_else(|| {
                ModelStoreError::BackendUnavailable(format!(
                    "Backend '{}' not registered for model_{}/round_{}",
                    entry.storage_backend, model_id, round_id
                ))
            })?;

        let data = backend.load(&entry.storage_location)?;

        // Verify integrity
        let actual_commitment: [u8; 32] = Sha256::digest(&data).into();
        if actual_commitment != entry.commitment {
            return Err(ModelStoreError::IntegrityError {
                expected: hex::encode(entry.commitment),
                actual: hex::encode(actual_commitment),
            });
        }

        Ok(data)
    }

    /// Returns the manifest entry for a specific (model_id, round_id), if it exists.
    pub fn get_manifest_entry(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Option<ModelManifestEntry> {
        self.manifest.get_entry(model_id, round_id).cloned()
    }

    /// Returns the latest manifest entry for the given model ID, if any.
    pub fn get_latest_entry(&self, model_id: u64) -> Option<ModelManifestEntry> {
        self.manifest.get_latest(model_id).cloned()
    }
}

/// Additional metadata provided by the caller when storing a model.
///
/// These fields are not derivable from the raw weight data alone.
pub struct ModelStoreMetadata {
    /// Number of workers that contributed to this round.
    pub num_contributors: u32,
    /// Final training loss.
    pub loss: f64,
    /// Accumulated error bound.
    pub error_bound: f64,
    /// Total training steps completed.
    pub steps_completed: u64,
    /// On-chain transaction hash, if finalized.
    pub tx_hash: Option<String>,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Generates a unique temp-file path to prevent collisions between concurrent
/// processes and threads.
fn unique_tmp_path(base: &Path) -> PathBuf {
    use rand::Rng;
    let pid = std::process::id();
    let rand_suffix: u32 = rand::thread_rng().gen();
    let tmp_name = format!(
        ".{}.{}.{}.tmp",
        base.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("data"),
        pid,
        rand_suffix,
    );
    base.with_file_name(tmp_name)
}

/// Atomically writes data to a file (write to unique temp path, then rename).
fn atomic_write(path: &Path, data: &[u8]) -> Result<(), ModelStoreError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = unique_tmp_path(path);
    fs::write(&tmp_path, data)?;

    // Rename is atomic on the same filesystem (POSIX guarantee)
    if let Err(e) = fs::rename(&tmp_path, path) {
        // Clean up temp file on failure
        let _ = fs::remove_file(&tmp_path);
        return Err(ModelStoreError::Io(e));
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ModelManifest tests ----

    #[test]
    fn test_manifest_add_and_get_entry() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_path = dir.path().join("manifest.json");
        let mut manifest = ModelManifest::new(&manifest_path);

        let entry = make_entry(1, 1);
        manifest.add_entry(entry.clone());

        let found = manifest.get_entry(1, 1).unwrap();
        assert_eq!(found.model_id, 1);
        assert_eq!(found.round_id, 1);
    }

    #[test]
    fn test_manifest_get_entry_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = ModelManifest::new(dir.path().join("manifest.json"));

        assert!(manifest.get_entry(1, 1).is_none());
    }

    #[test]
    fn test_manifest_get_latest() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = ModelManifest::new(dir.path().join("manifest.json"));

        manifest.add_entry(make_entry(1, 1));
        manifest.add_entry(make_entry(1, 3));
        manifest.add_entry(make_entry(1, 2));
        manifest.add_entry(make_entry(2, 5)); // different model

        let latest = manifest.get_latest(1).unwrap();
        assert_eq!(latest.round_id, 3);
    }

    #[test]
    fn test_manifest_get_latest_no_entries() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = ModelManifest::new(dir.path().join("manifest.json"));

        assert!(manifest.get_latest(42).is_none());
    }

    #[test]
    fn test_manifest_get_all() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = ModelManifest::new(dir.path().join("manifest.json"));

        manifest.add_entry(make_entry(1, 3));
        manifest.add_entry(make_entry(1, 1));
        manifest.add_entry(make_entry(2, 2)); // different model
        manifest.add_entry(make_entry(1, 2));

        let entries = manifest.get_all(1);
        assert_eq!(entries.len(), 3);
        // Should be sorted by round_id ascending
        assert_eq!(entries[0].round_id, 1);
        assert_eq!(entries[1].round_id, 2);
        assert_eq!(entries[2].round_id, 3);
    }

    #[test]
    fn test_manifest_get_all_empty() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = ModelManifest::new(dir.path().join("manifest.json"));

        let entries = manifest.get_all(1);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_manifest_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_path = dir.path().join("manifest.json");

        // Create and save manifest
        let mut manifest = ModelManifest::new(&manifest_path);
        manifest.add_entry(make_entry(1, 1));
        manifest.add_entry(make_entry(1, 2));
        manifest.add_entry(make_entry(2, 1));
        manifest.save().unwrap();

        // Load into a new manifest instance
        let mut loaded = ModelManifest::new(&manifest_path);
        loaded.load().unwrap();

        assert_eq!(loaded.get_all(1).len(), 2);
        assert_eq!(loaded.get_all(2).len(), 1);
        assert!(loaded.get_entry(1, 1).is_some());
        assert!(loaded.get_entry(1, 2).is_some());
        assert!(loaded.get_entry(2, 1).is_some());
    }

    #[test]
    fn test_manifest_load_nonexistent_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_path = dir.path().join("does_not_exist.json");

        let mut manifest = ModelManifest::new(&manifest_path);
        manifest.add_entry(make_entry(1, 1)); // existing entry
        manifest.load().unwrap(); // should clear entries

        // After loading a nonexistent file, entries should be empty
        assert!(manifest.get_entry(1, 1).is_none());
    }

    #[test]
    fn test_manifest_save_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_path = dir.path().join("nested").join("deep").join("manifest.json");

        let mut manifest = ModelManifest::new(&manifest_path);
        manifest.add_entry(make_entry(1, 1));
        manifest.save().unwrap();

        assert!(manifest_path.exists());
    }

    #[test]
    fn test_manifest_entry_serialization() {
        let entry = make_entry(42, 7);
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: ModelManifestEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.model_id, 42);
        assert_eq!(deserialized.round_id, 7);
        assert_eq!(deserialized.storage_backend, "local");
        assert_eq!(deserialized.commitment, entry.commitment);
        assert_eq!(deserialized.size_bytes, entry.size_bytes);
    }

    // ---- LocalModelStore tests ----

    #[test]
    fn test_local_store_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let data = b"model weight data here";
        let location = store.store(1, 1, data).unwrap();

        let loaded = store.load(&location).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_local_store_creates_directory_structure() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        store.store(42, 7, b"data").unwrap();

        let model_dir = dir.path().join("models").join("model_42");
        assert!(model_dir.exists());
        assert!(model_dir.join("round_7.bin").exists());
        assert!(model_dir.join("round_7.sha256").exists());
    }

    #[test]
    fn test_local_store_checksum_file_content() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let data = b"checksum test data";
        store.store(1, 1, data).unwrap();

        let checksum_path = dir
            .path()
            .join("models")
            .join("model_1")
            .join("round_1.sha256");
        let checksum = fs::read_to_string(&checksum_path).unwrap();

        let expected = hex::encode(Sha256::digest(data));
        assert_eq!(checksum, expected);
    }

    #[test]
    fn test_local_store_integrity_verification_passes() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let data = b"integrity test";
        let location = store.store(1, 1, data).unwrap();

        store.verify_integrity(Path::new(&location)).unwrap();
    }

    #[test]
    fn test_local_store_integrity_verification_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let data = b"original data";
        let location = store.store(1, 1, data).unwrap();

        // Corrupt the data file
        fs::write(&location, b"corrupted data").unwrap();

        let result = store.verify_integrity(Path::new(&location));
        assert!(matches!(result, Err(ModelStoreError::IntegrityError { .. })));
    }

    #[test]
    fn test_local_store_load_nonexistent_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let result = store.load("/nonexistent/path.bin");
        assert!(matches!(result, Err(ModelStoreError::Io(_))));
    }

    #[test]
    fn test_local_store_exists() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let location = store.store(1, 1, b"data").unwrap();

        assert!(store.exists(&location));
        assert!(!store.exists("/nonexistent/path.bin"));
    }

    #[test]
    fn test_local_store_backend_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();
        assert_eq!(store.backend_name(), "local");
    }

    #[test]
    fn test_local_store_multiple_rounds_same_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let data1 = b"round 1 weights";
        let data2 = b"round 2 weights";
        let data3 = b"round 3 weights";

        let loc1 = store.store(1, 1, data1).unwrap();
        let loc2 = store.store(1, 2, data2).unwrap();
        let loc3 = store.store(1, 3, data3).unwrap();

        assert_eq!(store.load(&loc1).unwrap(), data1);
        assert_eq!(store.load(&loc2).unwrap(), data2);
        assert_eq!(store.load(&loc3).unwrap(), data3);
    }

    #[test]
    fn test_local_store_different_models() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        let loc1 = store.store(1, 1, b"model 1").unwrap();
        let loc2 = store.store(2, 1, b"model 2").unwrap();

        assert_eq!(store.load(&loc1).unwrap(), b"model 1");
        assert_eq!(store.load(&loc2).unwrap(), b"model 2");
    }

    #[test]
    fn test_local_store_overwrite_same_round() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        store.store(1, 1, b"old data").unwrap();
        let location = store.store(1, 1, b"new data").unwrap();

        assert_eq!(store.load(&location).unwrap(), b"new data");
    }

    #[test]
    fn test_local_store_large_data() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path()).unwrap();

        // 1MB of data simulating real model weights
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let location = store.store(1, 1, &data).unwrap();

        let loaded = store.load(&location).unwrap();
        assert_eq!(loaded, data);
    }

    // ---- ModelStore facade tests ----

    #[test]
    fn test_model_store_default_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelStoreConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };

        let store = ModelStore::new(config).unwrap();
        assert_eq!(store.backends.len(), 1); // local only
        assert_eq!(store.backends[0].backend_name(), "local");
    }

    #[test]
    fn test_model_store_store_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelStoreConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut store = ModelStore::new(config).unwrap();

        let data = b"test model weights";
        let metadata = ModelStoreMetadata {
            num_contributors: 3,
            loss: 0.05,
            error_bound: 0.001,
            steps_completed: 100,
            tx_hash: Some("0xabc123".into()),
        };

        let entry = store.store_model(1, 1, data, metadata).unwrap();

        assert_eq!(entry.model_id, 1);
        assert_eq!(entry.round_id, 1);
        assert_eq!(entry.storage_backend, "local");
        assert_eq!(entry.size_bytes, data.len());
        assert_eq!(entry.num_contributors, 3);
        assert!((entry.loss - 0.05).abs() < f64::EPSILON);
        assert!((entry.error_bound - 0.001).abs() < f64::EPSILON);
        assert_eq!(entry.steps_completed, 100);
        assert_eq!(entry.tx_hash, Some("0xabc123".to_string()));
        assert!(entry.completed_at > 0);

        // Verify commitment matches SHA-256 of data
        let expected_commitment: [u8; 32] = Sha256::digest(data).into();
        assert_eq!(entry.commitment, expected_commitment);

        // Load back
        let loaded = store.load_model(1, 1).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_model_store_load_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelStoreConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let store = ModelStore::new(config).unwrap();

        let result = store.load_model(999, 1);
        assert!(matches!(result, Err(ModelStoreError::NotFound { .. })));
    }

    #[test]
    fn test_model_store_integrity_check_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelStoreConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut store = ModelStore::new(config).unwrap();

        let data = b"original weights";
        let metadata = ModelStoreMetadata {
            num_contributors: 1,
            loss: 0.1,
            error_bound: 0.0,
            steps_completed: 10,
            tx_hash: None,
        };
        let entry = store.store_model(1, 1, data, metadata).unwrap();

        // Corrupt the stored file
        fs::write(&entry.storage_location, b"corrupted").unwrap();
        // Also update the checksum sidecar to match corrupted data (so LocalModelStore
        // doesn't catch it), forcing the facade's commitment check to catch it
        let checksum_path = Path::new(&entry.storage_location).with_extension("sha256");
        fs::write(&checksum_path, hex::encode(Sha256::digest(b"corrupted"))).unwrap();

        let result = store.load_model(1, 1);
        assert!(matches!(result, Err(ModelStoreError::IntegrityError { .. })));
    }

    #[test]
    fn test_model_store_manifest_persists_across_instances() {
        let dir = tempfile::tempdir().unwrap();

        // Store a model
        {
            let config = ModelStoreConfig {
                data_dir: dir.path().to_path_buf(),
                ..Default::default()
            };
            let mut store = ModelStore::new(config).unwrap();
            let metadata = ModelStoreMetadata {
                num_contributors: 2,
                loss: 0.03,
                error_bound: 0.002,
                steps_completed: 50,
                tx_hash: None,
            };
            store.store_model(1, 1, b"weights v1", metadata).unwrap();
        }

        // Load from a new instance
        {
            let config = ModelStoreConfig {
                data_dir: dir.path().to_path_buf(),
                ..Default::default()
            };
            let store = ModelStore::new(config).unwrap();

            let entry = store.get_manifest_entry(1, 1);
            assert!(entry.is_some());

            let loaded = store.load_model(1, 1).unwrap();
            assert_eq!(loaded, b"weights v1");
        }
    }

    #[test]
    fn test_model_store_get_latest_entry() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelStoreConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut store = ModelStore::new(config).unwrap();

        let metadata = || ModelStoreMetadata {
            num_contributors: 1,
            loss: 0.0,
            error_bound: 0.0,
            steps_completed: 0,
            tx_hash: None,
        };

        store.store_model(1, 1, b"r1", metadata()).unwrap();
        store.store_model(1, 3, b"r3", metadata()).unwrap();
        store.store_model(1, 2, b"r2", metadata()).unwrap();

        let latest = store.get_latest_entry(1).unwrap();
        assert_eq!(latest.round_id, 3);
    }

    #[test]
    fn test_model_store_get_latest_entry_none() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelStoreConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let store = ModelStore::new(config).unwrap();

        assert!(store.get_latest_entry(999).is_none());
    }

    #[test]
    fn test_model_store_multiple_models() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelStoreConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut store = ModelStore::new(config).unwrap();

        let metadata = || ModelStoreMetadata {
            num_contributors: 1,
            loss: 0.0,
            error_bound: 0.0,
            steps_completed: 0,
            tx_hash: None,
        };

        store.store_model(1, 1, b"model1-r1", metadata()).unwrap();
        store.store_model(2, 1, b"model2-r1", metadata()).unwrap();

        assert_eq!(store.load_model(1, 1).unwrap(), b"model1-r1");
        assert_eq!(store.load_model(2, 1).unwrap(), b"model2-r1");

        let m1_latest = store.get_latest_entry(1).unwrap();
        let m2_latest = store.get_latest_entry(2).unwrap();
        assert_eq!(m1_latest.model_id, 1);
        assert_eq!(m2_latest.model_id, 2);
    }

    // ---- Atomic write tests ----

    #[test]
    fn test_atomic_write_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_file");

        atomic_write(&path, b"hello").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello");
    }

    #[test]
    fn test_atomic_write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a").join("b").join("c").join("file.bin");

        atomic_write(&path, b"nested").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"nested");
    }

    #[test]
    fn test_atomic_write_no_temp_file_left() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean_write");

        atomic_write(&path, b"data").unwrap();

        // No .tmp files should remain
        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(".tmp")
            })
            .collect();
        assert!(entries.is_empty());
    }

    // ---- Error Display tests ----

    #[test]
    fn test_error_display_io() {
        let err = ModelStoreError::Io(io::Error::new(io::ErrorKind::NotFound, "gone"));
        let msg = format!("{}", err);
        assert!(msg.contains("IO error"));
        assert!(msg.contains("gone"));
    }

    #[test]
    fn test_error_display_serialization() {
        let err = ModelStoreError::Serialization("bad json".into());
        let msg = format!("{}", err);
        assert!(msg.contains("serialization"));
        assert!(msg.contains("bad json"));
    }

    #[test]
    fn test_error_display_backend_unavailable() {
        let err = ModelStoreError::BackendUnavailable("ipfs".into());
        let msg = format!("{}", err);
        assert!(msg.contains("ipfs"));
        assert!(msg.contains("not available"));
    }

    #[test]
    fn test_error_display_not_found() {
        let err = ModelStoreError::NotFound {
            model_id: 42,
            round_id: 7,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
        assert!(msg.contains("7"));
    }

    #[test]
    fn test_error_display_integrity() {
        let err = ModelStoreError::IntegrityError {
            expected: "aaa".into(),
            actual: "bbb".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("aaa"));
        assert!(msg.contains("bbb"));
    }

    // ---- Stub backend tests (feature-gated) ----

    #[cfg(not(feature = "ipfs-fetch"))]
    #[test]
    fn test_ipfs_stub_returns_unavailable() {
        let result = IpfsModelStore::new("http://localhost:5001", "/tmp/cache");
        assert!(matches!(result, Err(ModelStoreError::BackendUnavailable(_))));
    }

    #[cfg(not(feature = "s3-fetch"))]
    #[test]
    fn test_s3_stub_returns_unavailable() {
        let result = S3ModelStore::new("bucket", "us-east-1", "/tmp/cache");
        assert!(matches!(result, Err(ModelStoreError::BackendUnavailable(_))));
    }

    // ---- Helper functions ----

    fn make_entry(model_id: u64, round_id: u64) -> ModelManifestEntry {
        ModelManifestEntry {
            model_id,
            round_id,
            storage_backend: "local".to_string(),
            storage_location: format!("/tmp/models/model_{}/round_{}.bin", model_id, round_id),
            commitment: [0u8; 32],
            size_bytes: 1024,
            num_contributors: 3,
            loss: 0.05,
            error_bound: 0.001,
            steps_completed: 100,
            completed_at: 1700000000,
            tx_hash: None,
        }
    }
}
