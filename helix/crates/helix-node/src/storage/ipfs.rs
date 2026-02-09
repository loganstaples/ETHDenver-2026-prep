//! IPFS Storage Backend.
//!
//! Stores and retrieves model checkpoints and node state via the IPFS HTTP API
//! (typically Kubo running at `http://localhost:5001`).
//!
//! Data is stored as opaque blobs. The IPFS content-hash (CID) is returned as
//! the storage key, providing built-in integrity verification.

use std::time::Duration;

use sha2::{Digest, Sha256};

use super::StorageBackend;

/// IPFS storage backend using the HTTP API.
pub struct IpfsStorage {
    /// Base URL of the IPFS HTTP API (e.g. `http://localhost:5001`).
    api_url: String,
    /// HTTP client (shared, connection-pooled).
    client: reqwest::Client,
    /// Request timeout.
    timeout: Duration,
}

/// Errors from IPFS storage operations.
#[derive(Debug)]
pub enum IpfsStorageError {
    /// HTTP request failed.
    Http(String),
    /// IPFS API returned an error.
    Api(String),
    /// Serialization/deserialization error.
    Serialization(String),
}

impl std::fmt::Display for IpfsStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(e) => write!(f, "IPFS HTTP error: {}", e),
            Self::Api(e) => write!(f, "IPFS API error: {}", e),
            Self::Serialization(e) => write!(f, "IPFS serialization error: {}", e),
        }
    }
}

impl std::error::Error for IpfsStorageError {}

/// Response from IPFS `/api/v0/add`.
#[derive(serde::Deserialize)]
struct IpfsAddResponse {
    #[serde(rename = "Hash")]
    hash: String,
    #[serde(rename = "Size")]
    size: String,
}

impl IpfsStorage {
    /// Creates a new IPFS storage backend.
    ///
    /// `api_url` should point to a Kubo (go-ipfs) HTTP API endpoint,
    /// e.g. `http://localhost:5001`.
    pub fn new(api_url: impl Into<String>) -> Self {
        Self {
            api_url: api_url.into(),
            client: reqwest::Client::new(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Creates an IPFS storage with custom timeout.
    pub fn with_timeout(api_url: impl Into<String>, timeout: Duration) -> Self {
        Self {
            api_url: api_url.into(),
            client: reqwest::Client::new(),
            timeout,
        }
    }

    /// Stores data on IPFS via `POST /api/v0/add`.
    /// Returns the CID (content hash) of the stored data.
    pub async fn store(&self, data: &[u8]) -> Result<String, IpfsStorageError> {
        let url = format!("{}/api/v0/add", self.api_url);

        let part = reqwest::multipart::Part::bytes(data.to_vec())
            .file_name("data");
        let form = reqwest::multipart::Form::new()
            .part("file", part);

        let response = self.client
            .post(&url)
            .multipart(form)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| IpfsStorageError::Http(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(IpfsStorageError::Api(format!(
                "IPFS add failed ({}): {}", status, body
            )));
        }

        let add_response: IpfsAddResponse = response
            .json()
            .await
            .map_err(|e| IpfsStorageError::Serialization(e.to_string()))?;

        Ok(add_response.hash)
    }

    /// Retrieves data from IPFS via `POST /api/v0/cat?arg=<cid>`.
    pub async fn fetch(&self, cid: &str) -> Result<Vec<u8>, IpfsStorageError> {
        let url = format!("{}/api/v0/cat?arg={}", self.api_url, cid);

        let response = self.client
            .post(&url)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| IpfsStorageError::Http(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(IpfsStorageError::Api(format!(
                "IPFS cat failed ({}): {}", status, body
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| IpfsStorageError::Http(e.to_string()))?;

        Ok(bytes.to_vec())
    }

    /// Pins a CID to prevent garbage collection via `POST /api/v0/pin/add`.
    pub async fn pin(&self, cid: &str) -> Result<(), IpfsStorageError> {
        let url = format!("{}/api/v0/pin/add?arg={}", self.api_url, cid);

        let response = self.client
            .post(&url)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| IpfsStorageError::Http(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(IpfsStorageError::Api(format!(
                "IPFS pin failed ({}): {}", status, body
            )));
        }

        Ok(())
    }

    /// Unpins a CID via `POST /api/v0/pin/rm`.
    pub async fn unpin(&self, cid: &str) -> Result<(), IpfsStorageError> {
        let url = format!("{}/api/v0/pin/rm?arg={}", self.api_url, cid);

        let response = self.client
            .post(&url)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| IpfsStorageError::Http(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            // Pin not found is OK — already unpinned
            if !body.contains("not pinned") {
                return Err(IpfsStorageError::Api(format!(
                    "IPFS unpin failed ({}): {}", status, body
                )));
            }
        }

        Ok(())
    }

    /// Stores a model checkpoint on IPFS and pins it.
    /// Returns the CID.
    pub async fn store_checkpoint(&self, round: u64, data: &[u8]) -> Result<String, IpfsStorageError> {
        // Prefix data with round number for identification
        let mut payload = round.to_le_bytes().to_vec();
        payload.extend_from_slice(data);

        let cid = self.store(&payload).await?;
        self.pin(&cid).await?;

        log::info!("Stored checkpoint for round {} on IPFS: {}", round, cid);
        Ok(cid)
    }

    /// Fetches a model checkpoint from IPFS by CID.
    pub async fn fetch_checkpoint(&self, cid: &str) -> Result<(u64, Vec<u8>), IpfsStorageError> {
        let payload = self.fetch(cid).await?;

        if payload.len() < 8 {
            return Err(IpfsStorageError::Serialization(
                "Checkpoint payload too small".to_string(),
            ));
        }

        let round = u64::from_le_bytes(payload[..8].try_into().unwrap());
        let data = payload[8..].to_vec();

        Ok((round, data))
    }
}

/// StorageBackend implementation for IPFS.
///
/// Keys are mapped to IPFS CIDs. The `save_state` method stores data and
/// returns immediately (the CID is logged but not returned via this trait).
/// For CID-based retrieval, use the `store`/`fetch` methods directly.
///
/// Note: This implementation uses `tokio::runtime::Handle::current()` to
/// bridge async IPFS calls into the sync StorageBackend trait. It requires
/// a running tokio runtime.
impl StorageBackend for IpfsStorage {
    type Error = IpfsStorageError;

    fn save_state(&self, key: &str, data: &[u8]) -> Result<(), Self::Error> {
        let rt = tokio::runtime::Handle::current();
        let data = data.to_vec();
        let self_api = self.api_url.clone();
        let client = self.client.clone();
        let timeout = self.timeout;

        // Use block_in_place to avoid deadlocks in tokio
        let cid = tokio::task::block_in_place(|| {
            rt.block_on(async {
                let storage = IpfsStorage {
                    api_url: self_api,
                    client,
                    timeout,
                };
                storage.store(&data).await
            })
        })?;

        log::info!("Saved state '{}' to IPFS: {}", key, cid);
        Ok(())
    }

    fn load_state(&self, _key: &str) -> Result<Option<Vec<u8>>, Self::Error> {
        // IPFS is content-addressed — loading by arbitrary key is not supported.
        // Use fetch(cid) directly for content retrieval.
        Ok(None)
    }

    fn delete_state(&self, _key: &str) -> Result<(), Self::Error> {
        // IPFS content is immutable; deletion happens via unpin + GC.
        Ok(())
    }
}
