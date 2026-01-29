//! IPFS Integration Module.
//!
//! Provides functionality for storing and retrieving data from IPFS
//! for decentralized model and dataset storage.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// IPFS Content Identifier (CID).
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Cid(pub String);

impl Cid {
    /// Creates a CID from a string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Creates a mock CID from content hash.
    pub fn from_content(content: &[u8]) -> Self {
        // Simplified - would use actual CID generation in production
        let hash = simple_hash(content);
        Self(format!("Qm{}", hex_encode(&hash[..20])))
    }
}

impl std::fmt::Display for Cid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// IPFS configuration.
#[derive(Debug, Clone)]
pub struct IpfsConfig {
    /// IPFS API endpoint.
    pub api_endpoint: String,
    /// Gateway URL for retrieval.
    pub gateway_url: String,
    /// Connection timeout (seconds).
    pub timeout_secs: u64,
    /// Maximum file size (bytes).
    pub max_file_size: usize,
    /// Pin content after upload.
    pub pin_content: bool,
}

impl Default for IpfsConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "http://localhost:5001/api/v0".to_string(),
            gateway_url: "https://ipfs.io/ipfs".to_string(),
            timeout_secs: 30,
            max_file_size: 100 * 1024 * 1024, // 100MB
            pin_content: true,
        }
    }
}

/// IPFS client for storing and retrieving content.
pub struct IpfsClient {
    /// Configuration.
    config: IpfsConfig,
    /// Local cache of content.
    cache: HashMap<Cid, Vec<u8>>,
    /// Upload statistics.
    stats: IpfsStats,
}

/// IPFS operation statistics.
#[derive(Debug, Clone, Default)]
pub struct IpfsStats {
    /// Total bytes uploaded.
    pub bytes_uploaded: u64,
    /// Total bytes downloaded.
    pub bytes_downloaded: u64,
    /// Number of uploads.
    pub num_uploads: u64,
    /// Number of downloads.
    pub num_downloads: u64,
    /// Cache hits.
    pub cache_hits: u64,
}

impl IpfsClient {
    /// Creates a new IPFS client.
    pub fn new(config: IpfsConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: IpfsStats::default(),
        }
    }

    /// Creates a client with default config.
    pub fn default_client() -> Self {
        Self::new(IpfsConfig::default())
    }

    /// Uploads content to IPFS.
    pub async fn upload(&mut self, content: &[u8]) -> Result<Cid, IpfsError> {
        if content.len() > self.config.max_file_size {
            return Err(IpfsError::FileTooLarge {
                size: content.len(),
                max: self.config.max_file_size,
            });
        }

        // Generate CID
        let cid = Cid::from_content(content);

        // Store in cache
        self.cache.insert(cid.clone(), content.to_vec());

        // Update stats
        self.stats.bytes_uploaded += content.len() as u64;
        self.stats.num_uploads += 1;

        // In production, would make HTTP request to IPFS API
        // For now, just return the CID
        Ok(cid)
    }

    /// Downloads content from IPFS.
    pub async fn download(&mut self, cid: &Cid) -> Result<Vec<u8>, IpfsError> {
        // Check cache first
        if let Some(content) = self.cache.get(cid) {
            self.stats.cache_hits += 1;
            return Ok(content.clone());
        }

        // In production, would make HTTP request to IPFS gateway
        // For now, return error since not in cache
        Err(IpfsError::NotFound(cid.clone()))
    }

    /// Pins content so it won't be garbage collected.
    pub async fn pin(&self, cid: &Cid) -> Result<(), IpfsError> {
        // Would make API call in production
        Ok(())
    }

    /// Unpins content.
    pub async fn unpin(&self, cid: &Cid) -> Result<(), IpfsError> {
        // Would make API call in production
        Ok(())
    }

    /// Checks if content exists.
    pub async fn exists(&self, cid: &Cid) -> bool {
        self.cache.contains_key(cid)
    }

    /// Gets the gateway URL for a CID.
    pub fn gateway_url(&self, cid: &Cid) -> String {
        format!("{}/{}", self.config.gateway_url, cid)
    }

    /// Gets statistics.
    pub fn stats(&self) -> &IpfsStats {
        &self.stats
    }

    /// Clears the local cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// IPFS operation errors.
#[derive(Debug)]
pub enum IpfsError {
    /// Content not found.
    NotFound(Cid),
    /// File too large.
    FileTooLarge { size: usize, max: usize },
    /// Connection error.
    ConnectionError(String),
    /// Timeout.
    Timeout,
    /// Invalid CID.
    InvalidCid(String),
}

impl std::fmt::Display for IpfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpfsError::NotFound(cid) => write!(f, "Content not found: {}", cid),
            IpfsError::FileTooLarge { size, max } => {
                write!(f, "File too large: {} bytes (max: {})", size, max)
            }
            IpfsError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            IpfsError::Timeout => write!(f, "Operation timed out"),
            IpfsError::InvalidCid(cid) => write!(f, "Invalid CID: {}", cid),
        }
    }
}

/// Model storage on IPFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfsModelRef {
    /// Model CID.
    pub cid: Cid,
    /// Model name.
    pub name: String,
    /// Version.
    pub version: String,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Upload timestamp.
    pub uploaded_at: u64,
    /// Hash of the model.
    pub model_hash: [u8; 32],
}

/// Dataset storage on IPFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfsDatasetRef {
    /// Dataset CID.
    pub cid: Cid,
    /// Dataset name.
    pub name: String,
    /// Number of samples.
    pub num_samples: usize,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Shard CIDs (if sharded).
    pub shard_cids: Vec<Cid>,
}

/// IPFS-backed model registry.
pub struct IpfsModelRegistry {
    /// IPFS client.
    client: IpfsClient,
    /// Registered models.
    models: HashMap<String, Vec<IpfsModelRef>>,
    /// Registered datasets.
    datasets: HashMap<String, IpfsDatasetRef>,
}

impl IpfsModelRegistry {
    /// Creates a new registry.
    pub fn new(client: IpfsClient) -> Self {
        Self {
            client,
            models: HashMap::new(),
            datasets: HashMap::new(),
        }
    }

    /// Registers a model.
    pub async fn register_model(
        &mut self,
        name: String,
        version: String,
        data: &[u8],
        model_hash: [u8; 32],
    ) -> Result<Cid, IpfsError> {
        let cid = self.client.upload(data).await?;

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let model_ref = IpfsModelRef {
            cid: cid.clone(),
            name: name.clone(),
            version,
            size_bytes: data.len(),
            uploaded_at: now,
            model_hash,
        };

        self.models.entry(name).or_default().push(model_ref);

        Ok(cid)
    }

    /// Gets the latest version of a model.
    pub fn get_latest_model(&self, name: &str) -> Option<&IpfsModelRef> {
        self.models.get(name)?.last()
    }

    /// Gets a specific version of a model.
    pub fn get_model_version(&self, name: &str, version: &str) -> Option<&IpfsModelRef> {
        self.models
            .get(name)?
            .iter()
            .find(|m| m.version == version)
    }

    /// Lists all model versions.
    pub fn list_model_versions(&self, name: &str) -> Vec<&IpfsModelRef> {
        self.models.get(name).map(|v| v.iter().collect()).unwrap_or_default()
    }

    /// Downloads model data.
    pub async fn download_model(&mut self, cid: &Cid) -> Result<Vec<u8>, IpfsError> {
        self.client.download(cid).await
    }

    /// Registers a dataset.
    pub async fn register_dataset(
        &mut self,
        name: String,
        data: &[u8],
        num_samples: usize,
    ) -> Result<Cid, IpfsError> {
        let cid = self.client.upload(data).await?;

        let dataset_ref = IpfsDatasetRef {
            cid: cid.clone(),
            name: name.clone(),
            num_samples,
            size_bytes: data.len(),
            shard_cids: Vec::new(),
        };

        self.datasets.insert(name, dataset_ref);

        Ok(cid)
    }

    /// Gets a dataset reference.
    pub fn get_dataset(&self, name: &str) -> Option<&IpfsDatasetRef> {
        self.datasets.get(name)
    }

    /// Lists all models.
    pub fn list_models(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Lists all datasets.
    pub fn list_datasets(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }
}

// Helper functions

fn simple_hash(data: &[u8]) -> [u8; 32] {
    let mut hash = [0u8; 32];
    for (i, byte) in data.iter().enumerate() {
        hash[i % 32] ^= byte;
    }
    hash
}

fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ipfs_upload_download() {
        let mut client = IpfsClient::default_client();
        
        let content = b"Hello, IPFS!";
        let cid = client.upload(content).await.unwrap();
        
        let downloaded = client.download(&cid).await.unwrap();
        assert_eq!(content.to_vec(), downloaded);
    }

    #[tokio::test]
    async fn test_ipfs_cid_generation() {
        let cid = Cid::from_content(b"test content");
        assert!(cid.0.starts_with("Qm"));
    }

    #[tokio::test]
    async fn test_model_registry() {
        let client = IpfsClient::default_client();
        let mut registry = IpfsModelRegistry::new(client);
        
        let model_data = b"model weights here";
        let cid = registry
            .register_model(
                "my_model".to_string(),
                "1.0".to_string(),
                model_data,
                [0; 32],
            )
            .await
            .unwrap();

        let model = registry.get_latest_model("my_model").unwrap();
        assert_eq!(model.name, "my_model");
        assert_eq!(model.cid, cid);
    }
}
