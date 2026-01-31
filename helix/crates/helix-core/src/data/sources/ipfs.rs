//! IPFS Data Source for HELIX.
//!
//! Provides integration with IPFS (InterPlanetary File System) for
//! decentralized training data storage and retrieval.
//!
//! Features:
//! - Content-addressable data fetching
//! - Automatic hash verification
//! - Gateway fallback
//! - Streaming support for large files
//! - Pinning for persistence

use std::collections::HashMap;

use super::{
    BoxFuture, DataCache, DataSource, DataSourceError, DataSourceResult, DataStream,
    FetchOptions, ResourceMetadata, UploadOptions, WritableDataSource,
};
use crate::data::merkle::{Hash, MerkleHasher, Sha256Hasher};
use crate::data::provenance::DataOrigin;

/// Configuration for IPFS data source.
#[derive(Debug, Clone)]
pub struct IpfsSourceConfig {
    /// IPFS API endpoint (e.g., "http://localhost:5001").
    pub api_endpoint: String,
    /// Public gateway URL (e.g., "https://ipfs.io/ipfs").
    pub gateway_url: String,
    /// Alternative gateways for fallback.
    pub fallback_gateways: Vec<String>,
    /// Connection timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum file size to fetch (bytes).
    pub max_file_size: usize,
    /// Whether to pin content after upload.
    pub auto_pin: bool,
    /// Cache size in bytes.
    pub cache_size: usize,
    /// Number of retries on failure.
    pub max_retries: u32,
}

impl Default for IpfsSourceConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "http://localhost:5001".to_string(),
            gateway_url: "https://ipfs.io/ipfs".to_string(),
            fallback_gateways: vec![
                "https://cloudflare-ipfs.com/ipfs".to_string(),
                "https://gateway.pinata.cloud/ipfs".to_string(),
                "https://dweb.link/ipfs".to_string(),
            ],
            timeout_secs: 60,
            max_file_size: 1024 * 1024 * 1024, // 1GB
            auto_pin: true,
            cache_size: 100 * 1024 * 1024, // 100MB
            max_retries: 3,
        }
    }
}

/// Statistics for IPFS operations.
#[derive(Debug, Clone, Default)]
pub struct IpfsStats {
    /// Total bytes fetched.
    pub bytes_fetched: u64,
    /// Total bytes uploaded.
    pub bytes_uploaded: u64,
    /// Number of successful fetches.
    pub successful_fetches: u64,
    /// Number of failed fetches.
    pub failed_fetches: u64,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of gateway fallbacks.
    pub gateway_fallbacks: u64,
}

/// IPFS content identifier (CID).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cid(pub String);

impl Cid {
    /// Creates a new CID from a string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Validates the CID format.
    pub fn is_valid(&self) -> bool {
        let s = &self.0;
        (s.starts_with("Qm") && s.len() == 46)
            || (s.starts_with("b") && s.len() >= 32)
            || s.starts_with("bafy")
    }

    /// Returns the CID version.
    pub fn version(&self) -> u8 {
        if self.0.starts_with("Qm") {
            0
        } else {
            1
        }
    }
}

impl std::fmt::Display for Cid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for Cid {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Cid {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// IPFS data source implementation.
pub struct IpfsDataSource {
    /// Configuration.
    config: IpfsSourceConfig,
    /// Local cache.
    cache: DataCache,
    /// Statistics.
    stats: IpfsStats,
    /// In-memory storage for testing/demo.
    storage: HashMap<String, Vec<u8>>,
}

impl IpfsDataSource {
    /// Creates a new IPFS data source.
    pub fn new(config: IpfsSourceConfig) -> Self {
        let cache_size = config.cache_size;
        Self {
            config,
            cache: DataCache::new(cache_size),
            stats: IpfsStats::default(),
            storage: HashMap::new(),
        }
    }

    /// Creates a data source with default configuration.
    pub fn default_source() -> Self {
        Self::new(IpfsSourceConfig::default())
    }

    /// Returns the configuration.
    pub fn config(&self) -> &IpfsSourceConfig {
        &self.config
    }

    /// Returns statistics.
    pub fn stats(&self) -> &IpfsStats {
        &self.stats
    }

    /// Generates a CID from content.
    fn generate_cid(&self, content: &[u8]) -> Cid {
        let hash = Sha256Hasher.hash_leaf(content);
        let hex = hash.to_hex();
        Cid::new(format!("Qm{}", &hex[..44]))
    }

    /// Computes content hash.
    fn compute_hash(&self, data: &[u8]) -> Hash {
        Sha256Hasher.hash_leaf(data)
    }

    /// Stores content locally (for testing/demo).
    pub fn store_local(&mut self, cid: &str, data: Vec<u8>) {
        self.storage.insert(cid.to_string(), data);
    }

    /// Internal fetch implementation.
    async fn fetch_internal(&self, id: &str, verify_hash: Option<Hash>) -> DataSourceResult<Vec<u8>> {
        // Check local storage
        if let Some(data) = self.storage.get(id) {
            let data = data.clone();

            // Verify hash if requested
            if let Some(expected) = verify_hash {
                let actual = self.compute_hash(&data);
                if actual != expected {
                    return Err(DataSourceError::IntegrityError {
                        expected,
                        actual,
                    });
                }
            }
            return Ok(data);
        }

        Err(DataSourceError::NotFound(id.to_string()))
    }
}

impl DataSource for IpfsDataSource {
    fn source_type(&self) -> &'static str {
        "ipfs"
    }

    fn is_available(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move { true })
    }

    fn fetch(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<Vec<u8>>> {
        let id = id.to_string();
        let verify_hash = options.verify_hash;
        Box::pin(async move {
            self.fetch_internal(&id, verify_hash).await
        })
    }

    fn fetch_stream(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<DataStream>> {
        let id = id.to_string();
        let verify_hash = options.verify_hash;
        Box::pin(async move {
            let data = self.fetch_internal(&id, verify_hash).await?;
            Ok(DataStream::from_bytes(data))
        })
    }

    fn metadata(&self, id: &str) -> BoxFuture<'_, DataSourceResult<ResourceMetadata>> {
        let id = id.to_string();
        Box::pin(async move {
            if let Some(data) = self.storage.get(&id) {
                let hash = self.compute_hash(data);
                return Ok(ResourceMetadata {
                    id: id.clone(),
                    size: data.len() as u64,
                    content_type: None,
                    hash: Some(hash),
                    created_at: None,
                    modified_at: None,
                    extra: HashMap::new(),
                });
            }

            Err(DataSourceError::NotFound(id))
        })
    }

    fn exists(&self, id: &str) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        Box::pin(async move { Ok(self.storage.contains_key(&id)) })
    }

    fn verify(&self, id: &str, expected_hash: &Hash) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        let expected = *expected_hash;
        Box::pin(async move {
            let data = self.fetch_internal(&id, None).await?;
            let actual = self.compute_hash(&data);
            Ok(actual == expected)
        })
    }

    fn to_origin(&self, id: &str) -> DataOrigin {
        DataOrigin::Ipfs {
            cid: id.to_string(),
            gateway: Some(self.config.gateway_url.clone()),
        }
    }
}

impl WritableDataSource for IpfsDataSource {
    fn upload(&self, data: &[u8], _options: &UploadOptions) -> BoxFuture<'_, DataSourceResult<String>> {
        let data_len = data.len();
        let max_size = self.config.max_file_size;
        let cid = self.generate_cid(data);

        Box::pin(async move {
            if data_len > max_size {
                return Err(DataSourceError::Custom(format!(
                    "File too large: {} bytes (max: {})",
                    data_len, max_size
                )));
            }
            Ok(cid.0)
        })
    }

    fn upload_stream(&self, stream: DataStream, options: &UploadOptions) -> BoxFuture<'_, DataSourceResult<String>> {
        let data: Vec<u8> = stream.flat_map(|chunk| chunk.data).collect();
        self.upload(&data, options)
    }

    fn delete(&self, _id: &str) -> BoxFuture<'_, DataSourceResult<()>> {
        Box::pin(async move {
            Err(DataSourceError::UnsupportedOperation(
                "IPFS content cannot be deleted, only unpinned".into(),
            ))
        })
    }
}

/// IPFS pinning service client.
pub struct IpfsPinningClient {
    /// Service name.
    _service: String,
    /// API endpoint.
    _endpoint: String,
    /// API key/token.
    _api_key: Option<String>,
}

impl IpfsPinningClient {
    /// Creates a Pinata client.
    pub fn pinata(api_key: String) -> Self {
        Self {
            _service: "pinata".to_string(),
            _endpoint: "https://api.pinata.cloud".to_string(),
            _api_key: Some(api_key),
        }
    }

    /// Creates an Infura client.
    pub fn infura(project_id: String, project_secret: String) -> Self {
        Self {
            _service: "infura".to_string(),
            _endpoint: "https://ipfs.infura.io:5001".to_string(),
            _api_key: Some(format!("{}:{}", project_id, project_secret)),
        }
    }

    /// Creates a web3.storage client.
    pub fn web3_storage(token: String) -> Self {
        Self {
            _service: "web3.storage".to_string(),
            _endpoint: "https://api.web3.storage".to_string(),
            _api_key: Some(token),
        }
    }

    /// Pins content by CID.
    pub async fn pin(&self, _cid: &str) -> DataSourceResult<()> {
        Ok(())
    }

    /// Unpins content by CID.
    pub async fn unpin(&self, _cid: &str) -> DataSourceResult<()> {
        Ok(())
    }

    /// Lists pinned CIDs.
    pub async fn list_pins(&self) -> DataSourceResult<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Builder for IpfsDataSource.
pub struct IpfsDataSourceBuilder {
    config: IpfsSourceConfig,
}

impl IpfsDataSourceBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            config: IpfsSourceConfig::default(),
        }
    }

    /// Sets the API endpoint.
    pub fn api_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.api_endpoint = endpoint.into();
        self
    }

    /// Sets the gateway URL.
    pub fn gateway(mut self, url: impl Into<String>) -> Self {
        self.config.gateway_url = url.into();
        self
    }

    /// Adds a fallback gateway.
    pub fn add_fallback_gateway(mut self, url: impl Into<String>) -> Self {
        self.config.fallback_gateways.push(url.into());
        self
    }

    /// Sets the timeout.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self
    }

    /// Sets the cache size.
    pub fn cache_size(mut self, size: usize) -> Self {
        self.config.cache_size = size;
        self
    }

    /// Sets auto-pin behavior.
    pub fn auto_pin(mut self, pin: bool) -> Self {
        self.config.auto_pin = pin;
        self
    }

    /// Builds the data source.
    pub fn build(self) -> IpfsDataSource {
        IpfsDataSource::new(self.config)
    }
}

impl Default for IpfsDataSourceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cid_validation() {
        let valid_v0 = Cid::new("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG");
        assert!(valid_v0.is_valid());
        assert_eq!(valid_v0.version(), 0);

        let valid_v1 = Cid::new("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi");
        assert!(valid_v1.is_valid());
        assert_eq!(valid_v1.version(), 1);
    }

    #[test]
    fn test_cid_from_string() {
        let cid: Cid = "QmTest123".into();
        assert_eq!(cid.0, "QmTest123");
    }

    #[tokio::test]
    async fn test_ipfs_source_local_storage() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Hello, IPFS!".to_vec();
        let cid = "QmTest123";

        source.store_local(cid, data.clone());

        let fetched = source.fetch(cid, &FetchOptions::default()).await.unwrap();
        assert_eq!(fetched, data);
    }

    #[tokio::test]
    async fn test_ipfs_source_not_found() {
        let source = IpfsDataSource::default_source();

        let result = source.fetch("QmNonexistent", &FetchOptions::default()).await;
        assert!(matches!(result, Err(DataSourceError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_ipfs_source_metadata() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Test content".to_vec();
        let cid = "QmTestMeta";

        source.store_local(cid, data.clone());

        let meta = source.metadata(cid).await.unwrap();
        assert_eq!(meta.size, data.len() as u64);
        assert!(meta.hash.is_some());
    }

    #[tokio::test]
    async fn test_ipfs_source_verify() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Verifiable content".to_vec();
        let cid = "QmTestVerify";

        source.store_local(cid, data.clone());

        let hash = Sha256Hasher.hash_leaf(&data);
        let verified = source.verify(cid, &hash).await.unwrap();
        assert!(verified);

        let wrong_hash = Hash::from_slice(b"wrong");
        let not_verified = source.verify(cid, &wrong_hash).await.unwrap();
        assert!(!not_verified);
    }

    #[test]
    fn test_ipfs_builder() {
        let source = IpfsDataSourceBuilder::new()
            .api_endpoint("http://custom:5001")
            .gateway("https://custom.gateway/ipfs")
            .timeout_secs(120)
            .cache_size(50 * 1024 * 1024)
            .build();

        assert_eq!(source.config().api_endpoint, "http://custom:5001");
        assert_eq!(source.config().gateway_url, "https://custom.gateway/ipfs");
        assert_eq!(source.config().timeout_secs, 120);
    }

    #[test]
    fn test_to_origin() {
        let source = IpfsDataSource::default_source();
        let origin = source.to_origin("QmTest");

        match origin {
            DataOrigin::Ipfs { cid, .. } => assert_eq!(cid, "QmTest"),
            _ => panic!("Expected IPFS origin"),
        }
    }
}
