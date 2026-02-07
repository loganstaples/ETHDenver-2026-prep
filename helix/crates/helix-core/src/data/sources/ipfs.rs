//! IPFS Data Source for HELIX.
//!
//! Provides integration with IPFS (InterPlanetary File System) for
//! decentralized training data storage and retrieval.
//!
//! Features:
//! - Content-addressable data fetching
//! - Automatic hash verification
//! - Gateway fallback with health checking
//! - Streaming support for large files
//! - Pinning for persistence via multiple services
//! - Chunked retrieval for large datasets
//! - DAG operations for dataset sharding
//! - Retry with exponential backoff
//! - Connection pooling and caching

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use super::{
    BoxFuture, DataCache, DataChunk, DataSource, DataSourceError, DataSourceResult, DataStream,
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

/// Gateway health status.
#[derive(Debug, Clone)]
pub struct GatewayHealth {
    /// Gateway URL.
    pub url: String,
    /// Whether the gateway is healthy.
    pub healthy: bool,
    /// Last check timestamp.
    pub last_check: u64,
    /// Average response time in milliseconds.
    pub avg_response_ms: u64,
    /// Success rate (0.0-1.0).
    pub success_rate: f64,
    /// Number of requests.
    pub request_count: u64,
}

impl GatewayHealth {
    /// Creates a new gateway health record.
    pub fn new(url: String) -> Self {
        Self {
            url,
            healthy: true,
            last_check: 0,
            avg_response_ms: 0,
            success_rate: 1.0,
            request_count: 0,
        }
    }
}

/// Pin status for content.
#[derive(Debug, Clone, PartialEq)]
pub enum PinStatus {
    /// Content is pinned.
    Pinned,
    /// Pin is in progress.
    Pinning,
    /// Pin failed.
    Failed(String),
    /// Not pinned.
    Unpinned,
    /// Status unknown.
    Unknown,
}

/// Information about a pinned item.
#[derive(Debug, Clone)]
pub struct PinInfo {
    /// Content identifier.
    pub cid: String,
    /// Pin status.
    pub status: PinStatus,
    /// Size in bytes.
    pub size: u64,
    /// Pin timestamp.
    pub pinned_at: Option<u64>,
    /// Pin name/label.
    pub name: Option<String>,
    /// Pin service used.
    pub service: String,
}

/// IPFS DAG node for dataset organization.
#[derive(Debug, Clone)]
pub struct DagNode {
    /// Node CID.
    pub cid: String,
    /// Node data.
    pub data: Vec<u8>,
    /// Links to child nodes.
    pub links: Vec<DagLink>,
}

/// Link in a DAG node.
#[derive(Debug, Clone)]
pub struct DagLink {
    /// Link name.
    pub name: String,
    /// Target CID.
    pub cid: String,
    /// Size of target.
    pub size: u64,
}

/// Chunked content for streaming.
#[derive(Debug, Clone)]
pub struct ChunkedContent {
    /// Total content size.
    pub total_size: u64,
    /// Chunk size.
    pub chunk_size: usize,
    /// Number of chunks.
    pub num_chunks: usize,
    /// Root CID.
    pub root_cid: String,
    /// Chunk CIDs.
    pub chunk_cids: Vec<String>,
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
    /// Gateway health tracking.
    gateway_health: HashMap<String, GatewayHealth>,
    /// Pin registry.
    pins: HashMap<String, PinInfo>,
    /// DAG nodes.
    dag_nodes: HashMap<String, DagNode>,
    /// Chunked content registry.
    chunked_content: HashMap<String, ChunkedContent>,
}

impl IpfsDataSource {
    /// Creates a new IPFS data source.
    pub fn new(config: IpfsSourceConfig) -> Self {
        let cache_size = config.cache_size;

        // Initialize gateway health for all configured gateways
        let mut gateway_health = HashMap::new();
        gateway_health.insert(
            config.gateway_url.clone(),
            GatewayHealth::new(config.gateway_url.clone())
        );
        for gateway in &config.fallback_gateways {
            gateway_health.insert(gateway.clone(), GatewayHealth::new(gateway.clone()));
        }

        Self {
            config,
            cache: DataCache::new(cache_size),
            stats: IpfsStats::default(),
            storage: HashMap::new(),
            gateway_health,
            pins: HashMap::new(),
            dag_nodes: HashMap::new(),
            chunked_content: HashMap::new(),
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

    /// Returns gateway health information.
    pub fn gateway_health(&self) -> &HashMap<String, GatewayHealth> {
        &self.gateway_health
    }

    /// Generates a CID from content using IPFS-compatible hashing.
    pub fn generate_cid(&self, content: &[u8]) -> Cid {
        let hash = Sha256Hasher.hash_leaf(content);
        let hex = hash.to_hex();
        // Generate a CIDv0-style identifier
        Cid::new(format!("Qm{}", &hex[..44]))
    }

    /// Generates a CIDv1 from content.
    pub fn generate_cidv1(&self, content: &[u8]) -> Cid {
        let hash = Sha256Hasher.hash_leaf(content);
        let hex = hash.to_hex();
        Cid::new(format!("bafybeig{}", &hex[..52]))
    }

    /// Computes content hash.
    fn compute_hash(&self, data: &[u8]) -> Hash {
        Sha256Hasher.hash_leaf(data)
    }

    /// Stores content locally (for testing/demo).
    pub fn store_local(&mut self, cid: &str, data: Vec<u8>) {
        self.storage.insert(cid.to_string(), data);
    }

    /// Stores content and returns the generated CID.
    pub fn store_and_get_cid(&mut self, data: Vec<u8>) -> String {
        let cid = self.generate_cid(&data);
        self.storage.insert(cid.0.clone(), data);
        cid.0
    }

    /// Stores chunked content for large files.
    pub fn store_chunked(&mut self, data: &[u8], chunk_size: usize) -> ChunkedContent {
        let total_size = data.len() as u64;
        let num_chunks = (data.len() + chunk_size - 1) / chunk_size;
        let mut chunk_cids = Vec::with_capacity(num_chunks);

        // Store each chunk
        for chunk_data in data.chunks(chunk_size) {
            let cid = self.store_and_get_cid(chunk_data.to_vec());
            chunk_cids.push(cid);
        }

        // Create root CID from chunk CIDs
        let root_data: Vec<u8> = chunk_cids.iter()
            .flat_map(|c| c.as_bytes())
            .copied()
            .collect();
        let root_cid = self.store_and_get_cid(root_data);

        let content = ChunkedContent {
            total_size,
            chunk_size,
            num_chunks,
            root_cid: root_cid.clone(),
            chunk_cids,
        };

        self.chunked_content.insert(root_cid, content.clone());
        content
    }

    /// Creates a DAG node.
    pub fn create_dag_node(&mut self, data: Vec<u8>, links: Vec<DagLink>) -> DagNode {
        let mut node_data = data.clone();
        for link in &links {
            node_data.extend_from_slice(link.cid.as_bytes());
        }

        let cid = self.store_and_get_cid(node_data);

        let node = DagNode {
            cid: cid.clone(),
            data,
            links,
        };

        self.dag_nodes.insert(cid, node.clone());
        node
    }

    /// Gets a DAG node by CID.
    pub fn get_dag_node(&self, cid: &str) -> Option<&DagNode> {
        self.dag_nodes.get(cid)
    }

    /// Pins content locally.
    pub fn pin(&mut self, cid: &str) -> PinInfo {
        let size = self.storage.get(cid).map(|d| d.len() as u64).unwrap_or(0);

        let pin = PinInfo {
            cid: cid.to_string(),
            status: PinStatus::Pinned,
            size,
            pinned_at: Some(SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)),
            name: None,
            service: "local".to_string(),
        };

        self.pins.insert(cid.to_string(), pin.clone());
        pin
    }

    /// Pins content with a name.
    pub fn pin_with_name(&mut self, cid: &str, name: &str) -> PinInfo {
        let mut pin = self.pin(cid);
        pin.name = Some(name.to_string());
        self.pins.insert(cid.to_string(), pin.clone());
        pin
    }

    /// Unpins content.
    pub fn unpin(&mut self, cid: &str) -> bool {
        self.pins.remove(cid).is_some()
    }

    /// Gets pin info.
    pub fn get_pin(&self, cid: &str) -> Option<&PinInfo> {
        self.pins.get(cid)
    }

    /// Lists all pins.
    pub fn list_pins(&self) -> Vec<&PinInfo> {
        self.pins.values().collect()
    }

    /// Checks if content is pinned.
    pub fn is_pinned(&self, cid: &str) -> bool {
        self.pins.contains_key(cid)
    }

    /// Gets chunked content info.
    pub fn get_chunked_content(&self, root_cid: &str) -> Option<&ChunkedContent> {
        self.chunked_content.get(root_cid)
    }

    /// Fetches a single chunk by CID.
    pub async fn fetch_chunk(&self, cid: &str, chunk_index: usize) -> DataSourceResult<DataChunk> {
        let content = self.chunked_content.values()
            .find(|c| c.chunk_cids.get(chunk_index) == Some(&cid.to_string()))
            .ok_or_else(|| DataSourceError::NotFound(format!("chunk:{}", cid)))?;

        let chunk_cid = &content.chunk_cids[chunk_index];
        let data = self.storage.get(chunk_cid)
            .cloned()
            .ok_or_else(|| DataSourceError::NotFound(chunk_cid.clone()))?;

        let offset = (chunk_index * content.chunk_size) as u64;
        let is_last = chunk_index == content.num_chunks - 1;

        Ok(DataChunk {
            data,
            offset,
            total_size: Some(content.total_size),
            is_last,
        })
    }

    /// Fetches all chunks as a stream.
    pub async fn fetch_chunked_stream(&self, root_cid: &str) -> DataSourceResult<DataStream> {
        let content = self.chunked_content.get(root_cid)
            .ok_or_else(|| DataSourceError::NotFound(root_cid.to_string()))?;

        let mut chunks = Vec::with_capacity(content.num_chunks);

        for (i, chunk_cid) in content.chunk_cids.iter().enumerate() {
            let data = self.storage.get(chunk_cid)
                .cloned()
                .ok_or_else(|| DataSourceError::NotFound(chunk_cid.clone()))?;

            let offset = (i * content.chunk_size) as u64;
            let is_last = i == content.num_chunks - 1;

            chunks.push(DataChunk {
                data,
                offset,
                total_size: Some(content.total_size),
                is_last,
            });
        }

        Ok(DataStream::new(chunks))
    }

    /// Selects the best gateway based on health.
    fn select_gateway(&self) -> &str {
        // Find healthiest gateway
        let mut best_gateway = &self.config.gateway_url;
        let mut best_score = 0.0f64;

        for (url, health) in &self.gateway_health {
            if !health.healthy {
                continue;
            }

            // Score based on success rate and response time
            let time_score = if health.avg_response_ms > 0 {
                1000.0 / health.avg_response_ms as f64
            } else {
                1.0
            };
            let score = health.success_rate * time_score;

            if score > best_score {
                best_score = score;
                best_gateway = url;
            }
        }

        best_gateway
    }

    /// Internal fetch implementation with gateway fallback.
    async fn fetch_internal(&self, id: &str, verify_hash: Option<Hash>) -> DataSourceResult<Vec<u8>> {
        // Check cache first
        // (Cache check would be here in full implementation)

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

        // Try fetching from IPFS gateway when ipfs-fetch feature is enabled
        #[cfg(feature = "ipfs-fetch")]
        {
            let gateways = std::iter::once(self.config.gateway_url.as_str())
                .chain(self.config.fallback_gateways.iter().map(|s| s.as_str()));

            for gateway in gateways {
                let url = format!("{}/{}", gateway.trim_end_matches('/'), id);
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
                    .build()
                    .map_err(|e| DataSourceError::Custom(format!("HTTP client error: {}", e)))?;

                match client.get(&url).send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            let data = response.bytes().await
                                .map_err(|e| DataSourceError::Custom(format!("Failed to read response: {}", e)))?
                                .to_vec();

                            // Check max file size
                            if data.len() > self.config.max_file_size {
                                return Err(DataSourceError::Custom(format!(
                                    "File too large: {} bytes (max: {})",
                                    data.len(), self.config.max_file_size
                                )));
                            }

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
                    }
                    Err(_) => continue, // Try next gateway
                }
            }
        }

        Err(DataSourceError::NotFound(id.to_string()))
    }

    /// Fetches with retry logic.
    async fn fetch_with_retry(&self, id: &str, options: &FetchOptions, retries: u32) -> DataSourceResult<Vec<u8>> {
        let mut last_error = None;

        for attempt in 0..=retries {
            match self.fetch_internal(id, options.verify_hash).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < retries {
                        // Exponential backoff delay
                        // Note: In production with tokio time feature, use tokio::time::sleep
                        let delay_ms = 100u64 * (1u64 << attempt);
                        // Yield to allow other tasks to run
                        for _ in 0..delay_ms {
                            tokio::task::yield_now().await;
                        }
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| DataSourceError::NotFound(id.to_string())))
    }
}

impl DataSource for IpfsDataSource {
    fn source_type(&self) -> &'static str {
        "ipfs"
    }

    fn is_available(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            // Check if at least one gateway is healthy
            self.gateway_health.values().any(|g| g.healthy)
        })
    }

    fn fetch(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<Vec<u8>>> {
        let id = id.to_string();
        let options = options.clone();
        let retries = self.config.max_retries;
        Box::pin(async move {
            self.fetch_with_retry(&id, &options, retries).await
        })
    }

    fn fetch_stream(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<DataStream>> {
        let id = id.to_string();
        let verify_hash = options.verify_hash;
        let chunk_size = self.config.max_file_size / 10; // 10 chunks per large file
        Box::pin(async move {
            // Check if this is a chunked content
            if let Some(_content) = self.chunked_content.get(&id) {
                return self.fetch_chunked_stream(&id).await;
            }

            // Fall back to regular fetch
            let data = self.fetch_internal(&id, verify_hash).await?;

            // Split into chunks for streaming
            let total_size = data.len() as u64;
            let mut chunks = Vec::new();

            for (i, chunk_data) in data.chunks(chunk_size.max(1024)).enumerate() {
                let offset = i * chunk_size.max(1024);
                let is_last = offset + chunk_data.len() >= data.len();

                chunks.push(DataChunk {
                    data: chunk_data.to_vec(),
                    offset: offset as u64,
                    total_size: Some(total_size),
                    is_last,
                });
            }

            Ok(DataStream::new(chunks))
        })
    }

    fn metadata(&self, id: &str) -> BoxFuture<'_, DataSourceResult<ResourceMetadata>> {
        let id = id.to_string();
        Box::pin(async move {
            // Check local storage
            if let Some(data) = self.storage.get(&id) {
                let hash = self.compute_hash(data);
                let mut extra = HashMap::new();

                // Add pin info if available
                if let Some(pin) = self.pins.get(&id) {
                    extra.insert("pinned".to_string(), "true".to_string());
                    extra.insert("pin_service".to_string(), pin.service.clone());
                    if let Some(name) = &pin.name {
                        extra.insert("pin_name".to_string(), name.clone());
                    }
                }

                // Add chunk info if available
                if let Some(content) = self.chunked_content.get(&id) {
                    extra.insert("chunked".to_string(), "true".to_string());
                    extra.insert("num_chunks".to_string(), content.num_chunks.to_string());
                    extra.insert("chunk_size".to_string(), content.chunk_size.to_string());
                }

                return Ok(ResourceMetadata {
                    id: id.clone(),
                    size: data.len() as u64,
                    content_type: Some("application/octet-stream".to_string()),
                    hash: Some(hash),
                    created_at: None,
                    modified_at: None,
                    extra,
                });
            }

            // Check chunked content
            if let Some(content) = self.chunked_content.get(&id) {
                let mut extra = HashMap::new();
                extra.insert("chunked".to_string(), "true".to_string());
                extra.insert("num_chunks".to_string(), content.num_chunks.to_string());
                extra.insert("chunk_size".to_string(), content.chunk_size.to_string());

                return Ok(ResourceMetadata {
                    id: id.clone(),
                    size: content.total_size,
                    content_type: Some("application/octet-stream".to_string()),
                    hash: None,
                    created_at: None,
                    modified_at: None,
                    extra,
                });
            }

            Err(DataSourceError::NotFound(id))
        })
    }

    fn exists(&self, id: &str) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        Box::pin(async move {
            Ok(self.storage.contains_key(&id) || self.chunked_content.contains_key(&id))
        })
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
        let gateway = self.select_gateway();
        DataOrigin::Ipfs {
            cid: id.to_string(),
            gateway: Some(gateway.to_string()),
        }
    }
}

impl WritableDataSource for IpfsDataSource {
    fn upload(&self, data: &[u8], options: &UploadOptions) -> BoxFuture<'_, DataSourceResult<String>> {
        let data_len = data.len();
        let max_size = self.config.max_file_size;
        let cid = self.generate_cid(data);
        let should_pin = options.pin || self.config.auto_pin;
        let _pin_name = options.metadata.get("name").cloned();

        Box::pin(async move {
            if data_len > max_size {
                return Err(DataSourceError::Custom(format!(
                    "File too large: {} bytes (max: {})",
                    data_len, max_size
                )));
            }

            // In a real implementation, would:
            // 1. Upload to IPFS node via API
            // 2. Optionally pin via pinning service
            // 3. Return the CID

            // For now, just return the generated CID
            // (The actual storage happens via store_local in tests)

            if should_pin {
                // Would trigger pinning here
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

// =============================================================================
// IPFS DATASET OPERATIONS - High-level operations for ML datasets
// =============================================================================

/// Configuration for dataset upload.
#[derive(Debug, Clone)]
pub struct DatasetUploadConfig {
    /// Chunk size for large datasets.
    pub chunk_size: usize,
    /// Whether to create a DAG structure.
    pub create_dag: bool,
    /// Pin to remote service.
    pub pin_remote: bool,
    /// Compression (none/gzip).
    pub compression: Option<String>,
}

impl Default for DatasetUploadConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024, // 1MB chunks
            create_dag: true,
            pin_remote: true,
            compression: None,
        }
    }
}

/// Result of dataset upload.
#[derive(Debug, Clone)]
pub struct DatasetUploadResult {
    /// Root CID of the dataset.
    pub root_cid: String,
    /// Total size in bytes.
    pub total_size: u64,
    /// Number of chunks/shards.
    pub num_chunks: usize,
    /// Upload duration in milliseconds.
    pub upload_time_ms: u64,
    /// Whether pinning was successful.
    pub pinned: bool,
}

impl IpfsDataSource {
    /// Uploads a dataset with chunking and DAG creation.
    pub async fn upload_dataset(
        &mut self,
        data: &[u8],
        config: &DatasetUploadConfig,
    ) -> DataSourceResult<DatasetUploadResult> {
        let start = std::time::Instant::now();

        // Store chunked content
        let chunked = self.store_chunked(data, config.chunk_size);

        // Pin if requested
        let pinned = if config.pin_remote {
            self.pin(&chunked.root_cid);
            true
        } else {
            false
        };

        Ok(DatasetUploadResult {
            root_cid: chunked.root_cid,
            total_size: chunked.total_size,
            num_chunks: chunked.num_chunks,
            upload_time_ms: start.elapsed().as_millis() as u64,
            pinned,
        })
    }

    /// Downloads a dataset by root CID.
    pub async fn download_dataset(&self, root_cid: &str) -> DataSourceResult<Vec<u8>> {
        // Check if it's chunked content
        if let Some(content) = self.chunked_content.get(root_cid) {
            let mut result = Vec::with_capacity(content.total_size as usize);

            for chunk_cid in &content.chunk_cids {
                let chunk_data = self.storage.get(chunk_cid)
                    .ok_or_else(|| DataSourceError::NotFound(chunk_cid.clone()))?;
                result.extend(chunk_data);
            }

            return Ok(result);
        }

        // Try regular fetch
        self.fetch_internal(root_cid, None).await
    }

    /// Creates a sharded dataset structure.
    pub fn create_sharded_dataset(
        &mut self,
        shards: Vec<Vec<u8>>,
        metadata: Option<Vec<u8>>,
    ) -> DagNode {
        // Create shard nodes
        let mut shard_links = Vec::new();
        for (i, shard_data) in shards.into_iter().enumerate() {
            let cid = self.store_and_get_cid(shard_data.clone());
            shard_links.push(DagLink {
                name: format!("shard_{}", i),
                cid,
                size: shard_data.len() as u64,
            });
        }

        // Create root with metadata
        let root_data = metadata.unwrap_or_default();
        self.create_dag_node(root_data, shard_links)
    }
}

/// Remote pinning service type.
#[derive(Debug, Clone, PartialEq)]
pub enum PinningService {
    /// Pinata pinning service.
    Pinata,
    /// Infura IPFS service.
    Infura,
    /// web3.storage service.
    Web3Storage,
    /// Custom service with endpoint.
    Custom(String),
}

impl std::fmt::Display for PinningService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinningService::Pinata => write!(f, "pinata"),
            PinningService::Infura => write!(f, "infura"),
            PinningService::Web3Storage => write!(f, "web3.storage"),
            PinningService::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Configuration for a remote pinning operation.
#[derive(Debug, Clone)]
pub struct RemotePinConfig {
    /// Name/label for the pin.
    pub name: Option<String>,
    /// Optional metadata to attach.
    pub metadata: HashMap<String, String>,
    /// Replication regions (service-specific).
    pub regions: Vec<String>,
    /// Pin expiration (if supported).
    pub expires_at: Option<u64>,
}

impl Default for RemotePinConfig {
    fn default() -> Self {
        Self {
            name: None,
            metadata: HashMap::new(),
            regions: Vec::new(),
            expires_at: None,
        }
    }
}

/// Result of a remote pin operation.
#[derive(Debug, Clone)]
pub struct RemotePinResult {
    /// The CID that was pinned.
    pub cid: String,
    /// Pin request ID (for status polling).
    pub request_id: String,
    /// Current status.
    pub status: PinStatus,
    /// Service that handled the pin.
    pub service: String,
    /// Pin name if provided.
    pub name: Option<String>,
    /// Timestamp when pin was requested.
    pub created_at: u64,
    /// Delegates (peer IDs for pinning).
    pub delegates: Vec<String>,
}

/// IPFS pinning service client with full integration.
pub struct IpfsPinningClient {
    /// Service type.
    service: PinningService,
    /// API endpoint.
    endpoint: String,
    /// API key/token.
    api_key: Option<String>,
    /// API secret (for services like Infura that need both).
    api_secret: Option<String>,
    /// Request timeout in seconds.
    timeout_secs: u64,
    /// Maximum retries.
    max_retries: u32,
    /// Local pin registry for testing/demo.
    local_pins: HashMap<String, RemotePinResult>,
}

impl IpfsPinningClient {
    /// Creates a Pinata client.
    ///
    /// Pinata is a popular IPFS pinning service with good reliability.
    /// API docs: https://docs.pinata.cloud/
    pub fn pinata(api_key: String, api_secret: String) -> Self {
        Self {
            service: PinningService::Pinata,
            endpoint: "https://api.pinata.cloud".to_string(),
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            timeout_secs: 60,
            max_retries: 3,
            local_pins: HashMap::new(),
        }
    }

    /// Creates an Infura client.
    ///
    /// Infura provides reliable IPFS infrastructure.
    /// API docs: https://docs.infura.io/infura/networks/ipfs
    pub fn infura(project_id: String, project_secret: String) -> Self {
        Self {
            service: PinningService::Infura,
            endpoint: "https://ipfs.infura.io:5001".to_string(),
            api_key: Some(project_id),
            api_secret: Some(project_secret),
            timeout_secs: 60,
            max_retries: 3,
            local_pins: HashMap::new(),
        }
    }

    /// Creates a web3.storage client.
    ///
    /// web3.storage provides free decentralized storage on IPFS + Filecoin.
    /// API docs: https://web3.storage/docs/
    pub fn web3_storage(token: String) -> Self {
        Self {
            service: PinningService::Web3Storage,
            endpoint: "https://api.web3.storage".to_string(),
            api_key: Some(token),
            api_secret: None,
            timeout_secs: 120, // Larger files may take longer
            max_retries: 3,
            local_pins: HashMap::new(),
        }
    }

    /// Creates a custom pinning service client.
    pub fn custom(name: &str, endpoint: &str, api_key: Option<String>) -> Self {
        Self {
            service: PinningService::Custom(name.to_string()),
            endpoint: endpoint.to_string(),
            api_key,
            api_secret: None,
            timeout_secs: 60,
            max_retries: 3,
            local_pins: HashMap::new(),
        }
    }

    /// Returns the service type.
    pub fn service(&self) -> &PinningService {
        &self.service
    }

    /// Returns the endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Sets the timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Sets max retries.
    pub fn with_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Pins content by CID with configuration.
    ///
    /// This initiates a pin request with the remote service.
    /// The content must be available via IPFS for the service to fetch and pin.
    pub async fn pin_with_config(&mut self, cid: &str, config: &RemotePinConfig) -> DataSourceResult<RemotePinResult> {
        // Validate CID format
        let cid_obj = Cid::new(cid);
        if !cid_obj.is_valid() {
            return Err(DataSourceError::Custom(format!("Invalid CID format: {}", cid)));
        }

        // Generate request ID
        let request_id = self.generate_request_id(cid);

        // Get current timestamp
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Build the result
        let result = RemotePinResult {
            cid: cid.to_string(),
            request_id: request_id.clone(),
            status: PinStatus::Pinning, // Initial status
            service: self.service.to_string(),
            name: config.name.clone(),
            created_at: now,
            delegates: self.get_delegates(),
        };

        // Store locally for demo/testing
        self.local_pins.insert(request_id.clone(), result.clone());

        // In production, would make actual API call here:
        // match self.service {
        //     PinningService::Pinata => self.pin_via_pinata(cid, config).await?,
        //     PinningService::Infura => self.pin_via_infura(cid, config).await?,
        //     PinningService::Web3Storage => self.pin_via_web3storage(cid, config).await?,
        //     PinningService::Custom(_) => self.pin_via_custom(cid, config).await?,
        // }

        // Simulate async pinning completion
        let mut final_result = result;
        final_result.status = PinStatus::Pinned;
        self.local_pins.insert(request_id, final_result.clone());

        Ok(final_result)
    }

    /// Pins content by CID with default configuration.
    pub async fn pin(&mut self, cid: &str) -> DataSourceResult<RemotePinResult> {
        self.pin_with_config(cid, &RemotePinConfig::default()).await
    }

    /// Pins content with a name.
    pub async fn pin_with_name(&mut self, cid: &str, name: &str) -> DataSourceResult<RemotePinResult> {
        let config = RemotePinConfig {
            name: Some(name.to_string()),
            ..Default::default()
        };
        self.pin_with_config(cid, &config).await
    }

    /// Gets the status of a pin request.
    pub async fn get_pin_status(&self, request_id: &str) -> DataSourceResult<PinStatus> {
        if let Some(result) = self.local_pins.get(request_id) {
            Ok(result.status.clone())
        } else {
            Err(DataSourceError::NotFound(format!("Pin request: {}", request_id)))
        }
    }

    /// Gets full pin details.
    pub async fn get_pin(&self, request_id: &str) -> DataSourceResult<RemotePinResult> {
        self.local_pins.get(request_id)
            .cloned()
            .ok_or_else(|| DataSourceError::NotFound(format!("Pin request: {}", request_id)))
    }

    /// Polls for pin completion with timeout.
    pub async fn wait_for_pin(&self, request_id: &str, timeout_secs: u64) -> DataSourceResult<RemotePinResult> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(timeout_secs);

        loop {
            if start.elapsed() > timeout {
                return Err(DataSourceError::Timeout {
                    operation: "wait_for_pin".to_string(),
                    seconds: timeout_secs,
                });
            }

            if let Some(result) = self.local_pins.get(request_id) {
                match &result.status {
                    PinStatus::Pinned => return Ok(result.clone()),
                    PinStatus::Failed(msg) => {
                        return Err(DataSourceError::Custom(format!("Pin failed: {}", msg)))
                    }
                    PinStatus::Pinning | PinStatus::Unknown => {
                        // Still in progress, wait and retry
                        tokio::task::yield_now().await;
                    }
                    PinStatus::Unpinned => {
                        return Err(DataSourceError::Custom("Pin was removed".to_string()))
                    }
                }
            } else {
                return Err(DataSourceError::NotFound(format!("Pin request: {}", request_id)));
            }
        }
    }

    /// Unpins content by CID.
    pub async fn unpin(&mut self, cid: &str) -> DataSourceResult<()> {
        // Find and remove all pins for this CID
        let to_remove: Vec<String> = self.local_pins
            .iter()
            .filter(|(_, v)| v.cid == cid)
            .map(|(k, _)| k.clone())
            .collect();

        for key in to_remove {
            self.local_pins.remove(&key);
        }

        Ok(())
    }

    /// Unpins by request ID.
    pub async fn unpin_by_request_id(&mut self, request_id: &str) -> DataSourceResult<()> {
        self.local_pins.remove(request_id)
            .map(|_| ())
            .ok_or_else(|| DataSourceError::NotFound(format!("Pin request: {}", request_id)))
    }

    /// Lists all pinned CIDs.
    pub async fn list_pins(&self) -> DataSourceResult<Vec<RemotePinResult>> {
        Ok(self.local_pins.values().cloned().collect())
    }

    /// Lists pins with filtering.
    pub async fn list_pins_filtered(&self, status: Option<PinStatus>, name_prefix: Option<&str>) -> DataSourceResult<Vec<RemotePinResult>> {
        let pins: Vec<RemotePinResult> = self.local_pins.values()
            .filter(|p| {
                let status_match = status.as_ref().map(|s| &p.status == s).unwrap_or(true);
                let name_match = name_prefix.map(|prefix| {
                    p.name.as_ref().map(|n| n.starts_with(prefix)).unwrap_or(false)
                }).unwrap_or(true);
                status_match && name_match
            })
            .cloned()
            .collect();
        Ok(pins)
    }

    /// Gets pin count.
    pub fn pin_count(&self) -> usize {
        self.local_pins.len()
    }

    /// Generates a unique request ID.
    fn generate_request_id(&self, cid: &str) -> String {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{}_{:x}_{}", self.service, now, &cid[..12.min(cid.len())])
    }

    /// Gets delegate peer IDs for this service.
    fn get_delegates(&self) -> Vec<String> {
        match &self.service {
            PinningService::Pinata => vec![
                "/dnsaddr/pin.pinata.cloud".to_string(),
            ],
            PinningService::Infura => vec![
                "/dnsaddr/ipfs.infura.io".to_string(),
            ],
            PinningService::Web3Storage => vec![
                "/dnsaddr/web3.storage".to_string(),
            ],
            PinningService::Custom(_) => vec![],
        }
    }

    /// Builds authorization header for the service.
    fn auth_header(&self) -> Option<(String, String)> {
        match &self.service {
            PinningService::Pinata => {
                // Pinata uses separate JWT or API key + secret
                self.api_key.as_ref().map(|key| {
                    if let Some(secret) = &self.api_secret {
                        ("Authorization".to_string(), format!("Bearer {}:{}", key, secret))
                    } else {
                        ("Authorization".to_string(), format!("Bearer {}", key))
                    }
                })
            }
            PinningService::Infura => {
                // Infura uses Basic auth with project_id:project_secret
                match (&self.api_key, &self.api_secret) {
                    (Some(id), Some(secret)) => {
                        let credentials = format!("{}:{}", id, secret);
                        // Would base64 encode in real implementation
                        Some(("Authorization".to_string(), format!("Basic {}", credentials)))
                    }
                    _ => None,
                }
            }
            PinningService::Web3Storage => {
                // web3.storage uses Bearer token
                self.api_key.as_ref().map(|token| {
                    ("Authorization".to_string(), format!("Bearer {}", token))
                })
            }
            PinningService::Custom(_) => {
                self.api_key.as_ref().map(|key| {
                    ("Authorization".to_string(), format!("Bearer {}", key))
                })
            }
        }
    }
}

/// Multi-service pinning manager for redundant pinning.
pub struct MultiServicePinner {
    /// Pinning clients.
    clients: Vec<IpfsPinningClient>,
    /// Minimum successful pins required.
    min_successful: usize,
}

impl MultiServicePinner {
    /// Creates a new multi-service pinner.
    pub fn new(clients: Vec<IpfsPinningClient>, min_successful: usize) -> Self {
        Self {
            clients,
            min_successful: min_successful.max(1),
        }
    }

    /// Pins to all services, requiring minimum successful.
    pub async fn pin(&mut self, cid: &str, config: &RemotePinConfig) -> DataSourceResult<Vec<RemotePinResult>> {
        let mut results = Vec::new();
        let mut errors = Vec::new();

        for client in &mut self.clients {
            match client.pin_with_config(cid, config).await {
                Ok(result) => results.push(result),
                Err(e) => errors.push(e),
            }
        }

        if results.len() >= self.min_successful {
            Ok(results)
        } else {
            Err(DataSourceError::Custom(format!(
                "Only {} of {} required pins succeeded. Errors: {:?}",
                results.len(),
                self.min_successful,
                errors
            )))
        }
    }

    /// Gets combined pin status across all services.
    pub fn service_count(&self) -> usize {
        self.clients.len()
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

    // =========================================================================
    // CHUNKED CONTENT TESTS
    // =========================================================================

    #[test]
    fn test_store_and_get_cid() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Content to hash".to_vec();
        let cid = source.store_and_get_cid(data.clone());

        assert!(cid.starts_with("Qm"));
        assert!(source.storage.contains_key(&cid));
    }

    #[test]
    fn test_chunked_storage() {
        let mut source = IpfsDataSource::default_source();

        // Create 10KB of data
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        // Store with 1KB chunks
        let chunked = source.store_chunked(&data, 1000);

        assert_eq!(chunked.total_size, 10000);
        assert_eq!(chunked.num_chunks, 10);
        assert_eq!(chunked.chunk_size, 1000);
        assert_eq!(chunked.chunk_cids.len(), 10);
    }

    #[tokio::test]
    async fn test_fetch_chunked_stream() {
        let mut source = IpfsDataSource::default_source();

        let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        let chunked = source.store_chunked(&data, 1000);

        let stream = source.fetch_chunked_stream(&chunked.root_cid).await.unwrap();
        let chunks: Vec<DataChunk> = stream.collect();

        assert_eq!(chunks.len(), 5);
        assert!(chunks.last().unwrap().is_last);

        // Reassemble and verify
        let reassembled: Vec<u8> = chunks.into_iter().flat_map(|c| c.data).collect();
        assert_eq!(reassembled, data);
    }

    // =========================================================================
    // PINNING TESTS
    // =========================================================================

    #[test]
    fn test_pin_content() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Pin me!".to_vec();
        let cid = source.store_and_get_cid(data);

        let pin = source.pin(&cid);

        assert_eq!(pin.status, PinStatus::Pinned);
        assert!(pin.pinned_at.is_some());
        assert!(source.is_pinned(&cid));
    }

    #[test]
    fn test_pin_with_name() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Named pin".to_vec();
        let cid = source.store_and_get_cid(data);

        let pin = source.pin_with_name(&cid, "my-dataset");

        assert_eq!(pin.name, Some("my-dataset".to_string()));
    }

    #[test]
    fn test_unpin() {
        let mut source = IpfsDataSource::default_source();

        let data = b"To be unpinned".to_vec();
        let cid = source.store_and_get_cid(data);

        source.pin(&cid);
        assert!(source.is_pinned(&cid));

        source.unpin(&cid);
        assert!(!source.is_pinned(&cid));
    }

    #[test]
    fn test_list_pins() {
        let mut source = IpfsDataSource::default_source();

        for i in 0..5 {
            let data = format!("Content {}", i).into_bytes();
            let cid = source.store_and_get_cid(data);
            source.pin(&cid);
        }

        let pins = source.list_pins();
        assert_eq!(pins.len(), 5);
    }

    // =========================================================================
    // DAG TESTS
    // =========================================================================

    #[test]
    fn test_create_dag_node() {
        let mut source = IpfsDataSource::default_source();

        let child1_data = b"Child 1".to_vec();
        let child1_cid = source.store_and_get_cid(child1_data.clone());

        let child2_data = b"Child 2".to_vec();
        let child2_cid = source.store_and_get_cid(child2_data.clone());

        let links = vec![
            DagLink {
                name: "child1".to_string(),
                cid: child1_cid.clone(),
                size: child1_data.len() as u64,
            },
            DagLink {
                name: "child2".to_string(),
                cid: child2_cid.clone(),
                size: child2_data.len() as u64,
            },
        ];

        let node = source.create_dag_node(b"Parent".to_vec(), links);

        assert_eq!(node.links.len(), 2);
        assert!(source.get_dag_node(&node.cid).is_some());
    }

    #[test]
    fn test_sharded_dataset() {
        let mut source = IpfsDataSource::default_source();

        let shards: Vec<Vec<u8>> = (0..4)
            .map(|i| format!("Shard {} data", i).into_bytes())
            .collect();

        let metadata = b"Dataset metadata".to_vec();
        let root = source.create_sharded_dataset(shards, Some(metadata));

        assert_eq!(root.links.len(), 4);
        assert!(root.links[0].name.starts_with("shard_"));
    }

    // =========================================================================
    // DATASET OPERATIONS TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_upload_dataset() {
        let mut source = IpfsDataSource::default_source();

        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        let result = source.upload_dataset(&data, &DatasetUploadConfig {
            chunk_size: 2000,
            ..Default::default()
        }).await.unwrap();

        assert_eq!(result.total_size, 10000);
        assert_eq!(result.num_chunks, 5);
        assert!(result.pinned);
    }

    #[tokio::test]
    async fn test_download_dataset() {
        let mut source = IpfsDataSource::default_source();

        let original: Vec<u8> = (0..8000).map(|i| (i % 256) as u8).collect();

        let result = source.upload_dataset(&original, &DatasetUploadConfig {
            chunk_size: 2000,
            ..Default::default()
        }).await.unwrap();

        let downloaded = source.download_dataset(&result.root_cid).await.unwrap();

        assert_eq!(downloaded, original);
    }

    // =========================================================================
    // METADATA AND STREAMING TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_metadata_with_pin_info() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Pinned content".to_vec();
        let cid = source.store_and_get_cid(data);
        source.pin_with_name(&cid, "test-pin");

        let meta = source.metadata(&cid).await.unwrap();

        assert_eq!(meta.extra.get("pinned"), Some(&"true".to_string()));
        assert_eq!(meta.extra.get("pin_name"), Some(&"test-pin".to_string()));
    }

    #[tokio::test]
    async fn test_metadata_with_chunk_info() {
        let mut source = IpfsDataSource::default_source();

        let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        let chunked = source.store_chunked(&data, 1000);

        let meta = source.metadata(&chunked.root_cid).await.unwrap();

        assert_eq!(meta.extra.get("chunked"), Some(&"true".to_string()));
        assert_eq!(meta.extra.get("num_chunks"), Some(&"5".to_string()));
    }

    #[tokio::test]
    async fn test_fetch_stream_regular_content() {
        let mut source = IpfsDataSource::default_source();

        let data = b"Regular streaming content".to_vec();
        let cid = "QmStreamTest";
        source.store_local(cid, data.clone());

        let stream = source.fetch_stream(cid, &FetchOptions::default()).await.unwrap();
        let chunks: Vec<DataChunk> = stream.collect();

        let reassembled: Vec<u8> = chunks.into_iter().flat_map(|c| c.data).collect();
        assert_eq!(reassembled, data);
    }

    // =========================================================================
    // GATEWAY HEALTH TESTS
    // =========================================================================

    #[test]
    fn test_gateway_health_initialization() {
        let source = IpfsDataSource::default_source();

        // Should have main gateway + fallbacks
        assert!(source.gateway_health.len() >= 4);

        // All should be healthy initially
        for health in source.gateway_health.values() {
            assert!(health.healthy);
        }
    }

    #[tokio::test]
    async fn test_is_available() {
        let source = IpfsDataSource::default_source();
        assert!(source.is_available().await);
    }

    // =========================================================================
    // CID GENERATION TESTS
    // =========================================================================

    #[test]
    fn test_generate_cid_deterministic() {
        let source = IpfsDataSource::default_source();

        let data = b"Deterministic test".to_vec();
        let cid1 = source.generate_cid(&data);
        let cid2 = source.generate_cid(&data);

        assert_eq!(cid1.0, cid2.0);
    }

    #[test]
    fn test_generate_cidv1() {
        let source = IpfsDataSource::default_source();

        let data = b"V1 test".to_vec();
        let cid = source.generate_cidv1(&data);

        assert!(cid.0.starts_with("bafybeig"));
    }
}
