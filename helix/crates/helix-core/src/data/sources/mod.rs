//! Data Source Abstractions for HELIX.
//!
//! Provides a unified interface for fetching training data from various sources:
//! - IPFS (InterPlanetary File System)
//! - Filecoin (distributed storage network)
//! - S3-compatible storage (AWS S3, MinIO, etc.)
//! - Arweave (permanent storage)
//!
//! All sources implement the `DataSource` trait for consistent data fetching
//! and verification.

pub mod filecoin;
pub mod ipfs;
pub mod s3;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use super::merkle::Hash;
use super::provenance::DataOrigin;

/// Re-export source implementations
pub use filecoin::{FilecoinClient, FilecoinConfig, FilecoinDataSource};
pub use ipfs::{IpfsDataSource, IpfsSourceConfig};
pub use s3::{S3Config, S3DataSource};

/// Result type for data source operations.
pub type DataSourceResult<T> = Result<T, DataSourceError>;

/// Type alias for boxed futures returned by DataSource methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Errors from data source operations.
#[derive(Debug, Clone)]
pub enum DataSourceError {
    /// Connection failed.
    ConnectionFailed(String),
    /// Resource not found.
    NotFound(String),
    /// Authentication failed.
    AuthenticationFailed(String),
    /// Permission denied.
    PermissionDenied(String),
    /// Data integrity error.
    IntegrityError { expected: Hash, actual: Hash },
    /// Timeout.
    Timeout { operation: String, seconds: u64 },
    /// Rate limited.
    RateLimited { retry_after: Option<u64> },
    /// Invalid configuration.
    InvalidConfig(String),
    /// Network error.
    NetworkError(String),
    /// I/O error.
    IoError(String),
    /// Unsupported operation.
    UnsupportedOperation(String),
    /// Custom error.
    Custom(String),
}

impl std::fmt::Display for DataSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSourceError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            DataSourceError::NotFound(id) => write!(f, "Not found: {}", id),
            DataSourceError::AuthenticationFailed(msg) => write!(f, "Auth failed: {}", msg),
            DataSourceError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            DataSourceError::IntegrityError { expected, actual } => {
                write!(f, "Integrity error: expected {}, got {}", expected, actual)
            }
            DataSourceError::Timeout { operation, seconds } => {
                write!(f, "Timeout after {}s: {}", seconds, operation)
            }
            DataSourceError::RateLimited { retry_after } => {
                write!(f, "Rate limited, retry after: {:?}s", retry_after)
            }
            DataSourceError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            DataSourceError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            DataSourceError::IoError(msg) => write!(f, "I/O error: {}", msg),
            DataSourceError::UnsupportedOperation(msg) => write!(f, "Unsupported: {}", msg),
            DataSourceError::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for DataSourceError {}

impl From<std::io::Error> for DataSourceError {
    fn from(e: std::io::Error) -> Self {
        DataSourceError::IoError(e.to_string())
    }
}

/// Metadata about a data resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetadata {
    /// Resource identifier (CID, key, etc.).
    pub id: String,
    /// Size in bytes.
    pub size: u64,
    /// Content type (MIME type).
    pub content_type: Option<String>,
    /// Hash of the content.
    pub hash: Option<Hash>,
    /// Creation timestamp.
    pub created_at: Option<u64>,
    /// Last modification timestamp.
    pub modified_at: Option<u64>,
    /// Additional metadata.
    pub extra: HashMap<String, String>,
}

impl ResourceMetadata {
    /// Creates a new resource metadata.
    pub fn new(id: String, size: u64) -> Self {
        Self {
            id,
            size,
            content_type: None,
            hash: None,
            created_at: None,
            modified_at: None,
            extra: HashMap::new(),
        }
    }
}

/// Options for fetching data.
#[derive(Debug, Clone, Default)]
pub struct FetchOptions {
    /// Verify content hash after fetch.
    pub verify_hash: Option<Hash>,
    /// Timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// Range start (for partial fetches).
    pub range_start: Option<u64>,
    /// Range end (for partial fetches).
    pub range_end: Option<u64>,
    /// Whether to cache the result.
    pub cache: bool,
}

impl FetchOptions {
    /// Creates options with hash verification.
    pub fn with_verification(hash: Hash) -> Self {
        Self {
            verify_hash: Some(hash),
            ..Default::default()
        }
    }

    /// Creates options for streaming.
    pub fn streaming() -> Self {
        Self {
            cache: false,
            ..Default::default()
        }
    }
}

/// A chunk of data for streaming.
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// Chunk data.
    pub data: Vec<u8>,
    /// Offset in the full data.
    pub offset: u64,
    /// Total size (if known).
    pub total_size: Option<u64>,
    /// Whether this is the last chunk.
    pub is_last: bool,
}

/// A stream of data chunks.
pub struct DataStream {
    /// Internal receiver for chunks.
    chunks: Vec<DataChunk>,
    /// Current position.
    position: usize,
}

impl DataStream {
    /// Creates a new data stream.
    pub fn new(chunks: Vec<DataChunk>) -> Self {
        Self {
            chunks,
            position: 0,
        }
    }

    /// Creates an empty stream.
    pub fn empty() -> Self {
        Self {
            chunks: Vec::new(),
            position: 0,
        }
    }

    /// Creates a stream from a single chunk.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let chunk = DataChunk {
            data,
            offset: 0,
            total_size: None,
            is_last: true,
        };
        Self {
            chunks: vec![chunk],
            position: 0,
        }
    }
}

impl Iterator for DataStream {
    type Item = DataChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.chunks.len() {
            let chunk = self.chunks[self.position].clone();
            self.position += 1;
            Some(chunk)
        } else {
            None
        }
    }
}

/// Trait for data sources.
///
/// Provides a unified interface for fetching and verifying data from
/// various storage backends.
///
/// Methods return boxed futures for async operation without requiring
/// the async_trait macro.
pub trait DataSource: Send + Sync {
    /// Returns the source type identifier.
    fn source_type(&self) -> &'static str;

    /// Checks if the source is available/connected.
    fn is_available(&self) -> BoxFuture<'_, bool>;

    /// Fetches data by its identifier.
    fn fetch(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<Vec<u8>>>;

    /// Fetches data as a stream (for large datasets).
    fn fetch_stream(
        &self,
        id: &str,
        options: &FetchOptions,
    ) -> BoxFuture<'_, DataSourceResult<DataStream>>;

    /// Gets metadata about a resource.
    fn metadata(&self, id: &str) -> BoxFuture<'_, DataSourceResult<ResourceMetadata>>;

    /// Checks if a resource exists.
    fn exists(&self, id: &str) -> BoxFuture<'_, DataSourceResult<bool>>;

    /// Verifies the integrity of a resource.
    fn verify(&self, id: &str, expected_hash: &Hash) -> BoxFuture<'_, DataSourceResult<bool>>;

    /// Returns the data origin for provenance tracking.
    fn to_origin(&self, id: &str) -> DataOrigin;
}

/// Upload options for storing data.
#[derive(Debug, Clone, Default)]
pub struct UploadOptions {
    /// Content type (MIME type).
    pub content_type: Option<String>,
    /// Pin/persist the data.
    pub pin: bool,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Trait for data sources that support writing.
pub trait WritableDataSource: DataSource {
    /// Uploads data and returns its identifier.
    fn upload(&self, data: &[u8], options: &UploadOptions) -> BoxFuture<'_, DataSourceResult<String>>;

    /// Uploads from a stream.
    fn upload_stream(
        &self,
        stream: DataStream,
        options: &UploadOptions,
    ) -> BoxFuture<'_, DataSourceResult<String>>;

    /// Deletes a resource.
    fn delete(&self, id: &str) -> BoxFuture<'_, DataSourceResult<()>>;
}

/// Configuration for data source connection pooling.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections.
    pub max_connections: usize,
    /// Connection timeout in seconds.
    pub connect_timeout_secs: u64,
    /// Idle timeout in seconds.
    pub idle_timeout_secs: u64,
    /// Maximum retries.
    pub max_retries: u32,
    /// Retry backoff base in milliseconds.
    pub retry_backoff_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            connect_timeout_secs: 30,
            idle_timeout_secs: 300,
            max_retries: 3,
            retry_backoff_ms: 100,
        }
    }
}

/// A multi-source data fetcher that can fetch from multiple sources.
pub struct MultiSourceFetcher {
    /// Available sources.
    sources: Vec<Box<dyn DataSource>>,
    /// Source priority (index into sources).
    priority: Vec<usize>,
    /// Fallback behavior.
    fallback: FallbackBehavior,
}

/// Fallback behavior when a source fails.
#[derive(Debug, Clone, Copy)]
pub enum FallbackBehavior {
    /// Try next source in priority order.
    TryNext,
    /// Try all sources in parallel.
    ParallelRace,
    /// Fail immediately.
    FailFast,
}

impl MultiSourceFetcher {
    /// Creates a new multi-source fetcher.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            priority: Vec::new(),
            fallback: FallbackBehavior::TryNext,
        }
    }

    /// Adds a source.
    pub fn add_source(mut self, source: Box<dyn DataSource>) -> Self {
        self.priority.push(self.sources.len());
        self.sources.push(source);
        self
    }

    /// Sets fallback behavior.
    pub fn with_fallback(mut self, behavior: FallbackBehavior) -> Self {
        self.fallback = behavior;
        self
    }

    /// Fetches data, trying sources in priority order.
    pub async fn fetch(&self, id: &str, options: &FetchOptions) -> DataSourceResult<Vec<u8>> {
        match self.fallback {
            FallbackBehavior::TryNext => self.fetch_sequential(id, options).await,
            FallbackBehavior::ParallelRace => self.fetch_parallel(id, options).await,
            FallbackBehavior::FailFast => {
                if self.sources.is_empty() {
                    return Err(DataSourceError::Custom("No sources configured".into()));
                }
                self.sources[self.priority[0]].fetch(id, options).await
            }
        }
    }

    async fn fetch_sequential(
        &self,
        id: &str,
        options: &FetchOptions,
    ) -> DataSourceResult<Vec<u8>> {
        let mut last_error = None;

        for &idx in &self.priority {
            let source = &self.sources[idx];
            match source.fetch(id, options).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| DataSourceError::Custom("No sources configured".into())))
    }

    async fn fetch_parallel(&self, id: &str, options: &FetchOptions) -> DataSourceResult<Vec<u8>> {
        if self.sources.is_empty() {
            return Err(DataSourceError::Custom("No sources configured".into()));
        }

        // In a real implementation, would use tokio::select! to race futures
        // For now, just try the first source
        self.sources[0].fetch(id, options).await
    }
}

impl Default for MultiSourceFetcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for data source results.
pub struct DataCache {
    /// Cached data.
    cache: HashMap<String, CacheEntry>,
    /// Maximum cache size in bytes.
    max_size: usize,
    /// Current cache size.
    current_size: usize,
    /// Cache statistics.
    stats: CacheStats,
}

/// A cache entry.
#[derive(Clone)]
struct CacheEntry {
    /// Cached data.
    data: Vec<u8>,
    /// Content hash.
    hash: Hash,
    /// Last access time.
    accessed_at: u64,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of hits.
    pub hits: u64,
    /// Number of misses.
    pub misses: u64,
    /// Number of evictions.
    pub evictions: u64,
    /// Total bytes read from cache.
    pub bytes_read: u64,
}

impl DataCache {
    /// Creates a new cache with maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            current_size: 0,
            stats: CacheStats::default(),
        }
    }

    /// Gets data from cache.
    pub fn get(&mut self, key: &str) -> Option<(Vec<u8>, Hash)> {
        if let Some(entry) = self.cache.get_mut(key) {
            entry.accessed_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            self.stats.hits += 1;
            self.stats.bytes_read += entry.data.len() as u64;
            Some((entry.data.clone(), entry.hash))
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Puts data in cache.
    pub fn put(&mut self, key: String, data: Vec<u8>, hash: Hash) {
        let size = data.len();

        // Evict if necessary
        while self.current_size + size > self.max_size && !self.cache.is_empty() {
            self.evict_oldest();
        }

        // Don't cache if item is larger than max cache size
        if size > self.max_size {
            return;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.cache.insert(
            key,
            CacheEntry {
                data,
                hash,
                accessed_at: now,
            },
        );
        self.current_size += size;
    }

    /// Evicts the oldest entry.
    fn evict_oldest(&mut self) {
        let oldest = self
            .cache
            .iter()
            .min_by_key(|(_, e)| e.accessed_at)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest {
            if let Some(entry) = self.cache.remove(&key) {
                self.current_size -= entry.data.len();
                self.stats.evictions += 1;
            }
        }
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_size = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_metadata() {
        let meta = ResourceMetadata::new("test-id".to_string(), 1024);
        assert_eq!(meta.id, "test-id");
        assert_eq!(meta.size, 1024);
    }

    #[test]
    fn test_fetch_options() {
        let hash = Hash::from_slice(b"test");
        let opts = FetchOptions::with_verification(hash);
        assert_eq!(opts.verify_hash, Some(hash));
    }

    #[test]
    fn test_data_stream() {
        let data = vec![1, 2, 3, 4, 5];
        let stream = DataStream::from_bytes(data.clone());

        let chunks: Vec<_> = stream.collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].data, data);
        assert!(chunks[0].is_last);
    }

    #[test]
    fn test_cache() {
        let mut cache = DataCache::new(1000);

        let data = vec![1, 2, 3, 4, 5];
        let hash = Hash::from_slice(&data);

        cache.put("test".to_string(), data.clone(), hash);

        let (cached_data, cached_hash) = cache.get("test").unwrap();
        assert_eq!(cached_data, data);
        assert_eq!(cached_hash, hash);

        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = DataCache::new(100);

        // Fill cache
        for i in 0..5 {
            let data = vec![i as u8; 30];
            let hash = Hash::from_slice(&data);
            cache.put(format!("item{}", i), data, hash);
        }

        // Cache should have evicted some items
        assert!(cache.current_size <= 100);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = DataCache::new(1000);
        assert!(cache.get("nonexistent").is_none());
        assert_eq!(cache.stats().misses, 1);
    }
}
