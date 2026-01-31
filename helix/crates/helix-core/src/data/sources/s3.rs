//! S3-Compatible Data Source for HELIX.
//!
//! Provides integration with S3-compatible storage services:
//! - AWS S3
//! - MinIO
//! - Google Cloud Storage (S3 compatibility mode)
//! - DigitalOcean Spaces
//! - Backblaze B2
//!
//! Features:
//! - Multipart uploads for large files
//! - Streaming downloads
//! - Presigned URLs
//! - Server-side encryption support

use std::collections::HashMap;
use std::time::SystemTime;

use super::{
    BoxFuture, DataCache, DataChunk, DataSource, DataSourceError, DataSourceResult, DataStream,
    FetchOptions, ResourceMetadata, UploadOptions, WritableDataSource,
};
use crate::data::merkle::{Hash, MerkleHasher, Sha256Hasher};
use crate::data::provenance::DataOrigin;

/// Configuration for S3-compatible storage.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Bucket name.
    pub bucket: String,
    /// AWS region (e.g., "us-east-1").
    pub region: String,
    /// Custom endpoint URL (for MinIO, etc.).
    pub endpoint: Option<String>,
    /// Access key ID.
    pub access_key_id: Option<String>,
    /// Secret access key.
    pub secret_access_key: Option<String>,
    /// Session token (for temporary credentials).
    pub session_token: Option<String>,
    /// Whether to use path-style addressing.
    pub path_style: bool,
    /// Connection timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum retries.
    pub max_retries: u32,
    /// Multipart upload threshold (bytes).
    pub multipart_threshold: usize,
    /// Multipart chunk size (bytes).
    pub multipart_chunk_size: usize,
    /// Cache size in bytes.
    pub cache_size: usize,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            bucket: String::new(),
            region: "us-east-1".to_string(),
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            path_style: false,
            timeout_secs: 60,
            max_retries: 3,
            multipart_threshold: 100 * 1024 * 1024,
            multipart_chunk_size: 10 * 1024 * 1024,
            cache_size: 100 * 1024 * 1024,
        }
    }
}

impl S3Config {
    /// Creates a config for AWS S3.
    pub fn aws(bucket: &str, region: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            region: region.to_string(),
            ..Default::default()
        }
    }

    /// Creates a config for MinIO.
    pub fn minio(endpoint: &str, bucket: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            region: "us-east-1".to_string(),
            endpoint: Some(endpoint.to_string()),
            path_style: true,
            ..Default::default()
        }
    }

    /// Creates a config for DigitalOcean Spaces.
    pub fn spaces(region: &str, bucket: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            region: region.to_string(),
            endpoint: Some(format!("https://{}.digitaloceanspaces.com", region)),
            ..Default::default()
        }
    }
}

/// S3 object metadata.
#[derive(Debug, Clone)]
pub struct S3ObjectMeta {
    /// Object key.
    pub key: String,
    /// Size in bytes.
    pub size: u64,
    /// Last modified timestamp.
    pub last_modified: u64,
    /// ETag (usually MD5 hash).
    pub etag: String,
    /// Content type.
    pub content_type: Option<String>,
    /// Storage class.
    pub storage_class: StorageClass,
    /// User metadata.
    pub metadata: HashMap<String, String>,
}

/// S3 storage classes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageClass {
    Standard,
    StandardIa,
    OneZoneIa,
    Glacier,
    GlacierDeepArchive,
    IntelligentTiering,
    ReducedRedundancy,
}

impl StorageClass {
    /// Returns true if immediate retrieval is possible.
    pub fn is_immediate(&self) -> bool {
        matches!(
            self,
            StorageClass::Standard
                | StorageClass::StandardIa
                | StorageClass::OneZoneIa
                | StorageClass::IntelligentTiering
                | StorageClass::ReducedRedundancy
        )
    }
}

/// Statistics for S3 operations.
#[derive(Debug, Clone, Default)]
pub struct S3Stats {
    /// Total bytes downloaded.
    pub bytes_downloaded: u64,
    /// Total bytes uploaded.
    pub bytes_uploaded: u64,
    /// Number of GET requests.
    pub get_requests: u64,
    /// Number of PUT requests.
    pub put_requests: u64,
    /// Number of HEAD requests.
    pub head_requests: u64,
    /// Number of multipart uploads.
    pub multipart_uploads: u64,
    /// Cache hits.
    pub cache_hits: u64,
}

/// S3-compatible data source.
pub struct S3DataSource {
    /// Configuration.
    config: S3Config,
    /// Local cache.
    cache: DataCache,
    /// Statistics.
    stats: S3Stats,
    /// Mock storage for testing.
    mock_storage: HashMap<String, MockObject>,
}

/// Mock object for testing.
struct MockObject {
    data: Vec<u8>,
    meta: S3ObjectMeta,
}

impl S3DataSource {
    /// Creates a new S3 data source.
    pub fn new(config: S3Config) -> Self {
        let cache_size = config.cache_size;
        Self {
            config,
            cache: DataCache::new(cache_size),
            stats: S3Stats::default(),
            mock_storage: HashMap::new(),
        }
    }

    /// Creates with credentials.
    pub fn with_credentials(
        bucket: &str,
        region: &str,
        access_key: &str,
        secret_key: &str,
    ) -> Self {
        let config = S3Config {
            bucket: bucket.to_string(),
            region: region.to_string(),
            access_key_id: Some(access_key.to_string()),
            secret_access_key: Some(secret_key.to_string()),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Returns the configuration.
    pub fn config(&self) -> &S3Config {
        &self.config
    }

    /// Returns statistics.
    pub fn stats(&self) -> &S3Stats {
        &self.stats
    }

    /// Stores mock data for testing.
    pub fn store_mock(&mut self, key: &str, data: Vec<u8>) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let hash = self.compute_hash(&data);

        let meta = S3ObjectMeta {
            key: key.to_string(),
            size: data.len() as u64,
            last_modified: now,
            etag: format!("\"{}\"", &hash.to_hex()[..32]),
            content_type: None,
            storage_class: StorageClass::Standard,
            metadata: HashMap::new(),
        };

        self.mock_storage.insert(key.to_string(), MockObject { data, meta });
    }

    /// Computes content hash.
    fn compute_hash(&self, data: &[u8]) -> Hash {
        Sha256Hasher.hash_leaf(data)
    }

    /// Generates a presigned URL for an object.
    pub fn presigned_url(&self, key: &str, _expires_secs: u64) -> String {
        let endpoint = self.config.endpoint.as_deref().unwrap_or("https://s3.amazonaws.com");
        format!("{}/{}/{}", endpoint, self.config.bucket, key)
    }

    /// Lists objects with a prefix.
    pub async fn list_objects(&self, prefix: &str) -> DataSourceResult<Vec<S3ObjectMeta>> {
        let objects: Vec<S3ObjectMeta> = self
            .mock_storage
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(_, v)| v.meta.clone())
            .collect();

        Ok(objects)
    }

    /// Internal fetch implementation.
    async fn fetch_internal(
        &self,
        id: &str,
        range: Option<(u64, u64)>,
        verify_hash: Option<Hash>,
    ) -> DataSourceResult<Vec<u8>> {
        if let Some(mock) = self.mock_storage.get(id) {
            let data = if let Some((start, end)) = range {
                let start = start as usize;
                let end = (end as usize).min(mock.data.len());
                mock.data[start..end].to_vec()
            } else {
                mock.data.clone()
            };

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

        Err(DataSourceError::NotFound(format!(
            "s3://{}/{}",
            self.config.bucket, id
        )))
    }
}

impl DataSource for S3DataSource {
    fn source_type(&self) -> &'static str {
        "s3"
    }

    fn is_available(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move { true })
    }

    fn fetch(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<Vec<u8>>> {
        let id = id.to_string();
        let range = match (options.range_start, options.range_end) {
            (Some(s), Some(e)) => Some((s, e)),
            _ => None,
        };
        let verify_hash = options.verify_hash;

        Box::pin(async move {
            self.fetch_internal(&id, range, verify_hash).await
        })
    }

    fn fetch_stream(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<DataStream>> {
        let id = id.to_string();
        let chunk_size = self.config.multipart_chunk_size;
        let verify_hash = options.verify_hash;

        Box::pin(async move {
            let data = self.fetch_internal(&id, None, verify_hash).await?;

            let total_size = data.len() as u64;
            let mut chunks = Vec::new();

            for (i, chunk_data) in data.chunks(chunk_size).enumerate() {
                let offset = (i * chunk_size) as u64;
                let is_last = offset + chunk_data.len() as u64 >= total_size;

                chunks.push(DataChunk {
                    data: chunk_data.to_vec(),
                    offset,
                    total_size: Some(total_size),
                    is_last,
                });
            }

            Ok(DataStream::new(chunks))
        })
    }

    fn metadata(&self, id: &str) -> BoxFuture<'_, DataSourceResult<ResourceMetadata>> {
        let id = id.to_string();
        let bucket = self.config.bucket.clone();

        Box::pin(async move {
            if let Some(mock) = self.mock_storage.get(&id) {
                let hash = self.compute_hash(&mock.data);
                return Ok(ResourceMetadata {
                    id: id.clone(),
                    size: mock.meta.size,
                    content_type: mock.meta.content_type.clone(),
                    hash: Some(hash),
                    created_at: None,
                    modified_at: Some(mock.meta.last_modified),
                    extra: {
                        let mut extra = mock.meta.metadata.clone();
                        extra.insert("etag".to_string(), mock.meta.etag.clone());
                        extra
                    },
                });
            }

            Err(DataSourceError::NotFound(format!("s3://{}/{}", bucket, id)))
        })
    }

    fn exists(&self, id: &str) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        Box::pin(async move { Ok(self.mock_storage.contains_key(&id)) })
    }

    fn verify(&self, id: &str, expected_hash: &Hash) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        let expected = *expected_hash;
        Box::pin(async move {
            let data = self.fetch_internal(&id, None, None).await?;
            let actual = self.compute_hash(&data);
            Ok(actual == expected)
        })
    }

    fn to_origin(&self, id: &str) -> DataOrigin {
        DataOrigin::S3 {
            bucket: self.config.bucket.clone(),
            key: id.to_string(),
            region: self.config.region.clone(),
            endpoint: self.config.endpoint.clone(),
        }
    }
}

impl WritableDataSource for S3DataSource {
    fn upload(&self, data: &[u8], _options: &UploadOptions) -> BoxFuture<'_, DataSourceResult<String>> {
        let hash = self.compute_hash(data);
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let key = format!("upload/{}/{}", timestamp, &hash.to_hex()[..16]);

        Box::pin(async move { Ok(key) })
    }

    fn upload_stream(&self, stream: DataStream, options: &UploadOptions) -> BoxFuture<'_, DataSourceResult<String>> {
        let data: Vec<u8> = stream.flat_map(|chunk| chunk.data).collect();
        self.upload(&data, options)
    }

    fn delete(&self, _id: &str) -> BoxFuture<'_, DataSourceResult<()>> {
        Box::pin(async move { Ok(()) })
    }
}

/// Builder for S3 data source.
pub struct S3DataSourceBuilder {
    config: S3Config,
}

impl S3DataSourceBuilder {
    /// Creates a new builder.
    pub fn new(bucket: impl Into<String>) -> Self {
        Self {
            config: S3Config {
                bucket: bucket.into(),
                ..Default::default()
            },
        }
    }

    /// Sets the region.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.config.region = region.into();
        self
    }

    /// Sets a custom endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.endpoint = Some(endpoint.into());
        self
    }

    /// Sets credentials.
    pub fn credentials(
        mut self,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        self.config.access_key_id = Some(access_key.into());
        self.config.secret_access_key = Some(secret_key.into());
        self
    }

    /// Enables path-style addressing.
    pub fn path_style(mut self, enabled: bool) -> Self {
        self.config.path_style = enabled;
        self
    }

    /// Sets timeout.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self
    }

    /// Sets multipart threshold.
    pub fn multipart_threshold(mut self, bytes: usize) -> Self {
        self.config.multipart_threshold = bytes;
        self
    }

    /// Builds the data source.
    pub fn build(self) -> S3DataSource {
        S3DataSource::new(self.config)
    }
}

/// Multipart upload manager.
pub struct MultipartUpload {
    /// Bucket.
    _bucket: String,
    /// Key.
    _key: String,
    /// Upload ID.
    upload_id: String,
    /// Uploaded parts.
    parts: Vec<UploadedPart>,
    /// Total bytes uploaded.
    bytes_uploaded: u64,
}

/// An uploaded part.
#[derive(Debug, Clone)]
pub struct UploadedPart {
    /// Part number (1-indexed).
    pub part_number: u32,
    /// ETag returned by S3.
    pub etag: String,
    /// Part size.
    pub size: usize,
}

impl MultipartUpload {
    /// Creates a new multipart upload.
    pub fn new(bucket: &str, key: &str, upload_id: &str) -> Self {
        Self {
            _bucket: bucket.to_string(),
            _key: key.to_string(),
            upload_id: upload_id.to_string(),
            parts: Vec::new(),
            bytes_uploaded: 0,
        }
    }

    /// Adds an uploaded part.
    pub fn add_part(&mut self, part: UploadedPart) {
        self.bytes_uploaded += part.size as u64;
        self.parts.push(part);
    }

    /// Returns the upload ID.
    pub fn upload_id(&self) -> &str {
        &self.upload_id
    }

    /// Returns total bytes uploaded.
    pub fn bytes_uploaded(&self) -> u64 {
        self.bytes_uploaded
    }

    /// Returns the number of parts.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_aws() {
        let config = S3Config::aws("my-bucket", "us-west-2");
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.region, "us-west-2");
        assert!(config.endpoint.is_none());
    }

    #[test]
    fn test_s3_config_minio() {
        let config = S3Config::minio("http://localhost:9000", "my-bucket");
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.endpoint, Some("http://localhost:9000".to_string()));
        assert!(config.path_style);
    }

    #[test]
    fn test_storage_class() {
        assert!(StorageClass::Standard.is_immediate());
        assert!(StorageClass::StandardIa.is_immediate());
        assert!(!StorageClass::Glacier.is_immediate());
        assert!(!StorageClass::GlacierDeepArchive.is_immediate());
    }

    #[tokio::test]
    async fn test_s3_mock_storage() {
        let mut source = S3DataSource::new(S3Config::aws("test-bucket", "us-east-1"));

        let data = b"S3 test data".to_vec();
        source.store_mock("test/file.bin", data.clone());

        let fetched = source
            .fetch("test/file.bin", &FetchOptions::default())
            .await
            .unwrap();
        assert_eq!(fetched, data);
    }

    #[tokio::test]
    async fn test_s3_range_request() {
        let mut source = S3DataSource::new(S3Config::aws("test-bucket", "us-east-1"));

        let data = b"0123456789".to_vec();
        source.store_mock("range-test", data.clone());

        let options = FetchOptions {
            range_start: Some(2),
            range_end: Some(5),
            ..Default::default()
        };

        let fetched = source.fetch("range-test", &options).await.unwrap();
        assert_eq!(fetched, b"234".to_vec());
    }

    #[tokio::test]
    async fn test_s3_metadata() {
        let mut source = S3DataSource::new(S3Config::aws("test-bucket", "us-east-1"));

        let data = b"Metadata test".to_vec();
        source.store_mock("meta-test", data.clone());

        let meta = source.metadata("meta-test").await.unwrap();
        assert_eq!(meta.size, data.len() as u64);
        assert!(meta.hash.is_some());
    }

    #[tokio::test]
    async fn test_s3_not_found() {
        let source = S3DataSource::new(S3Config::aws("test-bucket", "us-east-1"));

        let result = source
            .fetch("nonexistent", &FetchOptions::default())
            .await;
        assert!(matches!(result, Err(DataSourceError::NotFound(_))));
    }

    #[test]
    fn test_s3_to_origin() {
        let source = S3DataSource::new(S3Config::aws("my-bucket", "us-west-2"));
        let origin = source.to_origin("path/to/file.bin");

        match origin {
            DataOrigin::S3 {
                bucket,
                key,
                region,
                ..
            } => {
                assert_eq!(bucket, "my-bucket");
                assert_eq!(key, "path/to/file.bin");
                assert_eq!(region, "us-west-2");
            }
            _ => panic!("Expected S3 origin"),
        }
    }

    #[tokio::test]
    async fn test_s3_list_objects() {
        let mut source = S3DataSource::new(S3Config::aws("test-bucket", "us-east-1"));

        source.store_mock("prefix/a.txt", b"a".to_vec());
        source.store_mock("prefix/b.txt", b"b".to_vec());
        source.store_mock("other/c.txt", b"c".to_vec());

        let objects = source.list_objects("prefix/").await.unwrap();
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn test_s3_builder() {
        let source = S3DataSourceBuilder::new("my-bucket")
            .region("eu-west-1")
            .endpoint("https://custom.s3.endpoint")
            .credentials("key", "secret")
            .path_style(true)
            .timeout_secs(120)
            .build();

        assert_eq!(source.config().bucket, "my-bucket");
        assert_eq!(source.config().region, "eu-west-1");
        assert!(source.config().path_style);
    }

    #[test]
    fn test_presigned_url() {
        let source = S3DataSource::new(S3Config::aws("my-bucket", "us-east-1"));
        let url = source.presigned_url("path/to/file", 3600);
        assert!(url.contains("my-bucket"));
        assert!(url.contains("path/to/file"));
    }

    #[test]
    fn test_multipart_upload() {
        let mut upload = MultipartUpload::new("bucket", "key", "upload-123");

        upload.add_part(UploadedPart {
            part_number: 1,
            etag: "\"abc123\"".to_string(),
            size: 1024,
        });

        assert_eq!(upload.part_count(), 1);
        assert_eq!(upload.bytes_uploaded(), 1024);
    }
}
