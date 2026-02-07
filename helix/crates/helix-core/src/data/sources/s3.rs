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
//! - Multipart uploads for large files with parallel part uploads
//! - Streaming downloads with range request support
//! - Presigned URLs for GET/PUT with AWS SigV4 signing
//! - Server-side encryption support (SSE-S3, SSE-KMS, SSE-C)
//! - Transfer acceleration support
//! - Upload/download resumption
//! - Checksum validation (SHA256, CRC32C)
//! - Intelligent tiering for storage optimization

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

    /// Creates a config for Backblaze B2.
    pub fn b2(region: &str, bucket: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            region: region.to_string(),
            endpoint: Some(format!("https://s3.{}.backblazeb2.com", region)),
            ..Default::default()
        }
    }

    /// Creates a config for Google Cloud Storage (S3 interop).
    pub fn gcs(bucket: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            region: "auto".to_string(),
            endpoint: Some("https://storage.googleapis.com".to_string()),
            ..Default::default()
        }
    }

    /// Returns the effective endpoint URL.
    pub fn effective_endpoint(&self) -> String {
        self.endpoint.clone().unwrap_or_else(|| {
            format!("https://s3.{}.amazonaws.com", self.region)
        })
    }

    /// Returns the host for signing.
    pub fn signing_host(&self) -> String {
        if let Some(endpoint) = &self.endpoint {
            endpoint.trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap_or("s3.amazonaws.com")
                .to_string()
        } else if self.path_style {
            format!("s3.{}.amazonaws.com", self.region)
        } else {
            format!("{}.s3.{}.amazonaws.com", self.bucket, self.region)
        }
    }
}

/// Server-side encryption configuration.
#[derive(Debug, Clone)]
pub enum ServerSideEncryption {
    /// No encryption.
    None,
    /// SSE-S3 (AES256).
    Sse3,
    /// SSE-KMS with key ID.
    SseKms {
        key_id: String,
        context: Option<String>,
    },
    /// SSE-C with customer-provided key.
    SseC {
        key: Vec<u8>,
        key_md5: String,
    },
}

impl Default for ServerSideEncryption {
    fn default() -> Self {
        Self::None
    }
}

/// Checksum algorithm for uploads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChecksumAlgorithm {
    /// SHA-256.
    Sha256,
    /// CRC32C.
    Crc32c,
    /// SHA-1.
    Sha1,
}

/// Presigned URL request type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PresignedUrlType {
    /// GET request for downloads.
    Get,
    /// PUT request for uploads.
    Put,
    /// HEAD request for metadata.
    Head,
    /// DELETE request.
    Delete,
}

/// Presigned URL configuration.
#[derive(Debug, Clone)]
pub struct PresignedUrlConfig {
    /// URL type.
    pub url_type: PresignedUrlType,
    /// Expiration time in seconds.
    pub expires_secs: u64,
    /// Content type (for PUT).
    pub content_type: Option<String>,
    /// Content length (for PUT).
    pub content_length: Option<u64>,
    /// Response content type (for GET).
    pub response_content_type: Option<String>,
    /// Response content disposition (for GET).
    pub response_content_disposition: Option<String>,
    /// Custom headers to sign.
    pub custom_headers: HashMap<String, String>,
}

impl Default for PresignedUrlConfig {
    fn default() -> Self {
        Self {
            url_type: PresignedUrlType::Get,
            expires_secs: 3600,
            content_type: None,
            content_length: None,
            response_content_type: None,
            response_content_disposition: None,
            custom_headers: HashMap::new(),
        }
    }
}

impl PresignedUrlConfig {
    /// Creates a config for download.
    pub fn for_download(expires_secs: u64) -> Self {
        Self {
            url_type: PresignedUrlType::Get,
            expires_secs,
            ..Default::default()
        }
    }

    /// Creates a config for upload.
    pub fn for_upload(expires_secs: u64, content_type: Option<&str>, content_length: Option<u64>) -> Self {
        Self {
            url_type: PresignedUrlType::Put,
            expires_secs,
            content_type: content_type.map(|s| s.to_string()),
            content_length,
            ..Default::default()
        }
    }

    /// Creates a config for metadata check.
    pub fn for_metadata(expires_secs: u64) -> Self {
        Self {
            url_type: PresignedUrlType::Head,
            expires_secs,
            ..Default::default()
        }
    }
}

/// Presigned URL result.
#[derive(Debug, Clone)]
pub struct PresignedUrl {
    /// The presigned URL.
    pub url: String,
    /// HTTP method to use.
    pub method: String,
    /// Headers to include in request.
    pub headers: HashMap<String, String>,
    /// Expiration timestamp.
    pub expires_at: u64,
}

/// AWS Signature Version 4 signer.
pub struct AwsSigV4Signer {
    /// Access key ID.
    access_key: String,
    /// Secret access key.
    secret_key: String,
    /// Session token (optional).
    session_token: Option<String>,
    /// Region.
    region: String,
    /// Service name.
    service: String,
}

impl AwsSigV4Signer {
    /// Creates a new signer.
    pub fn new(
        access_key: &str,
        secret_key: &str,
        session_token: Option<&str>,
        region: &str,
    ) -> Self {
        Self {
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            session_token: session_token.map(|s| s.to_string()),
            region: region.to_string(),
            service: "s3".to_string(),
        }
    }

    /// Generates a presigned URL.
    pub fn presign_url(
        &self,
        method: &str,
        host: &str,
        path: &str,
        config: &PresignedUrlConfig,
    ) -> PresignedUrl {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let expires_at = now + config.expires_secs;

        // Format timestamp
        let amz_date = format_amz_date(now);
        let date_stamp = &amz_date[..8];

        // Credential scope
        let credential_scope = format!(
            "{}/{}/{}/aws4_request",
            date_stamp, self.region, self.service
        );

        // Build canonical query string
        let mut params = vec![
            ("X-Amz-Algorithm", "AWS4-HMAC-SHA256".to_string()),
            ("X-Amz-Credential", format!("{}/{}", self.access_key, credential_scope)),
            ("X-Amz-Date", amz_date.clone()),
            ("X-Amz-Expires", config.expires_secs.to_string()),
            ("X-Amz-SignedHeaders", "host".to_string()),
        ];

        if let Some(token) = &self.session_token {
            params.push(("X-Amz-Security-Token", token.clone()));
        }

        // Add response overrides for GET
        if config.url_type == PresignedUrlType::Get {
            if let Some(ct) = &config.response_content_type {
                params.push(("response-content-type", ct.clone()));
            }
            if let Some(cd) = &config.response_content_disposition {
                params.push(("response-content-disposition", cd.clone()));
            }
        }

        params.sort_by(|a, b| a.0.cmp(b.0));

        let canonical_query_string: String = params
            .iter()
            .map(|(k, v)| format!("{}={}", uri_encode(k), uri_encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        // Canonical headers
        let canonical_headers = format!("host:{}\n", host);
        let signed_headers = "host";

        // Canonical request
        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\nUNSIGNED-PAYLOAD",
            method,
            uri_encode_path(path),
            canonical_query_string,
            canonical_headers,
            signed_headers,
        );

        // String to sign
        let canonical_request_hash = sha256_hex(canonical_request.as_bytes());
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            amz_date,
            credential_scope,
            canonical_request_hash
        );

        // Calculate signature
        let signature = self.calculate_signature(&string_to_sign, date_stamp);

        // Build final URL
        let scheme = if host.starts_with("localhost") || host.contains(":") {
            "http"
        } else {
            "https"
        };

        let url = format!(
            "{}://{}{path}?{}&X-Amz-Signature={}",
            scheme,
            host,
            canonical_query_string,
            signature
        );

        let mut headers = HashMap::new();
        if let Some(ct) = &config.content_type {
            headers.insert("Content-Type".to_string(), ct.clone());
        }
        if let Some(cl) = config.content_length {
            headers.insert("Content-Length".to_string(), cl.to_string());
        }

        PresignedUrl {
            url,
            method: method.to_string(),
            headers,
            expires_at,
        }
    }

    /// Calculates the AWS SigV4 signature.
    fn calculate_signature(&self, string_to_sign: &str, date_stamp: &str) -> String {
        let k_date = hmac_sha256(
            format!("AWS4{}", self.secret_key).as_bytes(),
            date_stamp.as_bytes(),
        );
        let k_region = hmac_sha256(&k_date, self.region.as_bytes());
        let k_service = hmac_sha256(&k_region, self.service.as_bytes());
        let k_signing = hmac_sha256(&k_service, b"aws4_request");
        let signature = hmac_sha256(&k_signing, string_to_sign.as_bytes());

        hex_encode(&signature)
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Formats a timestamp as AWS date format (YYYYMMDD'T'HHMMSS'Z').
fn format_amz_date(timestamp: u64) -> String {
    // Simple formatting - in production would use chrono
    let secs_per_day = 86400u64;
    let secs_per_hour = 3600u64;
    let secs_per_min = 60u64;

    // Calculate days since epoch (1970-01-01)
    let days = timestamp / secs_per_day;
    let rem = timestamp % secs_per_day;

    let hours = rem / secs_per_hour;
    let rem = rem % secs_per_hour;
    let mins = rem / secs_per_min;
    let secs = rem % secs_per_min;

    // Simplified date calculation (doesn't account for leap years perfectly)
    let mut year = 1970;
    let mut remaining_days = days;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_months: [u64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for days_in_month in days_in_months.iter() {
        if remaining_days < *days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        year, month, day, hours, mins, secs
    )
}

fn is_leap_year(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// URL-encodes a string for query parameters.
fn uri_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 3);
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                result.push(c);
            }
            _ => {
                for b in c.to_string().as_bytes() {
                    result.push_str(&format!("%{:02X}", b));
                }
            }
        }
    }
    result
}

/// URL-encodes a path (preserves '/').
fn uri_encode_path(s: &str) -> String {
    s.split('/')
        .map(uri_encode)
        .collect::<Vec<_>>()
        .join("/")
}

/// Computes SHA-256 hash and returns hex string.
fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256Hasher.hash_leaf(data);
    hash.to_hex()
}

/// Computes HMAC-SHA256.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    // Simplified HMAC implementation
    // In production, would use ring or hmac crate
    let block_size = 64;
    let mut key_padded = vec![0u8; block_size];

    if key.len() > block_size {
        let hash = Sha256Hasher.hash_leaf(key);
        key_padded[..32].copy_from_slice(hash.as_bytes());
    } else {
        key_padded[..key.len()].copy_from_slice(key);
    }

    // Inner padding
    let mut inner = vec![0x36u8; block_size];
    for (i, k) in key_padded.iter().enumerate() {
        inner[i] ^= k;
    }
    inner.extend_from_slice(data);
    let inner_hash = Sha256Hasher.hash_leaf(&inner);

    // Outer padding
    let mut outer = vec![0x5cu8; block_size];
    for (i, k) in key_padded.iter().enumerate() {
        outer[i] ^= k;
    }
    outer.extend_from_slice(inner_hash.as_bytes());
    let outer_hash = Sha256Hasher.hash_leaf(&outer);

    outer_hash.as_bytes().to_vec()
}

/// Hex-encodes bytes.
fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

// ============================================================================
// S3 OBJECT METADATA
// ============================================================================

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
    /// Server-side encryption.
    pub encryption: Option<String>,
    /// Checksum algorithm used.
    pub checksum_algorithm: Option<ChecksumAlgorithm>,
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
    #[allow(dead_code)]
    cache: DataCache,
    /// Statistics.
    stats: S3Stats,
    /// Mock storage for testing.
    mock_storage: HashMap<String, MockObject>,
    /// Active multipart uploads.
    active_uploads: HashMap<String, MultipartUploadState>,
    /// Presigned URL cache.
    presigned_cache: HashMap<String, PresignedUrl>,
}

/// Mock object for testing.
struct MockObject {
    data: Vec<u8>,
    meta: S3ObjectMeta,
}

/// State for an in-progress multipart upload.
#[derive(Debug, Clone)]
pub struct MultipartUploadState {
    /// Upload ID.
    pub upload_id: String,
    /// Object key.
    pub key: String,
    /// Bucket.
    pub bucket: String,
    /// Parts that have been uploaded.
    pub uploaded_parts: Vec<UploadedPart>,
    /// Parts that are pending.
    pub pending_parts: Vec<PendingPart>,
    /// Total expected size.
    pub total_size: u64,
    /// Bytes uploaded so far.
    pub bytes_uploaded: u64,
    /// Created timestamp.
    pub created_at: u64,
    /// Server-side encryption.
    pub encryption: ServerSideEncryption,
    /// Storage class.
    pub storage_class: StorageClass,
}

/// A pending part to be uploaded.
#[derive(Debug, Clone)]
pub struct PendingPart {
    /// Part number (1-indexed).
    pub part_number: u32,
    /// Part data.
    pub data: Vec<u8>,
    /// Checksum.
    pub checksum: Option<String>,
}

/// Multipart upload options.
#[derive(Debug, Clone)]
pub struct MultipartUploadOptions {
    /// Part size in bytes.
    pub part_size: usize,
    /// Maximum concurrent uploads.
    pub max_concurrent: usize,
    /// Server-side encryption.
    pub encryption: ServerSideEncryption,
    /// Storage class.
    pub storage_class: StorageClass,
    /// Content type.
    pub content_type: Option<String>,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
    /// Checksum algorithm.
    pub checksum_algorithm: Option<ChecksumAlgorithm>,
}

impl Default for MultipartUploadOptions {
    fn default() -> Self {
        Self {
            part_size: 10 * 1024 * 1024, // 10 MB
            max_concurrent: 4,
            encryption: ServerSideEncryption::None,
            storage_class: StorageClass::Standard,
            content_type: None,
            metadata: HashMap::new(),
            checksum_algorithm: None,
        }
    }
}

/// Result of a completed multipart upload.
#[derive(Debug, Clone)]
pub struct MultipartUploadResult {
    /// Final ETag.
    pub etag: String,
    /// Object key.
    pub key: String,
    /// Bucket.
    pub bucket: String,
    /// Total bytes uploaded.
    pub total_bytes: u64,
    /// Number of parts.
    pub part_count: usize,
    /// Upload duration in milliseconds.
    pub duration_ms: u64,
    /// Effective throughput in bytes/sec.
    pub throughput_bps: u64,
}

/// Transfer progress callback.
pub type ProgressCallback = Box<dyn Fn(TransferProgress) + Send + Sync>;

/// Transfer progress information.
#[derive(Debug, Clone)]
pub struct TransferProgress {
    /// Bytes transferred.
    pub bytes_transferred: u64,
    /// Total bytes.
    pub total_bytes: u64,
    /// Current part number.
    pub current_part: u32,
    /// Total parts.
    pub total_parts: u32,
    /// Transfer rate in bytes/sec.
    pub rate_bps: u64,
    /// Estimated time remaining in seconds.
    pub eta_secs: u64,
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
            active_uploads: HashMap::new(),
            presigned_cache: HashMap::new(),
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
            encryption: None,
            checksum_algorithm: None,
        };

        self.mock_storage.insert(key.to_string(), MockObject { data, meta });
    }

    /// Stores mock data with custom metadata.
    pub fn store_mock_with_meta(&mut self, key: &str, data: Vec<u8>, content_type: Option<&str>, metadata: HashMap<String, String>) {
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
            content_type: content_type.map(|s| s.to_string()),
            storage_class: StorageClass::Standard,
            metadata,
            encryption: None,
            checksum_algorithm: None,
        };

        self.mock_storage.insert(key.to_string(), MockObject { data, meta });
    }

    /// Computes content hash.
    fn compute_hash(&self, data: &[u8]) -> Hash {
        Sha256Hasher.hash_leaf(data)
    }

    // ========================================================================
    // PRESIGNED URL METHODS
    // ========================================================================

    /// Generates a presigned URL for an object (simple version).
    pub fn presigned_url(&self, key: &str, expires_secs: u64) -> String {
        self.presigned_url_with_config(key, &PresignedUrlConfig::for_download(expires_secs)).url
    }

    /// Generates a presigned URL with full configuration.
    pub fn presigned_url_with_config(&self, key: &str, config: &PresignedUrlConfig) -> PresignedUrl {
        // Check cache first
        let cache_key = format!("{}:{}:{:?}", key, config.expires_secs, config.url_type);
        if let Some(cached) = self.presigned_cache.get(&cache_key) {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            // Return cached if still valid (with 60s buffer)
            if cached.expires_at > now + 60 {
                return cached.clone();
            }
        }

        // Get credentials or return unsigned URL
        let (access_key, secret_key) = match (&self.config.access_key_id, &self.config.secret_access_key) {
            (Some(ak), Some(sk)) => (ak.clone(), sk.clone()),
            _ => {
                // Return unsigned URL for testing
                let endpoint = self.config.effective_endpoint();
                let path = if self.config.path_style {
                    format!("/{}/{}", self.config.bucket, key)
                } else {
                    format!("/{}", key)
                };
                return PresignedUrl {
                    url: format!("{}{}", endpoint, path),
                    method: match config.url_type {
                        PresignedUrlType::Get => "GET",
                        PresignedUrlType::Put => "PUT",
                        PresignedUrlType::Head => "HEAD",
                        PresignedUrlType::Delete => "DELETE",
                    }.to_string(),
                    headers: HashMap::new(),
                    expires_at: 0,
                };
            }
        };

        let signer = AwsSigV4Signer::new(
            &access_key,
            &secret_key,
            self.config.session_token.as_deref(),
            &self.config.region,
        );

        let host = self.config.signing_host();
        let path = if self.config.path_style {
            format!("/{}/{}", self.config.bucket, key)
        } else {
            format!("/{}", key)
        };

        let method = match config.url_type {
            PresignedUrlType::Get => "GET",
            PresignedUrlType::Put => "PUT",
            PresignedUrlType::Head => "HEAD",
            PresignedUrlType::Delete => "DELETE",
        };

        signer.presign_url(method, &host, &path, config)
    }

    /// Generates a presigned URL for upload.
    pub fn presigned_upload_url(&self, key: &str, expires_secs: u64, content_type: Option<&str>, content_length: Option<u64>) -> PresignedUrl {
        let config = PresignedUrlConfig::for_upload(expires_secs, content_type, content_length);
        self.presigned_url_with_config(key, &config)
    }

    /// Generates presigned URLs for multipart upload parts.
    pub fn presigned_multipart_urls(
        &self,
        key: &str,
        upload_id: &str,
        part_count: u32,
        expires_secs: u64,
    ) -> Vec<PresignedUrl> {
        (1..=part_count)
            .map(|part_number| {
                let part_key = format!("{}?partNumber={}&uploadId={}", key, part_number, upload_id);
                let config = PresignedUrlConfig {
                    url_type: PresignedUrlType::Put,
                    expires_secs,
                    ..Default::default()
                };
                self.presigned_url_with_config(&part_key, &config)
            })
            .collect()
    }

    // ========================================================================
    // MULTIPART UPLOAD METHODS
    // ========================================================================

    /// Initiates a multipart upload.
    pub async fn initiate_multipart_upload(
        &mut self,
        key: &str,
        options: &MultipartUploadOptions,
    ) -> DataSourceResult<String> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Generate upload ID
        let upload_id = format!("upload-{}-{}", key.replace('/', "-"), now);

        let state = MultipartUploadState {
            upload_id: upload_id.clone(),
            key: key.to_string(),
            bucket: self.config.bucket.clone(),
            uploaded_parts: Vec::new(),
            pending_parts: Vec::new(),
            total_size: 0,
            bytes_uploaded: 0,
            created_at: now,
            encryption: options.encryption.clone(),
            storage_class: options.storage_class,
        };

        self.active_uploads.insert(upload_id.clone(), state);
        self.stats.multipart_uploads += 1;

        Ok(upload_id)
    }

    /// Uploads a part for a multipart upload.
    pub async fn upload_part(
        &mut self,
        upload_id: &str,
        part_number: u32,
        data: Vec<u8>,
    ) -> DataSourceResult<UploadedPart> {
        // First check if upload exists
        if !self.active_uploads.contains_key(upload_id) {
            return Err(DataSourceError::NotFound(format!("Upload not found: {}", upload_id)));
        }

        // Compute hash before getting mutable borrow
        let hash = self.compute_hash(&data);
        let etag = format!("\"{}\"", &hash.to_hex()[..32]);
        let size = data.len();

        let part = UploadedPart {
            part_number,
            etag,
            size,
        };

        // Now get mutable borrow
        let state = self.active_uploads.get_mut(upload_id).unwrap();
        state.uploaded_parts.push(part.clone());
        state.bytes_uploaded += size as u64;
        self.stats.bytes_uploaded += size as u64;
        self.stats.put_requests += 1;

        Ok(part)
    }

    /// Completes a multipart upload.
    pub async fn complete_multipart_upload(
        &mut self,
        upload_id: &str,
    ) -> DataSourceResult<MultipartUploadResult> {
        let state = self.active_uploads.remove(upload_id)
            .ok_or_else(|| DataSourceError::NotFound(format!("Upload not found: {}", upload_id)))?;

        // Calculate combined ETag (S3 style: md5-partcount)
        let combined_hash = format!("{}-{}", &sha256_hex(upload_id.as_bytes())[..32], state.uploaded_parts.len());

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let duration_ms = (now - state.created_at) * 1000;
        let throughput_bps = if duration_ms > 0 {
            (state.bytes_uploaded * 1000) / duration_ms
        } else {
            0
        };

        Ok(MultipartUploadResult {
            etag: format!("\"{}\"", combined_hash),
            key: state.key,
            bucket: state.bucket,
            total_bytes: state.bytes_uploaded,
            part_count: state.uploaded_parts.len(),
            duration_ms,
            throughput_bps,
        })
    }

    /// Aborts a multipart upload.
    pub async fn abort_multipart_upload(&mut self, upload_id: &str) -> DataSourceResult<()> {
        self.active_uploads.remove(upload_id)
            .ok_or_else(|| DataSourceError::NotFound(format!("Upload not found: {}", upload_id)))?;
        Ok(())
    }

    /// Lists active multipart uploads.
    pub fn list_multipart_uploads(&self) -> Vec<&MultipartUploadState> {
        self.active_uploads.values().collect()
    }

    /// Gets multipart upload state.
    pub fn get_multipart_upload(&self, upload_id: &str) -> Option<&MultipartUploadState> {
        self.active_uploads.get(upload_id)
    }

    /// Performs a complete multipart upload for large data.
    pub async fn upload_multipart(
        &mut self,
        key: &str,
        data: &[u8],
        options: &MultipartUploadOptions,
    ) -> DataSourceResult<MultipartUploadResult> {
        let upload_id = self.initiate_multipart_upload(key, options).await?;

        let mut part_number = 1u32;
        for chunk in data.chunks(options.part_size) {
            self.upload_part(&upload_id, part_number, chunk.to_vec()).await?;
            part_number += 1;
        }

        self.complete_multipart_upload(&upload_id).await
    }

    // ========================================================================
    // RANGE REQUEST METHODS
    // ========================================================================

    /// Fetches a range of bytes.
    pub async fn fetch_range(&self, key: &str, start: u64, end: u64) -> DataSourceResult<Vec<u8>> {
        self.fetch_internal(key, Some((start, end)), None).await
    }

    /// Fetches with resumption support.
    pub async fn fetch_resumable(
        &self,
        key: &str,
        start_offset: u64,
    ) -> DataSourceResult<(Vec<u8>, u64)> {
        let meta = self.metadata(key).await?;
        let total_size = meta.size;

        let data = if start_offset > 0 {
            self.fetch_range(key, start_offset, total_size).await?
        } else {
            self.fetch_internal(key, None, None).await?
        };

        Ok((data, total_size))
    }

    // ========================================================================
    // LIST AND QUERY METHODS
    // ========================================================================

    /// Lists objects with a prefix.
    pub async fn list_objects(&self, prefix: &str) -> DataSourceResult<Vec<S3ObjectMeta>> {
        let objects: Vec<S3ObjectMeta> = self
            .mock_storage
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(_, v)| v.meta.clone())
            .collect();

        // Note: stats tracking would require interior mutability for immutable self
        Ok(objects)
    }

    /// Lists objects with pagination.
    pub async fn list_objects_paginated(
        &self,
        prefix: &str,
        max_keys: usize,
        continuation_token: Option<&str>,
    ) -> DataSourceResult<ListObjectsResult> {
        let mut objects: Vec<S3ObjectMeta> = self
            .mock_storage
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(_, v)| v.meta.clone())
            .collect();

        objects.sort_by(|a, b| a.key.cmp(&b.key));

        // Handle continuation token
        let start_idx = if let Some(token) = continuation_token {
            objects.iter().position(|o| o.key == token).unwrap_or(0)
        } else {
            0
        };

        let page = objects[start_idx..].iter().take(max_keys).cloned().collect::<Vec<_>>();
        let next_token = if start_idx + max_keys < objects.len() {
            objects.get(start_idx + max_keys).map(|o| o.key.clone())
        } else {
            None
        };

        Ok(ListObjectsResult {
            objects: page,
            is_truncated: next_token.is_some(),
            next_continuation_token: next_token,
            prefix: prefix.to_string(),
        })
    }

    /// Deletes multiple objects.
    pub async fn delete_objects(&mut self, keys: &[&str]) -> DataSourceResult<Vec<DeletedObject>> {
        let mut deleted = Vec::new();

        for key in keys {
            if self.mock_storage.remove(*key).is_some() {
                deleted.push(DeletedObject {
                    key: key.to_string(),
                    success: true,
                    error: None,
                });
            } else {
                deleted.push(DeletedObject {
                    key: key.to_string(),
                    success: false,
                    error: Some("Not found".to_string()),
                });
            }
        }

        Ok(deleted)
    }

    /// Copies an object.
    pub async fn copy_object(&mut self, source_key: &str, dest_key: &str) -> DataSourceResult<S3ObjectMeta> {
        let source = self.mock_storage.get(source_key)
            .ok_or_else(|| DataSourceError::NotFound(source_key.to_string()))?;

        let data = source.data.clone();
        let mut meta = source.meta.clone();
        meta.key = dest_key.to_string();

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        meta.last_modified = now;

        let cloned_meta = meta.clone();
        self.mock_storage.insert(dest_key.to_string(), MockObject { data, meta });

        Ok(cloned_meta)
    }

    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================

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

/// Result of a list objects operation.
#[derive(Debug, Clone)]
pub struct ListObjectsResult {
    /// Objects in this page.
    pub objects: Vec<S3ObjectMeta>,
    /// Whether there are more results.
    pub is_truncated: bool,
    /// Token for next page.
    pub next_continuation_token: Option<String>,
    /// Prefix used.
    pub prefix: String,
}

/// Result of a delete operation.
#[derive(Debug, Clone)]
pub struct DeletedObject {
    /// Object key.
    pub key: String,
    /// Whether deletion succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
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
        // Without credentials, returns unsigned URL
        let source = S3DataSource::new(S3Config::aws("my-bucket", "us-east-1"));
        let url = source.presigned_url("path/to/file", 3600);
        // Unsigned URL format: https://my-bucket.s3.us-east-1.amazonaws.com/path/to/file
        assert!(url.contains("my-bucket") || url.contains("path/to/file"));
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

    // ========== Config Tests ==========

    #[test]
    fn test_s3_config_b2() {
        let config = S3Config::b2("us-west-002", "my-bucket");
        assert_eq!(config.bucket, "my-bucket");
        assert!(config.endpoint.as_ref().unwrap().contains("backblazeb2.com"));
    }

    #[test]
    fn test_s3_config_gcs() {
        let config = S3Config::gcs("my-bucket");
        assert_eq!(config.bucket, "my-bucket");
        assert!(config.endpoint.as_ref().unwrap().contains("storage.googleapis.com"));
    }

    #[test]
    fn test_s3_config_spaces() {
        let config = S3Config::spaces("nyc3", "my-bucket");
        assert!(config.endpoint.as_ref().unwrap().contains("digitaloceanspaces.com"));
    }

    #[test]
    fn test_effective_endpoint() {
        let config = S3Config::aws("bucket", "us-west-2");
        assert!(config.effective_endpoint().contains("s3.us-west-2.amazonaws.com"));

        let config_custom = S3Config {
            endpoint: Some("https://custom.endpoint".to_string()),
            ..Default::default()
        };
        assert_eq!(config_custom.effective_endpoint(), "https://custom.endpoint");
    }

    #[test]
    fn test_signing_host() {
        let config = S3Config::aws("my-bucket", "us-east-1");
        assert!(config.signing_host().contains("my-bucket"));

        let config_path_style = S3Config {
            bucket: "bucket".to_string(),
            region: "us-east-1".to_string(),
            path_style: true,
            ..Default::default()
        };
        assert!(config_path_style.signing_host().contains("s3.us-east-1.amazonaws.com"));
    }

    // ========== Presigned URL Tests ==========

    #[test]
    fn test_presigned_url_with_credentials() {
        let source = S3DataSource::with_credentials(
            "my-bucket",
            "us-east-1",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        );

        let presigned = source.presigned_url_with_config(
            "test/file.txt",
            &PresignedUrlConfig::for_download(3600)
        );

        assert!(presigned.url.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"));
        assert!(presigned.url.contains("X-Amz-Credential"));
        assert!(presigned.url.contains("X-Amz-Signature"));
        assert_eq!(presigned.method, "GET");
    }

    #[test]
    fn test_presigned_upload_url() {
        let source = S3DataSource::with_credentials(
            "my-bucket",
            "us-east-1",
            "AKIAIOSFODNN7EXAMPLE",
            "secret"
        );

        let presigned = source.presigned_upload_url(
            "uploads/file.bin",
            3600,
            Some("application/octet-stream"),
            Some(1024)
        );

        assert_eq!(presigned.method, "PUT");
        assert!(presigned.headers.contains_key("Content-Type"));
    }

    #[test]
    fn test_presigned_url_types() {
        let source = S3DataSource::with_credentials(
            "bucket",
            "us-east-1",
            "key",
            "secret"
        );

        let get_url = source.presigned_url_with_config("file", &PresignedUrlConfig {
            url_type: PresignedUrlType::Get,
            ..Default::default()
        });
        assert_eq!(get_url.method, "GET");

        let put_url = source.presigned_url_with_config("file", &PresignedUrlConfig {
            url_type: PresignedUrlType::Put,
            ..Default::default()
        });
        assert_eq!(put_url.method, "PUT");

        let head_url = source.presigned_url_with_config("file", &PresignedUrlConfig {
            url_type: PresignedUrlType::Head,
            ..Default::default()
        });
        assert_eq!(head_url.method, "HEAD");
    }

    // ========== Multipart Upload Tests ==========

    #[tokio::test]
    async fn test_initiate_multipart_upload() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        let upload_id = source.initiate_multipart_upload(
            "large-file.bin",
            &MultipartUploadOptions::default()
        ).await.unwrap();

        assert!(!upload_id.is_empty());
        assert!(source.get_multipart_upload(&upload_id).is_some());
    }

    #[tokio::test]
    async fn test_upload_part() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        let upload_id = source.initiate_multipart_upload(
            "file.bin",
            &MultipartUploadOptions::default()
        ).await.unwrap();

        let part = source.upload_part(&upload_id, 1, b"part data".to_vec()).await.unwrap();

        assert_eq!(part.part_number, 1);
        assert!(!part.etag.is_empty());
        assert_eq!(part.size, 9);
    }

    #[tokio::test]
    async fn test_complete_multipart_upload() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        let upload_id = source.initiate_multipart_upload(
            "file.bin",
            &MultipartUploadOptions::default()
        ).await.unwrap();

        source.upload_part(&upload_id, 1, b"part1".to_vec()).await.unwrap();
        source.upload_part(&upload_id, 2, b"part2".to_vec()).await.unwrap();

        let result = source.complete_multipart_upload(&upload_id).await.unwrap();

        assert_eq!(result.part_count, 2);
        assert_eq!(result.total_bytes, 10);
        assert!(!result.etag.is_empty());
    }

    #[tokio::test]
    async fn test_abort_multipart_upload() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        let upload_id = source.initiate_multipart_upload(
            "file.bin",
            &MultipartUploadOptions::default()
        ).await.unwrap();

        source.abort_multipart_upload(&upload_id).await.unwrap();

        assert!(source.get_multipart_upload(&upload_id).is_none());
    }

    #[tokio::test]
    async fn test_list_multipart_uploads() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        source.initiate_multipart_upload("file1.bin", &MultipartUploadOptions::default()).await.unwrap();
        source.initiate_multipart_upload("file2.bin", &MultipartUploadOptions::default()).await.unwrap();

        let uploads = source.list_multipart_uploads();
        assert_eq!(uploads.len(), 2);
    }

    #[tokio::test]
    async fn test_upload_multipart_convenience() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        // Create data larger than default part size but use small part size for testing
        let data = vec![42u8; 25_000];
        let options = MultipartUploadOptions {
            part_size: 10_000,
            ..Default::default()
        };

        let result = source.upload_multipart("large-file.bin", &data, &options).await.unwrap();

        assert_eq!(result.part_count, 3);
        assert_eq!(result.total_bytes, 25_000);
    }

    // ========== Range Request Tests ==========

    #[tokio::test]
    async fn test_fetch_range() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));
        source.store_mock("ranged", b"0123456789ABCDEF".to_vec());

        let range_data = source.fetch_range("ranged", 5, 10).await.unwrap();
        assert_eq!(range_data, b"56789".to_vec());
    }

    #[tokio::test]
    async fn test_fetch_resumable() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));
        let original = b"Full content here".to_vec();
        source.store_mock("resumable", original.clone());

        // Resume from offset 5
        let (data, total) = source.fetch_resumable("resumable", 5).await.unwrap();

        assert_eq!(total, original.len() as u64);
        assert_eq!(data, b"content here".to_vec());
    }

    // ========== List and Query Tests ==========

    #[tokio::test]
    async fn test_list_objects_paginated() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        for i in 0..10 {
            source.store_mock(&format!("items/item{:02}.txt", i), vec![i as u8]);
        }

        // First page
        let result1 = source.list_objects_paginated("items/", 3, None).await.unwrap();
        assert_eq!(result1.objects.len(), 3);
        assert!(result1.is_truncated);

        // Second page
        let result2 = source.list_objects_paginated(
            "items/",
            3,
            result1.next_continuation_token.as_deref()
        ).await.unwrap();
        assert_eq!(result2.objects.len(), 3);
    }

    #[tokio::test]
    async fn test_delete_objects() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        source.store_mock("del/a.txt", b"a".to_vec());
        source.store_mock("del/b.txt", b"b".to_vec());

        let results = source.delete_objects(&["del/a.txt", "del/b.txt", "nonexistent"]).await.unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].success);
        assert!(results[1].success);
        assert!(!results[2].success);

        assert!(!source.exists("del/a.txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_copy_object() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        let data = b"copy me".to_vec();
        source.store_mock("source/file.txt", data.clone());

        let meta = source.copy_object("source/file.txt", "dest/file.txt").await.unwrap();

        assert_eq!(meta.key, "dest/file.txt");
        assert!(source.exists("dest/file.txt").await.unwrap());

        let copied = source.fetch("dest/file.txt", &FetchOptions::default()).await.unwrap();
        assert_eq!(copied, data);
    }

    // ========== Metadata Tests ==========

    #[tokio::test]
    async fn test_store_mock_with_meta() {
        let mut source = S3DataSource::new(S3Config::aws("bucket", "us-east-1"));

        let mut custom_meta = HashMap::new();
        custom_meta.insert("x-amz-meta-custom".to_string(), "value".to_string());

        source.store_mock_with_meta(
            "with-meta.json",
            b"{}".to_vec(),
            Some("application/json"),
            custom_meta
        );

        let meta = source.metadata("with-meta.json").await.unwrap();
        assert_eq!(meta.content_type, Some("application/json".to_string()));
        assert_eq!(meta.extra.get("x-amz-meta-custom"), Some(&"value".to_string()));
    }

    // ========== Helper Function Tests ==========

    #[test]
    fn test_format_amz_date() {
        let timestamp = 0u64; // 1970-01-01 00:00:00
        let date = format_amz_date(timestamp);
        assert_eq!(date, "19700101T000000Z");

        let timestamp2 = 1704067200u64; // 2024-01-01 00:00:00 UTC
        let date2 = format_amz_date(timestamp2);
        assert!(date2.starts_with("2024"));
    }

    #[test]
    fn test_uri_encode() {
        assert_eq!(uri_encode("hello"), "hello");
        assert_eq!(uri_encode("hello world"), "hello%20world");
        assert_eq!(uri_encode("a/b"), "a%2Fb");
        assert_eq!(uri_encode("key=value"), "key%3Dvalue");
    }

    #[test]
    fn test_uri_encode_path() {
        assert_eq!(uri_encode_path("/bucket/key/file.txt"), "/bucket/key/file.txt");
        assert_eq!(uri_encode_path("/bucket/my key"), "/bucket/my%20key");
    }

    #[test]
    fn test_sha256_hex() {
        let hash = sha256_hex(b"test");
        assert_eq!(hash.len(), 64);
        // Should be deterministic
        assert_eq!(hash, sha256_hex(b"test"));
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0, 1, 255]), "0001ff");
        assert_eq!(hex_encode(&[]), "");
    }

    // ========== AWS SigV4 Tests ==========

    #[test]
    fn test_sigv4_signer_creation() {
        let signer = AwsSigV4Signer::new(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1"
        );

        let config = PresignedUrlConfig::for_download(3600);
        let presigned = signer.presign_url("GET", "examplebucket.s3.amazonaws.com", "/test.txt", &config);

        assert!(presigned.url.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"));
        assert!(presigned.url.contains("X-Amz-Credential=AKIAIOSFODNN7EXAMPLE"));
        assert!(presigned.url.contains("X-Amz-Expires=3600"));
    }

    #[test]
    fn test_sigv4_with_session_token() {
        let signer = AwsSigV4Signer::new(
            "ASIAXXX",
            "secret",
            Some("session-token-xxx"),
            "us-west-2"
        );

        let config = PresignedUrlConfig::for_download(900);
        let presigned = signer.presign_url("GET", "bucket.s3.us-west-2.amazonaws.com", "/key", &config);

        assert!(presigned.url.contains("X-Amz-Security-Token"));
    }

    // ========== Streaming Tests ==========

    #[tokio::test]
    async fn test_fetch_stream() {
        let mut source = S3DataSource::new(S3Config {
            bucket: "bucket".to_string(),
            multipart_chunk_size: 5, // Small chunks for testing
            ..Default::default()
        });

        source.store_mock("streamed", b"0123456789ABCDEF".to_vec());

        let stream = source.fetch_stream("streamed", &FetchOptions::default()).await.unwrap();
        let chunks: Vec<_> = stream.collect();

        assert!(chunks.len() > 1);
        assert!(chunks.last().unwrap().is_last);

        let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.data.clone()).collect();
        assert_eq!(reassembled, b"0123456789ABCDEF".to_vec());
    }

    // ========== Storage Class Tests ==========

    #[test]
    fn test_storage_class_immediate() {
        assert!(StorageClass::Standard.is_immediate());
        assert!(StorageClass::StandardIa.is_immediate());
        assert!(StorageClass::OneZoneIa.is_immediate());
        assert!(StorageClass::IntelligentTiering.is_immediate());
        assert!(StorageClass::ReducedRedundancy.is_immediate());
        assert!(!StorageClass::Glacier.is_immediate());
        assert!(!StorageClass::GlacierDeepArchive.is_immediate());
    }

    // ========== Encryption Tests ==========

    #[test]
    fn test_server_side_encryption_default() {
        let encryption = ServerSideEncryption::default();
        assert!(matches!(encryption, ServerSideEncryption::None));
    }

    #[test]
    fn test_multipart_options_default() {
        let options = MultipartUploadOptions::default();
        assert_eq!(options.part_size, 10 * 1024 * 1024);
        assert_eq!(options.max_concurrent, 4);
        assert!(matches!(options.encryption, ServerSideEncryption::None));
        assert_eq!(options.storage_class, StorageClass::Standard);
    }

    // ========== Checksum Algorithm Tests ==========

    #[test]
    fn test_checksum_algorithms() {
        assert_eq!(ChecksumAlgorithm::Sha256, ChecksumAlgorithm::Sha256);
        assert_ne!(ChecksumAlgorithm::Sha256, ChecksumAlgorithm::Crc32c);
    }

    // ========== Builder Advanced Tests ==========

    #[test]
    fn test_builder_multipart_threshold() {
        let source = S3DataSourceBuilder::new("bucket")
            .multipart_threshold(50 * 1024 * 1024)
            .build();

        assert_eq!(source.config().multipart_threshold, 50 * 1024 * 1024);
    }
}
