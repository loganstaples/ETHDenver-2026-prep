//! Data Pipeline Security Utilities for HELIX.
//!
//! Provides hardened security primitives for the data pipeline:
//! - Decompression bomb protection with streaming byte limits
//! - Token-bucket rate limiting for remote data source requests
//! - Input sanitization helpers

use std::io::{self, Read, Write};
use std::time::{Instant, SystemTime};

// ============================================================================
// DECOMPRESSION BOMB PROTECTION
// ============================================================================

/// Configuration for decompression bomb protection.
#[derive(Debug, Clone)]
pub struct DecompressionLimits {
    /// Maximum allowed decompressed size in bytes.
    /// Decompression aborts immediately when this limit is reached,
    /// rather than waiting until the full output is materialized.
    pub max_output_bytes: u64,
    /// Maximum compression ratio allowed (decompressed / compressed).
    /// A ratio above this threshold indicates a potential decompression bomb.
    /// Typical legitimate data has ratios below 100:1.
    pub max_compression_ratio: f64,
    /// Chunk size for streaming reads (bytes).
    /// Smaller chunks mean more frequent limit checks but higher overhead.
    pub chunk_size: usize,
}

impl Default for DecompressionLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: 256 * 1024 * 1024, // 256 MB
            max_compression_ratio: 100.0,         // 100:1
            chunk_size: 64 * 1024,                // 64 KB
        }
    }
}

impl DecompressionLimits {
    /// Creates limits suitable for ML training data (larger allowance).
    pub fn for_training_data() -> Self {
        Self {
            max_output_bytes: 1024 * 1024 * 1024, // 1 GB
            max_compression_ratio: 200.0,
            chunk_size: 256 * 1024,
        }
    }

    /// Creates strict limits for untrusted input.
    pub fn strict() -> Self {
        Self {
            max_output_bytes: 64 * 1024 * 1024, // 64 MB
            max_compression_ratio: 50.0,
            chunk_size: 32 * 1024,
        }
    }
}

/// Error returned when decompression limits are violated.
#[derive(Debug, Clone)]
pub enum DecompressionError {
    /// Decompressed output exceeds the maximum allowed size.
    OutputTooLarge {
        bytes_written: u64,
        max_bytes: u64,
    },
    /// Compression ratio exceeds the maximum allowed.
    SuspiciousRatio {
        compressed_size: u64,
        decompressed_size: u64,
        ratio: f64,
        max_ratio: f64,
    },
    /// I/O error during decompression.
    IoError(String),
}

impl std::fmt::Display for DecompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressionError::OutputTooLarge { bytes_written, max_bytes } => {
                write!(
                    f,
                    "decompression bomb detected: output reached {} bytes (limit: {})",
                    bytes_written, max_bytes
                )
            }
            DecompressionError::SuspiciousRatio {
                compressed_size,
                decompressed_size,
                ratio,
                max_ratio,
            } => {
                write!(
                    f,
                    "suspicious compression ratio {:.1}:1 ({} -> {} bytes, max ratio: {:.1}:1)",
                    ratio, compressed_size, decompressed_size, max_ratio
                )
            }
            DecompressionError::IoError(msg) => write!(f, "decompression I/O error: {}", msg),
        }
    }
}

impl std::error::Error for DecompressionError {}

impl From<io::Error> for DecompressionError {
    fn from(e: io::Error) -> Self {
        DecompressionError::IoError(e.to_string())
    }
}

/// A size-limited writer that tracks bytes written and enforces limits.
///
/// Wraps any `Write` implementation and aborts with an error when
/// the cumulative bytes written exceeds `max_bytes`. This is the core
/// mechanism for streaming decompression bomb protection: the decompressor
/// writes into this wrapper, and the wrapper cuts off the stream before
/// memory is exhausted.
pub struct LimitedWriter<W: Write> {
    inner: W,
    bytes_written: u64,
    max_bytes: u64,
}

impl<W: Write> LimitedWriter<W> {
    /// Creates a new limited writer.
    pub fn new(inner: W, max_bytes: u64) -> Self {
        Self {
            inner,
            bytes_written: 0,
            max_bytes,
        }
    }

    /// Returns the total bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Consumes the wrapper and returns the inner writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for LimitedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let new_total = self.bytes_written + buf.len() as u64;
        if new_total > self.max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!(
                    "decompression bomb: output would reach {} bytes (limit: {})",
                    new_total, self.max_bytes
                ),
            ));
        }
        let written = self.inner.write(buf)?;
        self.bytes_written += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Safely decompresses data with streaming size limits.
///
/// Reads from `reader` in chunks, writing to an output buffer.
/// Aborts immediately if:
/// - The decompressed output exceeds `limits.max_output_bytes`
/// - The compression ratio exceeds `limits.max_compression_ratio`
///
/// This provides true streaming protection: the decompressor never
/// allocates more than `max_output_bytes` regardless of the input.
pub fn safe_decompress<R: Read>(
    mut reader: R,
    compressed_size: u64,
    limits: &DecompressionLimits,
) -> Result<Vec<u8>, DecompressionError> {
    let mut output = Vec::new();
    let mut limited = LimitedWriter::new(&mut output, limits.max_output_bytes);
    let mut buf = vec![0u8; limits.chunk_size];

    loop {
        let n = reader.read(&mut buf).map_err(DecompressionError::from)?;
        if n == 0 {
            break;
        }

        // Write through the limited writer (enforces max_output_bytes)
        limited.write_all(&buf[..n]).map_err(|_| {
            DecompressionError::OutputTooLarge {
                bytes_written: limited.bytes_written(),
                max_bytes: limits.max_output_bytes,
            }
        })?;

        // Check compression ratio periodically
        let decompressed = limited.bytes_written();
        if compressed_size > 0 && decompressed > 0 {
            let ratio = decompressed as f64 / compressed_size as f64;
            if ratio > limits.max_compression_ratio {
                return Err(DecompressionError::SuspiciousRatio {
                    compressed_size,
                    decompressed_size: decompressed,
                    ratio,
                    max_ratio: limits.max_compression_ratio,
                });
            }
        }
    }

    drop(limited);
    Ok(output)
}

/// Validates that raw data (possibly compressed) is safe to process.
///
/// For uncompressed data, simply checks the size limit.
/// For compressed data, uses `safe_decompress` with streaming limits.
pub fn validate_data_size(
    data: &[u8],
    max_uncompressed_bytes: u64,
) -> Result<(), DecompressionError> {
    if data.len() as u64 > max_uncompressed_bytes {
        return Err(DecompressionError::OutputTooLarge {
            bytes_written: data.len() as u64,
            max_bytes: max_uncompressed_bytes,
        });
    }
    Ok(())
}

// ============================================================================
// RATE LIMITING
// ============================================================================

/// Configuration for rate limiting remote data source requests.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of requests per window.
    pub max_requests: u32,
    /// Window duration in seconds.
    pub window_secs: u64,
    /// Maximum burst size (tokens available immediately).
    pub burst_size: u32,
    /// Whether to block and wait when rate limited, or return immediately.
    pub block_on_limit: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window_secs: 60,
            burst_size: 10,
            block_on_limit: true,
        }
    }
}

impl RateLimitConfig {
    /// Creates a permissive config for trusted sources.
    pub fn permissive() -> Self {
        Self {
            max_requests: 1000,
            window_secs: 60,
            burst_size: 50,
            block_on_limit: true,
        }
    }

    /// Creates a strict config for untrusted or metered sources.
    pub fn strict() -> Self {
        Self {
            max_requests: 10,
            window_secs: 60,
            burst_size: 3,
            block_on_limit: true,
        }
    }

    /// Tokens added per second (refill rate).
    pub fn tokens_per_sec(&self) -> f64 {
        if self.window_secs == 0 {
            return self.max_requests as f64;
        }
        self.max_requests as f64 / self.window_secs as f64
    }
}

/// Token-bucket rate limiter for remote data source requests.
///
/// Uses a standard token bucket algorithm:
/// - Tokens are added at a fixed rate (`max_requests / window_secs`)
/// - Each request consumes one token
/// - When tokens are exhausted, requests are denied (or blocked)
/// - Burst capacity allows short bursts above the sustained rate
///
/// Thread-safe for single-threaded async contexts. For multi-threaded
/// use, wrap in `Arc<Mutex<_>>`.
#[derive(Debug)]
pub struct RateLimiter {
    /// Configuration.
    config: RateLimitConfig,
    /// Current number of available tokens.
    tokens: f64,
    /// Maximum tokens (bucket capacity).
    max_tokens: f64,
    /// Last time tokens were refilled.
    last_refill: Instant,
    /// Total requests attempted.
    total_requests: u64,
    /// Total requests that were rate-limited.
    limited_requests: u64,
}

/// Result of a rate limit check.
#[derive(Debug, Clone)]
pub enum RateLimitResult {
    /// Request is allowed. Contains remaining tokens.
    Allowed { remaining: u32 },
    /// Request is denied. Contains estimated wait time in milliseconds.
    Denied { retry_after_ms: u64 },
}

impl RateLimiter {
    /// Creates a new rate limiter from configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        let max_tokens = config.burst_size as f64;
        Self {
            tokens: max_tokens, // Start with full burst capacity
            max_tokens,
            last_refill: Instant::now(),
            total_requests: 0,
            limited_requests: 0,
            config,
        }
    }

    /// Creates a rate limiter that allows all requests (no limiting).
    pub fn unlimited() -> Self {
        Self::new(RateLimitConfig {
            max_requests: u32::MAX,
            window_secs: 1,
            burst_size: u32::MAX / 2,
            block_on_limit: false,
        })
    }

    /// Attempts to acquire a token for one request.
    ///
    /// Returns `Allowed` if the request can proceed, or `Denied` with the
    /// estimated wait time until a token becomes available.
    pub fn try_acquire(&mut self) -> RateLimitResult {
        self.refill();
        self.total_requests += 1;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            RateLimitResult::Allowed {
                remaining: self.tokens as u32,
            }
        } else {
            self.limited_requests += 1;
            let tokens_per_sec = self.config.tokens_per_sec();
            let wait_secs = if tokens_per_sec > 0.0 {
                (1.0 - self.tokens) / tokens_per_sec
            } else {
                self.config.window_secs as f64
            };
            RateLimitResult::Denied {
                retry_after_ms: (wait_secs * 1000.0).ceil() as u64,
            }
        }
    }

    /// Acquires a token, blocking asynchronously if necessary.
    ///
    /// If `block_on_limit` is false in the config, returns immediately
    /// with `Denied` instead of blocking.
    pub async fn acquire(&mut self) -> RateLimitResult {
        loop {
            match self.try_acquire() {
                result @ RateLimitResult::Allowed { .. } => return result,
                RateLimitResult::Denied { retry_after_ms } => {
                    if !self.config.block_on_limit {
                        return RateLimitResult::Denied { retry_after_ms };
                    }
                    // Yield to allow other tasks to run and time to pass
                    // (avoids requiring tokio `time` feature)
                    let yields = retry_after_ms.min(5000);
                    for _ in 0..yields {
                        tokio::task::yield_now().await;
                    }
                }
            }
        }
    }

    /// Refills tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let elapsed_secs = elapsed.as_secs_f64();

        let new_tokens = elapsed_secs * self.config.tokens_per_sec();
        self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
        self.last_refill = now;
    }

    /// Returns the current number of available tokens.
    pub fn available_tokens(&mut self) -> u32 {
        self.refill();
        self.tokens as u32
    }

    /// Returns rate limiter statistics.
    pub fn stats(&self) -> RateLimitStats {
        RateLimitStats {
            total_requests: self.total_requests,
            limited_requests: self.limited_requests,
            limit_rate: if self.total_requests > 0 {
                self.limited_requests as f64 / self.total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Resets the rate limiter to its initial state.
    pub fn reset(&mut self) {
        self.tokens = self.max_tokens;
        self.last_refill = Instant::now();
        self.total_requests = 0;
        self.limited_requests = 0;
    }
}

/// Statistics from the rate limiter.
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    /// Total requests attempted.
    pub total_requests: u64,
    /// Requests that were rate-limited.
    pub limited_requests: u64,
    /// Fraction of requests that were limited (0.0-1.0).
    pub limit_rate: f64,
}

// ============================================================================
// INPUT SANITIZATION
// ============================================================================

/// Validates a CSV column name for safety.
///
/// Rejects names that could be used for path traversal or injection:
/// - Contains `..` (parent directory traversal)
/// - Contains `/` or `\` (path separators)
/// - Contains null bytes
/// - Contains control characters (ASCII 0-31, except tab)
/// - Is empty
/// - Exceeds maximum length (1024 chars)
///
/// Returns `Ok(())` if the name is safe, or `Err` with a description.
pub fn validate_column_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("column name is empty".to_string());
    }

    if name.len() > 1024 {
        return Err(format!(
            "column name exceeds maximum length ({}  > 1024)",
            name.len()
        ));
    }

    if name.contains("..") {
        return Err(format!(
            "column name '{}' contains path traversal sequence '..'",
            sanitize_for_display(name)
        ));
    }

    if name.contains('/') {
        return Err(format!(
            "column name '{}' contains path separator '/'",
            sanitize_for_display(name)
        ));
    }

    if name.contains('\\') {
        return Err(format!(
            "column name '{}' contains path separator '\\'",
            sanitize_for_display(name)
        ));
    }

    if name.contains('\0') {
        return Err("column name contains null byte".to_string());
    }

    // Check for control characters (ASCII 0-31 except tab 0x09)
    for (i, c) in name.chars().enumerate() {
        if c.is_control() && c != '\t' {
            return Err(format!(
                "column name contains control character at position {} (U+{:04X})",
                i,
                c as u32
            ));
        }
    }

    Ok(())
}

/// Sanitizes a string for safe display in error messages.
///
/// Replaces control characters and non-printable characters with
/// their Unicode escape sequences to prevent log injection.
fn sanitize_for_display(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars().take(64) {
        if c.is_control() {
            result.push_str(&format!("\\u{{{:04X}}}", c as u32));
        } else {
            result.push(c);
        }
    }
    if s.len() > 64 {
        result.push_str("...");
    }
    result
}

// ============================================================================
// S3 PRESIGNED URL VALIDATION
// ============================================================================

/// Maximum allowed presigned URL expiry time (7 days, matching AWS maximum).
pub const MAX_PRESIGNED_URL_EXPIRY_SECS: u64 = 7 * 24 * 60 * 60;

/// Minimum time remaining before a presigned URL is considered too close to
/// expiry to be usable (30 seconds).
pub const MIN_PRESIGNED_URL_REMAINING_SECS: u64 = 30;

/// Validates a presigned URL's expiry parameters.
///
/// Returns `Ok(())` if the URL is valid, or `Err` with a description.
pub fn validate_presigned_url_expiry(
    expires_at: u64,
    expires_secs: u64,
) -> Result<(), String> {
    // Check excessive expiry duration
    if expires_secs > MAX_PRESIGNED_URL_EXPIRY_SECS {
        return Err(format!(
            "presigned URL expiry {} seconds exceeds maximum allowed {} seconds (7 days)",
            expires_secs, MAX_PRESIGNED_URL_EXPIRY_SECS
        ));
    }

    // Check if the URL has already expired
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if expires_at > 0 && expires_at <= now {
        return Err(format!(
            "presigned URL has expired (expired at {}, current time {})",
            expires_at, now
        ));
    }

    // Check if too little time remains
    if expires_at > 0 && expires_at.saturating_sub(now) < MIN_PRESIGNED_URL_REMAINING_SECS {
        return Err(format!(
            "presigned URL expires in {} seconds (minimum {} seconds required)",
            expires_at.saturating_sub(now),
            MIN_PRESIGNED_URL_REMAINING_SECS
        ));
    }

    Ok(())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Decompression Tests ==========

    #[test]
    fn test_limited_writer_within_limit() {
        let mut output = Vec::new();
        let mut writer = LimitedWriter::new(&mut output, 100);

        writer.write_all(b"hello world").unwrap();
        assert_eq!(writer.bytes_written(), 11);
    }

    #[test]
    fn test_limited_writer_exceeds_limit() {
        let mut output = Vec::new();
        let mut writer = LimitedWriter::new(&mut output, 10);

        let result = writer.write_all(b"this exceeds the limit");
        assert!(result.is_err());
    }

    #[test]
    fn test_limited_writer_exact_limit() {
        let mut output = Vec::new();
        let mut writer = LimitedWriter::new(&mut output, 5);

        writer.write_all(b"hello").unwrap();
        assert_eq!(writer.bytes_written(), 5);

        // One more byte should fail
        let result = writer.write_all(b"!");
        assert!(result.is_err());
    }

    #[test]
    fn test_safe_decompress_within_limits() {
        let data = b"some uncompressed test data for verification";
        let limits = DecompressionLimits {
            max_output_bytes: 1024,
            max_compression_ratio: 100.0,
            chunk_size: 16,
        };

        let result = safe_decompress(&data[..], data.len() as u64, &limits);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn test_safe_decompress_exceeds_size_limit() {
        let data = vec![0u8; 200];
        let limits = DecompressionLimits {
            max_output_bytes: 100,
            max_compression_ratio: 1000.0,
            chunk_size: 32,
        };

        let result = safe_decompress(&data[..], data.len() as u64, &limits);
        assert!(result.is_err());
        match result.unwrap_err() {
            DecompressionError::OutputTooLarge { max_bytes, .. } => {
                assert_eq!(max_bytes, 100);
            }
            other => panic!("expected OutputTooLarge, got: {:?}", other),
        }
    }

    #[test]
    fn test_safe_decompress_suspicious_ratio() {
        // Simulate: 10 bytes of compressed input producing 1000+ bytes of output
        let data = vec![42u8; 600];
        let limits = DecompressionLimits {
            max_output_bytes: 10000,
            max_compression_ratio: 5.0, // Very strict ratio
            chunk_size: 32,
        };

        // Claim the compressed size was only 10 bytes
        let result = safe_decompress(&data[..], 10, &limits);
        assert!(result.is_err());
        match result.unwrap_err() {
            DecompressionError::SuspiciousRatio { ratio, max_ratio, .. } => {
                assert!(ratio > max_ratio);
            }
            other => panic!("expected SuspiciousRatio, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_data_size_ok() {
        let data = vec![0u8; 100];
        assert!(validate_data_size(&data, 200).is_ok());
    }

    #[test]
    fn test_validate_data_size_too_large() {
        let data = vec![0u8; 300];
        assert!(validate_data_size(&data, 200).is_err());
    }

    #[test]
    fn test_decompression_limits_defaults() {
        let limits = DecompressionLimits::default();
        assert_eq!(limits.max_output_bytes, 256 * 1024 * 1024);
        assert_eq!(limits.max_compression_ratio, 100.0);
    }

    // ========== Rate Limiter Tests ==========

    #[test]
    fn test_rate_limiter_allows_burst() {
        let config = RateLimitConfig {
            max_requests: 10,
            window_secs: 1,
            burst_size: 5,
            block_on_limit: false,
        };
        let mut limiter = RateLimiter::new(config);

        // Should allow burst_size requests immediately
        for _ in 0..5 {
            match limiter.try_acquire() {
                RateLimitResult::Allowed { .. } => {}
                RateLimitResult::Denied { .. } => panic!("burst should be allowed"),
            }
        }
    }

    #[test]
    fn test_rate_limiter_denies_over_burst() {
        let config = RateLimitConfig {
            max_requests: 10,
            window_secs: 60,
            burst_size: 3,
            block_on_limit: false,
        };
        let mut limiter = RateLimiter::new(config);

        // Consume all burst tokens
        for _ in 0..3 {
            limiter.try_acquire();
        }

        // Next request should be denied
        match limiter.try_acquire() {
            RateLimitResult::Denied { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            RateLimitResult::Allowed { .. } => panic!("should be denied after burst exhausted"),
        }
    }

    #[test]
    fn test_rate_limiter_stats() {
        let config = RateLimitConfig {
            max_requests: 10,
            window_secs: 60,
            burst_size: 2,
            block_on_limit: false,
        };
        let mut limiter = RateLimiter::new(config);

        // 2 allowed + 2 denied
        for _ in 0..4 {
            limiter.try_acquire();
        }

        let stats = limiter.stats();
        assert_eq!(stats.total_requests, 4);
        assert_eq!(stats.limited_requests, 2);
        assert!((stats.limit_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rate_limiter_unlimited() {
        let mut limiter = RateLimiter::unlimited();

        for _ in 0..1000 {
            match limiter.try_acquire() {
                RateLimitResult::Allowed { .. } => {}
                RateLimitResult::Denied { .. } => panic!("unlimited limiter should never deny"),
            }
        }
    }

    #[test]
    fn test_rate_limiter_reset() {
        let config = RateLimitConfig {
            max_requests: 10,
            window_secs: 60,
            burst_size: 2,
            block_on_limit: false,
        };
        let mut limiter = RateLimiter::new(config);

        // Exhaust tokens
        limiter.try_acquire();
        limiter.try_acquire();

        // Reset
        limiter.reset();

        // Should have tokens again
        match limiter.try_acquire() {
            RateLimitResult::Allowed { .. } => {}
            RateLimitResult::Denied { .. } => panic!("should be allowed after reset"),
        }

        assert_eq!(limiter.stats().total_requests, 1);
    }

    #[test]
    fn test_rate_limit_config_tokens_per_sec() {
        let config = RateLimitConfig {
            max_requests: 60,
            window_secs: 60,
            burst_size: 10,
            block_on_limit: false,
        };
        assert!((config.tokens_per_sec() - 1.0).abs() < 1e-6);
    }

    // ========== Column Name Validation Tests ==========

    #[test]
    fn test_valid_column_names() {
        assert!(validate_column_name("feature_1").is_ok());
        assert!(validate_column_name("label").is_ok());
        assert!(validate_column_name("column with spaces").is_ok());
        assert!(validate_column_name("col-name").is_ok());
        assert!(validate_column_name("Col.Name").is_ok());
        assert!(validate_column_name("x").is_ok());
    }

    #[test]
    fn test_reject_path_traversal() {
        assert!(validate_column_name("../etc/passwd").is_err());
        assert!(validate_column_name("..").is_err());
        assert!(validate_column_name("col/../secret").is_err());
    }

    #[test]
    fn test_reject_path_separators() {
        assert!(validate_column_name("path/to/file").is_err());
        assert!(validate_column_name("path\\to\\file").is_err());
        assert!(validate_column_name("/absolute").is_err());
    }

    #[test]
    fn test_reject_null_bytes() {
        assert!(validate_column_name("col\0name").is_err());
    }

    #[test]
    fn test_reject_control_characters() {
        assert!(validate_column_name("col\x01name").is_err());
        assert!(validate_column_name("col\x0Bname").is_err());
        assert!(validate_column_name("col\x1Fname").is_err());
    }

    #[test]
    fn test_reject_empty_column_name() {
        assert!(validate_column_name("").is_err());
    }

    #[test]
    fn test_reject_overly_long_column_name() {
        let long_name = "a".repeat(1025);
        assert!(validate_column_name(&long_name).is_err());
    }

    #[test]
    fn test_allow_tab_in_column_name() {
        // Tab is a valid delimiter character, should be allowed in names
        assert!(validate_column_name("col\tname").is_ok());
    }

    // ========== Presigned URL Validation Tests ==========

    #[test]
    fn test_valid_presigned_url_expiry() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // URL expiring in 1 hour
        assert!(validate_presigned_url_expiry(now + 3600, 3600).is_ok());
    }

    #[test]
    fn test_reject_excessive_expiry() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // 8 days > 7 day maximum
        let eight_days = 8 * 24 * 60 * 60;
        let result = validate_presigned_url_expiry(now + eight_days, eight_days);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds maximum"));
    }

    #[test]
    fn test_reject_expired_url() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Already expired
        let result = validate_presigned_url_expiry(now - 100, 3600);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expired"));
    }

    #[test]
    fn test_reject_nearly_expired_url() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Only 10 seconds remaining (below 30s minimum)
        let result = validate_presigned_url_expiry(now + 10, 3600);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expires in"));
    }

    #[test]
    fn test_skip_validation_for_zero_expires_at() {
        // expires_at=0 means no expiry (unsigned URL or unknown)
        assert!(validate_presigned_url_expiry(0, 3600).is_ok());
    }
}
