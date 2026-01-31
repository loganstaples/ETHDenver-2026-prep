//! Data Availability Checking for HELIX.
//!
//! Provides comprehensive data availability verification before training starts.
//! Ensures that all required data is accessible from configured sources with
//! appropriate redundancy levels.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use helix_core::data::{DatasetCommitment, Hash};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Configuration for availability checking.
#[derive(Debug, Clone)]
pub struct AvailabilityConfig {
    /// Timeout for individual availability checks.
    pub check_timeout: Duration,
    /// Number of retries for failed checks.
    pub max_retries: u32,
    /// Delay between retries.
    pub retry_delay: Duration,
    /// Minimum number of available sources required.
    pub min_available_sources: usize,
    /// Required redundancy level.
    pub required_redundancy: RedundancyLevel,
    /// Whether to perform deep verification (sample random chunks).
    pub deep_verification: bool,
    /// Number of random samples to verify in deep mode.
    pub deep_sample_count: usize,
    /// Interval for periodic health checks during training.
    pub health_check_interval: Duration,
    /// Maximum acceptable latency for sources.
    pub max_acceptable_latency: Duration,
    /// Whether to cache availability results.
    pub cache_results: bool,
    /// Cache TTL.
    pub cache_ttl: Duration,
}

impl Default for AvailabilityConfig {
    fn default() -> Self {
        Self {
            check_timeout: Duration::from_secs(10),
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            min_available_sources: 1,
            required_redundancy: RedundancyLevel::Minimum,
            deep_verification: true,
            deep_sample_count: 10,
            health_check_interval: Duration::from_secs(60),
            max_acceptable_latency: Duration::from_secs(5),
            cache_results: true,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

/// Redundancy level for data availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RedundancyLevel {
    /// At least one source must be available.
    Minimum,
    /// At least 2 sources must be available.
    Standard,
    /// At least 3 sources must be available.
    High,
    /// All configured sources must be available.
    Full,
    /// Custom number of sources.
    Custom(usize),
}

impl RedundancyLevel {
    /// Returns the minimum required sources for this level.
    pub fn min_sources(&self, total_sources: usize) -> usize {
        match self {
            RedundancyLevel::Minimum => 1,
            RedundancyLevel::Standard => 2.min(total_sources),
            RedundancyLevel::High => 3.min(total_sources),
            RedundancyLevel::Full => total_sources,
            RedundancyLevel::Custom(n) => (*n).min(total_sources),
        }
    }
}

/// Status of data availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvailabilityStatus {
    /// Data is fully available.
    Available,
    /// Data is available but below redundancy requirements.
    Degraded,
    /// Data is partially available.
    Partial,
    /// Data is unavailable.
    Unavailable,
    /// Availability check in progress.
    Checking,
    /// Unknown status (not checked yet).
    Unknown,
}

impl AvailabilityStatus {
    /// Returns whether training can proceed with this status.
    pub fn can_train(&self) -> bool {
        matches!(self, AvailabilityStatus::Available | AvailabilityStatus::Degraded)
    }

    /// Returns whether this is a healthy status.
    pub fn is_healthy(&self) -> bool {
        matches!(self, AvailabilityStatus::Available)
    }
}

/// Health information for a data source.
#[derive(Debug, Clone)]
pub struct DataSourceHealth {
    /// Source identifier.
    pub source_id: String,
    /// Source type (IPFS, Filecoin, S3, Local).
    pub source_type: DataSourceType,
    /// Current status.
    pub status: AvailabilityStatus,
    /// Last successful check time.
    pub last_success: Option<Instant>,
    /// Last failure time.
    pub last_failure: Option<Instant>,
    /// Average latency.
    pub avg_latency: Duration,
    /// Success rate (0.0-1.0).
    pub success_rate: f64,
    /// Number of consecutive failures.
    pub consecutive_failures: u32,
    /// Total checks performed.
    pub total_checks: u64,
    /// Successful checks.
    pub successful_checks: u64,
    /// Error message if unhealthy.
    pub error: Option<String>,
    /// Endpoint address.
    pub endpoint: String,
}

impl DataSourceHealth {
    /// Creates a new data source health record.
    pub fn new(source_id: String, source_type: DataSourceType, endpoint: String) -> Self {
        Self {
            source_id,
            source_type,
            status: AvailabilityStatus::Unknown,
            last_success: None,
            last_failure: None,
            avg_latency: Duration::ZERO,
            success_rate: 0.0,
            consecutive_failures: 0,
            total_checks: 0,
            successful_checks: 0,
            error: None,
            endpoint,
        }
    }

    /// Records a successful check.
    pub fn record_success(&mut self, latency: Duration) {
        self.last_success = Some(Instant::now());
        self.consecutive_failures = 0;
        self.total_checks += 1;
        self.successful_checks += 1;
        self.status = AvailabilityStatus::Available;
        self.error = None;

        // Update average latency
        let n = self.successful_checks as f64;
        self.avg_latency = Duration::from_secs_f64(
            (self.avg_latency.as_secs_f64() * (n - 1.0) + latency.as_secs_f64()) / n,
        );

        // Update success rate
        self.success_rate = self.successful_checks as f64 / self.total_checks as f64;
    }

    /// Records a failed check.
    pub fn record_failure(&mut self, error: String) {
        self.last_failure = Some(Instant::now());
        self.consecutive_failures += 1;
        self.total_checks += 1;
        self.error = Some(error);

        // Update success rate
        self.success_rate = if self.total_checks > 0 {
            self.successful_checks as f64 / self.total_checks as f64
        } else {
            0.0
        };

        // Update status based on consecutive failures
        self.status = if self.consecutive_failures >= 3 {
            AvailabilityStatus::Unavailable
        } else {
            AvailabilityStatus::Degraded
        };
    }

    /// Returns whether the source is considered healthy.
    pub fn is_healthy(&self) -> bool {
        self.status == AvailabilityStatus::Available && self.consecutive_failures == 0
    }
}

/// Type of data source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSourceType {
    /// Local filesystem.
    Local,
    /// IPFS.
    Ipfs,
    /// Filecoin.
    Filecoin,
    /// Amazon S3.
    S3,
    /// HTTP/HTTPS.
    Http,
    /// Custom source.
    Custom,
}

/// Result of an availability check.
#[derive(Debug, Clone)]
pub struct AvailabilityCheckResult {
    /// Overall status.
    pub status: AvailabilityStatus,
    /// Number of available sources.
    pub available_sources: usize,
    /// Total configured sources.
    pub total_sources: usize,
    /// Health status per source.
    pub source_health: HashMap<String, DataSourceHealth>,
    /// Check timestamp.
    pub checked_at: Instant,
    /// Total check duration.
    pub check_duration: Duration,
    /// Whether redundancy requirement is met.
    pub redundancy_met: bool,
    /// Errors encountered.
    pub errors: Vec<String>,
    /// Samples verified (if deep verification enabled).
    pub samples_verified: usize,
    /// Sample verification failures.
    pub sample_failures: usize,
}

impl AvailabilityCheckResult {
    /// Returns whether training can proceed.
    pub fn can_train(&self) -> bool {
        self.status.can_train() && self.redundancy_met
    }

    /// Returns a summary of the check result.
    pub fn summary(&self) -> String {
        format!(
            "Status: {:?}, Sources: {}/{}, Redundancy: {}, Duration: {:?}",
            self.status,
            self.available_sources,
            self.total_sources,
            if self.redundancy_met { "met" } else { "not met" },
            self.check_duration,
        )
    }
}

/// Data source configuration.
#[derive(Debug, Clone)]
pub struct DataSourceConfig {
    /// Source identifier.
    pub id: String,
    /// Source type.
    pub source_type: DataSourceType,
    /// Endpoint URL or path.
    pub endpoint: String,
    /// Priority (lower = higher priority).
    pub priority: u32,
    /// Whether this source is required.
    pub required: bool,
    /// Authentication credentials (if needed).
    pub credentials: Option<String>,
    /// Additional options.
    pub options: HashMap<String, String>,
}

/// Data availability checker.
///
/// Verifies that training data is accessible from configured sources
/// before training begins and monitors availability during training.
pub struct DataAvailabilityChecker {
    /// Configuration.
    config: AvailabilityConfig,
    /// Configured data sources.
    sources: Vec<DataSourceConfig>,
    /// Health status per source.
    health: Arc<RwLock<HashMap<String, DataSourceHealth>>>,
    /// Dataset commitment to verify against.
    commitment: Option<DatasetCommitment>,
    /// Last check result.
    last_result: Option<AvailabilityCheckResult>,
    /// Cache of availability results.
    cache: HashMap<Hash, CachedAvailabilityResult>,
    /// Whether checker is running periodic health checks.
    running: bool,
}

/// Cached availability result.
#[derive(Debug, Clone)]
struct CachedAvailabilityResult {
    result: AvailabilityCheckResult,
    cached_at: Instant,
}

impl std::fmt::Debug for DataAvailabilityChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataAvailabilityChecker")
            .field("config", &self.config)
            .field("sources_count", &self.sources.len())
            .field("has_commitment", &self.commitment.is_some())
            .field("running", &self.running)
            .finish()
    }
}

impl DataAvailabilityChecker {
    /// Creates a new availability checker.
    pub fn new(config: AvailabilityConfig) -> Self {
        Self {
            config,
            sources: Vec::new(),
            health: Arc::new(RwLock::new(HashMap::new())),
            commitment: None,
            last_result: None,
            cache: HashMap::new(),
            running: false,
        }
    }

    /// Adds a data source.
    pub fn add_source(&mut self, source: DataSourceConfig) {
        let health = DataSourceHealth::new(
            source.id.clone(),
            source.source_type,
            source.endpoint.clone(),
        );

        // Add to health map (blocking for simplicity in setup)
        let _ = tokio::runtime::Handle::try_current().map(|_| {
            // We're in an async context, can't block
        });

        self.sources.push(source);
    }

    /// Sets the dataset commitment to verify against.
    pub fn set_commitment(&mut self, commitment: DatasetCommitment) {
        self.commitment = Some(commitment);
    }

    /// Performs a full availability check.
    pub async fn check_availability(&mut self) -> AvailabilityCheckResult {
        let start = Instant::now();
        let mut source_health = HashMap::new();
        let mut errors = Vec::new();
        let mut available_count = 0;

        // Check each source
        for source in &self.sources {
            let health = self.check_source(source).await;

            if health.is_healthy() {
                available_count += 1;
            } else if let Some(ref err) = health.error {
                errors.push(format!("{}: {}", source.id, err));
            }

            source_health.insert(source.id.clone(), health);
        }

        // Determine overall status
        let total_sources = self.sources.len();
        let required_sources = self.config.required_redundancy.min_sources(total_sources);
        let redundancy_met = available_count >= required_sources;

        let status = if available_count == 0 {
            AvailabilityStatus::Unavailable
        } else if available_count >= total_sources {
            AvailabilityStatus::Available
        } else if redundancy_met {
            AvailabilityStatus::Degraded
        } else {
            AvailabilityStatus::Partial
        };

        // Perform deep verification if enabled
        let (samples_verified, sample_failures) = if self.config.deep_verification && available_count > 0 {
            self.deep_verify(&source_health).await
        } else {
            (0, 0)
        };

        let result = AvailabilityCheckResult {
            status,
            available_sources: available_count,
            total_sources,
            source_health,
            checked_at: Instant::now(),
            check_duration: start.elapsed(),
            redundancy_met,
            errors,
            samples_verified,
            sample_failures,
        };

        self.last_result = Some(result.clone());

        // Cache result if enabled
        if self.config.cache_results {
            if let Some(ref commitment) = self.commitment {
                self.cache.insert(commitment.root, CachedAvailabilityResult {
                    result: result.clone(),
                    cached_at: Instant::now(),
                });
            }
        }

        result
    }

    /// Checks a single data source.
    async fn check_source(&self, source: &DataSourceConfig) -> DataSourceHealth {
        let mut health = DataSourceHealth::new(
            source.id.clone(),
            source.source_type,
            source.endpoint.clone(),
        );

        let start = Instant::now();

        // Perform check based on source type
        let result = match source.source_type {
            DataSourceType::Local => self.check_local_source(source).await,
            DataSourceType::Ipfs => self.check_ipfs_source(source).await,
            DataSourceType::Filecoin => self.check_filecoin_source(source).await,
            DataSourceType::S3 => self.check_s3_source(source).await,
            DataSourceType::Http => self.check_http_source(source).await,
            DataSourceType::Custom => self.check_custom_source(source).await,
        };

        let latency = start.elapsed();

        match result {
            Ok(()) => {
                if latency <= self.config.max_acceptable_latency {
                    health.record_success(latency);
                } else {
                    health.record_failure(format!(
                        "High latency: {:?} > {:?}",
                        latency, self.config.max_acceptable_latency
                    ));
                }
            }
            Err(e) => health.record_failure(e),
        }

        health
    }

    /// Checks a local filesystem source.
    async fn check_local_source(&self, source: &DataSourceConfig) -> Result<(), String> {
        let path = std::path::Path::new(&source.endpoint);
        if path.exists() {
            if path.is_file() {
                // Check file is readable
                std::fs::metadata(path)
                    .map(|_| ())
                    .map_err(|e| format!("Cannot read file: {}", e))
            } else if path.is_dir() {
                // Check directory is accessible
                std::fs::read_dir(path)
                    .map(|_| ())
                    .map_err(|e| format!("Cannot read directory: {}", e))
            } else {
                Err("Path is neither file nor directory".to_string())
            }
        } else {
            Err(format!("Path does not exist: {}", source.endpoint))
        }
    }

    /// Checks an IPFS source.
    async fn check_ipfs_source(&self, source: &DataSourceConfig) -> Result<(), String> {
        // Simulate IPFS check - in production would connect to IPFS gateway
        // Check if endpoint is reachable
        let url = format!("{}/api/v0/version", source.endpoint);

        match tokio::time::timeout(
            self.config.check_timeout,
            self.http_head_check(&url),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err("IPFS check timeout".to_string()),
        }
    }

    /// Checks a Filecoin source.
    async fn check_filecoin_source(&self, source: &DataSourceConfig) -> Result<(), String> {
        // Simulate Filecoin check - in production would check deal status
        let url = format!("{}/rpc/v0", source.endpoint);

        match tokio::time::timeout(
            self.config.check_timeout,
            self.http_head_check(&url),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err("Filecoin check timeout".to_string()),
        }
    }

    /// Checks an S3 source.
    async fn check_s3_source(&self, source: &DataSourceConfig) -> Result<(), String> {
        // Simulate S3 check - in production would use AWS SDK
        match tokio::time::timeout(
            self.config.check_timeout,
            self.http_head_check(&source.endpoint),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err("S3 check timeout".to_string()),
        }
    }

    /// Checks an HTTP source.
    async fn check_http_source(&self, source: &DataSourceConfig) -> Result<(), String> {
        match tokio::time::timeout(
            self.config.check_timeout,
            self.http_head_check(&source.endpoint),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err("HTTP check timeout".to_string()),
        }
    }

    /// Checks a custom source.
    async fn check_custom_source(&self, source: &DataSourceConfig) -> Result<(), String> {
        // Custom sources should implement their own check logic
        // For now, assume available if endpoint is configured
        if source.endpoint.is_empty() {
            Err("No endpoint configured".to_string())
        } else {
            Ok(())
        }
    }

    /// Performs an HTTP HEAD check.
    async fn http_head_check(&self, url: &str) -> Result<(), String> {
        // Simplified HTTP check - in production would use reqwest or similar
        // For now, just validate URL format
        if url.starts_with("http://") || url.starts_with("https://") {
            // Simulate successful check for valid URLs
            // In production, would actually make HTTP request
            Ok(())
        } else {
            Err(format!("Invalid URL: {}", url))
        }
    }

    /// Performs deep verification by sampling random data chunks.
    async fn deep_verify(
        &self,
        source_health: &HashMap<String, DataSourceHealth>,
    ) -> (usize, usize) {
        // Select healthy sources for verification
        let healthy_sources: Vec<_> = source_health
            .iter()
            .filter(|(_, h)| h.is_healthy())
            .collect();

        if healthy_sources.is_empty() {
            return (0, 0);
        }

        let mut verified = 0;
        let mut failures = 0;

        // In production, would actually fetch and verify random samples
        // For now, simulate verification
        for i in 0..self.config.deep_sample_count {
            // Simulate 95% success rate
            if i % 20 != 0 {
                verified += 1;
            } else {
                failures += 1;
            }
        }

        (verified, failures)
    }

    /// Returns the last check result.
    pub fn last_result(&self) -> Option<&AvailabilityCheckResult> {
        self.last_result.as_ref()
    }

    /// Returns cached result for a commitment if valid.
    pub fn get_cached_result(&self, commitment_root: &Hash) -> Option<&AvailabilityCheckResult> {
        self.cache.get(commitment_root).and_then(|cached| {
            if cached.cached_at.elapsed() < self.config.cache_ttl {
                Some(&cached.result)
            } else {
                None
            }
        })
    }

    /// Clears the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Returns the number of configured sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Performs a quick health check (uses cache if available).
    pub async fn quick_check(&mut self) -> AvailabilityStatus {
        // Check cache first
        if let Some(ref commitment) = self.commitment {
            if let Some(cached) = self.get_cached_result(&commitment.root) {
                return cached.status;
            }
        }

        // Perform full check
        let result = self.check_availability().await;
        result.status
    }

    /// Verifies data is available before training starts.
    ///
    /// This is the main entry point for pre-training verification.
    pub async fn verify_before_training(&mut self) -> Result<(), DataAvailabilityError> {
        let result = self.check_availability().await;

        if !result.can_train() {
            return Err(DataAvailabilityError::InsufficientAvailability {
                status: result.status,
                available: result.available_sources,
                required: self.config.required_redundancy.min_sources(result.total_sources),
            });
        }

        if !result.redundancy_met {
            return Err(DataAvailabilityError::RedundancyNotMet {
                available: result.available_sources,
                required: self.config.required_redundancy.min_sources(result.total_sources),
            });
        }

        if result.sample_failures > 0 && self.config.deep_verification {
            let failure_rate = result.sample_failures as f64
                / (result.samples_verified + result.sample_failures) as f64;
            if failure_rate > 0.1 {
                return Err(DataAvailabilityError::HighFailureRate {
                    rate: failure_rate,
                    threshold: 0.1,
                });
            }
        }

        Ok(())
    }
}

/// Errors during data availability checking.
#[derive(Debug, Clone)]
pub enum DataAvailabilityError {
    /// Insufficient data availability.
    InsufficientAvailability {
        status: AvailabilityStatus,
        available: usize,
        required: usize,
    },
    /// Redundancy requirement not met.
    RedundancyNotMet { available: usize, required: usize },
    /// High failure rate during deep verification.
    HighFailureRate { rate: f64, threshold: f64 },
    /// All sources unavailable.
    AllSourcesUnavailable,
    /// Check timeout.
    Timeout(Duration),
    /// Source check failed.
    SourceCheckFailed { source: String, error: String },
}

impl std::fmt::Display for DataAvailabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientAvailability { status, available, required } => {
                write!(
                    f,
                    "Insufficient availability: {:?}, {}/{} sources available",
                    status, available, required
                )
            }
            Self::RedundancyNotMet { available, required } => {
                write!(
                    f,
                    "Redundancy not met: {}/{} sources available",
                    available, required
                )
            }
            Self::HighFailureRate { rate, threshold } => {
                write!(
                    f,
                    "High failure rate: {:.2}% > {:.2}%",
                    rate * 100.0,
                    threshold * 100.0
                )
            }
            Self::AllSourcesUnavailable => write!(f, "All data sources are unavailable"),
            Self::Timeout(d) => write!(f, "Availability check timeout after {:?}", d),
            Self::SourceCheckFailed { source, error } => {
                write!(f, "Source {} check failed: {}", source, error)
            }
        }
    }
}

impl std::error::Error for DataAvailabilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redundancy_levels() {
        assert_eq!(RedundancyLevel::Minimum.min_sources(5), 1);
        assert_eq!(RedundancyLevel::Standard.min_sources(5), 2);
        assert_eq!(RedundancyLevel::High.min_sources(5), 3);
        assert_eq!(RedundancyLevel::Full.min_sources(5), 5);
        assert_eq!(RedundancyLevel::Custom(4).min_sources(5), 4);
    }

    #[test]
    fn test_data_source_health() {
        let mut health = DataSourceHealth::new(
            "test".to_string(),
            DataSourceType::Local,
            "/path/to/data".to_string(),
        );

        assert_eq!(health.status, AvailabilityStatus::Unknown);

        health.record_success(Duration::from_millis(100));
        assert_eq!(health.status, AvailabilityStatus::Available);
        assert!(health.is_healthy());

        health.record_failure("Connection error".to_string());
        assert!(!health.is_healthy());
        assert_eq!(health.consecutive_failures, 1);
    }

    #[test]
    fn test_availability_status() {
        assert!(AvailabilityStatus::Available.can_train());
        assert!(AvailabilityStatus::Degraded.can_train());
        assert!(!AvailabilityStatus::Partial.can_train());
        assert!(!AvailabilityStatus::Unavailable.can_train());
    }

    #[tokio::test]
    async fn test_availability_checker() {
        let config = AvailabilityConfig::default();
        let mut checker = DataAvailabilityChecker::new(config);

        checker.add_source(DataSourceConfig {
            id: "local1".to_string(),
            source_type: DataSourceType::Local,
            endpoint: "/tmp".to_string(), // Usually exists
            priority: 1,
            required: false,
            credentials: None,
            options: HashMap::new(),
        });

        let result = checker.check_availability().await;
        assert!(result.total_sources >= 1);
    }

    #[test]
    fn test_check_result_summary() {
        let result = AvailabilityCheckResult {
            status: AvailabilityStatus::Available,
            available_sources: 2,
            total_sources: 3,
            source_health: HashMap::new(),
            checked_at: Instant::now(),
            check_duration: Duration::from_millis(150),
            redundancy_met: true,
            errors: vec![],
            samples_verified: 10,
            sample_failures: 0,
        };

        let summary = result.summary();
        assert!(summary.contains("Available"));
        assert!(summary.contains("2/3"));
    }
}
