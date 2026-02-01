//! Demo Pre-warming Module
//!
//! Ensures fast demo starts by pre-loading:
//! - Proving keys into memory
//! - Model weights
//! - Network connections
//! - GPU memory allocation

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock, Semaphore};

/// Pre-warming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrewarmConfig {
    /// Enable proving key pre-loading
    pub prewarm_proving_keys: bool,
    /// Enable model weight pre-loading
    pub prewarm_model_weights: bool,
    /// Enable network connection warming
    pub prewarm_network: bool,
    /// Enable GPU memory allocation
    pub prewarm_gpu: bool,
    /// Maximum memory to use for pre-warming (bytes)
    pub max_memory_bytes: u64,
    /// Timeout for pre-warming operations
    pub timeout: Duration,
    /// Parallel pre-warming operations
    pub parallelism: usize,
    /// Cache directory
    pub cache_dir: PathBuf,
}

impl Default for PrewarmConfig {
    fn default() -> Self {
        Self {
            prewarm_proving_keys: true,
            prewarm_model_weights: true,
            prewarm_network: true,
            prewarm_gpu: false,
            max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            timeout: Duration::from_secs(30),
            parallelism: 4,
            cache_dir: PathBuf::from(".helix/cache"),
        }
    }
}

impl PrewarmConfig {
    /// Create config for demo mode (aggressive pre-warming)
    pub fn demo_mode() -> Self {
        Self {
            prewarm_proving_keys: true,
            prewarm_model_weights: true,
            prewarm_network: true,
            prewarm_gpu: true,
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            timeout: Duration::from_secs(60),
            parallelism: 8,
            cache_dir: PathBuf::from(".helix/demo-cache"),
        }
    }

    /// Create config for minimal pre-warming
    pub fn minimal() -> Self {
        Self {
            prewarm_proving_keys: false,
            prewarm_model_weights: true,
            prewarm_network: true,
            prewarm_gpu: false,
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
            timeout: Duration::from_secs(10),
            parallelism: 2,
            cache_dir: PathBuf::from(".helix/cache"),
        }
    }
}

/// Pre-warming status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrewarmStatus {
    /// Whether pre-warming is complete
    pub complete: bool,
    /// Whether pre-warming is in progress
    pub in_progress: bool,
    /// Progress percentage (0-100)
    pub progress_percent: u8,
    /// Current operation
    pub current_operation: String,
    /// Time elapsed
    pub elapsed: Duration,
    /// Estimated time remaining
    pub estimated_remaining: Duration,
    /// Memory used (bytes)
    pub memory_used_bytes: u64,
    /// Items pre-warmed
    pub items_warmed: HashMap<String, bool>,
    /// Errors encountered
    pub errors: Vec<String>,
}

impl Default for PrewarmStatus {
    fn default() -> Self {
        Self {
            complete: false,
            in_progress: false,
            progress_percent: 0,
            current_operation: "Idle".to_string(),
            elapsed: Duration::ZERO,
            estimated_remaining: Duration::ZERO,
            memory_used_bytes: 0,
            items_warmed: HashMap::new(),
            errors: Vec::new(),
        }
    }
}

/// Pre-warming result
#[derive(Debug, Clone)]
pub struct PrewarmResult {
    /// Whether pre-warming succeeded
    pub success: bool,
    /// Total time taken
    pub total_time: Duration,
    /// Memory used
    pub memory_used_bytes: u64,
    /// Items successfully pre-warmed
    pub items_warmed: Vec<String>,
    /// Errors encountered
    pub errors: Vec<String>,
    /// Metrics
    pub metrics: PrewarmMetrics,
}

/// Pre-warming metrics
#[derive(Debug, Clone, Default)]
pub struct PrewarmMetrics {
    /// Time to load proving keys
    pub proving_key_load_ms: u64,
    /// Time to load model weights
    pub model_load_ms: u64,
    /// Time to establish connections
    pub connection_time_ms: u64,
    /// Time for GPU initialization
    pub gpu_init_ms: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Cache entry
#[derive(Debug)]
pub struct CacheEntry {
    /// Entry key
    pub key: String,
    /// Data
    pub data: Vec<u8>,
    /// Size in bytes
    pub size: usize,
    /// Last accessed
    pub last_accessed: Instant,
    /// Hit count
    pub hits: u64,
}

/// Demo pre-warmer
pub struct DemoPrewarmer {
    /// Configuration
    config: PrewarmConfig,
    /// Current status
    status: Arc<RwLock<PrewarmStatus>>,
    /// In-memory cache
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Semaphore for parallelism control
    semaphore: Arc<Semaphore>,
    /// Total memory used
    memory_used: Arc<RwLock<u64>>,
}

impl DemoPrewarmer {
    /// Create new pre-warmer
    pub fn new(config: PrewarmConfig) -> Self {
        let parallelism = config.parallelism;
        Self {
            config,
            status: Arc::new(RwLock::new(PrewarmStatus::default())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(parallelism)),
            memory_used: Arc::new(RwLock::new(0)),
        }
    }

    /// Create pre-warmer with default config
    pub fn default_prewarmer() -> Self {
        Self::new(PrewarmConfig::default())
    }

    /// Create pre-warmer for demo mode
    pub fn demo_mode() -> Self {
        Self::new(PrewarmConfig::demo_mode())
    }

    /// Get current status
    pub async fn status(&self) -> PrewarmStatus {
        self.status.read().await.clone()
    }

    /// Check if pre-warming is complete
    pub async fn is_ready(&self) -> bool {
        self.status.read().await.complete
    }

    /// Get cached data
    pub async fn get_cached(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.cache.write().await;
        if let Some(entry) = cache.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.hits += 1;
            Some(entry.data.clone())
        } else {
            None
        }
    }

    /// Add to cache
    async fn add_to_cache(&self, key: &str, data: Vec<u8>) -> Result<()> {
        let size = data.len() as u64;

        // Check memory limit
        let current_used = *self.memory_used.read().await;
        if current_used + size > self.config.max_memory_bytes {
            // Try to evict old entries
            self.evict_cache(size).await?;
        }

        let mut cache = self.cache.write().await;
        cache.insert(
            key.to_string(),
            CacheEntry {
                key: key.to_string(),
                data,
                size: size as usize,
                last_accessed: Instant::now(),
                hits: 0,
            },
        );

        *self.memory_used.write().await += size;

        Ok(())
    }

    /// Evict cache entries to free up space
    async fn evict_cache(&self, needed: u64) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut memory_used = self.memory_used.write().await;

        // Sort by last accessed (oldest first)
        let mut entries: Vec<_> = cache.values().map(|e| (e.key.clone(), e.last_accessed)).collect();
        entries.sort_by_key(|(_, accessed)| *accessed);

        let mut freed = 0u64;
        for (key, _) in entries {
            if freed >= needed {
                break;
            }
            if let Some(entry) = cache.remove(&key) {
                freed += entry.size as u64;
                *memory_used -= entry.size as u64;
            }
        }

        if freed < needed {
            return Err(anyhow!("Could not free enough memory"));
        }

        Ok(())
    }

    /// Update status
    async fn update_status(&self, operation: &str, progress: u8) {
        let mut status = self.status.write().await;
        status.current_operation = operation.to_string();
        status.progress_percent = progress;
    }

    /// Run pre-warming
    pub async fn prewarm(&self) -> Result<PrewarmResult> {
        let start = Instant::now();

        // Update status
        {
            let mut status = self.status.write().await;
            status.in_progress = true;
            status.progress_percent = 0;
            status.current_operation = "Starting pre-warm".to_string();
        }

        let mut metrics = PrewarmMetrics::default();
        let mut items_warmed = Vec::new();
        let mut errors = Vec::new();

        // Ensure cache directory exists
        std::fs::create_dir_all(&self.config.cache_dir).ok();

        // 1. Pre-warm proving keys
        if self.config.prewarm_proving_keys {
            self.update_status("Loading proving keys...", 10).await;

            match self.prewarm_proving_keys().await {
                Ok(time) => {
                    metrics.proving_key_load_ms = time.as_millis() as u64;
                    items_warmed.push("proving_keys".to_string());
                }
                Err(e) => {
                    errors.push(format!("Proving keys: {}", e));
                }
            }
        }

        // 2. Pre-warm model weights
        if self.config.prewarm_model_weights {
            self.update_status("Loading model weights...", 30).await;

            match self.prewarm_model_weights().await {
                Ok(time) => {
                    metrics.model_load_ms = time.as_millis() as u64;
                    items_warmed.push("model_weights".to_string());
                }
                Err(e) => {
                    errors.push(format!("Model weights: {}", e));
                }
            }
        }

        // 3. Pre-warm network connections
        if self.config.prewarm_network {
            self.update_status("Establishing connections...", 60).await;

            match self.prewarm_network().await {
                Ok(time) => {
                    metrics.connection_time_ms = time.as_millis() as u64;
                    items_warmed.push("network_connections".to_string());
                }
                Err(e) => {
                    errors.push(format!("Network: {}", e));
                }
            }
        }

        // 4. Pre-warm GPU
        if self.config.prewarm_gpu {
            self.update_status("Initializing GPU...", 80).await;

            match self.prewarm_gpu().await {
                Ok(time) => {
                    metrics.gpu_init_ms = time.as_millis() as u64;
                    items_warmed.push("gpu_memory".to_string());
                }
                Err(e) => {
                    errors.push(format!("GPU: {}", e));
                }
            }
        }

        // Calculate cache hit rate
        let cache = self.cache.read().await;
        let total_hits: u64 = cache.values().map(|e| e.hits).sum();
        let total_entries = cache.len() as u64;
        metrics.cache_hit_rate = if total_entries > 0 {
            total_hits as f64 / total_entries as f64
        } else {
            0.0
        };

        // Update final status
        {
            let mut status = self.status.write().await;
            status.in_progress = false;
            status.complete = errors.is_empty();
            status.progress_percent = 100;
            status.current_operation = "Complete".to_string();
            status.elapsed = start.elapsed();
            status.memory_used_bytes = *self.memory_used.read().await;

            for item in &items_warmed {
                status.items_warmed.insert(item.clone(), true);
            }
            status.errors = errors.clone();
        }

        Ok(PrewarmResult {
            success: errors.is_empty(),
            total_time: start.elapsed(),
            memory_used_bytes: *self.memory_used.read().await,
            items_warmed,
            errors,
            metrics,
        })
    }

    /// Pre-warm proving keys
    async fn prewarm_proving_keys(&self) -> Result<Duration> {
        let start = Instant::now();

        // Check for cached keys
        if self.get_cached("proving_keys_k15").await.is_some() {
            return Ok(start.elapsed());
        }

        let _permit = self.semaphore.acquire().await?;

        // Try to load proving keys from standard locations
        let key_paths = [
            self.config.cache_dir.join("proving_keys/k15.pk"),
            std::path::PathBuf::from(".helix/proving_keys/k15.pk"),
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".helix/proving_keys/k15.pk"),
        ];

        for path in &key_paths {
            if path.exists() {
                match tokio::fs::read(path).await {
                    Ok(key_data) => {
                        tracing::info!("Loaded proving key from {:?} ({} bytes)", path, key_data.len());
                        self.add_to_cache("proving_keys_k15", key_data).await?;
                        return Ok(start.elapsed());
                    }
                    Err(e) => {
                        tracing::debug!("Failed to load proving key from {:?}: {}", path, e);
                    }
                }
            }
        }

        // If no keys found on disk, create placeholder for demo mode
        tracing::info!("No proving keys found on disk, using demo mode with placeholder");
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Create placeholder data (1 MB - simulates key structure)
        let placeholder_keys = vec![0u8; 1024 * 1024];
        self.add_to_cache("proving_keys_k15", placeholder_keys).await?;

        Ok(start.elapsed())
    }

    /// Pre-warm model weights
    async fn prewarm_model_weights(&self) -> Result<Duration> {
        let start = Instant::now();

        // Check for cached weights
        if self.get_cached("demo_model_weights").await.is_some() {
            return Ok(start.elapsed());
        }

        let _permit = self.semaphore.acquire().await?;

        // Simulate loading model weights
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Create placeholder weights (real implementation would load from IPFS/disk)
        let placeholder_weights = vec![0u8; 2 * 1024 * 1024]; // 2 MB placeholder
        self.add_to_cache("demo_model_weights", placeholder_weights).await?;

        Ok(start.elapsed())
    }

    /// Pre-warm network connections
    async fn prewarm_network(&self) -> Result<Duration> {
        let start = Instant::now();

        let _permit = self.semaphore.acquire().await?;

        // Simulate establishing network connections
        // In real implementation, this would connect to peers
        tokio::time::sleep(Duration::from_millis(30)).await;

        // Mark network as warmed
        self.add_to_cache("network_warmed", vec![1]).await?;

        Ok(start.elapsed())
    }

    /// Pre-warm GPU memory
    async fn prewarm_gpu(&self) -> Result<Duration> {
        let start = Instant::now();

        let _permit = self.semaphore.acquire().await?;

        // Detect GPU availability
        let gpu_info = detect_gpu_capabilities().await;

        if gpu_info.available {
            tracing::info!(
                "GPU detected: {} ({} MB available)",
                gpu_info.device_name,
                gpu_info.memory_mb
            );

            // For real GPU initialization, we would allocate buffers here
            // In demo mode, we just record that GPU is available
            let info_bytes = serde_json::to_vec(&gpu_info).unwrap_or_else(|_| vec![1]);
            self.add_to_cache("gpu_warmed", info_bytes).await?;
        } else {
            tracing::info!("No GPU detected, proof generation will use CPU");
            // Mark as warmed even without GPU (indicates check was done)
            self.add_to_cache("gpu_warmed", vec![0]).await?;
        }

        Ok(start.elapsed())
    }

    /// Check if GPU pre-warming found available GPU
    pub async fn is_gpu_available(&self) -> bool {
        if let Some(data) = self.get_cached("gpu_warmed").await {
            // If we have more than 1 byte, it's GPU info (available)
            // If just 1 byte with value 0, GPU not available
            data.len() > 1 || (data.len() == 1 && data[0] == 1)
        } else {
            false
        }
    }
}

/// GPU capability information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuInfo {
    /// Whether GPU is available
    pub available: bool,
    /// Device name
    pub device_name: String,
    /// Available memory in MB
    pub memory_mb: u64,
    /// GPU type (cuda, metal, none)
    pub gpu_type: String,
}

/// Detect GPU capabilities
async fn detect_gpu_capabilities() -> GpuInfo {
    // Check for CUDA (NVIDIA)
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/dev/nvidia0").exists() {
            // Try to get GPU info from nvidia-smi
            if let Ok(output) = tokio::process::Command::new("nvidia-smi")
                .arg("--query-gpu=name,memory.total")
                .arg("--format=csv,noheader,nounits")
                .output()
                .await
            {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    let parts: Vec<&str> = info.trim().split(',').collect();
                    if parts.len() >= 2 {
                        return GpuInfo {
                            available: true,
                            device_name: parts[0].trim().to_string(),
                            memory_mb: parts[1].trim().parse().unwrap_or(0),
                            gpu_type: "cuda".to_string(),
                        };
                    }
                }
            }
        }
    }

    // Check for Metal (macOS)
    #[cfg(target_os = "macos")]
    {
        // Metal is always available on modern macOS
        // Try to get GPU info from system_profiler
        if let Ok(output) = tokio::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
            .await
        {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                // Parse GPU name from output
                for line in info.lines() {
                    if line.contains("Chipset Model:") || line.contains("Chip:") {
                        let name = line.split(':').nth(1)
                            .map(|s| s.trim().to_string())
                            .unwrap_or_else(|| "Apple GPU".to_string());
                        return GpuInfo {
                            available: true,
                            device_name: name,
                            memory_mb: 8192, // Assume 8GB for unified memory
                            gpu_type: "metal".to_string(),
                        };
                    }
                }
            }
        }

        // Default for macOS - Metal is available
        return GpuInfo {
            available: true,
            device_name: "Apple GPU".to_string(),
            memory_mb: 8192,
            gpu_type: "metal".to_string(),
        };
    }

    // No GPU detected
    GpuInfo {
        available: false,
        device_name: "None".to_string(),
        memory_mb: 0,
        gpu_type: "none".to_string(),
    }
}

impl DemoPrewarmer {
    /// Get memory statistics
    pub async fn memory_stats(&self) -> MemoryStats {
        let cache = self.cache.read().await;
        let memory_used = *self.memory_used.read().await;

        MemoryStats {
            total_cached_items: cache.len(),
            memory_used_bytes: memory_used,
            memory_limit_bytes: self.config.max_memory_bytes,
            utilization_percent: (memory_used as f64 / self.config.max_memory_bytes as f64 * 100.0) as u8,
        }
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        *self.memory_used.write().await = 0;
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_cached_items: usize,
    pub memory_used_bytes: u64,
    pub memory_limit_bytes: u64,
    pub utilization_percent: u8,
}

/// Quick pre-warm helper that runs in background
pub async fn quick_prewarm() -> Result<()> {
    let prewarmer = DemoPrewarmer::demo_mode();
    prewarmer.prewarm().await?;
    Ok(())
}

/// Pre-warm with timeout
pub async fn prewarm_with_timeout(timeout: Duration) -> Result<PrewarmResult> {
    let prewarmer = DemoPrewarmer::demo_mode();

    tokio::select! {
        result = prewarmer.prewarm() => result,
        _ = tokio::time::sleep(timeout) => {
            Err(anyhow!("Pre-warm timed out after {:?}", timeout))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PrewarmConfig::default();
        assert!(config.prewarm_proving_keys);
        assert!(config.prewarm_model_weights);
        assert_eq!(config.max_memory_bytes, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_config_demo_mode() {
        let config = PrewarmConfig::demo_mode();
        assert!(config.prewarm_gpu);
        assert_eq!(config.parallelism, 8);
    }

    #[tokio::test]
    async fn test_prewarmer_creation() {
        let prewarmer = DemoPrewarmer::demo_mode();
        let status = prewarmer.status().await;
        assert!(!status.complete);
        assert!(!status.in_progress);
    }

    #[tokio::test]
    async fn test_prewarm_minimal() {
        let prewarmer = DemoPrewarmer::new(PrewarmConfig::minimal());
        let result = prewarmer.prewarm().await.unwrap();
        assert!(result.success);
        assert!(!result.items_warmed.is_empty());
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let prewarmer = DemoPrewarmer::default_prewarmer();

        // Add to cache
        prewarmer.add_to_cache("test_key", vec![1, 2, 3, 4]).await.unwrap();

        // Retrieve from cache
        let data = prewarmer.get_cached("test_key").await;
        assert!(data.is_some());
        assert_eq!(data.unwrap(), vec![1, 2, 3, 4]);

        // Clear cache
        prewarmer.clear_cache().await;
        assert!(prewarmer.get_cached("test_key").await.is_none());
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let prewarmer = DemoPrewarmer::default_prewarmer();

        prewarmer.add_to_cache("key1", vec![0u8; 1000]).await.unwrap();
        prewarmer.add_to_cache("key2", vec![0u8; 2000]).await.unwrap();

        let stats = prewarmer.memory_stats().await;
        assert_eq!(stats.total_cached_items, 2);
        assert_eq!(stats.memory_used_bytes, 3000);
    }
}
