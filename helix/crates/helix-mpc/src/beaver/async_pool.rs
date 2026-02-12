//! Async Beaver triple pre-generation pool.
//!
//! Wraps the synchronous [`BeaverPipeline`] with a tokio-compatible interface
//! that pre-generates triples in the background. Training steps can request
//! triples without blocking — if the pool has enough, they're returned
//! immediately; otherwise the caller can await until enough are ready.
//!
//! # Usage
//!
//! ```rust,ignore
//! let pool = AsyncTriplePool::start(3, AsyncPoolConfig::default());
//! pool.prepare_for_step(64, 2).await?;
//! pool.ensure_available(100, Duration::from_secs(5)).await?;
//! // ... use triples from pool.pipeline() ...
//! pool.shutdown().await;
//! ```

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;

use crate::beaver::pipeline::{BeaverPipeline, PipelineConfig, PipelineStats};
use crate::error::{MPCError, MPCResult};

/// Configuration for the async triple pool.
#[derive(Debug, Clone)]
pub struct AsyncPoolConfig {
    /// Underlying pipeline config.
    pub pipeline: PipelineConfig,
    /// How often the background task checks pool levels (ms).
    pub poll_interval_ms: u64,
    /// Minimum scalar triples to maintain before starting training.
    pub warmup_count: usize,
    /// Timeout for warmup phase.
    pub warmup_timeout: Duration,
}

impl Default for AsyncPoolConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            poll_interval_ms: 50,
            warmup_count: 1000,
            warmup_timeout: Duration::from_secs(30),
        }
    }
}

impl AsyncPoolConfig {
    /// Config for small models (demo/testing).
    pub fn small_model() -> Self {
        Self {
            pipeline: PipelineConfig::small_model(),
            poll_interval_ms: 25,
            warmup_count: 500,
            warmup_timeout: Duration::from_secs(10),
        }
    }
}

/// Async wrapper around `BeaverPipeline` for non-blocking triple pre-generation.
pub struct AsyncTriplePool {
    /// The underlying synchronous pipeline.
    pipeline: Arc<BeaverPipeline>,
    /// Notify waiters when new triples become available.
    triples_ready: Arc<Notify>,
    /// Background task handle.
    _monitor_handle: tokio::task::JoinHandle<()>,
    /// Shutdown signal.
    shutdown: Arc<tokio::sync::Notify>,
}

impl AsyncTriplePool {
    /// Starts the async triple pool with background generation.
    pub fn start(num_parties: usize, config: AsyncPoolConfig) -> Self {
        let pipeline = Arc::new(BeaverPipeline::new(num_parties, config.pipeline));
        pipeline.start_replenishment();

        let triples_ready = Arc::new(Notify::new());
        let shutdown = Arc::new(Notify::new());

        let monitor_pipeline = pipeline.clone();
        let monitor_notify = triples_ready.clone();
        let monitor_shutdown = shutdown.clone();
        let poll_interval = Duration::from_millis(config.poll_interval_ms);

        let monitor_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = monitor_shutdown.notified() => break,
                    _ = tokio::time::sleep(poll_interval) => {
                        // Check if any triples were generated and notify waiters.
                        let stats = monitor_pipeline.stats();
                        if stats.triples_generated > 0 {
                            monitor_notify.notify_waiters();
                        }
                    }
                }
            }
        });

        Self {
            pipeline,
            triples_ready,
            _monitor_handle: monitor_handle,
            shutdown,
        }
    }

    /// Requests triple pre-generation for a training step.
    pub fn prepare_for_step(&self, hidden_dim: usize, num_layers: usize) -> MPCResult<()> {
        self.pipeline.prepare_for_step(hidden_dim, num_layers)
    }

    /// Waits until at least `count` scalar triples are available in each
    /// party's pool, or the timeout expires.
    pub async fn ensure_available(
        &self,
        count: usize,
        timeout: Duration,
    ) -> MPCResult<()> {
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            // Check synchronously first.
            match self.pipeline.ensure_available(count, Duration::from_millis(1)) {
                Ok(()) => return Ok(()),
                Err(_) => {
                    if tokio::time::Instant::now() >= deadline {
                        return Err(MPCError::BeaverPoolExhausted {
                            requested: count,
                            available: 0,
                        });
                    }
                    // Wait for notification or short sleep.
                    tokio::select! {
                        _ = self.triples_ready.notified() => continue,
                        _ = tokio::time::sleep(Duration::from_millis(50)) => continue,
                    }
                }
            }
        }
    }

    /// Returns a reference to the underlying pipeline.
    pub fn pipeline(&self) -> &BeaverPipeline {
        &self.pipeline
    }

    /// Returns current pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        self.pipeline.stats()
    }

    /// Records scalar triple consumption for demand prediction.
    pub fn record_consumption(&self, count: usize) {
        self.pipeline.record_scalar_consumption(count);
    }

    /// Shuts down the pool and background tasks.
    pub async fn shutdown(self) {
        self.shutdown.notify_one();
        self.pipeline.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_pool_start_stop() {
        let pool = AsyncTriplePool::start(3, AsyncPoolConfig::small_model());

        // Give workers a moment to spin up.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let stats = pool.stats();
        assert!(stats.active_workers > 0);

        pool.shutdown().await;
    }

    #[tokio::test]
    async fn test_async_pool_prepare_and_wait() {
        let pool = AsyncTriplePool::start(3, AsyncPoolConfig::small_model());

        pool.prepare_for_step(4, 1).unwrap();

        // Wait for some triples.
        let result = pool
            .ensure_available(10, Duration::from_secs(5))
            .await;
        assert!(result.is_ok());

        pool.shutdown().await;
    }

    #[tokio::test]
    async fn test_async_pool_consumption_tracking() {
        let pool = AsyncTriplePool::start(3, AsyncPoolConfig::small_model());

        pool.record_consumption(50);

        let stats = pool.stats();
        // Stats should be accessible.
        let _ = stats.throughput;

        pool.shutdown().await;
    }
}
