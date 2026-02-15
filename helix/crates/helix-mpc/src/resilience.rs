//! Beaver triple exhaustion recovery and chain retry logic.
//!
//! During MPC training, two resources can become exhausted or fail:
//!
//! 1. **Beaver triples**: Each multiplication consumes a triple. If the pool runs
//!    out, training must pause while new triples are generated.
//!
//! 2. **On-chain submissions**: Checkpoint attestations submitted to the smart
//!    contract can fail due to network issues, gas price spikes, or temporary
//!    chain congestion.
//!
//! This module provides:
//! - [`ResilientTriplePool`]: A wrapper around a triple vector that tracks
//!   consumption and signals when replenishment is needed.
//! - [`ChainRetrier`]: Exponential backoff retry logic for on-chain submissions.
//! - [`PendingCheckpoint`]: A checkpoint that failed to submit and needs retry.

use std::time::Duration;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::beaver::triple::BeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::field::Fr;

// ============================================================================
// Resilient Triple Pool
// ============================================================================

/// A Beaver triple pool with automatic exhaustion detection and replenishment
/// signaling.
///
/// Wraps a `Vec<BeaverTriple>` with consumption tracking and a low-watermark
/// threshold. When the pool drops below the threshold, the pool signals that
/// replenishment is needed.
#[derive(Debug)]
pub struct ResilientTriplePool {
    /// Available triples.
    triples: Vec<BeaverTriple>,
    /// Current consumption cursor.
    cursor: usize,
    /// Low-watermark: signal replenishment when remaining drops below this.
    low_watermark: usize,
    /// Total triples generated over the lifetime of this pool.
    total_generated: usize,
    /// Total triples consumed.
    total_consumed: usize,
    /// Batch size for replenishment.
    replenish_batch_size: usize,
    /// RNG for local triple generation (trusted dealer fallback for testing).
    rng: ChaCha20Rng,
}

impl ResilientTriplePool {
    /// Creates a new resilient triple pool.
    ///
    /// # Arguments
    ///
    /// * `initial_triples` - Pre-generated triples to start with.
    /// * `low_watermark` - Signal replenishment when remaining drops below this.
    /// * `replenish_batch_size` - How many triples to generate per replenishment.
    /// * `seed` - Random seed for local generation.
    pub fn new(
        initial_triples: Vec<BeaverTriple>,
        low_watermark: usize,
        replenish_batch_size: usize,
        seed: u64,
    ) -> Self {
        let total = initial_triples.len();
        Self {
            triples: initial_triples,
            cursor: 0,
            low_watermark,
            total_generated: total,
            total_consumed: 0,
            replenish_batch_size,
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Returns the number of remaining (unconsumed) triples.
    pub fn remaining(&self) -> usize {
        self.triples.len().saturating_sub(self.cursor)
    }

    /// Returns whether the pool needs replenishment (below low-watermark).
    pub fn needs_replenishment(&self) -> bool {
        self.remaining() < self.low_watermark
    }

    /// Returns whether the pool is completely exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }

    /// Takes the next triple from the pool.
    ///
    /// Returns `Err` if the pool is exhausted.
    pub fn take(&mut self) -> MPCResult<BeaverTriple> {
        if self.cursor >= self.triples.len() {
            return Err(MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            });
        }

        let triple = self.triples[self.cursor].clone();
        self.cursor += 1;
        self.total_consumed += 1;

        if self.needs_replenishment() {
            debug!(
                remaining = self.remaining(),
                watermark = self.low_watermark,
                "Triple pool below low watermark"
            );
        }

        Ok(triple)
    }

    /// Takes the next triple, generating new ones if exhausted.
    ///
    /// If the pool is exhausted, generates a new batch of triples locally
    /// using the trusted dealer pattern (for testing / single-party scenarios).
    /// In production, this would trigger distributed triple generation.
    pub fn take_or_replenish(&mut self) -> MPCResult<BeaverTriple> {
        if self.is_exhausted() {
            info!(
                batch_size = self.replenish_batch_size,
                total_consumed = self.total_consumed,
                "Triple pool exhausted — generating new batch"
            );
            self.replenish_local();
        }
        self.take()
    }

    /// Replenishes the pool with locally generated triples.
    ///
    /// This uses the trusted dealer pattern (suitable for testing only).
    /// In production, triples are generated distributedly via the transport.
    pub fn replenish_local(&mut self) {
        let new_triples = generate_local_triples(
            self.replenish_batch_size,
            &mut self.rng,
        );
        let count = new_triples.len();

        // Compact: move remaining triples to the front and append new ones.
        let remaining: Vec<BeaverTriple> = self.triples[self.cursor..].to_vec();
        self.triples = remaining;
        self.cursor = 0;
        self.triples.extend(new_triples);
        self.total_generated += count;

        info!(
            added = count,
            total_available = self.remaining(),
            total_generated = self.total_generated,
            "Triple pool replenished"
        );
    }

    /// Adds externally generated triples to the pool.
    ///
    /// Used when triples are generated distributedly via the transport.
    pub fn add_triples(&mut self, triples: Vec<BeaverTriple>) {
        let count = triples.len();
        self.triples.extend(triples);
        self.total_generated += count;

        debug!(
            added = count,
            total_available = self.remaining(),
            "External triples added to pool"
        );
    }

    /// Returns pool statistics.
    pub fn stats(&self) -> TriplePoolStats {
        TriplePoolStats {
            remaining: self.remaining(),
            total_generated: self.total_generated,
            total_consumed: self.total_consumed,
            low_watermark: self.low_watermark,
            needs_replenishment: self.needs_replenishment(),
        }
    }

    /// Returns the cursor position (for checkpoint/restore).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Returns a reference to all triples (including consumed ones).
    pub fn all_triples(&self) -> &[BeaverTriple] {
        &self.triples
    }
}

/// Statistics for a triple pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriplePoolStats {
    /// Remaining unconsumed triples.
    pub remaining: usize,
    /// Total triples generated over pool lifetime.
    pub total_generated: usize,
    /// Total triples consumed.
    pub total_consumed: usize,
    /// Low-watermark threshold.
    pub low_watermark: usize,
    /// Whether the pool needs replenishment.
    pub needs_replenishment: bool,
}

/// Generates triples locally using the trusted dealer pattern.
/// Each triple (a, b, c) satisfies c = a * b (using mpc_scale).
fn generate_local_triples(count: usize, rng: &mut ChaCha20Rng) -> Vec<BeaverTriple> {
    let mut triples = Vec::with_capacity(count);
    for _ in 0..count {
        let a = Fr::random(rng);
        let b = Fr::random(rng);
        let c = a.mpc_scale(&b);
        triples.push(BeaverTriple::new(a, b, c));
    }
    triples
}

// ============================================================================
// Chain Retrier
// ============================================================================

/// Retry configuration for on-chain submissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Base delay between retries (before exponential backoff).
    pub base_delay: Duration,
    /// Maximum delay between retries (cap for exponential backoff).
    pub max_delay: Duration,
    /// Jitter factor (0.0 = no jitter, 1.0 = full jitter).
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            jitter_factor: 0.1,
        }
    }
}

/// Handles exponential backoff retry logic for on-chain checkpoint submissions.
///
/// When a checkpoint attestation fails to submit to the smart contract,
/// the retrier holds it as pending and retries with exponential backoff.
pub struct ChainRetrier {
    /// Retry configuration.
    config: RetryConfig,
    /// Pending checkpoints that need retry.
    pending: Vec<PendingCheckpoint>,
}

impl ChainRetrier {
    /// Creates a new chain retrier with the given configuration.
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
        }
    }

    /// Creates a chain retrier with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RetryConfig::default())
    }

    /// Records a failed checkpoint submission for later retry.
    pub fn record_failure(&mut self, checkpoint: PendingCheckpoint) {
        warn!(
            step = checkpoint.step,
            attempts = checkpoint.attempts,
            "Checkpoint submission failed, queued for retry"
        );
        self.pending.push(checkpoint);
    }

    /// Returns the number of pending checkpoints.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Returns whether there are any pending checkpoints.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Computes the delay for the next retry attempt.
    ///
    /// Uses exponential backoff: delay = base * 2^(attempt-1), capped at max_delay.
    pub fn compute_delay(&self, attempt: u32) -> Duration {
        let exp = 2u64.saturating_pow(attempt.saturating_sub(1));
        let delay_ms = self.config.base_delay.as_millis() as u64 * exp;
        let capped = delay_ms.min(self.config.max_delay.as_millis() as u64);
        Duration::from_millis(capped)
    }

    /// Attempts to retry all pending checkpoints.
    ///
    /// The `submit_fn` is called for each pending checkpoint. If it returns
    /// Ok, the checkpoint is removed from the pending list. If it returns Err,
    /// the attempt counter is incremented and the checkpoint remains pending
    /// (unless max retries is exceeded, in which case it's dropped).
    ///
    /// Returns the number of successfully submitted checkpoints.
    pub async fn retry_pending<F, Fut>(
        &mut self,
        mut submit_fn: F,
    ) -> usize
    where
        F: FnMut(&PendingCheckpoint) -> Fut,
        Fut: std::future::Future<Output = Result<(), String>>,
    {
        let mut succeeded = 0;
        let mut remaining = Vec::new();
        let base_delay = self.config.base_delay;
        let max_delay = self.config.max_delay;
        let max_retries = self.config.max_retries;

        for mut checkpoint in self.pending.drain(..) {
            checkpoint.attempts += 1;

            let exp = 2u64.saturating_pow(checkpoint.attempts.saturating_sub(1));
            let delay_ms = base_delay.as_millis() as u64 * exp;
            let capped = delay_ms.min(max_delay.as_millis() as u64);
            let delay = Duration::from_millis(capped);
            debug!(
                step = checkpoint.step,
                attempt = checkpoint.attempts,
                delay_ms = delay.as_millis() as u64,
                "Retrying checkpoint submission"
            );

            // Apply delay.
            tokio::time::sleep(delay).await;

            match submit_fn(&checkpoint).await {
                Ok(()) => {
                    info!(
                        step = checkpoint.step,
                        attempts = checkpoint.attempts,
                        "Checkpoint submitted successfully on retry"
                    );
                    succeeded += 1;
                }
                Err(err) => {
                    if checkpoint.attempts >= max_retries {
                        warn!(
                            step = checkpoint.step,
                            attempts = checkpoint.attempts,
                            error = %err,
                            "Checkpoint permanently failed after max retries"
                        );
                        // Drop it — max retries exceeded.
                    } else {
                        warn!(
                            step = checkpoint.step,
                            attempt = checkpoint.attempts,
                            error = %err,
                            "Checkpoint retry failed, will retry again"
                        );
                        remaining.push(checkpoint);
                    }
                }
            }
        }

        self.pending = remaining;
        succeeded
    }

    /// Drains all pending checkpoints (e.g., for shutdown).
    pub fn drain_pending(&mut self) -> Vec<PendingCheckpoint> {
        self.pending.drain(..).collect()
    }

    /// Returns a reference to all pending checkpoints.
    pub fn pending_checkpoints(&self) -> &[PendingCheckpoint] {
        &self.pending
    }
}

/// A checkpoint that failed to submit on-chain and is pending retry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingCheckpoint {
    /// Training step number.
    pub step: u64,
    /// Session identifier.
    pub session_id: String,
    /// Serialized attestation data.
    pub attestation_data: Vec<u8>,
    /// Number of submission attempts so far.
    pub attempts: u32,
    /// Error from the last failed attempt.
    pub last_error: Option<String>,
}

impl PendingCheckpoint {
    /// Creates a new pending checkpoint.
    pub fn new(step: u64, session_id: &str, attestation_data: Vec<u8>) -> Self {
        Self {
            step,
            session_id: session_id.to_string(),
            attestation_data,
            attempts: 0,
            last_error: None,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triples(count: usize) -> Vec<BeaverTriple> {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        generate_local_triples(count, &mut rng)
    }

    #[test]
    fn test_resilient_pool_creation() {
        let pool = ResilientTriplePool::new(make_triples(100), 10, 50, 42);
        assert_eq!(pool.remaining(), 100);
        assert!(!pool.needs_replenishment());
        assert!(!pool.is_exhausted());
    }

    #[test]
    fn test_resilient_pool_take() {
        let mut pool = ResilientTriplePool::new(make_triples(10), 5, 10, 42);

        let triple = pool.take().unwrap();
        assert_eq!(pool.remaining(), 9);
        assert_eq!(pool.stats().total_consumed, 1);

        // Consume down to low watermark.
        for _ in 0..4 {
            pool.take().unwrap();
        }
        assert_eq!(pool.remaining(), 5);
        assert!(!pool.needs_replenishment());

        // One more puts us below watermark.
        pool.take().unwrap();
        assert_eq!(pool.remaining(), 4);
        assert!(pool.needs_replenishment());
    }

    #[test]
    fn test_resilient_pool_exhaustion() {
        let mut pool = ResilientTriplePool::new(make_triples(3), 1, 10, 42);

        pool.take().unwrap();
        pool.take().unwrap();
        pool.take().unwrap();

        assert!(pool.is_exhausted());
        assert!(pool.take().is_err());
    }

    #[test]
    fn test_resilient_pool_replenish() {
        let mut pool = ResilientTriplePool::new(make_triples(3), 1, 10, 42);

        // Consume all.
        pool.take().unwrap();
        pool.take().unwrap();
        pool.take().unwrap();
        assert!(pool.is_exhausted());

        // Replenish.
        pool.replenish_local();
        assert_eq!(pool.remaining(), 10);
        assert!(!pool.is_exhausted());

        let stats = pool.stats();
        assert_eq!(stats.total_generated, 13); // 3 initial + 10 replenished
        assert_eq!(stats.total_consumed, 3);
    }

    #[test]
    fn test_resilient_pool_take_or_replenish() {
        let mut pool = ResilientTriplePool::new(make_triples(2), 1, 5, 42);

        // Take all initial triples.
        pool.take_or_replenish().unwrap();
        pool.take_or_replenish().unwrap();

        // This should trigger auto-replenishment.
        let _triple = pool.take_or_replenish().unwrap();
        assert_eq!(pool.stats().total_generated, 7); // 2 initial + 5 replenished
    }

    #[test]
    fn test_resilient_pool_add_external() {
        let mut pool = ResilientTriplePool::new(make_triples(2), 5, 10, 42);
        assert!(pool.needs_replenishment());

        pool.add_triples(make_triples(10));
        assert_eq!(pool.remaining(), 12);
        assert!(!pool.needs_replenishment());
    }

    #[test]
    fn test_resilient_pool_stats() {
        let pool = ResilientTriplePool::new(make_triples(50), 10, 25, 42);
        let stats = pool.stats();

        assert_eq!(stats.remaining, 50);
        assert_eq!(stats.total_generated, 50);
        assert_eq!(stats.total_consumed, 0);
        assert_eq!(stats.low_watermark, 10);
        assert!(!stats.needs_replenishment);
    }

    #[test]
    fn test_retry_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(30));
    }

    #[test]
    fn test_chain_retrier_delay_computation() {
        let retrier = ChainRetrier::with_defaults();

        // First attempt: base * 2^0 = 1s.
        assert_eq!(retrier.compute_delay(1), Duration::from_secs(1));
        // Second attempt: base * 2^1 = 2s.
        assert_eq!(retrier.compute_delay(2), Duration::from_secs(2));
        // Third attempt: base * 2^2 = 4s.
        assert_eq!(retrier.compute_delay(3), Duration::from_secs(4));
    }

    #[test]
    fn test_chain_retrier_delay_cap() {
        let config = RetryConfig {
            base_delay: Duration::from_secs(10),
            max_delay: Duration::from_secs(30),
            ..Default::default()
        };
        let retrier = ChainRetrier::new(config);

        // Third attempt: 10 * 4 = 40s, capped at 30s.
        assert_eq!(retrier.compute_delay(3), Duration::from_secs(30));
    }

    #[test]
    fn test_chain_retrier_record_failure() {
        let mut retrier = ChainRetrier::with_defaults();
        assert_eq!(retrier.pending_count(), 0);

        let cp = PendingCheckpoint::new(10, "test", vec![1, 2, 3]);
        retrier.record_failure(cp);

        assert_eq!(retrier.pending_count(), 1);
        assert!(retrier.has_pending());
    }

    #[tokio::test]
    async fn test_chain_retrier_retry_success() {
        let mut retrier = ChainRetrier::new(RetryConfig {
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            ..Default::default()
        });

        let cp = PendingCheckpoint::new(10, "test", vec![1, 2, 3]);
        retrier.record_failure(cp);

        let succeeded = retrier.retry_pending(|_| async { Ok(()) }).await;
        assert_eq!(succeeded, 1);
        assert_eq!(retrier.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_chain_retrier_retry_failure_then_success() {
        let mut retrier = ChainRetrier::new(RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            jitter_factor: 0.0,
        });

        let cp = PendingCheckpoint::new(10, "test", vec![1, 2, 3]);
        retrier.record_failure(cp);

        // First retry fails.
        let succeeded = retrier.retry_pending(|_| async {
            Err("temporary error".to_string())
        }).await;
        assert_eq!(succeeded, 0);
        assert_eq!(retrier.pending_count(), 1);

        // Second retry succeeds.
        let succeeded = retrier.retry_pending(|_| async { Ok(()) }).await;
        assert_eq!(succeeded, 1);
        assert_eq!(retrier.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_chain_retrier_max_retries_exceeded() {
        let mut retrier = ChainRetrier::new(RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            jitter_factor: 0.0,
        });

        let cp = PendingCheckpoint::new(10, "test", vec![1, 2, 3]);
        retrier.record_failure(cp);

        // Fail twice (reaches max_retries=2).
        retrier.retry_pending(|_| async { Err("fail".to_string()) }).await;
        retrier.retry_pending(|_| async { Err("fail".to_string()) }).await;

        // Should be dropped after max retries.
        assert_eq!(retrier.pending_count(), 0);
    }

    #[test]
    fn test_pending_checkpoint_creation() {
        let cp = PendingCheckpoint::new(42, "session-1", vec![0xDE, 0xAD]);
        assert_eq!(cp.step, 42);
        assert_eq!(cp.session_id, "session-1");
        assert_eq!(cp.attempts, 0);
        assert!(cp.last_error.is_none());
    }

    #[test]
    fn test_drain_pending() {
        let mut retrier = ChainRetrier::with_defaults();
        retrier.record_failure(PendingCheckpoint::new(1, "s", vec![]));
        retrier.record_failure(PendingCheckpoint::new(2, "s", vec![]));

        let drained = retrier.drain_pending();
        assert_eq!(drained.len(), 2);
        assert_eq!(retrier.pending_count(), 0);
    }

    #[test]
    fn test_generate_local_triples() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let triples = generate_local_triples(5, &mut rng);
        assert_eq!(triples.len(), 5);

        // Verify c = a * b for each triple.
        for t in &triples {
            let expected_c = t.a.mpc_scale(&t.b);
            let diff = (expected_c.to_f64() - t.c.to_f64()).abs();
            assert!(diff < 1e-6, "triple c != a*b");
        }
    }
}
