//! MAC verification batching for improved throughput.
//!
//! This module extends the basic MAC verification with batching capabilities
//! to verify multiple MACs efficiently using random linear combinations.
//!
//! # Technique
//!
//! Instead of verifying each MAC individually:
//!   for each i: verify MAC(x_i) = α * x_i
//!
//! We use random linear combination:
//!   verify Σ r_i * MAC(x_i) = α * Σ r_i * x_i
//!
//! This reduces verification cost while maintaining security with high probability.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Configuration for batched MAC verification.
#[derive(Debug, Clone)]
pub struct BatchMACConfig {
    /// Maximum MACs to accumulate before forced verification.
    pub max_batch_size: usize,
    /// Security parameter (bits) for random linear combination.
    pub security_bits: usize,
    /// Tolerance for floating-point comparison.
    pub tolerance: f64,
    /// Enable adaptive batching based on throughput.
    pub adaptive_batching: bool,
    /// Target throughput (verifications/second).
    pub target_throughput: f64,
}

impl Default for BatchMACConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            security_bits: 128,
            tolerance: 1e-6,
            adaptive_batching: true,
            target_throughput: 10000.0,
        }
    }
}

/// A pending MAC verification.
#[derive(Debug, Clone)]
struct PendingMAC {
    /// Value share.
    value: f64,
    /// MAC share.
    mac: f64,
    /// Party ID.
    party: PartyId,
    /// Label for debugging.
    label: String,
    /// Timestamp queued.
    queued_at: Instant,
}

/// Batched MAC verifier with random linear combination.
pub struct BatchMACVerifier {
    /// Configuration.
    config: BatchMACConfig,
    /// Pending MACs by party.
    pending: RwLock<Vec<PendingMAC>>,
    /// Alpha shares from all parties.
    alpha_shares: RwLock<Vec<f64>>,
    /// Random generator.
    rng: RwLock<ChaCha20Rng>,
    /// Statistics.
    stats: BatchMACStats,
    /// Seed for deterministic verification (for reproducibility).
    verification_seed: RwLock<Option<u64>>,
}

/// Statistics for batched MAC verification.
#[derive(Debug, Default)]
pub struct BatchMACStats {
    /// Total MACs verified.
    pub macs_verified: AtomicU64,
    /// Total batches processed.
    pub batches_processed: AtomicU64,
    /// Total verification time (nanoseconds).
    pub verification_time_ns: AtomicU64,
    /// Failed verifications.
    pub failures: AtomicU64,
    /// Average batch size.
    pub avg_batch_size: RwLock<f64>,
}

impl BatchMACVerifier {
    /// Creates a new batched MAC verifier.
    pub fn new(config: BatchMACConfig) -> Self {
        Self {
            config,
            pending: RwLock::new(Vec::new()),
            alpha_shares: RwLock::new(Vec::new()),
            rng: RwLock::new(ChaCha20Rng::from_entropy()),
            stats: BatchMACStats::default(),
            verification_seed: RwLock::new(None),
        }
    }

    /// Sets the alpha shares for all parties.
    pub fn set_alpha_shares(&self, shares: Vec<f64>) {
        *self.alpha_shares.write() = shares;
    }

    /// Sets a fixed seed for deterministic verification.
    pub fn set_verification_seed(&self, seed: u64) {
        *self.verification_seed.write() = Some(seed);
    }

    /// Queues a MAC for batch verification.
    pub fn queue(&self, value: f64, mac: f64, party: PartyId, label: String) {
        let mut pending = self.pending.write();
        pending.push(PendingMAC {
            value,
            mac,
            party,
            label,
            queued_at: Instant::now(),
        });

        if pending.len() >= self.config.max_batch_size {
            drop(pending);
            let _ = self.flush();
        }
    }

    /// Queues multiple MACs.
    pub fn queue_batch(
        &self,
        values: &[f64],
        macs: &[f64],
        party: PartyId,
        label_prefix: &str,
    ) {
        let mut pending = self.pending.write();
        let now = Instant::now();

        for (i, (&v, &m)) in values.iter().zip(macs.iter()).enumerate() {
            pending.push(PendingMAC {
                value: v,
                mac: m,
                party: party.clone(),
                label: format!("{}-{}", label_prefix, i),
                queued_at: now,
            });
        }

        if pending.len() >= self.config.max_batch_size {
            drop(pending);
            let _ = self.flush();
        }
    }

    /// Verifies all pending MACs using random linear combination.
    pub fn flush(&self) -> MPCResult<BatchVerificationResult> {
        let start = Instant::now();
        let pending = std::mem::take(&mut *self.pending.write());

        if pending.is_empty() {
            return Ok(BatchVerificationResult {
                verified_count: 0,
                failed: Vec::new(),
                duration: Duration::ZERO,
            });
        }

        let alpha_shares = self.alpha_shares.read();
        let alpha: f64 = alpha_shares.iter().sum();

        // Generate random coefficients
        let seed = self.verification_seed.read().unwrap_or_else(|| {
            self.rng.write().gen()
        });
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let coefficients: Vec<f64> = (0..pending.len())
            .map(|_| rng.gen_range(1.0..2f64.powi(self.config.security_bits as i32 / 4)))
            .collect();

        // Compute random linear combination
        let mut combined_value = 0.0;
        let mut combined_mac = 0.0;

        for (i, p) in pending.iter().enumerate() {
            combined_value += coefficients[i] * p.value;
            combined_mac += coefficients[i] * p.mac;
        }

        // Verify: combined_mac should equal alpha * combined_value
        let expected_mac = alpha * combined_value;
        let batch_valid = (combined_mac - expected_mac).abs() <= self.config.tolerance * pending.len() as f64;

        let duration = start.elapsed();
        self.stats.verification_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::SeqCst);
        self.stats.batches_processed.fetch_add(1, Ordering::SeqCst);

        let mut avg = self.stats.avg_batch_size.write();
        *avg = (*avg * 0.9) + (pending.len() as f64 * 0.1);

        if batch_valid {
            self.stats.macs_verified.fetch_add(pending.len() as u64, Ordering::SeqCst);
            Ok(BatchVerificationResult {
                verified_count: pending.len(),
                failed: Vec::new(),
                duration,
            })
        } else {
            // Batch failed - identify culprit(s) by individual verification
            self.stats.failures.fetch_add(1, Ordering::SeqCst);
            let failed = self.identify_failures(&pending, alpha);

            self.stats.macs_verified.fetch_add((pending.len() - failed.len()) as u64, Ordering::SeqCst);

            Ok(BatchVerificationResult {
                verified_count: pending.len() - failed.len(),
                failed,
                duration,
            })
        }
    }

    /// Identifies individual MAC failures.
    fn identify_failures(&self, pending: &[PendingMAC], alpha: f64) -> Vec<MACFailure> {
        pending
            .iter()
            .filter_map(|p| {
                let expected = alpha * p.value;
                if (p.mac - expected).abs() > self.config.tolerance {
                    Some(MACFailure {
                        party: p.party.clone(),
                        label: p.label.clone(),
                        expected_mac: expected,
                        actual_mac: p.mac,
                        value: p.value,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns current statistics.
    pub fn stats(&self) -> BatchMACStatsSnapshot {
        BatchMACStatsSnapshot {
            macs_verified: self.stats.macs_verified.load(Ordering::SeqCst),
            batches_processed: self.stats.batches_processed.load(Ordering::SeqCst),
            failures: self.stats.failures.load(Ordering::SeqCst),
            avg_batch_size: *self.stats.avg_batch_size.read(),
            pending_count: self.pending.read().len(),
            avg_verification_time: if self.stats.batches_processed.load(Ordering::SeqCst) > 0 {
                Duration::from_nanos(
                    self.stats.verification_time_ns.load(Ordering::SeqCst) /
                    self.stats.batches_processed.load(Ordering::SeqCst)
                )
            } else {
                Duration::ZERO
            },
        }
    }

    /// Returns pending count.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }
}

/// Result of batch verification.
#[derive(Debug, Clone)]
pub struct BatchVerificationResult {
    /// Number of MACs successfully verified.
    pub verified_count: usize,
    /// Failed MACs.
    pub failed: Vec<MACFailure>,
    /// Verification duration.
    pub duration: Duration,
}

/// A MAC verification failure.
#[derive(Debug, Clone)]
pub struct MACFailure {
    /// Party that submitted the invalid MAC.
    pub party: PartyId,
    /// Label identifying the value.
    pub label: String,
    /// Expected MAC value.
    pub expected_mac: f64,
    /// Actual MAC value received.
    pub actual_mac: f64,
    /// The value that was MACed.
    pub value: f64,
}

/// Snapshot of MAC verification statistics.
#[derive(Debug, Clone)]
pub struct BatchMACStatsSnapshot {
    pub macs_verified: u64,
    pub batches_processed: u64,
    pub failures: u64,
    pub avg_batch_size: f64,
    pub pending_count: usize,
    pub avg_verification_time: Duration,
}

/// Streaming MAC verifier for continuous verification.
pub struct StreamingMACVerifier {
    /// Inner batch verifier.
    inner: BatchMACVerifier,
    /// Verification interval.
    interval: Duration,
    /// Last verification time.
    last_verification: RwLock<Instant>,
}

impl StreamingMACVerifier {
    /// Creates a new streaming verifier.
    pub fn new(config: BatchMACConfig, interval: Duration) -> Self {
        Self {
            inner: BatchMACVerifier::new(config),
            interval,
            last_verification: RwLock::new(Instant::now()),
        }
    }

    /// Sets alpha shares.
    pub fn set_alpha_shares(&self, shares: Vec<f64>) {
        self.inner.set_alpha_shares(shares);
    }

    /// Adds a MAC and potentially triggers verification.
    pub fn add(&self, value: f64, mac: f64, party: PartyId, label: String) -> Option<BatchVerificationResult> {
        self.inner.queue(value, mac, party, label);

        let should_verify = {
            let last = *self.last_verification.read();
            last.elapsed() >= self.interval
        };

        if should_verify {
            *self.last_verification.write() = Instant::now();
            self.inner.flush().ok()
        } else {
            None
        }
    }

    /// Forces verification of all pending MACs.
    pub fn flush(&self) -> MPCResult<BatchVerificationResult> {
        *self.last_verification.write() = Instant::now();
        self.inner.flush()
    }

    /// Gets statistics.
    pub fn stats(&self) -> BatchMACStatsSnapshot {
        self.inner.stats()
    }
}

/// Parallel MAC verifier using multiple threads.
pub struct ParallelMACVerifier {
    /// Number of workers.
    num_workers: usize,
    /// Per-worker verifiers.
    workers: Vec<BatchMACVerifier>,
    /// Round-robin counter.
    next_worker: AtomicU64,
}

impl ParallelMACVerifier {
    /// Creates a new parallel verifier.
    pub fn new(num_workers: usize, config: BatchMACConfig) -> Self {
        let workers = (0..num_workers)
            .map(|_| BatchMACVerifier::new(config.clone()))
            .collect();

        Self {
            num_workers,
            workers,
            next_worker: AtomicU64::new(0),
        }
    }

    /// Sets alpha shares for all workers.
    pub fn set_alpha_shares(&self, shares: Vec<f64>) {
        for worker in &self.workers {
            worker.set_alpha_shares(shares.clone());
        }
    }

    /// Queues a MAC to a worker.
    pub fn queue(&self, value: f64, mac: f64, party: PartyId, label: String) {
        let idx = (self.next_worker.fetch_add(1, Ordering::SeqCst) as usize) % self.num_workers;
        self.workers[idx].queue(value, mac, party, label);
    }

    /// Flushes all workers.
    pub fn flush_all(&self) -> MPCResult<Vec<BatchVerificationResult>> {
        self.workers.iter().map(|w| w.flush()).collect()
    }

    /// Gets combined statistics.
    pub fn stats(&self) -> BatchMACStatsSnapshot {
        let mut total = BatchMACStatsSnapshot {
            macs_verified: 0,
            batches_processed: 0,
            failures: 0,
            avg_batch_size: 0.0,
            pending_count: 0,
            avg_verification_time: Duration::ZERO,
        };

        for worker in &self.workers {
            let s = worker.stats();
            total.macs_verified += s.macs_verified;
            total.batches_processed += s.batches_processed;
            total.failures += s.failures;
            total.pending_count += s.pending_count;
        }

        if self.num_workers > 0 {
            total.avg_batch_size = self.workers.iter()
                .map(|w| w.stats().avg_batch_size)
                .sum::<f64>() / self.num_workers as f64;
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_verifier() -> BatchMACVerifier {
        let config = BatchMACConfig {
            max_batch_size: 100,
            tolerance: 1e-6,
            ..Default::default()
        };
        let verifier = BatchMACVerifier::new(config);
        // Set alpha = 5.0 split across 2 parties
        verifier.set_alpha_shares(vec![2.0, 3.0]);
        verifier
    }

    #[test]
    fn test_batch_mac_verification_success() {
        let verifier = setup_verifier();

        // Queue valid MACs (value, mac where mac = 5 * value)
        for i in 0..10 {
            let value = i as f64;
            let mac = 5.0 * value; // alpha = 5.0
            verifier.queue(value, mac, PartyId::from_index(0), format!("test-{}", i));
        }

        let result = verifier.flush().unwrap();
        assert_eq!(result.verified_count, 10);
        assert!(result.failed.is_empty());
    }

    #[test]
    fn test_batch_mac_verification_failure() {
        let verifier = setup_verifier();

        // Queue mostly valid MACs with one bad one
        for i in 0..9 {
            let value = i as f64;
            let mac = 5.0 * value;
            verifier.queue(value, mac, PartyId::from_index(0), format!("test-{}", i));
        }

        // Bad MAC
        verifier.queue(10.0, 100.0, PartyId::from_index(1), "bad".into()); // Should be 50.0

        let result = verifier.flush().unwrap();
        assert_eq!(result.verified_count, 9);
        assert_eq!(result.failed.len(), 1);
        assert_eq!(result.failed[0].label, "bad");
    }

    #[test]
    fn test_queue_batch() {
        let verifier = setup_verifier();

        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let macs: Vec<f64> = values.iter().map(|v| 5.0 * v).collect();

        verifier.queue_batch(&values, &macs, PartyId::from_index(0), "batch");

        let result = verifier.flush().unwrap();
        assert_eq!(result.verified_count, 20);
    }

    #[test]
    fn test_streaming_verifier() {
        let config = BatchMACConfig::default();
        let verifier = StreamingMACVerifier::new(config, Duration::from_millis(10));
        verifier.set_alpha_shares(vec![5.0]);

        // Add some MACs
        for i in 0..5 {
            let result = verifier.add(i as f64, 5.0 * i as f64, PartyId::from_index(0), format!("test-{}", i));
            // May or may not verify based on timing
            if let Some(r) = result {
                assert!(r.failed.is_empty());
            }
        }

        // Force flush
        let result = verifier.flush().unwrap();
        assert!(result.failed.is_empty());
    }

    #[test]
    fn test_parallel_verifier() {
        let config = BatchMACConfig::default();
        let verifier = ParallelMACVerifier::new(4, config);
        verifier.set_alpha_shares(vec![5.0]);

        // Distribute MACs across workers
        for i in 0..100 {
            verifier.queue(i as f64, 5.0 * i as f64, PartyId::from_index(0), format!("test-{}", i));
        }

        let results = verifier.flush_all().unwrap();
        let total_verified: usize = results.iter().map(|r| r.verified_count).sum();
        assert_eq!(total_verified, 100);
    }

    #[test]
    fn test_deterministic_verification() {
        let config = BatchMACConfig::default();
        let verifier = BatchMACVerifier::new(config);
        verifier.set_alpha_shares(vec![5.0]);
        verifier.set_verification_seed(12345);

        for i in 0..10 {
            verifier.queue(i as f64, 5.0 * i as f64, PartyId::from_index(0), format!("test-{}", i));
        }

        let result = verifier.flush().unwrap();
        assert_eq!(result.verified_count, 10);
    }
}
