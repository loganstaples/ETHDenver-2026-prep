//! Adaptive party selection based on latency and reliability.
//!
//! This module implements intelligent party selection for MPC operations,
//! optimizing for performance and reliability.
//!
//! # Features
//!
//! - **Latency Measurement**: Continuous RTT monitoring via heartbeats
//! - **Reliability Scoring**: Track success/failure rates
//! - **Adaptive Selection**: Prefer faster, more reliable parties
//! - **Failover**: Automatic exclusion of failing parties
//! - **Load Balancing**: Distribute work across parties
//!
//! # Scoring Algorithm
//!
//! Party score = w1 * (1/latency) + w2 * reliability + w3 * availability
//!
//! Where:
//! - latency: exponential moving average of RTT
//! - reliability: success rate over recent operations
//! - availability: fraction of time party has been reachable

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Configuration for party selection.
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Weight for latency in scoring (higher = more important).
    pub latency_weight: f64,
    /// Weight for reliability in scoring.
    pub reliability_weight: f64,
    /// Weight for availability in scoring.
    pub availability_weight: f64,
    /// Heartbeat interval for latency measurement.
    pub heartbeat_interval: Duration,
    /// Timeout for considering a party unresponsive.
    pub response_timeout: Duration,
    /// Number of historical samples for averaging.
    pub history_size: usize,
    /// Exponential moving average decay factor (0-1).
    pub ema_alpha: f64,
    /// Minimum score to be considered selectable.
    pub min_score: f64,
    /// Number of consecutive failures before exclusion.
    pub max_consecutive_failures: usize,
    /// Cooldown period after exclusion.
    pub exclusion_cooldown: Duration,
    /// Enable random jitter in selection (for load balancing).
    pub enable_jitter: bool,
    /// Jitter factor (0-1).
    pub jitter_factor: f64,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            latency_weight: 0.4,
            reliability_weight: 0.4,
            availability_weight: 0.2,
            heartbeat_interval: Duration::from_secs(5),
            response_timeout: Duration::from_secs(10),
            history_size: 100,
            ema_alpha: 0.3,
            min_score: 0.1,
            max_consecutive_failures: 5,
            exclusion_cooldown: Duration::from_secs(60),
            enable_jitter: true,
            jitter_factor: 0.1,
        }
    }
}

impl SelectionConfig {
    /// Strict configuration prioritizing reliability.
    pub fn strict() -> Self {
        Self {
            latency_weight: 0.2,
            reliability_weight: 0.6,
            availability_weight: 0.2,
            max_consecutive_failures: 3,
            min_score: 0.3,
            ..Default::default()
        }
    }

    /// Performance-focused configuration prioritizing latency.
    pub fn performance() -> Self {
        Self {
            latency_weight: 0.6,
            reliability_weight: 0.3,
            availability_weight: 0.1,
            heartbeat_interval: Duration::from_secs(1),
            ..Default::default()
        }
    }
}

/// Latency sample with timestamp.
#[derive(Debug, Clone, Copy)]
struct LatencySample {
    rtt: Duration,
    timestamp: Instant,
}

/// Metrics for a single party.
#[derive(Debug)]
pub struct PartyMetrics {
    /// Party identifier.
    party_id: PartyId,
    /// Latency samples.
    latency_history: VecDeque<LatencySample>,
    /// Exponential moving average of latency.
    latency_ema: Option<Duration>,
    /// Successful operations.
    successes: AtomicU64,
    /// Failed operations.
    failures: AtomicU64,
    /// Consecutive failures (for exclusion).
    consecutive_failures: usize,
    /// Total time available.
    uptime: Duration,
    /// Last seen timestamp.
    last_seen: Instant,
    /// First seen timestamp.
    first_seen: Instant,
    /// Whether party is currently excluded.
    excluded: bool,
    /// When exclusion ends.
    exclusion_end: Option<Instant>,
    /// Current computed score.
    current_score: f64,
    /// Pending heartbeats (seq -> send_time).
    pending_heartbeats: HashMap<u64, Instant>,
    /// Next heartbeat sequence.
    next_heartbeat_seq: AtomicU64,
}

impl PartyMetrics {
    fn new(party_id: PartyId) -> Self {
        let now = Instant::now();
        Self {
            party_id,
            latency_history: VecDeque::new(),
            latency_ema: None,
            successes: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            consecutive_failures: 0,
            uptime: Duration::ZERO,
            last_seen: now,
            first_seen: now,
            excluded: false,
            exclusion_end: None,
            current_score: 1.0, // Start with neutral score
            pending_heartbeats: HashMap::new(),
            next_heartbeat_seq: AtomicU64::new(0),
        }
    }

    /// Records a latency sample.
    fn record_latency(&mut self, rtt: Duration, history_size: usize, ema_alpha: f64) {
        self.latency_history.push_back(LatencySample {
            rtt,
            timestamp: Instant::now(),
        });

        // Prune old samples
        while self.latency_history.len() > history_size {
            self.latency_history.pop_front();
        }

        // Update EMA
        self.latency_ema = Some(match self.latency_ema {
            Some(current) => {
                let current_nanos = current.as_nanos() as f64;
                let new_nanos = rtt.as_nanos() as f64;
                let updated = ema_alpha * new_nanos + (1.0 - ema_alpha) * current_nanos;
                Duration::from_nanos(updated as u64)
            }
            None => rtt,
        });

        self.last_seen = Instant::now();
        self.consecutive_failures = 0;
    }

    /// Records a successful operation.
    fn record_success(&mut self) {
        self.successes.fetch_add(1, Ordering::SeqCst);
        self.consecutive_failures = 0;
        self.last_seen = Instant::now();
    }

    /// Records a failed operation.
    fn record_failure(&mut self, max_failures: usize, cooldown: Duration) {
        self.failures.fetch_add(1, Ordering::SeqCst);
        self.consecutive_failures += 1;

        if self.consecutive_failures >= max_failures {
            self.excluded = true;
            self.exclusion_end = Some(Instant::now() + cooldown);
        }
    }

    /// Checks if exclusion has expired.
    fn check_exclusion(&mut self) {
        if let Some(end) = self.exclusion_end {
            if Instant::now() >= end {
                self.excluded = false;
                self.exclusion_end = None;
                self.consecutive_failures = 0;
            }
        }
    }

    /// Computes reliability score (0-1).
    fn reliability_score(&self) -> f64 {
        let successes = self.successes.load(Ordering::SeqCst);
        let failures = self.failures.load(Ordering::SeqCst);
        let total = successes + failures;

        if total == 0 {
            return 1.0; // No history = neutral
        }

        successes as f64 / total as f64
    }

    /// Computes availability score (0-1).
    fn availability_score(&self) -> f64 {
        let total_time = self.first_seen.elapsed();
        if total_time.is_zero() {
            return 1.0;
        }

        // Estimate uptime based on last_seen
        let since_last = self.last_seen.elapsed();
        if since_last > Duration::from_secs(60) {
            // Hasn't been seen in a while
            0.5
        } else {
            1.0
        }
    }

    /// Computes latency score (0-1, higher is better = lower latency).
    fn latency_score(&self, max_latency: Duration) -> f64 {
        match self.latency_ema {
            Some(latency) => {
                let max_ms = max_latency.as_millis() as f64;
                let latency_ms = latency.as_millis() as f64;
                if latency_ms >= max_ms {
                    0.0
                } else {
                    1.0 - (latency_ms / max_ms)
                }
            }
            None => 0.5, // No data = neutral
        }
    }

    /// Generates a heartbeat sequence number.
    fn generate_heartbeat(&mut self) -> u64 {
        let seq = self.next_heartbeat_seq.fetch_add(1, Ordering::SeqCst);
        self.pending_heartbeats.insert(seq, Instant::now());
        seq
    }

    /// Processes a heartbeat response.
    fn process_heartbeat_response(
        &mut self,
        seq: u64,
        history_size: usize,
        ema_alpha: f64,
    ) -> Option<Duration> {
        if let Some(send_time) = self.pending_heartbeats.remove(&seq) {
            let rtt = send_time.elapsed();
            self.record_latency(rtt, history_size, ema_alpha);
            Some(rtt)
        } else {
            None
        }
    }

    /// Cleans up old pending heartbeats.
    fn cleanup_pending_heartbeats(&mut self, timeout: Duration) {
        let now = Instant::now();
        self.pending_heartbeats.retain(|_, &mut send_time| {
            now.duration_since(send_time) < timeout
        });
    }
}

/// Party selection manager.
pub struct PartySelector {
    /// Configuration.
    config: SelectionConfig,
    /// Metrics for each party.
    metrics: RwLock<HashMap<String, PartyMetrics>>,
    /// Our party ID (excluded from selection).
    our_party_id: PartyId,
    /// Random generator for jitter.
    rng: RwLock<ChaCha20Rng>,
    /// Maximum expected latency for scoring.
    max_expected_latency: Duration,
}

impl PartySelector {
    /// Creates a new party selector.
    pub fn new(our_party_id: PartyId, config: SelectionConfig) -> Self {
        Self {
            config,
            metrics: RwLock::new(HashMap::new()),
            our_party_id,
            rng: RwLock::new(ChaCha20Rng::from_entropy()),
            max_expected_latency: Duration::from_millis(1000),
        }
    }

    /// Registers a party for selection.
    pub fn register_party(&self, party_id: PartyId) {
        if party_id != self.our_party_id {
            let mut metrics = self.metrics.write();
            metrics.entry(party_id.0.clone()).or_insert_with(|| PartyMetrics::new(party_id));
        }
    }

    /// Removes a party from selection.
    pub fn unregister_party(&self, party_id: &PartyId) {
        self.metrics.write().remove(&party_id.0);
    }

    /// Records a successful operation for a party.
    pub fn record_success(&self, party_id: &PartyId) {
        if let Some(m) = self.metrics.write().get_mut(&party_id.0) {
            m.record_success();
        }
    }

    /// Records a failed operation for a party.
    pub fn record_failure(&self, party_id: &PartyId) {
        if let Some(m) = self.metrics.write().get_mut(&party_id.0) {
            m.record_failure(
                self.config.max_consecutive_failures,
                self.config.exclusion_cooldown,
            );
        }
    }

    /// Records a latency measurement.
    pub fn record_latency(&self, party_id: &PartyId, rtt: Duration) {
        if let Some(m) = self.metrics.write().get_mut(&party_id.0) {
            m.record_latency(rtt, self.config.history_size, self.config.ema_alpha);
        }
    }

    /// Generates a heartbeat for a party.
    pub fn generate_heartbeat(&self, party_id: &PartyId) -> Option<u64> {
        self.metrics.write().get_mut(&party_id.0).map(|m| m.generate_heartbeat())
    }

    /// Processes a heartbeat response.
    pub fn process_heartbeat_response(&self, party_id: &PartyId, seq: u64) -> Option<Duration> {
        self.metrics.write().get_mut(&party_id.0).and_then(|m| {
            m.process_heartbeat_response(seq, self.config.history_size, self.config.ema_alpha)
        })
    }

    /// Updates scores for all parties.
    pub fn update_scores(&self) {
        let mut metrics = self.metrics.write();

        for m in metrics.values_mut() {
            m.check_exclusion();
            m.cleanup_pending_heartbeats(self.config.response_timeout);

            let latency_score = m.latency_score(self.max_expected_latency);
            let reliability_score = m.reliability_score();
            let availability_score = m.availability_score();

            let mut score = self.config.latency_weight * latency_score
                + self.config.reliability_weight * reliability_score
                + self.config.availability_weight * availability_score;

            // Apply penalty for excluded parties
            if m.excluded {
                score = 0.0;
            }

            m.current_score = score;
        }
    }

    /// Selects the best party for an operation.
    pub fn select_best(&self) -> Option<PartyId> {
        self.update_scores();
        let metrics = self.metrics.read();

        let mut best: Option<(&str, f64)> = None;

        for (id, m) in metrics.iter() {
            if m.excluded || m.current_score < self.config.min_score {
                continue;
            }

            let mut score = m.current_score;

            // Apply jitter for load balancing
            if self.config.enable_jitter {
                let jitter = self.rng.write().gen_range(-self.config.jitter_factor..self.config.jitter_factor);
                score *= 1.0 + jitter;
            }

            match best {
                None => best = Some((id, score)),
                Some((_, best_score)) if score > best_score => best = Some((id, score)),
                _ => {}
            }
        }

        best.map(|(id, _)| PartyId::new(id))
    }

    /// Selects the top N parties.
    pub fn select_top_n(&self, n: usize) -> Vec<PartyId> {
        self.update_scores();
        let metrics = self.metrics.read();

        let mut scored: Vec<_> = metrics
            .iter()
            .filter(|(_, m)| !m.excluded && m.current_score >= self.config.min_score)
            .map(|(id, m)| {
                let mut score = m.current_score;
                if self.config.enable_jitter {
                    let jitter = self.rng.write().gen_range(-self.config.jitter_factor..self.config.jitter_factor);
                    score *= 1.0 + jitter;
                }
                (id.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n).map(|(id, _)| PartyId::new(id)).collect()
    }

    /// Selects parties for a quorum.
    pub fn select_quorum(&self, required: usize) -> MPCResult<Vec<PartyId>> {
        let selected = self.select_top_n(required);
        if selected.len() < required {
            return Err(MPCError::InsufficientParties {
                required,
                available: selected.len(),
            });
        }
        Ok(selected)
    }

    /// Gets metrics for a party.
    pub fn get_metrics(&self, party_id: &PartyId) -> Option<PartyMetricsSummary> {
        self.metrics.read().get(&party_id.0).map(|m| PartyMetricsSummary {
            party_id: m.party_id.clone(),
            latency_ema: m.latency_ema,
            reliability_score: m.reliability_score(),
            availability_score: m.availability_score(),
            current_score: m.current_score,
            is_excluded: m.excluded,
            successes: m.successes.load(Ordering::SeqCst),
            failures: m.failures.load(Ordering::SeqCst),
            last_seen: m.last_seen,
        })
    }

    /// Gets all party metrics.
    pub fn all_metrics(&self) -> Vec<PartyMetricsSummary> {
        self.update_scores();
        self.metrics
            .read()
            .values()
            .map(|m| PartyMetricsSummary {
                party_id: m.party_id.clone(),
                latency_ema: m.latency_ema,
                reliability_score: m.reliability_score(),
                availability_score: m.availability_score(),
                current_score: m.current_score,
                is_excluded: m.excluded,
                successes: m.successes.load(Ordering::SeqCst),
                failures: m.failures.load(Ordering::SeqCst),
                last_seen: m.last_seen,
            })
            .collect()
    }

    /// Gets parties that need heartbeats.
    pub fn parties_needing_heartbeat(&self) -> Vec<PartyId> {
        let metrics = self.metrics.read();
        let now = Instant::now();

        metrics
            .values()
            .filter(|m| {
                !m.excluded && now.duration_since(m.last_seen) >= self.config.heartbeat_interval
            })
            .map(|m| m.party_id.clone())
            .collect()
    }

    /// Manually excludes a party.
    pub fn exclude_party(&self, party_id: &PartyId) {
        if let Some(m) = self.metrics.write().get_mut(&party_id.0) {
            m.excluded = true;
            m.exclusion_end = Some(Instant::now() + self.config.exclusion_cooldown);
        }
    }

    /// Manually reinstates a party.
    pub fn reinstate_party(&self, party_id: &PartyId) {
        if let Some(m) = self.metrics.write().get_mut(&party_id.0) {
            m.excluded = false;
            m.exclusion_end = None;
            m.consecutive_failures = 0;
        }
    }

    /// Returns count of available (non-excluded) parties.
    pub fn available_count(&self) -> usize {
        self.update_scores();
        self.metrics
            .read()
            .values()
            .filter(|m| !m.excluded && m.current_score >= self.config.min_score)
            .count()
    }
}

/// Summary of party metrics.
#[derive(Debug, Clone)]
pub struct PartyMetricsSummary {
    pub party_id: PartyId,
    pub latency_ema: Option<Duration>,
    pub reliability_score: f64,
    pub availability_score: f64,
    pub current_score: f64,
    pub is_excluded: bool,
    pub successes: u64,
    pub failures: u64,
    pub last_seen: Instant,
}

impl std::fmt::Display for PartyMetricsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: score={:.3}, latency={:?}, reliability={:.2}, excluded={}",
            self.party_id,
            self.current_score,
            self.latency_ema,
            self.reliability_score,
            self.is_excluded
        )
    }
}

/// Heartbeat message for latency measurement.
#[derive(Debug, Clone)]
pub struct HeartbeatMessage {
    /// Sender party.
    pub from: PartyId,
    /// Sequence number.
    pub seq: u64,
    /// Whether this is a response.
    pub is_response: bool,
    /// Timestamp for one-way delay estimation.
    pub timestamp: u64,
}

impl HeartbeatMessage {
    /// Creates a new heartbeat request.
    pub fn request(from: PartyId, seq: u64) -> Self {
        Self {
            from,
            seq,
            is_response: false,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Creates a response to a heartbeat.
    pub fn response(from: PartyId, request: &HeartbeatMessage) -> Self {
        Self {
            from,
            seq: request.seq,
            is_response: true,
            timestamp: request.timestamp,
        }
    }
}

/// Round-robin selector for load distribution.
pub struct RoundRobinSelector {
    parties: Vec<PartyId>,
    current: std::sync::atomic::AtomicUsize,
}

impl RoundRobinSelector {
    pub fn new(parties: Vec<PartyId>) -> Self {
        Self {
            parties,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Selects the next party.
    pub fn next(&self) -> Option<PartyId> {
        if self.parties.is_empty() {
            return None;
        }
        let idx = self.current.fetch_add(1, Ordering::SeqCst) % self.parties.len();
        Some(self.parties[idx].clone())
    }

    /// Updates the party list.
    pub fn update_parties(&mut self, parties: Vec<PartyId>) {
        self.parties = parties;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_party_selector_creation() {
        let parties = test_parties(3);
        let selector = PartySelector::new(parties[0].clone(), SelectionConfig::default());

        for party in &parties[1..] {
            selector.register_party(party.clone());
        }

        assert_eq!(selector.available_count(), 2);
    }

    #[test]
    fn test_party_metrics() {
        let party = PartyId::from_index(0);
        let mut metrics = PartyMetrics::new(party);

        // Record some latencies
        metrics.record_latency(Duration::from_millis(10), 100, 0.3);
        metrics.record_latency(Duration::from_millis(15), 100, 0.3);
        metrics.record_latency(Duration::from_millis(12), 100, 0.3);

        assert!(metrics.latency_ema.is_some());

        // Record success/failure
        metrics.record_success();
        metrics.record_success();
        metrics.record_failure(5, Duration::from_secs(60));

        assert_eq!(metrics.reliability_score(), 2.0 / 3.0);
    }

    #[test]
    fn test_select_best() {
        let parties = test_parties(4);
        let selector = PartySelector::new(parties[0].clone(), SelectionConfig::default());

        for party in &parties[1..] {
            selector.register_party(party.clone());
        }

        // Give party 1 better metrics
        selector.record_latency(&parties[1], Duration::from_millis(5));
        selector.record_success(&parties[1]);
        selector.record_success(&parties[1]);

        // Give party 2 worse metrics
        selector.record_latency(&parties[2], Duration::from_millis(100));
        selector.record_failure(&parties[2]);

        let best = selector.select_best();
        assert!(best.is_some());
        // Party 1 should generally be selected (though jitter adds randomness)
    }

    #[test]
    fn test_select_top_n() {
        let parties = test_parties(5);
        let config = SelectionConfig {
            enable_jitter: false, // Disable for deterministic test
            ..Default::default()
        };
        let selector = PartySelector::new(parties[0].clone(), config);

        for party in &parties[1..] {
            selector.register_party(party.clone());
        }

        let top3 = selector.select_top_n(3);
        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn test_exclusion() {
        let parties = test_parties(3);
        let config = SelectionConfig {
            max_consecutive_failures: 2,
            exclusion_cooldown: Duration::from_millis(100),
            ..Default::default()
        };
        let selector = PartySelector::new(parties[0].clone(), config);

        selector.register_party(parties[1].clone());

        // Cause exclusion
        selector.record_failure(&parties[1]);
        selector.record_failure(&parties[1]);

        let metrics = selector.get_metrics(&parties[1]).unwrap();
        assert!(metrics.is_excluded);

        // Wait for cooldown
        std::thread::sleep(Duration::from_millis(150));

        selector.update_scores();
        let metrics = selector.get_metrics(&parties[1]).unwrap();
        assert!(!metrics.is_excluded);
    }

    #[test]
    fn test_heartbeat_roundtrip() {
        let parties = test_parties(2);
        let selector = PartySelector::new(parties[0].clone(), SelectionConfig::default());
        selector.register_party(parties[1].clone());

        // Generate heartbeat
        let seq = selector.generate_heartbeat(&parties[1]).unwrap();

        // Simulate delay
        std::thread::sleep(Duration::from_millis(5));

        // Process response
        let rtt = selector.process_heartbeat_response(&parties[1], seq);
        assert!(rtt.is_some());
        assert!(rtt.unwrap() >= Duration::from_millis(5));
    }

    #[test]
    fn test_round_robin_selector() {
        let parties = test_parties(3);
        let selector = RoundRobinSelector::new(parties.clone());

        let first = selector.next().unwrap();
        let second = selector.next().unwrap();
        let third = selector.next().unwrap();
        let fourth = selector.next().unwrap();

        assert_eq!(first, parties[0]);
        assert_eq!(second, parties[1]);
        assert_eq!(third, parties[2]);
        assert_eq!(fourth, parties[0]); // Wraps around
    }

    #[test]
    fn test_heartbeat_message() {
        let party = PartyId::from_index(0);
        let request = HeartbeatMessage::request(party.clone(), 42);

        assert_eq!(request.seq, 42);
        assert!(!request.is_response);

        let responder = PartyId::from_index(1);
        let response = HeartbeatMessage::response(responder, &request);

        assert_eq!(response.seq, 42);
        assert!(response.is_response);
        assert_eq!(response.timestamp, request.timestamp);
    }

    #[test]
    fn test_config_presets() {
        let strict = SelectionConfig::strict();
        let performance = SelectionConfig::performance();

        assert!(strict.reliability_weight > performance.reliability_weight);
        assert!(performance.latency_weight > strict.latency_weight);
    }
}
