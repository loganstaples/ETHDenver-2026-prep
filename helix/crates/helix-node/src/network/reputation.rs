//! Peer Reputation Scoring for HELIX Network.
//!
//! Implements a comprehensive peer reputation system that tracks peer behavior
//! and influences network decisions. Key mechanisms:
//! - Multi-dimensional reputation scoring
//! - Behavior tracking (responsiveness, validity, bandwidth)
//! - Reputation decay and recovery
//! - Reputation-based peer prioritization

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::messages::PeerId;
use serde::{Deserialize, Serialize};

/// Configuration for reputation scoring.
#[derive(Debug, Clone)]
pub struct ReputationConfig {
    /// Initial reputation score for new peers.
    pub initial_score: f64,
    /// Minimum reputation score.
    pub min_score: f64,
    /// Maximum reputation score.
    pub max_score: f64,
    /// Score threshold for "good" reputation.
    pub good_threshold: f64,
    /// Score threshold below which peer is banned.
    pub ban_threshold: f64,
    /// Decay rate per hour.
    pub decay_rate: f64,
    /// Recovery rate per hour when behaving well.
    pub recovery_rate: f64,
    /// Weight for responsiveness dimension.
    pub responsiveness_weight: f64,
    /// Weight for validity dimension.
    pub validity_weight: f64,
    /// Weight for bandwidth dimension.
    pub bandwidth_weight: f64,
    /// Weight for uptime dimension.
    pub uptime_weight: f64,
    /// Window size for recent behavior tracking.
    pub behavior_window: Duration,
    /// Maximum behavior events to track.
    pub max_behavior_events: usize,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            initial_score: 50.0,
            min_score: 0.0,
            max_score: 100.0,
            good_threshold: 60.0,
            ban_threshold: 10.0,
            decay_rate: 0.5,
            recovery_rate: 1.0,
            responsiveness_weight: 0.25,
            validity_weight: 0.35,
            bandwidth_weight: 0.20,
            uptime_weight: 0.20,
            behavior_window: Duration::from_secs(3600),
            max_behavior_events: 1000,
        }
    }
}

/// Reputation dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReputationDimension {
    /// How quickly peer responds to requests.
    Responsiveness,
    /// How often peer provides valid data/proofs.
    Validity,
    /// Peer's bandwidth contribution.
    Bandwidth,
    /// How long peer has been online.
    Uptime,
    /// Overall combined score.
    Overall,
}

/// Peer reputation record.
#[derive(Debug, Clone)]
pub struct PeerReputation {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Overall reputation score.
    pub score: f64,
    /// Dimension scores.
    pub dimensions: HashMap<ReputationDimension, f64>,
    /// First seen timestamp.
    pub first_seen: Instant,
    /// Last update timestamp.
    pub last_update: Instant,
    /// Last activity timestamp.
    pub last_activity: Instant,
    /// Behavior events.
    behavior_events: VecDeque<BehaviorEvent>,
    /// Whether peer is banned.
    pub is_banned: bool,
    /// Ban expiry (if banned).
    pub ban_expiry: Option<Instant>,
    /// Total interactions.
    pub total_interactions: u64,
    /// Positive interactions.
    pub positive_interactions: u64,
    /// Recent latencies (for responsiveness).
    recent_latencies: VecDeque<Duration>,
    /// Configuration snapshot.
    config: ReputationConfig,
}

impl PeerReputation {
    /// Creates a new peer reputation record.
    pub fn new(peer_id: PeerId, config: ReputationConfig) -> Self {
        let now = Instant::now();
        let mut dimensions = HashMap::new();

        dimensions.insert(ReputationDimension::Responsiveness, config.initial_score);
        dimensions.insert(ReputationDimension::Validity, config.initial_score);
        dimensions.insert(ReputationDimension::Bandwidth, config.initial_score);
        dimensions.insert(ReputationDimension::Uptime, config.initial_score);

        Self {
            peer_id,
            score: config.initial_score,
            dimensions,
            first_seen: now,
            last_update: now,
            last_activity: now,
            behavior_events: VecDeque::with_capacity(config.max_behavior_events),
            is_banned: false,
            ban_expiry: None,
            total_interactions: 0,
            positive_interactions: 0,
            recent_latencies: VecDeque::with_capacity(100),
            config,
        }
    }

    /// Records a behavior event.
    pub fn record_event(&mut self, event: BehaviorEvent) {
        // Prune old events
        let cutoff = Instant::now() - self.config.behavior_window;
        while let Some(front) = self.behavior_events.front() {
            if front.timestamp < cutoff {
                self.behavior_events.pop_front();
            } else {
                break;
            }
        }

        // Add new event
        if self.behavior_events.len() >= self.config.max_behavior_events {
            self.behavior_events.pop_front();
        }
        self.behavior_events.push_back(event.clone());

        // Update relevant dimension
        self.update_dimension_from_event(&event);

        // Recalculate overall score
        self.recalculate_score();

        self.total_interactions += 1;
        if event.is_positive() {
            self.positive_interactions += 1;
        }

        self.last_activity = Instant::now();
        self.last_update = Instant::now();

        // Check for ban
        if self.score < self.config.ban_threshold {
            self.ban(Duration::from_secs(3600)); // 1 hour ban
        }
    }

    /// Updates dimension score based on event.
    fn update_dimension_from_event(&mut self, event: &BehaviorEvent) {
        let (dimension, impact) = match event.event_type {
            BehaviorEventType::ResponseReceived { latency } => {
                self.recent_latencies.push_back(latency);
                if self.recent_latencies.len() > 100 {
                    self.recent_latencies.pop_front();
                }
                // Lower latency = higher score impact
                let latency_score = if latency < Duration::from_millis(100) {
                    2.0
                } else if latency < Duration::from_millis(500) {
                    1.0
                } else if latency < Duration::from_secs(2) {
                    0.0
                } else {
                    -1.0
                };
                (ReputationDimension::Responsiveness, latency_score)
            }
            BehaviorEventType::ResponseTimeout => {
                (ReputationDimension::Responsiveness, -5.0)
            }
            BehaviorEventType::ValidProof => {
                (ReputationDimension::Validity, 3.0)
            }
            BehaviorEventType::InvalidProof => {
                (ReputationDimension::Validity, -10.0)
            }
            BehaviorEventType::ValidGradient => {
                (ReputationDimension::Validity, 2.0)
            }
            BehaviorEventType::InvalidGradient => {
                (ReputationDimension::Validity, -8.0)
            }
            BehaviorEventType::DataServed { bytes: _ } => {
                (ReputationDimension::Bandwidth, 1.0)
            }
            BehaviorEventType::DataRequested { bytes: _ } => {
                (ReputationDimension::Bandwidth, -0.1)
            }
            BehaviorEventType::Heartbeat => {
                (ReputationDimension::Uptime, 0.5)
            }
            BehaviorEventType::Disconnect { graceful } => {
                let impact = if graceful { -1.0 } else { -5.0 };
                (ReputationDimension::Uptime, impact)
            }
            BehaviorEventType::ProtocolViolation { severity } => {
                let impact = -(severity as f64 * 3.0);
                (ReputationDimension::Validity, impact)
            }
            BehaviorEventType::Custom { dimension, impact } => {
                (dimension, impact)
            }
        };

        // Update dimension score
        let current = self.dimensions.get(&dimension).copied().unwrap_or(self.config.initial_score);
        let new_score = (current + impact).clamp(self.config.min_score, self.config.max_score);
        self.dimensions.insert(dimension, new_score);
    }

    /// Recalculates overall score from dimensions.
    fn recalculate_score(&mut self) {
        let responsiveness = self.dimensions.get(&ReputationDimension::Responsiveness).copied().unwrap_or(50.0);
        let validity = self.dimensions.get(&ReputationDimension::Validity).copied().unwrap_or(50.0);
        let bandwidth = self.dimensions.get(&ReputationDimension::Bandwidth).copied().unwrap_or(50.0);
        let uptime = self.dimensions.get(&ReputationDimension::Uptime).copied().unwrap_or(50.0);

        self.score = responsiveness * self.config.responsiveness_weight
            + validity * self.config.validity_weight
            + bandwidth * self.config.bandwidth_weight
            + uptime * self.config.uptime_weight;

        self.score = self.score.clamp(self.config.min_score, self.config.max_score);
    }

    /// Applies time-based decay.
    pub fn apply_decay(&mut self, hours_elapsed: f64) {
        let decay = self.config.decay_rate * hours_elapsed;

        for score in self.dimensions.values_mut() {
            // Decay towards initial score
            let diff = *score - self.config.initial_score;
            if diff > 0.0 {
                *score = (*score - decay).max(self.config.initial_score);
            } else if diff < 0.0 {
                *score = (*score + decay).min(self.config.initial_score);
            }
        }

        self.recalculate_score();
        self.last_update = Instant::now();
    }

    /// Bans the peer.
    pub fn ban(&mut self, duration: Duration) {
        self.is_banned = true;
        self.ban_expiry = Some(Instant::now() + duration);
    }

    /// Unbans the peer.
    pub fn unban(&mut self) {
        self.is_banned = false;
        self.ban_expiry = None;
    }

    /// Checks if ban has expired.
    pub fn check_ban_expiry(&mut self) -> bool {
        if let Some(expiry) = self.ban_expiry {
            if Instant::now() >= expiry {
                self.unban();
                return true;
            }
        }
        false
    }

    /// Returns whether this is a "good" peer.
    pub fn is_good(&self) -> bool {
        self.score >= self.config.good_threshold && !self.is_banned
    }

    /// Returns average latency.
    pub fn average_latency(&self) -> Option<Duration> {
        if self.recent_latencies.is_empty() {
            return None;
        }
        let sum: Duration = self.recent_latencies.iter().sum();
        Some(sum / self.recent_latencies.len() as u32)
    }

    /// Returns positive interaction rate.
    pub fn positive_rate(&self) -> f64 {
        if self.total_interactions == 0 {
            return 0.5;
        }
        self.positive_interactions as f64 / self.total_interactions as f64
    }

    /// Returns peer age.
    pub fn age(&self) -> Duration {
        self.first_seen.elapsed()
    }
}

/// A behavior event to record.
#[derive(Debug, Clone)]
pub struct BehaviorEvent {
    /// Event type.
    pub event_type: BehaviorEventType,
    /// Event timestamp.
    pub timestamp: Instant,
}

impl BehaviorEvent {
    /// Creates a new behavior event.
    pub fn new(event_type: BehaviorEventType) -> Self {
        Self {
            event_type,
            timestamp: Instant::now(),
        }
    }

    /// Returns whether this is a positive event.
    pub fn is_positive(&self) -> bool {
        matches!(
            self.event_type,
            BehaviorEventType::ResponseReceived { .. }
                | BehaviorEventType::ValidProof
                | BehaviorEventType::ValidGradient
                | BehaviorEventType::DataServed { .. }
                | BehaviorEventType::Heartbeat
        )
    }
}

/// Types of behavior events.
#[derive(Debug, Clone)]
pub enum BehaviorEventType {
    /// Response received with latency.
    ResponseReceived { latency: Duration },
    /// Response timeout.
    ResponseTimeout,
    /// Valid proof submitted.
    ValidProof,
    /// Invalid proof submitted.
    InvalidProof,
    /// Valid gradient submitted.
    ValidGradient,
    /// Invalid gradient submitted.
    InvalidGradient,
    /// Data served to network.
    DataServed { bytes: u64 },
    /// Data requested from network.
    DataRequested { bytes: u64 },
    /// Heartbeat received.
    Heartbeat,
    /// Disconnection.
    Disconnect { graceful: bool },
    /// Protocol violation.
    ProtocolViolation { severity: u32 },
    /// Custom event.
    Custom { dimension: ReputationDimension, impact: f64 },
}

/// Reputation manager for all peers.
pub struct ReputationManager {
    /// Configuration.
    config: ReputationConfig,
    /// Peer reputations.
    reputations: HashMap<PeerId, PeerReputation>,
    /// Last decay time.
    last_decay: Instant,
    /// Decay interval.
    decay_interval: Duration,
}

impl ReputationManager {
    /// Creates a new reputation manager.
    pub fn new(config: ReputationConfig) -> Self {
        Self {
            config,
            reputations: HashMap::new(),
            last_decay: Instant::now(),
            decay_interval: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Gets or creates reputation for a peer.
    pub fn get_or_create(&mut self, peer_id: &PeerId) -> &mut PeerReputation {
        self.reputations.entry(peer_id.clone()).or_insert_with(|| {
            PeerReputation::new(peer_id.clone(), self.config.clone())
        })
    }

    /// Gets reputation for a peer.
    pub fn get(&self, peer_id: &PeerId) -> Option<&PeerReputation> {
        self.reputations.get(peer_id)
    }

    /// Records a behavior event for a peer.
    pub fn record_event(&mut self, peer_id: &PeerId, event: BehaviorEvent) {
        let rep = self.get_or_create(peer_id);
        rep.record_event(event);
    }

    /// Records a response with latency.
    pub fn record_response(&mut self, peer_id: &PeerId, latency: Duration) {
        self.record_event(peer_id, BehaviorEvent::new(
            BehaviorEventType::ResponseReceived { latency },
        ));
    }

    /// Records a timeout.
    pub fn record_timeout(&mut self, peer_id: &PeerId) {
        self.record_event(peer_id, BehaviorEvent::new(BehaviorEventType::ResponseTimeout));
    }

    /// Records a valid proof.
    pub fn record_valid_proof(&mut self, peer_id: &PeerId) {
        self.record_event(peer_id, BehaviorEvent::new(BehaviorEventType::ValidProof));
    }

    /// Records an invalid proof.
    pub fn record_invalid_proof(&mut self, peer_id: &PeerId) {
        self.record_event(peer_id, BehaviorEvent::new(BehaviorEventType::InvalidProof));
    }

    /// Records a valid gradient.
    pub fn record_valid_gradient(&mut self, peer_id: &PeerId) {
        self.record_event(peer_id, BehaviorEvent::new(BehaviorEventType::ValidGradient));
    }

    /// Records an invalid gradient.
    pub fn record_invalid_gradient(&mut self, peer_id: &PeerId) {
        self.record_event(peer_id, BehaviorEvent::new(BehaviorEventType::InvalidGradient));
    }

    /// Records a heartbeat.
    pub fn record_heartbeat(&mut self, peer_id: &PeerId) {
        self.record_event(peer_id, BehaviorEvent::new(BehaviorEventType::Heartbeat));
    }

    /// Records a disconnect.
    pub fn record_disconnect(&mut self, peer_id: &PeerId, graceful: bool) {
        self.record_event(peer_id, BehaviorEvent::new(
            BehaviorEventType::Disconnect { graceful },
        ));
    }

    /// Applies decay to all reputations.
    pub fn apply_decay(&mut self) {
        if self.last_decay.elapsed() < self.decay_interval {
            return;
        }

        let hours_elapsed = self.last_decay.elapsed().as_secs_f64() / 3600.0;

        for rep in self.reputations.values_mut() {
            rep.apply_decay(hours_elapsed);
            rep.check_ban_expiry();
        }

        self.last_decay = Instant::now();
    }

    /// Returns all good peers (above threshold, not banned).
    pub fn good_peers(&self) -> Vec<PeerId> {
        self.reputations
            .iter()
            .filter(|(_, rep)| rep.is_good())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns peers sorted by reputation score.
    pub fn peers_by_score(&self) -> Vec<(PeerId, f64)> {
        let mut peers: Vec<_> = self.reputations
            .iter()
            .filter(|(_, rep)| !rep.is_banned)
            .map(|(id, rep)| (id.clone(), rep.score))
            .collect();

        peers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peers
    }

    /// Returns banned peers.
    pub fn banned_peers(&self) -> Vec<PeerId> {
        self.reputations
            .iter()
            .filter(|(_, rep)| rep.is_banned)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns score for a peer.
    pub fn score(&self, peer_id: &PeerId) -> f64 {
        self.reputations
            .get(peer_id)
            .map(|r| r.score)
            .unwrap_or(self.config.initial_score)
    }

    /// Bans a peer manually.
    pub fn ban(&mut self, peer_id: &PeerId, duration: Duration) {
        if let Some(rep) = self.reputations.get_mut(peer_id) {
            rep.ban(duration);
        }
    }

    /// Unbans a peer.
    pub fn unban(&mut self, peer_id: &PeerId) {
        if let Some(rep) = self.reputations.get_mut(peer_id) {
            rep.unban();
        }
    }

    /// Returns number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.reputations.len()
    }

    /// Removes a peer's reputation record.
    pub fn remove(&mut self, peer_id: &PeerId) {
        self.reputations.remove(peer_id);
    }

    /// Returns statistics summary.
    pub fn stats(&self) -> ReputationStats {
        let total = self.reputations.len();
        let good = self.reputations.values().filter(|r| r.is_good()).count();
        let banned = self.reputations.values().filter(|r| r.is_banned).count();

        let avg_score = if total > 0 {
            self.reputations.values().map(|r| r.score).sum::<f64>() / total as f64
        } else {
            0.0
        };

        ReputationStats {
            total_peers: total,
            good_peers: good,
            banned_peers: banned,
            average_score: avg_score,
        }
    }
}

/// Reputation statistics.
#[derive(Debug, Clone)]
pub struct ReputationStats {
    /// Total tracked peers.
    pub total_peers: usize,
    /// Peers with good reputation.
    pub good_peers: usize,
    /// Banned peers.
    pub banned_peers: usize,
    /// Average reputation score.
    pub average_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_reputation() {
        let config = ReputationConfig::default();
        let rep = PeerReputation::new(PeerId::from_string("peer1"), config.clone());

        assert_eq!(rep.score, config.initial_score);
        assert!(!rep.is_banned);
    }

    #[test]
    fn test_positive_events() {
        let config = ReputationConfig::default();
        let mut rep = PeerReputation::new(PeerId::from_string("peer1"), config);

        let initial_score = rep.score;

        // Record positive events
        rep.record_event(BehaviorEvent::new(BehaviorEventType::ValidProof));
        rep.record_event(BehaviorEvent::new(BehaviorEventType::ValidGradient));

        assert!(rep.score > initial_score);
    }

    #[test]
    fn test_negative_events() {
        let config = ReputationConfig::default();
        let mut rep = PeerReputation::new(PeerId::from_string("peer1"), config);

        let initial_score = rep.score;

        // Record negative events
        rep.record_event(BehaviorEvent::new(BehaviorEventType::InvalidProof));

        assert!(rep.score < initial_score);
    }

    #[test]
    fn test_automatic_ban() {
        let config = ReputationConfig {
            ban_threshold: 30.0,
            ..Default::default()
        };
        let mut rep = PeerReputation::new(PeerId::from_string("peer1"), config);

        // Record many negative events
        for _ in 0..10 {
            rep.record_event(BehaviorEvent::new(BehaviorEventType::InvalidProof));
        }

        assert!(rep.is_banned);
    }

    #[test]
    fn test_reputation_manager() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);

        let peer = PeerId::from_string("peer1");

        // Record events
        manager.record_valid_proof(&peer);
        manager.record_response(&peer, Duration::from_millis(100));

        let score = manager.score(&peer);
        assert!(score > 50.0);
    }

    #[test]
    fn test_good_peers_filter() {
        let config = ReputationConfig {
            good_threshold: 60.0,
            ..Default::default()
        };
        let mut manager = ReputationManager::new(config);

        // Add a good peer
        let good_peer = PeerId::from_string("good");
        for _ in 0..5 {
            manager.record_valid_proof(&good_peer);
        }

        // Add a bad peer
        let bad_peer = PeerId::from_string("bad");
        for _ in 0..5 {
            manager.record_invalid_proof(&bad_peer);
        }

        let good_peers = manager.good_peers();
        assert!(good_peers.contains(&good_peer));
        assert!(!good_peers.contains(&bad_peer));
    }

    #[test]
    fn test_latency_tracking() {
        let config = ReputationConfig::default();
        let mut rep = PeerReputation::new(PeerId::from_string("peer1"), config);

        rep.record_event(BehaviorEvent::new(
            BehaviorEventType::ResponseReceived { latency: Duration::from_millis(100) },
        ));
        rep.record_event(BehaviorEvent::new(
            BehaviorEventType::ResponseReceived { latency: Duration::from_millis(200) },
        ));

        let avg = rep.average_latency().unwrap();
        assert_eq!(avg, Duration::from_millis(150));
    }
}
