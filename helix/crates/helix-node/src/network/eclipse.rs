//! Eclipse Attack Prevention for HELIX Network.
//!
//! Prevents Eclipse attacks where an attacker surrounds a node with malicious
//! peers, isolating it from the honest network. Key mechanisms:
//! - Diverse peer sourcing from multiple discovery channels
//! - Geographic/network topology diversity requirements
//! - Outbound-only connection preferences
//! - Peer table partitioning
//! - Churn resistance

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::IpAddr;
use std::time::{Duration, Instant};

use super::messages::PeerId;
use serde::{Deserialize, Serialize};

/// Configuration for Eclipse attack prevention.
#[derive(Debug, Clone)]
pub struct EclipsePreventionConfig {
    /// Minimum number of distinct IP prefixes (/16) in peer table.
    pub min_ip_diversity: usize,
    /// Minimum number of distinct ASNs (Autonomous System Numbers).
    pub min_asn_diversity: usize,
    /// Maximum peers from same /16 prefix.
    pub max_peers_per_prefix: usize,
    /// Maximum peers from same ASN.
    pub max_peers_per_asn: usize,
    /// Minimum outbound connections.
    pub min_outbound_connections: usize,
    /// Maximum inbound connections.
    pub max_inbound_connections: usize,
    /// Number of protected peer slots (anchors).
    pub anchor_slots: usize,
    /// Minimum peer age before eviction eligible.
    pub min_peer_age: Duration,
    /// Peer rotation interval.
    pub rotation_interval: Duration,
    /// Number of peer table buckets.
    pub bucket_count: usize,
    /// Peers per bucket.
    pub peers_per_bucket: usize,
    /// Enable peer shuffling on startup.
    pub shuffle_on_startup: bool,
    /// Use multiple discovery sources.
    pub use_multiple_sources: bool,
    /// Minimum discovery sources required.
    pub min_discovery_sources: usize,
}

impl Default for EclipsePreventionConfig {
    fn default() -> Self {
        Self {
            min_ip_diversity: 8,
            min_asn_diversity: 4,
            max_peers_per_prefix: 4,
            max_peers_per_asn: 8,
            min_outbound_connections: 8,
            max_inbound_connections: 125,
            anchor_slots: 4,
            min_peer_age: Duration::from_secs(300), // 5 minutes
            rotation_interval: Duration::from_secs(3600), // 1 hour
            bucket_count: 16,
            peers_per_bucket: 8,
            shuffle_on_startup: true,
            use_multiple_sources: true,
            min_discovery_sources: 2,
        }
    }
}

/// Peer connection type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConnectionType {
    /// We initiated the connection.
    Outbound,
    /// They initiated the connection.
    Inbound,
}

/// Source of peer discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscoverySource {
    /// DNS seeds.
    DnsSeed,
    /// DHT (Kademlia).
    Dht,
    /// mDNS (local network).
    Mdns,
    /// Peer exchange.
    PeerExchange,
    /// Manual configuration.
    Manual,
    /// Bootstrap nodes.
    Bootstrap,
    /// Unknown source.
    Unknown,
}

/// Peer network information for diversity tracking.
#[derive(Debug, Clone)]
pub struct PeerNetworkInfo {
    /// Peer ID.
    pub peer_id: PeerId,
    /// IP address.
    pub ip: Option<IpAddr>,
    /// IP prefix (/16 for IPv4).
    pub ip_prefix: Option<String>,
    /// ASN (Autonomous System Number).
    pub asn: Option<u32>,
    /// Geographic region.
    pub region: Option<String>,
    /// Connection type.
    pub connection_type: ConnectionType,
    /// Discovery source.
    pub discovery_source: DiscoverySource,
    /// Connection timestamp.
    pub connected_at: Instant,
    /// Whether this is an anchor peer (protected).
    pub is_anchor: bool,
    /// Bucket assignment.
    pub bucket: Option<usize>,
    /// Last activity timestamp.
    pub last_activity: Instant,
}

impl PeerNetworkInfo {
    /// Creates new peer network info.
    pub fn new(
        peer_id: PeerId,
        ip: Option<IpAddr>,
        connection_type: ConnectionType,
        discovery_source: DiscoverySource,
    ) -> Self {
        let ip_prefix = ip.map(|addr| Self::extract_prefix(&addr));
        let now = Instant::now();

        Self {
            peer_id,
            ip,
            ip_prefix,
            asn: None, // Would be looked up from IP
            region: None,
            connection_type,
            discovery_source,
            connected_at: now,
            is_anchor: false,
            bucket: None,
            last_activity: now,
        }
    }

    /// Extracts /16 prefix from IP address.
    fn extract_prefix(ip: &IpAddr) -> String {
        match ip {
            IpAddr::V4(v4) => {
                let octets = v4.octets();
                format!("{}.{}", octets[0], octets[1])
            }
            IpAddr::V6(v6) => {
                let segments = v6.segments();
                format!("{:x}:{:x}", segments[0], segments[1])
            }
        }
    }

    /// Returns peer age.
    pub fn age(&self) -> Duration {
        self.connected_at.elapsed()
    }

    /// Updates last activity timestamp.
    pub fn update_activity(&mut self) {
        self.last_activity = Instant::now();
    }
}

/// Statistics about peer diversity.
#[derive(Debug, Clone, Default)]
pub struct DiversityStats {
    /// Number of unique IP prefixes.
    pub unique_prefixes: usize,
    /// Number of unique ASNs.
    pub unique_asns: usize,
    /// Number of unique regions.
    pub unique_regions: usize,
    /// Number of outbound connections.
    pub outbound_count: usize,
    /// Number of inbound connections.
    pub inbound_count: usize,
    /// Number of discovery sources used.
    pub discovery_sources: usize,
    /// Overall diversity score (0.0-1.0).
    pub diversity_score: f64,
    /// Whether diversity requirements are met.
    pub requirements_met: bool,
}

/// Peer eviction candidate.
#[derive(Debug, Clone)]
struct EvictionCandidate {
    peer_id: PeerId,
    score: f64,
    reason: EvictionReason,
}

/// Reason for eviction.
#[derive(Debug, Clone)]
pub enum EvictionReason {
    /// Too many peers from same prefix.
    PrefixOverflow,
    /// Too many peers from same ASN.
    AsnOverflow,
    /// Too many inbound connections.
    InboundOverflow,
    /// Lowest activity.
    LowActivity,
    /// Failed diversity check.
    DiversityViolation,
    /// Manual eviction.
    Manual,
    /// Connection timeout.
    Timeout,
}

/// Eclipse-resistant peer manager.
///
/// Manages peer connections to maintain network diversity and
/// prevent Eclipse attacks.
pub struct EclipseResistantPeerManager {
    /// Configuration.
    config: EclipsePreventionConfig,
    /// Connected peers.
    peers: HashMap<PeerId, PeerNetworkInfo>,
    /// Peer buckets for address partitioning.
    buckets: Vec<Vec<PeerId>>,
    /// Peers by IP prefix.
    by_prefix: HashMap<String, HashSet<PeerId>>,
    /// Peers by ASN.
    by_asn: HashMap<u32, HashSet<PeerId>>,
    /// Peers by discovery source.
    by_source: HashMap<DiscoverySource, HashSet<PeerId>>,
    /// Anchor peers (protected from eviction).
    anchors: HashSet<PeerId>,
    /// Eviction candidates queue.
    eviction_queue: VecDeque<EvictionCandidate>,
    /// Last rotation time.
    last_rotation: Instant,
    /// Statistics.
    stats: DiversityStats,
}

impl EclipseResistantPeerManager {
    /// Creates a new peer manager.
    pub fn new(config: EclipsePreventionConfig) -> Self {
        let buckets = vec![Vec::new(); config.bucket_count];

        Self {
            config,
            peers: HashMap::new(),
            buckets,
            by_prefix: HashMap::new(),
            by_asn: HashMap::new(),
            by_source: HashMap::new(),
            anchors: HashSet::new(),
            eviction_queue: VecDeque::new(),
            last_rotation: Instant::now(),
            stats: DiversityStats::default(),
        }
    }

    /// Attempts to add a new peer connection.
    ///
    /// Returns whether the peer was accepted.
    pub fn try_add_peer(&mut self, info: PeerNetworkInfo) -> Result<(), EclipseError> {
        // Check connection limits
        if info.connection_type == ConnectionType::Inbound {
            let inbound_count = self.count_by_type(ConnectionType::Inbound);
            if inbound_count >= self.config.max_inbound_connections {
                // Try to evict a peer to make room
                if !self.evict_for_new_peer(&info) {
                    return Err(EclipseError::ConnectionLimitReached {
                        current: inbound_count,
                        max: self.config.max_inbound_connections,
                    });
                }
            }
        }

        // Check prefix limits
        if let Some(ref prefix) = info.ip_prefix {
            let prefix_count = self.by_prefix
                .get(prefix)
                .map(|s| s.len())
                .unwrap_or(0);
            if prefix_count >= self.config.max_peers_per_prefix {
                return Err(EclipseError::PrefixLimitReached {
                    prefix: prefix.clone(),
                    count: prefix_count,
                    max: self.config.max_peers_per_prefix,
                });
            }
        }

        // Check ASN limits
        if let Some(asn) = info.asn {
            let asn_count = self.by_asn
                .get(&asn)
                .map(|s| s.len())
                .unwrap_or(0);
            if asn_count >= self.config.max_peers_per_asn {
                return Err(EclipseError::AsnLimitReached {
                    asn,
                    count: asn_count,
                    max: self.config.max_peers_per_asn,
                });
            }
        }

        self.add_peer_internal(info);
        self.update_stats();

        Ok(())
    }

    /// Adds peer without checks (internal use).
    fn add_peer_internal(&mut self, mut info: PeerNetworkInfo) {
        // Assign to bucket
        let bucket_idx = self.assign_bucket(&info.peer_id);
        info.bucket = Some(bucket_idx);

        // Update tracking maps
        if let Some(ref prefix) = info.ip_prefix {
            self.by_prefix
                .entry(prefix.clone())
                .or_insert_with(HashSet::new)
                .insert(info.peer_id.clone());
        }

        if let Some(asn) = info.asn {
            self.by_asn
                .entry(asn)
                .or_insert_with(HashSet::new)
                .insert(info.peer_id.clone());
        }

        self.by_source
            .entry(info.discovery_source)
            .or_insert_with(HashSet::new)
            .insert(info.peer_id.clone());

        // Add to bucket
        self.buckets[bucket_idx].push(info.peer_id.clone());

        // Store peer info
        self.peers.insert(info.peer_id.clone(), info);
    }

    /// Assigns a peer to a bucket based on peer ID hash.
    fn assign_bucket(&self, peer_id: &PeerId) -> usize {
        // Simple hash-based bucket assignment
        let hash: u64 = peer_id.0.bytes()
            .enumerate()
            .map(|(i, b)| (b as u64) << (i % 8 * 8))
            .sum();
        (hash as usize) % self.config.bucket_count
    }

    /// Removes a peer.
    pub fn remove_peer(&mut self, peer_id: &PeerId) {
        if let Some(info) = self.peers.remove(peer_id) {
            // Remove from tracking maps
            if let Some(ref prefix) = info.ip_prefix {
                if let Some(set) = self.by_prefix.get_mut(prefix) {
                    set.remove(peer_id);
                }
            }

            if let Some(asn) = info.asn {
                if let Some(set) = self.by_asn.get_mut(&asn) {
                    set.remove(peer_id);
                }
            }

            if let Some(set) = self.by_source.get_mut(&info.discovery_source) {
                set.remove(peer_id);
            }

            // Remove from bucket
            if let Some(bucket_idx) = info.bucket {
                self.buckets[bucket_idx].retain(|id| id != peer_id);
            }

            // Remove from anchors
            self.anchors.remove(peer_id);
        }

        self.update_stats();
    }

    /// Marks a peer as an anchor (protected from eviction).
    pub fn set_anchor(&mut self, peer_id: &PeerId, is_anchor: bool) {
        if is_anchor && self.anchors.len() < self.config.anchor_slots {
            self.anchors.insert(peer_id.clone());
            if let Some(info) = self.peers.get_mut(peer_id) {
                info.is_anchor = true;
            }
        } else if !is_anchor {
            self.anchors.remove(peer_id);
            if let Some(info) = self.peers.get_mut(peer_id) {
                info.is_anchor = false;
            }
        }
    }

    /// Attempts to evict a peer to make room for a new one.
    fn evict_for_new_peer(&mut self, new_peer: &PeerNetworkInfo) -> bool {
        let candidates = self.find_eviction_candidates(new_peer);

        for candidate in candidates {
            // Skip anchors
            if self.anchors.contains(&candidate.peer_id) {
                continue;
            }

            // Check minimum age
            if let Some(info) = self.peers.get(&candidate.peer_id) {
                if info.age() < self.config.min_peer_age {
                    continue;
                }
            }

            // Evict this peer
            self.remove_peer(&candidate.peer_id);
            return true;
        }

        false
    }

    /// Finds eviction candidates.
    fn find_eviction_candidates(&self, new_peer: &PeerNetworkInfo) -> Vec<EvictionCandidate> {
        let mut candidates = Vec::new();

        for (peer_id, info) in &self.peers {
            if info.is_anchor {
                continue;
            }

            let mut score = 0.0;
            let mut reason = EvictionReason::LowActivity;

            // Prefer evicting inbound connections
            if info.connection_type == ConnectionType::Inbound {
                score += 1.0;
                reason = EvictionReason::InboundOverflow;
            }

            // Prefer evicting from overrepresented prefixes
            if let (Some(ref new_prefix), Some(ref info_prefix)) = (&new_peer.ip_prefix, &info.ip_prefix) {
                if new_prefix != info_prefix {
                    let prefix_count = self.by_prefix
                        .get(info_prefix)
                        .map(|s| s.len())
                        .unwrap_or(0);
                    if prefix_count > 1 {
                        score += prefix_count as f64 * 0.5;
                        reason = EvictionReason::PrefixOverflow;
                    }
                }
            }

            // Prefer evicting from same discovery source
            if info.discovery_source == new_peer.discovery_source {
                score += 0.5;
            }

            // Prefer evicting older, less active peers
            let inactivity = info.last_activity.elapsed().as_secs() as f64 / 3600.0;
            score += inactivity * 0.1;

            candidates.push(EvictionCandidate {
                peer_id: peer_id.clone(),
                score,
                reason,
            });
        }

        // Sort by score (higher = more likely to evict)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        candidates
    }

    /// Counts peers by connection type.
    fn count_by_type(&self, conn_type: ConnectionType) -> usize {
        self.peers
            .values()
            .filter(|info| info.connection_type == conn_type)
            .count()
    }

    /// Updates diversity statistics.
    fn update_stats(&mut self) {
        self.stats.unique_prefixes = self.by_prefix.len();
        self.stats.unique_asns = self.by_asn.len();
        self.stats.outbound_count = self.count_by_type(ConnectionType::Outbound);
        self.stats.inbound_count = self.count_by_type(ConnectionType::Inbound);
        self.stats.discovery_sources = self.by_source.len();

        // Calculate diversity score
        let prefix_score = (self.stats.unique_prefixes as f64 / self.config.min_ip_diversity as f64).min(1.0);
        let asn_score = (self.stats.unique_asns as f64 / self.config.min_asn_diversity as f64).min(1.0);
        let outbound_ratio = if self.stats.outbound_count + self.stats.inbound_count > 0 {
            self.stats.outbound_count as f64 / (self.stats.outbound_count + self.stats.inbound_count) as f64
        } else {
            0.0
        };
        let source_score = (self.stats.discovery_sources as f64 / self.config.min_discovery_sources as f64).min(1.0);

        self.stats.diversity_score = (prefix_score + asn_score + outbound_ratio + source_score) / 4.0;

        self.stats.requirements_met =
            self.stats.unique_prefixes >= self.config.min_ip_diversity &&
            self.stats.unique_asns >= self.config.min_asn_diversity &&
            self.stats.outbound_count >= self.config.min_outbound_connections &&
            self.stats.discovery_sources >= self.config.min_discovery_sources;
    }

    /// Returns current diversity statistics.
    pub fn diversity_stats(&self) -> &DiversityStats {
        &self.stats
    }

    /// Checks if diversity requirements are met.
    pub fn check_diversity(&self) -> DiversityCheckResult {
        let mut violations = Vec::new();

        if self.stats.unique_prefixes < self.config.min_ip_diversity {
            violations.push(DiversityViolation::InsufficientPrefixes {
                current: self.stats.unique_prefixes,
                required: self.config.min_ip_diversity,
            });
        }

        if self.stats.unique_asns < self.config.min_asn_diversity {
            violations.push(DiversityViolation::InsufficientAsns {
                current: self.stats.unique_asns,
                required: self.config.min_asn_diversity,
            });
        }

        if self.stats.outbound_count < self.config.min_outbound_connections {
            violations.push(DiversityViolation::InsufficientOutbound {
                current: self.stats.outbound_count,
                required: self.config.min_outbound_connections,
            });
        }

        if self.stats.discovery_sources < self.config.min_discovery_sources {
            violations.push(DiversityViolation::InsufficientSources {
                current: self.stats.discovery_sources,
                required: self.config.min_discovery_sources,
            });
        }

        DiversityCheckResult {
            passed: violations.is_empty(),
            diversity_score: self.stats.diversity_score,
            violations,
        }
    }

    /// Performs peer rotation to maintain diversity.
    pub fn rotate_peers(&mut self) -> Vec<PeerId> {
        if self.last_rotation.elapsed() < self.config.rotation_interval {
            return Vec::new();
        }

        self.last_rotation = Instant::now();

        // Find peers to rotate (oldest non-anchor peers)
        let mut rotation_candidates: Vec<_> = self.peers
            .iter()
            .filter(|(id, info)| !info.is_anchor && info.age() > self.config.min_peer_age)
            .map(|(id, info)| (id.clone(), info.connected_at))
            .collect();

        rotation_candidates.sort_by_key(|(_, time)| *time);

        // Rotate oldest 10% of eligible peers
        let rotate_count = (rotation_candidates.len() / 10).max(1);
        let to_rotate: Vec<PeerId> = rotation_candidates
            .into_iter()
            .take(rotate_count)
            .map(|(id, _)| id)
            .collect();

        for peer_id in &to_rotate {
            self.remove_peer(peer_id);
        }

        to_rotate
    }

    /// Returns peer count.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Returns all connected peer IDs.
    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.peers.keys().cloned().collect()
    }

    /// Returns peer info.
    pub fn get_peer(&self, peer_id: &PeerId) -> Option<&PeerNetworkInfo> {
        self.peers.get(peer_id)
    }

    /// Updates peer activity.
    pub fn update_peer_activity(&mut self, peer_id: &PeerId) {
        if let Some(info) = self.peers.get_mut(peer_id) {
            info.update_activity();
        }
    }

    /// Gets peers from a specific discovery source.
    pub fn peers_from_source(&self, source: DiscoverySource) -> Vec<PeerId> {
        self.by_source
            .get(&source)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Suggests outbound connection targets for diversity.
    pub fn suggest_connections(&self, count: usize) -> Vec<ConnectionSuggestion> {
        let mut suggestions = Vec::new();

        // Prioritize discovery sources we're missing
        for source in [
            DiscoverySource::DnsSeed,
            DiscoverySource::Dht,
            DiscoverySource::Bootstrap,
            DiscoverySource::PeerExchange,
        ] {
            if !self.by_source.contains_key(&source) {
                suggestions.push(ConnectionSuggestion {
                    source,
                    reason: "Missing discovery source".to_string(),
                    priority: 1,
                });
            }
        }

        // Suggest adding outbound connections if below minimum
        if self.stats.outbound_count < self.config.min_outbound_connections {
            let needed = self.config.min_outbound_connections - self.stats.outbound_count;
            suggestions.push(ConnectionSuggestion {
                source: DiscoverySource::Dht,
                reason: format!("Need {} more outbound connections", needed),
                priority: 2,
            });
        }

        suggestions.truncate(count);
        suggestions
    }
}

/// Suggestion for new connections.
#[derive(Debug, Clone)]
pub struct ConnectionSuggestion {
    /// Suggested discovery source.
    pub source: DiscoverySource,
    /// Reason for suggestion.
    pub reason: String,
    /// Priority (lower = higher priority).
    pub priority: u32,
}

/// Result of diversity check.
#[derive(Debug, Clone)]
pub struct DiversityCheckResult {
    /// Whether all requirements passed.
    pub passed: bool,
    /// Overall diversity score.
    pub diversity_score: f64,
    /// List of violations.
    pub violations: Vec<DiversityViolation>,
}

/// Type of diversity violation.
#[derive(Debug, Clone)]
pub enum DiversityViolation {
    /// Insufficient IP prefix diversity.
    InsufficientPrefixes { current: usize, required: usize },
    /// Insufficient ASN diversity.
    InsufficientAsns { current: usize, required: usize },
    /// Insufficient outbound connections.
    InsufficientOutbound { current: usize, required: usize },
    /// Insufficient discovery sources.
    InsufficientSources { current: usize, required: usize },
}

/// Eclipse prevention errors.
#[derive(Debug, Clone)]
pub enum EclipseError {
    /// Connection limit reached.
    ConnectionLimitReached { current: usize, max: usize },
    /// Too many peers from same prefix.
    PrefixLimitReached { prefix: String, count: usize, max: usize },
    /// Too many peers from same ASN.
    AsnLimitReached { asn: u32, count: usize, max: usize },
    /// Diversity check failed.
    DiversityCheckFailed(Vec<DiversityViolation>),
}

impl std::fmt::Display for EclipseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionLimitReached { current, max } => {
                write!(f, "Connection limit reached: {}/{}", current, max)
            }
            Self::PrefixLimitReached { prefix, count, max } => {
                write!(f, "Prefix {} limit reached: {}/{}", prefix, count, max)
            }
            Self::AsnLimitReached { asn, count, max } => {
                write!(f, "ASN {} limit reached: {}/{}", asn, count, max)
            }
            Self::DiversityCheckFailed(violations) => {
                write!(f, "Diversity check failed: {} violations", violations.len())
            }
        }
    }
}

impl std::error::Error for EclipseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_peer_addition() {
        let config = EclipsePreventionConfig::default();
        let mut manager = EclipseResistantPeerManager::new(config);

        let peer = PeerNetworkInfo::new(
            PeerId::from_string("peer1"),
            Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))),
            ConnectionType::Outbound,
            DiscoverySource::Dht,
        );

        let result = manager.try_add_peer(peer);
        assert!(result.is_ok());
        assert_eq!(manager.peer_count(), 1);
    }

    #[test]
    fn test_prefix_limit() {
        let config = EclipsePreventionConfig {
            max_peers_per_prefix: 2,
            ..Default::default()
        };
        let mut manager = EclipseResistantPeerManager::new(config);

        // Add peers from same prefix
        for i in 0..3 {
            let peer = PeerNetworkInfo::new(
                PeerId::from_string(format!("peer{}", i)),
                Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, i as u8))),
                ConnectionType::Outbound,
                DiscoverySource::Dht,
            );

            let result = manager.try_add_peer(peer);
            if i < 2 {
                assert!(result.is_ok());
            } else {
                assert!(matches!(result, Err(EclipseError::PrefixLimitReached { .. })));
            }
        }
    }

    #[test]
    fn test_diversity_tracking() {
        let config = EclipsePreventionConfig::default();
        let mut manager = EclipseResistantPeerManager::new(config);

        // Add peers from different prefixes
        for i in 0..5 {
            let peer = PeerNetworkInfo::new(
                PeerId::from_string(format!("peer{}", i)),
                Some(IpAddr::V4(Ipv4Addr::new(10 + i as u8, 0, 0, 1))),
                ConnectionType::Outbound,
                DiscoverySource::Dht,
            );
            manager.try_add_peer(peer).ok();
        }

        let stats = manager.diversity_stats();
        assert_eq!(stats.unique_prefixes, 5);
        assert_eq!(stats.outbound_count, 5);
    }

    #[test]
    fn test_anchor_protection() {
        let config = EclipsePreventionConfig {
            anchor_slots: 2,
            ..Default::default()
        };
        let mut manager = EclipseResistantPeerManager::new(config);

        let peer_id = PeerId::from_string("anchor_peer");
        let peer = PeerNetworkInfo::new(
            peer_id.clone(),
            Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))),
            ConnectionType::Outbound,
            DiscoverySource::Bootstrap,
        );

        manager.try_add_peer(peer).ok();
        manager.set_anchor(&peer_id, true);

        assert!(manager.get_peer(&peer_id).unwrap().is_anchor);
    }

    #[test]
    fn test_connection_type_counting() {
        let config = EclipsePreventionConfig::default();
        let mut manager = EclipseResistantPeerManager::new(config);

        // Add outbound
        let peer1 = PeerNetworkInfo::new(
            PeerId::from_string("peer1"),
            Some(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))),
            ConnectionType::Outbound,
            DiscoverySource::Dht,
        );
        manager.try_add_peer(peer1).ok();

        // Add inbound
        let peer2 = PeerNetworkInfo::new(
            PeerId::from_string("peer2"),
            Some(IpAddr::V4(Ipv4Addr::new(20, 0, 0, 1))),
            ConnectionType::Inbound,
            DiscoverySource::PeerExchange,
        );
        manager.try_add_peer(peer2).ok();

        let stats = manager.diversity_stats();
        assert_eq!(stats.outbound_count, 1);
        assert_eq!(stats.inbound_count, 1);
    }
}
