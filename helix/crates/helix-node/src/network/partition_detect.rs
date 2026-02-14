//! Network Partition Detection
//!
//! Detects network partitions and connectivity issues that could affect
//! distributed training consensus and data availability.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use helix_core::Hash;

/// Errors during partition detection
#[derive(Debug, Error)]
pub enum PartitionError {
    #[error("Insufficient peers for partition detection: have {have}, need {need}")]
    InsufficientPeers { have: usize, need: usize },

    #[error("Network appears partitioned: {0}")]
    PartitionDetected(String),

    #[error("Connectivity check failed: {0}")]
    ConnectivityFailed(String),

    #[error("Consensus view diverged")]
    ConsensusDivergence,

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Configuration for partition detection
#[derive(Debug, Clone)]
pub struct PartitionDetectionConfig {
    /// Minimum peers required for partition detection
    pub min_peers_for_detection: usize,
    /// How often to run partition detection
    pub detection_interval: Duration,
    /// Timeout for peer connectivity checks
    pub connectivity_timeout: Duration,
    /// Number of recent heartbeats to track per peer
    pub heartbeat_history_size: usize,
    /// Maximum acceptable heartbeat miss ratio
    pub max_heartbeat_miss_ratio: f64,
    /// Minimum peers that must agree on consensus view
    pub min_consensus_agreement: f64,
    /// Duration to consider a peer unreachable
    pub unreachable_threshold: Duration,
    /// Enable active probing
    pub enable_active_probing: bool,
    /// Number of random peers to probe each interval
    pub probe_sample_size: usize,
    /// Threshold for declaring network partition
    pub partition_threshold: f64,
    /// Recovery check interval after partition detected
    pub recovery_check_interval: Duration,
}

impl Default for PartitionDetectionConfig {
    fn default() -> Self {
        Self {
            min_peers_for_detection: 3,
            detection_interval: Duration::from_secs(30),
            connectivity_timeout: Duration::from_secs(5),
            heartbeat_history_size: 20,
            max_heartbeat_miss_ratio: 0.3,
            min_consensus_agreement: 0.67,
            unreachable_threshold: Duration::from_secs(60),
            enable_active_probing: true,
            probe_sample_size: 5,
            partition_threshold: 0.5,
            recovery_check_interval: Duration::from_secs(10),
        }
    }
}

/// Connectivity status of a peer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerConnectivity {
    /// Peer is connected and responsive
    Connected,
    /// Peer connection is degraded (high latency, packet loss)
    Degraded,
    /// Peer is unreachable
    Unreachable,
    /// Peer status unknown (not enough data)
    Unknown,
}

/// Information about a peer's connectivity
#[derive(Debug, Clone)]
pub struct PeerConnectivityInfo {
    /// Peer address
    pub addr: SocketAddr,
    /// Current connectivity status
    pub status: PeerConnectivity,
    /// Last successful contact
    pub last_contact: Option<Instant>,
    /// Heartbeat history (true = received, false = missed)
    pub heartbeat_history: VecDeque<bool>,
    /// Average round-trip time
    pub avg_rtt: Option<Duration>,
    /// Last known consensus view hash
    pub last_consensus_view: Option<Hash>,
    /// Peers this peer reports as connected
    pub reported_peers: HashSet<SocketAddr>,
    /// When we last updated this info
    pub updated_at: Instant,
}

impl PeerConnectivityInfo {
    fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            status: PeerConnectivity::Unknown,
            last_contact: None,
            heartbeat_history: VecDeque::new(),
            avg_rtt: None,
            last_consensus_view: None,
            reported_peers: HashSet::new(),
            updated_at: Instant::now(),
        }
    }

    fn heartbeat_miss_ratio(&self) -> f64 {
        if self.heartbeat_history.is_empty() {
            return 0.0;
        }
        let missed = self.heartbeat_history.iter().filter(|&&h| !h).count();
        missed as f64 / self.heartbeat_history.len() as f64
    }

    fn record_heartbeat(&mut self, received: bool, history_size: usize) {
        self.heartbeat_history.push_back(received);
        while self.heartbeat_history.len() > history_size {
            self.heartbeat_history.pop_front();
        }
        if received {
            self.last_contact = Some(Instant::now());
        }
        self.updated_at = Instant::now();
    }
}

/// Result of partition detection
#[derive(Debug, Clone)]
pub struct PartitionStatus {
    /// Whether network appears partitioned
    pub is_partitioned: bool,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Number of connected peers
    pub connected_peers: usize,
    /// Number of degraded peers
    pub degraded_peers: usize,
    /// Number of unreachable peers
    pub unreachable_peers: usize,
    /// Detected partition groups (sets of mutually connected peers)
    pub partition_groups: Vec<PartitionGroup>,
    /// Consensus agreement ratio
    pub consensus_agreement: f64,
    /// When this status was computed
    pub computed_at: Instant,
    /// Recommended action
    pub recommended_action: PartitionAction,
}

/// A group of mutually connected peers (partition)
#[derive(Debug, Clone)]
pub struct PartitionGroup {
    /// Peers in this group
    pub peers: HashSet<SocketAddr>,
    /// Total stake in this group (if available)
    pub total_stake: u64,
    /// Whether we're in this group
    pub contains_self: bool,
}

/// Recommended action based on partition status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionAction {
    /// Continue normal operation
    Continue,
    /// Pause training until connectivity improves
    PauseTraining,
    /// Attempt to reconnect to specific peers
    ReconnectPeers,
    /// Alert operator - manual intervention may be needed
    AlertOperator,
    /// Halt all operations - severe partition detected
    Halt,
}

/// Event for partition status changes
#[derive(Debug, Clone)]
pub enum PartitionEvent {
    /// Partition detected
    PartitionDetected(PartitionStatus),
    /// Partition resolved
    PartitionResolved,
    /// Peer became unreachable
    PeerUnreachable(SocketAddr),
    /// Peer reconnected
    PeerReconnected(SocketAddr),
    /// Consensus view divergence detected
    ConsensusDivergence {
        expected: Hash,
        divergent_peers: Vec<SocketAddr>,
    },
}

/// Network partition detector
pub struct PartitionDetector {
    config: PartitionDetectionConfig,
    peers: Arc<RwLock<HashMap<SocketAddr, PeerConnectivityInfo>>>,
    our_addr: Option<SocketAddr>,
    current_status: Arc<RwLock<Option<PartitionStatus>>>,
    our_consensus_view: Arc<RwLock<Option<Hash>>>,
    event_sender: Option<mpsc::Sender<PartitionEvent>>,
    last_detection: Arc<RwLock<Instant>>,
    partition_start: Arc<RwLock<Option<Instant>>>,
}

impl PartitionDetector {
    /// Create a new partition detector
    pub fn new(config: PartitionDetectionConfig) -> Self {
        Self {
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            our_addr: None,
            current_status: Arc::new(RwLock::new(None)),
            our_consensus_view: Arc::new(RwLock::new(None)),
            event_sender: None,
            last_detection: Arc::new(RwLock::new(Instant::now())),
            partition_start: Arc::new(RwLock::new(None)),
        }
    }

    /// Set our network address
    pub fn set_our_addr(&mut self, addr: SocketAddr) {
        self.our_addr = Some(addr);
    }

    /// Subscribe to partition events
    pub fn subscribe(&mut self) -> mpsc::Receiver<PartitionEvent> {
        let (tx, rx) = mpsc::channel(100);
        self.event_sender = Some(tx);
        rx
    }

    /// Register a peer for monitoring
    pub fn register_peer(&self, addr: SocketAddr) {
        let mut peers = self.peers.write();
        peers.entry(addr).or_insert_with(|| PeerConnectivityInfo::new(addr));
        debug!("Registered peer for partition detection: {}", addr);
    }

    /// Remove a peer from monitoring
    pub fn unregister_peer(&self, addr: &SocketAddr) {
        self.peers.write().remove(addr);
        debug!("Unregistered peer from partition detection: {}", addr);
    }

    /// Record a heartbeat from a peer
    pub fn record_heartbeat(&self, addr: &SocketAddr, rtt: Option<Duration>) {
        let mut peers = self.peers.write();
        if let Some(info) = peers.get_mut(addr) {
            info.record_heartbeat(true, self.config.heartbeat_history_size);
            if let Some(rtt) = rtt {
                // Exponential moving average for RTT
                info.avg_rtt = Some(match info.avg_rtt {
                    Some(current) => Duration::from_secs_f64(
                        current.as_secs_f64() * 0.8 + rtt.as_secs_f64() * 0.2,
                    ),
                    None => rtt,
                });
            }
            self.update_peer_status(info);
        }
    }

    /// Record a missed heartbeat for a peer
    pub fn record_missed_heartbeat(&self, addr: &SocketAddr) {
        let mut peers = self.peers.write();
        if let Some(info) = peers.get_mut(addr) {
            info.record_heartbeat(false, self.config.heartbeat_history_size);
            self.update_peer_status(info);
        }
    }

    /// Update peer's reported connected peers
    pub fn update_peer_view(&self, addr: &SocketAddr, connected_peers: HashSet<SocketAddr>) {
        let mut peers = self.peers.write();
        if let Some(info) = peers.get_mut(addr) {
            info.reported_peers = connected_peers;
            info.updated_at = Instant::now();
        }
    }

    /// Update peer's consensus view
    pub fn update_peer_consensus(&self, addr: &SocketAddr, view_hash: Hash) {
        let mut peers = self.peers.write();
        if let Some(info) = peers.get_mut(addr) {
            info.last_consensus_view = Some(view_hash);
            info.updated_at = Instant::now();
        }
    }

    /// Set our current consensus view
    pub fn set_our_consensus_view(&self, view_hash: Hash) {
        *self.our_consensus_view.write() = Some(view_hash);
    }

    fn update_peer_status(&self, info: &mut PeerConnectivityInfo) {
        let miss_ratio = info.heartbeat_miss_ratio();
        let time_since_contact = info
            .last_contact
            .map(|t| t.elapsed())
            .unwrap_or(Duration::MAX);

        info.status = if time_since_contact > self.config.unreachable_threshold {
            PeerConnectivity::Unreachable
        } else if miss_ratio > self.config.max_heartbeat_miss_ratio {
            PeerConnectivity::Degraded
        } else if info.heartbeat_history.len() >= 3 {
            PeerConnectivity::Connected
        } else {
            PeerConnectivity::Unknown
        };
    }

    /// Run partition detection
    pub async fn detect_partition(&self) -> Result<PartitionStatus, PartitionError> {
        // Build the status inside a sync block so the RwLockReadGuard is dropped
        // before any .await (parking_lot guards are !Send).
        let status = {
            let peers = self.peers.read();

            if peers.len() < self.config.min_peers_for_detection {
                return Err(PartitionError::InsufficientPeers {
                    have: peers.len(),
                    need: self.config.min_peers_for_detection,
                });
            }

            // Count peer statuses
            let mut connected = 0;
            let mut degraded = 0;
            let mut unreachable = 0;

            for info in peers.values() {
                match info.status {
                    PeerConnectivity::Connected => connected += 1,
                    PeerConnectivity::Degraded => degraded += 1,
                    PeerConnectivity::Unreachable => unreachable += 1,
                    PeerConnectivity::Unknown => {}
                }
            }

            let total_known = connected + degraded + unreachable;
            let unreachable_ratio = if total_known > 0 {
                unreachable as f64 / total_known as f64
            } else {
                0.0
            };

            // Check consensus agreement
            let consensus_agreement = self.calculate_consensus_agreement(&peers);

            // Detect partition groups using reported peer views
            let partition_groups = self.detect_partition_groups(&peers);

            // Determine if partitioned
            let is_partitioned = unreachable_ratio >= self.config.partition_threshold
                || consensus_agreement < self.config.min_consensus_agreement
                || partition_groups.len() > 1;

            // Calculate confidence
            let confidence = self.calculate_confidence(
                total_known,
                unreachable_ratio,
                consensus_agreement,
                &partition_groups,
            );

            // Determine recommended action
            let recommended_action = self.determine_action(
                is_partitioned,
                confidence,
                unreachable_ratio,
                consensus_agreement,
            );

            PartitionStatus {
                is_partitioned,
                confidence,
                connected_peers: connected,
                degraded_peers: degraded,
                unreachable_peers: unreachable,
                partition_groups,
                consensus_agreement,
                computed_at: Instant::now(),
                recommended_action,
            }
            // peers guard dropped here
        };

        // Update state and emit events (safe to .await now)
        self.handle_status_change(&status).await;

        *self.last_detection.write() = Instant::now();
        *self.current_status.write() = Some(status.clone());

        Ok(status)
    }

    fn calculate_consensus_agreement(
        &self,
        peers: &HashMap<SocketAddr, PeerConnectivityInfo>,
    ) -> f64 {
        let our_view = self.our_consensus_view.read().clone();

        let Some(our_hash) = our_view else {
            return 1.0; // No consensus view to compare
        };

        let mut agreeing = 0;
        let mut total_with_view = 0;

        for info in peers.values() {
            if info.status == PeerConnectivity::Connected {
                if let Some(ref peer_hash) = info.last_consensus_view {
                    total_with_view += 1;
                    if *peer_hash == our_hash {
                        agreeing += 1;
                    }
                }
            }
        }

        if total_with_view == 0 {
            1.0
        } else {
            agreeing as f64 / total_with_view as f64
        }
    }

    fn detect_partition_groups(
        &self,
        peers: &HashMap<SocketAddr, PeerConnectivityInfo>,
    ) -> Vec<PartitionGroup> {
        // Build adjacency from reported peer views
        let mut adjacency: HashMap<SocketAddr, HashSet<SocketAddr>> = HashMap::new();

        // Add ourselves
        if let Some(our_addr) = self.our_addr {
            let our_connected: HashSet<_> = peers
                .iter()
                .filter(|(_, info)| info.status == PeerConnectivity::Connected)
                .map(|(addr, _)| *addr)
                .collect();
            adjacency.insert(our_addr, our_connected);
        }

        // Add peer-reported connections
        for (addr, info) in peers {
            if info.status == PeerConnectivity::Connected {
                adjacency.insert(*addr, info.reported_peers.clone());
            }
        }

        // Find connected components using union-find
        let mut groups: Vec<PartitionGroup> = Vec::new();
        let mut visited: HashSet<SocketAddr> = HashSet::new();

        for start_addr in adjacency.keys() {
            if visited.contains(start_addr) {
                continue;
            }

            // BFS to find all peers in this component
            let mut component: HashSet<SocketAddr> = HashSet::new();
            let mut queue: VecDeque<SocketAddr> = VecDeque::new();
            queue.push_back(*start_addr);

            while let Some(addr) = queue.pop_front() {
                if !visited.insert(addr) {
                    continue;
                }
                component.insert(addr);

                if let Some(neighbors) = adjacency.get(&addr) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            queue.push_back(*neighbor);
                        }
                    }
                }
            }

            let contains_self = self.our_addr.map(|a| component.contains(&a)).unwrap_or(false);

            groups.push(PartitionGroup {
                peers: component,
                total_stake: 0, // Would be populated from stake registry
                contains_self,
            });
        }

        groups
    }

    fn calculate_confidence(
        &self,
        total_peers: usize,
        unreachable_ratio: f64,
        consensus_agreement: f64,
        partition_groups: &[PartitionGroup],
    ) -> f64 {
        // More peers = higher confidence in detection
        let peer_factor = (total_peers as f64 / 10.0).min(1.0);

        // Clear signals = higher confidence
        let signal_clarity = if unreachable_ratio > 0.7 || unreachable_ratio < 0.1 {
            0.9
        } else if unreachable_ratio > 0.5 || unreachable_ratio < 0.2 {
            0.7
        } else {
            0.5
        };

        // Multiple partition groups = higher confidence in partition
        let group_factor = if partition_groups.len() > 1 { 0.9 } else { 0.7 };

        // Consensus disagreement = higher confidence
        let consensus_factor = if consensus_agreement < 0.5 {
            0.9
        } else if consensus_agreement < 0.8 {
            0.7
        } else {
            0.5
        };

        (peer_factor * 0.3 + signal_clarity * 0.3 + group_factor * 0.2 + consensus_factor * 0.2)
            .min(1.0)
    }

    fn determine_action(
        &self,
        is_partitioned: bool,
        confidence: f64,
        unreachable_ratio: f64,
        consensus_agreement: f64,
    ) -> PartitionAction {
        if !is_partitioned {
            return PartitionAction::Continue;
        }

        if confidence < 0.5 {
            // Low confidence, try reconnecting
            return PartitionAction::ReconnectPeers;
        }

        if unreachable_ratio > 0.8 {
            // Almost all peers unreachable
            return PartitionAction::Halt;
        }

        if consensus_agreement < 0.3 {
            // Severe consensus divergence
            return PartitionAction::AlertOperator;
        }

        if unreachable_ratio > 0.5 {
            // Significant partition
            return PartitionAction::PauseTraining;
        }

        PartitionAction::ReconnectPeers
    }

    async fn handle_status_change(&self, new_status: &PartitionStatus) {
        let previous = self.current_status.read().clone();

        let Some(ref sender) = self.event_sender else {
            return;
        };

        // Collect all events synchronously (no .await while guards are held)
        let mut events = Vec::new();

        match (&previous, new_status.is_partitioned) {
            (None, true) | (Some(PartitionStatus { is_partitioned: false, .. }), true) => {
                // Partition detected
                *self.partition_start.write() = Some(Instant::now());
                warn!(
                    "Network partition detected: {} unreachable peers, {:.1}% consensus agreement",
                    new_status.unreachable_peers,
                    new_status.consensus_agreement * 100.0
                );
                events.push(PartitionEvent::PartitionDetected(new_status.clone()));
            }
            (Some(PartitionStatus { is_partitioned: true, .. }), false) => {
                // Partition resolved
                if let Some(start) = *self.partition_start.read() {
                    info!(
                        "Network partition resolved after {:?}",
                        start.elapsed()
                    );
                }
                *self.partition_start.write() = None;
                events.push(PartitionEvent::PartitionResolved);
            }
            _ => {}
        }

        // Check for individual peer status changes
        if let Some(ref prev) = previous {
            let peers = self.peers.read();
            for (addr, info) in peers.iter() {
                let was_unreachable = prev.unreachable_peers > 0; // Simplified check
                let is_unreachable = info.status == PeerConnectivity::Unreachable;

                if is_unreachable && !was_unreachable {
                    events.push(PartitionEvent::PeerUnreachable(*addr));
                } else if !is_unreachable && was_unreachable {
                    events.push(PartitionEvent::PeerReconnected(*addr));
                }
            }
            // peers guard dropped here
        }

        // Check for consensus divergence
        let our_view = self.our_consensus_view.read().clone();
        if let Some(our_hash) = our_view {
            let peers = self.peers.read();
            let divergent: Vec<_> = peers
                .iter()
                .filter(|(_, info)| {
                    info.status == PeerConnectivity::Connected
                        && info.last_consensus_view.as_ref().map(|h| *h != our_hash).unwrap_or(false)
                })
                .map(|(addr, _)| *addr)
                .collect();
            // peers guard dropped here

            if !divergent.is_empty() {
                events.push(PartitionEvent::ConsensusDivergence {
                    expected: our_hash,
                    divergent_peers: divergent,
                });
            }
        }

        // Now send all collected events (no guards held across .await)
        for event in events {
            let _ = sender.send(event).await;
        }
    }

    /// Get current partition status
    pub fn get_status(&self) -> Option<PartitionStatus> {
        self.current_status.read().clone()
    }

    /// Check if network is currently partitioned
    pub fn is_partitioned(&self) -> bool {
        self.current_status
            .read()
            .as_ref()
            .map(|s| s.is_partitioned)
            .unwrap_or(false)
    }

    /// Get list of unreachable peers
    pub fn get_unreachable_peers(&self) -> Vec<SocketAddr> {
        self.peers
            .read()
            .iter()
            .filter(|(_, info)| info.status == PeerConnectivity::Unreachable)
            .map(|(addr, _)| *addr)
            .collect()
    }

    /// Get peers that should be probed
    pub fn get_peers_to_probe(&self) -> Vec<SocketAddr> {
        use rand::seq::SliceRandom;

        let peers = self.peers.read();
        let candidates: Vec<_> = peers
            .iter()
            .filter(|(_, info)| {
                info.status == PeerConnectivity::Degraded
                    || info.status == PeerConnectivity::Unknown
                    || (info.status == PeerConnectivity::Connected
                        && info.updated_at.elapsed() > Duration::from_secs(30))
            })
            .map(|(addr, _)| *addr)
            .collect();

        let mut rng = rand::thread_rng();
        let mut selected: Vec<_> = candidates
            .choose_multiple(&mut rng, self.config.probe_sample_size)
            .copied()
            .collect();
        selected.shuffle(&mut rng);
        selected
    }

    /// Get connectivity statistics
    pub fn get_stats(&self) -> PartitionDetectorStats {
        let peers = self.peers.read();
        let status = self.current_status.read();

        let mut connected = 0;
        let mut degraded = 0;
        let mut unreachable = 0;
        let mut unknown = 0;
        let mut total_rtt = Duration::ZERO;
        let mut rtt_count = 0;

        for info in peers.values() {
            match info.status {
                PeerConnectivity::Connected => connected += 1,
                PeerConnectivity::Degraded => degraded += 1,
                PeerConnectivity::Unreachable => unreachable += 1,
                PeerConnectivity::Unknown => unknown += 1,
            }
            if let Some(rtt) = info.avg_rtt {
                total_rtt += rtt;
                rtt_count += 1;
            }
        }

        let avg_rtt = if rtt_count > 0 {
            Some(total_rtt / rtt_count as u32)
        } else {
            None
        };

        PartitionDetectorStats {
            total_peers: peers.len(),
            connected_peers: connected,
            degraded_peers: degraded,
            unreachable_peers: unreachable,
            unknown_peers: unknown,
            avg_rtt,
            is_partitioned: status.as_ref().map(|s| s.is_partitioned).unwrap_or(false),
            partition_duration: self.partition_start.read().map(|s| s.elapsed()),
            last_detection: self.last_detection.read().elapsed(),
        }
    }

    /// Run a background detection loop
    pub async fn run_detection_loop(self: Arc<Self>, mut shutdown: mpsc::Receiver<()>) {
        info!("Starting partition detection loop");

        loop {
            tokio::select! {
                _ = tokio::time::sleep(self.config.detection_interval) => {
                    match self.detect_partition().await {
                        Ok(status) => {
                            if status.is_partitioned {
                                warn!(
                                    "Partition status: {} groups, action: {:?}",
                                    status.partition_groups.len(),
                                    status.recommended_action
                                );
                            } else {
                                debug!(
                                    "Network healthy: {} connected, {} degraded",
                                    status.connected_peers,
                                    status.degraded_peers
                                );
                            }
                        }
                        Err(e) => {
                            debug!("Partition detection skipped: {}", e);
                        }
                    }
                }
                _ = shutdown.recv() => {
                    info!("Partition detection loop shutting down");
                    break;
                }
            }
        }
    }
}

/// Statistics from partition detector
#[derive(Debug, Clone)]
pub struct PartitionDetectorStats {
    /// Total monitored peers
    pub total_peers: usize,
    /// Connected peers
    pub connected_peers: usize,
    /// Degraded peers
    pub degraded_peers: usize,
    /// Unreachable peers
    pub unreachable_peers: usize,
    /// Unknown status peers
    pub unknown_peers: usize,
    /// Average RTT to peers
    pub avg_rtt: Option<Duration>,
    /// Whether currently partitioned
    pub is_partitioned: bool,
    /// How long we've been partitioned
    pub partition_duration: Option<Duration>,
    /// Time since last detection run
    pub last_detection: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_detector() -> PartitionDetector {
        PartitionDetector::new(PartitionDetectionConfig {
            min_peers_for_detection: 2,
            ..Default::default()
        })
    }

    #[test]
    fn test_register_peer() {
        let detector = create_detector();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        detector.register_peer(addr);

        let peers = detector.peers.read();
        assert!(peers.contains_key(&addr));
        assert_eq!(peers[&addr].status, PeerConnectivity::Unknown);
    }

    #[test]
    fn test_heartbeat_recording() {
        let detector = create_detector();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        detector.register_peer(addr);

        // Record some heartbeats
        for _ in 0..5 {
            detector.record_heartbeat(&addr, Some(Duration::from_millis(50)));
        }

        let peers = detector.peers.read();
        let info = &peers[&addr];
        assert_eq!(info.heartbeat_history.len(), 5);
        assert!(info.heartbeat_history.iter().all(|&h| h));
        assert!(info.avg_rtt.is_some());
    }

    #[test]
    fn test_missed_heartbeat_degrades_status() {
        let detector = create_detector();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        detector.register_peer(addr);

        // Record mostly missed heartbeats
        for i in 0..10 {
            if i % 3 == 0 {
                detector.record_heartbeat(&addr, None);
            } else {
                detector.record_missed_heartbeat(&addr);
            }
        }

        let peers = detector.peers.read();
        let info = &peers[&addr];
        assert!(info.heartbeat_miss_ratio() > 0.5);
    }

    #[tokio::test]
    async fn test_partition_detection_insufficient_peers() {
        let detector = create_detector();

        // Only add one peer
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        detector.register_peer(addr);

        let result = detector.detect_partition().await;
        assert!(matches!(result, Err(PartitionError::InsufficientPeers { .. })));
    }

    #[tokio::test]
    async fn test_partition_detection_healthy_network() {
        let detector = create_detector();

        // Add multiple healthy peers
        let addrs: Vec<SocketAddr> = (0..5)
            .map(|i| format!("127.0.0.1:800{}", i).parse().unwrap())
            .collect();

        for &addr in &addrs {
            detector.register_peer(addr);
            for _ in 0..5 {
                detector.record_heartbeat(&addr, Some(Duration::from_millis(50)));
            }
        }

        // Update peer views so each peer reports all other peers as connected
        // This forms a fully connected network topology
        for &addr in &addrs {
            let other_peers: HashSet<_> = addrs.iter().copied().filter(|&a| a != addr).collect();
            detector.update_peer_view(&addr, other_peers);
        }

        let status = detector.detect_partition().await.unwrap();
        assert!(!status.is_partitioned);
        assert_eq!(status.connected_peers, 5);
        assert_eq!(status.unreachable_peers, 0);
    }

    #[test]
    fn test_get_unreachable_peers() {
        let config = PartitionDetectionConfig {
            unreachable_threshold: Duration::from_millis(10),
            ..Default::default()
        };
        let detector = PartitionDetector::new(config);

        let addr1: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:8001".parse().unwrap();

        detector.register_peer(addr1);
        detector.register_peer(addr2);

        // Make addr1 connected
        for _ in 0..5 {
            detector.record_heartbeat(&addr1, Some(Duration::from_millis(50)));
        }

        // Let addr2 timeout
        std::thread::sleep(Duration::from_millis(20));

        // Record missed heartbeats to trigger unreachable
        for _ in 0..5 {
            detector.record_missed_heartbeat(&addr2);
        }

        let unreachable = detector.get_unreachable_peers();
        assert!(unreachable.contains(&addr2));
        assert!(!unreachable.contains(&addr1));
    }

    #[test]
    fn test_partition_groups() {
        let detector = create_detector();

        // Simulate two disconnected groups
        let addr1: SocketAddr = "192.168.1.1:8000".parse().unwrap();
        let addr2: SocketAddr = "192.168.1.2:8000".parse().unwrap();
        let addr3: SocketAddr = "10.0.0.1:8000".parse().unwrap();
        let addr4: SocketAddr = "10.0.0.2:8000".parse().unwrap();

        detector.register_peer(addr1);
        detector.register_peer(addr2);
        detector.register_peer(addr3);
        detector.register_peer(addr4);

        // Group 1 sees each other
        detector.update_peer_view(&addr1, HashSet::from([addr2]));
        detector.update_peer_view(&addr2, HashSet::from([addr1]));

        // Group 2 sees each other
        detector.update_peer_view(&addr3, HashSet::from([addr4]));
        detector.update_peer_view(&addr4, HashSet::from([addr3]));

        // Mark all as connected
        for addr in [addr1, addr2, addr3, addr4] {
            for _ in 0..5 {
                detector.record_heartbeat(&addr, None);
            }
        }

        let peers = detector.peers.read();
        let groups = detector.detect_partition_groups(&peers);

        // Should detect 2 partition groups
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_determine_action() {
        let detector = create_detector();

        // Not partitioned - continue
        assert_eq!(
            detector.determine_action(false, 0.9, 0.0, 1.0),
            PartitionAction::Continue
        );

        // Low confidence partition - reconnect
        assert_eq!(
            detector.determine_action(true, 0.3, 0.4, 0.8),
            PartitionAction::ReconnectPeers
        );

        // Severe partition - halt
        assert_eq!(
            detector.determine_action(true, 0.9, 0.85, 0.5),
            PartitionAction::Halt
        );

        // Consensus divergence - alert
        assert_eq!(
            detector.determine_action(true, 0.9, 0.4, 0.2),
            PartitionAction::AlertOperator
        );

        // Moderate partition - pause
        assert_eq!(
            detector.determine_action(true, 0.8, 0.6, 0.7),
            PartitionAction::PauseTraining
        );
    }
}
