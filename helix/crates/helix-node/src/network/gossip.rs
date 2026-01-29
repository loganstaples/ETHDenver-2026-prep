//! Gossip Protocol for HELIX Network.
//!
//! Implements epidemic-style gossip for message propagation,
//! with configurable fanout, TTL, and deduplication.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use super::messages::{NetworkMessage, PeerId};

/// Configuration for the gossip protocol.
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// Number of peers to forward messages to.
    pub fanout: usize,
    /// Maximum message hops (TTL).
    pub max_hops: u8,
    /// Time to keep message IDs for dedup (seconds).
    pub message_cache_ttl_secs: u64,
    /// Maximum messages in cache.
    pub max_cache_size: usize,
    /// Gossip interval (ms).
    pub gossip_interval_ms: u64,
    /// Enable eager push gossip.
    pub eager_push: bool,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 6,
            max_hops: 10,
            message_cache_ttl_secs: 300,
            max_cache_size: 10000,
            gossip_interval_ms: 100,
            eager_push: true,
        }
    }
}

/// Entry in the message cache.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Message ID.
    id: String,
    /// Timestamp when added.
    timestamp: u64,
    /// Whether this was locally originated.
    local: bool,
    /// Peers we've forwarded to.
    forwarded_to: HashSet<PeerId>,
}

/// Gossip protocol manager.
pub struct GossipProtocol {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: GossipConfig,
    /// Seen message IDs (for dedup).
    seen: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Outbound message queue.
    outbound_queue: Arc<RwLock<VecDeque<(PeerId, NetworkMessage)>>>,
    /// Statistics.
    stats: Arc<RwLock<GossipStats>>,
}

/// Gossip protocol statistics.
#[derive(Debug, Clone, Default)]
pub struct GossipStats {
    /// Messages originated locally.
    pub messages_originated: u64,
    /// Messages received.
    pub messages_received: u64,
    /// Messages forwarded.
    pub messages_forwarded: u64,
    /// Messages dropped (duplicate).
    pub messages_dropped_duplicate: u64,
    /// Messages dropped (TTL).
    pub messages_dropped_ttl: u64,
}

impl GossipProtocol {
    /// Creates a new gossip protocol manager.
    pub fn new(local_id: PeerId, config: GossipConfig) -> Self {
        Self {
            local_id,
            config,
            seen: Arc::new(RwLock::new(HashMap::new())),
            outbound_queue: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(GossipStats::default())),
        }
    }

    /// Broadcasts a message to the network via gossip.
    pub async fn broadcast(&self, message: NetworkMessage, peers: &[PeerId]) {
        // Mark as seen
        self.mark_seen(&message.id, true).await;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.messages_originated += 1;
        }

        // Queue for selected peers
        let targets = self.select_peers(peers, &HashSet::new());
        let mut queue = self.outbound_queue.write().await;
        
        for peer in targets {
            queue.push_back((peer, message.clone()));
        }
    }

    /// Handles a received gossip message.
    /// Returns (should_process, peers_to_forward_to).
    pub async fn handle_message(
        &self,
        mut message: NetworkMessage,
        from: &PeerId,
        available_peers: &[PeerId],
    ) -> (bool, Vec<PeerId>) {
        // Check if we've seen this message
        if self.has_seen(&message.id).await {
            let mut stats = self.stats.write().await;
            stats.messages_dropped_duplicate += 1;
            return (false, vec![]);
        }

        // Check TTL
        if message.hops >= self.config.max_hops {
            let mut stats = self.stats.write().await;
            stats.messages_dropped_ttl += 1;
            return (false, vec![]);
        }

        // Mark as seen
        self.mark_seen(&message.id, false).await;
        {
            let mut stats = self.stats.write().await;
            stats.messages_received += 1;
        }

        // Increment hop count
        message.increment_hops();

        // Select peers to forward to (excluding sender)
        let mut exclude = HashSet::new();
        exclude.insert(from.clone());
        exclude.insert(self.local_id.clone());
        
        let forward_to = self.select_peers(available_peers, &exclude);

        // Queue for forwarding
        if !forward_to.is_empty() {
            let mut queue = self.outbound_queue.write().await;
            let mut stats = self.stats.write().await;
            
            for peer in &forward_to {
                queue.push_back((peer.clone(), message.clone()));
                stats.messages_forwarded += 1;
            }
        }

        (true, forward_to)
    }

    /// Takes messages from the outbound queue.
    pub async fn take_outbound(&self, max: usize) -> Vec<(PeerId, NetworkMessage)> {
        let mut queue = self.outbound_queue.write().await;
        let count = max.min(queue.len());
        queue.drain(..count).collect()
    }

    /// Clears old entries from the message cache.
    pub async fn cleanup_cache(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let ttl = self.config.message_cache_ttl_secs;
        let max_size = self.config.max_cache_size;

        let mut seen = self.seen.write().await;
        
        // Remove expired entries
        seen.retain(|_, entry| now - entry.timestamp < ttl);
        
        // If still over limit, remove oldest
        while seen.len() > max_size {
            if let Some(oldest_key) = seen
                .iter()
                .min_by_key(|(_, e)| e.timestamp)
                .map(|(k, _)| k.clone())
            {
                seen.remove(&oldest_key);
            } else {
                break;
            }
        }
    }

    /// Gets the current statistics.
    pub async fn stats(&self) -> GossipStats {
        self.stats.read().await.clone()
    }

    /// Checks if a message has been seen.
    async fn has_seen(&self, id: &str) -> bool {
        self.seen.read().await.contains_key(id)
    }

    /// Marks a message as seen.
    async fn mark_seen(&self, id: &str, local: bool) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = CacheEntry {
            id: id.to_string(),
            timestamp: now,
            local,
            forwarded_to: HashSet::new(),
        };

        let mut seen = self.seen.write().await;
        seen.insert(id.to_string(), entry);
    }

    /// Selects peers for gossip propagation.
    fn select_peers(&self, available: &[PeerId], exclude: &HashSet<PeerId>) -> Vec<PeerId> {
        use rand::seq::SliceRandom;

        let mut candidates: Vec<_> = available
            .iter()
            .filter(|p| !exclude.contains(*p))
            .cloned()
            .collect();

        candidates.shuffle(&mut rand::thread_rng());
        candidates.truncate(self.config.fanout);
        candidates
    }

    /// Returns the number of pending outbound messages.
    pub async fn outbound_queue_size(&self) -> usize {
        self.outbound_queue.read().await.len()
    }

    /// Returns the cache size.
    pub async fn cache_size(&self) -> usize {
        self.seen.read().await.len()
    }
}

/// Lazy push gossip variant.
pub struct LazyPushGossip {
    /// Base gossip protocol.
    base: GossipProtocol,
    /// Message summaries (IHAVE).
    summaries: Arc<RwLock<HashMap<String, Vec<PeerId>>>>,
}

impl LazyPushGossip {
    /// Creates a new lazy push gossip instance.
    pub fn new(local_id: PeerId, config: GossipConfig) -> Self {
        Self {
            base: GossipProtocol::new(local_id, config),
            summaries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Announces availability of a message (IHAVE).
    pub async fn announce_have(&self, message_id: &str, to_peers: &[PeerId]) {
        let mut summaries = self.summaries.write().await;
        summaries.insert(message_id.to_string(), to_peers.to_vec());
    }

    /// Records interest in a message (IWANT).
    pub async fn request_message(&self, message_id: &str) -> Option<Vec<PeerId>> {
        let summaries = self.summaries.read().await;
        summaries.get(message_id).cloned()
    }

    /// Gets the base gossip protocol.
    pub fn base(&self) -> &GossipProtocol {
        &self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::messages::{HeartbeatMessage, MessagePayload};

    fn make_test_message(sender: PeerId) -> NetworkMessage {
        NetworkMessage::new(
            sender,
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 50,
            }),
        )
    }

    #[tokio::test]
    async fn test_gossip_broadcast() {
        let local_id = PeerId::random();
        let gossip = GossipProtocol::new(local_id.clone(), GossipConfig::default());

        let peers: Vec<PeerId> = (0..10).map(|_| PeerId::random()).collect();
        let msg = make_test_message(local_id);

        gossip.broadcast(msg, &peers).await;

        // Should have queued messages for fanout peers
        let outbound = gossip.take_outbound(100).await;
        assert_eq!(outbound.len(), gossip.config.fanout.min(peers.len()));
    }

    #[tokio::test]
    async fn test_gossip_dedup() {
        let local_id = PeerId::random();
        let gossip = GossipProtocol::new(local_id.clone(), GossipConfig::default());

        let sender = PeerId::random();
        let msg = make_test_message(sender.clone());
        let msg_id = msg.id.clone();
        let peers: Vec<PeerId> = (0..10).map(|_| PeerId::random()).collect();

        // First receive should process
        let (should_process1, _) = gossip.handle_message(msg.clone(), &sender, &peers).await;
        assert!(should_process1);

        // Second receive should be dedupe
        let (should_process2, _) = gossip.handle_message(msg, &sender, &peers).await;
        assert!(!should_process2);
    }

    #[tokio::test]
    async fn test_gossip_ttl() {
        let local_id = PeerId::random();
        let config = GossipConfig {
            max_hops: 3,
            ..Default::default()
        };
        let gossip = GossipProtocol::new(local_id.clone(), config);

        let sender = PeerId::random();
        let mut msg = make_test_message(sender.clone());
        msg.hops = 5; // Exceed TTL

        let peers: Vec<PeerId> = (0..10).map(|_| PeerId::random()).collect();
        let (should_process, _) = gossip.handle_message(msg, &sender, &peers).await;

        assert!(!should_process);
    }
}
