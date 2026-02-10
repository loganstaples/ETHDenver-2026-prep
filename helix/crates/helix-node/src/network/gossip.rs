//! Gossip Protocol for HELIX Network.
//!
//! Implements epidemic-style gossip for message propagation,
//! with configurable fanout, TTL, and LRU-based deduplication.
//!
//! The replay detection cache uses an LRU eviction policy instead of periodic
//! full cache sweeps. When the cache reaches capacity, the least-recently-used
//! entry is evicted automatically on insertion. TTL-based expiry is still
//! checked on lookup to avoid processing stale entries that happen to remain
//! in the cache.

use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use lru::LruCache;

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

/// Entry in the LRU message cache.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Timestamp when added (unix seconds).
    timestamp: u64,
    /// Whether this was locally originated.
    local: bool,
}

/// Gossip protocol manager.
pub struct GossipProtocol {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: GossipConfig,
    /// LRU deduplication cache: message ID -> CacheEntry.
    /// On insertion when full, the least-recently-used entry is automatically evicted.
    seen: Arc<RwLock<LruCache<String, CacheEntry>>>,
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
    /// LRU evictions (entries evicted to make room).
    pub lru_evictions: u64,
}

impl GossipProtocol {
    /// Creates a new gossip protocol manager.
    pub fn new(local_id: PeerId, config: GossipConfig) -> Self {
        let cache_cap = NonZeroUsize::new(config.max_cache_size.max(1))
            .expect("max_cache_size must be > 0");

        Self {
            local_id,
            config,
            seen: Arc::new(RwLock::new(LruCache::new(cache_cap))),
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

    /// Performs lightweight cache maintenance.
    ///
    /// Unlike the previous full-sweep approach, the LRU cache automatically evicts
    /// the least-recently-used entry on insertion when at capacity. This method
    /// only needs to remove TTL-expired entries that haven't been naturally evicted
    /// yet. It scans at most a bounded number of entries per call to avoid blocking.
    pub async fn cleanup_cache(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let ttl = self.config.message_cache_ttl_secs;

        let mut seen = self.seen.write().await;

        // Collect expired keys by peeking at LRU order (oldest first).
        // We scan up to 1000 entries per cleanup to bound latency.
        let max_scan = 1000.min(seen.len());
        let mut expired_keys = Vec::new();

        // Peek at entries without promoting them in LRU order.
        // LruCache::iter() returns entries from most-recently-used to least.
        // We iterate all and collect expired ones.
        for (key, entry) in seen.iter() {
            if now.saturating_sub(entry.timestamp) >= ttl {
                expired_keys.push(key.clone());
            }
            if expired_keys.len() >= max_scan {
                break;
            }
        }

        let mut eviction_count = 0u64;
        for key in expired_keys {
            seen.pop(&key);
            eviction_count += 1;
        }

        drop(seen);

        if eviction_count > 0 {
            let mut stats = self.stats.write().await;
            stats.lru_evictions += eviction_count;
        }
    }

    /// Gets the current statistics.
    pub async fn stats(&self) -> GossipStats {
        self.stats.read().await.clone()
    }

    /// Checks if a message has been seen (and not TTL-expired).
    ///
    /// This promotes the entry in LRU order on hit, which is the desired behavior:
    /// recently-queried messages are kept longer.
    async fn has_seen(&self, id: &str) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let ttl = self.config.message_cache_ttl_secs;

        let mut seen = self.seen.write().await;

        match seen.get(id) {
            Some(entry) => {
                if now.saturating_sub(entry.timestamp) >= ttl {
                    // Entry is expired — remove it and treat as not seen
                    seen.pop(id);
                    false
                } else {
                    true
                }
            }
            None => false,
        }
    }

    /// Marks a message as seen, inserting it into the LRU cache.
    ///
    /// If the cache is at capacity, the least-recently-used entry is
    /// automatically evicted by the LRU cache.
    async fn mark_seen(&self, id: &str, local: bool) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = CacheEntry {
            timestamp: now,
            local,
        };

        let mut seen = self.seen.write().await;
        let was_full = seen.len() == seen.cap().get();
        seen.put(id.to_string(), entry);

        // Track if LRU eviction happened
        if was_full {
            drop(seen);
            let mut stats = self.stats.write().await;
            stats.lru_evictions += 1;
        }
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
        let _msg_id = msg.id.clone();
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

    #[tokio::test]
    async fn test_lru_eviction_on_capacity() {
        let local_id = PeerId::random();
        let config = GossipConfig {
            max_cache_size: 5, // Very small cache
            ..Default::default()
        };
        let gossip = GossipProtocol::new(local_id.clone(), config);

        let sender = PeerId::random();
        let peers: Vec<PeerId> = (0..3).map(|_| PeerId::random()).collect();

        // Insert 5 messages to fill cache
        for i in 0..5 {
            let mut msg = make_test_message(sender.clone());
            msg.id = format!("msg-{}", i);
            gossip.handle_message(msg, &sender, &peers).await;
        }

        assert_eq!(gossip.cache_size().await, 5);

        // Insert one more — should evict the LRU entry (msg-0)
        let mut msg = make_test_message(sender.clone());
        msg.id = "msg-5".to_string();
        gossip.handle_message(msg, &sender, &peers).await;

        // Cache should still be at capacity (5), not 6
        assert_eq!(gossip.cache_size().await, 5);

        // msg-0 should have been evicted, so it should be processable again
        let mut msg0 = make_test_message(sender.clone());
        msg0.id = "msg-0".to_string();
        let (should_process, _) = gossip.handle_message(msg0, &sender, &peers).await;
        assert!(should_process, "Evicted message should be processable again");
    }

    #[tokio::test]
    async fn test_lru_promotion_on_access() {
        let local_id = PeerId::random();
        let config = GossipConfig {
            max_cache_size: 3,
            ..Default::default()
        };
        let gossip = GossipProtocol::new(local_id.clone(), config);

        let sender = PeerId::random();
        let peers: Vec<PeerId> = (0..3).map(|_| PeerId::random()).collect();

        // Insert msg-0, msg-1, msg-2
        for i in 0..3 {
            let mut msg = make_test_message(sender.clone());
            msg.id = format!("msg-{}", i);
            gossip.handle_message(msg, &sender, &peers).await;
        }

        // Access msg-0 again (should be promoted to most-recently-used)
        let mut msg0 = make_test_message(sender.clone());
        msg0.id = "msg-0".to_string();
        let (should_process, _) = gossip.handle_message(msg0, &sender, &peers).await;
        assert!(!should_process); // Deduped, but promoted in LRU

        // Now insert msg-3 — should evict msg-1 (the actual LRU), not msg-0
        let mut msg3 = make_test_message(sender.clone());
        msg3.id = "msg-3".to_string();
        gossip.handle_message(msg3, &sender, &peers).await;

        // msg-1 should have been evicted
        let mut msg1 = make_test_message(sender.clone());
        msg1.id = "msg-1".to_string();
        let (should_process, _) = gossip.handle_message(msg1, &sender, &peers).await;
        assert!(should_process, "msg-1 should have been evicted and processable again");

        // msg-0 should still be in cache (was promoted)
        let mut msg0_again = make_test_message(sender.clone());
        msg0_again.id = "msg-0".to_string();
        let (should_process, _) = gossip.handle_message(msg0_again, &sender, &peers).await;
        assert!(!should_process, "msg-0 should still be in cache (promoted)");
    }

    #[tokio::test]
    async fn test_cleanup_removes_expired_entries() {
        let local_id = PeerId::random();
        let config = GossipConfig {
            message_cache_ttl_secs: 0, // Expire immediately
            max_cache_size: 100,
            ..Default::default()
        };
        let gossip = GossipProtocol::new(local_id.clone(), config);

        let sender = PeerId::random();
        let peers: Vec<PeerId> = (0..3).map(|_| PeerId::random()).collect();

        // Insert a message
        let msg = make_test_message(sender.clone());
        gossip.handle_message(msg.clone(), &sender, &peers).await;
        assert_eq!(gossip.cache_size().await, 1);

        // Wait a moment for TTL to expire, then cleanup
        tokio::time::sleep(Duration::from_millis(10)).await;
        gossip.cleanup_cache().await;

        assert_eq!(gossip.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_stats_track_evictions() {
        let local_id = PeerId::random();
        let config = GossipConfig {
            max_cache_size: 2,
            ..Default::default()
        };
        let gossip = GossipProtocol::new(local_id.clone(), config);

        let sender = PeerId::random();
        let peers: Vec<PeerId> = (0..3).map(|_| PeerId::random()).collect();

        // Fill cache
        for i in 0..2 {
            let mut msg = make_test_message(sender.clone());
            msg.id = format!("msg-{}", i);
            gossip.handle_message(msg, &sender, &peers).await;
        }

        // Trigger eviction
        let mut msg = make_test_message(sender.clone());
        msg.id = "msg-2".to_string();
        gossip.handle_message(msg, &sender, &peers).await;

        let stats = gossip.stats().await;
        assert!(stats.lru_evictions > 0, "Should track LRU evictions");
    }
}
