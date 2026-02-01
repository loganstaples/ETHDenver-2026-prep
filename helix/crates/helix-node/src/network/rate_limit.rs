//! Rate Limiting for HELIX Network.
//!
//! Implements per-peer rate limiting to prevent abuse and ensure fair
//! resource allocation. Key mechanisms:
//! - Token bucket rate limiting per peer
//! - Global rate limiting
//! - Message-type specific limits
//! - Adaptive rate limiting based on network conditions
//! - Connection and blacklist management for DDoS mitigation

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::messages::PeerId;
use serde::{Deserialize, Serialize};

/// Configuration for rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Default requests per second per peer.
    pub default_requests_per_second: f64,
    /// Default burst size (token bucket capacity).
    pub default_burst_size: u32,
    /// Global requests per second limit.
    pub global_requests_per_second: f64,
    /// Global burst size.
    pub global_burst_size: u32,
    /// Window size for rate calculation.
    pub rate_window: Duration,
    /// Cooldown period after rate limit hit.
    pub cooldown_period: Duration,
    /// Maximum cooldown period (exponential backoff cap).
    pub max_cooldown: Duration,
    /// Enable adaptive rate limiting.
    pub adaptive_enabled: bool,
    /// Target network utilization for adaptive limiting.
    pub target_utilization: f64,
    /// Minimum allowed rate (never go below this).
    pub min_rate: f64,
    /// Maximum connection attempts per minute per IP.
    pub max_connections_per_minute: u32,
    /// Blacklist duration.
    pub blacklist_duration: Duration,
    /// Auto-blacklist after this many violations.
    pub auto_blacklist_threshold: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            default_requests_per_second: 100.0,
            default_burst_size: 200,
            global_requests_per_second: 10000.0,
            global_burst_size: 20000,
            rate_window: Duration::from_secs(1),
            cooldown_period: Duration::from_secs(1),
            max_cooldown: Duration::from_secs(60),
            adaptive_enabled: true,
            target_utilization: 0.8,
            min_rate: 1.0,
            max_connections_per_minute: 10,
            blacklist_duration: Duration::from_secs(3600), // 1 hour
            auto_blacklist_threshold: 10,
        }
    }
}

/// Rate limit state for a single peer.
#[derive(Debug, Clone)]
pub struct PeerRateLimit {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Current token count.
    tokens: f64,
    /// Maximum tokens (bucket capacity).
    max_tokens: f64,
    /// Token refill rate per second.
    refill_rate: f64,
    /// Last update time.
    last_update: Instant,
    /// Current cooldown expiry.
    cooldown_until: Option<Instant>,
    /// Number of violations.
    violations: u32,
    /// Total requests.
    total_requests: u64,
    /// Requests denied.
    requests_denied: u64,
    /// Per-message-type limits.
    message_limits: HashMap<MessageType, MessageRateLimit>,
}

impl PeerRateLimit {
    /// Creates a new peer rate limit.
    pub fn new(peer_id: PeerId, max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            peer_id,
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_update: Instant::now(),
            cooldown_until: None,
            violations: 0,
            total_requests: 0,
            requests_denied: 0,
            message_limits: HashMap::new(),
        }
    }

    /// Attempts to consume tokens for a request.
    ///
    /// Returns true if allowed, false if rate limited.
    pub fn try_acquire(&mut self, cost: f64) -> bool {
        // Check cooldown
        if let Some(until) = self.cooldown_until {
            if Instant::now() < until {
                self.requests_denied += 1;
                return false;
            }
            self.cooldown_until = None;
        }

        // Refill tokens
        self.refill();

        self.total_requests += 1;

        // Check if we have enough tokens
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            self.requests_denied += 1;
            self.violations += 1;
            false
        }
    }

    /// Tries to acquire for a specific message type.
    pub fn try_acquire_typed(&mut self, msg_type: MessageType, cost: f64) -> bool {
        // Check message-specific limit first
        if let Some(limit) = self.message_limits.get_mut(&msg_type) {
            if !limit.try_acquire(cost) {
                self.requests_denied += 1;
                return false;
            }
        }

        // Then check overall limit
        self.try_acquire(cost)
    }

    /// Refills tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_update = now;
    }

    /// Sets cooldown.
    pub fn set_cooldown(&mut self, duration: Duration) {
        self.cooldown_until = Some(Instant::now() + duration);
    }

    /// Returns current token count.
    pub fn tokens(&self) -> f64 {
        self.tokens
    }

    /// Returns violation count.
    pub fn violations(&self) -> u32 {
        self.violations
    }

    /// Returns denial rate.
    pub fn denial_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.requests_denied as f64 / self.total_requests as f64
        }
    }

    /// Adjusts rate limit.
    pub fn adjust_rate(&mut self, factor: f64) {
        self.refill_rate = (self.refill_rate * factor).max(1.0);
        self.max_tokens = (self.max_tokens * factor).max(1.0);
    }

    /// Sets message-specific limit.
    pub fn set_message_limit(&mut self, msg_type: MessageType, limit: MessageRateLimit) {
        self.message_limits.insert(msg_type, limit);
    }

    /// Resets violations and stats.
    pub fn reset_stats(&mut self) {
        self.violations = 0;
        self.total_requests = 0;
        self.requests_denied = 0;
    }
}

/// Rate limit for specific message type.
#[derive(Debug, Clone)]
pub struct MessageRateLimit {
    /// Message type.
    msg_type: MessageType,
    /// Current tokens.
    tokens: f64,
    /// Max tokens.
    max_tokens: f64,
    /// Refill rate.
    refill_rate: f64,
    /// Last update.
    last_update: Instant,
}

impl MessageRateLimit {
    /// Creates a new message rate limit.
    pub fn new(msg_type: MessageType, max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            msg_type,
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_update: Instant::now(),
        }
    }

    /// Tries to acquire tokens.
    pub fn try_acquire(&mut self, cost: f64) -> bool {
        // Refill
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_update = now;

        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }
}

/// Message types for rate limiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    /// Discovery messages.
    Discovery,
    /// Training coordination.
    Training,
    /// Gradient exchange.
    Gradient,
    /// State sync.
    Sync,
    /// Heartbeat.
    Heartbeat,
    /// Generic/unknown.
    Generic,
}

/// Blacklist entry.
#[derive(Debug, Clone)]
pub struct BlacklistEntry {
    /// Blacklisted identifier (peer ID or IP).
    pub identifier: String,
    /// Reason for blacklisting.
    pub reason: BlacklistReason,
    /// When blacklisted.
    pub blacklisted_at: Instant,
    /// Expiry time.
    pub expires_at: Instant,
    /// Number of times blacklisted.
    pub blacklist_count: u32,
}

/// Reason for blacklisting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlacklistReason {
    /// Too many rate limit violations.
    RateLimitViolations { count: u32 },
    /// Too many connection attempts.
    ConnectionFlood { attempts: u32 },
    /// Malicious behavior detected.
    MaliciousBehavior { details: String },
    /// Manual blacklist.
    Manual { reason: String },
}

/// Global rate limiter state.
#[derive(Debug, Clone)]
struct GlobalRateLimit {
    /// Current tokens.
    tokens: f64,
    /// Max tokens.
    max_tokens: f64,
    /// Refill rate.
    refill_rate: f64,
    /// Last update.
    last_update: Instant,
}

impl GlobalRateLimit {
    fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_update: Instant::now(),
        }
    }

    fn try_acquire(&mut self, cost: f64) -> bool {
        // Refill
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_update = now;

        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }

    fn utilization(&self) -> f64 {
        1.0 - (self.tokens / self.max_tokens)
    }
}

/// Connection tracking for flood prevention.
#[derive(Debug, Clone)]
struct ConnectionTracker {
    /// Connections per IP.
    connections: HashMap<String, Vec<Instant>>,
    /// Window for tracking.
    window: Duration,
    /// Max connections per window.
    max_per_window: u32,
}

impl ConnectionTracker {
    fn new(window: Duration, max: u32) -> Self {
        Self {
            connections: HashMap::new(),
            window,
            max_per_window: max,
        }
    }

    fn record_connection(&mut self, ip: &str) -> bool {
        let now = Instant::now();
        let cutoff = now - self.window;

        let entry = self.connections.entry(ip.to_string()).or_insert_with(Vec::new);

        // Prune old entries
        entry.retain(|&t| t > cutoff);

        // Check limit
        if entry.len() >= self.max_per_window as usize {
            false
        } else {
            entry.push(now);
            true
        }
    }

    fn connection_count(&self, ip: &str) -> usize {
        self.connections.get(ip).map(|v| v.len()).unwrap_or(0)
    }
}

/// Rate limiter for network traffic.
///
/// Implements per-peer and global rate limiting with DDoS protection.
pub struct RateLimiter {
    /// Configuration.
    config: RateLimitConfig,
    /// Per-peer rate limits.
    peer_limits: HashMap<PeerId, PeerRateLimit>,
    /// Global rate limit.
    global_limit: GlobalRateLimit,
    /// Blacklist.
    blacklist: HashMap<String, BlacklistEntry>,
    /// Connection tracker.
    connection_tracker: ConnectionTracker,
    /// Message type costs.
    message_costs: HashMap<MessageType, f64>,
    /// Statistics.
    stats: RateLimitStats,
    /// Last adaptive adjustment.
    last_adaptive_check: Instant,
}

/// Rate limiting statistics.
#[derive(Debug, Clone, Default)]
pub struct RateLimitStats {
    /// Total requests.
    pub total_requests: u64,
    /// Requests allowed.
    pub requests_allowed: u64,
    /// Requests denied.
    pub requests_denied: u64,
    /// Peers rate limited.
    pub peers_limited: u32,
    /// Active blacklist entries.
    pub blacklist_size: usize,
    /// Current global utilization.
    pub global_utilization: f64,
}

impl RateLimiter {
    /// Creates a new rate limiter.
    pub fn new(config: RateLimitConfig) -> Self {
        let global_limit = GlobalRateLimit::new(
            config.global_burst_size as f64,
            config.global_requests_per_second,
        );

        let connection_tracker = ConnectionTracker::new(
            Duration::from_secs(60),
            config.max_connections_per_minute,
        );

        let mut message_costs = HashMap::new();
        message_costs.insert(MessageType::Discovery, 1.0);
        message_costs.insert(MessageType::Training, 2.0);
        message_costs.insert(MessageType::Gradient, 5.0);
        message_costs.insert(MessageType::Sync, 3.0);
        message_costs.insert(MessageType::Heartbeat, 0.5);
        message_costs.insert(MessageType::Generic, 1.0);

        Self {
            config,
            peer_limits: HashMap::new(),
            global_limit,
            blacklist: HashMap::new(),
            connection_tracker,
            message_costs,
            stats: RateLimitStats::default(),
            last_adaptive_check: Instant::now(),
        }
    }

    /// Checks if a request from a peer should be allowed.
    pub fn check_rate_limit(
        &mut self,
        peer_id: &PeerId,
        msg_type: MessageType,
    ) -> RateLimitResult {
        self.stats.total_requests += 1;

        // Check blacklist
        if self.is_blacklisted(&peer_id.0) {
            self.stats.requests_denied += 1;
            return RateLimitResult::Blacklisted;
        }

        // Get message cost
        let cost = self.message_costs.get(&msg_type).copied().unwrap_or(1.0);

        // Check global limit
        if !self.global_limit.try_acquire(cost) {
            self.stats.requests_denied += 1;
            return RateLimitResult::GlobalLimitExceeded;
        }

        // Check peer limit - use scoped borrow to avoid conflicts
        let (limit_exceeded, violations, should_blacklist) = {
            let peer_limit = self.peer_limits.entry(peer_id.clone()).or_insert_with(|| {
                PeerRateLimit::new(
                    peer_id.clone(),
                    self.config.default_burst_size as f64,
                    self.config.default_requests_per_second,
                )
            });

            if !peer_limit.try_acquire_typed(msg_type, cost) {
                let violations = peer_limit.violations();
                let should_blacklist = violations >= self.config.auto_blacklist_threshold;

                if !should_blacklist {
                    // Apply cooldown with exponential backoff
                    let cooldown = Duration::from_secs_f64(
                        self.config.cooldown_period.as_secs_f64() * 2.0_f64.powi(violations as i32 - 1),
                    ).min(self.config.max_cooldown);
                    peer_limit.set_cooldown(cooldown);
                }

                (true, violations, should_blacklist)
            } else {
                (false, 0, false)
            }
        };

        if limit_exceeded {
            self.stats.requests_denied += 1;

            if should_blacklist {
                self.blacklist_peer(
                    peer_id,
                    BlacklistReason::RateLimitViolations {
                        count: violations,
                    },
                );
                return RateLimitResult::AutoBlacklisted;
            }

            let cooldown = Duration::from_secs_f64(
                self.config.cooldown_period.as_secs_f64() * 2.0_f64.powi(violations as i32 - 1),
            ).min(self.config.max_cooldown);

            return RateLimitResult::PeerLimitExceeded {
                cooldown,
                violations,
            };
        }

        self.stats.requests_allowed += 1;
        self.stats.global_utilization = self.global_limit.utilization();

        // Run adaptive adjustment periodically
        if self.config.adaptive_enabled && self.last_adaptive_check.elapsed() > Duration::from_secs(10) {
            self.run_adaptive_adjustment();
        }

        RateLimitResult::Allowed
    }

    /// Checks a connection attempt.
    pub fn check_connection(&mut self, ip: &str) -> RateLimitResult {
        // Check blacklist
        if self.is_blacklisted(ip) {
            return RateLimitResult::Blacklisted;
        }

        // Check connection rate
        if !self.connection_tracker.record_connection(ip) {
            // Too many connections - blacklist
            self.blacklist.insert(
                ip.to_string(),
                BlacklistEntry {
                    identifier: ip.to_string(),
                    reason: BlacklistReason::ConnectionFlood {
                        attempts: self.connection_tracker.connection_count(ip) as u32,
                    },
                    blacklisted_at: Instant::now(),
                    expires_at: Instant::now() + self.config.blacklist_duration,
                    blacklist_count: 1,
                },
            );
            return RateLimitResult::ConnectionFlood;
        }

        RateLimitResult::Allowed
    }

    /// Checks if an identifier is blacklisted.
    pub fn is_blacklisted(&mut self, identifier: &str) -> bool {
        if let Some(entry) = self.blacklist.get(identifier) {
            if Instant::now() < entry.expires_at {
                return true;
            }
            // Expired, remove
            self.blacklist.remove(identifier);
        }
        false
    }

    /// Blacklists a peer.
    pub fn blacklist_peer(&mut self, peer_id: &PeerId, reason: BlacklistReason) {
        let existing_count = self.blacklist
            .get(&peer_id.0)
            .map(|e| e.blacklist_count)
            .unwrap_or(0);

        // Exponential backoff for duration
        let duration = self.config.blacklist_duration * 2u32.pow(existing_count);

        self.blacklist.insert(
            peer_id.0.clone(),
            BlacklistEntry {
                identifier: peer_id.0.clone(),
                reason,
                blacklisted_at: Instant::now(),
                expires_at: Instant::now() + duration,
                blacklist_count: existing_count + 1,
            },
        );

        self.stats.blacklist_size = self.blacklist.len();
    }

    /// Removes from blacklist.
    pub fn unblacklist(&mut self, identifier: &str) {
        self.blacklist.remove(identifier);
        self.stats.blacklist_size = self.blacklist.len();
    }

    /// Runs adaptive rate adjustment.
    fn run_adaptive_adjustment(&mut self) {
        self.last_adaptive_check = Instant::now();

        let utilization = self.global_limit.utilization();

        // Adjust rates based on utilization
        let factor = if utilization > self.config.target_utilization {
            // Too much traffic - reduce rates
            0.9
        } else if utilization < self.config.target_utilization * 0.5 {
            // Low traffic - can increase rates
            1.1
        } else {
            return; // No adjustment needed
        };

        for limit in self.peer_limits.values_mut() {
            limit.adjust_rate(factor);
        }
    }

    /// Gets rate limit info for a peer.
    pub fn get_peer_limit(&self, peer_id: &PeerId) -> Option<&PeerRateLimit> {
        self.peer_limits.get(peer_id)
    }

    /// Sets custom rate for a peer.
    pub fn set_peer_rate(&mut self, peer_id: &PeerId, rate: f64, burst: f64) {
        if let Some(limit) = self.peer_limits.get_mut(peer_id) {
            limit.refill_rate = rate;
            limit.max_tokens = burst;
        }
    }

    /// Sets cost for a message type.
    pub fn set_message_cost(&mut self, msg_type: MessageType, cost: f64) {
        self.message_costs.insert(msg_type, cost);
    }

    /// Returns statistics.
    pub fn stats(&self) -> &RateLimitStats {
        &self.stats
    }

    /// Resets all statistics.
    pub fn reset_stats(&mut self) {
        self.stats = RateLimitStats::default();
        for limit in self.peer_limits.values_mut() {
            limit.reset_stats();
        }
    }

    /// Cleans up expired entries.
    pub fn cleanup(&mut self) {
        let now = Instant::now();

        // Remove expired blacklist entries
        self.blacklist.retain(|_, entry| entry.expires_at > now);
        self.stats.blacklist_size = self.blacklist.len();

        // Count limited peers
        self.stats.peers_limited = self.peer_limits
            .values()
            .filter(|l| l.violations() > 0)
            .count() as u32;
    }

    /// Returns number of active peer limits.
    pub fn peer_count(&self) -> usize {
        self.peer_limits.len()
    }

    /// Returns current blacklist.
    pub fn blacklist(&self) -> &HashMap<String, BlacklistEntry> {
        &self.blacklist
    }
}

/// Result of a rate limit check.
#[derive(Debug, Clone)]
pub enum RateLimitResult {
    /// Request allowed.
    Allowed,
    /// Peer limit exceeded.
    PeerLimitExceeded {
        cooldown: Duration,
        violations: u32,
    },
    /// Global limit exceeded.
    GlobalLimitExceeded,
    /// Identifier is blacklisted.
    Blacklisted,
    /// Automatically blacklisted due to violations.
    AutoBlacklisted,
    /// Connection flood detected.
    ConnectionFlood,
}

impl RateLimitResult {
    /// Returns whether the request was allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitResult::Allowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rate_limiting() {
        let config = RateLimitConfig {
            default_requests_per_second: 10.0,
            default_burst_size: 10,
            ..Default::default()
        };
        let mut limiter = RateLimiter::new(config);

        let peer = PeerId::from_string("peer1");

        // First 10 requests should succeed (burst size)
        for _ in 0..10 {
            let result = limiter.check_rate_limit(&peer, MessageType::Generic);
            assert!(result.is_allowed());
        }

        // Next request should fail (exceeded burst)
        let result = limiter.check_rate_limit(&peer, MessageType::Generic);
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_blacklisting() {
        let config = RateLimitConfig {
            auto_blacklist_threshold: 3,
            default_burst_size: 1,
            default_requests_per_second: 0.1,
            // Zero cooldown so violations can accumulate immediately
            cooldown_period: Duration::from_millis(0),
            max_cooldown: Duration::from_millis(0),
            ..Default::default()
        };
        let mut limiter = RateLimiter::new(config);

        let peer = PeerId::from_string("peer1");

        // Exhaust limit (first request uses the 1 token)
        limiter.check_rate_limit(&peer, MessageType::Generic);

        // Trigger violations - need 3 to hit auto_blacklist_threshold
        for _ in 0..5 {
            let result = limiter.check_rate_limit(&peer, MessageType::Generic);
            if matches!(result, RateLimitResult::AutoBlacklisted) {
                break;
            }
        }

        // Should be blacklisted now
        assert!(limiter.is_blacklisted(&peer.0));
    }

    #[test]
    fn test_message_type_costs() {
        let config = RateLimitConfig {
            default_burst_size: 10,
            ..Default::default()
        };
        let mut limiter = RateLimiter::new(config);
        limiter.set_message_cost(MessageType::Gradient, 5.0);

        let peer = PeerId::from_string("peer1");

        // Gradient costs 5 tokens, so only 2 should fit in 10-token burst
        let result1 = limiter.check_rate_limit(&peer, MessageType::Gradient);
        assert!(result1.is_allowed());

        let result2 = limiter.check_rate_limit(&peer, MessageType::Gradient);
        assert!(result2.is_allowed());

        // Third should fail
        let result3 = limiter.check_rate_limit(&peer, MessageType::Gradient);
        assert!(!result3.is_allowed());
    }

    #[test]
    fn test_connection_flooding() {
        let config = RateLimitConfig {
            max_connections_per_minute: 5,
            ..Default::default()
        };
        let mut limiter = RateLimiter::new(config);

        let ip = "192.168.1.1";

        // First 5 should succeed
        for _ in 0..5 {
            let result = limiter.check_connection(ip);
            assert!(result.is_allowed());
        }

        // 6th should fail
        let result = limiter.check_connection(ip);
        assert!(matches!(result, RateLimitResult::ConnectionFlood));
    }

    #[test]
    fn test_unblacklist() {
        let config = RateLimitConfig::default();
        let mut limiter = RateLimiter::new(config);

        let peer = PeerId::from_string("peer1");
        limiter.blacklist_peer(&peer, BlacklistReason::Manual { reason: "test".to_string() });

        assert!(limiter.is_blacklisted(&peer.0));

        limiter.unblacklist(&peer.0);

        assert!(!limiter.is_blacklisted(&peer.0));
    }
}
