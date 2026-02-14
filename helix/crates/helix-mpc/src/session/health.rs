//! Party health monitoring and graceful disconnection handling.
//!
//! [`PartyHealthMonitor`] tracks the liveness of all parties in an MPC session
//! via heartbeats and message activity. When a party disconnects:
//!
//! 1. Detection — the monitor detects missing heartbeats or transport errors.
//! 2. Logging — the disconnection is logged with context (party ID, last seen,
//!    session phase, step number).
//! 3. Cleanup — the party is marked inactive in the session.
//! 4. Continuation (if threshold met) — for t-of-n Shamir sharing, the session
//!    can continue with the remaining parties.
//!
//! # Threshold Continuation
//!
//! If the session uses Shamir sharing with threshold `t` and at least `t`
//! parties remain active, computation can continue using Lagrange interpolation
//! with the reduced party set. For additive sharing, all parties must remain
//! (no threshold tolerance).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::error::{MPCError, MPCResult};
use crate::types::{MPCConfig, MPCPhase, PartyId};

/// Health status of a party.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Party is healthy and responsive.
    Healthy,
    /// Party missed some heartbeats but hasn't timed out.
    Degraded,
    /// Party has timed out and is considered disconnected.
    Disconnected,
    /// Party was explicitly removed from the session.
    Removed,
}

/// Per-party health tracking.
#[derive(Debug, Clone)]
pub struct PartyHealth {
    pub party_id: PartyId,
    pub status: HealthStatus,
    /// Last time we received any message from this party.
    pub last_seen: Instant,
    /// Number of consecutive missed heartbeats.
    pub missed_heartbeats: u32,
    /// Total messages received from this party.
    pub messages_received: u64,
    /// Total messages sent to this party.
    pub messages_sent: u64,
    /// Number of communication errors with this party.
    pub error_count: u32,
}

/// Configuration for health monitoring.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// How often to send heartbeats.
    pub heartbeat_interval: Duration,
    /// How long to wait before considering a party degraded.
    pub degraded_threshold: Duration,
    /// How long to wait before considering a party disconnected.
    pub disconnect_threshold: Duration,
    /// Maximum consecutive missed heartbeats before disconnection.
    pub max_missed_heartbeats: u32,
    /// Maximum communication errors before marking unhealthy.
    pub max_errors: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(5),
            degraded_threshold: Duration::from_secs(15),
            disconnect_threshold: Duration::from_secs(30),
            max_missed_heartbeats: 5,
            max_errors: 10,
        }
    }
}

/// Event emitted when a party's health status changes.
#[derive(Debug, Clone)]
pub struct HealthEvent {
    pub party_id: PartyId,
    pub old_status: HealthStatus,
    pub new_status: HealthStatus,
    pub reason: String,
    pub session_phase: MPCPhase,
    pub step_number: u64,
}

/// Result of a health check sweep.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Events generated during this check.
    pub events: Vec<HealthEvent>,
    /// Number of healthy parties.
    pub healthy_count: usize,
    /// Number of degraded parties.
    pub degraded_count: usize,
    /// Number of disconnected parties.
    pub disconnected_count: usize,
    /// Whether the session can continue (enough parties for threshold).
    pub can_continue: bool,
}

/// Party health monitor for an MPC session.
pub struct PartyHealthMonitor {
    /// Per-party health state.
    parties: HashMap<PartyId, PartyHealth>,
    /// Monitoring configuration.
    config: HealthConfig,
    /// MPC configuration (for threshold checking).
    mpc_config: MPCConfig,
    /// Current session phase.
    current_phase: MPCPhase,
    /// Current step number.
    current_step: u64,
}

impl PartyHealthMonitor {
    /// Creates a new health monitor for the given parties.
    pub fn new(
        party_ids: &[PartyId],
        config: HealthConfig,
        mpc_config: MPCConfig,
    ) -> Self {
        let now = Instant::now();
        let parties = party_ids
            .iter()
            .map(|id| {
                (
                    id.clone(),
                    PartyHealth {
                        party_id: id.clone(),
                        status: HealthStatus::Healthy,
                        last_seen: now,
                        missed_heartbeats: 0,
                        messages_received: 0,
                        messages_sent: 0,
                        error_count: 0,
                    },
                )
            })
            .collect();

        Self {
            parties,
            config,
            mpc_config,
            current_phase: MPCPhase::Preprocessing,
            current_step: 0,
        }
    }

    /// Updates the session context (phase and step).
    pub fn update_context(&mut self, phase: MPCPhase, step: u64) {
        self.current_phase = phase;
        self.current_step = step;
    }

    /// Records that we received a message from a party.
    pub fn record_message_received(&mut self, party_id: &PartyId) {
        if let Some(health) = self.parties.get_mut(party_id) {
            health.last_seen = Instant::now();
            health.messages_received += 1;
            health.missed_heartbeats = 0;

            // Recover from degraded state.
            if health.status == HealthStatus::Degraded {
                health.status = HealthStatus::Healthy;
            }
        }
    }

    /// Records that we sent a message to a party.
    pub fn record_message_sent(&mut self, party_id: &PartyId) {
        if let Some(health) = self.parties.get_mut(party_id) {
            health.messages_sent += 1;
        }
    }

    /// Records a heartbeat from a party.
    #[instrument(skip_all, level = "trace", fields(party = %party_id))]
    pub fn record_heartbeat(&mut self, party_id: &PartyId) {
        if let Some(health) = self.parties.get_mut(party_id) {
            health.last_seen = Instant::now();
            health.missed_heartbeats = 0;

            if health.status == HealthStatus::Degraded {
                debug!(party = %party_id, "Party recovered from degraded status");
                health.status = HealthStatus::Healthy;
            }
        }
    }

    /// Records a communication error with a party.
    pub fn record_error(&mut self, party_id: &PartyId) {
        if let Some(health) = self.parties.get_mut(party_id) {
            health.error_count += 1;
        }
    }

    /// Runs a health check sweep and returns status changes.
    #[instrument(skip_all, level = "debug", fields(phase = ?self.current_phase, step = self.current_step))]
    pub fn check(&mut self) -> HealthCheckResult {
        let now = Instant::now();
        let mut events = Vec::new();
        let mut healthy = 0;
        let mut degraded = 0;
        let mut disconnected = 0;

        for health in self.parties.values_mut() {
            // Skip already-removed parties.
            if health.status == HealthStatus::Removed {
                disconnected += 1;
                continue;
            }

            let elapsed = now.duration_since(health.last_seen);
            let old_status = health.status;

            // Check for disconnection.
            if elapsed > self.config.disconnect_threshold
                || health.missed_heartbeats >= self.config.max_missed_heartbeats
                || health.error_count >= self.config.max_errors
            {
                if health.status != HealthStatus::Disconnected {
                    health.status = HealthStatus::Disconnected;

                    let reason = if elapsed > self.config.disconnect_threshold {
                        format!(
                            "no activity for {:.1}s (threshold: {:.1}s)",
                            elapsed.as_secs_f64(),
                            self.config.disconnect_threshold.as_secs_f64(),
                        )
                    } else if health.missed_heartbeats >= self.config.max_missed_heartbeats {
                        format!(
                            "{} consecutive missed heartbeats",
                            health.missed_heartbeats,
                        )
                    } else {
                        format!(
                            "{} communication errors (max: {})",
                            health.error_count, self.config.max_errors,
                        )
                    };

                    tracing::warn!(
                        "Party {} disconnected: {} (phase={}, step={})",
                        health.party_id,
                        reason,
                        self.current_phase,
                        self.current_step,
                    );

                    events.push(HealthEvent {
                        party_id: health.party_id.clone(),
                        old_status,
                        new_status: HealthStatus::Disconnected,
                        reason,
                        session_phase: self.current_phase,
                        step_number: self.current_step,
                    });
                }
                disconnected += 1;
            } else if elapsed > self.config.degraded_threshold {
                if health.status != HealthStatus::Degraded {
                    health.status = HealthStatus::Degraded;
                    health.missed_heartbeats += 1;

                    tracing::info!(
                        "Party {} degraded: no activity for {:.1}s",
                        health.party_id,
                        elapsed.as_secs_f64(),
                    );

                    events.push(HealthEvent {
                        party_id: health.party_id.clone(),
                        old_status,
                        new_status: HealthStatus::Degraded,
                        reason: format!("no activity for {:.1}s", elapsed.as_secs_f64()),
                        session_phase: self.current_phase,
                        step_number: self.current_step,
                    });
                }
                degraded += 1;
            } else {
                healthy += 1;
            }
        }

        let active_count = healthy + degraded;
        let can_continue = self.can_continue_with(active_count);

        HealthCheckResult {
            events,
            healthy_count: healthy,
            degraded_count: degraded,
            disconnected_count: disconnected,
            can_continue,
        }
    }

    /// Checks if the session can continue with the given number of active parties.
    pub fn can_continue_with(&self, active_count: usize) -> bool {
        if self.mpc_config.use_shamir {
            // Shamir: need at least `threshold` parties.
            active_count >= self.mpc_config.threshold
        } else {
            // Additive: need all parties.
            active_count >= self.mpc_config.num_parties
        }
    }

    /// Explicitly marks a party as removed (won't recover).
    pub fn remove_party(&mut self, party_id: &PartyId) -> MPCResult<()> {
        let health = self.parties.get_mut(party_id).ok_or_else(|| {
            MPCError::UnknownParty(party_id.clone())
        })?;

        tracing::warn!(
            "Party {} explicitly removed from session (phase={}, step={})",
            party_id,
            self.current_phase,
            self.current_step,
        );

        health.status = HealthStatus::Removed;
        Ok(())
    }

    /// Returns IDs of all active (healthy or degraded) parties.
    pub fn active_parties(&self) -> Vec<PartyId> {
        self.parties
            .values()
            .filter(|h| h.status == HealthStatus::Healthy || h.status == HealthStatus::Degraded)
            .map(|h| h.party_id.clone())
            .collect()
    }

    /// Returns IDs of disconnected parties.
    pub fn disconnected_parties(&self) -> Vec<PartyId> {
        self.parties
            .values()
            .filter(|h| {
                h.status == HealthStatus::Disconnected || h.status == HealthStatus::Removed
            })
            .map(|h| h.party_id.clone())
            .collect()
    }

    /// Returns health info for a specific party.
    pub fn get(&self, party_id: &PartyId) -> Option<&PartyHealth> {
        self.parties.get(party_id)
    }

    /// Returns all party health states.
    pub fn all(&self) -> &HashMap<PartyId, PartyHealth> {
        &self.parties
    }

    /// Returns the number of active parties.
    pub fn active_count(&self) -> usize {
        self.parties
            .values()
            .filter(|h| h.status == HealthStatus::Healthy || h.status == HealthStatus::Degraded)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_health_monitor_creation() {
        let parties = test_parties(3);
        let monitor = PartyHealthMonitor::new(
            &parties,
            HealthConfig::default(),
            MPCConfig::three_party(),
        );

        assert_eq!(monitor.active_count(), 3);
        assert!(monitor.disconnected_parties().is_empty());
    }

    #[test]
    fn test_all_healthy() {
        let parties = test_parties(3);
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            HealthConfig::default(),
            MPCConfig::three_party(),
        );

        let result = monitor.check();
        assert_eq!(result.healthy_count, 3);
        assert_eq!(result.degraded_count, 0);
        assert_eq!(result.disconnected_count, 0);
        assert!(result.can_continue);
        assert!(result.events.is_empty());
    }

    #[test]
    fn test_degraded_detection() {
        let parties = test_parties(3);
        let config = HealthConfig {
            degraded_threshold: Duration::from_millis(1),
            disconnect_threshold: Duration::from_secs(60),
            ..Default::default()
        };
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            config,
            MPCConfig::three_party(),
        );

        // Only update party 0 and 1.
        std::thread::sleep(Duration::from_millis(5));
        monitor.record_heartbeat(&parties[0]);
        monitor.record_heartbeat(&parties[1]);

        let result = monitor.check();
        assert_eq!(result.healthy_count, 2);
        assert_eq!(result.degraded_count, 1);
        assert!(result.can_continue);
    }

    #[test]
    fn test_disconnection_detection() {
        let parties = test_parties(3);
        let config = HealthConfig {
            degraded_threshold: Duration::from_millis(1),
            disconnect_threshold: Duration::from_millis(2),
            ..Default::default()
        };
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            config,
            MPCConfig::three_party(),
        );

        std::thread::sleep(Duration::from_millis(5));
        monitor.record_heartbeat(&parties[0]);

        let result = monitor.check();
        assert_eq!(result.disconnected_count, 2);
        assert!(result.events.len() >= 2);

        let disconnected = monitor.disconnected_parties();
        assert_eq!(disconnected.len(), 2);
    }

    #[test]
    fn test_threshold_continuation_shamir() {
        let parties = test_parties(5);
        let config = HealthConfig {
            disconnect_threshold: Duration::from_millis(1),
            degraded_threshold: Duration::from_millis(1),
            ..Default::default()
        };
        let mpc_config = MPCConfig::shamir(5, 3);
        let mut monitor = PartyHealthMonitor::new(&parties, config, mpc_config);

        std::thread::sleep(Duration::from_millis(5));

        // Keep 3 parties alive (meets threshold of 3).
        monitor.record_heartbeat(&parties[0]);
        monitor.record_heartbeat(&parties[1]);
        monitor.record_heartbeat(&parties[2]);

        let result = monitor.check();
        assert_eq!(result.disconnected_count, 2);
        assert!(result.can_continue); // 3 active >= threshold 3.
    }

    #[test]
    fn test_threshold_failure_shamir() {
        let parties = test_parties(5);
        let config = HealthConfig {
            disconnect_threshold: Duration::from_millis(1),
            degraded_threshold: Duration::from_millis(1),
            ..Default::default()
        };
        let mpc_config = MPCConfig::shamir(5, 3);
        let mut monitor = PartyHealthMonitor::new(&parties, config, mpc_config);

        std::thread::sleep(Duration::from_millis(5));

        // Only 2 parties alive (below threshold of 3).
        monitor.record_heartbeat(&parties[0]);
        monitor.record_heartbeat(&parties[1]);

        let result = monitor.check();
        assert!(!result.can_continue); // 2 active < threshold 3.
    }

    #[test]
    fn test_additive_no_tolerance() {
        let parties = test_parties(3);
        let config = HealthConfig {
            disconnect_threshold: Duration::from_millis(1),
            degraded_threshold: Duration::from_millis(1),
            ..Default::default()
        };
        let mpc_config = MPCConfig::three_party(); // Additive, no threshold tolerance.
        let mut monitor = PartyHealthMonitor::new(&parties, config, mpc_config);

        std::thread::sleep(Duration::from_millis(5));

        // Keep all but one alive.
        monitor.record_heartbeat(&parties[0]);
        monitor.record_heartbeat(&parties[1]);

        let result = monitor.check();
        assert!(!result.can_continue); // Additive needs all 3.
    }

    #[test]
    fn test_recovery_from_degraded() {
        let parties = test_parties(2);
        let config = HealthConfig {
            degraded_threshold: Duration::from_millis(1),
            disconnect_threshold: Duration::from_secs(60),
            ..Default::default()
        };
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            config,
            MPCConfig::two_party(),
        );

        std::thread::sleep(Duration::from_millis(5));

        // Party 1 is degraded.
        monitor.record_heartbeat(&parties[0]);
        let result = monitor.check();
        assert_eq!(result.degraded_count, 1);

        // Party 1 recovers.
        monitor.record_heartbeat(&parties[1]);
        let result = monitor.check();
        assert_eq!(result.degraded_count, 0);
        assert_eq!(result.healthy_count, 2);
    }

    #[test]
    fn test_explicit_removal() {
        let parties = test_parties(3);
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            HealthConfig::default(),
            MPCConfig::three_party(),
        );

        monitor.remove_party(&parties[2]).unwrap();

        assert_eq!(monitor.active_count(), 2);
        assert_eq!(monitor.disconnected_parties().len(), 1);
    }

    #[test]
    fn test_error_count_disconnection() {
        let parties = test_parties(2);
        let config = HealthConfig {
            max_errors: 3,
            ..Default::default()
        };
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            config,
            MPCConfig::two_party(),
        );

        // Accumulate errors.
        for _ in 0..3 {
            monitor.record_error(&parties[1]);
        }

        let result = monitor.check();
        assert_eq!(result.disconnected_count, 1);
    }

    #[test]
    fn test_message_tracking() {
        let parties = test_parties(2);
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            HealthConfig::default(),
            MPCConfig::two_party(),
        );

        monitor.record_message_received(&parties[1]);
        monitor.record_message_received(&parties[1]);
        monitor.record_message_sent(&parties[1]);

        let health = monitor.get(&parties[1]).unwrap();
        assert_eq!(health.messages_received, 2);
        assert_eq!(health.messages_sent, 1);
    }

    #[test]
    fn test_context_update() {
        let parties = test_parties(2);
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            HealthConfig::default(),
            MPCConfig::two_party(),
        );

        monitor.update_context(MPCPhase::Computation, 42);

        // The context is used in health events.
        assert_eq!(monitor.current_phase, MPCPhase::Computation);
        assert_eq!(monitor.current_step, 42);
    }
}
