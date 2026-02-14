//! MPC session recovery for party dropout and session failure.
//!
//! When a party drops from an MPC session, this module handles:
//!
//! - **Shamir threshold continuation**: If >= threshold parties remain active,
//!   continue computation with the reduced set using Lagrange interpolation.
//! - **Clean abort**: If below threshold, save a checkpoint and abort the session
//!   so it can be retried with a fresh party set.
//! - **Session restart**: Resume from checkpoint with replacement parties.
//!
//! # Recovery Flow
//!
//! 1. Health monitor detects party disconnection (`HealthStatus::Disconnected`)
//! 2. `SessionRecoveryCoordinator::handle_party_dropout` is called
//! 3. Coordinator checks threshold viability
//! 4. If viable: marks party inactive, recomputes Lagrange coefficients, continues
//! 5. If not viable: checkpoints session state, aborts, returns `RecoveryPlan::Restart`

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::error::{MPCError, MPCResult};
use crate::types::{MPCConfig, MPCPhase, PartyId};
use super::checkpoint::SessionCheckpoint;
use super::health::{HealthCheckResult, HealthStatus, PartyHealthMonitor};

/// The outcome of a recovery attempt after a party dropout.
#[derive(Debug, Clone)]
pub enum RecoveryOutcome {
    /// Session continues with reduced party set (Shamir threshold met).
    ContinueWithReduced {
        /// Parties still active after dropout.
        active_parties: Vec<PartyId>,
        /// Parties that were removed.
        removed_parties: Vec<PartyId>,
        /// Updated Lagrange coefficients for the active party set.
        lagrange_indices: Vec<usize>,
    },
    /// Session must be restarted (below threshold).
    Restart {
        /// Checkpoint saved before abort.
        checkpoint: SessionCheckpoint,
        /// Minimum parties needed to restart.
        min_parties_needed: usize,
        /// Reason for restart.
        reason: String,
    },
    /// Session aborted permanently (unrecoverable).
    Abort {
        reason: String,
    },
}

/// Plan for restarting a failed session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartPlan {
    /// Session ID of the failed session.
    pub original_session_id: String,
    /// Checkpoint to resume from.
    pub checkpoint: SessionCheckpoint,
    /// Parties that were active at failure time.
    pub surviving_parties: Vec<PartyId>,
    /// Parties that dropped out.
    pub dropped_parties: Vec<PartyId>,
    /// Minimum replacement parties needed.
    pub replacements_needed: usize,
    /// The step to resume from.
    pub resume_step: u64,
    /// The phase to resume from.
    pub resume_phase: MPCPhase,
}

/// Configuration for session recovery behavior.
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Maximum number of restart attempts for a single session.
    pub max_restart_attempts: u32,
    /// Whether to allow Shamir threshold continuation.
    pub allow_threshold_continuation: bool,
    /// Whether to save checkpoints on abort.
    pub checkpoint_on_abort: bool,
    /// Maximum party drops before forcing a full restart (even if threshold met).
    pub max_drops_before_restart: usize,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_restart_attempts: 3,
            allow_threshold_continuation: true,
            checkpoint_on_abort: true,
            max_drops_before_restart: 2,
        }
    }
}

/// Tracks per-session recovery state.
#[derive(Debug)]
struct SessionRecoveryState {
    #[allow(dead_code)]
    session_id: String,
    total_drops: usize,
    restart_attempts: u32,
    dropped_parties: Vec<PartyId>,
    last_recovery_at: Instant,
}

/// Coordinates recovery actions when MPC parties drop from sessions.
pub struct SessionRecoveryCoordinator {
    config: RecoveryConfig,
    /// Per-session recovery tracking.
    sessions: HashMap<String, SessionRecoveryState>,
}

impl SessionRecoveryCoordinator {
    /// Creates a new recovery coordinator.
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            sessions: HashMap::new(),
        }
    }

    /// Creates a coordinator with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RecoveryConfig::default())
    }

    /// Handles a party dropout event from the health monitor.
    ///
    /// This is the main entry point for recovery. Call this when the
    /// `PartyHealthMonitor::check()` reports disconnected parties.
    pub fn handle_party_dropout(
        &mut self,
        session_id: &str,
        mpc_config: &MPCConfig,
        health_result: &HealthCheckResult,
        session_phase: MPCPhase,
        current_step: u64,
        active_parties: &[PartyId],
        dropped_parties: &[PartyId],
        checkpoint: Option<SessionCheckpoint>,
    ) -> RecoveryOutcome {
        // Track this dropout
        let state = self.sessions.entry(session_id.to_string()).or_insert_with(|| {
            SessionRecoveryState {
                session_id: session_id.to_string(),
                total_drops: 0,
                restart_attempts: 0,
                dropped_parties: Vec::new(),
                last_recovery_at: Instant::now(),
            }
        });

        state.total_drops += dropped_parties.len();
        state.dropped_parties.extend(dropped_parties.iter().cloned());
        state.last_recovery_at = Instant::now();

        info!(
            session_id,
            dropped = dropped_parties.len(),
            total_drops = state.total_drops,
            active = active_parties.len(),
            phase = %session_phase,
            step = current_step,
            "Handling party dropout"
        );

        // Check if too many drops have occurred (even if threshold still met)
        if state.total_drops > self.config.max_drops_before_restart {
            warn!(
                session_id,
                total_drops = state.total_drops,
                max = self.config.max_drops_before_restart,
                "Too many party drops, forcing restart"
            );
            return self.plan_restart(
                session_id,
                mpc_config,
                active_parties,
                dropped_parties,
                current_step,
                session_phase,
                checkpoint,
                "exceeded maximum allowed party drops".to_string(),
            );
        }

        // Check threshold continuation
        if health_result.can_continue && self.config.allow_threshold_continuation && mpc_config.use_shamir {
            self.continue_with_reduced(
                session_id,
                mpc_config,
                active_parties,
                dropped_parties,
            )
        } else if health_result.can_continue && !mpc_config.use_shamir {
            // Additive sharing: can_continue means all parties are active
            // (the health monitor already accounts for this), so this path
            // means no parties actually dropped — just degraded.
            info!(
                session_id,
                "Additive session: all parties still active (degraded but present)"
            );
            RecoveryOutcome::ContinueWithReduced {
                active_parties: active_parties.to_vec(),
                removed_parties: Vec::new(),
                lagrange_indices: (0..active_parties.len()).collect(),
            }
        } else {
            // Below threshold — must restart or abort
            self.plan_restart(
                session_id,
                mpc_config,
                active_parties,
                dropped_parties,
                current_step,
                session_phase,
                checkpoint,
                format!(
                    "below threshold: {} active < {} required",
                    active_parties.len(),
                    mpc_config.threshold,
                ),
            )
        }
    }

    /// Continues with a reduced party set for Shamir sessions.
    fn continue_with_reduced(
        &self,
        session_id: &str,
        mpc_config: &MPCConfig,
        active_parties: &[PartyId],
        removed_parties: &[PartyId],
    ) -> RecoveryOutcome {
        // Extract the original indices of the active parties for Lagrange interpolation.
        // Party IDs are "party-{index}", so we parse the index from each.
        let lagrange_indices: Vec<usize> = active_parties
            .iter()
            .filter_map(|p| {
                p.0.strip_prefix("party-")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .collect();

        info!(
            session_id,
            active = active_parties.len(),
            removed = removed_parties.len(),
            threshold = mpc_config.threshold,
            lagrange_indices = ?lagrange_indices,
            "Continuing session with reduced party set"
        );

        RecoveryOutcome::ContinueWithReduced {
            active_parties: active_parties.to_vec(),
            removed_parties: removed_parties.to_vec(),
            lagrange_indices,
        }
    }

    /// Plans a session restart with checkpoint.
    fn plan_restart(
        &mut self,
        session_id: &str,
        mpc_config: &MPCConfig,
        active_parties: &[PartyId],
        dropped_parties: &[PartyId],
        current_step: u64,
        session_phase: MPCPhase,
        checkpoint: Option<SessionCheckpoint>,
        reason: String,
    ) -> RecoveryOutcome {
        let state = self.sessions.get_mut(session_id);
        let restart_attempts = state.map(|s| {
            s.restart_attempts += 1;
            s.restart_attempts
        }).unwrap_or(1);

        if restart_attempts > self.config.max_restart_attempts {
            error!(
                session_id,
                restart_attempts,
                max = self.config.max_restart_attempts,
                "Maximum restart attempts exceeded, aborting session permanently"
            );
            return RecoveryOutcome::Abort {
                reason: format!(
                    "session {} exceeded {} restart attempts: {}",
                    session_id, self.config.max_restart_attempts, reason,
                ),
            };
        }

        match checkpoint {
            Some(ckpt) if self.config.checkpoint_on_abort => {
                warn!(
                    session_id,
                    step = current_step,
                    restart_attempt = restart_attempts,
                    reason = %reason,
                    "Session requires restart, checkpoint saved"
                );
                RecoveryOutcome::Restart {
                    checkpoint: ckpt,
                    min_parties_needed: mpc_config.num_parties.saturating_sub(active_parties.len()),
                    reason,
                }
            }
            _ => {
                warn!(
                    session_id,
                    step = current_step,
                    restart_attempt = restart_attempts,
                    reason = %reason,
                    "Session requires restart, no checkpoint available"
                );
                RecoveryOutcome::Abort {
                    reason: format!(
                        "session {} needs restart but no checkpoint available: {}",
                        session_id, reason,
                    ),
                }
            }
        }
    }

    /// Builds a restart plan from a recovery outcome.
    ///
    /// Call this after `handle_party_dropout` returns `RecoveryOutcome::Restart`
    /// to get a plan that can be used to form a new session.
    pub fn build_restart_plan(
        &self,
        session_id: &str,
        checkpoint: SessionCheckpoint,
        active_parties: &[PartyId],
        dropped_parties: &[PartyId],
        mpc_config: &MPCConfig,
    ) -> RestartPlan {
        let replacements_needed = mpc_config.num_parties
            .saturating_sub(active_parties.len());

        RestartPlan {
            original_session_id: session_id.to_string(),
            checkpoint: checkpoint.clone(),
            surviving_parties: active_parties.to_vec(),
            dropped_parties: dropped_parties.to_vec(),
            replacements_needed,
            resume_step: checkpoint.current_step,
            resume_phase: checkpoint.phase,
        }
    }

    /// Resets recovery state for a session (e.g., after successful restart).
    pub fn reset_session(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Returns recovery statistics for a session.
    pub fn session_stats(&self, session_id: &str) -> Option<RecoveryStats> {
        self.sessions.get(session_id).map(|state| RecoveryStats {
            total_drops: state.total_drops,
            restart_attempts: state.restart_attempts,
            dropped_parties: state.dropped_parties.clone(),
        })
    }

    /// Evaluates session health and returns a recovery action if needed.
    ///
    /// Convenience method that runs a health check and handles any dropouts.
    pub fn evaluate_health(
        &mut self,
        session_id: &str,
        mpc_config: &MPCConfig,
        monitor: &mut PartyHealthMonitor,
        session_phase: MPCPhase,
        current_step: u64,
        checkpoint: Option<SessionCheckpoint>,
    ) -> Option<RecoveryOutcome> {
        let health = monitor.check();

        if health.disconnected_count == 0 {
            return None;
        }

        let active = monitor.active_parties();
        let dropped = monitor.disconnected_parties();

        Some(self.handle_party_dropout(
            session_id,
            mpc_config,
            &health,
            session_phase,
            current_step,
            &active,
            &dropped,
            checkpoint,
        ))
    }
}

/// Recovery statistics for a session.
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub total_drops: usize,
    pub restart_attempts: u32,
    pub dropped_parties: Vec<PartyId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::health::HealthConfig;
    use std::time::Duration;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    fn make_health_result(healthy: usize, degraded: usize, disconnected: usize, can_continue: bool) -> HealthCheckResult {
        HealthCheckResult {
            events: Vec::new(),
            healthy_count: healthy,
            degraded_count: degraded,
            disconnected_count: disconnected,
            can_continue,
        }
    }

    fn make_checkpoint(session_id: &str, step: u64, phase: MPCPhase) -> SessionCheckpoint {
        SessionCheckpoint {
            session_id: session_id.to_string(),
            config: MPCConfig::three_party(),
            phase,
            current_step: step,
            participants: Vec::new(),
            pool_levels: Vec::new(),
            has_model_shares: Vec::new(),
            created_at: 0,
            version: 1,
        }
    }

    #[test]
    fn test_shamir_threshold_continuation() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        // 1 party drops from 5 (threshold=3, 4 remain >= 3)
        let health = make_health_result(4, 0, 1, true);
        let active = parties[..4].to_vec();
        let dropped = vec![parties[4].clone()];

        let outcome = coordinator.handle_party_dropout(
            "session-1",
            &config,
            &health,
            MPCPhase::Computation,
            10,
            &active,
            &dropped,
            None,
        );

        match outcome {
            RecoveryOutcome::ContinueWithReduced { active_parties, removed_parties, lagrange_indices } => {
                assert_eq!(active_parties.len(), 4);
                assert_eq!(removed_parties.len(), 1);
                assert_eq!(lagrange_indices, vec![0, 1, 2, 3]);
            }
            other => panic!("expected ContinueWithReduced, got {:?}", other),
        }
    }

    #[test]
    fn test_shamir_below_threshold_restarts() {
        // Use high max_drops_before_restart so we hit the threshold check, not drop limit
        let recovery_config = RecoveryConfig {
            max_drops_before_restart: 10,
            ..Default::default()
        };
        let mut coordinator = SessionRecoveryCoordinator::new(recovery_config);
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        // 3 parties drop (only 2 remain < threshold 3)
        let health = make_health_result(2, 0, 3, false);
        let active = parties[..2].to_vec();
        let dropped = parties[2..].to_vec();
        let ckpt = make_checkpoint("session-1", 10, MPCPhase::Computation);

        let outcome = coordinator.handle_party_dropout(
            "session-1",
            &config,
            &health,
            MPCPhase::Computation,
            10,
            &active,
            &dropped,
            Some(ckpt),
        );

        match outcome {
            RecoveryOutcome::Restart { checkpoint, min_parties_needed, reason } => {
                assert_eq!(checkpoint.current_step, 10);
                assert_eq!(min_parties_needed, 3); // 5 - 2
                assert!(reason.contains("below threshold"));
            }
            other => panic!("expected Restart, got {:?}", other),
        }
    }

    #[test]
    fn test_additive_no_tolerance() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::three_party(); // additive, use_shamir=false
        let parties = test_parties(3);

        // 1 party drops (additive needs ALL)
        let health = make_health_result(2, 0, 1, false);
        let active = parties[..2].to_vec();
        let dropped = vec![parties[2].clone()];
        let ckpt = make_checkpoint("session-2", 5, MPCPhase::Computation);

        let outcome = coordinator.handle_party_dropout(
            "session-2",
            &config,
            &health,
            MPCPhase::Computation,
            5,
            &active,
            &dropped,
            Some(ckpt),
        );

        match outcome {
            RecoveryOutcome::Restart { .. } => {} // Expected
            other => panic!("expected Restart for additive sharing, got {:?}", other),
        }
    }

    #[test]
    fn test_max_drops_forces_restart() {
        let config_recovery = RecoveryConfig {
            max_drops_before_restart: 1,
            ..Default::default()
        };
        let mut coordinator = SessionRecoveryCoordinator::new(config_recovery);
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        // First drop — allowed
        let health = make_health_result(4, 0, 1, true);
        let active = parties[..4].to_vec();
        let dropped = vec![parties[4].clone()];

        let outcome1 = coordinator.handle_party_dropout(
            "session-3",
            &config,
            &health,
            MPCPhase::Computation,
            5,
            &active,
            &dropped,
            None,
        );
        assert!(matches!(outcome1, RecoveryOutcome::ContinueWithReduced { .. }));

        // Second drop — exceeds max_drops_before_restart=1, forced restart
        let health2 = make_health_result(3, 0, 2, true);
        let active2 = parties[..3].to_vec();
        let dropped2 = vec![parties[3].clone()];
        let ckpt = make_checkpoint("session-3", 6, MPCPhase::Computation);

        let outcome2 = coordinator.handle_party_dropout(
            "session-3",
            &config,
            &health2,
            MPCPhase::Computation,
            6,
            &active2,
            &dropped2,
            Some(ckpt),
        );
        match outcome2 {
            RecoveryOutcome::Restart { reason, .. } => {
                assert!(reason.contains("maximum allowed party drops"));
            }
            other => panic!("expected Restart, got {:?}", other),
        }
    }

    #[test]
    fn test_max_restart_attempts_aborts() {
        let config_recovery = RecoveryConfig {
            max_restart_attempts: 2,
            ..Default::default()
        };
        let mut coordinator = SessionRecoveryCoordinator::new(config_recovery);
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        // Force restart conditions 3 times (exceeds max_restart_attempts=2)
        for i in 0..3 {
            let health = make_health_result(1, 0, 4, false);
            let active = vec![parties[0].clone()];
            let dropped = parties[1..].to_vec();
            let ckpt = make_checkpoint("session-4", i, MPCPhase::Computation);

            // Reset drops between attempts to avoid max_drops_before_restart
            if let Some(state) = coordinator.sessions.get_mut("session-4") {
                state.total_drops = 0;
            }

            let outcome = coordinator.handle_party_dropout(
                "session-4",
                &config,
                &health,
                MPCPhase::Computation,
                i,
                &active,
                &dropped,
                Some(ckpt),
            );

            if i < 2 {
                assert!(matches!(outcome, RecoveryOutcome::Restart { .. }), "attempt {} should restart", i);
            } else {
                match outcome {
                    RecoveryOutcome::Abort { reason } => {
                        assert!(reason.contains("exceeded"));
                    }
                    other => panic!("expected Abort on attempt {}, got {:?}", i, other),
                }
            }
        }
    }

    #[test]
    fn test_no_checkpoint_aborts() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        // Below threshold with no checkpoint
        let health = make_health_result(2, 0, 3, false);
        let active = parties[..2].to_vec();
        let dropped = parties[2..].to_vec();

        let outcome = coordinator.handle_party_dropout(
            "session-5",
            &config,
            &health,
            MPCPhase::Computation,
            10,
            &active,
            &dropped,
            None, // No checkpoint
        );

        match outcome {
            RecoveryOutcome::Abort { reason } => {
                assert!(reason.contains("no checkpoint available"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_build_restart_plan() {
        let coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);
        let ckpt = make_checkpoint("session-6", 15, MPCPhase::Computation);

        let plan = coordinator.build_restart_plan(
            "session-6",
            ckpt,
            &parties[..3],
            &parties[3..],
            &config,
        );

        assert_eq!(plan.original_session_id, "session-6");
        assert_eq!(plan.surviving_parties.len(), 3);
        assert_eq!(plan.dropped_parties.len(), 2);
        assert_eq!(plan.replacements_needed, 2);
        assert_eq!(plan.resume_step, 15);
        assert_eq!(plan.resume_phase, MPCPhase::Computation);
    }

    #[test]
    fn test_reset_session() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        let health = make_health_result(4, 0, 1, true);
        coordinator.handle_party_dropout(
            "session-7",
            &config,
            &health,
            MPCPhase::Computation,
            5,
            &parties[..4],
            &parties[4..],
            None,
        );

        assert!(coordinator.session_stats("session-7").is_some());
        coordinator.reset_session("session-7");
        assert!(coordinator.session_stats("session-7").is_none());
    }

    #[test]
    fn test_recovery_stats() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        let health = make_health_result(4, 0, 1, true);
        coordinator.handle_party_dropout(
            "session-8",
            &config,
            &health,
            MPCPhase::Computation,
            5,
            &parties[..4],
            &[parties[4].clone()],
            None,
        );

        let stats = coordinator.session_stats("session-8").unwrap();
        assert_eq!(stats.total_drops, 1);
        assert_eq!(stats.restart_attempts, 0);
        assert_eq!(stats.dropped_parties.len(), 1);
    }

    #[test]
    fn test_evaluate_health_no_issues() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        let mut monitor = PartyHealthMonitor::new(
            &parties,
            HealthConfig::default(),
            config.clone(),
        );

        // All healthy — no recovery needed
        let outcome = coordinator.evaluate_health(
            "session-9",
            &config,
            &mut monitor,
            MPCPhase::Computation,
            5,
            None,
        );

        assert!(outcome.is_none());
    }

    #[test]
    fn test_evaluate_health_with_dropout() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        let health_config = HealthConfig {
            disconnect_threshold: Duration::from_millis(1),
            degraded_threshold: Duration::from_millis(1),
            ..Default::default()
        };
        let mut monitor = PartyHealthMonitor::new(
            &parties,
            health_config,
            config.clone(),
        );

        // Wait for timeout, then keep 3 alive
        std::thread::sleep(Duration::from_millis(5));
        monitor.record_heartbeat(&parties[0]);
        monitor.record_heartbeat(&parties[1]);
        monitor.record_heartbeat(&parties[2]);

        let outcome = coordinator.evaluate_health(
            "session-10",
            &config,
            &mut monitor,
            MPCPhase::Computation,
            5,
            None,
        );

        assert!(outcome.is_some());
        match outcome.unwrap() {
            RecoveryOutcome::ContinueWithReduced { active_parties, removed_parties, .. } => {
                assert_eq!(active_parties.len(), 3);
                assert_eq!(removed_parties.len(), 2);
            }
            other => panic!("expected ContinueWithReduced, got {:?}", other),
        }
    }

    #[test]
    fn test_lagrange_indices_correct() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::shamir(5, 3);
        let parties = test_parties(5);

        // Drop parties 1 and 3, keep 0, 2, 4
        let health = make_health_result(3, 0, 2, true);
        let active = vec![parties[0].clone(), parties[2].clone(), parties[4].clone()];
        let dropped = vec![parties[1].clone(), parties[3].clone()];

        let outcome = coordinator.handle_party_dropout(
            "session-11",
            &config,
            &health,
            MPCPhase::Computation,
            5,
            &active,
            &dropped,
            None,
        );

        match outcome {
            RecoveryOutcome::ContinueWithReduced { lagrange_indices, .. } => {
                assert_eq!(lagrange_indices, vec![0, 2, 4]);
            }
            other => panic!("expected ContinueWithReduced, got {:?}", other),
        }
    }

    #[test]
    fn test_additive_all_present_continues() {
        let mut coordinator = SessionRecoveryCoordinator::with_defaults();
        let config = MPCConfig::three_party(); // additive
        let parties = test_parties(3);

        // All present but degraded (can_continue=true for additive only if all present)
        let health = make_health_result(1, 2, 0, true);
        let active = parties.clone();
        let dropped: Vec<PartyId> = Vec::new();

        let outcome = coordinator.handle_party_dropout(
            "session-12",
            &config,
            &health,
            MPCPhase::Computation,
            5,
            &active,
            &dropped,
            None,
        );

        match outcome {
            RecoveryOutcome::ContinueWithReduced { active_parties, removed_parties, .. } => {
                assert_eq!(active_parties.len(), 3);
                assert!(removed_parties.is_empty());
            }
            other => panic!("expected ContinueWithReduced, got {:?}", other),
        }
    }
}
