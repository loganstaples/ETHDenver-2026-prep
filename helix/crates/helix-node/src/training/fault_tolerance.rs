//! Fault Tolerance for Distributed Training.
//!
//! Provides mechanisms for handling worker failures:
//! - Heartbeat-based failure detection
//! - Worker replacement and recovery
//! - Minimum participant thresholds
//! - Graceful degradation
//! - Leader failover

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::sync::{broadcast, mpsc};
use tokio::time::interval;

use crate::network::messages::PeerId;

/// Worker health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    /// Worker is healthy (responding to heartbeats).
    Healthy,
    /// Worker is degraded (slow responses, high latency).
    Degraded,
    /// Worker is suspected (missed heartbeats).
    Suspected,
    /// Worker is failed (confirmed unreachable).
    Failed,
    /// Worker is recovering (rejoining after failure).
    Recovering,
}

/// Information about a worker's health.
#[derive(Debug, Clone)]
pub struct WorkerHealthInfo {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Current health status.
    pub health: WorkerHealth,
    /// Last heartbeat received.
    pub last_heartbeat: Instant,
    /// Heartbeat latency (rolling average).
    pub avg_latency_ms: f64,
    /// Consecutive missed heartbeats.
    pub missed_heartbeats: u32,
    /// Number of failures since joining.
    pub failure_count: u32,
    /// Time of last failure.
    pub last_failure: Option<Instant>,
    /// Whether the worker is excluded from rounds.
    pub excluded: bool,
}

impl WorkerHealthInfo {
    pub fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            health: WorkerHealth::Healthy,
            last_heartbeat: Instant::now(),
            avg_latency_ms: 0.0,
            missed_heartbeats: 0,
            failure_count: 0,
            last_failure: None,
            excluded: false,
        }
    }

    pub fn update_heartbeat(&mut self, latency_ms: f64) {
        self.last_heartbeat = Instant::now();
        self.missed_heartbeats = 0;

        // Update rolling average latency
        const ALPHA: f64 = 0.3;
        self.avg_latency_ms = ALPHA * latency_ms + (1.0 - ALPHA) * self.avg_latency_ms;

        // Update health based on latency
        if latency_ms < 100.0 {
            self.health = WorkerHealth::Healthy;
        } else if latency_ms < 500.0 {
            self.health = WorkerHealth::Degraded;
        }
    }

    pub fn miss_heartbeat(&mut self) {
        self.missed_heartbeats += 1;

        // Update health based on missed heartbeats
        if self.missed_heartbeats >= 3 {
            self.health = WorkerHealth::Failed;
        } else if self.missed_heartbeats >= 1 {
            self.health = WorkerHealth::Suspected;
        }
    }

    pub fn mark_failed(&mut self) {
        self.health = WorkerHealth::Failed;
        self.failure_count += 1;
        self.last_failure = Some(Instant::now());
    }

    pub fn mark_recovering(&mut self) {
        self.health = WorkerHealth::Recovering;
    }

    pub fn mark_healthy(&mut self) {
        self.health = WorkerHealth::Healthy;
        self.missed_heartbeats = 0;
    }

    pub fn is_available(&self) -> bool {
        !self.excluded
            && matches!(
                self.health,
                WorkerHealth::Healthy | WorkerHealth::Degraded | WorkerHealth::Recovering
            )
    }
}

/// Configuration for fault tolerance.
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Heartbeat interval.
    pub heartbeat_interval: Duration,
    /// Heartbeat timeout (before marking as suspected).
    pub heartbeat_timeout: Duration,
    /// Number of missed heartbeats before failure.
    pub max_missed_heartbeats: u32,
    /// Minimum healthy workers required.
    pub min_healthy_workers: usize,
    /// Maximum failures before exclusion.
    pub max_failures_before_exclusion: u32,
    /// Cooldown period after failure before rejoining.
    pub failure_cooldown: Duration,
    /// Enable automatic recovery.
    pub auto_recovery: bool,
    /// Enable leader failover.
    pub leader_failover: bool,
    /// Maximum degraded workers allowed (as fraction).
    pub max_degraded_fraction: f64,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(15),
            max_missed_heartbeats: 3,
            min_healthy_workers: 2,
            max_failures_before_exclusion: 3,
            failure_cooldown: Duration::from_secs(60),
            auto_recovery: true,
            leader_failover: true,
            max_degraded_fraction: 0.33,
        }
    }
}

/// Event from the fault tolerance system.
#[derive(Debug, Clone)]
pub enum FaultEvent {
    /// Worker health changed.
    HealthChanged {
        peer_id: PeerId,
        from: WorkerHealth,
        to: WorkerHealth,
    },
    /// Worker failed.
    WorkerFailed {
        peer_id: PeerId,
        reason: String,
        failure_count: u32,
    },
    /// Worker recovered.
    WorkerRecovered { peer_id: PeerId },
    /// Worker excluded.
    WorkerExcluded {
        peer_id: PeerId,
        reason: String,
    },
    /// Leader failed.
    LeaderFailed { old_leader: PeerId },
    /// New leader elected.
    NewLeader { leader: PeerId },
    /// Insufficient workers.
    InsufficientWorkers {
        healthy: usize,
        required: usize,
    },
    /// Recovery complete.
    RecoveryComplete { worker_count: usize },
    /// System degraded.
    SystemDegraded {
        degraded_count: usize,
        healthy_count: usize,
    },
}

/// Replacement action for a failed worker.
#[derive(Debug, Clone)]
pub struct WorkerReplacement {
    /// Failed worker's peer ID.
    pub failed_peer: PeerId,
    /// Replacement worker's peer ID (if available).
    pub replacement_peer: Option<PeerId>,
    /// Shard to reassign.
    pub shard_index: u32,
    /// Action to take.
    pub action: ReplacementAction,
}

/// Action to take for worker replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplacementAction {
    /// Wait for worker to recover.
    WaitForRecovery,
    /// Assign a replacement worker.
    AssignReplacement,
    /// Redistribute shard to existing workers.
    Redistribute,
    /// Continue without replacement (degraded mode).
    ContinueDegraded,
    /// Abort the round.
    AbortRound,
}

/// Failure detector using heartbeats.
pub struct FailureDetector {
    /// Configuration.
    config: FaultToleranceConfig,
    /// Worker health information.
    workers: Arc<RwLock<HashMap<PeerId, WorkerHealthInfo>>>,
    /// Standby workers (available for replacement).
    standby_workers: Arc<RwLock<VecDeque<PeerId>>>,
    /// Current leader.
    current_leader: Arc<RwLock<Option<PeerId>>>,
    /// Event sender.
    event_tx: broadcast::Sender<FaultEvent>,
    /// Running flag.
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl FailureDetector {
    /// Creates a new failure detector.
    pub fn new(config: FaultToleranceConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        Self {
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            standby_workers: Arc::new(RwLock::new(VecDeque::new())),
            current_leader: Arc::new(RwLock::new(None)),
            event_tx,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Subscribes to fault events.
    pub fn subscribe(&self) -> broadcast::Receiver<FaultEvent> {
        self.event_tx.subscribe()
    }

    /// Registers a worker.
    pub fn register_worker(&self, peer_id: PeerId) {
        let mut workers = self.workers.write();
        if !workers.contains_key(&peer_id) {
            workers.insert(peer_id.clone(), WorkerHealthInfo::new(peer_id));
        }
    }

    /// Registers a standby worker.
    pub fn register_standby(&self, peer_id: PeerId) {
        let mut standby = self.standby_workers.write();
        if !standby.contains(&peer_id) {
            standby.push_back(peer_id);
        }
    }

    /// Unregisters a worker.
    pub fn unregister_worker(&self, peer_id: &PeerId) {
        self.workers.write().remove(peer_id);
        self.standby_workers.write().retain(|p| p != peer_id);
    }

    /// Sets the current leader.
    pub fn set_leader(&self, peer_id: PeerId) {
        *self.current_leader.write() = Some(peer_id);
    }

    /// Records a heartbeat from a worker.
    pub fn record_heartbeat(&self, peer_id: &PeerId, latency_ms: f64) {
        let mut workers = self.workers.write();
        if let Some(info) = workers.get_mut(peer_id) {
            let old_health = info.health;
            info.update_heartbeat(latency_ms);

            if old_health != info.health {
                let _ = self.event_tx.send(FaultEvent::HealthChanged {
                    peer_id: peer_id.clone(),
                    from: old_health,
                    to: info.health,
                });
            }
        }
    }

    /// Checks for failures (call periodically).
    pub fn check_failures(&self) -> Vec<WorkerReplacement> {
        let mut replacements = Vec::new();
        let mut workers = self.workers.write();
        let timeout = self.config.heartbeat_timeout;

        let mut newly_failed = Vec::new();

        for (peer_id, info) in workers.iter_mut() {
            if info.excluded {
                continue;
            }

            let elapsed = info.last_heartbeat.elapsed();

            if elapsed > timeout {
                info.miss_heartbeat();

                if info.missed_heartbeats >= self.config.max_missed_heartbeats {
                    let old_health = info.health;
                    info.mark_failed();

                    if old_health != WorkerHealth::Failed {
                        newly_failed.push(peer_id.clone());
                    }
                }
            }
        }

        drop(workers);

        // Process newly failed workers
        for peer_id in newly_failed {
            let replacement = self.handle_worker_failure(&peer_id, "Heartbeat timeout");
            if let Some(r) = replacement {
                replacements.push(r);
            }
        }

        replacements
    }

    /// Handles a worker failure.
    fn handle_worker_failure(
        &self,
        peer_id: &PeerId,
        reason: &str,
    ) -> Option<WorkerReplacement> {
        let failure_count = {
            let workers = self.workers.read();
            workers.get(peer_id).map(|w| w.failure_count).unwrap_or(0)
        };

        let _ = self.event_tx.send(FaultEvent::WorkerFailed {
            peer_id: peer_id.clone(),
            reason: reason.to_string(),
            failure_count,
        });

        // Check if this is the leader
        let is_leader = {
            let leader = self.current_leader.read();
            leader.as_ref() == Some(peer_id)
        };

        if is_leader && self.config.leader_failover {
            let _ = self.event_tx.send(FaultEvent::LeaderFailed {
                old_leader: peer_id.clone(),
            });
            self.elect_new_leader();
        }

        // Check if should exclude
        if failure_count >= self.config.max_failures_before_exclusion {
            let mut workers = self.workers.write();
            if let Some(info) = workers.get_mut(peer_id) {
                info.excluded = true;
            }

            let _ = self.event_tx.send(FaultEvent::WorkerExcluded {
                peer_id: peer_id.clone(),
                reason: format!(
                    "Exceeded maximum failures ({})",
                    self.config.max_failures_before_exclusion
                ),
            });

            // Try to get replacement
            let replacement_peer = self.get_replacement_worker();
            return Some(WorkerReplacement {
                failed_peer: peer_id.clone(),
                replacement_peer,
                shard_index: 0, // Would be set by caller
                action: if replacement_peer.is_some() {
                    ReplacementAction::AssignReplacement
                } else {
                    ReplacementAction::ContinueDegraded
                },
            });
        }

        // Determine action based on config
        let healthy_count = self.healthy_worker_count();
        let action = if self.config.auto_recovery {
            ReplacementAction::WaitForRecovery
        } else if healthy_count < self.config.min_healthy_workers {
            let _ = self.event_tx.send(FaultEvent::InsufficientWorkers {
                healthy: healthy_count,
                required: self.config.min_healthy_workers,
            });
            ReplacementAction::AbortRound
        } else {
            ReplacementAction::ContinueDegraded
        };

        Some(WorkerReplacement {
            failed_peer: peer_id.clone(),
            replacement_peer: None,
            shard_index: 0,
            action,
        })
    }

    /// Gets a replacement worker from standby.
    fn get_replacement_worker(&self) -> Option<PeerId> {
        let mut standby = self.standby_workers.write();
        standby.pop_front()
    }

    /// Elects a new leader.
    fn elect_new_leader(&self) {
        let workers = self.workers.read();

        // Find healthy worker with lowest failure count
        let new_leader = workers
            .iter()
            .filter(|(_, info)| info.is_available())
            .min_by_key(|(_, info)| info.failure_count)
            .map(|(id, _)| id.clone());

        drop(workers);

        if let Some(leader) = new_leader {
            *self.current_leader.write() = Some(leader.clone());
            let _ = self.event_tx.send(FaultEvent::NewLeader { leader });
        }
    }

    /// Attempts to recover a failed worker.
    pub fn attempt_recovery(&self, peer_id: &PeerId) -> bool {
        let mut workers = self.workers.write();

        if let Some(info) = workers.get_mut(peer_id) {
            if info.health == WorkerHealth::Failed {
                // Check cooldown
                if let Some(last_failure) = info.last_failure {
                    if last_failure.elapsed() < self.config.failure_cooldown {
                        return false;
                    }
                }

                info.mark_recovering();
                let _ = self.event_tx.send(FaultEvent::WorkerRecovered {
                    peer_id: peer_id.clone(),
                });
                return true;
            }
        }

        false
    }

    /// Completes recovery for a worker.
    pub fn complete_recovery(&self, peer_id: &PeerId) {
        let mut workers = self.workers.write();
        if let Some(info) = workers.get_mut(peer_id) {
            info.mark_healthy();
        }
    }

    /// Returns the count of healthy workers.
    pub fn healthy_worker_count(&self) -> usize {
        self.workers
            .read()
            .values()
            .filter(|w| w.health == WorkerHealth::Healthy && !w.excluded)
            .count()
    }

    /// Returns the count of available workers.
    pub fn available_worker_count(&self) -> usize {
        self.workers.read().values().filter(|w| w.is_available()).count()
    }

    /// Returns whether the system is healthy.
    pub fn is_system_healthy(&self) -> bool {
        let workers = self.workers.read();
        let total = workers.len();
        let healthy = workers
            .values()
            .filter(|w| w.health == WorkerHealth::Healthy)
            .count();
        let degraded = workers
            .values()
            .filter(|w| w.health == WorkerHealth::Degraded)
            .count();

        if total == 0 {
            return false;
        }

        let degraded_fraction = degraded as f64 / total as f64;

        healthy >= self.config.min_healthy_workers
            && degraded_fraction <= self.config.max_degraded_fraction
    }

    /// Returns worker health information.
    pub fn get_worker_health(&self, peer_id: &PeerId) -> Option<WorkerHealthInfo> {
        self.workers.read().get(peer_id).cloned()
    }

    /// Returns all worker health information.
    pub fn all_worker_health(&self) -> HashMap<PeerId, WorkerHealthInfo> {
        self.workers.read().clone()
    }

    /// Returns the current leader.
    pub fn current_leader(&self) -> Option<PeerId> {
        self.current_leader.read().clone()
    }

    /// Starts the background failure detection loop.
    pub fn start_detection_loop(&self) -> tokio::task::JoinHandle<()> {
        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);

        let running = Arc::clone(&self.running);
        let workers = Arc::clone(&self.workers);
        let event_tx = self.event_tx.clone();
        let timeout = self.config.heartbeat_timeout;
        let max_missed = self.config.max_missed_heartbeats;
        let min_healthy = self.config.min_healthy_workers;
        let interval_duration = self.config.heartbeat_interval;

        tokio::spawn(async move {
            let mut check_interval = interval(interval_duration);

            while running.load(std::sync::atomic::Ordering::SeqCst) {
                check_interval.tick().await;

                let mut newly_failed = Vec::new();

                {
                    let mut workers_guard = workers.write();

                    for (peer_id, info) in workers_guard.iter_mut() {
                        if info.excluded {
                            continue;
                        }

                        let elapsed = info.last_heartbeat.elapsed();

                        if elapsed > timeout {
                            info.miss_heartbeat();

                            if info.missed_heartbeats >= max_missed
                                && info.health != WorkerHealth::Failed
                            {
                                info.mark_failed();
                                newly_failed.push((peer_id.clone(), info.failure_count));
                            }
                        }
                    }
                }

                // Emit events for newly failed workers
                for (peer_id, failure_count) in newly_failed {
                    let _ = event_tx.send(FaultEvent::WorkerFailed {
                        peer_id,
                        reason: "Heartbeat timeout".to_string(),
                        failure_count,
                    });
                }

                // Check system health
                let (healthy, degraded, total) = {
                    let workers_guard = workers.read();
                    let total = workers_guard.len();
                    let healthy = workers_guard
                        .values()
                        .filter(|w| w.health == WorkerHealth::Healthy)
                        .count();
                    let degraded = workers_guard
                        .values()
                        .filter(|w| w.health == WorkerHealth::Degraded)
                        .count();
                    (healthy, degraded, total)
                };

                if healthy < min_healthy {
                    let _ = event_tx.send(FaultEvent::InsufficientWorkers {
                        healthy,
                        required: min_healthy,
                    });
                }

                if degraded > 0 {
                    let _ = event_tx.send(FaultEvent::SystemDegraded {
                        degraded_count: degraded,
                        healthy_count: healthy,
                    });
                }
            }
        })
    }

    /// Stops the detection loop.
    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Recovery coordinator for handling round recovery.
pub struct RecoveryCoordinator {
    /// Configuration.
    config: FaultToleranceConfig,
    /// Active recovery attempts.
    recoveries: Arc<RwLock<HashMap<PeerId, RecoveryAttempt>>>,
    /// Maximum concurrent recoveries.
    max_concurrent_recoveries: usize,
}

/// State of a recovery attempt.
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    pub peer_id: PeerId,
    pub started_at: Instant,
    pub attempt_number: u32,
    pub state: RecoveryState,
}

/// State of a recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryState {
    /// Waiting for worker to respond.
    Waiting,
    /// Sending catch-up data.
    SendingCatchup,
    /// Verifying worker is caught up.
    Verifying,
    /// Recovery complete.
    Complete,
    /// Recovery failed.
    Failed,
}

impl RecoveryCoordinator {
    /// Creates a new recovery coordinator.
    pub fn new(config: FaultToleranceConfig, max_concurrent: usize) -> Self {
        Self {
            config,
            recoveries: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent_recoveries: max_concurrent,
        }
    }

    /// Starts a recovery attempt for a worker.
    pub fn start_recovery(&self, peer_id: PeerId) -> Result<(), String> {
        let mut recoveries = self.recoveries.write();

        if recoveries.len() >= self.max_concurrent_recoveries {
            return Err("Maximum concurrent recoveries reached".to_string());
        }

        if recoveries.contains_key(&peer_id) {
            return Err("Recovery already in progress".to_string());
        }

        let attempt = RecoveryAttempt {
            peer_id: peer_id.clone(),
            started_at: Instant::now(),
            attempt_number: 1,
            state: RecoveryState::Waiting,
        };

        recoveries.insert(peer_id, attempt);
        Ok(())
    }

    /// Updates recovery state.
    pub fn update_recovery(&self, peer_id: &PeerId, state: RecoveryState) {
        let mut recoveries = self.recoveries.write();
        if let Some(attempt) = recoveries.get_mut(peer_id) {
            attempt.state = state;
        }
    }

    /// Completes a recovery.
    pub fn complete_recovery(&self, peer_id: &PeerId) -> Option<RecoveryAttempt> {
        let mut recoveries = self.recoveries.write();
        if let Some(mut attempt) = recoveries.remove(peer_id) {
            attempt.state = RecoveryState::Complete;
            Some(attempt)
        } else {
            None
        }
    }

    /// Fails a recovery.
    pub fn fail_recovery(&self, peer_id: &PeerId) -> Option<RecoveryAttempt> {
        let mut recoveries = self.recoveries.write();
        if let Some(mut attempt) = recoveries.remove(peer_id) {
            attempt.state = RecoveryState::Failed;
            Some(attempt)
        } else {
            None
        }
    }

    /// Returns active recoveries.
    pub fn active_recoveries(&self) -> Vec<RecoveryAttempt> {
        self.recoveries.read().values().cloned().collect()
    }

    /// Checks for timed-out recoveries.
    pub fn check_timeouts(&self) -> Vec<PeerId> {
        let timeout = self.config.failure_cooldown;
        let recoveries = self.recoveries.read();

        recoveries
            .iter()
            .filter(|(_, attempt)| {
                attempt.started_at.elapsed() > timeout && attempt.state != RecoveryState::Complete
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
}

/// Complete fault tolerance manager.
pub struct FaultToleranceManager {
    /// Failure detector.
    pub detector: FailureDetector,
    /// Recovery coordinator.
    pub recovery: RecoveryCoordinator,
    /// Configuration.
    config: FaultToleranceConfig,
}

impl FaultToleranceManager {
    /// Creates a new fault tolerance manager.
    pub fn new(config: FaultToleranceConfig) -> Self {
        Self {
            detector: FailureDetector::new(config.clone()),
            recovery: RecoveryCoordinator::new(config.clone(), 5),
            config,
        }
    }

    /// Registers a worker.
    pub fn register_worker(&self, peer_id: PeerId) {
        self.detector.register_worker(peer_id);
    }

    /// Unregisters a worker.
    pub fn unregister_worker(&self, peer_id: &PeerId) {
        self.detector.unregister_worker(peer_id);
    }

    /// Records a heartbeat.
    pub fn record_heartbeat(&self, peer_id: &PeerId, latency_ms: f64) {
        self.detector.record_heartbeat(peer_id, latency_ms);
    }

    /// Checks for failures and handles them.
    pub fn check_and_handle_failures(&self) -> Vec<WorkerReplacement> {
        let mut replacements = self.detector.check_failures();

        // Start recovery for failed workers if configured
        if self.config.auto_recovery {
            for replacement in &replacements {
                if replacement.action == ReplacementAction::WaitForRecovery {
                    let _ = self.recovery.start_recovery(replacement.failed_peer.clone());
                }
            }
        }

        // Check for timed-out recoveries
        let timed_out = self.recovery.check_timeouts();
        for peer_id in timed_out {
            self.recovery.fail_recovery(&peer_id);

            // Update replacement action
            for replacement in &mut replacements {
                if replacement.failed_peer == peer_id {
                    replacement.action = ReplacementAction::ContinueDegraded;
                }
            }
        }

        replacements
    }

    /// Returns whether the system can continue.
    pub fn can_continue(&self) -> bool {
        self.detector.available_worker_count() >= self.config.min_healthy_workers
    }

    /// Returns the detector.
    pub fn detector(&self) -> &FailureDetector {
        &self.detector
    }

    /// Returns the recovery coordinator.
    pub fn recovery(&self) -> &RecoveryCoordinator {
        &self.recovery
    }

    /// Subscribes to fault events.
    pub fn subscribe(&self) -> broadcast::Receiver<FaultEvent> {
        self.detector.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_peer_id(idx: usize) -> PeerId {
        PeerId::from_string(&format!("test-peer-{}", idx))
    }

    #[test]
    fn test_worker_health_update() {
        let mut info = WorkerHealthInfo::new(create_peer_id(0));
        assert_eq!(info.health, WorkerHealth::Healthy);

        // Fast heartbeat
        info.update_heartbeat(50.0);
        assert_eq!(info.health, WorkerHealth::Healthy);

        // Slow heartbeat
        info.update_heartbeat(300.0);
        assert_eq!(info.health, WorkerHealth::Degraded);
    }

    #[test]
    fn test_missed_heartbeats() {
        let mut info = WorkerHealthInfo::new(create_peer_id(0));

        info.miss_heartbeat();
        assert_eq!(info.health, WorkerHealth::Suspected);

        info.miss_heartbeat();
        assert_eq!(info.health, WorkerHealth::Suspected);

        info.miss_heartbeat();
        assert_eq!(info.health, WorkerHealth::Failed);
    }

    #[test]
    fn test_failure_detector() {
        let config = FaultToleranceConfig {
            heartbeat_timeout: Duration::from_millis(100),
            max_missed_heartbeats: 2,
            ..Default::default()
        };
        let detector = FailureDetector::new(config);

        // Register workers
        detector.register_worker(create_peer_id(0));
        detector.register_worker(create_peer_id(1));

        // Record heartbeat for worker 0
        detector.record_heartbeat(&create_peer_id(0), 10.0);

        assert_eq!(detector.healthy_worker_count(), 2);
    }

    #[test]
    fn test_worker_replacement() {
        let config = FaultToleranceConfig::default();
        let detector = FailureDetector::new(config);

        // Register worker and standby
        detector.register_worker(create_peer_id(0));
        detector.register_standby(create_peer_id(1));

        // Get replacement
        let replacement = detector.get_replacement_worker();
        assert_eq!(replacement, Some(create_peer_id(1)));
    }

    #[test]
    fn test_recovery_coordinator() {
        let config = FaultToleranceConfig::default();
        let coordinator = RecoveryCoordinator::new(config, 5);

        // Start recovery
        coordinator.start_recovery(create_peer_id(0)).unwrap();
        assert_eq!(coordinator.active_recoveries().len(), 1);

        // Complete recovery
        let attempt = coordinator.complete_recovery(&create_peer_id(0));
        assert!(attempt.is_some());
        assert_eq!(attempt.unwrap().state, RecoveryState::Complete);

        assert_eq!(coordinator.active_recoveries().len(), 0);
    }

    #[test]
    fn test_fault_tolerance_manager() {
        let config = FaultToleranceConfig {
            min_healthy_workers: 2,
            ..Default::default()
        };
        let manager = FaultToleranceManager::new(config);

        // Register workers
        manager.register_worker(create_peer_id(0));
        manager.register_worker(create_peer_id(1));
        manager.register_worker(create_peer_id(2));

        // Record heartbeats
        manager.record_heartbeat(&create_peer_id(0), 10.0);
        manager.record_heartbeat(&create_peer_id(1), 10.0);
        manager.record_heartbeat(&create_peer_id(2), 10.0);

        assert!(manager.can_continue());
    }

    #[test]
    fn test_leader_election() {
        let config = FaultToleranceConfig::default();
        let detector = FailureDetector::new(config);

        detector.register_worker(create_peer_id(0));
        detector.register_worker(create_peer_id(1));
        detector.set_leader(create_peer_id(0));

        assert_eq!(detector.current_leader(), Some(create_peer_id(0)));

        // Simulate leader failure and election
        detector.elect_new_leader();

        // New leader should be elected (either 0 or 1)
        assert!(detector.current_leader().is_some());
    }
}
