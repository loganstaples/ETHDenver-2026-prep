//! Subsystem health tracking and graceful degradation.
//!
//! Each node subsystem is classified as either **critical** (failure triggers
//! node shutdown) or **non-critical** (failure is logged and tolerated). The
//! [`SubsystemMonitor`] tracks the health of every subsystem and provides a
//! single [`should_shutdown`](SubsystemMonitor::should_shutdown) query that the
//! runtime can poll.

use std::time::Instant;

use dashmap::DashMap;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Subsystem
// ---------------------------------------------------------------------------

/// Identifies a node subsystem by name and criticality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Subsystem {
    // -- Critical: failure causes node shutdown --
    /// Core training loop.
    Training,
    /// ZK proof generation pipeline.
    ProofGeneration,
    /// P2P networking layer.
    Networking,

    // -- Non-critical: failure is logged and tolerated --
    /// Prometheus / metrics export.
    Metrics,
    /// Periodic model checkpointing.
    Checkpointing,
    /// IPFS storage backend.
    IpfsStorage,
    /// S3-compatible storage backend.
    S3Storage,
    /// On-chain proof / commitment submission.
    ChainSubmission,
}

impl Subsystem {
    /// Returns `true` if this subsystem is critical (failure = shutdown).
    pub fn is_critical(self) -> bool {
        matches!(
            self,
            Subsystem::Training | Subsystem::ProofGeneration | Subsystem::Networking
        )
    }

    /// Human-readable label for logging.
    pub fn label(self) -> &'static str {
        match self {
            Subsystem::Training => "training",
            Subsystem::ProofGeneration => "proof_generation",
            Subsystem::Networking => "networking",
            Subsystem::Metrics => "metrics",
            Subsystem::Checkpointing => "checkpointing",
            Subsystem::IpfsStorage => "ipfs_storage",
            Subsystem::S3Storage => "s3_storage",
            Subsystem::ChainSubmission => "chain_submission",
        }
    }
}

// ---------------------------------------------------------------------------
// SubsystemStatus
// ---------------------------------------------------------------------------

/// Current health status of a subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemStatus {
    /// Subsystem is operating normally.
    Healthy,
    /// Subsystem is experiencing intermittent issues.
    Degraded {
        /// When the degraded state began.
        since: Instant,
    },
    /// Subsystem has stopped functioning.
    Failed {
        /// When the failure was first detected.
        since: Instant,
    },
}

// ---------------------------------------------------------------------------
// SubsystemAction
// ---------------------------------------------------------------------------

/// Action to take when a subsystem failure is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemAction {
    /// A critical subsystem failed -- the node should shut down.
    Shutdown,
    /// A non-critical subsystem failed -- log the failure and continue.
    LogAndContinue,
}

// ---------------------------------------------------------------------------
// SubsystemMonitor
// ---------------------------------------------------------------------------

/// Tracks health of all node subsystems and determines whether the node should
/// continue operating or shut down.
pub struct SubsystemMonitor {
    statuses: DashMap<Subsystem, SubsystemStatus>,
}

impl SubsystemMonitor {
    /// Create a new monitor with all subsystems starting as [`SubsystemStatus::Healthy`].
    pub fn new() -> Self {
        let statuses = DashMap::new();

        let all_subsystems = [
            Subsystem::Training,
            Subsystem::ProofGeneration,
            Subsystem::Networking,
            Subsystem::Metrics,
            Subsystem::Checkpointing,
            Subsystem::IpfsStorage,
            Subsystem::S3Storage,
            Subsystem::ChainSubmission,
        ];

        for sub in all_subsystems {
            statuses.insert(sub, SubsystemStatus::Healthy);
        }

        Self { statuses }
    }

    /// Mark a subsystem as healthy.
    pub fn report_healthy(&self, subsystem: Subsystem) {
        info!(subsystem = subsystem.label(), "subsystem reported healthy");
        self.statuses.insert(subsystem, SubsystemStatus::Healthy);
    }

    /// Mark a subsystem as degraded.
    pub fn report_degraded(&self, subsystem: Subsystem) {
        warn!(
            subsystem = subsystem.label(),
            critical = subsystem.is_critical(),
            "subsystem reported degraded"
        );
        self.statuses
            .insert(subsystem, SubsystemStatus::Degraded { since: Instant::now() });
    }

    /// Mark a subsystem as failed and return the recommended action.
    ///
    /// - Critical subsystems return [`SubsystemAction::Shutdown`].
    /// - Non-critical subsystems return [`SubsystemAction::LogAndContinue`].
    pub fn report_failed(&self, subsystem: Subsystem) -> SubsystemAction {
        let action = if subsystem.is_critical() {
            SubsystemAction::Shutdown
        } else {
            SubsystemAction::LogAndContinue
        };

        warn!(
            subsystem = subsystem.label(),
            critical = subsystem.is_critical(),
            ?action,
            "subsystem reported FAILED"
        );

        self.statuses
            .insert(subsystem, SubsystemStatus::Failed { since: Instant::now() });

        action
    }

    /// Returns `true` if the given subsystem is classified as critical.
    pub fn is_critical(&self, subsystem: Subsystem) -> bool {
        subsystem.is_critical()
    }

    /// Returns `true` if any **critical** subsystem is in the
    /// [`SubsystemStatus::Failed`] state.
    pub fn should_shutdown(&self) -> bool {
        self.statuses.iter().any(|entry| {
            let subsystem = *entry.key();
            let status = *entry.value();
            subsystem.is_critical() && matches!(status, SubsystemStatus::Failed { .. })
        })
    }

    /// Returns a snapshot of all subsystem statuses, sorted by subsystem name
    /// for deterministic output.
    pub fn health_summary(&self) -> Vec<(Subsystem, SubsystemStatus)> {
        let mut out: Vec<_> = self
            .statuses
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();
        out.sort_by_key(|(sub, _)| sub.label());
        out
    }
}

impl Default for SubsystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_failure_triggers_shutdown() {
        let monitor = SubsystemMonitor::new();

        let action = monitor.report_failed(Subsystem::Training);
        assert_eq!(action, SubsystemAction::Shutdown);

        let action = monitor.report_failed(Subsystem::ProofGeneration);
        assert_eq!(action, SubsystemAction::Shutdown);

        let action = monitor.report_failed(Subsystem::Networking);
        assert_eq!(action, SubsystemAction::Shutdown);
    }

    #[test]
    fn test_noncritical_failure_continues() {
        let monitor = SubsystemMonitor::new();

        let action = monitor.report_failed(Subsystem::Metrics);
        assert_eq!(action, SubsystemAction::LogAndContinue);

        let action = monitor.report_failed(Subsystem::IpfsStorage);
        assert_eq!(action, SubsystemAction::LogAndContinue);

        let action = monitor.report_failed(Subsystem::S3Storage);
        assert_eq!(action, SubsystemAction::LogAndContinue);

        let action = monitor.report_failed(Subsystem::Checkpointing);
        assert_eq!(action, SubsystemAction::LogAndContinue);

        let action = monitor.report_failed(Subsystem::ChainSubmission);
        assert_eq!(action, SubsystemAction::LogAndContinue);
    }

    #[test]
    fn test_health_summary() {
        let monitor = SubsystemMonitor::new();

        // All subsystems should start healthy.
        let summary = monitor.health_summary();
        assert_eq!(summary.len(), 8);
        for (_, status) in &summary {
            assert_eq!(*status, SubsystemStatus::Healthy);
        }

        // Degrade one, fail another.
        monitor.report_degraded(Subsystem::IpfsStorage);
        monitor.report_failed(Subsystem::Metrics);

        let summary = monitor.health_summary();
        let ipfs = summary
            .iter()
            .find(|(s, _)| *s == Subsystem::IpfsStorage)
            .unwrap();
        assert!(matches!(ipfs.1, SubsystemStatus::Degraded { .. }));

        let metrics = summary
            .iter()
            .find(|(s, _)| *s == Subsystem::Metrics)
            .unwrap();
        assert!(matches!(metrics.1, SubsystemStatus::Failed { .. }));
    }

    #[test]
    fn test_degraded_tracking() {
        let monitor = SubsystemMonitor::new();
        let before = Instant::now();

        monitor.report_degraded(Subsystem::S3Storage);

        let summary = monitor.health_summary();
        let s3 = summary
            .iter()
            .find(|(s, _)| *s == Subsystem::S3Storage)
            .unwrap();

        match s3.1 {
            SubsystemStatus::Degraded { since } => {
                assert!(since >= before);
            }
            other => panic!("expected Degraded, got {:?}", other),
        }

        // Reporting healthy should recover.
        monitor.report_healthy(Subsystem::S3Storage);
        let summary = monitor.health_summary();
        let s3 = summary
            .iter()
            .find(|(s, _)| *s == Subsystem::S3Storage)
            .unwrap();
        assert_eq!(s3.1, SubsystemStatus::Healthy);
    }

    #[test]
    fn test_should_shutdown() {
        let monitor = SubsystemMonitor::new();

        // Non-critical failures should NOT trigger shutdown.
        monitor.report_failed(Subsystem::Metrics);
        monitor.report_failed(Subsystem::IpfsStorage);
        assert!(!monitor.should_shutdown());

        // Critical failure should trigger shutdown.
        monitor.report_failed(Subsystem::Training);
        assert!(monitor.should_shutdown());
    }
}
