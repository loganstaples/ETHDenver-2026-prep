//! Integration tests for the resilient MPC training session at the node layer.
//!
//! Tests cover:
//! 1. Worker registration and health tracking
//! 2. Worker disconnect with grace period and reconnection
//! 3. Worker permanent removal after max reconnect attempts
//! 4. Insufficient workers detection
//! 5. Session state persistence (save/load snapshots)
//! 6. Graceful session shutdown with state save
//! 7. Snapshot resume after crash
//! 8. Explicit worker removal (cheater identified)
//! 9. Multi-worker lifecycle tracking
//! 10. Full session lifecycle integration test

use std::time::Duration;

use helix_node::network::messages::PeerId;
use helix_node::training::resilient_session::{
    ResilientSession, ResilientSessionConfig, WorkerSessionState,
};

fn test_peer_id() -> PeerId {
    PeerId::random()
}

// ============================================================================
// 1. Worker Registration and Health Tracking
// ============================================================================

#[test]
fn test_worker_registration() {
    let config = ResilientSessionConfig::for_testing();
    let mut session = ResilientSession::new("reg-test".to_string(), config);

    assert_eq!(session.active_worker_count(), 0);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);

    assert_eq!(session.active_worker_count(), 3);
    assert!(session.has_sufficient_workers());

    let active = session.active_party_indices();
    assert_eq!(active.len(), 3);
    assert!(active.contains(&0));
    assert!(active.contains(&1));
    assert!(active.contains(&2));
}

#[test]
fn test_heartbeat_tracking() {
    let config = ResilientSessionConfig::for_testing();
    let mut session = ResilientSession::new("hb-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.record_heartbeat(0);
    session.record_heartbeat(0);
    session.record_heartbeat(0);

    let worker = session.worker(0).unwrap();
    assert_eq!(worker.messages_received, 3);
    assert!(worker.is_available());
}

#[test]
fn test_step_tracking() {
    let config = ResilientSessionConfig::for_testing();
    let mut session = ResilientSession::new("step-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);

    session.update_step(1);
    session.update_step(2);
    session.update_step(3);

    let w0 = session.worker(0).unwrap();
    assert_eq!(w0.steps_completed, 3);

    let w1 = session.worker(1).unwrap();
    assert_eq!(w1.steps_completed, 3);
}

// ============================================================================
// 2. Worker Disconnect with Grace Period and Reconnection
// ============================================================================

#[test]
fn test_worker_disconnect_grace_period() {
    let config = ResilientSessionConfig {
        disconnect_grace_period: Duration::from_millis(10),
        max_reconnect_attempts: 3,
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("grace-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);

    // Let grace period expire for all workers.
    std::thread::sleep(Duration::from_millis(20));

    // Only worker 0 sends a heartbeat.
    session.record_heartbeat(0);

    // Check health — workers 1 and 2 should enter grace period.
    session.check_worker_health();

    let w1 = session.worker(1).unwrap();
    assert!(
        matches!(w1.state, WorkerSessionState::GracePeriod { .. }),
        "Worker 1 should be in grace period, got: {:?}",
        w1.state,
    );

    let w2 = session.worker(2).unwrap();
    assert!(
        matches!(w2.state, WorkerSessionState::GracePeriod { .. }),
        "Worker 2 should be in grace period, got: {:?}",
        w2.state,
    );

    // Worker 0 should still be active.
    let w0 = session.worker(0).unwrap();
    assert!(w0.is_available());
}

#[test]
fn test_worker_reconnection_during_grace_period() {
    let config = ResilientSessionConfig {
        disconnect_grace_period: Duration::from_millis(10),
        max_reconnect_attempts: 5,
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("reconnect-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);

    // Let worker 1 enter grace period.
    std::thread::sleep(Duration::from_millis(20));
    session.record_heartbeat(0);
    session.check_worker_health();

    // Worker 1 should be in grace period.
    let w1 = session.worker(1).unwrap();
    assert!(matches!(w1.state, WorkerSessionState::GracePeriod { .. }));

    // Worker 1 reconnects.
    session.record_heartbeat(1);

    // Worker 1 should now be in Reconnected state.
    let w1 = session.worker(1).unwrap();
    assert!(
        matches!(w1.state, WorkerSessionState::Reconnected),
        "Worker should be Reconnected, got: {:?}",
        w1.state,
    );
    assert!(w1.is_available());
}

// ============================================================================
// 3. Worker Permanent Removal After Max Reconnect Attempts
// ============================================================================

#[test]
fn test_worker_permanent_removal_after_max_attempts() {
    let config = ResilientSessionConfig {
        disconnect_grace_period: Duration::from_millis(10),
        max_reconnect_attempts: 1,
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("removal-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);

    // Let grace period expire.
    std::thread::sleep(Duration::from_millis(20));
    session.record_heartbeat(0);
    session.record_heartbeat(1);

    // First check: worker 2 enters grace period.
    session.check_worker_health();
    assert_eq!(session.active_worker_count(), 2);

    // Wait for grace period to expire again.
    std::thread::sleep(Duration::from_millis(20));
    session.record_heartbeat(0);
    session.record_heartbeat(1);

    // Second check: worker 2 should be permanently removed
    // (max_reconnect_attempts=1, already had 1 attempt).
    let removed = session.check_worker_health();
    assert!(
        removed.contains(&2),
        "Worker 2 should be permanently removed after max attempts"
    );

    let w2 = session.worker(2).unwrap();
    assert!(
        matches!(w2.state, WorkerSessionState::Removed { .. }),
        "Worker 2 should be in Removed state, got: {:?}",
        w2.state,
    );
}

// ============================================================================
// 4. Insufficient Workers Detection
// ============================================================================

#[test]
fn test_insufficient_workers_detection() {
    let config = ResilientSessionConfig {
        min_workers: 3,
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("insufficient-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);

    assert!(session.has_sufficient_workers());

    session.remove_worker(2, "cheater identified");
    assert!(!session.has_sufficient_workers());
    assert_eq!(session.active_worker_count(), 2);
}

#[test]
fn test_min_workers_boundary() {
    let config = ResilientSessionConfig {
        min_workers: 2,
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("boundary-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);

    // Remove one — still sufficient (3 → 2, min=2).
    session.remove_worker(2, "test");
    assert!(session.has_sufficient_workers());

    // Remove another — insufficient (2 → 1, min=2).
    session.remove_worker(1, "test");
    assert!(!session.has_sufficient_workers());
}

// ============================================================================
// 5. Session State Persistence
// ============================================================================

#[test]
fn test_session_snapshot_save_and_load() {
    let dir = tempfile::tempdir().unwrap();
    let config = ResilientSessionConfig {
        persistence_dir: dir.path().to_path_buf(),
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("snap-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);
    session.update_step(42);
    session.record_checkpoint_submitted();
    session.record_checkpoint_submitted();
    session.remove_worker(2, "cheater");

    let path = session.save_snapshot().unwrap();
    assert!(path.exists());

    let snapshot = ResilientSession::load_snapshot(&path).unwrap();
    assert_eq!(snapshot.session_id, "snap-test");
    assert_eq!(snapshot.current_step, 42);
    assert_eq!(snapshot.active_workers.len(), 2);
    assert_eq!(snapshot.removed_workers.len(), 1);
    assert_eq!(snapshot.removed_workers[0].0, 2);
    assert_eq!(snapshot.removed_workers[0].1, "cheater");
    assert_eq!(snapshot.checkpoints_submitted, 2);
}

#[test]
fn test_find_latest_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let config = ResilientSessionConfig {
        persistence_dir: dir.path().to_path_buf(),
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("latest-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);

    // Save at step 10.
    session.update_step(10);
    session.save_snapshot().unwrap();

    std::thread::sleep(Duration::from_millis(20));

    // Save at step 20.
    session.update_step(20);
    session.save_snapshot().unwrap();

    std::thread::sleep(Duration::from_millis(20));

    // Save at step 30.
    session.update_step(30);
    session.save_snapshot().unwrap();

    // Find latest should return step 30.
    let latest = ResilientSession::find_latest_snapshot(dir.path(), "latest-test");
    assert!(latest.is_some());

    let path = latest.unwrap();
    assert!(
        path.to_string_lossy().contains("step_30"),
        "Expected step_30 in path, got: {}",
        path.display(),
    );

    let snapshot = ResilientSession::load_snapshot(&path).unwrap();
    assert_eq!(snapshot.current_step, 30);
}

// ============================================================================
// 6. Graceful Session Shutdown
// ============================================================================

#[test]
fn test_graceful_session_shutdown() {
    let dir = tempfile::tempdir().unwrap();
    let config = ResilientSessionConfig {
        persistence_dir: dir.path().to_path_buf(),
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("shutdown-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.update_step(50);
    session.record_checkpoint_submitted();

    assert!(!session.is_shutdown_requested());

    session.request_shutdown();
    assert!(session.is_shutdown_requested());

    // Double shutdown request is safe.
    session.request_shutdown();
    assert!(session.is_shutdown_requested());

    let path = session.execute_shutdown().unwrap();
    assert!(path.exists());

    let snapshot = ResilientSession::load_snapshot(&path).unwrap();
    assert_eq!(snapshot.current_step, 50);
    assert_eq!(snapshot.checkpoints_submitted, 1);
}

// ============================================================================
// 7. Snapshot Resume After Crash
// ============================================================================

#[test]
fn test_snapshot_resume_after_crash() {
    let dir = tempfile::tempdir().unwrap();

    // Original session.
    {
        let config = ResilientSessionConfig {
            persistence_dir: dir.path().to_path_buf(),
            ..ResilientSessionConfig::for_testing()
        };
        let mut session = ResilientSession::new("crash-test".to_string(), config);

        session.register_worker(test_peer_id(), 0);
        session.register_worker(test_peer_id(), 1);
        session.register_worker(test_peer_id(), 2);
        session.update_step(75);
        session.record_checkpoint_submitted();
        session.record_checkpoint_submitted();
        session.record_checkpoint_submitted();
        session.remove_worker(2, "disconnect");
        session.save_snapshot().unwrap();
        // Session drops here (simulating crash).
    }

    // Recovery: load the latest snapshot.
    let latest = ResilientSession::find_latest_snapshot(dir.path(), "crash-test");
    assert!(latest.is_some());

    let snapshot = ResilientSession::load_snapshot(&latest.unwrap()).unwrap();
    assert_eq!(snapshot.session_id, "crash-test");
    assert_eq!(snapshot.current_step, 75);
    assert_eq!(snapshot.active_workers.len(), 2);
    assert_eq!(snapshot.removed_workers.len(), 1);
    assert_eq!(snapshot.removed_workers[0].0, 2);
    assert_eq!(snapshot.checkpoints_submitted, 3);
}

// ============================================================================
// 8. Explicit Worker Removal (Cheater Identified)
// ============================================================================

#[test]
fn test_explicit_worker_removal_cheater() {
    let config = ResilientSessionConfig::for_testing();
    let mut session = ResilientSession::new("cheater-test".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);

    assert_eq!(session.active_worker_count(), 3);

    // Identify worker 1 as a cheater and remove.
    session.remove_worker(1, "MAC verification failed at step 42");

    assert_eq!(session.active_worker_count(), 2);
    let w1 = session.worker(1).unwrap();
    assert!(matches!(w1.state, WorkerSessionState::Removed { .. }));
    assert!(!w1.is_available());

    // Remaining workers should still be active.
    let active = session.active_party_indices();
    assert_eq!(active.len(), 2);
    assert!(active.contains(&0));
    assert!(active.contains(&2));

    let removed = session.removed_party_indices();
    assert_eq!(removed, vec![1]);
}

// ============================================================================
// 9. Multi-Worker Lifecycle Tracking
// ============================================================================

#[test]
fn test_multi_worker_lifecycle() {
    let config = ResilientSessionConfig {
        disconnect_grace_period: Duration::from_millis(10),
        max_reconnect_attempts: 2,
        min_workers: 2,
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("lifecycle-test".to_string(), config);

    // Phase 1: Register all workers.
    for i in 0..5 {
        session.register_worker(test_peer_id(), i);
    }
    assert_eq!(session.active_worker_count(), 5);

    // Phase 2: Simulate training with heartbeats.
    session.update_step(1);
    for i in 0..5 {
        session.record_heartbeat(i);
    }

    // Phase 3: Worker 3 stops responding.
    std::thread::sleep(Duration::from_millis(20));
    for i in [0, 1, 2, 4] {
        session.record_heartbeat(i);
    }
    session.check_worker_health();
    assert_eq!(session.active_worker_count(), 4); // Worker 3 in grace period

    // Phase 4: Worker 3 reconnects.
    session.record_heartbeat(3);
    let w3 = session.worker(3).unwrap();
    assert!(matches!(w3.state, WorkerSessionState::Reconnected));
    assert_eq!(session.active_worker_count(), 5);

    // Phase 5: Workers 3 and 4 disconnect permanently.
    std::thread::sleep(Duration::from_millis(20));
    for i in [0, 1, 2] {
        session.record_heartbeat(i);
    }
    session.check_worker_health(); // grace period starts
    std::thread::sleep(Duration::from_millis(20));
    for i in [0, 1, 2] {
        session.record_heartbeat(i);
    }
    session.check_worker_health(); // another attempt
    std::thread::sleep(Duration::from_millis(20));
    for i in [0, 1, 2] {
        session.record_heartbeat(i);
    }
    let removed = session.check_worker_health(); // exceeded max attempts

    // Workers 3 and 4 should be permanently removed.
    assert!(removed.contains(&3) || removed.contains(&4));
    assert!(session.has_sufficient_workers()); // Still 3 workers, min=2
}

// ============================================================================
// 10. Full Session Lifecycle Integration Test
// ============================================================================

#[test]
fn test_full_session_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let config = ResilientSessionConfig {
        disconnect_grace_period: Duration::from_millis(10),
        max_reconnect_attempts: 1,
        min_workers: 2,
        persistence_dir: dir.path().to_path_buf(),
        ..ResilientSessionConfig::for_testing()
    };
    let mut session = ResilientSession::new("full-lifecycle".to_string(), config);

    // Phase 1: Register workers.
    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);
    assert_eq!(session.active_worker_count(), 3);
    assert!(session.has_sufficient_workers());

    // Phase 2: Training progresses.
    for step in 1..=10 {
        session.update_step(step);
        for i in 0..3 {
            session.record_heartbeat(i);
        }
    }

    // Phase 3: Checkpoint.
    session.record_checkpoint_submitted();
    session.save_snapshot().unwrap();

    // Phase 4: Worker 2 disconnects.
    std::thread::sleep(Duration::from_millis(20));
    session.record_heartbeat(0);
    session.record_heartbeat(1);
    session.check_worker_health();

    // Worker 2 enters grace period.
    let w2 = session.worker(2).unwrap();
    assert!(matches!(w2.state, WorkerSessionState::GracePeriod { .. }));

    // Grace period expires → worker removed.
    std::thread::sleep(Duration::from_millis(20));
    session.record_heartbeat(0);
    session.record_heartbeat(1);
    let removed = session.check_worker_health();
    assert!(removed.contains(&2));
    assert_eq!(session.active_worker_count(), 2);
    assert!(session.has_sufficient_workers()); // Still 2, min=2

    // Phase 5: Continue training with 2 workers.
    for step in 11..=20 {
        session.update_step(step);
        session.record_heartbeat(0);
        session.record_heartbeat(1);
    }

    // Phase 6: Graceful shutdown.
    session.request_shutdown();
    let shutdown_path = session.execute_shutdown().unwrap();
    assert!(shutdown_path.exists());

    // Verify final state.
    let final_snapshot = ResilientSession::load_snapshot(&shutdown_path).unwrap();
    assert_eq!(final_snapshot.session_id, "full-lifecycle");
    assert_eq!(final_snapshot.current_step, 20);
    assert_eq!(final_snapshot.active_workers.len(), 2);
    assert_eq!(final_snapshot.removed_workers.len(), 1);
    assert_eq!(final_snapshot.removed_workers[0].0, 2);
    assert_eq!(final_snapshot.checkpoints_submitted, 1);

    // Drain events.
    let mut events = Vec::new();
    while let Some(event) = session.try_next_event() {
        events.push(format!("{:?}", event));
    }
    assert!(!events.is_empty(), "Should have captured some events");
}

#[test]
fn test_snapshot_nonexistent_session() {
    let dir = tempfile::tempdir().unwrap();
    let latest = ResilientSession::find_latest_snapshot(dir.path(), "nonexistent-session");
    assert!(latest.is_none());
}

#[test]
fn test_all_workers_info() {
    let config = ResilientSessionConfig::for_testing();
    let mut session = ResilientSession::new("all-info".to_string(), config);

    session.register_worker(test_peer_id(), 0);
    session.register_worker(test_peer_id(), 1);
    session.register_worker(test_peer_id(), 2);

    let all = session.all_workers();
    assert_eq!(all.len(), 3);
    assert!(all.contains_key(&0));
    assert!(all.contains_key(&1));
    assert!(all.contains_key(&2));
}

#[test]
fn test_checkpoint_counter() {
    let config = ResilientSessionConfig::for_testing();
    let mut session = ResilientSession::new("cp-count".to_string(), config);

    session.record_checkpoint_submitted();
    session.record_checkpoint_submitted();
    session.record_checkpoint_submitted();

    // No direct public accessor for checkpoint count, but save_snapshot includes it.
    let dir = tempfile::tempdir().unwrap();
    let config2 = ResilientSessionConfig {
        persistence_dir: dir.path().to_path_buf(),
        ..ResilientSessionConfig::for_testing()
    };
    let mut session2 = ResilientSession::new("cp-count-2".to_string(), config2);
    session2.register_worker(test_peer_id(), 0);
    session2.record_checkpoint_submitted();
    session2.record_checkpoint_submitted();
    let path = session2.save_snapshot().unwrap();
    let snapshot = ResilientSession::load_snapshot(&path).unwrap();
    assert_eq!(snapshot.checkpoints_submitted, 2);
}
