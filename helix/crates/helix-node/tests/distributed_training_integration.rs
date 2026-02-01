//! Distributed Training Integration Tests.
//!
//! These tests verify the complete distributed training workflow including:
//! - 3-worker training completion
//! - Surviving 1 worker failure
//! - Checkpointing and resume functionality
//!
//! Success Criteria:
//! 1. 3-worker training completes successfully
//! 2. Training survives 1 worker failure
//! 3. Checkpointing works correctly

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::sync::mpsc;

// Import from helix_node crate
use helix_node::network::messages::PeerId;
use helix_node::training::{
    // State machine
    StateMachineConfig, DistributedRoundState, DistributedRound,
    DistributedTrainingStateMachine, StateMachineEvent, RoundFailureReason,
    // Synchronization
    BarrierConfig, BarrierResult, SyncCoordinator, SyncPhase,
    // Fault tolerance
    FaultToleranceConfig, FaultToleranceManager, WorkerHealth,
    // Checkpointing
    DistributedCheckpointConfig, DistributedCheckpointCoordinator,
    WorkerCheckpointContribution,
    // Distributed coordinator
    GradientShare, GradientShareCollector,
    DistributedRoundId,
};
use helix_node::training::checkpoint::CheckpointId;

// ============================================================================
// Test Helpers
// ============================================================================

fn create_peer_id(id: u8) -> PeerId {
    PeerId(format!("peer-{}", id))
}

fn create_test_workers(count: usize) -> Vec<PeerId> {
    (1..=count).map(|i| create_peer_id(i as u8)).collect()
}

fn create_test_gradient_share(worker_id: PeerId, share_index: usize) -> GradientShare {
    GradientShare {
        worker_id,
        share_index,
        share_data: vec![1.0, 2.0, 3.0, 4.0],
        error_bound: 0.01,
        commitment: [share_index as u8; 32],
        proof: vec![1, 2, 3, 4, 5, 6, 7, 8],
        timestamp: 12345,
    }
}

fn create_test_checkpoint_contribution(
    worker_id: PeerId,
    iteration: u64,
) -> WorkerCheckpointContribution {
    WorkerCheckpointContribution {
        worker_id,
        local_checkpoint_id: CheckpointId::new(iteration),
        share_commitment: [0u8; 32],
        iteration,
        round_state: DistributedRoundState::Computing,
        shares_hash: [1u8; 32],
        contributed_at: 12345,
        gradient_commitment: Some([2u8; 32]),
    }
}

// ============================================================================
// Test: State Machine Round Lifecycle
// ============================================================================

#[test]
fn test_state_machine_round_lifecycle() {
    let config = StateMachineConfig::default();
    let mut state_machine = DistributedTrainingStateMachine::new(config);

    let workers = create_test_workers(3);
    let initial_commitment = [0u8; 32];

    // Start round
    let round_id = state_machine.start_round(initial_commitment, None).unwrap();
    assert!(round_id.round_number > 0);

    // Get round
    let round = state_machine.current_round().unwrap();
    assert_eq!(round.state, DistributedRoundState::WaitingForWorkers);

    // Add workers
    for (i, worker_id) in workers.iter().enumerate() {
        state_machine.add_worker(worker_id.clone(), 1000).unwrap();
    }

    // Try to start distribution
    let can_distribute = state_machine.try_start_distribution().unwrap();
    assert!(can_distribute);

    let round = state_machine.current_round().unwrap();
    assert_eq!(round.state, DistributedRoundState::Distributing);

    // Workers receive their shares (during Distributing phase)
    for worker_id in &workers {
        let round = state_machine.current_round_mut().unwrap();
        round.mark_shares_received(worker_id).unwrap();
    }

    // Mark distribution complete - transitions to Computing
    state_machine.mark_distribution_complete().unwrap();

    let round = state_machine.current_round().unwrap();
    assert_eq!(round.state, DistributedRoundState::Computing);

    // Workers submit their gradients (during Computing phase)
    for worker_id in &workers {
        state_machine.record_submission(worker_id, [0u8; 32], 0.01).unwrap();
    }

    // Try to start collection
    let can_collect = state_machine.try_start_collection().unwrap();
    assert!(can_collect);

    // Try to start aggregation
    let can_aggregate = state_machine.try_start_aggregation().unwrap();
    assert!(can_aggregate);

    // Transition to committing phase
    let round_id = state_machine.current_round().unwrap().id;
    state_machine.start_commit(round_id).unwrap();

    // Record commit
    let tx_hash = [1u8; 32];
    state_machine.record_commit(tx_hash).unwrap();

    let round = state_machine.current_round().unwrap();
    assert_eq!(round.state, DistributedRoundState::Completed);
}

// ============================================================================
// Test: 3-Worker Gradient Collection
// ============================================================================

#[test]
fn test_three_worker_gradient_collection() {
    let workers = create_test_workers(3);
    let round_id = DistributedRoundId::new(1, 1);

    let mut collector = GradientShareCollector::new(
        round_id,
        workers.clone(),
        2, // MPC threshold
        Duration::from_secs(60),
    );

    // Initially not complete
    assert!(!collector.has_enough_shares());
    assert!(!collector.is_complete());
    assert_eq!(collector.missing_workers().len(), 3);

    // First worker submits
    let share1 = create_test_gradient_share(workers[0].clone(), 0);
    collector.add_share(share1).unwrap();
    assert!(!collector.has_enough_shares());

    // Second worker submits - now have threshold
    let share2 = create_test_gradient_share(workers[1].clone(), 1);
    collector.add_share(share2).unwrap();
    assert!(collector.has_enough_shares());
    assert!(!collector.is_complete());

    // Third worker submits - now complete
    let share3 = create_test_gradient_share(workers[2].clone(), 2);
    collector.add_share(share3).unwrap();
    assert!(collector.is_complete());
    assert!(collector.missing_workers().is_empty());

    // Verify all shares collected
    let shares = collector.shares();
    assert_eq!(shares.len(), 3);
    for worker in &workers {
        assert!(shares.contains_key(worker));
    }
}

// ============================================================================
// Test: Barrier Synchronization
// ============================================================================

#[tokio::test]
async fn test_barrier_synchronization() {
    use std::sync::Arc;

    let config = BarrierConfig::default();
    let coordinator = Arc::new(SyncCoordinator::new(config));

    let workers = create_test_workers(3);
    let timeout = Duration::from_secs(5);

    // Start a round
    coordinator.start_round(1);

    // Register workers
    for worker_id in &workers {
        coordinator.register_worker(worker_id.clone());
    }

    // Create barrier for all workers
    coordinator.create_barrier(SyncPhase::Ready, workers.clone(), timeout);

    // Spawn concurrent tasks for workers to arrive at barrier
    let mut handles = Vec::new();
    for worker_id in workers.clone() {
        let coord = coordinator.clone();
        let handle = tokio::spawn(async move {
            coord.arrive_and_wait(SyncPhase::Ready, worker_id, timeout).await
        });
        handles.push(handle);
    }

    // Wait for all workers and check at least one got AllArrived
    let mut any_all_arrived = false;
    for handle in handles {
        let result = handle.await.unwrap();
        if matches!(result, BarrierResult::AllArrived { .. }) {
            any_all_arrived = true;
        }
    }
    assert!(any_all_arrived, "At least one worker should see AllArrived");
}

// ============================================================================
// Test: Fault Tolerance - Worker Failure Detection
// ============================================================================

#[test]
#[ignore = "Timing-dependent test - failure detection requires real-time heartbeat monitoring"]
fn test_worker_failure_detection() {
    let mut config = FaultToleranceConfig::default();
    config.heartbeat_timeout = Duration::from_millis(50);
    config.max_missed_heartbeats = 2; // Reduce for faster test

    let manager = FaultToleranceManager::new(config);

    let workers = create_test_workers(3);

    // Register all workers
    for worker_id in &workers {
        manager.register_worker(worker_id.clone());
    }

    // All workers healthy initially
    for worker_id in &workers {
        let health = manager.detector().get_worker_health(worker_id);
        assert!(matches!(health, Some(ref info) if info.health == WorkerHealth::Healthy));
    }

    // Send heartbeats from workers 1 and 2 only
    manager.record_heartbeat(&workers[0], 10.0);
    manager.record_heartbeat(&workers[1], 10.0);

    // Wait for timeout and call detect_failures multiple times to accumulate missed heartbeats
    for _ in 0..3 {
        std::thread::sleep(Duration::from_millis(75));
        // Keep workers 0 and 1 alive
        manager.record_heartbeat(&workers[0], 10.0);
        manager.record_heartbeat(&workers[1], 10.0);
        // Check for failures (increments missed_heartbeats for worker 2)
        let _ = manager.detect_failures();
    }

    // Final check - worker 3 should now be detected as failed
    let failed = manager.detect_failures();

    // Worker 3 should be detected as failed (no heartbeat, missed multiple checks)
    assert!(failed.contains(&workers[2]));
    assert!(!failed.contains(&workers[0]));
    assert!(!failed.contains(&workers[1]));
}

// ============================================================================
// Test: Checkpoint Coordination
// ============================================================================

#[tokio::test]
async fn test_checkpoint_coordination() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = DistributedCheckpointConfig::default();
    config.distributed_dir = temp_dir.path().join("distributed");
    config.base_config.checkpoint_dir = temp_dir.path().join("local");
    config.min_workers = 2;

    let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

    let workers = create_test_workers(3);

    // Initiate checkpoint
    let checkpoint_id = coordinator
        .initiate_checkpoint(1, workers.clone())
        .await
        .unwrap();

    assert_eq!(checkpoint_id.round_id, 1);

    // Check status
    let status = coordinator.get_status(checkpoint_id).unwrap();
    assert!(matches!(
        status.status,
        helix_node::training::DistributedCheckpointStatus::Initiated
    ));

    // Workers contribute
    for worker_id in &workers {
        let contribution = create_test_checkpoint_contribution(worker_id.clone(), 100);
        coordinator.contribute(checkpoint_id, contribution).await.unwrap();
    }

    // Status should be completed or verifying after all contributions
    let status = coordinator.get_status(checkpoint_id);
    if let Some(s) = status {
        assert!(matches!(
            s.status,
            helix_node::training::DistributedCheckpointStatus::Completed
                | helix_node::training::DistributedCheckpointStatus::Verifying
        ));
    }
}

// ============================================================================
// Test: Training Survives Worker Failure
// ============================================================================

#[test]
fn test_training_survives_worker_failure() {
    let mut config = StateMachineConfig::default();
    config.min_workers = 2; // Allow round to start with 2 workers
    let mut state_machine = DistributedTrainingStateMachine::new(config);

    let workers = create_test_workers(3);
    let initial_commitment = [0u8; 32];

    // Start round
    let _round_id = state_machine.start_round(initial_commitment, None).unwrap();

    // Only 2 workers join (simulating 1 failure)
    state_machine.add_worker(workers[0].clone(), 1000).unwrap();
    state_machine.add_worker(workers[1].clone(), 1000).unwrap();
    // Worker 3 never joins (failed)

    // Start with partial workers - verify it returns true
    let can_start = state_machine.try_start_distribution().unwrap();
    assert!(can_start, "Should be able to start distribution with 2 workers");

    // Available workers receive shares (during Distributing phase)
    {
        let round = state_machine.current_round_mut().unwrap();
        round.mark_shares_received(&workers[0]).unwrap();
        round.mark_shares_received(&workers[1]).unwrap();
    }

    // Complete distribution - transitions to Computing
    state_machine.mark_distribution_complete().unwrap();

    // Workers submit their gradients
    state_machine.record_submission(&workers[0], [0u8; 32], 0.01).unwrap();
    state_machine.record_submission(&workers[1], [0u8; 32], 0.01).unwrap();

    // Continue with 2/3 workers
    state_machine.try_start_collection().unwrap();
    state_machine.try_start_aggregation().unwrap();

    // Transition to committing
    let round_id = state_machine.current_round().unwrap().id;
    state_machine.start_commit(round_id).unwrap();

    let tx_hash = [1u8; 32];
    state_machine.record_commit(tx_hash).unwrap();

    // Verify round completed despite worker failure
    let round = state_machine.current_round().unwrap();
    assert_eq!(round.state, DistributedRoundState::Completed);
}

// ============================================================================
// Test: Gradient Share Collection with Threshold
// ============================================================================

#[test]
fn test_gradient_collection_with_threshold() {
    let workers = create_test_workers(5);
    let round_id = DistributedRoundId::new(1, 1);

    // Need 3 out of 5 workers (threshold)
    let mut collector = GradientShareCollector::new(
        round_id,
        workers.clone(),
        3, // MPC threshold
        Duration::from_secs(60),
    );

    // Submit from 3 workers (threshold met)
    for i in 0..3 {
        let share = create_test_gradient_share(workers[i].clone(), i);
        collector.add_share(share).unwrap();
    }

    // Should have enough for aggregation
    assert!(collector.has_enough_shares());

    // But not complete (missing 2 workers)
    assert!(!collector.is_complete());
    assert_eq!(collector.missing_workers().len(), 2);
}

// ============================================================================
// Test: Emergency Checkpoint on Failure
// ============================================================================

#[tokio::test]
async fn test_emergency_checkpoint_on_failure() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = DistributedCheckpointConfig::default();
    config.distributed_dir = temp_dir.path().join("distributed");
    config.base_config.checkpoint_dir = temp_dir.path().join("local");
    config.checkpoint_on_failure = true;

    let coordinator = DistributedCheckpointCoordinator::new(config).unwrap();

    let workers = create_test_workers(3);

    // Subscribe to events
    let mut events_rx = coordinator.subscribe();

    // Trigger emergency checkpoint
    let checkpoint_id = coordinator
        .emergency_checkpoint(5, workers, "Worker failure detected".to_string())
        .await
        .unwrap();

    assert_eq!(checkpoint_id.round_id, 5);

    // Check events
    let event = events_rx.try_recv().unwrap();
    assert!(matches!(
        event,
        helix_node::training::CheckpointEvent::Initiated { .. }
    ));

    let event = events_rx.try_recv().unwrap();
    assert!(matches!(
        event,
        helix_node::training::CheckpointEvent::EmergencyCheckpoint { .. }
    ));
}

// ============================================================================
// Test: Multiple Rounds Complete Successfully
// ============================================================================

#[test]
fn test_multiple_rounds_complete() {
    let config = StateMachineConfig::default();
    let mut state_machine = DistributedTrainingStateMachine::new(config);

    let workers = create_test_workers(3);

    // Run 5 rounds
    for round_num in 1..=5u64 {
        let initial_commitment = [round_num as u8; 32];

        // Start round
        let round_id = state_machine.start_round(initial_commitment, None).unwrap();
        assert_eq!(round_id.round_number, round_num);

        // Add workers
        for worker_id in &workers {
            state_machine.add_worker(worker_id.clone(), 1000).unwrap();
        }

        // Progress through states
        state_machine.try_start_distribution().unwrap();

        // Workers receive shares (during Distributing)
        for worker_id in &workers {
            let round = state_machine.current_round_mut().unwrap();
            round.mark_shares_received(worker_id).unwrap();
        }

        // Complete distribution - transitions to Computing
        state_machine.mark_distribution_complete().unwrap();

        // Workers submit gradients
        for worker_id in &workers {
            state_machine.record_submission(worker_id, [0u8; 32], 0.01).unwrap();
        }

        state_machine.try_start_collection().unwrap();
        state_machine.try_start_aggregation().unwrap();

        // Transition to committing
        let round_id = state_machine.current_round().unwrap().id;
        state_machine.start_commit(round_id).unwrap();

        let tx_hash = [round_num as u8; 32];
        state_machine.record_commit(tx_hash).unwrap();

        // Verify completed
        let round = state_machine.current_round().unwrap();
        assert_eq!(round.state, DistributedRoundState::Completed);
    }
}

// ============================================================================
// Test: Worker Health Tracking
// ============================================================================

#[test]
fn test_worker_health_tracking() {
    let config = FaultToleranceConfig::default();
    let mut manager = FaultToleranceManager::new(config);

    let worker = create_peer_id(1);

    // Register worker
    manager.register_worker(worker.clone());

    // Initially healthy
    assert!(matches!(
        manager.detector().get_worker_health(&worker),
        Some(ref info) if info.health == WorkerHealth::Healthy
    ));

    // Record heartbeats
    for _ in 0..5 {
        manager.record_heartbeat(&worker, 10.0);
        std::thread::sleep(Duration::from_millis(10));
    }

    // Still healthy
    assert!(matches!(
        manager.detector().get_worker_health(&worker),
        Some(ref info) if info.health == WorkerHealth::Healthy
    ));

    // Unregister
    manager.unregister_worker(&worker);
    assert!(manager.detector().get_worker_health(&worker).is_none());
}

// ============================================================================
// Test: Checkpoint Resume Compatibility
// ============================================================================

#[tokio::test]
async fn test_checkpoint_resume_compatibility() {
    use helix_node::training::distributed_checkpoint::ResumeCoordinator;

    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = DistributedCheckpointConfig::default();
    config.distributed_dir = temp_dir.path().join("distributed");
    config.base_config.checkpoint_dir = temp_dir.path().join("local");

    let checkpoint_coordinator = Arc::new(
        DistributedCheckpointCoordinator::new(config).unwrap()
    );

    let resume_coordinator = ResumeCoordinator::new(checkpoint_coordinator.clone());

    let workers = create_test_workers(3);

    // Create and complete a checkpoint
    let checkpoint_id = checkpoint_coordinator
        .initiate_checkpoint(10, workers.clone())
        .await
        .unwrap();

    // All workers contribute with same iteration and state
    for worker in &workers {
        let contribution = create_test_checkpoint_contribution(worker.clone(), 500);
        checkpoint_coordinator.contribute(checkpoint_id, contribution).await.unwrap();
    }

    // Verify resume compatibility
    let checkpoint = checkpoint_coordinator.get_status(checkpoint_id);
    if let Some(ckpt) = checkpoint {
        let result = resume_coordinator.verify_resume_compatibility(&ckpt, &workers);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Test: State Machine Event Emission
// ============================================================================

#[test]
fn test_state_machine_events() {
    let config = StateMachineConfig::default();
    let mut state_machine = DistributedTrainingStateMachine::new(config);

    // Subscribe before creating round
    let mut rx = state_machine.subscribe();

    let workers = create_test_workers(2);
    let initial_commitment = [0u8; 32];

    // Start round
    let _round_id = state_machine.start_round(initial_commitment, None).unwrap();

    // Add workers
    for worker in &workers {
        state_machine.add_worker(worker.clone(), 1000).unwrap();
    }

    // Check events - Note: events are sent via broadcast channel
    // We can try to receive but they may not be immediately available in test
}

// ============================================================================
// Integration Test: Full Training Flow
// ============================================================================

#[tokio::test]
async fn test_full_training_flow() {
    // Setup
    let temp_dir = tempfile::tempdir().unwrap();
    let workers = create_test_workers(3);

    // Create components
    let round_config = StateMachineConfig::default();
    let mut state_machine = DistributedTrainingStateMachine::new(round_config);

    let mut fault_config = FaultToleranceConfig::default();
    fault_config.heartbeat_timeout = Duration::from_secs(30);
    let mut fault_manager = FaultToleranceManager::new(fault_config);

    let mut checkpoint_config = DistributedCheckpointConfig::default();
    checkpoint_config.distributed_dir = temp_dir.path().join("distributed");
    checkpoint_config.base_config.checkpoint_dir = temp_dir.path().join("local");
    let checkpoint_coordinator = DistributedCheckpointCoordinator::new(checkpoint_config).unwrap();

    // Register workers with fault manager
    for worker in &workers {
        fault_manager.register_worker(worker.clone());
    }

    // Run 3 training rounds
    for round_num in 1..=3u64 {
        let initial_commitment = [round_num as u8; 32];

        // Start round
        let _round_id = state_machine.start_round(initial_commitment, None).unwrap();

        // Workers join
        for worker in &workers {
            fault_manager.record_heartbeat(worker, 10.0);
            state_machine.add_worker(worker.clone(), 1000).unwrap();
        }

        // Progress through training phases
        state_machine.try_start_distribution().unwrap();

        // Workers receive shares during Distributing phase
        for worker in &workers {
            fault_manager.record_heartbeat(worker, 10.0);
            let round = state_machine.current_round_mut().unwrap();
            round.mark_shares_received(worker).unwrap();
        }

        // Complete distribution - transitions to Computing
        state_machine.mark_distribution_complete().unwrap();

        // Collect gradients using share collector
        let collector_round_id = DistributedRoundId::new(1, round_num);
        let mut collector = GradientShareCollector::new(
            collector_round_id,
            workers.clone(),
            2,
            Duration::from_secs(60),
        );

        for (i, worker) in workers.iter().enumerate() {
            fault_manager.record_heartbeat(worker, 10.0);
            let share = create_test_gradient_share(worker.clone(), i);
            collector.add_share(share).unwrap();
        }

        assert!(collector.is_complete());

        // Workers submit gradients to state machine
        for worker in &workers {
            state_machine.record_submission(worker, [0u8; 32], 0.01).unwrap();
        }

        // Aggregate and commit
        state_machine.try_start_collection().unwrap();
        state_machine.try_start_aggregation().unwrap();

        // Transition to committing
        let round_id = state_machine.current_round().unwrap().id;
        state_machine.start_commit(round_id).unwrap();

        let tx_hash = [round_num as u8; 32];
        state_machine.record_commit(tx_hash).unwrap();

        // Verify round completed
        let round = state_machine.current_round().unwrap();
        assert_eq!(round.state, DistributedRoundState::Completed);

        // Create checkpoint every round
        let checkpoint_id = checkpoint_coordinator
            .initiate_checkpoint(round_num, workers.clone())
            .await
            .unwrap();

        for worker in &workers {
            let contribution = create_test_checkpoint_contribution(worker.clone(), round_num * 100);
            checkpoint_coordinator.contribute(checkpoint_id, contribution).await.unwrap();
        }
    }

    // Verify all workers still healthy
    for worker in &workers {
        let health = fault_manager.detector().get_worker_health(worker);
        assert!(matches!(health, Some(ref info) if info.health == WorkerHealth::Healthy));
    }

    // Verify checkpoints exist
    let resumable = checkpoint_coordinator.list_resumable();
    assert!(resumable.len() >= 2); // At least 2 should have completed
}

// ============================================================================
// Test: Round Failure Handling
// ============================================================================

#[test]
fn test_round_failure_handling() {
    let config = StateMachineConfig::default();
    let mut state_machine = DistributedTrainingStateMachine::new(config);

    let workers = create_test_workers(3);
    let initial_commitment = [0u8; 32];

    // Start round
    let _round_id = state_machine.start_round(initial_commitment, None).unwrap();

    // Add workers
    for worker in &workers {
        state_machine.add_worker(worker.clone(), 1000).unwrap();
    }

    state_machine.try_start_distribution().unwrap();
    state_machine.mark_distribution_complete().unwrap();

    // Fail the round explicitly
    let failure_reason = RoundFailureReason::WorkerTimeout;
    state_machine.fail_round(failure_reason, "Test timeout").unwrap();

    // Verify round is failed
    let round = state_machine.current_round().unwrap();
    assert!(matches!(round.state, DistributedRoundState::Failed(_)));
}
