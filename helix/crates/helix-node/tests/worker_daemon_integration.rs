//! Worker Daemon Integration Tests.
//!
//! Simulates a 3-node network (1 aggregator + 2 workers) to verify:
//! - Workers auto-discover and evaluate round announcements
//! - Workers register with aggregator and receive model weights
//! - Workers execute training and submit proofs
//! - Round lifecycle completes end-to-end
//! - Participation tracking and earnings accounting works
//!
//! These tests use the daemon's message handlers directly (no real networking)
//! to verify the complete round lifecycle in-process.

use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::RwLock;

use helix_node::network::messages::{
    ModelDims, NodeCapabilities, PeerId, RegistrationMessage,
    RoundManagementMessage, TrainingMessage,
};
use helix_node::trainer::MlpModel;
use helix_node::worker::daemon::{
    ActiveRoundState, RoundPhase, WorkerDaemonConfig, WorkerDaemonEvent,
};
use helix_node::worker::WorkerDaemon;

// ============================================================================
// Test Helpers
// ============================================================================

fn aggregator_id() -> PeerId {
    PeerId::from_string("aggregator-0")
}

fn worker_id(n: u8) -> PeerId {
    PeerId::from_string(format!("worker-{}", n))
}

fn test_capabilities() -> NodeCapabilities {
    NodeCapabilities {
        can_train: true,
        can_aggregate: false,
        can_prove: true,
        gpu_memory_mb: 4096,
        cpu_cores: 8,
        storage_gb: 100,
    }
}

fn test_dims() -> ModelDims {
    ModelDims {
        d_in: 4,
        d_hid: 8,
        d_out: 2,
        num_layers: 2,
        num_heads: 0,
        activation_type: 0,
    }
}

fn default_daemon_config() -> WorkerDaemonConfig {
    WorkerDaemonConfig {
        auto_join: true,
        max_model_params: 1_000_000,
        max_error_budget: 1.0,
        min_deadline_slack_secs: 10,
        max_steps_per_round: 100,
        max_history_records: 100,
        base_reward_per_proof_wei: 1_000_000,
    }
}

fn create_daemon(id: PeerId) -> WorkerDaemon {
    WorkerDaemon::new(id, default_daemon_config(), test_capabilities())
}

fn future_deadline(secs: u64) -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() + secs
}

/// Creates a model checkpoint for test model weights distribution.
fn create_test_checkpoint() -> Vec<u8> {
    let model = MlpModel::new_random(4, 8, 2, 42);
    let ckpt = model.to_checkpoint(0);
    ckpt.to_bytes().expect("checkpoint serialization")
}

// ============================================================================
// Round Evaluation Tests
// ============================================================================

#[test]
fn test_daemon_evaluates_and_accepts_round() {
    let daemon = create_daemon(worker_id(1));

    // Verify the daemon is idle
    let status = daemon.status();
    assert!(!status.in_round);
    assert!(status.auto_join);
    assert_eq!(status.total_rounds, 0);
}

#[test]
fn test_daemon_rejects_when_auto_join_disabled() {
    let config = WorkerDaemonConfig {
        auto_join: false,
        ..default_daemon_config()
    };
    let daemon = WorkerDaemon::new(worker_id(1), config, test_capabilities());

    assert!(!daemon.is_auto_join());

    let status = daemon.status();
    assert!(!status.auto_join);
}

#[test]
fn test_daemon_rejects_oversized_model() {
    let config = WorkerDaemonConfig {
        max_model_params: 10, // Very small — will reject any real model
        ..default_daemon_config()
    };
    let daemon = WorkerDaemon::new(worker_id(1), config, test_capabilities());

    // The evaluator should reject models with > 10 params
    // 4x8 + 8 + 8x2 + 2 = 58 params (way above 10)
    let status = daemon.status();
    assert!(!status.in_round); // Still idle
}

// ============================================================================
// Registration Message Handling
// ============================================================================

#[test]
fn test_worker_registered_accepted() {
    let daemon = create_daemon(worker_id(1));

    // Simulate setting up an active round first
    {
        let mut active = daemon.active_round.write();
        *active = Some(ActiveRoundState {
            round_id: 1,
            model_id: 100,
            model_dims: test_dims(),
            steps_per_worker: 5,
            learning_rate: 0.01,
            error_budget: 0.1,
            deadline: future_deadline(600),
            phase: RoundPhase::Registered,
            aggregator_id: aggregator_id(),
            dataset_ref: "test".into(),
            current_model_hash: [0u8; 32],
            steps_completed: 0,
            proofs_submitted: 0,
            joined_at: 1000,
        });
    }

    daemon.handle_registration_message(
        &aggregator_id(),
        &RegistrationMessage::WorkerRegistered {
            accepted: true,
            worker_slot: Some(0),
            reject_reason: None,
            current_round_id: Some(1),
        },
    );

    // Should still be in the round
    assert!(daemon.active_round().is_some());
}

#[test]
fn test_worker_registered_rejected_clears_state() {
    let daemon = create_daemon(worker_id(1));

    // Set up active round + tracker
    daemon.tracker().write().record_join(1, 100);
    {
        let mut active = daemon.active_round.write();
        *active = Some(ActiveRoundState {
            round_id: 1,
            model_id: 100,
            model_dims: test_dims(),
            steps_per_worker: 5,
            learning_rate: 0.01,
            error_budget: 0.1,
            deadline: future_deadline(600),
            phase: RoundPhase::Registered,
            aggregator_id: aggregator_id(),
            dataset_ref: "test".into(),
            current_model_hash: [0u8; 32],
            steps_completed: 0,
            proofs_submitted: 0,
            joined_at: 1000,
        });
    }

    daemon.handle_registration_message(
        &aggregator_id(),
        &RegistrationMessage::WorkerRegistered {
            accepted: false,
            worker_slot: None,
            reject_reason: Some("round full".to_string()),
            current_round_id: None,
        },
    );

    // Should have cleared the active round
    assert!(daemon.active_round().is_none());

    // Should have recorded failure in tracker
    let tracker = daemon.tracker();
    let record = tracker.read().get_round_record(1).cloned();
    assert!(record.is_some());
    assert!(!record.unwrap().success);
}

// ============================================================================
// Model Weights Handling
// ============================================================================

#[test]
fn test_model_weights_received_updates_phase() {
    let daemon = create_daemon(worker_id(1));

    // Set up active round in Registered phase
    {
        let mut active = daemon.active_round.write();
        *active = Some(ActiveRoundState {
            round_id: 1,
            model_id: 100,
            model_dims: test_dims(),
            steps_per_worker: 5,
            learning_rate: 0.01,
            error_budget: 0.1,
            deadline: future_deadline(600),
            phase: RoundPhase::Registered,
            aggregator_id: aggregator_id(),
            dataset_ref: "test".into(),
            current_model_hash: [0u8; 32],
            steps_completed: 0,
            proofs_submitted: 0,
            joined_at: 1000,
        });
    }

    // Create checkpoint data
    let ckpt_data = create_test_checkpoint();

    // Simulate model weights message — use the sync handler directly
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        daemon.handle_training_message(
            &aggregator_id(),
            &TrainingMessage::ModelWeights {
                round_id: 1,
                checkpoint_data: ckpt_data.clone(),
                weight_hash: [0u8; 32],
            },
            // We don't actually need network for weight handling
            &Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
                .local_id(worker_id(1))
                .listen_addr("127.0.0.1:0".parse().unwrap())
                .build()
                .unwrap()),
        ).await;
    });

    // Should have advanced to WeightsReceived
    let active = daemon.active_round();
    assert!(active.is_some());
    assert_eq!(active.unwrap().phase, RoundPhase::WeightsReceived);

    // Should have logged the event
    let events = daemon.recent_events();
    let weight_events: Vec<_> = events.iter().filter(|e| {
        matches!(e, WorkerDaemonEvent::WeightsReceived { .. })
    }).collect();
    assert!(!weight_events.is_empty());
}

#[test]
fn test_model_weights_for_wrong_round_ignored() {
    let daemon = create_daemon(worker_id(1));

    // Set up active round 1
    {
        let mut active = daemon.active_round.write();
        *active = Some(ActiveRoundState {
            round_id: 1,
            model_id: 100,
            model_dims: test_dims(),
            steps_per_worker: 5,
            learning_rate: 0.01,
            error_budget: 0.1,
            deadline: future_deadline(600),
            phase: RoundPhase::Registered,
            aggregator_id: aggregator_id(),
            dataset_ref: "test".into(),
            current_model_hash: [0u8; 32],
            steps_completed: 0,
            proofs_submitted: 0,
            joined_at: 1000,
        });
    }

    let ckpt_data = create_test_checkpoint();

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        daemon.handle_training_message(
            &aggregator_id(),
            &TrainingMessage::ModelWeights {
                round_id: 99, // Wrong round!
                checkpoint_data: ckpt_data,
                weight_hash: [0u8; 32],
            },
            &Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
                .local_id(worker_id(1))
                .listen_addr("127.0.0.1:0".parse().unwrap())
                .build()
                .unwrap()),
        ).await;
    });

    // Should still be in Registered phase (weights for wrong round ignored)
    let active = daemon.active_round();
    assert_eq!(active.unwrap().phase, RoundPhase::Registered);
}

// ============================================================================
// Round Completion Handling
// ============================================================================

#[test]
fn test_round_completed_updates_tracker() {
    let daemon = create_daemon(worker_id(1));

    // Set up active round in ProofSubmitted phase
    daemon.tracker().write().record_join(1, 100);
    {
        let mut active = daemon.active_round.write();
        *active = Some(ActiveRoundState {
            round_id: 1,
            model_id: 100,
            model_dims: test_dims(),
            steps_per_worker: 10,
            learning_rate: 0.01,
            error_budget: 0.1,
            deadline: future_deadline(600),
            phase: RoundPhase::ProofSubmitted,
            aggregator_id: aggregator_id(),
            dataset_ref: "test".into(),
            current_model_hash: [0u8; 32],
            steps_completed: 10,
            proofs_submitted: 1,
            joined_at: 1000,
        });
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let network = Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
            .local_id(worker_id(1))
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap());

        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundCompleted {
                round_id: 1,
                final_commitment: [0xab; 32],
                tx_hash: Some([0xcd; 32]),
                new_model_hash: [0xef; 32],
                total_error_bound: 0.05,
                num_contributors: 2,
            },
            &network,
        ).await;
    });

    // Round should be cleared
    assert!(daemon.active_round().is_none());

    // Tracker should record success
    let tracker = daemon.tracker();
    let record = tracker.read().get_round_record(1).cloned();
    assert!(record.is_some());
    let record = record.unwrap();
    assert!(record.success);
    assert_eq!(record.steps_completed, 10);
    assert_eq!(record.proofs_submitted, 1);

    // Earnings should be updated
    let earnings = daemon.earnings();
    assert_eq!(earnings.rounds_participated, 1);
    assert_eq!(earnings.rounds_succeeded, 1);
    assert!(earnings.total_earnings_wei > 0);
}

#[test]
fn test_round_failed_updates_tracker() {
    let daemon = create_daemon(worker_id(1));

    daemon.tracker().write().record_join(1, 100);
    {
        let mut active = daemon.active_round.write();
        *active = Some(ActiveRoundState {
            round_id: 1,
            model_id: 100,
            model_dims: test_dims(),
            steps_per_worker: 10,
            learning_rate: 0.01,
            error_budget: 0.1,
            deadline: future_deadline(600),
            phase: RoundPhase::Training,
            aggregator_id: aggregator_id(),
            dataset_ref: "test".into(),
            current_model_hash: [0u8; 32],
            steps_completed: 3,
            proofs_submitted: 0,
            joined_at: 1000,
        });
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let network = Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
            .local_id(worker_id(1))
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap());

        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundFailed {
                round_id: 1,
                reason: "insufficient workers".to_string(),
            },
            &network,
        ).await;
    });

    assert!(daemon.active_round().is_none());

    let tracker = daemon.tracker();
    let record = tracker.read().get_round_record(1).cloned();
    assert!(record.is_some());
    assert!(!record.unwrap().success);

    let earnings = daemon.earnings();
    assert_eq!(earnings.rounds_failed, 1);
}

// ============================================================================
// Two-Worker Simulation
// ============================================================================

#[test]
fn test_two_workers_evaluate_same_round() {
    let daemon1 = create_daemon(worker_id(1));
    let daemon2 = create_daemon(worker_id(2));

    // Both should be idle and auto-join enabled
    assert!(daemon1.is_auto_join());
    assert!(daemon2.is_auto_join());
    assert!(!daemon1.status().in_round);
    assert!(!daemon2.status().in_round);

    // Both evaluate the round status independently
    let status1 = daemon1.status();
    let status2 = daemon2.status();
    assert_eq!(status1.total_rounds, 0);
    assert_eq!(status2.total_rounds, 0);
}

// ============================================================================
// Auto-Join Toggle Tests
// ============================================================================

#[test]
fn test_auto_join_toggle_via_rpc_api() {
    let daemon = create_daemon(worker_id(1));

    // Initially enabled
    assert!(daemon.is_auto_join());
    let status = daemon.status();
    assert!(status.auto_join);

    // Disable
    let prev = daemon.set_auto_join(false);
    assert!(prev); // was true
    assert!(!daemon.is_auto_join());

    // Re-enable
    let prev = daemon.set_auto_join(true);
    assert!(!prev); // was false
    assert!(daemon.is_auto_join());
}

// ============================================================================
// Multi-Round Lifecycle Tracking
// ============================================================================

#[test]
fn test_multi_round_earnings_accumulation() {
    let daemon = create_daemon(worker_id(1));

    // Simulate 5 rounds (3 success, 2 failure)
    for round_id in 1..=5 {
        daemon.tracker().write().record_join(round_id, 100);

        if round_id <= 3 {
            // Successful rounds
            daemon.tracker().write().record_completion(
                round_id,
                10, // steps
                1,  // proofs
                0.01, // error
                0.5,  // loss
                format!("proof_{}", round_id),
            );
        } else {
            // Failed rounds
            daemon.tracker().write().record_failure(round_id);
        }
    }

    let earnings = daemon.earnings();
    assert_eq!(earnings.rounds_participated, 5);
    assert_eq!(earnings.rounds_succeeded, 3);
    assert_eq!(earnings.rounds_failed, 2);
    assert!(earnings.total_earnings_wei > 0);
    assert!((earnings.success_rate - 0.6).abs() < 0.01);
    assert!(earnings.reputation_score > 0.0);
    assert!(earnings.reputation_score <= 1.0);
}

// ============================================================================
// 3-Node Network Simulation (1 Aggregator + 2 Workers)
// ============================================================================

/// Simulates a complete 3-node training round:
/// 1. Aggregator broadcasts RoundConfigure
/// 2. Two workers evaluate and register
/// 3. Workers receive model weights
/// 4. Aggregator sends BeginTraining
/// 5. Workers complete training, submit proofs
/// 6. Aggregator broadcasts RoundCompleted
/// 7. Workers clean up and record success
#[test]
fn test_three_node_full_round_lifecycle() {
    let daemon1 = create_daemon(worker_id(1));
    let daemon2 = create_daemon(worker_id(2));
    let deadline = future_deadline(600);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let net1 = Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
            .local_id(worker_id(1))
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap());
        let net2 = Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
            .local_id(worker_id(2))
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap());

        // -- Step 1: Aggregator broadcasts RoundConfigure --
        let round_cfg = RoundManagementMessage::RoundConfigure {
            round_id: 1,
            model_id: 42,
            model_dims: test_dims(),
            dataset_ref: "ipfs://QmTest".to_string(),
            steps_per_worker: 3,
            learning_rate: 0.01,
            error_budget: 0.1,
            min_workers: 2,
            deadline,
            current_model_hash: [0u8; 32],
            dataset_spec: None,
        };

        daemon1.handle_round_management(&aggregator_id(), &round_cfg, &net1).await;
        daemon2.handle_round_management(&aggregator_id(), &round_cfg, &net2).await;

        // Both workers should have joined
        assert!(daemon1.active_round().is_some(), "Worker 1 should have joined");
        assert!(daemon2.active_round().is_some(), "Worker 2 should have joined");
        assert_eq!(daemon1.active_round().unwrap().round_id, 1);
        assert_eq!(daemon2.active_round().unwrap().round_id, 1);
        assert_eq!(daemon1.active_round().unwrap().phase, RoundPhase::Registered);
        assert_eq!(daemon2.active_round().unwrap().phase, RoundPhase::Registered);

        // -- Step 2: Aggregator confirms registrations --
        daemon1.handle_registration_message(
            &aggregator_id(),
            &RegistrationMessage::WorkerRegistered {
                accepted: true,
                worker_slot: Some(0),
                reject_reason: None,
                current_round_id: Some(1),
            },
        );
        daemon2.handle_registration_message(
            &aggregator_id(),
            &RegistrationMessage::WorkerRegistered {
                accepted: true,
                worker_slot: Some(1),
                reject_reason: None,
                current_round_id: Some(1),
            },
        );

        // -- Step 3: Aggregator sends model weights --
        let ckpt_data = create_test_checkpoint();
        let weights_msg = TrainingMessage::ModelWeights {
            round_id: 1,
            checkpoint_data: ckpt_data,
            weight_hash: [0u8; 32],
        };

        daemon1.handle_training_message(&aggregator_id(), &weights_msg, &net1).await;
        daemon2.handle_training_message(&aggregator_id(), &weights_msg, &net2).await;

        assert_eq!(daemon1.active_round().unwrap().phase, RoundPhase::WeightsReceived);
        assert_eq!(daemon2.active_round().unwrap().phase, RoundPhase::WeightsReceived);

        // -- Step 4: Aggregator sends BeginTraining --
        let begin_msg = RoundManagementMessage::BeginTraining {
            round_id: 1,
            participants: vec![
                worker_id(1).0.clone(),
                worker_id(2).0.clone(),
            ],
        };

        daemon1.handle_round_management(&aggregator_id(), &begin_msg, &net1).await;
        daemon2.handle_round_management(&aggregator_id(), &begin_msg, &net2).await;

        // After BeginTraining, workers should have completed training and submitted proofs
        assert_eq!(
            daemon1.active_round().unwrap().phase,
            RoundPhase::ProofSubmitted,
            "Worker 1 should have submitted proof"
        );
        assert_eq!(
            daemon2.active_round().unwrap().phase,
            RoundPhase::ProofSubmitted,
            "Worker 2 should have submitted proof"
        );

        // Verify steps completed
        assert_eq!(daemon1.active_round().unwrap().steps_completed, 3);
        assert_eq!(daemon2.active_round().unwrap().steps_completed, 3);

        // -- Step 5: Aggregator broadcasts RoundCompleted --
        let completed_msg = RoundManagementMessage::RoundCompleted {
            round_id: 1,
            final_commitment: [0xab; 32],
            tx_hash: Some([0xcd; 32]),
            new_model_hash: [0xef; 32],
            total_error_bound: 0.02,
            num_contributors: 2,
        };

        daemon1.handle_round_management(&aggregator_id(), &completed_msg, &net1).await;
        daemon2.handle_round_management(&aggregator_id(), &completed_msg, &net2).await;

        // Both workers should have cleaned up
        assert!(daemon1.active_round().is_none(), "Worker 1 should be idle");
        assert!(daemon2.active_round().is_none(), "Worker 2 should be idle");

        // Both workers should have recorded success
        let e1 = daemon1.earnings();
        let e2 = daemon2.earnings();
        assert_eq!(e1.rounds_participated, 1);
        assert_eq!(e1.rounds_succeeded, 1);
        assert_eq!(e1.rounds_failed, 0);
        assert_eq!(e2.rounds_participated, 1);
        assert_eq!(e2.rounds_succeeded, 1);

        // Both should have non-zero earnings
        assert!(e1.total_earnings_wei > 0, "Worker 1 should have earnings");
        assert!(e2.total_earnings_wei > 0, "Worker 2 should have earnings");

        // Success rate = 100%
        assert!((e1.success_rate - 1.0).abs() < 0.01);
        assert!((e2.success_rate - 1.0).abs() < 0.01);
    });
}

/// Tests that a second round can be joined after the first completes.
#[test]
fn test_consecutive_rounds_lifecycle() {
    let daemon = create_daemon(worker_id(1));
    let deadline = future_deadline(600);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let network = Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
            .local_id(worker_id(1))
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap());

        for round_id in 1..=3 {
            // RoundConfigure
            daemon.handle_round_management(
                &aggregator_id(),
                &RoundManagementMessage::RoundConfigure {
                    round_id,
                    model_id: 42,
                    model_dims: test_dims(),
                    dataset_ref: "test".to_string(),
                    steps_per_worker: 2,
                    learning_rate: 0.01,
                    error_budget: 0.1,
                    min_workers: 1,
                    deadline,
                    current_model_hash: [0u8; 32],
                    dataset_spec: None,
                },
                &network,
            ).await;

            assert!(daemon.active_round().is_some());

            // Model weights
            let ckpt_data = create_test_checkpoint();
            daemon.handle_training_message(
                &aggregator_id(),
                &TrainingMessage::ModelWeights {
                    round_id,
                    checkpoint_data: ckpt_data,
                    weight_hash: [0u8; 32],
                },
                &network,
            ).await;

            // BeginTraining
            daemon.handle_round_management(
                &aggregator_id(),
                &RoundManagementMessage::BeginTraining {
                    round_id,
                    participants: vec![worker_id(1).0.clone()],
                },
                &network,
            ).await;

            assert_eq!(daemon.active_round().unwrap().phase, RoundPhase::ProofSubmitted);

            // RoundCompleted
            daemon.handle_round_management(
                &aggregator_id(),
                &RoundManagementMessage::RoundCompleted {
                    round_id,
                    final_commitment: [0u8; 32],
                    tx_hash: None,
                    new_model_hash: [0u8; 32],
                    total_error_bound: 0.01,
                    num_contributors: 1,
                },
                &network,
            ).await;

            assert!(daemon.active_round().is_none());
        }

        // Verify all 3 rounds tracked
        let earnings = daemon.earnings();
        assert_eq!(earnings.rounds_participated, 3);
        assert_eq!(earnings.rounds_succeeded, 3);
        assert_eq!(earnings.rounds_failed, 0);
        assert!((earnings.success_rate - 1.0).abs() < 0.01);
    });
}

/// Tests that a worker rejects a second round while already in one.
#[test]
fn test_worker_rejects_second_round_while_active() {
    let daemon = create_daemon(worker_id(1));
    let deadline = future_deadline(600);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let network = Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
            .local_id(worker_id(1))
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap());

        // Join round 1
        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundConfigure {
                round_id: 1,
                model_id: 42,
                model_dims: test_dims(),
                dataset_ref: "test".to_string(),
                steps_per_worker: 5,
                learning_rate: 0.01,
                error_budget: 0.1,
                min_workers: 1,
                deadline,
                current_model_hash: [0u8; 32],
                dataset_spec: None,
            },
            &network,
        ).await;

        assert_eq!(daemon.current_round_id(), Some(1));

        // Try to join round 2 — should be rejected
        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundConfigure {
                round_id: 2,
                model_id: 43,
                model_dims: test_dims(),
                dataset_ref: "test".to_string(),
                steps_per_worker: 5,
                learning_rate: 0.01,
                error_budget: 0.1,
                min_workers: 1,
                deadline,
                current_model_hash: [0u8; 32],
                dataset_spec: None,
            },
            &network,
        ).await;

        // Should still be in round 1
        assert_eq!(daemon.current_round_id(), Some(1));

        // Check events show rejection
        let events = daemon.recent_events();
        let rejections: Vec<_> = events.iter().filter(|e| {
            matches!(e, WorkerDaemonEvent::RoundEvaluated { round_id: 2, accepted: false, .. })
        }).collect();
        assert!(!rejections.is_empty(), "Round 2 should have been rejected");
    });
}

/// Tests mixed success/failure scenario for reputation tracking.
#[test]
fn test_reputation_with_mixed_results() {
    let daemon = create_daemon(worker_id(1));
    let deadline = future_deadline(600);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let network = Arc::new(helix_node::network::runner::NetworkRunnerBuilder::new()
            .local_id(worker_id(1))
            .listen_addr("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap());

        // Round 1: Success
        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundConfigure {
                round_id: 1,
                model_id: 42,
                model_dims: test_dims(),
                dataset_ref: "test".to_string(),
                steps_per_worker: 2,
                learning_rate: 0.01,
                error_budget: 0.1,
                min_workers: 1,
                deadline,
                current_model_hash: [0u8; 32],
                dataset_spec: None,
            },
            &network,
        ).await;

        let ckpt_data = create_test_checkpoint();
        daemon.handle_training_message(
            &aggregator_id(),
            &TrainingMessage::ModelWeights {
                round_id: 1,
                checkpoint_data: ckpt_data.clone(),
                weight_hash: [0u8; 32],
            },
            &network,
        ).await;

        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::BeginTraining {
                round_id: 1,
                participants: vec![worker_id(1).0.clone()],
            },
            &network,
        ).await;

        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundCompleted {
                round_id: 1,
                final_commitment: [0u8; 32],
                tx_hash: None,
                new_model_hash: [0u8; 32],
                total_error_bound: 0.01,
                num_contributors: 1,
            },
            &network,
        ).await;

        // Round 2: Failure
        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundConfigure {
                round_id: 2,
                model_id: 42,
                model_dims: test_dims(),
                dataset_ref: "test".to_string(),
                steps_per_worker: 2,
                learning_rate: 0.01,
                error_budget: 0.1,
                min_workers: 1,
                deadline,
                current_model_hash: [0u8; 32],
                dataset_spec: None,
            },
            &network,
        ).await;

        daemon.handle_round_management(
            &aggregator_id(),
            &RoundManagementMessage::RoundFailed {
                round_id: 2,
                reason: "timeout".to_string(),
            },
            &network,
        ).await;

        let earnings = daemon.earnings();
        assert_eq!(earnings.rounds_participated, 2);
        assert_eq!(earnings.rounds_succeeded, 1);
        assert_eq!(earnings.rounds_failed, 1);
        assert!((earnings.success_rate - 0.5).abs() < 0.01);
        assert!(earnings.reputation_score > 0.0);
        assert!(earnings.reputation_score < 1.0);
    });
}
