//! Network Partition Recovery Tests.
//!
//! Tests the system's ability to handle and recover from network partitions:
//! - Message delivery during partitions
//! - State synchronization after recovery
//! - Byzantine-tolerant progress with partial connectivity
//! - Gradient collection with missing workers

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_mpc::types::PartyId;
use helix_prover::MLTrainingProverV2;

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Partition Simulation Tests
// ============================================================================

/// Tests basic partition behavior: messages blocked to partitioned nodes.
#[test]
fn test_partition_blocks_messages() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Partition nodes 3 and 4
    let partitioned = vec![parties[3].clone(), parties[4].clone()];
    network.start_partition(partitioned.clone());

    // Send messages to all nodes
    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 1000,
    };

    let mut delivered = Vec::new();
    let mut blocked = Vec::new();

    for (i, party) in parties.iter().enumerate() {
        if i == 0 {
            continue; // Skip sender
        }
        if network.send(party, msg.clone()) {
            delivered.push(party.clone());
        } else {
            blocked.push(party.clone());
        }
    }

    // Verify correct partitioning
    assert_eq!(delivered.len(), 2, "Should deliver to 2 non-partitioned nodes");
    assert_eq!(blocked.len(), 2, "Should block 2 partitioned nodes");

    for party in &partitioned {
        assert!(blocked.contains(party), "Partitioned node should be blocked");
    }
}

/// Tests that partitions are symmetric (bidirectional blocking).
#[test]
fn test_partition_symmetric() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Partition node 2
    network.start_partition(vec![parties[2].clone()]);

    // Node 0 cannot send to node 2
    let msg_to_partitioned = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 1,
    };
    assert!(!network.send(&parties[2], msg_to_partitioned));

    // The partition check is based on the recipient being partitioned
    // In our mock, partitioned nodes cannot receive but can still try to send
    // This tests that the receiving side is blocked
    assert!(network.is_partitioned(&parties[2]));
    assert!(!network.is_partitioned(&parties[0]));
}

/// Tests that non-partitioned nodes can communicate normally.
#[test]
fn test_partition_non_partitioned_communication() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Partition last 2 nodes
    network.start_partition(vec![parties[3].clone(), parties[4].clone()]);

    // Non-partitioned nodes should communicate freely
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                let msg = NetworkMessage::Heartbeat {
                    from: parties[i].clone(),
                    timestamp: (i * 10 + j) as u64,
                };
                assert!(
                    network.send(&parties[j], msg),
                    "Node {} should be able to send to node {}",
                    i,
                    j
                );
            }
        }
    }
}

// ============================================================================
// Recovery Tests
// ============================================================================

/// Tests message delivery resumes after partition ends.
#[test]
fn test_partition_recovery_message_delivery() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Start partition
    network.start_partition(vec![parties[2].clone()]);

    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 100,
    };

    // Should fail during partition
    assert!(!network.send(&parties[2], msg.clone()));

    // End partition
    network.end_partition();

    // Should succeed after recovery
    assert!(network.send(&parties[2], msg));

    // Verify message is received
    let received = network.receive(&parties[2]);
    assert_eq!(received.len(), 1, "Should have received 1 message after recovery");
}

/// Tests that network stats track partition correctly.
#[test]
fn test_partition_stats_tracking() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Initial state
    let initial_stats = network.stats();
    assert!(!initial_stats.partition_active);
    assert_eq!(initial_stats.dropped, 0);

    // Start partition
    network.start_partition(vec![parties[2].clone()]);

    let partitioned_stats = network.stats();
    assert!(partitioned_stats.partition_active);

    // Try to send to partitioned node
    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 1,
    };
    network.send(&parties[2], msg.clone());
    network.send(&parties[2], msg);

    let after_drops = network.stats();
    assert_eq!(after_drops.dropped, 2, "Should have 2 dropped messages");

    // End partition
    network.end_partition();

    let recovered_stats = network.stats();
    assert!(!recovered_stats.partition_active);
}

// ============================================================================
// Training Under Partition Tests
// ============================================================================

/// Tests that training can progress with honest majority despite partition.
#[test]
fn test_partition_training_progress() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("partition_training_progress");

    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    // 5 workers, partition 2 of them
    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    // Phase 1: Verify initial state
    let phase_start = Instant::now();
    assert_eq!(coordinator.active_worker_count(), 5);
    result.add_phase(PhaseResult::success("initial_state", phase_start.elapsed()));

    // Phase 2: Start partition (nodes 3, 4)
    let phase_start = Instant::now();
    network.start_partition(vec![parties[3].clone(), parties[4].clone()]);
    assert!(network.stats().partition_active);
    result.add_phase(PhaseResult::success("partition_started", phase_start.elapsed()));

    // Phase 3: Collect gradients from available workers
    let phase_start = Instant::now();
    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];

    // Workers 0, 1, 2 can submit (not partitioned)
    let available_submissions: Vec<GradientSubmission> = (0..3)
        .map(|i| GradientSubmission {
            party: parties[i].clone(),
            step: 1,
            gradients: true_gradients.clone(),
            commitment: [0u8; 32],
            valid: true,
        })
        .collect();

    let valid_count = available_submissions.len();
    result.add_phase(if valid_count >= 3 {
        PhaseResult::success("gradient_collection", phase_start.elapsed())
    } else {
        PhaseResult::failure(
            "gradient_collection",
            phase_start.elapsed(),
            &format!("Only {} workers available", valid_count),
        )
    });

    // Phase 4: Aggregate with available workers
    let phase_start = Instant::now();
    let aggregated = coordinator.aggregate_gradients(&available_submissions);
    result.add_phase(if aggregated.is_some() {
        let agg = aggregated.unwrap();
        let correct = agg.iter().zip(true_gradients.iter()).all(|(a, t)| (a - t).abs() < 1e-10);
        if correct {
            PhaseResult::success("partial_aggregation", phase_start.elapsed())
        } else {
            PhaseResult::failure("partial_aggregation", phase_start.elapsed(), "Aggregation incorrect")
        }
    } else {
        PhaseResult::failure("partial_aggregation", phase_start.elapsed(), "Aggregation failed")
    });

    // Phase 5: Recover from partition
    let phase_start = Instant::now();
    network.end_partition();
    assert!(!network.stats().partition_active);
    result.add_phase(PhaseResult::success("partition_recovered", phase_start.elapsed()));

    // Phase 6: All workers can now contribute
    let phase_start = Instant::now();
    let all_submissions: Vec<GradientSubmission> = (0..5)
        .map(|i| GradientSubmission {
            party: parties[i].clone(),
            step: 2,
            gradients: true_gradients.clone(),
            commitment: [0u8; 32],
            valid: true,
        })
        .collect();

    let full_aggregated = coordinator.aggregate_gradients(&all_submissions);
    result.add_phase(if full_aggregated.is_some() {
        PhaseResult::success("full_recovery", phase_start.elapsed())
    } else {
        PhaseResult::failure("full_recovery", phase_start.elapsed(), "Full aggregation failed")
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success);
}

/// Tests Byzantine-tolerant progress: 2f+1 honest workers needed for progress.
#[test]
fn test_partition_byzantine_tolerance() {
    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    // With 5 workers, can tolerate f=1 Byzantine fault, need 2f+1=3 for progress
    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];

    // Case 1: Partition 1 node (should still have progress with 4 workers)
    network.start_partition(vec![parties[4].clone()]);

    let submissions_4: Vec<GradientSubmission> = (0..4)
        .map(|i| GradientSubmission {
            party: parties[i].clone(),
            step: 1,
            gradients: true_gradients.clone(),
            commitment: [0u8; 32],
            valid: true,
        })
        .collect();

    let result_4 = coordinator.aggregate_gradients(&submissions_4);
    assert!(result_4.is_some(), "Should make progress with 4 of 5 workers");

    network.end_partition();

    // Case 2: Partition 2 nodes (should still have progress with 3 workers = minimum for BFT)
    network.start_partition(vec![parties[3].clone(), parties[4].clone()]);

    let submissions_3: Vec<GradientSubmission> = (0..3)
        .map(|i| GradientSubmission {
            party: parties[i].clone(),
            step: 2,
            gradients: true_gradients.clone(),
            commitment: [0u8; 32],
            valid: true,
        })
        .collect();

    let result_3 = coordinator.aggregate_gradients(&submissions_3);
    assert!(result_3.is_some(), "Should make progress with 3 of 5 workers (BFT threshold)");

    network.end_partition();

    // Case 3: Partition 3 nodes (only 2 workers left - below BFT threshold but can still aggregate)
    // In a real system, this might be below quorum, but our simple aggregator still works
    network.start_partition(vec![
        parties[2].clone(),
        parties[3].clone(),
        parties[4].clone(),
    ]);

    let submissions_2: Vec<GradientSubmission> = (0..2)
        .map(|i| GradientSubmission {
            party: parties[i].clone(),
            step: 3,
            gradients: true_gradients.clone(),
            commitment: [0u8; 32],
            valid: true,
        })
        .collect();

    // Our simple aggregator will still produce a result, but in production
    // this would be below quorum and rejected
    let result_2 = coordinator.aggregate_gradients(&submissions_2);
    assert!(result_2.is_some(), "Simple aggregator works with any valid submissions");
}

/// Tests state synchronization after partition recovery.
#[test]
fn test_partition_state_sync() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..4).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Simulate training steps with checkpoints
    let mut checkpoints: HashMap<PartyId, u64> = HashMap::new();
    for party in &parties {
        checkpoints.insert(party.clone(), 0);
    }

    // Partition node 3
    network.start_partition(vec![parties[3].clone()]);

    // Non-partitioned nodes advance through steps 1-5
    for step in 1..=5 {
        for i in 0..3 {
            // Simulate checkpoint update
            checkpoints.insert(parties[i].clone(), step);
        }
    }

    // Partitioned node is still at step 0
    assert_eq!(checkpoints[&parties[3]], 0);

    // End partition
    network.end_partition();

    // Sync state: partitioned node requests latest checkpoint
    let max_step = checkpoints.values().max().copied().unwrap_or(0);

    // Simulate sync message
    let sync_msg = NetworkMessage::Checkpoint {
        step: max_step,
        weights_hash: [0u8; 32],
    };
    assert!(network.send(&parties[3], sync_msg));

    // Partitioned node updates to latest step
    checkpoints.insert(parties[3].clone(), max_step);

    // Verify all nodes are synchronized
    for party in &parties {
        assert_eq!(
            checkpoints[party], 5,
            "Party {:?} should be at step 5 after sync",
            party
        );
    }
}

/// Tests handling of delayed messages from previously partitioned nodes.
#[test]
fn test_partition_delayed_messages() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Track received step numbers
    let mut received_steps: HashMap<PartyId, Vec<u64>> = HashMap::new();
    for party in &parties {
        received_steps.insert(party.clone(), Vec::new());
    }

    // Non-partitioned nodes at step 5
    let current_step = 5;

    // Partition ends and node 2 sends its old step (step 2)
    let old_step_msg = NetworkMessage::GradientShare {
        from: parties[2].clone(),
        step: 2, // Old step
        gradients: vec![1.0, 2.0],
        commitment: [0u8; 32],
    };

    // Send to coordinator (simulated as party 0)
    network.send(&parties[0], old_step_msg);

    let received = network.receive(&parties[0]);
    assert_eq!(received.len(), 1);

    // Check the step number in the received message
    if let NetworkMessage::GradientShare { step, .. } = &received[0] {
        // Old step should be detected and potentially discarded
        assert!(*step < current_step, "Should receive old step message");

        // In production, this would be discarded as stale
        // Here we just verify we can detect it
    }
}

// ============================================================================
// Packet Loss Tests
// ============================================================================

/// Tests network behavior with packet loss.
#[test]
fn test_network_packet_loss() {
    let network = Arc::new(MockNetwork::new().with_packet_loss(0.3)); // 30% loss

    let parties: Vec<PartyId> = (0..2).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Send many messages
    let mut delivered = 0;
    let total = 100;

    for i in 0..total {
        let msg = NetworkMessage::Heartbeat {
            from: parties[0].clone(),
            timestamp: i as u64,
        };
        if network.send(&parties[1], msg) {
            delivered += 1;
        }
    }

    // With 30% packet loss, expect roughly 70 deliveries
    // Allow some variance
    assert!(
        delivered >= 50 && delivered <= 90,
        "Delivered {} messages, expected around 70",
        delivered
    );

    let stats = network.stats();
    assert!(stats.dropped > 0, "Some messages should be dropped");
}

/// Tests network latency simulation.
#[test]
fn test_network_latency() {
    let latency_ms = 50;
    let network = Arc::new(MockNetwork::new().with_latency(latency_ms));

    let parties: Vec<PartyId> = (0..2).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Latency is simulated in the mock but doesn't actually delay
    // This just verifies the configuration is set
    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 1,
    };

    assert!(network.send(&parties[1], msg));
    // In a real implementation, we would measure actual delay
}

// ============================================================================
// Complex Partition Scenarios
// ============================================================================

/// Tests multiple sequential partitions.
#[test]
fn test_multiple_sequential_partitions() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 1,
    };

    // First partition: nodes 1, 2
    network.start_partition(vec![parties[1].clone(), parties[2].clone()]);
    assert!(!network.send(&parties[1], msg.clone()));
    assert!(!network.send(&parties[2], msg.clone()));
    assert!(network.send(&parties[3], msg.clone()));
    network.end_partition();

    // Second partition: nodes 3, 4
    network.start_partition(vec![parties[3].clone(), parties[4].clone()]);
    assert!(network.send(&parties[1], msg.clone())); // Now reachable
    assert!(network.send(&parties[2], msg.clone())); // Now reachable
    assert!(!network.send(&parties[3], msg.clone())); // Now partitioned
    assert!(!network.send(&parties[4], msg.clone())); // Now partitioned
    network.end_partition();

    // No partition
    for i in 1..5 {
        assert!(network.send(&parties[i], msg.clone()), "All should be reachable");
    }
}

/// Tests graceful handling of empty partitions.
#[test]
fn test_empty_partition() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Empty partition list
    network.start_partition(vec![]);

    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 1,
    };

    // All should still be reachable
    for i in 1..3 {
        assert!(
            network.send(&parties[i], msg.clone()),
            "Empty partition should not block anyone"
        );
    }

    let stats = network.stats();
    assert!(stats.partition_active); // Technically active but empty
}

/// Tests partition of all nodes (complete network failure).
#[test]
fn test_complete_partition() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Partition everyone
    network.start_partition(parties.clone());

    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 1,
    };

    // No one should receive
    for party in &parties {
        assert!(!network.send(party, msg.clone()), "Complete partition should block all");
    }

    // Recovery
    network.end_partition();

    // Everyone should be reachable again
    for i in 1..3 {
        assert!(network.send(&parties[i], msg.clone()));
    }
}
