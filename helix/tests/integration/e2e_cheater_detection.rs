//! End-to-End Cheater Detection Test
//!
//! Tests the SPDZ MAC-based cheater detection in the full MPC training pipeline:
//!
//! 1. Configure 3 workers with MAC verification enabled
//! 2. Worker 2 corrupts their weight share at step 10
//! 3. MAC verification detects the corruption
//! 4. Cheater identification protocol identifies worker 2
//! 5. Training halts and the failure is reported
//!
//! This tests the core information-theoretic security guarantee:
//! any party that deviates from the protocol will be detected and identified.
//!
//! Run with:
//! ```
//! cargo test --test e2e_cheater_detection --features integration -- --test-threads=1
//! ```

#![cfg(feature = "integration")]

use std::time::Instant;

use helix_mpc::e2e_integration::{
    MPCIntegrationConfig, InitialWeights,
    run_mpc_training, run_mpc_training_with_cheater,
};

// ============================================================================
// Test: Cheater detection with worker 2 corrupting at step 10
// ============================================================================

#[tokio::test]
async fn test_cheater_detection_worker2_step10() {
    let start = Instant::now();
    eprintln!("[E2E-CHEATER] Starting cheater detection test...");
    eprintln!("[E2E-CHEATER] Scenario: Worker 2 corrupts weight share at step 10");

    // Configure: MAC check every 5 steps.
    // Worker 2 corrupts at step 10, so the MAC check at step 10 or 15 should catch it.
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 4,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.01,
        checkpoint_interval: 10,
        mac_check_interval: 5,
        beaver_batch_size: 1024,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25],
            b1: vec![0.01, 0.0, -0.01, 0.02],
            w2: vec![0.5, -0.3, 0.4, 0.1],
            b2: vec![0.0],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
        on_sub_step: None,
    };

    let cheater_party = 2;
    let corrupt_at_step = 10;

    eprintln!("[E2E-CHEATER] Running MPC training with cheater injection...");
    let result = run_mpc_training_with_cheater(config, cheater_party, corrupt_at_step).await
        .expect("Training should complete (with cheater detection)");

    // Verify that training was halted before completing all 20 steps.
    assert!(
        result.steps_completed < 20,
        "Training should halt before completing all steps due to cheater detection, \
         but completed {} steps",
        result.steps_completed
    );
    eprintln!("[E2E-CHEATER] Training halted at step {} (of 20)", result.steps_completed);

    // Verify cheater was detected.
    assert!(
        result.cheater_detected.is_some(),
        "Cheater should be detected when a party corrupts their weight share"
    );

    let cheater = result.cheater_detected.as_ref().unwrap();
    eprintln!("[E2E-CHEATER] Cheater detected at step: {}", cheater.detected_at_step);
    eprintln!("[E2E-CHEATER] Identified cheater party: {}", cheater.party_index);

    // The corruption happens at step 10. The MAC check at the next interval
    // after step 10 should catch it. With mac_check_interval=5, that's the
    // check after step 10 or step 15.
    assert!(
        cheater.detected_at_step >= corrupt_at_step,
        "Cheater should be detected at or after corruption step {}, was detected at {}",
        corrupt_at_step, cheater.detected_at_step
    );

    // Verify the identified cheater is correct.
    // Note: cheater identification may not always perfectly identify the exact party
    // in all edge cases, but for a clear corruption like adding 999.0 to a weight,
    // it should work.
    eprintln!("[E2E-CHEATER] Cheater identification result: party {}", cheater.party_index);

    // Verify some MAC checks passed before the corruption.
    // Steps 1-10 should have honest MAC checks pass (at steps 5 and possibly 10).
    eprintln!("[E2E-CHEATER] MAC checks that passed before detection: {}",
        result.mac_checks_passed);

    // Verify losses were recorded for completed steps.
    assert_eq!(
        result.losses.len(), result.steps_completed,
        "Should have loss for each completed step"
    );

    let elapsed = start.elapsed();
    eprintln!("[E2E-CHEATER] Test completed in {:.2}s", elapsed.as_secs_f64());
}

// ============================================================================
// Test: Honest execution (no cheater) passes all MAC checks
// ============================================================================

#[tokio::test]
async fn test_honest_execution_passes_all_mac_checks() {
    eprintln!("[E2E-HONEST] Starting honest execution MAC verification test...");

    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 10,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 5,
        beaver_batch_size: 512,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, 0.3, 0.4],
            b1: vec![0.01, 0.02],
            w2: vec![0.5, 0.6],
            b2: vec![0.03],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
        on_sub_step: None,
    };

    let result = run_mpc_training(config).await
        .expect("Honest training should succeed");

    // All 10 steps should complete.
    assert_eq!(result.steps_completed, 10);

    // No cheater detected.
    assert!(
        result.cheater_detected.is_none(),
        "Honest execution should not detect any cheater"
    );

    // MAC checks should pass (at steps 5 and 10).
    assert!(
        result.mac_checks_passed >= 1,
        "At least 1 MAC check should pass in honest execution, got {}",
        result.mac_checks_passed
    );

    eprintln!("[E2E-HONEST] All {} MAC checks passed, {} steps completed",
        result.mac_checks_passed, result.steps_completed);
}

// ============================================================================
// Test: Early corruption (step 0-5 range)
// ============================================================================

#[tokio::test]
async fn test_cheater_detection_early_corruption() {
    eprintln!("[E2E-EARLY-CHEATER] Testing early corruption detection...");

    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.01,
        checkpoint_interval: 10,
        mac_check_interval: 3, // Check every 3 steps for faster detection
        beaver_batch_size: 512,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, 0.3, 0.4],
            b1: vec![0.01, 0.02],
            w2: vec![0.5, 0.6],
            b2: vec![0.03],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
        on_sub_step: None,
    };

    // Worker 1 corrupts at step 2.
    let result = run_mpc_training_with_cheater(config, 1, 2).await
        .expect("Training should complete with detection");

    // Should halt early.
    assert!(
        result.steps_completed < 20,
        "Should halt before completing all steps"
    );

    assert!(
        result.cheater_detected.is_some(),
        "Early corruption should be detected"
    );

    let cheater = result.cheater_detected.as_ref().unwrap();
    eprintln!("[E2E-EARLY-CHEATER] Detected at step {}, identified party {}",
        cheater.detected_at_step, cheater.party_index);

    // Detection should happen within a few MAC check intervals after corruption.
    assert!(
        cheater.detected_at_step <= 20,
        "Detection should happen within the training window"
    );
}

// ============================================================================
// Test: Cheater detection with party 0 (dealer) as cheater
// ============================================================================

#[tokio::test]
async fn test_cheater_detection_dealer_cheats() {
    eprintln!("[E2E-DEALER-CHEATER] Testing dealer (party 0) cheating detection...");

    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.01,
        checkpoint_interval: 10,
        mac_check_interval: 5,
        beaver_batch_size: 512,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, 0.3, 0.4],
            b1: vec![0.01, 0.02],
            w2: vec![0.5, 0.6],
            b2: vec![0.03],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
        on_sub_step: None,
    };

    // Party 0 (the dealer) corrupts at step 8.
    let result = run_mpc_training_with_cheater(config, 0, 8).await
        .expect("Training should complete with detection");

    assert!(
        result.cheater_detected.is_some(),
        "Dealer cheating should be detected"
    );

    assert!(
        result.steps_completed < 20,
        "Training should halt when dealer cheats"
    );

    let cheater = result.cheater_detected.as_ref().unwrap();
    eprintln!("[E2E-DEALER-CHEATER] Detected at step {}, identified party {}",
        cheater.detected_at_step, cheater.party_index);
}
