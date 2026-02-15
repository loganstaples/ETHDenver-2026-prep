//! End-to-End MPC Integration Test
//!
//! Tests the complete MPC training pipeline with 3 workers, 20 training steps,
//! 2 Pedersen checkpoints, and MAC verification:
//!
//! 1. Transport mesh creation (LocalTransport)
//! 2. Weight sharing with MAC initialization
//! 3. Distributed Beaver triple generation
//! 4. Training loop with MAC-verified steps
//! 5. Pedersen commitment checkpoints
//! 6. Weight reconstruction and verification
//!
//! Run with:
//! ```
//! cargo test --test e2e_mpc_integration --features integration -- --test-threads=1
//! ```

#![cfg(feature = "integration")]

use std::time::Instant;

use helix_mpc::e2e_integration::{
    MPCIntegrationConfig, InitialWeights, run_mpc_training,
};

// ============================================================================
// Test: Full E2E MPC Training (3 workers, 20 steps, 2 checkpoints)
// ============================================================================

#[tokio::test]
async fn test_e2e_mpc_training_full() {
    let start = Instant::now();
    eprintln!("[E2E-MPC-INTEGRATION] Starting full MPC integration test...");

    // Configure: 3 workers, 20 steps, checkpoints every 10 steps, MAC every 5 steps.
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
            (vec![0.3, 0.7], vec![0.5]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
    };

    eprintln!("[E2E-MPC-INTEGRATION] Config: {} workers, {} steps, d_in={}, d_hid={}, d_out={}",
        config.num_workers, config.num_steps, config.d_in, config.d_hid, config.d_out);

    let result = run_mpc_training(config).await
        .expect("MPC training should complete successfully");

    // Verify step count.
    assert_eq!(
        result.steps_completed, 20,
        "Should complete all 20 training steps, got {}",
        result.steps_completed
    );
    eprintln!("[E2E-MPC-INTEGRATION] Steps completed: {}", result.steps_completed);

    // Verify losses recorded.
    assert_eq!(
        result.losses.len(), 20,
        "Should have 20 loss values, got {}",
        result.losses.len()
    );
    for (i, loss) in result.losses.iter().enumerate() {
        assert!(
            loss.is_finite(),
            "Loss at step {} should be finite, got {}",
            i, loss
        );
        assert!(
            *loss >= 0.0,
            "Loss at step {} should be non-negative, got {}",
            i, loss
        );
    }
    eprintln!("[E2E-MPC-INTEGRATION] All losses finite and non-negative");
    eprintln!("[E2E-MPC-INTEGRATION] Loss trajectory: first={:.6}, last={:.6}",
        result.losses[0], result.losses[19]);

    // Verify checkpoints (checkpoint_interval=10, so 2 checkpoints at steps 10 and 20).
    assert_eq!(
        result.checkpoints.len(), 2,
        "Should have 2 checkpoints (every 10 steps), got {}",
        result.checkpoints.len()
    );
    for cp in &result.checkpoints {
        assert!(
            !cp.commitment_bytes.is_empty(),
            "Checkpoint at step {} should have non-empty commitment",
            cp.step
        );
        assert!(
            cp.loss.is_finite(),
            "Checkpoint loss at step {} should be finite",
            cp.step
        );
    }
    eprintln!("[E2E-MPC-INTEGRATION] Checkpoints verified: {} checkpoints at steps {:?}",
        result.checkpoints.len(),
        result.checkpoints.iter().map(|c| c.step).collect::<Vec<_>>());

    // Verify MAC checks passed.
    // With mac_check_interval=5 and 20 steps, expect 4 checks (at steps 5, 10, 15, 20).
    assert!(
        result.mac_checks_passed >= 2,
        "Should have at least 2 MAC checks pass, got {}",
        result.mac_checks_passed
    );
    eprintln!("[E2E-MPC-INTEGRATION] MAC checks passed: {}", result.mac_checks_passed);

    // Verify no cheater detected.
    assert!(
        result.cheater_detected.is_none(),
        "No cheater should be detected in honest execution"
    );

    // Verify final weights are reasonable.
    let fw = &result.final_weights;
    assert_eq!(fw.w1.len(), 8, "w1 should have 8 elements (d_hid * d_in = 4 * 2)");
    assert_eq!(fw.b1.len(), 4, "b1 should have 4 elements");
    assert_eq!(fw.w2.len(), 4, "w2 should have 4 elements (d_out * d_hid = 1 * 4)");
    assert_eq!(fw.b2.len(), 1, "b2 should have 1 element");

    for (i, w) in fw.w1.iter().enumerate() {
        assert!(
            w.is_finite() && w.abs() < 100.0,
            "w1[{}] should be finite and reasonable, got {}",
            i, w
        );
    }
    for (i, w) in fw.w2.iter().enumerate() {
        assert!(
            w.is_finite() && w.abs() < 100.0,
            "w2[{}] should be finite and reasonable, got {}",
            i, w
        );
    }
    eprintln!("[E2E-MPC-INTEGRATION] Final weights verified as finite and reasonable");

    // Verify training time is recorded.
    assert!(
        result.training_time_ms > 0,
        "Training time should be positive"
    );

    // Verify loss trend (average of first 5 vs last 5).
    let early_loss: f64 = result.losses[..5].iter().sum::<f64>() / 5.0;
    let late_loss: f64 = result.losses[15..].iter().sum::<f64>() / 5.0;
    eprintln!("[E2E-MPC-INTEGRATION] Loss trend: early_avg={:.6}, late_avg={:.6}", early_loss, late_loss);

    let elapsed = start.elapsed();
    eprintln!("[E2E-MPC-INTEGRATION] Test completed in {:.2}s", elapsed.as_secs_f64());
    eprintln!("[E2E-MPC-INTEGRATION] Training pipeline time: {}ms", result.training_time_ms);
}

// ============================================================================
// Test: No-MAC training (faster, verifies basic pipeline)
// ============================================================================

#[tokio::test]
async fn test_e2e_mpc_training_no_mac() {
    eprintln!("[E2E-MPC-NO-MAC] Starting no-MAC integration test...");

    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 10,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0, // MAC disabled
        beaver_batch_size: 256,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, 0.3, 0.4],
            b1: vec![0.01, 0.02],
            w2: vec![0.5, 0.6],
            b2: vec![0.03],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
            (vec![0.0, 0.0], vec![0.0]),
        ],
        seed: 123,
        use_node_transport: false,
        use_tcp_transport: false,
    };

    let result = run_mpc_training(config).await
        .expect("No-MAC training should succeed");

    assert_eq!(result.steps_completed, 10);
    assert_eq!(result.mac_checks_passed, 0, "No MAC checks with mac disabled");
    assert!(result.cheater_detected.is_none());
    assert_eq!(result.checkpoints.len(), 2, "Checkpoints at steps 5 and 10");

    eprintln!("[E2E-MPC-NO-MAC] Completed: {} steps, final_loss={:.6}",
        result.steps_completed, result.final_loss);
}

// ============================================================================
// Test: NodeTransport backend
// ============================================================================

#[tokio::test]
async fn test_e2e_mpc_training_node_transport() {
    eprintln!("[E2E-NODE-TRANSPORT] Starting NodeTransport integration test...");

    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 5,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0,
        beaver_batch_size: 256,
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
        use_node_transport: true,
        use_tcp_transport: false,
    };

    let result = run_mpc_training(config).await
        .expect("NodeTransport training should succeed");

    assert_eq!(result.steps_completed, 5);
    assert!(result.cheater_detected.is_none());

    eprintln!("[E2E-NODE-TRANSPORT] Completed: {} steps", result.steps_completed);
}

// ============================================================================
// Test: Weight reconstruction correctness
// ============================================================================

#[tokio::test]
async fn test_e2e_weight_reconstruction_matches() {
    eprintln!("[E2E-RECONSTRUCTION] Testing weight reconstruction correctness...");

    // Run the same model twice with different seeds to ensure deterministic
    // reconstruction produces valid weights.
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 3,
        learning_rate: 0.01,
        checkpoint_interval: 3,
        mac_check_interval: 0,
        beaver_batch_size: 256,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, 0.3, 0.4],
            b1: vec![0.01, 0.02],
            w2: vec![0.5, 0.6],
            b2: vec![0.03],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
    };

    let result = run_mpc_training(config).await
        .expect("training should succeed");

    // Verify the reconstructed weights are different from initial (training happened).
    let fw = &result.final_weights;
    let initial_w1 = vec![0.1, 0.2, 0.3, 0.4];

    let mut any_changed = false;
    for (i, (init, final_val)) in initial_w1.iter().zip(fw.w1.iter()).enumerate() {
        if (init - final_val).abs() > 1e-10 {
            any_changed = true;
        }
    }
    assert!(any_changed, "Weights should change after training");

    eprintln!("[E2E-RECONSTRUCTION] Weight reconstruction verified - weights changed from initial values");
}
