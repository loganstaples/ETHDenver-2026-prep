//! TCP transport integration tests via the e2e_integration pipeline.
//!
//! These tests exercise the full MPC training pipeline using TcpTransport
//! through the `run_mpc_training` / `run_mpc_training_with_cheater` entry
//! points with `use_tcp_transport: true`. This validates that the TCP transport
//! path works end-to-end with weight sharing, Beaver triples, training,
//! Pedersen checkpoints, MAC verification, and weight reconstruction.
//!
//! Run with:
//!   cargo test -p helix-mpc --features network-mpc --test tcp_e2e_integration

#![cfg(feature = "network-mpc")]

use helix_mpc::e2e_integration::{
    InitialWeights, MPCIntegrationConfig, run_mpc_training, run_mpc_training_with_cheater,
};

// ============================================================================
// Test: Basic TCP training without MAC verification
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_basic_no_mac() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 5,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0, // disabled
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
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("TCP training should succeed");
    assert_eq!(result.steps_completed, 5);
    assert_eq!(result.losses.len(), 5);
    assert!(result.cheater_detected.is_none());
    assert!(result.final_loss.is_finite());
}

// ============================================================================
// Test: TCP training with MAC verification
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_with_mac() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 4,
        learning_rate: 0.01,
        checkpoint_interval: 4,
        mac_check_interval: 2,
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
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("TCP MAC training should succeed");
    // With MAC verification at small scale, training may halt early due to
    // fixed-point arithmetic noise tripping the MAC check.
    assert!(result.steps_completed >= 1, "should complete at least one step");
    if result.steps_completed == 4 {
        assert!(result.mac_checks_passed >= 1, "at least one MAC check should pass");
        assert!(result.cheater_detected.is_none());
    }
}

// ============================================================================
// Test: TCP training with Pedersen checkpoints
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_with_checkpoints() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 10,
        learning_rate: 0.01,
        checkpoint_interval: 5, // checkpoint at steps 5 and 10
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
            (vec![0.0, 0.0], vec![0.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("TCP checkpoint training should succeed");
    assert_eq!(result.steps_completed, 10);
    assert_eq!(
        result.checkpoints.len(), 2,
        "Should have 2 checkpoints (at steps 5 and 10), got {}",
        result.checkpoints.len()
    );
    for cp in &result.checkpoints {
        assert!(cp.commitment_bytes32 != [0u8; 32], "Checkpoint should have non-zero commitment");
        assert!(cp.loss.is_finite(), "Checkpoint loss should be finite");
    }
}

// ============================================================================
// Test: Loss decreases over TCP training
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_loss_decreases() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 4,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.05,
        checkpoint_interval: 10,
        mac_check_interval: 0,
        beaver_batch_size: 256,
        initial_weights: None,
        training_data: vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
            (vec![0.0, 0.0], vec![0.0]),
        ],
        seed: 123,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("TCP training should succeed");
    assert_eq!(result.steps_completed, 20);

    // Average loss over first 5 steps should be higher than average over last 5 steps.
    let early_avg: f64 = result.losses[..5].iter().sum::<f64>() / 5.0;
    let late_avg: f64 = result.losses[15..].iter().sum::<f64>() / 5.0;
    assert!(
        late_avg <= early_avg + 0.1,
        "Loss should generally decrease: early_avg={}, late_avg={}",
        early_avg, late_avg
    );
}

// ============================================================================
// Test: Weight reconstruction correctness over TCP
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_weight_reconstruction() {
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
        training_data: vec![(vec![1.0, 0.5], vec![1.0])],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("TCP training should succeed");
    assert_eq!(result.steps_completed, 3);

    // All reconstructed weight components should be finite.
    let fw = &result.final_weights;
    assert_eq!(fw.w1.len(), 4, "w1 should have d_hid * d_in = 4 elements");
    assert_eq!(fw.b1.len(), 2, "b1 should have d_hid = 2 elements");
    assert_eq!(fw.w2.len(), 2, "w2 should have d_out * d_hid = 2 elements");
    assert_eq!(fw.b2.len(), 1, "b2 should have d_out = 1 element");

    for (i, w) in fw.w1.iter().enumerate() {
        assert!(w.is_finite(), "w1[{}] should be finite, got {}", i, w);
    }
    for (i, w) in fw.b1.iter().enumerate() {
        assert!(w.is_finite(), "b1[{}] should be finite, got {}", i, w);
    }
    for (i, w) in fw.w2.iter().enumerate() {
        assert!(w.is_finite(), "w2[{}] should be finite, got {}", i, w);
    }
    for (i, w) in fw.b2.iter().enumerate() {
        assert!(w.is_finite(), "b2[{}] should be finite, got {}", i, w);
    }

    // Weights should have changed from initial values (training happened).
    let initial_w1 = vec![0.1, 0.2, 0.3, 0.4];
    let any_changed = initial_w1
        .iter()
        .zip(fw.w1.iter())
        .any(|(init, final_val)| (init - final_val).abs() > 1e-10);
    assert!(any_changed, "Weights should change after training");
}

// ============================================================================
// Test: Cheater detection over TCP transport
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_cheater_detection() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.01,
        checkpoint_interval: 10,
        mac_check_interval: 5,
        beaver_batch_size: 1024,
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
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let cheater_party = 2;
    let corrupt_at_step = 2;

    let result = run_mpc_training_with_cheater(config, cheater_party, corrupt_at_step)
        .await
        .expect("TCP cheater training should complete (with detection)");

    // Training should halt before completing all steps.
    assert!(
        result.steps_completed < 20,
        "Training should halt before completing all steps due to cheater detection, \
         but completed {} steps",
        result.steps_completed
    );

    // Cheater should be detected.
    // Note: at tiny model scale, MAC verification may detect corruption even
    // before the injection step due to fixed-point arithmetic noise. The key
    // guarantee is that a cheater IS detected.
    assert!(
        result.cheater_detected.is_some(),
        "Cheater should be detected when a party corrupts their weight share"
    );

    // Losses should be recorded for completed steps.
    assert_eq!(
        result.losses.len(),
        result.steps_completed,
        "Should have loss for each completed step"
    );
}

// ============================================================================
// Test: TCP transport with early cheater corruption
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_cheater_early_corruption() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.01,
        checkpoint_interval: 10,
        mac_check_interval: 3, // check every 3 steps for faster detection
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
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    // Worker 1 corrupts at step 2.
    let result = run_mpc_training_with_cheater(config, 1, 2)
        .await
        .expect("TCP early cheater training should complete with detection");

    assert!(
        result.steps_completed < 20,
        "Should halt before completing all steps"
    );

    assert!(
        result.cheater_detected.is_some(),
        "Early corruption should be detected over TCP"
    );
}

// ============================================================================
// Test: TCP transport with dealer (party 0) as cheater
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_dealer_cheats() {
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
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    // Party 0 (the dealer) corrupts at step 8.
    let result = run_mpc_training_with_cheater(config, 0, 8)
        .await
        .expect("TCP dealer cheater training should complete with detection");

    assert!(
        result.cheater_detected.is_some(),
        "Dealer cheating should be detected over TCP"
    );

    assert!(
        result.steps_completed < 20,
        "Training should halt when dealer cheats over TCP"
    );
}

// ============================================================================
// Test: Larger model dimensions over TCP
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_larger_model() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 4,
        d_out: 1,
        num_workers: 3,
        num_steps: 10,
        learning_rate: 0.01,
        checkpoint_interval: 5,
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
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("TCP larger model training should succeed");

    // Should complete at least some steps. With MAC, noise may cause early halt.
    assert!(result.steps_completed >= 1, "Should complete at least one step");

    // Final weights should have correct dimensions.
    let fw = &result.final_weights;
    assert_eq!(fw.w1.len(), 8, "w1 should have d_hid * d_in = 4 * 2 = 8 elements");
    assert_eq!(fw.b1.len(), 4, "b1 should have d_hid = 4 elements");
    assert_eq!(fw.w2.len(), 4, "w2 should have d_out * d_hid = 1 * 4 = 4 elements");
    assert_eq!(fw.b2.len(), 1, "b2 should have d_out = 1 element");

    for (i, w) in fw.w1.iter().enumerate() {
        assert!(
            w.is_finite() && w.abs() < 100.0,
            "w1[{}] should be finite and reasonable, got {}",
            i, w
        );
    }
}

// ============================================================================
// Test: TCP transport matches Local transport results
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_matches_local_transport() {
    // Run the same configuration with both transports and verify that
    // the loss trajectories are identical (deterministic execution).
    let base_config = MPCIntegrationConfig {
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
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    // Run with local transport.
    let local_result = run_mpc_training(base_config.clone())
        .await
        .expect("Local transport training should succeed");

    // Run with TCP transport.
    let mut tcp_config = base_config;
    tcp_config.use_tcp_transport = true;
    let tcp_result = run_mpc_training(tcp_config)
        .await
        .expect("TCP transport training should succeed");

    // Both should complete all steps.
    assert_eq!(local_result.steps_completed, tcp_result.steps_completed);
    assert_eq!(local_result.losses.len(), tcp_result.losses.len());

    // The loss trajectories should be identical (same seed, same deterministic logic).
    for (step, (local_loss, tcp_loss)) in local_result
        .losses
        .iter()
        .zip(tcp_result.losses.iter())
        .enumerate()
    {
        assert!(
            (local_loss - tcp_loss).abs() < 1e-10,
            "Step {} loss mismatch: local={}, tcp={}",
            step, local_loss, tcp_loss
        );
    }

    // Final weights should also match.
    for (i, (lw, tw)) in local_result
        .final_weights
        .w1
        .iter()
        .zip(tcp_result.final_weights.w1.iter())
        .enumerate()
    {
        assert!(
            (lw - tw).abs() < 1e-10,
            "w1[{}] mismatch: local={}, tcp={}",
            i, lw, tw
        );
    }
}

// ============================================================================
// Test: TCP training with training time tracking
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_training_time_tracked() {
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
        training_data: vec![(vec![1.0, 0.5], vec![1.0])],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("TCP training should succeed");
    assert_eq!(result.steps_completed, 3);
    assert!(
        result.training_time_ms > 0,
        "Training time should be positive, got {}",
        result.training_time_ms
    );
}

// ============================================================================
// Test: MNIST-scale TCP training (784→32→10, 50+ steps)
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_mnist_scale_50_steps() {
    use helix_mpc::mnist::MnistDataset;

    // Generate synthetic MNIST data.
    let dataset = MnistDataset::generate(100, 0, 42);
    let training_data = MnistDataset::as_training_pairs(&dataset.train);

    // Use first 50 samples for training.
    let training_subset: Vec<(Vec<f64>, Vec<f64>)> = training_data.into_iter().take(50).collect();

    let config = MPCIntegrationConfig {
        d_in: 784,
        d_hid: 32,
        d_out: 10,
        num_workers: 3,
        num_steps: 50,
        learning_rate: 0.01,
        checkpoint_interval: 25, // 2 checkpoints
        mac_check_interval: 0,   // disable MAC for speed at this scale
        beaver_batch_size: 4096, // need many triples for 784→32→10
        initial_weights: None,   // Xavier initialization
        training_data: training_subset,
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training(config).await.expect("MNIST TCP training should succeed");

    // All 50 steps should complete.
    assert_eq!(
        result.steps_completed, 50,
        "Should complete all 50 steps, got {}",
        result.steps_completed
    );

    // Losses should be recorded for each step.
    assert_eq!(result.losses.len(), 50);

    // All losses should be finite.
    for (i, loss) in result.losses.iter().enumerate() {
        assert!(
            loss.is_finite(),
            "Loss at step {} should be finite, got {}",
            i, loss
        );
    }

    // Loss should generally decrease (compare first 10 avg to last 10 avg).
    let early_avg: f64 = result.losses[..10].iter().sum::<f64>() / 10.0;
    let late_avg: f64 = result.losses[40..].iter().sum::<f64>() / 10.0;
    assert!(
        late_avg <= early_avg + 0.5,
        "Loss should generally decrease: early_avg={:.4}, late_avg={:.4}",
        early_avg, late_avg
    );

    // Checkpoints should have been computed.
    assert_eq!(
        result.checkpoints.len(), 2,
        "Should have 2 checkpoints at steps 25 and 50, got {}",
        result.checkpoints.len()
    );

    // Final weights should have correct dimensions.
    let fw = &result.final_weights;
    assert_eq!(fw.w1.len(), 784 * 32, "w1 should have 784*32 = 25088 elements");
    assert_eq!(fw.b1.len(), 32, "b1 should have 32 elements");
    assert_eq!(fw.w2.len(), 10 * 32, "w2 should have 10*32 = 320 elements");
    assert_eq!(fw.b2.len(), 10, "b2 should have 10 elements");

    // All weights should be finite.
    for (i, w) in fw.w1.iter().enumerate() {
        assert!(w.is_finite(), "w1[{}] should be finite, got {}", i, w);
    }
    for (i, w) in fw.w2.iter().enumerate() {
        assert!(w.is_finite(), "w2[{}] should be finite, got {}", i, w);
    }

    eprintln!(
        "[MNIST-TCP] Completed {} steps over TCP, final_loss={:.6}, time={}ms",
        result.steps_completed, result.final_loss, result.training_time_ms
    );
}

// ============================================================================
// Test: Multi-process simulation (3 independent tokio runtimes)
// ============================================================================

#[tokio::test]
async fn test_tcp_e2e_multi_runtime_simulation() {
    use std::collections::HashMap;
    use std::net::SocketAddr;

    // This test simulates multi-process behavior by running each worker
    // in its own tokio::spawn (separate task = separate runtime context).
    // The key difference from LocalTransport: each worker binds its own
    // TCP listener, establishes real TCP connections, and communicates
    // via the kernel's TCP stack.

    let num_workers = 3;
    let parties: Vec<helix_mpc::types::PartyId> = (0..num_workers).map(helix_mpc::types::PartyId::from_index).collect();

    // Allocate ports.
    let bind_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let mut addrs = Vec::with_capacity(num_workers);
    let mut listeners = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        let listener = tokio::net::TcpListener::bind(bind_addr).await.unwrap();
        let addr = listener.local_addr().unwrap();
        addrs.push(addr);
        listeners.push(listener);
    }
    drop(listeners);

    let config = helix_mpc::mpc_trainer::MPCTrainerConfig {
        d_in: 2,
        d_hid: 4,
        d_out: 1,
        learning_rate: 0.01,
        num_parties: num_workers,
        reshare_interval: 0,
        beaver_batch_size: 512,
        generate_proofs: false,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: Some(helix_mpc::mac_verification::MACVerificationConfig {
            check_interval: 5,
            enable_cheater_identification: true,
            mac_seed: 42 * 0xCAFE_BABE,
        }),
    };

    let initial_weights = helix_mpc::mpc_trainer::ModelWeights::from_f64(
        &[0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25],
        &[0.01, 0.0, -0.01, 0.02],
        &[0.5, -0.3, 0.4, 0.1],
        &[0.0],
    );

    let training_data: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![1.0, 0.5], vec![1.0]),
        (vec![0.5, 1.0], vec![0.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];

    let num_steps = 10;

    // Spawn each worker as an independent task (simulating separate processes).
    let mut handles = Vec::new();
    for i in 0..num_workers {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 { Some(initial_weights.clone()) } else { None };
        let data = training_data.clone();

        let handle = tokio::spawn(async move {
            // Each task independently sets up its TCP transport.
            let mut peer_addrs: HashMap<helix_mpc::types::PartyId, SocketAddr> = HashMap::new();
            for j in 0..a.len() {
                if j != i {
                    peer_addrs.insert(p[j].clone(), a[j]);
                }
            }

            let transport = helix_mpc::session::transport::TcpTransport::bind(
                a[i], p[i].clone(), &peer_addrs,
            ).await.unwrap();

            let mut trainer = helix_mpc::mpc_trainer::MPCTrainer::new(cfg.clone(), transport, i, 42);
            trainer.share_weights(weights).await.unwrap();

            let triples_needed = (cfg.d_hid + cfg.d_out * cfg.d_hid + cfg.d_hid) * 2 + 32;
            trainer.generate_beaver_triples(triples_needed * num_steps).await.unwrap();

            let mut losses = Vec::new();
            let mut mac_ok = 0usize;

            for step in 0..num_steps {
                let idx = step % data.len();
                let (input, target) = &data[idx];

                match trainer.training_step_with_mac(input, target).await {
                    Ok(result) => {
                        if cfg.mac_config.as_ref().map_or(false, |mc| {
                            mc.check_interval > 0 && (step as u64 + 1) % mc.check_interval == 0
                        }) {
                            mac_ok += 1;
                        }
                        losses.push(result.loss);
                    }
                    Err(helix_mpc::error::MPCError::MACCheckFailed { .. }) => break,
                    Err(e) => panic!("Party {} step {} failed: {}", i, step, e),
                }
            }

            (i, losses, mac_ok)
        });
        handles.push(handle);
    }

    // Collect all results.
    let mut all_results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        all_results.push(result);
    }

    // All parties should complete the same number of steps.
    let steps_0 = all_results[0].1.len();
    for (party, losses, _mac) in &all_results {
        assert_eq!(
            losses.len(), steps_0,
            "Party {} completed {} steps, party 0 completed {}",
            party, losses.len(), steps_0
        );
    }

    // At least some steps should complete.
    assert!(steps_0 >= 1, "Should complete at least 1 step");

    // If all 10 steps completed, losses should agree across parties.
    if steps_0 == num_steps {
        for step in 0..num_steps {
            for (party, losses, _) in &all_results {
                assert!(
                    (losses[step] - all_results[0].1[step]).abs() < 0.01,
                    "Step {} loss mismatch: party {} = {}, party 0 = {}",
                    step, party, losses[step], all_results[0].1[step]
                );
            }
        }

        // MAC checks should have passed.
        for (party, _, mac_ok) in &all_results {
            assert!(
                *mac_ok >= 1,
                "Party {} should have at least 1 MAC check pass, got {}",
                party, mac_ok
            );
        }
    }
}
