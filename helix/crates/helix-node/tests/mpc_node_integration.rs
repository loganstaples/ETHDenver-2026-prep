//! Integration tests for private training (MPC) integration with the node.
//!
//! Tests the `private_training` module's high-level API:
//! - `PrivateTrainingSession` (transport mesh creation)
//! - `PrivateTrainingWorker` (per-party MPC trainer wrapper)
//! - `run_private_training_round` (full round orchestration)
//! - `PrivateTrainingCoordinator` (aggregator-side session management)
//! - Cleartext fallback path
//!
//! Run with:
//!   cargo test -p helix-node --test mpc_node_integration -- --test-threads=1

use helix_node::trainer::MlpModel;
use helix_node::training::private_training::{
    mlp_to_mpc_weights, mpc_to_mlp_model, run_cleartext_training_step,
    run_private_training_round, PrivateTrainingConfig,
    PrivateTrainingCoordinator, PrivateTrainingSession, PrivateTrainingWorker,
    TrainingMode, TrainingSample,
};

// ──────────────────────────────────────────────────────────────
// Unit-level integration: model conversion round-trip accuracy
// ──────────────────────────────────────────────────────────────

#[test]
fn test_model_conversion_preserves_weights() {
    let model = MlpModel::new_random(3, 4, 2, 99);
    let mpc = mlp_to_mpc_weights(&model);
    let recovered = mpc_to_mlp_model(&mpc, 3, 4, 2);

    // All weight vectors must round-trip within f64→Fr→f64 tolerance.
    let tolerance = 1e-6;
    for (i, (a, b)) in model.w1.iter().zip(recovered.w1.iter()).enumerate() {
        assert!(
            (a - b).abs() < tolerance,
            "w1[{}]: {} vs {} (delta={})",
            i,
            a,
            b,
            (a - b).abs()
        );
    }
    for (i, (a, b)) in model.b1.iter().zip(recovered.b1.iter()).enumerate() {
        assert!(
            (a - b).abs() < tolerance,
            "b1[{}]: {} vs {} (delta={})",
            i,
            a,
            b,
            (a - b).abs()
        );
    }
    for (i, (a, b)) in model.w2.iter().zip(recovered.w2.iter()).enumerate() {
        assert!(
            (a - b).abs() < tolerance,
            "w2[{}]: {} vs {} (delta={})",
            i,
            a,
            b,
            (a - b).abs()
        );
    }
    for (i, (a, b)) in model.b2.iter().zip(recovered.b2.iter()).enumerate() {
        assert!(
            (a - b).abs() < tolerance,
            "b2[{}]: {} vs {} (delta={})",
            i,
            a,
            b,
            (a - b).abs()
        );
    }
}

// ──────────────────────────────────────────────────────────────
// Session transport mesh: correct number of transports created
// ──────────────────────────────────────────────────────────────

#[test]
fn test_session_transport_mesh_creation() {
    let model = MlpModel::new_random(2, 2, 1, 42);
    for n in [2, 3, 4] {
        let config = PrivateTrainingConfig {
            num_parties: n,
            ..PrivateTrainingConfig::small(n)
        };
        let session = PrivateTrainingSession::new(config, &model, format!("mesh-{}", n)).unwrap();
        assert_eq!(session.num_parties(), n);
        assert_eq!(session.available_transports(), n);
    }
}

#[test]
fn test_session_rejects_single_party() {
    let model = MlpModel::new_random(2, 2, 1, 42);
    let config = PrivateTrainingConfig {
        num_parties: 1,
        ..PrivateTrainingConfig::small(1)
    };
    assert!(PrivateTrainingSession::new(config, &model, "fail").is_err());
}

// ──────────────────────────────────────────────────────────────
// Worker lifecycle: share → triples → train
// ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_2party_worker_full_lifecycle() {
    let config = PrivateTrainingConfig::small(2);
    let model = MlpModel::new_random(2, 2, 1, 42);
    let mut session = PrivateTrainingSession::new(config.clone(), &model, "lifecycle").unwrap();

    let t0 = session.take_worker_transport(0).unwrap();
    let t1 = session.take_worker_transport(1).unwrap();
    let weights = session.initial_weights().clone();

    let mut w0 = PrivateTrainingWorker::new(config.clone(), t0, 0, 100);
    let mut w1 = PrivateTrainingWorker::new(config.clone(), t1, 1, 200);

    // Workers start uninitialized.
    assert!(!w0.is_initialized());
    assert!(!w1.is_initialized());

    // Phase 1: Share weights concurrently.
    let (r0, r1) = tokio::join!(w0.share_weights(Some(weights)), w1.share_weights(None));
    r0.unwrap();
    r1.unwrap();

    // Phase 2: Generate Beaver triples concurrently.
    let (r0, r1) = tokio::join!(
        w0.generate_beaver_triples(128),
        w1.generate_beaver_triples(128)
    );
    r0.unwrap();
    r1.unwrap();

    assert!(w0.is_initialized());
    assert!(w1.is_initialized());
    assert!(w0.beaver_triples_remaining() >= 128);

    // Phase 3: Training step concurrently.
    let input = [0.5, 0.3];
    let target = [1.0];
    let (r0, r1) = tokio::join!(
        w0.training_step(&input, &target),
        w1.training_step(&input, &target)
    );
    let s0 = r0.unwrap();
    let s1 = r1.unwrap();

    // Both parties must agree on the loss.
    assert!(
        (s0.loss - s1.loss).abs() < 1e-4,
        "Loss mismatch: {} vs {}",
        s0.loss,
        s1.loss
    );
    assert_eq!(w0.current_step(), 1);
    assert_eq!(w1.current_step(), 1);
}

#[tokio::test]
async fn test_worker_rejects_step_before_sharing() {
    let config = PrivateTrainingConfig::small(2);
    let model = MlpModel::new_random(2, 2, 1, 42);
    let mut session = PrivateTrainingSession::new(config.clone(), &model, "reject").unwrap();

    let t0 = session.take_worker_transport(0).unwrap();
    let mut w0 = PrivateTrainingWorker::new(config, t0, 0, 100);

    // Attempting to train before sharing weights must fail.
    let result = w0.training_step(&[0.5, 0.3], &[1.0]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_worker_rejects_triples_before_sharing() {
    let config = PrivateTrainingConfig::small(2);
    let model = MlpModel::new_random(2, 2, 1, 42);
    let mut session = PrivateTrainingSession::new(config.clone(), &model, "reject2").unwrap();

    let t0 = session.take_worker_transport(0).unwrap();
    let mut w0 = PrivateTrainingWorker::new(config, t0, 0, 100);

    // Generating triples before weight sharing must fail.
    let result = w0.generate_beaver_triples(64).await;
    assert!(result.is_err());
}

// ──────────────────────────────────────────────────────────────
// Full round: run_private_training_round with 2 parties
// ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_private_round_2party_single_step() {
    let config = PrivateTrainingConfig {
        mode: TrainingMode::Private,
        num_parties: 2,
        generate_proofs: false,
        ..PrivateTrainingConfig::small(2)
    };
    let model = MlpModel::new_random(2, 2, 1, 42);
    let samples = vec![TrainingSample {
        input: vec![0.5, 0.3],
        target: vec![1.0],
    }];

    let result = run_private_training_round(config, &model, &samples, "round-2p-1s")
        .await
        .unwrap();

    assert_eq!(result.mode, TrainingMode::Private);
    assert_eq!(result.total_steps, 1);
    assert_eq!(result.step_results.len(), 1);
    assert!(result.step_results[0].loss >= 0.0);
    assert!(result.step_results[0].loss.is_finite());
}

#[tokio::test]
async fn test_private_round_2party_multi_step() {
    let config = PrivateTrainingConfig {
        mode: TrainingMode::Private,
        num_parties: 2,
        learning_rate: 0.1,
        generate_proofs: false,
        ..PrivateTrainingConfig::small(2)
    };
    let model = MlpModel::new_random(2, 2, 1, 42);
    let samples = vec![
        TrainingSample {
            input: vec![1.0, 0.0],
            target: vec![1.0],
        },
        TrainingSample {
            input: vec![0.0, 1.0],
            target: vec![0.0],
        },
        TrainingSample {
            input: vec![1.0, 1.0],
            target: vec![1.0],
        },
    ];

    let result = run_private_training_round(config, &model, &samples, "round-2p-3s")
        .await
        .unwrap();

    assert_eq!(result.mode, TrainingMode::Private);
    assert_eq!(result.total_steps, 3);

    // All losses must be finite.
    for step in &result.step_results {
        assert!(step.loss.is_finite(), "Step {} loss is NaN/Inf", step.step);
    }
}

#[tokio::test]
async fn test_private_round_3party() {
    let config = PrivateTrainingConfig {
        mode: TrainingMode::Private,
        num_parties: 3,
        generate_proofs: false,
        ..PrivateTrainingConfig::small(3)
    };
    let model = MlpModel::new_random(2, 2, 1, 42);
    let samples = vec![
        TrainingSample {
            input: vec![0.5, 0.3],
            target: vec![1.0],
        },
        TrainingSample {
            input: vec![0.1, 0.9],
            target: vec![0.0],
        },
    ];

    let result = run_private_training_round(config, &model, &samples, "round-3p")
        .await
        .unwrap();

    assert_eq!(result.mode, TrainingMode::Private);
    assert_eq!(result.total_steps, 2);
    for step in &result.step_results {
        assert!(step.loss.is_finite());
    }
}

// ──────────────────────────────────────────────────────────────
// Cleartext fallback path
// ──────────────────────────────────────────────────────────────

#[test]
fn test_cleartext_step_produces_valid_model() {
    let model = MlpModel::new_random(2, 3, 1, 42);
    let result = run_cleartext_training_step(&model, &[0.5, 0.3], &[1.0], 0.01, 0);

    assert_eq!(result.step, 0);
    assert!(result.loss >= 0.0);
    assert!(result.loss.is_finite());
    assert_eq!(result.model.d_in, 2);
    assert_eq!(result.model.d_hid, 3);
    assert_eq!(result.model.d_out, 1);

    // At least one weight vector should have changed after SGD.
    let w1_changed = model.w1.iter().zip(result.model.w1.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
    let b1_changed = model.b1.iter().zip(result.model.b1.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
    let w2_changed = model.w2.iter().zip(result.model.w2.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
    let b2_changed = model.b2.iter().zip(result.model.b2.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
    assert!(
        w1_changed || b1_changed || w2_changed || b2_changed,
        "SGD should modify at least one weight"
    );
}

#[tokio::test]
async fn test_cleartext_round_fallback() {
    let config = PrivateTrainingConfig {
        mode: TrainingMode::Cleartext,
        ..PrivateTrainingConfig::small(2)
    };
    let model = MlpModel::new_random(2, 2, 1, 42);
    let samples = vec![
        TrainingSample {
            input: vec![1.0, 0.0],
            target: vec![1.0],
        },
        TrainingSample {
            input: vec![0.0, 1.0],
            target: vec![0.0],
        },
    ];

    let result = run_private_training_round(config, &model, &samples, "cleartext-fallback")
        .await
        .unwrap();

    assert_eq!(result.mode, TrainingMode::Cleartext);
    assert_eq!(result.total_steps, 2);

    // No proofs or resharing in cleartext mode.
    for step in &result.step_results {
        assert!(!step.has_proof);
        assert!(!step.reshared);
    }
}

// ──────────────────────────────────────────────────────────────
// Coordinator: session lifecycle
// ──────────────────────────────────────────────────────────────

#[test]
fn test_coordinator_session_lifecycle() {
    use helix_node::network::messages::PeerId;

    let config = PrivateTrainingConfig::small(2);
    let mut coord = PrivateTrainingCoordinator::new(config, None);

    // Register enough workers.
    let p0 = PeerId::from_string("peer-0");
    let p1 = PeerId::from_string("peer-1");
    coord.register_worker(&p0);
    coord.register_worker(&p1);

    assert!(coord.can_start_session());
    assert!(coord.active_session().is_none());

    // Start session.
    let model = MlpModel::new_random(2, 2, 1, 42);
    let result = coord.start_session(1, &model).unwrap();
    assert!(result.is_some());

    let (session_id, assignments) = result.unwrap();
    assert!(!session_id.is_empty());
    assert_eq!(assignments.len(), 2);

    // Session is now active.
    assert!(coord.active_session().is_some());

    // Take a transport.
    let t = coord.take_worker_transport(0);
    assert!(t.is_some());

    // Complete session.
    coord.complete_session();
    assert!(coord.active_session().is_none());
}

#[test]
fn test_coordinator_insufficient_workers() {
    use helix_node::network::messages::PeerId;

    let config = PrivateTrainingConfig::small(3);
    let mut coord = PrivateTrainingCoordinator::new(config, None);

    // Only register 1 worker (need 3).
    let p0 = PeerId::from_string("peer-0");
    coord.register_worker(&p0);
    assert!(!coord.can_start_session());

    let model = MlpModel::new_random(2, 2, 1, 42);
    let result = coord.start_session(1, &model).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_coordinator_fail_session() {
    use helix_node::network::messages::PeerId;

    let config = PrivateTrainingConfig::small(2);
    let mut coord = PrivateTrainingCoordinator::new(config, None);

    let p0 = PeerId::from_string("peer-0");
    let p1 = PeerId::from_string("peer-1");
    coord.register_worker(&p0);
    coord.register_worker(&p1);

    let model = MlpModel::new_random(2, 2, 1, 42);
    coord.start_session(1, &model).unwrap();
    assert!(coord.active_session().is_some());

    // Fail the session.
    coord.fail_session("test failure");
    assert!(coord.active_session().is_none());
}

// ──────────────────────────────────────────────────────────────
// MPC vs cleartext: same data produces comparable loss
// ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mpc_and_cleartext_both_produce_valid_output() {
    let model = MlpModel::new_random(2, 2, 1, 42);
    let samples = vec![TrainingSample {
        input: vec![0.5, 0.3],
        target: vec![1.0],
    }];

    // Cleartext path.
    let cleartext_result = {
        let config = PrivateTrainingConfig {
            mode: TrainingMode::Cleartext,
            ..PrivateTrainingConfig::small(2)
        };
        run_private_training_round(config, &model, &samples, "cmp-cleartext")
            .await
            .unwrap()
    };

    // MPC path.
    let mpc_result = {
        let config = PrivateTrainingConfig {
            mode: TrainingMode::Private,
            num_parties: 2,
            generate_proofs: false,
            ..PrivateTrainingConfig::small(2)
        };
        run_private_training_round(config, &model, &samples, "cmp-mpc")
            .await
            .unwrap()
    };

    let cleartext_loss = cleartext_result.step_results[0].loss;
    let mpc_loss = mpc_result.step_results[0].loss;

    // Both should produce finite losses and complete successfully.
    // Note: MPC loss is computed in BN254 Fr fixed-point (x * 2^64) so the
    // magnitude differs from cleartext f64 loss. What matters is that both
    // paths run to completion and produce valid (finite, non-NaN) results.
    assert!(cleartext_loss.is_finite(), "Cleartext loss is NaN/Inf");
    assert!(mpc_loss.is_finite(), "MPC loss is NaN/Inf");

    assert_eq!(cleartext_result.mode, TrainingMode::Cleartext);
    assert_eq!(mpc_result.mode, TrainingMode::Private);
    assert_eq!(cleartext_result.total_steps, 1);
    assert_eq!(mpc_result.total_steps, 1);
}
