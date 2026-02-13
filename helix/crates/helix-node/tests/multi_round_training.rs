//! Multi-Round Training Integration Tests.
//!
//! Tests that the MultiRoundController and WorkerStateManager correctly
//! drive consecutive training rounds with:
//!
//! - Commitment chaining across rounds (each round's initial = previous final)
//! - Weight persistence and reload via LocalWeightStore
//! - Crash recovery mid-session (aggregator and worker)
//! - Worker weight download & verification
//! - Training summary accuracy

use std::collections::HashMap;

use helix_core::ModelCheckpoint;
use helix_node::trainer::{MlpModel, average_models};
use helix_node::training::multi_round::{
    compute_weight_commitment, LocalWeightStore, MultiRoundConfig, MultiRoundController,
    WeightStore, WorkerStateManager,
};
use tempfile::tempdir;

/// Helper: serialize model to checkpoint bytes.
fn model_to_bytes(model: &MlpModel, round: u64) -> Vec<u8> {
    model.to_checkpoint(round).to_bytes().unwrap()
}

/// Helper: deserialize model from checkpoint bytes.
fn model_from_bytes(bytes: &[u8]) -> MlpModel {
    let ckpt = ModelCheckpoint::from_bytes(bytes).unwrap();
    MlpModel::from_checkpoint(&ckpt).unwrap()
}

/// Helper: perturb model weights by a position-dependent amount.
fn perturb_model(model: &mut MlpModel, delta: f64) {
    for w in model.w1.iter_mut() {
        *w += delta;
    }
    for b in model.b1.iter_mut() {
        *b += delta;
    }
    for w in model.w2.iter_mut() {
        *w += delta;
    }
    for b in model.b2.iter_mut() {
        *b += delta;
    }
}

fn test_config(dir: &std::path::Path) -> MultiRoundConfig {
    MultiRoundConfig {
        max_rounds: 10,
        weight_store_dir: dir.join("weights"),
        max_weight_files: 10,
        snapshot_dir: Some(dir.join("snapshots")),
        max_snapshots: 5,
        verify_commitments: true,
        d_in: 4,
        d_hid: 8,
        d_out: 2,
        model_seed: 42,
        learning_rate: 0.01,
    }
}

// ============================================================================
// Test 1: Three consecutive rounds with commitment chaining
// ============================================================================

#[test]
fn test_three_rounds_commitment_chaining_with_real_model() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // Create initial model and serialize
    let initial_model = MlpModel::new_random(4, 8, 2, 42);
    let initial_bytes = model_to_bytes(&initial_model, 0);
    let initial_commitment = compute_weight_commitment(&initial_bytes);

    let mut controller = MultiRoundController::new(
        config,
        initial_commitment,
        Some(initial_bytes.clone()),
    )
    .unwrap();

    // Store initial weights so workers can access them
    let weight_store = LocalWeightStore::new(dir.path().join("weights"), 10).unwrap();
    let (initial_uri, _) = weight_store.store_weights(999, &initial_bytes).unwrap();

    // Create 3 workers
    let mut workers: Vec<WorkerStateManager> = (0..3)
        .map(|i| {
            WorkerStateManager::new(
                format!("worker-{}", i),
                Some(dir.path().join(format!("worker_{}_state", i))),
                Box::new(
                    LocalWeightStore::new(dir.path().join("weights"), 10).unwrap(),
                ),
            )
            .unwrap()
        })
        .collect();

    let mut current_uri = initial_uri;
    let mut current_commitment = initial_commitment;
    let mut commitments_per_round: Vec<([u8; 32], [u8; 32])> = Vec::new();

    for round in 0..3u64 {
        // --- Workers download and verify weights ---
        let mut worker_models: Vec<MlpModel> = Vec::new();
        for worker in &mut workers {
            let bytes = worker
                .prepare_for_round(&current_uri, current_commitment)
                .unwrap();
            assert_eq!(
                compute_weight_commitment(&bytes),
                current_commitment,
                "Worker weight commitment mismatch at round {}",
                round
            );
            worker_models.push(model_from_bytes(&bytes));
        }

        // --- Workers train (perturb in different directions) ---
        for (i, model) in worker_models.iter_mut().enumerate() {
            let delta = 0.001 * (i as f64 + 1.0) * (round as f64 + 1.0);
            perturb_model(model, delta);
        }

        // --- Aggregator averages worker models ---
        let aggregated = average_models(&worker_models).unwrap();
        let agg_bytes = model_to_bytes(&aggregated, round + 1);

        // --- Record round completion ---
        let round_initial = controller.next_round_initial_commitment();
        assert_eq!(
            round_initial, current_commitment,
            "Next round initial commitment should match previous final"
        );

        let result = controller
            .complete_round(
                agg_bytes.clone(),
                0.5 - (round as f64 * 0.15),
                0.02 - (round as f64 * 0.005),
                100,
                3,
                Some([(round + 1) as u8; 32]),
                workers
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (format!("worker-{}", i), 90.0 + i as f64))
                    .collect(),
            )
            .unwrap();

        // Verify round result
        assert_eq!(result.round_number, round);
        assert_eq!(result.initial_commitment, current_commitment);
        assert_eq!(result.num_contributors, 3);

        commitments_per_round.push((result.initial_commitment, result.final_commitment));

        // --- Workers record completion ---
        for (i, worker) in workers.iter_mut().enumerate() {
            let worker_bytes = model_to_bytes(&worker_models[i], round + 1);
            worker.complete_round(worker_bytes, 100, None).unwrap();
        }

        current_uri = result.weights_uri;
        current_commitment = result.final_commitment;
    }

    // === Verify commitment chain ===
    assert_eq!(controller.completed_rounds(), 3);
    controller.validate_state().unwrap();

    let history = controller.round_history();
    assert_eq!(history.len(), 3);

    // First round's initial = original model commitment
    assert_eq!(history[0].initial_commitment, initial_commitment);

    // Each round's initial = previous round's final
    for i in 1..3 {
        assert_eq!(
            history[i].initial_commitment,
            history[i - 1].final_commitment,
            "Commitment chain broken between round {} and {}",
            i - 1,
            i,
        );
    }

    // All commitments are unique (model actually changed each round)
    let all_commitments: Vec<[u8; 32]> = std::iter::once(initial_commitment)
        .chain(history.iter().map(|r| r.final_commitment))
        .collect();
    for i in 0..all_commitments.len() {
        for j in (i + 1)..all_commitments.len() {
            assert_ne!(
                all_commitments[i], all_commitments[j],
                "Commitments at positions {} and {} should differ",
                i, j,
            );
        }
    }

    // === Verify workers tracked correctly ===
    for worker in &workers {
        assert_eq!(worker.last_completed_round(), 3);
        assert_eq!(worker.total_steps(), 300);
    }

    // === Verify training summary ===
    let summary = controller.training_summary();
    assert_eq!(summary.rounds_completed, 3);
    assert_eq!(summary.total_steps, 300);
    assert_eq!(summary.total_contributors, 9);
    assert_eq!(summary.initial_commitment, initial_commitment);
    assert_eq!(summary.current_commitment, current_commitment);
    // Loss should be decreasing
    assert!(summary.loss_history[0] > summary.loss_history[2]);
}

// ============================================================================
// Test 2: Weight persistence and reload across rounds
// ============================================================================

#[test]
fn test_weight_persistence_and_reload_across_rounds() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let initial_model = MlpModel::new_random(4, 8, 2, 42);
    let initial_bytes = model_to_bytes(&initial_model, 0);
    let initial_commitment = compute_weight_commitment(&initial_bytes);

    let mut controller = MultiRoundController::new(
        config.clone(),
        initial_commitment,
        Some(initial_bytes),
    )
    .unwrap();

    // Run 3 rounds, storing weights after each
    let mut round_results = Vec::new();
    let mut model = initial_model;
    for round in 0..3u64 {
        perturb_model(&mut model, 0.005 * (round as f64 + 1.0));
        let bytes = model_to_bytes(&model, round + 1);

        let result = controller
            .complete_round(bytes, 0.5 - round as f64 * 0.1, 0.01, 50, 2, None, HashMap::new())
            .unwrap();
        round_results.push(result);
    }

    // Verify each round's weights can be loaded from storage and match commitment
    let weight_store = LocalWeightStore::new(dir.path().join("weights"), 10).unwrap();
    for result in &round_results {
        let loaded = weight_store.load_weights(&result.weights_uri).unwrap();
        let loaded_commitment = compute_weight_commitment(&loaded);
        assert_eq!(
            loaded_commitment, result.final_commitment,
            "Stored weights for round {} don't match commitment",
            result.round_number,
        );
    }

    // Verify load_latest_weights returns the last round's weights
    let latest = controller.load_latest_weights().unwrap().unwrap();
    let latest_commitment = compute_weight_commitment(&latest);
    assert_eq!(latest_commitment, round_results[2].final_commitment);

    // Verify a new worker can download and verify each round's weights
    let mut worker = WorkerStateManager::new(
        "late-joiner".to_string(),
        Some(dir.path().join("late_joiner_state")),
        Box::new(LocalWeightStore::new(dir.path().join("weights"), 10).unwrap()),
    )
    .unwrap();

    // Late-joining worker downloads the latest weights
    let last_result = &round_results[2];
    let downloaded = worker
        .prepare_for_round(&last_result.weights_uri, last_result.final_commitment)
        .unwrap();
    assert_eq!(compute_weight_commitment(&downloaded), last_result.final_commitment);
}

// ============================================================================
// Test 3: Crash recovery mid-training
// ============================================================================

#[test]
fn test_crash_recovery_mid_multi_round_training() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let initial_model = MlpModel::new_random(4, 8, 2, 42);
    let initial_bytes = model_to_bytes(&initial_model, 0);
    let initial_commitment = compute_weight_commitment(&initial_bytes);

    // Phase 1: Run 2 rounds then "crash"
    let commitment_after_r2;
    let bytes_after_r2;
    {
        let mut controller = MultiRoundController::new(
            config.clone(),
            initial_commitment,
            Some(initial_bytes.clone()),
        )
        .unwrap();

        let mut model = initial_model.clone();

        // Round 0
        perturb_model(&mut model, 0.01);
        let r0_bytes = model_to_bytes(&model, 1);
        controller
            .complete_round(r0_bytes, 0.5, 0.02, 100, 3, None, HashMap::new())
            .unwrap();

        // Round 1
        perturb_model(&mut model, 0.01);
        bytes_after_r2 = model_to_bytes(&model, 2);
        let result = controller
            .complete_round(bytes_after_r2.clone(), 0.35, 0.015, 100, 3, None, HashMap::new())
            .unwrap();
        commitment_after_r2 = result.final_commitment;
    }
    // Controller dropped - simulates crash

    // Phase 2: Recover from snapshot
    let mut recovered_controller = MultiRoundController::new(
        config.clone(),
        [0u8; 32], // dummy, will be overwritten
        None,
    )
    .unwrap();

    let recovered = recovered_controller
        .recover_from_snapshot()
        .unwrap()
        .expect("Should recover from snapshot");

    assert_eq!(recovered.round_number, 2, "Should resume at round 2");
    assert_eq!(recovered.total_rounds_completed, 2);
    assert_eq!(recovered.commitment, commitment_after_r2);
    assert_eq!(recovered.weight_bytes, bytes_after_r2);
    assert_eq!(recovered_controller.current_round(), 2);

    // Phase 3: Continue with round 2 (third round overall)
    let mut model = model_from_bytes(&recovered.weight_bytes);
    perturb_model(&mut model, 0.01);
    let r2_bytes = model_to_bytes(&model, 3);
    let r2_commitment = compute_weight_commitment(&r2_bytes);

    let result = recovered_controller
        .complete_round(r2_bytes, 0.2, 0.01, 100, 3, None, HashMap::new())
        .unwrap();

    assert_eq!(result.round_number, 2);
    assert_eq!(result.initial_commitment, commitment_after_r2);
    assert_eq!(result.final_commitment, r2_commitment);
    assert_eq!(recovered_controller.current_round(), 3);
}

// ============================================================================
// Test 4: Worker crash recovery with commitment validation
// ============================================================================

#[test]
fn test_worker_crash_recovery_with_commitment_validation() {
    let dir = tempdir().unwrap();

    let initial_model = MlpModel::new_random(4, 8, 2, 42);
    let initial_bytes = model_to_bytes(&initial_model, 0);

    // Store initial weights
    let weight_store = LocalWeightStore::new(dir.path().join("weights"), 10).unwrap();
    let (_initial_uri, _initial_commitment) =
        weight_store.store_weights(0, &initial_bytes).unwrap();

    // Phase 1: Worker completes 2 rounds then crashes
    let final_bytes_r2;
    let final_commitment_r2;
    {
        let mut worker = WorkerStateManager::new(
            "worker-crash".to_string(),
            Some(dir.path().join("worker_state")),
            Box::new(LocalWeightStore::new(dir.path().join("weights"), 10).unwrap()),
        )
        .unwrap();

        let mut model = initial_model.clone();
        perturb_model(&mut model, 0.01);
        let r0_bytes = model_to_bytes(&model, 1);
        worker.complete_round(r0_bytes, 50, None).unwrap();

        perturb_model(&mut model, 0.01);
        final_bytes_r2 = model_to_bytes(&model, 2);
        final_commitment_r2 = compute_weight_commitment(&final_bytes_r2);
        worker
            .complete_round(final_bytes_r2.clone(), 50, Some(vec![0xAA; 16]))
            .unwrap();
    }
    // Worker dropped - crash

    // Phase 2: Recover worker
    let mut recovered_worker = WorkerStateManager::new(
        "worker-crash".to_string(),
        Some(dir.path().join("worker_state")),
        Box::new(LocalWeightStore::new(dir.path().join("weights"), 10).unwrap()),
    )
    .unwrap();

    let state = recovered_worker
        .recover()
        .unwrap()
        .expect("Should recover");

    assert_eq!(state.peer_id, "worker-crash");
    assert_eq!(state.last_completed_round, 2);
    assert_eq!(state.total_steps, 100);
    assert!(state.weight_bytes.is_some());
    assert_eq!(state.commitment, Some(final_commitment_r2));

    // Phase 3: Validate against on-chain commitment
    assert!(recovered_worker
        .validate_against_commitment(final_commitment_r2)
        .unwrap());

    // Wrong commitment should fail
    assert!(!recovered_worker
        .validate_against_commitment([0xFF; 32])
        .unwrap());
}

// ============================================================================
// Test 5: FedAvg with multi-round controller (E2E simulation)
// ============================================================================

#[test]
fn test_fedavg_three_rounds_e2e() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let initial_model = MlpModel::new_random(4, 8, 2, 42);
    let initial_bytes = model_to_bytes(&initial_model, 0);
    let initial_commitment = compute_weight_commitment(&initial_bytes);

    let mut controller = MultiRoundController::new(
        config,
        initial_commitment,
        Some(initial_bytes.clone()),
    )
    .unwrap();

    // Store initial weights
    let weight_store = LocalWeightStore::new(dir.path().join("weights"), 10).unwrap();
    let (mut current_uri, _) = weight_store.store_weights(999, &initial_bytes).unwrap();
    let mut current_commitment = initial_commitment;

    let num_workers = 3;
    let num_rounds = 3;

    for round in 0..num_rounds {
        // Each worker downloads weights, trains independently
        let mut worker_models = Vec::new();
        for w in 0..num_workers {
            // Verify downloaded weights match commitment
            let bytes = weight_store.load_weights(&current_uri).unwrap();
            assert_eq!(compute_weight_commitment(&bytes), current_commitment);

            let mut model = model_from_bytes(&bytes);
            // Each worker perturbs differently (simulates different data shards)
            let delta = 0.001 * ((w as f64) - 1.0) * (round as f64 + 1.0);
            perturb_model(&mut model, delta);
            worker_models.push(model);
        }

        // Aggregator averages
        let aggregated = average_models(&worker_models).unwrap();
        let agg_bytes = model_to_bytes(&aggregated, (round + 1) as u64);

        let result = controller
            .complete_round(
                agg_bytes,
                0.5 - (round as f64 * 0.1),
                0.02 - (round as f64 * 0.003),
                100,
                num_workers as u32,
                None,
                HashMap::new(),
            )
            .unwrap();

        current_uri = result.weights_uri;
        current_commitment = result.final_commitment;
    }

    // Verify 3 rounds completed with valid chain
    assert_eq!(controller.completed_rounds(), 3);
    controller.validate_state().unwrap();

    let history = controller.round_history();

    // Initial commitment matches
    assert_eq!(history[0].initial_commitment, initial_commitment);

    // Chain is unbroken
    for i in 1..history.len() {
        assert_eq!(
            history[i].initial_commitment,
            history[i - 1].final_commitment,
        );
    }

    // Loss decreased over rounds
    assert!(
        history[2].final_loss < history[0].final_loss,
        "Loss should decrease: {} < {}",
        history[2].final_loss,
        history[0].final_loss,
    );

    // Final model differs from initial
    let final_bytes = controller.load_latest_weights().unwrap().unwrap();
    let final_commitment = compute_weight_commitment(&final_bytes);
    assert_ne!(final_commitment, initial_commitment);
    assert_eq!(final_commitment, history[2].final_commitment);
}

// ============================================================================
// Test 6: Commitment mismatch is detected
// ============================================================================

#[test]
fn test_commitment_mismatch_rejected() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let initial_bytes = model_to_bytes(&MlpModel::new_random(4, 8, 2, 42), 0);
    let initial_commitment = compute_weight_commitment(&initial_bytes);

    let mut controller = MultiRoundController::new(
        config,
        initial_commitment,
        Some(initial_bytes),
    )
    .unwrap();

    // Complete round 0
    let final_bytes = model_to_bytes(&MlpModel::new_random(4, 8, 2, 99), 1);
    let result = controller
        .complete_round(final_bytes, 0.5, 0.01, 50, 2, None, HashMap::new())
        .unwrap();

    // Try to load with wrong commitment
    let wrong_commitment = [0xDE; 32];
    let load_result =
        controller.load_and_verify_weights(&result.weights_uri, wrong_commitment);
    assert!(
        load_result.is_err(),
        "Should reject weights with wrong commitment"
    );
}

// ============================================================================
// Test 7: Weight file corruption detected
// ============================================================================

#[test]
fn test_corrupted_weights_detected() {
    let dir = tempdir().unwrap();

    let weight_store = LocalWeightStore::new(dir.path().join("weights"), 10).unwrap();
    let bytes = model_to_bytes(&MlpModel::new_random(4, 8, 2, 42), 0);
    let (uri, _commitment) = weight_store.store_weights(0, &bytes).unwrap();

    // Corrupt the weight file on disk
    let weight_path = dir.path().join("weights").join("round_000000.weights");
    std::fs::write(&weight_path, b"corrupted data").unwrap();

    // Loading should fail because checksum won't match
    let result = weight_store.load_weights(&uri);
    assert!(
        result.is_err(),
        "Should detect corrupted weights via checksum mismatch"
    );
}
