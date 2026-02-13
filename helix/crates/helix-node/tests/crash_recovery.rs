//! Crash Recovery Integration Test.
//!
//! Tests that the aggregator can persist full state after each round and
//! resume from the exact point after a simulated crash.
//!
//! Test flow:
//! 1. Start aggregator state -> run 3 rounds -> checkpoint after each
//! 2. Simulate crash (drop all in-memory state)
//! 3. Restore from checkpoint -> verify resumes at round 3
//! 4. Run 2 more rounds -> verify total is 5
//! 5. Verify checkpoint pruning keeps only N checkpoints
//!
//! Also tests worker checkpoint persistence and restore.

use std::collections::HashMap;

use helix_core::ModelCheckpoint;
use helix_node::trainer::MlpModel;
use helix_node::training::persistence::{AggregatorSnapshot, StatePersistence, WorkerSnapshot};
use tempfile::tempdir;

/// Helper: perturb all weights in an MlpModel by a fixed delta.
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

/// Helper: perturb all weights with a position-dependent delta.
fn perturb_model_indexed(model: &mut MlpModel, round: u64) {
    let mut idx = 0usize;
    for w in model.w1.iter_mut() {
        idx += 1;
        *w += 0.001 * (round as f64) * (idx as f64);
    }
    for b in model.b1.iter_mut() {
        idx += 1;
        *b += 0.001 * (round as f64) * (idx as f64);
    }
    for w in model.w2.iter_mut() {
        idx += 1;
        *w += 0.001 * (round as f64) * (idx as f64);
    }
    for b in model.b2.iter_mut() {
        idx += 1;
        *b += 0.001 * (round as f64) * (idx as f64);
    }
}

// ============================================================================
// Aggregator Crash Recovery
// ============================================================================

#[test]
fn test_aggregator_crash_recovery_full_cycle() {
    // Simulate: 3 rounds -> crash -> restore -> 2 more rounds -> verify total = 5
    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 5, "aggregator").unwrap();

    let d_in = 4;
    let d_hid = 8;
    let d_out = 2;
    let model_seed = 42;
    let learning_rate = 0.01;

    // Phase 1: Run 3 rounds, checkpointing after each
    let mut model = MlpModel::new_random(d_in, d_hid, d_out, model_seed);
    let mut round_number = 0u64;
    let mut total_rounds = 0u64;
    let mut error_bounds: HashMap<u64, f64> = HashMap::new();
    let mut proof_hashes: HashMap<u64, Vec<[u8; 32]>> = HashMap::new();

    for _ in 0..3 {
        perturb_model(&mut model, 0.001);

        round_number += 1;
        total_rounds += 1;

        error_bounds.insert(round_number, 0.05 * round_number as f64);
        proof_hashes.insert(round_number, vec![model.commitment()]);

        let ckpt = model.to_checkpoint(round_number);
        let ckpt_bytes = ckpt.to_bytes().unwrap();

        let mut snapshot = AggregatorSnapshot::new(
            ckpt_bytes,
            round_number,
            total_rounds,
            d_in, d_hid, d_out,
            model_seed,
            learning_rate,
        );
        snapshot.participants = vec!["worker-1".to_string(), "worker-2".to_string()];
        snapshot.error_bounds = error_bounds.clone();
        snapshot.proof_hashes = proof_hashes.clone();

        persistence.save_aggregator(&snapshot).unwrap();
    }

    assert_eq!(round_number, 3);
    assert_eq!(total_rounds, 3);

    // Note: to_checkpoint() converts f64→f32 and from_checkpoint() converts f32→f64,
    // so the commitment after a round-trip won't match the original f64 commitment.
    // Instead we verify the round-trip is idempotent: save→restore→save→restore = same.
    let ckpt_rt = model.to_checkpoint(round_number);
    let model_rt = MlpModel::from_checkpoint(&ckpt_rt).unwrap();
    let commitment_after_roundtrip = model_rt.commitment();

    // Phase 2: Simulate crash (drop all in-memory state)
    drop(model);
    drop(model_rt);
    drop(error_bounds);
    drop(proof_hashes);

    // Phase 3: Restore from checkpoint
    let restored_snapshot = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(restored_snapshot.last_completed_round, 3);
    assert_eq!(restored_snapshot.total_rounds_completed, 3);
    assert_eq!(restored_snapshot.d_in, d_in);
    assert_eq!(restored_snapshot.d_hid, d_hid);
    assert_eq!(restored_snapshot.d_out, d_out);
    assert_eq!(restored_snapshot.model_seed, model_seed);
    assert_eq!(restored_snapshot.participants.len(), 2);
    assert_eq!(restored_snapshot.error_bounds.len(), 3);
    assert_eq!(restored_snapshot.proof_hashes.len(), 3);

    let restored_ckpt =
        ModelCheckpoint::from_bytes(&restored_snapshot.model_checkpoint_bytes).unwrap();
    let mut restored_model = MlpModel::from_checkpoint(&restored_ckpt).unwrap();

    assert_eq!(
        restored_model.commitment(),
        commitment_after_roundtrip,
        "Model commitment mismatch after restore (idempotent round-trip)"
    );

    // Resume state
    let mut round_number = restored_snapshot.last_completed_round;
    let mut total_rounds = restored_snapshot.total_rounds_completed;
    let mut error_bounds = restored_snapshot.error_bounds;
    let mut proof_hashes = restored_snapshot.proof_hashes;

    // Phase 4: Run 2 more rounds
    for _ in 0..2 {
        perturb_model(&mut restored_model, 0.001);

        round_number += 1;
        total_rounds += 1;

        error_bounds.insert(round_number, 0.05 * round_number as f64);
        proof_hashes.insert(round_number, vec![restored_model.commitment()]);

        let ckpt = restored_model.to_checkpoint(round_number);
        let ckpt_bytes = ckpt.to_bytes().unwrap();

        let mut snapshot = AggregatorSnapshot::new(
            ckpt_bytes,
            round_number,
            total_rounds,
            d_in, d_hid, d_out,
            model_seed,
            learning_rate,
        );
        snapshot.participants = vec!["worker-1".to_string(), "worker-2".to_string()];
        snapshot.error_bounds = error_bounds.clone();
        snapshot.proof_hashes = proof_hashes.clone();

        persistence.save_aggregator(&snapshot).unwrap();
    }

    // Phase 5: Verify final state
    assert_eq!(round_number, 5, "Should be at round 5");
    assert_eq!(total_rounds, 5, "Should have 5 total rounds");
    assert_eq!(error_bounds.len(), 5, "Should have 5 error bound entries");
    assert_eq!(proof_hashes.len(), 5, "Should have 5 proof hash entries");

    let final_snapshot = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(final_snapshot.last_completed_round, 5);
    assert_eq!(final_snapshot.total_rounds_completed, 5);

    let fresh_model = MlpModel::new_random(d_in, d_hid, d_out, model_seed);
    assert_ne!(
        restored_model.commitment(),
        fresh_model.commitment(),
        "Restored model should differ from fresh model"
    );
}

#[test]
fn test_aggregator_checkpoint_pruning() {
    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 3, "aggregator").unwrap();

    let model = MlpModel::new_random(4, 8, 2, 42);

    for i in 1..=7u64 {
        let ckpt = model.to_checkpoint(i);
        let ckpt_bytes = ckpt.to_bytes().unwrap();

        let mut snapshot = AggregatorSnapshot::new(ckpt_bytes, i, i, 4, 8, 2, 42, 0.01);
        snapshot.timestamp += i;
        persistence.save_aggregator(&snapshot).unwrap();
    }

    let files = persistence.list_checkpoints().unwrap();
    assert_eq!(files.len(), 3, "Expected 3 checkpoints after pruning, got {}", files.len());

    let latest = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(latest.last_completed_round, 7);
}

// ============================================================================
// Worker Crash Recovery
// ============================================================================

#[test]
fn test_worker_crash_recovery() {
    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 5, "worker_test").unwrap();

    let model = MlpModel::new_random(4, 8, 2, 42);

    let ckpt = model.to_checkpoint(10);
    let ckpt_bytes = ckpt.to_bytes().unwrap();

    let mut snapshot = WorkerSnapshot::new("worker-1".to_string(), 3, 10);
    snapshot.model_checkpoint_bytes = Some(ckpt_bytes);

    persistence.save_worker(&snapshot).unwrap();

    let restored = persistence.load_latest_worker().unwrap().unwrap();
    assert_eq!(restored.last_completed_round, 3);
    assert_eq!(restored.steps_completed, 10);
    assert_eq!(restored.peer_id, "worker-1");
    assert!(restored.model_checkpoint_bytes.is_some());

    let restored_ckpt =
        ModelCheckpoint::from_bytes(restored.model_checkpoint_bytes.as_ref().unwrap()).unwrap();
    let restored_model = MlpModel::from_checkpoint(&restored_ckpt).unwrap();

    // f64→f32→f64 round-trip loses precision, so compare against same round-trip
    let model_rt = MlpModel::from_checkpoint(&model.to_checkpoint(10)).unwrap();
    assert_eq!(
        restored_model.commitment(),
        model_rt.commitment(),
        "Worker model commitment should match after round-trip"
    );
}

#[test]
fn test_worker_sends_last_round_on_reconnect() {
    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 5, "worker_reconn").unwrap();

    for round in 1..=5u64 {
        let snapshot = WorkerSnapshot::new("worker-1".to_string(), round, round * 3);
        persistence.save_worker(&snapshot).unwrap();
    }

    let restored = persistence.load_latest_worker().unwrap().unwrap();
    assert_eq!(restored.last_completed_round, 5);
    assert_eq!(restored.steps_completed, 15);
}

// ============================================================================
// Multi-crash recovery (crash twice, still recovers)
// ============================================================================

#[test]
fn test_double_crash_recovery() {
    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 10, "aggregator").unwrap();

    let mut model = MlpModel::new_random(4, 8, 2, 42);

    // Run 2 rounds
    for i in 1..=2u64 {
        perturb_model(&mut model, 0.001);
        let ckpt = model.to_checkpoint(i);
        let snapshot = AggregatorSnapshot::new(ckpt.to_bytes().unwrap(), i, i, 4, 8, 2, 42, 0.01);
        persistence.save_aggregator(&snapshot).unwrap();
    }

    // First crash + restore
    let snap1 = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(snap1.last_completed_round, 2);
    let ckpt1 = ModelCheckpoint::from_bytes(&snap1.model_checkpoint_bytes).unwrap();
    let mut model = MlpModel::from_checkpoint(&ckpt1).unwrap();

    // Run 2 more rounds
    for i in 3..=4u64 {
        perturb_model(&mut model, 0.001);
        let ckpt = model.to_checkpoint(i);
        let snapshot = AggregatorSnapshot::new(ckpt.to_bytes().unwrap(), i, i, 4, 8, 2, 42, 0.01);
        persistence.save_aggregator(&snapshot).unwrap();
    }

    // Second crash + restore
    let snap2 = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(snap2.last_completed_round, 4);
    let ckpt2 = ModelCheckpoint::from_bytes(&snap2.model_checkpoint_bytes).unwrap();
    let mut model = MlpModel::from_checkpoint(&ckpt2).unwrap();

    // Run 1 final round
    perturb_model(&mut model, 0.001);
    let ckpt = model.to_checkpoint(5);
    let snapshot = AggregatorSnapshot::new(ckpt.to_bytes().unwrap(), 5, 5, 4, 8, 2, 42, 0.01);
    persistence.save_aggregator(&snapshot).unwrap();

    let final_snap = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(final_snap.last_completed_round, 5);
}

// ============================================================================
// Concurrent aggregator + worker checkpoint isolation
// ============================================================================

#[test]
fn test_aggregator_and_worker_checkpoints_isolated() {
    let dir = tempdir().unwrap();

    let agg_persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 5, "aggregator").unwrap();
    let worker_persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 5, "worker_0").unwrap();

    let agg_snapshot = AggregatorSnapshot::new(vec![1, 2, 3], 10, 10, 4, 8, 2, 42, 0.01);
    agg_persistence.save_aggregator(&agg_snapshot).unwrap();

    let worker_snapshot = WorkerSnapshot::new("worker-0".to_string(), 10, 50);
    worker_persistence.save_worker(&worker_snapshot).unwrap();

    let loaded_agg = agg_persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(loaded_agg.last_completed_round, 10);

    let loaded_worker = worker_persistence.load_latest_worker().unwrap().unwrap();
    assert_eq!(loaded_worker.steps_completed, 50);

    let agg_files = agg_persistence.list_checkpoints().unwrap();
    let worker_files = worker_persistence.list_checkpoints().unwrap();
    assert_eq!(agg_files.len(), 1);
    assert_eq!(worker_files.len(), 1);
}

// ============================================================================
// Model weight continuity across checkpoints
// ============================================================================

#[test]
fn test_model_weights_continuous_across_checkpoints() {
    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 10, "aggregator").unwrap();

    let mut model = MlpModel::new_random(4, 8, 2, 42);
    let mut commitments_after_roundtrip = Vec::new();

    for i in 1..=5u64 {
        perturb_model_indexed(&mut model, i);

        let ckpt = model.to_checkpoint(i);
        // Track the commitment after checkpoint round-trip (f64→f32→f64)
        let model_rt = MlpModel::from_checkpoint(&ckpt).unwrap();
        commitments_after_roundtrip.push(model_rt.commitment());

        let snapshot = AggregatorSnapshot::new(
            ckpt.to_bytes().unwrap(), i, i, 4, 8, 2, 42, 0.01,
        );
        persistence.save_aggregator(&snapshot).unwrap();
    }

    // All round-tripped commitments should be different
    for i in 0..commitments_after_roundtrip.len() {
        for j in (i + 1)..commitments_after_roundtrip.len() {
            assert_ne!(
                commitments_after_roundtrip[i], commitments_after_roundtrip[j],
                "Commitments at rounds {} and {} should differ",
                i + 1, j + 1
            );
        }
    }

    // Restore and verify we get the round 5 model
    let snap = persistence.load_latest_aggregator().unwrap().unwrap();
    let ckpt = ModelCheckpoint::from_bytes(&snap.model_checkpoint_bytes).unwrap();
    let restored = MlpModel::from_checkpoint(&ckpt).unwrap();

    assert_eq!(
        restored.commitment(),
        commitments_after_roundtrip[4],
        "Restored model should have round 5 commitment"
    );
}

// ============================================================================
// Graceful shutdown saves state
// ============================================================================

#[test]
fn test_graceful_shutdown_preserves_state() {
    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 5, "aggregator").unwrap();

    let model = MlpModel::new_random(4, 8, 2, 42);

    let ckpt = model.to_checkpoint(3);
    let mut snapshot = AggregatorSnapshot::new(ckpt.to_bytes().unwrap(), 3, 3, 4, 8, 2, 42, 0.01);
    snapshot.participants = vec!["w1".to_string(), "w2".to_string(), "w3".to_string()];
    snapshot.error_bounds.insert(1, 0.01);
    snapshot.error_bounds.insert(2, 0.008);
    snapshot.error_bounds.insert(3, 0.006);
    snapshot.last_on_chain_proof_hash = Some([0xAA; 32]);
    snapshot.on_chain_model_id = Some(42);

    persistence.save_aggregator(&snapshot).unwrap();

    // New process starts, loads state
    let new_persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 5, "aggregator").unwrap();
    let restored = new_persistence.load_latest_aggregator().unwrap().unwrap();

    assert_eq!(restored.last_completed_round, 3);
    assert_eq!(restored.total_rounds_completed, 3);
    assert_eq!(restored.participants.len(), 3);
    assert_eq!(restored.error_bounds.len(), 3);
    assert_eq!(restored.last_on_chain_proof_hash, Some([0xAA; 32]));
    assert_eq!(restored.on_chain_model_id, Some(42));
    assert!(
        (restored.error_bounds[&3] - 0.006).abs() < 1e-10,
        "Error bound for round 3 should be 0.006"
    );
}

// ============================================================================
// FedAvg simulation with crash recovery
// ============================================================================

#[test]
fn test_fedavg_crash_recovery_with_real_aggregation() {
    use helix_node::trainer::average_models;

    let dir = tempdir().unwrap();
    let persistence =
        StatePersistence::new(Some(dir.path().to_path_buf()), 10, "aggregator").unwrap();

    let d_in = 4;
    let d_hid = 8;
    let d_out = 2;

    let mut aggregator_model = MlpModel::new_random(d_in, d_hid, d_out, 42);
    let mut round_number = 0u64;

    // Phase 1: Run 3 rounds with real FedAvg
    for _ in 0..3 {
        let mut worker1 = aggregator_model.clone();
        let mut worker2 = aggregator_model.clone();

        // Worker 1: perturb weights up
        perturb_model(&mut worker1, 0.01);
        // Worker 2: perturb weights down
        perturb_model(&mut worker2, -0.005);

        let averaged = average_models(&[worker1, worker2]).unwrap();
        aggregator_model = averaged;
        round_number += 1;

        let ckpt = aggregator_model.to_checkpoint(round_number);
        let snapshot = AggregatorSnapshot::new(
            ckpt.to_bytes().unwrap(),
            round_number, round_number,
            d_in, d_hid, d_out, 42, 0.01,
        );
        persistence.save_aggregator(&snapshot).unwrap();
    }

    // Compute round-trip commitment (f64→f32→f64 changes weights)
    let commitment_at_round_3_rt =
        MlpModel::from_checkpoint(&aggregator_model.to_checkpoint(round_number))
            .unwrap()
            .commitment();

    // Crash!
    drop(aggregator_model);

    // Restore
    let snap = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(snap.last_completed_round, 3);
    let ckpt = ModelCheckpoint::from_bytes(&snap.model_checkpoint_bytes).unwrap();
    let mut aggregator_model = MlpModel::from_checkpoint(&ckpt).unwrap();
    assert_eq!(aggregator_model.commitment(), commitment_at_round_3_rt);

    // Phase 2: Run 2 more rounds
    for _ in 0..2 {
        let mut worker1 = aggregator_model.clone();
        let mut worker2 = aggregator_model.clone();

        perturb_model(&mut worker1, 0.01);
        perturb_model(&mut worker2, -0.005);

        let averaged = average_models(&[worker1, worker2]).unwrap();
        aggregator_model = averaged;
        round_number += 1;

        let ckpt = aggregator_model.to_checkpoint(round_number);
        let snapshot = AggregatorSnapshot::new(
            ckpt.to_bytes().unwrap(),
            round_number, round_number,
            d_in, d_hid, d_out, 42, 0.01,
        );
        persistence.save_aggregator(&snapshot).unwrap();
    }

    assert_eq!(round_number, 5);
    let final_snap = persistence.load_latest_aggregator().unwrap().unwrap();
    assert_eq!(final_snap.last_completed_round, 5);

    let fresh = MlpModel::new_random(d_in, d_hid, d_out, 42);
    assert_ne!(
        aggregator_model.commitment(),
        fresh.commitment(),
        "Model should differ from fresh after 5 rounds of FedAvg"
    );
}
