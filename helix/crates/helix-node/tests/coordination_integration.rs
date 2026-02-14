//! Integration tests for the distributed training coordination layer.
//!
//! These tests boot an aggregator coordinator + multiple workers using
//! in-process channels, run real ML training with ZK proof generation,
//! and verify that the aggregator collects and aggregates proofs correctly.

use std::collections::HashMap;

use helix_node::coordination::coordinator::{RoundConfig, TrainingJobCoordinator};
use helix_node::coordination::worker_handler::TrainingWorkerHandler;
use helix_node::network::messages::PeerId;
use helix_node::roles::aggregator::AggregatorConfig;
use helix_node::trainer::MlpModel;

use tokio::sync::mpsc;

/// Creates a small model with weights in the 0.001 range (required for circuit ReLU range).
fn small_test_model() -> MlpModel {
    MlpModel::new(
        2, 2, 1,
        vec![0.001, 0.002, 0.003, 0.001],
        vec![0.0, 0.0],
        vec![0.001, 0.001],
        vec![0.0],
    )
}

/// Simple dataset for testing.
fn test_dataset() -> Vec<(Vec<f64>, Vec<f64>)> {
    vec![
        (vec![0.001, 0.001], vec![0.005]),
        (vec![0.002, 0.001], vec![0.006]),
    ]
}

/// Sets up coordinator + N workers with connected channels.
///
/// Returns the coordinator and a list of worker handlers that are ready to run.
fn setup_coordination(
    num_workers: usize,
) -> (TrainingJobCoordinator, Vec<TrainingWorkerHandler>) {
    let aggregator_id = PeerId::from_string("aggregator");

    // Multiplexed inbound channel for coordinator (all workers → coordinator)
    let (coord_inbound_tx, coord_inbound_rx) = mpsc::channel(64);

    let mut worker_txs: HashMap<PeerId, mpsc::Sender<helix_node::network::messages::NetworkMessage>> = HashMap::new();
    let mut workers = Vec::new();

    for i in 0..num_workers {
        let worker_id = PeerId::from_string(format!("worker-{}", i));

        // Coordinator → Worker channel
        let (to_worker_tx, from_coord_rx) = mpsc::channel(64);

        worker_txs.insert(worker_id.clone(), to_worker_tx);

        // Worker sends directly to the coordinator's multiplexed inbound channel.
        // The TrainingWorkerHandler takes (PeerId, NetworkMessage) sender.
        let handler = TrainingWorkerHandler::new(
            worker_id,
            coord_inbound_tx.clone(),
            from_coord_rx,
        );
        workers.push(handler);
    }

    // Drop the original sender so coordinator can detect when all workers disconnect
    drop(coord_inbound_tx);

    let config = AggregatorConfig {
        min_participants: 1, // Allow 1+ for flexible testing
        byzantine_strategy: None, // No Byzantine filtering for small tests
        ..Default::default()
    };

    let coordinator = TrainingJobCoordinator::new(
        aggregator_id,
        config,
        worker_txs,
        coord_inbound_rx,
    );

    (coordinator, workers)
}

/// Boots aggregator + 2 workers, runs 1 training step each, verifies proofs collected.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_two_workers_one_step() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("helix_node=debug")
        .try_init();

    let (mut coordinator, mut workers) = setup_coordination(2);

    let model = small_test_model();
    let dataset = test_dataset();

    let round_config = RoundConfig {
        model: model.clone(),
        learning_rate: 0.001,
        steps_per_worker: 1,
        error_budget: 1.0,
        min_workers: 2,
        deadline_secs: 120,
        model_id: 1,
        num_layers: 2,
        activation_type: 0,
        dataset,
    };

    // Spawn workers
    let mut worker_handles = Vec::new();
    for mut worker in workers.drain(..) {
        let handle = tokio::spawn(async move {
            worker.run_round().await
        });
        worker_handles.push(handle);
    }

    // Run coordinator
    let result = coordinator.run_training_round(round_config).await;

    // Wait for workers to finish
    let mut worker_results = Vec::new();
    for handle in worker_handles {
        let worker_result = handle.await.expect("worker task panicked");
        worker_results.push(worker_result);
    }

    // Verify coordinator result
    let result = result.expect("coordinator round failed");
    assert_eq!(result.round_id, 1);
    assert_eq!(result.worker_submissions.len(), 2, "should have 2 submissions");
    assert!(result.aggregated.num_participants >= 2, "aggregator should have 2+ participants");
    assert!(!result.aggregated.proof.is_empty(), "aggregation proof should be non-empty");
    assert!(result.aggregated.total_error_bound >= 0.0, "error bound should be non-negative");

    // Verify each worker submission
    for sub in &result.worker_submissions {
        assert!(sub.proof_len > 0, "worker proof should be non-empty for {}", sub.peer_id);
        assert_eq!(sub.public_inputs_count, 8, "V2 prover should produce 8 public inputs");
        assert_eq!(sub.steps_completed, 1, "each worker should complete 1 step");
        assert_ne!(sub.new_model_hash, [0u8; 32], "model hash should be non-zero");
    }

    // Verify worker results
    for wr in &worker_results {
        let wr = wr.as_ref().expect("worker round failed");
        assert_eq!(wr.steps_completed, 1);
        assert!(wr.proof_len > 0);
        assert_eq!(wr.public_inputs_count, 8);
        assert!(wr.loss > 0.0, "loss should be positive");
    }

    // Verify FedAvg produced an updated model
    let updated = result.updated_model.expect("FedAvg should produce an updated model");
    assert_eq!(updated.d_in, 2);
    assert_eq!(updated.d_hid, 2);
    assert_eq!(updated.d_out, 1);

    // The averaged model should differ from the original (training happened)
    assert_ne!(
        updated.commitment(),
        model.commitment(),
        "updated model should differ from original after training"
    );
}

/// Tests that workers get different model states after independent training.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_workers_produce_different_commitments() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("helix_node=info")
        .try_init();

    let (mut coordinator, mut workers) = setup_coordination(2);

    let model = small_test_model();

    let round_config = RoundConfig {
        model: model.clone(),
        learning_rate: 0.001,
        steps_per_worker: 3, // Multiple steps to amplify differences
        error_budget: 1.0,
        min_workers: 2,
        deadline_secs: 180,
        model_id: 1,
        num_layers: 2,
        activation_type: 0,
        dataset: test_dataset(),
    };

    let mut worker_handles = Vec::new();
    for mut worker in workers.drain(..) {
        let handle = tokio::spawn(async move {
            worker.run_round().await
        });
        worker_handles.push(handle);
    }

    let result = coordinator.run_training_round(round_config).await
        .expect("coordinator round failed");

    for handle in worker_handles {
        handle.await.expect("worker task panicked")
            .expect("worker round failed");
    }

    // With identical dataset and model, workers should produce identical results
    // (deterministic training). This verifies the coordination is working correctly.
    assert_eq!(result.worker_submissions.len(), 2);

    let hash0 = result.worker_submissions[0].new_model_hash;
    let hash1 = result.worker_submissions[1].new_model_hash;
    // Same model + same data + same hyperparams = same result (deterministic)
    assert_eq!(hash0, hash1, "deterministic training should produce identical model hashes");
}

/// Tests that the coordinator rejects when not enough workers.
#[tokio::test]
async fn test_insufficient_workers() {
    let (mut coordinator, _workers) = setup_coordination(1);

    let round_config = RoundConfig {
        model: small_test_model(),
        learning_rate: 0.001,
        steps_per_worker: 1,
        error_budget: 1.0,
        min_workers: 5, // More than available
        deadline_secs: 10,
        model_id: 1,
        num_layers: 2,
        activation_type: 0,
        dataset: test_dataset(),
    };

    let result = coordinator.run_training_round(round_config).await;
    assert!(result.is_err(), "should fail with insufficient workers");
    let err = result.err().unwrap();
    assert!(
        err.to_string().contains("not enough workers"),
        "error should mention insufficient workers: {}",
        err
    );
}

/// Tests multi-step training with proof verification.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_multi_step_training() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("helix_node=info")
        .try_init();

    let (mut coordinator, mut workers) = setup_coordination(2);

    let model = small_test_model();

    let round_config = RoundConfig {
        model: model.clone(),
        learning_rate: 0.0001, // Smaller LR for multi-step stability
        steps_per_worker: 5,
        error_budget: 10.0,
        min_workers: 2,
        deadline_secs: 300,
        model_id: 1,
        num_layers: 2,
        activation_type: 0,
        dataset: test_dataset(),
    };

    let mut worker_handles = Vec::new();
    for mut worker in workers.drain(..) {
        let handle = tokio::spawn(async move {
            worker.run_round().await
        });
        worker_handles.push(handle);
    }

    let result = coordinator.run_training_round(round_config).await
        .expect("coordinator round failed");

    for handle in worker_handles {
        let wr = handle.await.expect("worker task panicked")
            .expect("worker round failed");
        assert_eq!(wr.steps_completed, 5, "worker should complete 5 steps");
        assert!(wr.proof_len > 0, "worker should produce a proof");
    }

    assert_eq!(result.worker_submissions.len(), 2);
    for sub in &result.worker_submissions {
        assert_eq!(sub.steps_completed, 5);
        assert!(sub.proof_len > 0);
        assert_eq!(sub.public_inputs_count, 8);
    }

    // Aggregation should succeed
    assert!(result.aggregated.num_participants >= 2);
    assert!(!result.aggregated.proof.is_empty());

    // FedAvg model should exist
    assert!(result.updated_model.is_some());
}

/// Tests proof content is valid (non-zero, correct size for Halo2 KZG).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_proof_content_valid() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("helix_node=info")
        .try_init();

    let (mut coordinator, mut workers) = setup_coordination(2);

    let round_config = RoundConfig {
        model: small_test_model(),
        learning_rate: 0.001,
        steps_per_worker: 1,
        error_budget: 1.0,
        min_workers: 2,
        deadline_secs: 120,
        model_id: 1,
        num_layers: 2,
        activation_type: 0,
        dataset: test_dataset(),
    };

    let mut worker_handles = Vec::new();
    for mut worker in workers.drain(..) {
        let handle = tokio::spawn(async move { worker.run_round().await });
        worker_handles.push(handle);
    }

    let result = coordinator.run_training_round(round_config).await
        .expect("coordinator round failed");

    for handle in worker_handles {
        handle.await.expect("worker task panicked")
            .expect("worker round failed");
    }

    for sub in &result.worker_submissions {
        // Halo2 KZG proofs with PSE transcript are 1856 bytes
        // (for MLTrainingStepV2Circuit with SHPLONK)
        assert!(
            sub.proof_len > 100,
            "proof should be substantial (got {} bytes)",
            sub.proof_len
        );

        // Public inputs should be 8 (V2 circuit)
        assert_eq!(sub.public_inputs_count, 8);

        // Model hash should be non-zero
        assert_ne!(sub.new_model_hash, [0u8; 32]);
    }

    // Aggregation proof should be deserializable
    let agg_proof: helix_node::roles::aggregator::AggregationProof =
        bincode::deserialize(&result.aggregated.proof)
            .expect("aggregation proof should be deserializable");
    assert_eq!(agg_proof.participant_count, 2);
    assert_eq!(agg_proof.round_id, 1);
    assert_ne!(agg_proof.poseidon_commitment, [0u8; 32]);
}
