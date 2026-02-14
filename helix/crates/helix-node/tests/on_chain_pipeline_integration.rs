//! On-Chain Pipeline Integration Tests.
//!
//! Tests the aggregator's proof-to-chain pipeline including:
//! - ChainConfig construction and validation
//! - Proof queue processing and marking
//! - AggregatorNode round completion with on-chain pipeline
//! - VerifyAll policy configuration via model dimensions
//! - RLC aggregation proof building from CollectedGradients
//!
//! These tests run without a real chain (unit-style integration),
//! verifying the data flow and configuration correctness.

use std::sync::Arc;

use helix_node::config::{ChainConfig, NodeConfig, NodeRole};
use helix_node::network::messages::PeerId;
use helix_node::on_chain_pipeline::PipelineState;
use helix_node::roles::aggregator::{AggregatorConfig, AggregatorNode, CollectedGradient};
use helix_node::roles::verifier::{VerificationPolicy, VerifierConfig, VerifierNode};
use helix_node::api::rpc::QueuedProof;

// ============================================================================
// ChainConfig Tests
// ============================================================================

#[test]
fn test_chain_config_from_json() {
    let json = r#"{
        "coordinator_address": "0xDEADBEEF",
        "d_in": 4,
        "d_hid": 8,
        "d_out": 2
    }"#;
    let config: ChainConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.coordinator_address, "0xDEADBEEF");
    assert_eq!(config.d_in, 4);
    assert_eq!(config.d_hid, 8);
    assert_eq!(config.d_out, 2);
    assert!(config.model_id.is_none());
    assert_eq!(config.round_duration_secs, 600);
    assert_eq!(config.proof_queue_interval_secs, 5);
}

#[test]
fn test_chain_config_with_model_id() {
    let json = r#"{
        "coordinator_address": "0x1234",
        "model_id": 42,
        "d_in": 2,
        "d_hid": 2,
        "d_out": 1
    }"#;
    let config: ChainConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.model_id, Some(42));
}

#[test]
fn test_node_config_with_chain() {
    let json = r#"{
        "listen_addr": "0.0.0.0:9000",
        "role": "aggregator",
        "rpc_url": "http://localhost:8545",
        "chain": {
            "coordinator_address": "0xABC",
            "d_in": 2,
            "d_hid": 2,
            "d_out": 1,
            "model_id": 1
        }
    }"#;
    let config: NodeConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.role, NodeRole::Aggregator);
    assert!(config.chain.is_some());
    let chain = config.chain.unwrap();
    assert_eq!(chain.coordinator_address, "0xABC");
    assert_eq!(chain.model_id, Some(1));
    assert_eq!(chain.d_in, 2);
}

#[test]
fn test_node_config_without_chain() {
    let config = NodeConfig::default();
    assert!(config.chain.is_none());
}

#[test]
fn test_chain_config_serialization_roundtrip() {
    let config = ChainConfig {
        coordinator_address: "0x1234567890abcdef".to_string(),
        model_ipfs_hash: "QmTest123".to_string(),
        model_id: Some(5),
        min_stake_wei: "2000000000000000000".to_string(),
        stake_amount_wei: "3000000000000000000".to_string(),
        round_duration_secs: 300,
        d_in: 4,
        d_hid: 8,
        d_out: 2,
        proof_queue_interval_secs: 10,
        watcher: Default::default(),
    };
    let json = serde_json::to_string(&config).unwrap();
    let loaded: ChainConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.coordinator_address, config.coordinator_address);
    assert_eq!(loaded.model_id, Some(5));
    assert_eq!(loaded.d_in, 4);
    assert_eq!(loaded.round_duration_secs, 300);
}

// ============================================================================
// VerifyAll Policy Tests
// ============================================================================

#[test]
fn test_verifier_config_for_model_sets_verify_all() {
    let config = VerifierConfig::for_model(2, 2, 1);
    assert_eq!(config.policy, VerificationPolicy::VerifyAll);
    assert_eq!(config.model_dims, Some((2, 2, 1)));
}

#[test]
fn test_verifier_config_demo_sets_verify_all() {
    let config = VerifierConfig::demo();
    assert_eq!(config.policy, VerificationPolicy::VerifyAll);
    assert_eq!(config.model_dims, Some((2, 2, 1)));
}

#[test]
fn test_default_verifier_is_verify_all() {
    // Secure-by-default: VerifierConfig::default() now uses VerifyAll
    let config = VerifierConfig::default();
    assert_eq!(config.policy, VerificationPolicy::VerifyAll);
    assert!(config.model_dims.is_some());
}

#[test]
fn test_structural_constructor_for_testing() {
    // Use VerifierConfig::structural() for test environments
    let config = VerifierConfig::structural();
    assert_eq!(config.policy, VerificationPolicy::Structural);
    assert!(config.model_dims.is_none());
}

#[tokio::test]
async fn test_verifier_node_with_model_dims_rejects_fake_proof() {
    // VerifyAll with model_dims should reject random bytes (proves real Halo2 is active)
    let config = VerifierConfig::for_model(2, 2, 1);
    let node = VerifierNode::new(PeerId::random(), config);

    let fake_proof = vec![0xAB; 1856]; // Correct size but wrong content
    let inputs = helix_node::sc_client::TrainingProofInputs {
        old_hash_lo: ethers::types::U256::from(1),
        old_hash_hi: ethers::types::U256::from(2),
        new_hash_lo: ethers::types::U256::from(3),
        new_hash_hi: ethers::types::U256::from(4),
        loss: ethers::types::U256::from(100),
        error_bound: ethers::types::U256::from(10),
        step_number: ethers::types::U256::from(1),
        error_checksum: ethers::types::U256::zero(),
    };

    let result = node.verify_proof("test-1".into(), &fake_proof, &inputs).await;
    assert!(
        !result.is_valid(),
        "Fake proof should be rejected by VerifyAll"
    );
}

// ============================================================================
// Pipeline State Tests
// ============================================================================

#[test]
fn test_pipeline_state_initial() {
    let state = PipelineState {
        model_id: None,
        staked: false,
        current_round: 0,
        proofs_submitted: 0,
    };
    assert!(state.model_id.is_none());
    assert!(!state.staked);
}

#[test]
fn test_pipeline_state_after_init() {
    let state = PipelineState {
        model_id: Some(42),
        staked: true,
        current_round: 5,
        proofs_submitted: 10,
    };
    assert_eq!(state.model_id, Some(42));
    assert!(state.staked);
    assert_eq!(state.current_round, 5);
    assert_eq!(state.proofs_submitted, 10);
}

// ============================================================================
// Proof Queue Tests
// ============================================================================

#[test]
fn test_proof_queue_unsubmitted_filtering() {
    let queue = vec![
        QueuedProof {
            model_id: 1,
            round_id: 1,
            proof_hex: "0xaabb".to_string(),
            public_inputs_hex: vec!["0x01".to_string()],
            submitted: false,
            tx_hash: None,
        },
        QueuedProof {
            model_id: 1,
            round_id: 2,
            proof_hex: "0xccdd".to_string(),
            public_inputs_hex: vec!["0x02".to_string()],
            submitted: true,
            tx_hash: Some("0xabc".to_string()),
        },
        QueuedProof {
            model_id: 1,
            round_id: 3,
            proof_hex: "0xeeff".to_string(),
            public_inputs_hex: vec!["0x03".to_string()],
            submitted: false,
            tx_hash: None,
        },
    ];

    let unsubmitted: Vec<_> = queue
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.submitted)
        .collect();

    assert_eq!(unsubmitted.len(), 2);
    assert_eq!(unsubmitted[0].0, 0); // index 0
    assert_eq!(unsubmitted[1].0, 2); // index 2
}

#[test]
fn test_proof_queue_marking_submitted() {
    let queue = Arc::new(parking_lot::RwLock::new(vec![
        QueuedProof {
            model_id: 1,
            round_id: 1,
            proof_hex: "0xdeadbeef".to_string(),
            public_inputs_hex: vec!["0x01".to_string()],
            submitted: false,
            tx_hash: None,
        },
    ]));

    // Mark as submitted
    {
        let mut q = queue.write();
        q[0].submitted = true;
        q[0].tx_hash = Some("0x1234".to_string());
    }

    // Verify
    let q = queue.read();
    assert!(q[0].submitted);
    assert_eq!(q[0].tx_hash.as_deref(), Some("0x1234"));
}

// ============================================================================
// AggregatorNode with Pipeline Tests
// ============================================================================

fn default_params() -> helix_node::network::messages::TrainingParams {
    helix_node::network::messages::TrainingParams {
        learning_rate: 0.001,
        batch_size: 32,
        local_epochs: 5,
        max_error_bound: 0.1,
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        model_seed: 42,
        num_layers: 2,
        activation_type: 0,
    }
}

#[tokio::test]
async fn test_aggregator_without_pipeline_completes_normally() {
    // Aggregator without on-chain pipeline should still work as before
    let config = AggregatorConfig {
        min_participants: 1,
        byzantine_strategy: None,
        ..Default::default()
    };
    let node = AggregatorNode::new(PeerId::random(), config);

    node.start_round([1; 32], default_params()).await;

    let p1 = PeerId::random();
    let p2 = PeerId::random();
    node.handle_participate_request(p1.clone(), 1).await;
    node.handle_participate_request(p2.clone(), 1).await;
    node.start_collection().await;

    node.handle_gradient_share(p1, 1, [0xAA; 32], 0.01, vec![1, 2, 3]).await;
    node.handle_gradient_share(p2, 1, [0xBB; 32], 0.02, vec![4, 5, 6]).await;

    let result = node.aggregate().await.expect("aggregation should succeed");
    assert_eq!(result.num_participants, 2);
    assert!(!result.proof.is_empty());
}

#[tokio::test]
async fn test_aggregator_with_public_inputs_builds_training_results() {
    // Test that when gradients have public_inputs_hex, they can be converted
    // to TrainingProofResultV2 for RLC aggregation
    let config = AggregatorConfig {
        min_participants: 1,
        byzantine_strategy: None,
        ..Default::default()
    };
    let node = AggregatorNode::new(PeerId::random(), config);

    node.start_round([1; 32], default_params()).await;

    let p1 = PeerId::random();
    node.handle_participate_request(p1.clone(), 1).await;
    node.start_collection().await;

    // Submit gradient with public inputs
    let public_inputs = vec![
        "0x0100000000000000000000000000000000000000000000000000000000000000".to_string(),
        "0x0200000000000000000000000000000000000000000000000000000000000000".to_string(),
        "0x0300000000000000000000000000000000000000000000000000000000000000".to_string(),
        "0x0400000000000000000000000000000000000000000000000000000000000000".to_string(),
        "0x6400000000000000000000000000000000000000000000000000000000000000".to_string(), // 100
        "0x0a00000000000000000000000000000000000000000000000000000000000000".to_string(), // 10
        "0x0100000000000000000000000000000000000000000000000000000000000000".to_string(), // step 1
        "0x0000000000000000000000000000000000000000000000000000000000000000".to_string(), // checksum
    ];

    node.handle_gradient_share_with_data(
        p1.clone(),
        1,
        [0xAA; 32],
        0.01,
        vec![0xAB; 512], // fake proof bytes
        None,
        Some(public_inputs),
    ).await;

    // Aggregation should still work (pipeline is not set, so no on-chain submission)
    let result = node.aggregate().await.expect("aggregation should succeed");
    assert_eq!(result.num_participants, 1);
}

#[tokio::test]
async fn test_collected_gradient_with_public_inputs() {
    // Verify the CollectedGradient struct correctly stores public inputs
    let gradient = CollectedGradient {
        participant: PeerId::random(),
        commitment: [0xAA; 32],
        error_bound: 0.01,
        proof: vec![1, 2, 3],
        received_at: 1000,
        gradient_data: None,
        public_inputs_hex: Some(vec!["0x01".to_string(), "0x02".to_string()]),
    };

    assert!(gradient.public_inputs_hex.is_some());
    assert_eq!(gradient.public_inputs_hex.as_ref().unwrap().len(), 2);
}

#[tokio::test]
async fn test_collected_gradient_without_public_inputs() {
    let gradient = CollectedGradient {
        participant: PeerId::random(),
        commitment: [0xBB; 32],
        error_bound: 0.02,
        proof: vec![4, 5, 6],
        received_at: 2000,
        gradient_data: None,
        public_inputs_hex: None,
    };

    assert!(gradient.public_inputs_hex.is_none());
}

// ============================================================================
// Full Pipeline Flow Test (no real chain)
// ============================================================================

#[tokio::test]
async fn test_aggregator_round_completion_flow() {
    // Simulate a full round: start → collect → aggregate
    // Without a real chain, the on-chain submission will be skipped
    // but we verify the data flow is correct
    let config = AggregatorConfig {
        min_participants: 1,
        byzantine_strategy: None,
        ..Default::default()
    };
    let node = AggregatorNode::new(PeerId::random(), config);

    // Start round
    let _msg = node.start_round([0x42; 32], default_params()).await;

    // Register 3 workers
    let workers: Vec<PeerId> = (0..3).map(|_| PeerId::random()).collect();
    for w in &workers {
        node.handle_participate_request(w.clone(), 1).await;
    }

    // Start collection
    node.start_collection().await;

    // Each worker submits a gradient with proof bytes and public inputs
    for (i, w) in workers.iter().enumerate() {
        let public_inputs: Vec<String> = (0..8)
            .map(|j| format!("0x{:064x}", (i * 8 + j) + 1))
            .collect();

        node.handle_gradient_share_with_data(
            w.clone(),
            1,
            [(i as u8) + 1; 32],
            0.01,
            vec![(i as u8) + 1; 1856], // Unique proof bytes per worker
            None,
            Some(public_inputs),
        ).await;
    }

    // Aggregate
    let result = node.aggregate().await.expect("aggregation should succeed");
    assert_eq!(result.num_participants, 3);
    assert_eq!(result.round_id, 1);
    assert!(!result.proof.is_empty()); // Poseidon commitment proof
    assert!(result.excluded_participants.is_empty()); // No Byzantine filtering

    // Verify stats
    let stats = node.get_stats().await;
    assert_eq!(stats.rounds_successful, 1);
    assert_eq!(stats.gradients_aggregated, 3);
}

#[test]
fn test_chain_config_default_stake_values() {
    let config: ChainConfig = serde_json::from_str(r#"{
        "coordinator_address": "0x1234",
        "d_in": 2,
        "d_hid": 2,
        "d_out": 1
    }"#).unwrap();

    // Default should be 1 ETH in wei
    assert_eq!(config.min_stake_wei, "1000000000000000000");
    assert_eq!(config.stake_amount_wei, "1000000000000000000");
}
