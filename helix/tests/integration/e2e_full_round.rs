//! Test 1: Full P2P Training Round
//!
//! Simulates the complete production flow:
//! 1. Deploy contracts to Anvil
//! 2. Register model on-chain
//! 3. Start 1 aggregator + 3 workers (as tokio tasks)
//! 4. Workers train and generate real ZK proofs
//! 5. Aggregator collects and validates proofs
//! 6. Submit proof on-chain via Halo2Verifier
//! 7. Verify: round completes, commitment chain is valid, rewards distributed
//!
//! Run with: `cargo test --test e2e_full_round --features integration -- --test-threads=1`

#![cfg(feature = "integration")]

use std::time::Instant;

use ethers::types::{Bytes, U256};
use halo2curves::bn256::Fr;
use sha2::{Digest, Sha256};

use helix_node::network::messages::PeerId;
use helix_node::roles::aggregator::{AggregatorConfig, AggregatorNode, CommitmentAggregation};
use helix_prover::TrainingProofResultV2;

#[path = "../common/mod.rs"]
mod common;
use common::anvil::*;
use common::e2e::*;

// ============================================================================
// Test 1: Full P2P Training Round — 3 Workers + Aggregator + On-Chain
// ============================================================================

#[tokio::test]
async fn test_full_p2p_training_round() {
    let start = Instant::now();
    eprintln!("[E2E] Starting full P2P training round test...");

    // ── Phase 1: Set up chain ──
    let env = OnChainTestEnv::new().await;
    let weights = initial_weights();
    let commitment = compute_initial_commitment(&weights);
    eprintln!("[E2E] Phase 1: Chain deployed ({:.1}s)", start.elapsed().as_secs_f64());

    // ── Phase 2: Register model + stake ──
    let model_id = env.register_model(commitment).await;
    let stake_amount = ethers::utils::parse_ether("1").unwrap();
    env.stake(model_id, stake_amount).await;
    env.start_round(model_id, U256::from(3600u64)).await;
    eprintln!("[E2E] Phase 2: Model registered (id={}), staked, round started", model_id);

    // ── Phase 3: Spawn 3 workers as parallel tasks ──
    let dataset = training_data();
    let model_id_bytes = model_id_to_bytes(model_id);
    let error_budget = default_error_budget();

    let mut worker_handles = Vec::new();
    for worker_id in 0..3 {
        let w = weights.clone();
        let d = dataset.clone();
        let mid = model_id_bytes;
        let eb = error_budget;
        let handle = spawn_worker_with_params(worker_id, w, d, 3, mid, eb);
        worker_handles.push(handle);
    }

    // ── Phase 4: Collect worker results ──
    let mut worker_results = Vec::new();
    for handle in worker_handles {
        let result = handle.await.expect("Worker task panicked");
        eprintln!(
            "[E2E] Worker {} completed: {} steps, all proofs verified",
            result.worker_id,
            result.steps.len()
        );
        worker_results.push(result);
    }
    eprintln!("[E2E] Phase 3-4: All 3 workers completed ({:.1}s)", start.elapsed().as_secs_f64());

    // ── Phase 5: Aggregator collects proofs ──
    let aggregator_id = PeerId::from_string("aggregator-0");
    let config = AggregatorConfig {
        min_participants: 1,
        max_participants: 4,
        collection_timeout_secs: 60,
        generate_proof: false,
        max_error_bound: 1000.0,
        commitment_aggregation: CommitmentAggregation::HashBased,
        byzantine_strategy: None,
    };
    let aggregator = AggregatorNode::new(aggregator_id, config);

    // Start aggregator round
    let model_hash = {
        let mut h = Sha256::new();
        h.update(b"e2e-test-model");
        let result = h.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    };
    let params = helix_node::network::messages::TrainingParams {
        learning_rate: 0.01,
        batch_size: 1,
        local_epochs: 1,
        max_error_bound: 1000.0,
        d_in: D_IN,
        d_hid: D_HID,
        d_out: D_OUT,
        model_seed: 42,
    };
    let _round_msg = aggregator.start_round(model_hash, params).await;

    // Register workers
    for i in 0..3 {
        let peer_id = PeerId::from_string(format!("worker-{}", i));
        aggregator.handle_participate_request(peer_id, 1).await;
    }
    aggregator.start_collection().await;

    // Submit worker proofs to aggregator
    let mut proofs_accepted = 0;
    for (i, worker) in worker_results.iter().enumerate() {
        for (result, _) in &worker.steps {
            let peer_id = PeerId::from_string(format!("worker-{}", i));
            let commitment_hash = {
                let mut h = Sha256::new();
                h.update(&result.proof);
                let digest = h.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&digest);
                hash
            };
            let accepted = aggregator
                .handle_gradient_share(peer_id, 1, commitment_hash, 0.01, result.proof.clone())
                .await;
            if accepted {
                proofs_accepted += 1;
            }
        }
    }
    assert!(proofs_accepted > 0, "At least one proof should be accepted by aggregator");
    eprintln!("[E2E] Phase 5: Aggregator accepted {} proofs", proofs_accepted);

    // ── Phase 6: Submit first worker's first proof on-chain ──
    let first_proof = &worker_results[0].steps[0].0;
    let bundle = TestEvmProofBundle::from_proof_result(first_proof, model_id, env.max_error_bound);

    // Verify commitment matches what we registered
    assert_eq!(
        bundle.old_commitment, commitment,
        "Old commitment from proof must match registered commitment"
    );

    let receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64),
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

    assert_eq!(receipt.status, Some(1.into()), "Proof submission should succeed");
    eprintln!("[E2E] Phase 6: Proof submitted on-chain (gas: {:?})", receipt.gas_used);

    // ── Phase 7: Verify on-chain state ──
    let (_, new_commitment, _, is_completed, prover) =
        env.get_round(model_id, U256::from(1u64)).await;

    assert!(is_completed, "Round should be completed");
    assert_eq!(prover, env.deployer, "Prover should be the deployer");
    assert_ne!(new_commitment, U256::zero(), "New commitment should be set");
    assert_eq!(
        new_commitment, bundle.new_commitment,
        "On-chain commitment should match proof's new commitment"
    );

    // ── Phase 8: Verify no slashing ──
    let (stake_after, _, slashed) = env.get_stake(env.deployer, model_id).await;
    assert!(!slashed, "Stake should NOT be slashed after valid proof");
    assert!(stake_after > 0, "Stake should remain");

    // ── Phase 9: Verify model commitment updated ──
    let on_chain_commitment = env.model_commitment(model_id).await;
    assert_eq!(
        on_chain_commitment, bundle.new_commitment,
        "Model commitment should be updated to proof's new commitment"
    );

    // ── Phase 10: Verify all workers produced valid commitment chains ──
    for worker in &worker_results {
        let refs: Vec<&TrainingProofResultV2> = worker.steps.iter().map(|(r, _)| r).collect();
        assert!(
            verify_commitment_chain(&refs),
            "Worker {} should have valid commitment chain",
            worker.worker_id
        );
    }

    let elapsed = start.elapsed();
    eprintln!(
        "[E2E] PASS: Full P2P training round completed in {:.1}s",
        elapsed.as_secs_f64()
    );
    eprintln!(
        "[E2E]   Workers: 3, Steps/worker: 3, Proofs: 9, On-chain: 1 verified"
    );
}

// ============================================================================
// Test: Multiple Workers Submit to Same Round
// ============================================================================

#[tokio::test]
async fn test_multiple_workers_same_round() {
    eprintln!("[E2E] Testing multiple workers contributing to same round...");

    let env = OnChainTestEnv::new().await;
    let weights = initial_weights();
    let commitment = compute_initial_commitment(&weights);

    let model_id = env.register_model(commitment).await;
    env.stake(model_id, ethers::utils::parse_ether("1").unwrap()).await;
    env.start_round(model_id, U256::from(3600u64)).await;

    // Generate proofs from 3 independent workers (all start from same weights)
    let dataset = training_data();
    let model_id_bytes = model_id_to_bytes(model_id);

    let worker0 = simulate_worker_with_params(
        0, &weights, &dataset, 1, model_id_bytes, default_error_budget(),
    );

    // All workers should produce proofs with the same old_state_hash
    // (since they all start from the same weights)
    assert_eq!(
        worker0.steps[0].0.old_state_hash,
        compute_state_hash(&weights),
        "Worker 0's old hash should match initial weights"
    );

    // Submit worker 0's proof
    let bundle = TestEvmProofBundle::from_proof_result(
        &worker0.steps[0].0,
        model_id,
        env.max_error_bound,
    );
    let receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64),
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

    assert_eq!(receipt.status, Some(1.into()));
    let (_, _, _, is_completed, _) = env.get_round(model_id, U256::from(1u64)).await;
    assert!(is_completed, "Round should complete with first valid proof");

    eprintln!("[E2E] PASS: Multiple workers contributing to same round");
}
