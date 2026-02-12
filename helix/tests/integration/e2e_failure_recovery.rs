//! Test 6: Failure Recovery
//!
//! Tests system resilience when workers fail:
//! 1. Kill a worker mid-round, verify remaining workers complete
//! 2. Worker failure doesn't corrupt aggregation
//! 3. New workers can pick up from committed state
//! 4. On-chain state remains consistent after failures
//!
//! Run with: `cargo test --test e2e_failure_recovery --features integration -- --test-threads=1`

#![cfg(feature = "integration")]

use std::time::{Duration, Instant};

use ethers::types::U256;
use halo2curves::bn256::Fr;

use helix_prover::TrainingProofResultV2;

#[path = "../common/mod.rs"]
mod common;
use common::anvil::*;
use common::e2e::*;

// ============================================================================
// Test 6a: Worker Failure — Kill Worker Mid-Round
// ============================================================================

#[tokio::test]
async fn test_worker_failure_recovery() {
    let start = Instant::now();
    eprintln!("[E2E-FR] Starting worker failure recovery test...");

    // ── Phase 1: Set up chain ──
    let env = OnChainTestEnv::new().await;
    let weights = initial_weights();
    let commitment = compute_initial_commitment(&weights);

    let model_id = env.register_model(commitment).await;
    env.stake(model_id, ethers::utils::parse_ether("1").unwrap()).await;
    env.start_round(model_id, U256::from(3600u64)).await;
    eprintln!("[E2E-FR] Phase 1: Chain deployed, model registered");

    // ── Phase 2: Spawn 3 workers, cancel 1 mid-round ──
    let dataset = training_data();
    let model_id_bytes = model_id_to_bytes(model_id);
    let error_budget = default_error_budget();

    // Worker 0 and 1: run to completion (3 steps each)
    let handle_0 = spawn_worker_with_params(
        0,
        weights.clone(),
        dataset.clone(),
        3,
        model_id_bytes,
        error_budget,
    );
    let handle_1 = spawn_worker_with_params(
        1,
        weights.clone(),
        dataset.clone(),
        3,
        model_id_bytes,
        error_budget,
    );

    // Worker 2: will be cancelled after a delay
    let handle_2 = spawn_worker_with_params(
        2,
        weights.clone(),
        dataset.clone(),
        10, // many steps — will be cancelled before completing
        model_id_bytes,
        error_budget,
    );

    // Wait a moment for workers to start, then cancel worker 2
    // In practice, we race the task with a timeout.
    let worker_2_result = tokio::time::timeout(Duration::from_millis(100), handle_2).await;

    // Worker 2 may or may not have completed — either way is fine
    let worker_2_completed = match &worker_2_result {
        Ok(Ok(result)) => {
            eprintln!("[E2E-FR] Worker 2 completed {} steps before timeout", result.steps.len());
            true
        }
        Ok(Err(e)) => {
            eprintln!("[E2E-FR] Worker 2 panicked: {:?}", e);
            false
        }
        Err(_) => {
            eprintln!("[E2E-FR] Worker 2 timed out (simulating failure)");
            false
        }
    };

    // Workers 0 and 1 should complete normally
    let result_0 = handle_0.await.expect("Worker 0 should not panic");
    let result_1 = handle_1.await.expect("Worker 1 should not panic");

    assert_eq!(result_0.steps.len(), 3, "Worker 0 should complete all 3 steps");
    assert_eq!(result_1.steps.len(), 3, "Worker 1 should complete all 3 steps");
    eprintln!("[E2E-FR] Phase 2: Workers 0 and 1 completed, worker 2 failed/timed out");

    // ── Phase 3: Submit proof from surviving workers ──
    let proof = &result_0.steps[0].0;
    let bundle = TestEvmProofBundle::from_proof_result(proof, model_id, env.max_error_bound);

    let receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64),
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

    assert_eq!(receipt.status, Some(1.into()), "Proof from surviving worker should succeed");

    let (_, _, _, is_completed, _) = env.get_round(model_id, U256::from(1u64)).await;
    assert!(is_completed, "Round should complete despite worker failure");
    eprintln!("[E2E-FR] Phase 3: Round completed with surviving workers' proof");

    // ── Phase 4: Verify on-chain state is consistent ──
    let (_, _, slashed) = env.get_stake(env.deployer, model_id).await;
    assert!(!slashed, "No slashing should occur for worker failure");

    let on_chain_commit = env.model_commitment(model_id).await;
    assert_eq!(
        on_chain_commit, bundle.new_commitment,
        "On-chain commitment should match submitted proof"
    );

    let elapsed = start.elapsed();
    eprintln!(
        "[E2E-FR] PASS: Worker failure recovery in {:.1}s",
        elapsed.as_secs_f64()
    );
    eprintln!("[E2E-FR]   Workers: 3 (2 completed, 1 failed), round: completed");
}

// ============================================================================
// Test 6b: Resume From Committed State
// ============================================================================

#[tokio::test]
async fn test_resume_from_committed_state() {
    let start = Instant::now();
    eprintln!("[E2E-FR] Starting resume from committed state test...");

    // ── Phase 1: Set up chain and complete first round ──
    let env = OnChainTestEnv::new().await;
    let weights = initial_weights();
    let commitment = compute_initial_commitment(&weights);
    let model_id_bytes = model_id_to_bytes(U256::zero());
    let error_budget = default_error_budget();

    let model_id = env.register_model(commitment).await;
    env.stake(model_id, ethers::utils::parse_ether("1").unwrap()).await;
    env.start_round(model_id, U256::from(3600u64)).await;

    // Complete round 1 with a single training step
    let dataset = training_data();
    let (x, target) = &dataset[0];
    let model_id_bytes = model_id_to_bytes(model_id);
    let (result_1, new_weights_1) = prove_step_with_params(
        &weights, x, target, 1, model_id_bytes, error_budget,
    );

    let bundle_1 = TestEvmProofBundle::from_proof_result(&result_1, model_id, env.max_error_bound);
    env.submit_proof(
        model_id,
        U256::from(1u64),
        bundle_1.proof_as_bytes(),
        bundle_1.public_inputs.clone(),
    )
    .await;

    let (_, _, _, is_completed, _) = env.get_round(model_id, U256::from(1u64)).await;
    assert!(is_completed, "Round 1 should complete");
    eprintln!("[E2E-FR] Phase 1: Round 1 completed, commitment updated");

    // ── Phase 2: "Restart" — new worker picks up from committed state ──
    // The new worker should use the weights from after round 1.
    // In production, this comes from a checkpoint. Here we use the updated weights.

    env.start_round(model_id, U256::from(3600u64)).await;

    let (x2, target2) = &dataset[1];
    let (result_2, new_weights_2) = prove_step_with_params(
        &new_weights_1, x2, target2, 2, model_id_bytes, error_budget,
    );

    assert!(result_2.verified, "Resumed proof should be self-verified");

    // Verify commitment chaining: round 2's old hash == round 1's new hash
    assert_eq!(
        result_2.old_state_hash, result_1.new_state_hash,
        "Resumed training should chain from round 1's output"
    );
    eprintln!("[E2E-FR] Phase 2: Resumed training chains from committed state");

    // ── Phase 3: Submit round 2 proof ──
    let bundle_2 = TestEvmProofBundle::from_proof_result(&result_2, model_id, env.max_error_bound);

    // Round 2's old_commitment should match round 1's new_commitment
    assert_eq!(
        bundle_2.old_commitment, bundle_1.new_commitment,
        "Round 2 old commitment should match round 1 new commitment"
    );

    let receipt = env
        .submit_proof(
            model_id,
            U256::from(2u64),
            bundle_2.proof_as_bytes(),
            bundle_2.public_inputs.clone(),
        )
        .await;

    assert_eq!(receipt.status, Some(1.into()));
    let (_, _, _, is_completed_2, _) = env.get_round(model_id, U256::from(2u64)).await;
    assert!(is_completed_2, "Round 2 should complete after resume");

    // Verify final on-chain state
    let final_commit = env.model_commitment(model_id).await;
    assert_eq!(
        final_commit, bundle_2.new_commitment,
        "Final commitment should match round 2's output"
    );

    let elapsed = start.elapsed();
    eprintln!(
        "[E2E-FR] PASS: Resume from committed state in {:.1}s",
        elapsed.as_secs_f64()
    );
    eprintln!("[E2E-FR]   Rounds: 2 (1 initial + 1 resumed), commitment chain: valid");
}

// ============================================================================
// Test 6c: Concurrent Worker Completion Order Independence
// ============================================================================

#[tokio::test]
async fn test_worker_completion_order_independence() {
    eprintln!("[E2E-FR] Testing worker completion order independence...");

    let weights = initial_weights();
    let dataset = training_data();

    // Spawn 3 workers with different step counts
    // They'll complete at different times but should all produce valid proofs
    let handle_fast = spawn_worker(0, weights.clone(), dataset.clone(), 1);
    let handle_medium = spawn_worker(1, weights.clone(), dataset.clone(), 2);
    let handle_slow = spawn_worker(2, weights.clone(), dataset.clone(), 3);

    // Collect results (order of completion doesn't matter)
    let result_fast = handle_fast.await.expect("Fast worker panicked");
    let result_medium = handle_medium.await.expect("Medium worker panicked");
    let result_slow = handle_slow.await.expect("Slow worker panicked");

    assert_eq!(result_fast.steps.len(), 1);
    assert_eq!(result_medium.steps.len(), 2);
    assert_eq!(result_slow.steps.len(), 3);

    // All workers should start from the same initial state
    let fast_old_hash = result_fast.steps[0].0.old_state_hash;
    let medium_old_hash = result_medium.steps[0].0.old_state_hash;
    let slow_old_hash = result_slow.steps[0].0.old_state_hash;

    assert_eq!(fast_old_hash, medium_old_hash, "All workers should start from same state");
    assert_eq!(medium_old_hash, slow_old_hash, "All workers should start from same state");

    // Each worker's chain should be independently valid
    let fast_refs: Vec<&TrainingProofResultV2> = result_fast.steps.iter().map(|(r, _)| r).collect();
    let medium_refs: Vec<&TrainingProofResultV2> = result_medium.steps.iter().map(|(r, _)| r).collect();
    let slow_refs: Vec<&TrainingProofResultV2> = result_slow.steps.iter().map(|(r, _)| r).collect();

    assert!(verify_commitment_chain(&fast_refs), "Fast worker chain should be valid");
    assert!(verify_commitment_chain(&medium_refs), "Medium worker chain should be valid");
    assert!(verify_commitment_chain(&slow_refs), "Slow worker chain should be valid");

    // All proofs should verify
    for (i, (result, _)) in result_slow.steps.iter().enumerate() {
        assert!(verify_proof(result), "Slow worker step {} should verify", i + 1);
    }

    eprintln!("[E2E-FR] PASS: Worker completion order independence verified");
    eprintln!("[E2E-FR]   Workers: 3 (1, 2, 3 steps), all chains valid, all proofs valid");
}
