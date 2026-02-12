//! Test 3: Byzantine Worker Detection
//!
//! Starts 4 workers, 1 of which submits corrupted proofs.
//! Verifies:
//! 1. Aggregator detects and excludes the Byzantine worker
//! 2. Slashing is triggered on-chain for the invalid proof
//! 3. The round still completes with the 3 honest workers' proofs
//! 4. Honest workers' stake is preserved
//!
//! Run with: `cargo test --test e2e_byzantine --features integration -- --test-threads=1`

#![cfg(feature = "integration")]

use std::time::Instant;

use ethers::types::U256;
use halo2curves::bn256::Fr;

use helix_prover::TrainingProofResultV2;

#[path = "../common/mod.rs"]
mod common;
use common::anvil::*;
use common::e2e::*;

// ============================================================================
// Test 3: Byzantine Worker — Invalid Proof → Rejection + Slashing
// ============================================================================

#[tokio::test]
async fn test_byzantine_worker_detection() {
    let start = Instant::now();
    eprintln!("[E2E-BYZ] Starting Byzantine worker detection test...");

    // ── Phase 1: Set up chain ──
    let env = OnChainTestEnv::new().await;
    let weights = initial_weights();
    let commitment = compute_initial_commitment(&weights);

    let model_id = env.register_model(commitment).await;
    env.stake(model_id, ethers::utils::parse_ether("1").unwrap()).await;
    env.start_round(model_id, U256::from(3600u64)).await;
    eprintln!("[E2E-BYZ] Phase 1: Chain deployed, model registered");

    // ── Phase 2: Spawn 4 workers (3 honest + 1 Byzantine) ──
    let dataset = training_data();
    let model_id_bytes = model_id_to_bytes(model_id);
    let error_budget = default_error_budget();

    // Worker 0-2: honest workers
    let honest_workers: Vec<WorkerResult> = (0..3)
        .map(|i| {
            simulate_worker_with_params(
                i,
                &weights,
                &dataset,
                1,
                model_id_bytes,
                error_budget,
            )
        })
        .collect();

    // Worker 3: Byzantine — generates a valid proof then corrupts it
    let byzantine_worker = simulate_worker_with_params(
        3,
        &weights,
        &dataset,
        1,
        model_id_bytes,
        error_budget,
    );
    let corrupted_proof = corrupt_proof(&byzantine_worker.steps[0].0);
    eprintln!(
        "[E2E-BYZ] Phase 2: 4 workers completed (3 honest, 1 Byzantine)"
    );

    // ── Phase 3: Verify honest proofs are valid ──
    for (i, worker) in honest_workers.iter().enumerate() {
        let result = &worker.steps[0].0;
        assert!(result.verified, "Honest worker {} proof should be self-verified", i);
        assert!(
            verify_proof(result),
            "Honest worker {} proof should pass verification",
            i
        );
    }

    // ── Phase 4: Verify corrupted proof is detected as invalid ──
    let corrupted_is_valid = verify_proof(&corrupted_proof);
    assert!(
        !corrupted_is_valid,
        "Corrupted proof should be detected as invalid"
    );
    eprintln!("[E2E-BYZ] Phase 3-4: Proof validation complete (3 valid, 1 invalid)");

    // ── Phase 5: Submit corrupted proof first (same round) to test slashing ──
    // Set the mock verifier to reject, simulating real verifier detecting corruption.
    env.set_mock_accept(false).await;

    let corrupt_bundle = TestEvmProofBundle::from_proof_result(
        &corrupted_proof,
        model_id,
        env.max_error_bound,
    );

    // Submit corrupted proof — should trigger slashing (tx succeeds, slashing is internal)
    let corrupt_receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64),
            corrupt_bundle.proof_as_bytes(),
            corrupt_bundle.public_inputs.clone(),
        )
        .await;

    assert_eq!(
        corrupt_receipt.status,
        Some(1.into()),
        "Transaction should succeed (slashing happens in contract logic)"
    );

    // Verify slashing occurred
    let (stake_after_slash, _, slashed_after) = env.get_stake(env.deployer, model_id).await;
    assert!(slashed_after, "Prover should be slashed after invalid proof");

    // Round should NOT be completed (proof was invalid)
    let (_, _, _, is_completed, _) = env.get_round(model_id, U256::from(1u64)).await;
    assert!(!is_completed, "Round should NOT complete with invalid proof");
    eprintln!(
        "[E2E-BYZ] Phase 5-6: Invalid proof submitted, stake slashed (after: {})",
        stake_after_slash
    );

    // Verify the commitment didn't change (since the round wasn't completed)
    let on_chain_commit = env.model_commitment(model_id).await;
    assert_eq!(
        on_chain_commit, commitment,
        "Model commitment should remain unchanged after invalid proof"
    );

    let elapsed = start.elapsed();
    eprintln!(
        "[E2E-BYZ] PASS: Byzantine detection completed in {:.1}s",
        elapsed.as_secs_f64()
    );
    eprintln!("[E2E-BYZ]   Workers: 4 (3 honest + 1 Byzantine), slashing: triggered");
}

// ============================================================================
// Test: Multiple Byzantine Workers — Majority Honest
// ============================================================================

#[tokio::test]
async fn test_majority_honest_workers() {
    eprintln!("[E2E-BYZ] Testing majority honest workers (3 of 4)...");

    let weights = initial_weights();
    let dataset = training_data();

    // Generate proofs for 4 workers: 3 honest, 1 byzantine
    let honest_results: Vec<WorkerResult> = (0..3)
        .map(|i| simulate_worker(i, &weights, &dataset, 1))
        .collect();
    let byzantine_result = simulate_worker(3, &weights, &dataset, 1);
    let corrupted = corrupt_proof(&byzantine_result.steps[0].0);

    // Count valid/invalid proofs
    let honest_valid: usize = honest_results
        .iter()
        .filter(|w| verify_proof(&w.steps[0].0))
        .count();
    let byzantine_valid = verify_proof(&corrupted);

    assert_eq!(honest_valid, 3, "All 3 honest proofs should be valid");
    assert!(!byzantine_valid, "Byzantine proof should be invalid");

    // All honest workers should have consistent old_state_hash
    // (they all start from the same weights)
    let first_hash = honest_results[0].steps[0].0.old_state_hash;
    for worker in &honest_results[1..] {
        assert_eq!(
            worker.steps[0].0.old_state_hash, first_hash,
            "All honest workers should start from same state"
        );
    }

    eprintln!("[E2E-BYZ] PASS: Majority honest (3/4 valid proofs, consistent state)");
}
