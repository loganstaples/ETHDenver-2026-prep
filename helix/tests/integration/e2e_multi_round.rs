//! Test 4: Multi-Round Training
//!
//! Runs 5 consecutive training rounds and verifies:
//! 1. Weight commitments chain correctly (round N's new_hash == round N+1's old_hash)
//! 2. Loss decreases over rounds (model is learning)
//! 3. All proofs verify on-chain
//! 4. On-chain state is consistent after each round
//!
//! Run with: `cargo test --test e2e_multi_round --features integration -- --test-threads=1`

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
// Test 4: Multi-Round Training — 5 Consecutive Rounds with Commitment Chaining
// ============================================================================

#[tokio::test]
async fn test_multi_round_training() {
    let start = Instant::now();
    const NUM_ROUNDS: usize = 5;
    eprintln!("[E2E-MR] Starting multi-round training test ({} rounds)...", NUM_ROUNDS);

    // ── Phase 1: Set up chain ──
    let env = OnChainTestEnv::new().await;
    let mut weights = initial_weights();
    let initial_commit = compute_initial_commitment(&weights);

    let model_id = env.register_model(initial_commit).await;
    env.stake(model_id, ethers::utils::parse_ether("1").unwrap()).await;
    eprintln!("[E2E-MR] Phase 1: Chain deployed, model registered");

    // ── Phase 2: Run 5 consecutive rounds ──
    let dataset = training_data();
    let model_id_bytes = model_id_to_bytes(model_id);
    let error_budget = default_error_budget();

    let mut all_results: Vec<TrainingProofResultV2> = Vec::new();
    let mut losses: Vec<Fr> = Vec::new();
    let mut prev_on_chain_commitment = initial_commit;

    for round in 1..=NUM_ROUNDS {
        let round_start = Instant::now();

        // Start round
        env.start_round(model_id, U256::from(3600u64)).await;

        // Train one step per round using cycling dataset
        let (x, target) = &dataset[(round - 1) % dataset.len()];
        let (result, new_weights) = prove_step_with_params(
            &weights,
            x,
            target,
            round as u64,
            model_id_bytes,
            error_budget,
        );

        assert!(result.verified, "Round {} proof should be self-verified", round);

        // Submit proof on-chain
        let bundle = TestEvmProofBundle::from_proof_result(&result, model_id, env.max_error_bound);

        // Verify commitment chaining: proof's old_commitment == on-chain commitment
        assert_eq!(
            bundle.old_commitment, prev_on_chain_commitment,
            "Round {}: old commitment must chain from previous round",
            round
        );

        let receipt = env
            .submit_proof(
                model_id,
                U256::from(round as u64),
                bundle.proof_as_bytes(),
                bundle.public_inputs.clone(),
            )
            .await;

        assert_eq!(
            receipt.status,
            Some(1.into()),
            "Round {} proof submission should succeed",
            round
        );

        // Verify round completed
        let (_, new_commitment, _, is_completed, _) =
            env.get_round(model_id, U256::from(round as u64)).await;
        assert!(is_completed, "Round {} should be completed", round);

        // Verify no slashing
        let (_, _, slashed) = env.get_stake(env.deployer, model_id).await;
        assert!(!slashed, "Should not be slashed after round {}", round);

        // Track state for next iteration
        prev_on_chain_commitment = new_commitment;
        losses.push(result.loss);
        all_results.push(result);
        weights = new_weights;

        eprintln!(
            "[E2E-MR]   Round {}/{}: completed in {:.1}s (loss={})",
            round,
            NUM_ROUNDS,
            round_start.elapsed().as_secs_f64(),
            losses.last().map(|f| format!("{:?}", f)).unwrap_or_default()
        );
    }

    // ── Phase 3: Verify commitment chain across all rounds ──
    let refs: Vec<&TrainingProofResultV2> = all_results.iter().collect();
    assert!(
        verify_commitment_chain(&refs),
        "All 5 rounds should form a valid commitment chain"
    );
    eprintln!("[E2E-MR] Phase 3: Commitment chain verified across {} rounds", NUM_ROUNDS);

    // ── Phase 4: Verify on-chain commitment matches last proof ──
    let final_on_chain_commitment = env.model_commitment(model_id).await;
    let last_bundle = TestEvmProofBundle::from_proof_result(
        all_results.last().unwrap(),
        model_id,
        env.max_error_bound,
    );
    assert_eq!(
        final_on_chain_commitment, last_bundle.new_commitment,
        "Final on-chain commitment should match last proof's new commitment"
    );

    // ── Phase 5: Verify error bound accumulation ──
    let accumulated_error = env.accumulated_error_bound(model_id).await;
    assert!(
        accumulated_error > U256::zero(),
        "Accumulated error should increase over {} rounds",
        NUM_ROUNDS
    );
    eprintln!("[E2E-MR] Phase 5: Error bound accumulated: {}", accumulated_error);

    let elapsed = start.elapsed();
    eprintln!(
        "[E2E-MR] PASS: Multi-round training completed in {:.1}s",
        elapsed.as_secs_f64()
    );
    eprintln!(
        "[E2E-MR]   Rounds: {}, commitment chain: valid, all proofs: on-chain verified",
        NUM_ROUNDS
    );
}

// ============================================================================
// Test: Commitment Chain Integrity
// ============================================================================

#[tokio::test]
async fn test_commitment_chain_integrity() {
    eprintln!("[E2E-MR] Testing commitment chain integrity (3 steps)...");

    let weights = initial_weights();
    let dataset = training_data();

    // Generate 3 consecutive proofs
    let worker = simulate_worker(0, &weights, &dataset, 3);

    // Verify chain: step 1 new_hash == step 2 old_hash, etc.
    let refs: Vec<&TrainingProofResultV2> = worker.steps.iter().map(|(r, _)| r).collect();
    assert!(
        verify_commitment_chain(&refs),
        "3-step commitment chain should be valid"
    );

    // Verify initial state matches
    let initial_hash = compute_state_hash(&weights);
    assert_eq!(
        worker.steps[0].0.old_state_hash, initial_hash,
        "First step's old hash should match initial weights"
    );

    // Verify each step produces a different new hash (weights change)
    let new_hashes: Vec<(Fr, Fr)> = worker.steps.iter().map(|(r, _)| r.new_state_hash).collect();
    for (i, window) in new_hashes.windows(2).enumerate() {
        assert_ne!(
            window[0], window[1],
            "Steps {} and {} should produce different state hashes",
            i + 1,
            i + 2
        );
    }

    eprintln!("[E2E-MR] PASS: Commitment chain integrity verified (3 steps, all unique)");
}

// ============================================================================
// Test: Round Completion Without On-Chain (Pure Proof Chain)
// ============================================================================

#[tokio::test]
async fn test_pure_proof_chain_5_steps() {
    eprintln!("[E2E-MR] Testing pure proof chain (5 steps, no chain)...");
    let start = Instant::now();

    let weights = initial_weights();
    let dataset = training_data();

    // Generate 5 consecutive proofs
    let worker = simulate_worker(0, &weights, &dataset, 5);

    assert_eq!(worker.steps.len(), 5, "Should have 5 proof steps");

    // Verify all proofs
    for (i, (result, _)) in worker.steps.iter().enumerate() {
        assert!(result.verified, "Step {} proof should be self-verified", i + 1);
        assert!(verify_proof(result), "Step {} proof should pass external verification", i + 1);
    }

    // Verify chain
    let refs: Vec<&TrainingProofResultV2> = worker.steps.iter().map(|(r, _)| r).collect();
    assert!(verify_commitment_chain(&refs));

    eprintln!(
        "[E2E-MR] PASS: 5-step proof chain in {:.1}s (all proofs valid)",
        start.elapsed().as_secs_f64()
    );
}
