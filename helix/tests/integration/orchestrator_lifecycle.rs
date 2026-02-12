//! Orchestrator Lifecycle Integration Tests.
//!
//! Tests the full training round state machine lifecycle:
//! Initializing → Collecting → Aggregating → Committing → Completed
//!
//! Uses the `TrainingRound` and `RoundManager` from helix-node directly
//! to test orchestration logic without requiring real networking.

#![allow(unused_imports)]

use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::ml::training_step_v2::compute_state_hash_v2;
use helix_prover::{
    BatchTrainingProverV2, MLTrainingProverV2, TrainingProofResultV2, TrainingWeights,
};

use helix_node::training::{
    AggregationResult, GradientSubmission, Participant, RoundConfig, RoundError, RoundFailure,
    RoundId, RoundManager, RoundState, RoundSummary, TrainingRound,
};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Helpers
// ============================================================================

fn test_round_config(min_participants: usize) -> RoundConfig {
    RoundConfig {
        min_participants,
        max_participants: 10,
        collection_timeout: Duration::from_secs(300),
        aggregation_timeout: Duration::from_secs(60),
        commit_timeout: Duration::from_secs(120),
        min_stake: 100,
        byzantine_tolerant: true,
        max_byzantine_fraction: 0.33,
    }
}

fn make_gradient_submission(
    participant_id: &str,
    gradient_data: &[f64],
    step: u64,
) -> GradientSubmission {
    let mut hasher = Sha256::new();
    for g in gradient_data {
        hasher.update(g.to_le_bytes());
    }
    let gradient_hash: [u8; 32] = hasher.finalize().into();

    let serialized: Vec<u8> = gradient_data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    GradientSubmission {
        participant_id: participant_id.to_string(),
        gradient_hash,
        gradient_data: serialized,
        proof: vec![0u8; 64], // placeholder proof
        public_inputs: vec![],
        error_bound: 0.01,
        timestamp: step,
    }
}

fn make_aggregation_result(
    submissions: &[GradientSubmission],
    num_contributors: usize,
) -> AggregationResult {
    let mut hasher = Sha256::new();
    for sub in submissions {
        hasher.update(&sub.gradient_hash);
    }
    let aggregated_hash: [u8; 32] = hasher.finalize().into();

    AggregationResult {
        aggregated_hash,
        aggregated_gradient: vec![0u8; 32],
        combined_error_bound: submissions.iter().map(|s| s.error_bound).sum(),
        total_stake: (num_contributors * 1000) as u64,
        num_contributors,
        excluded_participants: Vec::new(),
        aggregated_proof: vec![],
        aggregated_proof_public_inputs: vec![],
    }
}

fn compute_commitment(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

// ============================================================================
// Single Round Lifecycle Tests
// ============================================================================

/// Tests complete round lifecycle: Initializing → Collecting → Aggregating → Committing → Completed.
#[test]
fn test_round_full_lifecycle() {
    let config = test_round_config(2);
    let initial_commitment = compute_commitment(b"initial_model_v0");

    let mut round = TrainingRound::new(RoundId::new(0), config, initial_commitment);
    assert_eq!(round.state, RoundState::Initializing);

    // Register 3 participants
    round
        .register_participant("worker-0".to_string(), 1000)
        .unwrap();
    round
        .register_participant("worker-1".to_string(), 2000)
        .unwrap();
    round
        .register_participant("aggregator".to_string(), 5000)
        .unwrap();
    assert_eq!(round.participants.len(), 3);

    // Transition to Collecting
    round.start_collection().unwrap();
    assert_eq!(round.state, RoundState::Collecting);

    // Workers submit gradients
    let grad0 = make_gradient_submission("worker-0", &[0.1, 0.2, 0.3, 0.4], 1);
    let grad1 = make_gradient_submission("worker-1", &[0.15, 0.25, 0.35, 0.45], 1);
    round.submit_gradient(grad0.clone()).unwrap();
    round.submit_gradient(grad1.clone()).unwrap();
    assert_eq!(round.submission_count(), 2);

    // Aggregator also submits
    let grad_agg = make_gradient_submission("aggregator", &[0.12, 0.22, 0.32, 0.42], 1);
    round.submit_gradient(grad_agg.clone()).unwrap();
    assert_eq!(round.submission_count(), 3);
    assert!((round.submission_progress() - 1.0).abs() < 1e-10);

    // Transition to Aggregating
    round.start_aggregation().unwrap();
    assert_eq!(round.state, RoundState::Aggregating);

    // Set aggregation result
    let agg_result = make_aggregation_result(&[grad0, grad1, grad_agg], 3);
    round.set_aggregation_result(agg_result).unwrap();
    assert_eq!(round.state, RoundState::Committing);

    // Complete with new commitment
    let final_commitment = compute_commitment(b"model_v1_after_round_0");
    round.complete(final_commitment).unwrap();
    assert_eq!(round.state, RoundState::Completed);
    assert!(round.is_terminal());
    assert_eq!(round.final_commitment, Some(final_commitment));

    // Verify state history has all transitions
    assert_eq!(round.state_history.len(), 5); // Init, Collecting, Aggregating, Committing, Completed
}

/// Tests multi-round lifecycle with commitment chain continuity via RoundManager.
#[test]
fn test_round_manager_three_rounds_commitment_chain() {
    let config = test_round_config(2);
    let mut manager = RoundManager::new(config);

    let mut prev_commitment = compute_commitment(b"genesis_model");
    let mut commitments = vec![prev_commitment];

    for round_idx in 0..3 {
        // Start new round
        let round_id = manager.start_round(prev_commitment).unwrap();
        assert_eq!(round_id, RoundId::new(round_idx));

        let round = manager.current_mut().unwrap();

        // Register workers
        round
            .register_participant("worker-0".to_string(), 1000)
            .unwrap();
        round
            .register_participant("worker-1".to_string(), 1000)
            .unwrap();

        // Start collection
        round.start_collection().unwrap();

        // Submit gradients
        let grad0 = make_gradient_submission(
            "worker-0",
            &[0.1 * (round_idx as f64 + 1.0), 0.2],
            round_idx + 1,
        );
        let grad1 = make_gradient_submission(
            "worker-1",
            &[0.15 * (round_idx as f64 + 1.0), 0.25],
            round_idx + 1,
        );
        round.submit_gradient(grad0.clone()).unwrap();
        round.submit_gradient(grad1.clone()).unwrap();

        // Aggregate
        round.start_aggregation().unwrap();
        let agg_result = make_aggregation_result(&[grad0, grad1], 2);
        round.set_aggregation_result(agg_result).unwrap();

        // Complete with new commitment derived from round data
        let new_commitment =
            compute_commitment(format!("model_after_round_{}", round_idx).as_bytes());
        round.complete(new_commitment).unwrap();

        // Verify commitment chain: this round's initial matches previous round's final
        let completed_round = manager.current().unwrap();
        assert_eq!(completed_round.initial_commitment, prev_commitment);
        assert_eq!(completed_round.final_commitment, Some(new_commitment));

        prev_commitment = new_commitment;
        commitments.push(new_commitment);
    }

    // Verify round manager state
    assert_eq!(manager.completed_count(), 2); // Rounds 0 and 1 archived (round 2 still current)
    assert!((manager.success_rate() - 1.0).abs() < 1e-10);

    // Verify commitment chain integrity
    for i in 1..commitments.len() {
        assert_ne!(
            commitments[i], commitments[i - 1],
            "Each round should produce a different commitment"
        );
    }
}

// ============================================================================
// Worker Failure & Recovery Tests
// ============================================================================

/// Tests that a round can still succeed if one worker fails to submit
/// (as long as min_participants is met).
#[test]
fn test_round_worker_failure_recovery() {
    let config = test_round_config(2); // Only 2 required
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    // Register 3 workers (1 more than minimum)
    round
        .register_participant("worker-0".to_string(), 1000)
        .unwrap();
    round
        .register_participant("worker-1".to_string(), 1000)
        .unwrap();
    round
        .register_participant("worker-2-unreliable".to_string(), 1000)
        .unwrap();

    round.start_collection().unwrap();

    // Only 2 of 3 workers submit (worker-2 "fails")
    let grad0 = make_gradient_submission("worker-0", &[0.1, 0.2], 1);
    let grad1 = make_gradient_submission("worker-1", &[0.15, 0.25], 1);
    round.submit_gradient(grad0.clone()).unwrap();
    round.submit_gradient(grad1.clone()).unwrap();

    // Should still be able to aggregate with 2 submissions (>= min_participants=2)
    round.start_aggregation().unwrap();

    let agg_result = make_aggregation_result(&[grad0, grad1], 2);
    round.set_aggregation_result(agg_result).unwrap();

    let final_commitment = compute_commitment(b"model_v1");
    round.complete(final_commitment).unwrap();
    assert_eq!(round.state, RoundState::Completed);

    // Verify the failed worker is still registered but didn't submit
    let unreliable = round.participants.get("worker-2-unreliable").unwrap();
    assert!(!unreliable.submitted);
}

/// Tests that insufficient submissions prevent aggregation.
#[test]
fn test_round_insufficient_submissions_rejected() {
    let config = test_round_config(3); // Need at least 3
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    // Register exactly 3 workers
    round
        .register_participant("worker-0".to_string(), 1000)
        .unwrap();
    round
        .register_participant("worker-1".to_string(), 1000)
        .unwrap();
    round
        .register_participant("worker-2".to_string(), 1000)
        .unwrap();

    round.start_collection().unwrap();

    // Only 2 submit — below minimum
    let grad0 = make_gradient_submission("worker-0", &[0.1], 1);
    let grad1 = make_gradient_submission("worker-1", &[0.15], 1);
    round.submit_gradient(grad0).unwrap();
    round.submit_gradient(grad1).unwrap();

    // Attempting aggregation should fail
    let err = round.start_aggregation().unwrap_err();
    match err {
        RoundError::InsufficientSubmissions {
            required,
            received,
        } => {
            assert_eq!(required, 3);
            assert_eq!(received, 2);
        }
        other => panic!("Expected InsufficientSubmissions, got {:?}", other),
    }
}

// ============================================================================
// State Machine Invariant Tests
// ============================================================================

/// Tests that state transitions are enforced (wrong-state errors).
#[test]
fn test_round_state_transition_enforcement() {
    let config = test_round_config(2);
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    // Can't submit gradient in Initializing state
    let grad = make_gradient_submission("worker-0", &[0.1], 1);
    let err = round.submit_gradient(grad).unwrap_err();
    assert!(matches!(err, RoundError::WrongState { .. }));

    // Can't aggregate in Initializing state
    let err = round.start_aggregation().unwrap_err();
    assert!(matches!(err, RoundError::WrongState { .. }));

    // Can't complete in Initializing state
    let err = round.complete([0u8; 32]).unwrap_err();
    assert!(matches!(err, RoundError::WrongState { .. }));

    // Register workers and move to Collecting
    round
        .register_participant("w0".to_string(), 1000)
        .unwrap();
    round
        .register_participant("w1".to_string(), 1000)
        .unwrap();
    round.start_collection().unwrap();

    // Can't register in Collecting state
    let err = round.register_participant("w2".to_string(), 1000).unwrap_err();
    assert!(matches!(err, RoundError::WrongState { .. }));

    // Can't complete in Collecting state
    let err = round.complete([0u8; 32]).unwrap_err();
    assert!(matches!(err, RoundError::WrongState { .. }));
}

/// Tests duplicate participant registration is rejected.
#[test]
fn test_round_duplicate_registration_rejected() {
    let config = test_round_config(2);
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    round
        .register_participant("worker-0".to_string(), 1000)
        .unwrap();
    let err = round
        .register_participant("worker-0".to_string(), 1000)
        .unwrap_err();
    assert!(matches!(err, RoundError::AlreadyRegistered(_)));
}

/// Tests double gradient submission is rejected.
#[test]
fn test_round_double_submission_rejected() {
    let config = test_round_config(2);
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    round
        .register_participant("worker-0".to_string(), 1000)
        .unwrap();
    round
        .register_participant("worker-1".to_string(), 1000)
        .unwrap();
    round.start_collection().unwrap();

    let grad = make_gradient_submission("worker-0", &[0.1], 1);
    round.submit_gradient(grad.clone()).unwrap();

    // Second submission from same worker should fail
    let err = round.submit_gradient(grad).unwrap_err();
    assert!(matches!(err, RoundError::AlreadySubmitted(_)));
}

/// Tests insufficient stake is rejected.
#[test]
fn test_round_insufficient_stake_rejected() {
    let config = test_round_config(2);
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    // Config has min_stake=100, try with 50
    let err = round
        .register_participant("poor-worker".to_string(), 50)
        .unwrap_err();
    assert!(matches!(
        err,
        RoundError::InsufficientStake {
            required: 100,
            provided: 50
        }
    ));
}

/// Tests that unknown participant submissions are rejected.
#[test]
fn test_round_unknown_participant_rejected() {
    let config = test_round_config(2);
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    round
        .register_participant("worker-0".to_string(), 1000)
        .unwrap();
    round
        .register_participant("worker-1".to_string(), 1000)
        .unwrap();
    round.start_collection().unwrap();

    // Unknown participant tries to submit
    let grad = make_gradient_submission("unknown-worker", &[0.1], 1);
    let err = round.submit_gradient(grad).unwrap_err();
    assert!(matches!(err, RoundError::UnknownParticipant(_)));
}

// ============================================================================
// Round Manager Multi-Round Resilience Tests
// ============================================================================

/// Tests that RoundManager tracks success rate across mixed success/failure rounds.
#[test]
fn test_round_manager_mixed_success_failure() {
    let config = test_round_config(2);
    let mut manager = RoundManager::new(config);

    // Round 0: Success
    let commitment = compute_commitment(b"genesis");
    manager.start_round(commitment).unwrap();
    {
        let round = manager.current_mut().unwrap();
        round
            .register_participant("w0".to_string(), 1000)
            .unwrap();
        round
            .register_participant("w1".to_string(), 1000)
            .unwrap();
        round.start_collection().unwrap();
        let g0 = make_gradient_submission("w0", &[0.1], 1);
        let g1 = make_gradient_submission("w1", &[0.2], 1);
        round.submit_gradient(g0.clone()).unwrap();
        round.submit_gradient(g1.clone()).unwrap();
        round.start_aggregation().unwrap();
        round
            .set_aggregation_result(make_aggregation_result(&[g0, g1], 2))
            .unwrap();
        let new_commit = compute_commitment(b"model_v1");
        round.complete(new_commit).unwrap();
    }

    // Round 1: Failure (not enough submissions)
    let commitment = compute_commitment(b"model_v1");
    manager.start_round(commitment).unwrap();
    {
        let round = manager.current_mut().unwrap();
        round
            .register_participant("w0".to_string(), 1000)
            .unwrap();
        round
            .register_participant("w1".to_string(), 1000)
            .unwrap();
        round.start_collection().unwrap();
        // Only 1 submission, force failure
        round.fail(RoundFailure::CollectionTimeout);
        assert!(round.is_terminal());
    }

    // Round 2: Success
    let commitment = compute_commitment(b"model_v1"); // Same as before since round 1 failed
    manager.start_round(commitment).unwrap();
    {
        let round = manager.current_mut().unwrap();
        round
            .register_participant("w0".to_string(), 1000)
            .unwrap();
        round
            .register_participant("w1".to_string(), 1000)
            .unwrap();
        round.start_collection().unwrap();
        let g0 = make_gradient_submission("w0", &[0.3], 3);
        let g1 = make_gradient_submission("w1", &[0.4], 3);
        round.submit_gradient(g0.clone()).unwrap();
        round.submit_gradient(g1.clone()).unwrap();
        round.start_aggregation().unwrap();
        round
            .set_aggregation_result(make_aggregation_result(&[g0, g1], 2))
            .unwrap();
        let new_commit = compute_commitment(b"model_v2");
        round.complete(new_commit).unwrap();
    }

    // Round 0 and 1 are archived; round 2 is current
    assert_eq!(manager.completed_count(), 2);
    // 1 success + 1 failure = 50% success rate
    assert!((manager.success_rate() - 0.5).abs() < 1e-10);
}

/// Tests that starting a new round while another is active fails.
#[test]
fn test_round_manager_active_round_blocks_new() {
    let config = test_round_config(2);
    let mut manager = RoundManager::new(config);

    let commitment = compute_commitment(b"genesis");
    manager.start_round(commitment).unwrap();

    // Round is in Initializing state (not terminal) — can't start another
    let err = manager.start_round(commitment).unwrap_err();
    assert!(matches!(err, RoundError::WrongState { .. }));
}

/// Tests stake-weighted submission progress tracking.
#[test]
fn test_round_submission_progress_stake_weighted() {
    let config = test_round_config(2);
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    // Register workers with different stakes
    round
        .register_participant("whale".to_string(), 8000)
        .unwrap();
    round
        .register_participant("small".to_string(), 2000)
        .unwrap();
    assert_eq!(round.total_stake(), 10000);

    round.start_collection().unwrap();

    // Small worker submits: 2000/10000 = 20% progress
    let grad = make_gradient_submission("small", &[0.1], 1);
    round.submit_gradient(grad).unwrap();
    assert!((round.submission_progress() - 0.2).abs() < 1e-10);
    assert_eq!(round.submitted_stake(), 2000);

    // Whale submits: 10000/10000 = 100% progress
    let grad = make_gradient_submission("whale", &[0.2], 1);
    round.submit_gradient(grad).unwrap();
    assert!((round.submission_progress() - 1.0).abs() < 1e-10);
}

// ============================================================================
// Orchestrator + Real Proof Generation Integration Tests
// ============================================================================

/// Simulates a worker that generates a real ZK proof for a training step.
fn worker_prove_step(
    weights: &TrainingWeights,
    x: &[Fr],
    target: &[Fr],
    step_number: u64,
) -> (TrainingProofResultV2, TrainingWeights) {
    let prover = MLTrainingProverV2::new(weights.d_in, weights.d_hid, weights.d_out);

    let witness = MLTrainingProverV2::build_witness(
        weights.d_in,
        weights.d_hid,
        weights.d_out,
        x,
        target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        step_number,
        Fr::from(1u64),
    );

    let result = prover.prove(&witness).expect("Proof generation failed");

    let new_weights = TrainingWeights::new(
        weights.d_in,
        weights.d_hid,
        weights.d_out,
        witness.w1_new.clone(),
        witness.b1_new.clone(),
        witness.w2_new.clone(),
        witness.b2_new.clone(),
    );

    (result, new_weights)
}

/// Converts a proof result into a gradient submission for the round manager.
fn proof_to_submission(
    participant_id: &str,
    proof_result: &TrainingProofResultV2,
    step: u64,
) -> GradientSubmission {
    let mut hasher = Sha256::new();
    hasher.update(&proof_result.proof);
    let gradient_hash: [u8; 32] = hasher.finalize().into();

    GradientSubmission {
        participant_id: participant_id.to_string(),
        gradient_hash,
        gradient_data: proof_result.proof.clone(),
        proof: proof_result.proof.clone(),
        public_inputs: vec![],
        error_bound: 0.01,
        timestamp: step,
    }
}

/// Full orchestrator lifecycle with real proof generation: spawn 2 workers,
/// run 3 rounds, verify commitment chain is valid and proofs verify.
#[test]
fn test_orchestrator_with_real_proofs_three_rounds() {
    let config = test_round_config(2);
    let mut manager = RoundManager::new(config);

    let d_in = 2;
    let d_hid = 2;
    let d_out = 1;

    // Initial model weights (small values for circuit safety)
    let mut weights = TrainingWeights::new(
        d_in, d_hid, d_out,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    );

    let training_samples: Vec<(Vec<Fr>, Vec<Fr>)> = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(3u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(4u64)]),
    ];

    // Track state hash chain for verification
    let initial_hash = compute_state_hash_v2(
        &weights.w1, &weights.b1, &weights.w2, &weights.b2,
    );
    let mut prev_commitment = compute_commitment(initial_hash.0.to_repr().as_ref());
    let mut all_proofs: Vec<TrainingProofResultV2> = Vec::new();

    for round_idx in 0..3 {
        let step_number = (round_idx + 1) as u64;
        let (x, target) = &training_samples[round_idx];

        // Start round in manager
        let round_id = manager.start_round(prev_commitment).unwrap();
        assert_eq!(round_id, RoundId::new(round_idx as u64));

        let round = manager.current_mut().unwrap();

        // Register 2 workers
        round.register_participant("worker-0".to_string(), 1000).unwrap();
        round.register_participant("worker-1".to_string(), 1000).unwrap();
        round.start_collection().unwrap();

        // Both workers generate real proofs with the same model state
        let (proof_result, new_weights) = worker_prove_step(&weights, x, target, step_number);
        assert!(proof_result.verified, "Round {}: proof should self-verify", round_idx);

        // Workers submit gradient (containing real proof data)
        let sub0 = proof_to_submission("worker-0", &proof_result, step_number);
        let sub1 = proof_to_submission("worker-1", &proof_result, step_number);
        round.submit_gradient(sub0.clone()).unwrap();
        round.submit_gradient(sub1.clone()).unwrap();
        assert_eq!(round.submission_count(), 2);

        // Aggregate
        round.start_aggregation().unwrap();
        let agg_result = make_aggregation_result(&[sub0, sub1], 2);
        round.set_aggregation_result(agg_result).unwrap();

        // Complete round with new commitment
        let new_hash = compute_state_hash_v2(
            &new_weights.w1, &new_weights.b1, &new_weights.w2, &new_weights.b2,
        );
        let new_commitment = compute_commitment(new_hash.0.to_repr().as_ref());
        round.complete(new_commitment).unwrap();
        assert_eq!(round.state, RoundState::Completed);

        // Verify the proof chain: this step's old hash matches previous state
        if round_idx > 0 {
            assert_eq!(
                proof_result.old_state_hash, all_proofs[round_idx - 1].new_state_hash,
                "Round {}: old state hash must chain from previous round's new state hash",
                round_idx
            );
        }

        all_proofs.push(proof_result);
        prev_commitment = new_commitment;
        weights = new_weights;
    }

    // Verify full proof chain integrity
    for i in 0..all_proofs.len() - 1 {
        assert_eq!(
            all_proofs[i].new_state_hash, all_proofs[i + 1].old_state_hash,
            "Proof chain broken at step {}: new_hash != next old_hash",
            i
        );
    }

    // Manager should have 2 archived rounds + 1 current
    assert_eq!(manager.completed_count(), 2);
    assert!((manager.success_rate() - 1.0).abs() < 1e-10);

    eprintln!(
        "Orchestrator + real proofs test PASSED: 3 rounds with commitment chain verified"
    );
}

/// Worker failure recovery test: one worker fails mid-round but the round
/// still completes because min_participants is met.
#[test]
fn test_orchestrator_worker_failure_with_real_proofs() {
    let config = test_round_config(2); // min_participants = 2
    let mut round = TrainingRound::new(
        RoundId::new(0),
        config,
        compute_commitment(b"model_v0"),
    );

    let d_in = 2;
    let d_hid = 2;
    let d_out = 1;

    let weights = TrainingWeights::new(
        d_in, d_hid, d_out,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    );

    // Register 3 workers (1 more than minimum)
    round.register_participant("worker-0".to_string(), 1000).unwrap();
    round.register_participant("worker-1".to_string(), 1000).unwrap();
    round.register_participant("worker-2-unreliable".to_string(), 1000).unwrap();
    round.start_collection().unwrap();

    // Workers 0 and 1 generate real proofs and submit
    let (proof_result, _) = worker_prove_step(
        &weights,
        &[Fr::from(1u64), Fr::from(1u64)],
        &[Fr::from(5u64)],
        1,
    );
    assert!(proof_result.verified, "Proof should verify");

    let sub0 = proof_to_submission("worker-0", &proof_result, 1);
    let sub1 = proof_to_submission("worker-1", &proof_result, 1);
    round.submit_gradient(sub0.clone()).unwrap();
    round.submit_gradient(sub1.clone()).unwrap();

    // Worker-2 "dies" — never submits
    assert_eq!(round.submission_count(), 2);

    // Should still be able to aggregate with 2 submissions (>= min_participants)
    round.start_aggregation().unwrap();
    let agg_result = make_aggregation_result(&[sub0, sub1], 2);
    round.set_aggregation_result(agg_result).unwrap();

    let final_commitment = compute_commitment(b"model_v1_after_recovery");
    round.complete(final_commitment).unwrap();
    assert_eq!(round.state, RoundState::Completed);

    // Verify the failed worker is registered but didn't submit
    let unreliable = round.participants.get("worker-2-unreliable").unwrap();
    assert!(!unreliable.submitted, "Unreliable worker should not have submitted");

    // Verify round completed despite one worker failing
    assert!(round.is_terminal());
    assert_eq!(round.final_commitment, Some(final_commitment));

    eprintln!(
        "Worker failure recovery with real proofs PASSED: round completed with 2/3 workers"
    );
}

/// Multi-round orchestrator test where round 2 fails (timeout) but round 3
/// recovers using the last successful commitment.
#[test]
fn test_orchestrator_failure_recovery_with_commitment_chain() {
    let config = test_round_config(2);
    let mut manager = RoundManager::new(config);

    let d_in = 2;
    let d_hid = 2;
    let d_out = 1;

    let mut weights = TrainingWeights::new(
        d_in, d_hid, d_out,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    );

    // Round 0: Success
    let initial_hash = compute_state_hash_v2(
        &weights.w1, &weights.b1, &weights.w2, &weights.b2,
    );
    let mut prev_commitment = compute_commitment(initial_hash.0.to_repr().as_ref());
    let saved_commitment = prev_commitment; // Save for recovery

    manager.start_round(prev_commitment).unwrap();
    {
        let round = manager.current_mut().unwrap();
        round.register_participant("w0".to_string(), 1000).unwrap();
        round.register_participant("w1".to_string(), 1000).unwrap();
        round.start_collection().unwrap();

        let (proof, new_w) = worker_prove_step(
            &weights,
            &[Fr::from(1u64), Fr::from(1u64)],
            &[Fr::from(5u64)],
            1,
        );

        let s0 = proof_to_submission("w0", &proof, 1);
        let s1 = proof_to_submission("w1", &proof, 1);
        round.submit_gradient(s0.clone()).unwrap();
        round.submit_gradient(s1.clone()).unwrap();
        round.start_aggregation().unwrap();
        round.set_aggregation_result(make_aggregation_result(&[s0, s1], 2)).unwrap();

        let new_hash = compute_state_hash_v2(
            &new_w.w1, &new_w.b1, &new_w.w2, &new_w.b2,
        );
        let new_commitment = compute_commitment(new_hash.0.to_repr().as_ref());
        round.complete(new_commitment).unwrap();

        prev_commitment = new_commitment;
        weights = new_w;
    }

    // Round 1: Failure (collection timeout — simulated)
    let pre_failure_commitment = prev_commitment;
    manager.start_round(prev_commitment).unwrap();
    {
        let round = manager.current_mut().unwrap();
        round.register_participant("w0".to_string(), 1000).unwrap();
        round.register_participant("w1".to_string(), 1000).unwrap();
        round.start_collection().unwrap();
        // No submissions — timeout
        round.fail(RoundFailure::CollectionTimeout);
        assert!(round.is_terminal());
    }
    // Commitment should NOT advance after failure
    // prev_commitment stays the same

    // Round 2: Recovery — use the last successful commitment
    manager.start_round(pre_failure_commitment).unwrap();
    {
        let round = manager.current_mut().unwrap();
        round.register_participant("w0".to_string(), 1000).unwrap();
        round.register_participant("w1".to_string(), 1000).unwrap();
        round.start_collection().unwrap();

        let (proof, _) = worker_prove_step(
            &weights,
            &[Fr::from(2u64), Fr::from(1u64)],
            &[Fr::from(3u64)],
            2,
        );

        let s0 = proof_to_submission("w0", &proof, 2);
        let s1 = proof_to_submission("w1", &proof, 2);
        round.submit_gradient(s0.clone()).unwrap();
        round.submit_gradient(s1.clone()).unwrap();
        round.start_aggregation().unwrap();
        round.set_aggregation_result(make_aggregation_result(&[s0, s1], 2)).unwrap();

        let recovery_commitment = compute_commitment(b"model_after_recovery");
        round.complete(recovery_commitment).unwrap();
    }

    // Should have 2 archived (success + failure), 1 current (success)
    assert_eq!(manager.completed_count(), 2);
    // 1 success + 1 failure in archived = 50%
    assert!((manager.success_rate() - 0.5).abs() < 1e-10);

    eprintln!(
        "Orchestrator failure recovery test PASSED: round 1 failed, round 2 recovered"
    );
}
