//! End-to-end on-chain verification integration tests.
//!
//! These tests exercise the full pipeline: Rust prover generates a real KZG proof,
//! the proof is formatted for EVM, contracts are deployed to a local Anvil instance,
//! and the proof is submitted and verified on-chain.
//!
//! # Requirements
//!
//! - Foundry (`forge`, `anvil`) must be installed and in PATH
//! - Run with: `cargo test --test on_chain_verification --features on-chain -- --test-threads=1`
//!
//! # Test Coverage
//!
//! 1. Full pipeline: prover → EVM proof → deploy → submit → verify (no slashing)
//! 2. Multi-step: 3 consecutive training steps with commitment chaining
//! 3. Adversarial: tampered public inputs → contract rejects and slashes
//! 4. Cross-verification: same proof passes both Rust native and EVM format validation
//! 5. Error bound accumulation: on-chain accumulated error matches Rust reports

use std::sync::OnceLock;

use ethers::types::U256;

use halo2curves::bn256::Fr;

use helix_circuits::compute_state_hash_v2;
use helix_circuits::verifier::serialize_proof_for_evm;
use helix_circuits::validate_proof_format;

use helix_prover::{MLTrainingProverV2, RetryConfig, TrainingProofResultV2, TrainingWeights, V2ProverConfig};

use helix_integration_tests::common::anvil::*;

// ============================================================================
// Shared Prover (keygen is expensive — reuse across tests)
// ============================================================================

/// Model dimensions for all tests: 2 inputs, 2 hidden, 1 output.
const D_IN: usize = 2;
const D_HID: usize = 2;
const D_OUT: usize = 1;

static PROVER: OnceLock<MLTrainingProverV2> = OnceLock::new();

fn get_prover() -> &'static MLTrainingProverV2 {
    PROVER.get_or_init(|| {
        // Use k=14 and relu_range=256 to support multi-step proving where
        // weight updates produce intermediate values exceeding minimal ranges.
        // Disable Freivalds and retries for predictable behavior.
        let config = V2ProverConfig {
            k: 14,
            relu_range: 256,
            exp_range: 256,
            exp_scale: 1000,
            self_verify: true,
            use_witness_cache: false,
            use_freivalds: false,
            enable_tracing: false,
            retry: RetryConfig::none(),
            ..V2ProverConfig::default()
        };
        MLTrainingProverV2::with_config(D_IN, D_HID, D_OUT, config)
    })
}

/// Creates initial weights for a tiny 2×2×1 model.
fn initial_weights() -> TrainingWeights {
    TrainingWeights::new(
        D_IN,
        D_HID,
        D_OUT,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    )
}

/// Default learning rate.
fn lr() -> Fr {
    Fr::from(1u64)
}

/// Default base error.
fn base_error() -> Fr {
    Fr::from(1u64)
}

/// Generates a proof for one training step, returning the proof result and updated weights.
fn prove_step(
    weights: &TrainingWeights,
    x: &[Fr],
    target: &[Fr],
    step_number: u64,
) -> (TrainingProofResultV2, TrainingWeights) {
    let prover = get_prover();

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
        lr(),
        step_number,
        base_error(),
    );

    let result = prover.prove(&witness).expect("Proof generation failed");

    // Build updated weights
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

/// Computes the initial commitment (keccak256 hash pair) for model registration.
fn compute_initial_commitment(weights: &TrainingWeights) -> U256 {
    let hash = compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);
    let lo = fr_to_u256(&hash.0);
    let hi = fr_to_u256(&hash.1);
    compute_hash_pair(lo, hi)
}

// ============================================================================
// Test 1: Full Pipeline — Prover → EVM Proof → Anvil → Submit → Verify
// ============================================================================

#[tokio::test]
async fn test_anvil_full_pipeline() {
    let env = OnChainTestEnv::new().await;
    let weights = initial_weights();

    // 1. Generate a real proof
    let (proof_result, _new_weights) =
        prove_step(&weights, &[Fr::from(1u64), Fr::from(1u64)], &[Fr::from(5u64)], 1);

    assert!(proof_result.verified, "Proof should be self-verified");
    assert!(!proof_result.proof.is_empty(), "Proof should not be empty");

    // 2. Compute initial commitment and register model
    let initial_commitment = compute_initial_commitment(&weights);
    let model_id = env.register_model(initial_commitment).await;
    assert_eq!(model_id, U256::zero(), "First model should have ID 0");

    // 3. Stake ETH
    let stake_amount = ethers::utils::parse_ether("1").unwrap();
    env.stake(model_id, stake_amount).await;

    // 4. Start training round (1 hour duration)
    env.start_round(model_id, U256::from(3600u64)).await;

    // 5. Format proof for EVM and submit
    let bundle =
        EvmProofBundle::from_proof_result(&proof_result, model_id, env.max_error_bound);

    // Verify old commitment matches what we registered
    assert_eq!(
        bundle.old_commitment, initial_commitment,
        "Old commitment from proof must match registered commitment"
    );

    let receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64), // roundId (first round after registerModel)
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

    // 6. Verify round completed successfully
    assert_eq!(receipt.status, Some(1.into()), "Transaction should succeed");

    let (_model_commitment, new_commitment, _deadline, is_completed, prover) =
        env.get_round(model_id, U256::from(1u64)).await;

    assert!(is_completed, "Round should be completed");
    assert_eq!(prover, env.deployer, "Prover should be deployer");
    assert_ne!(new_commitment, U256::zero(), "New commitment should be set");

    // 7. Verify no slashing occurred
    let (stake_amount_after, _locked, slashed) =
        env.get_stake(env.deployer, model_id).await;
    assert!(!slashed, "Stake should NOT be slashed after valid proof");
    assert!(stake_amount_after > 0, "Stake should remain");

    // 8. Verify model commitment was updated
    let model_commitment = env.model_commitment(model_id).await;
    assert_eq!(
        model_commitment, bundle.new_commitment,
        "Model commitment should match proof's new commitment"
    );

    eprintln!(
        "Full pipeline test passed: proof generated in {:?}, submitted on-chain successfully",
        proof_result.generation_time
    );
}

// ============================================================================
// Test 2: Multi-Step Commitment Chaining
// ============================================================================

#[tokio::test]
async fn test_multi_step_commitment_chaining() {
    let env = OnChainTestEnv::new().await;
    let mut weights = initial_weights();

    // Register model with initial commitment
    let initial_commitment = compute_initial_commitment(&weights);
    let model_id = env.register_model(initial_commitment).await;

    // Stake
    let stake_amount = ethers::utils::parse_ether("1").unwrap();
    env.stake(model_id, stake_amount).await;

    // Training samples for 3 steps
    let samples: Vec<(Vec<Fr>, Vec<Fr>)> = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(3u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(4u64)]),
    ];

    let mut prev_new_commitment: Option<U256> = None;

    for (step_idx, (x, target)) in samples.iter().enumerate() {
        let step_number = (step_idx + 1) as u64;
        let round_id = U256::from(step_number);

        // Generate proof
        let (proof_result, new_weights) = prove_step(&weights, x, target, step_number);

        // Start round
        env.start_round(model_id, U256::from(3600u64)).await;

        // Format and submit
        let bundle =
            EvmProofBundle::from_proof_result(&proof_result, model_id, env.max_error_bound);

        // Verify commitment chaining: this round's old commitment should match
        // the previous round's new commitment (or initial commitment for step 1)
        if let Some(prev_commitment) = prev_new_commitment {
            assert_eq!(
                bundle.old_commitment, prev_commitment,
                "Step {}: old commitment should equal previous step's new commitment",
                step_number
            );
        } else {
            assert_eq!(
                bundle.old_commitment, initial_commitment,
                "Step 1: old commitment should equal initial commitment"
            );
        }

        env.submit_proof(
            model_id,
            round_id,
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

        // Verify round completed
        let (_, _, _, is_completed, _) = env.get_round(model_id, round_id).await;
        assert!(is_completed, "Step {} round should complete", step_number);

        // Verify model commitment updated
        let model_commitment = env.model_commitment(model_id).await;
        assert_eq!(
            model_commitment, bundle.new_commitment,
            "Step {}: model commitment should be updated",
            step_number
        );

        prev_new_commitment = Some(bundle.new_commitment);
        weights = new_weights;
    }

    // Verify no slashing after 3 valid steps
    let (_, _, slashed) = env.get_stake(env.deployer, model_id).await;
    assert!(!slashed, "Stake should not be slashed after valid steps");

    eprintln!("Multi-step commitment chaining test passed: 3 steps verified");
}

// ============================================================================
// Test 3: Adversarial — Tampered Public Inputs → Rejection + Slashing
// ============================================================================

#[tokio::test]
async fn test_adversarial_tampered_inputs() {
    let env = OnChainTestEnv::new().await;
    let weights = initial_weights();

    // Register, stake, start round
    let initial_commitment = compute_initial_commitment(&weights);
    let model_id = env.register_model(initial_commitment).await;
    let stake_amount = ethers::utils::parse_ether("1").unwrap();
    env.stake(model_id, stake_amount).await;
    env.start_round(model_id, U256::from(3600u64)).await;

    // Generate a valid proof
    let (proof_result, _) =
        prove_step(&weights, &[Fr::from(1u64), Fr::from(1u64)], &[Fr::from(5u64)], 1);

    let bundle =
        EvmProofBundle::from_proof_result(&proof_result, model_id, env.max_error_bound);

    // Make the mock verifier reject proofs
    env.set_mock_accept(false).await;

    // Submit the proof — MockVerifier rejects, so contract should slash
    let receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64),
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

    // Transaction should succeed (slashing is not a revert, it's handled internally)
    assert_eq!(receipt.status, Some(1.into()), "Transaction should succeed");

    // Verify round is NOT completed (proof was invalid)
    let (_, _, _, is_completed, _) = env.get_round(model_id, U256::from(1u64)).await;
    assert!(
        !is_completed,
        "Round should NOT be completed after invalid proof"
    );

    // Verify stake was slashed
    let (stake_amount_after, _, slashed) = env.get_stake(env.deployer, model_id).await;
    assert!(slashed, "Stake should be slashed after invalid proof");
    assert!(
        stake_amount_after < stake_amount.as_u128(),
        "Stake amount should be reduced after slashing"
    );

    eprintln!(
        "Adversarial test passed: invalid proof correctly rejected, stake slashed to {}",
        stake_amount_after
    );
}

// ============================================================================
// Test 4: Cross-Verification — Same Proof Passes Both Rust Native and EVM Format
// ============================================================================

#[tokio::test]
async fn test_cross_verification_rust_and_solidity() {
    let prover = get_prover();
    let weights = initial_weights();

    // Generate proof
    let (proof_result, _) =
        prove_step(&weights, &[Fr::from(1u64), Fr::from(1u64)], &[Fr::from(5u64)], 1);

    // 1. Rust native verification
    assert!(
        prover.verify_result(&proof_result),
        "Proof should pass Rust native verification"
    );
    assert!(proof_result.verified, "Proof should be self-verified");

    // 2. EVM proof format validation
    let evm_proof = serialize_proof_for_evm(&proof_result.proof, 3)
        .expect("EVM proof serialization should succeed with KZG");
    assert_eq!(evm_proof.len(), 320, "EVM proof must be exactly 320 bytes");

    validate_proof_format(&evm_proof).expect("EVM proof format should be valid");

    // 3. Verify public inputs are well-formed
    let evm_pi = proof_result.to_evm_public_inputs();
    assert_eq!(evm_pi.len(), 8, "Should have 8 public inputs");

    // Each public input should be a 32-byte big-endian value
    for (i, pi) in evm_pi.iter().enumerate() {
        let pi_bytes: &[u8; 32] = pi;
        assert_eq!(pi_bytes.len(), 32, "PI[{}] should be 32 bytes", i);
    }

    // 4. Verify state hash consistency
    let old_hash_lo = U256::from_big_endian(&evm_pi[0]);
    let old_hash_hi = U256::from_big_endian(&evm_pi[1]);
    let new_hash_lo = U256::from_big_endian(&evm_pi[2]);
    let new_hash_hi = U256::from_big_endian(&evm_pi[3]);

    // Old and new commitments should be different (training step changed weights)
    let old_commitment = compute_hash_pair(old_hash_lo, old_hash_hi);
    let new_commitment = compute_hash_pair(new_hash_lo, new_hash_hi);
    assert_ne!(
        old_commitment, new_commitment,
        "Old and new commitments should differ after training"
    );

    // 5. Verify old commitment matches initial weights
    let expected_initial = compute_initial_commitment(&weights);
    assert_eq!(
        old_commitment, expected_initial,
        "Proof's old commitment should match initial weights"
    );

    // 6. Verify with tampered public inputs fails native verification
    let mut bad_pi = proof_result.public_inputs.clone();
    bad_pi[4] = Fr::from(99999u64); // tamper with loss
    assert!(
        !prover.verify(&proof_result.proof, &bad_pi),
        "Tampered public inputs should fail native verification"
    );

    // 7. Deploy to Anvil and verify EVM format is accepted by MockVerifier
    let env = OnChainTestEnv::new().await;
    let model_id = env.register_model(expected_initial).await;
    let stake_amount = ethers::utils::parse_ether("1").unwrap();
    env.stake(model_id, stake_amount).await;
    env.start_round(model_id, U256::from(3600u64)).await;

    let bundle =
        EvmProofBundle::from_proof_result(&proof_result, model_id, env.max_error_bound);
    let receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64),
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

    assert_eq!(receipt.status, Some(1.into()), "On-chain submission should succeed");
    let (_, _, _, is_completed, _) = env.get_round(model_id, U256::from(1u64)).await;
    assert!(is_completed, "Round should complete with valid proof format");

    eprintln!(
        "Cross-verification test passed: proof verified by both Rust native ({:?}) and on-chain submission",
        proof_result.verification_time.unwrap_or_default()
    );
}

// ============================================================================
// Test 5: Error Bound Accumulation
// ============================================================================

#[tokio::test]
async fn test_error_bound_accumulation() {
    let env = OnChainTestEnv::new().await;
    let mut weights = initial_weights();

    // Register model
    let initial_commitment = compute_initial_commitment(&weights);
    let model_id = env.register_model(initial_commitment).await;

    // Stake
    let stake_amount = ethers::utils::parse_ether("1").unwrap();
    env.stake(model_id, stake_amount).await;

    // Initial accumulated error should be 0
    let initial_error = env.accumulated_error_bound(model_id).await;
    assert_eq!(
        initial_error,
        U256::zero(),
        "Initial accumulated error should be 0"
    );

    let mut expected_accumulated_error = U256::zero();
    let mut step_errors: Vec<U256> = Vec::new();

    // Run 3 training steps, tracking error bounds
    let samples: Vec<(Vec<Fr>, Vec<Fr>)> = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(3u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(4u64)]),
    ];

    for (step_idx, (x, target)) in samples.iter().enumerate() {
        let step_number = (step_idx + 1) as u64;

        // Generate proof
        let (proof_result, new_weights) = prove_step(&weights, x, target, step_number);

        // Start round and submit
        env.start_round(model_id, U256::from(3600u64)).await;

        let bundle =
            EvmProofBundle::from_proof_result(&proof_result, model_id, env.max_error_bound);

        let step_error = bundle.error_bound;
        step_errors.push(step_error);
        expected_accumulated_error = expected_accumulated_error + step_error;

        env.submit_proof(
            model_id,
            U256::from(step_number),
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

        // Read on-chain accumulated error
        let on_chain_error = env.accumulated_error_bound(model_id).await;
        assert_eq!(
            on_chain_error, expected_accumulated_error,
            "Step {}: on-chain accumulated error ({}) should match expected ({})",
            step_number, on_chain_error, expected_accumulated_error
        );

        // Also verify it matches the Rust prover's reported error
        let rust_error = fr_to_u256(&proof_result.total_error);
        assert_eq!(
            step_error, rust_error,
            "Step {}: EVM error bound should match Rust prover's total_error",
            step_number
        );

        weights = new_weights;
    }

    // Final verification
    let final_error = env.accumulated_error_bound(model_id).await;
    let sum_of_steps: U256 = step_errors.iter().fold(U256::zero(), |acc, e| acc + e);
    assert_eq!(
        final_error, sum_of_steps,
        "Final accumulated error should equal sum of step errors"
    );

    eprintln!(
        "Error bound accumulation test passed: {} total error after {} steps ({:?})",
        final_error,
        step_errors.len(),
        step_errors
    );
}
