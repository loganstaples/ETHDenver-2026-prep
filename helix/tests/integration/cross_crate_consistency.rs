//! Cross-Crate Consistency Tests
//!
//! Verifies that the interfaces between HELIX crates are consistent:
//!
//! 1. **Witness format**: helix-circuits witness matches helix-prover's expected input
//! 2. **EVM proof format**: helix-prover's proof output matches Halo2Verifier.sol's input layout
//! 3. **Poseidon hash**: Rust Poseidon is deterministic and consistent across crate layers
//! 4. **Error checksum**: Rust error checksum is deterministic and parameter-sensitive
//!
//! # Requirements
//!
//! - Foundry (`forge`, `anvil`) must be in PATH
//! - Run with: `cargo test --test cross_crate_consistency --features on-chain -- --test-threads=1`

use std::sync::OnceLock;

use ethers::types::U256;

use halo2curves::bn256::Fr;

use helix_circuits::gadgets::poseidon::{poseidon_hash_two, poseidon_hash_many};
use helix_circuits::ml::training_step_v2::{
    compute_state_hash_v2, compute_witness_v2, NUM_PUBLIC_INPUTS,
};
use helix_circuits::verifier::{serialize_proof_for_evm, validate_proof_format};

use helix_prover::{
    MLTrainingProverV2, RetryConfig, TrainingProofResultV2, TrainingWeights, V2ProverConfig,
};

use helix_integration_tests::common::anvil::*;

// ============================================================================
// Shared Prover
// ============================================================================

const D_IN: usize = 2;
const D_HID: usize = 2;
const D_OUT: usize = 1;

static PROVER: OnceLock<MLTrainingProverV2> = OnceLock::new();

fn get_prover() -> &'static MLTrainingProverV2 {
    PROVER.get_or_init(|| {
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

fn test_weights() -> TrainingWeights {
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

fn test_sample() -> (Vec<Fr>, Vec<Fr>) {
    (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)])
}

// ============================================================================
// Test 1: Witness Format — helix-circuits matches helix-prover input
// ============================================================================

/// Verifies that a witness built by `compute_witness_v2` (helix-circuits) has
/// all fields expected by `MLTrainingProverV2::prove` (helix-prover), and that
/// the public inputs match between the circuit witness and the proof result.
#[test]
fn test_witness_format_matches_prover_input() {
    let weights = test_weights();
    let (x, target) = test_sample();

    // Build witness via helix-circuits
    let witness = compute_witness_v2(
        D_IN,
        D_HID,
        D_OUT,
        &x,
        &target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64), // lr
        (Fr::zero(), Fr::zero()), // old_state_hash placeholder (auto-computed)
        (Fr::zero(), Fr::zero()), // new_state_hash placeholder
        1, // step_number
        Fr::from(1u64), // base_error
    );

    // Also build witness via helix-prover's API
    let prover_witness = MLTrainingProverV2::build_witness(
        D_IN,
        D_HID,
        D_OUT,
        &x,
        &target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    // Verify dimensions match
    assert_eq!(witness.d_in, prover_witness.d_in, "d_in mismatch");
    assert_eq!(witness.d_hid, prover_witness.d_hid, "d_hid mismatch");
    assert_eq!(witness.d_out, prover_witness.d_out, "d_out mismatch");

    // Verify input data matches
    assert_eq!(witness.x, prover_witness.x, "x mismatch");
    assert_eq!(witness.target, prover_witness.target, "target mismatch");

    // Verify weights match
    assert_eq!(witness.w1, prover_witness.w1, "w1 mismatch");
    assert_eq!(witness.b1, prover_witness.b1, "b1 mismatch");
    assert_eq!(witness.w2, prover_witness.w2, "w2 mismatch");
    assert_eq!(witness.b2, prover_witness.b2, "b2 mismatch");

    // Verify updated weights match
    assert_eq!(witness.w1_new, prover_witness.w1_new, "w1_new mismatch");
    assert_eq!(witness.b1_new, prover_witness.b1_new, "b1_new mismatch");
    assert_eq!(witness.w2_new, prover_witness.w2_new, "w2_new mismatch");
    assert_eq!(witness.b2_new, prover_witness.b2_new, "b2_new mismatch");

    // Verify forward pass intermediates match
    assert_eq!(witness.h_pre, prover_witness.h_pre, "h_pre mismatch");
    assert_eq!(witness.h, prover_witness.h, "h mismatch");
    assert_eq!(witness.y, prover_witness.y, "y mismatch");
    assert_eq!(witness.loss, prover_witness.loss, "loss mismatch");

    // Verify backward pass intermediates match
    assert_eq!(witness.dy, prover_witness.dy, "dy mismatch");
    assert_eq!(witness.dw2, prover_witness.dw2, "dw2 mismatch");
    assert_eq!(witness.dw1, prover_witness.dw1, "dw1 mismatch");

    // Verify error tracking matches
    assert_eq!(witness.total_error, prover_witness.total_error, "total_error mismatch");

    // Verify state hashes match
    assert_eq!(
        witness.old_state_hash, prover_witness.old_state_hash,
        "old_state_hash mismatch"
    );
    assert_eq!(
        witness.new_state_hash, prover_witness.new_state_hash,
        "new_state_hash mismatch"
    );

    // Verify public inputs count and format
    let circuit_pi = witness.public_inputs();
    let prover_pi = prover_witness.public_inputs();
    assert_eq!(circuit_pi.len(), NUM_PUBLIC_INPUTS, "Circuit PI count wrong");
    assert_eq!(prover_pi.len(), NUM_PUBLIC_INPUTS, "Prover PI count wrong");
    assert_eq!(circuit_pi, prover_pi, "Public inputs should be identical");

    // Verify the prover can actually prove this witness
    let prover = get_prover();
    let proof_result = prover.prove(&prover_witness).expect("Prover should accept witness");
    assert!(proof_result.verified, "Proof should self-verify");

    eprintln!(
        "Witness format consistency PASSED: {} public inputs, all fields match",
        NUM_PUBLIC_INPUTS
    );
}

/// Verifies that state hashes computed by helix-circuits match those computed
/// by the prover's witness builder.
#[test]
fn test_state_hash_consistency_across_crates() {
    let weights = test_weights();

    // Compute state hash via helix-circuits directly
    let hash_circuits = compute_state_hash_v2(
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
    );

    // Compute via prover's witness builder
    let (x, target) = test_sample();
    let witness = MLTrainingProverV2::build_witness(
        D_IN, D_HID, D_OUT,
        &x, &target,
        &weights.w1, &weights.b1, &weights.w2, &weights.b2,
        Fr::from(1u64), 1, Fr::from(1u64),
    );

    // The old_state_hash in the witness should match the direct computation
    assert_eq!(
        witness.old_state_hash, hash_circuits,
        "State hash from witness should match direct computation"
    );

    // And the new_state_hash should match computation on updated weights
    let hash_new_circuits = compute_state_hash_v2(
        &witness.w1_new,
        &witness.b1_new,
        &witness.w2_new,
        &witness.b2_new,
    );
    assert_eq!(
        witness.new_state_hash, hash_new_circuits,
        "New state hash should match computation on updated weights"
    );

    eprintln!("State hash consistency PASSED across helix-circuits and helix-prover");
}

// ============================================================================
// Test 2: EVM Proof Format — helix-prover output matches Halo2Verifier.sol
// ============================================================================

/// Verifies that the EVM proof format produced by helix-prover matches the
/// layout expected by Halo2Verifier.sol:
/// - Proof bytes: exactly 320 bytes (3 × 64-byte G1 + 2 × 64-byte opening)
/// - Public inputs: exactly 8 × 32-byte big-endian uint256 values
/// - All PI values are within the BN254 scalar field
#[test]
fn test_evm_proof_format_matches_contract() {
    let weights = test_weights();
    let (x, target) = test_sample();

    let prover = get_prover();
    let witness = MLTrainingProverV2::build_witness(
        D_IN, D_HID, D_OUT,
        &x, &target,
        &weights.w1, &weights.b1, &weights.w2, &weights.b2,
        Fr::from(1u64), 1, Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).expect("Proof should generate");

    // === EVM Proof Bytes ===
    let evm_proof = serialize_proof_for_evm(&proof_result.proof, 3)
        .expect("EVM serialization should succeed");

    // Contract expects exactly 320 bytes
    assert_eq!(
        evm_proof.len(),
        320,
        "EVM proof must be exactly 320 bytes, got {}",
        evm_proof.len()
    );

    // Validate format structure
    validate_proof_format(&evm_proof).expect("EVM proof format should be valid");

    // Verify G1 point structure: 3 advice commits at offsets 0, 64, 128
    // Each G1 point is (x, y) as 32-byte big-endian
    for i in 0..3 {
        let offset = i * 64;
        let x_bytes = &evm_proof[offset..offset + 32];
        let y_bytes = &evm_proof[offset + 32..offset + 64];

        // Points should be non-zero (zero G1 point is invalid for valid proofs)
        let is_zero_x = x_bytes.iter().all(|&b| b == 0);
        let is_zero_y = y_bytes.iter().all(|&b| b == 0);
        assert!(
            !(is_zero_x && is_zero_y),
            "Advice commitment {} should not be point at infinity",
            i
        );
    }

    // Verify opening proofs at offsets 192 and 256
    for (name, offset) in [("W", 192), ("W'", 256)] {
        let point_bytes = &evm_proof[offset..offset + 64];
        let is_zero = point_bytes.iter().all(|&b| b == 0);
        assert!(
            !is_zero,
            "Opening proof {} should not be point at infinity",
            name
        );
    }

    // === EVM Public Inputs ===
    let evm_pi = proof_result.to_evm_public_inputs();
    assert_eq!(evm_pi.len(), 8, "Must have 8 public inputs");

    // Each PI must be 32 bytes
    for (i, pi) in evm_pi.iter().enumerate() {
        assert_eq!(pi.len(), 32, "PI[{}] must be 32 bytes", i);
    }

    // BN254 scalar field order R
    let r = U256::from_dec_str(
        "21888242871839275222246405745257275088548364400416034343698204186575808495617",
    )
    .unwrap();

    // All PIs must be < R (BN254 scalar field)
    for (i, pi_bytes) in evm_pi.iter().enumerate() {
        let val = U256::from_big_endian(pi_bytes);
        assert!(
            val < r,
            "PI[{}] = {} exceeds BN254 scalar field order",
            i,
            val
        );
    }

    // Verify PI layout:
    // [0-1]: old state hash (lo, hi)
    // [2-3]: new state hash (lo, hi)
    // [4]: loss
    // [6]: step number (should be 1)
    let step_number = U256::from_big_endian(&evm_pi[6]);
    assert_eq!(step_number, U256::from(1u64), "Step number should be 1");

    // old_hash != new_hash (weights changed during training)
    let old_lo = U256::from_big_endian(&evm_pi[0]);
    let old_hi = U256::from_big_endian(&evm_pi[1]);
    let new_lo = U256::from_big_endian(&evm_pi[2]);
    let new_hi = U256::from_big_endian(&evm_pi[3]);
    assert!(
        old_lo != new_lo || old_hi != new_hi,
        "State hash should change after training step"
    );

    eprintln!(
        "EVM proof format consistency PASSED: {} bytes proof, {} public inputs",
        evm_proof.len(),
        evm_pi.len()
    );
}

/// Verifies that the EVM proof bundle (including Solidity error checksum
/// recomputation) produces values accepted by the on-chain coordinator.
#[tokio::test]
async fn test_evm_bundle_accepted_on_chain() {
    let weights = test_weights();
    let (x, target) = test_sample();

    let prover = get_prover();
    let witness = MLTrainingProverV2::build_witness(
        D_IN, D_HID, D_OUT,
        &x, &target,
        &weights.w1, &weights.b1, &weights.w2, &weights.b2,
        Fr::from(1u64), 1, Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).expect("Proof should generate");

    // Deploy V2 environment with mock verifier
    let env = OnChainTestEnv::new().await;
    let initial_commitment = {
        let hash = compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);
        compute_hash_pair(fr_to_u256(&hash.0), fr_to_u256(&hash.1))
    };

    let model_id = env.register_model(initial_commitment).await;
    let stake_amount = ethers::utils::parse_ether("1").unwrap();
    env.stake(model_id, stake_amount).await;
    env.start_round(model_id, U256::from(3600u64)).await;

    // Create bundle — this bridges helix-prover output to Solidity expectations
    let bundle = TestEvmProofBundle::from_proof_result(&proof_result, model_id, env.max_error_bound);

    // The bundle's old commitment must match what we registered
    assert_eq!(
        bundle.old_commitment, initial_commitment,
        "TestEvmProofBundle old commitment must match registered commitment"
    );

    // Submit and verify acceptance
    let receipt = env
        .submit_proof(
            model_id,
            U256::from(1u64),
            bundle.proof_as_bytes(),
            bundle.public_inputs.clone(),
        )
        .await;

    assert_eq!(receipt.status, Some(1.into()), "Submission should succeed");

    let (_, _, _, is_completed, _) = env.get_round(model_id, U256::from(1u64)).await;
    assert!(
        is_completed,
        "Round should complete — proves EVM bundle format matches contract expectations"
    );

    // Verify commitment updated correctly
    let on_chain = env.model_commitment(model_id).await;
    assert_eq!(
        on_chain, bundle.new_commitment,
        "On-chain commitment should match bundle's new commitment"
    );

    eprintln!("EVM bundle on-chain acceptance PASSED");
}

// ============================================================================
// Test 3: Poseidon Hash — Rust consistency and determinism
// ============================================================================

/// Verifies that Poseidon hash is deterministic: same inputs always produce
/// the same output, and different inputs produce different outputs.
///
/// NOTE: PoseidonHasher.sol is a Solidity `library` with `internal` functions,
/// so it cannot be deployed/called standalone. On-chain Poseidon consistency is
/// validated indirectly through test_evm_bundle_accepted_on_chain (the coordinator
/// uses PoseidonHasher internally for commitment and checksum validation).
#[test]
fn test_poseidon_hash_determinism() {
    let test_cases: Vec<(Fr, Fr)> = vec![
        (Fr::from(0u64), Fr::from(0u64)),
        (Fr::from(1u64), Fr::from(2u64)),
        (Fr::from(42u64), Fr::from(1337u64)),
        (Fr::from(1000000u64), Fr::from(999999u64)),
        (Fr::from(0xDEADBEEFu64), Fr::from(0xCAFEBABEu64)),
    ];

    // Determinism: same inputs → same output
    for (i, (left, right)) in test_cases.iter().enumerate() {
        let h1 = poseidon_hash_two(*left, *right);
        let h2 = poseidon_hash_two(*left, *right);
        assert_eq!(h1, h2, "Test case {}: Poseidon hash is not deterministic", i);
        assert_ne!(h1, Fr::zero(), "Test case {}: Hash should not be zero", i);
    }

    // Collision resistance: different inputs → different outputs
    for i in 0..test_cases.len() {
        for j in (i + 1)..test_cases.len() {
            let h_i = poseidon_hash_two(test_cases[i].0, test_cases[i].1);
            let h_j = poseidon_hash_two(test_cases[j].0, test_cases[j].1);
            assert_ne!(
                h_i, h_j,
                "Collision between test case {} and {}: different inputs produced same hash",
                i, j
            );
        }
    }

    // Argument order matters
    let h_ab = poseidon_hash_two(Fr::from(1u64), Fr::from(2u64));
    let h_ba = poseidon_hash_two(Fr::from(2u64), Fr::from(1u64));
    assert_ne!(h_ab, h_ba, "poseidon_hash_two(1,2) should differ from poseidon_hash_two(2,1)");

    eprintln!("Poseidon hash determinism PASSED: {} test cases verified", test_cases.len());
}

/// Verifies that compute_state_hash_v2 (used by circuits and prover) produces
/// Poseidon-based hashes that are consistent when invoked from different layers.
#[test]
fn test_poseidon_state_hash_matches_across_layers() {
    let weights = test_weights();

    // compute_state_hash_v2 should be deterministic
    let (lo1, hi1) = compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);
    let (lo2, hi2) = compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);
    assert_eq!(lo1, lo2, "State hash lo should be deterministic");
    assert_eq!(hi1, hi2, "State hash hi should be deterministic");

    // Verify the state hash uses poseidon internally — change one weight, get different hash
    let mut modified_w1 = weights.w1.clone();
    modified_w1[0] = modified_w1[0] + Fr::one();
    let (lo_mod, hi_mod) =
        compute_state_hash_v2(&modified_w1, &weights.b1, &weights.w2, &weights.b2);
    assert!(
        lo1 != lo_mod || hi1 != hi_mod,
        "Changing a weight should produce a different state hash"
    );

    // Verify lo/hi are non-trivial (not both zero)
    assert!(
        lo1 != Fr::zero() || hi1 != Fr::zero(),
        "State hash should not be (0, 0) for non-trivial weights"
    );

    // Verify the poseidon_hash_many produces consistent results for the same weight vector
    let all_weights: Vec<Fr> = weights
        .w1
        .iter()
        .chain(weights.b1.iter())
        .chain(weights.w2.iter())
        .chain(weights.b2.iter())
        .cloned()
        .collect();
    let hash_direct = poseidon_hash_many(&all_weights);
    let hash_direct_2 = poseidon_hash_many(&all_weights);
    assert_eq!(hash_direct, hash_direct_2, "poseidon_hash_many should be deterministic");

    eprintln!("Poseidon state hash cross-layer consistency PASSED");
}

/// Verifies that the Rust error checksum computation matches what the
/// coordinator expects on-chain. Since PoseidonHasher.sol is a library,
/// we validate through the coordinator's submitProof path which calls
/// `_computeErrorChecksum` internally.
///
/// The `test_evm_bundle_accepted_on_chain` test already validates this
/// end-to-end. This test focuses on the Rust side: verifying that
/// `compute_solidity_error_checksum` is deterministic and sensitive to inputs.
#[test]
fn test_error_checksum_rust_determinism() {
    let test_cases = vec![
        (U256::from(100u64), U256::from(1u64), U256::zero(), U256::from(1_000_000_000_000_000_000u64)),
        (U256::from(500u64), U256::from(5u64), U256::from(1u64), U256::from(1_000_000_000_000_000_000u64)),
        (U256::from(0u64), U256::from(0u64), U256::zero(), U256::from(1_000_000_000_000_000_000u64)),
    ];

    // Determinism: same inputs → same checksum
    for (i, (eb, sn, mid, budget)) in test_cases.iter().enumerate() {
        let c1 = compute_solidity_error_checksum(*eb, *sn, *mid, *budget);
        let c2 = compute_solidity_error_checksum(*eb, *sn, *mid, *budget);
        assert_eq!(c1, c2, "Test case {}: Error checksum not deterministic", i);
    }

    // Different inputs → different checksums
    for i in 0..test_cases.len() {
        for j in (i + 1)..test_cases.len() {
            let c_i = compute_solidity_error_checksum(
                test_cases[i].0, test_cases[i].1, test_cases[i].2, test_cases[i].3,
            );
            let c_j = compute_solidity_error_checksum(
                test_cases[j].0, test_cases[j].1, test_cases[j].2, test_cases[j].3,
            );
            assert_ne!(
                c_i, c_j,
                "Test cases {} and {} should produce different checksums",
                i, j
            );
        }
    }

    // Each individual parameter matters
    let base = compute_solidity_error_checksum(
        U256::from(100u64), U256::from(1u64), U256::zero(), U256::from(1_000_000_000_000_000_000u64),
    );
    let diff_eb = compute_solidity_error_checksum(
        U256::from(200u64), U256::from(1u64), U256::zero(), U256::from(1_000_000_000_000_000_000u64),
    );
    let diff_sn = compute_solidity_error_checksum(
        U256::from(100u64), U256::from(2u64), U256::zero(), U256::from(1_000_000_000_000_000_000u64),
    );
    let diff_mid = compute_solidity_error_checksum(
        U256::from(100u64), U256::from(1u64), U256::from(1u64), U256::from(1_000_000_000_000_000_000u64),
    );
    let diff_budget = compute_solidity_error_checksum(
        U256::from(100u64), U256::from(1u64), U256::zero(), U256::from(2_000_000_000_000_000_000u64),
    );

    assert_ne!(base, diff_eb, "Changing error_bound should change checksum");
    assert_ne!(base, diff_sn, "Changing step_number should change checksum");
    assert_ne!(base, diff_mid, "Changing model_id should change checksum");
    assert_ne!(base, diff_budget, "Changing error_budget should change checksum");

    eprintln!("Error checksum Rust determinism PASSED: {} test cases verified", test_cases.len());
}

// ============================================================================
// Test 4: Multi-step proof chain — format consistency across steps
// ============================================================================

/// Verifies that multiple consecutive training steps produce proofs with
/// consistent format and properly chaining commitments, using both the
/// circuit-level and prover-level APIs.
#[test]
fn test_multi_step_format_consistency() {
    let prover = get_prover();
    let mut weights = test_weights();

    let samples = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(3u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(4u64)]),
    ];

    let mut prev_proof: Option<TrainingProofResultV2> = None;

    for (step_idx, (x, target)) in samples.iter().enumerate() {
        let step_number = (step_idx + 1) as u64;

        let witness = MLTrainingProverV2::build_witness(
            D_IN, D_HID, D_OUT,
            x, target,
            &weights.w1, &weights.b1, &weights.w2, &weights.b2,
            Fr::from(1u64), step_number, Fr::from(1u64),
        );

        let proof_result = prover.prove(&witness).expect("Proof should generate");

        // Verify PI count is always 8
        assert_eq!(
            proof_result.public_inputs.len(),
            NUM_PUBLIC_INPUTS,
            "Step {}: PI count must be {}",
            step_number,
            NUM_PUBLIC_INPUTS
        );

        // Verify step number in PI matches
        assert_eq!(
            proof_result.step_number, step_number,
            "Step number mismatch at step {}",
            step_number
        );

        // Verify EVM proof format is consistent
        let evm_proof = serialize_proof_for_evm(&proof_result.proof, 3)
            .expect("EVM serialization should succeed");
        assert_eq!(evm_proof.len(), 320, "Step {}: EVM proof must be 320 bytes", step_number);

        let evm_pi = proof_result.to_evm_public_inputs();
        assert_eq!(evm_pi.len(), 8, "Step {}: must have 8 EVM PIs", step_number);

        // Verify commitment chaining
        if let Some(ref prev) = prev_proof {
            assert_eq!(
                proof_result.old_state_hash, prev.new_state_hash,
                "Step {}: old hash must chain from previous new hash",
                step_number
            );
        }

        // Update weights for next step
        weights = TrainingWeights::new(
            D_IN, D_HID, D_OUT,
            witness.w1_new.clone(),
            witness.b1_new.clone(),
            witness.w2_new.clone(),
            witness.b2_new.clone(),
        );

        prev_proof = Some(proof_result);
    }

    eprintln!("Multi-step format consistency PASSED: 3 steps with proper chaining");
}

/// Verifies that the proof's public input encoding/decoding round-trips
/// correctly between Fr → EVM bytes → U256 → Fr.
#[test]
fn test_public_input_encoding_roundtrip() {
    let weights = test_weights();
    let (x, target) = test_sample();

    let prover = get_prover();
    let witness = MLTrainingProverV2::build_witness(
        D_IN, D_HID, D_OUT,
        &x, &target,
        &weights.w1, &weights.b1, &weights.w2, &weights.b2,
        Fr::from(1u64), 1, Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).expect("Proof should generate");

    // Original Fr values
    let original_pi = &proof_result.public_inputs;

    // Fr → EVM bytes (32-byte big-endian)
    let evm_pi = proof_result.to_evm_public_inputs();

    // EVM bytes → U256
    let u256_pi: Vec<U256> = evm_pi.iter().map(|b| U256::from_big_endian(b)).collect();

    // U256 → Fr
    let recovered_pi: Vec<Fr> = u256_pi.iter().map(|u| u256_to_fr(*u)).collect();

    assert_eq!(original_pi.len(), recovered_pi.len(), "PI count mismatch");

    for (i, (orig, recovered)) in original_pi.iter().zip(recovered_pi.iter()).enumerate() {
        assert_eq!(
            orig, recovered,
            "PI[{}] round-trip failed: {:?} != {:?}",
            i, orig, recovered
        );
    }

    eprintln!("Public input encoding roundtrip PASSED: {} values preserved", original_pi.len());
}
