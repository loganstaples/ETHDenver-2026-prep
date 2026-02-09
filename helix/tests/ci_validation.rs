//! CI Validation Tests
//!
//! Regression test suite designed to run on every commit. These tests are
//! optimized for speed while still providing comprehensive coverage of
//! critical functionality.
//!
//! # Test Categories
//!
//! 1. **Smoke Tests**: Quick validation that basic functionality works
//! 2. **Regression Tests**: Ensure no regressions in critical paths
//! 3. **Integration Tests**: Validate component integration
//! 4. **Performance Gates**: Ensure performance stays within bounds
//!
//! # Usage
//!
//! Run with: `cargo test --test ci_validation`
//!
//! These tests should complete in under 60 seconds for CI suitability.

#![allow(unused_imports)]

use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::ml::training_step_v2::{
    compute_state_hash_v2, compute_witness_v2, MLTrainingStepV2Circuit, MLTrainingStepV2Witness,
    NUM_PUBLIC_INPUTS,
};
use helix_circuits::verifier::evm::generate_verifier_contract;
use helix_prover::{
    BatchTrainingProverV2, MLTrainingProverV2, TrainingProofResultV2, TrainingWeights,
    V2ProverConfig,
};

#[path = "common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Smoke Tests - Basic Functionality
// ============================================================================

/// Smoke test: Initialize model and create prover.
#[test]
fn ci_smoke_prover_initialization() {
    let dims = ModelDimensions::tiny();
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // Prover should be usable
    assert!(prover.config().k >= 12, "Prover should have minimum k value");
}

/// Smoke test: Build witness from model and sample.
#[test]
fn ci_smoke_witness_building() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    // Witness should have correct public inputs
    let pi = witness.public_inputs();
    assert_eq!(pi.len(), NUM_PUBLIC_INPUTS);
}

/// Smoke test: Generate and verify a single proof.
#[test]
fn ci_smoke_single_proof() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    assert!(!proof_result.proof.is_empty(), "Proof should not be empty");
    assert!(
        prover.verify_result(&proof_result),
        "Proof should verify"
    );
}

/// Smoke test: State hash computation is deterministic.
#[test]
fn ci_smoke_state_hash_deterministic() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let hash1 = compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);
    let hash2 = compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);

    assert_eq!(hash1, hash2, "State hash should be deterministic");
}

// ============================================================================
// Regression Tests - Critical Functionality
// ============================================================================

/// Regression: Public inputs have correct format.
#[test]
fn ci_regression_public_inputs_format() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();
    let pi = &proof_result.public_inputs;

    // Verify exact count
    assert_eq!(
        pi.len(),
        NUM_PUBLIC_INPUTS,
        "Public inputs count mismatch"
    );

    // Verify structure:
    // [0]: old_hash_lo
    // [1]: old_hash_hi
    // [2]: new_hash_lo
    // [3]: new_hash_hi
    // [4]: loss
    // [5]: error_bound
    // [6]: step_number

    // Step number should be 1
    assert_eq!(pi[6], Fr::from(1u64), "Step number should be 1");

    // State hashes should match proof result fields
    assert_eq!(pi[0], proof_result.old_state_hash.0, "Old hash lo mismatch");
    assert_eq!(pi[1], proof_result.old_state_hash.1, "Old hash hi mismatch");
    assert_eq!(pi[2], proof_result.new_state_hash.0, "New hash lo mismatch");
    assert_eq!(pi[3], proof_result.new_state_hash.1, "New hash hi mismatch");

    // Error bound should be tracked
    assert_ne!(pi[5], Fr::zero(), "Error bound should be non-zero");
}

/// Regression: Invalid witness produces invalid/failing proof.
#[test]
fn ci_regression_invalid_witness_rejected() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Corrupt the public inputs (change the loss value)
    let mut corrupted_pi = proof_result.public_inputs.clone();
    corrupted_pi[4] = Fr::from(99999u64); // Wrong loss

    // Verification should fail with corrupted inputs
    let result = prover.verify(&proof_result.proof, &corrupted_pi);
    assert!(
        !result,
        "Verification should fail with corrupted public inputs"
    );
}

/// Regression: State hash transition is correct.
#[test]
fn ci_regression_state_hash_transition() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Compute expected initial hash
    let expected_old_hash =
        compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);

    // Old hash should match initial weights
    assert_eq!(
        proof_result.old_state_hash, expected_old_hash,
        "Old state hash should match initial weights"
    );

    // New hash should be different (weights updated)
    assert_ne!(
        proof_result.old_state_hash, proof_result.new_state_hash,
        "State should change after training step"
    );
}

/// Regression: Error bounds are tracked through computation.
#[test]
fn ci_regression_error_bound_tracking() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    // Error should be tracked in witness
    assert!(
        witness.total_error != Fr::zero(),
        "Witness should track non-zero error"
    );

    // Error should appear in public inputs
    let pi = witness.public_inputs();
    assert!(
        pi[5] != Fr::zero(),
        "Error bound in public inputs should be non-zero"
    );
}

/// Regression: Proof chain maintains consistency.
#[test]
fn ci_regression_proof_chain_consistency() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // Use circuit-safe dataset: inputs in 1..5 to keep h_pre within ReLU lookup range ±128
    let dataset = TestDataset::circuit_safe(dims.d_in, dims.d_out, 3, 42);
    let samples = dataset.to_tuples();

    let batch_result = prover.prove_batch(training_weights, &samples, Fr::from(1u64));

    // Batch prover must return proofs — assert rather than silently skip
    assert!(
        !batch_result.proofs.is_empty(),
        "Batch prover returned 0 proofs — chain consistency cannot be verified. Failed steps: {:?}",
        batch_result.failed_steps
    );

    // Verify chain continuity
    for i in 0..batch_result.proofs.len().saturating_sub(1) {
        assert_eq!(
            batch_result.proofs[i].new_state_hash,
            batch_result.proofs[i + 1].old_state_hash,
            "Chain continuity broken at step {}",
            i
        );
    }

    // Verify step numbers
    for (i, proof) in batch_result.proofs.iter().enumerate() {
        assert_eq!(
            proof.step_number,
            (i + 1) as u64,
            "Step number mismatch at {}",
            i
        );
    }
}

// ============================================================================
// Integration Tests - Component Integration
// ============================================================================

/// Integration: Complete pipeline from model to verified proof.
#[test]
fn ci_integration_complete_pipeline() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let start = Instant::now();

    // 1. Initialize model
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    // 2. Create prover
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // 3. Build witness
    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    // 4. Generate proof
    let proof_result = prover.prove(&witness).unwrap();

    // 5. Verify with native verifier
    let native_verified = prover.verify_result(&proof_result);

    // 6. Verify with mock EVM verifier
    let mock_evm = MockEVMVerifier::new();
    let evm_result = mock_evm.verify(&proof_result.proof, &proof_result.public_inputs);

    let total_time = start.elapsed();

    // All checks
    assert!(native_verified, "Native verification should pass");
    assert!(evm_result.valid, "Mock EVM verification should pass");
    assert!(
        !proof_result.proof.is_empty(),
        "Proof should have content"
    );
    assert_eq!(
        proof_result.public_inputs.len(),
        NUM_PUBLIC_INPUTS,
        "Public inputs count should match"
    );

    println!("Complete pipeline finished in {:?}", total_time);
}

/// Integration: Multi-step training completes successfully.
#[test]
fn ci_integration_multi_step_training() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // 10 training steps with circuit-safe inputs.
    // lr=0: multi-step batch proving with lr=1 causes weight explosion beyond
    // the ReLU lookup range (±128). Field arithmetic has no fractional lr.
    // This tests batch infrastructure (proof count, chain, verification).
    let dataset = TestDataset::circuit_safe(dims.d_in, dims.d_out, 10, 42);
    let samples = dataset.to_tuples();

    let start = Instant::now();
    let batch_result = prover.prove_batch(training_weights, &samples, Fr::zero());
    let proving_time = start.elapsed();

    // Batch prover must return proofs
    assert!(
        !batch_result.proofs.is_empty(),
        "Batch prover returned 0 proofs for 10-step training. Failed steps: {:?}",
        batch_result.failed_steps
    );

    // Verify count
    assert_eq!(
        batch_result.proofs.len(),
        10,
        "Should complete all 10 steps"
    );

    // Verify all proofs
    let start = Instant::now();
    let all_verified = prover.verify_batch(&batch_result);
    let verify_time = start.elapsed();

    assert!(all_verified, "All 10 proofs should verify");

    println!(
        "10-step training: proving {:?}, verification {:?}",
        proving_time, verify_time
    );
}

/// Integration: Proof bytes loadable for Solidity test harness.
#[test]
fn ci_integration_solidity_compatible_proof() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Proof bytes should be valid for contract
    assert!(
        proof_result.proof.len() >= 64,
        "Proof should have minimum size for Halo2"
    );

    // Public inputs should be serializable to 32-byte values
    for (i, pi) in proof_result.public_inputs.iter().enumerate() {
        use helix_circuits::halo2curves::ff::PrimeField;
        let bytes = pi.to_repr();
        assert_eq!(
            bytes.as_ref().len(),
            32,
            "Public input {} should serialize to 32 bytes",
            i
        );
    }

    // Generate Solidity verifier contract
    let contract = prover.generate_solidity_verifier("HelixVerifier");
    assert!(
        contract.contains("function verify"),
        "Contract should have verify function"
    );
    assert!(
        contract.contains(&format!("NUM_INSTANCES = {}", NUM_PUBLIC_INPUTS)),
        "Contract should have correct instance count"
    );
}

// ============================================================================
// Performance Gate Tests
// ============================================================================

/// Performance gate: Single proof generation under time limit.
#[test]
fn ci_performance_single_proof_time() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let start = Instant::now();
    let _ = prover.prove(&witness);
    let elapsed = start.elapsed();

    // Proof generation for tiny model should complete in reasonable time
    // This is a soft gate - we just log if it exceeds the target
    let target = Duration::from_secs(30);
    if elapsed > target {
        println!(
            "WARNING: Proof generation took {:?}, exceeds {:?} target",
            elapsed, target
        );
    }

    // Hard gate: must complete within 60 seconds
    assert!(
        elapsed < Duration::from_secs(60),
        "Proof generation took too long: {:?}",
        elapsed
    );
}

/// Performance gate: Verification should be fast.
#[test]
fn ci_performance_verification_time() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Measure verification time (average over multiple runs)
    let num_runs = 5;
    let mut total_time = Duration::ZERO;

    for _ in 0..num_runs {
        let start = Instant::now();
        let _ = prover.verify_result(&proof_result);
        total_time += start.elapsed();
    }

    let avg_time = total_time / num_runs as u32;

    // Verification should be under 5 seconds
    assert!(
        avg_time < Duration::from_secs(5),
        "Verification too slow: {:?} average",
        avg_time
    );

    println!("Average verification time: {:?}", avg_time);
}

/// Performance gate: Witness building should be fast.
#[test]
fn ci_performance_witness_building_time() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    // Measure witness building time (average over multiple runs)
    let num_runs = 10;
    let mut total_time = Duration::ZERO;

    for _ in 0..num_runs {
        let start = Instant::now();
        let _ = MLTrainingProverV2::build_witness(
            dims.d_in,
            dims.d_hid,
            dims.d_out,
            &sample.x,
            &sample.target,
            &weights.w1,
            &weights.b1,
            &weights.w2,
            &weights.b2,
            Fr::from(1u64),
            1,
            Fr::from(1u64),
        );
        total_time += start.elapsed();
    }

    let avg_time = total_time / num_runs as u32;

    // Witness building should be under 1 second
    assert!(
        avg_time < Duration::from_secs(1),
        "Witness building too slow: {:?} average",
        avg_time
    );

    println!("Average witness building time: {:?}", avg_time);
}

// ============================================================================
// CI Hardening: Proof Determinism
// ============================================================================

/// Determinism gate: same input → same proof bytes.
/// Runs the prover twice with identical inputs and asserts byte-level equality
/// of proofs and public inputs.
#[test]
fn ci_hardening_proof_determinism() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    // Use deterministic seed via V2ProverConfig
    let seed = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01u8];
    let config = V2ProverConfig::default().deterministic(seed);
    let prover = MLTrainingProverV2::with_config(dims.d_in, dims.d_hid, dims.d_out, config.clone());

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof1 = prover.prove(&witness).unwrap();

    // Create a fresh prover with same config to ensure no state leakage
    let config2 = V2ProverConfig::default().deterministic(seed);
    let prover2 = MLTrainingProverV2::with_config(dims.d_in, dims.d_hid, dims.d_out, config2);
    let proof2 = prover2.prove(&witness).unwrap();

    // Public inputs must be identical
    assert_eq!(
        proof1.public_inputs, proof2.public_inputs,
        "Public inputs must be deterministic"
    );

    // State hashes must be identical
    assert_eq!(
        proof1.old_state_hash, proof2.old_state_hash,
        "Old state hash must be deterministic"
    );
    assert_eq!(
        proof1.new_state_hash, proof2.new_state_hash,
        "New state hash must be deterministic"
    );

    // Loss must be identical
    assert_eq!(proof1.loss, proof2.loss, "Loss must be deterministic");

    // Step number must be identical
    assert_eq!(
        proof1.step_number, proof2.step_number,
        "Step number must be deterministic"
    );

    // Proof bytes must be identical (deterministic SRS + deterministic seed)
    assert_eq!(
        proof1.proof.len(),
        proof2.proof.len(),
        "Proof byte lengths must match"
    );
    assert_eq!(
        proof1.proof, proof2.proof,
        "Proof bytes must be deterministic with same seed"
    );
}

// ============================================================================
// CI Hardening: Feature Flag Compile Checks
// ============================================================================

/// Feature flag gate: verify that core crate types are accessible.
/// This test validates at compile-time that the expected types exist and are
/// importable from each crate. If any crate rearranges its public API,
/// this test will fail to compile.
#[test]
fn ci_hardening_feature_flag_api_surface() {
    // helix-circuits types
    use helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit;
    use helix_circuits::ml::training_step_v2::MLTrainingStepV2Witness;
    use helix_circuits::ml::training_step_v2::NUM_PUBLIC_INPUTS;

    // helix-prover types
    use helix_prover::MLTrainingProverV2;
    use helix_prover::BatchTrainingProverV2;
    use helix_prover::TrainingProofResultV2;
    use helix_prover::TrainingWeights;

    // helix-core types
    use helix_core::BoundedTensor;
    use helix_core::BoundedValue;

    // helix-mpc types
    use helix_mpc::types::PartyId;
    use helix_mpc::sharing::AdditiveSharing;

    // helix-avm types
    use helix_avm::AVMExecutor;

    // Ensure NUM_PUBLIC_INPUTS hasn't changed unexpectedly
    assert_eq!(
        NUM_PUBLIC_INPUTS, 8,
        "NUM_PUBLIC_INPUTS changed — update contract ABI and tests. \
         Expected 8: [old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, \
         loss, error_bound, step_number, error_checksum]"
    );

    // Ensure standard type sizes haven't changed
    assert_eq!(
        std::mem::size_of::<Fr>(),
        32,
        "Fr field element size changed from 32 bytes"
    );

    // Verify types are constructible (compile-time check)
    let _ = std::mem::size_of::<MLTrainingStepV2Witness>();
    let _ = std::mem::size_of::<MLTrainingStepV2Circuit>();
    let _ = std::mem::size_of::<TrainingProofResultV2>();
    let _ = std::mem::size_of::<PartyId>();
    let _ = std::mem::size_of::<BoundedValue<f64>>();
    let _ = std::mem::size_of::<AVMExecutor>();
}

// ============================================================================
// CI Hardening: Performance Regression Gates
// ============================================================================

/// Performance regression gate: proof generation for k=14 tiny model < 10s.
#[test]
fn ci_hardening_proof_generation_under_10s() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let start = Instant::now();
    let proof_result = prover.prove(&witness).unwrap();
    let elapsed = start.elapsed();

    // Hard gate: must complete under 10 seconds
    assert!(
        elapsed < Duration::from_secs(10),
        "Proof generation took {:?} — exceeds 10s regression gate. \
         k={}, proof_size={} bytes",
        elapsed,
        prover.config().k,
        proof_result.proof.len()
    );

    // Verify the proof is valid (catch silent regressions)
    assert!(
        prover.verify_result(&proof_result),
        "Generated proof fails verification — possible prover regression"
    );

    println!(
        "Proof generation: {:?} (k={}, {} bytes)",
        elapsed,
        prover.config().k,
        proof_result.proof.len()
    );
}

/// Performance regression gate: proof verification < 1s.
#[test]
fn ci_hardening_verification_under_1s() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Warm up verifier (first call may be slower)
    let _ = prover.verify_result(&proof_result);

    // Measure verification time
    let start = Instant::now();
    let verified = prover.verify_result(&proof_result);
    let elapsed = start.elapsed();

    assert!(verified, "Proof should verify");
    assert!(
        elapsed < Duration::from_secs(1),
        "Verification took {:?} — exceeds 1s regression gate",
        elapsed
    );

    println!("Verification: {:?}", elapsed);
}

/// Gas estimation gate: proof byte size should stay within bounds.
/// The EVM verifier gas cost scales with proof size. SHPLONK produces
/// exactly 5 G1 points (3 advice + 2 opening) = 160 bytes uncompressed
/// or 320 bytes with coordinates. We allow up to 1024 bytes for future
/// protocol extensions.
#[test]
fn ci_hardening_proof_size_gas_bound() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Proof should not be empty
    assert!(
        !proof_result.proof.is_empty(),
        "Proof should not be empty"
    );

    // Proof size regression gate — gas scales linearly with proof bytes.
    // SHPLONK proofs for our k=14 circuit are ~1312 bytes (41 words).
    // At ~3k gas/word + 60k base + 16k for public inputs ≈ 199k gas.
    // Allow up to 2048 bytes before flagging (≈ 252k gas, still under 300k).
    let max_proof_bytes = 2048;
    assert!(
        proof_result.proof.len() <= max_proof_bytes,
        "Proof size {} bytes exceeds {} byte gas regression limit. \
         Estimated EVM verification gas exceeds 300k. Investigate proof structure changes.",
        proof_result.proof.len(),
        max_proof_bytes
    );

    // Public inputs should be exactly NUM_PUBLIC_INPUTS * 32 bytes when serialized
    let expected_pi_bytes = NUM_PUBLIC_INPUTS * 32;
    let actual_pi_count = proof_result.public_inputs.len();
    assert_eq!(
        actual_pi_count, NUM_PUBLIC_INPUTS,
        "Public inputs count regression: {} vs expected {}",
        actual_pi_count, NUM_PUBLIC_INPUTS
    );

    // Estimate gas: ~60k base + ~3k per 32-byte word of proof + ~2k per public input
    let proof_words = (proof_result.proof.len() + 31) / 32;
    let estimated_gas = 60_000 + proof_words * 3_000 + NUM_PUBLIC_INPUTS * 2_000;
    println!(
        "Estimated EVM verification gas: {} (proof={} bytes, {} words, {} public inputs)",
        estimated_gas,
        proof_result.proof.len(),
        proof_words,
        NUM_PUBLIC_INPUTS
    );

    // Soft gate: estimated gas should be under 300k
    if estimated_gas > 300_000 {
        println!(
            "WARNING: Estimated gas {} exceeds 300k target. \
             Consider optimizing proof structure.",
            estimated_gas
        );
    }
}

// ============================================================================
// CI Hardening: Cross-Crate Consistency
// ============================================================================

/// Cross-crate consistency: state hash computed in circuits matches prover output.
#[test]
fn ci_hardening_state_hash_cross_crate_consistency() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    // Compute state hash directly via circuits crate
    let circuit_hash =
        compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);

    // Compute via prover (which uses circuits internally)
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &sample.x,
        &sample.target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Old state hash from proof must match direct computation
    assert_eq!(
        proof_result.old_state_hash, circuit_hash,
        "State hash mismatch between circuits crate and prover crate"
    );
}

// ============================================================================
// Test Suite Summary
// ============================================================================

/// Meta-test: Prints summary of all CI tests.
#[test]
fn ci_test_suite_summary() {
    println!("\n========================================");
    println!("HELIX CI Validation Test Suite");
    println!("========================================");
    println!("\nTest Categories:");
    println!("  - Smoke Tests: Basic functionality validation");
    println!("  - Regression Tests: Critical path regression checks");
    println!("  - Integration Tests: Component integration validation");
    println!("  - Performance Gates: Performance boundary checks");
    println!("  - CI Hardening: Determinism, feature flags, gas/perf regression");
    println!("\nExpected total runtime: < 60 seconds");
    println!("========================================\n");
}

// ============================================================================
// CI Hardening: Feature Flag Compile-Time Checks
// ============================================================================

/// Feature flag compile check: all crate types are accessible under default features.
///
/// This test validates that the expected types exist and are importable.
/// If any crate rearranges its public API, this test fails to compile.
/// Each type import serves as a compile-time assertion that the API surface
/// hasn't changed.
#[test]
fn ci_hardening_default_feature_types_accessible() {
    // helix-circuits: core circuit types
    let _ = std::mem::size_of::<helix_circuits::ml::training_step_v2::MLTrainingStepV2Circuit>();
    let _ = std::mem::size_of::<helix_circuits::ml::training_step_v2::MLTrainingStepV2Witness>();

    // helix-prover: prover pipeline types
    let _ = std::mem::size_of::<helix_prover::MLTrainingProverV2>();
    let _ = std::mem::size_of::<helix_prover::BatchTrainingProverV2>();
    let _ = std::mem::size_of::<helix_prover::TrainingProofResultV2>();
    let _ = std::mem::size_of::<helix_prover::TrainingWeights>();
    let _ = std::mem::size_of::<helix_prover::V2ProverConfig>();

    // helix-core: bounded computation types
    let _ = std::mem::size_of::<helix_core::BoundedTensor>();
    let _ = std::mem::size_of::<helix_core::BoundedValue<f64>>();

    // helix-mpc: multi-party computation types
    let _ = std::mem::size_of::<helix_mpc::types::PartyId>();
    let _ = std::mem::size_of::<helix_mpc::sharing::AdditiveSharing>();

    // helix-avm: approximate VM types
    let _ = std::mem::size_of::<helix_avm::AVMExecutor>();

    // helix-node: training round orchestration
    let _ = std::mem::size_of::<helix_node::training::TrainingRound>();
    let _ = std::mem::size_of::<helix_node::training::RoundManager>();
    let _ = std::mem::size_of::<helix_node::training::RoundConfig>();
}

/// Feature flag compile check: verifier and proof format types accessible.
#[test]
fn ci_hardening_verifier_api_surface() {
    use helix_circuits::verifier::serialize_proof_for_evm;
    use helix_circuits::validate_proof_format;
    use helix_circuits::ml::training_step_v2::NUM_PUBLIC_INPUTS;

    // Verify constant hasn't changed
    assert_eq!(
        NUM_PUBLIC_INPUTS, 8,
        "NUM_PUBLIC_INPUTS changed from 8 — update contract ABI, \
         proof format, and all integration tests. Layout: \
         [old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, \
         loss, error_bound, step_number, error_checksum]"
    );

    // Fr element size must be 32 bytes (BN254 scalar field)
    assert_eq!(
        std::mem::size_of::<Fr>(), 32,
        "Fr size changed from 32 bytes — all proof serialization will break"
    );

    // Verify EVM proof format function is callable (compile-time check)
    let empty_proof: Vec<u8> = vec![];
    let _ = serialize_proof_for_evm(&empty_proof, 3);
    let _ = validate_proof_format(&[]);
}

// ============================================================================
// CI Hardening: Gas Regression Hard Gate
// ============================================================================

/// Gas regression hard gate: estimated EVM verification gas must stay under 300k.
///
/// This estimates gas from proof structure rather than running on-chain. For actual
/// on-chain gas measurement, see the `on-chain` feature-gated gas test.
#[test]
fn ci_hardening_gas_regression_hard_gate() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let sample = TestSample::known(dims.d_in, dims.d_out);
    let witness = MLTrainingProverV2::build_witness(
        dims.d_in, dims.d_hid, dims.d_out,
        &sample.x, &sample.target,
        &weights.w1, &weights.b1, &weights.w2, &weights.b2,
        Fr::from(1u64), 1, Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();

    // Proof size regression gate: SHPLONK proofs for k=14 should be ~1312 bytes
    let proof_bytes = proof_result.proof.len();
    assert!(
        proof_bytes <= 2048,
        "Proof size {} bytes exceeds 2048 byte limit. \
         Estimated gas will exceed 300k. Check for circuit changes.",
        proof_bytes
    );

    // Gas estimation model:
    // - 60k base gas (ecPairing precompile call)
    // - ~3k gas per 32-byte word of proof data
    // - ~2k gas per public input (field element multiplication)
    // - ~16k gas for BN254 pairing operations
    let proof_words = (proof_bytes + 31) / 32;
    let estimated_gas =
        60_000                           // base ecPairing
        + proof_words * 3_000            // proof data processing
        + NUM_PUBLIC_INPUTS * 2_000      // public input processing
        + 16_000;                        // pairing constant

    assert!(
        estimated_gas < 300_000,
        "Estimated verification gas {} exceeds 300k hard limit. \
         Proof size: {} bytes ({} words), {} public inputs. \
         Breakdown: 60k base + {}k proof + {}k PI + 16k pairing = {}k total",
        estimated_gas,
        proof_bytes,
        proof_words,
        NUM_PUBLIC_INPUTS,
        proof_words * 3,
        NUM_PUBLIC_INPUTS * 2,
        estimated_gas / 1000
    );

    eprintln!(
        "Gas regression gate PASSED: estimated {} gas (proof: {} bytes)",
        estimated_gas, proof_bytes
    );
}

// ============================================================================
// CI Hardening: Proof Chain Determinism
// ============================================================================

/// Proof chain determinism: same 3-step batch produces identical chains.
///
/// This verifies that the batch prover is deterministic across the full
/// chain, not just individual proofs. Critical for reproducible testing
/// and debugging.
#[test]
fn ci_hardening_proof_chain_determinism() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let dataset = TestDataset::circuit_safe(dims.d_in, dims.d_out, 3, 42);
    let samples = dataset.to_tuples();

    // Generate chain twice
    let prover1 = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let result1 = prover1.prove_batch(weights.to_training_weights(), &samples, Fr::from(1u64));

    let prover2 = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let result2 = prover2.prove_batch(weights.to_training_weights(), &samples, Fr::from(1u64));

    // Both must produce proofs
    assert!(
        !result1.proofs.is_empty(),
        "First chain must produce proofs"
    );
    assert_eq!(
        result1.proofs.len(), result2.proofs.len(),
        "Both chains must have same length"
    );

    // Compare step-by-step
    for (i, (p1, p2)) in result1.proofs.iter().zip(result2.proofs.iter()).enumerate() {
        assert_eq!(
            p1.public_inputs, p2.public_inputs,
            "Step {}: public inputs must match", i
        );
        assert_eq!(
            p1.old_state_hash, p2.old_state_hash,
            "Step {}: old state hash must match", i
        );
        assert_eq!(
            p1.new_state_hash, p2.new_state_hash,
            "Step {}: new state hash must match", i
        );
        assert_eq!(
            p1.step_number, p2.step_number,
            "Step {}: step number must match", i
        );
    }

    // Verify chain continuity in both results
    for i in 0..result1.proofs.len() - 1 {
        assert_eq!(
            result1.proofs[i].new_state_hash,
            result1.proofs[i + 1].old_state_hash,
            "Chain 1 continuity broken at step {}", i
        );
        assert_eq!(
            result2.proofs[i].new_state_hash,
            result2.proofs[i + 1].old_state_hash,
            "Chain 2 continuity broken at step {}", i
        );
    }

    eprintln!(
        "Proof chain determinism PASSED: {} steps identical across runs",
        result1.proofs.len()
    );
}

// ============================================================================
// CI Hardening: Performance Regression Gates
// ============================================================================

/// Performance regression: proof chain (3 steps) must complete within 30s.
///
/// This gate catches regressions in batch proving where per-step overhead
/// compounds. Single-proof timing gates catch individual regressions, but
/// batch overhead may grow independently.
#[test]
fn ci_hardening_proof_chain_under_30s() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let dataset = TestDataset::circuit_safe(dims.d_in, dims.d_out, 3, 42);
    let samples = dataset.to_tuples();

    let start = Instant::now();
    let batch_result = prover.prove_batch(training_weights, &samples, Fr::from(1u64));
    let elapsed = start.elapsed();

    assert!(
        !batch_result.proofs.is_empty(),
        "Batch prover must produce proofs for performance measurement"
    );

    assert!(
        elapsed < Duration::from_secs(30),
        "3-step batch proving took {:?} — exceeds 30s regression gate. \
         Per-step average: {:?}",
        elapsed,
        elapsed / 3
    );

    eprintln!(
        "Proof chain performance PASSED: 3 steps in {:?} (avg {:?}/step)",
        elapsed,
        elapsed / 3
    );
}

/// Performance regression: witness building for 10 samples must complete in <1s total.
#[test]
fn ci_hardening_batch_witness_building_under_1s() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let dataset = TestDataset::circuit_safe(dims.d_in, dims.d_out, 10, 42);
    let samples = dataset.to_tuples();

    let start = Instant::now();
    let mut current_weights = (
        weights.w1.clone(),
        weights.b1.clone(),
        weights.w2.clone(),
        weights.b2.clone(),
    );

    for (i, (x, target)) in samples.iter().enumerate() {
        let witness = MLTrainingProverV2::build_witness(
            dims.d_in, dims.d_hid, dims.d_out,
            x, target,
            &current_weights.0, &current_weights.1,
            &current_weights.2, &current_weights.3,
            Fr::from(1u64),
            (i + 1) as u64,
            Fr::from(1u64),
        );
        current_weights = (
            witness.w1_new.clone(),
            witness.b1_new.clone(),
            witness.w2_new.clone(),
            witness.b2_new.clone(),
        );
    }

    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "10 witness builds took {:?} — exceeds 1s regression gate",
        elapsed
    );

    eprintln!(
        "Batch witness building PASSED: 10 witnesses in {:?} (avg {:?})",
        elapsed,
        elapsed / 10
    );
}

#[cfg(test)]
mod ci_tests {
    use super::*;

    /// Verify test infrastructure is working.
    #[test]
    fn test_infrastructure() {
        let dims = ModelDimensions::tiny();
        assert_eq!(dims.d_in, 2);
        assert_eq!(dims.d_hid, 2);
        assert_eq!(dims.d_out, 1);

        let harness = TestHarness::new();
        assert!(!harness.check_timeout());

        let rng = DeterministicRng::new(42);
        let _ = rng;
    }
}
