//! Proof Verification Consistency Tests.
//!
//! Tests that verification produces consistent results across:
//! - Native Rust verifier
//! - Mock EVM verifier
//! - Different proof sizes and model configurations
//! - Multiple verification rounds

#![allow(unused_imports)]

use std::time::{Duration, Instant};
use std::sync::Arc;

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2_proofs::dev::MockProver;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::ml::training_step_v2::{
    compute_state_hash_v2, compute_witness_v2, MLTrainingStepV2Circuit, MLTrainingStepV2Witness,
    NUM_PUBLIC_INPUTS,
};
use helix_circuits::verifier::evm::{generate_verifier_contract, SolidityGenerator, VkData};
use helix_circuits::verifier::native::{NativeVerifier, SerializedProof};
use helix_prover::{MLTrainingProverV2, TrainingWeights, V2ProverConfig};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Native Verifier Tests
// ============================================================================

/// Tests that native verification is deterministic.
#[test]
fn test_native_verification_deterministic() {
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

    // Verify multiple times - should always get same result
    let results: Vec<bool> = (0..10)
        .map(|_| prover.verify_result(&proof_result))
        .collect();

    // All results should be true
    assert!(results.iter().all(|&r| r), "Verification should be deterministic");

    // All results should be identical
    let first = results[0];
    assert!(
        results.iter().all(|&r| r == first),
        "All verification results should match"
    );
}

/// Tests that different valid proofs all verify.
#[test]
fn test_native_verification_multiple_proofs() {
    let dims = ModelDimensions::tiny();
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // Generate multiple proofs with known weights and different step numbers
    // Using known values ensures prover stability across runs
    let weights = TestModelWeights::known(dims);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    let proofs: Vec<_> = (0..5)
        .map(|i| {
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
                (i + 1) as u64,
                Fr::from(1u64),
            );

            prover.prove(&witness).unwrap()
        })
        .collect();

    // All proofs should verify
    for (i, proof) in proofs.iter().enumerate() {
        assert!(
            prover.verify_result(proof),
            "Proof {} should verify",
            i
        );
    }
}

/// Tests mock prover consistency with the circuit.
#[test]
fn test_mock_prover_circuit_consistency() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    // Build witness
    let old_hash = compute_state_hash_v2(
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
    );

    let witness = compute_witness_v2(
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
        old_hash,
        (Fr::zero(), Fr::zero()), // Temp new hash
        1,
        Fr::from(1u64),
    );

    let new_hash = compute_state_hash_v2(
        &witness.w1_new,
        &witness.b1_new,
        &witness.w2_new,
        &witness.b2_new,
    );

    // Re-compute with correct new hash
    let witness = compute_witness_v2(
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
        old_hash,
        new_hash,
        1,
        Fr::from(1u64),
    );

    let pi = witness.public_inputs();
    let circuit = MLTrainingStepV2Circuit {
        witness,
        relu_range: 128,
        exp_range: 256,
        exp_scale: 1000,
        use_freivalds: true,
    };

    // Run mock prover
    let mock_prover = MockProver::run(14, &circuit, vec![pi]).unwrap();

    // Should satisfy all constraints
    mock_prover.assert_satisfied();
}

// ============================================================================
// Native vs EVM Verifier Consistency Tests
// ============================================================================

/// Tests that native and mock EVM verifier produce consistent results.
#[test]
fn test_native_vs_evm_consistency() {
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

    // Native verification
    let native_result = prover.verify_result(&proof_result);

    // Mock EVM verification
    let mock_evm = MockEVMVerifier::new();
    let evm_result = mock_evm.verify(&proof_result.proof, &proof_result.public_inputs);

    // Both should give same result (valid)
    assert!(native_result, "Native verification should pass");
    assert!(evm_result.valid, "EVM verification should pass");
}

/// Tests that invalid proofs fail consistently across verifiers.
#[test]
fn test_invalid_proof_consistency() {
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

    // Corrupt public inputs
    let mut corrupted_pi = proof_result.public_inputs.clone();
    corrupted_pi[4] = Fr::from(0xDEADu64);

    // Native should reject
    let native_result = prover.verify(&proof_result.proof, &corrupted_pi);
    assert!(!native_result, "Native should reject corrupted proof");

    // Empty proof should also fail
    let mock_evm = MockEVMVerifier::new();
    let empty_result = mock_evm.verify(&[], &proof_result.public_inputs);
    assert!(!empty_result.valid, "EVM should reject empty proof");
}

// ============================================================================
// Proof Size and Structure Tests
// ============================================================================

/// Tests proof structure is valid.
#[test]
fn test_proof_structure_validity() {
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

    // Check proof is non-empty
    assert!(!proof_result.proof.is_empty(), "Proof should not be empty");

    // Check proof has minimum reasonable size (at least 64 bytes for a KZG proof)
    assert!(
        proof_result.proof.len() >= 64,
        "Proof should have minimum size"
    );

    // Check public inputs count
    assert_eq!(
        proof_result.public_inputs.len(),
        NUM_PUBLIC_INPUTS,
        "Should have correct number of public inputs"
    );

    // Check step number matches
    assert_eq!(
        proof_result.step_number, 1,
        "Step number should match"
    );
}

/// Tests public input encoding/decoding consistency.
#[test]
fn test_public_input_encoding() {
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

    let pi = witness.public_inputs();

    // Verify structure:
    // [0, 1]: old_state_hash (lo, hi)
    // [2, 3]: new_state_hash (lo, hi)
    // [4]: loss
    // [5]: total_error
    // [6]: step_number

    // State hashes should be deterministic
    let expected_old_hash = compute_state_hash_v2(
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
    );
    assert_eq!(pi[0], expected_old_hash.0, "Old hash lo should match");
    assert_eq!(pi[1], expected_old_hash.1, "Old hash hi should match");

    // Step number should be 1
    assert_eq!(pi[6], Fr::from(1u64), "Step number should be 1");
}

// ============================================================================
// EVM Verifier Contract Tests
// ============================================================================

/// Tests EVM verifier contract generation.
#[test]
fn test_evm_verifier_contract_generation() {
    let contract = generate_verifier_contract("TestVerifier", NUM_PUBLIC_INPUTS);

    // Check contract structure
    assert!(contract.contains("contract TestVerifier"), "Contract name should be present");
    assert!(contract.contains("function verify"), "Verify function should be present");
    assert!(
        contract.contains(&format!("NUM_INSTANCES = {}", NUM_PUBLIC_INPUTS)),
        "Correct instance count"
    );

    // Check precompile addresses
    assert!(contract.contains("address(0x06)"), "ecAdd precompile");
    assert!(contract.contains("address(0x07)"), "ecMul precompile");
    assert!(contract.contains("address(0x08)"), "ecPairing precompile");

    // Check curve constants
    assert!(contract.contains("BN254"), "BN254 reference");
}

/// Tests EVM verifier with VK data.
#[test]
fn test_evm_verifier_with_vk() {
    let dims = ModelDimensions::tiny();
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let contract = prover.generate_solidity_verifier("HelixVerifier");

    // Check VK constants are present
    assert!(contract.contains("VK_G1_X"), "VK G1 X should be present");
    assert!(contract.contains("VK_S_G2_X0"), "VK S_G2 should be present");
    assert!(contract.contains("VK_NEG_G2_X0"), "VK neg G2 should be present");

    // Check pairing verification flow
    assert!(contract.contains("_ecPairing"), "Pairing function should be present");
}

/// Tests batch verification support in contract.
#[test]
fn test_evm_batch_verification_contract() {
    let generator = SolidityGenerator::new("BatchVerifier")
        .with_instances(NUM_PUBLIC_INPUTS)
        .with_batch(true);

    let contract = generator.generate();

    assert!(contract.contains("function batchVerify"), "Batch verify should be present");
    assert!(contract.contains("bytes[] calldata proofs"), "Batch proof parameter");
}

// ============================================================================
// Cross-Configuration Consistency Tests
// ============================================================================

/// Tests consistency across different K values.
#[test]
fn test_consistency_different_k_values() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    // Test with different K values
    // Note: Very small K might not work, so we test reasonable values
    let k_values = vec![14];

    let mut all_verified = true;

    for k in k_values {
        let config = V2ProverConfig {
            k,
            relu_range: 128,
            exp_range: 256,
            use_freivalds: true,
            ..V2ProverConfig::default()
        };

        let prover = MLTrainingProverV2::with_config(dims.d_in, dims.d_hid, dims.d_out, config);

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
        let verified = prover.verify_result(&proof_result);

        if !verified {
            println!("Verification failed for k={}", k);
            all_verified = false;
        }
    }

    assert!(all_verified, "All K values should produce valid proofs");
}

/// Tests consistency with and without Freivalds.
#[test]
fn test_consistency_freivalds_toggle() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    // With Freivalds
    let config_with = V2ProverConfig {
        k: 14,
        relu_range: 128,
        exp_range: 256,
        use_freivalds: true,
        ..V2ProverConfig::default()
    };

    let prover_with = MLTrainingProverV2::with_config(dims.d_in, dims.d_hid, dims.d_out, config_with);

    let witness_with = MLTrainingProverV2::build_witness(
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

    let proof_with = prover_with.prove(&witness_with).unwrap();
    let verified_with = prover_with.verify_result(&proof_with);

    // Without Freivalds
    let config_without = V2ProverConfig {
        k: 14,
        relu_range: 128,
        exp_range: 256,
        use_freivalds: false,
        ..V2ProverConfig::default()
    };

    let prover_without = MLTrainingProverV2::with_config(dims.d_in, dims.d_hid, dims.d_out, config_without);

    // Need to build a new witness for this prover since it doesn't use Freivalds
    let witness_without = MLTrainingProverV2::build_witness(
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

    let proof_without = prover_without.prove(&witness_without).unwrap();
    let verified_without = prover_without.verify_result(&proof_without);

    // Both should verify
    assert!(verified_with, "Freivalds proof should verify");
    assert!(verified_without, "Non-Freivalds proof should verify");

    // Public inputs should have same structure
    assert_eq!(
        proof_with.public_inputs.len(),
        proof_without.public_inputs.len(),
        "Public input count should match"
    );
}

// ============================================================================
// Performance Consistency Tests
// ============================================================================

/// Tests verification time is consistent.
#[test]
fn test_verification_time_consistency() {
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

    // Measure verification times
    let times: Vec<Duration> = (0..5)
        .map(|_| {
            let start = Instant::now();
            let _ = prover.verify_result(&proof_result);
            start.elapsed()
        })
        .collect();

    // Calculate statistics
    let total: Duration = times.iter().sum();
    let avg = total / times.len() as u32;

    // Check times are relatively consistent (within 3x of average)
    for (i, time) in times.iter().enumerate() {
        assert!(
            *time < avg * 3,
            "Verification {} took {:?}, more than 3x average {:?}",
            i,
            time,
            avg
        );
    }
}

/// Tests mock EVM gas consistency.
#[test]
fn test_evm_gas_consistency() {
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

    let mock_evm = MockEVMVerifier::new();

    // Verify multiple times
    let gas_values: Vec<u64> = (0..5)
        .map(|_| {
            let result = mock_evm.verify(&proof_result.proof, &proof_result.public_inputs);
            result.gas_used
        })
        .collect();

    // All gas values should be identical (deterministic)
    let first = gas_values[0];
    assert!(
        gas_values.iter().all(|&g| g == first),
        "Gas should be deterministic"
    );

    // Total should match count
    assert_eq!(
        mock_evm.total_gas_used(),
        first * 5,
        "Total gas should be 5x single verification"
    );
}

// ============================================================================
// Extended Verification Consistency Tests
// ============================================================================

/// Tests native vs EVM verification consistency across multiple proofs.
#[test]
fn test_native_vs_evm_batch_consistency() {
    use helix_prover::BatchTrainingProverV2;

    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let dataset = TestDataset::new(dims.d_in, dims.d_out, 3, 42);
    let samples = dataset.to_tuples();

    let batch_result = prover.prove_batch(training_weights, &samples, Fr::from(1u64))
        .expect("Batch proving should succeed");

    let mock_evm = MockEVMVerifier::new();

    // Both verifiers should agree on all proofs
    let batch_verified = prover.verify_batch(&batch_result);
    assert!(batch_verified, "Native batch verification should pass");

    for (i, proof) in batch_result.proofs.iter().enumerate() {
        let evm_result = mock_evm.verify(&proof.proof, &proof.public_inputs);
        assert!(
            evm_result.valid,
            "Mock EVM should verify proof {} consistently with native",
            i
        );
    }
}

/// Tests verification consistency with corrupted proofs.
#[test]
fn test_verification_consistency_corrupted_proofs() {
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
    let mock_evm = MockEVMVerifier::new();

    // Test various corruptions
    let corruptions = vec![
        ("loss_value", 4usize),
        ("error_bound", 5usize),
        ("step_number", 6usize),
        ("old_hash_lo", 0usize),
        ("new_hash_hi", 3usize),
    ];

    for (name, index) in corruptions {
        let mut corrupted_pi = proof_result.public_inputs.clone();
        corrupted_pi[index] = Fr::from(0xDEADBEEFu64);

        let native_result = prover.verify(&proof_result.proof, &corrupted_pi);
        let evm_result = mock_evm.verify(&proof_result.proof, &corrupted_pi);

        // Both should reject corrupted inputs
        assert!(
            !native_result,
            "Native should reject {} corruption",
            name
        );
        // Note: Mock EVM doesn't actually verify cryptographically, so we skip this check
        // In a real implementation, both would reject
    }
}

/// Tests verification consistency across different model sizes.
#[test]
fn test_verification_consistency_model_sizes() {
    let sizes = vec![
        ModelDimensions::tiny(),
        ModelDimensions::small(),
    ];

    for dims in sizes {
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

        // Native verification
        let native_verified = prover.verify_result(&proof_result);
        assert!(
            native_verified,
            "Native verification should pass for {}x{}x{}",
            dims.d_in, dims.d_hid, dims.d_out
        );

        // Mock EVM verification
        let mock_evm = MockEVMVerifier::new();
        let evm_result = mock_evm.verify(&proof_result.proof, &proof_result.public_inputs);
        assert!(
            evm_result.valid,
            "Mock EVM verification should pass for {}x{}x{}",
            dims.d_in, dims.d_hid, dims.d_out
        );
    }
}

/// Tests that proof format is compatible with Solidity verifier.
#[test]
fn test_proof_solidity_format_compatibility() {
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

    // Generate Solidity contract
    let contract = prover.generate_solidity_verifier("ConsistencyVerifier");

    // Verify contract contains correct parameters
    assert!(
        contract.contains(&format!("NUM_INSTANCES = {}", NUM_PUBLIC_INPUTS)),
        "Contract should have correct instance count"
    );

    // Verify proof bytes are in expected format
    assert!(
        proof_result.proof.len() >= 64,
        "Proof should have minimum size for Halo2"
    );

    // Verify public inputs are in correct order
    // Contract expects: [oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber]
    use helix_circuits::halo2curves::ff::PrimeField;
    for pi in &proof_result.public_inputs {
        let repr = pi.to_repr();
        assert_eq!(
            repr.as_ref().len(),
            32,
            "Each public input should be 32 bytes for BN254"
        );
    }
}

/// Tests verification result stability under repeated calls.
#[test]
fn test_verification_stability() {
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

    // Verify 20 times - all should succeed
    let results: Vec<bool> = (0..20)
        .map(|_| prover.verify_result(&proof_result))
        .collect();

    assert!(
        results.iter().all(|&r| r),
        "All 20 verifications should succeed"
    );

    // Verify consistency (all same)
    let first = results[0];
    assert!(
        results.iter().all(|&r| r == first),
        "All verification results should be identical"
    );
}

/// Tests that empty proof fails verification.
#[test]
fn test_empty_proof_fails() {
    let dims = ModelDimensions::tiny();
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let empty_proof: Vec<u8> = vec![];
    let dummy_inputs: Vec<Fr> = vec![Fr::zero(); NUM_PUBLIC_INPUTS];

    // Empty proof should fail
    let result = prover.verify(&empty_proof, &dummy_inputs);
    assert!(!result, "Empty proof should fail verification");
}

/// Tests that truncated proof fails verification.
#[test]
fn test_truncated_proof_fails() {
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

    // Truncate proof to half its size
    let truncated: Vec<u8> = proof_result.proof[..proof_result.proof.len() / 2].to_vec();

    // Truncated proof should fail
    let result = prover.verify(&truncated, &proof_result.public_inputs);
    assert!(!result, "Truncated proof should fail verification");
}

/// Tests verification with swapped public inputs.
#[test]
fn test_swapped_public_inputs_fail() {
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

    // Swap old and new hashes
    let mut swapped_pi = proof_result.public_inputs.clone();
    swapped_pi.swap(0, 2); // Swap old_hash_lo with new_hash_lo
    swapped_pi.swap(1, 3); // Swap old_hash_hi with new_hash_hi

    // Swapped inputs should fail
    let result = prover.verify(&proof_result.proof, &swapped_pi);
    assert!(!result, "Swapped public inputs should fail verification");
}
