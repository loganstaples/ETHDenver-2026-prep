//! Cross-Platform Compatibility Tests.
//!
//! Tests that HELIX components work correctly across different:
//! - System architectures (endianness, word size)
//! - Serialization formats
//! - Numeric representations
//! - Data encoding schemes

#![allow(unused_imports)]

use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::ml::training_step_v2::compute_state_hash_v2;
use helix_prover::{MLTrainingProverV2, TrainingWeights};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Endianness Tests
// ============================================================================

/// Tests that field element serialization is endian-consistent.
#[test]
fn test_fr_serialization_endianness() {
    let value = Fr::from(0x0102030405060708u64);

    // Serialize to bytes
    let repr = value.to_repr();
    let bytes = repr.as_ref();

    // Verify little-endian encoding (standard for Fr)
    // The low bytes should contain the value
    assert_eq!(bytes[0], 0x08, "Lowest byte should be 0x08");
    assert_eq!(bytes[1], 0x07, "Second byte should be 0x07");

    // Deserialize and verify roundtrip
    let recovered = Fr::from_repr_vartime(repr).unwrap();
    assert_eq!(value, recovered, "Roundtrip should preserve value");
}

/// Tests that hash computation is platform-independent.
#[test]
fn test_hash_computation_deterministic() {
    let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)];
    let b1 = vec![Fr::zero(), Fr::zero()];
    let w2 = vec![Fr::from(5u64), Fr::from(6u64)];
    let b2 = vec![Fr::zero()];

    // Compute hash multiple times
    let hashes: Vec<_> = (0..5)
        .map(|_| compute_state_hash_v2(&w1, &b1, &w2, &b2))
        .collect();

    // All hashes should be identical
    for (i, hash) in hashes.iter().enumerate().skip(1) {
        assert_eq!(hashes[0], *hash, "Hash {} should match hash 0", i);
    }

    // Hash should be non-trivial
    assert!(
        hashes[0].0 != Fr::zero() || hashes[0].1 != Fr::zero(),
        "Hash should be non-trivial"
    );
}

// ============================================================================
// Numeric Representation Tests
// ============================================================================

/// Tests that small integers are correctly represented in field elements.
#[test]
fn test_small_integer_representation() {
    for i in 0u64..100 {
        let fr = Fr::from(i);

        // Converting back should give same value
        let repr = fr.to_repr();
        let bytes = repr.as_ref();

        // First 8 bytes should contain the value in little-endian
        let recovered = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);

        assert_eq!(i, recovered, "Integer {} should roundtrip correctly", i);
    }
}

/// Tests that large integers near field modulus work correctly.
#[test]
fn test_large_integer_representation() {
    // Test some large values
    let large_values = [
        u64::MAX,
        u64::MAX - 1,
        0xFFFFFFFF_00000000u64,
        0x00000000_FFFFFFFFu64,
    ];

    for val in large_values {
        let fr = Fr::from(val);

        // Should be able to serialize and deserialize
        let repr = fr.to_repr();
        let recovered = Fr::from_repr_vartime(repr);

        assert!(recovered.is_some(), "Should deserialize large value {}", val);
        assert_eq!(fr, recovered.unwrap(), "Large value {} should roundtrip", val);
    }
}

/// Tests field element arithmetic consistency.
#[test]
fn test_field_arithmetic_consistency() {
    let a = Fr::from(1000u64);
    let b = Fr::from(2000u64);

    // Addition
    let sum = a + b;
    let expected_sum = Fr::from(3000u64);
    assert_eq!(sum, expected_sum, "Addition should be consistent");

    // Multiplication
    let prod = a * b;
    let expected_prod = Fr::from(2_000_000u64);
    assert_eq!(prod, expected_prod, "Multiplication should be consistent");

    // Subtraction
    let diff = b - a;
    let expected_diff = Fr::from(1000u64);
    assert_eq!(diff, expected_diff, "Subtraction should be consistent");

    // Division (multiply by inverse)
    let b_inv = b.invert().unwrap();
    let quot = a * b_inv;
    // a/b should equal a * (1/b)
    // Can't easily verify the value, but it should be consistent
    let check = quot * b;
    assert_eq!(check, a, "Division should be consistent");
}

// ============================================================================
// Proof Serialization Tests
// ============================================================================

/// Tests that proof bytes are platform-independent.
#[test]
fn test_proof_serialization_platform_independent() {
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

    // Proof should be non-empty
    assert!(!proof_result.proof.is_empty());

    // Proof should verify with original bytes
    assert!(prover.verify_result(&proof_result));

    // Clone proof bytes and verify they still work
    let cloned_proof = proof_result.proof.clone();
    assert!(prover.verify(&cloned_proof, &proof_result.public_inputs));
}

/// Tests public input encoding consistency.
#[test]
fn test_public_input_encoding() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    // Build witness twice with same inputs
    let witness1 = MLTrainingProverV2::build_witness(
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

    let witness2 = MLTrainingProverV2::build_witness(
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

    // Public inputs should be identical
    let pi1 = witness1.public_inputs();
    let pi2 = witness2.public_inputs();

    assert_eq!(pi1.len(), pi2.len(), "Public input count should match");
    for (i, (p1, p2)) in pi1.iter().zip(pi2.iter()).enumerate() {
        assert_eq!(p1, p2, "Public input {} should be identical", i);
    }
}

// ============================================================================
// Data Format Compatibility Tests
// ============================================================================

/// Tests JSON serialization compatibility.
#[test]
fn test_json_compatibility() {
    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct TestData {
        name: String,
        values: Vec<u64>,
        nested: NestedData,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct NestedData {
        flag: bool,
        count: i32,
    }

    let data = TestData {
        name: "test".to_string(),
        values: vec![1, 2, 3, 4, 5],
        nested: NestedData { flag: true, count: 42 },
    };

    // Serialize
    let json = serde_json::to_string(&data).expect("Should serialize");

    // Deserialize
    let recovered: TestData = serde_json::from_str(&json).expect("Should deserialize");

    assert_eq!(data, recovered, "JSON roundtrip should preserve data");
}

/// Tests bincode serialization compatibility.
#[test]
fn test_bincode_compatibility() {
    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct BinaryData {
        id: u64,
        bytes: Vec<u8>,
        flag: bool,
    }

    let data = BinaryData {
        id: 12345,
        bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
        flag: true,
    };

    // Serialize
    let binary = bincode::serialize(&data).expect("Should serialize");

    // Deserialize
    let recovered: BinaryData = bincode::deserialize(&binary).expect("Should deserialize");

    assert_eq!(data, recovered, "Bincode roundtrip should preserve data");
}

// ============================================================================
// Circuit Determinism Tests
// ============================================================================

/// Tests that circuit computation is deterministic.
#[test]
fn test_circuit_determinism() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let sample = TestSample::known(dims.d_in, dims.d_out);

    // Generate proofs multiple times with same input
    let results: Vec<_> = (0..3)
        .map(|_| {
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
            prover.prove(&witness).unwrap()
        })
        .collect();

    // All should have same public inputs
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            results[0].public_inputs, result.public_inputs,
            "Public inputs {} should match",
            i
        );
        assert_eq!(results[0].loss, result.loss, "Loss {} should match", i);
        assert_eq!(
            results[0].old_state_hash, result.old_state_hash,
            "Old hash {} should match",
            i
        );
        assert_eq!(
            results[0].new_state_hash, result.new_state_hash,
            "New hash {} should match",
            i
        );
    }

    // All proofs should verify
    for result in &results {
        assert!(prover.verify_result(result), "Proof should verify");
    }
}

// ============================================================================
// Memory Alignment Tests
// ============================================================================

/// Tests that data structures have expected sizes.
#[test]
fn test_data_structure_sizes() {
    // Fr should be 32 bytes
    assert_eq!(
        std::mem::size_of::<Fr>(),
        32,
        "Fr should be 32 bytes"
    );

    // Vec should be 24 bytes (pointer, length, capacity)
    assert_eq!(
        std::mem::size_of::<Vec<u8>>(),
        24,
        "Vec<u8> should be 24 bytes on 64-bit"
    );
}

/// Tests that Fr arrays can be safely transmitted.
#[test]
fn test_fr_array_transmission() {
    let original: Vec<Fr> = (0..10).map(|i| Fr::from(i as u64)).collect();

    // Convert to bytes
    let bytes: Vec<u8> = original
        .iter()
        .flat_map(|fr| fr.to_repr().as_ref().to_vec())
        .collect();

    // Convert back
    let recovered: Vec<Fr> = bytes
        .chunks(32)
        .map(|chunk| {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(chunk);
            Fr::from_repr_vartime(arr.into()).unwrap()
        })
        .collect();

    assert_eq!(original.len(), recovered.len());
    for (orig, rec) in original.iter().zip(recovered.iter()) {
        assert_eq!(orig, rec, "Fr element should survive transmission");
    }
}

// ============================================================================
// RNG Consistency Tests
// ============================================================================

/// Tests that DeterministicRng produces consistent results.
#[test]
fn test_deterministic_rng_consistency() {
    let seed = 42u64;

    // Generate sequence twice with same seed
    let mut rng1 = DeterministicRng::new(seed);
    let mut rng2 = DeterministicRng::new(seed);

    for i in 0..100 {
        let val1 = rng1.next_fr();
        let val2 = rng2.next_fr();
        assert_eq!(val1, val2, "RNG value {} should be deterministic", i);
    }
}

/// Tests that different seeds produce different sequences.
#[test]
fn test_deterministic_rng_different_seeds() {
    let mut rng1 = DeterministicRng::new(42);
    let mut rng2 = DeterministicRng::new(43);

    let seq1: Vec<Fr> = (0..10).map(|_| rng1.next_fr()).collect();
    let seq2: Vec<Fr> = (0..10).map(|_| rng2.next_fr()).collect();

    // Sequences should be different
    let any_different = seq1.iter().zip(seq2.iter()).any(|(a, b)| a != b);
    assert!(any_different, "Different seeds should produce different sequences");
}

// ============================================================================
// Comprehensive Compatibility Test
// ============================================================================

/// Comprehensive cross-platform compatibility test.
#[test]
fn test_comprehensive_compatibility() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("comprehensive_compatibility");

    // Phase 1: Field element operations
    let phase_start = Instant::now();
    let a = Fr::from(12345u64);
    let b = Fr::from(67890u64);
    let c = a + b;
    let d = a * b;
    let e = b - a;
    let _ = (c, d, e); // Use values
    result.add_phase(PhaseResult::success("field_arithmetic", phase_start.elapsed()));

    // Phase 2: Serialization
    let phase_start = Instant::now();
    let repr = a.to_repr();
    let recovered = Fr::from_repr_vartime(repr);
    result.add_phase(if recovered == Some(a) {
        PhaseResult::success("fr_serialization", phase_start.elapsed())
    } else {
        PhaseResult::failure("fr_serialization", phase_start.elapsed(), "Roundtrip failed")
    });

    // Phase 3: Hash computation
    let phase_start = Instant::now();
    let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)];
    let b1 = vec![Fr::zero(), Fr::zero()];
    let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
    let b2 = vec![Fr::zero()];
    let hash1 = compute_state_hash_v2(&w1, &b1, &w2, &b2);
    let hash2 = compute_state_hash_v2(&w1, &b1, &w2, &b2);
    result.add_phase(if hash1 == hash2 {
        PhaseResult::success("hash_determinism", phase_start.elapsed())
    } else {
        PhaseResult::failure("hash_determinism", phase_start.elapsed(), "Hash not deterministic")
    });

    // Phase 4: Proof generation and verification
    let phase_start = Instant::now();
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

    let proof = prover.prove(&witness).unwrap();
    let verified = prover.verify_result(&proof);
    result.add_phase(if verified {
        PhaseResult::success("proof_roundtrip", phase_start.elapsed())
    } else {
        PhaseResult::failure("proof_roundtrip", phase_start.elapsed(), "Verification failed")
    });

    // Phase 5: RNG determinism
    let phase_start = Instant::now();
    let mut rng1 = DeterministicRng::new(42);
    let mut rng2 = DeterministicRng::new(42);
    let rng_match = (0..100).all(|_| rng1.next_fr() == rng2.next_fr());
    result.add_phase(if rng_match {
        PhaseResult::success("rng_determinism", phase_start.elapsed())
    } else {
        PhaseResult::failure("rng_determinism", phase_start.elapsed(), "RNG not deterministic")
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success);
}
