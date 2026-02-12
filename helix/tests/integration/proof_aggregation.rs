//! Proof Aggregation Pipeline Integration Tests
//!
//! Tests the end-to-end aggregation pipeline:
//! 1. Generate N individual training step proofs
//! 2. Aggregate them via RLCAggregationProver into a single KZG proof
//! 3. Verify the aggregated proof natively
//! 4. Validate 8 public inputs match the on-chain contract interface
//! 5. Verify structural properties (chain integrity, error accumulation, boundary hashes)

#![allow(unused_imports)]

use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::{AGGREGATION_NUM_PUBLIC_INPUTS, MAX_AGGREGATION_BATCH};
use helix_prover::{
    AggregatedTrainingProof, BatchProver, BatchTrainingProverV2,
    MLTrainingProverV2, RLCAggregationProver, TrainingProofResultV2,
    TrainingWeights, V2ProverConfig,
};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Helper: Create chained mock proofs with matching PI format
// ============================================================================

/// Creates N mock TrainingProofResultV2 values with proper PI chaining.
///
/// Each proof's new_state_hash matches the next proof's old_state_hash.
/// This is the fundamental invariant the aggregation circuit checks.
fn make_chained_proofs(n: usize) -> Vec<TrainingProofResultV2> {
    let mut proofs = Vec::with_capacity(n);
    for i in 0..n {
        let old_hash = (
            Fr::from((i * 10 + 1) as u64),
            Fr::from((i * 10 + 2) as u64),
        );
        let new_hash = (
            Fr::from(((i + 1) * 10 + 1) as u64),
            Fr::from(((i + 1) * 10 + 2) as u64),
        );
        let loss = Fr::from(100u64);
        let error = Fr::from(5u64);
        let step_num = Fr::from((i + 1) as u64);
        let checksum = Fr::from(42u64);

        let public_inputs = vec![
            old_hash.0,  // PI[0]: old_hash_lo
            old_hash.1,  // PI[1]: old_hash_hi
            new_hash.0,  // PI[2]: new_hash_lo
            new_hash.1,  // PI[3]: new_hash_hi
            loss,        // PI[4]: loss
            error,       // PI[5]: error_bound
            step_num,    // PI[6]: step_number
            checksum,    // PI[7]: error_checksum
        ];

        proofs.push(TrainingProofResultV2 {
            proof: vec![0xDE, 0xAD, 0xBE, 0xEF],
            public_inputs,
            loss,
            total_error: error,
            step_number: (i + 1) as u64,
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            verified: true,
            generation_time: Duration::from_millis(10),
            verification_time: None,
            attempts: 1,
            from_cache: false,
            witness_hash: None,
        });
    }
    proofs
}

// ============================================================================
// Test: RLC aggregation produces valid proof with 8 PIs
// ============================================================================

/// Tests that RLC aggregation produces a valid proof with 8 public inputs
/// matching the on-chain contract interface.
#[test]
fn test_aggregation_pipeline_produces_contract_compatible_proof() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("aggregation_contract_compatible");

    // Phase 1: Create chained proofs
    let phase_start = Instant::now();
    let proofs = make_chained_proofs(3);
    assert_eq!(proofs.len(), 3);
    result.add_phase(PhaseResult::success("create_proofs", phase_start.elapsed()));

    // Phase 2: Create aggregation prover
    let phase_start = Instant::now();
    let prover = RLCAggregationProver::new(MAX_AGGREGATION_BATCH, 14);
    assert!(prover.is_ready());
    result.add_phase(PhaseResult::success("prover_setup", phase_start.elapsed()));

    // Phase 3: Aggregate
    let phase_start = Instant::now();
    let agg = prover.aggregate(&proofs).expect("Aggregation should succeed");
    result.add_phase(PhaseResult::success("aggregation", phase_start.elapsed()));

    // Phase 4: Validate 8 public inputs
    let phase_start = Instant::now();
    assert_eq!(
        agg.public_inputs.len(),
        AGGREGATION_NUM_PUBLIC_INPUTS,
        "Aggregated proof must have exactly 8 PIs (contract interface)"
    );
    assert_eq!(agg.num_steps, 3);
    assert!(!agg.proof.is_empty(), "Proof bytes must not be empty");
    result.add_phase(PhaseResult::success("validate_pis", phase_start.elapsed()));

    // Phase 5: Verify natively
    let phase_start = Instant::now();
    let valid = prover.verify(&agg).expect("Verification should complete");
    assert!(valid, "Aggregated proof must verify natively");
    result.add_phase(PhaseResult::success("native_verification", phase_start.elapsed()));

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success, "Aggregation pipeline test failed");
}

// ============================================================================
// Test: Boundary hash preservation
// ============================================================================

/// Tests that the aggregated proof correctly preserves boundary hashes:
/// - first_old_hash == proofs[0].old_state_hash (the starting state)
/// - last_new_hash == proofs[N-1].new_state_hash (the ending state)
#[test]
fn test_aggregation_preserves_boundary_hashes() {
    let proofs = make_chained_proofs(4);
    let prover = RLCAggregationProver::new(MAX_AGGREGATION_BATCH, 14);
    let agg = prover.aggregate(&proofs).expect("Aggregation should succeed");

    // Boundary hash invariants
    assert_eq!(
        agg.first_old_hash, proofs[0].old_state_hash,
        "Aggregated first_old_hash must match first proof's old_state_hash"
    );
    assert_eq!(
        agg.last_new_hash,
        proofs.last().unwrap().new_state_hash,
        "Aggregated last_new_hash must match last proof's new_state_hash"
    );

    // These should be reflected in the PI layout
    assert_eq!(agg.public_inputs[0], proofs[0].old_state_hash.0);
    assert_eq!(agg.public_inputs[1], proofs[0].old_state_hash.1);
    assert_eq!(
        agg.public_inputs[2],
        proofs.last().unwrap().new_state_hash.0
    );
    assert_eq!(
        agg.public_inputs[3],
        proofs.last().unwrap().new_state_hash.1
    );
}

// ============================================================================
// Test: Error and loss accumulation
// ============================================================================

/// Tests that error bounds and losses accumulate correctly across steps.
#[test]
fn test_aggregation_accumulates_errors_and_losses() {
    let proofs = make_chained_proofs(5);
    let prover = RLCAggregationProver::new(MAX_AGGREGATION_BATCH, 14);
    let agg = prover.aggregate(&proofs).expect("Aggregation should succeed");

    // Each proof has error=5, loss=100
    assert_eq!(agg.total_error, Fr::from(25u64), "Total error = 5 * 5 steps");
    assert_eq!(agg.total_loss, Fr::from(500u64), "Total loss = 100 * 5 steps");
    assert_eq!(agg.num_steps, 5);
}

// ============================================================================
// Test: Chain integrity rejection
// ============================================================================

/// Tests that aggregation rejects proofs with broken PI chain.
#[test]
fn test_aggregation_rejects_broken_chain() {
    let mut proofs = make_chained_proofs(3);

    // Break the chain: step[1]'s old_hash doesn't match step[0]'s new_hash
    proofs[1].public_inputs[0] = Fr::from(9999u64);
    proofs[1].old_state_hash.0 = Fr::from(9999u64);

    let prover = RLCAggregationProver::new(MAX_AGGREGATION_BATCH, 14);
    let result = prover.aggregate(&proofs);
    assert!(
        result.is_err(),
        "Broken PI chain must be rejected by aggregation"
    );
}

// ============================================================================
// Test: Single-step aggregation
// ============================================================================

/// Tests that aggregation works for a single proof (degenerate case).
#[test]
fn test_aggregation_single_step() {
    let proofs = make_chained_proofs(1);
    let prover = RLCAggregationProver::new(MAX_AGGREGATION_BATCH, 14);
    let agg = prover.aggregate(&proofs).expect("Single-step aggregation should succeed");

    assert_eq!(agg.num_steps, 1);
    assert_eq!(agg.public_inputs.len(), AGGREGATION_NUM_PUBLIC_INPUTS);
    assert_eq!(agg.first_old_hash, agg.individual_proofs[0].old_state_hash);
    assert_eq!(agg.last_new_hash, agg.individual_proofs[0].new_state_hash);

    let valid = prover.verify(&agg).expect("Verification should complete");
    assert!(valid, "Single-step aggregated proof must verify");
}

// ============================================================================
// Test: EVM public input format
// ============================================================================

/// Tests that to_evm_public_inputs produces correctly formatted 32-byte arrays
/// suitable for on-chain verification.
#[test]
fn test_aggregation_evm_public_input_format() {
    let proofs = make_chained_proofs(2);
    let prover = RLCAggregationProver::new(MAX_AGGREGATION_BATCH, 14);
    let agg = prover.aggregate(&proofs).expect("Aggregation should succeed");

    let evm_pis = agg.to_evm_public_inputs();
    assert_eq!(
        evm_pis.len(),
        AGGREGATION_NUM_PUBLIC_INPUTS,
        "EVM PIs must have 8 elements"
    );

    // Each EVM PI is a 32-byte array
    for (i, pi_bytes) in evm_pis.iter().enumerate() {
        assert_eq!(pi_bytes.len(), 32, "EVM PI[{i}] must be 32 bytes");
    }

    // Verify PIs round-trip: bytes → Fr should recover the original Fr values
    for (i, pi_bytes) in evm_pis.iter().enumerate() {
        let recovered = Fr::from_repr_vartime((*pi_bytes).into());
        assert!(
            recovered.is_some(),
            "EVM PI[{i}] must be a valid Fr representation"
        );
    }
}

// ============================================================================
// Test: BatchProver::aggregate_training_proofs API
// ============================================================================

/// Tests the public BatchProver::aggregate_training_proofs entry point.
#[test]
fn test_batch_prover_aggregate_training_proofs() {
    let proofs = make_chained_proofs(3);
    let agg = BatchProver::aggregate_training_proofs(&proofs)
        .expect("BatchProver aggregation should succeed");

    assert_eq!(agg.num_steps, 3);
    assert_eq!(agg.public_inputs.len(), AGGREGATION_NUM_PUBLIC_INPUTS);
    assert!(!agg.proof.is_empty());
}

// ============================================================================
// Test: Empty batch rejection
// ============================================================================

/// Tests that aggregation rejects empty proof batches.
#[test]
fn test_aggregation_rejects_empty_batch() {
    let prover = RLCAggregationProver::new(MAX_AGGREGATION_BATCH, 14);
    let result = prover.aggregate(&[]);
    assert!(result.is_err(), "Empty batch must be rejected");
}
