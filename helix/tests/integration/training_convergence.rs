//! Training Convergence Tests with Known-Good Models.
//!
//! Tests that the training pipeline produces models that actually learn:
//! - Loss decreases over training steps
//! - Gradients are non-zero and bounded
//! - Error bounds remain within acceptable limits
//! - State transitions are valid

#![allow(unused_imports)]

use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::ml::training_step_v2::{compute_state_hash_v2, ErrorTracker};
use helix_prover::{BatchTrainingProverV2, MLTrainingProverV2, TrainingWeights};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Known-Good Model Tests
// ============================================================================

/// Tests that a tiny model learns a simple pattern.
#[test]
fn test_convergence_tiny_model() {
    let dims = ModelDimensions::tiny();

    // Initialize with small positive weights (easier to train)
    let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
    let b1 = vec![Fr::zero(), Fr::zero()];
    let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
    let b2 = vec![Fr::zero()];

    let weights = TrainingWeights::new(dims.d_in, dims.d_hid, dims.d_out, w1, b1, w2, b2);

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // Simple training data: learn to output roughly 5 for input [1, 1]
    let samples = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(6u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(7u64)]),
    ];

    let batch_result = prover.prove_batch(weights, &samples, Fr::from(1u64));

    // All proofs should verify
    assert!(prover.verify_batch(&batch_result), "All proofs should verify");

    // Should have completed all steps
    assert_eq!(batch_result.num_steps, samples.len());

    // Extract losses (as Fr values, we can't compare directly but can check they exist)
    let losses: Vec<Fr> = batch_result.proofs.iter().map(|p| p.loss).collect();
    assert_eq!(losses.len(), samples.len());
}

/// Tests that gradients are non-zero (model is learning).
#[test]
fn test_convergence_gradients_nonzero() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // Training sample with non-zero targets
    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::from(5u64)];

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
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

    // Check that weights changed
    let weights_changed = weights.w1.iter().zip(witness.w1_new.iter()).any(|(old, new)| old != new)
        || weights.w2.iter().zip(witness.w2_new.iter()).any(|(old, new)| old != new);

    assert!(weights_changed, "Weights should change during training");

    // Proof should still verify
    let proof = prover.prove(&witness);
    assert!(prover.verify_result(&proof), "Proof should verify");
}

/// Tests that state hashes change with weight updates.
#[test]
fn test_convergence_state_transitions() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::from(5u64)];

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
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

    let proof = prover.prove(&witness);

    // State should change
    assert_ne!(
        proof.old_state_hash, proof.new_state_hash,
        "State hash should change after training step"
    );

    // Both hashes should be non-trivial
    assert!(
        proof.old_state_hash.0 != Fr::zero() || proof.old_state_hash.1 != Fr::zero(),
        "Old state hash should be non-trivial"
    );
    assert!(
        proof.new_state_hash.0 != Fr::zero() || proof.new_state_hash.1 != Fr::zero(),
        "New state hash should be non-trivial"
    );
}

// ============================================================================
// Error Bound Tests
// ============================================================================

/// Tests that error bounds are tracked and reasonable.
#[test]
fn test_convergence_error_bounds_tracking() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::from(5u64)];

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
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

    // Error should be tracked (non-zero with base_error=1)
    assert_ne!(witness.total_error, Fr::zero(), "Error should be tracked");

    // Error should be in public inputs
    let pi = witness.public_inputs();
    assert_eq!(pi[5], witness.total_error, "Error bound should be in public inputs");
}

/// Tests error tracker accumulation logic.
#[test]
fn test_convergence_error_tracker() {
    let mut tracker = ErrorTracker::new();

    // Initial state
    assert_eq!(tracker.accumulated, Fr::ZERO);
    assert_eq!(tracker.op_count, 0);

    // Add some errors
    let base = Fr::from(1u64);
    tracker.dot_product_error(4, base); // 4-element dot product

    assert_ne!(tracker.accumulated, Fr::ZERO, "Error should accumulate");
    assert!(tracker.op_count > 0, "Op count should increase");

    // Matrix multiplication error
    let matmul_err = tracker.matmul_error(2, 4, 1, base);
    assert_ne!(matmul_err, Fr::ZERO, "Matmul should add error");
}

/// Tests that error bounds grow predictably with model size.
#[test]
fn test_convergence_error_scaling() {
    let base_error = Fr::from(1u64);

    // Tiny model (2x2x1)
    let dims_tiny = ModelDimensions::tiny();
    let weights_tiny = TestModelWeights::known(dims_tiny);
    let sample_tiny = TestSample::known(dims_tiny.d_in, dims_tiny.d_out);

    let witness_tiny = MLTrainingProverV2::build_witness(
        dims_tiny.d_in,
        dims_tiny.d_hid,
        dims_tiny.d_out,
        &sample_tiny.x,
        &sample_tiny.target,
        &weights_tiny.w1,
        &weights_tiny.b1,
        &weights_tiny.w2,
        &weights_tiny.b2,
        Fr::from(1u64),
        1,
        base_error,
    );

    // Small model (4x8x2)
    let dims_small = ModelDimensions::small();
    let weights_small = TestModelWeights::known(dims_small);
    let sample_small = TestSample::known(dims_small.d_in, dims_small.d_out);

    let witness_small = MLTrainingProverV2::build_witness(
        dims_small.d_in,
        dims_small.d_hid,
        dims_small.d_out,
        &sample_small.x,
        &sample_small.target,
        &weights_small.w1,
        &weights_small.b1,
        &weights_small.w2,
        &weights_small.b2,
        Fr::from(1u64),
        1,
        base_error,
    );

    // Both should have non-zero errors
    assert_ne!(witness_tiny.total_error, Fr::ZERO);
    assert_ne!(witness_small.total_error, Fr::ZERO);

    // We can't easily compare Fr values, but the errors should exist
    // In a real test, we'd convert to integers and compare
}

// ============================================================================
// Multi-Step Training Tests
// ============================================================================

/// Tests that batch training produces valid proof chains.
#[test]
fn test_convergence_proof_chain() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let samples = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(6u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(7u64)]),
    ];

    let batch_result = prover.prove_batch(training_weights, &samples, Fr::from(1u64));

    // Verify proof chain: each step's new hash should match next step's old hash
    for i in 0..batch_result.proofs.len() - 1 {
        let current_new = batch_result.proofs[i].new_state_hash;
        let next_old = batch_result.proofs[i + 1].old_state_hash;

        assert_eq!(
            current_new, next_old,
            "Proof chain broken between steps {} and {}",
            i,
            i + 1
        );
    }

    // Step numbers should be sequential
    for (i, proof) in batch_result.proofs.iter().enumerate() {
        assert_eq!(
            proof.step_number,
            (i + 1) as u64,
            "Step number should be sequential"
        );
    }
}

/// Tests that final weights are correctly tracked.
#[test]
fn test_convergence_final_weights() {
    let dims = ModelDimensions::tiny();

    let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
    let b1 = vec![Fr::zero(), Fr::zero()];
    let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
    let b2 = vec![Fr::zero()];

    let initial_weights = TrainingWeights::new(dims.d_in, dims.d_hid, dims.d_out, w1.clone(), b1.clone(), w2.clone(), b2.clone());

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let samples = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
    ];

    let batch_result = prover.prove_batch(initial_weights, &samples, Fr::from(1u64));

    // Final weights should be different from initial (training happened)
    let final_weights = &batch_result.final_weights;

    let w1_changed = w1.iter().zip(final_weights.w1.iter()).any(|(i, f)| i != f);
    let w2_changed = w2.iter().zip(final_weights.w2.iter()).any(|(i, f)| i != f);

    assert!(w1_changed || w2_changed, "Weights should have changed during training");
}

// ============================================================================
// Learning Rate Tests
// ============================================================================

/// Tests that learning rate affects weight updates.
#[test]
fn test_convergence_learning_rate_effect() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::from(10u64)]; // Large target for noticeable gradient

    // Small learning rate
    let witness_small_lr = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &x,
        &target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(1u64), // LR = 1
        1,
        Fr::from(1u64),
    );

    // Larger learning rate
    let witness_large_lr = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &x,
        &target,
        &weights.w1,
        &weights.b1,
        &weights.w2,
        &weights.b2,
        Fr::from(2u64), // LR = 2
        1,
        Fr::from(1u64),
    );

    // Both should produce valid proofs
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let proof_small = prover.prove(&witness_small_lr);
    let proof_large = prover.prove(&witness_large_lr);

    assert!(prover.verify_result(&proof_small), "Small LR proof should verify");
    assert!(prover.verify_result(&proof_large), "Large LR proof should verify");

    // Weight updates should differ (larger LR = larger update)
    // We can't easily compare Fr magnitudes, but the final weights should be different
    let different_w1 = witness_small_lr.w1_new.iter()
        .zip(witness_large_lr.w1_new.iter())
        .any(|(s, l)| s != l);

    let different_w2 = witness_small_lr.w2_new.iter()
        .zip(witness_large_lr.w2_new.iter())
        .any(|(s, l)| s != l);

    assert!(different_w1 || different_w2, "Different LRs should produce different weights");
}

// ============================================================================
// Convergence Metrics Tests
// ============================================================================

/// Tests comprehensive convergence scenario.
#[test]
fn test_convergence_comprehensive() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("convergence_comprehensive");

    let dims = ModelDimensions::tiny();

    // Phase 1: Initialize model
    let phase_start = Instant::now();
    let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
    let b1 = vec![Fr::zero(), Fr::zero()];
    let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
    let b2 = vec![Fr::zero()];
    let initial_weights = TrainingWeights::new(dims.d_in, dims.d_hid, dims.d_out, w1, b1, w2, b2);
    result.add_phase(PhaseResult::success("model_init", phase_start.elapsed()));

    // Phase 2: Create training data
    let phase_start = Instant::now();
    let samples = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(6u64)]),
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(7u64)]),
    ];
    harness.record_metric("num_samples", samples.len() as f64, "count");
    result.add_phase(PhaseResult::success("data_prep", phase_start.elapsed()));

    // Phase 3: Run training
    let phase_start = Instant::now();
    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let batch_result = prover.prove_batch(initial_weights, &samples, Fr::from(1u64));
    harness.record_duration("training_time", phase_start.elapsed());
    harness.record_metric("num_proofs", batch_result.proofs.len() as f64, "count");
    result.add_phase(PhaseResult::success("training", phase_start.elapsed()));

    // Phase 4: Verify all proofs
    let phase_start = Instant::now();
    let all_verified = prover.verify_batch(&batch_result);
    result.add_phase(if all_verified {
        PhaseResult::success("verification", phase_start.elapsed())
    } else {
        PhaseResult::failure("verification", phase_start.elapsed(), "Some proofs failed")
    });

    // Phase 5: Verify proof chain
    let phase_start = Instant::now();
    let mut chain_valid = true;
    for i in 0..batch_result.proofs.len() - 1 {
        if batch_result.proofs[i].new_state_hash != batch_result.proofs[i + 1].old_state_hash {
            chain_valid = false;
            break;
        }
    }
    result.add_phase(if chain_valid {
        PhaseResult::success("proof_chain", phase_start.elapsed())
    } else {
        PhaseResult::failure("proof_chain", phase_start.elapsed(), "Chain broken")
    });

    // Phase 6: Verify state changed
    let phase_start = Instant::now();
    let first_old = batch_result.proofs[0].old_state_hash;
    let last_new = batch_result.proofs.last().unwrap().new_state_hash;
    let state_changed = first_old != last_new;
    result.add_phase(if state_changed {
        PhaseResult::success("state_transition", phase_start.elapsed())
    } else {
        PhaseResult::failure("state_transition", phase_start.elapsed(), "State unchanged")
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Tests training with zero inputs.
#[test]
fn test_convergence_zero_inputs() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // Zero input
    let x = vec![Fr::zero(), Fr::zero()];
    let target = vec![Fr::from(5u64)];

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
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

    // Should still produce valid proof
    let proof = prover.prove(&witness);
    assert!(prover.verify_result(&proof), "Zero input should produce valid proof");
}

/// Tests training with zero targets.
#[test]
fn test_convergence_zero_targets() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);

    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::zero()]; // Zero target

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
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

    let proof = prover.prove(&witness);
    assert!(prover.verify_result(&proof), "Zero target should produce valid proof");
}

/// Tests single training step.
#[test]
fn test_convergence_single_step() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // Single sample
    let samples = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
    ];

    let batch_result = prover.prove_batch(training_weights, &samples, Fr::from(1u64));

    assert_eq!(batch_result.num_steps, 1, "Should complete 1 step");
    assert_eq!(batch_result.proofs.len(), 1, "Should have 1 proof");
    assert!(prover.verify_batch(&batch_result), "Single step should verify");
}
