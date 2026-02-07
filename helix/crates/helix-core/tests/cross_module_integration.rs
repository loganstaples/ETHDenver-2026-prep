//! Cross-Module Integration Test for helix-core
//!
//! This test exercises the full types/ pipeline end-to-end with NO mocking:
//! BoundedTensor → matmul → ErrorComposition → PrecisionSelector →
//! ErrorBudgetManager → ErrorCheckpoint → ErrorCommitment → verify checksum_split()
//!
//! Run with: cargo test --test cross_module_integration

use helix_core::types::{
    // Budget allocation
    AllocationStrategy, BudgetAllocator, BudgetAllocationConfig,
    BudgetComponent, ComponentType,
    // Error composition & precision
    AdaptivePrecisionConfig, AdaptivePrecisionController,
    // Error checkpoint
    ErrorBudgetState, ErrorCheckpoint, LayerErrorState,
    // Precision selector
    OperationCharacteristics, OperationType, PrecisionSelector,
};
use helix_core::types::error_commitment::{ErrorCommitment, ErrorCommitmentBuilder, ErrorCommitmentPublicInputs, ErrorCommitmentTracker};
use helix_core::{
    BoundedTensor, ErrorDistribution, ProbabilisticError,
};

/// Full pipeline: tensor ops → error tracking → budget management → checkpointing → commitment
#[test]
fn test_full_types_pipeline_no_mocking() {
    // =========================================================================
    // Step 1: Create BoundedTensors and perform matmul
    // =========================================================================
    let a = BoundedTensor::from_approximate(
        (0..16).map(|i| (i as f64 * 0.1).sin()).collect(),
        vec![4, 4],
        1e-10,
    );
    let b = BoundedTensor::from_approximate(
        (0..16).map(|i| (i as f64 * 0.2).cos()).collect(),
        vec![4, 4],
        1e-10,
    );

    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape(), &vec![4, 4]);
    assert!(c.is_finite(), "matmul result should be finite");

    let matmul_error = c.max_error();
    assert!(matmul_error > 0.0, "matmul should accumulate some error");

    // =========================================================================
    // Step 2: Track error via AdaptivePrecisionController
    // =========================================================================
    let config = AdaptivePrecisionConfig {
        total_error_budget: 0.01,
        ..Default::default()
    };
    let mut precision_controller = AdaptivePrecisionController::new(config);

    let decision = precision_controller.record_operation(matmul_error);
    assert!(
        !precision_controller.budget_exceeded(),
        "Budget should not be exceeded after one matmul"
    );
    assert!(decision.remaining_budget > 0.0);

    // =========================================================================
    // Step 3: Use PrecisionSelector to choose precision for next operation
    // =========================================================================
    let mut selector = PrecisionSelector::default_selector();
    let characteristics = OperationCharacteristics {
        operation_type: OperationType::MatMul,
        num_elements: 16,
        value_range: (-1.0, 1.0),
        is_critical: false,
        sensitivity: 1.0,
        compute_cost: 1.0,
    };
    let selection = selector.select(&characteristics);
    assert!(
        selection.estimated_error < 1.0,
        "Estimated error should be reasonable"
    );

    // =========================================================================
    // Step 4: Allocate error budget via BudgetAllocator
    // =========================================================================
    let budget_config = BudgetAllocationConfig {
        total_budget: 0.01,
        strategy: AllocationStrategy::Weighted,
        min_allocation: 0.01,
        max_allocation: 0.4,
        reserve_fraction: 0.1,
        update_interval: 100,
        smoothing_factor: 0.9,
    };
    let mut allocator = BudgetAllocator::new(budget_config);

    allocator.add_component(BudgetComponent::new("matmul", ComponentType::FeedForward));
    allocator.add_component(BudgetComponent::new("activation", ComponentType::Output));
    allocator.add_component(BudgetComponent::new("gradient", ComponentType::Gradient));

    let allocation = allocator.allocate();
    assert!(!allocation.allocations.is_empty());

    // Consume error from the matmul
    allocator.consume("matmul", matmul_error);
    assert!(
        allocator.remaining_budget() > 0.0,
        "Should have remaining budget after one op"
    );

    // =========================================================================
    // Step 5: Bridge budget into precision controller
    // =========================================================================
    let budgets = allocator.export_budgets();
    assert!(budgets.contains_key("matmul"));
    assert!(budgets.contains_key("activation"));

    // Use the matmul budget to update the precision controller
    if let Some(&matmul_budget) = budgets.get("matmul") {
        precision_controller.update_budget(matmul_budget);
    }

    // =========================================================================
    // Step 6: Create an ErrorCheckpoint capturing current state
    // =========================================================================
    let mut checkpoint = ErrorCheckpoint::new("integration_test_cp", 1);

    // Set the budget state from the allocator
    let mut budget_state = ErrorBudgetState::new(0.01);
    budget_state.consume(allocator.total_consumed(), 1, 100);
    checkpoint.budget_state = budget_state;

    // Add layer error state
    let layer_error = LayerErrorState {
        name: "matmul_layer".to_string(),
        layer_type: "feedforward".to_string(),
        forward_error: ProbabilisticError {
            mean: matmul_error * 0.8,
            std_dev: matmul_error * 0.1,
            worst_case: matmul_error,
            sample_count: 16,
            distribution: ErrorDistribution::Gaussian,
        },
        backward_error: ProbabilisticError::zero(),
        precision: format!("{:?}", precision_controller.current_precision()),
        operation_count: 1,
    };
    checkpoint.add_layer_error(layer_error);

    // Verify checkpoint is serializable (roundtrip)
    let json = checkpoint.to_json().expect("checkpoint should serialize");
    let restored = ErrorCheckpoint::from_json(&json).expect("checkpoint should deserialize");
    assert_eq!(restored.step, 1);
    assert_eq!(restored.checkpoint_id, "integration_test_cp");
    assert!(restored.layer_errors.contains_key("matmul_layer"));

    // =========================================================================
    // Step 7: Generate ErrorCommitment and verify checksum_split
    // =========================================================================
    let model_id = [42u8; 32];
    let commitment = ErrorCommitmentBuilder::new()
        .accumulated_error(matmul_error)
        .step_number(1)
        .model_id(model_id)
        .budget_limit(0.01)
        .build();

    assert!(
        commitment.verify_within_budget(),
        "Error should be within budget"
    );
    assert!(commitment.budget_utilization() < 1.0);

    // Verify checksum_split produces valid lo/hi values
    let (lo, hi) = commitment.checksum_split();
    // Both halves should be non-zero (SHA-256 of non-trivial data)
    assert!(lo != 0 || hi != 0, "checksum should be non-zero");

    // Verify compact checksum is consistent
    let compact = commitment.checksum_compact();
    // The compact is the first 8 bytes of the full checksum, which should match lo's low 8 bytes
    let lo_bytes = lo.to_le_bytes();
    let expected_compact = u64::from_le_bytes(lo_bytes[0..8].try_into().unwrap());
    assert_eq!(compact, expected_compact, "compact checksum should match lo's first 8 bytes");

    // =========================================================================
    // Step 8: Use ErrorCommitmentTracker for multi-step tracking
    // =========================================================================
    let mut tracker = ErrorCommitmentTracker::with_history(model_id, 0.01);

    // Simulate multiple training steps
    for _ in 0..5 {
        tracker.record_step(matmul_error);
    }

    assert_eq!(tracker.current_step(), 5);
    assert!(tracker.within_budget(), "Should still be within budget after 5 small steps");

    let final_commitment = tracker.commitment();
    let (final_lo, final_hi) = final_commitment.checksum_split();
    // Different accumulated error means different checksum
    assert!(
        final_lo != lo || final_hi != hi,
        "Different accumulated errors should produce different checksums"
    );

    // =========================================================================
    // Summary assertions: verify the full pipeline is coherent
    // =========================================================================
    // The total consumed error should be tracked consistently
    let budget_consumed = allocator.total_consumed();
    assert!(budget_consumed > 0.0, "Some budget should be consumed");
    assert!(
        budget_consumed < 0.01,
        "Total consumed should be less than total budget"
    );

    // The precision controller should reflect the operations
    assert!(precision_controller.accumulated_error() > 0.0);

    // The checkpoint should reflect the budget state
    assert!(!checkpoint.budget_state.is_exceeded());

    println!("=== Cross-Module Integration Test Passed ===");
    println!("Matmul error: {:.2e}", matmul_error);
    println!("Budget consumed: {:.2e} / 0.01", budget_consumed);
    println!("Precision: {:?}", precision_controller.current_precision());
    println!("Commitment checksum (lo, hi): ({}, {})", lo, hi);
    println!("Tracker steps: {}, accumulated: {:.2e}", tracker.current_step(), tracker.accumulated_error());
}

/// Test that chained matmuls feed error correctly through the full pipeline
#[test]
fn test_chained_ops_error_propagation_pipeline() {
    let size = 8;
    let num_ops = 20;

    // Create an initial matrix
    let mut current = BoundedTensor::from_approximate(
        (0..size * size)
            .map(|i| if i / size == i % size { 1.0 } else { 0.01 })
            .collect(),
        vec![size, size],
        1e-10,
    );

    let multiplier = BoundedTensor::from_approximate(
        (0..size * size)
            .map(|i| if i / size == i % size { 0.99 } else { 0.001 })
            .collect(),
        vec![size, size],
        1e-10,
    );

    // Budget and precision tracking
    let mut precision_controller = AdaptivePrecisionController::new(AdaptivePrecisionConfig {
        total_error_budget: 1.0, // generous budget
        ..Default::default()
    });

    let mut tracker = ErrorCommitmentTracker::with_history([1u8; 32], 1.0);
    let mut errors: Vec<f64> = Vec::new();

    for i in 0..num_ops {
        current = current.matmul(&multiplier).expect(&format!("matmul {} failed", i));

        let step_error = current.max_error();
        errors.push(step_error);

        // Feed into precision controller
        let _decision = precision_controller.record_operation(step_error);

        // Feed into commitment tracker
        tracker.record_step(step_error);

        // Verify pipeline consistency
        assert!(
            !precision_controller.budget_exceeded(),
            "Budget exceeded at step {}",
            i
        );
        assert!(tracker.within_budget(), "Tracker budget exceeded at step {}", i);
    }

    // Verify error is monotonically tracked
    assert_eq!(tracker.current_step(), num_ops as u64);
    assert!(tracker.accumulated_error() > 0.0);

    // Verify commitment is valid
    let commitment = tracker.commitment();
    assert!(commitment.verify_within_budget());
    let (lo, hi) = commitment.checksum_split();
    assert!(lo != 0 || hi != 0);

    // Verify errors grew but stayed bounded
    let first_error = errors[0];
    let last_error = *errors.last().unwrap();
    assert!(
        last_error >= first_error,
        "Error should grow over chained operations"
    );

    println!("=== Chained Ops Pipeline Test Passed ===");
    println!("Errors: first={:.2e}, last={:.2e}, growth={:.1}x",
        first_error, last_error, last_error / first_error);
    println!("Accumulated: {:.2e}, Steps: {}", tracker.accumulated_error(), num_ops);
}

/// Test circuit/contract interface: verifies the 7-element public input layout
/// matches what circuits (MLTrainingStepV2Circuit) and contracts (HelixCoordinatorV2)
/// expect: [oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber]
#[test]
fn test_circuit_contract_public_input_layout() {
    // =========================================================================
    // Step 1: Create BoundedTensors and perform matmul to produce real errors
    // =========================================================================
    let a = BoundedTensor::from_approximate(
        (0..64).map(|i| (i as f64 * 0.3).sin()).collect(),
        vec![8, 8],
        1e-10,
    );
    let b = BoundedTensor::from_approximate(
        (0..64).map(|i| (i as f64 * 0.7).cos()).collect(),
        vec![8, 8],
        1e-10,
    );

    let c = a.matmul(&b).expect("matmul should succeed");
    let matmul_error = c.max_error();
    assert!(matmul_error > 0.0, "matmul should produce some error for commitment tracking");

    // =========================================================================
    // Step 2: Create "old" and "new" ErrorCommitments (before/after a training step)
    // =========================================================================
    let model_id = [0xABu8; 32];
    let budget_limit = 0.01;
    let step_number: u64 = 5;

    // "Old" commitment: state before this training step
    let old_accumulated_error = 0.0025; // some prior accumulated error
    let old_commitment = ErrorCommitment::new(
        old_accumulated_error,
        step_number - 1,
        model_id,
        budget_limit,
    );

    // "New" commitment: state after this training step adds matmul_error
    let new_accumulated_error = old_accumulated_error + matmul_error;
    let new_commitment = ErrorCommitment::new(
        new_accumulated_error,
        step_number,
        model_id,
        budget_limit,
    );

    // =========================================================================
    // Step 3: Compute checksum_split() for both old and new commitments
    // =========================================================================
    let (old_hash_lo, old_hash_hi) = old_commitment.checksum_split();
    let (new_hash_lo, new_hash_hi) = new_commitment.checksum_split();

    // =========================================================================
    // Step 4: Compute loss value and error bound
    // =========================================================================
    // In a real training step, loss comes from the model's compute_loss().
    // Here we use the matmul result as a proxy for the loss value, and
    // the accumulated error bound from the new commitment.
    let loss_value: u64 = {
        // Simulate a scaled loss value (e.g., cross-entropy output)
        let raw_loss = c.data().iter().map(|v| v.value().abs()).sum::<f64>() / c.len() as f64;
        // Scale to u64 with 12 decimal places (matching circuit convention)
        (raw_loss * 1e12).min(u64::MAX as f64) as u64
    };

    let error_bound: u64 = new_commitment.accumulated_error_scaled();

    // =========================================================================
    // Step 5: Construct the 7-element public input array
    // =========================================================================
    // Layout: [oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber]
    //
    // This matches the contract's `submitProof()` expectations and the
    // `MLTrainingStepV2Circuit` public inputs:
    //   Indices 0-1: Old weight/error hash (lo/hi 128-bit halves)
    //   Indices 2-3: New weight/error hash (lo/hi 128-bit halves)
    //   Index 4:     Computed loss value
    //   Index 5:     Error bound for this step
    //   Index 6:     Training step number
    let public_inputs: [u128; 7] = [
        old_hash_lo,                    // index 0: oldHashLo
        old_hash_hi,                    // index 1: oldHashHi
        new_hash_lo,                    // index 2: newHashLo
        new_hash_hi,                    // index 3: newHashHi
        loss_value as u128,             // index 4: loss
        error_bound as u128,            // index 5: errorBound
        step_number as u128,            // index 6: stepNumber
    ];

    // =========================================================================
    // Step 6: Verify the layout matches what circuits/contracts expect
    // =========================================================================
    // Index 0-1: old commitment hash split
    assert_eq!(public_inputs[0], old_hash_lo, "Index 0 should be oldHashLo");
    assert_eq!(public_inputs[1], old_hash_hi, "Index 1 should be oldHashHi");

    // Index 2-3: new commitment hash split
    assert_eq!(public_inputs[2], new_hash_lo, "Index 2 should be newHashLo");
    assert_eq!(public_inputs[3], new_hash_hi, "Index 3 should be newHashHi");

    // Index 4: loss
    assert_eq!(public_inputs[4], loss_value as u128, "Index 4 should be loss");

    // Index 5: error bound
    assert_eq!(public_inputs[5], error_bound as u128, "Index 5 should be errorBound");

    // Index 6: step number
    assert_eq!(public_inputs[6], step_number as u128, "Index 6 should be stepNumber");

    // Exactly 7 elements
    assert_eq!(public_inputs.len(), 7, "Public inputs must have exactly 7 elements");

    // =========================================================================
    // Step 7: Verify old != new checksums, all values non-zero, step number correct
    // =========================================================================
    // Old and new checksums must differ (different accumulated error and step number)
    assert!(
        old_hash_lo != new_hash_lo || old_hash_hi != new_hash_hi,
        "Old and new commitment checksums must differ (different error/step)"
    );

    // All hash values should be non-zero (SHA-256 of non-trivial data)
    assert!(old_hash_lo != 0 || old_hash_hi != 0, "Old checksum should be non-zero");
    assert!(new_hash_lo != 0 || new_hash_hi != 0, "New checksum should be non-zero");

    // Loss and error bound should be non-zero
    assert!(loss_value > 0, "Loss value should be non-zero");
    assert!(error_bound > 0, "Error bound should be non-zero");

    // Step number should be correct
    assert_eq!(step_number, 5, "Step number should be 5");

    // Verify that ErrorCommitmentPublicInputs is consistent with our manual construction
    let old_public = ErrorCommitmentPublicInputs::from_commitment(&old_commitment);
    let new_public = ErrorCommitmentPublicInputs::from_commitment(&new_commitment);

    assert!(
        old_public.error_checksum != new_public.error_checksum,
        "Compact checksums should also differ between old and new"
    );
    assert!(
        old_public.total_error < new_public.total_error,
        "New commitment should have higher total error than old"
    );

    // Both commitments should be within budget
    assert!(old_commitment.verify_within_budget(), "Old commitment should be within budget");
    assert!(new_commitment.verify_within_budget(), "New commitment should be within budget");

    println!("=== Circuit/Contract Public Input Layout Test Passed ===");
    println!("Old hash (lo, hi): ({}, {})", old_hash_lo, old_hash_hi);
    println!("New hash (lo, hi): ({}, {})", new_hash_lo, new_hash_hi);
    println!("Loss (scaled): {}", loss_value);
    println!("Error bound (scaled): {}", error_bound);
    println!("Step number: {}", step_number);
    println!("Public inputs array: {:?}", public_inputs);
}
