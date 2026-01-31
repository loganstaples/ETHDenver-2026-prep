//! Adversarial Node Simulation Tests.
//!
//! Tests the system's resilience against various attack vectors:
//! - Fake gradient submissions
//! - Invalid proofs
//! - Commitment mismatches
//! - Gradient norm violations
//! - Lazy workers (zero gradients)
//! - Delayed/unresponsive workers

#![allow(unused_imports)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_mpc::types::PartyId;
use helix_prover::{MLTrainingProverV2, TrainingWeights};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Adversarial Scenario Tests
// ============================================================================

/// Tests detection and slashing of workers submitting random garbage gradients.
#[test]
fn test_adversarial_random_gradients() {
    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    // Create workers: 3 honest, 2 adversarial with random gradients
    let workers: Vec<MockWorker> = vec![
        MockWorker::new(parties[0].clone(), 0, Arc::clone(&network)),
        MockWorker::new(parties[1].clone(), 1, Arc::clone(&network)),
        MockWorker::new(parties[2].clone(), 2, Arc::clone(&network)),
        MockWorker::adversarial(
            parties[3].clone(),
            3,
            Arc::clone(&network),
            AdversaryType::RandomGradients,
        ),
        MockWorker::adversarial(
            parties[4].clone(),
            4,
            Arc::clone(&network),
            AdversaryType::RandomGradients,
        ),
    ];

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];

    // Simulate training round
    let submissions: Vec<GradientSubmission> = workers
        .iter()
        .map(|w| w.submit_gradients(1, &true_gradients))
        .collect();

    // Count valid submissions
    let valid_count = submissions.iter().filter(|s| s.valid).count();
    assert_eq!(valid_count, 3, "Only honest workers should have valid submissions");

    // Verify adversarial submissions are marked invalid
    assert!(!submissions[3].valid, "Random gradient submission should be invalid");
    assert!(!submissions[4].valid, "Random gradient submission should be invalid");

    // Aggregation should exclude invalid submissions
    let aggregated = coordinator.aggregate_gradients(&submissions);
    assert!(aggregated.is_some(), "Aggregation should succeed with honest majority");

    let agg = aggregated.unwrap();
    // Should be exactly the true gradients (average of identical values)
    for (i, (a, t)) in agg.iter().zip(true_gradients.iter()).enumerate() {
        assert!(
            (a - t).abs() < 1e-10,
            "Aggregated gradient {} should match true gradient: {} vs {}",
            i,
            a,
            t
        );
    }
}

/// Tests detection of workers submitting zero gradients (lazy attack).
#[test]
fn test_adversarial_zero_gradients() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();

    let workers: Vec<MockWorker> = vec![
        MockWorker::new(parties[0].clone(), 0, Arc::clone(&network)),
        MockWorker::new(parties[1].clone(), 1, Arc::clone(&network)),
        MockWorker::adversarial(
            parties[2].clone(),
            2,
            Arc::clone(&network),
            AdversaryType::ZeroGradients,
        ),
    ];

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];
    let submissions: Vec<GradientSubmission> = workers
        .iter()
        .map(|w| w.submit_gradients(1, &true_gradients))
        .collect();

    // Zero gradient submission should be detected as invalid
    assert!(!submissions[2].valid, "Zero gradient submission should be invalid");

    // Verify the zero gradient worker actually submitted zeros
    if submissions[2].gradients.len() == true_gradients.len() {
        for g in &submissions[2].gradients {
            assert_eq!(*g, 0.0, "Lazy worker should submit zeros");
        }
    }
}

/// Tests detection of commitment mismatches (man-in-the-middle attack).
#[test]
fn test_adversarial_wrong_commitment() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();

    let honest_worker = MockWorker::new(parties[0].clone(), 0, Arc::clone(&network));
    let adversarial_worker = MockWorker::adversarial(
        parties[1].clone(),
        1,
        Arc::clone(&network),
        AdversaryType::WrongCommitment,
    );

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];

    let honest_sub = honest_worker.submit_gradients(1, &true_gradients);
    let adv_sub = adversarial_worker.submit_gradients(1, &true_gradients);

    // Honest submission should be valid
    assert!(honest_sub.valid, "Honest submission should be valid");

    // Wrong commitment should be detected
    assert!(!adv_sub.valid, "Wrong commitment should be detected as invalid");

    // Verify the commitment is actually wrong
    assert_eq!(adv_sub.commitment, [0xDE; 32], "Adversarial commitment should be forged");
}

/// Tests detection of gradient norm violations.
#[test]
fn test_adversarial_large_gradients() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();

    let honest_worker = MockWorker::new(parties[0].clone(), 0, Arc::clone(&network));
    let adversarial_worker = MockWorker::adversarial(
        parties[1].clone(),
        1,
        Arc::clone(&network),
        AdversaryType::LargeGradients,
    );

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];

    let honest_sub = honest_worker.submit_gradients(1, &true_gradients);
    let adv_sub = adversarial_worker.submit_gradients(1, &true_gradients);

    assert!(honest_sub.valid, "Honest submission should be valid");
    assert!(!adv_sub.valid, "Large gradient submission should be invalid");

    // Verify gradients are actually large
    let max_gradient = adv_sub.gradients.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    assert!(max_gradient > 100.0, "Adversarial gradients should be large");
}

/// Tests handling of unresponsive workers.
#[test]
fn test_adversarial_unresponsive_worker() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();

    let workers: Vec<MockWorker> = vec![
        MockWorker::new(parties[0].clone(), 0, Arc::clone(&network)),
        MockWorker::new(parties[1].clone(), 1, Arc::clone(&network)),
        MockWorker::adversarial(
            parties[2].clone(),
            2,
            Arc::clone(&network),
            AdversaryType::Unresponsive,
        ),
    ];

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];
    let submissions: Vec<GradientSubmission> = workers
        .iter()
        .map(|w| w.submit_gradients(1, &true_gradients))
        .collect();

    // Unresponsive worker submits empty
    assert!(submissions[2].gradients.is_empty(), "Unresponsive worker should submit nothing");
    assert!(!submissions[2].valid, "Unresponsive submission should be invalid");

    // The other workers should still be valid
    assert!(submissions[0].valid);
    assert!(submissions[1].valid);
}

/// Tests slashing mechanism for adversarial workers.
#[test]
fn test_adversarial_slashing() {
    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    assert_eq!(coordinator.active_worker_count(), 5);

    // Detect and slash adversarial workers
    for (i, party) in parties.iter().enumerate() {
        if i >= 3 {
            // Parties 3 and 4 are adversarial
            coordinator.slash_worker(party, "Invalid gradient submission");
        }
    }

    // Verify slashing
    assert!(!coordinator.is_slashed(&parties[0]));
    assert!(!coordinator.is_slashed(&parties[1]));
    assert!(!coordinator.is_slashed(&parties[2]));
    assert!(coordinator.is_slashed(&parties[3]));
    assert!(coordinator.is_slashed(&parties[4]));

    assert_eq!(coordinator.active_worker_count(), 3);
}

/// Tests that slashed workers cannot contribute to aggregation.
#[test]
fn test_slashed_workers_excluded() {
    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    // Party 1 gets slashed
    coordinator.slash_worker(&parties[1], "Previous violation");

    // All parties submit gradients
    let submissions = vec![
        GradientSubmission {
            party: parties[0].clone(),
            step: 1,
            gradients: vec![1.0, 2.0],
            commitment: [0u8; 32],
            valid: true,
        },
        GradientSubmission {
            party: parties[1].clone(),
            step: 1,
            gradients: vec![100.0, 200.0], // Would corrupt average
            commitment: [0u8; 32],
            valid: true, // Marked valid but should be excluded due to slashing
        },
        GradientSubmission {
            party: parties[2].clone(),
            step: 1,
            gradients: vec![1.0, 2.0],
            commitment: [0u8; 32],
            valid: true,
        },
    ];

    let aggregated = coordinator.aggregate_gradients(&submissions);
    assert!(aggregated.is_some());

    let agg = aggregated.unwrap();
    // Should only average parties 0 and 2, excluding slashed party 1
    assert!((agg[0] - 1.0).abs() < 1e-10, "Slashed party should be excluded from aggregation");
    assert!((agg[1] - 2.0).abs() < 1e-10, "Slashed party should be excluded from aggregation");
}

// ============================================================================
// ZK Proof Adversarial Tests
// ============================================================================

/// Tests that proofs with corrupted public inputs are rejected.
#[test]
fn test_adversarial_corrupted_public_inputs() {
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

    let proof_result = prover.prove(&witness);

    // Valid verification should pass
    assert!(prover.verify_result(&proof_result), "Valid proof should verify");

    // Test various corruptions
    let corruptions = vec![
        ("old_hash_lo", 0),
        ("old_hash_hi", 1),
        ("new_hash_lo", 2),
        ("new_hash_hi", 3),
        ("loss", 4),
        ("error_bound", 5),
        ("step_number", 6),
    ];

    for (name, idx) in corruptions {
        let mut corrupted_pi = proof_result.public_inputs.clone();
        corrupted_pi[idx] = Fr::from(0xDEADBEEFu64);

        let verification = prover.verify(&proof_result.proof, &corrupted_pi);
        assert!(
            !verification,
            "Corrupted {} (index {}) should fail verification",
            name,
            idx
        );
    }
}

/// Tests that proofs with wrong proof bytes are rejected.
#[test]
fn test_adversarial_corrupted_proof_bytes() {
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

    let proof_result = prover.prove(&witness);

    // Corrupt various parts of the proof
    let mut corrupted_proof = proof_result.proof.clone();
    if corrupted_proof.len() > 32 {
        // Flip some bits
        corrupted_proof[16] ^= 0xFF;
        corrupted_proof[32] ^= 0xFF;
    }

    let verification = prover.verify(&corrupted_proof, &proof_result.public_inputs);
    assert!(!verification, "Corrupted proof bytes should fail verification");
}

/// Tests that empty proofs are rejected.
#[test]
fn test_adversarial_empty_proof() {
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

    let proof_result = prover.prove(&witness);

    // Empty proof should fail
    let empty_proof: Vec<u8> = vec![];
    let verification = prover.verify(&empty_proof, &proof_result.public_inputs);
    assert!(!verification, "Empty proof should fail verification");
}

/// Tests that proofs with wrong number of public inputs are rejected.
#[test]
fn test_adversarial_wrong_pi_count() {
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

    let proof_result = prover.prove(&witness);

    // Too few public inputs
    let few_pi: Vec<Fr> = proof_result.public_inputs[..3].to_vec();
    let verification = prover.verify(&proof_result.proof, &few_pi);
    assert!(!verification, "Too few public inputs should fail");

    // Too many public inputs
    let mut many_pi = proof_result.public_inputs.clone();
    many_pi.push(Fr::from(999u64));
    let verification = prover.verify(&proof_result.proof, &many_pi);
    assert!(!verification, "Too many public inputs should fail");
}

// ============================================================================
// Network Adversarial Tests
// ============================================================================

/// Tests that partitioned adversaries cannot disrupt honest parties.
#[test]
fn test_adversarial_network_partition() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Partition adversarial nodes (3 and 4)
    network.start_partition(vec![parties[3].clone(), parties[4].clone()]);

    // Honest nodes can still communicate
    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 123,
    };

    assert!(network.send(&parties[1], msg.clone()), "Honest party should receive");
    assert!(network.send(&parties[2], msg.clone()), "Honest party should receive");
    assert!(!network.send(&parties[3], msg.clone()), "Partitioned party should not receive");
    assert!(!network.send(&parties[4], msg.clone()), "Partitioned party should not receive");

    // Verify network stats
    let stats = network.stats();
    assert!(stats.partition_active);
    assert!(stats.dropped > 0);
}

/// Tests recovery after adversarial partition ends.
#[test]
fn test_adversarial_partition_recovery() {
    let network = Arc::new(MockNetwork::new());

    let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
    for party in &parties {
        network.register_party(party.clone());
    }

    // Start partition
    network.start_partition(vec![parties[2].clone()]);

    let msg = NetworkMessage::Heartbeat {
        from: parties[0].clone(),
        timestamp: 123,
    };

    assert!(!network.send(&parties[2], msg.clone()), "Should not deliver during partition");

    // End partition
    network.end_partition();

    assert!(network.send(&parties[2], msg), "Should deliver after partition ends");
    assert!(!network.is_partitioned(&parties[2]));
}

// ============================================================================
// Comprehensive Adversarial Scenario
// ============================================================================

/// Tests a comprehensive adversarial scenario with multiple attack types.
#[test]
fn test_adversarial_comprehensive_scenario() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("adversarial_comprehensive");

    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    let parties: Vec<PartyId> = (0..7).map(PartyId::from_index).collect();
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    // Create diverse worker pool
    let workers: Vec<MockWorker> = vec![
        MockWorker::new(parties[0].clone(), 0, Arc::clone(&network)),
        MockWorker::new(parties[1].clone(), 1, Arc::clone(&network)),
        MockWorker::new(parties[2].clone(), 2, Arc::clone(&network)),
        MockWorker::new(parties[3].clone(), 3, Arc::clone(&network)),
        MockWorker::adversarial(parties[4].clone(), 4, Arc::clone(&network), AdversaryType::RandomGradients),
        MockWorker::adversarial(parties[5].clone(), 5, Arc::clone(&network), AdversaryType::LargeGradients),
        MockWorker::adversarial(parties[6].clone(), 6, Arc::clone(&network), AdversaryType::ZeroGradients),
    ];

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];

    // Phase 1: Collect submissions
    let phase_start = Instant::now();
    let submissions: Vec<GradientSubmission> = workers
        .iter()
        .map(|w| w.submit_gradients(1, &true_gradients))
        .collect();

    let valid_count = submissions.iter().filter(|s| s.valid).count();
    result.add_phase(if valid_count == 4 {
        PhaseResult::success("gradient_collection", phase_start.elapsed())
    } else {
        PhaseResult::failure(
            "gradient_collection",
            phase_start.elapsed(),
            &format!("Expected 4 valid, got {}", valid_count),
        )
    });

    // Phase 2: Detect and slash adversaries
    let phase_start = Instant::now();
    let mut slashed_count = 0;
    for (i, sub) in submissions.iter().enumerate() {
        if !sub.valid && i >= 4 {
            coordinator.slash_worker(&parties[i], "Invalid submission");
            slashed_count += 1;
        }
    }
    result.add_phase(if slashed_count == 3 {
        PhaseResult::success("adversary_detection", phase_start.elapsed())
    } else {
        PhaseResult::failure(
            "adversary_detection",
            phase_start.elapsed(),
            &format!("Expected 3 slashed, got {}", slashed_count),
        )
    });

    // Phase 3: Aggregate with only honest contributions
    let phase_start = Instant::now();
    let aggregated = coordinator.aggregate_gradients(&submissions);
    result.add_phase(if aggregated.is_some() {
        let agg = aggregated.unwrap();
        // Verify aggregation is correct (average of 4 identical valid submissions)
        let is_correct = agg.iter().zip(true_gradients.iter()).all(|(a, t)| (a - t).abs() < 1e-10);
        if is_correct {
            PhaseResult::success("honest_aggregation", phase_start.elapsed())
        } else {
            PhaseResult::failure("honest_aggregation", phase_start.elapsed(), "Aggregation corrupted")
        }
    } else {
        PhaseResult::failure("honest_aggregation", phase_start.elapsed(), "Aggregation failed")
    });

    // Phase 4: Verify honest majority maintained
    let phase_start = Instant::now();
    let active_count = coordinator.active_worker_count();
    result.add_phase(if active_count == 4 {
        PhaseResult::success("honest_majority", phase_start.elapsed())
    } else {
        PhaseResult::failure(
            "honest_majority",
            phase_start.elapsed(),
            &format!("Expected 4 active, got {}", active_count),
        )
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success, "Comprehensive adversarial test failed");
}
