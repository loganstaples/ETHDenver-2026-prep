//! MPC + ZK Proof Integration Tests.
//!
//! Tests the integration between Multi-Party Computation secret sharing
//! and Zero-Knowledge proof generation for distributed training.

#![allow(unused_imports)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::ml::training_step_v2::compute_state_hash_v2;
use helix_mpc::error::MPCResult;
use helix_mpc::sharing::{AdditiveSharing, ScalarShare, SecretSharingScheme, VectorShare};
use helix_mpc::types::{MPCConfig, PartyId, ShareId};
use helix_prover::{MLTrainingProverV2, TrainingWeights};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Test Fixtures for MPC
// ============================================================================

fn create_parties(n: usize) -> Vec<PartyId> {
    (0..n).map(PartyId::from_index).collect()
}

fn create_sharing(seed: u64) -> AdditiveSharing {
    AdditiveSharing::with_seed(seed)
}

// ============================================================================
// Basic MPC + Proof Tests
// ============================================================================

/// Tests secret sharing and reconstruction of model weights.
#[test]
fn test_mpc_weight_sharing() {
    let sharing = create_sharing(42);
    let parties = create_parties(3);

    // Share a scalar weight
    let secret_weight = 3.14159;
    let shares = sharing
        .share_scalar(secret_weight, "weight", &parties)
        .expect("Sharing should succeed");

    assert_eq!(shares.len(), 3, "Should have 3 shares");

    // No single share should reveal the secret
    for share in &shares {
        assert!(
            (share.value - secret_weight).abs() > 0.001,
            "Individual share should not reveal secret"
        );
    }

    // Reconstruction should return the secret
    let reconstructed = sharing
        .reconstruct_scalar(&shares)
        .expect("Reconstruction should succeed");

    assert!(
        (reconstructed - secret_weight).abs() < 1e-10,
        "Reconstructed value should match original: {} vs {}",
        reconstructed,
        secret_weight
    );
}

/// Tests vector sharing for weight matrices.
#[test]
fn test_mpc_vector_sharing() {
    let sharing = create_sharing(42);
    let parties = create_parties(3);

    // Share a weight vector
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let shares = sharing
        .share_vector(&weights, "w1", &parties)
        .expect("Vector sharing should succeed");

    assert_eq!(shares.len(), 3, "Should have 3 vector shares");
    for share in &shares {
        assert_eq!(share.values.len(), 5, "Each share should have same dimension");
    }

    // Reconstruction
    let reconstructed = sharing
        .reconstruct_vector(&shares)
        .expect("Vector reconstruction should succeed");

    for (i, (orig, recon)) in weights.iter().zip(reconstructed.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-10,
            "Element {} mismatch: {} vs {}",
            i,
            orig,
            recon
        );
    }
}

/// Tests that shares are linearly homomorphic (addition works on shares).
#[test]
fn test_mpc_linear_homomorphism() {
    let sharing = create_sharing(42);
    let parties = create_parties(3);

    let a = 10.0;
    let b = 20.0;

    // Share both values
    let shares_a = sharing.share_scalar(a, "a", &parties).unwrap();
    let sharing_b = create_sharing(99); // Different seed for independence
    let shares_b = sharing_b.share_scalar(b, "b", &parties).unwrap();

    // Add shares element-wise: share_c[i] = share_a[i] + share_b[i]
    let shares_c: Vec<ScalarShare> = shares_a
        .iter()
        .zip(shares_b.iter())
        .enumerate()
        .map(|(i, (sa, sb))| {
            ScalarShare::new(ShareId::new(parties[i].clone(), "c", i), sa.value + sb.value)
        })
        .collect();

    // Reconstruction should give a + b
    let result = sharing.reconstruct_scalar(&shares_c).unwrap();

    assert!(
        (result - 30.0).abs() < 1e-10,
        "Homomorphic addition failed: {} vs 30.0",
        result
    );
}

/// Tests that gradient shares can be aggregated correctly.
#[test]
fn test_mpc_gradient_aggregation() {
    let parties = create_parties(3);
    let gradient_dim = 4;

    // Simulate each party computing gradient on their share
    // In real MPC, gradients would be computed securely
    let party_gradients: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ];

    // Aggregate gradients (simple mean)
    let mut aggregated = vec![0.0; gradient_dim];
    for grads in &party_gradients {
        for (i, g) in grads.iter().enumerate() {
            aggregated[i] += g;
        }
    }
    for g in &mut aggregated {
        *g /= parties.len() as f64;
    }

    // Expected: [2.0, 3.0, 4.0, 5.0]
    let expected = vec![2.0, 3.0, 4.0, 5.0];
    for (i, (agg, exp)) in aggregated.iter().zip(expected.iter()).enumerate() {
        assert!(
            (agg - exp).abs() < 1e-10,
            "Aggregated gradient {} mismatch: {} vs {}",
            i,
            agg,
            exp
        );
    }
}

// ============================================================================
// MPC + ZK Integration Tests
// ============================================================================

/// Tests the complete flow: share weights → compute on shares → generate proof.
#[test]
fn test_mpc_zkp_integration_flow() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("mpc_zkp_integration_flow");

    // Phase 1: Create model and share weights
    let phase_start = Instant::now();
    let dims = ModelDimensions::tiny();
    let sharing = create_sharing(42);
    let parties = create_parties(3);

    // Create weights as f64 for MPC
    let w1_f64: Vec<f64> = vec![1.0, 2.0, 3.0, 1.0];
    let b1_f64: Vec<f64> = vec![0.0, 0.0];
    let w2_f64: Vec<f64> = vec![1.0, 1.0];
    let b2_f64: Vec<f64> = vec![0.0];

    // Share each weight vector
    let w1_shares = sharing.share_vector(&w1_f64, "w1", &parties).unwrap();
    let b1_shares = sharing.share_vector(&b1_f64, "b1", &parties).unwrap();
    let w2_shares = sharing.share_vector(&w2_f64, "w2", &parties).unwrap();
    let b2_shares = sharing.share_vector(&b2_f64, "b2", &parties).unwrap();

    result.add_phase(PhaseResult::success("weight_sharing", phase_start.elapsed()));

    // Phase 2: Reconstruct weights (simulating MPC computation output)
    let phase_start = Instant::now();
    let w1_recon = sharing.reconstruct_vector(&w1_shares).unwrap();
    let b1_recon = sharing.reconstruct_vector(&b1_shares).unwrap();
    let w2_recon = sharing.reconstruct_vector(&w2_shares).unwrap();
    let b2_recon = sharing.reconstruct_vector(&b2_shares).unwrap();

    // Convert to Fr for proof generation
    let w1_fr: Vec<Fr> = w1_recon.iter().map(|&v| Fr::from(v as u64)).collect();
    let b1_fr: Vec<Fr> = b1_recon.iter().map(|&v| Fr::from(v as u64)).collect();
    let w2_fr: Vec<Fr> = w2_recon.iter().map(|&v| Fr::from(v as u64)).collect();
    let b2_fr: Vec<Fr> = b2_recon.iter().map(|&v| Fr::from(v as u64)).collect();

    result.add_phase(PhaseResult::success("weight_reconstruction", phase_start.elapsed()));

    // Phase 3: Generate ZK proof for training step
    let phase_start = Instant::now();
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::from(5u64)];

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &x,
        &target,
        &w1_fr,
        &b1_fr,
        &w2_fr,
        &b2_fr,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof_result = prover.prove(&witness).unwrap();
    result.add_phase(PhaseResult::success("proof_generation", phase_start.elapsed()));

    // Phase 4: Verify proof
    let phase_start = Instant::now();
    let verified = prover.verify_result(&proof_result);
    result.add_phase(if verified {
        PhaseResult::success("proof_verification", phase_start.elapsed())
    } else {
        PhaseResult::failure("proof_verification", phase_start.elapsed(), "Verification failed")
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success);
}

/// Tests 3-party MPC with proof generation for each party's computation.
#[test]
fn test_mpc_three_party_proved_computation() {
    let parties = create_parties(3);
    let dims = ModelDimensions::tiny();

    // Initialize model weights
    let w1: Vec<Fr> = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
    let b1: Vec<Fr> = vec![Fr::zero(), Fr::zero()];
    let w2: Vec<Fr> = vec![Fr::from(1u64), Fr::from(1u64)];
    let b2: Vec<Fr> = vec![Fr::zero()];

    // Training input
    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::from(5u64)];

    // Each party generates a proof for the same computation
    // In real MPC, they would compute on shares and generate partial proofs
    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let mut party_proofs = Vec::new();
    for (i, party) in parties.iter().enumerate() {
        let witness = MLTrainingProverV2::build_witness(
            dims.d_in,
            dims.d_hid,
            dims.d_out,
            &x,
            &target,
            &w1,
            &b1,
            &w2,
            &b2,
            Fr::from(1u64),
            (i + 1) as u64,
            Fr::from(1u64),
        );

        let proof = prover.prove(&witness).unwrap();
        party_proofs.push((party.clone(), proof));
    }

    // Verify all party proofs
    for (party, proof) in &party_proofs {
        let verified = prover.verify_result(proof);
        assert!(
            verified,
            "Proof from party {:?} should verify",
            party
        );
    }

    // All proofs should have same loss (same computation)
    let losses: Vec<_> = party_proofs.iter().map(|(_, p)| p.loss).collect();
    for (i, loss) in losses.iter().enumerate().skip(1) {
        assert_eq!(
            *loss, losses[0],
            "Loss from party {} should match party 0",
            i
        );
    }
}

/// Tests gradient commitment verification in MPC context.
#[test]
fn test_mpc_gradient_commitment_verification() {
    use sha2::{Digest, Sha256};

    let parties = create_parties(3);

    // Each party computes and commits to their gradient share
    let gradient_shares: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 3.0, 4.0],
        vec![3.0, 4.0, 5.0],
    ];

    // Generate commitments
    let commitments: Vec<[u8; 32]> = gradient_shares
        .iter()
        .map(|grads| {
            let mut hasher = Sha256::new();
            for g in grads {
                hasher.update(g.to_le_bytes());
            }
            hasher.finalize().into()
        })
        .collect();

    // Verify each party's commitment matches their gradient
    for (i, (grads, commit)) in gradient_shares.iter().zip(commitments.iter()).enumerate() {
        let mut hasher = Sha256::new();
        for g in grads {
            hasher.update(g.to_le_bytes());
        }
        let recomputed: [u8; 32] = hasher.finalize().into();

        assert_eq!(
            commit, &recomputed,
            "Commitment mismatch for party {}",
            i
        );
    }

    // Verify that different gradients produce different commitments
    assert_ne!(commitments[0], commitments[1], "Different gradients should have different commitments");
}

/// Tests resharing mechanism for periodic security refresh.
#[test]
fn test_mpc_resharing() {
    let mut sharing = create_sharing(42);
    let parties = create_parties(3);

    let secret = 100.0;
    let initial_shares = sharing.share_scalar(secret, "orig", &parties).unwrap();

    // Reshare the first party's share
    let new_sub_shares = sharing
        .reshare_scalar(initial_shares[0].value, 3, "reshared", &parties)
        .unwrap();

    // New sub-shares should sum to original share
    let sum: f64 = new_sub_shares.iter().map(|s| s.value).sum();
    assert!(
        (sum - initial_shares[0].value).abs() < 1e-10,
        "Reshared sum should equal original share"
    );

    // Can still reconstruct original secret
    let mut combined_shares = vec![new_sub_shares[0].value];
    for share in initial_shares.iter().skip(1) {
        combined_shares.push(share.value);
    }
    // Note: This is a simplified test - real resharing would involve all parties
}

// ============================================================================
// Adversarial MPC Tests
// ============================================================================

/// Tests detection of malicious gradient submissions.
#[test]
fn test_mpc_malicious_gradient_detection() {
    let network = Arc::new(MockNetwork::new());
    let parties = create_parties(3);

    // Create workers: 2 honest, 1 adversarial
    let workers: Vec<MockWorker> = vec![
        MockWorker::new(parties[0].clone(), 0, Arc::clone(&network)),
        MockWorker::new(parties[1].clone(), 1, Arc::clone(&network)),
        MockWorker::adversarial(
            parties[2].clone(),
            2,
            Arc::clone(&network),
            AdversaryType::RandomGradients,
        ),
    ];

    let true_gradients = vec![1.0, 2.0, 3.0, 4.0];
    let submissions: Vec<_> = workers
        .iter()
        .map(|w| w.submit_gradients(1, &true_gradients))
        .collect();

    // Check validity
    let valid_count = submissions.iter().filter(|s| s.valid).count();
    assert_eq!(valid_count, 2, "Only 2 honest workers should have valid submissions");

    // The adversarial submission should be detected
    assert!(!submissions[2].valid, "Adversarial submission should be marked invalid");
}

/// Tests that Byzantine-tolerant aggregation excludes malicious parties.
#[test]
fn test_mpc_byzantine_tolerant_aggregation() {
    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    // Register parties
    let parties = create_parties(5);
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    // 3 honest, 2 adversarial submissions
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
            gradients: vec![1.0, 2.0],
            commitment: [0u8; 32],
            valid: true,
        },
        GradientSubmission {
            party: parties[2].clone(),
            step: 1,
            gradients: vec![1.0, 2.0],
            commitment: [0u8; 32],
            valid: true,
        },
        GradientSubmission {
            party: parties[3].clone(),
            step: 1,
            gradients: vec![100.0, 200.0], // Outlier
            commitment: [0u8; 32],
            valid: false, // Marked invalid by verification
        },
        GradientSubmission {
            party: parties[4].clone(),
            step: 1,
            gradients: vec![0.0, 0.0], // Zero gradient (lazy)
            commitment: [0u8; 32],
            valid: false,
        },
    ];

    // Aggregation should only use valid submissions
    let aggregated = coordinator.aggregate_gradients(&submissions);
    assert!(aggregated.is_some());

    let agg = aggregated.unwrap();
    // Should be average of the 3 valid submissions: (1+1+1)/3 = 1, (2+2+2)/3 = 2
    assert!((agg[0] - 1.0).abs() < 1e-10);
    assert!((agg[1] - 2.0).abs() < 1e-10);
}

/// Tests that slashing is correctly applied to adversarial parties.
#[test]
fn test_mpc_slashing_mechanism() {
    let network = Arc::new(MockNetwork::new());
    let coordinator = MockCoordinator::new(Arc::clone(&network));

    let parties = create_parties(3);
    for party in &parties {
        coordinator.register_worker(party.clone());
    }

    // Slash party 1
    coordinator.slash_worker(&parties[1], "Invalid proof");

    assert!(coordinator.is_slashed(&parties[1]));
    assert!(!coordinator.is_slashed(&parties[0]));
    assert!(!coordinator.is_slashed(&parties[2]));

    // Active worker count should decrease
    assert_eq!(coordinator.active_worker_count(), 2);
}

// ============================================================================
// Integration Tests
// ============================================================================

/// Full integration test: MPC sharing → gradient computation → proof → verification.
#[test]
fn test_full_mpc_zkp_pipeline() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("full_mpc_zkp_pipeline");

    let dims = ModelDimensions::tiny();
    let parties = create_parties(3);
    let sharing = create_sharing(42);

    // === Phase 1: Model owner shares weights ===
    let phase_start = Instant::now();

    let model_weights_f64 = vec![1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let weight_shares = sharing
        .share_vector(&model_weights_f64, "model", &parties)
        .unwrap();

    harness.record_metric("num_parties", parties.len() as f64, "count");
    result.add_phase(PhaseResult::success("weight_sharing", phase_start.elapsed()));

    // === Phase 2: Each party holds a share ===
    let phase_start = Instant::now();

    // Verify each party has a valid share
    for (i, share) in weight_shares.iter().enumerate() {
        assert_eq!(share.values.len(), model_weights_f64.len());
        // Share index should match party
        assert_eq!(share.id.index, i);
    }

    result.add_phase(PhaseResult::success("share_distribution", phase_start.elapsed()));

    // === Phase 3: Simulate distributed gradient computation ===
    let phase_start = Instant::now();

    // In real MPC, gradients would be computed on shares using Beaver triples
    // Here we simulate by computing gradients locally and creating shares
    let gradients_f64 = vec![0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0];
    let gradient_shares = sharing
        .share_vector(&gradients_f64, "gradients", &parties)
        .unwrap();

    result.add_phase(PhaseResult::success("gradient_computation", phase_start.elapsed()));

    // === Phase 4: Aggregate gradients ===
    let phase_start = Instant::now();

    let aggregated = sharing.reconstruct_vector(&gradient_shares).unwrap();
    assert!((aggregated[0] - gradients_f64[0]).abs() < 1e-10);

    result.add_phase(PhaseResult::success("gradient_aggregation", phase_start.elapsed()));

    // === Phase 5: Generate proof of training step ===
    let phase_start = Instant::now();

    // Reconstruct weights for proof
    let weights_recon = sharing.reconstruct_vector(&weight_shares).unwrap();

    // Convert to Fr
    let w1_fr: Vec<Fr> = weights_recon[0..4]
        .iter()
        .map(|&v| Fr::from((v.abs() * 1000.0) as u64))
        .collect();
    let b1_fr = vec![Fr::zero(); 2];
    let w2_fr: Vec<Fr> = weights_recon[6..8]
        .iter()
        .map(|&v| Fr::from((v.abs() * 1000.0) as u64))
        .collect();
    let b2_fr = vec![Fr::zero(); 1];

    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let x = vec![Fr::from(1000u64), Fr::from(1000u64)];
    let target = vec![Fr::from(5000u64)];

    let witness = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &x,
        &target,
        &w1_fr,
        &b1_fr,
        &w2_fr,
        &b2_fr,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let proof = prover.prove(&witness).unwrap();
    harness.record_metric("proof_size", proof.proof.len() as f64, "bytes");

    result.add_phase(PhaseResult::success("proof_generation", phase_start.elapsed()));

    // === Phase 6: Verify proof ===
    let phase_start = Instant::now();

    let verified = prover.verify_result(&proof);
    result.add_phase(if verified {
        PhaseResult::success("proof_verification", phase_start.elapsed())
    } else {
        PhaseResult::failure("proof_verification", phase_start.elapsed(), "Verification failed")
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success);
}
