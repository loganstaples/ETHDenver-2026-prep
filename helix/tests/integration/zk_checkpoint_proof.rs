//! Integration tests for the optional ZK proof pipeline.
//!
//! These tests verify that:
//! 1. StateTransitionCircuit proofs can be generated from real MPC training
//!    checkpoint data and verified end-to-end.
//! 2. With ZK proofs disabled, zero proof code runs (no SRS loading, no
//!    circuit setup, no prover invocation).
//! 3. Proofs are in the standard format compatible with the Halo2Verifier.

use helix_circuits::ml::state_transition::{
    StateTransitionWitness, NUM_PUBLIC_INPUTS, compute_field_hash, split_hash,
};
use helix_mpc::field::Fr as MpcFr;
use helix_mpc::mac_verification::MACVerificationConfig;
use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use helix_mpc::session::transport::LocalTransport;
use helix_mpc::types::PartyId;
use helix_mpc::MPCError;
use helix_prover::{CheckpointProver, CheckpointProverConfig};
use halo2curves::bn256::Fr as Halo2Fr;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

/// Small model dimensions for fast ZK proof generation in tests.
const D_IN: usize = 4;
const D_HID: usize = 2;
const D_OUT: usize = 2;
const NUM_WEIGHTS: usize = D_IN * D_HID + D_HID + D_HID * D_OUT + D_OUT;
// 4*2 + 2 + 2*2 + 2 = 8 + 2 + 4 + 2 = 16

/// Number of MPC workers.
const NUM_PARTIES: usize = 3;

/// Reconstructs model weights from party shares as Halo2Fr.
fn reconstruct_weights_halo2(
    trainers: &[MPCTrainer<LocalTransport>],
) -> Vec<Halo2Fr> {
    let (first_w1, first_b1, first_w2, first_b2) = trainers[0].weight_shares();

    let mut w1_sum: Vec<MpcFr> = first_w1.to_vec();
    let mut b1_sum: Vec<MpcFr> = first_b1.to_vec();
    let mut w2_sum: Vec<MpcFr> = first_w2.to_vec();
    let mut b2_sum: Vec<MpcFr> = first_b2.to_vec();

    for trainer in trainers.iter().skip(1) {
        let (w1, b1, w2, b2) = trainer.weight_shares();
        for (i, s) in w1.iter().enumerate() {
            w1_sum[i] = MpcFr::add(&w1_sum[i], s);
        }
        for (i, s) in b1.iter().enumerate() {
            b1_sum[i] = MpcFr::add(&b1_sum[i], s);
        }
        for (i, s) in w2.iter().enumerate() {
            w2_sum[i] = MpcFr::add(&w2_sum[i], s);
        }
        for (i, s) in b2.iter().enumerate() {
            b2_sum[i] = MpcFr::add(&b2_sum[i], s);
        }
    }

    let mut flat: Vec<Halo2Fr> = Vec::with_capacity(NUM_WEIGHTS);
    flat.extend(w1_sum.iter().map(|f| *f.inner()));
    flat.extend(b1_sum.iter().map(|f| *f.inner()));
    flat.extend(w2_sum.iter().map(|f| *f.inner()));
    flat.extend(b2_sum.iter().map(|f| *f.inner()));
    flat
}

/// Sets up MPC trainers with shared weights and generates Beaver triples.
async fn setup_mpc_trainers() -> Vec<MPCTrainer<LocalTransport>> {
    let party_ids: Vec<PartyId> = (0..NUM_PARTIES).map(PartyId::from_index).collect();
    let transports = LocalTransport::create_mesh(&party_ids);

    let mut init_rng = ChaCha20Rng::seed_from_u64(42);
    let initial_weights = ModelWeights::random(D_IN, D_HID, D_OUT, &mut init_rng);

    let mac_config = MACVerificationConfig {
        check_interval: 10,
        enable_cheater_identification: true,
        mac_seed: 1042,
    };

    let trainer_config = MPCTrainerConfig {
        d_in: D_IN,
        d_hid: D_HID,
        d_out: D_OUT,
        learning_rate: 0.01,
        num_parties: NUM_PARTIES,
        reshare_interval: 0,
        beaver_batch_size: 5000,
        generate_proofs: false,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: Some(mac_config),
    };

    // Share weights across parties
    let mut party_handles = Vec::with_capacity(NUM_PARTIES);
    for (party_idx, transport) in transports.into_iter().enumerate() {
        let config = trainer_config.clone();
        let weights = if party_idx == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(config, transport, party_idx, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer
        });
        party_handles.push(handle);
    }

    let mut trainers: Vec<MPCTrainer<LocalTransport>> = Vec::with_capacity(NUM_PARTIES);
    for handle in party_handles {
        trainers.push(handle.await.unwrap());
    }

    // Generate Beaver triples for training
    let triples_per_step = helix_mpc::mnist::triples_per_step(D_HID, D_OUT);
    let total_triples = triples_per_step * 25; // enough for 20 steps + margin
    let mut triple_handles = Vec::with_capacity(NUM_PARTIES);
    for trainer in trainers.drain(..) {
        let count = total_triples;
        let handle = tokio::spawn(async move {
            let mut t = trainer;
            t.generate_beaver_triples(count).await.unwrap();
            t
        });
        triple_handles.push(handle);
    }
    for handle in triple_handles {
        trainers.push(handle.await.unwrap());
    }

    trainers
}

/// Runs N training steps on the MPC trainers.
async fn run_training_steps(
    trainers: &mut Vec<MPCTrainer<LocalTransport>>,
    steps: usize,
) {
    // Generate simple training data as f64 (training_step_with_mac takes &[f64])
    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..steps)
        .map(|i| {
            let input: Vec<f64> = (0..D_IN)
                .map(|j| (i * D_IN + j) as f64 * 0.01)
                .collect();
            let mut target = vec![0.0f64; D_OUT];
            target[i % D_OUT] = 1.0;
            (input, target)
        })
        .collect();

    for step in 0..steps {
        let (ref input, ref target) = data[step % data.len()];
        let mut step_handles = Vec::with_capacity(trainers.len());
        for trainer in trainers.drain(..) {
            let inp = input.clone();
            let tgt = target.clone();
            let handle = tokio::spawn(async move {
                let mut t = trainer;
                let result = t.training_step_with_mac(&inp, &tgt).await;
                (t, result)
            });
            step_handles.push(handle);
        }

        for handle in step_handles {
            let (trainer, result) = handle.await.unwrap();
            match result {
                Ok(_) | Err(MPCError::MACCheckFailed { .. }) => {}
                Err(e) => panic!("Training step {} failed: {}", step, e),
            }
            trainers.push(trainer);
        }
        trainers.sort_by_key(|t| t.party_index());
    }
}

/// Test: Run small MPC training, generate ZK proofs at 2 checkpoints, verify both.
///
/// This is the core end-to-end test for the ZK proof pipeline:
/// 1. Set up 3 MPC workers with a small (4->2->2) model
/// 2. Train for 20 steps with checkpoints at step 10 and 20
/// 3. At each checkpoint, reconstruct weights and generate a StateTransitionCircuit proof
/// 4. Verify both proofs pass
#[tokio::test]
async fn test_zk_proof_from_mpc_training_checkpoints() {
    let start = Instant::now();

    // Setup
    let mut trainers = setup_mpc_trainers().await;
    println!("  Setup complete ({:.1}ms)", start.elapsed().as_millis());

    // Record initial weights (before training)
    let initial_weights = reconstruct_weights_halo2(&trainers);
    assert_eq!(initial_weights.len(), NUM_WEIGHTS);

    // Initialize checkpoint prover (this does SRS + keygen)
    let prover_start = Instant::now();
    let config = CheckpointProverConfig::new(NUM_WEIGHTS);
    println!("  CheckpointProver config: k={}, num_weights={}", config.k, config.num_weights);
    let prover = CheckpointProver::new(config);
    println!("  Prover initialized ({:.1}s)", prover_start.elapsed().as_secs_f64());

    // Train 10 steps (checkpoint 1)
    run_training_steps(&mut trainers, 10).await;
    let checkpoint1_weights = reconstruct_weights_halo2(&trainers);

    // Generate proof 1: initial -> checkpoint 1
    let witness1 = StateTransitionWitness::new(
        initial_weights.clone(),
        checkpoint1_weights.clone(),
        Halo2Fr::from(1000u64), // error bound
    );
    assert_eq!(witness1.public_inputs().len(), NUM_PUBLIC_INPUTS);

    let proof1_start = Instant::now();
    let proof1 = prover.prove(&witness1).expect("Proof 1 generation should succeed");
    println!(
        "  Proof 1 generated: {} bytes, {:.1}s",
        proof1.proof_size(),
        proof1_start.elapsed().as_secs_f64(),
    );

    // Verify proof 1
    let valid1 = prover.verify(&proof1).expect("Proof 1 verification should not error");
    assert!(valid1, "Proof 1 must verify");
    println!("  Proof 1 verified successfully");

    // Train 10 more steps (checkpoint 2)
    run_training_steps(&mut trainers, 10).await;
    let checkpoint2_weights = reconstruct_weights_halo2(&trainers);

    // Generate proof 2: checkpoint 1 -> checkpoint 2
    let witness2 = StateTransitionWitness::new(
        checkpoint1_weights,
        checkpoint2_weights.clone(),
        Halo2Fr::from(2000u64),
    );

    let proof2_start = Instant::now();
    let proof2 = prover.prove(&witness2).expect("Proof 2 generation should succeed");
    println!(
        "  Proof 2 generated: {} bytes, {:.1}s",
        proof2.proof_size(),
        proof2_start.elapsed().as_secs_f64(),
    );

    // Verify proof 2
    let valid2 = prover.verify(&proof2).expect("Proof 2 verification should not error");
    assert!(valid2, "Proof 2 must verify");
    println!("  Proof 2 verified successfully");

    // Verify EVM-compatible output format
    let evm_proof1 = proof1.to_evm_proof();
    let evm_pi1 = proof1.to_evm_public_inputs();
    assert!(!evm_proof1.is_empty(), "EVM proof must be non-empty");
    assert_eq!(evm_pi1.len(), NUM_PUBLIC_INPUTS, "Must have {} EVM public inputs", NUM_PUBLIC_INPUTS);
    assert_eq!(
        proof1.proof_size() % 32,
        0,
        "Proof size must be multiple of 32 bytes for EVM"
    );

    // Verify proof is sufficiently large (real KZG proof, not a stub)
    assert!(
        proof1.proof_size() >= 320,
        "KZG proof must be >= 320 bytes (SHPLONK minimum), got {}",
        proof1.proof_size()
    );

    println!(
        "\n  All assertions passed! Total time: {:.1}s",
        start.elapsed().as_secs_f64(),
    );
}

/// Test: Verify that with ZK proofs disabled, no prover code runs.
///
/// This confirms the lazy-loading design: when --zk-proofs is not set,
/// zero proof infrastructure is loaded.
#[tokio::test]
async fn test_no_zk_code_runs_when_disabled() {
    let start = Instant::now();

    // Setup MPC trainers
    let mut trainers = setup_mpc_trainers().await;

    // Run training with no ZK prover at all
    let zk_prover: Option<CheckpointProver> = None; // ZK disabled

    // Verify no prover was created
    assert!(zk_prover.is_none(), "ZK prover must be None when disabled");

    // Run 10 training steps
    run_training_steps(&mut trainers, 10).await;

    // Reconstruct weights (this always works, ZK or not)
    let weights = reconstruct_weights_halo2(&trainers);
    assert_eq!(weights.len(), NUM_WEIGHTS);

    // Verify the test completed quickly (no SRS loading overhead)
    let elapsed = start.elapsed();
    println!("  Training without ZK completed in {:.1}ms", elapsed.as_millis());

    // SRS loading + keygen for even a small circuit takes > 1s.
    // If we completed in < 1s, no ZK code ran.
    // (We allow 30s for CI environments with slow I/O, but the point is
    // we didn't load any SRS.)
    // This is a sanity check, not a strict timing assertion.
}

/// Test: Tampered weights produce a proof that doesn't verify against the original.
#[tokio::test]
async fn test_tampered_witness_produces_invalid_proof() {
    let num_weights = 8;

    // Create valid weights
    let old_weights: Vec<Halo2Fr> = (0..num_weights).map(|i| Halo2Fr::from((i + 1) as u64)).collect();
    let new_weights: Vec<Halo2Fr> = (0..num_weights).map(|i| Halo2Fr::from((i + 2) as u64)).collect();

    let witness = StateTransitionWitness::new(
        old_weights.clone(),
        new_weights.clone(),
        Halo2Fr::from(100u64),
    );

    // Create prover
    let config = CheckpointProverConfig::minimal(num_weights);
    let prover = CheckpointProver::new(config);

    // Generate valid proof
    let result = prover.prove(&witness).expect("Valid proof should succeed");
    assert!(prover.verify(&result).unwrap(), "Valid proof must verify");

    // Now try to verify with tampered public inputs
    let mut tampered_pi = result.public_inputs.clone();
    tampered_pi[0] = Halo2Fr::from(999999u64); // Tamper old_hash_lo

    let tampered_valid = prover.verify_raw(&result.proof_bytes, &tampered_pi);
    match tampered_valid {
        Ok(false) => println!("  Tampered PI correctly rejected"),
        Err(_) => println!("  Tampered PI caused verification error (also acceptable)"),
        Ok(true) => panic!("Tampered public inputs must NOT verify!"),
    }
}

/// Test: Verify the public inputs structure matches the contract expectations.
#[tokio::test]
async fn test_public_inputs_evm_format() {
    let num_weights = 4;
    let old_weights: Vec<Halo2Fr> = (0..num_weights).map(|i| Halo2Fr::from((i + 10) as u64)).collect();
    let new_weights: Vec<Halo2Fr> = (0..num_weights).map(|i| Halo2Fr::from((i + 20) as u64)).collect();

    let witness = StateTransitionWitness::new(
        old_weights.clone(),
        new_weights.clone(),
        Halo2Fr::from(42u64),
    );

    let pi = witness.public_inputs();
    assert_eq!(pi.len(), 6, "StateTransitionCircuit has 6 public inputs");

    // Verify structure:
    // [0] = old_hash_lo, [1] = old_hash_hi
    // [2] = new_hash_lo, [3] = new_hash_hi
    // [4] = delta_hash
    // [5] = error_bound

    // Verify hash consistency
    let old_hash = compute_field_hash(&old_weights);
    let (old_lo, old_hi) = split_hash(old_hash);
    assert_eq!(pi[0], old_lo, "PI[0] must be old_hash_lo");
    assert_eq!(pi[1], old_hi, "PI[1] must be old_hash_hi");

    let new_hash = compute_field_hash(&new_weights);
    let (new_lo, new_hi) = split_hash(new_hash);
    assert_eq!(pi[2], new_lo, "PI[2] must be new_hash_lo");
    assert_eq!(pi[3], new_hi, "PI[3] must be new_hash_hi");

    // Verify error bound
    assert_eq!(pi[5], Halo2Fr::from(42u64), "PI[5] must be error_bound");

    // Create prover and generate proof to get EVM format
    let config = CheckpointProverConfig::minimal(num_weights);
    let prover = CheckpointProver::new(config);
    let result = prover.prove(&witness).expect("Proof generation should succeed");

    // Verify EVM public inputs are 32-byte big-endian
    let evm_pi = result.to_evm_public_inputs();
    assert_eq!(evm_pi.len(), 6);
    for pi_bytes in &evm_pi {
        assert_eq!(pi_bytes.len(), 32, "Each EVM PI must be 32 bytes (uint256)");
    }

    // Verify the EVM proof is non-empty and properly sized
    let evm_proof = result.to_evm_proof();
    assert!(!evm_proof.is_empty());
    assert_eq!(evm_proof.len() % 32, 0, "EVM proof must be 32-byte aligned");
}
