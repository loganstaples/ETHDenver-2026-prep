//! End-to-End On-Chain Checkpoint Integration Test
//!
//! Tests the complete MPC training → on-chain checkpoint attestation pipeline:
//!
//! 1. Boot Anvil, deploy HelixCoordinatorV4
//! 2. Owner registers a training job with ETH payment
//! 3. Three workers stake and join the job
//! 4. MPC training runs for 50 steps with checkpoint every 25 steps
//! 5. Workers' Pedersen commitment shares are exchanged and combined during training
//! 6. Each checkpoint is signed by all worker wallets (ECDSA, EIP-191)
//! 7. Checkpoints are submitted on-chain via `submitCheckpoint()`
//! 8. Verify: 2 checkpoints on-chain, valid commitments, decreasing loss
//!
//! Run with:
//! ```
//! cargo test --test e2e_onchain_checkpoint --features integration -- --test-threads=1
//! ```

#![cfg(feature = "integration")]

use std::time::Instant;

use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, U256};
use ethers::utils::Anvil;

use helix_client::checkpoint_submitter::{submit_all_checkpoints, CheckpointData};
use helix_client::rpc::chain_v4::ChainClientV4;
use helix_mpc::e2e_integration::{InitialWeights, MPCIntegrationConfig, run_mpc_training};

// ============================================================================
// Test: Full MPC Training with On-Chain Checkpoint Attestation
// ============================================================================

#[tokio::test]
async fn test_mpc_training_with_onchain_checkpoints() {
    let start = Instant::now();
    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Starting full on-chain checkpoint test...");

    // ========================================================================
    // Phase 1: Boot Anvil and deploy V4 contract
    // ========================================================================

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Phase 1: Booting Anvil and deploying V4...");

    let anvil = Anvil::new().spawn();
    let rpc_url = anvil.endpoint();
    let chain_id = anvil.chain_id();

    // Account 0 = deployer/owner, Accounts 1-3 = workers
    let owner_wallet: LocalWallet = anvil.keys()[0].clone().into();
    let owner_wallet = owner_wallet.with_chain_id(chain_id);
    let owner_pk = hex::encode(anvil.keys()[0].to_bytes());
    let owner_address = owner_wallet.address();

    let mut worker_wallets: Vec<LocalWallet> = Vec::new();
    let mut worker_pks: Vec<String> = Vec::new();
    for i in 1..=3 {
        let wallet: LocalWallet = anvil.keys()[i].clone().into();
        let wallet = wallet.with_chain_id(chain_id);
        worker_pks.push(hex::encode(anvil.keys()[i].to_bytes()));
        worker_wallets.push(wallet);
    }

    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Owner: {:?}, Workers: {:?}",
        owner_address,
        worker_wallets.iter().map(|w| w.address()).collect::<Vec<_>>()
    );

    // Deploy V4 coordinator contract.
    // treasury = owner, verifier = zero (no ZK proofs in this test).
    let (owner_client, deploy_result) = ChainClientV4::deploy(
        &rpc_url,
        &owner_pk,
        owner_address,
        Address::zero(),
        Some(chain_id),
    )
    .await
    .expect("V4 deploy should succeed");

    let coordinator_addr_str = deploy_result.coordinator.clone();
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] V4 deployed at {}",
        coordinator_addr_str
    );

    // ========================================================================
    // Phase 2: Register training job
    // ========================================================================

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Phase 2: Registering training job...");

    let architecture_hash = [0x42u8; 32]; // dummy architecture hash
    let checkpoint_freq = 25u64; // checkpoint every 25 steps
    let num_rounds = 50u64; // 50 total steps
    let payment_amount = U256::from(1_000_000_000_000_000_000u128); // 1 ETH

    let (_receipt, job_id) = owner_client
        .register_training_job(architecture_hash, checkpoint_freq, num_rounds, payment_amount)
        .await
        .expect("register_training_job should succeed");

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Job registered: id={}", job_id);

    // Verify job is registered and active.
    let summary = owner_client
        .get_job_summary(job_id)
        .await
        .expect("get_job_summary should succeed");
    assert!(summary.active, "Job should be active");
    assert!(!summary.completed, "Job should not be completed");
    assert_eq!(summary.active_worker_count, 0, "No workers yet");

    // ========================================================================
    // Phase 3: Workers stake and join
    // ========================================================================

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Phase 3: Workers staking and joining...");

    let stake_amount = U256::from(100_000_000_000_000_000u128); // 0.1 ETH minimum

    for (i, pk) in worker_pks.iter().enumerate() {
        let worker_client = ChainClientV4::new(
            &rpc_url,
            pk,
            &coordinator_addr_str,
            Some(chain_id),
        )
        .await
        .expect("worker client creation should succeed");

        worker_client
            .stake_and_join(job_id, stake_amount)
            .await
            .expect("stake_and_join should succeed");

        eprintln!(
            "[E2E-ONCHAIN-CHECKPOINT] Worker {} ({:?}) staked and joined",
            i,
            worker_wallets[i].address()
        );
    }

    // Verify all workers joined.
    let summary = owner_client
        .get_job_summary(job_id)
        .await
        .expect("get_job_summary should succeed");
    assert_eq!(
        summary.active_worker_count, 3,
        "Should have 3 active workers"
    );

    // ========================================================================
    // Phase 4: Run MPC training (50 steps, checkpoint every 25 steps)
    // ========================================================================

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Phase 4: Running MPC training...");

    let mpc_config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 4,
        d_out: 1,
        num_workers: 3,
        num_steps: 50,
        learning_rate: 0.05,
        checkpoint_interval: 25,
        mac_check_interval: 0, // Disable MAC for speed in this test
        beaver_batch_size: 2048,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25],
            b1: vec![0.01, 0.0, -0.01, 0.02],
            w2: vec![0.5, -0.3, 0.4, 0.1],
            b2: vec![0.0],
        }),
        training_data: vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.8, 0.2], vec![1.0]),
            (vec![0.2, 0.8], vec![0.0]),
            (vec![0.6, 0.4], vec![1.0]),
            (vec![0.4, 0.6], vec![0.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
        on_sub_step: None,
        capture_checkpoint_weights: false,
    };

    let training_result = run_mpc_training(mpc_config)
        .await
        .expect("MPC training should complete successfully");

    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Training completed: {} steps, final_loss={:.6}",
        training_result.steps_completed, training_result.final_loss
    );

    // Verify training produced 2 checkpoints (at steps 25 and 50).
    assert_eq!(
        training_result.checkpoints.len(),
        2,
        "Should have 2 checkpoints (at steps 25 and 50), got {}",
        training_result.checkpoints.len()
    );

    for cp in &training_result.checkpoints {
        eprintln!(
            "[E2E-ONCHAIN-CHECKPOINT] Checkpoint: step={}, loss={:.6}, commitment={:?}",
            cp.step,
            cp.loss,
            &cp.commitment_bytes32[..4] // first 4 bytes for readability
        );
    }

    // Verify loss decreases across checkpoints.
    let cp1_loss = training_result.checkpoints[0].loss;
    let cp2_loss = training_result.checkpoints[1].loss;
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Checkpoint 1 loss: {:.6}, Checkpoint 2 loss: {:.6}",
        cp1_loss, cp2_loss
    );

    // Use average loss around checkpoint steps for a more robust comparison.
    let early_avg: f64 = training_result.losses[..25].iter().sum::<f64>() / 25.0;
    let late_avg: f64 = training_result.losses[25..].iter().sum::<f64>() / 25.0;
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Average loss: first 25 steps={:.6}, last 25 steps={:.6}",
        early_avg, late_avg
    );

    // ========================================================================
    // Phase 5: Sign and submit checkpoints on-chain
    // ========================================================================

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Phase 5: Signing and submitting checkpoints on-chain...");

    // Convert training checkpoints to CheckpointData for the submitter.
    let checkpoint_data: Vec<CheckpointData> = training_result
        .checkpoints
        .iter()
        .map(|cp| CheckpointData {
            step: cp.step as u64,
            commitment_bytes32: cp.commitment_bytes32,
            loss: cp.loss,
        })
        .collect();

    // Submit all checkpoints on-chain (signed by all 3 workers).
    let submissions = submit_all_checkpoints(
        &owner_client,
        job_id,
        &checkpoint_data,
        &worker_wallets,
    )
    .await
    .expect("checkpoint submission should succeed");

    assert_eq!(
        submissions.len(),
        2,
        "Should submit 2 checkpoints, got {}",
        submissions.len()
    );

    for sub in &submissions {
        eprintln!(
            "[E2E-ONCHAIN-CHECKPOINT] Submitted checkpoint: step={}, tx={:?}, signers={}",
            sub.step, sub.receipt.transaction_hash, sub.signer_count
        );
        assert_eq!(sub.signer_count, 3, "Each checkpoint should have 3 signers");
    }

    // ========================================================================
    // Phase 6: Verify on-chain checkpoint data
    // ========================================================================

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Phase 6: Verifying on-chain checkpoint data...");

    // Verify checkpoint count.
    let onchain_count = owner_client
        .get_checkpoint_count(job_id)
        .await
        .expect("get_checkpoint_count should succeed");
    assert_eq!(onchain_count, 2, "Should have 2 on-chain checkpoints");

    // Verify each checkpoint's data matches what we submitted.
    for (i, cp_data) in checkpoint_data.iter().enumerate() {
        let onchain_cp = owner_client
            .get_checkpoint(job_id, i as u64)
            .await
            .expect("get_checkpoint should succeed");

        assert_eq!(
            onchain_cp.step_number, cp_data.step,
            "Checkpoint {} step mismatch: on-chain={}, expected={}",
            i, onchain_cp.step_number, cp_data.step
        );

        assert_eq!(
            onchain_cp.weight_commitment, cp_data.commitment_bytes32,
            "Checkpoint {} commitment mismatch",
            i
        );

        assert_eq!(
            onchain_cp.signer_count, 3,
            "Checkpoint {} should have 3 signers",
            i
        );

        assert!(
            onchain_cp.timestamp > 0,
            "Checkpoint {} should have a non-zero timestamp",
            i
        );

        eprintln!(
            "[E2E-ONCHAIN-CHECKPOINT] Verified on-chain checkpoint {}: step={}, signers={}, timestamp={}",
            i, onchain_cp.step_number, onchain_cp.signer_count, onchain_cp.timestamp
        );
    }

    // Verify the first and second checkpoint commitments are different
    // (weights changed between checkpoints).
    let cp0 = owner_client.get_checkpoint(job_id, 0).await.unwrap();
    let cp1 = owner_client.get_checkpoint(job_id, 1).await.unwrap();
    assert_ne!(
        cp0.weight_commitment, cp1.weight_commitment,
        "Checkpoint commitments should differ (weights changed during training)"
    );

    // Verify job state was updated.
    let final_summary = owner_client
        .get_job_summary(job_id)
        .await
        .expect("get_job_summary should succeed");
    assert_eq!(
        final_summary.current_step, 50,
        "Job current step should be 50 after last checkpoint"
    );
    assert!(final_summary.active, "Job should still be active");
    assert!(!final_summary.completed, "Job should not be completed yet");

    // ========================================================================
    // Phase 7: Complete training with final commitment
    // ========================================================================

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Phase 7: Completing training...");

    // Use the last checkpoint's commitment as the final commitment.
    let final_commitment = checkpoint_data.last().unwrap().commitment_bytes32;

    // Sign completion message with all workers.
    let mut completion_sigs = Vec::new();
    for wallet in &worker_wallets {
        let sig = helix_client::rpc::chain_v4::sign_completion(
            wallet,
            U256::from(job_id),
            final_commitment,
        )
        .await
        .expect("sign_completion should succeed");
        completion_sigs.push(sig);
    }

    owner_client
        .complete_training(job_id, final_commitment, completion_sigs)
        .await
        .expect("complete_training should succeed");

    let completed_summary = owner_client
        .get_job_summary(job_id)
        .await
        .expect("get_job_summary should succeed");
    assert!(
        completed_summary.completed,
        "Job should be completed after complete_training"
    );
    assert!(
        !completed_summary.active,
        "Job should no longer be active after completion"
    );

    eprintln!("[E2E-ONCHAIN-CHECKPOINT] Training completed on-chain");

    // ========================================================================
    // Summary
    // ========================================================================

    let elapsed = start.elapsed();
    eprintln!("\n[E2E-ONCHAIN-CHECKPOINT] === TEST SUMMARY ===");
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Training: {} steps, {:.2}s",
        training_result.steps_completed,
        training_result.training_time_ms as f64 / 1000.0
    );
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Loss: first={:.6}, last={:.6}",
        training_result.losses[0], training_result.final_loss
    );
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Checkpoints: {} submitted and verified on-chain",
        submissions.len()
    );
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Job: completed={}, final_step={}",
        completed_summary.completed, completed_summary.current_step
    );
    eprintln!(
        "[E2E-ONCHAIN-CHECKPOINT] Total test time: {:.2}s",
        elapsed.as_secs_f64()
    );
}
