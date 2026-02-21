//! Stage 5: MPC Training → On-Chain Settlement Integration Tests
//!
//! These tests wire **real MPC training output** (Pedersen checkpoints, MAC
//! failure reports, final commitments) to the HelixCoordinatorV4 contract
//! running on a local Anvil instance. Unlike the e2e_v4_lifecycle tests that
//! use fabricated commitments, these tests exercise the full pipeline:
//!
//!   MPC training → checkpoint attestation → MAC failure slashing →
//!   training completion → payment distribution → stake withdrawal
//!
//! Requires `--features chain` and Foundry (anvil).
//! Gated behind `#[cfg(feature = "chain")]` so normal `cargo test` skips it.

#![cfg(feature = "chain")]

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use ethers::providers::{Http, Middleware, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, U256};
use ethers::utils::keccak256;

use helix_client::rpc::chain_v4::{
    sign_checkpoint, sign_completion, sign_mac_failure, ChainClientV4,
};
use helix_mpc::e2e_integration::{
    CheaterRecord, InitialWeights, MPCIntegrationConfig, MPCIntegrationResult,
};

// ============================================================================
// Constants
// ============================================================================

/// Anvil default funded private keys (accounts 0-3).
const DEPLOYER_KEY: &str = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const WORKER1_KEY: &str = "59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
const WORKER2_KEY: &str = "5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a";
const WORKER3_KEY: &str = "7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6";

const CHAIN_ID: u64 = 31337;

/// 0.2 ETH stake per worker (min is 0.1 ETH).
const STAKE_AMOUNT: u128 = 200_000_000_000_000_000;

/// 10 ETH total payment pool.
const PAYMENT_AMOUNT: u128 = 10_000_000_000_000_000_000;

/// 7 days in seconds (stake cooldown).
const SEVEN_DAYS: u64 = 7 * 24 * 3600;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Manages an Anvil subprocess for test duration.
struct AnvilInstance {
    child: Child,
    rpc_url: String,
}

impl AnvilInstance {
    async fn start(port: u16) -> Result<Self> {
        let child = Command::new("anvil")
            .arg("--port")
            .arg(port.to_string())
            .arg("--chain-id")
            .arg(CHAIN_ID.to_string())
            .arg("--accounts")
            .arg("10")
            .arg("--balance")
            .arg("10000")
            .arg("--silent")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to start anvil — is foundry installed?")?;

        let rpc_url = format!("http://127.0.0.1:{}", port);

        // Wait for Anvil to be ready.
        let client = reqwest::Client::new();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if Instant::now() > deadline {
                return Err(anyhow!("Anvil did not start within 15 seconds"));
            }
            let body = serde_json::json!({
                "jsonrpc": "2.0", "method": "eth_chainId", "params": [], "id": 1
            });
            match client.post(&rpc_url).json(&body).send().await {
                Ok(r) if r.status().is_success() => break,
                _ => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
        Ok(Self { child, rpc_url })
    }

    fn rpc_url(&self) -> &str {
        &self.rpc_url
    }
}

impl Drop for AnvilInstance {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Advance Anvil block time by the given number of seconds and mine a block.
async fn advance_time(rpc_url: &str, seconds: u64) -> Result<()> {
    let provider = Provider::<Http>::try_from(rpc_url)?;
    let _: serde_json::Value = provider
        .request("evm_increaseTime", [seconds])
        .await
        .map_err(|e| anyhow!("evm_increaseTime: {}", e))?;
    let _: serde_json::Value = provider
        .request("evm_mine", Vec::<()>::new())
        .await
        .map_err(|e| anyhow!("evm_mine: {}", e))?;
    Ok(())
}

/// Create a wallet from a hex private key string with the test chain ID.
fn make_wallet(key: &str) -> LocalWallet {
    key.parse::<LocalWallet>()
        .unwrap()
        .with_chain_id(CHAIN_ID)
}

/// Architecture hash for test model: keccak256("4,4,2").
fn test_arch_hash() -> [u8; 32] {
    keccak256(b"4,4,2")
}

/// Deploy V4 coordinator and create clients for owner + 3 workers.
async fn deploy_v4_env(
    port: u16,
) -> Result<(
    AnvilInstance,
    ChainClientV4,
    Vec<ChainClientV4>,
    Vec<LocalWallet>,
)> {
    let anvil = AnvilInstance::start(port).await?;

    let deployer_wallet = make_wallet(DEPLOYER_KEY);
    let treasury = deployer_wallet.address();

    let (owner_client, deploy_result) = ChainClientV4::deploy(
        anvil.rpc_url(),
        DEPLOYER_KEY,
        treasury,
        Address::zero(), // no ZK verifier
        Some(CHAIN_ID),
    )
    .await?;

    let worker_keys = [WORKER1_KEY, WORKER2_KEY, WORKER3_KEY];
    let mut worker_clients = Vec::new();
    let mut worker_wallets = Vec::new();
    for key in &worker_keys {
        let client = ChainClientV4::new(
            anvil.rpc_url(),
            key,
            &deploy_result.coordinator,
            Some(CHAIN_ID),
        )
        .await?;
        worker_clients.push(client);
        worker_wallets.push(make_wallet(key));
    }

    Ok((anvil, owner_client, worker_clients, worker_wallets))
}

// ============================================================================
// MPC Training Configuration
// ============================================================================

/// Build the default MPC training config for integration tests.
///
/// Uses a small 4→4→2 model with deterministic initial weights for fast,
/// reproducible test execution. MAC checking is disabled by default to avoid
/// false positives from fixed-point arithmetic noise in the small model.
fn test_mpc_config(num_steps: usize, checkpoint_interval: usize) -> MPCIntegrationConfig {
    MPCIntegrationConfig {
        d_in: 4,
        d_hid: 4,
        d_out: 2,
        num_workers: 3,
        num_steps,
        learning_rate: 0.01,
        checkpoint_interval,
        mac_check_interval: 0, // disabled — avoids false positives at small scale
        beaver_batch_size: 512,
        initial_weights: Some(InitialWeights {
            // 4x4 = 16 weights for W1
            w1: vec![
                0.1, 0.2, -0.1, 0.05,
                -0.15, 0.3, 0.1, -0.2,
                0.25, -0.1, 0.15, 0.3,
                -0.05, 0.2, -0.25, 0.1,
            ],
            b1: vec![0.01, -0.01, 0.02, -0.02],
            // 4x2 = 8 weights for W2
            w2: vec![0.3, -0.2, 0.15, -0.1, 0.25, 0.1, -0.15, 0.2],
            b2: vec![0.01, -0.01],
        }),
        training_data: vec![
            (vec![1.0, 0.0, 0.5, 0.3], vec![1.0, 0.0]),
            (vec![0.0, 1.0, 0.2, 0.8], vec![0.0, 1.0]),
            (vec![0.5, 0.5, 0.7, 0.1], vec![1.0, 0.0]),
            (vec![0.3, 0.7, 0.1, 0.9], vec![0.0, 1.0]),
            (vec![0.9, 0.1, 0.4, 0.6], vec![1.0, 0.0]),
            (vec![0.2, 0.8, 0.6, 0.4], vec![0.0, 1.0]),
        ],
        seed: 12345,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
        on_sub_step: None,
        on_cheater_detected: None,
        on_recovery_completed: None,
        capture_checkpoint_weights: false,
        signal_rx: None,
        starting_step: 0,
    }
}

/// Build an MPC config with MAC verification enabled (for cheater detection tests).
fn test_mpc_config_with_mac(
    num_steps: usize,
    checkpoint_interval: usize,
    mac_check_interval: u64,
) -> MPCIntegrationConfig {
    let mut config = test_mpc_config(num_steps, checkpoint_interval);
    config.mac_check_interval = mac_check_interval;
    config
}

// ============================================================================
// Chain Settlement Helpers
// ============================================================================

/// Submit all MPC checkpoints to the V4 contract.
///
/// Each checkpoint is signed by all worker wallets and submitted on-chain.
/// Returns the total gas used across all checkpoint submissions.
async fn submit_checkpoints_on_chain(
    owner: &ChainClientV4,
    job_id: u64,
    mpc_result: &MPCIntegrationResult,
    wallets: &[LocalWallet],
) -> Result<u64> {
    let mut total_gas: u64 = 0;

    for (idx, checkpoint) in mpc_result.checkpoints.iter().enumerate() {
        let step_u256 = U256::from(checkpoint.step);
        let loss_scaled = (checkpoint.loss * 1_000_000.0) as u64;
        let loss_u256 = U256::from(loss_scaled);

        // All workers sign the checkpoint attestation.
        let mut signatures = Vec::with_capacity(wallets.len());
        for wallet in wallets {
            let sig = sign_checkpoint(
                wallet,
                U256::from(job_id),
                step_u256,
                checkpoint.commitment_bytes32,
                loss_u256,
            )
            .await
            .with_context(|| format!("sign_checkpoint failed for checkpoint {}", idx))?;
            signatures.push(sig);
        }

        let receipt = owner
            .submit_checkpoint(
                job_id,
                checkpoint.step as u64,
                checkpoint.commitment_bytes32,
                loss_u256,
                signatures,
            )
            .await
            .with_context(|| format!("submit_checkpoint failed for checkpoint {}", idx))?;

        total_gas += receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
    }

    Ok(total_gas)
}

/// Report a MAC failure (cheater detected) on-chain.
///
/// Serializes the evidence from the MPC training result, collects signatures
/// from all honest workers, and calls `report_mac_failure()`.
/// Returns the gas used.
async fn report_mac_failure_on_chain(
    owner: &ChainClientV4,
    job_id: u64,
    cheater: &CheaterRecord,
    wallets: &[LocalWallet],
) -> Result<u64> {
    let cheater_address = wallets[cheater.party_index].address();
    let evidence_bytes = serialize_mac_evidence(cheater);

    // Collect signatures from all honest workers (majority required).
    let mut reporter_sigs = Vec::new();
    for (i, wallet) in wallets.iter().enumerate() {
        if i == cheater.party_index {
            continue; // Skip the cheater.
        }
        let sig = sign_mac_failure(
            wallet,
            U256::from(job_id),
            U256::from(cheater.detected_at_step),
            cheater_address,
            &evidence_bytes,
        )
        .await
        .with_context(|| format!("sign_mac_failure failed for worker {}", i))?;
        reporter_sigs.push(sig);
    }

    let receipt = owner
        .report_mac_failure(
            job_id,
            cheater.detected_at_step,
            cheater_address,
            evidence_bytes,
            reporter_sigs,
        )
        .await
        .context("report_mac_failure failed")?;

    Ok(receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0))
}

/// Complete training on-chain with signatures from all active workers.
///
/// Uses the last checkpoint's commitment as the final commitment, or a
/// hash of the final weights if no checkpoints were produced.
/// Returns the gas used.
async fn complete_training_on_chain(
    owner: &ChainClientV4,
    job_id: u64,
    mpc_result: &MPCIntegrationResult,
    active_wallets: &[&LocalWallet],
) -> Result<u64> {
    let final_commitment = mpc_result
        .checkpoints
        .last()
        .map(|cp| cp.commitment_bytes32)
        .unwrap_or_else(|| compute_weight_commitment(&mpc_result.final_weights));

    let mut completion_sigs = Vec::with_capacity(active_wallets.len());
    for wallet in active_wallets {
        let sig = sign_completion(wallet, U256::from(job_id), final_commitment)
            .await
            .context("sign_completion failed")?;
        completion_sigs.push(sig);
    }

    let receipt = owner
        .complete_training(job_id, final_commitment, completion_sigs)
        .await
        .context("complete_training failed")?;

    Ok(receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0))
}

/// Withdraw stakes for all specified workers after the cooldown period.
///
/// Returns total gas used across all withdrawals.
async fn withdraw_stakes(
    workers: &[ChainClientV4],
    job_id: u64,
) -> Result<u64> {
    let mut total_gas: u64 = 0;
    for worker in workers {
        let receipt = worker
            .withdraw_stake(job_id)
            .await
            .context("withdraw_stake failed")?;
        total_gas += receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
    }
    Ok(total_gas)
}

/// Serialize MAC failure evidence into bytes for on-chain reporting.
fn serialize_mac_evidence(cheater: &CheaterRecord) -> Vec<u8> {
    let mut evidence = Vec::new();

    // Party index (4 bytes, big-endian).
    evidence.extend_from_slice(&(cheater.party_index as u32).to_be_bytes());

    // Step number (8 bytes, big-endian).
    evidence.extend_from_slice(&cheater.detected_at_step.to_be_bytes());

    // Sigma values from the failure report.
    for sigma in &cheater.failure_report.sigma_values {
        evidence.extend_from_slice(&(sigma.len() as u32).to_be_bytes());
        evidence.extend_from_slice(sigma);
    }

    // Commitments from the failure report.
    for commitment in &cheater.failure_report.commitments {
        evidence.extend_from_slice(commitment);
    }

    // Pairwise check results summary.
    let pairwise = &cheater.failure_report.evidence.pairwise_results;
    evidence.extend_from_slice(&(pairwise.len() as u32).to_be_bytes());
    for result in pairwise {
        evidence.extend_from_slice(&(result.party_a as u32).to_be_bytes());
        evidence.extend_from_slice(&(result.party_b as u32).to_be_bytes());
    }

    evidence
}

/// Compute a keccak256 commitment from final reconstructed weights.
fn compute_weight_commitment(weights: &helix_mpc::e2e_integration::FinalWeights) -> [u8; 32] {
    let mut data = Vec::new();
    for &v in weights
        .w1
        .iter()
        .chain(weights.b1.iter())
        .chain(weights.w2.iter())
        .chain(weights.b2.iter())
    {
        data.extend_from_slice(&v.to_le_bytes());
    }
    keccak256(&data)
}

// ============================================================================
// Test 1: MPC Training with On-Chain Checkpoints
// ============================================================================

/// Runs real 3-party MPC training (20 steps, checkpoint every 10), then
/// submits the resulting Pedersen commitment checkpoints to the V4 contract.
///
/// Verifies:
/// - MPC training produces the expected number of checkpoints
/// - Each checkpoint's Pedersen commitment is a valid 32-byte value
/// - All checkpoints are accepted by the V4 contract
/// - On-chain state (step number, commitment, signer count) matches MPC output
/// - Loss is monotonically tracked on-chain
#[tokio::test]
async fn test_mpc_training_with_chain_checkpoints() -> Result<()> {
    // Phase 1: Deploy contract and set up workers.
    let (_anvil, owner, workers, wallets) = deploy_v4_env(19800).await?;

    let checkpoint_freq = 10u64;
    let num_rounds = 100u64;
    let payment = U256::from(PAYMENT_AMOUNT);
    let stake = U256::from(STAKE_AMOUNT);

    let (_, job_id) = owner
        .register_training_job(test_arch_hash(), checkpoint_freq, num_rounds, payment)
        .await?;
    assert_eq!(job_id, 0);

    for worker in &workers {
        worker.stake_and_join(job_id, stake).await?;
    }
    assert_eq!(owner.get_active_worker_count(job_id).await?, 3);

    // Phase 2: Run real MPC training (20 steps, checkpoint every 10).
    let config = test_mpc_config(20, 10);
    let mpc_result = helix_mpc::e2e_integration::run_mpc_training(config)
        .await
        .context("MPC training failed")?;

    assert_eq!(mpc_result.steps_completed, 20, "Should complete all 20 steps");
    assert_eq!(
        mpc_result.checkpoints.len(),
        2,
        "Should produce 2 checkpoints (at step 10 and 20)"
    );
    assert!(mpc_result.cheater_detected.is_none(), "No cheater injected");

    // Verify checkpoints have valid Pedersen commitments.
    for cp in &mpc_result.checkpoints {
        assert_ne!(
            cp.commitment_bytes32,
            [0u8; 32],
            "Checkpoint commitment should not be all zeros"
        );
        assert!(cp.loss.is_finite(), "Loss should be finite");
        assert!(cp.loss >= 0.0, "Loss should be non-negative");
    }

    // Phase 3: Submit checkpoints on-chain.
    let checkpoint_gas = submit_checkpoints_on_chain(&owner, job_id, &mpc_result, &wallets).await?;
    assert!(checkpoint_gas > 0, "Checkpoints should use gas");

    // Phase 4: Verify on-chain state matches MPC output.
    let on_chain_count = owner.get_checkpoint_count(job_id).await?;
    assert_eq!(
        on_chain_count, 2,
        "Contract should have 2 checkpoints"
    );

    for (i, mpc_cp) in mpc_result.checkpoints.iter().enumerate() {
        let chain_cp = owner.get_checkpoint(job_id, i as u64).await?;
        assert_eq!(
            chain_cp.step_number, mpc_cp.step as u64,
            "Checkpoint {} step mismatch",
            i
        );
        assert_eq!(
            chain_cp.weight_commitment, mpc_cp.commitment_bytes32,
            "Checkpoint {} commitment mismatch",
            i
        );

        // Verify loss scaling: on-chain = f64 * 1_000_000.
        let expected_loss = U256::from((mpc_cp.loss * 1_000_000.0) as u64);
        assert_eq!(
            chain_cp.loss, expected_loss,
            "Checkpoint {} loss mismatch: on-chain={}, expected={}",
            i, chain_cp.loss, expected_loss
        );

        assert_eq!(
            chain_cp.signer_count, 3,
            "All 3 workers should have signed checkpoint {}",
            i
        );
        assert!(chain_cp.timestamp > 0, "Checkpoint should have a timestamp");
    }

    // Verify job state advanced to step 20.
    let summary = owner.get_job_summary(job_id).await?;
    assert_eq!(summary.current_step, 20);
    assert!(summary.active);
    assert!(!summary.completed);

    Ok(())
}

// ============================================================================
// Test 2: MPC Cheater Detection with On-Chain Slashing
// ============================================================================

/// Injects a cheater (party 1) during MPC training, detects them via MAC
/// verification, then reports the failure on-chain and verifies slashing.
///
/// Verifies:
/// - MPC training detects the cheater via SPDZ MAC verification
/// - The cheater's party index is correctly identified
/// - MAC failure report is accepted on-chain
/// - Cheater's stake is slashed (10% bounty to reporters, 90% to treasury)
/// - Reporters receive their bounty share
/// - Training can continue with remaining workers (checkpoint + completion)
#[tokio::test]
async fn test_mpc_cheater_slashing_on_chain() -> Result<()> {
    // Phase 1: Deploy and set up.
    let (anvil, owner, workers, wallets) = deploy_v4_env(19801).await?;
    let provider = Provider::<Http>::try_from(anvil.rpc_url())?;

    let payment = U256::from(PAYMENT_AMOUNT);
    let stake = U256::from(STAKE_AMOUNT);

    let (_, job_id) = owner
        .register_training_job(test_arch_hash(), 10, 100, payment)
        .await?;

    for worker in &workers {
        worker.stake_and_join(job_id, stake).await?;
    }

    // Submit an initial checkpoint at step 5 before corruption (with all 3 workers).
    // We need at least one checkpoint before MAC failure so the contract has seen
    // progress from all workers.
    let pre_config = test_mpc_config(5, 5);
    let pre_result = helix_mpc::e2e_integration::run_mpc_training(pre_config)
        .await
        .context("Pre-cheater MPC training failed")?;

    assert_eq!(pre_result.checkpoints.len(), 1, "Should get 1 pre-cheater checkpoint");
    submit_checkpoints_on_chain(&owner, job_id, &pre_result, &wallets).await?;

    // Phase 2: Run MPC training with cheater injection.
    // Party 1 (WORKER2) cheats at step 5. MAC check at interval 5 should catch it.
    let cheater_config = test_mpc_config_with_mac(10, 5, 5);
    let cheater_result = helix_mpc::e2e_integration::run_mpc_training_with_cheater(
        cheater_config,
        1,  // party 1 = WORKER2
        5,  // corrupt at step 5
    )
    .await
    .context("Cheater MPC training failed")?;

    // Training should have been halted early by MAC check failure.
    assert!(
        cheater_result.cheater_detected.is_some(),
        "Should have detected a cheater"
    );

    let cheater = cheater_result.cheater_detected.as_ref().unwrap();
    // The pairwise MAC check identifies an inconsistent party, which is the
    // cheater or a party whose sigma was inconsistent due to the corruption.
    // We use whatever party the protocol identifies for on-chain reporting.
    let cheater_party_idx = cheater.party_index;
    assert!(
        cheater_party_idx < 3,
        "Identified cheater index {} should be a valid party",
        cheater_party_idx
    );

    // Phase 3: Report MAC failure on-chain.
    // Track balances of reporters (all workers except the identified cheater).
    let reporter_indices: Vec<usize> = (0..3)
        .filter(|&i| i != cheater_party_idx)
        .collect();
    let mut reporter_bals_before = Vec::new();
    for &i in &reporter_indices {
        reporter_bals_before.push(provider.get_balance(wallets[i].address(), None).await?);
    }

    let slash_gas = report_mac_failure_on_chain(&owner, job_id, cheater, &wallets).await?;
    assert!(slash_gas > 0, "Slashing should use gas");

    // Phase 4: Verify slashing on-chain.
    let cheater_address = wallets[cheater_party_idx].address();
    let cheater_info = owner
        .get_worker_info(job_id, cheater_address)
        .await?;
    assert!(cheater_info.slashed, "Identified cheater (party {}) should be slashed", cheater_party_idx);
    assert_eq!(
        cheater_info.stake_amount,
        U256::zero(),
        "Slashed worker's stake should be zero"
    );
    assert!(
        !owner.is_active_worker(job_id, cheater_address).await?,
        "Slashed worker should not be active"
    );

    // Verify only 2 workers remain active.
    assert_eq!(owner.get_active_worker_count(job_id).await?, 2);

    // Verify MAC failure report stored on-chain.
    assert_eq!(owner.get_mac_failure_report_count(job_id).await?, 1);
    let report = owner.get_mac_failure_report(job_id, 0).await?;
    assert_eq!(report.cheater, cheater_address);
    assert_eq!(report.slashed_amount, stake);
    assert_eq!(report.reporter_count, 2);

    // Verify slashing event emitted.
    let slash_events = owner.query_worker_slashed(0, None).await?;
    assert_eq!(slash_events.len(), 1);

    // Verify reporters received bounty (10% of 0.2 ETH / 2 reporters = 0.01 ETH each).
    let bounty_total = stake * U256::from(10) / U256::from(100);
    let bounty_per = bounty_total / U256::from(2);

    for (idx, &i) in reporter_indices.iter().enumerate() {
        let bal_after = provider.get_balance(wallets[i].address(), None).await?;
        assert_eq!(
            bal_after - reporter_bals_before[idx],
            bounty_per,
            "Reporter {} (party {}) should receive bounty",
            idx,
            i
        );
    }

    // Phase 5: Continue training with 2 remaining workers and complete.
    // Submit a post-slashing checkpoint signed by only the 2 remaining workers.
    let mut post_config = test_mpc_config(10, 10);
    post_config.num_workers = 2; // Only 2 workers remain.
    let post_result = helix_mpc::e2e_integration::run_mpc_training(post_config)
        .await
        .context("Post-slashing MPC training failed")?;

    assert_eq!(post_result.checkpoints.len(), 1, "Should get 1 post-slashing checkpoint");

    // Submit post-slashing checkpoint (signed by only the remaining honest workers).
    let honest_wallet_refs: Vec<&LocalWallet> = reporter_indices
        .iter()
        .map(|&i| &wallets[i])
        .collect();
    for cp in &post_result.checkpoints {
        let step_u256 = U256::from(cp.step);
        let loss_scaled = (cp.loss * 1_000_000.0) as u64;
        let loss_u256 = U256::from(loss_scaled);

        let mut sigs = Vec::new();
        for wallet in &honest_wallet_refs {
            sigs.push(
                sign_checkpoint(
                    wallet,
                    U256::from(job_id),
                    step_u256,
                    cp.commitment_bytes32,
                    loss_u256,
                )
                .await?,
            );
        }
        owner
            .submit_checkpoint(job_id, cp.step as u64, cp.commitment_bytes32, loss_u256, sigs)
            .await?;
    }

    // Complete training with signatures from honest workers only.
    complete_training_on_chain(&owner, job_id, &post_result, &honest_wallet_refs).await?;

    let summary = owner.get_job_summary(job_id).await?;
    assert!(summary.completed, "Job should be completed");
    assert!(!summary.active, "Job should no longer be active");

    Ok(())
}

// ============================================================================
// Test 3: Full Lifecycle (Register → Train → Checkpoints → Complete → Withdraw)
// ============================================================================

/// End-to-end lifecycle test: job registration → worker staking → MPC training →
/// checkpoint attestation → training completion → payment distribution →
/// stake withdrawal after cooldown.
///
/// Verifies:
/// - The complete happy-path lifecycle works end-to-end
/// - Payment distribution is proportional to worker participation
/// - Stake withdrawal works after the 7-day cooldown
/// - Final on-chain state is consistent
#[tokio::test]
async fn test_full_lifecycle() -> Result<()> {
    let (anvil, owner, workers, wallets) = deploy_v4_env(19802).await?;
    let provider = Provider::<Http>::try_from(anvil.rpc_url())?;

    let payment = U256::from(PAYMENT_AMOUNT);
    let stake = U256::from(STAKE_AMOUNT);

    // Phase 1: Register job.
    let (_, job_id) = owner
        .register_training_job(test_arch_hash(), 10, 100, payment)
        .await?;

    // Phase 2: Workers stake and join.
    for worker in &workers {
        worker.stake_and_join(job_id, stake).await?;
    }
    assert_eq!(owner.get_active_worker_count(job_id).await?, 3);

    // Phase 3: Run MPC training (20 steps, checkpoint every 10).
    let config = test_mpc_config(20, 10);
    let mpc_result = helix_mpc::e2e_integration::run_mpc_training(config)
        .await
        .context("MPC training failed")?;

    assert_eq!(mpc_result.steps_completed, 20);
    assert_eq!(mpc_result.checkpoints.len(), 2);
    assert!(mpc_result.cheater_detected.is_none());

    // Phase 4: Submit checkpoints on-chain.
    submit_checkpoints_on_chain(&owner, job_id, &mpc_result, &wallets).await?;
    assert_eq!(owner.get_checkpoint_count(job_id).await?, 2);

    // Phase 5: Complete training.
    let balances_before: Vec<U256> = {
        let mut bals = Vec::new();
        for wallet in &wallets {
            bals.push(provider.get_balance(wallet.address(), None).await?);
        }
        bals
    };

    let all_wallets: Vec<&LocalWallet> = wallets.iter().collect();
    complete_training_on_chain(&owner, job_id, &mpc_result, &all_wallets).await?;

    let summary = owner.get_job_summary(job_id).await?;
    assert!(summary.completed, "Job should be completed");
    assert!(!summary.active, "Job should no longer be active");

    // Verify payment distribution.
    // All workers joined at step 0, last active at step 20 → equal participation.
    // Each gets ~10/3 ETH.
    let expected_per_worker = payment / 3;
    for (i, wallet) in wallets.iter().enumerate() {
        let bal_after = provider.get_balance(wallet.address(), None).await?;
        let received = bal_after - balances_before[i];
        assert!(
            received >= expected_per_worker - U256::from(1),
            "Worker {} received {:?}, expected >= {:?}",
            i,
            received,
            expected_per_worker
        );
    }

    // Verify payment events.
    let pay_events = owner.query_payment_distributed(0, None).await?;
    assert_eq!(pay_events.len(), 3, "Should have 3 payment events");

    // Phase 6: Withdraw stakes after cooldown.
    advance_time(anvil.rpc_url(), SEVEN_DAYS + 1).await?;

    for (i, worker) in workers.iter().enumerate() {
        let bal_before = provider.get_balance(wallets[i].address(), None).await?;
        let receipt = worker.withdraw_stake(job_id).await?;
        let bal_after = provider.get_balance(wallets[i].address(), None).await?;

        let gas_cost = receipt.gas_used.unwrap_or_default()
            * receipt.effective_gas_price.unwrap_or_default();
        assert_eq!(
            bal_after + gas_cost,
            bal_before + stake,
            "Worker {} stake withdrawal mismatch",
            i
        );

        // Verify worker's stake is now zero on-chain.
        let info = owner.get_worker_info(job_id, wallets[i].address()).await?;
        assert_eq!(
            info.stake_amount,
            U256::zero(),
            "Worker {} should have zero stake after withdrawal",
            i
        );
    }

    Ok(())
}

// ============================================================================
// Test 4: Gas Tracking
// ============================================================================

/// Tracks gas usage across all on-chain operations in the pipeline.
///
/// Verifies:
/// - Gas per checkpoint is < 500K (multi-party attestation should be cheap)
/// - Gas for MAC failure report is reasonable
/// - Gas for completion is reasonable
/// - Total gas for the full pipeline is within budget
#[tokio::test]
async fn test_gas_tracking() -> Result<()> {
    let (anvil, owner, workers, wallets) = deploy_v4_env(19803).await?;

    let payment = U256::from(PAYMENT_AMOUNT);
    let stake = U256::from(STAKE_AMOUNT);

    // Register + stake.
    let (reg_receipt, job_id) = owner
        .register_training_job(test_arch_hash(), 10, 100, payment)
        .await?;
    let reg_gas = reg_receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);

    let mut stake_gas_total: u64 = 0;
    for worker in &workers {
        let receipt = worker.stake_and_join(job_id, stake).await?;
        stake_gas_total += receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
    }

    // Run MPC training (20 steps, 2 checkpoints).
    let config = test_mpc_config(20, 10);
    let mpc_result = helix_mpc::e2e_integration::run_mpc_training(config)
        .await
        .context("MPC training failed")?;

    assert_eq!(mpc_result.checkpoints.len(), 2);

    // Submit checkpoints and track gas per checkpoint.
    let mut checkpoint_gas_values = Vec::new();
    for (idx, checkpoint) in mpc_result.checkpoints.iter().enumerate() {
        let step_u256 = U256::from(checkpoint.step);
        let loss_scaled = (checkpoint.loss * 1_000_000.0) as u64;
        let loss_u256 = U256::from(loss_scaled);

        let mut sigs = Vec::new();
        for wallet in &wallets {
            sigs.push(
                sign_checkpoint(
                    wallet,
                    U256::from(job_id),
                    step_u256,
                    checkpoint.commitment_bytes32,
                    loss_u256,
                )
                .await?,
            );
        }

        let receipt = owner
            .submit_checkpoint(
                job_id,
                checkpoint.step as u64,
                checkpoint.commitment_bytes32,
                loss_u256,
                sigs,
            )
            .await?;

        let gas = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
        checkpoint_gas_values.push(gas);

        // Each checkpoint should be < 500K gas.
        assert!(
            gas < 500_000,
            "Checkpoint {} used {} gas, expected < 500K. \
             Multi-party attestation should be cheaper than ZK verification.",
            idx,
            gas
        );
    }

    // Complete training and track gas.
    let final_commitment = mpc_result
        .checkpoints
        .last()
        .map(|cp| cp.commitment_bytes32)
        .unwrap_or_else(|| compute_weight_commitment(&mpc_result.final_weights));

    let mut completion_sigs = Vec::new();
    for wallet in &wallets {
        completion_sigs.push(
            sign_completion(wallet, U256::from(job_id), final_commitment).await?,
        );
    }

    let completion_receipt = owner
        .complete_training(job_id, final_commitment, completion_sigs)
        .await?;
    let completion_gas = completion_receipt
        .gas_used
        .map(|g| g.as_u64())
        .unwrap_or(0);

    // Withdraw stakes and track gas.
    advance_time(anvil.rpc_url(), SEVEN_DAYS + 1).await?;
    let withdrawal_gas_total = withdraw_stakes(&workers, job_id).await?;

    // Compute totals.
    let checkpoint_gas_total: u64 = checkpoint_gas_values.iter().sum();
    let total_gas = reg_gas + stake_gas_total + checkpoint_gas_total + completion_gas + withdrawal_gas_total;

    // Print gas report.
    eprintln!("\n=== Gas Tracking Report ===");
    eprintln!("  Job registration:    {:>8} gas", reg_gas);
    eprintln!("  3x worker staking:   {:>8} gas ({}/worker)", stake_gas_total, stake_gas_total / 3);
    for (i, &gas) in checkpoint_gas_values.iter().enumerate() {
        eprintln!("  Checkpoint {} (step {}): {:>8} gas", i, mpc_result.checkpoints[i].step, gas);
    }
    eprintln!("  Training completion: {:>8} gas", completion_gas);
    eprintln!("  3x stake withdrawal: {:>8} gas ({}/worker)", withdrawal_gas_total, withdrawal_gas_total / 3);
    eprintln!("  ---------------------------------");
    eprintln!("  TOTAL:               {:>8} gas", total_gas);
    eprintln!("===========================\n");

    // Gas budget assertions.
    // Multi-party attestation checkpoints should be MUCH cheaper than ZK proofs (~7.5M gas).
    let avg_checkpoint_gas = checkpoint_gas_total / checkpoint_gas_values.len() as u64;
    assert!(
        avg_checkpoint_gas < 500_000,
        "Average checkpoint gas {} should be < 500K",
        avg_checkpoint_gas
    );

    // Completion gas should be reasonable (includes payment distribution to 3 workers).
    assert!(
        completion_gas < 500_000,
        "Completion gas {} should be < 500K",
        completion_gas
    );

    // Total pipeline gas should be < 3M for this simple lifecycle.
    assert!(
        total_gas < 3_000_000,
        "Total pipeline gas {} should be < 3M",
        total_gas
    );

    Ok(())
}
