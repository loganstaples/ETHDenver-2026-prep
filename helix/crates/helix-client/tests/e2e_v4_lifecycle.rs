//! V4 Full Lifecycle Integration Tests
//!
//! Tests the V4 MPC-primary coordinator via ethers:
//!   Deploy → Job registration → Worker staking → Checkpoint attestation →
//!   MAC failure reporting → Training completion → Stake withdrawal
//!
//! Requires `--features chain` and foundry (anvil).
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

/// Checkpoint every 10 steps.
const CHECKPOINT_FREQ: u64 = 10;

/// 100 total training rounds.
const NUM_ROUNDS: u64 = 100;

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

        // Wait for Anvil to be ready
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

/// Architecture hash for test model: keccak256("784,32,10").
fn test_arch_hash() -> [u8; 32] {
    keccak256(b"784,32,10")
}

/// Generate a deterministic weight commitment for a given step.
fn weight_commitment(step: u64) -> [u8; 32] {
    keccak256(format!("weights_at_step_{}", step).as_bytes())
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
// Tests
// ============================================================================

/// Basic: deploy V4 from Rust, register a job, verify on-chain state.
#[tokio::test]
async fn test_v4_deploy_and_register() -> Result<()> {
    let (_anvil, owner, _workers, _wallets) = deploy_v4_env(19700).await?;

    // Register a training job
    let payment = U256::from(PAYMENT_AMOUNT);
    let (receipt, job_id) = owner
        .register_training_job(test_arch_hash(), CHECKPOINT_FREQ, NUM_ROUNDS, payment)
        .await?;

    assert_eq!(job_id, 0);
    assert!(receipt
        .status
        .map(|s| s.as_u64() == 1)
        .unwrap_or(false));

    // Verify job state via view function
    let summary = owner.get_job_summary(0).await?;
    assert_eq!(summary.owner, owner.signer_address());
    assert_eq!(summary.current_step, 0);
    assert_eq!(summary.num_rounds, NUM_ROUNDS);
    assert_eq!(summary.payment_amount, payment);
    assert!(summary.active);
    assert!(!summary.completed);
    assert_eq!(summary.active_worker_count, 0);

    // Query events
    let events = owner.query_job_registered(0, None).await?;
    assert_eq!(events.len(), 1);

    Ok(())
}

/// Full happy path: register → stake → checkpoints → complete → withdraw.
#[tokio::test]
async fn test_v4_full_happy_path() -> Result<()> {
    let (anvil, owner, workers, wallets) = deploy_v4_env(19701).await?;
    let provider = Provider::<Http>::try_from(anvil.rpc_url())?;

    // 1. Register job
    let payment = U256::from(PAYMENT_AMOUNT);
    let (_, job_id) = owner
        .register_training_job(test_arch_hash(), CHECKPOINT_FREQ, NUM_ROUNDS, payment)
        .await?;
    assert_eq!(job_id, 0);

    // 2. Workers stake and join
    let stake = U256::from(STAKE_AMOUNT);
    for worker in &workers {
        worker.stake_and_join(job_id, stake).await?;
    }

    // Verify 3 active workers
    assert_eq!(owner.get_active_worker_count(job_id).await?, 3);

    // Verify each worker's info
    for wallet in &wallets {
        let info = owner.get_worker_info(job_id, wallet.address()).await?;
        assert_eq!(info.stake_amount, stake);
        assert!(info.registered);
        assert!(!info.slashed);
        assert!(owner.is_active_worker(job_id, wallet.address()).await?);
    }

    // 3. Submit 3 checkpoints
    for step in [10u64, 20, 30] {
        let commitment = weight_commitment(step);
        let loss = U256::from(1000 - step * 10);

        // All workers sign
        let mut sigs = Vec::new();
        for wallet in &wallets {
            let sig = sign_checkpoint(
                wallet,
                U256::from(job_id),
                U256::from(step),
                commitment,
                loss,
            )
            .await?;
            sigs.push(sig);
        }

        owner
            .submit_checkpoint(job_id, step, commitment, loss, sigs)
            .await?;

        // Verify state updated
        let summary = owner.get_job_summary(job_id).await?;
        assert_eq!(summary.current_step, step);
    }

    // Verify 3 checkpoints stored
    assert_eq!(owner.get_checkpoint_count(job_id).await?, 3);

    // Verify first checkpoint data
    let cp = owner.get_checkpoint(job_id, 0).await?;
    assert_eq!(cp.step_number, 10);
    assert_eq!(cp.weight_commitment, weight_commitment(10));
    assert_eq!(cp.signer_count, 3);

    // 4. Complete training
    let final_commitment = weight_commitment(999);
    let mut completion_sigs = Vec::new();
    for wallet in &wallets {
        let sig =
            sign_completion(wallet, U256::from(job_id), final_commitment).await?;
        completion_sigs.push(sig);
    }

    // Record worker balances before completion
    let mut balances_before = Vec::new();
    for wallet in &wallets {
        balances_before.push(provider.get_balance(wallet.address(), None).await?);
    }

    owner
        .complete_training(job_id, final_commitment, completion_sigs)
        .await?;

    // Verify job completed
    let summary = owner.get_job_summary(job_id).await?;
    assert!(summary.completed);
    assert!(!summary.active);

    // Verify workers received payment
    // All joined at step 0, last active at 30 → equal participation
    // Each should receive ~10/3 = 3.333... ETH
    let expected_per_worker = payment / 3;
    for (i, wallet) in wallets.iter().enumerate() {
        let bal_after = provider.get_balance(wallet.address(), None).await?;
        let received = bal_after - balances_before[i];
        // Allow 1 wei rounding for the last worker (gets remainder)
        assert!(
            received >= expected_per_worker - U256::from(1),
            "Worker {} received {:?}, expected >= {:?}",
            i,
            received,
            expected_per_worker
        );
    }

    // Verify payment events
    let pay_events = owner.query_payment_distributed(0, None).await?;
    assert_eq!(pay_events.len(), 3);

    // 5. Withdraw stakes after cooldown
    advance_time(anvil.rpc_url(), SEVEN_DAYS + 1).await?;

    for (i, worker) in workers.iter().enumerate() {
        let bal_before = provider.get_balance(wallets[i].address(), None).await?;
        let receipt = worker.withdraw_stake(job_id).await?;
        let bal_after = provider.get_balance(wallets[i].address(), None).await?;

        // bal_after = bal_before - gas_cost + stake
        let gas_cost = receipt.gas_used.unwrap_or_default()
            * receipt.effective_gas_price.unwrap_or_default();
        assert_eq!(
            bal_after + gas_cost,
            bal_before + stake,
            "Worker {} stake withdrawal mismatch",
            i
        );

        // Verify worker's stake is now zero
        let info = owner.get_worker_info(job_id, wallets[i].address()).await?;
        assert_eq!(info.stake_amount, U256::zero());
    }

    Ok(())
}

/// Slashing flow: register → stake → checkpoint → report MAC failure →
/// continue with remaining workers → complete → verify payment proportions.
#[tokio::test]
async fn test_v4_mac_failure_and_recovery() -> Result<()> {
    let (anvil, owner, workers, wallets) = deploy_v4_env(19702).await?;
    let provider = Provider::<Http>::try_from(anvil.rpc_url())?;

    // Setup: register job, stake all 3 workers
    let payment = U256::from(PAYMENT_AMOUNT);
    let stake = U256::from(STAKE_AMOUNT);
    let (_, job_id) = owner
        .register_training_job(test_arch_hash(), CHECKPOINT_FREQ, NUM_ROUNDS, payment)
        .await?;

    for worker in &workers {
        worker.stake_and_join(job_id, stake).await?;
    }

    // Submit checkpoint at step 10 (all 3 workers)
    let commitment = weight_commitment(10);
    let loss = U256::from(900);
    let mut sigs = Vec::new();
    for wallet in &wallets {
        sigs.push(
            sign_checkpoint(
                wallet,
                U256::from(job_id),
                U256::from(10u64),
                commitment,
                loss,
            )
            .await?,
        );
    }
    owner
        .submit_checkpoint(job_id, 10, commitment, loss, sigs)
        .await?;

    // ---- Report MAC failure on worker3 ----
    let cheater = wallets[2].address();
    let evidence = b"mac_verification_failed_sigma_mismatch";
    let step = 10u64;

    // Workers 1 & 2 sign the report (majority: 2 out of 3)
    let mut reporter_sigs = Vec::new();
    for wallet in &wallets[0..2] {
        let sig = sign_mac_failure(
            wallet,
            U256::from(job_id),
            U256::from(step),
            cheater,
            evidence,
        )
        .await?;
        reporter_sigs.push(sig);
    }

    // Record reporter balances before slashing
    let r1_bal_before = provider.get_balance(wallets[0].address(), None).await?;
    let r2_bal_before = provider.get_balance(wallets[1].address(), None).await?;

    owner
        .report_mac_failure(
            job_id,
            step,
            cheater,
            evidence.to_vec(),
            reporter_sigs,
        )
        .await?;

    // Verify worker3 slashed
    let info = owner.get_worker_info(job_id, cheater).await?;
    assert!(info.slashed);
    assert_eq!(info.stake_amount, U256::zero());
    assert!(!owner.is_active_worker(job_id, cheater).await?);
    assert_eq!(owner.get_active_worker_count(job_id).await?, 2);

    // Verify MAC failure report stored
    assert_eq!(owner.get_mac_failure_report_count(job_id).await?, 1);
    let report = owner.get_mac_failure_report(job_id, 0).await?;
    assert_eq!(report.step_number, step);
    assert_eq!(report.cheater, cheater);
    assert_eq!(report.slashed_amount, stake);
    assert_eq!(report.reporter_count, 2);

    // Verify slashing event
    let slash_events = owner.query_worker_slashed(0, None).await?;
    assert_eq!(slash_events.len(), 1);

    // Verify reporters received bounty (10% of 0.2 ETH / 2 = 0.01 ETH each)
    let bounty_total = stake * U256::from(10) / U256::from(100);
    let bounty_per = bounty_total / U256::from(2);

    let r1_bal_after = provider.get_balance(wallets[0].address(), None).await?;
    let r2_bal_after = provider.get_balance(wallets[1].address(), None).await?;
    assert_eq!(r1_bal_after - r1_bal_before, bounty_per);
    assert_eq!(r2_bal_after - r2_bal_before, bounty_per);

    // ---- Continue training with 2 remaining workers ----
    let commitment2 = weight_commitment(20);
    let loss2 = U256::from(800);
    let mut sigs2 = Vec::new();
    for wallet in &wallets[0..2] {
        sigs2.push(
            sign_checkpoint(
                wallet,
                U256::from(job_id),
                U256::from(20u64),
                commitment2,
                loss2,
            )
            .await?,
        );
    }
    owner
        .submit_checkpoint(job_id, 20, commitment2, loss2, sigs2)
        .await?;

    // Verify checkpoint count increased
    assert_eq!(owner.get_checkpoint_count(job_id).await?, 2);

    // ---- Complete training with 2 workers ----
    let final_commitment = weight_commitment(999);
    let mut completion_sigs = Vec::new();
    for wallet in &wallets[0..2] {
        completion_sigs.push(
            sign_completion(wallet, U256::from(job_id), final_commitment).await?,
        );
    }

    let w1_bal_before = provider.get_balance(wallets[0].address(), None).await?;
    let w2_bal_before = provider.get_balance(wallets[1].address(), None).await?;

    owner
        .complete_training(job_id, final_commitment, completion_sigs)
        .await?;

    let summary = owner.get_job_summary(job_id).await?;
    assert!(summary.completed);

    // Verify payment distribution
    // Worker1: joined=0, lastActive=20 → weight=20
    // Worker2: joined=0, lastActive=20 → weight=20
    // Total weight=40, each gets 50% = 5 ETH
    let w1_bal_after = provider.get_balance(wallets[0].address(), None).await?;
    let w2_bal_after = provider.get_balance(wallets[1].address(), None).await?;
    let w1_received = w1_bal_after - w1_bal_before;
    let w2_received = w2_bal_after - w2_bal_before;

    let expected = payment / 2;
    assert_eq!(w1_received, expected, "Worker1 payment mismatch");
    // Worker2 gets remainder — should be equal or within 1 wei
    assert!(
        w2_received >= expected - U256::from(1)
            && w2_received <= expected + U256::from(1),
        "Worker2 payment mismatch: got {:?}, expected ~{:?}",
        w2_received,
        expected
    );

    Ok(())
}

/// Verify all view functions return correct data at each lifecycle stage.
#[tokio::test]
async fn test_v4_view_functions() -> Result<()> {
    let (_anvil, owner, workers, wallets) = deploy_v4_env(19703).await?;

    // Register job
    let payment = U256::from(PAYMENT_AMOUNT);
    let (_, job_id) = owner
        .register_training_job(test_arch_hash(), CHECKPOINT_FREQ, NUM_ROUNDS, payment)
        .await?;

    // Pre-stake: no workers
    assert_eq!(owner.get_active_worker_count(job_id).await?, 0);
    assert!(owner.get_active_workers(job_id).await?.is_empty());
    assert!(!owner.is_active_worker(job_id, wallets[0].address()).await?);

    // Unregistered worker info
    let info = owner
        .get_worker_info(job_id, wallets[0].address())
        .await?;
    assert!(!info.registered);
    assert_eq!(info.stake_amount, U256::zero());

    // Stake workers
    let stake = U256::from(STAKE_AMOUNT);
    for worker in &workers {
        worker.stake_and_join(job_id, stake).await?;
    }

    // Post-stake verification
    let active = owner.get_active_workers(job_id).await?;
    assert_eq!(active.len(), 3);
    for wallet in &wallets {
        assert!(active.contains(&wallet.address()));
    }

    assert_eq!(owner.get_active_worker_count(job_id).await?, 3);

    for wallet in &wallets {
        let info = owner.get_worker_info(job_id, wallet.address()).await?;
        assert_eq!(info.stake_amount, stake);
        assert_eq!(info.joined_at_step, 0);
        assert_eq!(info.last_active_step, 0);
        assert!(info.registered);
        assert!(!info.slashed);
        assert!(owner.is_active_worker(job_id, wallet.address()).await?);
    }

    // Submit a checkpoint
    let commitment = weight_commitment(10);
    let loss = U256::from(900);
    let mut sigs = Vec::new();
    for wallet in &wallets {
        sigs.push(
            sign_checkpoint(
                wallet,
                U256::from(job_id),
                U256::from(10u64),
                commitment,
                loss,
            )
            .await?,
        );
    }
    owner
        .submit_checkpoint(job_id, 10, commitment, loss, sigs)
        .await?;

    // Post-checkpoint verification
    let summary = owner.get_job_summary(job_id).await?;
    assert_eq!(summary.owner, owner.signer_address());
    assert_eq!(summary.current_step, 10);
    assert_eq!(summary.num_rounds, NUM_ROUNDS);
    assert_eq!(summary.payment_amount, payment);
    assert_eq!(summary.active_worker_count, 3);
    assert!(summary.active);
    assert!(!summary.completed);

    // Workers' lastActiveStep should be updated
    for wallet in &wallets {
        let info = owner.get_worker_info(job_id, wallet.address()).await?;
        assert_eq!(info.last_active_step, 10);
    }

    // Checkpoint data
    assert_eq!(owner.get_checkpoint_count(job_id).await?, 1);
    let cp = owner.get_checkpoint(job_id, 0).await?;
    assert_eq!(cp.step_number, 10);
    assert_eq!(cp.weight_commitment, commitment);
    assert_eq!(cp.loss, loss);
    assert_eq!(cp.signer_count, 3);
    assert!(cp.timestamp > 0);

    // No MAC failure reports
    assert_eq!(owner.get_mac_failure_report_count(job_id).await?, 0);

    // Events
    let job_events = owner.query_job_registered(0, None).await?;
    assert_eq!(job_events.len(), 1);

    Ok(())
}
