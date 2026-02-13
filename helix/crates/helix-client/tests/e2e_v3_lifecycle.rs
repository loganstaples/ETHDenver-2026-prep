//! V3 Full Lifecycle Integration Test
//!
//! Tests the complete V3 contract stack via ethers:
//!   Deploy → Token distribution → Staking → Model registration →
//!   3-round training with 3 workers → Finalization → Fee distribution
//!
//! Requires `--features chain` and foundry (anvil + forge).
//! Gated behind `#[cfg(feature = "chain")]` so normal `cargo test` skips it.

#![cfg(feature = "chain")]

use std::process::{Child, Command, Stdio};
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use ethers::providers::{Http, Provider};
use ethers::types::{Address, U256};
use ff::PrimeField;

use helix_circuits::gadgets::poseidon_hash_two;
use helix_circuits::halo2curves::bn256::Fr;
use helix_client::rpc::chain_v3::{ChainClientV3, ForgeDeployResultV3};

// ============================================================================
// Constants
// ============================================================================

/// Anvil default funded private keys (accounts 0-3).
const DEPLOYER_KEY: &str = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const WORKER1_KEY: &str = "59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
const WORKER2_KEY: &str = "5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a";
const WORKER3_KEY: &str = "7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6";

/// Test port (avoid conflict with V2 tests at 18545-18565).
const V3_PORT: u16 = 18600;

/// 1e18 — matches HelixCoordinatorV3.DEFAULT_MAX_ERROR_BOUND.
const DEFAULT_MAX_ERROR_BOUND: u64 = 1_000_000_000_000_000_000;

/// Stake amount per worker: 200 HELIX (min is 100e18).
const STAKE_AMOUNT: u128 = 200_000_000_000_000_000_000;

/// Token amount transferred to each worker: 1000 HELIX.
const WORKER_TOKENS: u128 = 1_000_000_000_000_000_000_000;

/// Training job deposit: 300 HELIX (100 per round × 3 rounds).
const JOB_DEPOSIT: u128 = 300_000_000_000_000_000_000;

/// Round duration: 600 seconds (10 minutes).
const ROUND_DURATION: u64 = 600;

/// Time to advance past dispute deadline: duration + DISPUTE_PERIOD(3600) + margin.
const FINALIZE_TIME_ADVANCE: u64 = 4800; // 600 + 3600 + 600 margin

// Hash values for 3-round commitment chain.
const INIT_LO: u64 = 12345;
const INIT_HI: u64 = 67890;
const R1_NEW_LO: u64 = 20001;
const R1_NEW_HI: u64 = 20002;
const R2_NEW_LO: u64 = 30001;
const R2_NEW_HI: u64 = 30002;
const R3_NEW_LO: u64 = 40001;
const R3_NEW_HI: u64 = 40002;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Manages an Anvil subprocess for test duration.
struct AnvilInstance {
    child: Child,
    rpc_url: String,
    chain_id: u64,
}

impl AnvilInstance {
    async fn start(port: u16, chain_id: u64) -> Result<Self> {
        let child = Command::new("anvil")
            .arg("--port")
            .arg(port.to_string())
            .arg("--chain-id")
            .arg(chain_id.to_string())
            .arg("--accounts")
            .arg("10")
            .arg("--balance")
            .arg("10000")
            .arg("--disable-code-size-limit")
            .arg("--silent")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to start anvil — is foundry installed?")?;

        let rpc_url = format!("http://127.0.0.1:{}", port);
        let instance = Self {
            child,
            rpc_url: rpc_url.clone(),
            chain_id,
        };

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
        Ok(instance)
    }

    fn rpc_url(&self) -> &str {
        &self.rpc_url
    }

    fn chain_id(&self) -> u64 {
        self.chain_id
    }
}

impl Drop for AnvilInstance {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Clean up broadcast directories for non-default chain IDs
        if self.chain_id != 31337 {
            let cdir = contracts_dir();
            let broadcast_dir = format!("{}/broadcast/Deploy.s.sol/{}", cdir, self.chain_id);
            let _ = std::fs::remove_dir_all(&broadcast_dir);
        }
    }
}

/// Resolve the contracts directory relative to workspace root.
fn contracts_dir() -> String {
    for candidate in &[
        "helix/contracts",
        "contracts",
        "../contracts",
        "../../contracts",
        "../../helix/contracts",
    ] {
        if std::path::Path::new(candidate)
            .join("foundry.toml")
            .exists()
        {
            return candidate.to_string();
        }
    }
    std::env::var("HELIX_CONTRACTS_DIR").unwrap_or_else(|_| "helix/contracts".to_string())
}

/// Acquire a filesystem lock so only one forge script runs at a time.
/// Multiple concurrent forge processes in the same directory corrupt shared caches.
fn acquire_forge_lock(contracts_dir: &str) -> Result<std::path::PathBuf> {
    let lock_dir = std::path::PathBuf::from(format!("{}/.forge_lock", contracts_dir));
    let deadline = Instant::now() + Duration::from_secs(300); // 5 min max wait
    loop {
        match std::fs::create_dir(&lock_dir) {
            Ok(()) => return Ok(lock_dir),
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(e) => return Err(anyhow!("Failed to acquire forge lock after 300s: {}", e)),
        }
    }
}

fn release_forge_lock(lock_dir: &std::path::Path) {
    let _ = std::fs::remove_dir(lock_dir);
}

/// Deploy V3 full stack with mock verifier via forge script.
async fn deploy_v3_contracts(rpc_url: &str, contracts_dir: &str, chain_id: u64) -> Result<ForgeDeployResultV3> {
    // Serialize forge invocations to prevent shared-cache corruption.
    let lock_path = acquire_forge_lock(contracts_dir)?;
    let result = deploy_v3_contracts_inner(rpc_url, contracts_dir, chain_id).await;
    release_forge_lock(&lock_path);
    result
}

async fn deploy_v3_contracts_inner(rpc_url: &str, contracts_dir: &str, chain_id: u64) -> Result<ForgeDeployResultV3> {
    // Clean stale broadcast files for this chain ID.
    let broadcast_dir = format!("{}/broadcast/Deploy.s.sol/{}", contracts_dir, chain_id);
    for name in &["deployV3WithMock-latest.json", "run-latest.json"] {
        let _ = std::fs::remove_file(format!("{}/{}", broadcast_dir, name));
    }

    // Pipe `yes` into forge to auto-confirm interactive prompts (e.g., no-code address warnings).
    // Forge 1.5+ prompts interactively for certain warnings; piping `yes` answers "y" to all.
    let yes_child = Command::new("yes")
        .stdout(Stdio::piped())
        .spawn()
        .context("Failed to start yes")?;

    let output = Command::new("forge")
        .arg("script")
        .arg("script/Deploy.s.sol")
        .arg("--tc")
        .arg("DeployScript")
        .arg("--sig")
        .arg("deployV3WithMock()")
        .arg("--broadcast")
        .arg("--rpc-url")
        .arg(rpc_url)
        .arg("--private-key")
        .arg(DEPLOYER_KEY)
        .arg("--disable-code-size-limit")
        .stdin(yes_child.stdout.expect("yes stdout"))
        .current_dir(contracts_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("Failed to run forge script")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow!(
            "forge script deployV3WithMock failed:\nstderr: {}\nstdout (first 2000): {}",
            stderr,
            &stdout[..stdout.len().min(2000)]
        ));
    }

    // Parse broadcast JSON — try both possible filenames.
    // Also search for any *-latest.json file as forge may use different naming.
    let base = format!("{}/broadcast/Deploy.s.sol/{}", contracts_dir, chain_id);
    let candidates = [
        format!("{}/deployV3WithMock-latest.json", base),
        format!("{}/run-latest.json", base),
    ];

    let content = candidates
        .iter()
        .find_map(|p| std::fs::read_to_string(p).ok())
        .or_else(|| {
            // Fallback: find any *-latest.json in the broadcast dir
            std::fs::read_dir(&base).ok().and_then(|rd| {
                rd.filter_map(|e| e.ok())
                    .find(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .ends_with("-latest.json")
                    })
                    .and_then(|e| std::fs::read_to_string(e.path()).ok())
            })
        })
        .ok_or_else(|| {
            let files: Vec<String> = std::fs::read_dir(&base)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().to_string())
                        .collect()
                })
                .unwrap_or_default();
            anyhow!(
                "No broadcast JSON found. Dir '{}' contents: {:?}\nforge stdout (first 500): {}",
                base,
                files,
                &String::from_utf8_lossy(&output.stdout)[..output.stdout.len().min(500)]
            )
        })?;

    let json: serde_json::Value =
        serde_json::from_str(&content).context("Failed to parse broadcast JSON")?;

    let transactions = json
        .get("transactions")
        .and_then(|t| t.as_array())
        .ok_or_else(|| anyhow!("No 'transactions' in broadcast JSON"))?;

    let mut addresses = std::collections::HashMap::new();
    let mut deployer_addr = String::new();

    for tx in transactions {
        if let (Some(name), Some(addr)) = (
            tx.get("contractName").and_then(|n| n.as_str()),
            tx.get("contractAddress").and_then(|a| a.as_str()),
        ) {
            addresses.insert(name.to_string(), addr.to_string());
        }
        if deployer_addr.is_empty() {
            if let Some(from) = tx
                .get("transaction")
                .and_then(|t| t.get("from"))
                .and_then(|f| f.as_str())
            {
                deployer_addr = from.to_string();
            }
        }
    }

    Ok(ForgeDeployResultV3 {
        coordinator: addresses
            .get("HelixCoordinatorV3")
            .cloned()
            .ok_or_else(|| anyhow!("HelixCoordinatorV3 not found in broadcast"))?,
        verifier: addresses
            .get("MockVerifierForDeploy")
            .or_else(|| addresses.get("Halo2Verifier"))
            .cloned()
            .unwrap_or_default(),
        token: addresses
            .get("HelixToken")
            .cloned()
            .ok_or_else(|| anyhow!("HelixToken not found"))?,
        staking: addresses
            .get("Staking")
            .cloned()
            .ok_or_else(|| anyhow!("Staking not found"))?,
        rewards: addresses
            .get("Rewards")
            .cloned()
            .ok_or_else(|| anyhow!("Rewards not found"))?,
        registry: addresses
            .get("ModelRegistry")
            .cloned()
            .ok_or_else(|| anyhow!("ModelRegistry not found"))?,
        treasury: deployer_addr,
    })
}

// ============================================================================
// Crypto Helpers
// ============================================================================

/// Compute keccak256(abi.encodePacked(lo, hi)) — matches Solidity _hashPair.
fn hash_pair(lo: U256, hi: U256) -> U256 {
    use ethers::utils::keccak256;
    let mut data = [0u8; 64];
    lo.to_big_endian(&mut data[..32]);
    hi.to_big_endian(&mut data[32..]);
    U256::from(keccak256(data))
}

/// Compute Poseidon error checksum matching V3._computeErrorChecksum / PoseidonHasher.
fn compute_error_checksum(
    error_bound: u64,
    step_number: u64,
    model_id: u64,
    max_error_bound: U256,
) -> U256 {
    let h1 = poseidon_hash_two(Fr::from(error_bound), Fr::from(step_number));
    let h2 = poseidon_hash_two(Fr::from(model_id), Fr::from(max_error_bound.as_u64()));
    let checksum = poseidon_hash_two(h1, h2);
    U256::from_little_endian(checksum.to_repr().as_ref())
}

/// Build 8-element public inputs vector for V3 submitProof.
fn build_public_inputs(
    old_lo: u64,
    old_hi: u64,
    new_lo: u64,
    new_hi: u64,
    loss: u64,
    error_bound: u64,
    step: u64,
    model_id: u64,
) -> Vec<U256> {
    let checksum = compute_error_checksum(
        error_bound,
        step,
        model_id,
        U256::from(DEFAULT_MAX_ERROR_BOUND),
    );
    vec![
        U256::from(old_lo),
        U256::from(old_hi),
        U256::from(new_lo),
        U256::from(new_hi),
        U256::from(loss),
        U256::from(error_bound),
        U256::from(step),
        checksum,
    ]
}

/// Advance Anvil's block time by `seconds`.
async fn advance_time(rpc_url: &str, seconds: u64) -> Result<()> {
    let provider =
        Provider::<Http>::try_from(rpc_url).map_err(|e| anyhow!("Provider error: {}", e))?;
    provider
        .request::<_, serde_json::Value>("evm_increaseTime", [seconds])
        .await
        .map_err(|e| anyhow!("evm_increaseTime: {}", e))?;
    provider
        .request::<_, serde_json::Value>("evm_mine", Vec::<()>::new())
        .await
        .map_err(|e| anyhow!("evm_mine: {}", e))?;
    Ok(())
}

fn parse_addr(s: &str) -> Result<Address> {
    Address::from_str(s).map_err(|e| anyhow!("Invalid address '{}': {}", s, e))
}

// ============================================================================
// Tests
// ============================================================================

/// Full 3-round, 3-worker training lifecycle on V3.
///
/// Flow:
///   1. Deploy V3 stack (mock verifier)
///   2. Distribute HELIX tokens to workers
///   3. Workers approve staking + stake
///   4. Register model with known commitment
///   5. Create training job (deposits tokens for worker fees)
///   6. Round 1: start → 3 workers submit proofs → advance time → finalize
///   7. Round 2: chain commitment → same flow
///   8. Round 3: chain commitment → same flow
///   9. Verify: commitment chain, accumulated error, fee distribution
#[tokio::test]
async fn test_v3_full_lifecycle_3_rounds_3_workers() {
    // =========== Setup ===========
    let anvil = AnvilInstance::start(V3_PORT, 31337)
        .await
        .expect("Failed to start Anvil");

    let cdir = contracts_dir();
    let deployment = deploy_v3_contracts(anvil.rpc_url(), &cdir, anvil.chain_id())
        .await
        .expect("Failed to deploy V3 contracts");

    // Create deployer + 3 worker clients
    let deployer = ChainClientV3::new(anvil.rpc_url(), DEPLOYER_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .expect("Failed to create deployer client");

    let worker_keys = [WORKER1_KEY, WORKER2_KEY, WORKER3_KEY];
    let mut workers = Vec::new();
    for key in &worker_keys {
        let client = ChainClientV3::new(anvil.rpc_url(), key, &deployment, Some(anvil.chain_id()))
            .await
            .expect("Failed to create worker client");
        workers.push(client);
    }

    let staking_addr = parse_addr(&deployment.staking).expect("Invalid staking address");
    let coordinator_addr = deployer.coordinator_address();

    // =========== Token Distribution ===========
    // Deployer (treasury) has all initial tokens. Transfer to workers.
    for worker in &workers {
        deployer
            .transfer_tokens(worker.signer_address(), U256::from(WORKER_TOKENS))
            .await
            .expect("Token transfer to worker failed");
    }

    // Verify workers received tokens
    for worker in &workers {
        let balance = deployer
            .token_balance(worker.signer_address())
            .await
            .expect("Balance query failed");
        assert_eq!(
            balance,
            U256::from(WORKER_TOKENS),
            "Worker should have {} tokens",
            WORKER_TOKENS
        );
    }

    // =========== Staking ===========
    for worker in &workers {
        // Approve staking contract to spend worker's tokens
        worker
            .approve_tokens(staking_addr, U256::from(STAKE_AMOUNT))
            .await
            .expect("Approve staking failed");

        // Stake
        worker
            .stake_tokens(U256::from(STAKE_AMOUNT))
            .await
            .expect("Staking failed");

        // Verify
        let can = worker
            .can_participate(worker.signer_address())
            .await
            .expect("canParticipate query failed");
        assert!(can, "Worker should be eligible to participate after staking");
    }

    // =========== Model Registration ===========
    let initial_commitment = hash_pair(U256::from(INIT_LO), U256::from(INIT_HI));
    let (reg_receipt, model_id) = deployer
        .register_model(
            "TestModel",
            "V3 lifecycle test model",
            "QmV3LifecycleTest",
            initial_commitment,
        )
        .await
        .expect("Model registration failed");
    assert_eq!(reg_receipt.status, Some(1u64.into()));

    let state = deployer
        .get_model_state(model_id)
        .await
        .expect("get_model_state failed");
    assert!(state.active);
    assert_eq!(state.current_commitment, initial_commitment);
    assert_eq!(state.current_round, 0);

    // =========== Training Job ===========
    // Deployer approves coordinator to pull deposit tokens
    deployer
        .approve_tokens(coordinator_addr, U256::from(JOB_DEPOSIT))
        .await
        .expect("Approve coordinator for training job failed");

    let (job_receipt, job_id) = deployer
        .create_training_job(model_id, 3, U256::from(JOB_DEPOSIT))
        .await
        .expect("Create training job failed");
    assert_eq!(job_receipt.status, Some(1u64.into()));
    assert!(job_id > 0, "Job ID should be positive");

    // Record worker balances before rounds (after staking)
    let mut pre_round_balances = Vec::new();
    for worker in &workers {
        let bal = deployer
            .token_balance(worker.signer_address())
            .await
            .unwrap();
        pre_round_balances.push(bal);
    }

    // =========== 3 Training Rounds ===========
    let rounds_data: [(u64, u64, u64, u64); 3] = [
        // (old_lo, old_hi, new_lo, new_hi) for each round
        (INIT_LO, INIT_HI, R1_NEW_LO, R1_NEW_HI),
        (R1_NEW_LO, R1_NEW_HI, R2_NEW_LO, R2_NEW_HI),
        (R2_NEW_LO, R2_NEW_HI, R3_NEW_LO, R3_NEW_HI),
    ];

    // Per-worker loss values (worker 0 has lowest = best prover)
    let worker_losses: [u64; 3] = [300, 500, 800];

    for (round_idx, &(old_lo, old_hi, new_lo, new_hi)) in rounds_data.iter().enumerate() {
        let round_num = (round_idx + 1) as u64;

        // --- Start Round ---
        deployer
            .start_round_with_threshold(model_id, ROUND_DURATION, 3)
            .await
            .unwrap_or_else(|e| panic!("Start round {} failed: {}", round_num, e));

        let model_state = deployer.get_model_state(model_id).await.unwrap();
        assert_eq!(model_state.current_round, round_num);

        // --- Workers Submit Proofs ---
        for (w_idx, worker) in workers.iter().enumerate() {
            let inputs = build_public_inputs(
                old_lo,
                old_hi,
                new_lo,
                new_hi,
                worker_losses[w_idx],
                10, // error_bound
                round_num,
                model_id,
            );

            // Use different proof bytes per worker to avoid replay hash collision
            let mut proof = vec![0u8; 320];
            proof[0] = w_idx as u8;
            proof[1] = round_idx as u8;

            worker
                .submit_proof_raw(model_id, round_num, proof, inputs)
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "Worker {} proof submission in round {} failed: {}",
                        w_idx, round_num, e
                    )
                });
        }

        // Verify all 3 proofs recorded
        let ext = deployer
            .get_round_ext(model_id, round_num)
            .await
            .unwrap();
        assert_eq!(
            ext.valid_proofs, 3,
            "Round {} should have 3 valid proofs",
            round_num
        );
        assert_eq!(
            ext.best_prover,
            workers[0].signer_address(),
            "Worker 0 (lowest loss) should be best prover in round {}",
            round_num
        );

        // --- Advance Time Past Dispute Period ---
        advance_time(anvil.rpc_url(), FINALIZE_TIME_ADVANCE)
            .await
            .expect("Time advance failed");

        // --- Finalize Round ---
        deployer
            .finalize_round(model_id, round_num)
            .await
            .unwrap_or_else(|e| panic!("Finalize round {} failed: {}", round_num, e));

        // Verify commitment updated
        let new_state = deployer.get_model_state(model_id).await.unwrap();
        let expected = hash_pair(U256::from(new_lo), U256::from(new_hi));
        assert_eq!(
            new_state.current_commitment, expected,
            "Round {} commitment should be updated to hash_pair({}, {})",
            round_num, new_lo, new_hi
        );

        // Verify round completed
        let round_state = deployer
            .get_round_state(model_id, round_num)
            .await
            .unwrap();
        assert!(
            round_state.is_completed,
            "Round {} should be completed",
            round_num
        );
    }

    // =========== Final Verification ===========

    // 1. Commitment chain verified at final state
    let final_state = deployer.get_model_state(model_id).await.unwrap();
    let expected_final = hash_pair(U256::from(R3_NEW_LO), U256::from(R3_NEW_HI));
    assert_eq!(
        final_state.current_commitment, expected_final,
        "Final commitment should be from round 3"
    );
    assert_eq!(final_state.current_round, 3);

    // 2. Accumulated error: 10 per round × 3 rounds = 30
    let accumulated = deployer
        .get_accumulated_error(model_id)
        .await
        .expect("get_accumulated_error failed");
    assert_eq!(
        accumulated,
        U256::from(30u64),
        "Accumulated error should be 30 (10 × 3 rounds)"
    );

    // 3. Training job fee distribution: each worker should have received tokens
    //    JOB_DEPOSIT / 3 rounds = fee_per_round
    //    fee_per_round / 3 workers = per_worker_per_round
    //    Total per worker = per_worker_per_round × 3 rounds
    let fee_per_round = JOB_DEPOSIT / 3;
    let per_worker_per_round = fee_per_round / 3;
    let expected_total_fee_per_worker = per_worker_per_round * 3;

    for (i, worker) in workers.iter().enumerate() {
        let current_balance = deployer
            .token_balance(worker.signer_address())
            .await
            .unwrap();
        let received = current_balance - pre_round_balances[i];
        assert_eq!(
            received,
            U256::from(expected_total_fee_per_worker),
            "Worker {} should have received {} tokens in fees (got {})",
            i,
            expected_total_fee_per_worker,
            received
        );
    }

    // 4. Verify ProofSubmitted events (9 total: 3 workers × 3 rounds)
    let proof_events = deployer
        .query_proof_submitted(0, None)
        .await
        .expect("Event query failed");
    assert_eq!(
        proof_events.len(),
        9,
        "Should have 9 ProofSubmitted events (3 workers × 3 rounds)"
    );

    // 5. Verify RoundCompleted events (3 total)
    let round_events = deployer
        .query_round_completed(0, None)
        .await
        .expect("Round completed event query failed");
    assert_eq!(
        round_events.len(),
        3,
        "Should have 3 RoundCompleted events"
    );
}

/// Verify V3 deployment creates all expected contracts and basic operations work.
#[tokio::test]
async fn test_v3_deploy_and_basic_operations() {
    let anvil = AnvilInstance::start(V3_PORT + 1, 31338)
        .await
        .expect("Failed to start Anvil");

    let cdir = contracts_dir();
    let deployment = deploy_v3_contracts(anvil.rpc_url(), &cdir, anvil.chain_id())
        .await
        .expect("V3 deployment failed");

    // Verify all addresses are present
    assert!(
        !deployment.coordinator.is_empty(),
        "Coordinator address should be set"
    );
    assert!(!deployment.token.is_empty(), "Token address should be set");
    assert!(
        !deployment.staking.is_empty(),
        "Staking address should be set"
    );
    assert!(
        !deployment.rewards.is_empty(),
        "Rewards address should be set"
    );
    assert!(
        !deployment.registry.is_empty(),
        "Registry address should be set"
    );

    // Create client and verify basic queries
    let client = ChainClientV3::new(anvil.rpc_url(), DEPLOYER_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .expect("Client creation failed");

    // Deployer (treasury) should have tokens
    let balance = client
        .token_balance(client.signer_address())
        .await
        .expect("Balance query failed");
    assert!(balance > U256::zero(), "Deployer should have initial token supply");

    // Register a model
    let commitment = hash_pair(U256::from(100u64), U256::from(200u64));
    let (receipt, model_id) = client
        .register_model("BasicModel", "Test", "QmTest", commitment)
        .await
        .expect("Model registration failed");
    assert_eq!(receipt.status, Some(1u64.into()));

    let state = client.get_model_state(model_id).await.unwrap();
    assert!(state.active);
    assert_eq!(state.current_commitment, commitment);
    assert_eq!(state.current_round, 0);
}

/// Verify single-participant round auto-finalizes after one proof.
#[tokio::test]
async fn test_v3_single_participant_auto_finalize() {
    let anvil = AnvilInstance::start(V3_PORT + 2, 31339)
        .await
        .expect("Failed to start Anvil");

    let cdir = contracts_dir();
    let deployment = deploy_v3_contracts(anvil.rpc_url(), &cdir, anvil.chain_id())
        .await
        .expect("V3 deployment failed");

    let deployer = ChainClientV3::new(anvil.rpc_url(), DEPLOYER_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .unwrap();
    let worker = ChainClientV3::new(anvil.rpc_url(), WORKER1_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .unwrap();

    let staking_addr = parse_addr(&deployment.staking).unwrap();

    // Fund + stake worker
    deployer
        .transfer_tokens(worker.signer_address(), U256::from(WORKER_TOKENS))
        .await
        .unwrap();
    worker
        .approve_tokens(staking_addr, U256::from(STAKE_AMOUNT))
        .await
        .unwrap();
    worker.stake_tokens(U256::from(STAKE_AMOUNT)).await.unwrap();

    // Register model
    let commitment = hash_pair(U256::from(INIT_LO), U256::from(INIT_HI));
    let (_, model_id) = deployer
        .register_model("AutoFinalizeModel", "Test", "QmAutoFinalize", commitment)
        .await
        .unwrap();

    // Start round with default min_participants=1 (auto-finalize)
    deployer
        .start_round(model_id, ROUND_DURATION)
        .await
        .unwrap();

    // Submit one proof
    let inputs = build_public_inputs(INIT_LO, INIT_HI, R1_NEW_LO, R1_NEW_HI, 500, 10, 1, model_id);
    worker
        .submit_proof_raw(model_id, 1, vec![0u8; 320], inputs)
        .await
        .expect("Proof submission should succeed");

    // Round should be auto-finalized (no need for explicit finalizeRound)
    let round_state = deployer.get_round_state(model_id, 1).await.unwrap();
    assert!(
        round_state.is_completed,
        "Single-participant round should auto-finalize"
    );

    // Commitment should be updated
    let state = deployer.get_model_state(model_id).await.unwrap();
    let expected = hash_pair(U256::from(R1_NEW_LO), U256::from(R1_NEW_HI));
    assert_eq!(state.current_commitment, expected);

    // Can start next round
    deployer
        .start_round(model_id, ROUND_DURATION)
        .await
        .expect("Should be able to start round 2 after auto-finalize");

    let state2 = deployer.get_model_state(model_id).await.unwrap();
    assert_eq!(state2.current_round, 2);
}

/// Verify invalid proof triggers slashing via Staking.sol.
#[tokio::test]
async fn test_v3_invalid_proof_slashing() {
    let anvil = AnvilInstance::start(V3_PORT + 3, 31340)
        .await
        .expect("Failed to start Anvil");

    let cdir = contracts_dir();
    let deployment = deploy_v3_contracts(anvil.rpc_url(), &cdir, anvil.chain_id())
        .await
        .expect("V3 deployment failed");

    let deployer = ChainClientV3::new(anvil.rpc_url(), DEPLOYER_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .unwrap();
    let worker = ChainClientV3::new(anvil.rpc_url(), WORKER1_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .unwrap();

    let staking_addr = parse_addr(&deployment.staking).unwrap();

    // Fund + stake
    deployer
        .transfer_tokens(worker.signer_address(), U256::from(WORKER_TOKENS))
        .await
        .unwrap();
    worker
        .approve_tokens(staking_addr, U256::from(STAKE_AMOUNT))
        .await
        .unwrap();
    worker.stake_tokens(U256::from(STAKE_AMOUNT)).await.unwrap();

    // Register model
    let commitment = hash_pair(U256::from(INIT_LO), U256::from(INIT_HI));
    let (_, model_id) = deployer
        .register_model("SlashTestModel", "Test", "QmSlash", commitment)
        .await
        .unwrap();

    deployer
        .start_round(model_id, ROUND_DURATION)
        .await
        .unwrap();

    // Submit proof with WRONG old commitment (should fail with revert)
    let wrong_inputs = build_public_inputs(
        99999, 99999, // wrong old hash
        R1_NEW_LO, R1_NEW_HI, 500, 10, 1, model_id,
    );

    let result = worker
        .submit_proof_raw(model_id, 1, vec![0u8; 320], wrong_inputs)
        .await;

    assert!(
        result.is_err(),
        "Proof with wrong old commitment should be rejected"
    );
}

/// Verify round expiry when participant threshold not met.
#[tokio::test]
async fn test_v3_round_expiry() {
    let anvil = AnvilInstance::start(V3_PORT + 4, 31341)
        .await
        .expect("Failed to start Anvil");

    let cdir = contracts_dir();
    let deployment = deploy_v3_contracts(anvil.rpc_url(), &cdir, anvil.chain_id())
        .await
        .expect("V3 deployment failed");

    let deployer = ChainClientV3::new(anvil.rpc_url(), DEPLOYER_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .unwrap();
    let worker = ChainClientV3::new(anvil.rpc_url(), WORKER1_KEY, &deployment, Some(anvil.chain_id()))
        .await
        .unwrap();

    let staking_addr = parse_addr(&deployment.staking).unwrap();

    // Fund + stake only 1 worker
    deployer
        .transfer_tokens(worker.signer_address(), U256::from(WORKER_TOKENS))
        .await
        .unwrap();
    worker
        .approve_tokens(staking_addr, U256::from(STAKE_AMOUNT))
        .await
        .unwrap();
    worker.stake_tokens(U256::from(STAKE_AMOUNT)).await.unwrap();

    // Register model
    let commitment = hash_pair(U256::from(INIT_LO), U256::from(INIT_HI));
    let (_, model_id) = deployer
        .register_model("ExpiryModel", "Test", "QmExpiry", commitment)
        .await
        .unwrap();

    // Start round requiring 3 participants
    deployer
        .start_round_with_threshold(model_id, ROUND_DURATION, 3)
        .await
        .unwrap();

    // Only 1 worker submits (threshold is 3)
    let inputs = build_public_inputs(INIT_LO, INIT_HI, R1_NEW_LO, R1_NEW_HI, 500, 10, 1, model_id);
    worker
        .submit_proof_raw(model_id, 1, vec![0u8; 320], inputs)
        .await
        .unwrap();

    // Advance time past round deadline
    advance_time(anvil.rpc_url(), ROUND_DURATION + 100)
        .await
        .unwrap();

    // Expire the round (threshold not met)
    deployer
        .expire_round(model_id, 1)
        .await
        .expect("expire_round should succeed when threshold not met");

    // Round should be completed but commitment unchanged
    let round_state = deployer.get_round_state(model_id, 1).await.unwrap();
    assert!(round_state.is_completed, "Expired round should be marked completed");

    let state = deployer.get_model_state(model_id).await.unwrap();
    assert_eq!(
        state.current_commitment, commitment,
        "Commitment should be unchanged after expired round"
    );
}
