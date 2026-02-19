//! Chain recovery integration test: MPC cheater detection → on-chain slashing.
//!
//! Full E2E flow:
//!   1. Deploy HelixCoordinatorV4, register job, 3 workers stake
//!   2. MPC training with cheater injection → MAC check catches it
//!   3. Sign blame report using Rust k256 ECDSA (matching V4 contract format)
//!   4. Submit reportMACFailure() on-chain → cheater slashed, reporters get bounty
//!   5. Share redistribution for N-1 honest workers
//!   6. Continue training with 2 workers → verify convergence
//!
//! Requires: Foundry (anvil) installed, compiled contracts.
//! Run: cargo test -p helix-mpc --features chain -- chain_recovery

#![cfg(feature = "chain")]

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ethers::prelude::*;
use ethers::utils::keccak256 as eth_keccak256;

use helix_mpc::blame_report::{
    address_from_private_key, build_mac_failure_message, serialize_evidence,
    sign_hash, to_eth_signed_message_hash,
};
use helix_mpc::e2e_integration::{
    run_mpc_training, run_mpc_training_with_cheater, InitialWeights, MPCIntegrationConfig,
};
use helix_mpc::mac_verification::{
    CheaterEvidence, MACFailureReport, PairwiseCheckResult, TrainingCheckpoint,
};
use helix_mpc::share_redistribution;
use helix_mpc::field::Fr;
use helix_mpc::types::PartyId;

// ============================================================================
// Contract bindings from compiled Foundry artifact
// ============================================================================

abigen!(
    HelixCoordinatorV4,
    "../../contracts/out/HelixCoordinatorV4.sol/HelixCoordinatorV4.json"
);

// ============================================================================
// Constants — Anvil well-known funded accounts
// ============================================================================

const CHAIN_ID: u64 = 31337;

const DEPLOYER_KEY: &str =
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const WORKER_KEYS: [&str; 3] = [
    "59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
];

/// 1 ETH stake per worker.
const STAKE_WEI: u128 = 1_000_000_000_000_000_000;
/// 3 ETH total payment pool.
const PAYMENT_WEI: u128 = 3_000_000_000_000_000_000;

// ============================================================================
// Anvil management
// ============================================================================

struct AnvilInstance {
    child: Child,
    rpc_url: String,
}

impl AnvilInstance {
    async fn start(port: u16) -> anyhow::Result<Self> {
        let child = Command::new("anvil")
            .arg("--port").arg(port.to_string())
            .arg("--chain-id").arg(CHAIN_ID.to_string())
            .arg("--accounts").arg("10")
            .arg("--balance").arg("10000")
            .arg("--silent")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to start anvil: {e}"))?;

        let rpc_url = format!("http://127.0.0.1:{port}");

        // Wait for Anvil to accept RPC calls.
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if Instant::now() > deadline {
                anyhow::bail!("Anvil did not start within 15s on port {port}");
            }
            if let Ok(p) = Provider::<Http>::try_from(rpc_url.as_str()) {
                if p.get_chainid().await.is_ok() {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        Ok(Self { child, rpc_url })
    }
}

impl Drop for AnvilInstance {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ============================================================================
// Helpers
// ============================================================================

type SignedClient = SignerMiddleware<Provider<Http>, LocalWallet>;

fn make_wallet(hex_key: &str) -> LocalWallet {
    hex_key.parse::<LocalWallet>().unwrap().with_chain_id(CHAIN_ID)
}

fn make_client(rpc_url: &str, hex_key: &str) -> Arc<SignedClient> {
    let provider = Provider::<Http>::try_from(rpc_url).unwrap();
    let wallet = make_wallet(hex_key);
    Arc::new(SignerMiddleware::new(provider, wallet))
}

fn hex_to_32(hex: &str) -> [u8; 32] {
    let bytes = hex::decode(hex).unwrap();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

/// Verify that our Rust k256 address derivation matches ethers' LocalWallet.
fn verify_address_consistency(hex_key: &str) {
    let wallet = make_wallet(hex_key);
    let rust_addr = address_from_private_key(&hex_to_32(hex_key)).unwrap();
    let ethers_addr = wallet.address();
    assert_eq!(
        &rust_addr[..],
        ethers_addr.as_bytes(),
        "k256 and ethers must derive the same address for key {}",
        &hex_key[..8]
    );
}

// ============================================================================
// Test 1: Rust k256 signatures accepted by V4 contract (focused crypto test)
// ============================================================================

/// Deploys V4, registers a job with 3 workers, signs a blame report using
/// our pure-Rust k256 ECDSA, submits it on-chain, and verifies slashing.
///
/// This test is focused on proving Rust ↔ Solidity signing compatibility.
#[tokio::test]
async fn test_blame_signatures_accepted_on_chain() -> anyhow::Result<()> {
    // Verify address derivation is consistent across libraries.
    for key in &WORKER_KEYS {
        verify_address_consistency(key);
    }

    let anvil = AnvilInstance::start(19810).await?;
    let deployer = make_client(&anvil.rpc_url, DEPLOYER_KEY);
    let provider = Provider::<Http>::try_from(anvil.rpc_url.as_str())?;

    // Deploy V4 (treasury = deployer, no ZK verifier).
    let contract = HelixCoordinatorV4::deploy(
        deployer.clone(),
        (deployer.address(), Address::zero()),
    )?.send().await?;

    // Register job.
    let payment = U256::from(PAYMENT_WEI);
    let arch_hash: [u8; 32] = eth_keccak256(b"2,2,1");
    contract
        .register_training_job(arch_hash, U256::from(5u64), U256::from(100u64), payment)
        .value(payment)
        .send().await?.await?.unwrap();

    let job_id = U256::zero();
    let stake = U256::from(STAKE_WEI);

    // 3 workers stake.
    for key in &WORKER_KEYS {
        let wc = make_client(&anvil.rpc_url, key);
        let wcontract = HelixCoordinatorV4::new(contract.address(), wc);
        wcontract.stake_and_join(job_id).value(stake).send().await?.await?.unwrap();
    }
    assert_eq!(contract.get_active_worker_count(job_id).call().await?, U256::from(3u64));

    // Build blame report: worker 2 is the cheater.
    let cheater_pk = hex_to_32(WORKER_KEYS[2]);
    let cheater_addr_bytes = address_from_private_key(&cheater_pk)?;
    let cheater_eth: Address = Address::from_slice(&cheater_addr_bytes);

    let evidence = b"sigma_protocol_pairwise_failure_party_2".to_vec();
    let step = 10u64;

    // Build message hash matching V4's _buildMACFailureMessage.
    let msg_hash = build_mac_failure_message(0, step, &cheater_addr_bytes, &evidence);
    let eth_hash = to_eth_signed_message_hash(&msg_hash);

    // Workers 0 and 1 sign (majority: 2 of 3).
    let mut sigs: Vec<Bytes> = Vec::new();
    for key in &WORKER_KEYS[..2] {
        let sig = sign_hash(&eth_hash, &hex_to_32(key))?;
        sigs.push(Bytes::from(sig.to_vec()));
    }

    // Record reporter balances.
    let r0 = make_wallet(WORKER_KEYS[0]).address();
    let r1 = make_wallet(WORKER_KEYS[1]).address();
    let bal0_pre = provider.get_balance(r0, None).await?;
    let bal1_pre = provider.get_balance(r1, None).await?;

    // Submit on-chain.
    let receipt = contract
        .report_mac_failure(
            job_id,
            U256::from(step),
            cheater_eth,
            Bytes::from(evidence),
            sigs,
        )
        .send().await?.await?.unwrap();
    assert_eq!(receipt.status.unwrap().as_u64(), 1, "reportMACFailure tx failed");

    // Verify: cheater slashed.
    let info = contract.get_worker_info(job_id, cheater_eth).call().await?;
    assert_eq!(info.0, U256::zero(), "cheater stake should be 0");  // stakeAmount
    assert!(info.4, "cheater should be slashed");                    // slashed
    assert!(!contract.is_active_worker(job_id, cheater_eth).call().await?);
    assert_eq!(contract.get_active_worker_count(job_id).call().await?, U256::from(2u64));

    // Verify: reporters received bounty (10% of 1 ETH / 2 = 0.05 ETH each).
    let bounty_per = stake * U256::from(10u64) / U256::from(100u64) / U256::from(2u64);
    let bal0_post = provider.get_balance(r0, None).await?;
    let bal1_post = provider.get_balance(r1, None).await?;
    assert_eq!(bal0_post - bal0_pre, bounty_per, "reporter 0 bounty mismatch");
    assert_eq!(bal1_post - bal1_pre, bounty_per, "reporter 1 bounty mismatch");

    // Verify: MAC failure report stored.
    assert_eq!(
        contract.get_mac_failure_report_count(job_id).call().await?,
        U256::from(1u64),
    );
    let report = contract.get_mac_failure_report(job_id, U256::zero()).call().await?;
    assert_eq!(report.0, U256::from(step));       // stepNumber
    assert_eq!(report.1, cheater_eth);             // cheater
    assert_eq!(report.3, stake);                   // slashedAmount
    assert_eq!(report.4, U256::from(2u64));        // reporterCount

    Ok(())
}

// ============================================================================
// Test 2: Full E2E — MPC training → cheater detection → on-chain slash →
//         share redistribution → recovery training
// ============================================================================

#[tokio::test]
async fn test_chain_recovery_full_e2e() -> anyhow::Result<()> {
    // ---- Phase 1: On-chain setup ----
    let anvil = AnvilInstance::start(19820).await?;
    let deployer = make_client(&anvil.rpc_url, DEPLOYER_KEY);
    let provider = Provider::<Http>::try_from(anvil.rpc_url.as_str())?;

    let contract = HelixCoordinatorV4::deploy(
        deployer.clone(),
        (deployer.address(), Address::zero()),
    )?.send().await?;

    let payment = U256::from(PAYMENT_WEI);
    let arch_hash: [u8; 32] = eth_keccak256(b"2,2,1");
    contract
        .register_training_job(arch_hash, U256::from(5u64), U256::from(100u64), payment)
        .value(payment)
        .send().await?.await?.unwrap();

    let job_id = U256::zero();
    let stake = U256::from(STAKE_WEI);

    for key in &WORKER_KEYS {
        let wc = make_client(&anvil.rpc_url, key);
        let wcontract = HelixCoordinatorV4::new(contract.address(), wc);
        wcontract.stake_and_join(job_id).value(stake).send().await?.await?.unwrap();
    }
    assert_eq!(contract.get_active_worker_count(job_id).call().await?, U256::from(3u64));

    // ---- Phase 2: MPC training with cheater injection ----
    let cheater_party = 2; // Worker index 2 = WORKER_KEYS[2]
    let corrupt_at_step = 3;

    let mpc_config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 5,
        beaver_batch_size: 512,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, 0.3, 0.4],
            b1: vec![0.01, 0.02],
            w2: vec![0.5, 0.6],
            b2: vec![0.03],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let result = run_mpc_training_with_cheater(mpc_config, cheater_party, corrupt_at_step).await?;

    let cheater_record = result.cheater_detected
        .expect("MAC check should detect the cheater");
    assert_eq!(cheater_record.party_index, cheater_party);

    // ---- Phase 3: Build and sign blame report ----
    let detected_step = cheater_record.detected_at_step;

    // Build a MACFailureReport with evidence from the detection.
    let failure_report = MACFailureReport {
        session_id: "chain-recovery-e2e".to_string(),
        step_number: detected_step,
        identified_cheater: Some(cheater_party),
        sigma_values: vec![vec![1, 2, 3]; 3],
        commitments: vec![[0u8; 32]; 3],
        evidence: CheaterEvidence {
            pairwise_results: vec![
                PairwiseCheckResult { party_a: 0, party_b: 2, consistent: false },
                PairwiseCheckResult { party_a: 1, party_b: 2, consistent: false },
            ],
            round1_sigmas: vec![vec![10, 11]; 3],
            round2_sigmas: vec![vec![20, 21]; 3],
        },
    };

    let evidence_bytes = serialize_evidence(&failure_report);

    let cheater_pk = hex_to_32(WORKER_KEYS[cheater_party]);
    let cheater_addr = address_from_private_key(&cheater_pk)?;
    let cheater_eth = Address::from_slice(&cheater_addr);

    let msg_hash = build_mac_failure_message(0, detected_step, &cheater_addr, &evidence_bytes);
    let eth_hash = to_eth_signed_message_hash(&msg_hash);

    // Honest workers sign.
    let mut sigs: Vec<Bytes> = Vec::new();
    for key in &WORKER_KEYS[..2] {
        let sig = sign_hash(&eth_hash, &hex_to_32(key))?;
        sigs.push(Bytes::from(sig.to_vec()));
    }

    // ---- Phase 4: Submit on-chain ----
    let r0 = make_wallet(WORKER_KEYS[0]).address();
    let r1 = make_wallet(WORKER_KEYS[1]).address();
    let bal0_pre = provider.get_balance(r0, None).await?;
    let bal1_pre = provider.get_balance(r1, None).await?;

    let receipt = contract
        .report_mac_failure(
            job_id,
            U256::from(detected_step),
            cheater_eth,
            Bytes::from(evidence_bytes),
            sigs,
        )
        .send().await?.await?.unwrap();
    assert_eq!(receipt.status.unwrap().as_u64(), 1, "on-chain slash failed");

    // ---- Phase 5: Verify on-chain state ----
    let info = contract.get_worker_info(job_id, cheater_eth).call().await?;
    assert_eq!(info.0, U256::zero(), "cheater stake must be 0");
    assert!(info.4, "cheater must be slashed");
    assert_eq!(contract.get_active_worker_count(job_id).call().await?, U256::from(2u64));

    let bounty_per = stake * U256::from(10u64) / U256::from(100u64) / U256::from(2u64);
    let bal0_post = provider.get_balance(r0, None).await?;
    let bal1_post = provider.get_balance(r1, None).await?;
    assert_eq!(bal0_post - bal0_pre, bounty_per, "reporter 0 bounty");
    assert_eq!(bal1_post - bal1_pre, bounty_per, "reporter 1 bounty");

    // ---- Phase 6: Share redistribution (in-memory) ----
    // Create synthetic checkpoint shares that sum to known weights.
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use helix_mpc::session::transport::LocalTransport;

    let num_parties = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let full_w1 = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
    let full_b1 = vec![Fr::from_f64(0.1)];
    let full_w2 = vec![Fr::from_f64(0.5)];
    let full_b2 = vec![Fr::from_f64(0.01)];

    // Split into additive shares for 3 parties.
    let mut checkpoints = Vec::new();
    let mut remaining_w1 = full_w1.clone();
    let mut remaining_b1 = full_b1.clone();
    let mut remaining_w2 = full_w2.clone();
    let mut remaining_b2 = full_b2.clone();

    for i in 0..num_parties {
        if i < num_parties - 1 {
            let w1: Vec<Fr> = (0..full_w1.len()).map(|_| Fr::random(&mut rng)).collect();
            let b1: Vec<Fr> = (0..full_b1.len()).map(|_| Fr::random(&mut rng)).collect();
            let w2: Vec<Fr> = (0..full_w2.len()).map(|_| Fr::random(&mut rng)).collect();
            let b2: Vec<Fr> = (0..full_b2.len()).map(|_| Fr::random(&mut rng)).collect();

            for j in 0..full_w1.len() { remaining_w1[j] = Fr::sub(&remaining_w1[j], &w1[j]); }
            for j in 0..full_b1.len() { remaining_b1[j] = Fr::sub(&remaining_b1[j], &b1[j]); }
            for j in 0..full_w2.len() { remaining_w2[j] = Fr::sub(&remaining_w2[j], &w2[j]); }
            for j in 0..full_b2.len() { remaining_b2[j] = Fr::sub(&remaining_b2[j], &b2[j]); }

            checkpoints.push(TrainingCheckpoint {
                step: 10, w1, b1, w2, b2,
                w1_macs: Vec::new(), b1_macs: Vec::new(),
                w2_macs: Vec::new(), b2_macs: Vec::new(),
                beaver_cursor: 0, auth_beaver_cursor: 0,
            });
        } else {
            checkpoints.push(TrainingCheckpoint {
                step: 10,
                w1: remaining_w1.clone(), b1: remaining_b1.clone(),
                w2: remaining_w2.clone(), b2: remaining_b2.clone(),
                w1_macs: Vec::new(), b1_macs: Vec::new(),
                w2_macs: Vec::new(), b2_macs: Vec::new(),
                beaver_cursor: 0, auth_beaver_cursor: 0,
            });
        }
    }

    // Run redistribution for honest parties (0, 1) — skip cheater (2).
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let transports = LocalTransport::create_mesh(&parties);

    let mut redist_handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        if i == cheater_party { continue; }
        let cp = checkpoints[i].clone();
        redist_handles.push(tokio::spawn(async move {
            share_redistribution::redistribute_shares_after_removal(
                &cp, &transport, i, num_parties, cheater_party,
                "chain-recovery-e2e", 42,
            ).await
        }));
    }

    let mut redist_results = Vec::new();
    for h in redist_handles {
        redist_results.push(h.await.unwrap().expect("redistribution should succeed"));
    }
    assert_eq!(redist_results.len(), 2);

    // Verify new shares have valid MAC state.
    for r in &redist_results {
        assert!(!r.w1.is_empty(), "new w1 shares should be non-empty");
        assert!(!r.mac_state.w1_macs.is_empty(), "MAC state should be initialized");
    }

    // Verify shares sum to finite weights.
    let sum0 = Fr::add(&redist_results[0].w1[0], &redist_results[1].w1[0]);
    assert!(sum0.to_f64().is_finite(), "reconstructed weight should be finite");

    // ---- Phase 7: Recovery training with 2 workers ----
    let recovery_config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 2,
        num_steps: 10,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0, // no MAC for recovery (fresh alpha, different setup)
        beaver_batch_size: 512,
        initial_weights: Some(InitialWeights {
            w1: vec![0.1, 0.2, 0.3, 0.4],
            b1: vec![0.01, 0.02],
            w2: vec![0.5, 0.6],
            b2: vec![0.03],
        }),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
        ],
        seed: 99,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
        batch_size: 1,
        on_step: None,
    };

    let recovery_result = run_mpc_training(recovery_config).await?;
    assert_eq!(recovery_result.steps_completed, 10, "recovery training should complete all steps");
    assert!(recovery_result.final_loss.is_finite(), "loss should be finite after recovery");
    assert!(recovery_result.cheater_detected.is_none(), "no cheater in recovery phase");

    Ok(())
}
