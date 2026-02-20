//! On-chain client for HelixCoordinatorV4 (MPC-primary architecture).
//!
//! Provides typed methods for the V4 training lifecycle:
//!   Job registration → Worker staking → Checkpoint attestation →
//!   MAC failure reporting → Training completion → Stake withdrawal
//!
//! Deploys directly from Rust using the compiled JSON artifact (no forge script needed).
//! Gated behind the `chain` feature flag.
//!
//! **Build prerequisite**: Run `forge build` in `helix/contracts/` before compiling
//! this crate so the JSON artifact exists.

use std::str::FromStr;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, U256};
use ethers::utils::keccak256;
use tokio::sync::RwLock;

use super::chain::ChainCircuitBreaker;

// Generate type-safe bindings + deploy from compiled JSON artifact.
abigen!(
    HelixCoordinatorV4Contract,
    "../../contracts/out/HelixCoordinatorV4.sol/HelixCoordinatorV4.json"
);

// Halo2Verifier binding for deploying the real on-chain verifier.
abigen!(
    Halo2VerifierContract,
    "../../contracts/out/Halo2Verifier.sol/Halo2Verifier.json"
);

// MockVerifier binding for demo mode (accepts all proofs, off-chain verification is primary).
abigen!(
    MockVerifierContract,
    "../../contracts/out/MockVerifier.sol/MockVerifier.json"
);

// HelixModelStore (ERC-721) binding for minting model NFTs after training completion.
abigen!(
    HelixModelStoreContract,
    "../../contracts/out/HelixModelStore.sol/HelixModelStore.json"
);

// ============ Data Types ============

/// V4 deployment result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct V4DeployResult {
    pub coordinator: String,
    pub treasury: String,
}

/// Job summary from getJobSummary().
#[derive(Debug, Clone)]
pub struct V4JobSummary {
    pub owner: Address,
    pub current_step: u64,
    pub num_rounds: u64,
    pub payment_amount: U256,
    pub active_worker_count: u64,
    pub active: bool,
    pub completed: bool,
    pub zk_enabled: bool,
    pub zk_activated_by_risk: bool,
}

/// Worker info from getWorkerInfo().
#[derive(Debug, Clone)]
pub struct V4WorkerInfo {
    pub stake_amount: U256,
    pub joined_at_step: u64,
    pub last_active_step: u64,
    pub registered: bool,
    pub slashed: bool,
}

/// Checkpoint data from getCheckpoint().
#[derive(Debug, Clone)]
pub struct V4Checkpoint {
    pub step_number: u64,
    pub weight_commitment: [u8; 32],
    pub loss: U256,
    pub timestamp: u64,
    pub signer_count: u64,
}

/// MAC failure report from getMACFailureReport().
#[derive(Debug, Clone)]
pub struct V4MACFailureReport {
    pub step_number: u64,
    pub cheater: Address,
    pub timestamp: u64,
    pub slashed_amount: U256,
    pub reporter_count: u64,
}

/// Pool worker info from getPoolWorkerInfo().
#[derive(Debug, Clone)]
pub struct V4PoolWorkerInfo {
    pub endpoint: String,
    pub stake_amount: U256,
    pub registered_at: u64,
    pub available: bool,
    pub active_job_id: u64,
}

// ============ Message Builders ============

/// Build the checkpoint attestation message hash.
/// Matches Solidity: `keccak256(abi.encodePacked("HELIX_CHECKPOINT", jobId, stepNumber, weightCommitment, loss))`
pub fn build_checkpoint_message(
    job_id: U256,
    step_number: U256,
    weight_commitment: [u8; 32],
    loss: U256,
) -> [u8; 32] {
    let mut data = Vec::with_capacity(144);
    data.extend_from_slice(b"HELIX_CHECKPOINT");
    let mut buf = [0u8; 32];
    job_id.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    step_number.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    data.extend_from_slice(&weight_commitment);
    loss.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    keccak256(&data)
}

/// Build the MAC failure report message hash.
/// Matches Solidity: `keccak256(abi.encodePacked("HELIX_MAC_FAILURE", jobId, stepNumber, cheater, evidence))`
pub fn build_mac_failure_message(
    job_id: U256,
    step_number: U256,
    cheater: Address,
    evidence: &[u8],
) -> [u8; 32] {
    let mut data = Vec::with_capacity(101 + evidence.len());
    data.extend_from_slice(b"HELIX_MAC_FAILURE");
    let mut buf = [0u8; 32];
    job_id.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    step_number.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    data.extend_from_slice(cheater.as_bytes()); // 20 bytes for address
    data.extend_from_slice(evidence);
    keccak256(&data)
}

/// Build the training completion message hash.
/// Matches Solidity: `keccak256(abi.encodePacked("HELIX_COMPLETE", jobId, finalCommitment))`
pub fn build_completion_message(
    job_id: U256,
    final_commitment: [u8; 32],
) -> [u8; 32] {
    let mut data = Vec::with_capacity(78);
    data.extend_from_slice(b"HELIX_COMPLETE");
    let mut buf = [0u8; 32];
    job_id.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    data.extend_from_slice(&final_commitment);
    keccak256(&data)
}

/// Build the inference attestation message hash.
/// Matches Solidity: `keccak256(abi.encodePacked("HELIX_INFERENCE", jobId, prediction, inputHash, outputHash))`
pub fn build_inference_message(
    job_id: U256,
    prediction: U256,
    input_hash: [u8; 32],
    output_hash: [u8; 32],
) -> [u8; 32] {
    let mut data = Vec::with_capacity(15 + 32 + 32 + 32 + 32);
    data.extend_from_slice(b"HELIX_INFERENCE");
    let mut buf = [0u8; 32];
    job_id.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    prediction.to_big_endian(&mut buf);
    data.extend_from_slice(&buf);
    data.extend_from_slice(&input_hash);
    data.extend_from_slice(&output_hash);
    keccak256(&data)
}

/// Sign an inference attestation message with a wallet.
pub async fn sign_inference(
    wallet: &LocalWallet,
    job_id: U256,
    prediction: U256,
    input_hash: [u8; 32],
    output_hash: [u8; 32],
) -> Result<Bytes> {
    let message = build_inference_message(job_id, prediction, input_hash, output_hash);
    let sig = wallet
        .sign_message(&message)
        .await
        .map_err(|e| anyhow!("sign_inference: {}", e))?;
    let sig_bytes: [u8; 65] = sig.into();
    Ok(Bytes::from(sig_bytes.to_vec()))
}

// ============ Signing Helpers ============

/// Sign a checkpoint attestation message with a wallet.
/// Returns 65-byte ECDSA signature (r || s || v) compatible with the V4 contract.
pub async fn sign_checkpoint(
    wallet: &LocalWallet,
    job_id: U256,
    step_number: U256,
    weight_commitment: [u8; 32],
    loss: U256,
) -> Result<Bytes> {
    let message = build_checkpoint_message(job_id, step_number, weight_commitment, loss);
    let sig = wallet
        .sign_message(&message)
        .await
        .map_err(|e| anyhow!("sign_checkpoint: {}", e))?;
    let sig_bytes: [u8; 65] = sig.into();
    Ok(Bytes::from(sig_bytes.to_vec()))
}

/// Sign a MAC failure report message with a wallet.
pub async fn sign_mac_failure(
    wallet: &LocalWallet,
    job_id: U256,
    step_number: U256,
    cheater: Address,
    evidence: &[u8],
) -> Result<Bytes> {
    let message = build_mac_failure_message(job_id, step_number, cheater, evidence);
    let sig = wallet
        .sign_message(&message)
        .await
        .map_err(|e| anyhow!("sign_mac_failure: {}", e))?;
    let sig_bytes: [u8; 65] = sig.into();
    Ok(Bytes::from(sig_bytes.to_vec()))
}

/// Sign a training completion message with a wallet.
pub async fn sign_completion(
    wallet: &LocalWallet,
    job_id: U256,
    final_commitment: [u8; 32],
) -> Result<Bytes> {
    let message = build_completion_message(job_id, final_commitment);
    let sig = wallet
        .sign_message(&message)
        .await
        .map_err(|e| anyhow!("sign_completion: {}", e))?;
    let sig_bytes: [u8; 65] = sig.into();
    Ok(Bytes::from(sig_bytes.to_vec()))
}

// ============ Client ============

type SignedClient = SignerMiddleware<Provider<Http>, LocalWallet>;

/// On-chain client for HelixCoordinatorV4 (MPC-primary architecture).
///
/// Provides typed methods for the complete V4 training lifecycle:
/// job registration → worker staking → multi-party checkpoint attestation →
/// MAC failure reporting with slashing → training completion → stake withdrawal.
pub struct ChainClientV4 {
    client: Arc<SignedClient>,
    coordinator: HelixCoordinatorV4Contract<SignedClient>,
    coordinator_address: Address,
    circuit_breaker: Arc<RwLock<ChainCircuitBreaker>>,
}

impl ChainClientV4 {
    /// Create a client connected to an existing V4 coordinator contract.
    pub async fn new(
        rpc_url: &str,
        private_key: &str,
        coordinator_address: &str,
        chain_id: Option<u64>,
    ) -> Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url)
            .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;

        let cid = match chain_id {
            Some(id) => id,
            None => provider
                .get_chainid()
                .await
                .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?
                .as_u64(),
        };

        let pk = private_key.strip_prefix("0x").unwrap_or(private_key);
        let wallet = LocalWallet::from_str(pk)
            .map_err(|e| anyhow!("Invalid private key: {}", e))?
            .with_chain_id(cid);

        let client = Arc::new(SignerMiddleware::new(provider, wallet));
        let addr = Address::from_str(coordinator_address)
            .map_err(|e| anyhow!("Invalid coordinator address: {}", e))?;
        let coordinator = HelixCoordinatorV4Contract::new(addr, client.clone());

        Ok(Self {
            client,
            coordinator,
            coordinator_address: addr,
            circuit_breaker: Arc::new(RwLock::new(ChainCircuitBreaker::default())),
        })
    }

    /// Deploy a new V4 coordinator contract and return a connected client.
    ///
    /// Uses compiled bytecode from the forge build artifact (no forge script needed).
    /// Pass `Address::zero()` for verifier to disable optional ZK proofs.
    pub async fn deploy(
        rpc_url: &str,
        private_key: &str,
        treasury: Address,
        verifier: Address,
        chain_id: Option<u64>,
    ) -> Result<(Self, V4DeployResult)> {
        let provider = Provider::<Http>::try_from(rpc_url)
            .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;

        let cid = match chain_id {
            Some(id) => id,
            None => provider
                .get_chainid()
                .await
                .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?
                .as_u64(),
        };

        let pk = private_key.strip_prefix("0x").unwrap_or(private_key);
        let wallet = LocalWallet::from_str(pk)
            .map_err(|e| anyhow!("Invalid private key: {}", e))?
            .with_chain_id(cid);

        let client = Arc::new(SignerMiddleware::new(provider, wallet));

        let contract = HelixCoordinatorV4Contract::deploy(client.clone(), (treasury, verifier))
            .map_err(|e| anyhow!("V4 deploy prepare: {}", e))?
            .send()
            .await
            .map_err(|e| anyhow!("V4 deploy send: {}", e))?;

        let coordinator_address = contract.address();
        let deployment = V4DeployResult {
            coordinator: format!("{:?}", coordinator_address),
            treasury: format!("{:?}", treasury),
        };

        Ok((
            Self {
                client,
                coordinator: contract,
                coordinator_address,
                circuit_breaker: Arc::new(RwLock::new(ChainCircuitBreaker::default())),
            },
            deployment,
        ))
    }

    /// Create a client with a pre-constructed wallet.
    pub async fn with_wallet(
        rpc_url: &str,
        wallet: LocalWallet,
        coordinator_address: &str,
    ) -> Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url)
            .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;
        let client = Arc::new(SignerMiddleware::new(provider, wallet));
        let addr = Address::from_str(coordinator_address)
            .map_err(|e| anyhow!("Invalid coordinator address: {}", e))?;
        let coordinator = HelixCoordinatorV4Contract::new(addr, client.clone());

        Ok(Self {
            client,
            coordinator,
            coordinator_address: addr,
            circuit_breaker: Arc::new(RwLock::new(ChainCircuitBreaker::default())),
        })
    }

    /// Returns the signer's Ethereum address.
    pub fn signer_address(&self) -> Address {
        self.client.signer().address()
    }

    /// Returns the coordinator contract address.
    pub fn coordinator_address(&self) -> Address {
        self.coordinator_address
    }

    /// Get a reference to the circuit breaker.
    pub fn circuit_breaker(&self) -> &Arc<RwLock<ChainCircuitBreaker>> {
        &self.circuit_breaker
    }

    /// Get the underlying ethers provider (for direct RPC calls, e.g. balance queries).
    pub fn ethers_client(&self) -> &Arc<SignedClient> {
        &self.client
    }

    // ============ Circuit Breaker Helpers ============

    async fn check_cb(&self) -> Result<()> {
        let mut cb = self.circuit_breaker.write().await;
        if !cb.allow_request() {
            return Err(anyhow!(
                "RPC circuit breaker is open — endpoint unavailable (failures: {})",
                cb.failure_count()
            ));
        }
        Ok(())
    }

    async fn ok(&self) {
        self.circuit_breaker.write().await.record_success();
    }

    async fn fail(&self) {
        self.circuit_breaker.write().await.record_failure();
    }

    // ============ Job Registration ============

    /// Register a new MPC training job with ETH payment deposit.
    ///
    /// New parameters for risk-based ZK activation:
    /// - `zk_enabled`: If true, always require ZK proofs for checkpoints.
    /// - `zk_checkpoint_freq`: ZK proof every N checkpoints (0 = end only).
    /// - `risk_zk_enabled`: If true, auto-activate ZK when workers drop below threshold.
    /// - `min_workers_for_mpc`: Minimum worker count for MPC-only mode (0 defaults to 2).
    ///
    /// Returns `(receipt, job_id)`.
    pub async fn register_training_job(
        &self,
        architecture_hash: [u8; 32],
        checkpoint_freq: u64,
        num_rounds: u64,
        payment_amount: U256,
    ) -> Result<(TransactionReceipt, u64)> {
        self.register_training_job_with_zk(
            architecture_hash,
            checkpoint_freq,
            num_rounds,
            payment_amount,
            false,
            0,
            false,
            0,
            Address::zero(),
        )
        .await
    }

    /// Register a new MPC training job with full ZK/risk configuration.
    ///
    /// `operator` can call `assignPoolWorkers` on behalf of the job owner.
    /// Pass `Address::zero()` if not needed (owner-only).
    ///
    /// Returns `(receipt, job_id)`.
    pub async fn register_training_job_with_zk(
        &self,
        architecture_hash: [u8; 32],
        checkpoint_freq: u64,
        num_rounds: u64,
        payment_amount: U256,
        zk_enabled: bool,
        zk_checkpoint_freq: u64,
        risk_zk_enabled: bool,
        min_workers_for_mpc: u64,
        operator: Address,
    ) -> Result<(TransactionReceipt, u64)> {
        self.check_cb().await?;
        let result: Result<(TransactionReceipt, u64)> = async {
            let call = self
                .coordinator
                .register_training_job(
                    architecture_hash,
                    U256::from(checkpoint_freq),
                    U256::from(num_rounds),
                    payment_amount,
                    zk_enabled,
                    U256::from(zk_checkpoint_freq),
                    risk_zk_enabled,
                    U256::from(min_workers_for_mpc),
                    operator,
                )
                .value(payment_amount);
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("register_training_job send: {}", e))?;
            let receipt = pending
                .await
                .map_err(|e| anyhow!("register_training_job receipt: {}", e))?
                .ok_or_else(|| anyhow!("register_training_job: tx dropped"))?;

            // Parse job ID from JobRegistered event (first indexed topic)
            let coord_addr = self.coordinator_address;
            let job_id = receipt
                .logs
                .iter()
                .find_map(|log| {
                    if log.address == coord_addr && log.topics.len() >= 2 {
                        Some(U256::from(log.topics[1].as_bytes()).as_u64())
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            Ok((receipt, job_id))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Worker Registration ============

    /// Stake ETH and join a training job as a worker.
    ///
    /// Includes pre-flight checks for testnet compatibility:
    /// - Verifies contract code exists at coordinator address
    /// - Verifies sender balance covers stake + gas
    /// - Uses explicit gas limit to avoid testnet gas-estimation quirks
    pub async fn stake_and_join(
        &self,
        job_id: u64,
        stake_amount: U256,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            // Pre-flight: verify contract code exists at the coordinator address.
            // Calling a function on an EOA (no code) reverts with empty data (0x).
            let code = self
                .client
                .get_code(self.coordinator_address, None)
                .await
                .map_err(|e| anyhow!("Failed to query contract code: {}", e))?;
            if code.is_empty() {
                return Err(anyhow!(
                    "No contract code at coordinator address {:?}. \
                     Verify the contract was deployed on this chain and the address is correct.",
                    self.coordinator_address
                ));
            }

            // Pre-flight: verify sender has sufficient balance for stake + gas.
            // On testnets with load-balanced RPCs, recently-funded wallets may show
            // stale (zero) balances if the funding tx hasn't propagated yet.
            let sender = self.client.signer().address();
            let balance = self
                .client
                .get_balance(sender, None)
                .await
                .map_err(|e| anyhow!("Failed to check sender balance: {}", e))?;
            // Conservative gas buffer: 300K gas at up to 100 gwei
            let gas_buffer = U256::from(300_000u64) * U256::from(100_000_000_000u64);
            if balance < stake_amount {
                return Err(anyhow!(
                    "Insufficient balance for staking. Sender {:?} has {} wei \
                     but stake requires {} wei. The funding transaction may not \
                     have propagated yet — try again in a few seconds.",
                    sender, balance, stake_amount
                ));
            }
            if balance < stake_amount + gas_buffer {
                tracing::warn!(
                    sender = ?sender,
                    balance = %balance,
                    stake = %stake_amount,
                    "Worker balance is low — may not cover gas on high-fee chains"
                );
            }

            // Let ethers estimate gas naturally — explicit gas limits can
            // over-reserve on L2s with high gas prices, causing the max fee to
            // exceed the worker's balance.
            let call = self
                .coordinator
                .stake_and_join(U256::from(job_id))
                .value(stake_amount);
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("stake_and_join send: {}", e))?;
            let receipt = pending
                .await
                .map_err(|e| anyhow!("stake_and_join receipt: {}", e))?
                .ok_or_else(|| anyhow!("stake_and_join: tx dropped"))?;

            // Check on-chain execution status.
            if receipt.status == Some(ethers::types::U64::from(0)) {
                return Err(anyhow!(
                    "stake_and_join reverted on-chain (tx {:?}). \
                     Check: job {} exists and is active, worker {:?} not already registered, \
                     stake {} >= minStake (0.001 ETH).",
                    receipt.transaction_hash, job_id, sender, stake_amount
                ));
            }

            Ok(receipt)
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Checkpoint Submission ============

    /// Submit a multi-party attestation checkpoint.
    ///
    /// `signatures` must contain ECDSA signatures from ALL active workers,
    /// each signing the checkpoint message (see `sign_checkpoint`).
    pub async fn submit_checkpoint(
        &self,
        job_id: u64,
        step_number: u64,
        weight_commitment: [u8; 32],
        loss: U256,
        signatures: Vec<Bytes>,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.submit_checkpoint(
                U256::from(job_id),
                U256::from(step_number),
                weight_commitment,
                loss,
                signatures,
            );
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("submit_checkpoint send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("submit_checkpoint receipt: {}", e))?
                .ok_or_else(|| anyhow!("submit_checkpoint: tx dropped"))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ MAC Failure Reporting ============

    /// Report a MAC failure (cheating worker) with majority signatures.
    ///
    /// `reporter_signatures` must contain ECDSA signatures from a majority of
    /// active workers (NOT including the cheater).
    pub async fn report_mac_failure(
        &self,
        job_id: u64,
        step_number: u64,
        cheater: Address,
        evidence: Vec<u8>,
        reporter_signatures: Vec<Bytes>,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.report_mac_failure(
                U256::from(job_id),
                U256::from(step_number),
                cheater,
                Bytes::from(evidence),
                reporter_signatures,
            );
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("report_mac_failure send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("report_mac_failure receipt: {}", e))?
                .ok_or_else(|| anyhow!("report_mac_failure: tx dropped"))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Training Completion ============

    /// Complete training with a final checkpoint signed by all active workers.
    /// Triggers proportional payment distribution to workers.
    pub async fn complete_training(
        &self,
        job_id: u64,
        final_commitment: [u8; 32],
        signatures: Vec<Bytes>,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.complete_training(
                U256::from(job_id),
                final_commitment,
                signatures,
            );
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("complete_training send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("complete_training receipt: {}", e))?
                .ok_or_else(|| anyhow!("complete_training: tx dropped"))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Stake Withdrawal ============

    /// Withdraw stake after job completion and 7-day cooldown.
    pub async fn withdraw_stake(&self, job_id: u64) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.withdraw_stake(U256::from(job_id));
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("withdraw_stake send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("withdraw_stake receipt: {}", e))?
                .ok_or_else(|| anyhow!("withdraw_stake: tx dropped"))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Optional ZK Checkpoint ============

    /// Submit a checkpoint with a ZK proof (requires verifier to be set on contract).
    pub async fn submit_checkpoint_with_proof(
        &self,
        job_id: u64,
        step_number: u64,
        weight_commitment: [u8; 32],
        loss: U256,
        proof: Vec<u8>,
        public_inputs: Vec<U256>,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.submit_checkpoint_with_proof(
                U256::from(job_id),
                U256::from(step_number),
                weight_commitment,
                loss,
                Bytes::from(proof),
                public_inputs,
            );
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("submit_checkpoint_with_proof send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("submit_checkpoint_with_proof receipt: {}", e))?
                .ok_or_else(|| anyhow!("submit_checkpoint_with_proof: tx dropped"))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Inference Attestation ============

    /// Submit a multi-party inference result attestation on-chain.
    ///
    /// `signatures` must contain ECDSA signatures from active workers,
    /// each signing the inference message (see `sign_inference`).
    /// Returns `(receipt, inference_id)`.
    pub async fn submit_inference_result(
        &self,
        job_id: u64,
        prediction: u64,
        input_hash: [u8; 32],
        output_hash: [u8; 32],
        signatures: Vec<Bytes>,
    ) -> Result<(TransactionReceipt, u64)> {
        self.check_cb().await?;
        let result: Result<(TransactionReceipt, u64)> = async {
            let call = self.coordinator.submit_inference_result(
                U256::from(job_id),
                U256::from(prediction),
                input_hash,
                output_hash,
                signatures,
            );
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("submit_inference_result send: {}", e))?;
            let receipt = pending
                .await
                .map_err(|e| anyhow!("submit_inference_result receipt: {}", e))?
                .ok_or_else(|| anyhow!("submit_inference_result: tx dropped"))?;

            // Parse inference ID from InferenceResultCommitted event
            let coord_addr = self.coordinator_address;
            let inference_id = receipt
                .logs
                .iter()
                .find_map(|log| {
                    if log.address == coord_addr && log.topics.len() >= 2 {
                        Some(U256::from(log.topics[1].as_bytes()).as_u64())
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            Ok((receipt, inference_id))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get an inference result by ID.
    pub async fn get_inference_result(
        &self,
        inference_id: u64,
    ) -> Result<(u64, u64, [u8; 32], [u8; 32], u64, u64)> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_inference_result(U256::from(inference_id))
            .call()
            .await
            .map(|resp| {
                (
                    resp.0.as_u64(), // jobId
                    resp.1.as_u64(), // prediction
                    resp.2,          // inputHash
                    resp.3,          // outputHash
                    resp.4.as_u64(), // timestamp
                    resp.5.as_u64(), // signerCount
                )
            })
            .map_err(|e| anyhow!("get_inference_result: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Halo2Verifier Deployment ============

    /// Deploy the real Halo2Verifier contract for on-chain ZK proof verification.
    ///
    /// The Halo2Verifier deploys its own `Halo2VerifierCore` and `Halo2VerifyingKey`
    /// contracts in its constructor (no-args). Returns the deployed verifier address.
    ///
    /// After deployment, call `set_verifier(verifier_address)` on the coordinator
    /// to enable ZK proof verification for `submitCheckpointWithProof()`.
    pub async fn deploy_halo2_verifier(&self) -> Result<Address> {
        let contract = Halo2VerifierContract::deploy(self.client.clone(), ())
            .map_err(|e| anyhow!("Halo2Verifier deploy prepare: {}", e))?
            .send()
            .await
            .map_err(|e| anyhow!("Halo2Verifier deploy send: {}", e))?;

        Ok(contract.address())
    }

    // ============ Admin ============

    /// Update the treasury address (owner only).
    pub async fn set_treasury(&self, treasury: Address) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.set_treasury(treasury);
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("set_treasury send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("set_treasury receipt: {}", e))?
                .ok_or_else(|| anyhow!("set_treasury: tx dropped"))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Update the ZK verifier address (owner only).
    pub async fn set_verifier(&self, verifier: Address) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.set_verifier(verifier);
            let pending = call
                .send()
                .await
                .map_err(|e| anyhow!("set_verifier send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("set_verifier receipt: {}", e))?
                .ok_or_else(|| anyhow!("set_verifier: tx dropped"))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ View Functions ============

    /// Get job summary.
    pub async fn get_job_summary(&self, job_id: u64) -> Result<V4JobSummary> {
        self.check_cb().await?;
        let result: Result<V4JobSummary> = async {
            let resp = self
                .coordinator
                .get_job_summary(U256::from(job_id))
                .call()
                .await
                .map_err(|e| anyhow!("get_job_summary: {}", e))?;
            Ok(V4JobSummary {
                owner: resp.0,
                current_step: resp.1.as_u64(),
                num_rounds: resp.2.as_u64(),
                payment_amount: resp.3,
                active_worker_count: resp.4.as_u64(),
                active: resp.5,
                completed: resp.6,
                zk_enabled: resp.7,
                zk_activated_by_risk: resp.8,
            })
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get the list of active workers for a job.
    pub async fn get_active_workers(&self, job_id: u64) -> Result<Vec<Address>> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_active_workers(U256::from(job_id))
            .call()
            .await
            .map_err(|e| anyhow!("get_active_workers: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get the number of active workers.
    pub async fn get_active_worker_count(&self, job_id: u64) -> Result<u64> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_active_worker_count(U256::from(job_id))
            .call()
            .await
            .map(|v| v.as_u64())
            .map_err(|e| anyhow!("get_active_worker_count: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get worker info for a job.
    pub async fn get_worker_info(
        &self,
        job_id: u64,
        worker: Address,
    ) -> Result<V4WorkerInfo> {
        self.check_cb().await?;
        let result: Result<V4WorkerInfo> = async {
            let resp = self
                .coordinator
                .get_worker_info(U256::from(job_id), worker)
                .call()
                .await
                .map_err(|e| anyhow!("get_worker_info: {}", e))?;
            Ok(V4WorkerInfo {
                stake_amount: resp.0,
                joined_at_step: resp.1.as_u64(),
                last_active_step: resp.2.as_u64(),
                registered: resp.3,
                slashed: resp.4,
            })
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get checkpoint count for a job.
    pub async fn get_checkpoint_count(&self, job_id: u64) -> Result<u64> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_checkpoint_count(U256::from(job_id))
            .call()
            .await
            .map(|v| v.as_u64())
            .map_err(|e| anyhow!("get_checkpoint_count: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get a checkpoint by index.
    pub async fn get_checkpoint(&self, job_id: u64, index: u64) -> Result<V4Checkpoint> {
        self.check_cb().await?;
        let result: Result<V4Checkpoint> = async {
            let resp = self
                .coordinator
                .get_checkpoint(U256::from(job_id), U256::from(index))
                .call()
                .await
                .map_err(|e| anyhow!("get_checkpoint: {}", e))?;
            Ok(V4Checkpoint {
                step_number: resp.0.as_u64(),
                weight_commitment: resp.1,
                loss: resp.2,
                timestamp: resp.3.as_u64(),
                signer_count: resp.4.as_u64(),
            })
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get MAC failure report count for a job.
    pub async fn get_mac_failure_report_count(&self, job_id: u64) -> Result<u64> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_mac_failure_report_count(U256::from(job_id))
            .call()
            .await
            .map(|v| v.as_u64())
            .map_err(|e| anyhow!("get_mac_failure_report_count: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get a MAC failure report by index.
    pub async fn get_mac_failure_report(
        &self,
        job_id: u64,
        index: u64,
    ) -> Result<V4MACFailureReport> {
        self.check_cb().await?;
        let result: Result<V4MACFailureReport> = async {
            let resp = self
                .coordinator
                .get_mac_failure_report(U256::from(job_id), U256::from(index))
                .call()
                .await
                .map_err(|e| anyhow!("get_mac_failure_report: {}", e))?;
            Ok(V4MACFailureReport {
                step_number: resp.0.as_u64(),
                cheater: resp.1,
                timestamp: resp.2.as_u64(),
                slashed_amount: resp.3,
                reporter_count: resp.4.as_u64(),
            })
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Check if ZK proofs are required for a job (either user-enabled or risk-activated).
    pub async fn is_zk_required(&self, job_id: u64) -> Result<bool> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .is_zk_required(U256::from(job_id))
            .call()
            .await
            .map_err(|e| anyhow!("is_zk_required: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Check if an address is an active worker for a job.
    pub async fn is_active_worker(&self, job_id: u64, worker: Address) -> Result<bool> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .is_active_worker(U256::from(job_id), worker)
            .call()
            .await
            .map_err(|e| anyhow!("is_active_worker: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Global Worker Pool ============

    /// Register in the global worker pool with an endpoint and ETH stake.
    pub async fn register_in_pool(&self, endpoint: &str, stake: U256) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .register_in_pool(endpoint.to_string())
            .value(stake)
            .send()
            .await
            .map_err(|e| anyhow!("register_in_pool send: {}", e))?
            .await
            .map_err(|e| anyhow!("register_in_pool receipt: {}", e))?
            .ok_or_else(|| anyhow!("register_in_pool: no receipt"));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Deregister from the worker pool and withdraw stake.
    pub async fn deregister_from_pool(&self) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .deregister_from_pool()
            .send()
            .await
            .map_err(|e| anyhow!("deregister_from_pool send: {}", e))?
            .await
            .map_err(|e| anyhow!("deregister_from_pool receipt: {}", e))?
            .ok_or_else(|| anyhow!("deregister_from_pool: no receipt"));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Assign available pool workers to a job.
    pub async fn assign_pool_workers(&self, job_id: u64, count: u64) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .assign_pool_workers(U256::from(job_id), U256::from(count))
            .send()
            .await
            .map_err(|e| anyhow!("assign_pool_workers send: {}", e))?
            .await
            .map_err(|e| anyhow!("assign_pool_workers receipt: {}", e))?
            .ok_or_else(|| anyhow!("assign_pool_workers: no receipt"));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get list of all pool worker addresses.
    pub async fn get_pool_workers(&self) -> Result<Vec<Address>> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_pool_workers()
            .call()
            .await
            .map_err(|e| anyhow!("get_pool_workers: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get total pool worker count.
    pub async fn get_pool_worker_count(&self) -> Result<u64> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_pool_worker_count()
            .call()
            .await
            .map(|v| v.as_u64())
            .map_err(|e| anyhow!("get_pool_worker_count: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get available (idle) pool worker count.
    pub async fn get_available_pool_worker_count(&self) -> Result<u64> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_available_pool_worker_count()
            .call()
            .await
            .map(|v| v.as_u64())
            .map_err(|e| anyhow!("get_available_pool_worker_count: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Get pool worker info.
    pub async fn get_pool_worker_info(&self, worker: Address) -> Result<V4PoolWorkerInfo> {
        self.check_cb().await?;
        let result = self
            .coordinator
            .get_pool_worker_info(worker)
            .call()
            .await
            .map(|(endpoint, stake_amount, registered_at, available, active_job_id)| {
                V4PoolWorkerInfo {
                    endpoint,
                    stake_amount,
                    registered_at: registered_at.as_u64(),
                    available,
                    active_job_id: active_job_id.as_u64(),
                }
            })
            .map_err(|e| anyhow!("get_pool_worker_info: {}", e));
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    // ============ Event Queries ============

    /// Query JobRegistered events.
    pub async fn query_job_registered(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<JobRegisteredFilter>> {
        self.check_cb().await?;
        let result: Result<Vec<JobRegisteredFilter>> = async {
            let filter = self
                .coordinator
                .job_registered_filter()
                .from_block(from_block);
            let filter = if let Some(to) = to_block {
                filter.to_block(to)
            } else {
                filter
            };
            filter
                .query()
                .await
                .map_err(|e| anyhow!("query_job_registered: {}", e))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Query WorkerSlashed events.
    pub async fn query_worker_slashed(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<WorkerSlashedFilter>> {
        self.check_cb().await?;
        let result: Result<Vec<WorkerSlashedFilter>> = async {
            let filter = self
                .coordinator
                .worker_slashed_filter()
                .from_block(from_block);
            let filter = if let Some(to) = to_block {
                filter.to_block(to)
            } else {
                filter
            };
            filter
                .query()
                .await
                .map_err(|e| anyhow!("query_worker_slashed: {}", e))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }

    /// Query PaymentDistributed events.
    pub async fn query_payment_distributed(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<PaymentDistributedFilter>> {
        self.check_cb().await?;
        let result: Result<Vec<PaymentDistributedFilter>> = async {
            let filter = self
                .coordinator
                .payment_distributed_filter()
                .from_block(from_block);
            let filter = if let Some(to) = to_block {
                filter.to_block(to)
            } else {
                filter
            };
            filter
                .query()
                .await
                .map_err(|e| anyhow!("query_payment_distributed: {}", e))
        }
        .await;
        if result.is_ok() {
            self.ok().await;
        } else {
            self.fail().await;
        }
        result
    }
}

// ============ Model Store Client ============

/// On-chain client for HelixModelStore (ERC-721 model NFTs).
///
/// Used after training completion to mint a model NFT with version metadata.
pub struct ModelStoreClient {
    client: Arc<SignedClient>,
    store: HelixModelStoreContract<SignedClient>,
    store_address: Address,
}

/// Result of deploying a new HelixModelStore contract.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelStoreDeployResult {
    pub address: String,
}

/// Result of creating a model NFT.
#[derive(Debug, Clone)]
pub struct CreateModelResult {
    pub token_id: u64,
    pub receipt: TransactionReceipt,
}

impl ModelStoreClient {
    /// Connect to an existing HelixModelStore contract.
    pub async fn new(
        rpc_url: &str,
        private_key: &str,
        store_address: &str,
        chain_id: Option<u64>,
    ) -> Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url)
            .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;

        let cid = match chain_id {
            Some(id) => id,
            None => provider
                .get_chainid()
                .await
                .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?
                .as_u64(),
        };

        let pk = private_key.strip_prefix("0x").unwrap_or(private_key);
        let wallet = LocalWallet::from_str(pk)
            .map_err(|e| anyhow!("Invalid private key: {}", e))?
            .with_chain_id(cid);

        let client = Arc::new(SignerMiddleware::new(provider, wallet));
        let addr = Address::from_str(store_address)
            .map_err(|e| anyhow!("Invalid model store address: {}", e))?;
        let store = HelixModelStoreContract::new(addr, client.clone());

        Ok(Self {
            client,
            store,
            store_address: addr,
        })
    }

    /// Connect using a pre-constructed ethers client (shares provider with ChainClientV4).
    pub fn with_client(client: Arc<SignedClient>, store_address: Address) -> Self {
        let store = HelixModelStoreContract::new(store_address, client.clone());
        Self {
            client,
            store,
            store_address,
        }
    }

    /// Deploy a new HelixModelStore contract.
    pub async fn deploy(
        rpc_url: &str,
        private_key: &str,
        chain_id: Option<u64>,
    ) -> Result<(Self, ModelStoreDeployResult)> {
        let provider = Provider::<Http>::try_from(rpc_url)
            .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;

        let cid = match chain_id {
            Some(id) => id,
            None => provider
                .get_chainid()
                .await
                .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?
                .as_u64(),
        };

        let pk = private_key.strip_prefix("0x").unwrap_or(private_key);
        let wallet = LocalWallet::from_str(pk)
            .map_err(|e| anyhow!("Invalid private key: {}", e))?
            .with_chain_id(cid);

        let client = Arc::new(SignerMiddleware::new(provider, wallet));

        let contract = HelixModelStoreContract::deploy(client.clone(), ())
            .map_err(|e| anyhow!("ModelStore deploy prepare: {}", e))?
            .send()
            .await
            .map_err(|e| anyhow!("ModelStore deploy send: {}", e))?;

        let store_address = contract.address();
        let result = ModelStoreDeployResult {
            address: format!("{:?}", store_address),
        };

        Ok((
            Self {
                client,
                store: contract,
                store_address,
            },
            result,
        ))
    }

    /// Returns the store contract address.
    pub fn address(&self) -> Address {
        self.store_address
    }

    /// Returns the underlying ethers client.
    pub fn ethers_client(&self) -> &Arc<SignedClient> {
        &self.client
    }

    /// Create a new model NFT with a unique slug.
    /// Returns the token ID and transaction receipt.
    pub async fn create_model(
        &self,
        slug: &str,
        name: &str,
        description: &str,
        architecture: &str,
    ) -> Result<CreateModelResult> {
        let call = self.store.create_model(
            slug.to_string(),
            name.to_string(),
            description.to_string(),
            architecture.to_string(),
        );
        let pending = call
            .send()
            .await
            .map_err(|e| anyhow!("create_model send: {}", e))?;
        let receipt = pending
            .await
            .map_err(|e| anyhow!("create_model receipt: {}", e))?
            .ok_or_else(|| anyhow!("create_model: tx dropped"))?;

        // Parse token ID from ModelCreated event (first indexed topic after event sig)
        let token_id = receipt
            .logs
            .iter()
            .find_map(|log| {
                if log.address == self.store_address && log.topics.len() >= 2 {
                    Some(U256::from(log.topics[1].as_bytes()).as_u64())
                } else {
                    None
                }
            })
            .ok_or_else(|| anyhow!("ModelCreated event not found in receipt logs"))?;

        Ok(CreateModelResult { token_id, receipt })
    }

    /// Add a version to an existing model NFT.
    pub async fn add_version(
        &self,
        token_id: u64,
        semver: &str,
        root_hash: &str,
        accuracy: u64,
        session_id: &str,
        weights_stored: bool,
    ) -> Result<TransactionReceipt> {
        let call = self.store.add_version(
            U256::from(token_id),
            semver.to_string(),
            root_hash.to_string(),
            accuracy as u128, // u96 fits in u128
            session_id.to_string(),
            weights_stored,
        );
        let pending = call
            .send()
            .await
            .map_err(|e| anyhow!("add_version send: {}", e))?;
        pending
            .await
            .map_err(|e| anyhow!("add_version receipt: {}", e))?
            .ok_or_else(|| anyhow!("add_version: tx dropped"))
    }

    /// Set the model as public (accessible to all for inference).
    pub async fn set_public(
        &self,
        token_id: u64,
        is_public: bool,
    ) -> Result<TransactionReceipt> {
        let call = self.store.set_public(U256::from(token_id), is_public);
        let pending = call
            .send()
            .await
            .map_err(|e| anyhow!("set_public send: {}", e))?;
        pending
            .await
            .map_err(|e| anyhow!("set_public receipt: {}", e))?
            .ok_or_else(|| anyhow!("set_public: tx dropped"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_checkpoint_message_deterministic() {
        let msg = build_checkpoint_message(
            U256::from(0),
            U256::from(10),
            [0xAB; 32],
            U256::from(500),
        );
        assert_eq!(msg.len(), 32);
        let msg2 = build_checkpoint_message(
            U256::from(0),
            U256::from(10),
            [0xAB; 32],
            U256::from(500),
        );
        assert_eq!(msg, msg2);
    }

    #[test]
    fn test_build_checkpoint_message_varies_with_input() {
        let msg1 = build_checkpoint_message(
            U256::from(0),
            U256::from(10),
            [0xAB; 32],
            U256::from(500),
        );
        let msg2 = build_checkpoint_message(
            U256::from(1),
            U256::from(10),
            [0xAB; 32],
            U256::from(500),
        );
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_build_mac_failure_message() {
        let msg = build_mac_failure_message(
            U256::from(0),
            U256::from(5),
            Address::zero(),
            b"evidence_data",
        );
        assert_eq!(msg.len(), 32);
    }

    #[test]
    fn test_build_completion_message() {
        let msg = build_completion_message(U256::from(0), [0xCD; 32]);
        assert_eq!(msg.len(), 32);
    }

    #[test]
    fn test_v4_deploy_result_serialization() {
        let result = V4DeployResult {
            coordinator: "0x1234".to_string(),
            treasury: "0x5678".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: V4DeployResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.coordinator, "0x1234");
        assert_eq!(parsed.treasury, "0x5678");
    }
}
