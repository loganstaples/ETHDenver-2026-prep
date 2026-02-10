//! On-chain client for HelixCoordinatorV2.
//!
//! Wraps ethers Provider<Http> + LocalWallet to interact with the coordinator
//! contract. Reuses the same ABI as helix-node's sc_client.rs.
//!
//! Gated behind the `chain` feature flag.

use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, U256};
use tokio::sync::RwLock;

/// Addresses of contracts deployed via `deploy_with_forge`.
///
/// Intentionally self-contained so chain.rs compiles in both the lib *and*
/// bin crate contexts (main.rs re-declares `mod rpc;`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForgeDeployResult {
    pub coordinator: String,
    pub verifier: String,
    pub token: Option<String>,
    pub staking: Option<String>,
    pub rewards: Option<String>,
    pub registry: Option<String>,
    pub treasury: String,
}

// Generate type-safe bindings (same ABI as helix-node sc_client.rs)
abigen!(
    HelixCoordinatorV2,
    r#"[
        function models(uint256) external view returns (string ipfsHash, uint256 currentCommitment, uint256 currentRound, address owner, uint256 minStake, bool active)
        function rounds(uint256 modelId, uint256 roundId) external view returns (uint256 modelCommitment, uint256 newCommitment, bool isCompleted, uint256 deadline, address prover)
        function stakes(address prover, uint256 modelId) external view returns (uint256 amount, uint256 lockedUntil, bool slashed)
        function getModelState(uint256 modelId) external view returns (uint256 currentRound, uint256 currentCommitment, bool active)
        function getStake(address prover, uint256 modelId) external view returns (uint256 amount, uint256 lockedUntil, bool slashed)
        function registerModel(string ipfsHash, uint256 initialCommitment, uint256 minStake) external returns (uint256 modelId)
        function startRound(uint256 modelId, uint256 duration) external
        function stake(uint256 modelId) external payable
        function unstake(uint256 modelId) external
        function submitProof(uint256 modelId, uint256 roundId, bytes proof, uint256[] publicInputs) external
        function challengeProof(uint256 modelId, uint256 roundId, bytes proof, uint256[] publicInputs) external
        event ProofSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, uint256 newCommitment)
        event Slashed(address indexed prover, uint256 indexed modelId, uint256 amount, string reason)
        event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, uint256 newCommitment)
    ]"#
);

/// Model state returned from the coordinator.
#[derive(Debug, Clone)]
pub struct ChainModelState {
    pub current_round: u64,
    pub current_commitment: U256,
    pub active: bool,
}

/// Round state returned from the coordinator.
#[derive(Debug, Clone)]
pub struct ChainRoundState {
    pub model_commitment: U256,
    pub new_commitment: U256,
    pub is_completed: bool,
    pub deadline: u64,
    pub prover: Address,
}

/// Stake info for a prover.
#[derive(Debug, Clone)]
pub struct ChainStakeInfo {
    pub amount: U256,
    pub locked_until: u64,
    pub slashed: bool,
}

/// Public inputs for MLTrainingStepV2Circuit (8 elements).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingProofInputs {
    pub old_hash_lo: U256,
    pub old_hash_hi: U256,
    pub new_hash_lo: U256,
    pub new_hash_hi: U256,
    pub loss: U256,
    pub error_bound: U256,
    pub step_number: U256,
}

impl TrainingProofInputs {
    /// Converts to the format expected by the contract.
    pub fn to_vec(&self) -> Vec<U256> {
        vec![
            self.old_hash_lo,
            self.old_hash_hi,
            self.new_hash_lo,
            self.new_hash_hi,
            self.loss,
            self.error_bound,
            self.step_number,
        ]
    }

    /// Creates from field element bytes (Fr serialized as 32-byte big-endian).
    pub fn from_bytes(
        old_hash_lo: [u8; 32],
        old_hash_hi: [u8; 32],
        new_hash_lo: [u8; 32],
        new_hash_hi: [u8; 32],
        loss: [u8; 32],
        error_bound: [u8; 32],
        step_number: u64,
    ) -> Self {
        Self {
            old_hash_lo: U256::from_big_endian(&old_hash_lo),
            old_hash_hi: U256::from_big_endian(&old_hash_hi),
            new_hash_lo: U256::from_big_endian(&new_hash_lo),
            new_hash_hi: U256::from_big_endian(&new_hash_hi),
            loss: U256::from_big_endian(&loss),
            error_bound: U256::from_big_endian(&error_bound),
            step_number: U256::from(step_number),
        }
    }
}

/// Circuit breaker state for the on-chain RPC endpoint.
///
/// Prevents hammering a down/flaky RPC by tracking consecutive failures
/// and temporarily rejecting calls when the failure threshold is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainCircuitState {
    /// Normal operation — calls flow through
    Closed,
    /// Breaker tripped — calls are rejected immediately
    Open,
    /// Testing recovery — one probe call allowed
    HalfOpen,
}

/// Circuit breaker for the on-chain RPC endpoint.
#[derive(Debug)]
pub struct ChainCircuitBreaker {
    state: ChainCircuitState,
    failure_count: u32,
    threshold: u32,
    last_failure_time: Option<Instant>,
    reset_timeout: Duration,
}

impl ChainCircuitBreaker {
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: ChainCircuitState::Closed,
            failure_count: 0,
            threshold,
            last_failure_time: None,
            reset_timeout,
        }
    }

    /// Check whether a call should be allowed through.
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            ChainCircuitState::Closed => true,
            ChainCircuitState::Open => {
                if let Some(last) = self.last_failure_time {
                    if last.elapsed() >= self.reset_timeout {
                        self.state = ChainCircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            ChainCircuitState::HalfOpen => true,
        }
    }

    /// Record a successful call — resets the breaker.
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.state = ChainCircuitState::Closed;
    }

    /// Record a failed call — may trip the breaker.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());
        if self.failure_count >= self.threshold {
            self.state = ChainCircuitState::Open;
        }
    }

    /// Get the current state.
    pub fn state(&self) -> ChainCircuitState {
        self.state
    }

    /// Get the consecutive failure count.
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }
}

impl Default for ChainCircuitBreaker {
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30))
    }
}

/// On-chain client for the HELIX coordinator contract.
///
/// Wraps an ethers `SignerMiddleware<Provider<Http>, LocalWallet>` to make
/// typed contract calls against `HelixCoordinatorV2`.
///
/// Includes a circuit breaker that trips after repeated RPC failures,
/// preventing the client from hammering a downed endpoint.
pub struct ChainClient {
    client: Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
    coordinator: HelixCoordinatorV2<SignerMiddleware<Provider<Http>, LocalWallet>>,
    coordinator_address: Address,
    circuit_breaker: Arc<RwLock<ChainCircuitBreaker>>,
}

impl ChainClient {
    /// Create a new ChainClient from explicit parameters.
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
        let coordinator = HelixCoordinatorV2::new(addr, client.clone());

        Ok(Self {
            client,
            coordinator,
            coordinator_address: addr,
            circuit_breaker: Arc::new(RwLock::new(ChainCircuitBreaker::default())),
        })
    }

    /// Create from a HelixConfig's chain fields.
    pub async fn from_config(
        rpc_url: &str,
        chain_id: u64,
        coordinator_address: &str,
        private_key: &str,
    ) -> Result<Self> {
        Self::new(rpc_url, private_key, coordinator_address, Some(chain_id)).await
    }

    /// Create from an already-constructed ethers LocalWallet.
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
        let coordinator = HelixCoordinatorV2::new(addr, client.clone());

        Ok(Self {
            client,
            coordinator,
            coordinator_address: addr,
            circuit_breaker: Arc::new(RwLock::new(ChainCircuitBreaker::default())),
        })
    }

    /// Check the circuit breaker before making an RPC call.
    /// Returns an error if the breaker is open.
    async fn check_circuit_breaker(&self) -> Result<()> {
        let mut cb = self.circuit_breaker.write().await;
        if !cb.allow_request() {
            return Err(anyhow!(
                "RPC circuit breaker is open — endpoint unavailable (failures: {}, cooldown: {}s)",
                cb.failure_count(),
                cb.reset_timeout.as_secs()
            ));
        }
        Ok(())
    }

    /// Record a successful RPC call.
    async fn record_success(&self) {
        self.circuit_breaker.write().await.record_success();
    }

    /// Record a failed RPC call.
    async fn record_failure(&self) {
        self.circuit_breaker.write().await.record_failure();
    }

    /// Execute an async operation with circuit breaker protection.
    async fn with_circuit_breaker<F, Fut, T>(&self, op_name: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        self.check_circuit_breaker().await?;
        match f().await {
            Ok(result) => {
                self.record_success().await;
                Ok(result)
            }
            Err(e) => {
                self.record_failure().await;
                Err(anyhow!("{}: {}", op_name, e))
            }
        }
    }

    /// Get a reference to the circuit breaker (for monitoring/testing).
    pub fn circuit_breaker(&self) -> &Arc<RwLock<ChainCircuitBreaker>> {
        &self.circuit_breaker
    }

    /// Returns the signer's Ethereum address.
    pub fn signer_address(&self) -> Address {
        self.client.signer().address()
    }

    /// Returns the coordinator contract address.
    pub fn coordinator_address(&self) -> Address {
        self.coordinator_address
    }

    // ======================== Model Management ========================

    /// Register a new model on-chain.
    ///
    /// Returns `(receipt, model_id)`.
    pub async fn register_model(
        &self,
        ipfs_hash: &str,
        initial_commitment: U256,
        min_stake: U256,
    ) -> Result<(TransactionReceipt, u64)> {
        self.check_circuit_breaker().await?;
        let result: Result<(TransactionReceipt, u64)> = async {
            let call = self.coordinator.register_model(
                ipfs_hash.to_string(),
                initial_commitment,
                min_stake,
            );
            let pending = call.send().await.map_err(|e| anyhow!("register_model send: {}", e))?;
            let receipt = pending
                .await
                .map_err(|e| anyhow!("register_model receipt: {}", e))?
                .ok_or_else(|| anyhow!("register_model: tx dropped"))?;

            // Parse model ID from event logs (first indexed topic after event sig)
            let model_id = receipt
                .logs
                .iter()
                .find_map(|log| {
                    if log.topics.len() >= 2 {
                        Some(U256::from(log.topics[1].as_bytes()).as_u64())
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            Ok((receipt, model_id))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Get the current state of a model.
    pub async fn get_model_state(&self, model_id: u64) -> Result<ChainModelState> {
        self.check_circuit_breaker().await?;
        let result: Result<ChainModelState> = async {
            let (current_round, current_commitment, active) = self
                .coordinator
                .get_model_state(U256::from(model_id))
                .call()
                .await
                .map_err(|e| anyhow!("get_model_state: {}", e))?;

            Ok(ChainModelState {
                current_round: current_round.as_u64(),
                current_commitment,
                active,
            })
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Start a new training round.
    pub async fn start_round(
        &self,
        model_id: u64,
        duration_secs: u64,
    ) -> Result<TransactionReceipt> {
        self.check_circuit_breaker().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.start_round(
                U256::from(model_id),
                U256::from(duration_secs),
            );
            let pending = call.send().await.map_err(|e| anyhow!("start_round send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("start_round receipt: {}", e))?
                .ok_or_else(|| anyhow!("start_round: tx dropped"))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Get the state of a specific round.
    pub async fn get_round_state(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<ChainRoundState> {
        self.check_circuit_breaker().await?;
        let result: Result<ChainRoundState> = async {
            let (model_commitment, new_commitment, is_completed, deadline, prover) = self
                .coordinator
                .rounds(U256::from(model_id), U256::from(round_id))
                .call()
                .await
                .map_err(|e| anyhow!("get_round_state: {}", e))?;

            Ok(ChainRoundState {
                model_commitment,
                new_commitment,
                is_completed,
                deadline: deadline.as_u64(),
                prover,
            })
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    // ======================== Staking ========================

    /// Stake ETH for a model.
    pub async fn stake(
        &self,
        model_id: u64,
        amount: U256,
    ) -> Result<TransactionReceipt> {
        self.check_circuit_breaker().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.stake(U256::from(model_id)).value(amount);
            let pending = call.send().await.map_err(|e| anyhow!("stake send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("stake receipt: {}", e))?
                .ok_or_else(|| anyhow!("stake: tx dropped"))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Unstake from a model.
    pub async fn unstake(&self, model_id: u64) -> Result<TransactionReceipt> {
        self.check_circuit_breaker().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.unstake(U256::from(model_id));
            let pending = call.send().await.map_err(|e| anyhow!("unstake send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("unstake receipt: {}", e))?
                .ok_or_else(|| anyhow!("unstake: tx dropped"))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Get stake info for an address.
    pub async fn get_stake(
        &self,
        prover: Address,
        model_id: u64,
    ) -> Result<ChainStakeInfo> {
        self.check_circuit_breaker().await?;
        let result: Result<ChainStakeInfo> = async {
            let (amount, locked_until, slashed) = self
                .coordinator
                .get_stake(prover, U256::from(model_id))
                .call()
                .await
                .map_err(|e| anyhow!("get_stake: {}", e))?;

            Ok(ChainStakeInfo {
                amount,
                locked_until: locked_until.as_u64(),
                slashed,
            })
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    // ======================== Proof Submission ========================

    /// Submit a training proof for a round.
    pub async fn submit_proof(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        inputs: &TrainingProofInputs,
    ) -> Result<TransactionReceipt> {
        self.check_circuit_breaker().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.submit_proof(
                U256::from(model_id),
                U256::from(round_id),
                Bytes::from(proof),
                inputs.to_vec(),
            );
            let pending = call.send().await.map_err(|e| anyhow!("submit_proof send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("submit_proof receipt: {}", e))?
                .ok_or_else(|| anyhow!("submit_proof: tx dropped"))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Submit a proof with raw U256 public inputs.
    pub async fn submit_proof_raw(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        public_inputs: Vec<U256>,
    ) -> Result<TransactionReceipt> {
        self.check_circuit_breaker().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator.submit_proof(
                U256::from(model_id),
                U256::from(round_id),
                Bytes::from(proof),
                public_inputs,
            );
            let pending = call.send().await.map_err(|e| anyhow!("submit_proof_raw send: {}", e))?;
            pending
                .await
                .map_err(|e| anyhow!("submit_proof_raw receipt: {}", e))?
                .ok_or_else(|| anyhow!("submit_proof_raw: tx dropped"))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    // ======================== Event Querying ========================

    /// Query ProofSubmitted events.
    pub async fn query_proof_submitted(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<ProofSubmittedFilter>> {
        self.check_circuit_breaker().await?;
        let result: Result<Vec<ProofSubmittedFilter>> = async {
            let filter = self.coordinator.proof_submitted_filter().from_block(from_block);
            let filter = if let Some(to) = to_block {
                filter.to_block(to)
            } else {
                filter
            };
            filter.query().await.map_err(|e| anyhow!("query_proof_submitted: {}", e))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Query RoundCompleted events.
    pub async fn query_round_completed(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<RoundCompletedFilter>> {
        self.check_circuit_breaker().await?;
        let result: Result<Vec<RoundCompletedFilter>> = async {
            let filter = self.coordinator.round_completed_filter().from_block(from_block);
            let filter = if let Some(to) = to_block {
                filter.to_block(to)
            } else {
                filter
            };
            filter.query().await.map_err(|e| anyhow!("query_round_completed: {}", e))
        }.await;
        if result.is_ok() { self.record_success().await; } else { self.record_failure().await; }
        result
    }

    /// Deploy contracts via `forge script` and return a `ChainClient`
    /// connected to the freshly deployed coordinator.
    ///
    /// This is a convenience method that:
    /// 1. Runs `forge script script/Deploy.s.sol --sig "runWithMock()" --broadcast`
    /// 2. Parses the broadcast JSON for contract addresses
    /// 3. Returns `(ChainClient, DeploymentResult)`
    pub async fn deploy_with_forge(
        rpc_url: &str,
        private_key: &str,
        contracts_dir: &std::path::Path,
    ) -> Result<(Self, ForgeDeployResult)> {
        use std::process::{Command, Stdio};

        let pk = private_key.strip_prefix("0x").unwrap_or(private_key);

        let output = Command::new("forge")
            .arg("script")
            .arg("script/Deploy.s.sol")
            .arg("--sig")
            .arg("runWithMock()")
            .arg("--broadcast")
            .arg("--rpc-url")
            .arg(rpc_url)
            .arg("--private-key")
            .arg(pk)
            .current_dir(contracts_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| anyhow!("Failed to run forge: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("forge script failed: {}", stderr));
        }

        // Parse broadcast JSON
        let broadcast_path = contracts_dir
            .join("broadcast")
            .join("Deploy.s.sol")
            .join("31337")
            .join("run-latest.json");

        let content = std::fs::read_to_string(&broadcast_path)
            .map_err(|e| anyhow!("Failed to read broadcast JSON: {}", e))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse broadcast JSON: {}", e))?;

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
                if let Some(from) = tx.get("transaction")
                    .and_then(|t| t.get("from"))
                    .and_then(|f| f.as_str())
                {
                    deployer_addr = from.to_string();
                }
            }
        }

        let coordinator_addr = addresses
            .get("HelixCoordinatorV2")
            .or_else(|| addresses.get("HelixCoordinatorV3"))
            .cloned()
            .ok_or_else(|| anyhow!("Coordinator not found in broadcast"))?;

        let verifier_addr = addresses
            .get("MockVerifierForDeploy")
            .or_else(|| addresses.get("Halo2Verifier"))
            .cloned()
            .unwrap_or_default();

        let deployment = ForgeDeployResult {
            coordinator: coordinator_addr.clone(),
            verifier: verifier_addr,
            token: addresses.get("HelixToken").cloned(),
            staking: addresses.get("Staking").cloned(),
            rewards: addresses.get("Rewards").cloned(),
            registry: addresses.get("ModelRegistry").cloned(),
            treasury: deployer_addr,
        };

        let client = Self::new(rpc_url, private_key, &coordinator_addr, None).await?;

        Ok((client, deployment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_proof_inputs() {
        let inputs = TrainingProofInputs {
            old_hash_lo: U256::from(1),
            old_hash_hi: U256::from(2),
            new_hash_lo: U256::from(3),
            new_hash_hi: U256::from(4),
            loss: U256::from(100),
            error_bound: U256::from(10),
            step_number: U256::from(1),
        };
        let vec = inputs.to_vec();
        assert_eq!(vec.len(), 7);
        assert_eq!(vec[0], U256::from(1));
        assert_eq!(vec[6], U256::from(1));
    }

    #[test]
    fn test_training_proof_inputs_from_bytes() {
        let mut lo = [0u8; 32];
        lo[31] = 42;
        let hi = [0u8; 32];
        let inputs = TrainingProofInputs::from_bytes(lo, hi, lo, hi, lo, lo, 5);
        assert_eq!(inputs.old_hash_lo, U256::from(42));
        assert_eq!(inputs.step_number, U256::from(5));
    }
}
