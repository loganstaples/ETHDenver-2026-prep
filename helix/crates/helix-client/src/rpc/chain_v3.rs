//! On-chain client for HelixCoordinatorV3, Staking, Rewards, HelixToken, and ModelRegistry.
//!
//! Wraps ethers `SignerMiddleware<Provider<Http>, LocalWallet>` to interact with
//! the full V3 contract stack. Extends `ChainClient` (V2) with V3-specific methods:
//! round finalization, token staking, reward claiming, training jobs, and model registry.
//!
//! Gated behind the `chain` feature flag.

use std::str::FromStr;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, U256};

use super::chain::{
    ChainCircuitBreaker, ChainModelState, ChainRoundState, TrainingProofInputs,
};
use tokio::sync::RwLock;

// ============ Contract ABIs ============

abigen!(
    HelixCoordinatorV3Contract,
    r#"[
        function registerModel(string name, string description, string ipfsHash, uint256 initialCommitment) external returns (uint256 modelId)
        function startRound(uint256 modelId, uint256 duration) external
        function startRoundWithThreshold(uint256 modelId, uint256 duration, uint32 minParticipants) external
        function submitProof(uint256 modelId, uint256 roundId, bytes proof, uint256[] publicInputs) external
        function finalizeRound(uint256 modelId, uint256 roundId) external
        function expireRound(uint256 modelId, uint256 roundId) external
        function createTrainingJob(uint256 modelId, uint256 rounds, uint256 depositAmount) external returns (uint256 jobId)
        function cancelTrainingJob(uint256 jobId) external
        function challengeProof(uint256 modelId, uint256 roundId, bytes proof, uint256[] publicInputs) external
        function getModelState(uint256 modelId) external view returns (uint256 currentRound, uint256 currentCommitment, bool active)
        function rounds(uint256 modelId, uint256 roundId) external view returns (uint256 modelCommitment, uint256 newCommitment, uint40 deadline, bool isCompleted, address prover)
        function getRoundExt(uint256 modelId, uint256 roundId) external view returns (uint32 minParticipants, uint32 validProofs, uint40 disputeDeadline, uint40 startedAt, bool finalized, address bestProver, uint256 bestLoss)
        function getRoundParticipants(uint256 modelId, uint256 roundId) external view returns (address[])
        function getAccumulatedErrorBound(uint256 modelId) external view returns (uint256)
        function getSlashingRecordCount() external view returns (uint256)
        function isProofUsed(bytes32 proofHash) external view returns (bool)
        function maxErrorBound() external view returns (uint256)
        function paused() external view returns (bool)
        event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 initialCommitment, string ipfsHash)
        event RoundStarted(uint256 indexed modelId, uint256 indexed roundId, uint256 deadline, uint256 modelCommitment)
        event ProofSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, uint256 newCommitment, uint256 errorBound)
        event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, uint256 newCommitment, uint256 totalErrorBound)
        event RoundFinalized(uint256 indexed modelId, uint256 indexed roundId, address indexed bestProver, uint256 bestLoss)
        event TrainingJobCreated(uint256 indexed modelId, uint256 indexed jobId, uint256 deposit, uint256 rounds)
    ]"#
);

abigen!(
    HelixStakingContract,
    r#"[
        function stake(uint256 amount) external
        function startUnbonding() external
        function unstake() external
        function getStakeInfo(address staker) external view returns (uint256 amount, bool isActive, bool isUnbonding, uint256 unbondingEndTime)
        function canParticipate(address staker) external view returns (bool)
        function helixToken() external view returns (address)
        function minStake() external view returns (uint256)
        function totalStaked() external view returns (uint256)
    ]"#
);

abigen!(
    HelixRewardsContract,
    r#"[
        function claimRoundRewards(uint256[] modelIds, uint256[] roundIds) external
        function fundRewardPool(uint256 amount, uint256 rewardsPerRound, uint256 duration) external
        function getClaimableRewards(address account) external view returns (uint256)
        function getRewardPoolInfo() external view returns (uint256 total, uint256 distributed, uint256 remaining, uint256 perRound, bool isActive)
        function getParticipantStats(address account) external view returns (uint256 totalEarned, uint256 totalClaimed, uint256 roundsParticipated, uint256 pendingAmount)
        event RewardsClaimed(address indexed claimer, uint256 amount)
        event RewardsAllocated(uint256 indexed modelId, uint256 indexed roundId, uint256 totalAmount, uint256 participantCount)
    ]"#
);

abigen!(
    HelixTokenContract,
    r#"[
        function approve(address spender, uint256 amount) external returns (bool)
        function balanceOf(address account) external view returns (uint256)
        function transfer(address to, uint256 amount) external returns (bool)
        function allowance(address owner, address spender) external view returns (uint256)
        function mint(address to, uint256 amount) external
        function totalSupply() external view returns (uint256)
    ]"#
);

abigen!(
    HelixModelRegistryContract,
    r#"[
        function getModel(uint256 modelId) external view returns (string name, string description, bytes32 currentCommitment, address owner, uint256 version, bool isActive)
        function getCheckpointCount(uint256 modelId) external view returns (uint256)
        function getOwnerModels(address owner) external view returns (uint256[])
    ]"#
);

// ============ Types ============

/// Extended round state for V3 multi-participant rounds.
#[derive(Debug, Clone)]
pub struct ChainRoundExtState {
    pub min_participants: u32,
    pub valid_proofs: u32,
    pub dispute_deadline: u64,
    pub started_at: u64,
    pub finalized: bool,
    pub best_prover: Address,
    pub best_loss: U256,
}

/// Staking info from V3's Staking.sol.
#[derive(Debug, Clone)]
pub struct ChainV3StakeInfo {
    pub amount: U256,
    pub is_active: bool,
    pub is_unbonding: bool,
    pub unbonding_end_time: u64,
}

/// Reward pool info.
#[derive(Debug, Clone)]
pub struct ChainRewardPoolInfo {
    pub total: U256,
    pub distributed: U256,
    pub remaining: U256,
    pub per_round: U256,
    pub is_active: bool,
}

/// Participant reward stats.
#[derive(Debug, Clone)]
pub struct ChainParticipantStats {
    pub total_earned: U256,
    pub total_claimed: U256,
    pub rounds_participated: U256,
    pub pending_amount: U256,
}

/// V3 deployment result with all contract addresses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForgeDeployResultV3 {
    pub coordinator: String,
    pub verifier: String,
    pub token: String,
    pub staking: String,
    pub rewards: String,
    pub registry: String,
    pub treasury: String,
}

type SignedClient = SignerMiddleware<Provider<Http>, LocalWallet>;

/// On-chain client for the full V3 contract stack.
///
/// Provides typed methods for the complete training lifecycle:
/// model registration → token staking → round management → proof submission →
/// round finalization → reward claiming.
pub struct ChainClientV3 {
    client: Arc<SignedClient>,
    coordinator: HelixCoordinatorV3Contract<SignedClient>,
    staking: HelixStakingContract<SignedClient>,
    rewards: HelixRewardsContract<SignedClient>,
    token: HelixTokenContract<SignedClient>,
    registry: HelixModelRegistryContract<SignedClient>,
    coordinator_address: Address,
    circuit_breaker: Arc<RwLock<ChainCircuitBreaker>>,
}

impl ChainClientV3 {
    /// Create a V3 client from explicit addresses.
    pub async fn new(
        rpc_url: &str,
        private_key: &str,
        addrs: &ForgeDeployResultV3,
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

        let coordinator_addr = Address::from_str(&addrs.coordinator)
            .map_err(|e| anyhow!("Invalid coordinator address: {}", e))?;
        let staking_addr = Address::from_str(&addrs.staking)
            .map_err(|e| anyhow!("Invalid staking address: {}", e))?;
        let rewards_addr = Address::from_str(&addrs.rewards)
            .map_err(|e| anyhow!("Invalid rewards address: {}", e))?;
        let token_addr = Address::from_str(&addrs.token)
            .map_err(|e| anyhow!("Invalid token address: {}", e))?;
        let registry_addr = Address::from_str(&addrs.registry)
            .map_err(|e| anyhow!("Invalid registry address: {}", e))?;

        Ok(Self {
            coordinator: HelixCoordinatorV3Contract::new(coordinator_addr, client.clone()),
            staking: HelixStakingContract::new(staking_addr, client.clone()),
            rewards: HelixRewardsContract::new(rewards_addr, client.clone()),
            token: HelixTokenContract::new(token_addr, client.clone()),
            registry: HelixModelRegistryContract::new(registry_addr, client.clone()),
            coordinator_address: coordinator_addr,
            client,
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

    // ============ Circuit breaker helpers ============

    async fn check_cb(&self) -> Result<()> {
        let mut cb = self.circuit_breaker.write().await;
        if !cb.allow_request() {
            return Err(anyhow!("RPC circuit breaker is open"));
        }
        Ok(())
    }

    async fn ok(&self) {
        self.circuit_breaker.write().await.record_success();
    }

    async fn fail(&self) {
        self.circuit_breaker.write().await.record_failure();
    }

    /// Get the staking contract address.
    pub fn staking_address(&self) -> Address {
        self.staking.address()
    }

    /// Get the rewards contract address.
    pub fn rewards_address(&self) -> Address {
        self.rewards.address()
    }

    /// Get the token contract address.
    pub fn token_address(&self) -> Address {
        self.token.address()
    }

    // ============ Token Operations ============

    /// Mint tokens to an address (requires minter role).
    pub async fn mint_tokens(
        &self,
        to: Address,
        amount: U256,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.token.mint(to, amount);
            let pending = call.send().await.map_err(|e| anyhow!("mint send: {}", e))?;
            pending.await.map_err(|e| anyhow!("mint receipt: {}", e))?
                .ok_or_else(|| anyhow!("mint: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Approve a spender for the given amount of HELIX tokens.
    pub async fn approve_tokens(
        &self,
        spender: Address,
        amount: U256,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.token.approve(spender, amount);
            let pending = call.send().await.map_err(|e| anyhow!("approve send: {}", e))?;
            pending.await.map_err(|e| anyhow!("approve receipt: {}", e))?
                .ok_or_else(|| anyhow!("approve: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Transfer HELIX tokens to a recipient.
    pub async fn transfer_tokens(
        &self,
        to: Address,
        amount: U256,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.token.transfer(to, amount);
            let pending = call.send().await.map_err(|e| anyhow!("transfer send: {}", e))?;
            pending.await.map_err(|e| anyhow!("transfer receipt: {}", e))?
                .ok_or_else(|| anyhow!("transfer: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get token balance.
    pub async fn token_balance(&self, account: Address) -> Result<U256> {
        self.check_cb().await?;
        let result = self.token.balance_of(account).call().await
            .map_err(|e| anyhow!("balance_of: {}", e));
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Staking Operations ============

    /// Stake HELIX tokens. Must have approved the staking contract first.
    pub async fn stake_tokens(&self, amount: U256) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.staking.stake(amount);
            let pending = call.send().await.map_err(|e| anyhow!("stake send: {}", e))?;
            pending.await.map_err(|e| anyhow!("stake receipt: {}", e))?
                .ok_or_else(|| anyhow!("stake: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get stake info for an address.
    pub async fn get_stake_info(&self, staker: Address) -> Result<ChainV3StakeInfo> {
        self.check_cb().await?;
        let result: Result<ChainV3StakeInfo> = async {
            let (amount, is_active, is_unbonding, unbonding_end_time) = self.staking
                .get_stake_info(staker).call().await
                .map_err(|e| anyhow!("get_stake_info: {}", e))?;
            Ok(ChainV3StakeInfo {
                amount,
                is_active,
                is_unbonding,
                unbonding_end_time: unbonding_end_time.as_u64(),
            })
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Check if a staker can participate.
    pub async fn can_participate(&self, staker: Address) -> Result<bool> {
        self.check_cb().await?;
        let result = self.staking.can_participate(staker).call().await
            .map_err(|e| anyhow!("can_participate: {}", e));
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get the staking contract's token address.
    pub async fn staking_token_address(&self) -> Result<Address> {
        self.check_cb().await?;
        let result = self.staking.helix_token().call().await
            .map_err(|e| anyhow!("helix_token: {}", e));
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Model Management ============

    /// Register a new model on V3.
    pub async fn register_model(
        &self,
        name: &str,
        description: &str,
        ipfs_hash: &str,
        initial_commitment: U256,
    ) -> Result<(TransactionReceipt, u64)> {
        self.check_cb().await?;
        let result: Result<(TransactionReceipt, u64)> = async {
            let call = self.coordinator.register_model(
                name.to_string(),
                description.to_string(),
                ipfs_hash.to_string(),
                initial_commitment,
            );
            let pending = call.send().await.map_err(|e| anyhow!("register_model send: {}", e))?;
            let receipt = pending.await
                .map_err(|e| anyhow!("register_model receipt: {}", e))?
                .ok_or_else(|| anyhow!("register_model: tx dropped"))?;

            // Filter by coordinator address to avoid picking up ModelRegistry events
            let coord_addr = self.coordinator_address;
            let model_id = receipt.logs.iter().find_map(|log| {
                if log.address == coord_addr && log.topics.len() >= 2 {
                    Some(U256::from(log.topics[1].as_bytes()).as_u64())
                } else {
                    None
                }
            }).unwrap_or(0);

            Ok((receipt, model_id))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get model state.
    pub async fn get_model_state(&self, model_id: u64) -> Result<ChainModelState> {
        self.check_cb().await?;
        let result: Result<ChainModelState> = async {
            let (current_round, current_commitment, active) = self.coordinator
                .get_model_state(U256::from(model_id)).call().await
                .map_err(|e| anyhow!("get_model_state: {}", e))?;
            Ok(ChainModelState {
                current_round: current_round.as_u64(),
                current_commitment,
                active,
            })
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Round Management ============

    /// Start a new training round (single-participant auto-finalize).
    pub async fn start_round(
        &self,
        model_id: u64,
        duration_secs: u64,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator
                .start_round(U256::from(model_id), U256::from(duration_secs));
            let pending = call.send().await.map_err(|e| anyhow!("start_round send: {}", e))?;
            pending.await.map_err(|e| anyhow!("start_round receipt: {}", e))?
                .ok_or_else(|| anyhow!("start_round: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Start a multi-participant round.
    pub async fn start_round_with_threshold(
        &self,
        model_id: u64,
        duration_secs: u64,
        min_participants: u32,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator
                .start_round_with_threshold(
                    U256::from(model_id),
                    U256::from(duration_secs),
                    min_participants,
                );
            let pending = call.send().await.map_err(|e| anyhow!("start_round_with_threshold send: {}", e))?;
            pending.await.map_err(|e| anyhow!("start_round_with_threshold receipt: {}", e))?
                .ok_or_else(|| anyhow!("start_round_with_threshold: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Finalize a multi-participant round after dispute period.
    pub async fn finalize_round(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator
                .finalize_round(U256::from(model_id), U256::from(round_id));
            let pending = call.send().await.map_err(|e| anyhow!("finalize_round send: {}", e))?;
            pending.await.map_err(|e| anyhow!("finalize_round receipt: {}", e))?
                .ok_or_else(|| anyhow!("finalize_round: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Expire a round that didn't meet participant threshold.
    pub async fn expire_round(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator
                .expire_round(U256::from(model_id), U256::from(round_id));
            let pending = call.send().await.map_err(|e| anyhow!("expire_round send: {}", e))?;
            pending.await.map_err(|e| anyhow!("expire_round receipt: {}", e))?
                .ok_or_else(|| anyhow!("expire_round: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get round state.
    pub async fn get_round_state(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<ChainRoundState> {
        self.check_cb().await?;
        let result: Result<ChainRoundState> = async {
            let (model_commitment, new_commitment, deadline, is_completed, prover) = self
                .coordinator
                .rounds(U256::from(model_id), U256::from(round_id))
                .call().await
                .map_err(|e| anyhow!("get_round_state: {}", e))?;
            Ok(ChainRoundState {
                model_commitment,
                new_commitment,
                is_completed,
                deadline: deadline as u64,
                prover,
            })
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get extended round state for multi-participant rounds.
    pub async fn get_round_ext(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<ChainRoundExtState> {
        self.check_cb().await?;
        let result: Result<ChainRoundExtState> = async {
            let (min_participants, valid_proofs, dispute_deadline, started_at, finalized, best_prover, best_loss) =
                self.coordinator
                    .get_round_ext(U256::from(model_id), U256::from(round_id))
                    .call().await
                    .map_err(|e| anyhow!("get_round_ext: {}", e))?;
            Ok(ChainRoundExtState {
                min_participants,
                valid_proofs,
                dispute_deadline: dispute_deadline as u64,
                started_at: started_at as u64,
                finalized,
                best_prover,
                best_loss,
            })
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get participants for a round.
    pub async fn get_round_participants(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<Vec<Address>> {
        self.check_cb().await?;
        let result = self.coordinator
            .get_round_participants(U256::from(model_id), U256::from(round_id))
            .call().await
            .map_err(|e| anyhow!("get_round_participants: {}", e));
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Proof Submission ============

    /// Submit a training proof.
    pub async fn submit_proof(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        inputs: &TrainingProofInputs,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator
                .submit_proof(
                    U256::from(model_id),
                    U256::from(round_id),
                    Bytes::from(proof),
                    inputs.to_vec(),
                );
            let pending = call.send().await.map_err(|e| anyhow!("submit_proof send: {}", e))?;
            pending.await.map_err(|e| anyhow!("submit_proof receipt: {}", e))?
                .ok_or_else(|| anyhow!("submit_proof: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
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
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator
                .submit_proof(
                    U256::from(model_id),
                    U256::from(round_id),
                    Bytes::from(proof),
                    public_inputs,
                );
            let pending = call.send().await.map_err(|e| anyhow!("submit_proof_raw send: {}", e))?;
            pending.await.map_err(|e| anyhow!("submit_proof_raw receipt: {}", e))?
                .ok_or_else(|| anyhow!("submit_proof_raw: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Training Jobs ============

    /// Create a training job with token deposit to pay workers.
    pub async fn create_training_job(
        &self,
        model_id: u64,
        rounds: u64,
        deposit_amount: U256,
    ) -> Result<(TransactionReceipt, u64)> {
        self.check_cb().await?;
        let result: Result<(TransactionReceipt, u64)> = async {
            let call = self.coordinator
                .create_training_job(
                    U256::from(model_id),
                    U256::from(rounds),
                    deposit_amount,
                );
            let pending = call.send().await.map_err(|e| anyhow!("create_training_job send: {}", e))?;
            let receipt = pending.await
                .map_err(|e| anyhow!("create_training_job receipt: {}", e))?
                .ok_or_else(|| anyhow!("create_training_job: tx dropped"))?;

            // Parse job ID from TrainingJobCreated event (filter by coordinator to
            // avoid ERC20 Transfer events whose topics contain addresses, not IDs)
            let coord_addr = self.coordinator_address;
            let job_id = receipt.logs.iter().find_map(|log| {
                if log.address == coord_addr && log.topics.len() >= 3 {
                    Some(U256::from(log.topics[2].as_bytes()).as_u64())
                } else {
                    None
                }
            }).unwrap_or(0);

            Ok((receipt, job_id))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Cancel a training job and refund remaining balance.
    pub async fn cancel_training_job(
        &self,
        job_id: u64,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.coordinator
                .cancel_training_job(U256::from(job_id));
            let pending = call.send().await.map_err(|e| anyhow!("cancel_training_job send: {}", e))?;
            pending.await.map_err(|e| anyhow!("cancel_training_job receipt: {}", e))?
                .ok_or_else(|| anyhow!("cancel_training_job: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Rewards ============

    /// Claim rewards for specific rounds.
    pub async fn claim_rewards(
        &self,
        model_ids: Vec<u64>,
        round_ids: Vec<u64>,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let m_ids: Vec<U256> = model_ids.iter().map(|&id| U256::from(id)).collect();
            let r_ids: Vec<U256> = round_ids.iter().map(|&id| U256::from(id)).collect();
            let call = self.rewards
                .claim_round_rewards(m_ids, r_ids);
            let pending = call.send().await.map_err(|e| anyhow!("claim_rewards send: {}", e))?;
            pending.await.map_err(|e| anyhow!("claim_rewards receipt: {}", e))?
                .ok_or_else(|| anyhow!("claim_rewards: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get claimable rewards for an address.
    pub async fn get_claimable_rewards(&self, account: Address) -> Result<U256> {
        self.check_cb().await?;
        let result = self.rewards.get_claimable_rewards(account).call().await
            .map_err(|e| anyhow!("get_claimable_rewards: {}", e));
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get reward pool info.
    pub async fn get_reward_pool_info(&self) -> Result<ChainRewardPoolInfo> {
        self.check_cb().await?;
        let result: Result<ChainRewardPoolInfo> = async {
            let (total, distributed, remaining, per_round, is_active) = self.rewards
                .get_reward_pool_info().call().await
                .map_err(|e| anyhow!("get_reward_pool_info: {}", e))?;
            Ok(ChainRewardPoolInfo {
                total,
                distributed,
                remaining,
                per_round,
                is_active,
            })
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get participant reward stats.
    pub async fn get_participant_stats(&self, account: Address) -> Result<ChainParticipantStats> {
        self.check_cb().await?;
        let result: Result<ChainParticipantStats> = async {
            let (total_earned, total_claimed, rounds_participated, pending_amount) = self.rewards
                .get_participant_stats(account).call().await
                .map_err(|e| anyhow!("get_participant_stats: {}", e))?;
            Ok(ChainParticipantStats {
                total_earned,
                total_claimed,
                rounds_participated,
                pending_amount,
            })
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Fund the reward pool.
    pub async fn fund_reward_pool(
        &self,
        amount: U256,
        rewards_per_round: U256,
        duration_secs: u64,
    ) -> Result<TransactionReceipt> {
        self.check_cb().await?;
        let result: Result<TransactionReceipt> = async {
            let call = self.rewards
                .fund_reward_pool(amount, rewards_per_round, U256::from(duration_secs));
            let pending = call.send().await.map_err(|e| anyhow!("fund_reward_pool send: {}", e))?;
            pending.await.map_err(|e| anyhow!("fund_reward_pool receipt: {}", e))?
                .ok_or_else(|| anyhow!("fund_reward_pool: tx dropped"))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Query ============

    /// Get accumulated error bound for a model.
    pub async fn get_accumulated_error(&self, model_id: u64) -> Result<U256> {
        self.check_cb().await?;
        let result = self.coordinator
            .get_accumulated_error_bound(U256::from(model_id)).call().await
            .map_err(|e| anyhow!("get_accumulated_error: {}", e));
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Get model checkpoint count from registry.
    pub async fn get_checkpoint_count(&self, model_id: u64) -> Result<u64> {
        self.check_cb().await?;
        let result = self.registry
            .get_checkpoint_count(U256::from(model_id)).call().await
            .map(|v| v.as_u64())
            .map_err(|e| anyhow!("get_checkpoint_count: {}", e));
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Event Queries ============

    /// Query ProofSubmitted events.
    pub async fn query_proof_submitted(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<ProofSubmittedFilter>> {
        self.check_cb().await?;
        let result: Result<Vec<ProofSubmittedFilter>> = async {
            let filter = self.coordinator.proof_submitted_filter().from_block(from_block);
            let filter = if let Some(to) = to_block {
                filter.to_block(to)
            } else {
                filter
            };
            filter.query().await.map_err(|e| anyhow!("query_proof_submitted: {}", e))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    /// Query RoundCompleted events.
    pub async fn query_round_completed(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<RoundCompletedFilter>> {
        self.check_cb().await?;
        let result: Result<Vec<RoundCompletedFilter>> = async {
            let filter = self.coordinator.round_completed_filter().from_block(from_block);
            let filter = if let Some(to) = to_block {
                filter.to_block(to)
            } else {
                filter
            };
            filter.query().await.map_err(|e| anyhow!("query_round_completed: {}", e))
        }.await;
        if result.is_ok() { self.ok().await; } else { self.fail().await; }
        result
    }

    // ============ Deployment ============

    /// Deploy V3 contracts via `forge script` and return a connected client.
    pub async fn deploy_with_forge(
        rpc_url: &str,
        private_key: &str,
        contracts_dir: &std::path::Path,
    ) -> Result<(Self, ForgeDeployResultV3)> {
        use std::process::{Command, Stdio};

        let pk = private_key.strip_prefix("0x").unwrap_or(private_key);

        let output = Command::new("forge")
            .arg("script")
            .arg("script/Deploy.s.sol")
            .arg("--sig")
            .arg("deployV3WithMock()")
            .arg("--broadcast")
            .arg("--rpc-url")
            .arg(rpc_url)
            .arg("--private-key")
            .arg(pk)
            .arg("--code-size-limit")
            .arg("99999") // V3 coordinator exceeds 24KB EIP-170 limit
            .current_dir(contracts_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| anyhow!("Failed to run forge: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("forge script deployV3WithMock failed: {}", stderr));
        }

        // Parse broadcast JSON (forge names file after --sig function)
        let broadcast_base = contracts_dir
            .join("broadcast")
            .join("Deploy.s.sol")
            .join("31337");
        let broadcast_path = {
            let specific = broadcast_base.join("deployV3WithMock-latest.json");
            if specific.exists() {
                specific
            } else {
                broadcast_base.join("run-latest.json")
            }
        };

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

        let deployment = ForgeDeployResultV3 {
            coordinator: addresses.get("HelixCoordinatorV3")
                .cloned().ok_or_else(|| anyhow!("HelixCoordinatorV3 not found in broadcast"))?,
            verifier: addresses.get("MockVerifierForDeploy")
                .or_else(|| addresses.get("Halo2Verifier"))
                .cloned().unwrap_or_default(),
            token: addresses.get("HelixToken")
                .cloned().ok_or_else(|| anyhow!("HelixToken not found in broadcast"))?,
            staking: addresses.get("Staking")
                .cloned().ok_or_else(|| anyhow!("Staking not found in broadcast"))?,
            rewards: addresses.get("Rewards")
                .cloned().ok_or_else(|| anyhow!("Rewards not found in broadcast"))?,
            registry: addresses.get("ModelRegistry")
                .cloned().ok_or_else(|| anyhow!("ModelRegistry not found in broadcast"))?,
            treasury: deployer_addr,
        };

        let client = Self::new(rpc_url, private_key, &deployment, None).await?;

        Ok((client, deployment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forge_deploy_result_v3_serialization() {
        let result = ForgeDeployResultV3 {
            coordinator: "0x1234".to_string(),
            verifier: "0x5678".to_string(),
            token: "0xaaaa".to_string(),
            staking: "0xbbbb".to_string(),
            rewards: "0xcccc".to_string(),
            registry: "0xdddd".to_string(),
            treasury: "0xeeee".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ForgeDeployResultV3 = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.coordinator, "0x1234");
        assert_eq!(parsed.token, "0xaaaa");
    }

    #[test]
    fn test_chain_v3_stake_info() {
        let info = ChainV3StakeInfo {
            amount: U256::from(100),
            is_active: true,
            is_unbonding: false,
            unbonding_end_time: 0,
        };
        assert!(info.is_active);
        assert_eq!(info.amount, U256::from(100));
    }

    #[test]
    fn test_chain_round_ext_state() {
        let ext = ChainRoundExtState {
            min_participants: 3,
            valid_proofs: 2,
            dispute_deadline: 1000,
            started_at: 500,
            finalized: false,
            best_prover: Address::zero(),
            best_loss: U256::MAX,
        };
        assert_eq!(ext.min_participants, 3);
        assert!(!ext.finalized);
    }
}
