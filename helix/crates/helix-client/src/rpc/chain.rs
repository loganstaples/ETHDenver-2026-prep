//! On-chain client for HelixCoordinatorV2.
//!
//! Wraps ethers Provider<Http> + LocalWallet to interact with the coordinator
//! contract. Reuses the same ABI as helix-node's sc_client.rs.
//!
//! Gated behind the `chain` feature flag.

use std::str::FromStr;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, U256};

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

/// Public inputs for MLTrainingStepCircuit (7 elements).
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

/// On-chain client for the HELIX coordinator contract.
///
/// Wraps an ethers `SignerMiddleware<Provider<Http>, LocalWallet>` to make
/// typed contract calls against `HelixCoordinatorV2`.
pub struct ChainClient {
    client: Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
    coordinator: HelixCoordinatorV2<SignerMiddleware<Provider<Http>, LocalWallet>>,
    coordinator_address: Address,
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
    }

    /// Get the current state of a model.
    pub async fn get_model_state(&self, model_id: u64) -> Result<ChainModelState> {
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
    }

    /// Start a new training round.
    pub async fn start_round(
        &self,
        model_id: u64,
        duration_secs: u64,
    ) -> Result<TransactionReceipt> {
        let call = self.coordinator.start_round(
            U256::from(model_id),
            U256::from(duration_secs),
        );
        let pending = call.send().await.map_err(|e| anyhow!("start_round send: {}", e))?;
        pending
            .await
            .map_err(|e| anyhow!("start_round receipt: {}", e))?
            .ok_or_else(|| anyhow!("start_round: tx dropped"))
    }

    /// Get the state of a specific round.
    pub async fn get_round_state(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> Result<ChainRoundState> {
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
    }

    // ======================== Staking ========================

    /// Stake ETH for a model.
    pub async fn stake(
        &self,
        model_id: u64,
        amount: U256,
    ) -> Result<TransactionReceipt> {
        let call = self.coordinator.stake(U256::from(model_id)).value(amount);
        let pending = call.send().await.map_err(|e| anyhow!("stake send: {}", e))?;
        pending
            .await
            .map_err(|e| anyhow!("stake receipt: {}", e))?
            .ok_or_else(|| anyhow!("stake: tx dropped"))
    }

    /// Unstake from a model.
    pub async fn unstake(&self, model_id: u64) -> Result<TransactionReceipt> {
        let call = self.coordinator.unstake(U256::from(model_id));
        let pending = call.send().await.map_err(|e| anyhow!("unstake send: {}", e))?;
        pending
            .await
            .map_err(|e| anyhow!("unstake receipt: {}", e))?
            .ok_or_else(|| anyhow!("unstake: tx dropped"))
    }

    /// Get stake info for an address.
    pub async fn get_stake(
        &self,
        prover: Address,
        model_id: u64,
    ) -> Result<ChainStakeInfo> {
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
    }

    /// Submit a proof with raw U256 public inputs.
    pub async fn submit_proof_raw(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        public_inputs: Vec<U256>,
    ) -> Result<TransactionReceipt> {
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
    }

    // ======================== Event Querying ========================

    /// Query ProofSubmitted events.
    pub async fn query_proof_submitted(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<ProofSubmittedFilter>> {
        let filter = self.coordinator.proof_submitted_filter().from_block(from_block);
        let filter = if let Some(to) = to_block {
            filter.to_block(to)
        } else {
            filter
        };
        filter.query().await.map_err(|e| anyhow!("query_proof_submitted: {}", e))
    }

    /// Query RoundCompleted events.
    pub async fn query_round_completed(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> Result<Vec<RoundCompletedFilter>> {
        let filter = self.coordinator.round_completed_filter().from_block(from_block);
        let filter = if let Some(to) = to_block {
            filter.to_block(to)
        } else {
            filter
        };
        filter.query().await.map_err(|e| anyhow!("query_round_completed: {}", e))
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
