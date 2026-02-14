//! Smart Contract Client for HelixCoordinatorV2.
//!
//! Provides typed Rust bindings to interact with the on-chain coordinator:
//! - Model registration and state queries
//! - Staking for proof submission rights
//! - ZK proof submission with MLTrainingStepCircuit public inputs
//! - Slashing and challenge mechanisms

use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, U256};
use std::convert::TryFrom;
use std::env;
use std::str::FromStr;
use std::sync::Arc;

// Abigen to generate type-safe bindings for HelixCoordinatorV2
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

/// Public inputs for MLTrainingStepV2Circuit (8 elements).
///
/// All 8 public inputs are required by `HelixCoordinatorV2` and `V3`
/// (`EXPECTED_PUBLIC_INPUTS = 8`). The 8th (`error_checksum`) is the
/// Poseidon hash of the error state, verified both in-circuit and on-chain.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingProofInputs {
    /// Old state hash (lower 128 bits).
    pub old_hash_lo: U256,
    /// Old state hash (upper 128 bits).
    pub old_hash_hi: U256,
    /// New state hash (lower 128 bits).
    pub new_hash_lo: U256,
    /// New state hash (upper 128 bits).
    pub new_hash_hi: U256,
    /// Loss value.
    pub loss: U256,
    /// Accumulated error bound.
    pub error_bound: U256,
    /// Training step number.
    pub step_number: U256,
    /// Error checksum (8th public input for Halo2 verification).
    /// Defaults to zero for backward compatibility with on-chain-only flows.
    #[serde(default)]
    pub error_checksum: U256,
}

impl TrainingProofInputs {
    /// Converts to the 8-element format required by the on-chain contract
    /// (`EXPECTED_PUBLIC_INPUTS = 8` in HelixCoordinatorV2/V3).
    pub fn to_vec(&self) -> Vec<U256> {
        vec![
            self.old_hash_lo,
            self.old_hash_hi,
            self.new_hash_lo,
            self.new_hash_hi,
            self.loss,
            self.error_bound,
            self.step_number,
            self.error_checksum,
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
            error_checksum: U256::zero(),
        }
    }
}

/// Model state from the coordinator.
#[derive(Debug, Clone)]
pub struct ModelState {
    pub current_round: u64,
    pub current_commitment: U256,
    pub active: bool,
}

/// Round state from the coordinator.
#[derive(Debug, Clone)]
pub struct RoundState {
    pub model_commitment: U256,
    pub new_commitment: U256,
    pub is_completed: bool,
    pub deadline: u64,
    pub prover: Address,
}

/// Stake info for a prover.
#[derive(Debug, Clone)]
pub struct StakeInfo {
    pub amount: U256,
    pub locked_until: u64,
    pub slashed: bool,
}

/// Smart contract client for HelixCoordinatorV2.
pub struct SCClient {
    pub client: Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
    pub coordinator: HelixCoordinatorV2<SignerMiddleware<Provider<Http>, LocalWallet>>,
    pub address: Address,
}

impl SCClient {
    /// Creates a new SCClient from environment variables.
    ///
    /// Required env vars:
    /// - `PRIVATE_KEY`: Hex-encoded private key
    /// - `COORDINATOR_ADDRESS`: Deployed HelixCoordinatorV2 address
    ///
    /// Optional:
    /// - `RPC_URL`: Defaults to `http://localhost:8545`
    pub async fn new() -> anyhow::Result<Self> {
        dotenv::dotenv().ok();

        let rpc_url = env::var("RPC_URL").unwrap_or_else(|_| "http://localhost:8545".to_string());
        let private_key = env::var("PRIVATE_KEY").expect("PRIVATE_KEY must be set");
        let contract_addr = env::var("COORDINATOR_ADDRESS").expect("COORDINATOR_ADDRESS must be set");

        Self::with_config(&rpc_url, &private_key, &contract_addr).await
    }

    /// Creates a new SCClient with explicit configuration.
    pub async fn with_config(
        rpc_url: &str,
        private_key: &str,
        coordinator_address: &str,
    ) -> anyhow::Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url)?;
        let chain_id = provider.get_chainid().await?;

        let wallet = LocalWallet::from_str(private_key)?.with_chain_id(chain_id.as_u64());
        let client = Arc::new(SignerMiddleware::new(provider, wallet));

        let address = Address::from_str(coordinator_address)?;
        let coordinator = HelixCoordinatorV2::new(address, client.clone());

        Ok(Self {
            client,
            coordinator,
            address,
        })
    }

    /// Returns the signer's address.
    pub fn signer_address(&self) -> Address {
        self.client.signer().address()
    }

    // ============ Model Management ============

    /// Registers a new model on-chain.
    pub async fn register_model(
        &self,
        ipfs_hash: &str,
        initial_commitment: U256,
        min_stake: U256,
    ) -> anyhow::Result<(TransactionReceipt, u64)> {
        let call = self.coordinator.register_model(
            ipfs_hash.to_string(),
            initial_commitment,
            min_stake,
        );
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;

        // Parse the ModelRegistered event to get the model ID
        let model_id = receipt
            .logs
            .iter()
            .find_map(|log| {
                // First topic is event signature, second is indexed modelId
                if log.topics.len() >= 2 {
                    Some(U256::from(log.topics[1].as_bytes()).as_u64())
                } else {
                    None
                }
            })
            .unwrap_or(0);

        Ok((receipt, model_id))
    }

    /// Gets the current state of a model.
    pub async fn get_model_state(&self, model_id: u64) -> anyhow::Result<ModelState> {
        let (current_round, current_commitment, active) = self
            .coordinator
            .get_model_state(U256::from(model_id))
            .call()
            .await?;
        Ok(ModelState {
            current_round: current_round.as_u64(),
            current_commitment,
            active,
        })
    }

    /// Starts a new training round.
    pub async fn start_round(&self, model_id: u64, duration_secs: u64) -> anyhow::Result<TransactionReceipt> {
        let call = self.coordinator.start_round(
            U256::from(model_id),
            U256::from(duration_secs),
        );
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }

    /// Gets the state of a specific round.
    pub async fn get_round_state(&self, model_id: u64, round_id: u64) -> anyhow::Result<RoundState> {
        let (model_commitment, new_commitment, is_completed, deadline, prover) = self
            .coordinator
            .rounds(U256::from(model_id), U256::from(round_id))
            .call()
            .await?;
        Ok(RoundState {
            model_commitment,
            new_commitment,
            is_completed,
            deadline: deadline.as_u64(),
            prover,
        })
    }

    // ============ Staking ============

    /// Stakes ETH to participate in a model's training.
    pub async fn stake(&self, model_id: u64, amount: U256) -> anyhow::Result<TransactionReceipt> {
        let call = self.coordinator.stake(U256::from(model_id)).value(amount);
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }

    /// Withdraws stake after lock period.
    pub async fn unstake(&self, model_id: u64) -> anyhow::Result<TransactionReceipt> {
        let call = self.coordinator.unstake(U256::from(model_id));
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }

    /// Gets stake info for a prover.
    pub async fn get_stake(&self, prover: Address, model_id: u64) -> anyhow::Result<StakeInfo> {
        let (amount, locked_until, slashed) = self
            .coordinator
            .get_stake(prover, U256::from(model_id))
            .call()
            .await?;
        Ok(StakeInfo {
            amount,
            locked_until: locked_until.as_u64(),
            slashed,
        })
    }

    // ============ Proof Submission ============

    /// Submits a training proof for a round.
    ///
    /// The proof must be from MLTrainingStepCircuit and the public inputs must match.
    /// Invalid proofs will result in stake slashing.
    pub async fn submit_proof(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        inputs: &TrainingProofInputs,
    ) -> anyhow::Result<TransactionReceipt> {
        let call = self.coordinator.submit_proof(
            U256::from(model_id),
            U256::from(round_id),
            ethers::types::Bytes::from(proof),
            inputs.to_vec(),
        );
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }

    /// Submits a proof using raw U256 public inputs.
    pub async fn submit_proof_raw(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        public_inputs: Vec<U256>,
    ) -> anyhow::Result<TransactionReceipt> {
        let call = self.coordinator.submit_proof(
            U256::from(model_id),
            U256::from(round_id),
            ethers::types::Bytes::from(proof),
            public_inputs,
        );
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }

    /// Submits an aggregated proof via `submitAggregatedProof()` on V3.
    ///
    /// This calls `HelixCoordinatorV3.submitAggregatedProof(modelId, roundId, proof, publicInputs, numSteps)`
    /// which verifies the single aggregated proof against the aggregation verifier contract.
    /// Uses raw calldata encoding since the abigen bindings are for V2 only.
    pub async fn submit_aggregated_proof(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        public_inputs: Vec<U256>,
        num_steps: u64,
    ) -> anyhow::Result<TransactionReceipt> {
        // submitAggregatedProof(uint256,uint256,bytes,uint256[],uint256)
        let selector =
            ethers::utils::id("submitAggregatedProof(uint256,uint256,bytes,uint256[],uint256)");
        let encoded = ethers::abi::encode(&[
            ethers::abi::Token::Uint(U256::from(model_id)),
            ethers::abi::Token::Uint(U256::from(round_id)),
            ethers::abi::Token::Bytes(proof),
            ethers::abi::Token::Array(
                public_inputs
                    .iter()
                    .map(|pi| ethers::abi::Token::Uint(*pi))
                    .collect(),
            ),
            ethers::abi::Token::Uint(U256::from(num_steps)),
        ]);

        let mut calldata = selector[..4].to_vec();
        calldata.extend_from_slice(&encoded);

        let tx = ethers::types::TransactionRequest::new()
            .to(self.address)
            .data(calldata);

        let pending_tx = self.client.send_transaction(tx, None).await?;
        let receipt = pending_tx
            .await?
            .ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }

    /// Challenges a previously submitted proof (for fraud detection).
    pub async fn challenge_proof(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        public_inputs: Vec<U256>,
    ) -> anyhow::Result<TransactionReceipt> {
        let call = self.coordinator.challenge_proof(
            U256::from(model_id),
            U256::from(round_id),
            ethers::types::Bytes::from(proof),
            public_inputs,
        );
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }

    // ============ Event Querying ============

    /// Queries ProofSubmitted events from a block range.
    pub async fn query_proof_submitted(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> anyhow::Result<Vec<ProofSubmittedFilter>> {
        let filter = self.coordinator.proof_submitted_filter()
            .from_block(from_block);
        let filter = if let Some(to) = to_block {
            filter.to_block(to)
        } else {
            filter
        };
        let events = filter.query().await?;
        Ok(events)
    }

    /// Queries Slashed events from a block range.
    pub async fn query_slashed(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> anyhow::Result<Vec<SlashedFilter>> {
        let filter = self.coordinator.slashed_filter()
            .from_block(from_block);
        let filter = if let Some(to) = to_block {
            filter.to_block(to)
        } else {
            filter
        };
        let events = filter.query().await?;
        Ok(events)
    }

    /// Queries RoundCompleted events from a block range.
    pub async fn query_round_completed(
        &self,
        from_block: u64,
        to_block: Option<u64>,
    ) -> anyhow::Result<Vec<RoundCompletedFilter>> {
        let filter = self.coordinator.round_completed_filter()
            .from_block(from_block);
        let filter = if let Some(to) = to_block {
            filter.to_block(to)
        } else {
            filter
        };
        let events = filter.query().await?;
        Ok(events)
    }
}

/// Helper to convert Fr field element to U256.
#[cfg(feature = "halo2")]
pub fn fr_to_u256(fr: &helix_circuits::halo2curves::bn256::Fr) -> U256 {
    use helix_circuits::halo2curves::ff::PrimeField;
    let repr = fr.to_repr();
    U256::from_little_endian(repr.as_ref())
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
            error_checksum: U256::from(42),
        };
        // On-chain format has 8 elements (matching EXPECTED_PUBLIC_INPUTS = 8)
        let vec = inputs.to_vec();
        assert_eq!(vec.len(), 8);
        assert_eq!(vec[0], U256::from(1));
        assert_eq!(vec[6], U256::from(1));
        assert_eq!(vec[7], U256::from(42));
    }
}
