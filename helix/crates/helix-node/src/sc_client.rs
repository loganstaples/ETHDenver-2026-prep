use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, U256};
use std::sync::Arc;
use std::str::FromStr;
use std::convert::TryFrom;
use dotenv::dotenv;
use std::env;

// Abigen to generate type-safe bindings
abigen!(
    HelixCoordinator,
    r#"[
        function models(uint256) external view returns (string, uint256, uint256, address)
        function submitGradient(uint256 modelId, uint256 roundId, bytes calldata proof, uint256[] memory publicInputs) external
    ]"#
);

pub struct SCClient {
    pub client: Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
    pub coordinator: HelixCoordinator<SignerMiddleware<Provider<Http>, LocalWallet>>,
}

impl SCClient {
    pub async fn new() -> anyhow::Result<Self> {
        dotenv().ok();
        
        let rpc_url = env::var("RPC_URL").unwrap_or_else(|_| "http://localhost:8545".to_string());
        let private_key = env::var("PRIVATE_KEY").expect("PRIVATE_KEY must be set");
        let contract_addr = env::var("COORDINATOR_ADDRESS").expect("COORDINATOR_ADDRESS must be set");

        let provider = Provider::<Http>::try_from(rpc_url)?;
        let chain_id = provider.get_chainid().await?;
        
        let wallet = LocalWallet::from_str(&private_key)?.with_chain_id(chain_id.as_u64());
        let client = Arc::new(SignerMiddleware::new(provider, wallet));

        let address = Address::from_str(&contract_addr)?;
        let coordinator = HelixCoordinator::new(address, client.clone());

        Ok(Self { client, coordinator })
    }

    pub async fn get_model_state(&self, model_id: U256) -> anyhow::Result<(U256, U256)> {
        // models struct: (ipfsHash, currentCommitment, currentRound, owner)
        let (_ipfs, commitment, round, _owner) = self.coordinator.models(model_id).call().await?;
        Ok((round, commitment))
    }

    pub async fn submit_update(
        &self,
        model_id: U256,
        round_id: U256,
        proof: Vec<u8>,
        commitment: [u8; 32], // This is the new commitment
        old_commitment: [u8; 32] // We likely need the old commitment too for public inputs
    ) -> anyhow::Result<TransactionReceipt> {
        let commitment_u256 = U256::from(commitment);
        let old_commitment_u256 = U256::from(old_commitment);
        
        // Public Inputs: [oldCommitment, newCommitment, gradientCommitment(0 for now)]
        // The contract expects: verifyProof(proof, [old, new, ...]) 
        // Wait, HelixCoordinator.sol `submitGradient` inputs:
        // function submitGradient(uint256 modelId, uint256 roundId, bytes memory proof, uint256[] memory publicInputs)
        // verifyProof checks publicInputs.
        
        // For this fake test, let's assume public inputs are [old_commitment, new_commitment].
        // In reality, it should match the circuit's public inputs.
        let public_inputs = vec![old_commitment_u256, commitment_u256];

        let call = self.coordinator.submit_gradient(
            model_id,
            round_id,
            ethers::types::Bytes::from(proof),
            public_inputs,
        );
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }
}
