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
        function currentRound() external view returns (uint256)
        function models(bytes32) external view returns (address, uint256, uint256)
        function updateModel(bytes32 modelId, bytes calldata proof, bytes32 newCommitment) external
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

    pub async fn get_current_round(&self) -> anyhow::Result<U256> {
        let round = self.coordinator.current_round().call().await?;
        Ok(round)
    }

    pub async fn submit_update(
        &self,
        model_id: [u8; 32],
        proof: Vec<u8>,
        new_commitment: [u8; 32],
    ) -> anyhow::Result<TransactionReceipt> {
        let call = self.coordinator.update_model(
            model_id,
            ethers::types::Bytes::from(proof),
            new_commitment,
        );
        let pending_tx = call.send().await?;
        let receipt = pending_tx.await?.ok_or_else(|| anyhow::anyhow!("Tx dropped"))?;
        Ok(receipt)
    }
}
