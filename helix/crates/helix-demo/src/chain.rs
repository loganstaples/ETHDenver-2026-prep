//! On-chain infrastructure: Anvil, contract deployment, proof submission.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ethers::abi::{Abi, Token};
use ethers::contract::{Contract, ContractFactory};
use ethers::middleware::SignerMiddleware;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, TransactionReceipt, U256};
use ethers::utils::Anvil;
use crate::worker::EvmBundle;

type SignedClient = Arc<SignerMiddleware<Provider<Http>, LocalWallet>>;

/// Complete on-chain environment for the demo.
pub struct ChainEnv {
    /// Anvil instance (killed on drop).
    #[allow(dead_code)]
    anvil: ethers::utils::AnvilInstance,
    /// Signed provider.
    #[allow(dead_code)]
    client: SignedClient,
    /// Anvil HTTP endpoint.
    pub anvil_endpoint: String,
    /// MockVerifier address.
    pub mock_verifier_addr: Address,
    /// HelixCoordinatorV2 address.
    pub coordinator_addr: Address,
    /// MockVerifier contract.
    mock_verifier: Contract<SignerMiddleware<Provider<Http>, LocalWallet>>,
    /// Coordinator contract.
    coordinator: Contract<SignerMiddleware<Provider<Http>, LocalWallet>>,
    /// Deployer address.
    #[allow(dead_code)]
    deployer: Address,
    /// Model ID (set after registration).
    #[allow(dead_code)]
    pub model_id: Option<U256>,
    /// Max error bound from contract.
    #[allow(dead_code)]
    pub max_error_bound: U256,
}

/// Returns the path to the contracts directory.
fn contracts_dir() -> PathBuf {
    // Navigate from crates/helix-demo to contracts
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates dir")
        .parent()
        .expect("helix dir")
        .join("contracts")
}

/// Ensures contracts are compiled, returns the output directory.
fn ensure_compiled() -> Result<PathBuf> {
    let dir = contracts_dir();
    let out = dir.join("out");

    let coord_artifact = out
        .join("HelixCoordinatorV2.sol")
        .join("HelixCoordinatorV2.json");

    if coord_artifact.exists() {
        return Ok(out);
    }

    crate::display::info("Compiling contracts with forge build...");
    let status = std::process::Command::new("forge")
        .arg("build")
        .current_dir(&dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .context("Failed to run forge build. Is Foundry installed?")?;

    if !status.success() {
        bail!("forge build failed");
    }
    Ok(out)
}

/// Loads a contract artifact from the Foundry output.
fn load_artifact(out_dir: &Path, sol_file: &str, contract_name: &str) -> Result<(Abi, Bytes)> {
    let path = out_dir
        .join(sol_file)
        .join(format!("{}.json", contract_name));

    let json: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&path)
            .with_context(|| format!("Failed to open artifact {:?}", path))?,
    )
    .context("Failed to parse artifact JSON")?;

    let abi: Abi = serde_json::from_value(json["abi"].clone()).context("Failed to parse ABI")?;

    let bytecode_hex = json["bytecode"]["object"]
        .as_str()
        .context("No bytecode in artifact")?;
    let hex_str = bytecode_hex.strip_prefix("0x").unwrap_or(bytecode_hex);
    let bytecode = Bytes::from(hex::decode(hex_str).context("Invalid bytecode hex")?);

    Ok((abi, bytecode))
}

/// Spawns Anvil, deploys MockVerifier + HelixCoordinatorV2.
pub async fn setup_chain() -> Result<ChainEnv> {
    let out_dir = ensure_compiled()?;

    // Spawn Anvil
    let anvil = Anvil::new().spawn();
    let endpoint = anvil.endpoint();
    let provider =
        Provider::<Http>::try_from(&endpoint).context("Failed to connect to Anvil")?;

    let wallet: LocalWallet = anvil.keys()[0].clone().into();
    let wallet = wallet.with_chain_id(anvil.chain_id());
    let deployer = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider, wallet));

    // Deploy MockVerifier
    let (mock_abi, mock_bytecode) =
        load_artifact(&out_dir, "Deploy.s.sol", "MockVerifierForDeploy")?;
    let mock_factory = ContractFactory::new(mock_abi.clone(), mock_bytecode, client.clone());
    let mock_contract = mock_factory
        .deploy(())
        .context("MockVerifier deploy args")?
        .send()
        .await
        .context("MockVerifier deploy failed")?;
    let mock_verifier_addr = mock_contract.address();

    // Deploy HelixCoordinatorV2
    let (coord_abi, coord_bytecode) =
        load_artifact(&out_dir, "HelixCoordinatorV2.sol", "HelixCoordinatorV2")?;
    let coord_factory = ContractFactory::new(coord_abi.clone(), coord_bytecode, client.clone());
    let coord_contract = coord_factory
        .deploy((
            Token::Address(mock_verifier_addr),
            Token::Address(deployer),
        ))
        .context("Coordinator deploy args")?
        .send()
        .await
        .context("Coordinator deploy failed")?;
    let coordinator_addr = coord_contract.address();

    let mock_verifier = Contract::new(mock_verifier_addr, mock_abi, client.clone());
    let coordinator = Contract::new(coordinator_addr, coord_abi, client.clone());

    let max_error_bound: U256 = coordinator
        .method::<_, U256>("maxErrorBound", ())
        .context("maxErrorBound method")?
        .call()
        .await
        .context("maxErrorBound call failed")?;

    Ok(ChainEnv {
        anvil,
        client,
        anvil_endpoint: endpoint,
        mock_verifier_addr,
        coordinator_addr,
        mock_verifier,
        coordinator,
        deployer,
        model_id: None,
        max_error_bound,
    })
}

/// Registers a model on-chain and returns the model ID.
pub async fn register_model(env: &ChainEnv) -> Result<U256> {
    let next_id: u32 = env
        .coordinator
        .method::<_, u32>("nextModelId", ())
        .context("nextModelId method")?
        .call()
        .await
        .context("nextModelId call failed")?;
    let model_id = U256::from(next_id);

    let _receipt: TransactionReceipt = env
        .coordinator
        .method::<_, ()>(
            "registerModel",
            (
                "ipfs://helix-demo-model".to_string(),
                U256::zero(),
                U256::zero(),
            ),
        )
        .context("registerModel method")?
        .send()
        .await
        .context("registerModel send failed")?
        .await
        .context("registerModel confirm failed")?
        .context("registerModel receipt missing")?;

    Ok(model_id)
}

/// Stakes ETH for workers.
pub async fn stake_for_workers(env: &ChainEnv, model_id: U256, num_workers: usize) -> Result<()> {
    let stake_amount = ethers::utils::parse_ether(1u64).unwrap();

    for _ in 0..num_workers {
        let _receipt: TransactionReceipt = env
            .coordinator
            .method::<_, ()>("stake", model_id)
            .context("stake method")?
            .value(stake_amount)
            .send()
            .await
            .context("stake send failed")?
            .await
            .context("stake confirm failed")?
            .context("stake receipt missing")?;
    }
    Ok(())
}

/// Starts a training round.
pub async fn start_round(env: &ChainEnv, model_id: U256) -> Result<()> {
    let duration = U256::from(3600u64); // 1 hour
    let _receipt: TransactionReceipt = env
        .coordinator
        .method::<_, ()>("startRound", (model_id, duration))
        .context("startRound method")?
        .send()
        .await
        .context("startRound send failed")?
        .await
        .context("startRound confirm failed")?
        .context("startRound receipt missing")?;

    Ok(())
}

/// Submits a proof on-chain. Returns gas used.
pub async fn submit_proof(env: &ChainEnv, bundle: &EvmBundle) -> Result<u64> {
    let proof_bytes = Bytes::from(bundle.proof_bytes.clone());
    let public_inputs: Vec<U256> = bundle
        .public_inputs_u256
        .iter()
        .map(|pi| U256::from_big_endian(pi))
        .collect();

    let model_id = U256::zero();
    let round_id = U256::zero();

    let receipt: TransactionReceipt = env
        .coordinator
        .method::<_, ()>(
            "submitProof",
            (model_id, round_id, proof_bytes, public_inputs),
        )
        .context("submitProof method")?
        .gas(5_000_000u64)
        .send()
        .await
        .context("submitProof send failed")?
        .await
        .context("submitProof confirm failed")?
        .context("submitProof receipt missing")?;

    Ok(receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0))
}

/// Demonstrates slashing by submitting a bad proof.
pub async fn demonstrate_slashing(env: &ChainEnv) -> Result<()> {
    crate::display::info("Injecting adversarial worker with fake gradient proof...");

    // Set mock verifier to reject
    let _: TransactionReceipt = env
        .mock_verifier
        .method::<_, ()>("setAccept", false)
        .context("setAccept method")?
        .send()
        .await
        .context("setAccept send failed")?
        .await
        .context("setAccept confirm failed")?
        .context("setAccept receipt missing")?;

    // Submit a garbage proof
    let fake_proof = Bytes::from(vec![0xDE; 320]);
    let fake_pi = vec![U256::from(1u64); 8];
    let model_id = U256::zero();
    let round_id = U256::zero();

    let call = env
        .coordinator
        .method::<_, ()>(
            "submitProof",
            (model_id, round_id, fake_proof, fake_pi),
        )
        .context("submitProof method")?
        .gas(5_000_000u64);

    match call.send().await {
        Err(_) => {
            crate::display::alert("Proof verification FAILED (reverted on-chain)");
            crate::display::success("Contract correctly rejected invalid proof");
        }
        Ok(pending_tx) => {
            let result = pending_tx.await;
            match result {
                Ok(Some(receipt)) => {
                    if receipt.status == Some(ethers::types::U64::from(0)) {
                        crate::display::alert("Transaction reverted - invalid proof rejected");
                    } else {
                        crate::display::warn("Transaction succeeded (mock verifier may have accepted)");
                    }
                }
                _ => {
                    crate::display::alert("Proof submission reverted");
                }
            }
        }
    }

    // Reset mock verifier
    let _: TransactionReceipt = env
        .mock_verifier
        .method::<_, ()>("setAccept", true)
        .context("setAccept method")?
        .send()
        .await
        .context("setAccept send failed")?
        .await
        .context("setAccept confirm failed")?
        .context("setAccept receipt missing")?;

    Ok(())
}
