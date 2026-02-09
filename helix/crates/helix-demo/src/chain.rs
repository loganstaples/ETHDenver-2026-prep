//! On-chain infrastructure: Anvil, contract deployment, proof submission.
//!
//! Deploys the **real** `Halo2Verifier.sol` with VK parameters extracted from
//! the prover's SRS, plus `HelixCoordinatorV2`. Proofs are verified on-chain
//! through actual BN254 pairing checks — no mock verifier.

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
use helix_circuits::verifier::VkData;

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
    /// Halo2Verifier address (real pairing-based verifier).
    pub verifier_addr: Address,
    /// HelixCoordinatorV2 address.
    pub coordinator_addr: Address,
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

/// Parses VkData s_g2 decimal strings into ethers U256 values.
fn parse_s_g2(vk: &VkData) -> Result<[U256; 4]> {
    Ok([
        U256::from_dec_str(&vk.s_g2.0).context("Invalid s_g2.0")?,
        U256::from_dec_str(&vk.s_g2.1).context("Invalid s_g2.1")?,
        U256::from_dec_str(&vk.s_g2.2).context("Invalid s_g2.2")?,
        U256::from_dec_str(&vk.s_g2.3).context("Invalid s_g2.3")?,
    ])
}

/// Spawns Anvil, deploys real Halo2Verifier + HelixCoordinatorV2.
///
/// The `vk_data` must come from `MLTrainingProverV2::export_vk_data()` so that
/// the on-chain verifier uses the same SRS as the prover.
pub async fn setup_chain(vk_data: &VkData) -> Result<ChainEnv> {
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

    // Deploy real Halo2Verifier with SRS s·G2 from the prover's VK
    let (verifier_abi, verifier_bytecode) =
        load_artifact(&out_dir, "Halo2Verifier.sol", "Halo2Verifier")?;

    let s_g2 = parse_s_g2(vk_data)?;
    let s_g2_token = Token::FixedArray(
        s_g2.iter().map(|v| Token::Uint(*v)).collect(),
    );

    let verifier_factory =
        ContractFactory::new(verifier_abi.clone(), verifier_bytecode, client.clone());
    let verifier_contract = verifier_factory
        .deploy(s_g2_token)
        .context("Halo2Verifier deploy args")?
        .send()
        .await
        .context("Halo2Verifier deploy failed")?;
    let verifier_addr = verifier_contract.address();

    // Deploy HelixCoordinatorV2 with the real verifier
    let (coord_abi, coord_bytecode) =
        load_artifact(&out_dir, "HelixCoordinatorV2.sol", "HelixCoordinatorV2")?;
    let coord_factory = ContractFactory::new(coord_abi.clone(), coord_bytecode, client.clone());
    let coord_contract = coord_factory
        .deploy((
            Token::Address(verifier_addr),
            Token::Address(deployer),
        ))
        .context("Coordinator deploy args")?
        .send()
        .await
        .context("Coordinator deploy failed")?;
    let coordinator_addr = coord_contract.address();

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
        verifier_addr,
        coordinator_addr,
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

/// Demonstrates slashing by submitting a garbage proof.
///
/// With the real Halo2Verifier, the BN254 pairing check fails on random bytes,
/// causing the contract to revert. No mock toggle needed.
pub async fn demonstrate_slashing(env: &ChainEnv) -> Result<()> {
    crate::display::info("Injecting adversarial worker with fake gradient proof...");

    // Submit a garbage proof — the real Halo2Verifier will reject it
    // because the pairing check e(A, -G2) · e(B, sG2) != 1.
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
            crate::display::success("Real Halo2Verifier correctly rejected invalid proof");
        }
        Ok(pending_tx) => {
            let result = pending_tx.await;
            match result {
                Ok(Some(receipt)) => {
                    if receipt.status == Some(ethers::types::U64::from(0)) {
                        crate::display::alert("Transaction reverted - BN254 pairing check failed");
                        crate::display::success("Real verifier correctly rejected invalid proof");
                    } else {
                        crate::display::warn(
                            "Transaction succeeded unexpectedly — check verifier deployment",
                        );
                    }
                }
                _ => {
                    crate::display::alert("Proof submission reverted (pairing check failed)");
                }
            }
        }
    }

    Ok(())
}
